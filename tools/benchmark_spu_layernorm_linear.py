#!/usr/bin/env python3
"""Benchmark equivalent LayerNorm/linear rewrites on production-shaped tensors."""

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_json(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_counter(interface, name):
    return int(
        Path("/sys/class/net")
        .joinpath(interface, "statistics", name)
        .read_text(encoding="utf-8")
    )


def array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=197)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--output-dim", type=int, default=1536)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--fold-affine", action="store_true")
    parser.add_argument(
        "--post-linear-normalization",
        action="store_true",
        help=(
            "After affine folding, move the per-token inverse-standard-deviation "
            "multiplication after the linear map. This helps only when output-dim "
            "is smaller than embed-dim."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    import jax.numpy as jnp
    import numpy as np
    import spu
    import spu.utils.distributed as ppd

    from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
        fold_layer_norm_affine_into_linear,
    )

    dimensions = (
        args.batch_size,
        args.token_count,
        args.embed_dim,
        args.output_dim,
        args.repeats,
    )
    if min(dimensions) <= 0:
        raise ValueError("all dimensions and repeats must be positive")
    if args.post_linear_normalization and not args.fold_affine:
        raise ValueError("post-linear normalization requires --fold-affine")

    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    rng = np.random.default_rng(args.seed)
    input_shape = (args.batch_size, args.token_count, args.embed_dim)
    x = rng.normal(0.0, 0.25, size=input_shape).astype(np.float32)
    gamma = rng.normal(1.0, 0.05, size=(args.embed_dim,)).astype(np.float32)
    beta = rng.normal(0.0, 0.02, size=(args.embed_dim,)).astype(np.float32)
    weight = rng.normal(
        0.0,
        0.02,
        size=(args.output_dim, args.embed_dim),
    ).astype(np.float32)
    bias = rng.normal(0.0, 0.02, size=(args.output_dim,)).astype(np.float32)

    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + args.epsilon)
    reference = (normalized * gamma + beta) @ weight.T + bias

    if args.fold_affine:
        _, _, runtime_weight, runtime_bias = fold_layer_norm_affine_into_linear(
            gamma,
            beta,
            weight,
            bias,
        )
    else:
        runtime_weight = weight
        runtime_bias = bias

    p1 = ppd.device("P1")
    p2 = ppd.device("P2")
    spu_device = ppd.device("SPU")
    identity = lambda value: value
    x_ref = p1(identity)(x)
    weight_ref = p2(identity)(runtime_weight)
    bias_ref = p2(identity)(runtime_bias)

    if args.post_linear_normalization:

        def layernorm_linear(input_value, fc_weight, fc_bias):
            local_mean = jnp.mean(input_value, axis=-1, keepdims=True)
            local_centered = input_value - local_mean
            local_variance = jnp.mean(
                local_centered * local_centered,
                axis=-1,
                keepdims=True,
            )
            local_inverse_std = 1.0 / jnp.sqrt(
                local_variance + args.epsilon
            )
            projected = jnp.matmul(
                local_centered,
                jnp.swapaxes(fc_weight, -1, -2),
            )
            return projected * local_inverse_std + fc_bias

        secure_function = spu_device(layernorm_linear)
        function_args = (x_ref, weight_ref, bias_ref)
    elif args.fold_affine:

        def layernorm_linear(input_value, fc_weight, fc_bias):
            local_mean = jnp.mean(input_value, axis=-1, keepdims=True)
            local_centered = input_value - local_mean
            local_variance = jnp.mean(
                local_centered * local_centered,
                axis=-1,
                keepdims=True,
            )
            local_normalized = local_centered / jnp.sqrt(
                local_variance + args.epsilon
            )
            return jnp.matmul(
                local_normalized,
                jnp.swapaxes(fc_weight, -1, -2),
            ) + fc_bias

        secure_function = spu_device(layernorm_linear)
        function_args = (x_ref, weight_ref, bias_ref)
    else:
        gamma_ref = p2(identity)(gamma)
        beta_ref = p2(identity)(beta)

        def layernorm_linear(input_value, norm_gamma, norm_beta, fc_weight, fc_bias):
            local_mean = jnp.mean(input_value, axis=-1, keepdims=True)
            local_centered = input_value - local_mean
            local_variance = jnp.mean(
                local_centered * local_centered,
                axis=-1,
                keepdims=True,
            )
            local_normalized = local_centered / jnp.sqrt(
                local_variance + args.epsilon
            )
            norm_output = local_normalized * norm_gamma + norm_beta
            return jnp.matmul(
                norm_output,
                jnp.swapaxes(fc_weight, -1, -2),
            ) + fc_bias

        secure_function = spu_device(layernorm_linear)
        function_args = (x_ref, gamma_ref, beta_ref, weight_ref, bias_ref)

    repetitions = []
    for repeat_index in range(args.repeats):
        tx_before = read_counter(args.interface, "tx_bytes")
        rx_before = read_counter(args.interface, "rx_bytes")
        started = time.perf_counter()
        output = np.asarray(ppd.get(secure_function(*function_args)), dtype=np.float32)
        elapsed = time.perf_counter() - started
        tx_after = read_counter(args.interface, "tx_bytes")
        rx_after = read_counter(args.interface, "rx_bytes")
        absolute_error = np.abs(output - reference)
        repetitions.append(
            {
                "repeat_index": repeat_index,
                "elapsed_sec": elapsed,
                "loopback_tx_bytes": tx_after - tx_before,
                "loopback_rx_bytes": rx_after - rx_before,
                "output_sha256": array_sha256(output),
                "finite": bool(np.isfinite(output).all()),
                "reference_max_abs_error": float(absolute_error.max()),
                "reference_mean_abs_error": float(absolute_error.mean()),
            }
        )

    runtime_config = config["devices"]["SPU"]["config"]["runtime_config"]
    payload = {
        "manifest_type": "transshield_spu_layernorm_linear_benchmark_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "spu_version": getattr(spu, "__version__", "unknown"),
        "config": str(config_path),
        "runtime_config": runtime_config,
        "fold_layer_norm_affine": bool(args.fold_affine),
        "post_linear_normalization": bool(args.post_linear_normalization),
        "ownership": {
            "input": "P1",
            "model_parameters": "P2",
            "reveal": "full synthetic final linear output only",
        },
        "shape": {
            "input": list(input_shape),
            "linear": [
                args.batch_size * args.token_count,
                args.embed_dim,
                args.output_dim,
            ],
        },
        "epsilon": float(args.epsilon),
        "seed": int(args.seed),
        "reference_output_sha256": array_sha256(reference),
        "repetitions": repetitions,
        "acceptance": {
            "real_valued_graph_equivalent": True,
            "all_finite": all(item["finite"] for item in repetitions),
        },
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
