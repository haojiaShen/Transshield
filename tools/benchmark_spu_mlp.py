#!/usr/bin/env python3
"""Benchmark TransShield's exact secret MLP matrix path on an SPU runtime.

The benchmark keeps the production MLP graph intact:

    y = x + ((x @ W1.T + b1) ** 2 * alpha) @ W2.T + b2

The input belongs to P1 and all model parameters belong to P2.  Only synthetic
values are used.  This tool is intended for runtime/kernel A/B tests; it keeps
the model width, weights, activation, and arithmetic graph unchanged while the
runtime field and fixed-point precision are checked against explicit values.
"""

import argparse
import hashlib
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_counter(interface, name):
    counter_path = Path("/sys/class/net").joinpath(interface, "statistics", name)
    return int(counter_path.read_text(encoding="utf-8"))


def array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=197)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--mlp-dim", type=int, default=1536)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--expected-field", default="FM64")
    parser.add_argument("--expected-fxp-fraction-bits", type=int, default=16)
    return parser


def main():
    args = build_parser().parse_args()

    import jax.numpy as jnp
    import numpy as np
    import spu
    import spu.utils.distributed as ppd

    if min(args.batch_size, args.token_count, args.embed_dim, args.mlp_dim, args.repeats) <= 0:
        raise ValueError("all shape dimensions and repeats must be positive")
    if args.expected_fxp_fraction_bits <= 0:
        raise ValueError("expected fixed-point fraction bits must be positive")

    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    rng = np.random.default_rng(args.seed)
    input_shape = (args.batch_size, args.token_count, args.embed_dim)
    x = rng.normal(0.0, 0.25, size=input_shape).astype(np.float32)
    weight1 = rng.normal(0.0, 0.02, size=(args.mlp_dim, args.embed_dim)).astype(np.float32)
    bias1 = rng.normal(0.0, 0.02, size=(args.mlp_dim,)).astype(np.float32)
    alpha = np.asarray(1.0, dtype=np.float32)
    weight2 = rng.normal(0.0, 0.02, size=(args.embed_dim, args.mlp_dim)).astype(np.float32)
    bias2 = rng.normal(0.0, 0.02, size=(args.embed_dim,)).astype(np.float32)

    # Compute the float32 reference outside the timed and network-counted SPU region.
    hidden_reference = np.matmul(x, weight1.T) + bias1
    hidden_reference = (hidden_reference * hidden_reference) * alpha
    output_reference = x + np.matmul(hidden_reference, weight2.T) + bias2

    p1 = ppd.device("P1")
    p2 = ppd.device("P2")
    spu_device = ppd.device("SPU")

    identity = lambda value: value
    x_ref = p1(identity)(x)
    weight1_ref = p2(identity)(weight1)
    bias1_ref = p2(identity)(bias1)
    alpha_ref = p2(identity)(alpha)
    weight2_ref = p2(identity)(weight2)
    bias2_ref = p2(identity)(bias2)

    def exact_mlp(input_value, fc1_weight, fc1_bias, act_alpha, fc2_weight, fc2_bias):
        hidden = jnp.matmul(input_value, jnp.swapaxes(fc1_weight, -1, -2)) + fc1_bias
        hidden = act_alpha * (hidden * hidden)
        projected = jnp.matmul(hidden, jnp.swapaxes(fc2_weight, -1, -2)) + fc2_bias
        return input_value + projected

    secure_mlp = spu_device(exact_mlp)
    repetitions = []
    for repeat_index in range(args.repeats):
        tx_before = read_counter(args.interface, "tx_bytes")
        rx_before = read_counter(args.interface, "rx_bytes")
        started = time.perf_counter()
        output_ref = secure_mlp(x_ref, weight1_ref, bias1_ref, alpha_ref, weight2_ref, bias2_ref)
        output = np.asarray(ppd.get(output_ref), dtype=np.float32)
        elapsed = time.perf_counter() - started
        tx_after = read_counter(args.interface, "tx_bytes")
        rx_after = read_counter(args.interface, "rx_bytes")
        absolute_error = np.abs(output - output_reference)
        repetitions.append(
            {
                "repeat_index": repeat_index,
                "elapsed_sec": elapsed,
                "loopback_tx_bytes": tx_after - tx_before,
                "loopback_rx_bytes": rx_after - rx_before,
                "output_sha256": array_sha256(output),
                "output_min": float(output.min()),
                "output_max": float(output.max()),
                "output_mean": float(output.mean()),
                "finite": bool(np.isfinite(output).all()),
                "reference_max_abs_error": float(absolute_error.max()),
                "reference_mean_abs_error": float(absolute_error.mean()),
            }
        )

    runtime_config = config["devices"]["SPU"]["config"]["runtime_config"]
    payload = {
        "manifest_type": "transshield_spu_exact_mlp_benchmark_v2",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "spu_version": getattr(spu, "__version__", "unknown"),
        "config": str(config_path),
        "runtime_config": runtime_config,
        "ownership": {
            "input": "P1",
            "model_parameters": "P2",
            "reveal": "full synthetic final MLP output only",
        },
        "graph": "x + (alpha * square(x @ W1.T + b1)) @ W2.T + b2",
        "shape": {
            "input": list(input_shape),
            "flattened_tokens": args.batch_size * args.token_count,
            "fc1": [args.batch_size * args.token_count, args.embed_dim, args.mlp_dim],
            "fc2": [args.batch_size * args.token_count, args.mlp_dim, args.embed_dim],
        },
        "seed": args.seed,
        "interface": args.interface,
        "expected_runtime_config": {
            "field": args.expected_field,
            "fxp_fraction_bits": args.expected_fxp_fraction_bits,
        },
        "reference_output_sha256": array_sha256(output_reference),
        "repetitions": repetitions,
        "acceptance": {
            "unchanged_math_graph": True,
            "runtime_config_matches_expectation": (
                runtime_config.get("field") == args.expected_field
                and int(runtime_config.get("fxp_fraction_bits", 0))
                == args.expected_fxp_fraction_bits
            ),
            "all_finite": all(item["finite"] for item in repetitions),
        },
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
