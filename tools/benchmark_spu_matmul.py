#!/usr/bin/env python3
"""Benchmark one production-shaped secret matrix multiplication on SPU."""

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
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--inner-dim", type=int, required=True)
    parser.add_argument("--output-dim", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
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

    dimensions = (args.rows, args.inner_dim, args.output_dim, args.repeats)
    if min(dimensions) <= 0:
        raise ValueError("all dimensions and repeats must be positive")

    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    rng = np.random.default_rng(args.seed)
    lhs = rng.normal(0.0, 0.25, size=(args.rows, args.inner_dim)).astype(np.float32)
    rhs = rng.normal(
        0.0,
        0.02,
        size=(args.output_dim, args.inner_dim),
    ).astype(np.float32)
    reference = lhs @ rhs.T

    p1 = ppd.device("P1")
    p2 = ppd.device("P2")
    spu_device = ppd.device("SPU")
    identity = lambda value: value
    lhs_ref = p1(identity)(lhs)
    rhs_ref = p2(identity)(rhs)

    def matmul(input_value, weight):
        return jnp.matmul(input_value, jnp.swapaxes(weight, -1, -2))

    secure_matmul = spu_device(matmul)
    repetitions = []
    for repeat_index in range(args.repeats):
        tx_before = read_counter(args.interface, "tx_bytes")
        rx_before = read_counter(args.interface, "rx_bytes")
        started = time.perf_counter()
        output = np.asarray(ppd.get(secure_matmul(lhs_ref, rhs_ref)), dtype=np.float32)
        elapsed = time.perf_counter() - started
        tx_after = read_counter(args.interface, "tx_bytes")
        rx_after = read_counter(args.interface, "rx_bytes")
        absolute_error = np.abs(output - reference)
        repetitions.append(
            {
                "repeat_index": repeat_index,
                "elapsed_sec": float(elapsed),
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
        "manifest_type": "transshield_spu_matmul_benchmark_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "spu_version": getattr(spu, "__version__", "unknown"),
        "config": str(config_path),
        "runtime_config": runtime_config,
        "ownership": {
            "left_matrix": "P1",
            "right_matrix": "P2",
            "reveal": "full synthetic final matrix output only",
        },
        "shape": {
            "lhs": [args.rows, args.inner_dim],
            "rhs": [args.output_dim, args.inner_dim],
            "matmul": [args.rows, args.inner_dim, args.output_dim],
        },
        "seed": int(args.seed),
        "interface": args.interface,
        "reference_output_sha256": array_sha256(reference),
        "repetitions": repetitions,
        "acceptance": {
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
