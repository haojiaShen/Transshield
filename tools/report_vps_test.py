#!/usr/bin/env python3
"""Build detailed, VPS-only evidence for the test scopes frozen in the report.

This tool never updates the formal ``results/final`` or ``results/communication``
files.  Put every new run below ``results/vps_report_tests/<run-name>``.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "report_vps_test_matrix.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_sample_path(recorded_path):
    normalized = str(recorded_path).replace("\\", "/")
    marker = "/val/"
    if marker not in normalized:
        raise ValueError("sample path does not contain /val/: {}".format(recorded_path))
    return normalized.split(marker, 1)[1]


def load_sample_records(list_file, data_root, inspect_images=True):
    list_path = Path(list_file).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    recorded = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    aggregate = hashlib.sha256()
    seen = set()
    class_counts = {}
    image_reader = None
    if inspect_images:
        try:
            from PIL import Image

            image_reader = Image
        except ImportError:
            image_reader = None
    for index, source_path in enumerate(recorded):
        relative_path = relative_sample_path(source_path)
        actual_path = root / relative_path
        class_name = relative_path.split("/", 1)[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        duplicate = relative_path in seen
        seen.add(relative_path)
        record = {
            "index": index,
            "recorded_path": source_path,
            "relative_path": relative_path,
            "actual_path": str(actual_path),
            "class": class_name,
            "duplicate": duplicate,
            "exists": actual_path.is_file(),
        }
        if actual_path.is_file():
            record["size_bytes"] = actual_path.stat().st_size
            record["sha256"] = sha256_file(actual_path)
            aggregate.update(
                "{}\0{}\0{}\n".format(relative_path, record["sha256"], record["size_bytes"]).encode("utf-8")
            )
            if image_reader is not None:
                with image_reader.open(actual_path) as image:
                    record["image_mode"] = image.mode
                    record["image_size"] = list(image.size)
        records.append(record)
    return {
        "list_file": str(list_path),
        "list_file_sha256": sha256_file(list_path),
        "data_root": str(root),
        "sample_count": len(records),
        "class_counts": class_counts,
        "missing_count": sum(not item["exists"] for item in records),
        "duplicate_count": sum(item["duplicate"] for item in records),
        "dataset_content_manifest_sha256": aggregate.hexdigest(),
        "image_metadata_available": image_reader is not None,
        "records": records,
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return "unavailable: {}".format(exc)


def collect_environment(runtime_config):
    config_path = Path(runtime_config).expanduser().resolve() if runtime_config else None
    config_payload = read_json(config_path) if config_path and config_path.is_file() else None
    memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal"}:
                memory[key] = value.strip()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python": platform.python_version(),
        "packages": {
            "torch": package_version("torch"),
            "jax": package_version("jax"),
            "spu": package_version("spu"),
            "numpy": package_version("numpy"),
            "Pillow": package_version("Pillow"),
        },
        "cpu": command_output(["lscpu"]),
        "memory": memory,
        "runtime_config": {
            "path": str(config_path) if config_path else None,
            "exists": bool(config_path and config_path.is_file()),
            "sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
            "payload": config_payload,
        },
    }


def check_equal(checks, name, actual, expected):
    passed = actual == expected
    checks.append({"name": name, "actual": actual, "expected": expected, "passed": passed})
    return passed


def command_inventory(args):
    repo_root = Path(args.repo_root).expanduser().resolve()
    matrix = read_json(args.matrix)
    datasets = matrix["datasets"]
    roots = {
        "medical_full_validation": args.medical_data_root,
        "medical_secure_deployment_batch": args.medical_data_root,
        "finance_boundary_stress": args.finance_data_root,
    }
    output_dir = Path(args.materialized_list_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {}
    checks = []
    for key, data_root in roots.items():
        spec = datasets[key]
        list_file = repo_root / spec["list_file"]
        detail = load_sample_records(list_file, data_root, inspect_images=not args.skip_image_metadata)
        inventory[key] = detail
        check_equal(checks, key + ".sample_count", detail["sample_count"], int(spec["sample_count"]))
        check_equal(checks, key + ".class_counts", detail["class_counts"], spec["class_counts"])
        check_equal(checks, key + ".missing_count", detail["missing_count"], 0)
        check_equal(checks, key + ".duplicate_count", detail["duplicate_count"], 0)
        if spec.get("list_file_sha256"):
            check_equal(checks, key + ".list_file_sha256", detail["list_file_sha256"], spec["list_file_sha256"])
        materialized = output_dir / (key + ".txt")
        materialized.write_text(
            "\n".join(item["actual_path"] for item in detail["records"]) + "\n",
            encoding="utf-8",
        )
        detail["materialized_list_file"] = str(materialized)
        detail["materialized_list_file_sha256"] = sha256_file(materialized)

    bundles = {}
    for key, spec in matrix["bundles"].items():
        weight_path = repo_root / spec["weight"]
        actual_hash = sha256_file(weight_path) if weight_path.is_file() else None
        bundles[key] = {
            "path": str(weight_path),
            "exists": weight_path.is_file(),
            "size_bytes": weight_path.stat().st_size if weight_path.is_file() else None,
            "sha256": actual_hash,
            "expected_sha256": spec["weight_sha256"],
        }
        check_equal(checks, "bundle." + key + ".sha256", actual_hash, spec["weight_sha256"])

    payload = {
        "manifest_type": "transshield_report_vps_inventory_v1",
        "matrix": str(Path(args.matrix).expanduser().resolve()),
        "matrix_sha256": sha256_file(args.matrix),
        "report": matrix["report"],
        "environment": collect_environment(args.runtime_config),
        "preprocessing": matrix["preprocessing"],
        "datasets": inventory,
        "bundles": bundles,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    write_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "passed": payload["passed"], "checks": checks}, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


def read_counter(interface, name):
    path = Path("/sys/class/net") / interface / "statistics" / name
    return int(path.read_text(encoding="utf-8").strip())


def process_snapshot(pid):
    if not pid:
        return None
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.is_file():
        return {"pid": pid, "available": False}
    wanted = {"VmRSS", "VmHWM", "Threads"}
    fields = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in wanted:
            fields[key] = value.strip()
    fd_root = Path("/proc") / str(pid) / "fd"
    fds = list(fd_root.iterdir()) if fd_root.is_dir() else []
    socket_fds = 0
    for fd in fds:
        try:
            if os.readlink(str(fd)).startswith("socket:"):
                socket_fds += 1
        except OSError:
            pass
    return {
        "pid": pid,
        "available": True,
        "fd_count": len(fds),
        "socket_fd_count": socket_fds,
        "status": fields,
    }


def command_network_snapshot(args):
    payload = {
        "manifest_type": "transshield_vps_network_snapshot_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "interface": args.interface,
        "tx_bytes": read_counter(args.interface, "tx_bytes"),
        "rx_bytes": read_counter(args.interface, "rx_bytes"),
        "process": process_snapshot(args.pid),
    }
    write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def infer_target(relative_path):
    class_name = relative_path.split("/", 1)[0].lower()
    if class_name in {"0", "fraud"}:
        return 0
    if class_name in {"1", "normal"}:
        return 1
    raise ValueError("unsupported class directory: {}".format(class_name))


def binary_auc(scores, targets):
    positives = [float(score) for score, target in zip(scores, targets) if int(target) == 1]
    negatives = [float(score) for score, target in zip(scores, targets) if int(target) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def mean(values):
    return sum(values) / len(values) if values else None


def max_or_none(values):
    return max(values) if values else None


def load_inference_payload(path):
    import torch

    return torch.load(Path(path).expanduser().resolve(), map_location="cpu")


def tensor_rows(tensor):
    if tensor is None:
        return None
    return [[float(value) for value in row] for row in tensor.detach().cpu().float().tolist()]


def scalar_predictions(payload, threshold):
    import torch

    logits = payload["logits"].detach().cpu().float()
    probabilities = payload.get("probabilities")
    if probabilities is None:
        probabilities = torch.softmax(logits, dim=-1)
    else:
        probabilities = probabilities.detach().cpu().float()
    argmax_predictions = logits.argmax(dim=1)
    threshold_predictions = (probabilities[:, 1] >= float(threshold)).long()
    return logits, probabilities, argmax_predictions, threshold_predictions


def command_summarize(args):
    import torch

    matrix = read_json(args.matrix)
    spec = matrix["datasets"][args.dataset_key]
    candidate_path = Path(args.candidate_pt).expanduser().resolve()
    candidate = load_inference_payload(candidate_path)
    threshold = float(args.threshold if args.threshold is not None else candidate.get("threshold"))
    logits, probabilities, argmax_predictions, threshold_predictions = scalar_predictions(candidate, threshold)

    list_file = Path(args.sample_list).expanduser().resolve()
    relative_paths = [relative_sample_path(line.strip()) for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = [infer_target(path) for path in relative_paths]
    if len(targets) != int(logits.shape[0]):
        raise ValueError("sample list count {} != candidate count {}".format(len(targets), logits.shape[0]))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    argmax_accuracy = float((argmax_predictions == target_tensor).float().mean().item())
    threshold_accuracy = float((threshold_predictions == target_tensor).float().mean().item())
    auc = binary_auc(probabilities[:, 1].tolist(), targets)

    reference = None
    reference_metrics = None
    per_sample_reference = [None] * len(targets)
    if args.reference_pt:
        reference = load_inference_payload(args.reference_pt)
        ref_logits, ref_probabilities, ref_argmax, ref_threshold = scalar_predictions(reference, threshold)
        if tuple(ref_logits.shape) != tuple(logits.shape):
            raise ValueError("reference shape {} != candidate shape {}".format(tuple(ref_logits.shape), tuple(logits.shape)))
        logit_abs = (logits - ref_logits).abs()
        probability_abs = (probabilities - ref_probabilities).abs()
        reference_metrics = {
            "argmax_match_ratio": float((argmax_predictions == ref_argmax).float().mean().item()),
            "threshold_match_ratio": float((threshold_predictions == ref_threshold).float().mean().item()),
            "logits_max_abs_error": float(logit_abs.max().item()),
            "logits_mean_abs_error": float(logit_abs.mean().item()),
            "probabilities_max_abs_error": float(probability_abs.max().item()),
            "probabilities_mean_abs_error": float(probability_abs.mean().item()),
            "reference_argmax_accuracy": float((ref_argmax == target_tensor).float().mean().item()),
            "reference_threshold_accuracy": float((ref_threshold == target_tensor).float().mean().item()),
            "reference_auc": binary_auc(ref_probabilities[:, 1].tolist(), targets),
        }
        for index in range(len(targets)):
            per_sample_reference[index] = {
                "logits": [float(value) for value in ref_logits[index].tolist()],
                "probabilities": [float(value) for value in ref_probabilities[index].tolist()],
                "argmax_prediction": int(ref_argmax[index].item()),
                "threshold_prediction": int(ref_threshold[index].item()),
                "logits_max_abs_error": float(logit_abs[index].max().item()),
                "probabilities_max_abs_error": float(probability_abs[index].max().item()),
            }

    candidate_json = read_json(args.candidate_json) if args.candidate_json else {}
    network = None
    if args.network_before and args.network_after:
        before = read_json(args.network_before)
        after = read_json(args.network_after)
        tx_delta = int(after["tx_bytes"]) - int(before["tx_bytes"])
        rx_delta = int(after["rx_bytes"]) - int(before["rx_bytes"])
        network = {
            "interface": before["interface"],
            "before": before,
            "after": after,
            "tx_delta_bytes": tx_delta,
            "rx_delta_bytes": rx_delta,
            "loopback_total_bytes_rule": "use tx_delta once; loopback tx and rx mirror the same transferred bytes",
            "total_bytes": tx_delta,
            "total_gib": tx_delta / (1024.0 ** 3),
            "per_sample_gib": tx_delta / len(targets) / (1024.0 ** 3),
        }

    elapsed_sec = candidate_json.get("elapsed_sec")
    if elapsed_sec is None:
        elapsed_sec = candidate.get("elapsed_sec")
    result = {
        "manifest_type": "transshield_report_vps_inference_summary_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": args.dataset_key,
        "report_scope": spec,
        "candidate_pt": str(candidate_path),
        "candidate_pt_sha256": sha256_file(candidate_path),
        "candidate_json": str(Path(args.candidate_json).resolve()) if args.candidate_json else None,
        "reference_pt": str(Path(args.reference_pt).resolve()) if args.reference_pt else None,
        "sample_list": str(list_file),
        "sample_list_sha256": sha256_file(list_file),
        "sample_count": len(targets),
        "class_counts": {str(value): targets.count(value) for value in sorted(set(targets))},
        "threshold": threshold,
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "finite_probabilities": bool(torch.isfinite(probabilities).all().item()),
        "argmax_accuracy": argmax_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "auc": auc,
        "elapsed_sec": elapsed_sec,
        "sec_per_sample": float(elapsed_sec) / len(targets) if elapsed_sec is not None else None,
        "prediction_counts": {
            "argmax": {str(value): int((argmax_predictions == value).sum().item()) for value in [0, 1]},
            "threshold": {str(value): int((threshold_predictions == value).sum().item()) for value in [0, 1]},
        },
        "class1_probability": {
            "min": float(probabilities[:, 1].min().item()),
            "max": float(probabilities[:, 1].max().item()),
            "mean": float(probabilities[:, 1].mean().item()),
        },
        "reference_comparison": reference_metrics,
        "network": network,
        "runtime_metadata": candidate_json,
        "privacy_facts": {
            "runner_host_model_params_materialized": candidate_json.get("host_model_params_materialized"),
            "host_plaintext_pixel_values_materialized": candidate_json.get("host_plaintext_pixel_values_materialized"),
            "host_private_share_tensors_loaded": candidate_json.get("host_private_share_tensors_loaded"),
            "model_params_secret_inside_spu": candidate_json.get("spu", {}).get("model_params_secret_inside_spu"),
            "predictor_params_secret_inside_spu": candidate_json.get("spu", {}).get("predictor_params_secret_inside_spu"),
            "reveal_policy": candidate_json.get("reveal_policy"),
            "input_mode": candidate_json.get("input_mode"),
        },
        "per_sample": [],
    }
    for index, relative_path in enumerate(relative_paths):
        result["per_sample"].append(
            {
                "index": index,
                "relative_path": relative_path,
                "target": targets[index],
                "logits": [float(value) for value in logits[index].tolist()],
                "probabilities": [float(value) for value in probabilities[index].tolist()],
                "argmax_prediction": int(argmax_predictions[index].item()),
                "threshold_prediction": int(threshold_predictions[index].item()),
                "argmax_correct": bool(argmax_predictions[index].item() == targets[index]),
                "threshold_correct": bool(threshold_predictions[index].item() == targets[index]),
                "reference": per_sample_reference[index],
            }
        )
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "sample_count": result["sample_count"],
                "argmax_accuracy": argmax_accuracy,
                "threshold_accuracy": threshold_accuracy,
                "auc": auc,
                "sec_per_sample": result["sec_per_sample"],
                "reference_comparison": reference_metrics,
                "network": network,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_compare_preprocessed(args):
    import torch

    current = load_inference_payload(args.current_pt)
    frozen = load_inference_payload(args.frozen_pt)
    current_tensor = current["pixel_values"].detach().cpu().float()
    frozen_tensor = frozen["pixel_values"].detach().cpu().float()
    if tuple(current_tensor.shape) != tuple(frozen_tensor.shape):
        raise ValueError("pixel tensor shape mismatch")
    absolute = (current_tensor - frozen_tensor).abs()
    payload = {
        "manifest_type": "transshield_report_vps_preprocessing_compare_v1",
        "current_pt": str(Path(args.current_pt).resolve()),
        "current_pt_sha256": sha256_file(args.current_pt),
        "frozen_pt": str(Path(args.frozen_pt).resolve()),
        "frozen_pt_sha256": sha256_file(args.frozen_pt),
        "shape": list(current_tensor.shape),
        "dtype": str(current["pixel_values"].dtype),
        "exact_tensor_equal": bool(torch.equal(current_tensor, frozen_tensor)),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "targets_equal": bool(torch.equal(current.get("targets"), frozen.get("targets"))),
        "sample_ids_equal": list(current.get("sample_ids") or []) == list(frozen.get("sample_ids") or []),
    }
    payload["passed"] = payload["exact_tensor_equal"] and payload["targets_equal"] and payload["sample_ids_equal"]
    write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


def build_parser():
    parser = argparse.ArgumentParser(description="VPS-only report-scope evidence helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="verify exact report data and bundle assets")
    inventory.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    inventory.add_argument("--repo-root", default=str(REPO_ROOT))
    inventory.add_argument("--medical-data-root", required=True)
    inventory.add_argument("--finance-data-root", required=True)
    inventory.add_argument("--runtime-config", default="")
    inventory.add_argument("--materialized-list-dir", required=True)
    inventory.add_argument("--skip-image-metadata", action="store_true")
    inventory.add_argument("--out", required=True)
    inventory.set_defaults(func=command_inventory)

    snapshot = subparsers.add_parser("network-snapshot", help="capture VPS interface and optional process counters")
    snapshot.add_argument("--interface", default="lo")
    snapshot.add_argument("--pid", type=int, default=0)
    snapshot.add_argument("--out", required=True)
    snapshot.set_defaults(func=command_network_snapshot)

    summarize = subparsers.add_parser("summarize", help="summarize one VPS inference run with per-sample evidence")
    summarize.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    summarize.add_argument("--dataset-key", required=True, choices=[
        "medical_full_validation",
        "medical_secure_deployment_batch",
        "finance_boundary_stress",
    ])
    summarize.add_argument("--candidate-pt", required=True)
    summarize.add_argument("--candidate-json", default="")
    summarize.add_argument("--reference-pt", default="")
    summarize.add_argument("--sample-list", required=True)
    summarize.add_argument("--threshold", type=float, default=None)
    summarize.add_argument("--network-before", default="")
    summarize.add_argument("--network-after", default="")
    summarize.add_argument("--out", required=True)
    summarize.set_defaults(func=command_summarize)

    compare = subparsers.add_parser("compare-preprocessed", help="compare regenerated VPS tensors to frozen inputs")
    compare.add_argument("--current-pt", required=True)
    compare.add_argument("--frozen-pt", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=command_compare_preprocessed)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
