#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_run_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--run label must be non-empty")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "e2e_secure_poc" / "e2e_approx_eval_metrics.json"
    if not path.exists():
        raise argparse.ArgumentTypeError(f"metrics JSON does not exist: {path}")
    return label, path.resolve()


def privacy_ok(payload) -> bool:
    fields = payload.get("privacy_fields") or {}
    return (
        fields.get("input_mode") == "party_local_debug_share_load"
        and fields.get("host_plaintext_pixel_values_materialized") is False
        and fields.get("host_private_share_tensors_loaded") is False
        and fields.get("private_input_paths_redacted") is True
    )


def summarize(label: str, path: Path):
    payload = load_json(path)
    sample_count = payload.get("sample_count")
    elapsed = payload.get("e2e_elapsed_sec")
    sec_per_sample = None
    if isinstance(sample_count, (int, float)) and sample_count and isinstance(elapsed, (int, float)):
        sec_per_sample = float(elapsed) / float(sample_count)
    communication = payload.get("e2e_communication_from_spu_node_logs") or {}
    return {
        "label": label,
        "metrics_json": str(path),
        "sample_count": sample_count,
        "finite_logits": payload.get("finite_logits"),
        "e2e_argmax_accuracy": payload.get("e2e_argmax_accuracy"),
        "e2e_threshold_accuracy": payload.get("e2e_threshold_accuracy"),
        "original_plaintext_same_subset_argmax_accuracy": payload.get("original_plaintext_same_subset_argmax_accuracy"),
        "original_plaintext_same_subset_threshold_accuracy": payload.get("original_plaintext_same_subset_threshold_accuracy"),
        "static_whole_forward_same_subset_argmax_accuracy": payload.get("static_whole_forward_same_subset_argmax_accuracy"),
        "static_whole_forward_same_subset_threshold_accuracy": payload.get("static_whole_forward_same_subset_threshold_accuracy"),
        "prediction_match_vs_original_plaintext": payload.get("prediction_match_vs_original_plaintext"),
        "prediction_match_vs_static_whole_forward": payload.get("prediction_match_vs_static_whole_forward"),
        "raw_secure_graph_before_output_calibration": payload.get("raw_secure_graph_before_output_calibration"),
        "e2e_elapsed_sec": elapsed,
        "e2e_sec_per_sample": sec_per_sample,
        "privacy_fields": payload.get("privacy_fields"),
        "privacy_ok": privacy_ok(payload),
        "aggregate_total_bytes": communication.get("aggregate_total_bytes"),
    }


def add_speedups(items):
    isolated_by_count = {}
    for item in items:
        label = item["label"].lower()
        if "isolated" in label and "nonisolated" not in label and item.get("sample_count") is not None:
            isolated_by_count[item["sample_count"]] = item
    for item in items:
        baseline = isolated_by_count.get(item.get("sample_count"))
        speedup = None
        if (
            baseline
            and baseline["label"] != item["label"]
            and item.get("e2e_elapsed_sec")
            and baseline.get("e2e_elapsed_sec")
        ):
            speedup = float(baseline["e2e_elapsed_sec"]) / float(item["e2e_elapsed_sec"])
        item["speedup_vs_same_sample_isolated"] = speedup
        item["speedup_baseline_label"] = None if speedup is None else baseline["label"]


def write_markdown(path: Path, report) -> None:
    lines = [
        "# E2E Runtime Efficiency Report",
        "",
        f"- Label: `{report['label']}`",
        f"- Best speedup: `{report['best_speedup_vs_same_sample_isolated']}`",
        "",
        "| Run | Samples | Finite | Threshold Acc | Elapsed (s) | Sec/sample | Speedup vs isolated | Privacy OK |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in report["runs"]:
        lines.append(
            "| {label} | {sample_count} | {finite_logits} | {acc} | {elapsed} | {sps} | {speedup} | {privacy} |".format(
                label=item["label"],
                sample_count=item.get("sample_count"),
                finite_logits=item.get("finite_logits"),
                acc=fmt(item.get("e2e_threshold_accuracy")),
                elapsed=fmt(item.get("e2e_elapsed_sec")),
                sps=fmt(item.get("e2e_sec_per_sample")),
                speedup=fmt(item.get("speedup_vs_same_sample_isolated")),
                privacy=item.get("privacy_ok"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `speedup_vs_same_sample_isolated` is computed only when an isolated run with the same sample count is present.",
            "- If present, `static_whole_forward_same_subset_*` is the primary comparator for the current secure-static whole-forward path.",
            "- `original_plaintext_*` is retained only as context against the full bundle forward.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E2E runtime metrics across isolated/non-isolated runs.")
    parser.add_argument("--run", action="append", type=parse_run_arg, required=True, help="label=metrics_json_or_run_dir")
    parser.add_argument("--label", default="e2e_runtime_efficiency")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    items = [summarize(label, path) for label, path in args.run]
    add_speedups(items)
    speedups = [
        float(item["speedup_vs_same_sample_isolated"])
        for item in items
        if item.get("speedup_vs_same_sample_isolated") is not None
    ]
    report = {
        "manifest_type": "transshield_e2e_runtime_efficiency_report_v0",
        "label": args.label,
        "run_count": len(items),
        "runs": items,
        "best_speedup_vs_same_sample_isolated": max(speedups) if speedups else None,
        "all_finite_logits": all(item.get("finite_logits") is True for item in items),
        "all_privacy_ok": all(item.get("privacy_ok") is True for item in items),
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)


if __name__ == "__main__":
    main()
