#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def maybe_existing_path(value):
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.exists() else path


def resolve_run_artifact(run_dir: Path, value, fallback_name: str = ""):
    if value:
        raw_path = Path(str(value)).expanduser()
        if raw_path.exists():
            return raw_path.resolve()
        candidate = run_dir / raw_path.name
        if candidate.exists():
            return candidate.resolve()
    if fallback_name:
        fallback = run_dir / fallback_name
        if fallback.exists():
            return fallback.resolve()
    return maybe_existing_path(value)


def resolve_run_arg(raw_value: str):
    if "=" in raw_value:
        label, raw_path = raw_value.split("=", 1)
        label = label.strip()
    else:
        raw_path = raw_value
        label = ""
    path = Path(raw_path).expanduser()
    if path.is_file():
        if path.name != "e2e_approx_eval_metrics.json":
            raise argparse.ArgumentTypeError(f"expected run dir or e2e_approx_eval_metrics.json: {path}")
        run_dir = path.parent
        metrics_json = path
    elif path.is_dir():
        direct_metrics = path / "e2e_approx_eval_metrics.json"
        nested_metrics = path / "e2e_secure_poc" / "e2e_approx_eval_metrics.json"
        if direct_metrics.exists():
            run_dir = path
            metrics_json = direct_metrics
        elif nested_metrics.exists():
            run_dir = path / "e2e_secure_poc"
            metrics_json = nested_metrics
        else:
            raise argparse.ArgumentTypeError(f"missing e2e_approx_eval_metrics.json under {path}")
    else:
        raise argparse.ArgumentTypeError(f"missing path: {path}")
    if not label:
        label = run_dir.parent.name if run_dir.name == "e2e_secure_poc" else run_dir.name
    return {
        "label": label,
        "run_dir": run_dir.resolve(),
        "metrics_json": metrics_json.resolve(),
    }


def parse_run_arg(raw_value: str):
    return resolve_run_arg(raw_value)


def float_or_none(value):
    if value is None:
        return None
    return float(value)


def guess_profile(calibration_json: Path):
    if calibration_json is None:
        return "unknown"
    name = calibration_json.name.lower()
    if "spuaware" in name:
        return "accuracy_first"
    if "temperature" in name:
        return "loss_first_temperature"
    if "affine" in name:
        return "loss_first_affine"
    if "bridge" in name:
        return "bridge_best"
    if name == "e2e_static_output_calibration_public_logit_bias.json":
        return "static_bias"
    return calibration_json.stem


def binary_cross_entropy_from_scores(scores, targets):
    total = 0.0
    for score, target in zip(scores, targets):
        if score >= 0.0:
            total += math.log1p(math.exp(-score)) + (1 - target) * score
        else:
            total += math.log1p(math.exp(score)) - target * score
    return float(total / len(targets))


def load_candidate_loss_block(candidate_pt: Path, reference_json: Path):
    if candidate_pt is None or reference_json is None:
        return None
    if not candidate_pt.exists() or not reference_json.exists():
        return None
    import torch

    payload = torch.load(candidate_pt, map_location="cpu")
    rows = load_json(reference_json).get("per_sample") or []
    if not rows:
        return None
    targets = [int(row["target"]) for row in rows]
    logits = payload.get("logits")
    if logits is None:
        return None
    logits = logits.detach().cpu().float()
    count = min(int(logits.shape[0]), len(targets))
    logits = logits[:count]
    targets = targets[:count]
    calibrated_scores = (logits[:, 1] - logits[:, 0]).tolist()
    result = {
        "calibrated_bce": binary_cross_entropy_from_scores(calibrated_scores, targets),
        "score_count": count,
    }
    raw_logits = payload.get("raw_logits_before_output_calibration")
    if raw_logits is not None:
        raw_logits = raw_logits.detach().cpu().float()[:count]
        raw_scores = (raw_logits[:, 1] - raw_logits[:, 0]).tolist()
        result["raw_bce"] = binary_cross_entropy_from_scores(raw_scores, targets)
        result["bce_delta_calibrated_minus_raw"] = result["calibrated_bce"] - result["raw_bce"]
    return result


def sample_signature(reference_json: Path):
    if reference_json is None or not reference_json.exists():
        return None
    rows = load_json(reference_json).get("per_sample") or []
    if not rows:
        return None
    sample_ids = [str(row.get("sample_id")) for row in rows]
    targets = [int(row.get("target")) for row in rows]
    digest = hashlib.sha256("\n".join(sample_ids + ["--targets--"] + [str(item) for item in targets]).encode("utf-8")).hexdigest()
    return {
        "sample_count": len(sample_ids),
        "sample_ids_sha256": digest,
    }


def summarize_run(run_item):
    label = run_item["label"]
    run_dir = run_item["run_dir"]
    metrics_json = run_item["metrics_json"]
    metrics = load_json(metrics_json)
    gap_json = run_dir / "plaintext_static_gap.json"
    gap = load_json(gap_json) if gap_json.exists() else None
    candidate_pt = resolve_run_artifact(run_dir, metrics.get("candidate_pt"))
    reference_json = resolve_run_artifact(run_dir, metrics.get("plaintext_reference_json"), "plaintext_same_images_reference.json")
    calibration_json = maybe_existing_path((metrics.get("output_calibration") or {}).get("calibration_json"))
    raw_block = metrics.get("raw_secure_graph_before_output_calibration") or {}
    comm = metrics.get("e2e_communication_from_spu_node_logs") or {}
    losses = load_candidate_loss_block(candidate_pt, reference_json)
    signature = sample_signature(reference_json)
    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "metrics_json": str(metrics_json),
        "sample_signature": signature,
        "sample_count": metrics.get("sample_count"),
        "finite_logits": metrics.get("finite_logits"),
        "profile_guess": guess_profile(calibration_json),
        "output_calibration_json": str(calibration_json) if calibration_json is not None else None,
        "output_calibration_note": (metrics.get("output_calibration") or {}).get("note"),
        "e2e_argmax_accuracy": float_or_none(metrics.get("e2e_argmax_accuracy")),
        "e2e_threshold_accuracy": float_or_none(metrics.get("e2e_threshold_accuracy")),
        "e2e_elapsed_sec": float_or_none(metrics.get("e2e_elapsed_sec")),
        "aggregate_total_bytes": (comm or {}).get("aggregate_total_bytes"),
        "static_whole_forward_same_subset_argmax_accuracy": float_or_none(metrics.get("static_whole_forward_same_subset_argmax_accuracy")),
        "static_whole_forward_same_subset_threshold_accuracy": float_or_none(metrics.get("static_whole_forward_same_subset_threshold_accuracy")),
        "original_plaintext_same_subset_argmax_accuracy": float_or_none(metrics.get("original_plaintext_same_subset_argmax_accuracy")),
        "original_plaintext_same_subset_threshold_accuracy": float_or_none(metrics.get("original_plaintext_same_subset_threshold_accuracy")),
        "raw_secure_graph_before_output_calibration": {
            "present": bool(raw_block.get("present")),
            "same_subset_argmax_accuracy": float_or_none(raw_block.get("same_subset_argmax_accuracy")),
            "same_subset_threshold_accuracy": float_or_none(raw_block.get("same_subset_threshold_accuracy")),
            "logits_mean_abs_error_vs_static": float_or_none(((raw_block.get("logits_error_vs_static_whole_forward") or {}).get("mean_abs_error"))),
            "probabilities_mean_abs_error_vs_static": float_or_none(((raw_block.get("probabilities_error_vs_static_whole_forward") or {}).get("mean_abs_error"))),
        },
        "losses": losses,
        "plaintext_static_gap": None,
    }
    if gap is not None:
        judgement = gap.get("judgement") or {}
        compare = gap.get("compare") or {}
        summary["plaintext_static_gap"] = {
            "status": judgement.get("status"),
            "reason": judgement.get("reason"),
            "score_correlation": float_or_none(judgement.get("score_correlation")),
            "same_sign_ratio": float_or_none(judgement.get("same_sign_ratio")),
            "affine_boundary_shift_x_at_y0": float_or_none(judgement.get("affine_boundary_shift_x_at_y0")),
            "plaintext_best_threshold_accuracy": float_or_none(judgement.get("plaintext_best_threshold_accuracy")),
            "plaintext_to_static_best_match_threshold": float_or_none(judgement.get("plaintext_to_static_best_match_threshold")),
            "metric_delta_static_minus_plaintext": compare.get("metric_delta_static_minus_plaintext"),
        }
    return summary


def numeric_delta(value, baseline):
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def build_deltas(runs, baseline_label):
    baseline = next((item for item in runs if item["label"] == baseline_label), None)
    if baseline is None:
        raise SystemExit(f"baseline label not found: {baseline_label}")
    fields = [
        "e2e_argmax_accuracy",
        "e2e_threshold_accuracy",
        "e2e_elapsed_sec",
        "aggregate_total_bytes",
        "static_whole_forward_same_subset_argmax_accuracy",
        "static_whole_forward_same_subset_threshold_accuracy",
        "original_plaintext_same_subset_argmax_accuracy",
        "original_plaintext_same_subset_threshold_accuracy",
    ]
    deltas = {}
    for item in runs:
        row = {}
        for field in fields:
            row[field] = numeric_delta(item.get(field), baseline.get(field))
        row["raw_logits_mean_abs_error_vs_static"] = numeric_delta(
            (item.get("raw_secure_graph_before_output_calibration") or {}).get("logits_mean_abs_error_vs_static"),
            (baseline.get("raw_secure_graph_before_output_calibration") or {}).get("logits_mean_abs_error_vs_static"),
        )
        row["calibrated_bce"] = numeric_delta(
            ((item.get("losses") or {}).get("calibrated_bce")),
            ((baseline.get("losses") or {}).get("calibrated_bce")),
        )
        deltas[item["label"]] = row
    return deltas


def choose_best(runs, metric, higher_is_better=True):
    candidates = [item for item in runs if item.get(metric) is not None]
    if not candidates:
        return None

    def key(item):
        value = float(item[metric])
        primary = -value if higher_is_better else value
        return (
            primary,
            -(item.get("e2e_argmax_accuracy") or float("-inf")),
            item.get("e2e_elapsed_sec") if item.get("e2e_elapsed_sec") is not None else float("inf"),
            item["label"],
        )

    winner = sorted(candidates, key=key)[0]
    return {"label": winner["label"], "metric": metric, "value": winner[metric]}


def build_report(runs, baseline_label):
    signatures = [((item.get("sample_signature") or {}).get("sample_ids_sha256")) for item in runs if item.get("sample_signature")]
    sample_counts = [item.get("sample_count") for item in runs if item.get("sample_count") is not None]
    report = {
        "manifest_type": "transshield_e2e_output_profile_compare_v0",
        "run_count": len(runs),
        "baseline_label": baseline_label,
        "all_runs_share_same_sample_signature": bool(signatures) and len(set(signatures)) == 1,
        "all_runs_share_same_sample_count": bool(sample_counts) and len(set(sample_counts)) == 1,
        "best_by_metric": {
            "threshold_accuracy": choose_best(runs, "e2e_threshold_accuracy", higher_is_better=True),
            "argmax_accuracy": choose_best(runs, "e2e_argmax_accuracy", higher_is_better=True),
            "fastest": choose_best(runs, "e2e_elapsed_sec", higher_is_better=False),
            "lowest_total_bytes": choose_best(runs, "aggregate_total_bytes", higher_is_better=False),
            "lowest_calibrated_bce": choose_best(
                [{**item, "calibrated_bce": ((item.get("losses") or {}).get("calibrated_bce"))} for item in runs],
                "calibrated_bce",
                higher_is_better=False,
            ),
            "lowest_raw_logits_mae_vs_static": choose_best(
                [
                    {
                        **item,
                        "raw_logits_mean_abs_error_vs_static": (
                            (item.get("raw_secure_graph_before_output_calibration") or {}).get("logits_mean_abs_error_vs_static")
                        ),
                    }
                    for item in runs
                ],
                "raw_logits_mean_abs_error_vs_static",
                higher_is_better=False,
            ),
        },
        "runs": runs,
    }
    if baseline_label:
        report["delta_vs_baseline"] = build_deltas(runs, baseline_label)
    return report


def render_value(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown(report):
    lines = [
        "# E2E Output Profile Compare",
        "",
        f"- run_count: `{report['run_count']}`",
        f"- same_sample_signature: `{str(report['all_runs_share_same_sample_signature']).lower()}`",
        f"- same_sample_count: `{str(report['all_runs_share_same_sample_count']).lower()}`",
    ]
    if report.get("baseline_label"):
        lines.append(f"- baseline_label: `{report['baseline_label']}`")
    best = report.get("best_by_metric") or {}
    for name in (
        "threshold_accuracy",
        "argmax_accuracy",
        "lowest_calibrated_bce",
        "fastest",
        "lowest_total_bytes",
        "lowest_raw_logits_mae_vs_static",
    ):
        item = best.get(name)
        if item is not None:
            lines.append(f"- best_{name}: `{item['label']}` ({render_value(item['value'])})")
    lines.extend(
        [
            "",
            "| label | profile | th_acc | argmax_acc | calibrated_bce | raw_th_acc | raw_argmax_acc | raw_logits_mae | elapsed_sec | total_bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["runs"]:
        raw = item.get("raw_secure_graph_before_output_calibration") or {}
        losses = item.get("losses") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    item["label"],
                    item.get("profile_guess") or "-",
                    render_value(item.get("e2e_threshold_accuracy")),
                    render_value(item.get("e2e_argmax_accuracy")),
                    render_value(losses.get("calibrated_bce")),
                    render_value(raw.get("same_subset_threshold_accuracy")),
                    render_value(raw.get("same_subset_argmax_accuracy")),
                    render_value(raw.get("logits_mean_abs_error_vs_static")),
                    render_value(item.get("e2e_elapsed_sec")),
                    render_value(item.get("aggregate_total_bytes"), digits=0),
                ]
            )
            + " |"
        )
    if report.get("delta_vs_baseline"):
        lines.extend(["", "## Delta Vs Baseline", "", f"Baseline: `{report['baseline_label']}`", ""])
        lines.append("| label | d_th_acc | d_argmax_acc | d_calibrated_bce | d_elapsed_sec | d_total_bytes | d_raw_logits_mae |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, row in report["delta_vs_baseline"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        render_value(row.get("e2e_threshold_accuracy")),
                        render_value(row.get("e2e_argmax_accuracy")),
                        render_value(row.get("calibrated_bce")),
                        render_value(row.get("e2e_elapsed_sec")),
                        render_value(row.get("aggregate_total_bytes"), digits=0),
                        render_value(row.get("raw_logits_mean_abs_error_vs_static")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Compare completed real E2E runs across output-calibration profiles.")
    parser.add_argument("--run", action="append", required=True, type=parse_run_arg, help="label=run_dir_or_metrics_json")
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main():
    args = build_parser().parse_args()
    runs = [summarize_run(item) for item in args.run]
    report = build_report(runs, args.baseline_label.strip() or None)
    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)
    if args.output_md:
        write_text(Path(args.output_md).expanduser().resolve(), render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
