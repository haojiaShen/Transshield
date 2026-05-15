import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt_float(value, digits=4):
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def bool_str(value):
    if value is None:
        return "null"
    return str(bool(value)).lower()


def display_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return bool_str(value)
    return value


def run_label(summary: dict):
    run_name = Path(summary["run_dir"]).name
    prefix = "keepmask_wholeforward_wrapper_spu_"
    if prefix in run_name:
        return run_name.split(prefix, 1)[1]
    return run_name


def normalize_run(summary: dict):
    compare = summary.get("compare_metrics") or {}
    privacy = summary.get("privacy_fields") or {}
    keepmask = summary.get("keepmask_fields") or {}
    return {
        "run_label": run_label(summary),
        "run_dir": summary.get("run_dir"),
        "sample_count": summary.get("sample_count"),
        "elapsed_sec": summary.get("elapsed_sec"),
        "sec_per_sample": summary.get("sec_per_sample"),
        "finite_logits": summary.get("finite_logits"),
        "argmax_match_ratio": compare.get("argmax_match_ratio"),
        "threshold_match_ratio": compare.get("threshold_match_ratio"),
        "logits_max_abs_error": compare.get("logits_max_abs_error"),
        "logits_mean_abs_error": compare.get("logits_mean_abs_error"),
        "probabilities_max_abs_error": compare.get("probabilities_max_abs_error"),
        "probabilities_mean_abs_error": compare.get("probabilities_mean_abs_error"),
        "privacy_fields": privacy,
        "keepmask_fields": keepmask,
    }


def compare_key(item):
    sample_count = item.get("sample_count")
    if sample_count is None:
        return (1, 0, item["run_label"])
    return (0, int(sample_count), item["run_label"])


def build_privacy_comparison(runs):
    reference = dict(runs[0].get("privacy_fields") or {})
    keepmask_ref = dict(runs[0].get("keepmask_fields") or {})
    reference["runtime_pruning_keep_mask_stage_count"] = keepmask_ref.get(
        "runtime_pruning_keep_mask_stage_count"
    )
    mismatches = {}
    fields = sorted(reference.keys())
    for field in fields:
        values = {}
        for run in runs:
            value = run.get("privacy_fields", {}).get(field)
            if field == "runtime_pruning_keep_mask_stage_count":
                value = run.get("keepmask_fields", {}).get(field)
            values[run["run_label"]] = value
        unique_values = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(unique_values) > 1:
            mismatches[field] = values
    return {
        "reference": reference,
        "consistent": not mismatches,
        "mismatches": mismatches,
    }


def mean(values):
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def build_pairwise_scaling(runs):
    rows = []
    for prev, curr in zip(runs, runs[1:]):
        prev_samples = prev.get("sample_count")
        curr_samples = curr.get("sample_count")
        prev_elapsed = prev.get("elapsed_sec")
        curr_elapsed = curr.get("elapsed_sec")
        prev_sec_per_sample = prev.get("sec_per_sample")
        curr_sec_per_sample = curr.get("sec_per_sample")
        sample_delta = (
            int(curr_samples) - int(prev_samples)
            if prev_samples is not None and curr_samples is not None
            else None
        )
        elapsed_delta = (
            float(curr_elapsed) - float(prev_elapsed)
            if prev_elapsed is not None and curr_elapsed is not None
            else None
        )
        rows.append(
            {
                "from_run_label": prev["run_label"],
                "to_run_label": curr["run_label"],
                "from_sample_count": prev_samples,
                "to_sample_count": curr_samples,
                "sample_ratio": (
                    float(curr_samples) / float(prev_samples)
                    if prev_samples not in (None, 0) and curr_samples is not None
                    else None
                ),
                "elapsed_ratio": (
                    float(curr_elapsed) / float(prev_elapsed)
                    if prev_elapsed not in (None, 0) and curr_elapsed is not None
                    else None
                ),
                "sec_per_sample_ratio": (
                    float(curr_sec_per_sample) / float(prev_sec_per_sample)
                    if prev_sec_per_sample not in (None, 0) and curr_sec_per_sample is not None
                    else None
                ),
                "sample_delta": sample_delta,
                "elapsed_delta_sec": elapsed_delta,
                "incremental_sec_per_new_sample": (
                    float(elapsed_delta) / float(sample_delta)
                    if elapsed_delta is not None and sample_delta not in (None, 0)
                    else None
                ),
            }
        )
    return rows


def build_aggregate_metrics(runs):
    sample_counts = [run.get("sample_count") for run in runs if run.get("sample_count") is not None]
    elapsed = [run.get("elapsed_sec") for run in runs if run.get("elapsed_sec") is not None]
    sec_per_sample = [run.get("sec_per_sample") for run in runs if run.get("sec_per_sample") is not None]
    logits_error = [
        run.get("logits_max_abs_error")
        for run in runs
        if run.get("logits_max_abs_error") is not None
    ]
    probs_error = [
        run.get("probabilities_max_abs_error")
        for run in runs
        if run.get("probabilities_max_abs_error") is not None
    ]
    argmax = [run.get("argmax_match_ratio") for run in runs]
    threshold = [run.get("threshold_match_ratio") for run in runs]
    return {
        "run_count": len(runs),
        "sample_count_min": min(sample_counts) if sample_counts else None,
        "sample_count_max": max(sample_counts) if sample_counts else None,
        "sample_count_total": sum(sample_counts) if sample_counts else None,
        "elapsed_sec_total": sum(float(value) for value in elapsed) if elapsed else None,
        "sec_per_sample_mean": mean(sec_per_sample),
        "sec_per_sample_min": min(sec_per_sample) if sec_per_sample else None,
        "sec_per_sample_max": max(sec_per_sample) if sec_per_sample else None,
        "sec_per_sample_spread_ratio": (
            max(sec_per_sample) / min(sec_per_sample)
            if sec_per_sample and min(sec_per_sample) not in (None, 0)
            else None
        ),
        "logits_max_abs_error_max": max(logits_error) if logits_error else None,
        "probabilities_max_abs_error_max": max(probs_error) if probs_error else None,
        "all_finite_logits": all(run.get("finite_logits") is True for run in runs),
        "all_argmax_match_ratio_one": all(value == 1.0 for value in argmax),
        "all_threshold_match_ratio_one": all(value == 1.0 for value in threshold),
    }


def build_judgement(privacy, aggregate, pairwise):
    spread_ratio = aggregate.get("sec_per_sample_spread_ratio")
    all_matches = aggregate.get("all_argmax_match_ratio_one") and aggregate.get(
        "all_threshold_match_ratio_one"
    )
    if not privacy.get("consistent"):
        return {
            "status": "privacy_boundary_inconsistent",
            "reason": "privacy fields differ across keep-mask runs; do not compare scaling until the boundary is fixed.",
        }
    if not aggregate.get("all_finite_logits"):
        return {
            "status": "non_finite_logits_observed",
            "reason": "at least one keep-mask wrapper run produced non-finite logits.",
        }
    if not all_matches:
        return {
            "status": "decision_drift_observed",
            "reason": "at least one keep-mask wrapper run lost argmax or threshold agreement with the reference.",
        }
    if spread_ratio is not None and spread_ratio <= 1.18:
        steady_rows = [
            row for row in pairwise if row.get("incremental_sec_per_new_sample") is not None
        ]
        if steady_rows:
            last_row = steady_rows[-1]
            return {
                "status": "near_linear_scaling_with_stable_accuracy",
                "reason": (
                    "privacy boundary stays fixed, all compared runs keep exact decision agreement, "
                    "and sec/sample remains close to flat; latest incremental cost is about "
                    f"{fmt_float(last_row.get('incremental_sec_per_new_sample'))} sec/new-sample."
                ),
            }
    return {
        "status": "scaling_observed_but_needs_more_points",
        "reason": (
            "current keep-mask wrapper runs are valid and privacy-consistent, "
            "but more sample-count points are still needed to judge scaling stability with confidence."
        ),
    }


def build_report(summaries):
    runs = sorted([normalize_run(summary) for summary in summaries], key=compare_key)
    privacy = build_privacy_comparison(runs)
    aggregate = build_aggregate_metrics(runs)
    pairwise = build_pairwise_scaling(runs)
    judgement = build_judgement(privacy=privacy, aggregate=aggregate, pairwise=pairwise)
    return {
        "manifest_type": "transshield_e2e_keepmask_scaling_report_v0",
        "runs": runs,
        "privacy_boundary": privacy,
        "aggregate_metrics": aggregate,
        "pairwise_scaling": pairwise,
        "judgement": judgement,
    }


def build_markdown(report: dict):
    privacy = report["privacy_boundary"]
    aggregate = report["aggregate_metrics"]
    lines = []
    lines.append("# Keep-mask Whole-forward Wrapper Scaling Report")
    lines.append("")
    lines.append("## Judgement")
    lines.append(
        f"- `status = {report['judgement']['status']}`: {report['judgement']['reason']}"
    )
    lines.append("")
    lines.append("## Privacy Boundary")
    for key in sorted(privacy["reference"].keys()):
        lines.append(f"- `{key} = {display_value(privacy['reference'][key])}`")
    lines.append(f"- `privacy_consistent = {bool_str(privacy['consistent'])}`")
    if privacy["mismatches"]:
        for key, value in sorted(privacy["mismatches"].items()):
            lines.append(f"- `mismatch[{key}] = {json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- `run_count = {aggregate['run_count']}`")
    lines.append(
        f"- `sample_count_min/max/total = {aggregate['sample_count_min']} / {aggregate['sample_count_max']} / {aggregate['sample_count_total']}`"
    )
    lines.append(f"- `elapsed_sec_total = {fmt_float(aggregate['elapsed_sec_total'])}`")
    lines.append(
        "- `sec_per_sample mean/min/max = "
        f"{fmt_float(aggregate['sec_per_sample_mean'])} / "
        f"{fmt_float(aggregate['sec_per_sample_min'])} / "
        f"{fmt_float(aggregate['sec_per_sample_max'])}`"
    )
    lines.append(
        f"- `sec_per_sample_spread_ratio = {fmt_float(aggregate['sec_per_sample_spread_ratio'])}`"
    )
    lines.append(
        f"- `logits_max_abs_error_max = {fmt_float(aggregate['logits_max_abs_error_max'], digits=6)}`"
    )
    lines.append(
        "- `probabilities_max_abs_error_max = "
        f"{fmt_float(aggregate['probabilities_max_abs_error_max'], digits=6)}`"
    )
    lines.append(f"- `all_finite_logits = {bool_str(aggregate['all_finite_logits'])}`")
    lines.append(
        f"- `all_argmax_match_ratio_one = {bool_str(aggregate['all_argmax_match_ratio_one'])}`"
    )
    lines.append(
        f"- `all_threshold_match_ratio_one = {bool_str(aggregate['all_threshold_match_ratio_one'])}`"
    )
    lines.append("")
    lines.append("## Runs")
    for run in report["runs"]:
        lines.append(f"- `{run['run_label']}`: `{run['run_dir']}/`")
        lines.append(f"  - `sample_count = {run['sample_count']}`")
        lines.append(f"  - `elapsed_sec = {fmt_float(run['elapsed_sec'])}`")
        lines.append(f"  - `sec_per_sample = {fmt_float(run['sec_per_sample'])}`")
        lines.append(
            "  - `logits/probabilities max_abs_error = "
            f"{fmt_float(run['logits_max_abs_error'], digits=6)} / "
            f"{fmt_float(run['probabilities_max_abs_error'], digits=6)}`"
        )
        lines.append(
            "  - `argmax/threshold match = "
            f"{fmt_float(run['argmax_match_ratio'], digits=1)} / "
            f"{fmt_float(run['threshold_match_ratio'], digits=1)}`"
        )
    lines.append("")
    lines.append("## Pairwise Scaling")
    for row in report["pairwise_scaling"]:
        lines.append(
            "- "
            f"`{row['from_run_label']} -> {row['to_run_label']}`: "
            f"`sample_ratio = {fmt_float(row['sample_ratio'])}`, "
            f"`elapsed_ratio = {fmt_float(row['elapsed_ratio'])}`, "
            f"`sec_per_sample_ratio = {fmt_float(row['sec_per_sample_ratio'])}`, "
            f"`incremental_sec_per_new_sample = {fmt_float(row['incremental_sec_per_new_sample'])}`"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple keep-mask whole-forward wrapper summaries into scaling reports."
    )
    parser.add_argument("--summary-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    summaries = [load_json(Path(path).expanduser().resolve()) for path in args.summary_json]
    report = build_report(summaries)
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    write_json(output_json, report)
    write_text(output_md, build_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
