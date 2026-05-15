#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def parse_transfer_report(label: str, path: Path) -> Dict:
    report = load_json(path)
    rows = []
    for calibration, item in sorted((report.get("calibration_results") or {}).items()):
        score_summary = item.get("score_summary") or {}
        rows.append(
            {
                "split": label,
                "sample_count": report.get("sample_count"),
                "calibration": calibration,
                "accuracy": item.get("accuracy"),
                "binary_cross_entropy": item.get("binary_cross_entropy"),
                "wrong_count": score_summary.get("wrong_count"),
                "low_margin_count_abs_lt_0_25": score_summary.get("low_margin_count_abs_lt_0_25"),
                "mean_abs_margin": score_summary.get("mean_abs_margin"),
            }
        )
    return {
        "split": label,
        "source_report": str(path),
        "sample_count": report.get("sample_count"),
        "rows": rows,
    }


def best_by(rows: List[Dict], metric: str, *, higher_is_better: bool) -> Dict:
    valid = [row for row in rows if isinstance(row.get(metric), (int, float))]
    if not valid:
        return {}
    return sorted(valid, key=lambda row: row[metric], reverse=higher_is_better)[0]


def average_metric_by_calibration(rows: List[Dict], metric: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            buckets.setdefault(row["calibration"], []).append(float(value))
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def weighted_average_metric_by_calibration(rows: List[Dict], metric: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for row in rows:
        value = row.get(metric)
        sample_count = row.get("sample_count")
        if not isinstance(value, (int, float)) or not isinstance(sample_count, (int, float)):
            continue
        weight = float(sample_count)
        totals[row["calibration"]] = totals.get(row["calibration"], 0.0) + float(value) * weight
        weights[row["calibration"]] = weights.get(row["calibration"], 0.0) + weight
    return {key: totals[key] / weights[key] for key in totals if weights.get(key, 0.0) > 0.0}


def calibration_metric(rows: List[Dict], calibration: str, metric: str) -> Optional[float]:
    values = [row.get(metric) for row in rows if row.get("calibration") == calibration]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def split_accuracy_deltas(rows: List[Dict], candidate: str, baseline: str) -> List[Dict]:
    splits = sorted({row["split"] for row in rows})
    deltas = []
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        candidate_acc = calibration_metric(split_rows, candidate, "accuracy")
        baseline_acc = calibration_metric(split_rows, baseline, "accuracy")
        delta = None if candidate_acc is None or baseline_acc is None else candidate_acc - baseline_acc
        deltas.append(
            {
                "split": split,
                "candidate_calibration": candidate,
                "baseline_calibration": baseline,
                "candidate_accuracy": candidate_acc,
                "baseline_accuracy": baseline_acc,
                "accuracy_delta": delta,
            }
        )
    return deltas


def choose_accuracy_first(averages: Dict[str, float]) -> str:
    if not averages:
        return None
    max_value = max(averages.values())
    tied = sorted(key for key, value in averages.items() if abs(value - max_value) <= 1e-9)
    for preferred in ("spuaware_bias", "e2e_smoke32_bias", "e2e_smoke32_affine", "e2e_smoke32_temperature"):
        if preferred in tied:
            return preferred
    return tied[0]


def choose_loss_first(averages: Dict[str, float]) -> str:
    if not averages:
        return None
    min_value = min(averages.values())
    tied = sorted(key for key, value in averages.items() if abs(value - min_value) <= 1e-9)
    for preferred in ("e2e_smoke32_affine", "e2e_smoke32_temperature", "spuaware_bias"):
        if preferred in tied:
            return preferred
    return tied[0]


def build_judgement(all_rows: List[Dict]) -> Dict:
    heldout_rows = [row for row in all_rows if row["split"].startswith("heldout")]
    heldout_splits = sorted({row["split"] for row in heldout_rows})
    has_heldout238 = "heldout238" in heldout_splits
    accuracy_averages = average_metric_by_calibration(heldout_rows, "accuracy")
    loss_averages = average_metric_by_calibration(heldout_rows, "binary_cross_entropy")
    weighted_accuracy_averages = weighted_average_metric_by_calibration(heldout_rows, "accuracy")
    weighted_loss_averages = weighted_average_metric_by_calibration(heldout_rows, "binary_cross_entropy")
    accuracy_first = choose_accuracy_first(weighted_accuracy_averages or accuracy_averages)
    loss_first = choose_loss_first(weighted_loss_averages or loss_averages)
    static_acc = (weighted_accuracy_averages or accuracy_averages).get("static_bias")
    spuaware_acc = (weighted_accuracy_averages or accuracy_averages).get("spuaware_bias")
    accuracy_delta = None if static_acc is None or spuaware_acc is None else spuaware_acc - static_acc
    deltas = split_accuracy_deltas(heldout_rows, "spuaware_bias", "static_bias")
    heldout238_delta = next((row.get("accuracy_delta") for row in deltas if row["split"] == "heldout238"), None)
    any_spuaware_regression = any(
        isinstance(row.get("accuracy_delta"), (int, float)) and row["accuracy_delta"] < -1e-9 for row in deltas
    )
    promotion_gain_threshold = 0.5
    promotion_gate_satisfied = bool(
        has_heldout238
        and accuracy_delta is not None
        and accuracy_delta >= promotion_gain_threshold
        and (heldout238_delta is None or heldout238_delta >= -1e-9)
        and not any_spuaware_regression
    )

    if promotion_gate_satisfied:
        status = "promote_spuaware_bias_as_accuracy_first_default"
        reason = (
            "SPU-aware bias has cleared the heldout238 gate: sample-weighted held-out accuracy improves over "
            "static bias, heldout238 is non-regressive, and no held-out split regresses."
        )
        default_gate = "Cleared: SPU-aware bias can replace static bias for accuracy-first E2E output calibration."
    elif accuracy_delta is not None and accuracy_delta > 0.0:
        status = "keep_spuaware_bias_as_accuracy_first_candidate"
        if has_heldout238:
            reason = (
                "SPU-aware bias improves aggregate held-out accuracy over static bias, but it has not cleared the "
                "promotion gate because the gain is below threshold or at least one held-out split regresses."
            )
            default_gate = (
                f"Not cleared: require sample-weighted gain >= {promotion_gain_threshold} percentage point, "
                "heldout238 non-regression, and no held-out split regression."
            )
        else:
            reason = (
                "SPU-aware bias improves average held-out accuracy over static bias, but evidence is still based on "
                "small-to-medium E2E subsets and should wait for the running heldout238 check before becoming default."
            )
            default_gate = "Wait for heldout238; promote only if it confirms non-regression and meaningful aggregate accuracy gain."
    else:
        status = "do_not_promote_spuaware_bias"
        reason = "SPU-aware bias has not yet shown a held-out accuracy advantage over static bias."
        default_gate = "Not cleared: held-out accuracy advantage over static bias is absent."
    return {
        "status": status,
        "reason": reason,
        "heldout_splits": heldout_splits,
        "has_heldout238": has_heldout238,
        "accuracy_first_choice": accuracy_first,
        "loss_first_choice": loss_first,
        "average_accuracy_by_calibration": accuracy_averages,
        "average_bce_by_calibration": loss_averages,
        "sample_weighted_accuracy_by_calibration": weighted_accuracy_averages,
        "sample_weighted_bce_by_calibration": weighted_loss_averages,
        "average_static_bias_heldout_accuracy": static_acc,
        "average_spuaware_bias_heldout_accuracy": spuaware_acc,
        "average_spuaware_minus_static_accuracy": accuracy_delta,
        "spuaware_vs_static_split_accuracy_deltas": deltas,
        "promotion_gain_threshold_percentage_point": promotion_gain_threshold,
        "heldout238_spuaware_minus_static_accuracy": heldout238_delta,
        "any_spuaware_heldout_regression": any_spuaware_regression,
        "promotion_gate_satisfied": promotion_gate_satisfied,
        "default_promotion_gate": default_gate,
        "confidence_loss_recovery_note": (
            "E2E-smoke32 affine/temperature are loss-first candidates: they reduce BCE and low-margin counts, "
            "but should be selected separately from the accuracy-first SPU-aware bias decision."
        ),
    }


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E Calibration Decision Report",
        "",
        f"- status: `{report['judgement']['status']}`",
        f"- reason: {report['judgement']['reason']}",
        f"- accuracy-first choice: `{report['judgement']['accuracy_first_choice']}`",
        f"- loss-first choice: `{report['judgement']['loss_first_choice']}`",
        f"- default gate: {report['judgement']['default_promotion_gate']}",
        "",
        "| split | calibration | accuracy | BCE loss | wrong | low margin <0.25 | mean abs margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["flat_rows"]:
        lines.append(
            "| {split} | {calibration} | {accuracy} | {loss} | {wrong} | {low} | {margin} |".format(
                split=f"{row['split']} (n={row.get('sample_count')})",
                calibration=row["calibration"],
                accuracy=fmt(row.get("accuracy")),
                loss=fmt(row.get("binary_cross_entropy")),
                wrong=fmt(row.get("wrong_count")),
                low=fmt(row.get("low_margin_count_abs_lt_0_25")),
                margin=fmt(row.get("mean_abs_margin")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use `spuaware_bias` when the priority is threshold / argmax accuracy.",
            "- Use E2E-smoke32 affine or temperature when the priority is lower BCE loss and fewer low-margin outputs.",
            "- Aggregate choices use sample-weighted held-out averages when sample counts are available.",
            "- None of these public output calibrations changes the secret-sharing 2PC computation graph.",
            "- None of these should be described as fixing late-block numeric drift; they are post-reveal public calibration layers.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an E2E output calibration decision report.")
    parser.add_argument("--transfer-report", action="append", required=True, help="label=path")
    parser.add_argument("--label", default="e2e_calibration_decision")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    transfer_reports = []
    flat_rows = []
    for item in args.transfer_report:
        if "=" not in item:
            raise ValueError("--transfer-report must use label=path")
        label, raw_path = item.split("=", 1)
        parsed = parse_transfer_report(label, Path(raw_path).expanduser().resolve())
        transfer_reports.append(parsed)
        flat_rows.extend(parsed["rows"])

    report = {
        "manifest_type": "transshield_e2e_calibration_decision_report_v0",
        "label": args.label,
        "transfer_reports": transfer_reports,
        "flat_rows": flat_rows,
        "judgement": build_judgement(flat_rows),
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps(report["judgement"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
