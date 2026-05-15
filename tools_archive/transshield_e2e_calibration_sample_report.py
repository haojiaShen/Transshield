#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_float(value: str):
    if value in (None, ""):
        return None
    return float(value)


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            row["index"] = int(row["index"])
            row["target"] = int(row["target"])
            for key in list(row):
                if key.endswith("_correct"):
                    row[key] = parse_bool(row[key])
                elif key.endswith("_prediction"):
                    row[key] = int(row[key])
                elif key.endswith("_score") or key.endswith("_abs_margin") or key.startswith("raw_"):
                    row[key] = parse_float(row[key])
            rows.append(row)
    return rows


def slim(row: Dict) -> Dict:
    keys = [
        "index",
        "image",
        "target",
        "raw_score_logit1_minus_logit0",
        "static_bias_prediction",
        "static_bias_correct",
        "static_bias_score",
        "static_bias_abs_margin",
        "spuaware_bias_prediction",
        "spuaware_bias_correct",
        "spuaware_bias_score",
        "spuaware_bias_abs_margin",
        "e2e_smoke32_affine_prediction",
        "e2e_smoke32_affine_correct",
        "e2e_smoke32_affine_score",
        "e2e_smoke32_affine_abs_margin",
        "e2e_smoke32_temperature_score",
        "e2e_smoke32_temperature_abs_margin",
    ]
    return {key: row.get(key) for key in keys if key in row}


def count_correct(rows: List[Dict], calibration: str) -> int:
    return sum(1 for row in rows if row.get(f"{calibration}_correct") is True)


def low_margin_count(rows: List[Dict], calibration: str, threshold: float) -> int:
    key = f"{calibration}_abs_margin"
    return sum(1 for row in rows if isinstance(row.get(key), (int, float)) and row[key] < threshold)


def sort_by_abs_margin(rows: List[Dict], calibration: str, reverse: bool = False) -> List[Dict]:
    key = f"{calibration}_abs_margin"
    return sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=reverse)


def build_report(rows: List[Dict], label: str, top_k: int) -> Dict:
    sample_count = len(rows)
    calibrations = [
        "static_bias",
        "spuaware_bias",
        "e2e_smoke32_affine",
        "e2e_smoke32_temperature",
    ]
    summary = {}
    for calibration in calibrations:
        correct = count_correct(rows, calibration)
        summary[calibration] = {
            "correct_count": correct,
            "wrong_count": sample_count - correct,
            "accuracy": None if sample_count == 0 else correct / sample_count * 100.0,
            "low_margin_count_abs_lt_0_25": low_margin_count(rows, calibration, 0.25),
            "low_margin_count_abs_lt_0_50": low_margin_count(rows, calibration, 0.50),
        }

    static_wrong_spuaware_correct = [
        row for row in rows if not row.get("static_bias_correct") and row.get("spuaware_bias_correct")
    ]
    spuaware_wrong = [row for row in rows if not row.get("spuaware_bias_correct")]
    spuaware_wrong_low_margin = sort_by_abs_margin(spuaware_wrong, "spuaware_bias")[:top_k]
    spuaware_wrong_high_margin = sort_by_abs_margin(spuaware_wrong, "spuaware_bias", reverse=True)[:top_k]
    static_wrong_spuaware_correct_sorted = sort_by_abs_margin(static_wrong_spuaware_correct, "static_bias")[:top_k]

    affine_margin_gain_rows = []
    for row in rows:
        spu_margin = row.get("spuaware_bias_abs_margin")
        affine_margin = row.get("e2e_smoke32_affine_abs_margin")
        if not isinstance(spu_margin, (int, float)) or not isinstance(affine_margin, (int, float)):
            continue
        enriched = dict(row)
        enriched["affine_minus_spuaware_abs_margin"] = affine_margin - spu_margin
        affine_margin_gain_rows.append(enriched)
    affine_margin_gain_rows.sort(key=lambda row: row["affine_minus_spuaware_abs_margin"], reverse=True)

    return {
        "manifest_type": "transshield_e2e_calibration_sample_report_v0",
        "label": label,
        "sample_count": sample_count,
        "summary_by_calibration": summary,
        "category_counts": {
            "static_wrong_spuaware_correct": len(static_wrong_spuaware_correct),
            "static_correct_spuaware_wrong": sum(
                1 for row in rows if row.get("static_bias_correct") and not row.get("spuaware_bias_correct")
            ),
            "spuaware_wrong": len(spuaware_wrong),
        },
        "probe_recommendations": {
            "static_wrong_spuaware_correct": [slim(row) for row in static_wrong_spuaware_correct_sorted],
            "spuaware_wrong_low_margin": [slim(row) for row in spuaware_wrong_low_margin],
            "spuaware_wrong_high_margin": [slim(row) for row in spuaware_wrong_high_margin],
            "largest_affine_margin_gain": [slim(row) | {"affine_minus_spuaware_abs_margin": row["affine_minus_spuaware_abs_margin"]} for row in affine_margin_gain_rows[:top_k]],
        },
        "interpretation": {
            "accuracy_axis": (
                "Rows in static_wrong_spuaware_correct explain why SPU-aware bias is now the accuracy-first default: "
                "they are boundary cases recovered by the SPU-aware public threshold."
            ),
            "drift_probe_axis": (
                "Use spuaware_wrong_high_margin for late-block numeric drift probes, because public threshold changes "
                "are unlikely to recover high-margin wrong decisions."
            ),
            "loss_axis": (
                "Affine/temperature margin expansion is a post-reveal confidence/loss calibration layer; it does not "
                "change the secret-sharing SPU computation graph."
            ),
        },
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E Calibration Sample Report",
        "",
        f"- label: `{report['label']}`",
        f"- sample_count: `{report['sample_count']}`",
        "",
        "| calibration | accuracy | wrong | low margin <0.25 | low margin <0.50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for calibration, summary in report["summary_by_calibration"].items():
        lines.append(
            f"| {calibration} | {fmt(summary['accuracy'])} | {summary['wrong_count']} | "
            f"{summary['low_margin_count_abs_lt_0_25']} | {summary['low_margin_count_abs_lt_0_50']} |"
        )
    lines.extend(["", "## Category Counts", ""])
    for key, value in report["category_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    for category, rows in report["probe_recommendations"].items():
        lines.extend(
            [
                "",
                f"## {category}",
                "",
                "| index | target | static | spuaware | affine | image |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| {index} | {target} | {static} | {spuaware} | {affine} | {image} |".format(
                    index=row.get("index"),
                    target=row.get("target"),
                    static=fmt(row.get("static_bias_score")),
                    spuaware=fmt(row.get("spuaware_bias_score")),
                    affine=fmt(row.get("e2e_smoke32_affine_score")),
                    image=row.get("image"),
                )
            )
    lines.extend(["", "## Interpretation", ""])
    for value in report["interpretation"].values():
        lines.append(f"- {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-sample E2E calibration transfer CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--label", default="e2e_calibration_sample_report")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv).expanduser().resolve())
    report = build_report(rows, args.label, args.top_k)
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps(report["category_counts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
