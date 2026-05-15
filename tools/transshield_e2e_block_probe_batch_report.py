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


def parse_probe_arg(raw: str) -> Dict:
    if "=" not in raw:
        raise ValueError("--probe-summary must use label=path")
    label, path = raw.split("=", 1)
    probe_path = Path(path).expanduser().resolve()
    payload = load_json(probe_path)
    sample = payload.get("sample") or {}
    interpretation = payload.get("interpretation") or {}
    block_output_series = []
    attn_cosine_series = []
    final_logits_max_abs_series = []
    for row in payload.get("blocks") or []:
      block_output = (row.get("key_stage_errors") or {}).get("block_output_cls") or {}
      attn = (row.get("key_stage_errors") or {}).get("attn_out_cls") or {}
      final_logits = (row.get("key_stage_errors") or {}).get("final_logits") or {}
      block_output_series.append(
          {
              "block_ordinal": row.get("block_ordinal"),
              "max_abs_error": block_output.get("max_abs_error"),
              "relative_l2_error": block_output.get("relative_l2_error"),
          }
      )
      attn_cosine_series.append(
          {
              "block_ordinal": row.get("block_ordinal"),
              "cosine_similarity": attn.get("cosine_similarity"),
          }
      )
      final_logits_max_abs_series.append(
          {
              "block_ordinal": row.get("block_ordinal"),
              "max_abs_error": final_logits.get("max_abs_error"),
          }
      )
    return {
        "label": label,
        "probe_summary_json": str(probe_path),
        "sample": sample,
        "interpretation": interpretation,
        "block_output_series": block_output_series,
        "attn_cosine_series": attn_cosine_series,
        "final_logits_max_abs_series": final_logits_max_abs_series,
    }


def finite_values(rows: List[Dict], key: str) -> List[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def summarize(items: List[Dict]) -> Dict:
    growths = [
        float(item["interpretation"]["block_output_max_abs_growth_first_to_last"])
        for item in items
        if isinstance(item.get("interpretation", {}).get("block_output_max_abs_growth_first_to_last"), (int, float))
    ]
    min_attn_cosines = [
        float(item["interpretation"]["min_attn_out_cls_cosine"])
        for item in items
        if isinstance(item.get("interpretation", {}).get("min_attn_out_cls_cosine"), (int, float))
    ]
    max_final_logits_abs = [
        float(item["interpretation"]["max_debug_final_logits_abs_error"])
        for item in items
        if isinstance(item.get("interpretation", {}).get("max_debug_final_logits_abs_error"), (int, float))
    ]
    all_consistent = bool(items) and all(
        item.get("interpretation", {}).get("status") == "late_block_cumulative_numeric_drift_with_high_cosine_alignment"
        for item in items
    )
    if all_consistent:
        status = "consistent_late_block_cumulative_drift_pattern_observed"
        reason = (
            "Every selected sample shows high attn cosine alignment plus late-block block_output drift growth; "
            "the residual error pattern is consistent with cumulative numeric drift, not a one-off attention-direction failure."
        )
    else:
        status = "mixed_block_probe_patterns"
        reason = "Not all selected samples share the same late-block cumulative drift interpretation."
    return {
        "status": status,
        "reason": reason,
        "sample_count": len(items),
        "growth_stats": {
            "min": min(growths) if growths else None,
            "max": max(growths) if growths else None,
            "mean": (sum(growths) / len(growths)) if growths else None,
        },
        "min_attn_cosine_stats": {
            "min": min(min_attn_cosines) if min_attn_cosines else None,
            "max": max(min_attn_cosines) if min_attn_cosines else None,
        },
        "max_debug_final_logits_abs_error_stats": {
            "min": min(max_final_logits_abs) if max_final_logits_abs else None,
            "max": max(max_final_logits_abs) if max_final_logits_abs else None,
        },
    }


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E Block Probe Batch Report",
        "",
        f"- label: `{report['label']}`",
        f"- status: `{report['summary']['status']}`",
        f"- reason: {report['summary']['reason']}",
        "",
        "| probe | source index | target | spuaware score | growth first->last | min attn cosine | max final logits abs | image |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["items"]:
        sample = item.get("sample") or {}
        interpretation = item.get("interpretation") or {}
        label = item.get("label")
        source_index = None
        image = sample.get("image") or ""
        if "idx" in label:
            try:
                source_index = int(label.split("idx", 1)[1].split("_", 1)[0])
            except Exception:
                source_index = None
        lines.append(
            "| {label} | {source_index} | {target} | {score} | {growth} | {cos} | {logits} | {image} |".format(
                label=label,
                source_index="" if source_index is None else source_index,
                target=sample.get("target", ""),
                score=fmt(sample.get("bias_score")),
                growth=fmt(interpretation.get("block_output_max_abs_growth_first_to_last")),
                cos=fmt(interpretation.get("min_attn_out_cls_cosine")),
                logits=fmt(interpretation.get("max_debug_final_logits_abs_error")),
                image=image,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple E2E single-sample block probe summaries.")
    parser.add_argument("--probe-summary", action="append", default=[], help="label=path/to/block_probe_summary.json")
    parser.add_argument("--label", default="e2e_block_probe_batch")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    if not args.probe_summary:
        raise SystemExit("at least one --probe-summary is required")
    items = [parse_probe_arg(raw) for raw in args.probe_summary]
    report = {
        "manifest_type": "transshield_e2e_block_probe_batch_report_v0",
        "label": args.label,
        "items": items,
        "summary": summarize(items),
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
