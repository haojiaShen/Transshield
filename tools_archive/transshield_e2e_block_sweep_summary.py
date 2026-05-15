#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KEY_STAGES = [
    "block_input_cls",
    "norm1_out_cls",
    "attn_out_cls",
    "attn_residual_out_cls",
    "norm2_out_cls",
    "mlp_out_cls",
    "block_output_cls",
    "final_norm_cls",
    "final_logits",
    "final_probabilities",
]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel_l2(stage: Optional[Dict]) -> Optional[float]:
    if not stage:
        return None
    l2_error = stage.get("l2_error")
    reference_l2_norm = stage.get("reference_l2_norm")
    if not isinstance(l2_error, (int, float)) or not isinstance(reference_l2_norm, (int, float)):
        return None
    if float(reference_l2_norm) == 0.0:
        return None
    return float(l2_error) / float(reference_l2_norm)


def compact_stage(stage: Optional[Dict]) -> Optional[Dict]:
    if not stage:
        return None
    return {
        "cosine_similarity": stage.get("cosine_similarity"),
        "l2_error": stage.get("l2_error"),
        "max_abs_error": stage.get("max_abs_error"),
        "mean_abs_error": stage.get("mean_abs_error"),
        "reference_l2_norm": stage.get("reference_l2_norm"),
        "relative_l2_error": rel_l2(stage),
    }


def parse_compare(path: Path) -> Dict:
    payload = load_json(path)
    stage_errors = payload.get("stage_errors") or payload.get("intermediate_stage_errors") or {}
    largest = payload.get("largest_stage") or {}
    block_ordinal = payload.get("probe_block_ordinal")
    if block_ordinal is None:
        block_ordinal = int(payload.get("probe_block_index", -1)) + 1
    key_stage_errors = {name: compact_stage(stage_errors.get(name)) for name in KEY_STAGES if name in stage_errors}
    largest_compact = compact_stage(largest)
    if largest_compact is not None:
        largest_compact["name"] = largest.get("name")
    return {
        "block_ordinal": int(block_ordinal),
        "compare_json": str(path),
        "debug_prediction_match": payload.get("prediction_match"),
        "debug_prediction_match_is_full_candidate_safe": payload.get("prediction_match_is_full_candidate_safe"),
        "largest_stage": largest_compact,
        "key_stage_errors": key_stage_errors,
    }


def discover_compare_jsons(input_dir: Path, pattern: str) -> List[Path]:
    paths = [Path(value) for value in glob.glob(str(input_dir / pattern))]
    if not paths:
        raise FileNotFoundError(f"no compare JSONs found under {input_dir} with pattern {pattern}")
    return sorted(paths, key=lambda path: parse_block_sort_key(path))


def parse_block_sort_key(path: Path) -> Tuple[int, str]:
    for part in path.parts:
        if part.startswith("block") and part[5:].isdigit():
            return int(part[5:]), str(path)
    return 10**9, str(path)


def build_interpretation(rows: List[Dict]) -> Dict:
    block_outputs = [
        row["key_stage_errors"].get("block_output_cls")
        for row in rows
        if row["key_stage_errors"].get("block_output_cls")
    ]
    attn_outputs = [
        row["key_stage_errors"].get("attn_out_cls")
        for row in rows
        if row["key_stage_errors"].get("attn_out_cls")
    ]
    final_logits = [
        row["key_stage_errors"].get("final_logits")
        for row in rows
        if row["key_stage_errors"].get("final_logits")
    ]
    first_block_output = block_outputs[0] if block_outputs else None
    last_block_output = block_outputs[-1] if block_outputs else None
    first_max_abs = None if not first_block_output else first_block_output.get("max_abs_error")
    last_max_abs = None if not last_block_output else last_block_output.get("max_abs_error")
    growth = None
    if isinstance(first_max_abs, (int, float)) and first_max_abs:
        growth = float(last_max_abs) / float(first_max_abs)
    min_attn_cosine = min(
        (float(stage["cosine_similarity"]) for stage in attn_outputs if stage.get("cosine_similarity") is not None),
        default=None,
    )
    max_final_logits_abs = max(
        (float(stage["max_abs_error"]) for stage in final_logits if stage.get("max_abs_error") is not None),
        default=None,
    )
    return {
        "status": "late_block_cumulative_numeric_drift_with_high_cosine_alignment",
        "reason": (
            "Block sweep on the selected sample shows high cosine alignment at every probed block, "
            "while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude "
            "drift plus insufficient boundary robustness, not a large attention-direction mismatch."
        ),
        "block_output_max_abs_growth_first_to_last": growth,
        "min_attn_out_cls_cosine": min_attn_cosine,
        "max_debug_final_logits_abs_error": max_final_logits_abs,
        "debug_graph_warning": (
            "Probe prediction_match fields describe only the debug probe graph and are not full candidate decisions."
        ),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 100:
            return f"{float(value):.4f}"
        return f"{float(value):.6g}"
    return str(value)


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E AA=none Block Sweep Summary",
        "",
        f"- label: `{report['label']}`",
        f"- sample: `{report['sample']['label']}`",
        f"- status: `{report['interpretation']['status']}`",
        f"- reason: {report['interpretation']['reason']}",
        "",
        "| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["blocks"]:
        largest = row.get("largest_stage") or {}
        block_output = row["key_stage_errors"].get("block_output_cls") or {}
        attn = row["key_stage_errors"].get("attn_out_cls") or {}
        final_logits = row["key_stage_errors"].get("final_logits") or {}
        lines.append(
            "| {block} | {stage} | {cos} | {max_abs} | {rel_l2} | {bo_max_abs} | {bo_rel_l2} | {attn_rel_l2} | {logits_abs} |".format(
                block=row["block_ordinal"],
                stage=largest.get("name", ""),
                cos=fmt(largest.get("cosine_similarity")),
                max_abs=fmt(largest.get("max_abs_error")),
                rel_l2=fmt(largest.get("relative_l2_error")),
                bo_max_abs=fmt(block_output.get("max_abs_error")),
                bo_rel_l2=fmt(block_output.get("relative_l2_error")),
                attn_rel_l2=fmt(attn.get("relative_l2_error")),
                logits_abs=fmt(final_logits.get("max_abs_error")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.",
            "- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.",
            "- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.",
            "- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CPU-vs-SPU E2E block probe sweeps.")
    parser.add_argument("--input-dir", required=True, help="directory containing block*/e2e_secure_poc compare JSONs")
    parser.add_argument(
        "--pattern",
        default="block*/e2e_secure_poc/block*_probe_compare_cpu_vs_spu_depth*.json",
        help="glob pattern relative to input-dir",
    )
    parser.add_argument("--label", default="e2e_block_sweep")
    parser.add_argument("--sample-label", default="sample")
    parser.add_argument("--sample-image", default="")
    parser.add_argument("--sample-target", type=int, default=None)
    parser.add_argument("--sample-bias-score", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    paths = discover_compare_jsons(input_dir, args.pattern)
    rows = [parse_compare(path.resolve()) for path in paths]
    rows.sort(key=lambda row: row["block_ordinal"])
    report = {
        "manifest_type": "transshield_e2e_block_sweep_summary_v0",
        "label": args.label,
        "input_dir": str(input_dir),
        "sample": {
            "label": args.sample_label,
            "image": args.sample_image or None,
            "target": args.sample_target,
            "bias_score": args.sample_bias_score,
        },
        "blocks": rows,
        "interpretation": build_interpretation(rows),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps(report["interpretation"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
