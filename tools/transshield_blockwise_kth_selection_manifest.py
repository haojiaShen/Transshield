import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_network_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    network_definition = manifest.get("network_definition", {})
    if network_definition.get("network_type") != "fixed_odd_even_compare_swap_desc":
        raise ValueError(f"unsupported network type: {network_definition.get('network_type')}")
    return manifest


def odd_even_compare_schedule(token_count: int):
    token_count = int(token_count)
    total_comparators = 0
    for pass_index in range(token_count):
        start_index = pass_index % 2
        total_comparators += len(range(start_index, token_count - 1, 2))
    return {
        "pass_count": token_count,
        "total_comparators": total_comparators,
        "comparators_per_pass_mean": float(total_comparators / token_count) if token_count > 0 else 0.0,
        "depth": token_count,
    }


def parse_index_set(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    return {int(item) for item in raw.replace(",", " ").split()}


def build_contiguous_blocks(token_count: int, block_count: int):
    token_count = int(token_count)
    block_count = int(block_count)
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if block_count > token_count:
        block_count = token_count

    base = token_count // block_count
    remainder = token_count % block_count
    blocks = []
    start_index = 0
    for block_index in range(block_count):
        block_len = base + (1 if block_index < remainder else 0)
        end_index = start_index + block_len
        blocks.append(
            {
                "block_index": block_index,
                "start_index": start_index,
                "end_index_exclusive": end_index,
                "block_token_count": block_len,
            }
        )
        start_index = end_index
    return blocks


def estimate_blockwise_plan(active_token_count: int, keep_count: int, block_count: int):
    active_token_count = int(active_token_count)
    keep_count = int(keep_count)
    lower_tail_count = active_token_count - keep_count
    candidate_rank = lower_tail_count + 1
    blocks = build_contiguous_blocks(active_token_count, block_count)

    flat_schedule = odd_even_compare_schedule(active_token_count)
    local_total_comparators = 0
    candidate_count = 0
    for block in blocks:
        local_total_comparators += odd_even_compare_schedule(block["block_token_count"])["total_comparators"]
        candidate_count += min(block["block_token_count"], candidate_rank)
    merge_schedule = odd_even_compare_schedule(candidate_count)
    total_comparators = local_total_comparators + merge_schedule["total_comparators"]

    return {
        "block_count": len(blocks),
        "block_size": max(block["block_token_count"] for block in blocks) if blocks else 0,
        "blocks": blocks,
        "candidate_rank": candidate_rank,
        "candidate_count": candidate_count,
        "candidate_side": "lower_tail",
        "candidate_side_meaning": "compute the same kth-largest threshold as the lower-tail rank",
        "candidate_strategy": "smaller_exact_side",
        "current_flat_network_definition": {
            "input_token_count": active_token_count,
            "schedule_summary": flat_schedule,
        },
        "merge_network_definition": {
            "input_token_count": candidate_count,
            "network_type": "fixed_odd_even_compare_swap",
            "schedule_summary": merge_schedule,
            "selection_goal": "recover exact kth threshold from reduced candidate set",
        },
        "expected_reduction_vs_current_flat": {
            "candidate_ratio": float(candidate_count / active_token_count) if active_token_count > 0 else 1.0,
            "merge_total_comparator_ratio": (
                float(merge_schedule["total_comparators"] / flat_schedule["total_comparators"])
                if flat_schedule["total_comparators"] > 0
                else 1.0
            ),
            "total_comparator_ratio": (
                float(total_comparators / flat_schedule["total_comparators"])
                if flat_schedule["total_comparators"] > 0
                else 1.0
            ),
        },
        "estimated_total_comparators": {
            "flat_total": int(flat_schedule["total_comparators"]),
            "local_block_total": int(local_total_comparators),
            "merge_total": int(merge_schedule["total_comparators"]),
            "combined_total": int(total_comparators),
        },
    }


def choose_stage_plan(
    stage,
    enabled_stage_indices,
    min_block_count: int,
    max_block_count: int,
    min_improvement_ratio: float,
):
    stage_index = int(stage["stage_index"])
    pruning_layer = int(stage["pruning_layer"])
    active_token_count = int(stage.get("effective_input_token_count", stage["input_token_count"]))
    keep_count = int(stage["keep_count"])

    flat_schedule = odd_even_compare_schedule(active_token_count)
    stage_entry = {
        "stage_index": stage_index,
        "pruning_layer": pruning_layer,
        "active_token_count": active_token_count,
        "keep_count": keep_count,
        "lower_tail_count": int(active_token_count - keep_count),
        "constraints": {
            "must_match_existing_kth_threshold_exactly": True,
            "must_not_change_tie_policy_semantics": True,
            "must_pass_existing_kth_checker_before_replay": True,
        },
        "current_flat_network_definition": {
            "input_token_count": active_token_count,
            "schedule_summary": flat_schedule,
        },
    }

    if enabled_stage_indices is not None and stage_index not in enabled_stage_indices:
        stage_entry["stage_selection_kind"] = "flat_odd_even"
        stage_entry["selection_reason"] = "disabled_by_enabled_stage_indices"
        return stage_entry

    best_candidate = None
    for block_count in range(max(int(min_block_count), 2), max(int(max_block_count), 2) + 1):
        if block_count >= active_token_count:
            continue
        candidate = estimate_blockwise_plan(active_token_count, keep_count, block_count)
        if best_candidate is None:
            best_candidate = candidate
            continue
        if candidate["estimated_total_comparators"]["combined_total"] < best_candidate["estimated_total_comparators"]["combined_total"]:
            best_candidate = candidate

    if best_candidate is None:
        stage_entry["stage_selection_kind"] = "flat_odd_even"
        stage_entry["selection_reason"] = "no_valid_block_count"
        return stage_entry

    total_ratio = best_candidate["expected_reduction_vs_current_flat"]["total_comparator_ratio"]
    improvement = 1.0 - total_ratio
    if improvement < float(min_improvement_ratio):
        stage_entry["stage_selection_kind"] = "flat_odd_even"
        stage_entry["selection_reason"] = "estimated_gain_below_threshold"
        stage_entry["best_blockwise_candidate"] = best_candidate
        return stage_entry

    stage_entry.update(best_candidate)
    stage_entry["stage_selection_kind"] = "blockwise_lower_tail_exact"
    stage_entry["selection_reason"] = "best_estimated_total_comparators"
    return stage_entry


def build_manifest(
    network_manifest,
    enabled_stage_indices,
    min_block_count: int,
    max_block_count: int,
    min_improvement_ratio: float,
):
    stage_plan = [
        choose_stage_plan(stage, enabled_stage_indices, min_block_count, max_block_count, min_improvement_ratio)
        for stage in network_manifest["stage_plan"]
    ]
    blockwise_stage_count = sum(1 for stage in stage_plan if stage["stage_selection_kind"] == "blockwise_lower_tail_exact")
    return {
        "manifest_type": "blockwise_exact_kth_selection",
        "design_status": "generated_local_manifest_only",
        "production_bridge_replaced": False,
        "source_current_manifest_json": network_manifest.get("single_source_of_truth", {}).get("bundle_dir"),
        "source_network_manifest_json": network_manifest.get("single_source_of_truth", {}).get("bundle_dir"),
        "generation_policy": {
            "enabled_stage_indices": sorted(enabled_stage_indices) if enabled_stage_indices is not None else "all",
            "min_block_count": int(min_block_count),
            "max_block_count": int(max_block_count),
            "min_improvement_ratio": float(min_improvement_ratio),
            "objective": "minimize estimated total comparators while preserving exact kth semantics",
        },
        "summary": {
            "stage_count": len(stage_plan),
            "blockwise_stage_count": blockwise_stage_count,
            "flat_stage_count": len(stage_plan) - blockwise_stage_count,
        },
        "stage_plan": stage_plan,
        "next_step": {
            "summary": "Run the bridge with selection-mode=blockwise_exact_kth, then compare checker / replay and SPU profile against flat_odd_even.",
            "not_yet_claimed": [
                "no secure runtime speedup claim from manifest alone",
                "no production default replacement implied by this file alone",
            ],
        },
    }


def render_markdown(manifest, output_json_path: Path):
    lines = [
        "# Blockwise exact-kth selection manifest",
        "",
        f"- 输出 JSON：`{output_json_path}`",
        f"- stage 总数：`{manifest['summary']['stage_count']}`",
        f"- blockwise stage 数：`{manifest['summary']['blockwise_stage_count']}`",
        "",
        "| Stage | Layer | Selection | Active | Keep | Candidate | Total comparator ratio |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for stage in manifest["stage_plan"]:
        selection = stage["stage_selection_kind"]
        candidate_count = stage.get("candidate_count")
        total_ratio = None
        if "expected_reduction_vs_current_flat" in stage:
            total_ratio = stage["expected_reduction_vs_current_flat"].get("total_comparator_ratio")
        lines.append(
            "| {stage} | {layer} | {selection} | {active} | {keep} | {candidate} | {ratio} |".format(
                stage=stage["stage_index"],
                layer=stage["pruning_layer"],
                selection=selection,
                active=stage["active_token_count"],
                keep=stage["keep_count"],
                candidate=candidate_count if candidate_count is not None else "N/A",
                ratio=f"{total_ratio:.4f}x" if total_ratio is not None else "N/A",
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a stage-selective blockwise exact-kth manifest for Transshield network-kth experiments."
    )
    parser.add_argument("--network-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--enabled-stage-indices", default="", help="Comma-separated stage indices to enable blockwise mode; empty means all stages.")
    parser.add_argument("--min-block-count", type=int, default=2)
    parser.add_argument("--max-block-count", type=int, default=4)
    parser.add_argument(
        "--min-improvement-ratio",
        type=float,
        default=0.0,
        help="Minimum estimated total comparator reduction ratio required to keep a stage in blockwise mode.",
    )
    args = parser.parse_args()

    network_manifest_path = Path(args.network_manifest_json).resolve()
    output_json = Path(args.output_json).resolve()
    enabled_stage_indices = parse_index_set(args.enabled_stage_indices)

    network_manifest = load_network_manifest(network_manifest_path)
    manifest = build_manifest(
        network_manifest,
        enabled_stage_indices,
        args.min_block_count,
        args.max_block_count,
        args.min_improvement_ratio,
    )
    manifest["source_network_manifest_json"] = str(network_manifest_path)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(manifest, output_json), encoding="utf-8")

    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
