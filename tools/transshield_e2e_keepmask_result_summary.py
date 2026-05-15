import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pick(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_summary(run_dir: Path, candidate: dict, compare: dict):
    sample_count = candidate.get("sample_count", compare.get("sample_count"))
    elapsed_sec = candidate.get("elapsed_sec")
    reveal_policy = candidate.get("reveal_policy", pick(candidate, "spu", "reveal_policy"))
    if reveal_policy is None and candidate.get("runtime") == "spu":
        reveal_policy = "final_logits_only"
    return {
        "manifest_type": "transshield_e2e_keepmask_result_summary_v0",
        "run_dir": str(run_dir),
        "sample_count": sample_count,
        "runtime": candidate.get("runtime"),
        "backend": candidate.get("backend"),
        "elapsed_sec": elapsed_sec,
        "sec_per_sample": (
            float(elapsed_sec) / int(sample_count)
            if elapsed_sec is not None and sample_count not in (None, 0)
            else None
        ),
        "finite_logits": candidate.get("finite_logits"),
        "privacy_fields": {
            "input_pt": candidate.get("input_pt"),
            "input_mode": candidate.get("input_mode", pick(candidate, "spu", "input_mode")),
            "host_plaintext_pixel_values_materialized": candidate.get(
                "host_plaintext_pixel_values_materialized",
                pick(candidate, "spu", "host_plaintext_pixel_values_materialized"),
            ),
            "host_private_share_tensors_loaded": candidate.get(
                "host_private_share_tensors_loaded",
                pick(candidate, "spu", "host_private_share_tensors_loaded"),
            ),
            "private_input_paths_redacted": candidate.get(
                "private_input_paths_redacted",
                pick(candidate, "spu", "private_input_paths_redacted"),
            ),
            "reveal_policy": reveal_policy,
            "spu_params_mode": candidate.get("spu_params_mode", pick(candidate, "spu", "spu_params_mode")),
            "spu_forward_graph_mode": candidate.get(
                "spu_forward_graph_mode",
                pick(candidate, "spu", "spu_forward_graph_mode"),
            ),
        },
        "keepmask_fields": {
            "runtime_pruning_keep_mask_pt": candidate.get("runtime_pruning_keep_mask_pt"),
            "runtime_pruning_keep_mask_stage_count": candidate.get("runtime_pruning_keep_mask_stage_count"),
        },
        "compare_metrics": {
            "argmax_match_ratio": pick(compare, "prediction_match", "argmax_match_ratio"),
            "threshold_match_ratio": pick(compare, "prediction_match", "threshold_match_ratio"),
            "logits_max_abs_error": pick(compare, "logits_error", "max_abs_error"),
            "logits_mean_abs_error": pick(compare, "logits_error", "mean_abs_error"),
            "probabilities_max_abs_error": pick(compare, "probabilities_error", "max_abs_error"),
            "probabilities_mean_abs_error": pick(compare, "probabilities_error", "mean_abs_error"),
        },
        "files": {
            "candidate_json": str(run_dir / "e2e_static_whole_forward_candidate_from_server.json"),
            "compare_json": str(run_dir / "e2e_static_whole_forward_compare.json"),
            "runtime_pruning_reference_json": str(run_dir / "runtime_pruning_reference.json"),
            "runtime_pruning_keep_mask_json": str(run_dir / "runtime_pruning_keep_mask_payload.json"),
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize a keep-mask whole-forward SPU run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--candidate-json-name",
        default="e2e_static_whole_forward_candidate_from_server.json",
    )
    parser.add_argument(
        "--compare-json-name",
        default="e2e_static_whole_forward_compare.json",
    )
    parser.add_argument("--output-json", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    candidate_path = run_dir / args.candidate_json_name
    compare_path = run_dir / args.compare_json_name
    if not candidate_path.is_file():
        raise SystemExit(f"missing candidate JSON: {candidate_path}")
    if not compare_path.is_file():
        raise SystemExit(f"missing compare JSON: {compare_path}")
    candidate = load_json(candidate_path)
    compare = load_json(compare_path)
    payload = build_summary(run_dir=run_dir, candidate=candidate, compare=compare)
    write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
