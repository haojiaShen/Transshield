import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_label_from_summary(summary: dict):
    run_name = Path(summary["run_dir"]).name
    if "keepmask_wholeforward_wrapper_spu_" in run_name:
        return run_name.split("keepmask_wholeforward_wrapper_spu_", 1)[1]
    return run_name


def fmt(value, digits=4):
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def bool_str(value):
    if value is None:
        return "null"
    return str(bool(value)).lower()


def build_common_privacy_lines(summary: dict):
    privacy = summary.get("privacy_fields") or {}
    keepmask = summary.get("keepmask_fields") or {}
    return [
        f"- `input_pt = {privacy.get('input_pt')}`",
        f"- `input_mode = {privacy.get('input_mode')}`",
        f"- `host_plaintext_pixel_values_materialized = {bool_str(privacy.get('host_plaintext_pixel_values_materialized'))}`",
        f"- `host_private_share_tensors_loaded = {bool_str(privacy.get('host_private_share_tensors_loaded'))}`",
        f"- `private_input_paths_redacted = {bool_str(privacy.get('private_input_paths_redacted'))}`",
        f"- `spu_params_mode = {privacy.get('spu_params_mode')}`",
        f"- `spu_forward_graph_mode = {privacy.get('spu_forward_graph_mode')}`",
        f"- `reveal_policy = {privacy.get('reveal_policy')}`",
        f"- `runtime_pruning_keep_mask_stage_count = {keepmask.get('runtime_pruning_keep_mask_stage_count')}`",
    ]


def build_run_lines(summary: dict):
    compare = summary.get("compare_metrics") or {}
    run_dir = Path(summary["run_dir"])
    return [
        f"- `{run_label_from_summary(summary)}`：`{run_dir.as_posix()}/`",
        f"  - `sample_count = {summary.get('sample_count')}`",
        f"  - `elapsed_sec = {fmt(summary.get('elapsed_sec'), digits=4)}`",
        (
            "  - `logits/probabilities max_abs_error = "
            f"{fmt(compare.get('logits_max_abs_error'), digits=4)} / "
            f"{fmt(compare.get('probabilities_max_abs_error'), digits=4)}`"
        ),
        (
            "  - `argmax / threshold match = "
            f"{fmt(compare.get('argmax_match_ratio'), digits=1)} / "
            f"{fmt(compare.get('threshold_match_ratio'), digits=1)}`"
        ),
    ]


def build_block(summaries):
    if not summaries:
        raise SystemExit("no summaries provided")
    lines = []
    lines.append("- 共同隐私边界：")
    for item in build_common_privacy_lines(summaries[0]):
        lines.append(f"  {item}")
    for summary in summaries:
        for item in build_run_lines(summary):
            lines.append(f"  {item}")
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Render multiple keep-mask run summaries into one doc block.")
    parser.add_argument("--summary-json", action="append", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    summaries = [load_json(Path(path).expanduser().resolve()) for path in args.summary_json]
    block = build_block(summaries)
    write_text(Path(args.output_md).expanduser().resolve(), block)
    print(block, end="")


if __name__ == "__main__":
    main()
