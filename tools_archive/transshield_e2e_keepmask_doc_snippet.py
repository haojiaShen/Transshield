import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_float(value, digits=4):
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def run_label_from_path(run_dir: Path):
    name = run_dir.name
    if "keepmask_wholeforward_wrapper_spu_" in name:
        return name.split("keepmask_wholeforward_wrapper_spu_", 1)[1]
    return name


def build_snippet(summary: dict):
    run_dir = Path(summary["run_dir"])
    privacy = summary.get("privacy_fields") or {}
    compare = summary.get("compare_metrics") or {}
    keepmask = summary.get("keepmask_fields") or {}
    lines = []
    lines.append(f"- `{run_label_from_path(run_dir)}`：`{run_dir.as_posix()}/`")
    lines.append(f"  - `sample_count = {summary.get('sample_count')}`")
    lines.append(f"  - `elapsed_sec = {format_float(summary.get('elapsed_sec'), digits=4)}`")
    lines.append(
        "  - `logits/probabilities max_abs_error = "
        f"{format_float(compare.get('logits_max_abs_error'), digits=4)} / "
        f"{format_float(compare.get('probabilities_max_abs_error'), digits=4)}`"
    )
    lines.append(
        "  - `argmax / threshold match = "
        f"{format_float(compare.get('argmax_match_ratio'), digits=1)} / "
        f"{format_float(compare.get('threshold_match_ratio'), digits=1)}`"
    )
    lines.append(f"  - `input_pt = {privacy.get('input_pt')}`")
    lines.append(f"  - `input_mode = {privacy.get('input_mode')}`")
    lines.append(
        "  - `host_plaintext_pixel_values_materialized = "
        f"{str(privacy.get('host_plaintext_pixel_values_materialized')).lower()}`"
    )
    lines.append(
        "  - `host_private_share_tensors_loaded = "
        f"{str(privacy.get('host_private_share_tensors_loaded')).lower()}`"
    )
    lines.append(
        "  - `private_input_paths_redacted = "
        f"{str(privacy.get('private_input_paths_redacted')).lower()}`"
    )
    lines.append(f"  - `spu_params_mode = {privacy.get('spu_params_mode')}`")
    lines.append(f"  - `spu_forward_graph_mode = {privacy.get('spu_forward_graph_mode')}`")
    lines.append(f"  - `reveal_policy = {privacy.get('reveal_policy')}`")
    lines.append(
        "  - `runtime_pruning_keep_mask_stage_count = "
        f"{keepmask.get('runtime_pruning_keep_mask_stage_count')}`"
    )
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Render a keep-mask run summary into doc-ready Markdown snippet.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    summary = load_json(Path(args.summary_json).expanduser().resolve())
    snippet = build_snippet(summary)
    write_text(Path(args.output_md).expanduser().resolve(), snippet)
    print(snippet, end="")


if __name__ == "__main__":
    main()
