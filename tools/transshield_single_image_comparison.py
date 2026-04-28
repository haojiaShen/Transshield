import argparse
import importlib
import json
import sys
import types
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_plaintext_checkpoint_eval import (
    build_eval_transform,
    build_model,
    checkpoint_args_to_dict,
    find_threshold_json,
    load_json,
)
from tools.transshield_pruning_trace import collect_predictor_outputs, reconstruct_eval_masks
from tools.transshield_stage2_bundle import (
    load_frozen_bundle,
    postprocess_binary_output,
    preprocess_image,
    resolve_threshold,
)
from tools.transshield_token_pruning_visualization import (
    draw_mask_overlay,
    draw_original_panel,
    fit_display_image,
    load_font,
    make_contact_sheet,
    write_json,
    write_text,
)


def parse_class_names(raw_value: str):
    values = [item.strip() for item in raw_value.split(',') if item.strip()]
    if len(values) != 2:
        raise ValueError('--class-names must contain exactly two comma-separated names')
    return values


def label_name(class_names, index):
    if index is None:
        return None
    return class_names[index] if 0 <= index < len(class_names) else f'class_{index}'


def summarize_args_snapshot(args_snapshot):
    return {
        'model': args_snapshot.get('model'),
        'base_rate': float(args_snapshot.get('base_rate', 0.7)),
        'use_mask_pruning': bool(args_snapshot.get('use_mask_pruning', False)),
        'use_square_gelu': bool(args_snapshot.get('use_square_gelu', False)),
        'use_approx_attn': bool(args_snapshot.get('use_approx_attn', False)),
    }


def build_trace_report(source_key, source_value, image_path, logits, probs, trace, trace_style, args_snapshot, threshold):
    return {
        source_key: str(Path(source_value).resolve()),
        'image_path': str(image_path),
        'probabilities': [float(value) for value in probs.squeeze(0).detach().cpu().tolist()],
        'logits': [float(value) for value in logits.squeeze(0).detach().cpu().tolist()],
        **postprocess_binary_output(probs, threshold=threshold),
        'pruning_trace': trace,
        'trace_style': trace_style,
        'args_snapshot_summary': summarize_args_snapshot(args_snapshot),
    }


def import_repo_modules_fresh(repo_root: Path):
    if 'torch._six' not in sys.modules:
        torch_six = types.ModuleType('torch._six')
        torch_six.inf = math.inf
        sys.modules['torch._six'] = torch_six
    removed_modules = {}
    module_names = ['datasets', 'models', 'models.dyvit', 'utils']
    for module_name in module_names:
        if module_name in sys.modules:
            removed_modules[module_name] = sys.modules.pop(module_name)

    sys.path.insert(0, str(repo_root))
    try:
        datasets_mod = importlib.import_module('datasets')
        dyvit_mod = importlib.import_module('models.dyvit')
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)
        for module_name in ['datasets', 'models', 'models.dyvit', 'utils']:
            sys.modules.pop(module_name, None)
        for module_name, module_value in removed_modules.items():
            sys.modules[module_name] = module_value

    return datasets_mod, dyvit_mod


def build_dynamic_shape_trace(stage_token_counts):
    init_tokens = int(stage_token_counts[0]) if stage_token_counts else 0
    previous_count = init_tokens
    stages = []
    for stage_index, token_count in enumerate(stage_token_counts):
        stages.append({
            'stage_index': stage_index,
            'pruning_layer': [3, 6, 9][stage_index],
            'configured_keep_count': int(token_count),
            'active_before_per_sample': [float(previous_count)],
            'active_after_per_sample': [float(token_count)],
            'active_after_density_per_sample': [float(token_count / init_tokens) if init_tokens else 0.0],
            'trace_style': 'dynamic_shape_direct_pruning',
        })
        previous_count = token_count
    return {
        'init_spatial_tokens': init_tokens,
        'stages': stages,
        'trace_style': 'dynamic_shape_direct_pruning',
    }


def build_baseline_trace(predictor_outputs, base_rate):
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    stage_token_counts = [int(output.reshape(output.shape[0], -1, 2).shape[1]) for output in predictor_outputs]
    if stage_token_counts and all(count == stage_token_counts[0] for count in stage_token_counts):
        return reconstruct_eval_masks(predictor_outputs, token_ratio), 'mask_grid'
    return build_dynamic_shape_trace(stage_token_counts), 'dynamic_shape_direct_pruning'


def load_checkpoint_trace(repo_root, checkpoint_path, image_path, device='cpu', threshold_json=''):
    repo_root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    datasets_mod, dyvit_mod = import_repo_modules_fresh(repo_root)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args_snapshot = checkpoint_args_to_dict(checkpoint.get('args'))
    model = build_model(args_snapshot, dyvit_mod.VisionTransformerDiffPruning).to(device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, 'missing_keys', None) or getattr(load_result, 'unexpected_keys', None):
        raise ValueError(
            f'non-strict load result: missing={getattr(load_result, "missing_keys", None)} '
            f'unexpected={getattr(load_result, "unexpected_keys", None)}'
        )
    model.eval()

    transform = build_eval_transform(args_snapshot, datasets_mod.build_transform)
    image_path, input_tensor = preprocess_image(image_path, transform, device)
    logits, probs, predictor_outputs = collect_predictor_outputs(model, input_tensor)
    trace, trace_style = build_baseline_trace(
        predictor_outputs,
        float(args_snapshot.get('base_rate', 0.7)),
    )

    threshold = None
    threshold_path = find_threshold_json(checkpoint_path, threshold_json)
    if threshold_path is not None and threshold_path.exists():
        threshold_payload = load_json(threshold_path)
        threshold = float(threshold_payload['eval_binary_threshold'])

    report = build_trace_report(
        'checkpoint_path',
        checkpoint_path,
        image_path,
        logits,
        probs,
        trace,
        trace_style,
        args_snapshot,
        threshold,
    )
    report['repo_root'] = str(repo_root)
    return report


def build_modified_trace(bundle_dir, image_path, device='cpu'):
    bundle = load_frozen_bundle(bundle_dir, device)
    threshold = resolve_threshold(Path(bundle_dir).resolve(), None)
    image_path, input_tensor = preprocess_image(image_path, bundle['transform'], device)
    logits, probs, predictor_outputs = collect_predictor_outputs(bundle['model'], input_tensor)
    base_rate = float(bundle['args_snapshot']['base_rate'])
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    trace = reconstruct_eval_masks(predictor_outputs, token_ratio)
    return build_trace_report(
        'bundle_dir',
        bundle_dir,
        image_path,
        logits,
        probs,
        trace,
        'mask_grid',
        bundle['args_snapshot'],
        threshold,
    )


def build_prediction_lines(title, report, class_names):
    probs = report['probabilities']
    return [
        title,
        f"argmax: {label_name(class_names, report['argmax_class'])}",
        f"threshold: {label_name(class_names, report.get('threshold_class'))}",
        f"prob_0: {probs[0]:.6f}",
        f"prob_1: {probs[1]:.6f}",
        f"base_rate: {report['args_snapshot_summary']['base_rate']:.3f}",
        f"use_mask_pruning: {report['args_snapshot_summary']['use_mask_pruning']}",
    ]


def draw_prediction_panel(title, report, class_names, width, height):
    panel = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(panel)
    title_font = load_font(18)
    body_font = load_font(14)
    lines = build_prediction_lines(title, report, class_names)
    y = 12
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        draw.text((12, y), line, fill='black', font=font)
        y += 28 if index == 0 else 22
    return panel


def build_stage_overlay_title(prefix, stage_index, stage):
    return f"{prefix} Stage {stage_index + 1} | layer {stage['pruning_layer']}"


def build_stage_overlay_subtitle(stage, init_tokens):
    return f"keep={int(stage['active_after_per_sample'][0])}/{init_tokens}"


def draw_stage_count_panel(title, stage, total_tokens, width, height):
    panel = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(panel)
    title_font = load_font(18)
    body_font = load_font(14)
    keep_count = int(stage['active_after_per_sample'][0])
    pruned_count = int(stage['active_before_per_sample'][0] - stage['active_after_per_sample'][0])
    density = float(stage['active_after_density_per_sample'][0])
    lines = [
        title,
        f"layer: {stage['pruning_layer']}",
        f"keep: {keep_count}/{total_tokens}",
        f"pruned: {pruned_count}",
        f"density: {density:.4f}",
        "style: direct pruning",
    ]
    y = 12
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        draw.text((12, y), line, fill='black', font=font)
        y += 28 if index == 0 else 22
    return panel


def build_report_stage_panel(prefix, report, stage, stage_index, base_image, panel_size):
    width, height = panel_size
    if report['trace_style'] == 'mask_grid' and 'first_sample_mask_grid' in stage:
        return draw_mask_overlay(
            base_image=base_image,
            mask_grid=stage['first_sample_mask_grid'],
            title=build_stage_overlay_title(prefix, stage_index, stage),
            subtitle=build_stage_overlay_subtitle(stage, report['pruning_trace']['init_spatial_tokens']),
        )
    return draw_stage_count_panel(
        title=f'{prefix} Stage {stage_index + 1}',
        stage=stage,
        total_tokens=report['pruning_trace']['init_spatial_tokens'],
        width=width,
        height=height,
    )


def build_stage_summary_rows(report, include_representation=False):
    rows = []
    for stage in report['pruning_trace']['stages']:
        row = [
            f"stage_{stage['stage_index'] + 1}",
            str(stage['pruning_layer']),
            str(int(stage['active_after_per_sample'][0])),
            f"{stage['active_after_density_per_sample'][0]:.4f}",
        ]
        if include_representation:
            row.append(report['trace_style'])
        rows.append(row)
    return rows


def render_markdown(image_path, output_dir, class_names, baseline_report, modified_report):
    lines = [
        '# Baseline vs Modified Single-Image Comparison',
        '',
        f"- Image: `{image_path}`",
        f"- Summary board: `{(output_dir / 'baseline_vs_modified_summary.png').resolve()}`",
        f"- Comparison JSON: `{(output_dir / 'baseline_vs_modified_comparison.json').resolve()}`",
        f"- Class names: `{', '.join(class_names)}`",
        '',
        '## Prediction summary',
        '',
        '| Model | Argmax | Threshold | prob_0 | prob_1 |',
        '|---|---|---|---:|---:|',
        f"| baseline | {label_name(class_names, baseline_report['argmax_class'])} | "
        f"{label_name(class_names, baseline_report.get('threshold_class'))} | "
        f"{baseline_report['probabilities'][0]:.6f} | {baseline_report['probabilities'][1]:.6f} |",
        f"| modified | {label_name(class_names, modified_report['argmax_class'])} | "
        f"{label_name(class_names, modified_report.get('threshold_class'))} | "
        f"{modified_report['probabilities'][0]:.6f} | {modified_report['probabilities'][1]:.6f} |",
        '',
        '## Modified stage summary',
        '',
        '| Stage | Layer | Keep Count | Keep Density |',
        '|---|---:|---:|---:|',
    ]
    for row in build_stage_summary_rows(modified_report):
        lines.append(f'| {" | ".join(row)} |')
    lines.extend(
        [
            '',
            '## Baseline stage summary',
            '',
            '| Stage | Layer | Keep Count | Keep Density | Representation |',
            '|---|---:|---:|---:|---|',
        ]
    )
    for row in build_stage_summary_rows(baseline_report, include_representation=True):
        lines.append(f'| {" | ".join(row)} |')
    lines.extend(
        [
            '',
            '## How to explain',
            '',
            '- baseline 面板给出原始参考模型对这张图的输出，以及 direct pruning 的阶段保留数量。',
            '- modified 面板给出当前主模型的输出与 stagewise token masking。',
            '- 这页图最适合解释：baseline 更接近直接裁剪，而 modified 改写成了 masking 表达。',
            '- 如果需要强调 secure 友好性，重点解释 modified 采用的是 masking，而不是直接裁剪 token。',
            '',
        ]
    )
    return '\n'.join(lines)


def build_summary_panels(image_path, image_size, class_names, baseline_report, modified_report):
    base_image = fit_display_image(image_path, image_size)
    original_panel = draw_original_panel(
        base_image,
        title='Original Image',
        subtitle='Single-image comparison input',
    )
    panel_size = (original_panel.width, original_panel.height)
    panels = [
        original_panel,
        draw_prediction_panel('Baseline Prediction', baseline_report, class_names, *panel_size),
        draw_prediction_panel('Modified Prediction', modified_report, class_names, *panel_size),
    ]
    for stage_index, modified_stage in enumerate(modified_report['pruning_trace']['stages']):
        baseline_stage = baseline_report['pruning_trace']['stages'][stage_index]
        panels.append(
            build_report_stage_panel(
                'Baseline',
                baseline_report,
                baseline_stage,
                stage_index,
                base_image,
                panel_size,
            )
        )
        panels.append(
            build_report_stage_panel(
                'Modified',
                modified_report,
                modified_stage,
                stage_index,
                base_image,
                panel_size,
            )
        )
    return panels


def build_parser():
    parser = argparse.ArgumentParser(description='Generate a baseline-vs-modified single-image comparison board.')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--baseline-repo-root', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--baseline-threshold-json', default='')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--class-names', default='class_0,class_1')
    return parser


def main():
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = parse_class_names(args.class_names)

    baseline_report = load_checkpoint_trace(
        repo_root=args.baseline_repo_root,
        checkpoint_path=args.baseline_checkpoint,
        image_path=args.image_path,
        device=args.device,
        threshold_json=args.baseline_threshold_json,
    )
    modified_report = build_modified_trace(
        bundle_dir=args.bundle_dir,
        image_path=args.image_path,
        device=args.device,
    )

    comparison = {
        'image_path': str(Path(args.image_path).resolve()),
        'class_names': class_names,
        'baseline': baseline_report,
        'modified': modified_report,
    }
    write_json(output_dir / 'baseline_vs_modified_comparison.json', comparison)

    panels = build_summary_panels(
        Path(args.image_path).resolve(),
        args.image_size,
        class_names,
        baseline_report,
        modified_report,
    )
    summary_board = make_contact_sheet(panels, columns=3)
    summary_board.save(output_dir / 'baseline_vs_modified_summary.png')
    write_text(
        output_dir / 'baseline_vs_modified_comparison.md',
        render_markdown(Path(args.image_path).resolve(), output_dir, class_names, baseline_report, modified_report) + '\n',
    )
    print(json.dumps({
        'comparison_json': str((output_dir / 'baseline_vs_modified_comparison.json').resolve()),
        'summary_png': str((output_dir / 'baseline_vs_modified_summary.png').resolve()),
        'markdown': str((output_dir / 'baseline_vs_modified_comparison.md').resolve()),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
