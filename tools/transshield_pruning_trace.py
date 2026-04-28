import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import (
    infer_logits,
    input_tensor_stats,
    load_frozen_bundle,
    postprocess_binary_output,
    preprocess_image,
    resolve_threshold,
)


def tensor_stats(tensor):
    tensor = tensor.detach().float().cpu()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {
            'shape': list(tensor.shape),
            'finite_count': 0,
        }
    return {
        'shape': list(tensor.shape),
        'finite_count': int(finite.numel()),
        'min': float(finite.min().item()),
        'max': float(finite.max().item()),
        'mean': float(finite.mean().item()),
        'std': float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
    }


def infer_token_grid_size(token_count: int):
    grid_size = int(token_count ** 0.5)
    if grid_size * grid_size != token_count:
        raise ValueError(f'expected square token grid, got token_count={token_count}')
    return grid_size


def collect_predictor_outputs(model, input_tensor):
    predictor_outputs = []
    hooks = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            predictor_outputs.append((index, output.detach().cpu()))
        return hook

    for index, predictor in enumerate(model.score_predictor):
        hooks.append(predictor.register_forward_hook(make_hook(index)))

    try:
        logits, probs = infer_logits(model, input_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    predictor_outputs.sort(key=lambda item: item[0])
    return logits, probs, [output for _, output in predictor_outputs]


def f_mux_equivalence(mask):
    mask = mask.detach().float()
    token_values = torch.linspace(
        -1.0, 1.0, steps=mask.shape[1] * 4, dtype=mask.dtype, device=mask.device
    ).reshape(mask.shape[0], mask.shape[1], 4)
    multiply_form = token_values * mask
    mux_form = torch.where(mask > 0.5, token_values, torch.zeros_like(token_values))
    return float((multiply_form - mux_form).abs().max().item())


def reconstruct_eval_masks(predictor_outputs, token_ratio):
    if not predictor_outputs:
        raise ValueError('no score_predictor outputs captured')

    first_output = predictor_outputs[0].reshape(predictor_outputs[0].shape[0], -1, 2)
    batch_size = first_output.shape[0]
    init_tokens = first_output.shape[1]
    grid_size = infer_token_grid_size(init_tokens)
    prev_decision = torch.ones(batch_size, init_tokens, 1)
    stage_reports = []

    for stage_index, pred_score_raw in enumerate(predictor_outputs):
        pred_score = pred_score_raw.reshape(batch_size, -1, 2).float()
        score = pred_score[:, :, 0]
        active_before = prev_decision.sum(dim=1).squeeze(-1)
        masked_score = score.masked_fill(prev_decision.squeeze(-1) == 0, float('-inf'))
        keep_count = int(init_tokens * float(token_ratio[stage_index]))
        keep_policy = torch.argsort(masked_score, dim=1, descending=True)[:, :keep_count]
        new_mask = torch.zeros_like(prev_decision)
        new_mask.scatter_(1, keep_policy.unsqueeze(-1), 1.0)
        prev_decision = new_mask * prev_decision
        active_after = prev_decision.sum(dim=1).squeeze(-1)
        first_indices = keep_policy[0, : min(12, keep_policy.shape[1])].tolist()
        first_sample_keep_indices = keep_policy[0].detach().cpu().tolist()
        first_sample_mask = prev_decision[0, :, 0].detach().cpu().tolist()
        first_sample_pruned_indices = [
            index for index, keep_flag in enumerate(first_sample_mask) if float(keep_flag) < 0.5
        ]
        first_sample_mask_grid = [
            [int(round(float(value))) for value in first_sample_mask[row_start: row_start + grid_size]]
            for row_start in range(0, init_tokens, grid_size)
        ]

        stage_reports.append({
            'stage_index': stage_index,
            'pruning_layer': [3, 6, 9][stage_index],
            'token_ratio': float(token_ratio[stage_index]),
            'token_grid_size': grid_size,
            'configured_keep_count': keep_count,
            'active_before_per_sample': [float(value) for value in active_before.tolist()],
            'active_after_per_sample': [float(value) for value in active_after.tolist()],
            'active_after_density_per_sample': [
                float(value / init_tokens) for value in active_after.tolist()
            ],
            'score_stats': tensor_stats(score),
            'mask_stats': tensor_stats(prev_decision),
            'first_sample_keep_indices_preview': [int(value) for value in first_indices],
            'first_sample_keep_indices': [int(value) for value in first_sample_keep_indices],
            'first_sample_pruned_indices': [int(value) for value in first_sample_pruned_indices],
            'first_sample_mask_grid': first_sample_mask_grid,
            'f_mux_equivalence_max_abs_error': f_mux_equivalence(prev_decision),
        })

    return {
        'init_spatial_tokens': init_tokens,
        'token_grid_size': grid_size,
        'stages': stage_reports,
        'final_mask_stats': tensor_stats(prev_decision),
    }


def build_trace_report(bundle_dir, image_path, device='cpu', threshold_override=None):
    bundle_dir = Path(bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, device)
    threshold = resolve_threshold(bundle_dir, threshold_override)
    image_path, input_tensor = preprocess_image(image_path, bundle['transform'], device)

    logits, probs, predictor_outputs = collect_predictor_outputs(bundle['model'], input_tensor)
    token_ratio = [
        float(bundle['args_snapshot']['base_rate']),
        float(bundle['args_snapshot']['base_rate']) ** 2,
        float(bundle['args_snapshot']['base_rate']) ** 3,
    ]
    mask_report = reconstruct_eval_masks(predictor_outputs, token_ratio)
    prediction_summary = postprocess_binary_output(probs, threshold=threshold)

    return {
        'bundle_dir': str(bundle_dir),
        'image_path': str(image_path),
        'device': device,
        'input_shape': list(input_tensor.shape),
        'input_stats': input_tensor_stats(input_tensor),
        'logits': [float(value) for value in logits.squeeze(0).detach().cpu().tolist()],
        **prediction_summary,
        'pruning_trace': mask_report,
        'scheme_c_status': {
            'model_semantics_changed': False,
            'kept_current_topk_reference': True,
            'f_mux_checked_as_mask_multiply_equivalent': True,
            'pruning_keep_generation_rewritten_to_f_less': False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Trace frozen Transshield top-k pruning masks without changing model semantics.')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    report = build_trace_report(
        bundle_dir=args.bundle_dir,
        image_path=args.image_path,
        device=args.device,
        threshold_override=args.threshold,
    )

    report_text = json.dumps(report, indent=2, sort_keys=True)
    print(report_text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(report_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
