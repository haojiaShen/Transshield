import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_frozen_bundle


@torch.no_grad()
def collect_stagewise_scores(model, inputs):
    model.eval()
    batch_size = inputs.shape[0]
    x = model.patch_embed(inputs)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    init_n = x.shape[1] - 1
    prev_decision = torch.ones(batch_size, init_n, 1, dtype=x.dtype, device=x.device)
    policy = torch.ones(batch_size, init_n + 1, 1, dtype=x.dtype, device=x.device)

    stage_reports = []
    stage_index = 0
    for block_index, blk in enumerate(model.blocks):
        if block_index in model.pruning_loc:
            if model.use_mask_pruning:
                x = model._apply_spatial_mask(x, prev_decision)
            spatial_x = x[:, 1:]
            pred_score = model.score_predictor[stage_index](spatial_x, prev_decision).reshape(batch_size, -1, 2)
            score = pred_score[:, :, 0]
            active_before = prev_decision.squeeze(-1) > 0
            masked_score = score.masked_fill(~active_before, float('-inf'))
            keep_count = int(init_n * model.token_ratio[stage_index])
            keep_policy = torch.argsort(masked_score, dim=1, descending=True)[:, :keep_count]
            new_mask = torch.zeros_like(prev_decision)
            new_mask.scatter_(1, keep_policy.unsqueeze(-1), 1.0)
            updated_mask = new_mask * prev_decision

            kept_scores = torch.gather(score, 1, keep_policy)
            boundary_keep_score = kept_scores[:, -1]

            next_score = []
            for sample_index in range(batch_size):
                active_scores = score[sample_index][active_before[sample_index]]
                sorted_scores, _ = torch.sort(active_scores, descending=True)
                if keep_count < sorted_scores.numel():
                    next_score.append(float(sorted_scores[keep_count].item()))
                else:
                    next_score.append(float('-inf'))

            stage_reports.append(
                {
                    'stage_index': stage_index,
                    'pruning_layer': int(block_index),
                    'keep_count': keep_count,
                    'active_count_before': [int(v) for v in active_before.sum(dim=1).tolist()],
                    'score': score.detach().cpu(),
                    'active_before': active_before.detach().cpu(),
                    'topk_mask': updated_mask.squeeze(-1).detach().cpu() > 0,
                    'boundary_keep_score': boundary_keep_score.detach().cpu(),
                    'boundary_next_score': torch.tensor(next_score, dtype=torch.float32),
                }
            )

            prev_decision = updated_mask
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
            cls_policy = torch.ones(batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
            stage_index += 1
        else:
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

    return stage_reports


def compute_global_threshold(active_scores, keep_fraction):
    if active_scores.numel() == 0:
        return float('nan')
    sorted_scores, _ = torch.sort(active_scores, descending=True)
    keep_total = int(round(sorted_scores.numel() * keep_fraction))
    keep_total = max(1, min(keep_total, sorted_scores.numel()))
    return float(sorted_scores[keep_total - 1].item())


def summarize_stage(stage_payload):
    score = stage_payload['score']
    active_before = stage_payload['active_before']
    topk_mask = stage_payload['topk_mask']
    boundary_keep_score = stage_payload['boundary_keep_score']
    boundary_next_score = stage_payload['boundary_next_score']

    active_scores = score[active_before]
    keep_fraction = stage_payload['keep_count'] / float(active_before.sum(dim=1)[0].item())
    global_threshold = compute_global_threshold(active_scores, keep_fraction)

    threshold_mask = (score >= global_threshold) & active_before
    topk_mask_bool = topk_mask.bool()

    per_sample_threshold_keep = threshold_mask.sum(dim=1)
    per_sample_topk_keep = topk_mask_bool.sum(dim=1)
    exact_count_match = (per_sample_threshold_keep == per_sample_topk_keep).float().mean().item()
    exact_mask_match = (threshold_mask == topk_mask_bool).float().mean().item()

    intersection = (threshold_mask & topk_mask_bool).sum(dim=1).float()
    union = (threshold_mask | topk_mask_bool).sum(dim=1).float().clamp_min(1.0)
    jaccard = (intersection / union).mean().item()

    false_keep = ((threshold_mask == 1) & (topk_mask_bool == 0)).sum(dim=1).float().mean().item()
    false_drop = ((threshold_mask == 0) & (topk_mask_bool == 1)).sum(dim=1).float().mean().item()

    margin = boundary_keep_score - boundary_next_score
    finite_next = torch.isfinite(boundary_next_score)
    greater_mask = (score > boundary_keep_score.unsqueeze(1)) & active_before
    equal_mask = (score == boundary_keep_score.unsqueeze(1)) & active_before
    tie_keep_quota = stage_payload['keep_count'] - greater_mask.sum(dim=1)
    tie_excess = (equal_mask.sum(dim=1) - tie_keep_quota).clamp_min(0)
    score_std = active_scores.std(unbiased=False)
    finite_margin = margin[finite_next]
    margin_percentiles = None
    margin_small_ratio_abs = None
    margin_small_ratio_rel = None
    if finite_margin.numel() > 0:
        margin_percentiles = {
            'p10': float(torch.quantile(finite_margin, 0.10).item()),
            'p50': float(torch.quantile(finite_margin, 0.50).item()),
            'p90': float(torch.quantile(finite_margin, 0.90).item()),
        }
        margin_small_ratio_abs = {
            'lte_1e-05': float((finite_margin <= 1e-5).float().mean().item()),
            'lte_1e-04': float((finite_margin <= 1e-4).float().mean().item()),
            'lte_1e-03': float((finite_margin <= 1e-3).float().mean().item()),
        }
        std_scale = float(score_std.item())
        if std_scale > 0:
            normalized_margin = finite_margin / std_scale
            margin_small_ratio_rel = {
                'lte_0p01_score_std': float((normalized_margin <= 0.01).float().mean().item()),
                'lte_0p05_score_std': float((normalized_margin <= 0.05).float().mean().item()),
                'lte_0p10_score_std': float((normalized_margin <= 0.10).float().mean().item()),
            }

    active_counts = active_before.sum(dim=1).float()
    estimated_score_bytes = active_counts.sum().item() * 4.0

    return {
        'stage_index': stage_payload['stage_index'],
        'pruning_layer': stage_payload['pruning_layer'],
        'keep_count': int(stage_payload['keep_count']),
        'active_count_before': int(active_before.sum(dim=1)[0].item()),
        'relative_keep_fraction': keep_fraction,
        'global_threshold_candidate': global_threshold,
        'topk_boundary_keep_score_mean': float(boundary_keep_score.mean().item()),
        'topk_boundary_keep_score_std': float(boundary_keep_score.std(unbiased=False).item()),
        'topk_boundary_next_score_mean': float(boundary_next_score[finite_next].mean().item()) if finite_next.any() else None,
        'topk_boundary_margin_mean': float(margin[finite_next].mean().item()) if finite_next.any() else None,
        'topk_boundary_margin_std': float(margin[finite_next].std(unbiased=False).item()) if finite_next.any() else None,
        'topk_boundary_margin_percentiles': margin_percentiles,
        'topk_boundary_small_margin_ratio_abs': margin_small_ratio_abs,
        'topk_boundary_small_margin_ratio_rel': margin_small_ratio_rel,
        'global_threshold_eval': {
            'mean_threshold_keep_count': float(per_sample_threshold_keep.float().mean().item()),
            'std_threshold_keep_count': float(per_sample_threshold_keep.float().std(unbiased=False).item()),
            'exact_count_match_ratio': float(exact_count_match),
            'exact_mask_match_ratio': float(exact_mask_match),
            'mean_jaccard_vs_topk': float(jaccard),
            'mean_false_keep_count': float(false_keep),
            'mean_false_drop_count': float(false_drop),
        },
        'score_distribution': {
            'min': float(active_scores.min().item()),
            'max': float(active_scores.max().item()),
            'mean': float(active_scores.mean().item()),
            'std': float(active_scores.std(unbiased=False).item()),
        },
        'protocol_risk_signals': {
            'boundary_tie_sample_ratio': float((equal_mask.sum(dim=1) > tie_keep_quota).float().mean().item()),
            'mean_boundary_equal_count': float(equal_mask.sum(dim=1).float().mean().item()),
            'max_boundary_equal_count': int(equal_mask.sum(dim=1).max().item()),
            'mean_tie_keep_quota': float(tie_keep_quota.float().mean().item()),
            'mean_tie_excess_count': float(tie_excess.float().mean().item()),
            'max_tie_excess_count': int(tie_excess.max().item()),
        },
        'payload_estimate': {
            'estimated_active_score_bytes_float32': float(estimated_score_bytes),
            'estimated_active_score_bytes_per_sample_float32': float((estimated_score_bytes / max(1, score.shape[0]))),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Compute read-only stage-wise threshold candidates against the frozen top-k pruning reference.')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--data-path', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    data_path = Path(args.data_path).resolve() if args.data_path else Path(bundle['args_snapshot']['eval_data_path']).resolve()

    dataset = ImageFolder(root=str(data_path), transform=bundle['transform'])
    if args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.imgs = dataset.imgs[: args.max_samples]
        dataset.targets = dataset.targets[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    aggregated = None
    total_samples = 0
    for inputs, _targets in loader:
        inputs = inputs.to(args.device)
        batch_reports = collect_stagewise_scores(bundle['model'], inputs)
        if aggregated is None:
            aggregated = batch_reports
        else:
            for stage_index, stage_payload in enumerate(batch_reports):
                aggregated_stage = aggregated[stage_index]
                for key in ['score', 'active_before', 'topk_mask', 'boundary_keep_score', 'boundary_next_score']:
                    aggregated_stage[key] = torch.cat([aggregated_stage[key], stage_payload[key]], dim=0)
                aggregated_stage['active_count_before'].extend(stage_payload['active_count_before'])
        total_samples += inputs.shape[0]

    if aggregated is None:
        raise ValueError('no samples processed')

    stage_summaries = [summarize_stage(stage_payload) for stage_payload in aggregated]
    report = {
        'bundle_dir': str(bundle_dir),
        'data_path': str(data_path),
        'device': args.device,
        'sample_count': total_samples,
        'reference_semantics': {
            'topk_pruning_preserved': True,
            'model_semantics_changed': False,
            'report_is_read_only': True,
        },
        'stage_summaries': stage_summaries,
        'recommendation': {
            'ready_for_pruning_f_less_design': True,
            'ready_to_replace_topk_with_global_threshold_now': False,
            'next_step': 'Use these stage-wise threshold statistics as read-only calibration evidence before any margin-aware training or kth protocol rewrite.',
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
