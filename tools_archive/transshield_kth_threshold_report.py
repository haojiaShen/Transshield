import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_frozen_bundle
from tools.transshield_stagewise_threshold_report import collect_stagewise_scores


def summarize_tensor(values):
    values = values.detach().float().cpu()
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
    }


def mask_summary(lhs, rhs):
    lhs = lhs.bool()
    rhs = rhs.bool()
    exact_count_match = (lhs.sum(dim=1) == rhs.sum(dim=1)).float().mean().item()
    exact_mask_match = (lhs == rhs).float().mean().item()
    intersection = (lhs & rhs).sum(dim=1).float()
    union = (lhs | rhs).sum(dim=1).float().clamp_min(1.0)
    jaccard = (intersection / union).mean().item()
    return {
        'exact_count_match_ratio': float(exact_count_match),
        'exact_mask_match_ratio': float(exact_mask_match),
        'mean_jaccard_vs_topk': float(jaccard),
        'mean_lhs_keep_count': float(lhs.sum(dim=1).float().mean().item()),
        'mean_rhs_keep_count': float(rhs.sum(dim=1).float().mean().item()),
    }


def summarize_stage(stage_payload):
    score = stage_payload['score'].float()
    active_before = stage_payload['active_before'].bool()
    topk_mask = stage_payload['topk_mask'].bool()
    keep_count = int(stage_payload['keep_count'])
    kth_threshold = stage_payload['boundary_keep_score'].float().unsqueeze(1)

    ge_mask = (score >= kth_threshold) & active_before
    gt_mask = (score > kth_threshold) & active_before
    eq_mask = (score == kth_threshold) & active_before

    gt_count = gt_mask.sum(dim=1)
    eq_count = eq_mask.sum(dim=1)
    ge_count = ge_mask.sum(dim=1)
    tie_keep_quota = keep_count - gt_count
    overflow_if_plain_ge = ge_count - keep_count
    boundary_tie_present = eq_count > tie_keep_quota

    selected_equal_from_topk = topk_mask & eq_mask
    selected_equal_count = selected_equal_from_topk.sum(dim=1)

    return {
        'stage_index': int(stage_payload['stage_index']),
        'pruning_layer': int(stage_payload['pruning_layer']),
        'keep_count': keep_count,
        'active_count_before': int(active_before.sum(dim=1)[0].item()),
        'kth_threshold_summary': summarize_tensor(kth_threshold.squeeze(1)),
        'plain_score_ge_kth_metrics': mask_summary(ge_mask, topk_mask),
        'strict_score_gt_kth_metrics': {
            'mean_strict_keep_count': float(gt_count.float().mean().item()),
            'min_strict_keep_count': int(gt_count.min().item()),
            'max_strict_keep_count': int(gt_count.max().item()),
        },
        'tie_statistics': {
            'boundary_tie_sample_ratio': float(boundary_tie_present.float().mean().item()),
            'mean_equal_token_count_at_boundary': float(eq_count.float().mean().item()),
            'max_equal_token_count_at_boundary': int(eq_count.max().item()),
            'mean_tie_keep_quota': float(tie_keep_quota.float().mean().item()),
            'min_tie_keep_quota': int(tie_keep_quota.min().item()),
            'max_tie_keep_quota': int(tie_keep_quota.max().item()),
            'mean_selected_equal_count_from_topk': float(selected_equal_count.float().mean().item()),
            'mean_overflow_if_plain_ge': float(overflow_if_plain_ge.float().mean().item()),
            'max_overflow_if_plain_ge': int(overflow_if_plain_ge.max().item()),
        },
        'design_contract': {
            'masked_score_source': 'score.masked_fill(~active_before, -inf)',
            'kth_threshold_tensor': {
                'shape': ['B', 1],
                'definition': 'per-sample kth kept score from the current top-k reference',
            },
            'primary_compare': 'score > kth_threshold',
            'boundary_compare': 'score == kth_threshold',
            'tie_keep_quota': 'keep_count - count(score > kth_threshold)',
            'final_note': 'plain score >= kth_threshold is already top-k-compatible enough for stage-2 design, but exact top-k reproduction still needs tie handling when boundary ties exist',
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Emit a read-only stage-2 kth-threshold design report against the frozen Transshield top-k pruning reference.'
    )
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

    stage_reports = [summarize_stage(stage_payload) for stage_payload in aggregated]
    report = {
        'bundle_dir': str(bundle_dir),
        'data_path': str(data_path),
        'device': args.device,
        'sample_count': total_samples,
        'design_status': {
            'topk_compatible_kth_threshold_reference_available': True,
            'frozen_baseline_replaced': False,
            'report_is_read_only': True,
            'secure_kth_selection_still_required': True,
        },
        'design_summary': {
            'pruning_keep_generation_can_be_factored_as': [
                'masked_score from predictor score and active mask',
                'per-sample kth threshold extraction',
                'pointwise compare against kth threshold',
                'boundary tie handling when equal-score ties exist',
                'mask application via F_mux-style token masking',
            ],
            'current_best_use': 'treat this as a stage-2 design reference for future secure selection work, not as a promoted replacement for the frozen top-k baseline',
        },
        'stage_reports': stage_reports,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
