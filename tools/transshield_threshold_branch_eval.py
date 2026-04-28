import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_frozen_bundle, load_json, resolve_threshold


def accuracy_from_logits(logits, targets):
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item() * 100.0)


def threshold_accuracy_from_logits(logits, targets, threshold):
    probs = torch.softmax(logits, dim=-1)
    pred = (probs[:, 1] >= threshold).long()
    return float((pred == targets).float().mean().item() * 100.0)


def binary_auc(scores, targets):
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().long()
    pos = scores[targets == 1]
    neg = scores[targets == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[targets == 1]
    auc = (pos_ranks.sum() - pos.numel() * (pos.numel() + 1) / 2.0) / (pos.numel() * neg.numel())
    return float(auc.item())


def mask_metrics(threshold_mask, topk_mask):
    threshold_mask = threshold_mask.bool()
    topk_mask = topk_mask.bool()
    exact_count_match = (threshold_mask.sum(dim=1) == topk_mask.sum(dim=1)).float().mean().item()
    exact_mask_match = (threshold_mask == topk_mask).float().mean().item()
    intersection = (threshold_mask & topk_mask).sum(dim=1).float()
    union = (threshold_mask | topk_mask).sum(dim=1).float().clamp_min(1.0)
    jaccard = (intersection / union).mean().item()
    return {
        'exact_count_match_ratio': float(exact_count_match),
        'exact_mask_match_ratio': float(exact_mask_match),
        'mean_jaccard_vs_topk': float(jaccard),
        'mean_threshold_keep_count': float(threshold_mask.sum(dim=1).float().mean().item()),
        'mean_topk_keep_count': float(topk_mask.sum(dim=1).float().mean().item()),
    }


def tensor_summary(values):
    values = values.detach().float().cpu()
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
    }


def fixed_threshold_summary(threshold):
    return {
        'min': float(threshold),
        'max': float(threshold),
        'mean': float(threshold),
        'std': 0.0,
    }


def select_equal_by_index(equal_mask, tie_keep_quota, mode):
    batch_size, token_count = equal_mask.shape
    selected = torch.zeros_like(equal_mask, dtype=torch.bool)
    base_indices = torch.arange(token_count, device=equal_mask.device)
    if mode == 'secure_tie_lowest_index':
        ordered_indices = base_indices
    elif mode == 'secure_tie_highest_index':
        ordered_indices = torch.flip(base_indices, dims=[0])
    else:
        raise ValueError(f'unsupported tie policy mode: {mode}')

    for sample_index in range(batch_size):
        quota = int(tie_keep_quota[sample_index].item())
        if quota <= 0:
            continue
        equal_indices = ordered_indices[equal_mask[sample_index, ordered_indices]]
        selected_indices = equal_indices[:quota]
        selected[sample_index, selected_indices] = True
    return selected


def normalize_tie_policy_mode(mode):
    if mode in ('lowest_index', 'secure_tie_lowest_index'):
        return 'secure_tie_lowest_index'
    if mode in ('highest_index', 'secure_tie_highest_index'):
        return 'secure_tie_highest_index'
    raise ValueError(f'unsupported tie policy mode: {mode}')


def build_reference_topk_mask(masked_score, active_before, keep_count, tie_policy_mode='lowest_index'):
    tie_policy_mode = normalize_tie_policy_mode(tie_policy_mode)
    keep_policy = torch.argsort(masked_score, dim=1, descending=True)[:, :keep_count]
    kth_threshold = masked_score.gather(1, keep_policy[:, -1:].contiguous())
    greater_mask = (masked_score > kth_threshold) & active_before
    equal_mask = (masked_score == kth_threshold) & active_before
    tie_keep_quota = keep_count - greater_mask.sum(dim=1)
    selected_equal_mask = select_equal_by_index(equal_mask, tie_keep_quota, tie_policy_mode)
    topk_mask = greater_mask | selected_equal_mask
    return {
        'keep_policy': keep_policy,
        'kth_threshold': kth_threshold,
        'greater_mask': greater_mask,
        'equal_mask': equal_mask,
        'tie_keep_quota': tie_keep_quota,
        'selected_equal_mask': selected_equal_mask,
        'topk_mask': topk_mask,
    }


def next_power_of_two(value):
    power = 1
    while power < value:
        power *= 2
    return power


def odd_even_sort_desc(values):
    token_count = values.shape[1]
    sorted_values = values
    for pass_index in range(token_count):
        start = pass_index % 2
        left_indices = torch.arange(start, token_count - 1, 2, device=values.device)
        if left_indices.numel() == 0:
            continue
        right_indices = left_indices + 1
        left_values = sorted_values[:, left_indices]
        right_values = sorted_values[:, right_indices]
        keep_left = left_values >= right_values
        high_values = torch.where(keep_left, left_values, right_values)
        low_values = torch.where(keep_left, right_values, left_values)
        sorted_values = sorted_values.clone()
        sorted_values[:, left_indices] = high_values
        sorted_values[:, right_indices] = low_values
    return sorted_values


def kth_threshold_from_compare_network(masked_score, keep_count):
    token_count = masked_score.shape[1]
    padded_count = next_power_of_two(token_count)
    if padded_count == token_count:
        sortable = masked_score
    else:
        pad = torch.full(
            (masked_score.shape[0], padded_count - token_count),
            float('-inf'),
            dtype=masked_score.dtype,
            device=masked_score.device,
        )
        sortable = torch.cat([masked_score, pad], dim=1)
    sorted_values = odd_even_sort_desc(sortable)
    return sorted_values[:, keep_count - 1 : keep_count]


def build_threshold_report(
    threshold,
    threshold_summary,
    requires_secure_selection_design,
    uses_reference_tie_payload=False,
    tie_statistics=None,
    kth_threshold_compare=None,
):
    return {
        'threshold': threshold,
        'threshold_summary': threshold_summary,
        'dynamic_threshold_source_requires_secure_selection_design': requires_secure_selection_design,
        'uses_reference_tie_payload': uses_reference_tie_payload,
        'tie_statistics': tie_statistics,
        'kth_threshold_compare': kth_threshold_compare,
    }


def build_tie_statistics(eq_mask, selected_equal_mask, tie_keep_quota):
    equal_token_count = eq_mask.sum(dim=1)
    selected_equal_count = selected_equal_mask.sum(dim=1)
    tie_sample_mask = equal_token_count > tie_keep_quota
    return {
        'sample_count': int(eq_mask.shape[0]),
        'tie_sample_count': int(tie_sample_mask.sum().item()),
        'sum_equal_token_count_at_boundary': float(equal_token_count.float().sum().item()),
        'sum_selected_equal_count': float(selected_equal_count.float().sum().item()),
        'sum_tie_keep_quota': float(tie_keep_quota.float().sum().item()),
        'boundary_tie_sample_ratio': float(tie_sample_mask.float().mean().item()),
        'mean_equal_token_count_at_boundary': float(equal_token_count.float().mean().item()),
        'mean_selected_equal_count': float(selected_equal_count.float().mean().item()),
        'mean_tie_keep_quota': float(tie_keep_quota.float().mean().item()),
    }


def build_mask_from_kth_threshold(score, kth_threshold, active_before, keep_count, tie_policy_mode):
    gt_mask = (score > kth_threshold) & active_before
    eq_mask = (score == kth_threshold) & active_before
    tie_keep_quota = keep_count - gt_mask.sum(dim=1)
    selected_equal_mask = select_equal_by_index(eq_mask, tie_keep_quota, tie_policy_mode)
    return {
        'gt_mask': gt_mask,
        'eq_mask': eq_mask,
        'tie_keep_quota': tie_keep_quota,
        'selected_equal_mask': selected_equal_mask,
        'threshold_mask': gt_mask | selected_equal_mask,
    }


def build_kth_threshold_compare(kth_threshold, reference_kth_threshold, source):
    return {
        'max_abs_error_vs_argsort_reference': float((kth_threshold - reference_kth_threshold).abs().max().item()),
        'source': source,
    }


def build_threshold_mask(score, masked_score, active_before, keep_policy, topk_reference, stage_thresholds, stage_index, threshold_mode):
    if threshold_mode == 'fixed_stage_threshold':
        threshold = stage_thresholds[stage_index]
        threshold_mask = (score >= threshold) & active_before
        return threshold_mask, build_threshold_report(
            threshold=float(threshold),
            threshold_summary=fixed_threshold_summary(threshold),
            requires_secure_selection_design=False,
        )

    if threshold_mode == 'per_sample_kth_score':
        kth_threshold = masked_score.gather(1, keep_policy[:, -1:].contiguous())
        threshold_mask = (score >= kth_threshold) & active_before
        return threshold_mask, build_threshold_report(
            threshold=None,
            threshold_summary=tensor_summary(kth_threshold.squeeze(1)),
            requires_secure_selection_design=True,
        )

    if threshold_mode == 'secure_kth_reference':
        kth_threshold = topk_reference['kth_threshold']
        eq_mask = topk_reference['equal_mask']
        selected_equal_mask = topk_reference['selected_equal_mask']
        tie_keep_quota = topk_reference['tie_keep_quota']
        return topk_reference['topk_mask'], build_threshold_report(
            threshold=None,
            threshold_summary=tensor_summary(kth_threshold.squeeze(1)),
            requires_secure_selection_design=True,
            uses_reference_tie_payload=True,
            tie_statistics=build_tie_statistics(eq_mask, selected_equal_mask, tie_keep_quota),
        )

    if threshold_mode in ('secure_tie_lowest_index', 'secure_tie_highest_index'):
        kth_threshold = masked_score.gather(1, keep_policy[:, -1:].contiguous())
        tie_payload = build_mask_from_kth_threshold(score, kth_threshold, active_before, keep_policy.shape[1], threshold_mode)
        return tie_payload['threshold_mask'], build_threshold_report(
            threshold=None,
            threshold_summary=tensor_summary(kth_threshold.squeeze(1)),
            requires_secure_selection_design=True,
            tie_statistics=build_tie_statistics(
                tie_payload['eq_mask'],
                tie_payload['selected_equal_mask'],
                tie_payload['tie_keep_quota'],
            ),
        )

    if threshold_mode == 'secure_network_kth_lowest_index':
        keep_count = keep_policy.shape[1]
        kth_threshold = kth_threshold_from_compare_network(masked_score, keep_count)
        tie_payload = build_mask_from_kth_threshold(
            score,
            kth_threshold,
            active_before,
            keep_count,
            'secure_tie_lowest_index',
        )
        reference_kth_threshold = masked_score.gather(1, keep_policy[:, -1:].contiguous())
        return tie_payload['threshold_mask'], build_threshold_report(
            threshold=None,
            threshold_summary=tensor_summary(kth_threshold.squeeze(1)),
            requires_secure_selection_design=False,
            tie_statistics=build_tie_statistics(
                tie_payload['eq_mask'],
                tie_payload['selected_equal_mask'],
                tie_payload['tie_keep_quota'],
            ),
            kth_threshold_compare=build_kth_threshold_compare(
                kth_threshold,
                reference_kth_threshold,
                'fixed_odd_even_compare_swap_network',
            ),
        )

    raise ValueError(f'unsupported threshold mode: {threshold_mode}')


@torch.no_grad()
def forward_threshold_pruning(model, inputs, stage_thresholds, threshold_mode):
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
            topk_reference = build_reference_topk_mask(
                masked_score=masked_score,
                active_before=active_before,
                keep_count=keep_count,
                tie_policy_mode=getattr(model, 'eval_tie_policy', 'lowest_index'),
            )
            keep_policy = topk_reference['keep_policy']
            topk_mask = topk_reference['topk_mask']

            threshold_mask, threshold_report = build_threshold_mask(
                score=score,
                masked_score=masked_score,
                active_before=active_before,
                keep_policy=keep_policy,
                topk_reference=topk_reference,
                stage_thresholds=stage_thresholds,
                stage_index=stage_index,
                threshold_mode=threshold_mode,
            )
            prev_decision = threshold_mask.unsqueeze(-1).to(dtype=x.dtype)

            cls_policy = torch.ones(batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

            stage_reports.append(
                {
                    'stage_index': stage_index,
                    'threshold_mode': threshold_mode,
                    **threshold_report,
                    'metrics': mask_metrics(threshold_mask.detach().cpu(), topk_mask.detach().cpu()),
                }
            )
            stage_index += 1
        else:
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

    x = model.norm(x)
    x = x[:, 0]
    x = model.pre_logits(x)
    logits = model.head(x)
    return logits, stage_reports


def aggregate_stage_reports(all_stage_reports):
    stage_count = len(all_stage_reports[0])
    aggregated = []
    for stage_index in range(stage_count):
        stage_items = [batch[stage_index] for batch in all_stage_reports]
        metrics = {
            key: float(sum(item['metrics'][key] for item in stage_items) / len(stage_items))
            for key in stage_items[0]['metrics']
        }
        threshold_summary = {
            key: float(sum(item['threshold_summary'][key] for item in stage_items) / len(stage_items))
            for key in stage_items[0]['threshold_summary']
        }

        tie_statistics = stage_items[0].get('tie_statistics')
        if tie_statistics is not None:
            total_samples = sum(item['tie_statistics']['sample_count'] for item in stage_items)
            tie_statistics = {
                'boundary_tie_sample_ratio': float(
                    sum(item['tie_statistics']['tie_sample_count'] for item in stage_items) / total_samples
                ),
                'mean_equal_token_count_at_boundary': float(
                    sum(item['tie_statistics']['sum_equal_token_count_at_boundary'] for item in stage_items) / total_samples
                ),
                'mean_selected_equal_count': float(
                    sum(item['tie_statistics']['sum_selected_equal_count'] for item in stage_items) / total_samples
                ),
                'mean_tie_keep_quota': float(
                    sum(item['tie_statistics']['sum_tie_keep_quota'] for item in stage_items) / total_samples
                ),
            }
        kth_threshold_compare = stage_items[0].get('kth_threshold_compare')
        if kth_threshold_compare is not None:
            kth_threshold_compare = {
                'max_abs_error_vs_argsort_reference': float(
                    max(item['kth_threshold_compare']['max_abs_error_vs_argsort_reference'] for item in stage_items)
                ),
                'source': kth_threshold_compare['source'],
            }

        stage_report = {
            'stage_index': stage_index,
            'threshold_mode': stage_items[0]['threshold_mode'],
            'threshold': stage_items[0]['threshold'],
            'threshold_summary': threshold_summary,
            'dynamic_threshold_source_requires_secure_selection_design': bool(
                stage_items[0]['dynamic_threshold_source_requires_secure_selection_design']
            ),
            'uses_reference_tie_payload': bool(stage_items[0].get('uses_reference_tie_payload', False)),
            'tie_statistics': tie_statistics,
            'kth_threshold_compare': kth_threshold_compare,
            'metrics': metrics,
        }
        aggregated.append(stage_report)
    return aggregated


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run an isolated eval-only threshold-pruning branch against the frozen Transshield bundle.'
    )
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--threshold-json', default='')
    parser.add_argument(
        '--threshold-mode',
        default='fixed_stage_threshold',
        choices=[
            'fixed_stage_threshold',
            'per_sample_kth_score',
            'secure_kth_reference',
            'secure_tie_lowest_index',
            'secure_tie_highest_index',
            'secure_network_kth_lowest_index',
        ],
        help='Use fixed stage thresholds from --threshold-json, or derive a per-sample kth-score threshold from the top-k boundary.',
    )
    parser.add_argument('--data-path', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output-json', required=True)
    return parser


def resolve_stage_thresholds(bundle_dir: Path, threshold_mode: str, threshold_json: str):
    if threshold_mode != 'fixed_stage_threshold':
        return [], None
    if not threshold_json:
        raise ValueError('--threshold-json is required when --threshold-mode fixed_stage_threshold')
    threshold_json_path = Path(threshold_json).resolve()
    threshold_payload = load_json(threshold_json_path)
    return [stage['threshold'] for stage in threshold_payload['stages']], threshold_json_path


def resolve_data_path(args, bundle):
    if args.data_path:
        return Path(args.data_path).resolve()
    return Path(bundle['args_snapshot']['eval_data_path']).resolve()


def build_dataset(bundle, data_path: Path, max_samples: int):
    dataset = ImageFolder(root=str(data_path), transform=bundle['transform'])
    if max_samples > 0:
        dataset.samples = dataset.samples[:max_samples]
        dataset.imgs = dataset.imgs[:max_samples]
        dataset.targets = dataset.targets[:max_samples]
    return dataset


def evaluate_threshold_branch(bundle, loader, stage_thresholds, threshold_mode, device: str):
    all_logits = []
    all_targets = []
    all_stage_reports = []
    finite_logits = True

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, stage_reports = forward_threshold_pruning(bundle['model'], inputs, stage_thresholds, threshold_mode)
        finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_stage_reports.append(stage_reports)

    return {
        'logits': torch.cat(all_logits, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'stage_reports': aggregate_stage_reports(all_stage_reports),
        'finite_logits': finite_logits,
    }


def build_eval_metrics(logits, targets, class_threshold):
    loss = F.cross_entropy(logits, targets).item()
    probs = torch.softmax(logits, dim=-1)
    pred_counts = torch.bincount(logits.argmax(dim=1), minlength=2).tolist()
    return {
        'eval_loss': float(loss),
        'argmax_accuracy': accuracy_from_logits(logits, targets),
        'threshold_accuracy': threshold_accuracy_from_logits(logits, targets, class_threshold),
        'class1_auc': binary_auc(probs[:, 1], targets),
        'pred_counts': [int(value) for value in pred_counts],
        'class_threshold': float(class_threshold),
    }


def main():
    args = build_parser().parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    class_threshold = resolve_threshold(bundle_dir, None)
    stage_thresholds, threshold_json_path = resolve_stage_thresholds(
        bundle_dir,
        args.threshold_mode,
        args.threshold_json,
    )
    data_path = resolve_data_path(args, bundle)
    dataset = build_dataset(bundle, data_path, args.max_samples)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    eval_output = evaluate_threshold_branch(
        bundle,
        loader,
        stage_thresholds,
        args.threshold_mode,
        args.device,
    )

    report = {
        'bundle_dir': str(bundle_dir),
        'threshold_json': str(threshold_json_path) if threshold_json_path else None,
        'data_path': str(data_path),
        'device': args.device,
        'sample_count': int(eval_output['targets'].numel()),
        'branch_type': 'isolated_eval_only_threshold_pruning',
        'pruning_threshold_mode': args.threshold_mode,
        'metrics': {
            'finite_logits': eval_output['finite_logits'],
            **build_eval_metrics(eval_output['logits'], eval_output['targets'], class_threshold),
        },
        'stage_reports': eval_output['stage_reports'],
        'constraints': {
            'model_semantics_changed_in_core_code': False,
            'frozen_candidate_replaced': False,
            'branch_is_eval_only': True,
            'topk_reference_preserved_outside_this_isolated_eval': True,
            'dynamic_threshold_source_requires_secure_selection_design': args.threshold_mode in (
                'per_sample_kth_score',
                'secure_kth_reference',
                'secure_tie_lowest_index',
                'secure_tie_highest_index',
            ),
            'uses_reference_tie_payload': args.threshold_mode == 'secure_kth_reference',
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
