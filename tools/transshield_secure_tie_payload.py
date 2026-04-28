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

from tools.transshield_input_selection import (
    SelectedImageDataset,
    add_input_selection_args,
    resolve_selected_sample_paths,
)
from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold
from tools.transshield_stagewise_threshold_report import collect_stagewise_scores
from tools.transshield_threshold_branch_eval import (
    accuracy_from_logits,
    binary_auc,
    build_reference_topk_mask,
    mask_metrics,
    threshold_accuracy_from_logits,
)


DEFAULT_THRESHOLD_TOLERANCE = 5e-5


def print_and_optionally_write(payload, output_json: str = ''):
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_json:
        Path(output_json).resolve().write_text(text + '\n', encoding='utf-8')


def summarize_bool_tensor(tensor):
    tensor = tensor.bool()
    counts = tensor.sum(dim=1).float()
    return {
        'shape': list(tensor.shape),
        'mean_true_count': float(counts.mean().item()),
        'max_true_count': int(counts.max().item()),
        'min_true_count': int(counts.min().item()),
    }


def resolve_dataset(bundle, args):
    selection = resolve_selected_sample_paths(
        data_path=args.data_path,
        default_data_path=bundle['args_snapshot'].get('eval_data_path', ''),
        image_paths=args.image,
        image_list=args.image_list,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        max_samples=args.max_samples,
    )
    return selection, SelectedImageDataset(selection['sample_paths'], bundle['transform'])


def export_tie_sidecar(args):
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    selection, dataset = resolve_dataset(bundle, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    stage_buffers = None
    sample_count = 0
    for inputs, _paths, _targets in loader:
        inputs = inputs.to(args.device)
        batch_reports = collect_stagewise_scores(bundle['model'], inputs)
        if stage_buffers is None:
            stage_buffers = [
                {
                    'stage_index': stage['stage_index'],
                    'pruning_layer': stage['pruning_layer'],
                    'keep_count': stage['keep_count'],
                    'selected_equal_mask': [],
                    'tie_keep_quota': [],
                }
                for stage in batch_reports
            ]

        for stage_index, stage in enumerate(batch_reports):
            score = stage['score']
            active_before = stage['active_before'].bool()
            topk_mask = stage['topk_mask'].bool()
            kth_threshold = stage['boundary_keep_score'].unsqueeze(1)
            equal_mask = (score == kth_threshold) & active_before
            greater_mask = (score > kth_threshold) & active_before
            tie_keep_quota = stage['keep_count'] - greater_mask.sum(dim=1)
            selected_equal_mask = equal_mask & topk_mask
            stage_buffers[stage_index]['selected_equal_mask'].append(selected_equal_mask.detach().cpu())
            stage_buffers[stage_index]['tie_keep_quota'].append(tie_keep_quota.detach().cpu())

        sample_count += inputs.shape[0]

    if stage_buffers is None:
        raise ValueError('no samples processed')

    payload_stages = []
    summary_stages = []
    for stage in stage_buffers:
        selected_equal_mask = torch.cat(stage['selected_equal_mask'], dim=0)
        tie_keep_quota = torch.cat(stage['tie_keep_quota'], dim=0)
        payload_stages.append(
            {
                'stage_index': int(stage['stage_index']),
                'pruning_layer': int(stage['pruning_layer']),
                'keep_count': int(stage['keep_count']),
                'selected_equal_mask': selected_equal_mask,
                'tie_keep_quota': tie_keep_quota,
            }
        )
        summary_stages.append(
            {
                'stage_index': int(stage['stage_index']),
                'pruning_layer': int(stage['pruning_layer']),
                'keep_count': int(stage['keep_count']),
                'selected_equal_mask': summarize_bool_tensor(selected_equal_mask),
                'tie_keep_quota': {
                    'shape': list(tie_keep_quota.shape),
                    'mean': float(tie_keep_quota.float().mean().item()),
                    'max': int(tie_keep_quota.max().item()),
                    'min': int(tie_keep_quota.min().item()),
                },
            }
        )

    pt_payload = {
        'bundle_dir': str(bundle_dir),
        'data_path': selection['data_path'],
        'selection_mode': selection['selection_mode'],
        'sample_paths': [str(path) for path in dataset.sample_paths],
        'stages': payload_stages,
        'candidate_metadata': {
            'format_purpose': 'secure_tie_payload_reference_topk_sidecar',
            'source_reference_topk': True,
            'tie_policy': 'lowest_index',
            'model_semantics_changed': False,
        },
    }
    torch.save(pt_payload, Path(args.output_pt).resolve())

    summary = {
        'bundle_dir': str(bundle_dir),
        'data_path': selection['data_path'],
        'selection_mode': selection['selection_mode'],
        'sample_count': sample_count,
        'output_pt': str(Path(args.output_pt).resolve()),
        'stages': summary_stages,
        'constraints': {
            'source_reference_topk': True,
            'tie_policy_lowest_index': True,
            'sidecar_is_read_only_reference': True,
            'model_semantics_changed': False,
        },
    }
    print_and_optionally_write(summary, args.output_json)


def load_payload(path: Path):
    payload = torch.load(path.resolve(), map_location='cpu', weights_only=False)
    if 'stages' not in payload:
        raise ValueError(f'missing stages in tie payload: {path}')
    return payload


def resolve_boundary_threshold(masked_score, kth_threshold, active_before, keep_count, tolerance):
    threshold = kth_threshold.view(-1)
    resolved = threshold.clone()
    distance = torch.full_like(threshold, float('inf'))

    for sample_index in range(masked_score.shape[0]):
        active_scores = masked_score[sample_index][active_before[sample_index]]
        if active_scores.numel() == 0:
            continue

        candidate_scores = torch.unique(active_scores, sorted=True)
        gt_counts = (active_scores.unsqueeze(0) > candidate_scores.unsqueeze(1)).sum(dim=1)
        ge_counts = (active_scores.unsqueeze(0) >= candidate_scores.unsqueeze(1)).sum(dim=1)
        valid_mask = (gt_counts < keep_count) & (keep_count <= ge_counts)
        valid_scores = candidate_scores[valid_mask]
        if valid_scores.numel() == 0:
            continue

        valid_distance = (valid_scores - threshold[sample_index]).abs()
        best_index = valid_distance.argmin()
        best_distance = valid_distance[best_index]
        if bool(best_distance <= tolerance):
            resolved[sample_index] = valid_scores[best_index]
            distance[sample_index] = best_distance

    return resolved.view(-1, 1), distance


def tensor_compare(lhs, rhs):
    lhs = lhs.detach().cpu()
    rhs = rhs.detach().cpu()
    if lhs.shape != rhs.shape:
        return {
            'shape_match': False,
            'lhs_shape': list(lhs.shape),
            'rhs_shape': list(rhs.shape),
            'exact_match': False,
        }
    if lhs.dtype == torch.bool or rhs.dtype == torch.bool:
        lhs_cmp = lhs.bool()
        rhs_cmp = rhs.bool()
        mismatch_count = int((lhs_cmp != rhs_cmp).sum().item())
    else:
        mismatch_count = int((lhs != rhs).sum().item())
    return {
        'shape_match': True,
        'shape': list(lhs.shape),
        'mismatch_count': mismatch_count,
        'exact_match': bool(mismatch_count == 0),
    }


def validate_candidate_stage(stage):
    selected_equal_mask = stage['selected_equal_mask'].bool()
    tie_keep_quota = stage['tie_keep_quota'].long()
    selected_counts = selected_equal_mask.sum(dim=1).long()
    count_mismatch = int((selected_counts != tie_keep_quota).sum().item())
    return {
        'selected_equal_count_matches_tie_keep_quota': bool(count_mismatch == 0),
        'count_mismatch_sample_count': count_mismatch,
        'selected_equal_mask_shape': list(selected_equal_mask.shape),
        'tie_keep_quota_shape': list(tie_keep_quota.shape),
    }


def build_semantic_stage_report(input_stage, kth_stage, candidate_stage, threshold_tolerance, reference_tie_policy):
    masked_score = input_stage['masked_score'].detach().float().cpu()
    active_before = input_stage.get('active_before')
    if active_before is None:
        active_before = torch.isfinite(masked_score)
    else:
        active_before = active_before.detach().bool().cpu()

    keep_count = int(input_stage['keep_count'])
    masked_score_active = masked_score.masked_fill(~active_before, float('-inf'))
    topk_reference = build_reference_topk_mask(
        masked_score=masked_score_active,
        active_before=active_before,
        keep_count=keep_count,
        tie_policy_mode=reference_tie_policy,
    )
    topk_mask = topk_reference['topk_mask']

    kth_threshold = kth_stage['kth_threshold'].detach().float().cpu().view(-1, 1)
    snapped_threshold, snap_distance = resolve_boundary_threshold(
        masked_score,
        kth_threshold,
        active_before,
        keep_count,
        threshold_tolerance,
    )
    greater_mask = (masked_score > snapped_threshold) & active_before
    equal_mask = (masked_score == snapped_threshold) & active_before

    selected_equal_mask = candidate_stage['selected_equal_mask'].detach().bool().cpu()
    tie_keep_quota = candidate_stage['tie_keep_quota'].detach().long().cpu()
    branch_mask = greater_mask | selected_equal_mask

    subset_ok = bool(((selected_equal_mask & ~equal_mask).sum() == 0).item())
    disjoint_ok = bool(((selected_equal_mask & greater_mask).sum() == 0).item())
    quota_ok = bool(torch.equal(selected_equal_mask.sum(dim=1).long(), tie_keep_quota))
    reconstructed_topk_ok = bool(torch.equal(branch_mask, topk_mask))

    finite_distance = snap_distance[torch.isfinite(snap_distance)]
    return {
        'threshold_tolerance': float(threshold_tolerance),
        'selected_equal_subset_of_equal_mask': subset_ok,
        'selected_equal_disjoint_from_greater_mask': disjoint_ok,
        'selected_equal_count_matches_tie_keep_quota': quota_ok,
        'reconstructed_branch_matches_topk_reference': reconstructed_topk_ok,
        'threshold_snap': {
            'max_distance': float(finite_distance.max().item()) if finite_distance.numel() else None,
            'mean_distance': float(finite_distance.mean().item()) if finite_distance.numel() else None,
            'snapped_count': int((snap_distance <= threshold_tolerance).sum().item()),
            'sample_count': int(snap_distance.numel()),
        },
        'passed': all([subset_ok, disjoint_ok, quota_ok, reconstructed_topk_ok]),
    }


def build_report(reference_payload, candidate_payload, reference_path, candidate_path, input_payload, kth_payload, threshold_tolerance):
    reference_stages = {int(stage['stage_index']): stage for stage in reference_payload['stages']}
    candidate_stages = {int(stage['stage_index']): stage for stage in candidate_payload['stages']}
    input_stages = {int(stage['stage_index']): stage for stage in input_payload['stages']} if input_payload else {}
    kth_stages = {int(stage['stage_index']): stage for stage in kth_payload['stages']} if kth_payload else {}
    use_semantic_check = bool(input_payload is not None and kth_payload is not None)
    reference_tie_policy = candidate_payload.get('candidate_metadata', {}).get('tie_policy', 'lowest_index')

    stage_reports = []
    all_passed = True
    for stage_index, reference_stage in reference_stages.items():
        candidate_stage = candidate_stages.get(stage_index)
        if candidate_stage is None:
            stage_reports.append(
                {
                    'stage_index': stage_index,
                    'available': False,
                    'passed': False,
                    'reason': 'missing candidate stage',
                }
            )
            all_passed = False
            continue

        selected_equal_compare = tensor_compare(reference_stage['selected_equal_mask'], candidate_stage['selected_equal_mask'])
        tie_quota_compare = tensor_compare(reference_stage['tie_keep_quota'], candidate_stage['tie_keep_quota'])
        candidate_self_check = validate_candidate_stage(candidate_stage)
        semantic_report = None
        if use_semantic_check:
            input_stage = input_stages.get(stage_index)
            kth_stage = kth_stages.get(stage_index)
            if input_stage is None or kth_stage is None:
                semantic_report = {
                    'passed': False,
                    'reason': 'missing semantic input stage or kth stage',
                }
            else:
                semantic_report = build_semantic_stage_report(
                    input_stage,
                    kth_stage,
                    candidate_stage,
                    threshold_tolerance,
                    reference_tie_policy,
                )

        stage_passed = (
            semantic_report['passed']
            if semantic_report is not None
            else (
                selected_equal_compare['exact_match']
                and tie_quota_compare['exact_match']
                and candidate_self_check['selected_equal_count_matches_tie_keep_quota']
            )
        )
        all_passed = all_passed and stage_passed
        stage_reports.append(
            {
                'stage_index': stage_index,
                'available': True,
                'pruning_layer': int(reference_stage['pruning_layer']),
                'keep_count': int(reference_stage['keep_count']),
                'selected_equal_mask_compare': selected_equal_compare,
                'tie_keep_quota_compare': tie_quota_compare,
                'candidate_self_check': candidate_self_check,
                'semantic_check': semantic_report,
                'passed': stage_passed,
            }
        )

    return {
        'reference_pt': str(reference_path),
        'candidate_pt': str(candidate_path),
        'bundle_dir': reference_payload.get('bundle_dir'),
        'data_path': reference_payload.get('data_path'),
        'overall_passed': bool(all_passed),
        'stage_reports': stage_reports,
        'threshold_tolerance': threshold_tolerance if use_semantic_check else None,
        'constraints': {
            'read_only_checker': True,
            'checks_tie_payload_sidecar_only': True,
            'does_not_run_model_forward': True,
            'uses_semantic_boundary_check': use_semantic_check,
        },
    }


def run_check(args):
    reference_path = Path(args.reference_pt).resolve()
    candidate_path = Path(args.candidate_pt).resolve()
    input_payload = load_payload(Path(args.input_pt).resolve()) if args.input_pt else None
    kth_payload = load_payload(Path(args.kth_payload_pt).resolve()) if args.kth_payload_pt else None
    report = build_report(
        load_payload(reference_path),
        load_payload(candidate_path),
        reference_path,
        candidate_path,
        input_payload,
        kth_payload,
        args.threshold_tolerance,
    )
    print_and_optionally_write(report, args.output_json)


def infer_payload_sample_count(tie_payload):
    stages = tie_payload.get('stages') or []
    if not stages:
        return 0
    first_stage = stages[0]
    selected_equal_mask = first_stage.get('selected_equal_mask')
    if selected_equal_mask is None:
        return 0
    return int(selected_equal_mask.shape[0])


@torch.no_grad()
def forward_secure_tie_payload_branch(model, inputs, tie_payload_stages, sample_offset, tie_policy='lowest_index'):
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
                tie_policy_mode=tie_policy,
            )
            topk_mask = topk_reference['topk_mask']
            kth_threshold = topk_reference['kth_threshold']
            gt_mask = topk_reference['greater_mask']
            eq_mask = topk_reference['equal_mask']

            stage_payload = tie_payload_stages[stage_index]
            selected_equal_mask = stage_payload['selected_equal_mask'][sample_offset : sample_offset + batch_size].to(inputs.device).bool()
            tie_keep_quota = stage_payload['tie_keep_quota'][sample_offset : sample_offset + batch_size].to(inputs.device).long()
            branch_mask = gt_mask | selected_equal_mask

            subset_ok = bool(((selected_equal_mask & ~eq_mask).sum() == 0).item())
            disjoint_ok = bool(((selected_equal_mask & gt_mask).sum() == 0).item())
            quota_ok = bool(torch.equal(selected_equal_mask.sum(dim=1).long(), tie_keep_quota))

            prev_decision = branch_mask.unsqueeze(-1).to(dtype=x.dtype)
            cls_policy = torch.ones(batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

            stage_reports.append(
                {
                    'stage_index': stage_index,
                    'threshold_mode': 'secure_tie_payload_candidate',
                    'threshold': None,
                    'threshold_summary': {
                        'min': float(kth_threshold.min().item()),
                        'max': float(kth_threshold.max().item()),
                        'mean': float(kth_threshold.mean().item()),
                        'std': float(kth_threshold.std(unbiased=False).item()),
                    },
                    'metrics': mask_metrics(branch_mask.detach().cpu(), topk_mask.detach().cpu()),
                    'tie_payload_checks': {
                        'selected_equal_subset_of_equal_mask': subset_ok,
                        'selected_equal_disjoint_from_greater_mask': disjoint_ok,
                        'selected_equal_count_matches_tie_keep_quota': quota_ok,
                    },
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
        metrics = {}
        for key in stage_items[0]['metrics']:
            metrics[key] = float(sum(item['metrics'][key] for item in stage_items) / len(stage_items))
        threshold_summary = {}
        for key in stage_items[0]['threshold_summary']:
            threshold_summary[key] = float(sum(item['threshold_summary'][key] for item in stage_items) / len(stage_items))
        tie_payload_checks = {
            key: all(item['tie_payload_checks'][key] for item in stage_items) for key in stage_items[0]['tie_payload_checks']
        }
        aggregated.append(
            {
                'stage_index': stage_index,
                'threshold_mode': 'secure_tie_payload_candidate',
                'threshold': None,
                'threshold_summary': threshold_summary,
                'metrics': metrics,
                'tie_payload_checks': tie_payload_checks,
            }
        )
    return aggregated


def run_branch_eval(args):
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    class_threshold = resolve_threshold(bundle_dir, None)
    tie_payload = load_payload(Path(args.tie_payload_pt))
    payload_sample_count = infer_payload_sample_count(tie_payload)
    data_path = Path(args.data_path).resolve() if args.data_path else Path(bundle['args_snapshot']['eval_data_path']).resolve()

    dataset = ImageFolder(root=str(data_path), transform=bundle['transform'])
    effective_max_samples = payload_sample_count
    if args.max_samples > 0:
        effective_max_samples = min(args.max_samples, payload_sample_count) if payload_sample_count > 0 else args.max_samples
    if effective_max_samples > 0:
        dataset.samples = dataset.samples[: effective_max_samples]
        dataset.imgs = dataset.imgs[: effective_max_samples]
        dataset.targets = dataset.targets[: effective_max_samples]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    all_logits = []
    all_targets = []
    all_stage_reports = []
    finite_logits = True
    sample_offset = 0
    for inputs, targets in loader:
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        logits, stage_reports = forward_secure_tie_payload_branch(
            bundle['model'],
            inputs,
            tie_payload['stages'],
            sample_offset,
            tie_payload.get('candidate_metadata', {}).get(
                'tie_policy',
                bundle['args_snapshot'].get('eval_tie_policy', 'lowest_index'),
            ),
        )
        finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_stage_reports.append(stage_reports)
        sample_offset += inputs.shape[0]

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    loss = F.cross_entropy(logits, targets).item()
    probs = torch.softmax(logits, dim=-1)
    pred_counts = torch.bincount(logits.argmax(dim=1), minlength=2).tolist()

    report = {
        'bundle_dir': str(bundle_dir),
        'tie_payload_pt': str(Path(args.tie_payload_pt).resolve()),
        'data_path': str(data_path),
        'device': args.device,
        'sample_count': int(targets.numel()),
        'payload_sample_count': payload_sample_count,
        'branch_type': 'isolated_eval_only_secure_tie_payload_branch',
        'pruning_threshold_mode': 'secure_tie_payload_candidate',
        'metrics': {
            'finite_logits': finite_logits,
            'eval_loss': float(loss),
            'argmax_accuracy': accuracy_from_logits(logits, targets),
            'threshold_accuracy': threshold_accuracy_from_logits(logits, targets, class_threshold),
            'class1_auc': binary_auc(probs[:, 1], targets),
            'pred_counts': [int(value) for value in pred_counts],
            'class_threshold': float(class_threshold),
        },
        'stage_reports': aggregate_stage_reports(all_stage_reports),
        'constraints': {
            'model_semantics_changed_in_core_code': False,
            'frozen_candidate_replaced': False,
            'branch_is_eval_only': True,
            'topk_reference_preserved_outside_this_isolated_eval': True,
            'dynamic_threshold_source_requires_secure_selection_design': True,
            'uses_reference_tie_payload': bool(tie_payload.get('candidate_metadata', {}).get('source_reference_topk', False)),
        },
    }
    print_and_optionally_write(report, args.output_json)


def main():
    parser = argparse.ArgumentParser(description='Unified secure tie-payload sidecar utility.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    export = subparsers.add_parser('export', help='export deterministic lowest-index tie payload')
    export.add_argument('--bundle-dir', required=True)
    add_input_selection_args(export)
    export.add_argument('--device', default='cpu')
    export.add_argument('--batch-size', type=int, default=16)
    export.add_argument('--num-workers', type=int, default=0)
    export.add_argument('--max-samples', type=int, default=0)
    export.add_argument('--output-pt', required=True)
    export.add_argument('--output-json', default='')

    checker = subparsers.add_parser('check', help='check tie payload candidate against reference')
    checker.add_argument('--reference-pt', required=True)
    checker.add_argument('--candidate-pt', required=True)
    checker.add_argument('--input-pt', default='')
    checker.add_argument('--kth-payload-pt', default='')
    checker.add_argument('--threshold-tolerance', type=float, default=DEFAULT_THRESHOLD_TOLERANCE)
    checker.add_argument('--output-json', default='')

    branch_eval = subparsers.add_parser('branch-eval', help='run isolated eval-only tie sidecar branch')
    branch_eval.add_argument('--bundle-dir', required=True)
    branch_eval.add_argument('--tie-payload-pt', required=True)
    branch_eval.add_argument('--data-path', default='')
    branch_eval.add_argument('--device', default='cpu')
    branch_eval.add_argument('--batch-size', type=int, default=16)
    branch_eval.add_argument('--num-workers', type=int, default=0)
    branch_eval.add_argument('--max-samples', type=int, default=0)
    branch_eval.add_argument('--output-json', required=True)

    args = parser.parse_args()
    dispatch = {
        'export': export_tie_sidecar,
        'check': run_check,
        'branch-eval': run_branch_eval,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
