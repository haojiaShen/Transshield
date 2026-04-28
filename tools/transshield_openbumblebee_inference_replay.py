import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_model_state_dict_path, resolve_threshold
from tools.transshield_threshold_branch_eval import build_reference_topk_mask, mask_metrics

DEFAULT_THRESHOLD_TOLERANCE = 5e-5


def write_json(path: Path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def load_payload(path: Path, expected_fields):
    payload = torch.load(path.resolve(), map_location='cpu', weights_only=False)
    missing_fields = [field for field in expected_fields if field not in payload]
    if missing_fields:
        raise ValueError(f'missing required fields {missing_fields} in payload: {path}')
    return payload


def stage_map(payload):
    return {int(stage['stage_index']): stage for stage in payload['stages']}


def load_stage_payloads(input_pt: Path, kth_payload_pt: Path, tie_payload_pt: Path):
    input_payload = load_payload(input_pt, ['stages'])
    kth_payload = load_payload(kth_payload_pt, ['stages'])
    tie_payload = load_payload(tie_payload_pt, ['stages'])
    return input_payload, kth_payload, tie_payload


def tensor_summary(values):
    values = values.detach().float().cpu()
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    return {
        'shape': list(values.shape),
        'finite_count': int(finite_mask.sum().item()),
        'nonfinite_count': int((~finite_mask).sum().item()),
        'min': float(finite_values.min().item()) if finite_values.numel() else None,
        'max': float(finite_values.max().item()) if finite_values.numel() else None,
        'mean': float(finite_values.mean().item()) if finite_values.numel() else None,
        'std': float(finite_values.std(unbiased=False).item()) if finite_values.numel() else 0.0,
    }


def bool_count_summary(values):
    values = values.detach().bool().cpu()
    counts = values.sum(dim=1).float()
    return {
        'shape': list(values.shape),
        'mean_true_count': float(counts.mean().item()) if counts.numel() else 0.0,
        'max_true_count': int(counts.max().item()) if counts.numel() else 0,
        'min_true_count': int(counts.min().item()) if counts.numel() else 0,
    }


def resolve_active_before(input_stage, masked_score):
    active_before = input_stage.get('active_before')
    if active_before is None:
        return torch.isfinite(masked_score)
    return active_before.detach().bool().cpu()


def resolve_boundary_threshold(masked_score, kth_threshold, active_before, keep_count, tolerance):
    threshold = kth_threshold.view(-1)
    masked_score_active = masked_score.masked_fill(~active_before, float('-inf'))
    reference_threshold = torch.topk(masked_score_active, k=keep_count, dim=1).values[:, -1]
    distance = (reference_threshold - threshold).abs()
    resolved = torch.where(distance <= tolerance, reference_threshold, threshold)
    unresolved_distance = torch.full_like(distance, float('inf'))
    distance = torch.where(distance <= tolerance, distance, unresolved_distance)
    return resolved.view(-1, 1), distance


def threshold_snap_summary(snap_distance, tolerance):
    snap_distance = snap_distance.detach().float().cpu()
    finite_distance = snap_distance[torch.isfinite(snap_distance)]
    snapped_mask = snap_distance <= tolerance
    return {
        'tolerance': float(tolerance),
        'snapped_count': int(snapped_mask.sum().item()),
        'sample_count': int(snap_distance.numel()),
        'max_distance': float(finite_distance.max().item()) if finite_distance.numel() else None,
        'mean_distance': float(finite_distance.mean().item()) if finite_distance.numel() else None,
    }


def resolve_sample_path(sample_path: str, sample_root_from: str, sample_root_to: str):
    path = Path(sample_path)
    if path.exists() or not sample_root_from or not sample_root_to:
        return path
    try:
        relative = path.relative_to(Path(sample_root_from))
    except ValueError:
        return path
    return Path(sample_root_to) / relative


def collect_sample_path_report(sample_paths, sample_root_from: str, sample_root_to: str):
    resolved = [resolve_sample_path(item, sample_root_from, sample_root_to) for item in sample_paths]
    missing = [str(path) for path in resolved if not path.exists()]
    return {
        'count': len(sample_paths),
        'sample_root_from': sample_root_from or None,
        'sample_root_to': sample_root_to or None,
        'missing_count': len(missing),
        'missing_examples': missing[:4],
        'resolved_paths_preview': [str(path) for path in resolved[:4]],
    }, resolved


def ensure_matching_stage_sets(input_stages, kth_stages, tie_stages):
    if set(input_stages) != set(kth_stages) or set(input_stages) != set(tie_stages):
        raise ValueError('stage index sets differ across input/kth/tie payloads')


def validate_stage_alignment(input_stage, kth_stage, tie_stage):
    expected_stage_index = int(input_stage['stage_index'])
    if int(kth_stage['stage_index']) != expected_stage_index:
        raise ValueError(f'kth stage index mismatch at stage {expected_stage_index}')
    if int(tie_stage['stage_index']) != expected_stage_index:
        raise ValueError(f'tie stage index mismatch at stage {expected_stage_index}')

    expected_layer = int(input_stage['pruning_layer'])
    if int(kth_stage['pruning_layer']) != expected_layer:
        raise ValueError(f'kth pruning layer mismatch at stage {expected_stage_index}')
    if int(tie_stage['pruning_layer']) != expected_layer:
        raise ValueError(f'tie pruning layer mismatch at stage {expected_stage_index}')

    expected_keep = int(input_stage['keep_count'])
    if int(kth_stage['keep_count']) != expected_keep:
        raise ValueError(f'kth keep_count mismatch at stage {expected_stage_index}')
    if int(tie_stage['keep_count']) != expected_keep:
        raise ValueError(f'tie keep_count mismatch at stage {expected_stage_index}')


def build_reference_topk(masked_score, active_before, keep_count, reference_tie_policy):
    masked_score_active = masked_score.masked_fill(~active_before, float('-inf'))
    topk_reference = build_reference_topk_mask(
        masked_score=masked_score_active,
        active_before=active_before,
        keep_count=keep_count,
        tie_policy_mode=reference_tie_policy,
    )
    reference_kth_threshold = masked_score_active.gather(1, topk_reference['keep_policy'][:, -1:].contiguous())
    return masked_score_active, topk_reference, reference_kth_threshold


def build_branch_masks(masked_score, snapped_threshold, active_before, selected_equal_mask):
    greater_mask = (masked_score > snapped_threshold) & active_before
    equal_mask = (masked_score == snapped_threshold) & active_before
    branch_mask = greater_mask | selected_equal_mask
    return greater_mask, equal_mask, branch_mask


def build_branch_checks(selected_equal_mask, equal_mask, greater_mask, tie_keep_quota, branch_mask, topk_mask):
    subset_ok = bool(((selected_equal_mask & ~equal_mask).sum() == 0).item())
    disjoint_ok = bool(((selected_equal_mask & greater_mask).sum() == 0).item())
    quota_ok = bool(torch.equal(selected_equal_mask.sum(dim=1).long(), tie_keep_quota))
    reconstructed_topk_ok = bool(torch.equal(branch_mask, topk_mask))
    return {
        'selected_equal_subset_of_equal_mask': subset_ok,
        'selected_equal_disjoint_from_greater_mask': disjoint_ok,
        'selected_equal_count_matches_tie_keep_quota': quota_ok,
        'reconstructed_branch_matches_topk_reference': reconstructed_topk_ok,
    }


def build_stage_boundary_contracts(
    input_stage,
    masked_score,
    active_before,
    kth_threshold,
    snapped_threshold,
    reference_kth_threshold,
    snap_distance,
    selected_equal_mask,
    tie_keep_quota,
    greater_mask,
    equal_mask,
    branch_mask,
    stage_metrics,
    threshold_tolerance,
    checks,
):
    boundary_tie_sample_mask = equal_mask.sum(dim=1) > tie_keep_quota
    return {
        'stage_index': int(input_stage['stage_index']),
        'pruning_layer': int(input_stage['pruning_layer']),
        'keep_count': int(input_stage['keep_count']),
        'sample_count': int(masked_score.shape[0]),
        'input_contract': {
            'masked_score': tensor_summary(masked_score),
            'active_before': bool_count_summary(active_before),
        },
        'kth_payload_contract': {
            'kth_threshold': tensor_summary(kth_threshold.squeeze(1)),
            'snapped_kth_threshold': tensor_summary(snapped_threshold.squeeze(1)),
            'kth_threshold_is_finite': bool(torch.isfinite(kth_threshold).all().item()),
            'max_abs_error_vs_argsort_reference': float((kth_threshold - reference_kth_threshold).abs().max().item()),
            'max_abs_error_after_snap_vs_argsort_reference': float((snapped_threshold - reference_kth_threshold).abs().max().item()),
            'threshold_snap': threshold_snap_summary(snap_distance, threshold_tolerance),
        },
        'tie_payload_contract': {
            'selected_equal_mask': bool_count_summary(selected_equal_mask),
            'tie_keep_quota': tensor_summary(tie_keep_quota),
            'selected_equal_subset_of_equal_mask': checks['selected_equal_subset_of_equal_mask'],
            'selected_equal_disjoint_from_greater_mask': checks['selected_equal_disjoint_from_greater_mask'],
            'selected_equal_count_matches_tie_keep_quota': checks['selected_equal_count_matches_tie_keep_quota'],
        },
        'reconstructed_branch_contract': {
            'greater_mask': bool_count_summary(greater_mask),
            'equal_mask': bool_count_summary(equal_mask),
            'branch_mask': bool_count_summary(branch_mask),
            'boundary_tie_sample_ratio': float(boundary_tie_sample_mask.float().mean().item()),
            'reconstructed_branch_matches_topk_reference': checks['reconstructed_branch_matches_topk_reference'],
        },
        'metrics': stage_metrics,
        'overall_passed': all(
            [
                checks['selected_equal_subset_of_equal_mask'],
                checks['selected_equal_disjoint_from_greater_mask'],
                checks['selected_equal_count_matches_tie_keep_quota'],
                checks['reconstructed_branch_matches_topk_reference'],
                bool(torch.isfinite(kth_threshold).all().item()),
                stage_metrics['exact_count_match_ratio'] == 1.0,
                stage_metrics['exact_mask_match_ratio'] == 1.0,
            ]
        ),
    }


def build_stage_boundary_report(input_stage, kth_stage, tie_stage, threshold_tolerance, reference_tie_policy):
    validate_stage_alignment(input_stage, kth_stage, tie_stage)

    masked_score = input_stage['masked_score'].detach().float().cpu()
    active_before = resolve_active_before(input_stage, masked_score)

    keep_count = int(input_stage['keep_count'])
    _masked_score_active, topk_reference, reference_kth_threshold = build_reference_topk(
        masked_score,
        active_before,
        keep_count,
        reference_tie_policy,
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

    selected_equal_mask = tie_stage['selected_equal_mask'].detach().bool().cpu()
    tie_keep_quota = tie_stage['tie_keep_quota'].detach().long().cpu()
    greater_mask, equal_mask, branch_mask = build_branch_masks(
        masked_score,
        snapped_threshold,
        active_before,
        selected_equal_mask,
    )
    checks = build_branch_checks(
        selected_equal_mask,
        equal_mask,
        greater_mask,
        tie_keep_quota,
        branch_mask,
        topk_mask,
    )
    stage_metrics = mask_metrics(branch_mask, topk_mask)
    return build_stage_boundary_contracts(
        input_stage,
        masked_score,
        active_before,
        kth_threshold,
        snapped_threshold,
        reference_kth_threshold,
        snap_distance,
        selected_equal_mask,
        tie_keep_quota,
        greater_mask,
        equal_mask,
        branch_mask,
        stage_metrics,
        threshold_tolerance,
        checks,
    )


def build_boundary_report(
    bundle_dir: Path,
    input_pt: Path,
    kth_payload_pt: Path,
    tie_payload_pt: Path,
    sample_root_from: str,
    sample_root_to: str,
    threshold_tolerance: float,
):
    input_payload, kth_payload, tie_payload = load_stage_payloads(input_pt, kth_payload_pt, tie_payload_pt)

    input_stages = stage_map(input_payload)
    kth_stages = stage_map(kth_payload)
    tie_stages = stage_map(tie_payload)
    ensure_matching_stage_sets(input_stages, kth_stages, tie_stages)

    sample_paths = input_payload.get('sample_paths', [])
    sample_path_report, resolved_sample_paths = collect_sample_path_report(sample_paths, sample_root_from, sample_root_to)
    reference_tie_policy = tie_payload.get('candidate_metadata', {}).get('tie_policy', 'lowest_index')
    stage_reports = [
        build_stage_boundary_report(
            input_stages[index],
            kth_stages[index],
            tie_stages[index],
            threshold_tolerance,
            reference_tie_policy,
        )
        for index in sorted(input_stages)
    ]

    metadata_checks = {
        'sample_count_consistent': len(sample_paths) == stage_reports[0]['sample_count'] if stage_reports else True,
        'stage_count': len(stage_reports),
    }

    return {
        'bundle_dir': str(bundle_dir.resolve()),
        'input_pt': str(input_pt.resolve()),
        'kth_payload_pt': str(kth_payload_pt.resolve()),
        'tie_payload_pt': str(tie_payload_pt.resolve()),
        'sample_count': len(sample_paths) if sample_paths else (stage_reports[0]['sample_count'] if stage_reports else 0),
        'boundary_map': {
            'externalized_secure_components': [
                'masked_score -> compare-network kth_threshold extraction',
                'kth_threshold -> boundary tie selection payload',
            ],
            'local_plaintext_components': [
                'patch embedding and transformer blocks',
                'score predictor forward that emits masked_score',
                'token masking application F_mux',
                'classification head and final binary threshold',
            ],
            'server_ready_entrypoint': 'tools/transshield_openbumblebee_pipeline.py replay',
        },
        'payload_metadata': {
            'input_candidate_metadata': input_payload.get('candidate_metadata', {}),
            'kth_candidate_metadata': kth_payload.get('candidate_metadata', {}),
            'tie_candidate_metadata': tie_payload.get('candidate_metadata', {}),
            'input_sample_paths': sample_path_report,
            'metadata_checks': metadata_checks,
            'threshold_tolerance': threshold_tolerance,
        },
        'stage_reports': stage_reports,
        'overall_passed': all(item['overall_passed'] for item in stage_reports) and metadata_checks['sample_count_consistent'],
        'resolved_sample_paths': [str(path) for path in resolved_sample_paths],
    }


def run_transformer_block(blk, x, prev_decision, policy):
    x = blk(x, policy=policy)
    return torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)


def slice_tie_payload_stage(tie_stage, sample_offset, batch_size, device):
    selected_equal_mask = tie_stage['selected_equal_mask'][sample_offset : sample_offset + batch_size].to(device).bool()
    tie_keep_quota = tie_stage['tie_keep_quota'][sample_offset : sample_offset + batch_size].to(device).long()
    return selected_equal_mask, tie_keep_quota


def build_replay_stage_report(
    input_stage,
    snapped_threshold,
    kth_threshold,
    snap_distance,
    branch_mask,
    topk_mask,
    active_before_matches_payload,
    checks,
    threshold_tolerance,
):
    return {
        'stage_index': int(input_stage['stage_index']),
        'pruning_layer': int(input_stage['pruning_layer']),
        'threshold_mode': 'external_secure_kth_plus_tie_payload',
        'threshold_summary': tensor_summary(snapped_threshold.squeeze(1)),
        'raw_threshold_summary': tensor_summary(kth_threshold.squeeze(1)),
        'threshold_snap': threshold_snap_summary(snap_distance, threshold_tolerance),
        'metrics': mask_metrics(branch_mask.detach().cpu(), topk_mask.detach().cpu()),
        'checks': {
            'active_before_matches_payload': active_before_matches_payload,
            **checks,
        },
    }


def process_external_pruning_stage(
    model,
    blk,
    x,
    prev_decision,
    policy,
    stage_index,
    input_stages,
    kth_stages,
    tie_stages,
    sample_offset,
    threshold_tolerance,
    reference_tie_policy,
):
    batch_size = x.shape[0]
    if model.use_mask_pruning:
        x = model._apply_spatial_mask(x, prev_decision)
    spatial_x = x[:, 1:]
    pred_score = model.score_predictor[stage_index](spatial_x, prev_decision).reshape(batch_size, -1, 2)
    score = pred_score[:, :, 0]
    active_before = prev_decision.squeeze(-1) > 0
    masked_score = score.masked_fill(~active_before, float('-inf'))

    input_stage = input_stages[stage_index]
    kth_stage = kth_stages[stage_index]
    tie_stage = tie_stages[stage_index]

    expected_keep = int(input_stage['keep_count'])
    keep_count = int((x.shape[1] - 1) * model.token_ratio[stage_index])
    if keep_count != expected_keep:
        raise ValueError(f'keep_count mismatch between model and payload at stage {stage_index}: {keep_count} vs {expected_keep}')

    _masked_score_active, topk_reference, _reference_kth_threshold = build_reference_topk(
        masked_score,
        active_before,
        keep_count,
        reference_tie_policy,
    )
    topk_mask = topk_reference['topk_mask']

    kth_threshold = kth_stage['kth_threshold'][sample_offset : sample_offset + batch_size].to(inputs_device := x.device).view(-1, 1)
    snapped_threshold, snap_distance = resolve_boundary_threshold(
        masked_score,
        kth_threshold,
        active_before,
        keep_count,
        threshold_tolerance,
    )
    selected_equal_mask, tie_keep_quota = slice_tie_payload_stage(
        tie_stage,
        sample_offset,
        batch_size,
        inputs_device,
    )
    greater_mask, equal_mask, branch_mask = build_branch_masks(
        masked_score,
        snapped_threshold,
        active_before,
        selected_equal_mask,
    )
    checks = build_branch_checks(
        selected_equal_mask,
        equal_mask,
        greater_mask,
        tie_keep_quota,
        branch_mask,
        topk_mask,
    )

    prev_decision = branch_mask.unsqueeze(-1).to(dtype=x.dtype)
    cls_policy = torch.ones(batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
    policy = torch.cat([cls_policy, prev_decision], dim=1)
    x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
    x = run_transformer_block(blk, x, prev_decision, policy)

    reference_active_before = input_stage.get('active_before')
    active_before_matches_payload = True
    if reference_active_before is not None:
        reference_active_before = reference_active_before[sample_offset : sample_offset + batch_size].to(inputs_device).bool()
        active_before_matches_payload = bool(torch.equal(reference_active_before, active_before))

    stage_report = build_replay_stage_report(
        input_stage,
        snapped_threshold,
        kth_threshold,
        snap_distance,
        branch_mask,
        topk_mask,
        active_before_matches_payload,
        checks,
        threshold_tolerance,
    )
    return x, prev_decision, policy, stage_report


def finalize_model_logits(model, x):
    x = model.norm(x)
    x = x[:, 0]
    x = model.pre_logits(x)
    return model.head(x)


@torch.no_grad()
def forward_external_secure_pruning(
    model,
    inputs,
    input_stages,
    kth_stages,
    tie_stages,
    sample_offset,
    threshold_tolerance,
    reference_tie_policy,
):
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
            x, prev_decision, policy, stage_report = process_external_pruning_stage(
                model,
                blk,
                x,
                prev_decision,
                policy,
                stage_index,
                input_stages,
                kth_stages,
                tie_stages,
                sample_offset,
                threshold_tolerance,
                reference_tie_policy,
            )
            stage_reports.append(stage_report)
            stage_index += 1
        else:
            x = run_transformer_block(blk, x, prev_decision, policy)

    return finalize_model_logits(model, x), stage_reports


class ReplaySamplePathDataset(Dataset):
    def __init__(self, sample_paths, transform):
        self.sample_paths = [Path(path) for path in sample_paths]
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        path = self.sample_paths[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        parent_name = path.parent.name
        target = int(parent_name) if parent_name.isdigit() else -1
        return tensor, str(path), target


def weighted_average(values, weights):
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def aggregate_tensor_summary(stage_items, summary_key, total_weight):
    first_shape = stage_items[0][summary_key]['shape']
    target_summary = {'shape': [int(total_weight), *first_shape[1:]] if first_shape else [int(total_weight)]}
    target_summary['finite_count'] = int(sum(int(item[summary_key]['finite_count']) for item in stage_items))
    target_summary['nonfinite_count'] = int(sum(int(item[summary_key]['nonfinite_count']) for item in stage_items))
    for key in ['min', 'max']:
        values = [item[summary_key][key] for item in stage_items if item[summary_key][key] is not None]
        target_summary[key] = float(min(values) if key == 'min' else max(values)) if values else None
    weights = [int(item.get('sample_count', 0) or 0) for item in stage_items]
    for key in ['mean', 'std']:
        values = [item[summary_key][key] for item in stage_items]
        present_values = [(float(value), weight) for value, weight in zip(values, weights) if value is not None]
        target_summary[key] = weighted_average(
            [value for value, _ in present_values],
            [weight for _, weight in present_values],
        )
    return target_summary


def aggregate_threshold_snap(stage_items):
    sample_counts = [int(item['threshold_snap']['sample_count']) for item in stage_items]
    threshold_snap = {
        'tolerance': float(stage_items[0]['threshold_snap']['tolerance']),
        'snapped_count': int(sum(int(item['threshold_snap']['snapped_count']) for item in stage_items)),
        'sample_count': int(sum(sample_counts)),
    }
    max_distances = [item['threshold_snap']['max_distance'] for item in stage_items if item['threshold_snap']['max_distance'] is not None]
    threshold_snap['max_distance'] = float(max(max_distances)) if max_distances else None
    mean_values = [item['threshold_snap']['mean_distance'] for item in stage_items]
    present_means = [(float(value), weight) for value, weight in zip(mean_values, sample_counts) if value is not None]
    threshold_snap['mean_distance'] = weighted_average(
        [value for value, _ in present_means],
        [weight for _, weight in present_means],
    )
    return threshold_snap


def aggregate_stage_metrics(stage_items, total_weight):
    weights = [int(item.get('sample_count', 0) or 0) for item in stage_items]
    metrics = {}
    for key in stage_items[0]['metrics']:
        metrics[key] = float(
            sum(float(item['metrics'][key]) * weight for item, weight in zip(stage_items, weights))
            / total_weight
        )
    return metrics


def aggregate_stage_reports(all_stage_reports):
    if not all_stage_reports:
        return []

    stage_count = len(all_stage_reports[0])
    aggregated = []
    for stage_index in range(stage_count):
        stage_items = [batch[stage_index] for batch in all_stage_reports]
        total_weight = sum(int(item.get('sample_count', 0) or 0) for item in stage_items)
        if total_weight <= 0:
            total_weight = len(stage_items)
        threshold_summary = aggregate_tensor_summary(stage_items, 'threshold_summary', total_weight)
        raw_threshold_summary = aggregate_tensor_summary(stage_items, 'raw_threshold_summary', total_weight)
        threshold_snap = aggregate_threshold_snap(stage_items)
        metrics = aggregate_stage_metrics(stage_items, total_weight)
        checks = {key: all(bool(item['checks'][key]) for item in stage_items) for key in stage_items[0]['checks']}

        aggregated.append(
            {
                'stage_index': int(stage_items[0]['stage_index']),
                'pruning_layer': int(stage_items[0]['pruning_layer']),
                'sample_count': int(total_weight),
                'threshold_mode': stage_items[0]['threshold_mode'],
                'threshold_summary': threshold_summary,
                'raw_threshold_summary': raw_threshold_summary,
                'threshold_snap': threshold_snap,
                'metrics': metrics,
                'checks': checks,
            }
        )
    return aggregated


def resolve_replay_sample_paths(boundary_report, max_samples: int):
    resolved_sample_paths = [Path(path) for path in boundary_report['resolved_sample_paths']]
    if not resolved_sample_paths:
        raise ValueError('model replay requires sample_paths in the input payload')
    if any(not path.exists() for path in resolved_sample_paths):
        missing = [str(path) for path in resolved_sample_paths if not path.exists()]
        raise FileNotFoundError(f'model replay sample paths are missing: {missing[:4]}')
    if max_samples > 0:
        resolved_sample_paths = resolved_sample_paths[:max_samples]
    return resolved_sample_paths


def load_replay_runtime_inputs(boundary_report):
    input_payload, kth_payload, tie_payload = load_stage_payloads(
        Path(boundary_report['input_pt']),
        Path(boundary_report['kth_payload_pt']),
        Path(boundary_report['tie_payload_pt']),
    )
    return stage_map(input_payload), stage_map(kth_payload), stage_map(tie_payload), tie_payload


def build_replay_loader(bundle, resolved_sample_paths, batch_size: int, num_workers: int):
    dataset = ReplaySamplePathDataset(resolved_sample_paths, bundle['transform'])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_replay_result(
    logits,
    probs,
    all_sample_paths,
    all_targets,
    stage_reports,
    device: str,
    weights_path: Path,
    class_threshold,
    threshold_tolerance: float,
    batch_size: int,
    num_workers: int,
):
    result = {
        'status': 'ok',
        'device': device,
        'sample_count': len(all_sample_paths),
        'weights_path': str(weights_path.resolve()),
        'class_threshold': float(class_threshold) if class_threshold is not None else None,
        'threshold_tolerance': threshold_tolerance,
        'batch_size': int(batch_size),
        'num_workers': int(num_workers),
        'logits': logits.detach().cpu().tolist(),
        'probabilities': probs.tolist(),
        'argmax_predictions': logits.argmax(dim=1).detach().cpu().tolist(),
        'threshold_predictions': (
            (probs[:, 1] >= class_threshold).long().tolist() if class_threshold is not None and probs.shape[1] == 2 else None
        ),
        'sample_paths': list(all_sample_paths),
        'stage_reports': stage_reports,
    }
    if all(target >= 0 for target in all_targets):
        targets = torch.tensor(all_targets, dtype=torch.long)
        result['targets'] = targets.tolist()
        result['argmax_accuracy'] = float((logits.detach().cpu().argmax(dim=1) == targets).float().mean().item() * 100.0)
        if class_threshold is not None and probs.shape[1] == 2:
            threshold_predictions = (probs[:, 1] >= class_threshold).long()
            result['threshold_accuracy'] = float((threshold_predictions == targets).float().mean().item() * 100.0)
    return result


def run_model_replay(
    boundary_report,
    bundle_dir: Path,
    device: str,
    max_samples: int,
    threshold_tolerance: float,
    batch_size: int,
    num_workers: int,
):
    weights_path = resolve_model_state_dict_path(bundle_dir)
    resolved_sample_paths = resolve_replay_sample_paths(boundary_report, max_samples)

    bundle = load_frozen_bundle(bundle_dir, device)
    class_threshold = resolve_threshold(bundle_dir, None)
    input_stages, kth_stages, tie_stages, tie_payload = load_replay_runtime_inputs(boundary_report)
    loader = build_replay_loader(bundle, resolved_sample_paths, batch_size, num_workers)
    replay_tie_policy = tie_payload.get('candidate_metadata', {}).get(
        'tie_policy',
        bundle['args_snapshot'].get('eval_tie_policy', 'lowest_index'),
    )

    all_logits = []
    all_sample_paths = []
    all_targets = []
    all_stage_reports = []
    sample_offset = 0
    for inputs, batch_paths, batch_targets in loader:
        inputs = inputs.to(device)
        logits, stage_reports = forward_external_secure_pruning(
            bundle['model'],
            inputs,
            input_stages,
            kth_stages,
            tie_stages,
            sample_offset,
            threshold_tolerance,
            replay_tie_policy,
        )
        for stage_report in stage_reports:
            stage_report['sample_count'] = int(inputs.shape[0])
        all_logits.append(logits.detach().cpu())
        all_sample_paths.extend(batch_paths)
        all_targets.extend(batch_targets.tolist())
        all_stage_reports.append(stage_reports)
        sample_offset += inputs.shape[0]

    logits = torch.cat(all_logits, dim=0)
    stage_reports = aggregate_stage_reports(all_stage_reports)
    probs = torch.softmax(logits, dim=-1).detach().cpu()

    return build_replay_result(
        logits,
        probs,
        all_sample_paths,
        all_targets,
        stage_reports,
        device,
        weights_path,
        class_threshold,
        threshold_tolerance,
        batch_size,
        num_workers,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Unified Transshield/OpenBumbleBee inference-boundary replay: validate masked_score -> kth payload -> tie payload and optionally replay the final model.'
    )
    parser.add_argument('--bundle-dir', default=str(REPO_ROOT / 'artifacts' / 'frozen_bundle'))
    parser.add_argument('--input-pt', required=True)
    parser.add_argument('--kth-payload-pt', required=True)
    parser.add_argument('--tie-payload-pt', required=True)
    parser.add_argument('--sample-root-from', default='')
    parser.add_argument('--sample-root-to', default='')
    parser.add_argument('--enable-model-replay', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threshold-tolerance', type=float, default=DEFAULT_THRESHOLD_TOLERANCE)
    parser.add_argument('--output-json', required=True)
    return parser


def main():
    args = build_parser().parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    report = build_boundary_report(
        bundle_dir=bundle_dir,
        input_pt=Path(args.input_pt).resolve(),
        kth_payload_pt=Path(args.kth_payload_pt).resolve(),
        tie_payload_pt=Path(args.tie_payload_pt).resolve(),
        sample_root_from=args.sample_root_from,
        sample_root_to=args.sample_root_to,
        threshold_tolerance=args.threshold_tolerance,
    )

    if args.enable_model_replay:
        report['model_replay'] = run_model_replay(
            report,
            bundle_dir,
            args.device,
            args.max_samples,
            args.threshold_tolerance,
            args.batch_size,
            args.num_workers,
        )
    else:
        report['model_replay'] = {
            'status': 'skipped',
            'reason': 'enable with --enable-model-replay after copying frozen weights and dataset paths into the standalone/server environment',
        }

    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    Path(args.output_json).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
