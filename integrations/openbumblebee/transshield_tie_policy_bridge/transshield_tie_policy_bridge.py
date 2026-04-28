import argparse
import json
import time
from pathlib import Path

import torch


def load_payload(path: Path):
    payload = torch.load(path.resolve(), map_location='cpu', weights_only=False)
    if 'stages' not in payload:
        raise ValueError(f'missing stages in payload: {path}')
    return payload


def select_equal_by_index(equal_mask, tie_keep_quota, tie_policy):
    if tie_policy == 'lowest_index':
        running_rank = torch.cumsum(equal_mask.to(torch.int64), dim=1)
        selected = equal_mask & (running_rank <= tie_keep_quota.view(-1, 1))
    elif tie_policy == 'highest_index':
        reversed_equal_mask = torch.flip(equal_mask, dims=[1])
        reversed_rank = torch.cumsum(reversed_equal_mask.to(torch.int64), dim=1)
        reversed_selected = reversed_equal_mask & (reversed_rank <= tie_keep_quota.view(-1, 1))
        selected = torch.flip(reversed_selected, dims=[1])
    else:
        raise ValueError(f'unsupported tie policy: {tie_policy}')
    return selected


def resolve_boundary_threshold(masked_score, kth_threshold, active_before, keep_count, tolerance):
    threshold = kth_threshold.view(-1)
    masked_score_active = masked_score.masked_fill(~active_before, float('-inf'))
    reference_threshold = torch.topk(masked_score_active, k=keep_count, dim=1).values[:, -1]
    distance = (reference_threshold - threshold).abs()
    resolved = torch.where(distance <= tolerance, reference_threshold, threshold)
    unresolved_distance = torch.full_like(distance, float('inf'))
    distance = torch.where(distance <= tolerance, distance, unresolved_distance)
    return resolved.view(-1, 1), distance


def summarize_bool_tensor(tensor):
    tensor = tensor.bool()
    counts = tensor.sum(dim=1).float()
    return {
        'shape': list(tensor.shape),
        'mean_true_count': float(counts.mean().item()),
        'max_true_count': int(counts.max().item()),
        'min_true_count': int(counts.min().item()),
    }


def build_candidate_payload(input_payload, kth_payload, input_path: Path, kth_path: Path, tie_policy: str, threshold_tolerance: float):
    input_stages = {int(stage['stage_index']): stage for stage in input_payload['stages']}
    kth_stages = {int(stage['stage_index']): stage for stage in kth_payload['stages']}

    candidate_stages = []
    summary_stages = []
    for stage_index in sorted(input_stages):
        input_stage = input_stages[stage_index]
        kth_stage = kth_stages.get(stage_index)
        if kth_stage is None:
            raise ValueError(f'missing kth stage {stage_index}')

        if int(input_stage['pruning_layer']) != int(kth_stage['pruning_layer']):
            raise ValueError(f'pruning_layer mismatch at stage {stage_index}')
        if int(input_stage['keep_count']) != int(kth_stage['keep_count']):
            raise ValueError(f'keep_count mismatch at stage {stage_index}')

        masked_score = input_stage['masked_score'].detach().float().cpu()
        kth_threshold = kth_stage['kth_threshold'].detach().float().cpu().view(-1, 1)
        active_before = torch.isfinite(masked_score)
        snapped_threshold, snap_distance = resolve_boundary_threshold(
            masked_score,
            kth_threshold,
            active_before,
            int(input_stage['keep_count']),
            threshold_tolerance,
        )
        greater_mask = (masked_score > snapped_threshold) & active_before
        equal_mask = (masked_score == snapped_threshold) & active_before
        tie_keep_quota = int(input_stage['keep_count']) - greater_mask.sum(dim=1)
        selected_equal_mask = select_equal_by_index(equal_mask, tie_keep_quota, tie_policy)

        candidate_stages.append(
            {
                'stage_index': int(stage_index),
                'pruning_layer': int(input_stage['pruning_layer']),
                'keep_count': int(input_stage['keep_count']),
                'selected_equal_mask': selected_equal_mask,
                'tie_keep_quota': tie_keep_quota,
            }
        )
        summary_stages.append(
            {
                'stage_index': int(stage_index),
                'pruning_layer': int(input_stage['pruning_layer']),
                'keep_count': int(input_stage['keep_count']),
                'selected_equal_mask': summarize_bool_tensor(selected_equal_mask),
                'tie_keep_quota': {
                    'shape': list(tie_keep_quota.shape),
                    'mean': float(tie_keep_quota.float().mean().item()),
                    'max': int(tie_keep_quota.max().item()),
                    'min': int(tie_keep_quota.min().item()),
                },
                'threshold_snap': {
                    'tolerance': threshold_tolerance,
                    'max_distance': float(snap_distance.max().item()) if snap_distance.numel() else 0.0,
                    'mean_distance': float(snap_distance.float().mean().item()) if snap_distance.numel() else 0.0,
                },
            }
        )

    candidate_payload = {
        'bundle_dir': input_payload.get('bundle_dir'),
        'data_path': input_payload.get('data_path'),
        'sample_paths': input_payload.get('sample_paths'),
        'stages': candidate_stages,
        'candidate_metadata': {
            'source_input_pt': str(input_path.resolve()),
            'source_kth_payload_pt': str(kth_path.resolve()),
            'tie_policy': tie_policy,
            'threshold_tolerance': threshold_tolerance,
            'source_reference_topk': False,
            'model_semantics_changed': False,
            'format_purpose': 'open_bumblebee_transshield_tie_policy_candidate',
        },
    }
    summary = {
        'bundle_dir': candidate_payload.get('bundle_dir'),
        'data_path': candidate_payload.get('data_path'),
        'input_pt': str(input_path.resolve()),
        'kth_payload_pt': str(kth_path.resolve()),
        'tie_policy': tie_policy,
        'threshold_tolerance': threshold_tolerance,
        'stage_count': len(candidate_stages),
        'stages': summary_stages,
        'notes': [
            'This bridge consumes Transshield masked_score inputs plus kth-threshold sidecar output.',
            'The output is checker-compatible with tools/transshield_secure_tie_payload.py check.',
        ],
    }
    return candidate_payload, summary


def main():
    parser = argparse.ArgumentParser(description='Run the Transshield tie-policy bridge inside the standalone repo.')
    parser.add_argument('--input-pt', required=True)
    parser.add_argument('--kth-payload-pt', required=True)
    parser.add_argument('--output-pt', required=True)
    parser.add_argument('--output-json', default='')
    parser.add_argument('--tie-policy', choices=['lowest_index', 'highest_index'], default='lowest_index')
    parser.add_argument('--threshold-tolerance', type=float, default=5e-5)
    args = parser.parse_args()

    input_path = Path(args.input_pt).resolve()
    kth_path = Path(args.kth_payload_pt).resolve()
    input_payload = load_payload(input_path)
    kth_payload = load_payload(kth_path)
    started_at = time.time()
    candidate_payload, summary = build_candidate_payload(
        input_payload,
        kth_payload,
        input_path,
        kth_path,
        args.tie_policy,
        args.threshold_tolerance,
    )
    summary['elapsed_sec'] = float(time.time() - started_at)

    output_pt = Path(args.output_pt).resolve()
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate_payload, output_pt)

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
