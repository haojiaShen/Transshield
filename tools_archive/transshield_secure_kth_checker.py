import argparse
import json
from pathlib import Path

import torch


def tensor_stats(tensor):
    tensor = tensor.detach().float().cpu()
    return {
        'shape': list(tensor.shape),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
    }


def max_abs_error(lhs, rhs):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    return float((lhs - rhs).abs().max().item())


def load_trace_payload(path):
    payload = torch.load(Path(path).resolve(), map_location='cpu', weights_only=False)
    if 'trace_tensors' not in payload:
        raise ValueError(f'missing trace_tensors in payload: {path}')
    return payload


def compare_optional(reference_tensors, candidate_tensors, key, tolerance):
    if candidate_tensors is None or key not in candidate_tensors:
        return {'available': False, 'passed': None}
    error = max_abs_error(reference_tensors[key], candidate_tensors[key])
    return {
        'available': True,
        'max_abs_error': error,
        'tolerance': tolerance,
        'passed': bool(error <= tolerance),
    }


def run_checks(reference_tensors, candidate_tensors=None, tolerance=1e-6):
    stage_indices = sorted(
        {
            int(key.split('_')[1])
            for key in reference_tensors
            if key.startswith('stage_') and key.endswith('_prev_decision')
        }
    )

    stage_checks = []
    overall_passed = True
    for stage_index in stage_indices:
        gt_key = f'stage_{stage_index}_greater_mask'
        eq_key = f'stage_{stage_index}_equal_mask'
        sel_key = f'stage_{stage_index}_selected_equal_mask'
        quota_key = f'stage_{stage_index}_tie_keep_quota'
        prev_key = f'stage_{stage_index}_prev_decision'

        gt_mask = reference_tensors[gt_key].bool()
        eq_mask = reference_tensors[eq_key].bool()
        sel_mask = reference_tensors[sel_key].bool()
        quota = reference_tensors[quota_key].long()
        prev_decision = reference_tensors[prev_key].squeeze(-1).bool()

        selected_subset_error = float((sel_mask & ~eq_mask).float().sum().item())
        disjoint_error = float((sel_mask & gt_mask).float().sum().item())
        reconstructed = gt_mask | sel_mask
        reconstruction_error = max_abs_error(reconstructed.float(), prev_decision.float())
        selected_count = sel_mask.sum(dim=1).long()
        quota_error = max_abs_error(selected_count.float(), quota.float())

        stage_passed = (
            selected_subset_error == 0.0
            and disjoint_error == 0.0
            and reconstruction_error <= tolerance
            and quota_error <= tolerance
        )
        overall_passed = overall_passed and stage_passed

        stage_checks.append(
            {
                'stage_index': stage_index,
                'reference_self_checks': {
                    'selected_equal_subset_of_equal_mask': {
                        'violation_count': selected_subset_error,
                        'passed': bool(selected_subset_error == 0.0),
                    },
                    'greater_mask_disjoint_from_selected_equal': {
                        'violation_count': disjoint_error,
                        'passed': bool(disjoint_error == 0.0),
                    },
                    'reconstructed_mask_matches_prev_decision': {
                        'max_abs_error': reconstruction_error,
                        'tolerance': tolerance,
                        'passed': bool(reconstruction_error <= tolerance),
                    },
                    'selected_equal_count_matches_tie_keep_quota': {
                        'max_abs_error': quota_error,
                        'tolerance': tolerance,
                        'passed': bool(quota_error <= tolerance),
                    },
                },
                'candidate_comparisons': {
                    'kth_threshold': compare_optional(reference_tensors, candidate_tensors, f'stage_{stage_index}_kth_threshold', tolerance),
                    'greater_mask': compare_optional(reference_tensors, candidate_tensors, gt_key, tolerance),
                    'equal_mask': compare_optional(reference_tensors, candidate_tensors, eq_key, tolerance),
                    'selected_equal_mask': compare_optional(reference_tensors, candidate_tensors, sel_key, tolerance),
                    'tie_keep_quota': compare_optional(reference_tensors, candidate_tensors, quota_key, tolerance),
                    'prev_decision': compare_optional(reference_tensors, candidate_tensors, prev_key, tolerance),
                },
                'reference_stats': {
                    'kth_threshold': tensor_stats(reference_tensors[f'stage_{stage_index}_kth_threshold']),
                    'greater_mask': tensor_stats(reference_tensors[gt_key].float()),
                    'equal_mask': tensor_stats(reference_tensors[eq_key].float()),
                    'selected_equal_mask': tensor_stats(reference_tensors[sel_key].float()),
                    'tie_keep_quota': tensor_stats(reference_tensors[quota_key].float()),
                },
                'passed': stage_passed,
            }
        )

    logits_compare = compare_optional(reference_tensors, candidate_tensors, 'logits', tolerance)
    probabilities_compare = compare_optional(reference_tensors, candidate_tensors, 'probabilities', tolerance)
    if logits_compare['available']:
        overall_passed = overall_passed and bool(logits_compare['passed'])
    if probabilities_compare['available']:
        overall_passed = overall_passed and bool(probabilities_compare['passed'])

    return {
        'overall_passed': bool(overall_passed),
        'tolerance': tolerance,
        'stage_checks': stage_checks,
        'final_output_comparisons': {
            'logits': logits_compare,
            'probabilities': probabilities_compare,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Check Transshield secure-kth reference trace relations and compare external candidate tensors when available.')
    parser.add_argument('--reference-pt', required=True)
    parser.add_argument('--candidate-pt', default='')
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    reference_payload = load_trace_payload(args.reference_pt)
    candidate_payload = load_trace_payload(args.candidate_pt) if args.candidate_pt else None
    candidate_tensors = None if candidate_payload is None else candidate_payload['trace_tensors']

    report = {
        'reference_pt': str(Path(args.reference_pt).resolve()),
        'candidate_pt': str(Path(args.candidate_pt).resolve()) if args.candidate_pt else None,
        'reference_bundle_dir': reference_payload.get('bundle_dir'),
        'image_path': reference_payload.get('image_path'),
        'check_status': run_checks(reference_payload['trace_tensors'], candidate_tensors, tolerance=args.tolerance),
        'constraints': {
            'model_semantics_changed': False,
            'read_only_checker': True,
            'supports_external_tensor_comparison': True,
            'secure_kth_design_reference_only': True,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
