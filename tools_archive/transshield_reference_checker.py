import argparse
import json
from pathlib import Path

import torch


def tensor_stats(tensor):
    tensor = tensor.detach().float().cpu()
    finite_mask = torch.isfinite(tensor)
    finite = tensor[finite_mask]
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


def max_abs_error(lhs, rhs):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    return float((lhs - rhs).abs().max().item())


def build_attn_policy_from_policy(policy):
    batch_size, token_count, _ = policy.shape
    attn_policy = policy.reshape(batch_size, 1, 1, token_count)
    eye = torch.eye(token_count, dtype=attn_policy.dtype).view(1, 1, token_count, token_count)
    return attn_policy + (1.0 - attn_policy) * eye


def apply_spatial_mask(sequence, prev_decision):
    return torch.cat([sequence[:, :1], sequence[:, 1:] * prev_decision], dim=1)


def load_trace_payload(path):
    payload = torch.load(Path(path).resolve(), map_location='cpu', weights_only=False)
    if 'trace_tensors' not in payload:
        raise ValueError(f'missing trace_tensors in payload: {path}')
    return payload


def compare_optional(reference_tensors, candidate_tensors, key, tolerance):
    if candidate_tensors is None or key not in candidate_tensors:
        return {
            'available': False,
            'passed': None,
        }
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

    checks = []
    all_passed = True

    for stage_index in stage_indices:
        prev_decision_key = f'stage_{stage_index}_prev_decision'
        policy_key = f'stage_{stage_index}_policy'
        attn_policy_key = f'stage_{stage_index}_attn_policy'
        block_output_key = f'stage_{stage_index}_block_output'
        masked_output_key = f'stage_{stage_index}_masked_output'
        masked_input_key = f'stage_{stage_index}_masked_input_seq'
        input_key = f'stage_{stage_index}_input_seq'

        prev_decision = reference_tensors[prev_decision_key]
        expected_policy = torch.cat(
            [torch.ones(prev_decision.shape[0], 1, 1, dtype=prev_decision.dtype), prev_decision], dim=1
        )
        expected_attn_policy = build_attn_policy_from_policy(reference_tensors[policy_key])
        expected_masked_output = apply_spatial_mask(reference_tensors[block_output_key], prev_decision)
        if stage_index == 0:
            prior_mask = torch.ones_like(prev_decision)
        else:
            prior_mask = reference_tensors[f'stage_{stage_index - 1}_prev_decision']
        expected_masked_input = apply_spatial_mask(reference_tensors[input_key], prior_mask)

        policy_error = max_abs_error(reference_tensors[policy_key], expected_policy)
        attn_policy_error = max_abs_error(reference_tensors[attn_policy_key], expected_attn_policy)
        masked_output_error = max_abs_error(reference_tensors[masked_output_key], expected_masked_output)
        masked_input_error = max_abs_error(reference_tensors[masked_input_key], expected_masked_input)

        stage_passed = all(
            error <= tolerance
            for error in (policy_error, attn_policy_error, masked_output_error, masked_input_error)
        )
        all_passed = all_passed and stage_passed

        checks.append(
            {
                'stage_index': stage_index,
                'reference_self_checks': {
                    'policy_from_prev_decision': {
                        'max_abs_error': policy_error,
                        'tolerance': tolerance,
                        'passed': bool(policy_error <= tolerance),
                    },
                    'attn_policy_from_policy': {
                        'max_abs_error': attn_policy_error,
                        'tolerance': tolerance,
                        'passed': bool(attn_policy_error <= tolerance),
                    },
                    'masked_output_from_block_output_and_mask': {
                        'max_abs_error': masked_output_error,
                        'tolerance': tolerance,
                        'passed': bool(masked_output_error <= tolerance),
                    },
                    'masked_input_from_input_seq_and_prior_mask': {
                        'max_abs_error': masked_input_error,
                        'tolerance': tolerance,
                        'passed': bool(masked_input_error <= tolerance),
                    },
                },
                'candidate_comparisons': {
                    'prev_decision': compare_optional(reference_tensors, candidate_tensors, prev_decision_key, tolerance),
                    'policy': compare_optional(reference_tensors, candidate_tensors, policy_key, tolerance),
                    'attn_policy': compare_optional(reference_tensors, candidate_tensors, attn_policy_key, tolerance),
                    'masked_output': compare_optional(reference_tensors, candidate_tensors, masked_output_key, tolerance),
                },
                'reference_stats': {
                    'prev_decision': tensor_stats(reference_tensors[prev_decision_key]),
                    'policy': tensor_stats(reference_tensors[policy_key]),
                    'attn_policy': tensor_stats(reference_tensors[attn_policy_key]),
                    'masked_output': tensor_stats(reference_tensors[masked_output_key]),
                },
                'passed': stage_passed,
            }
        )

    logits_compare = compare_optional(reference_tensors, candidate_tensors, 'logits', tolerance)
    probabilities_compare = compare_optional(reference_tensors, candidate_tensors, 'probabilities', tolerance)

    if logits_compare['available']:
        all_passed = all_passed and bool(logits_compare['passed'])
    if probabilities_compare['available']:
        all_passed = all_passed and bool(probabilities_compare['passed'])

    return {
        'overall_passed': bool(all_passed),
        'tolerance': tolerance,
        'stage_checks': checks,
        'final_output_comparisons': {
            'logits': logits_compare,
            'probabilities': probabilities_compare,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Check Transshield stage-2 reference trace relations and compare external tensors when available.')
    parser.add_argument('--reference-pt', required=True, help='Reference trace .pt produced by tools/transshield_forward_trace.py')
    parser.add_argument('--candidate-pt', default='', help='Optional external implementation tensor payload with matching trace_tensors keys')
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
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
