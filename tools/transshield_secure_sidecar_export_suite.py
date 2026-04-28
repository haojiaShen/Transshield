import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_input_selection import (
    SelectedImageDataset,
    add_input_selection_args,
    resolve_selected_sample_paths,
)
from tools.transshield_stage2_bundle import load_frozen_bundle
from tools.transshield_stagewise_threshold_report import collect_stagewise_scores


def summarize_masked_score(masked_score):
    masked_score = masked_score.detach().float().cpu()
    finite = torch.isfinite(masked_score)
    finite_values = masked_score[finite]
    return {
        'shape': list(masked_score.shape),
        'finite_value_count': int(finite.sum().item()),
        'non_finite_value_count': int((~finite).sum().item()),
        'finite_min': float(finite_values.min().item()) if finite_values.numel() > 0 else None,
        'finite_max': float(finite_values.max().item()) if finite_values.numel() > 0 else None,
        'finite_mean': float(finite_values.mean().item()) if finite_values.numel() > 0 else None,
        'finite_std': float(finite_values.std(unbiased=False).item()) if finite_values.numel() > 1 else 0.0,
    }


def summarize_active_before(active_before):
    active_before = active_before.detach().cpu().bool()
    active_counts = active_before.sum(dim=1).float()
    return {
        'shape': list(active_before.shape),
        'min_active_count': int(active_counts.min().item()),
        'max_active_count': int(active_counts.max().item()),
        'mean_active_count': float(active_counts.mean().item()),
        'std_active_count': float(active_counts.std(unbiased=False).item()) if active_counts.numel() > 1 else 0.0,
    }


def summarize_float_tensor(tensor):
    tensor = tensor.detach().float().cpu()
    return {
        'shape': list(tensor.shape),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
    }


def summarize_bool_tensor(tensor):
    tensor = tensor.bool()
    counts = tensor.sum(dim=1).float()
    return {
        'shape': list(tensor.shape),
        'mean_true_count': float(counts.mean().item()),
        'max_true_count': int(counts.max().item()),
        'min_true_count': int(counts.min().item()),
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Export Transshield secure sidecar inputs, reference kth payload, and reference tie payload in one forward pass.'
    )
    parser.add_argument('--bundle-dir', required=True)
    add_input_selection_args(parser)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--input-output-pt', required=True)
    parser.add_argument('--input-output-json', default='')
    parser.add_argument('--kth-output-pt', required=True)
    parser.add_argument('--kth-output-json', default='')
    parser.add_argument('--tie-output-pt', required=True)
    parser.add_argument('--tie-output-json', default='')
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    selection = resolve_selected_sample_paths(
        data_path=args.data_path,
        default_data_path=bundle['args_snapshot'].get('eval_data_path', ''),
        image_paths=args.image,
        image_list=args.image_list,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        max_samples=args.max_samples,
    )
    dataset = SelectedImageDataset(selection['sample_paths'], bundle['transform'])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

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
                    'masked_score': [],
                    'active_before': [],
                    'kth_threshold': [],
                    'selected_equal_mask': [],
                    'tie_keep_quota': [],
                }
                for stage in batch_reports
            ]

        for stage_index, stage in enumerate(batch_reports):
            score = stage['score']
            active_before = stage['active_before'].bool()
            topk_mask = stage['topk_mask'].bool()
            masked_score = score.masked_fill(~active_before, float('-inf'))
            kth_threshold = stage['boundary_keep_score'].detach().cpu()
            snapped_threshold = kth_threshold.unsqueeze(1)
            equal_mask = (score == snapped_threshold) & active_before
            greater_mask = (score > snapped_threshold) & active_before
            tie_keep_quota = stage['keep_count'] - greater_mask.sum(dim=1)
            selected_equal_mask = equal_mask & topk_mask

            buffer = stage_buffers[stage_index]
            buffer['masked_score'].append(masked_score.detach().cpu())
            buffer['active_before'].append(active_before.detach().cpu())
            buffer['kth_threshold'].append(kth_threshold)
            buffer['selected_equal_mask'].append(selected_equal_mask.detach().cpu())
            buffer['tie_keep_quota'].append(tie_keep_quota.detach().cpu())

        sample_count += inputs.shape[0]

    if stage_buffers is None:
        raise ValueError('no samples processed')

    sample_paths = [str(path) for path in dataset.sample_paths]

    input_payload_stages = []
    input_summary_stages = []
    kth_payload_stages = []
    kth_summary_stages = []
    tie_payload_stages = []
    tie_summary_stages = []

    for stage in stage_buffers:
        stage_index = int(stage['stage_index'])
        pruning_layer = int(stage['pruning_layer'])
        keep_count = int(stage['keep_count'])
        masked_score = torch.cat(stage['masked_score'], dim=0)
        active_before = torch.cat(stage['active_before'], dim=0)
        kth_threshold = torch.cat(stage['kth_threshold'], dim=0)
        selected_equal_mask = torch.cat(stage['selected_equal_mask'], dim=0)
        tie_keep_quota = torch.cat(stage['tie_keep_quota'], dim=0)

        input_payload_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'masked_score': masked_score,
                'active_before': active_before,
            }
        )
        input_summary_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'masked_score': summarize_masked_score(masked_score),
                'active_before': summarize_active_before(active_before),
            }
        )

        kth_payload_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'kth_threshold': kth_threshold,
            }
        )
        kth_summary_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'kth_threshold': summarize_float_tensor(kth_threshold),
            }
        )

        tie_payload_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'selected_equal_mask': selected_equal_mask,
                'tie_keep_quota': tie_keep_quota,
            }
        )
        tie_summary_stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'keep_count': keep_count,
                'selected_equal_mask': summarize_bool_tensor(selected_equal_mask),
                'tie_keep_quota': {
                    'shape': list(tie_keep_quota.shape),
                    'mean': float(tie_keep_quota.float().mean().item()),
                    'max': int(tie_keep_quota.max().item()),
                    'min': int(tie_keep_quota.min().item()),
                },
            }
        )

    common_payload = {
        'bundle_dir': str(bundle_dir),
        'data_path': selection['data_path'],
        'selection_mode': selection['selection_mode'],
        'sample_paths': sample_paths,
    }

    input_payload = {
        **common_payload,
        'stages': input_payload_stages,
        'candidate_metadata': {
            'format_purpose': 'secure_network_kth_external_input_sidecar',
            'generated_from_frozen_model_forward': True,
            'masked_score_required_by_manifest': True,
            'model_semantics_changed': False,
            'export_suite_single_forward_pass': True,
        },
    }
    kth_payload = {
        **common_payload,
        'stages': kth_payload_stages,
        'candidate_metadata': {
            'format_purpose': 'secure_network_kth_reference_topk_sidecar',
            'source_reference_topk': True,
            'source_compare_network': False,
            'model_semantics_changed': False,
            'export_suite_single_forward_pass': True,
        },
    }
    tie_payload = {
        **common_payload,
        'stages': tie_payload_stages,
        'candidate_metadata': {
            'format_purpose': 'secure_tie_payload_reference_sidecar',
            'source_reference_topk': True,
            'model_semantics_changed': False,
            'export_suite_single_forward_pass': True,
        },
    }

    input_output_pt = Path(args.input_output_pt).resolve()
    kth_output_pt = Path(args.kth_output_pt).resolve()
    tie_output_pt = Path(args.tie_output_pt).resolve()
    input_output_pt.parent.mkdir(parents=True, exist_ok=True)
    kth_output_pt.parent.mkdir(parents=True, exist_ok=True)
    tie_output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(input_payload, input_output_pt)
    torch.save(kth_payload, kth_output_pt)
    torch.save(tie_payload, tie_output_pt)

    if args.input_output_json:
        write_json(
            Path(args.input_output_json).resolve(),
            {
                'bundle_dir': str(bundle_dir),
                'data_path': selection['data_path'],
                'selection_mode': selection['selection_mode'],
                'sample_count': sample_count,
                'output_pt': str(input_output_pt),
                'stages': input_summary_stages,
                'constraints': {
                    'input_sidecar_is_read_only': True,
                    'exported_from_frozen_model_forward': True,
                    'contains_masked_score_not_kth_threshold': True,
                    'model_semantics_changed': False,
                    'export_suite_single_forward_pass': True,
                },
            },
        )

    if args.kth_output_json:
        write_json(
            Path(args.kth_output_json).resolve(),
            {
                'bundle_dir': str(bundle_dir),
                'data_path': selection['data_path'],
                'selection_mode': selection['selection_mode'],
                'sample_count': sample_count,
                'output_pt': str(kth_output_pt),
                'stages': kth_summary_stages,
                'constraints': {
                    'source_reference_topk': True,
                    'source_compare_network': False,
                    'sidecar_is_read_only_reference': True,
                    'model_semantics_changed': False,
                    'export_suite_single_forward_pass': True,
                },
            },
        )

    if args.tie_output_json:
        write_json(
            Path(args.tie_output_json).resolve(),
            {
                'bundle_dir': str(bundle_dir),
                'data_path': selection['data_path'],
                'selection_mode': selection['selection_mode'],
                'sample_count': sample_count,
                'output_pt': str(tie_output_pt),
                'stages': tie_summary_stages,
                'constraints': {
                    'source_reference_topk': True,
                    'sidecar_is_read_only_reference': True,
                    'model_semantics_changed': False,
                    'export_suite_single_forward_pass': True,
                },
            },
        )

    print(
        json.dumps(
            {
                'bundle_dir': str(bundle_dir),
                'data_path': selection['data_path'],
                'selection_mode': selection['selection_mode'],
                'sample_count': sample_count,
                'outputs': {
                    'input_pt': str(input_output_pt),
                    'kth_pt': str(kth_output_pt),
                    'tie_pt': str(tie_output_pt),
                },
                'constraints': {
                    'single_model_forward_pass_for_all_sidecars': True,
                    'model_semantics_changed': False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
