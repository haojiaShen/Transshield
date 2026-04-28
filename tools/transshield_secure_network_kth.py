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
    kth_threshold_from_compare_network,
    mask_metrics,
    select_equal_by_index,
    threshold_accuracy_from_logits,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def print_and_optionally_write(payload, output_json: str = ''):
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_json:
        Path(output_json).resolve().write_text(text + '\n', encoding='utf-8')


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


def export_input_sidecar(args):
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
                    'masked_score': [],
                    'active_before': [],
                }
                for stage in batch_reports
            ]

        for stage_index, stage in enumerate(batch_reports):
            active_before = stage['active_before'].bool()
            masked_score = stage['score'].masked_fill(~active_before, float('-inf'))
            stage_buffers[stage_index]['masked_score'].append(masked_score.detach().cpu())
            stage_buffers[stage_index]['active_before'].append(active_before.detach().cpu())

        sample_count += inputs.shape[0]

    if stage_buffers is None:
        raise ValueError('no samples processed')

    payload_stages = []
    summary_stages = []
    for stage in stage_buffers:
        masked_score = torch.cat(stage['masked_score'], dim=0)
        active_before = torch.cat(stage['active_before'], dim=0)
        payload_stages.append(
            {
                'stage_index': int(stage['stage_index']),
                'pruning_layer': int(stage['pruning_layer']),
                'keep_count': int(stage['keep_count']),
                'masked_score': masked_score,
                'active_before': active_before,
            }
        )
        summary_stages.append(
            {
                'stage_index': int(stage['stage_index']),
                'pruning_layer': int(stage['pruning_layer']),
                'keep_count': int(stage['keep_count']),
                'masked_score': summarize_masked_score(masked_score),
                'active_before': summarize_active_before(active_before),
            }
        )

    pt_payload = {
        'bundle_dir': str(bundle_dir),
        'data_path': selection['data_path'],
        'selection_mode': selection['selection_mode'],
        'sample_paths': [str(path) for path in dataset.sample_paths],
        'stages': payload_stages,
        'candidate_metadata': {
            'format_purpose': 'secure_network_kth_external_input_sidecar',
            'generated_from_frozen_model_forward': True,
            'masked_score_required_by_manifest': True,
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
            'input_sidecar_is_read_only': True,
            'exported_from_frozen_model_forward': True,
            'contains_masked_score_not_kth_threshold': True,
            'model_semantics_changed': False,
        },
    }
    print_and_optionally_write(summary, args.output_json)


def odd_even_compare_schedule(token_count):
    passes = []
    total_comparators = 0
    for pass_index in range(token_count):
        start_index = pass_index % 2
        pair_count = len(range(start_index, token_count - 1, 2))
        total_comparators += pair_count
        passes.append(
            {
                'pass_index': pass_index,
                'start_index': start_index,
                'pair_count': pair_count,
            }
        )
    return {
        'pass_count': token_count,
        'total_comparators': total_comparators,
        'passes': passes,
    }


def build_manifest(bundle_dir: Path):
    bundle = load_frozen_bundle(bundle_dir, device='cpu')
    model = bundle['model']

    input_token_count = int(model.pos_embed.shape[1] - 1)
    padded_token_count = input_token_count
    keep_counts = [int(input_token_count * ratio) for ratio in model.token_ratio]
    pruning_layers = [int(value) for value in model.pruning_loc]

    network_schedule = odd_even_compare_schedule(padded_token_count)
    stages = []
    previous_active_count = input_token_count
    for stage_index, (pruning_layer, keep_count) in enumerate(zip(pruning_layers, keep_counts)):
        effective_input_token_count = previous_active_count
        effective_padded_token_count = effective_input_token_count
        effective_schedule = odd_even_compare_schedule(effective_padded_token_count)
        stages.append(
            {
                'stage_index': stage_index,
                'pruning_layer': pruning_layer,
                'input_token_count': input_token_count,
                'padded_token_count': padded_token_count,
                'effective_input_token_count': effective_input_token_count,
                'effective_padded_token_count': effective_padded_token_count,
                'keep_count': keep_count,
                'kth_lane_index_in_desc_sorted_output': keep_count - 1,
                'inactive_lane_padding_value': '-inf',
                'tie_policy': 'lowest_index',
                'effective_schedule_summary': {
                    'pass_count': effective_schedule['pass_count'],
                    'total_comparators': effective_schedule['total_comparators'],
                    'comparators_per_pass_mean': float(
                        effective_schedule['total_comparators'] / effective_schedule['pass_count']
                    ),
                    'depth': effective_schedule['pass_count'],
                },
                'active_token_compaction_before_compare': stage_index > 0,
            }
        )
        previous_active_count = keep_count

    return {
        'experiment_family': 'secure_network_kth_threshold',
        'single_source_of_truth': {
            'bundle_dir': str(bundle_dir),
            'frozen_candidate_protected': True,
        },
        'network_definition': {
            'network_type': 'fixed_odd_even_compare_swap_desc',
            'compare_operator': 'max_to_left_min_to_right',
            'input_token_count': input_token_count,
            'padded_token_count': padded_token_count,
            'padding_rule': {
                'required': False,
                'pad_value': '-inf',
                'pad_count': 0,
            },
            'schedule_rule': {
                'pass_count_equals_padded_token_count': True,
                'pass_parity_start_index': 'pass_index % 2',
                'pair_pattern': '(start,start+1), (start+2,start+3), ...',
                'compact_schedule_without_extra_padding': True,
            },
            'schedule_summary': {
                'pass_count': network_schedule['pass_count'],
                'total_comparators': network_schedule['total_comparators'],
                'comparators_per_pass_mean': float(network_schedule['total_comparators'] / network_schedule['pass_count']),
                'depth': network_schedule['pass_count'],
            },
            'passes': network_schedule['passes'],
        },
        'stage_plan': stages,
        'runtime_contract': {
            'masked_score_required': True,
            'inactive_tokens_must_be_filled_before_network': True,
            'kth_threshold_output_per_stage': True,
            'default_online_tie_policy': 'lowest_index',
            'threshold_mask_reconstruction': 'greater_mask OR selected_equal_mask',
        },
        'current_reference_artifacts': {
            'threshold_branch_eval_val': str(
                bundle_dir.parent / 'threshold_branch_secure_network_kth_lowest_v1' / 'threshold_branch_eval_val.json'
            ),
            'threshold_branch_acceptance_val': str(
                bundle_dir.parent / 'threshold_branch_secure_network_kth_lowest_v1' / 'threshold_branch_acceptance_val.json'
            ),
            'secure_kth_contract': str(bundle_dir / 'stage2_secure_kth_contract.json'),
            'secure_tie_payload_contract': str(bundle_dir / 'stage2_secure_tie_payload_contract.json'),
        },
        'recommended_next_step': {
            'summary': 'Implement this exact compact fixed compare-swap network plus stage-wise active-token compaction and lowest-index tie policy in the external secure runtime, then validate with the existing isolated branch and acceptance gates.',
            'avoid': [
                'do not replace the frozen baseline',
                'do not reopen global threshold search',
                'do not change pruning semantics in core model code',
            ],
        },
    }


def run_manifest(args):
    output_json = Path(args.output_json).resolve()
    manifest = build_manifest(Path(args.bundle_dir).resolve())
    write_json(output_json, manifest)
    print(
        json.dumps(
            {
                'output_json': str(output_json),
                'network_type': manifest['network_definition']['network_type'],
                'input_token_count': manifest['network_definition']['input_token_count'],
                'padded_token_count': manifest['network_definition']['padded_token_count'],
                'pass_count': manifest['network_definition']['schedule_summary']['pass_count'],
                'total_comparators': manifest['network_definition']['schedule_summary']['total_comparators'],
                'stage_keep_counts': [stage['keep_count'] for stage in manifest['stage_plan']],
                'default_tie_policy': manifest['runtime_contract']['default_online_tie_policy'],
            },
            indent=2,
            sort_keys=True,
        )
    )


def export_reference_sidecar(args):
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
                    'kth_threshold': [],
                }
                for stage in batch_reports
            ]

        for stage_index, stage in enumerate(batch_reports):
            if args.source_mode == 'reference_topk':
                kth_threshold = stage['boundary_keep_score'].detach().cpu()
            else:
                score = stage['score'].to(args.device)
                active_before = stage['active_before'].to(args.device).bool()
                masked_score = score.masked_fill(~active_before, float('-inf'))
                kth_threshold = kth_threshold_from_compare_network(masked_score, stage['keep_count']).squeeze(1).detach().cpu()
            stage_buffers[stage_index]['kth_threshold'].append(kth_threshold)

        sample_count += inputs.shape[0]

    if stage_buffers is None:
        raise ValueError('no samples processed')

    payload_stages = []
    summary_stages = []
    for stage in stage_buffers:
        kth_threshold = torch.cat(stage['kth_threshold'], dim=0)
        payload_stages.append(
            {
                'stage_index': stage['stage_index'],
                'pruning_layer': stage['pruning_layer'],
                'keep_count': stage['keep_count'],
                'kth_threshold': kth_threshold,
            }
        )
        summary_stages.append(
            {
                'stage_index': stage['stage_index'],
                'pruning_layer': stage['pruning_layer'],
                'keep_count': stage['keep_count'],
                'kth_threshold': summarize_float_tensor(kth_threshold),
            }
        )

    pt_payload = {
        'bundle_dir': str(bundle_dir),
        'data_path': selection['data_path'],
        'selection_mode': selection['selection_mode'],
        'sample_paths': [str(path) for path in dataset.sample_paths],
        'stages': payload_stages,
        'candidate_metadata': {
            'format_purpose': f'secure_network_kth_{args.source_mode}_sidecar',
            'source_reference_topk': bool(args.source_mode == 'reference_topk'),
            'source_compare_network': bool(args.source_mode == 'compare_network'),
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
            'source_reference_topk': bool(args.source_mode == 'reference_topk'),
            'source_compare_network': bool(args.source_mode == 'compare_network'),
            'sidecar_is_read_only_reference': bool(args.source_mode == 'reference_topk'),
            'model_semantics_changed': False,
        },
    }
    print_and_optionally_write(summary, args.output_json)


def load_payload(path: Path):
    payload = torch.load(path.resolve(), map_location='cpu', weights_only=False)
    if 'stages' not in payload:
        raise ValueError(f'missing stages in kth-threshold payload: {path}')
    return payload


def tensor_compare(lhs, rhs, tolerance):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if lhs.shape != rhs.shape:
        return {
            'shape_match': False,
            'lhs_shape': list(lhs.shape),
            'rhs_shape': list(rhs.shape),
            'max_abs_error': None,
            'tolerance': tolerance,
            'passed': False,
        }
    error = float((lhs - rhs).abs().max().item()) if lhs.numel() > 0 else 0.0
    return {
        'shape_match': True,
        'shape': list(lhs.shape),
        'max_abs_error': error,
        'tolerance': tolerance,
        'passed': bool(error <= tolerance),
    }


def validate_candidate_stage(stage):
    kth_threshold = stage['kth_threshold'].detach().float().cpu()
    finite = torch.isfinite(kth_threshold)
    return {
        'kth_threshold_is_finite': bool(finite.all().item()),
        'non_finite_count': int((~finite).sum().item()),
        'shape': list(kth_threshold.shape),
    }


def build_report(reference_payload, candidate_payload, reference_path, candidate_path, tolerance):
    reference_stages = {int(stage['stage_index']): stage for stage in reference_payload['stages']}
    candidate_stages = {int(stage['stage_index']): stage for stage in candidate_payload['stages']}

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

        kth_compare = tensor_compare(reference_stage['kth_threshold'], candidate_stage['kth_threshold'], tolerance)
        metadata_match = {
            'pruning_layer_match': bool(int(reference_stage['pruning_layer']) == int(candidate_stage['pruning_layer'])),
            'keep_count_match': bool(int(reference_stage['keep_count']) == int(candidate_stage['keep_count'])),
        }
        candidate_self_check = validate_candidate_stage(candidate_stage)
        stage_passed = kth_compare['passed'] and all(metadata_match.values()) and candidate_self_check['kth_threshold_is_finite']
        all_passed = all_passed and stage_passed
        stage_reports.append(
            {
                'stage_index': stage_index,
                'available': True,
                'reference_pruning_layer': int(reference_stage['pruning_layer']),
                'candidate_pruning_layer': int(candidate_stage['pruning_layer']),
                'reference_keep_count': int(reference_stage['keep_count']),
                'candidate_keep_count': int(candidate_stage['keep_count']),
                'kth_threshold_compare': kth_compare,
                'metadata_match': metadata_match,
                'candidate_self_check': candidate_self_check,
                'passed': stage_passed,
            }
        )

    return {
        'reference_pt': str(reference_path),
        'candidate_pt': str(candidate_path),
        'bundle_dir': reference_payload.get('bundle_dir'),
        'data_path': reference_payload.get('data_path'),
        'overall_passed': bool(all_passed),
        'tolerance': tolerance,
        'stage_reports': stage_reports,
        'constraints': {
            'read_only_checker': True,
            'checks_kth_threshold_sidecar_only': True,
            'does_not_run_model_forward': True,
        },
    }


def run_check(args):
    reference_path = Path(args.reference_pt).resolve()
    candidate_path = Path(args.candidate_pt).resolve()
    report = build_report(
        load_payload(reference_path),
        load_payload(candidate_path),
        reference_path,
        candidate_path,
        args.tolerance,
    )
    print_and_optionally_write(report, args.output_json)


def infer_payload_sample_count(kth_payload):
    stages = kth_payload.get('stages') or []
    if not stages:
        return 0
    first_stage = stages[0]
    kth_threshold = first_stage.get('kth_threshold')
    if kth_threshold is None:
        return 0
    return int(kth_threshold.shape[0])


@torch.no_grad()
def forward_secure_network_kth_branch(model, inputs, kth_payload_stages, sample_offset):
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
            topk_mask = topk_reference['topk_mask']

            stage_payload = kth_payload_stages[stage_index]
            kth_threshold = stage_payload['kth_threshold'][sample_offset : sample_offset + batch_size].to(inputs.device).view(-1, 1)
            gt_mask = (score > kth_threshold) & active_before
            eq_mask = (score == kth_threshold) & active_before
            tie_keep_quota = keep_count - gt_mask.sum(dim=1)
            selected_equal_mask = select_equal_by_index(eq_mask, tie_keep_quota, 'secure_tie_lowest_index')
            branch_mask = gt_mask | selected_equal_mask

            reference_kth_threshold = topk_reference['kth_threshold']
            prev_decision = branch_mask.unsqueeze(-1).to(dtype=x.dtype)
            cls_policy = torch.ones(batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
            x = blk(x, policy=policy)
            x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

            stage_reports.append(
                {
                    'stage_index': stage_index,
                    'threshold_mode': 'secure_network_kth_candidate',
                    'threshold': None,
                    'threshold_summary': {
                        'min': float(kth_threshold.min().item()),
                        'max': float(kth_threshold.max().item()),
                        'mean': float(kth_threshold.mean().item()),
                        'std': float(kth_threshold.std(unbiased=False).item()),
                    },
                    'metrics': mask_metrics(branch_mask.detach().cpu(), topk_mask.detach().cpu()),
                    'kth_payload_checks': {
                        'kth_threshold_is_finite': bool(torch.isfinite(kth_threshold).all().item()),
                        'max_abs_error_vs_argsort_reference': float((kth_threshold - reference_kth_threshold).abs().max().item()),
                        'uses_lowest_index_tie_policy': True,
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
        kth_payload_checks = {
            'kth_threshold_is_finite': all(item['kth_payload_checks']['kth_threshold_is_finite'] for item in stage_items),
            'max_abs_error_vs_argsort_reference': float(
                max(item['kth_payload_checks']['max_abs_error_vs_argsort_reference'] for item in stage_items)
            ),
            'uses_lowest_index_tie_policy': True,
        }
        aggregated.append(
            {
                'stage_index': stage_index,
                'threshold_mode': 'secure_network_kth_candidate',
                'threshold': None,
                'threshold_summary': threshold_summary,
                'metrics': metrics,
                'kth_payload_checks': kth_payload_checks,
            }
        )
    return aggregated


def run_branch_eval(args):
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    class_threshold = resolve_threshold(bundle_dir, None)
    kth_payload = load_payload(Path(args.kth_payload_pt))
    payload_sample_count = infer_payload_sample_count(kth_payload)
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
        logits, stage_reports = forward_secure_network_kth_branch(bundle['model'], inputs, kth_payload['stages'], sample_offset)
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
        'kth_payload_pt': str(Path(args.kth_payload_pt).resolve()),
        'data_path': str(data_path),
        'device': args.device,
        'sample_count': int(targets.numel()),
        'payload_sample_count': payload_sample_count,
        'branch_type': 'isolated_eval_only_secure_network_kth_branch',
        'pruning_threshold_mode': 'secure_network_kth_candidate',
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
            'dynamic_threshold_source_requires_secure_selection_design': False,
            'uses_reference_tie_payload': False,
            'uses_reference_kth_payload': bool(kth_payload.get('candidate_metadata', {}).get('source_reference_topk', False)),
        },
    }
    print_and_optionally_write(report, args.output_json)


def add_export_common_args(parser):
    parser.add_argument('--bundle-dir', required=True)
    add_input_selection_args(parser)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output-pt', required=True)
    parser.add_argument('--output-json', default='')


def main():
    parser = argparse.ArgumentParser(description='Unified secure-network-kth sidecar utility.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    input_export = subparsers.add_parser('input-export', help='export masked-score input sidecar')
    add_export_common_args(input_export)

    manifest = subparsers.add_parser('manifest', help='emit compare-network kth manifest')
    manifest.add_argument('--bundle-dir', required=True)
    manifest.add_argument('--output-json', required=True)

    reference_export = subparsers.add_parser('export', help='export kth-threshold reference sidecar')
    reference_export.add_argument(
        '--source-mode',
        default='reference_topk',
        choices=['reference_topk', 'compare_network'],
        help='Use top-k boundary score as the reference export, or export kth_threshold from the fixed compare-network candidate path.',
    )
    add_export_common_args(reference_export)

    checker = subparsers.add_parser('check', help='check kth-threshold candidate against reference')
    checker.add_argument('--reference-pt', required=True)
    checker.add_argument('--candidate-pt', required=True)
    checker.add_argument('--tolerance', type=float, default=5e-5)
    checker.add_argument('--output-json', default='')

    branch_eval = subparsers.add_parser('branch-eval', help='run isolated eval-only kth sidecar branch')
    branch_eval.add_argument('--bundle-dir', required=True)
    branch_eval.add_argument('--kth-payload-pt', required=True)
    branch_eval.add_argument('--data-path', default='')
    branch_eval.add_argument('--device', default='cpu')
    branch_eval.add_argument('--batch-size', type=int, default=16)
    branch_eval.add_argument('--num-workers', type=int, default=0)
    branch_eval.add_argument('--max-samples', type=int, default=0)
    branch_eval.add_argument('--output-json', required=True)

    args = parser.parse_args()
    dispatch = {
        'input-export': export_input_sidecar,
        'manifest': run_manifest,
        'export': export_reference_sidecar,
        'check': run_check,
        'branch-eval': run_branch_eval,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
