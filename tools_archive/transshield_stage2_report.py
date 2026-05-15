import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_json, resolve_model_state_dict_path


def write_json_report(payload, output_json: str):
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_json:
        Path(output_json).resolve().write_text(text + '\n', encoding='utf-8')


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'This subcommand requires `torch`; please run it in the Transshield runtime environment.'
        ) from exc
    return torch


def build_pruning_semantics_report(bundle_dir: Path):
    args_snapshot = load_json(bundle_dir / 'args_snapshot.json')
    base_rate = float(args_snapshot['base_rate'])
    init_tokens = 14 * 14
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    keep_counts = [int(init_tokens * ratio) for ratio in token_ratio]
    return {
        'bundle_dir': str(bundle_dir),
        'model_state_dict': str(resolve_model_state_dict_path(bundle_dir)),
        'threshold_sidecar': str(bundle_dir / 'threshold_best.json'),
        'pruning_loc': [3, 6, 9],
        'init_spatial_tokens': init_tokens,
        'token_ratio': token_ratio,
        'keep_counts': keep_counts,
        'current_eval_keep_mask_generation': [
            {
                'step': 'score_select',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'score = pred_score[:, :, 0]',
            },
            {
                'step': 'drop_previous_tokens',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': "score.masked_fill(prev_decision.squeeze(-1) == 0, float('-inf'))",
            },
            {
                'step': 'topk_selection',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]',
            },
            {
                'step': 'scatter_mask',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'new_mask.scatter_(1, keep_policy.unsqueeze(-1), 1.0)',
            },
            {
                'step': 'history_mask_intersection',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'prev_decision = new_mask * prev_decision',
            },
        ],
        'f_mux_ready_sites': [
            {
                'location': 'models/dyvit.py::_apply_spatial_mask',
                'operation': 'x[:, 1:] * decision',
                'status': 'direct_mask_multiply',
            },
            {
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'x[:, 1:] * prev_decision',
                'status': 'direct_mask_multiply',
            },
        ],
        'f_less_ready_sites': [
            {
                'location': 'engine.py::_binary_threshold_accuracy',
                'operation': 'class1_prob >= threshold',
                'status': 'explicit_binary_compare',
            },
            {
                'location': 'tools/transshield_stage2_bundle.py::postprocess_binary_output',
                'operation': 'probs_cpu[1].item() >= threshold',
                'status': 'explicit_binary_compare',
            },
        ],
        'gap': {
            'pruning_keep_generation_is_explicit_f_less': False,
            'reason': 'eval pruning currently uses top-k argsort/scatter, not score >= threshold_stage',
            'safe_default_next_step': 'keep PyTorch top-k reference semantics and map only mask application to F_mux until threshold calibration is explicitly requested',
        },
        'candidate_future_paths': [
            'keep top-k semantics and implement secure top-k/selection later',
            'calibrate per-stage thresholds offline and validate score >= threshold_stage against frozen baseline',
            'defer pruning keep generation conversion and continue pure-forward packaging first',
        ],
    }


def build_f_mux_spec_report(bundle_dir: Path):
    trace = load_json(bundle_dir / 'stage2_pruning_trace_00003.json')
    semantics = load_json(bundle_dir / 'stage2_pruning_semantics_report.json')

    stages = []
    for stage in trace['pruning_trace']['stages']:
        stages.append({
            'stage_index': stage['stage_index'],
            'pruning_layer': stage['pruning_layer'],
            'configured_keep_count': stage['configured_keep_count'],
            'active_after_density': stage['active_after_density_per_sample'][0],
            'mask_shape': stage['mask_stats']['shape'],
            'mask_value_range': [stage['mask_stats']['min'], stage['mask_stats']['max']],
            'mask_mean': stage['mask_stats']['mean'],
            'mask_std': stage['mask_stats']['std'],
            'f_mux_equivalence_max_abs_error': stage['f_mux_equivalence_max_abs_error'],
        })

    return {
        'bundle_dir': str(bundle_dir),
        'single_source_of_truth': {
            'model_state_dict': str(resolve_model_state_dict_path(bundle_dir)),
            'threshold_sidecar': str(bundle_dir / 'threshold_best.json'),
        },
        'token_sequence_spec': {
            'full_sequence_shape': [1, 197, 384],
            'cls_token_shape': [1, 1, 384],
            'spatial_tokens_shape': [1, 196, 384],
            'cls_token_masked': False,
        },
        'mask_spec': {
            'shape': [1, 196, 1],
            'semantic_dtype': 'boolean_as_float_0_1',
            'broadcast_target_shape': [1, 196, 384],
            'apply_form': 'masked_tokens = tokens * mask',
            'safe_form': 'masked_tokens = F_mux(mask, tokens, zeros_like(tokens))',
        },
        'policy_spec': {
            'shape': [1, 197, 1],
            'cls_prefix_value': 1.0,
            'spatial_suffix_source': 'prev_decision',
        },
        'stages': stages,
        'f_mux_sites': semantics['f_mux_ready_sites'],
        'constraints': {
            'topk_keep_generation_preserved': True,
            'pruning_keep_generation_rewritten_to_f_less': False,
            'model_semantics_changed': False,
        },
    }


def build_policy_spec_report(bundle_dir: Path):
    args_snapshot = load_json(bundle_dir / 'args_snapshot.json')
    trace = load_json(bundle_dir / 'stage2_pruning_trace_00003.json')

    policy_shape = [1, 197, 1]
    attn_policy_shape = [1, 1, 197, 197]
    stage_reports = []
    for stage in trace['pruning_trace']['stages']:
        stage_reports.append({
            'stage_index': stage['stage_index'],
            'pruning_layer': stage['pruning_layer'],
            'prev_decision_shape': stage['mask_stats']['shape'],
            'policy_shape': policy_shape,
            'attn_policy_shape': attn_policy_shape,
            'mask_value_range': [stage['mask_stats']['min'], stage['mask_stats']['max']],
            'cls_policy_prefix': 1.0,
            'shared_across_heads': True,
            'inactive_token_kept_self_loop': True,
        })

    return {
        'bundle_dir': str(bundle_dir),
        'single_source_of_truth': {
            'model_state_dict': str(resolve_model_state_dict_path(bundle_dir)),
            'threshold_sidecar': str(bundle_dir / 'threshold_best.json'),
        },
        'frozen_attention_config': {
            'use_approx_attn': bool(args_snapshot['use_approx_attn']),
            'approx_attn_mode': args_snapshot['approx_attn_mode'],
            'fp32_attention': True,
            'path': 'softmax_with_policy' if not args_snapshot['use_approx_attn'] else '_relu_attention',
        },
        'policy_spec': {
            'source': 'concat(cls_policy, prev_decision)',
            'shape': policy_shape,
            'cls_prefix_value': 1.0,
            'spatial_suffix_shape': [1, 196, 1],
            'semantic_dtype': 'boolean_as_float_0_1',
        },
        'attn_policy_spec': {
            'builder': 'Attention._build_attn_policy',
            'shape': attn_policy_shape,
            'construction': 'policy.reshape(B,1,1,N) + (1-policy.reshape(B,1,1,N)) * eye(N)',
            'broadcast_over_heads': True,
            'inactive_token_non_diagonal_zero': True,
            'inactive_token_diagonal_one': True,
        },
        'stages': stage_reports,
        'constraints': {
            'token_f_mux_handled_separately': True,
            'attention_policy_not_reduced_to_plain_f_mux': True,
            'model_semantics_changed': False,
        },
    }


def build_forward_dataflow_report(bundle_dir: Path):
    args_snapshot = load_json(bundle_dir / 'args_snapshot.json')
    threshold_json = load_json(bundle_dir / 'threshold_best.json')
    pruning_semantics = load_json(bundle_dir / 'stage2_pruning_semantics_report.json')
    f_mux_spec = load_json(bundle_dir / 'stage2_f_mux_spec.json')

    keep_counts = [stage['configured_keep_count'] for stage in f_mux_spec['stages']]
    pruning_layers = [stage['pruning_layer'] for stage in f_mux_spec['stages']]
    input_size = int(args_snapshot['input_size'])
    nb_classes = int(args_snapshot['nb_classes'])

    return {
        'bundle_dir': str(bundle_dir),
        'single_source_of_truth': {
            'model_state_dict': str(resolve_model_state_dict_path(bundle_dir)),
            'threshold_sidecar': str(bundle_dir / 'threshold_best.json'),
        },
        'frozen_config_summary': {
            'input_size': input_size,
            'nb_classes': nb_classes,
            'base_rate': float(args_snapshot['base_rate']),
            'use_mask_pruning': bool(args_snapshot['use_mask_pruning']),
            'use_square_gelu': bool(args_snapshot['use_square_gelu']),
            'square_activation_mode': args_snapshot['square_activation_mode'],
            'use_approx_attn': bool(args_snapshot['use_approx_attn']),
        },
        'input_preprocess': {
            'image_mode': 'RGB',
            'transform_order': [
                'Resize/CenterCrop via build_transform(is_train=False)',
                'ToTensor',
                'Normalize',
            ],
            'tensor_shape': [1, 3, input_size, input_size],
            'tensor_value_range_before_normalize': '[0, 1]',
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
        },
        'forward_dataflow': [
            {
                'step_index': 0,
                'name': 'patch_embed',
                'location': 'models/dyvit.py::PatchEmbed.forward',
                'input_shape': [1, 3, input_size, input_size],
                'operation': 'Conv2d(kernel=16, stride=16) -> flatten -> transpose',
                'output_shape': [1, 196, 384],
            },
            {
                'step_index': 1,
                'name': 'cls_and_pos',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'concat(cls_token, patch_tokens) + pos_embed + pos_drop',
                'output_shape': [1, 197, 384],
            },
            {
                'step_index': 2,
                'name': 'init_mask_state',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'prev_decision=ones([B,196,1]); policy=ones([B,197,1])',
                'prev_decision_shape': [1, 196, 1],
                'policy_shape': [1, 197, 1],
            },
            {
                'step_index': 3,
                'name': 'non_pruning_blocks',
                'location': 'models/dyvit.py::Block.forward',
                'block_indices': [0, 1, 2, 4, 5, 7, 8, 10, 11],
                'operation': 'block(x, policy) then spatial token remask with prev_decision on eval path',
                'input_shape': [1, 197, 384],
                'output_shape': [1, 197, 384],
            },
            {
                'step_index': 4,
                'name': 'pruning_stages',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'pruning_layers': pruning_layers,
                'configured_keep_counts': keep_counts,
                'per_stage_flow': [
                    {
                        'substep': 'mask_apply_before_predictor',
                        'location': 'models/dyvit.py::_apply_spatial_mask',
                        'operation': 'concat(cls_token, spatial_tokens * prev_decision)',
                        'mapping': 'F_mux_ready',
                    },
                    {
                        'substep': 'score_predictor',
                        'location': 'models/dyvit.py::PredictorLG.forward',
                        'operation': 'pred_score -> reshape(B,196,2)',
                    },
                    {
                        'substep': 'keep_mask_generation_eval',
                        'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                        'operation': 'score -> masked_fill(-inf) -> argsort(desc) -> topk -> scatter -> prev_decision update',
                        'mapping': 'topk_reference_preserved',
                        'explicit_f_less_ready': False,
                    },
                    {
                        'substep': 'policy_build',
                        'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                        'operation': 'policy = concat(cls_policy, prev_decision)',
                        'policy_shape': [1, 197, 1],
                    },
                    {
                        'substep': 'attention_policy_gate',
                        'location': 'models/dyvit.py::Attention._build_attn_policy',
                        'operation': 'policy.reshape(B,1,1,N) + (1-policy)*eye(N)',
                        'output_shape': [1, 1, 197, 197],
                    },
                    {
                        'substep': 'mask_apply_after_block',
                        'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                        'operation': 'concat(cls_token, spatial_tokens * prev_decision)',
                        'mapping': 'F_mux_ready',
                    },
                ],
            },
            {
                'step_index': 5,
                'name': 'final_head',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'norm -> take cls token -> pre_logits -> head',
                'logits_shape': [1, nb_classes],
            },
            {
                'step_index': 6,
                'name': 'postprocess',
                'location': 'tools/transshield_stage2_bundle.py::postprocess_binary_output',
                'operation': 'softmax -> argmax -> class1_prob >= threshold',
                'mapping': 'final_F_less_ready',
                'threshold': float(threshold_json['eval_binary_threshold']),
            },
        ],
        'mapping_summary': {
            'explicit_f_mux_sites': pruning_semantics['f_mux_ready_sites'],
            'explicit_final_f_less_sites': pruning_semantics['f_less_ready_sites'],
            'pruning_keep_generation_rewritten_to_f_less': False,
            'attention_policy_requires_separate_mapping': True,
        },
        'related_bundle_sidecars': {
            'pruning_semantics': str(bundle_dir / 'stage2_pruning_semantics_report.json'),
            'f_mux_spec': str(bundle_dir / 'stage2_f_mux_spec.json'),
            'policy_spec': str(bundle_dir / 'stage2_policy_spec.json'),
        },
        'constraints': {
            'model_semantics_changed': False,
            'frozen_candidate_replaced': False,
            'topk_pruning_reference_preserved': True,
        },
    }


def build_tensor_contract_report(reference_pt: Path):
    torch = import_torch()
    payload = torch.load(reference_pt, map_location='cpu', weights_only=False)
    trace_tensors = payload['trace_tensors']

    descriptions = {
        'input_tensor': 'normalized model input tensor',
        'patch_embed': 'patch embedding output before cls concat',
        'seq_with_cls_pos': 'token sequence after cls concat, position embedding, and pos_drop',
        'norm_output': 'final normalized token sequence before cls extraction',
        'pre_logits': 'cls feature after pre_logits',
        'logits': 'final classifier output',
        'probabilities': 'softmax(logits)',
        'input_seq': 'stage input sequence before pre-stage mask apply',
        'masked_input_seq': 'stage input sequence after applying prior spatial mask',
        'pred_score': 'predictor raw output reshaped to [B, 196, 2]',
        'prev_decision': 'stage keep mask over spatial tokens',
        'policy': 'concat(cls_policy, prev_decision)',
        'attn_policy': 'attention gate built from policy with diagonal self-loop preservation',
        'block_output': 'block output before post-block spatial remask',
        'masked_output': 'block output after post-block spatial remask',
    }

    contract = {
        'reference_pt': str(reference_pt),
        'bundle_dir': payload.get('bundle_dir'),
        'image_path': payload.get('image_path'),
        'format': {
            'top_level_keys': ['bundle_dir', 'image_path', 'trace_tensors'],
            'trace_tensor_container_key': 'trace_tensors',
        },
        'groups': {
            'global': ['input_tensor', 'patch_embed', 'seq_with_cls_pos', 'norm_output', 'pre_logits', 'logits', 'probabilities'],
            'stage_required': ['prev_decision', 'policy', 'attn_policy', 'masked_output'],
            'stage_optional_debug': ['input_seq', 'masked_input_seq', 'pred_score', 'block_output'],
        },
        'tensors': {},
        'constraints': {
            'stage_indices': [0, 1, 2],
            'pruning_layers': [3, 6, 9],
            'stage_required_keys_for_checker_candidate_mode': [
                'stage_0_prev_decision',
                'stage_0_policy',
                'stage_0_attn_policy',
                'stage_0_masked_output',
                'stage_1_prev_decision',
                'stage_1_policy',
                'stage_1_attn_policy',
                'stage_1_masked_output',
                'stage_2_prev_decision',
                'stage_2_policy',
                'stage_2_attn_policy',
                'stage_2_masked_output',
            ],
            'final_optional_compare_keys': ['logits', 'probabilities'],
            'checker_default_tolerance': 1e-6,
        },
    }
    add_tensor_descriptions(contract, trace_tensors, descriptions)
    return contract


def build_secure_kth_contract_report(reference_pt: Path):
    torch = import_torch()
    payload = torch.load(reference_pt, map_location='cpu', weights_only=False)
    trace_tensors = payload['trace_tensors']

    descriptions = {
        'score': 'predictor keep score before masking inactive tokens',
        'active_before': 'active-token boolean mask before current stage selection',
        'masked_score': 'score after inactive tokens are filled with -inf',
        'kth_threshold': 'per-sample kth kept score from the current top-k reference',
        'greater_mask': 'score > kth_threshold',
        'equal_mask': 'score == kth_threshold',
        'selected_equal_mask': 'reference-selected subset of equal_mask needed to fill tie_keep_quota',
        'tie_keep_quota': 'keep_count - count(score > kth_threshold)',
        'prev_decision': 'final stage keep mask after combining greater_mask and selected_equal_mask',
        'logits': 'final classifier output',
        'probabilities': 'softmax(logits)',
    }

    contract = {
        'reference_pt': str(reference_pt),
        'bundle_dir': payload.get('bundle_dir'),
        'image_path': payload.get('image_path'),
        'format': {
            'top_level_keys': ['bundle_dir', 'image_path', 'trace_tensors'],
            'trace_tensor_container_key': 'trace_tensors',
        },
        'constraints': {
            'stage_indices': [0, 1, 2],
            'pruning_layers': [3, 6, 9],
            'stage_required_keys_for_checker_candidate_mode': [
                'stage_0_kth_threshold',
                'stage_0_greater_mask',
                'stage_0_equal_mask',
                'stage_0_selected_equal_mask',
                'stage_0_tie_keep_quota',
                'stage_0_prev_decision',
                'stage_1_kth_threshold',
                'stage_1_greater_mask',
                'stage_1_equal_mask',
                'stage_1_selected_equal_mask',
                'stage_1_tie_keep_quota',
                'stage_1_prev_decision',
                'stage_2_kth_threshold',
                'stage_2_greater_mask',
                'stage_2_equal_mask',
                'stage_2_selected_equal_mask',
                'stage_2_tie_keep_quota',
                'stage_2_prev_decision',
            ],
            'final_optional_compare_keys': ['logits', 'probabilities'],
            'checker_default_tolerance': 1e-6,
        },
        'tensors': {},
    }
    add_tensor_descriptions(contract, trace_tensors, descriptions)
    return contract


def add_tensor_descriptions(contract, trace_tensors, descriptions):
    for key, tensor in sorted(trace_tensors.items()):
        short_name = key
        stage_index = None
        if key.startswith('stage_'):
            _, stage_raw, suffix = key.split('_', 2)
            stage_index = int(stage_raw)
            short_name = suffix
        contract['tensors'][key] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'stage_index': stage_index,
            'description': descriptions.get(short_name, short_name),
        }


def add_bundle_report_parser(subparsers, name: str, description: str):
    parser = subparsers.add_parser(name, description=description)
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--output-json', default='')
    return parser


def add_reference_report_parser(subparsers, name: str, description: str):
    parser = subparsers.add_parser(name, description=description)
    parser.add_argument('--reference-pt', required=True)
    parser.add_argument('--output-json', default='')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='Unified Stage-2 specification and contract report generator for Transshield bundles.'
    )
    subparsers = parser.add_subparsers(dest='report_type', required=True)

    bundle_report_builders = {
        'pruning-semantics': build_pruning_semantics_report,
        'f-mux-spec': build_f_mux_spec_report,
        'policy-spec': build_policy_spec_report,
        'forward-dataflow': build_forward_dataflow_report,
    }
    for report_name in bundle_report_builders:
        add_bundle_report_parser(subparsers, report_name, f'Generate {report_name} report from a frozen bundle.')

    reference_report_builders = {
        'tensor-contract': build_tensor_contract_report,
        'secure-kth-contract': build_secure_kth_contract_report,
    }
    for report_name in reference_report_builders:
        add_reference_report_parser(subparsers, report_name, f'Generate {report_name} report from a reference tensor payload.')

    args = parser.parse_args()
    if args.report_type in bundle_report_builders:
        payload = bundle_report_builders[args.report_type](Path(args.bundle_dir).resolve())
    else:
        payload = reference_report_builders[args.report_type](Path(args.reference_pt).resolve())
    write_json_report(payload, args.output_json)


if __name__ == '__main__':
    main()
