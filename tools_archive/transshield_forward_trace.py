import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import (
    input_tensor_stats,
    load_frozen_bundle,
    postprocess_binary_output,
    preprocess_image,
    resolve_threshold,
)


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


def preview_tensor(tensor, limit=8):
    flat = tensor.detach().float().reshape(-1).cpu()
    return [float(value) for value in flat[:limit].tolist()]


@torch.no_grad()
def trace_eval_forward(model, input_tensor):
    model.eval()
    batch_size = input_tensor.shape[0]

    x = input_tensor
    trace_tensors = {
        'input_tensor': x.detach().cpu(),
    }
    trace_summary = {
        'input_tensor_stats': input_tensor_stats(input_tensor),
    }

    x = model.patch_embed(x)
    trace_tensors['patch_embed'] = x.detach().cpu()
    trace_summary['patch_embed_stats'] = tensor_stats(x)

    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    trace_tensors['seq_with_cls_pos'] = x.detach().cpu()
    trace_summary['seq_with_cls_pos_stats'] = tensor_stats(x)

    init_n = x.shape[1] - 1
    prev_decision = torch.ones(batch_size, init_n, 1, dtype=x.dtype, device=x.device)
    policy = torch.ones(batch_size, init_n + 1, 1, dtype=x.dtype, device=x.device)

    pruning_stage_traces = []
    pruning_loc_to_stage = {layer_index: stage_index for stage_index, layer_index in enumerate(model.pruning_loc)}

    for block_index, blk in enumerate(model.blocks):
        if block_index in pruning_loc_to_stage:
            stage_index = pruning_loc_to_stage[block_index]
            stage_trace = {
                'stage_index': stage_index,
                'pruning_layer': int(block_index),
                'configured_keep_count': int(init_n * model.token_ratio[stage_index]),
                'prev_decision_before_stats': tensor_stats(prev_decision),
            }

            stage_input = x
            stage_trace['input_seq_stats_before_mask'] = tensor_stats(stage_input)
            trace_tensors[f'stage_{stage_index}_input_seq'] = stage_input.detach().cpu()

            if model.use_mask_pruning:
                masked_input = model._apply_spatial_mask(stage_input, prev_decision)
            else:
                masked_input = stage_input
            stage_trace['masked_input_seq_stats'] = tensor_stats(masked_input)
            stage_trace['masked_spatial_tokens_stats'] = tensor_stats(masked_input[:, 1:])
            trace_tensors[f'stage_{stage_index}_masked_input_seq'] = masked_input.detach().cpu()

            spatial_x = masked_input[:, 1:]
            pred_score = model.score_predictor[stage_index](spatial_x, prev_decision).reshape(batch_size, -1, 2)
            score = pred_score[:, :, 0]
            masked_score = score.masked_fill(prev_decision.squeeze(-1) == 0, float('-inf'))

            keep_count = int(init_n * model.token_ratio[stage_index])
            keep_policy = torch.argsort(masked_score, dim=1, descending=True)[:, :keep_count]
            new_mask = torch.zeros_like(prev_decision)
            new_mask.scatter_(1, keep_policy.unsqueeze(-1), 1.0)
            prev_decision = new_mask * prev_decision

            cls_policy = torch.ones(
                batch_size, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device
            )
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            attn_policy = blk.attn._build_attn_policy(policy)

            block_output = blk(masked_input, policy=policy)
            masked_output = torch.cat([block_output[:, :1], block_output[:, 1:] * prev_decision], dim=1)
            x = masked_output

            stage_trace.update(
                {
                    'pred_score_stats': tensor_stats(pred_score),
                    'score_stats': tensor_stats(score),
                    'masked_score_finite_stats': tensor_stats(masked_score[torch.isfinite(masked_score)]),
                    'keep_indices_preview': [int(v) for v in keep_policy[0, : min(12, keep_policy.shape[1])].tolist()],
                    'prev_decision_after_stats': tensor_stats(prev_decision),
                    'policy_stats': tensor_stats(policy),
                    'attn_policy_stats': tensor_stats(attn_policy),
                    'block_output_stats': tensor_stats(block_output),
                    'masked_output_stats': tensor_stats(masked_output),
                    'policy_preview': preview_tensor(policy[0, :, 0], limit=12),
                    'prev_decision_preview': preview_tensor(prev_decision[0, :, 0], limit=12),
                }
            )

            trace_tensors[f'stage_{stage_index}_pred_score'] = pred_score.detach().cpu()
            trace_tensors[f'stage_{stage_index}_prev_decision'] = prev_decision.detach().cpu()
            trace_tensors[f'stage_{stage_index}_policy'] = policy.detach().cpu()
            trace_tensors[f'stage_{stage_index}_attn_policy'] = attn_policy.detach().cpu()
            trace_tensors[f'stage_{stage_index}_block_output'] = block_output.detach().cpu()
            trace_tensors[f'stage_{stage_index}_masked_output'] = masked_output.detach().cpu()
            pruning_stage_traces.append(stage_trace)
        else:
            block_output = blk(x, policy=policy)
            x = torch.cat([block_output[:, :1], block_output[:, 1:] * prev_decision], dim=1)

    norm_output = model.norm(x)
    cls_feature = norm_output[:, 0]
    pre_logits = model.pre_logits(cls_feature)
    logits = model.head(pre_logits)
    probs = torch.softmax(logits, dim=-1)

    direct_logits = model(input_tensor)
    max_abs_error = float((logits - direct_logits).abs().max().item())
    verification_tol = 1e-6

    trace_tensors['norm_output'] = norm_output.detach().cpu()
    trace_tensors['pre_logits'] = pre_logits.detach().cpu()
    trace_tensors['logits'] = logits.detach().cpu()
    trace_tensors['probabilities'] = probs.detach().cpu()

    trace_summary.update(
        {
            'pruning_stages': pruning_stage_traces,
            'norm_output_stats': tensor_stats(norm_output),
            'pre_logits_stats': tensor_stats(pre_logits),
            'logits_stats': tensor_stats(logits),
            'probabilities_stats': tensor_stats(probs),
            'manual_vs_model_max_abs_error': max_abs_error,
            'manual_trace_verification_tol': verification_tol,
            'manual_trace_matches_model': bool(max_abs_error <= verification_tol),
        }
    )

    return logits, probs, trace_summary, trace_tensors


def main():
    parser = argparse.ArgumentParser(description='Emit a read-only exact eval-forward trace for the frozen Transshield bundle.')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--output-json', default='')
    parser.add_argument('--output-pt', default='')
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    threshold = resolve_threshold(bundle_dir, args.threshold)
    image_path, input_tensor = preprocess_image(args.image_path, bundle['transform'], args.device)

    logits, probs, trace_summary, trace_tensors = trace_eval_forward(bundle['model'], input_tensor)
    prediction_summary = postprocess_binary_output(probs, threshold=threshold)

    report = {
        'bundle_dir': str(bundle_dir),
        'image_path': str(image_path),
        'device': args.device,
        'logits': [float(value) for value in logits.squeeze(0).detach().cpu().tolist()],
        **prediction_summary,
        'forward_trace': trace_summary,
        'tensor_artifact_path': str(Path(args.output_pt).resolve()) if args.output_pt else None,
        'trace_status': {
            'model_semantics_changed': False,
            'read_only_trace': True,
            'topk_pruning_reference_preserved': True,
            'manual_eval_trace_verification_tol': trace_summary['manual_trace_verification_tol'],
            'manual_eval_trace_verified_against_model': bool(trace_summary['manual_trace_matches_model']),
        },
    }

    report_text = json.dumps(report, indent=2, sort_keys=True)
    print(report_text)
    if args.output_json:
        Path(args.output_json).resolve().write_text(report_text + '\n', encoding='utf-8')
    if args.output_pt:
        tensor_payload = {
            'bundle_dir': str(bundle_dir),
            'image_path': str(image_path),
            'trace_tensors': trace_tensors,
        }
        torch.save(tensor_payload, Path(args.output_pt).resolve())


if __name__ == '__main__':
    main()
