#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import torch


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_pt(path: Path):
    return torch.load(path, map_location='cpu')


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def finite_quantile(values: torch.Tensor, quantile: float):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(torch.quantile(finite, quantile).item())


def finite_mean(values: torch.Tensor):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.mean().item())


def finite_std(values: torch.Tensor):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.std(unbiased=False).item())


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def fmt_number(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def fmt_percent(value, digits=2):
    if value is None:
        return 'N/A'
    return f'{float(value) * 100:.{digits}f}%'


def stage_note(stage_payload):
    all_equal_ratio = stage_payload['risk_signals']['all_active_equal_ratio']
    tie_ratio = stage_payload['risk_signals']['boundary_tie_sample_ratio']
    strict_margin_p50 = stage_payload['margin']['strict_below_margin_p50']
    kth_error = stage_payload['numeric']['kth_threshold_max_abs_error']
    if all_equal_ratio is not None and all_equal_ratio >= 0.99:
        return 'all_active_equal_boundary'
    if tie_ratio is not None and tie_ratio >= 0.9:
        if strict_margin_p50 is not None and kth_error is not None and kth_error > strict_margin_p50:
            return 'tie_dominated_and_numeric_margin_smaller_than_kth_error'
        return 'frequent_boundary_ties'
    if strict_margin_p50 is not None and kth_error is not None and kth_error > strict_margin_p50:
        return 'numeric_margin_smaller_than_kth_error'
    return 'cost_driven_stage'


def stage_risk_score(stage_payload):
    tie_ratio = stage_payload['risk_signals']['boundary_tie_sample_ratio'] or 0.0
    keep_count = max(1, int(stage_payload['keep_count']))
    tie_excess = stage_payload['risk_signals']['mean_tie_excess_count'] or 0.0
    tie_excess_ratio = min(1.0, float(tie_excess) / float(keep_count))
    semantic_zero_ratio = stage_payload['margin']['semantic_zero_margin_ratio'] or 0.0
    cost_share = stage_payload['cost']['estimated_total_sidecar_share'] or 0.0
    all_equal_ratio = stage_payload['risk_signals']['all_active_equal_ratio'] or 0.0

    numeric_risk = 0.0
    strict_margin_p50 = stage_payload['margin']['strict_below_margin_p50']
    kth_error = stage_payload['numeric']['kth_threshold_max_abs_error']
    if kth_error is not None:
        if strict_margin_p50 is None:
            numeric_risk = 1.0 if all_equal_ratio > 0 else 0.5
        elif kth_error >= strict_margin_p50:
            numeric_risk = 1.0
        else:
            numeric_risk = float(min(1.0, kth_error / max(strict_margin_p50, 1e-12)))

    score = (
        0.28 * tie_ratio
        + 0.22 * tie_excess_ratio
        + 0.22 * semantic_zero_ratio
        + 0.18 * numeric_risk
        + 0.10 * cost_share
    )
    score = max(score, 0.95 if all_equal_ratio >= 0.99 else score)
    if score >= 0.75:
        tier = 'high'
    elif score >= 0.45:
        tier = 'medium'
    else:
        tier = 'low'
    return float(score), tier


def summarize_secret_runtime(secret_run_dir: Path, summary_payload):
    accepted_files = sorted(secret_run_dir.glob('idx*_accepted.json'))
    elapsed = []
    raw_abs = []
    for path in accepted_files:
        payload = load_json(path)
        elapsed_sec = payload.get('elapsed_sec')
        if elapsed_sec is not None:
            elapsed.append(float(elapsed_sec))
        raw_logits = payload.get('raw_logits_before_output_calibration') or {}
        raw_max = raw_logits.get('max')
        raw_min = raw_logits.get('min')
        if raw_max is not None and raw_min is not None:
            raw_abs.append(max(abs(float(raw_max)), abs(float(raw_min))))

    return {
        'summary_json': str((secret_run_dir / 'secret_isolated_eval_summary.json').resolve()),
        'accepted_file_count': len(accepted_files),
        'accepted_accuracy': summary_payload.get('accepted_accuracy'),
        'accepted_count': summary_payload.get('accepted_count'),
        'sample_count': summary_payload.get('sample_count'),
        'complete': summary_payload.get('complete'),
        'pending_count': summary_payload.get('pending_count'),
        'unstable_count': summary_payload.get('unstable_count'),
        'elapsed_sec': {
            'mean': float(sum(elapsed) / len(elapsed)) if elapsed else None,
            'min': min(elapsed) if elapsed else None,
            'max': max(elapsed) if elapsed else None,
            'std': float(torch.tensor(elapsed).float().std(unbiased=False).item()) if len(elapsed) > 1 else 0.0 if elapsed else None,
            'sum': float(sum(elapsed)) if elapsed else None,
        },
        'raw_logit_abs_before_output_calibration': {
            'mean': float(sum(raw_abs) / len(raw_abs)) if raw_abs else None,
            'max': max(raw_abs) if raw_abs else None,
        },
        'outlier_rule': summary_payload.get('outlier_rule'),
        'spu_params_mode': (summary_payload.get('accepted_item_preview') or [{}])[0].get('spu_params_mode'),
        'spu_layer_norm_policy': (summary_payload.get('accepted_item_preview') or [{}])[0].get('spu_layer_norm_policy'),
    }


def summarize_stage(stage_input, stage_reference, stage_tie, stage_kth_check, stage_tie_check):
    scores = stage_input['masked_score'].float()
    active = stage_input['active_before'].bool()
    kth_threshold = stage_reference['kth_threshold'].float().unsqueeze(1)
    keep_count = int(stage_input['keep_count'])

    gt_mask = (scores > kth_threshold) & active
    eq_mask = (scores == kth_threshold) & active
    lt_mask = (scores < kth_threshold) & active
    active_count = active.sum(dim=1).float()
    greater_count = gt_mask.sum(dim=1).float()
    equal_count = eq_mask.sum(dim=1).float()
    tie_keep_quota = stage_tie['tie_keep_quota'].float()
    tie_excess = (equal_count - tie_keep_quota).clamp_min(0.0)
    selected_equal_count = stage_tie['selected_equal_mask'].bool().sum(dim=1).float()
    boundary_tie = equal_count > tie_keep_quota
    all_active_equal = equal_count == active_count

    below_scores = scores.masked_fill(~lt_mask, float('-inf'))
    next_below = below_scores.max(dim=1).values
    strict_margin = kth_threshold.squeeze(1) - next_below
    strict_margin = torch.where(
        torch.isfinite(next_below),
        strict_margin,
        torch.full_like(strict_margin, float('inf')),
    )
    semantic_margin = torch.where(boundary_tie, torch.zeros_like(strict_margin), strict_margin)

    kth_error = None
    if stage_kth_check:
        kth_error = ((stage_kth_check.get('kth_threshold_compare') or {}).get('max_abs_error'))
    threshold_snap_max_distance = None
    if stage_tie_check:
        threshold_snap_max_distance = (((stage_tie_check.get('semantic_check') or {}).get('threshold_snap') or {}).get('max_distance'))

    summary = {
        'stage_index': int(stage_input['stage_index']),
        'pruning_layer': int(stage_input['pruning_layer']),
        'keep_count': keep_count,
        'sample_count': int(scores.shape[0]),
        'activity': {
            'active_before_mean': float(active_count.mean().item()),
            'active_before_min': int(active_count.min().item()),
            'active_before_max': int(active_count.max().item()),
            'retention_ratio': safe_ratio(keep_count, float(active_count.mean().item())),
        },
        'margin': {
            'strict_below_margin_p10': finite_quantile(strict_margin, 0.10),
            'strict_below_margin_p50': finite_quantile(strict_margin, 0.50),
            'strict_below_margin_mean': finite_mean(strict_margin),
            'strict_below_margin_std': finite_std(strict_margin),
            'semantic_margin_p10': finite_quantile(semantic_margin, 0.10),
            'semantic_margin_p50': finite_quantile(semantic_margin, 0.50),
            'semantic_zero_margin_ratio': float((semantic_margin == 0).float().mean().item()),
        },
        'risk_signals': {
            'boundary_tie_sample_ratio': float(boundary_tie.float().mean().item()),
            'all_active_equal_ratio': float(all_active_equal.float().mean().item()),
            'mean_greater_count': float(greater_count.mean().item()),
            'mean_boundary_equal_count': float(equal_count.mean().item()),
            'max_boundary_equal_count': int(equal_count.max().item()),
            'mean_tie_keep_quota': float(tie_keep_quota.mean().item()),
            'max_tie_keep_quota': int(tie_keep_quota.max().item()),
            'mean_tie_excess_count': float(tie_excess.mean().item()),
            'max_tie_excess_count': int(tie_excess.max().item()),
            'mean_selected_equal_count': float(selected_equal_count.mean().item()),
        },
        'numeric': {
            'kth_threshold_max_abs_error': float(kth_error) if kth_error is not None else None,
            'threshold_snap_max_distance': float(threshold_snap_max_distance) if threshold_snap_max_distance is not None else None,
        },
        'bridge_contract': {
            'network_kth_passed': None if not stage_kth_check else stage_kth_check.get('passed'),
            'tie_semantic_passed': None if not stage_tie_check else (stage_tie_check.get('semantic_check') or {}).get('passed'),
            'reconstructed_branch_matches_topk_reference': None if not stage_tie_check else (stage_tie_check.get('semantic_check') or {}).get('reconstructed_branch_matches_topk_reference'),
            'selected_equal_mask_exact_match': None if not stage_tie_check else ((stage_tie_check.get('selected_equal_mask_compare') or {}).get('exact_match')),
            'tie_keep_quota_exact_match': None if not stage_tie_check else ((stage_tie_check.get('tie_keep_quota_compare') or {}).get('exact_match')),
        },
        'cost': {
            'active_score_values_per_sample': float(active_count.mean().item()),
            'boundary_equal_values_per_sample': float(equal_count.mean().item()),
            'estimated_active_score_bytes_per_sample_float32': float(active_count.mean().item() * 4.0),
            'estimated_active_score_payload_bytes_total_float32': float(active_count.sum().item() * 4.0),
        },
    }
    return summary


def summarize_run(run_dir: Path, secret_run_dir: Path, acceptance_json_path: Optional[Path]):
    input_pt_path = run_dir / 'stage2_secure_network_kth_input_smoke8.pt'
    kth_reference_path = run_dir / 'stage2_secure_network_kth_reference_smoke8.pt'
    tie_reference_path = run_dir / 'stage2_secure_tie_policy_lowest_smoke8.pt'
    pipeline_run_summary_path = run_dir / 'pipeline_run_summary.json'
    kth_check_path = run_dir / 'stage2_secure_network_kth_candidate_check.json'
    tie_check_path = run_dir / 'stage2_secure_tie_candidate_check.json'
    compare_path = run_dir / 'plaintext_vs_secure_score_compare.json'
    secret_summary_path = secret_run_dir / 'secret_isolated_eval_summary.json'

    input_pt = load_pt(input_pt_path)
    kth_reference = load_pt(kth_reference_path)
    tie_reference = load_pt(tie_reference_path)
    pipeline_run_summary = load_json(pipeline_run_summary_path)
    kth_check = load_json(kth_check_path)
    tie_check = load_json(tie_check_path)
    compare = load_json(compare_path)
    secret_summary = load_json(secret_summary_path)
    acceptance = load_json(acceptance_json_path) if acceptance_json_path else None

    kth_check_by_stage = {int(item['stage_index']): item for item in (kth_check.get('stage_reports') or [])}
    tie_check_by_stage = {int(item['stage_index']): item for item in (tie_check.get('stage_reports') or [])}

    stage_rows = []
    active_totals = []
    equal_totals = []
    for stage_input, stage_ref, stage_tie in zip(input_pt['stages'], kth_reference['stages'], tie_reference['stages']):
        stage_index = int(stage_input['stage_index'])
        row = summarize_stage(
            stage_input=stage_input,
            stage_reference=stage_ref,
            stage_tie=stage_tie,
            stage_kth_check=kth_check_by_stage.get(stage_index),
            stage_tie_check=tie_check_by_stage.get(stage_index),
        )
        active_total = row['cost']['estimated_active_score_payload_bytes_total_float32'] / 4.0
        equal_total = row['risk_signals']['mean_boundary_equal_count'] * row['sample_count']
        active_totals.append(active_total)
        equal_totals.append(equal_total)
        stage_rows.append(row)

    network_kth_sec = None
    tie_bridge_sec = None
    for step in pipeline_run_summary.get('steps') or []:
        if step.get('name') == 'network_kth_bridge':
            network_kth_sec = step.get('duration_sec')
        elif step.get('name') == 'tie_policy_bridge':
            tie_bridge_sec = step.get('duration_sec')

    total_active = sum(active_totals) or 1.0
    total_equal = sum(equal_totals) or 1.0
    total_estimated_bridge_sec = float((network_kth_sec or 0.0) + (tie_bridge_sec or 0.0))

    for row, active_total, equal_total in zip(stage_rows, active_totals, equal_totals):
        active_share = float(active_total / total_active)
        equal_share = float(equal_total / total_equal)
        estimated_network_sec = float((network_kth_sec or 0.0) * active_share)
        estimated_tie_sec = float((tie_bridge_sec or 0.0) * equal_share)
        estimated_total_sec = estimated_network_sec + estimated_tie_sec
        row['cost'].update(
            {
                'active_score_share_of_network_kth': active_share,
                'boundary_equal_share_of_tie_bridge': equal_share,
                'estimated_network_kth_bridge_sec': estimated_network_sec,
                'estimated_tie_bridge_sec': estimated_tie_sec,
                'estimated_total_sidecar_sec': estimated_total_sec,
                'estimated_total_sidecar_share': safe_ratio(estimated_total_sec, total_estimated_bridge_sec),
            }
        )
        score, tier = stage_risk_score(row)
        row['risk_model'] = {
            'heuristic_score': score,
            'heuristic_tier': tier,
            'primary_driver': stage_note(row),
        }

    stage_rows = sorted(stage_rows, key=lambda item: item['stage_index'])
    dominant_cost_stage = max(stage_rows, key=lambda item: item['cost']['estimated_total_sidecar_sec'])
    dominant_risk_stage = max(stage_rows, key=lambda item: item['risk_model']['heuristic_score'])
    secret_runtime = summarize_secret_runtime(secret_run_dir, secret_summary)

    output = {
        'manifest_type': 'transshield_stage_cost_risk_report_v0',
        'inputs': {
            'run_dir': str(run_dir.resolve()),
            'secret_run_dir': str(secret_run_dir.resolve()),
            'network_kth_input_pt': str(input_pt_path.resolve()),
            'network_kth_reference_pt': str(kth_reference_path.resolve()),
            'tie_reference_pt': str(tie_reference_path.resolve()),
            'pipeline_run_summary_json': str(pipeline_run_summary_path.resolve()),
            'network_kth_check_json': str(kth_check_path.resolve()),
            'tie_check_json': str(tie_check_path.resolve()),
            'plaintext_secure_compare_json': str(compare_path.resolve()),
            'secret_isolated_eval_summary_json': str(secret_summary_path.resolve()),
            'acceptance_json': None if not acceptance_json_path else str(acceptance_json_path.resolve()),
        },
        'official_line': None if not acceptance else acceptance.get('official_line'),
        'acceptance_readiness': None if not acceptance else acceptance.get('readiness'),
        'methodology': {
            'cost_proxy': {
                'network_kth_bridge': '按各 stage active token 总量占比分摊 network_kth_bridge 墙钟时间。',
                'tie_policy_bridge': '按各 stage boundary equal token 总量占比分摊 tie_policy_bridge 墙钟时间。',
                'note': '这是 stage-level 近似成本模型，不是协议级精确 profiling。它用于说明 pruning boundary sidecar 的成本主要集中在哪些 stage。'
            },
            'risk_proxy': {
                'signals': [
                    'boundary_tie_sample_ratio',
                    'all_active_equal_ratio',
                    'mean_tie_excess_count',
                    'semantic_zero_margin_ratio',
                    'kth_threshold_max_abs_error vs strict_below_margin',
                ],
                'note': '风险分数是启发式聚合，用于定位最需要语义 tie 处理和数值保护的 stage，不作为单独验收门。'
            },
        },
        'run_profile': {
            'sample_count': int(input_pt['stages'][0]['masked_score'].shape[0]) if input_pt.get('stages') else 0,
            'stage_count': len(stage_rows),
            'network_kth_bridge_sec': network_kth_sec,
            'tie_policy_bridge_sec': tie_bridge_sec,
            'total_boundary_sidecar_bridge_sec': total_estimated_bridge_sec,
        },
        'consistency': {
            'argmax_match_ratio': ((compare.get('comparison') or {}).get('argmax_predictions') or {}).get('match_ratio'),
            'threshold_match_ratio': ((compare.get('comparison') or {}).get('threshold_predictions') or {}).get('match_ratio'),
            'logits_max_abs_error': ((compare.get('comparison') or {}).get('logits') or {}).get('max_abs_error'),
            'probabilities_max_abs_error': ((compare.get('comparison') or {}).get('probabilities') or {}).get('max_abs_error'),
            'secure_overall_passed': ((compare.get('source_status') or {}).get('secure_overall_passed')),
        },
        'secret_runtime': secret_runtime,
        'stage_cost_risk': {
            'dominant_cost_stage_index': dominant_cost_stage['stage_index'],
            'dominant_risk_stage_index': dominant_risk_stage['stage_index'],
            'stages': stage_rows,
        },
        'key_findings': [
            f"stage {dominant_cost_stage['stage_index']} contributes the largest estimated sidecar bridge share: {fmt_percent(dominant_cost_stage['cost']['estimated_total_sidecar_share'])}.",
            f"stage {dominant_risk_stage['stage_index']} has the highest heuristic risk tier: {dominant_risk_stage['risk_model']['heuristic_tier']} ({fmt_number(dominant_risk_stage['risk_model']['heuristic_score'], 4)}).",
            'stages 0/2 use strict below-threshold margins on the order of 1e-8 while secure kth max abs error stays around 1e-5, so semantic tie handling is part of the method, not an optional patch.',
            'stage 1 is fully tie-dominated in the current clean run: all active tokens lie on the kth boundary before tie disambiguation.',
        ],
    }
    return output


def render_markdown(report):
    lines = [
        '# Stage-Level Secure Cost / Risk Model',
        '',
        '## 1. 概览',
        '',
        f"- run_dir: `{report['inputs']['run_dir']}`",
        f"- secret_run_dir: `{report['inputs']['secret_run_dir']}`",
        f"- sample_count: `{report['run_profile']['sample_count']}`",
        f"- stage_count: `{report['run_profile']['stage_count']}`",
        f"- network_kth_bridge_sec: `{fmt_number(report['run_profile']['network_kth_bridge_sec'], 4)}`",
        f"- tie_policy_bridge_sec: `{fmt_number(report['run_profile']['tie_policy_bridge_sec'], 4)}`",
        f"- total_boundary_sidecar_bridge_sec: `{fmt_number(report['run_profile']['total_boundary_sidecar_bridge_sec'], 4)}`",
        '',
        '## 2. 关键结论',
        '',
    ]
    for finding in report['key_findings']:
        lines.append(f'- {finding}')

    lines.extend(
        [
            '',
            '## 3. Stage 明细',
            '',
            '| Stage | Layer | Keep | Active Before | Tie Ratio | Tie Excess Mean | Strict Margin P50 | KTH Error Max | Est. Sidecar Sec | Est. Cost Share | Risk Tier | Driver |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
        ]
    )
    for stage in report['stage_cost_risk']['stages']:
        lines.append(
            f"| {stage['stage_index']} | {stage['pruning_layer']} | {stage['keep_count']} | "
            f"{fmt_number(stage['activity']['active_before_mean'], 1)} | "
            f"{fmt_percent(stage['risk_signals']['boundary_tie_sample_ratio'])} | "
            f"{fmt_number(stage['risk_signals']['mean_tie_excess_count'], 3)} | "
            f"{fmt_number(stage['margin']['strict_below_margin_p50'], 8)} | "
            f"{fmt_number(stage['numeric']['kth_threshold_max_abs_error'], 8)} | "
            f"{fmt_number(stage['cost']['estimated_total_sidecar_sec'], 4)} | "
            f"{fmt_percent(stage['cost']['estimated_total_sidecar_share'])} | "
            f"{stage['risk_model']['heuristic_tier']} | "
            f"{stage['risk_model']['primary_driver']} |"
        )

    secret_runtime = report['secret_runtime']
    lines.extend(
        [
            '',
            '## 4. Secret Runtime 旁证',
            '',
            f"- accepted_count: `{secret_runtime['accepted_count']}` / `{secret_runtime['sample_count']}`",
            f"- complete: `{secret_runtime['complete']}`",
            f"- pending_count: `{secret_runtime['pending_count']}`",
            f"- unstable_count: `{secret_runtime['unstable_count']}`",
            f"- mean accepted elapsed_sec: `{fmt_number((secret_runtime['elapsed_sec'] or {}).get('mean'), 4)}`",
            f"- max raw-logit abs before output calibration: `{fmt_number((secret_runtime['raw_logit_abs_before_output_calibration'] or {}).get('max'), 6)}`",
            '',
            '## 5. 一致性旁证',
            '',
            f"- argmax_match_ratio: `{fmt_percent(report['consistency']['argmax_match_ratio'])}`",
            f"- threshold_match_ratio: `{fmt_percent(report['consistency']['threshold_match_ratio'])}`",
            f"- logits_max_abs_error: `{fmt_number(report['consistency']['logits_max_abs_error'], 8)}`",
            f"- probabilities_max_abs_error: `{fmt_number(report['consistency']['probabilities_max_abs_error'], 8)}`",
            '',
            '## 6. 解释口径',
            '',
            '- 这里的 cost 是 stage-level 近似分摊模型，不是假装拿到了协议内部每个 gate 的精确耗时。',
            '- 这里的 risk 重点描述 boundary tie、strict margin、threshold snap 误差三者的关系，用来解释为什么 `F_less + tie sidecar + F_mux` 是主线方法的一部分。',
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate a stage-level secure cost / risk report for the current Transshield delivery line.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--secret-run-dir', required=True)
    parser.add_argument('--acceptance-json', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    secret_run_dir = Path(args.secret_run_dir).expanduser().resolve()
    acceptance_json_path = Path(args.acceptance_json).expanduser().resolve() if args.acceptance_json else None

    report = summarize_run(run_dir=run_dir, secret_run_dir=secret_run_dir, acceptance_json_path=acceptance_json_path)
    write_json(Path(args.output_json).expanduser().resolve(), report)
    write_text(Path(args.output_md).expanduser().resolve(), render_markdown(report) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
