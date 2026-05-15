#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


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


def round_sig(value: float, sig: int = 2) -> float:
    if value == 0:
        return 0.0
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)


def derive_stage_weight(stage_payload: dict, dominant_cost_stage: Optional[int], dominant_risk_stage: Optional[int]) -> float:
    stage_index = int(stage_payload['stage_index'])
    heuristic_tier = ((stage_payload.get('risk_model') or {}).get('heuristic_tier')) or 'low'
    tie_ratio = ((stage_payload.get('risk_signals') or {}).get('boundary_tie_sample_ratio')) or 0.0

    weight = 0.75
    if stage_index == dominant_cost_stage:
        weight += 0.50
    if stage_index == dominant_risk_stage:
        weight += 0.50
    if heuristic_tier == 'high':
        weight += 0.25
    elif heuristic_tier == 'medium':
        weight += 0.10
    if tie_ratio >= 0.85:
        weight += 0.15
    return min(2.0, round(weight, 2))


def derive_margin_target(stages: List[dict]) -> float:
    kth_errors = []
    for stage in stages:
        error = ((stage.get('numeric') or {}).get('kth_threshold_max_abs_error'))
        if error is not None and math.isfinite(float(error)):
            kth_errors.append(float(error))
    if not kth_errors:
        return 1e-5
    target = max(1e-5, max(kth_errors) * 2.0)
    target = min(1e-4, target)
    return round_sig(target, sig=2)


def format_stage_weights_csv(stage_weights: List[float]) -> str:
    return ','.join(f'{weight:.2f}' for weight in stage_weights)


def build_profiles(report: dict):
    stage_cost_risk = report['stage_cost_risk']
    stages = stage_cost_risk['stages']
    dominant_cost_stage = stage_cost_risk.get('dominant_cost_stage_index')
    dominant_risk_stage = stage_cost_risk.get('dominant_risk_stage_index')
    base_stage_weights = [
        derive_stage_weight(stage, dominant_cost_stage, dominant_risk_stage)
        for stage in stages
    ]
    focused_stage_weights = []
    for stage in stages:
        stage_index = int(stage['stage_index'])
        base_weight = base_stage_weights[stage_index]
        if stage_index == dominant_risk_stage:
            focused_stage_weights.append(round(min(2.25, base_weight + 0.25), 2))
        else:
            focused_stage_weights.append(round(max(0.75, base_weight - 0.10), 2))

    base_target = derive_margin_target(stages)
    focused_target = round_sig(min(1e-4, max(1e-5, base_target * 1.5)), sig=2)

    profiles = {
        'conservative': {
            'pruning_margin_weight': 1.0,
            'pruning_margin_target': base_target,
            'pruning_margin_mode': 'softplus',
            'pruning_margin_stage_weights': base_stage_weights,
            'pruning_margin_stage_weights_csv': format_stage_weights_csv(base_stage_weights),
            'pruning_margin_start_epoch': 0,
            'intended_use': 'first protocol-aware wiring check with immediate activation',
            'notes': [
                'uses a smooth penalty to avoid turning the tie-dominated boundary into an abrupt optimization target on the first short run',
                'keeps all pruning stages active, but emphasizes stage 1 first and stage 0 second',
            ],
        },
        'focused': {
            'pruning_margin_weight': 3.0,
            'pruning_margin_target': focused_target,
            'pruning_margin_mode': 'hinge',
            'pruning_margin_stage_weights': focused_stage_weights,
            'pruning_margin_stage_weights_csv': format_stage_weights_csv(focused_stage_weights),
            'pruning_margin_start_epoch': 0,
            'intended_use': 'after conservative wiring succeeds and you want stronger explicit margin pressure',
            'notes': [
                'pushes the dominant risk stage more aggressively',
                'switches to hinge to optimize a clearer minimum-margin target once short-run stability is confirmed',
            ],
        },
    }
    return profiles


def build_report(stage_cost_risk_path: Path):
    report = load_json(stage_cost_risk_path)
    stage_cost_risk = report['stage_cost_risk']
    stages = stage_cost_risk['stages']
    dominant_cost_stage = stage_cost_risk.get('dominant_cost_stage_index')
    dominant_risk_stage = stage_cost_risk.get('dominant_risk_stage_index')
    profiles = build_profiles(report)

    basis = []
    for stage in stages:
        risk_model = stage.get('risk_model') or {}
        numeric = stage.get('numeric') or {}
        basis.append(
            {
                'stage_index': stage['stage_index'],
                'pruning_layer': stage['pruning_layer'],
                'keep_count': stage['keep_count'],
                'estimated_total_sidecar_share': (stage.get('cost') or {}).get('estimated_total_sidecar_share'),
                'boundary_tie_sample_ratio': (stage.get('risk_signals') or {}).get('boundary_tie_sample_ratio'),
                'all_active_equal_ratio': (stage.get('risk_signals') or {}).get('all_active_equal_ratio'),
                'semantic_zero_margin_ratio': (stage.get('margin') or {}).get('semantic_zero_margin_ratio'),
                'kth_threshold_max_abs_error': numeric.get('kth_threshold_max_abs_error'),
                'risk_score': risk_model.get('heuristic_score'),
                'risk_tier': risk_model.get('heuristic_tier'),
                'primary_driver': risk_model.get('primary_driver'),
            }
        )

    recommended_profile = 'conservative'
    output = {
        'manifest_type': 'transshield_protocol_aware_pruning_recipe_v0',
        'inputs': {
            'stage_cost_risk_json': str(stage_cost_risk_path.resolve()),
        },
        'official_line': report.get('official_line'),
        'acceptance_readiness': report.get('acceptance_readiness'),
        'basis': {
            'dominant_cost_stage_index': dominant_cost_stage,
            'dominant_risk_stage_index': dominant_risk_stage,
            'consistency': report.get('consistency'),
            'stage_basis': basis,
            'key_findings': report.get('key_findings') or [],
        },
        'profiles': profiles,
        'recommended_profile': recommended_profile,
        'judgement': {
            'scope': (
                'this recipe only formalizes a protocol-aware pruning objective entry point for the current official line; '
                'it does not claim a new accuracy gain before fresh training runs are executed and compared.'
            ),
            'why_now': [
                'stage 1 is the dominant boundary-risk stage and should receive the strongest margin pressure',
                'stage 0 still carries the largest estimated sidecar cost share and should not be ignored',
                'stages 0 and 2 show strict margins far below the current secure kth numerical error scale, so immediate protocol-aware regularization should stay conservative first',
            ],
            'debug_cadence_note': (
                'the current loss print cadence is 100 training steps, so debug80 is for launch/stability only; '
                'epoch1 is the first short mode that should emit pruning margin stats on the current dataset.'
            ),
        },
    }
    return output


def render_markdown(report: dict):
    basis = report['basis']
    conservative = report['profiles']['conservative']
    focused = report['profiles']['focused']

    lines = [
        '# Protocol-Aware Pruning Recipe',
        '',
        '## 1. 结论',
        '',
        '- 当前 `P1-3` 的收口不是“已经证明更高精度”，而是把现有 margin-aware 接口变成正式可运行、可报告、可复验的 protocol-aware 训练入口。',
        f"- 推荐起点 profile：`{report['recommended_profile']}`",
        '',
        '## 2. 当前 clean 证据为什么指向 protocol-aware pruning objective',
        '',
        f"- dominant cost stage: `stage {basis['dominant_cost_stage_index']}`",
        f"- dominant risk stage: `stage {basis['dominant_risk_stage_index']}`",
        '',
        '| Stage | Layer | Cost Share | Tie Ratio | All-Equal Ratio | Risk Tier | Driver |',
        '|---|---:|---:|---:|---:|---|---|',
    ]

    for stage in basis['stage_basis']:
        lines.append(
            f"| {stage['stage_index']} | {stage['pruning_layer']} | "
            f"{fmt_percent(stage['estimated_total_sidecar_share'])} | "
            f"{fmt_percent(stage['boundary_tie_sample_ratio'])} | "
            f"{fmt_percent(stage['all_active_equal_ratio'])} | "
            f"{stage['risk_tier']} | {stage['primary_driver']} |"
        )

    lines.extend(
        [
            '',
            '## 3. 推荐 profile',
            '',
            '### conservative',
            '',
            f"- pruning_margin_weight: `{fmt_number(conservative['pruning_margin_weight'], 2)}`",
            f"- pruning_margin_target: `{fmt_number(conservative['pruning_margin_target'], 8)}`",
            f"- pruning_margin_mode: `{conservative['pruning_margin_mode']}`",
            f"- pruning_margin_stage_weights: `{conservative['pruning_margin_stage_weights_csv']}`",
            f"- pruning_margin_start_epoch: `{conservative['pruning_margin_start_epoch']}`",
            '',
            '### focused',
            '',
            f"- pruning_margin_weight: `{fmt_number(focused['pruning_margin_weight'], 2)}`",
            f"- pruning_margin_target: `{fmt_number(focused['pruning_margin_target'], 8)}`",
            f"- pruning_margin_mode: `{focused['pruning_margin_mode']}`",
            f"- pruning_margin_stage_weights: `{focused['pruning_margin_stage_weights_csv']}`",
            f"- pruning_margin_start_epoch: `{focused['pruning_margin_start_epoch']}`",
            '',
            '## 4. 运行建议',
            '',
            '- `debug80` 只检查命令接线、非有限值和早期稳定性；默认 80 step 不会触发 100-step 的 `loss info` 打印。',
            '- `epoch1` 是当前数据规模下第一条应当产出 `pruning_margin` / `margin_stats` 日志的短跑模式。',
            '- 如果 conservative `epoch1` 能稳定出日志，再考虑切换 `focused` 做更强约束。',
            '',
            '## 5. 推荐命令',
            '',
            '```bash',
            'bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh recipe',
            'export PROTOCOL_AWARE_PROFILE=conservative',
            'bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh debug80',
            'bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh epoch1',
            'bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_report.sh',
            '```',
        ]
    )
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a protocol-aware pruning recipe from the current stage cost/risk report.')
    parser.add_argument('--stage-cost-risk-json', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-md', required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args.stage_cost_risk_json)
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))


if __name__ == '__main__':
    main()
