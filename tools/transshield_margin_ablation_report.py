#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def fmt_float(value, digits=6):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.{digits}g}'
    except (TypeError, ValueError):
        return 'N/A'


def fmt_percent(value, digits=2):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value) * 100:.{digits}f}%'
    except (TypeError, ValueError):
        return 'N/A'


def fmt_ratio(value):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.3f}x'
    except (TypeError, ValueError):
        return 'N/A'


def bundle_metrics_from_report(report):
    bundle_dir = report.get('bundle_dir')
    if not bundle_dir:
        return {}
    manifest_path = Path(bundle_dir) / 'manifest.json'
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    threshold = (manifest.get('primary') or {}).get('threshold_metrics') or {}
    return {
        'default_argmax_acc1': threshold.get('default_argmax_acc1'),
        'threshold_acc1': threshold.get('eval_acc1'),
        'auc': threshold.get('auc'),
        'threshold': threshold.get('eval_binary_threshold'),
        'best_epoch_acc1': ((manifest.get('primary') or {}).get('last_log_entry') or {}).get('test_acc1'),
    }


def stage_by_index(report):
    return {int(stage['stage_index']): stage for stage in report.get('stage_summaries') or []}


def get_nested(payload, keys, default=None):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def extract_stage_metrics(stage):
    return {
        'stage_index': int(stage['stage_index']),
        'pruning_layer': int(stage.get('pruning_layer', -1)),
        'active_count_before': stage.get('active_count_before'),
        'keep_count': stage.get('keep_count'),
        'margin_mean': stage.get('topk_boundary_margin_mean'),
        'margin_p10': get_nested(stage, ['topk_boundary_margin_percentiles', 'p10']),
        'margin_p50': get_nested(stage, ['topk_boundary_margin_percentiles', 'p50']),
        'margin_p90': get_nested(stage, ['topk_boundary_margin_percentiles', 'p90']),
        'small_margin_lte_1e_4': get_nested(stage, ['topk_boundary_small_margin_ratio_abs', 'lte_1e-04']),
        'small_margin_lte_1e_3': get_nested(stage, ['topk_boundary_small_margin_ratio_abs', 'lte_1e-03']),
        'small_margin_rel_lte_0p05_std': get_nested(stage, ['topk_boundary_small_margin_ratio_rel', 'lte_0p05_score_std']),
        'boundary_tie_sample_ratio': get_nested(stage, ['protocol_risk_signals', 'boundary_tie_sample_ratio']),
        'mean_tie_excess_count': get_nested(stage, ['protocol_risk_signals', 'mean_tie_excess_count']),
        'payload_bytes_float32': get_nested(stage, ['payload_estimate', 'estimated_active_score_bytes_float32']),
    }


def compare_stage(baseline_stage, candidate_stage):
    baseline = extract_stage_metrics(baseline_stage)
    candidate = extract_stage_metrics(candidate_stage)

    baseline_margin = baseline['margin_mean']
    candidate_margin = candidate['margin_mean']
    margin_ratio = None
    if baseline_margin not in (None, 0) and candidate_margin is not None:
        margin_ratio = float(candidate_margin) / float(baseline_margin)

    baseline_small = baseline['small_margin_lte_1e_4']
    candidate_small = candidate['small_margin_lte_1e_4']
    small_delta = None
    if baseline_small is not None and candidate_small is not None:
        small_delta = float(candidate_small) - float(baseline_small)

    tie_delta = None
    if baseline['boundary_tie_sample_ratio'] is not None and candidate['boundary_tie_sample_ratio'] is not None:
        tie_delta = float(candidate['boundary_tie_sample_ratio']) - float(baseline['boundary_tie_sample_ratio'])

    return {
        'stage_index': baseline['stage_index'],
        'pruning_layer': baseline['pruning_layer'],
        'active_count_before': baseline['active_count_before'],
        'keep_count': baseline['keep_count'],
        'baseline': baseline,
        'candidate': candidate,
        'delta': {
            'margin_mean_ratio_candidate_over_baseline': margin_ratio,
            'small_margin_lte_1e_4_delta': small_delta,
            'boundary_tie_sample_ratio_delta': tie_delta,
        },
        'improved': {
            'margin_mean_increased': candidate_margin is not None and baseline_margin is not None and candidate_margin > baseline_margin,
            'small_margin_lte_1e_4_decreased': small_delta is not None and small_delta < 0,
            'tie_ratio_not_worse': tie_delta is None or tie_delta <= 0,
        },
    }


def candidate_label(path: Path, report):
    bundle_dir = report.get('bundle_dir') or ''
    if bundle_dir:
        return Path(bundle_dir).name
    return path.parent.name or path.stem


def build_candidate_summary(baseline_report, candidate_path: Path):
    candidate_report = load_json(candidate_path)
    baseline_stages = stage_by_index(baseline_report)
    candidate_stages = stage_by_index(candidate_report)
    stage_comparisons = []
    for stage_index in sorted(baseline_stages):
        if stage_index not in candidate_stages:
            continue
        stage_comparisons.append(compare_stage(baseline_stages[stage_index], candidate_stages[stage_index]))

    stage2 = next((item for item in stage_comparisons if item['stage_index'] == 1), None)
    stage2_improved = False
    if stage2:
        stage2_improved = (
            stage2['improved']['margin_mean_increased']
            and stage2['improved']['small_margin_lte_1e_4_decreased']
            and stage2['improved']['tie_ratio_not_worse']
        )

    all_margins_not_worse = all(
        item['delta']['margin_mean_ratio_candidate_over_baseline'] is None
        or item['delta']['margin_mean_ratio_candidate_over_baseline'] >= 1.0
        for item in stage_comparisons
    )
    return {
        'label': candidate_label(candidate_path, candidate_report),
        'candidate_json': str(candidate_path),
        'bundle_dir': candidate_report.get('bundle_dir'),
        'sample_count': candidate_report.get('sample_count'),
        'bundle_metrics': bundle_metrics_from_report(candidate_report),
        'stage_comparisons': stage_comparisons,
        'recommendation': {
            'stage2_boundary_improved': stage2_improved,
            'all_stage_margin_not_worse': all_margins_not_worse,
            'next_step': (
                '可以进入 secure replay / SPU 一致性检查'
                if stage2_improved and all_margins_not_worse
                else '先不要替换默认 bundle；继续调小/调大 margin 权重或只保留为实验记录'
            ),
        },
    }


def render_markdown(report):
    lines = [
        '# Margin-aware pruning ablation 对比',
        '',
        '## 口径',
        '',
        f"- Baseline risk JSON：`{report['baseline_json']}`",
        f"- Candidate 数量：`{len(report['candidates'])}`",
        '- 主要观察目标：Stage 2 boundary margin 是否变大、near-boundary 比例是否下降、tie 风险是否不变坏。',
        '- 这份报告只用于算法 ablation，不会自动替换 Web demo 默认 bundle。',
        '',
    ]

    for candidate in report['candidates']:
        metrics = candidate.get('bundle_metrics') or {}
        lines += [
            f"## Candidate：`{candidate['label']}`",
            '',
            f"- Bundle：`{candidate.get('bundle_dir') or 'N/A'}`",
            f"- 样本数：`{candidate.get('sample_count')}`",
            f"- Argmax Acc：`{fmt_float(metrics.get('default_argmax_acc1'))}`",
            f"- Threshold Acc：`{fmt_float(metrics.get('threshold_acc1'))}`",
            f"- AUC：`{fmt_float(metrics.get('auc'))}`",
            f"- 建议：{candidate['recommendation']['next_step']}",
            '',
            '| Stage | Layer | Baseline margin mean | Candidate margin mean | Margin ratio | Baseline <=1e-4 | Candidate <=1e-4 | Delta <=1e-4 | Tie delta |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for item in candidate['stage_comparisons']:
            base = item['baseline']
            cand = item['candidate']
            delta = item['delta']
            lines.append(
                '| {stage} | {layer} | {base_margin} | {cand_margin} | {ratio} | {base_small} | {cand_small} | {small_delta} | {tie_delta} |'.format(
                    stage=item['stage_index'] + 1,
                    layer=item['pruning_layer'],
                    base_margin=fmt_float(base['margin_mean']),
                    cand_margin=fmt_float(cand['margin_mean']),
                    ratio=fmt_ratio(delta['margin_mean_ratio_candidate_over_baseline']),
                    base_small=fmt_percent(base['small_margin_lte_1e_4']),
                    cand_small=fmt_percent(cand['small_margin_lte_1e_4']),
                    small_delta=fmt_percent(delta['small_margin_lte_1e_4_delta']),
                    tie_delta=fmt_percent(delta['boundary_tie_sample_ratio_delta']),
                )
            )
        lines.append('')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Compare margin-aware pruning ablation stage-wise protocol risk reports.')
    parser.add_argument('--baseline-json', required=True)
    parser.add_argument('--candidate-json', action='append', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json).resolve()
    baseline_report = load_json(baseline_path)
    candidates = [
        build_candidate_summary(baseline_report, Path(path).resolve())
        for path in args.candidate_json
    ]
    report = {
        'title': 'Margin-aware pruning ablation 对比',
        'baseline_json': str(baseline_path),
        'candidates': candidates,
    }
    write_json(Path(args.output_json).resolve(), report)
    Path(args.output_md).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).resolve().write_text(render_markdown(report), encoding='utf-8')
    print(json.dumps({'output_json': args.output_json, 'output_md': args.output_md}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
