#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def relpath_if_possible(path_str: str, repo_root: Path):
    if not path_str:
        return path_str
    path = Path(path_str).resolve()
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def build_external_comparison_from_fair_report(fair_report: dict):
    if not fair_report:
        return None
    fairness = fair_report.get('fairness_checks') or {}
    if not fairness.get('accuracy_comparison_is_fair'):
        return None
    transshield = fair_report.get('transshield') or {}
    trans_metrics = transshield.get('metrics') or {}
    mpcvit = ((fair_report.get('external_baselines') or {}).get('mpcvit') or {})
    mpcvit_metrics = mpcvit.get('metrics') or {}
    comparison = fair_report.get('comparison') or {}
    run_dir = transshield.get('run_dir') or ''
    source_run = Path(run_dir).name if run_dir else ''
    run_seeds = [str(item.get('seed')) for item in (mpcvit.get('runs') or []) if item.get('seed') is not None]
    seed_count = len(mpcvit.get('runs') or [])
    seed_label = f'MPCViT {seed_count} 个随机种子均值' if seed_count else 'MPCViT 多随机种子均值'
    seed_status = f'同数据集明文外部基线（seed {"/".join(run_seeds)} 均值）' if run_seeds else '同数据集明文外部基线'
    return {
        'title': '外部项目对比',
        'summary': f'主页面当前只保留与同数据集外部明文基线的公平对比；当前展示值来自 {source_run or "latest fair comparison"}。',
        'source_run': source_run,
        'fairness_passed': True,
        'fairness_note': fairness.get('accuracy_comparison_reason'),
        'transshield_verified': {
            'argmax_accuracy': trans_metrics.get('argmax_accuracy'),
            'threshold_accuracy': trans_metrics.get('threshold_accuracy'),
            'auc': trans_metrics.get('auc'),
        },
        'mpcvit_same_dataset': {
            'argmax_accuracy': mpcvit_metrics.get('argmax_accuracy'),
            'threshold_accuracy': mpcvit_metrics.get('threshold_accuracy'),
            'auc': mpcvit_metrics.get('auc'),
            'seed_count': seed_count,
            'status': seed_status,
        },
        'mpcvit_seed_mean': {
            'label': seed_label,
            'argmax_accuracy': mpcvit_metrics.get('argmax_accuracy'),
            'threshold_accuracy': mpcvit_metrics.get('threshold_accuracy'),
            'auc': mpcvit_metrics.get('auc'),
            'seed_count': seed_count,
            'status': '同路径同样本量公平对比均值',
        },
        'mpcvit_3seed_mean': {
            'label': seed_label,
            'argmax_accuracy': mpcvit_metrics.get('argmax_accuracy'),
            'threshold_accuracy': mpcvit_metrics.get('threshold_accuracy'),
            'auc': mpcvit_metrics.get('auc'),
            'seed_count': seed_count,
            'status': '兼容旧字段名；按最新公平对比结果回填',
        },
        'gap_to_mpcvit_reference': {
            'argmax_accuracy_gap': abs(comparison.get('transshield_minus_mpcvit_argmax_accuracy_pt') or 0.0),
            'threshold_accuracy_gap': abs(comparison.get('transshield_minus_mpcvit_threshold_accuracy_pt') or 0.0),
            'auc_gap': abs(comparison.get('transshield_minus_mpcvit_auc') or 0.0),
        },
        'gap_to_mpcvit_best_by_argmax': {
            'argmax_accuracy_gap': abs(comparison.get('transshield_minus_mpcvit_argmax_accuracy_pt') or 0.0),
            'threshold_accuracy_gap': abs(comparison.get('transshield_minus_mpcvit_threshold_accuracy_pt') or 0.0),
            'auc_gap': abs(comparison.get('transshield_minus_mpcvit_auc') or 0.0),
        },
        'note': '这里展示的是同数据集、同样本量、同路径下的公平效果对比，不表示外部方法已经在当前仓库完成同口径 secure 复现。',
    }


def build_standardized_benchmark_summary(benchmark_report: dict, source_path: Path, repo_root: Path):
    if not benchmark_report:
        return None
    profiles = benchmark_report.get('profiles') or []
    comparisons = benchmark_report.get('comparisons') or []
    compact_profiles = []
    for profile in profiles:
        compact_profiles.append(
            {
                'profile_id': profile.get('profile_id'),
                'display_name': profile.get('display_name'),
                'role': profile.get('role'),
                'comparison_group': profile.get('comparison_group'),
                'model_source': profile.get('model_source'),
                'scope_note': profile.get('scope_note'),
                'config': profile.get('config') or {},
                'metrics': profile.get('metrics') or {},
            }
        )
    return {
        'title': benchmark_report.get('title') or '统一 secure benchmark',
        'status': 'available' if profiles else 'empty',
        'summary': (benchmark_report.get('scope') or {}).get(
            'summary',
            '同一 secure transformer benchmark harness 下的通信 / 时间 profile。',
        ),
        'source_json': relpath_if_possible(str(source_path), repo_root),
        'scope': benchmark_report.get('scope') or {},
        'profiles': compact_profiles,
        'comparisons': comparisons,
        'run_entry': 'artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh',
    }


def main():
    parser = argparse.ArgumentParser(description='Update web demo summary JSON from the latest competition scorecard.')
    parser.add_argument('--scorecard-json', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--repo-root', default='')
    parser.add_argument('--fair-comparison-json', default='')
    parser.add_argument('--standardized-benchmark-json', default='')
    args = parser.parse_args()

    scorecard_path = Path(args.scorecard_json).resolve()
    output_path = Path(args.output_json).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else output_path.parents[2]

    scorecard = load_json(scorecard_path)
    current = load_json(output_path) if output_path.exists() else {}
    fair_path = Path(args.fair_comparison_json).resolve() if args.fair_comparison_json else None
    fair_report = load_json(fair_path) if fair_path and fair_path.exists() else None
    benchmark_path = Path(args.standardized_benchmark_json).resolve() if args.standardized_benchmark_json else None
    benchmark_report = load_json(benchmark_path) if benchmark_path and benchmark_path.exists() else None

    verified_plaintext = ((scorecard.get('verified_candidate') or {}).get('plaintext') or {})
    secure_summary = ((scorecard.get('verified_candidate') or {}).get('secure_summary') or {})
    artifact_paths = scorecard.get('artifact_paths') or {}
    external = scorecard.get('external_references') or {}
    gap_best = (external.get('gap_to_mpcvit_same_dataset') or {}).get(
        'transshield_verified_candidate_vs_mpcvit_best_by_argmax'
    ) or {}
    current['updated_at'] = dt.date.today().isoformat()
    current['default_bundle'] = {
        'title': '当前展示模型',
        'display_name': '当前主展示模型（已验证）',
        'bundle_name': Path(artifact_paths.get('verified_bundle_dir') or '').name or 'N/A',
        'bundle_dir': relpath_if_possible(artifact_paths.get('verified_bundle_dir') or '', repo_root),
        'status': '已验证候选（可展示）',
        'argmax_accuracy': verified_plaintext.get('argmax_accuracy'),
        'threshold_accuracy': verified_plaintext.get('threshold_accuracy'),
        'best_epoch': (scorecard.get('verified_candidate') or {}).get('best_epoch'),
        'argmax_match_ratio': secure_summary.get('argmax_match_ratio'),
        'threshold_match_ratio': secure_summary.get('threshold_match_ratio'),
        'spu_pipeline_overall_passed': secure_summary.get('spu_pipeline_overall_passed'),
        'spu_replay_overall_passed': secure_summary.get('spu_replay_overall_passed'),
        'communication_source': '本页面仅展示当前 SPU live run 通信量',
        'summary': '前端默认加载这份冻结展示包，里面放的是当前主展示模型的权重、阈值和运行配置。',
        'best_epoch_note': '最佳轮次表示训练过程中验证集效果最好的 epoch；当前离线成绩来自这一轮，单图在线结果会实时重新计算。',
    }
    fair_external = build_external_comparison_from_fair_report(fair_report)
    current['external_comparison'] = fair_external or {
        'title': '外部项目对比',
        'summary': '主页面当前只保留与同数据集外部明文基线的公平对比。',
        'transshield_verified': {
            'argmax_accuracy': verified_plaintext.get('argmax_accuracy'),
            'threshold_accuracy': verified_plaintext.get('threshold_accuracy'),
            'auc': verified_plaintext.get('auc'),
        },
        'mpcvit_same_dataset': external.get('mpcvit_same_dataset') or {},
        'mpcvit_seed_mean': external.get('mpcvit_seed_mean') or external.get('mpcvit_3seed_mean') or {},
        'mpcvit_3seed_mean': external.get('mpcvit_3seed_mean') or {},
        'gap_to_mpcvit_reference': gap_best,
        'gap_to_mpcvit_best_by_argmax': gap_best,
        'note': '这里展示的是同数据集模型效果对比，不表示外部方法已经在当前仓库完成同口径 secure 复现。',
    }
    current.pop('training_gain', None)

    communication_evidence = current.get('communication_evidence') or {}
    communication_evidence.pop('local_fastpath', None)
    communication_evidence.pop('archived_secure_profile', None)
    communication_evidence['title'] = communication_evidence.get('title') or '通信说明'
    communication_evidence['summary'] = '当前页面中的通信量只使用本次上传图片触发的 SPU live run；不再保留历史固定字节数。'
    external_reference = communication_evidence.get('external_reference') or {}
    external_reference['source'] = external_reference.get('source') or 'MPCFormer secure transformer benchmark'
    external_reference['note'] = (
        '该外部 secure benchmark 仅用于说明 secure Transformer 的通信热点通常集中在 '
        'attention / 非线性等环节，不用于当前页面的单图直接数字比较。'
    )
    communication_evidence['external_reference'] = external_reference
    current['communication_evidence'] = communication_evidence

    standardized_benchmark = build_standardized_benchmark_summary(benchmark_report, benchmark_path, repo_root) if benchmark_path else None
    if standardized_benchmark:
        current['standardized_secure_benchmark'] = standardized_benchmark
    else:
        current['standardized_secure_benchmark'] = {
            'title': '统一 secure benchmark',
            'status': 'not_run',
            'summary': (
                '当前前端没有写入统一 secure benchmark 数字；如果要比较外部模型的安全通信 / 时间，'
                '必须先用同一个 MPCFormer local 2PC benchmark harness 重新运行。'
            ),
            'run_entry': 'artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh',
            'scope_note': (
                '该 benchmark 只用于同一 secure transformer harness 下的开销对比，不等同于 full-val 医学图像 pipeline，'
                '也不能和网页单图 live run 混用。'
            ),
        }

    write_json(output_path, current)


if __name__ == '__main__':
    main()
