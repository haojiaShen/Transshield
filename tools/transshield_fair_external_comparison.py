#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def normalize_path(path):
    if not path:
        return None
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(Path(path))


def fmt(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def pct_gap(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def nested_get(payload, path, default=None):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def imagefolder_sample_paths(root):
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    allowed_suffixes = {
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.ppm',
        '.pgm',
        '.tif',
        '.tiff',
        '.webp',
    }
    sample_paths = []
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda item: item.name)
    for class_dir in class_dirs:
        for current_root, _dirs, files in os.walk(class_dir, followlinks=True):
            for filename in sorted(files):
                path = Path(current_root) / filename
                if path.suffix.lower() in allowed_suffixes:
                    sample_paths.append(str(path.resolve()))
    return sample_paths


def sample_paths_sha256(sample_paths):
    digest = hashlib.sha256()
    for path in sample_paths:
        digest.update(path.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def build_dataset_reference(train_data_path, val_data_path):
    train_path = normalize_path(train_data_path)
    val_path = normalize_path(val_data_path)
    train_samples = imagefolder_sample_paths(train_path) if train_path else []
    val_samples = imagefolder_sample_paths(val_path) if val_path else []
    return {
        'train_path': train_path,
        'val_path': val_path,
        'train_sample_count': len(train_samples),
        'val_sample_count': len(val_samples),
        'train_sample_paths_sha256': sample_paths_sha256(train_samples) if train_samples else None,
        'val_sample_paths_sha256': sample_paths_sha256(val_samples) if val_samples else None,
    }


def parse_transshield_run(run_dir):
    run_dir = Path(run_dir)
    plaintext_eval = load_json(run_dir / 'plaintext_modified_eval.json') or {}
    score_compare = load_json(run_dir / 'plaintext_vs_secure_score_compare.json') or {}
    secure_profile = load_json(run_dir / 'secure_profile_summary.json') or {}

    metrics = plaintext_eval.get('metrics') or {}
    comparison = score_compare.get('comparison') or secure_profile.get('score_compare') or {}
    communication_profile = secure_profile.get('communication_profile') or {}
    step_profile = secure_profile.get('step_profile') or {}

    aggregate_python = communication_profile.get('aggregate_python_fastpath_metrics') or {}
    python_profile = communication_profile.get('python_fastpath_profile') or {}
    rpc_total_bytes = (
        aggregate_python.get('rpc_total_bytes')
        or python_profile.get('rpc_total_bytes')
        or communication_profile.get('python_fastpath_rpc_total_bytes')
    )

    return {
        'run_dir': str(run_dir),
        'available': bool(plaintext_eval or score_compare or secure_profile),
        'plaintext_eval_json': str(run_dir / 'plaintext_modified_eval.json'),
        'score_compare_json': str(run_dir / 'plaintext_vs_secure_score_compare.json'),
        'secure_profile_json': str(run_dir / 'secure_profile_summary.json'),
        'data_path': plaintext_eval.get('data_path'),
        'data_path_resolved': normalize_path(plaintext_eval.get('data_path')),
        'sample_count': plaintext_eval.get('sample_count') or comparison.get('argmax_predictions', {}).get('shape', [None])[0],
        'sample_paths_sha256': plaintext_eval.get('sample_paths_sha256'),
        'metrics': {
            'argmax_accuracy': metrics.get('argmax_accuracy'),
            'threshold_accuracy': metrics.get('threshold_accuracy'),
            'auc': metrics.get('auc'),
            'threshold': metrics.get('threshold'),
            'eval_loss': metrics.get('eval_loss'),
        },
        'secure_consistency': {
            'argmax_match_ratio': nested_get(comparison, ['argmax_predictions', 'match_ratio']),
            'threshold_match_ratio': nested_get(comparison, ['threshold_predictions', 'match_ratio']),
            'logits_max_abs_error': nested_get(comparison, ['logits', 'max_abs_error']),
            'probabilities_max_abs_error': nested_get(comparison, ['probabilities', 'max_abs_error']),
            'overall_passed': secure_profile.get('overall_passed'),
            'replay_overall_passed': secure_profile.get('replay_overall_passed'),
        },
        'secure_runtime': {
            'runtime': secure_profile.get('runtime'),
            'total_pipeline_duration_sec': step_profile.get('total_pipeline_duration_sec'),
            'network_kth_bridge_sec': step_profile.get('network_kth_bridge_elapsed_sec')
            or nested_get(step_profile, ['durations_sec', 'network_kth_bridge']),
            'tie_bridge_sec': step_profile.get('tie_bridge_elapsed_sec')
            or nested_get(step_profile, ['durations_sec', 'tie_policy_bridge']),
            'communication_status': communication_profile.get('status'),
            'rpc_total_bytes': rpc_total_bytes,
            'communication_note': communication_profile.get('note') or communication_profile.get('warning'),
        },
    }


def extract_summary_payload(summary_json):
    payload = load_json(summary_json)
    return payload or {}


def parse_mpcvit_single(summary_json):
    payload = extract_summary_payload(summary_json)
    if not payload:
        return None
    best = payload.get('best') or {}
    val = best.get('val') or {}
    args = payload.get('args') or {}
    sample_counts = payload.get('sample_counts') or {}
    return {
        'type': 'single_seed',
        'summary_json': str(summary_json),
        'seed': args.get('seed'),
        'epoch': best.get('epoch'),
        'train_dir': args.get('train_dir'),
        'val_dir': args.get('val_dir'),
        'train_dir_resolved': normalize_path(args.get('train_dir')),
        'val_dir_resolved': normalize_path(args.get('val_dir')),
        'train_samples': sample_counts.get('train'),
        'val_samples': sample_counts.get('val') or val.get('sample_count'),
        'elapsed_sec': payload.get('elapsed_sec'),
        'metrics': {
            'argmax_accuracy': val.get('argmax_accuracy'),
            'threshold_accuracy': val.get('threshold_accuracy'),
            'auc': val.get('auc'),
            'threshold': val.get('threshold'),
            'loss': val.get('loss'),
        },
        'config': {
            'model': args.get('model'),
            'epochs': args.get('epochs'),
            'batch_size': args.get('batch_size'),
            'img_size': args.get('img_size'),
            'device': args.get('device'),
            'class_balanced_loss': args.get('class_balanced_loss'),
            'no_hflip': args.get('no_hflip'),
        },
    }


def parse_mpcvit_multiseed(summary_json):
    payload = extract_summary_payload(summary_json)
    if not payload:
        return None
    rows = []
    train_dir_values = []
    val_dir_values = []
    train_sample_values = []
    val_sample_values = []
    for row in payload.get('runs') or []:
        summary_path = Path(row.get('summary_json', ''))
        summary_payload = load_json(summary_path) or {}
        args = summary_payload.get('args') or {}
        sample_counts = summary_payload.get('sample_counts') or {}
        item = row.get('best_by_argmax') or {}
        val = item.get('val') or {}
        train_dir = normalize_path(args.get('train_dir'))
        val_dir = normalize_path(args.get('val_dir'))
        train_samples = row.get('train_samples') or sample_counts.get('train')
        val_samples = row.get('val_samples') or sample_counts.get('val')
        train_dir_values.append(train_dir)
        val_dir_values.append(val_dir)
        train_sample_values.append(train_samples)
        val_sample_values.append(val_samples)
        rows.append(
            {
                'seed': row.get('seed'),
                'summary_json': row.get('summary_json'),
                'epoch': item.get('epoch'),
                'train_dir': args.get('train_dir'),
                'val_dir': args.get('val_dir'),
                'train_dir_resolved': train_dir,
                'val_dir_resolved': val_dir,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'elapsed_sec': row.get('elapsed_sec'),
                'metrics': {
                    'argmax_accuracy': val.get('argmax_accuracy'),
                    'threshold_accuracy': val.get('threshold_accuracy'),
                    'auc': val.get('auc'),
                    'threshold': val.get('threshold'),
                    'loss': val.get('loss'),
                },
            }
        )
    aggregate = payload.get('aggregate', {}).get('best_by_argmax') or {}
    unique_train_dirs = sorted({value for value in train_dir_values if value})
    unique_val_dirs = sorted({value for value in val_dir_values if value})
    unique_train_samples = sorted({int(value) for value in train_sample_values if value is not None})
    unique_val_samples = sorted({int(value) for value in val_sample_values if value is not None})
    return {
        'type': 'multi_seed',
        'summary_json': str(summary_json),
        'output_root': payload.get('output_root'),
        'runs': rows,
        'sample_count': unique_val_samples[0] if len(unique_val_samples) == 1 else None,
        'train_dir': unique_train_dirs[0] if len(unique_train_dirs) == 1 else None,
        'val_dir': unique_val_dirs[0] if len(unique_val_dirs) == 1 else None,
        'train_dir_resolved': unique_train_dirs[0] if len(unique_train_dirs) == 1 else None,
        'val_dir_resolved': unique_val_dirs[0] if len(unique_val_dirs) == 1 else None,
        'train_samples': unique_train_samples[0] if len(unique_train_samples) == 1 else None,
        'val_samples': unique_val_samples[0] if len(unique_val_samples) == 1 else None,
        'metrics': {
            'argmax_accuracy': nested_get(aggregate, ['argmax_accuracy', 'mean']) or nested_get(aggregate, ['argmax_accuracy', 'max']),
            'threshold_accuracy': nested_get(aggregate, ['threshold_accuracy', 'mean']) or nested_get(aggregate, ['threshold_accuracy', 'max']),
            'auc': nested_get(aggregate, ['auc', 'mean']) or nested_get(aggregate, ['auc', 'max']),
        },
        'aggregate': aggregate,
        'fairness_metadata': {
            'all_runs_same_train_dir': len(unique_train_dirs) <= 1,
            'all_runs_same_val_dir': len(unique_val_dirs) <= 1,
            'all_runs_same_train_sample_count': len(unique_train_samples) <= 1,
            'all_runs_same_val_sample_count': len(unique_val_samples) <= 1,
            'unique_train_dirs': unique_train_dirs,
            'unique_val_dirs': unique_val_dirs,
            'unique_train_sample_counts': unique_train_samples,
            'unique_val_sample_counts': unique_val_samples,
        },
    }


def parse_mpcvit(mpcvit_summary_json=None, mpcvit_multiseed_json=None):
    if mpcvit_multiseed_json:
        parsed = parse_mpcvit_multiseed(mpcvit_multiseed_json)
        if parsed:
            return parsed
    if mpcvit_summary_json:
        return parse_mpcvit_single(mpcvit_summary_json)
    return None


def build_path_match(actual_path, requested_path):
    return bool(actual_path) and bool(requested_path) and actual_path == requested_path


def build_sample_count_match(actual_count, requested_count):
    return actual_count is not None and requested_count is not None and int(actual_count) == int(requested_count)


def build_hash_match(actual_hash, requested_hash):
    return actual_hash is not None and requested_hash is not None and actual_hash == requested_hash


def build_transshield_fairness_checks(transshield, dataset_reference):
    trans_val_path = transshield.get('data_path_resolved')
    trans_sample_count = transshield.get('sample_count')
    trans_sample_hash = transshield.get('sample_paths_sha256')
    requested_val_path = dataset_reference.get('val_path')
    requested_val_count = dataset_reference.get('val_sample_count')
    requested_val_hash = dataset_reference.get('val_sample_paths_sha256')
    return {
        'val_path': trans_val_path,
        'sample_count': trans_sample_count,
        'sample_paths_sha256': trans_sample_hash,
        'matches_requested_val_path': build_path_match(trans_val_path, requested_val_path),
        'matches_requested_val_sample_count': build_sample_count_match(trans_sample_count, requested_val_count),
        'matches_requested_val_sample_paths_sha256': build_hash_match(trans_sample_hash, requested_val_hash),
    }


def build_mpcvit_fairness_checks(mpcvit, dataset_reference):
    if not mpcvit:
        return {
            'type': None,
            'train_path': None,
            'val_path': None,
            'train_sample_count': None,
            'val_sample_count': None,
            'matches_requested_train_path': False,
            'matches_requested_val_path': False,
            'matches_requested_train_sample_count': False,
            'matches_requested_val_sample_count': False,
            'multi_seed_consistency': None,
        }
    mpc_train_path = mpcvit.get('train_dir_resolved')
    mpc_val_path = mpcvit.get('val_dir_resolved')
    mpc_train_sample_count = mpcvit.get('train_samples')
    mpc_val_sample_count = mpcvit.get('sample_count') or mpcvit.get('val_samples')
    return {
        'type': mpcvit.get('type'),
        'train_path': mpc_train_path,
        'val_path': mpc_val_path,
        'train_sample_count': mpc_train_sample_count,
        'val_sample_count': mpc_val_sample_count,
        'matches_requested_train_path': build_path_match(mpc_train_path, dataset_reference.get('train_path')),
        'matches_requested_val_path': build_path_match(mpc_val_path, dataset_reference.get('val_path')),
        'matches_requested_train_sample_count': build_sample_count_match(mpc_train_sample_count, dataset_reference.get('train_sample_count')),
        'matches_requested_val_sample_count': build_sample_count_match(mpc_val_sample_count, dataset_reference.get('val_sample_count')),
        'multi_seed_consistency': mpcvit.get('fairness_metadata'),
    }


def is_multiseed_consistent(mpcvit_checks):
    metadata = mpcvit_checks.get('multi_seed_consistency') or {}
    return all(
        metadata.get(key, True)
        for key in [
            'all_runs_same_train_dir',
            'all_runs_same_val_dir',
            'all_runs_same_train_sample_count',
            'all_runs_same_val_sample_count',
        ]
    )


def build_accuracy_fairness(transshield, mpcvit, transshield_checks, mpcvit_checks):
    trans_hash_ok = transshield_checks['matches_requested_val_sample_paths_sha256'] or transshield.get('sample_paths_sha256') is None
    accuracy_fair = bool(
        transshield.get('available')
        and mpcvit
        and transshield_checks['matches_requested_val_path']
        and transshield_checks['matches_requested_val_sample_count']
        and trans_hash_ok
        and mpcvit_checks['matches_requested_train_path']
        and mpcvit_checks['matches_requested_val_path']
        and mpcvit_checks['matches_requested_train_sample_count']
        and mpcvit_checks['matches_requested_val_sample_count']
        and is_multiseed_consistent(mpcvit_checks)
    )
    if not transshield.get('available') or not mpcvit:
        reason = 'Transshield 或 MPCViT 结果缺失，暂时无法做同口径外部对比。'
    elif accuracy_fair:
        reason = 'Transshield 与 MPCViT 都指向同一组 train/val 路径，样本量也一致，可做同数据集效果对比。'
    else:
        reason = '当前运行结果存在路径、样本量或多 seed 汇总口径不一致，不能把这组数字当作严格公平对比。'
    return accuracy_fair, reason


def build_accuracy_rows(transshield, mpcvit):
    rows = []
    trans_metrics = transshield.get('metrics') or {}
    mpc_metrics = (mpcvit or {}).get('metrics') or {}
    if transshield.get('available'):
        rows.append(
            {
                'method': 'Transshield modified + secure replay',
                'scope': '当前项目',
                'sample_count': transshield.get('sample_count'),
                'argmax_accuracy': trans_metrics.get('argmax_accuracy'),
                'threshold_accuracy': trans_metrics.get('threshold_accuracy'),
                'auc': trans_metrics.get('auc'),
                'source': transshield.get('plaintext_eval_json'),
            }
        )
    if mpcvit:
        rows.append(
            {
                'method': 'MPCViT',
                'scope': '外部同数据集明文 baseline',
                'sample_count': mpcvit.get('sample_count') or mpcvit.get('val_samples'),
                'argmax_accuracy': mpc_metrics.get('argmax_accuracy'),
                'threshold_accuracy': mpc_metrics.get('threshold_accuracy'),
                'auc': mpc_metrics.get('auc'),
                'source': mpcvit.get('summary_json'),
            }
        )
    return rows


def build_accuracy_comparison(transshield, mpcvit):
    if not (transshield.get('available') and mpcvit):
        return {}
    trans_metrics = transshield.get('metrics') or {}
    mpc_metrics = mpcvit.get('metrics') or {}
    return {
        'transshield_minus_mpcvit_argmax_accuracy_pt': pct_gap(
            trans_metrics.get('argmax_accuracy'), mpc_metrics.get('argmax_accuracy')
        ),
        'transshield_minus_mpcvit_threshold_accuracy_pt': pct_gap(
            trans_metrics.get('threshold_accuracy'), mpc_metrics.get('threshold_accuracy')
        ),
        'transshield_minus_mpcvit_auc': pct_gap(trans_metrics.get('auc'), mpc_metrics.get('auc')),
    }


def build_communication_rows(transshield, mpcvit):
    rows = []
    secure_runtime = transshield.get('secure_runtime') or {}
    if transshield.get('available'):
        rows.append(
            {
                'method': 'Transshield SPU secure sidecar',
                'sample_count': transshield.get('sample_count'),
                'runtime': secure_runtime.get('runtime'),
                'total_pipeline_duration_sec': secure_runtime.get('total_pipeline_duration_sec'),
                'rpc_total_bytes': secure_runtime.get('rpc_total_bytes'),
                'communication_status': secure_runtime.get('communication_status'),
                'source': transshield.get('secure_profile_json'),
                'comparable_to_external': False,
                'reason': '当前没有外部模型在同数据集、同输入、同协议路径下的 secure 通信结果。',
            }
        )
    if mpcvit:
        rows.append(
            {
                'method': 'MPCViT',
                'sample_count': mpcvit.get('sample_count') or mpcvit.get('val_samples'),
                'runtime': None,
                'total_pipeline_duration_sec': None,
                'rpc_total_bytes': None,
                'communication_status': 'not_run_secure',
                'source': mpcvit.get('summary_json'),
                'comparable_to_external': False,
                'reason': '当前 MPCViT 只有同数据集明文训练/评估结果，没有同协议 secure 通信结果。',
            }
        )
    return rows


def build_fairness_scope(args, accuracy_fair: bool, accuracy_fair_reason: str):
    return {
        'dataset': args.dataset_name,
        'train_data_path': args.train_data_path,
        'val_data_path': args.val_data_path,
        'same_dataset_accuracy_comparison': accuracy_fair,
        'same_protocol_secure_communication_comparison': False,
        'accuracy_comparison_reason': accuracy_fair_reason,
        'communication_comparison_reason': (
            '外部 MPCViT 当前没有同数据集、同输入、同 SPU/2PC 协议路径的 secure 通信结果；'
            '因此通信量只能展示 Transshield 自身 live/full-run 结果，不能和外部模型硬比。'
        ),
    }


def build_fairness_checks(dataset_reference, transshield_checks, mpcvit_checks, accuracy_fair: bool, accuracy_fair_reason: str):
    return {
        'requested_dataset': dataset_reference,
        'transshield': transshield_checks,
        'mpcvit': mpcvit_checks,
        'accuracy_comparison_is_fair': accuracy_fair,
        'accuracy_comparison_reason': accuracy_fair_reason,
    }


def build_payload(args):
    dataset_reference = build_dataset_reference(args.train_data_path, args.val_data_path)
    transshield = parse_transshield_run(args.transshield_run_dir)
    mpcvit = parse_mpcvit(args.mpcvit_summary_json, args.mpcvit_multiseed_json)
    transshield_checks = build_transshield_fairness_checks(transshield, dataset_reference)
    mpcvit_checks = build_mpcvit_fairness_checks(mpcvit, dataset_reference)
    accuracy_fair, accuracy_fair_reason = build_accuracy_fairness(
        transshield,
        mpcvit,
        transshield_checks,
        mpcvit_checks,
    )

    payload = {
        'title': 'Transshield 公平外部对比',
        'fairness_scope': build_fairness_scope(args, accuracy_fair, accuracy_fair_reason),
        'fairness_checks': build_fairness_checks(
            dataset_reference,
            transshield_checks,
            mpcvit_checks,
            accuracy_fair,
            accuracy_fair_reason,
        ),
        'transshield': transshield,
        'external_baselines': {
            'mpcvit': mpcvit,
        },
        'tables': {
            'accuracy': build_accuracy_rows(transshield, mpcvit),
            'communication': build_communication_rows(transshield, mpcvit),
        },
        'comparison': build_accuracy_comparison(transshield, mpcvit),
    }
    return payload


def build_fairness_markdown_lines(payload):
    multi_seed_consistency = nested_get(payload, ['fairness_checks', 'mpcvit', 'multi_seed_consistency'])
    multi_seed_consistency_display = f"`{multi_seed_consistency}`" if multi_seed_consistency else '`N/A (single_seed)`'
    return [
        '# Transshield 公平外部对比',
        '',
        '## 公平口径',
        '',
        f"- 数据集：`{payload['fairness_scope']['dataset']}`",
        f"- 训练集：`{payload['fairness_scope']['train_data_path']}`",
        f"- 验证集：`{payload['fairness_scope']['val_data_path']}`",
        f"- 准确率对比是否同数据集：`{payload['fairness_scope']['same_dataset_accuracy_comparison']}`",
        f"- 准确率口径说明：{payload['fairness_scope']['accuracy_comparison_reason']}",
        f"- secure 通信量是否同协议可比：`{payload['fairness_scope']['same_protocol_secure_communication_comparison']}`",
        f"- 通信量说明：{payload['fairness_scope']['communication_comparison_reason']}",
        '',
        '## 公平性自检',
        '',
        '| 检查项 | 当前值 | 结果 |',
        '|---|---|---|',
        f"| 请求的 train 样本量 | {fmt(nested_get(payload, ['fairness_checks', 'requested_dataset', 'train_sample_count']), 0)} | 基准 |",
        f"| 请求的 val 样本量 | {fmt(nested_get(payload, ['fairness_checks', 'requested_dataset', 'val_sample_count']), 0)} | 基准 |",
        f"| Transshield val 路径 | `{fmt(nested_get(payload, ['fairness_checks', 'transshield', 'val_path']))}` | {fmt(nested_get(payload, ['fairness_checks', 'transshield', 'matches_requested_val_path']))} |",
        f"| Transshield val 样本量 | {fmt(nested_get(payload, ['fairness_checks', 'transshield', 'sample_count']), 0)} | {fmt(nested_get(payload, ['fairness_checks', 'transshield', 'matches_requested_val_sample_count']))} |",
        f"| Transshield val 文件列表哈希 | `{fmt(nested_get(payload, ['fairness_checks', 'transshield', 'sample_paths_sha256']))}` | {fmt(nested_get(payload, ['fairness_checks', 'transshield', 'matches_requested_val_sample_paths_sha256']))} |",
        f"| MPCViT train 路径 | `{fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'train_path']))}` | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'matches_requested_train_path']))} |",
        f"| MPCViT val 路径 | `{fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'val_path']))}` | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'matches_requested_val_path']))} |",
        f"| MPCViT train 样本量 | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'train_sample_count']), 0)} | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'matches_requested_train_sample_count']))} |",
        f"| MPCViT val 样本量 | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'val_sample_count']), 0)} | {fmt(nested_get(payload, ['fairness_checks', 'mpcvit', 'matches_requested_val_sample_count']))} |",
        f"| 多 seed 口径一致 | {multi_seed_consistency_display} | 仅多 seed 有意义 |",
        '',
    ]


def build_accuracy_markdown_lines(payload):
    lines = [
        '## 准确率 / AUC 结果表（仅公平性通过时可用于主对比）',
        '',
        '| 方法 | 样本数 | Argmax Acc (%) | Threshold Acc (%) | AUC | 来源 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for row in payload['tables']['accuracy']:
        lines.append(
            f"| {row['method']} | {fmt(row.get('sample_count'), 0)} | "
            f"{fmt(row.get('argmax_accuracy'))} | {fmt(row.get('threshold_accuracy'))} | "
            f"{fmt(row.get('auc'))} | `{row.get('source')}` |"
        )
    return lines


def build_comparison_markdown_lines(payload):
    comparison = payload.get('comparison') or {}
    if not comparison:
        return []
    return [
        '',
        '## 差值',
        '',
        '| 指标 | Transshield - MPCViT |',
        '|---|---:|',
        f"| Argmax Acc | {fmt(comparison.get('transshield_minus_mpcvit_argmax_accuracy_pt'))} pt |",
        f"| Threshold Acc | {fmt(comparison.get('transshield_minus_mpcvit_threshold_accuracy_pt'))} pt |",
        f"| AUC | {fmt(comparison.get('transshield_minus_mpcvit_auc'))} |",
    ]


def build_communication_markdown_lines(payload):
    lines = [
        '',
        '## Secure 通信 / 运行开销（当前不做外部硬比）',
        '',
        '| 方法 | 样本数 | Runtime | Total Pipeline Sec | RPC Bytes | 状态 | 是否外部可比 | 原因 |',
        '|---|---:|---|---:|---:|---|---|---|',
    ]
    for row in payload['tables']['communication']:
        lines.append(
            f"| {row['method']} | {fmt(row.get('sample_count'), 0)} | {fmt(row.get('runtime'))} | "
            f"{fmt(row.get('total_pipeline_duration_sec'))} | {fmt(row.get('rpc_total_bytes'), 0)} | "
            f"{fmt(row.get('communication_status'))} | {fmt(row.get('comparable_to_external'))} | "
            f"{row.get('reason')} |"
        )
    return lines


def build_conclusion_markdown_lines():
    return [
        '',
        '## 结论',
        '',
        '- 当前仅在公平性自检通过时，才能把准确率 / AUC 当作正式外部对比口径。',
        '- 当前不能公平展示：外部模型 secure 通信量对比。',
        '- 如果要补 secure 通信外部对比，需要让外部模型也走同输入、同样本量、同 SPU/2PC 协议路径后再统计。',
        '',
    ]


def write_markdown(path, payload):
    lines = []
    lines.extend(build_fairness_markdown_lines(payload))
    lines.extend(build_accuracy_markdown_lines(payload))
    lines.extend(build_comparison_markdown_lines(payload))
    lines.extend(build_communication_markdown_lines(payload))
    lines.extend(build_conclusion_markdown_lines())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description='Build a fair external comparison report for Transshield.')
    parser.add_argument('--transshield-run-dir', required=True)
    parser.add_argument('--mpcvit-summary-json', default='')
    parser.add_argument('--mpcvit-multiseed-json', default='')
    parser.add_argument('--train-data-path', default='')
    parser.add_argument('--val-data-path', default='')
    parser.add_argument('--dataset-name', default='PneumoniaMNIST imagefolder subset')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    payload = build_payload(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)


if __name__ == '__main__':
    main()
