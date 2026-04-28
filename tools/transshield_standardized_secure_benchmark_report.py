#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


COMMUNICATION_KEYS = (
    'linear_comm_byte',
    'softmax_comm_byte',
    'act_comm_byte',
    'norm_comm_byte',
    'embed_comm_byte',
)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


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


def human_bytes(value):
    if value is None:
        return 'N/A'
    value = float(value)
    for suffix in ('B', 'KiB', 'MiB', 'GiB'):
        if abs(value) < 1024.0 or suffix == 'GiB':
            return f'{value:.2f} {suffix}' if suffix != 'B' else f'{value:.0f} {suffix}'
        value /= 1024.0
    return f'{value:.2f} GiB'


def metric_mean(summary, key):
    value = summary.get(f'{key}_mean')
    return float(value) if isinstance(value, (int, float)) else None


def module_comm_mean(summary):
    value = summary.get('module_comm_byte_mean')
    if isinstance(value, (int, float)):
        return float(value)
    values = [metric_mean(summary, key) for key in COMMUNICATION_KEYS]
    if any(value is None for value in values):
        return None
    return float(sum(values))


def parse_profile(profile_dir):
    profile_dir = Path(profile_dir)
    meta = load_json(profile_dir / 'profile_meta.json') or {}
    summary = load_json(profile_dir / 'summary.json') or {}
    if not summary:
        return None
    communication_breakdown = {
        key: metric_mean(summary, key)
        for key in COMMUNICATION_KEYS
    }
    total_comm_bytes = module_comm_mean(summary)
    total_time = metric_mean(summary, 'total_time')
    return {
        'profile_id': meta.get('profile_id') or profile_dir.name,
        'display_name': meta.get('display_name') or profile_dir.name,
        'role': meta.get('role') or 'reference',
        'comparison_group': meta.get('comparison_group') or 'standardized_secure_benchmark',
        'model_source': meta.get('model_source'),
        'scope_note': meta.get('scope_note'),
        'output_dir': str(profile_dir),
        'summary_json': str(profile_dir / 'summary.json'),
        'config': {
            'batch_size': meta.get('batch_size'),
            'num_hidden_layers': meta.get('num_hidden_layers'),
            'hidden_size': meta.get('hidden_size'),
            'intermediate_size': meta.get('intermediate_size'),
            'sequence_length': meta.get('sequence_length'),
            'num_attention_heads': meta.get('num_attention_heads'),
            'hidden_act': meta.get('hidden_act'),
            'softmax_act': meta.get('softmax_act'),
            'warmup': meta.get('warmup'),
            'repeats': meta.get('repeats'),
            'world_size': meta.get('world_size'),
        },
        'metrics': {
            'measurement_count': summary.get('measurement_count'),
            'total_time_mean_sec': total_time,
            'module_comm_byte_mean': total_comm_bytes,
            'module_comm_mib_mean': total_comm_bytes / (1024 ** 2) if total_comm_bytes is not None else None,
            'communication_breakdown_byte_mean': communication_breakdown,
        },
    }


def compare_profiles(left, right):
    left_comm = left.get('metrics', {}).get('module_comm_byte_mean')
    right_comm = right.get('metrics', {}).get('module_comm_byte_mean')
    left_time = left.get('metrics', {}).get('total_time_mean_sec')
    right_time = right.get('metrics', {}).get('total_time_mean_sec')
    return {
        'left_profile_id': left.get('profile_id'),
        'right_profile_id': right.get('profile_id'),
        'left_display_name': left.get('display_name'),
        'right_display_name': right.get('display_name'),
        'module_comm_delta_bytes': left_comm - right_comm if left_comm is not None and right_comm is not None else None,
        'module_comm_ratio_left_over_right': left_comm / right_comm if left_comm is not None and right_comm else None,
        'time_delta_sec': left_time - right_time if left_time is not None and right_time is not None else None,
        'time_ratio_left_over_right': left_time / right_time if left_time is not None and right_time else None,
    }


def build_report(output_root):
    output_root = Path(output_root)
    profiles = []
    for profile_dir in sorted(output_root.iterdir() if output_root.exists() else []):
        if not profile_dir.is_dir():
            continue
        profile = parse_profile(profile_dir)
        if profile:
            profiles.append(profile)

    groups = {}
    for profile in profiles:
        groups.setdefault(profile.get('comparison_group') or 'standardized_secure_benchmark', []).append(profile)

    comparisons = []
    for group_name, group_profiles in sorted(groups.items()):
        current_profile = next((item for item in group_profiles if item.get('role') == 'current_project'), None)
        external_profile = next((item for item in group_profiles if item.get('role') == 'external_baseline'), None)
        if current_profile and external_profile:
            comparison = compare_profiles(current_profile, external_profile)
            comparison['comparison_group'] = group_name
            if group_name == 'architecture_proxy':
                comparison['scope_note'] = (
                    '同一 MPCFormer local 2PC benchmark harness；各 profile 使用各自模型结构参数。'
                    '这不是 full-val 医学图像 pipeline 通信量。'
                )
            elif group_name == 'same_shape_operator_proxy':
                comparison['scope_note'] = (
                    '同一模型形状、同一 2PC benchmark harness；主要用于观察算子配置导致的安全开销差异。'
                )
            else:
                comparison['scope_note'] = '同一 benchmark harness 下的 profile 对比。'
            comparisons.append(comparison)

    return {
        'title': 'Transshield 统一 secure transformer benchmark',
        'updated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'output_root': str(output_root),
        'scope': {
            'benchmark_harness': 'MPCFormer local 2PC configurable transformer benchmark',
            'is_full_val_image_pipeline': False,
            'is_single_image_live_run': False,
            'summary': (
                '这些结果只表示同一 secure transformer benchmark harness 下的通信 / 时间 profile，'
                '不能和当前网页单图 live run 或 full-val Transshield SPU sidecar 通信量混为一谈。'
            ),
        },
        'profiles': profiles,
        'comparisons': comparisons,
    }


def markdown_table(report):
    lines = [
        '# Transshield 统一 secure transformer benchmark',
        '',
        '## 口径说明',
        '',
        f"- Benchmark harness：`{report['scope']['benchmark_harness']}`",
        f"- 是否 full-val 医学图像 pipeline：`{fmt(report['scope']['is_full_val_image_pipeline'])}`",
        f"- 是否网页单图 live run：`{fmt(report['scope']['is_single_image_live_run'])}`",
        f"- 说明：{report['scope']['summary']}",
        '',
        '## Profile 结果',
        '',
        '| Profile | 角色 | 结构 / 算子 | 平均总时间 | 平均模块通信 | 来源 |',
        '|---|---|---|---:|---:|---|',
    ]
    for profile in report.get('profiles') or []:
        config = profile.get('config') or {}
        metrics = profile.get('metrics') or {}
        shape = (
            f"L={fmt(config.get('num_hidden_layers'), 0)}, "
            f"H={fmt(config.get('hidden_size'), 0)}, "
            f"I={fmt(config.get('intermediate_size'), 0)}, "
            f"seq={fmt(config.get('sequence_length'), 0)}, "
            f"heads={fmt(config.get('num_attention_heads'), 0)}, "
            f"act={fmt(config.get('hidden_act'))}, "
            f"softmax={fmt(config.get('softmax_act'))}"
        )
        lines.append(
            f"| {fmt(profile.get('display_name'))} | {fmt(profile.get('role'))} | {shape} | "
            f"{fmt(metrics.get('total_time_mean_sec'), 4)}s | "
            f"{human_bytes(metrics.get('module_comm_byte_mean'))} | "
            f"`{fmt(profile.get('summary_json'))}` |"
        )
    lines.extend(['', '## 对比结果', '', '| 分组 | 左侧 | 右侧 | 通信差值 | 通信比例 | 时间差值 | 说明 |', '|---|---|---|---:|---:|---:|---|'])
    for comparison in report.get('comparisons') or []:
        lines.append(
            f"| {fmt(comparison.get('comparison_group'))} | {fmt(comparison.get('left_display_name'))} | "
            f"{fmt(comparison.get('right_display_name'))} | "
            f"{human_bytes(comparison.get('module_comm_delta_bytes'))} | "
            f"{fmt(comparison.get('module_comm_ratio_left_over_right'), 4)}x | "
            f"{fmt(comparison.get('time_delta_sec'), 4)}s | "
            f"{fmt(comparison.get('scope_note'))} |"
        )
    lines.extend(
        [
            '',
            '## 使用限制',
            '',
            '- 可以用于说明外部模型 / 算子在同一 secure benchmark harness 下的开销差异。',
            '- 不能写成 Transshield full-val SPU sidecar 与外部模型 full pipeline 的严格通信量对比。',
            '- 如果要做 full-val 通信量公平对比，外部模型也必须接入同输入、同样本量、同协议路径后重新统计。',
        ]
    )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Summarize standardized secure transformer benchmark runs.')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    report = build_report(args.output_root)
    write_json(args.output_json, report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(markdown_table(report), encoding='utf-8')
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == '__main__':
    main()
