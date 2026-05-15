#!/usr/bin/env python3
import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional


LOSS_INFO_RE = re.compile(
    r'loss info: .*?cls_kl=(?P<cls_kl>[-+0-9.eE]+),\s*token_kl=(?P<token_kl>[-+0-9.eE]+)'
)


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


def parse_command_flags(command_path: Optional[Path]) -> Dict[str, str]:
    if command_path is None or not command_path.is_file():
        return {}
    content = command_path.read_text(encoding='utf-8').strip()
    if not content:
        return {}
    tokens = shlex.split(content)
    flags: Dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith('--') and idx + 1 < len(tokens) and not tokens[idx + 1].startswith('--'):
            flags[token[2:]] = tokens[idx + 1]
            idx += 2
        else:
            idx += 1
    return flags


def mean(values: List[float]):
    if not values:
        return None
    return float(sum(values) / len(values))


def parse_float(text: Optional[str], default: float = 0.0) -> float:
    if text in (None, ''):
        return default
    return float(text)


def build_report(train_log_path: Path):
    lines = train_log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    run_dir = train_log_path.parent
    command_path = run_dir / 'command.sh'
    command_flags = parse_command_flags(command_path)

    cls_weight = parse_float(command_flags.get('cls_distill_weight'), default=0.0)
    token_weight = parse_float(command_flags.get('token_distill_weight'), default=0.0)

    loss_entries = []
    for line_number, raw_line in enumerate(lines, start=1):
        match = LOSS_INFO_RE.search(raw_line)
        if not match:
            continue
        cls_kl = float(match.group('cls_kl'))
        token_kl = float(match.group('token_kl'))
        loss_entries.append(
            {
                'line_number': line_number,
                'cls_kl': cls_kl,
                'token_kl': token_kl,
                'effective_cls_term': cls_weight * cls_kl,
                'effective_token_term': token_weight * token_kl,
            }
        )

    cls_kl_values = [entry['cls_kl'] for entry in loss_entries]
    token_kl_values = [entry['token_kl'] for entry in loss_entries]
    effective_cls_values = [entry['effective_cls_term'] for entry in loss_entries]
    effective_token_values = [entry['effective_token_term'] for entry in loss_entries]
    debug_max_steps = command_flags.get('debug_max_steps')

    if not loss_entries:
        if debug_max_steps is not None and int(debug_max_steps) < 100:
            interpretation = 'no_loss_info_lines_expected_under_debug80'
            interpretation_reason = (
                'command.sh shows debug_max_steps<100 while the current print cadence is 100 steps, '
                'so missing distill loss logs is expected for this short run.'
            )
        else:
            interpretation = 'no_loss_info_lines_found'
            interpretation_reason = 'train_stdout.log does not contain any loss info lines with cls_kl/token_kl output.'
    elif cls_weight <= 0 and token_weight <= 0:
        interpretation = 'distill_disabled_reference'
        interpretation_reason = 'command.sh keeps both cls/token distill weights at 0, so this run acts as the no-distill reference.'
    elif any(abs(value) > 0 for value in effective_cls_values + effective_token_values):
        interpretation = 'distill_terms_observed'
        interpretation_reason = 'distill loss lines were parsed successfully and the configured weights produce non-zero effective distill terms.'
    else:
        interpretation = 'distill_weights_positive_but_no_effective_terms'
        interpretation_reason = 'distill weights are positive, but parsed effective distill terms are all zero in the observed logging window.'

    report = {
        'manifest_type': 'transshield_distill_log_report_v0',
        'inputs': {
            'train_log': str(train_log_path.resolve()),
            'command_sh': str(command_path.resolve()) if command_path.is_file() else None,
        },
        'configured_distill': {
            'cls_distill_weight': command_flags.get('cls_distill_weight'),
            'token_distill_weight': command_flags.get('token_distill_weight'),
            'ratio_weight': command_flags.get('ratio_weight'),
            'secure_static_train_depth': command_flags.get('secure_static_train_depth'),
            'secure_static_skip_pruning': command_flags.get('secure_static_skip_pruning'),
            'debug_max_steps': command_flags.get('debug_max_steps'),
            'epochs': command_flags.get('epochs'),
            'batch_size': command_flags.get('batch_size'),
        },
        'log_summary': {
            'loss_info_line_count': len(loss_entries),
            'mean_cls_kl': mean(cls_kl_values),
            'max_cls_kl': max(cls_kl_values, default=None),
            'mean_token_kl': mean(token_kl_values),
            'max_token_kl': max(token_kl_values, default=None),
            'mean_effective_cls_term': mean(effective_cls_values),
            'max_effective_cls_term': max(effective_cls_values, default=None),
            'mean_effective_token_term': mean(effective_token_values),
            'max_effective_token_term': max(effective_token_values, default=None),
            'nonzero_effective_distill_line_count': int(
                sum(1 for entry in loss_entries if abs(entry['effective_cls_term']) > 0 or abs(entry['effective_token_term']) > 0)
            ),
        },
        'interpretation': {
            'status': interpretation,
            'reason': interpretation_reason,
        },
    }
    return report


def render_markdown(report: dict):
    configured = report['configured_distill']
    summary = report['log_summary']
    lines = [
        '# Distill Log Report',
        '',
        '## 1. 结论',
        '',
        f"- status: `{report['interpretation']['status']}`",
        f"- reason: {report['interpretation']['reason']}",
        '',
        '## 2. 当前配置',
        '',
        f"- cls_distill_weight: `{configured['cls_distill_weight']}`",
        f"- token_distill_weight: `{configured['token_distill_weight']}`",
        f"- ratio_weight: `{configured['ratio_weight']}`",
        f"- debug_max_steps: `{configured['debug_max_steps']}`",
        '',
        '## 3. 日志摘要',
        '',
        f"- loss_info_line_count: `{summary['loss_info_line_count']}`",
        f"- mean_cls_kl: `{fmt_number(summary['mean_cls_kl'], 8)}`",
        f"- max_cls_kl: `{fmt_number(summary['max_cls_kl'], 8)}`",
        f"- mean_token_kl: `{fmt_number(summary['mean_token_kl'], 8)}`",
        f"- max_token_kl: `{fmt_number(summary['max_token_kl'], 8)}`",
        f"- mean_effective_cls_term: `{fmt_number(summary['mean_effective_cls_term'], 8)}`",
        f"- mean_effective_token_term: `{fmt_number(summary['mean_effective_token_term'], 8)}`",
        f"- nonzero_effective_distill_line_count: `{summary['nonzero_effective_distill_line_count']}`",
        '',
    ]
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse distill-loss logs from a secure-static training run.')
    parser.add_argument('--train-log', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-md', required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args.train_log)
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))


if __name__ == '__main__':
    main()
