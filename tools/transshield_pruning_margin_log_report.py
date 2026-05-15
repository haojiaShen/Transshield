#!/usr/bin/env python3
import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional


LOSS_INFO_RE = re.compile(
    r'loss info: .*?pruning_margin=(?P<pruning_margin>[-+0-9.eE]+)'
    r'(?:\s+margin_stats=(?P<margin_stats>none|\[.*\]))?'
)
STAGE_OK_RE = re.compile(
    r's(?P<stage>\d+):w=(?P<weight>[-+0-9.eE]+),mean=(?P<mean>[-+0-9.eE]+),'
    r'viol=(?P<viol>[-+0-9.eE]+),loss=(?P<loss>[-+0-9.eE]+)'
)
STAGE_STATUS_RE = re.compile(r's(?P<stage>\d+):(?P<status>[a-zA-Z0-9_]+)')
GLOBAL_BEFORE_START_RE = re.compile(
    r'global:before_start_epoch\(epoch=(?P<epoch>[-+0-9]+),start=(?P<start>[-+0-9]+)\)'
)
GLOBAL_STATUS_RE = re.compile(r'global:(?P<status>[a-zA-Z0-9_]+)')


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


def parse_margin_stats(text: Optional[str]) -> List[dict]:
    if not text or text == 'none':
        return []
    body = text.strip()
    if not body.startswith('[') or not body.endswith(']'):
        return []
    entries = [part.strip() for part in body[1:-1].split(';') if part.strip()]
    parsed = []
    for entry in entries:
        match = STAGE_OK_RE.fullmatch(entry)
        if match:
            parsed.append(
                {
                    'type': 'stage',
                    'stage_index': int(match.group('stage')),
                    'status': 'ok',
                    'stage_weight': float(match.group('weight')),
                    'margin_mean': float(match.group('mean')),
                    'violation_ratio': float(match.group('viol')),
                    'stage_loss_mean': float(match.group('loss')),
                }
            )
            continue
        match = STAGE_STATUS_RE.fullmatch(entry)
        if match:
            parsed.append(
                {
                    'type': 'stage',
                    'stage_index': int(match.group('stage')),
                    'status': match.group('status'),
                }
            )
            continue
        match = GLOBAL_BEFORE_START_RE.fullmatch(entry)
        if match:
            parsed.append(
                {
                    'type': 'global',
                    'status': 'before_start_epoch',
                    'current_epoch': int(match.group('epoch')),
                    'start_epoch': int(match.group('start')),
                }
            )
            continue
        match = GLOBAL_STATUS_RE.fullmatch(entry)
        if match:
            parsed.append(
                {
                    'type': 'global',
                    'status': match.group('status'),
                }
            )
            continue
        parsed.append(
            {
                'type': 'global',
                'status': 'unparsed',
                'raw': entry,
            }
        )
    return parsed


def mean(values: List[float]):
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_stage_entries(entries: List[dict]) -> dict:
    status_counts: Dict[str, int] = {}
    ok_entries = [entry for entry in entries if entry.get('status') == 'ok']
    for entry in entries:
        status = entry.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        'entry_count': len(entries),
        'ok_entry_count': len(ok_entries),
        'status_counts': status_counts,
        'mean_stage_weight': mean([entry.get('stage_weight') for entry in ok_entries if entry.get('stage_weight') is not None]),
        'mean_margin_mean': mean([entry.get('margin_mean') for entry in ok_entries if entry.get('margin_mean') is not None]),
        'mean_violation_ratio': mean([entry.get('violation_ratio') for entry in ok_entries if entry.get('violation_ratio') is not None]),
        'max_violation_ratio': max([entry.get('violation_ratio') for entry in ok_entries if entry.get('violation_ratio') is not None], default=None),
        'mean_stage_loss_mean': mean([entry.get('stage_loss_mean') for entry in ok_entries if entry.get('stage_loss_mean') is not None]),
    }


def compare_stage_weights(configured_csv: Optional[str], recipe_csv: Optional[str]):
    if not configured_csv or not recipe_csv:
        return None
    configured = [item.strip() for item in configured_csv.split(',') if item.strip()]
    recipe = [item.strip() for item in recipe_csv.split(',') if item.strip()]
    return configured == recipe


def build_report(train_log_path: Path, recipe_json_path: Optional[Path], profile_name: Optional[str]):
    lines = train_log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    run_dir = train_log_path.parent
    command_path = run_dir / 'command.sh'
    command_flags = parse_command_flags(command_path)

    loss_entries = []
    stage_entries: Dict[int, List[dict]] = {}
    global_status_counts: Dict[str, int] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        match = LOSS_INFO_RE.search(raw_line)
        if not match:
            continue
        pruning_margin = float(match.group('pruning_margin'))
        margin_stats = parse_margin_stats(match.group('margin_stats'))
        loss_entries.append(
            {
                'line_number': line_number,
                'pruning_margin': pruning_margin,
                'margin_stats': margin_stats,
            }
        )
        for item in margin_stats:
            if item['type'] == 'stage':
                stage_entries.setdefault(int(item['stage_index']), []).append(item)
            else:
                status = item.get('status', 'unknown')
                global_status_counts[status] = global_status_counts.get(status, 0) + 1

    recipe = load_json(recipe_json_path) if recipe_json_path and recipe_json_path.is_file() else None
    selected_profile_name = profile_name or (None if recipe is None else recipe.get('recommended_profile'))
    selected_profile = None if recipe is None or selected_profile_name is None else (recipe.get('profiles') or {}).get(selected_profile_name)

    stage_summaries = {
        str(stage_index): summarize_stage_entries(entries)
        for stage_index, entries in sorted(stage_entries.items())
    }
    pruning_margin_values = [entry['pruning_margin'] for entry in loss_entries]
    debug_max_steps = command_flags.get('debug_max_steps')
    secure_static_skip_pruning = command_flags.get('secure_static_skip_pruning')
    secure_static_train_depth = command_flags.get('secure_static_train_depth')

    if not loss_entries:
        if debug_max_steps is not None and int(debug_max_steps) < 100:
            interpretation = 'no_loss_info_lines_expected_under_debug80'
            interpretation_reason = (
                'command.sh shows debug_max_steps<100 while the current print cadence is 100 steps, '
                'so missing pruning margin logs is expected for this short run.'
            )
        else:
            interpretation = 'no_loss_info_lines_found'
            interpretation_reason = 'train_stdout.log does not contain any loss info lines with pruning_margin output.'
    elif not stage_summaries:
        if secure_static_skip_pruning == 'true' and secure_static_train_depth not in (None, '', '0'):
            interpretation = 'pruning_objective_disabled_by_secure_static_skip_pruning'
            interpretation_reason = (
                'loss info lines exist, but command.sh keeps secure_static_skip_pruning=true under secure_static_train_depth>0, '
                'so predictor stages are bypassed and no pruning margin stats can be produced.'
            )
        else:
            interpretation = 'loss_info_present_but_no_stage_margin_stats'
            interpretation_reason = 'loss info lines exist, but margin_stats is empty/none in the parsed window.'
    else:
        interpretation = 'protocol_margin_stats_available'
        interpretation_reason = 'stage-wise pruning margin stats were parsed successfully from train_stdout.log.'

    report = {
        'manifest_type': 'transshield_pruning_margin_log_report_v0',
        'inputs': {
            'train_log': str(train_log_path.resolve()),
            'command_sh': str(command_path.resolve()) if command_path.is_file() else None,
            'recipe_json': None if recipe_json_path is None else str(recipe_json_path.resolve()),
            'selected_profile': selected_profile_name,
        },
        'configured_objective': {
            'pruning_margin_weight': command_flags.get('pruning_margin_weight'),
            'pruning_margin_target': command_flags.get('pruning_margin_target'),
            'pruning_margin_mode': command_flags.get('pruning_margin_mode'),
            'pruning_margin_stage_weights': command_flags.get('pruning_margin_stage_weights'),
            'pruning_margin_start_epoch': command_flags.get('pruning_margin_start_epoch'),
            'secure_static_train_depth': secure_static_train_depth,
            'secure_static_skip_pruning': secure_static_skip_pruning,
            'debug_max_steps': command_flags.get('debug_max_steps'),
            'epochs': command_flags.get('epochs'),
            'batch_size': command_flags.get('batch_size'),
        },
        'recipe_comparison': None if selected_profile is None else {
            'profile_name': selected_profile_name,
            'matches_weight': command_flags.get('pruning_margin_weight') == str(selected_profile.get('pruning_margin_weight')),
            'matches_target': command_flags.get('pruning_margin_target') == str(selected_profile.get('pruning_margin_target')),
            'matches_mode': command_flags.get('pruning_margin_mode') == str(selected_profile.get('pruning_margin_mode')),
            'matches_start_epoch': command_flags.get('pruning_margin_start_epoch') == str(selected_profile.get('pruning_margin_start_epoch')),
            'matches_stage_weights_csv': compare_stage_weights(
                command_flags.get('pruning_margin_stage_weights'),
                selected_profile.get('pruning_margin_stage_weights_csv'),
            ),
        },
        'log_summary': {
            'loss_info_line_count': len(loss_entries),
            'stage_margin_line_count': int(sum(1 for entry in loss_entries if entry['margin_stats'])),
            'mean_pruning_margin': mean(pruning_margin_values),
            'max_pruning_margin': max(pruning_margin_values, default=None),
            'nonzero_pruning_margin_line_count': int(sum(1 for value in pruning_margin_values if abs(value) > 0)),
        },
        'stage_summaries': stage_summaries,
        'global_status_counts': global_status_counts,
        'interpretation': {
            'status': interpretation,
            'reason': interpretation_reason,
        },
    }
    return report


def render_markdown(report: dict):
    configured = report['configured_objective']
    lines = [
        '# Pruning Margin Log Report',
        '',
        '## 1. 结论',
        '',
        f"- status: `{report['interpretation']['status']}`",
        f"- reason: {report['interpretation']['reason']}",
        '',
        '## 2. 当前配置',
        '',
        f"- pruning_margin_weight: `{configured['pruning_margin_weight']}`",
        f"- pruning_margin_target: `{configured['pruning_margin_target']}`",
        f"- pruning_margin_mode: `{configured['pruning_margin_mode']}`",
        f"- pruning_margin_stage_weights: `{configured['pruning_margin_stage_weights']}`",
        f"- pruning_margin_start_epoch: `{configured['pruning_margin_start_epoch']}`",
        f"- debug_max_steps: `{configured['debug_max_steps']}`",
        '',
        '## 3. 日志摘要',
        '',
        f"- loss_info_line_count: `{report['log_summary']['loss_info_line_count']}`",
        f"- stage_margin_line_count: `{report['log_summary']['stage_margin_line_count']}`",
        f"- mean_pruning_margin: `{fmt_number(report['log_summary']['mean_pruning_margin'], 8)}`",
        f"- max_pruning_margin: `{fmt_number(report['log_summary']['max_pruning_margin'], 8)}`",
        f"- nonzero_pruning_margin_line_count: `{report['log_summary']['nonzero_pruning_margin_line_count']}`",
        '',
        '## 4. Stage 汇总',
        '',
        '| Stage | Entries | OK | Mean Weight | Mean Margin | Mean Viol | Max Viol | Mean Stage Loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    if report['stage_summaries']:
        for stage_index, summary in report['stage_summaries'].items():
            lines.append(
                f"| {stage_index} | {summary['entry_count']} | {summary['ok_entry_count']} | "
                f"{fmt_number(summary['mean_stage_weight'], 4)} | "
                f"{fmt_number(summary['mean_margin_mean'], 8)} | "
                f"{fmt_number(summary['mean_violation_ratio'], 4)} | "
                f"{fmt_number(summary['max_violation_ratio'], 4)} | "
                f"{fmt_number(summary['mean_stage_loss_mean'], 8)} |"
            )
    else:
        lines.append('| N/A | 0 | 0 | N/A | N/A | N/A | N/A | N/A |')

    recipe_comparison = report.get('recipe_comparison')
    if recipe_comparison is not None:
        lines.extend(
            [
                '',
                '## 5. Recipe 对照',
                '',
                f"- profile_name: `{recipe_comparison['profile_name']}`",
                f"- matches_weight: `{fmt_number(recipe_comparison['matches_weight'])}`",
                f"- matches_target: `{fmt_number(recipe_comparison['matches_target'])}`",
                f"- matches_mode: `{fmt_number(recipe_comparison['matches_mode'])}`",
                f"- matches_start_epoch: `{fmt_number(recipe_comparison['matches_start_epoch'])}`",
                f"- matches_stage_weights_csv: `{fmt_number(recipe_comparison['matches_stage_weights_csv'])}`",
            ]
        )
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(description='Parse pruning margin logs from a protocol-aware training run.')
    parser.add_argument('--train-log', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-md', required=True, type=Path)
    parser.add_argument('--recipe-json', type=Path, default=None)
    parser.add_argument('--profile', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args.train_log, args.recipe_json, args.profile)
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))


if __name__ == '__main__':
    main()
