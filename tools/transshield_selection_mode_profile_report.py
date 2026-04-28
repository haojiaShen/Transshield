#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SCALAR_COMPARE_FIELDS = [
    'total_pipeline_duration_sec',
    'network_kth_bridge_elapsed_sec',
    'tie_bridge_elapsed_sec',
    'communication_total_bytes',
    'python_fastpath_rpc_request_total_bytes',
    'python_fastpath_rpc_response_total_bytes',
    'python_fastpath_make_shares_total_input_bytes',
    'payload_dense_masked_score_float32_bytes',
    'payload_compact_masked_score_float32_bytes',
    'payload_saved_float32_bytes',
    'payload_mixed_transport_total_bytes',
    'payload_mixed_base_half_total_bytes',
    'payload_mixed_boundary_index_total_bytes',
    'payload_mixed_boundary_value_total_bytes',
    'comparison_argmax_match_ratio',
    'comparison_threshold_match_ratio',
    'plaintext_argmax_accuracy',
    'secure_argmax_accuracy',
    'plaintext_threshold_accuracy',
    'secure_threshold_accuracy',
]

KEY_METRIC_ROWS = [
    ('total pipeline duration', 'total_pipeline_duration_sec', 'seconds'),
    ('network_kth bridge', 'network_kth_bridge_elapsed_sec', 'seconds'),
    ('tie bridge', 'tie_bridge_elapsed_sec', 'seconds'),
    ('communication total bytes', 'communication_total_bytes', 'bytes'),
]

PAYLOAD_METRIC_ROWS = [
    ('dense masked_score bytes into SPU', 'payload_dense_masked_score_float32_bytes'),
    ('compact masked_score bytes into SPU', 'payload_compact_masked_score_float32_bytes'),
    ('bytes saved by active compaction', 'payload_saved_float32_bytes'),
    ('mixed transport bytes into P1', 'payload_mixed_transport_total_bytes'),
    ('mixed base-half bytes', 'payload_mixed_base_half_total_bytes'),
    ('mixed boundary index bytes', 'payload_mixed_boundary_index_total_bytes'),
    ('mixed boundary value bytes', 'payload_mixed_boundary_value_total_bytes'),
]

ACCURACY_METRIC_ROWS = [
    ('plaintext argmax accuracy', 'plaintext_argmax_accuracy'),
    ('secure argmax accuracy', 'secure_argmax_accuracy'),
    ('plaintext threshold accuracy', 'plaintext_threshold_accuracy'),
    ('secure threshold accuracy', 'secure_threshold_accuracy'),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def maybe_load_json(path: Path):
    return load_json(path) if path.exists() else None


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def scalar_delta(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return float(rhs - lhs)


def scalar_ratio(lhs, rhs):
    if lhs in (None, 0) or rhs is None:
        return None
    return float(rhs / lhs)


def human_seconds(value):
    if value is None:
        return 'N/A'
    return f'{float(value):.4f}s'


def human_ratio(value):
    if value is None:
        return 'N/A'
    return f'{float(value):.3f}x'


def human_float(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def human_bytes(value):
    if value is None:
        return 'N/A'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(value)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f'{size:.2f} {units[unit_index]}'


def human_payload_config(summary):
    if not isinstance(summary, dict):
        return 'N/A'
    stage_overrides = summary.get('payload_stage_dtypes') or 'none'
    boundary_window = summary.get('payload_boundary_window')
    return (
        f"default={summary.get('payload_default_dtype') or 'float32'}, "
        f"stage_overrides={stage_overrides}, "
        f"boundary_window={boundary_window if boundary_window is not None else 0}"
    )


def first_truthy(*values, default=None):
    for value in values:
        if value:
            return value
    return default


def first_non_none(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def load_run_artifacts(run_dir: Path):
    return {
        'profile': load_json(run_dir / 'secure_profile_summary.json'),
        'verify': maybe_load_json(run_dir / 'pipeline_verify_summary.json') or {},
        'compare': maybe_load_json(run_dir / 'plaintext_vs_secure_score_compare.json') or {},
        'pipeline_run': maybe_load_json(run_dir / 'pipeline_run_summary.json') or {},
        'kth_candidate': maybe_load_json(run_dir / 'stage2_secure_network_kth_candidate_from_server.json') or {},
    }


def extract_communication_summary(profile):
    communication = profile.get('communication_profile') or {}
    fastpath = communication.get('aggregate_python_fastpath_metrics') or {}
    aggregate_link = communication.get('aggregate_link_metrics') or {}
    communication_source = (
        'python_distributed_rpc_cloudpickle'
        if fastpath.get('rpc_total_bytes') is not None
        else 'cpp_yacl_linkdetails'
    )
    communication_total_bytes = (
        fastpath.get('rpc_total_bytes')
        if communication_source == 'python_distributed_rpc_cloudpickle'
        else aggregate_link.get('sum_total_bytes')
    )
    return {
        'communication_status': communication.get('status'),
        'communication_source': communication_source,
        'communication_total_bytes': communication_total_bytes,
        'python_fastpath_rpc_request_total_bytes': fastpath.get('rpc_request_total_bytes'),
        'python_fastpath_rpc_response_total_bytes': fastpath.get('rpc_response_total_bytes'),
        'python_fastpath_make_shares_total_input_bytes': fastpath.get('make_shares_total_input_bytes'),
        'link_sum_total_bytes': aggregate_link.get('sum_total_bytes'),
    }


def extract_payload_summary(pipeline_run, payload_profile, payload_estimate):
    return {
        'payload_default_dtype': first_truthy(
            pipeline_run.get('payload_dtype'),
            payload_profile.get('payload_default_dtype'),
            payload_estimate.get('payload_default_dtype'),
        ),
        'payload_stage_dtypes': first_truthy(
            pipeline_run.get('payload_stage_dtypes'),
            payload_profile.get('payload_stage_dtypes'),
            default='',
        ),
        'payload_boundary_window': first_non_none(
            pipeline_run.get('payload_boundary_window'),
            payload_profile.get('payload_boundary_window'),
            payload_estimate.get('payload_boundary_window'),
        ),
        'payload_dense_masked_score_float32_bytes': payload_estimate.get('dense_masked_score_float32_bytes'),
        'payload_compact_masked_score_float32_bytes': payload_estimate.get('compact_masked_score_float32_bytes'),
        'payload_saved_float32_bytes': payload_estimate.get('saved_float32_bytes'),
        'payload_compact_ratio': payload_estimate.get('compact_ratio'),
        'payload_stage_count': payload_estimate.get('stage_count'),
        'payload_mixed_transport_total_bytes': first_truthy(
            payload_profile.get('mixed_transport_total_bytes'),
            payload_estimate.get('mixed_transport_total_bytes'),
        ),
        'payload_mixed_transport_ratio_vs_compact_float32': first_truthy(
            payload_profile.get('mixed_transport_ratio_vs_compact_float32'),
            payload_estimate.get('mixed_transport_ratio_vs_compact_float32'),
        ),
        'payload_mixed_base_half_total_bytes': first_truthy(
            payload_profile.get('mixed_base_half_total_bytes'),
            payload_estimate.get('mixed_base_half_total_bytes'),
        ),
        'payload_mixed_boundary_index_total_bytes': first_truthy(
            payload_profile.get('mixed_boundary_index_total_bytes'),
            payload_estimate.get('mixed_boundary_index_total_bytes'),
        ),
        'payload_mixed_boundary_value_total_bytes': first_truthy(
            payload_profile.get('mixed_boundary_value_total_bytes'),
            payload_estimate.get('mixed_boundary_value_total_bytes'),
        ),
        'selection_mode_changes_secure_input_shape': payload_estimate.get('selection_mode_changes_secure_input_shape'),
        'selection_mode_scope': payload_estimate.get('selection_mode_scope'),
        'payload_transport_scope': first_truthy(
            payload_profile.get('payload_transport_scope'),
            payload_estimate.get('payload_transport_scope'),
        ),
        'payload_rpc_total_over_compact_ratio': payload_profile.get('rpc_total_over_compact_payload_ratio'),
        'payload_make_shares_over_compact_ratio': payload_profile.get('make_shares_over_compact_payload_ratio'),
        'payload_rpc_total_over_mixed_ratio': payload_profile.get('rpc_total_over_mixed_transport_ratio'),
        'payload_make_shares_over_mixed_ratio': payload_profile.get('make_shares_over_mixed_transport_ratio'),
    }


def extract_compare_summary(compare):
    comparison = compare.get('comparison', {})
    argmax = comparison.get('argmax_predictions', {})
    threshold = comparison.get('threshold_predictions', {})
    return {
        'sample_count': compare.get('sample_count'),
        'comparison_argmax_match_ratio': argmax.get('match_ratio'),
        'comparison_threshold_match_ratio': threshold.get('match_ratio'),
        'plaintext_argmax_accuracy': comparison.get('plaintext_argmax_accuracy'),
        'secure_argmax_accuracy': comparison.get('secure_argmax_accuracy'),
        'plaintext_threshold_accuracy': comparison.get('plaintext_threshold_accuracy'),
        'secure_threshold_accuracy': comparison.get('secure_threshold_accuracy'),
    }


def extract_run_summary(run_dir: Path, label: str):
    artifacts = load_run_artifacts(run_dir)
    profile = artifacts['profile']
    verify = artifacts['verify']
    pipeline_run = artifacts['pipeline_run']
    kth_candidate = artifacts['kth_candidate']
    payload_profile = (profile.get('payload_profile') or {})
    step_profile = (profile.get('step_profile') or {})

    return {
        'label': label,
        'run_dir': str(run_dir),
        'runtime': profile.get('runtime'),
        'selection_mode': pipeline_run.get('selection_mode'),
        'overall_passed': profile.get('overall_passed'),
        'pipeline_verify_overall_passed': verify.get('overall_passed'),
        'total_pipeline_duration_sec': step_profile.get('total_pipeline_duration_sec'),
        'network_kth_bridge_elapsed_sec': step_profile.get('network_kth_bridge_elapsed_sec'),
        'tie_bridge_elapsed_sec': step_profile.get('tie_bridge_elapsed_sec'),
        'durations_sec': step_profile.get('durations_sec') or {},
        **extract_communication_summary(profile),
        **extract_payload_summary(
            pipeline_run,
            payload_profile,
            kth_candidate.get('payload_estimate') or {},
        ),
        **extract_compare_summary(artifacts['compare']),
    }


def build_scalar_compare_entry(left, right):
    return {
        'a': left,
        'b': right,
        'delta_b_minus_a': scalar_delta(left, right),
        'ratio_b_over_a': scalar_ratio(left, right),
    }


def compare_scalar_fields(summary_a, summary_b, field_names):
    return {
        field_name: build_scalar_compare_entry(summary_a.get(field_name), summary_b.get(field_name))
        for field_name in field_names
    }


def build_step_compare(summary_a, summary_b):
    step_names = sorted(set(summary_a.get('durations_sec', {})) | set(summary_b.get('durations_sec', {})))
    return {
        step_name: build_scalar_compare_entry(
            summary_a.get('durations_sec', {}).get(step_name),
            summary_b.get('durations_sec', {}).get(step_name),
        )
        for step_name in step_names
    }


def build_output(summary_a, summary_b):
    return {
        'summary_a': summary_a,
        'summary_b': summary_b,
        'scalar_compare': compare_scalar_fields(summary_a, summary_b, SCALAR_COMPARE_FIELDS),
        'step_compare': build_step_compare(summary_a, summary_b),
    }


def markdown_metric_row(label, payload, value_format):
    formatters = {
        'seconds': human_seconds,
        'bytes': human_bytes,
        'float': human_float,
    }
    formatter = formatters[value_format]
    return (
        f"| {label} | {formatter(payload['a'])} | {formatter(payload['b'])} | "
        f"{formatter(payload['delta_b_minus_a'])} | {human_ratio(payload['ratio_b_over_a'])} |"
    )


def markdown_accuracy_row(label, payload):
    return (
        f"| {label} | {human_float(payload['a'])} | {human_float(payload['b'])} | "
        f"{human_float(payload['delta_b_minus_a'])} |"
    )


def render_overview(summary_a, summary_b):
    return [
        '## Overview',
        '',
        f"- A label: `{summary_a['label']}`",
        f"- A run dir: `{summary_a['run_dir']}`",
        f"- A selection mode: `{summary_a.get('selection_mode') or summary_a['label']}`",
        f"- A payload config: `{human_payload_config(summary_a)}`",
        f"- A sample count: `{human_float(summary_a.get('sample_count'), 0)}`",
        f"- B label: `{summary_b['label']}`",
        f"- B run dir: `{summary_b['run_dir']}`",
        f"- B selection mode: `{summary_b.get('selection_mode') or summary_b['label']}`",
        f"- B payload config: `{human_payload_config(summary_b)}`",
        f"- B sample count: `{human_float(summary_b.get('sample_count'), 0)}`",
    ]


def render_correctness_guardrail(summary_a, summary_b):
    lines = [
        '## Correctness Guardrail',
        '',
        f"- A verify passed: `{human_float(summary_a.get('pipeline_verify_overall_passed'))}`",
        f"- B verify passed: `{human_float(summary_b.get('pipeline_verify_overall_passed'))}`",
    ]
    if summary_a.get('comparison_argmax_match_ratio') is None and summary_b.get('comparison_argmax_match_ratio') is None:
        return lines
    lines.extend(
        [
            f"- A argmax match ratio: `{human_float(summary_a.get('comparison_argmax_match_ratio'))}`",
            f"- B argmax match ratio: `{human_float(summary_b.get('comparison_argmax_match_ratio'))}`",
            f"- A threshold match ratio: `{human_float(summary_a.get('comparison_threshold_match_ratio'))}`",
            f"- B threshold match ratio: `{human_float(summary_b.get('comparison_threshold_match_ratio'))}`",
        ]
    )
    return lines


def render_table_section(title, rows, column_header='| Metric | A | B | Delta (B-A) | Ratio (B/A) |'):
    return [
        title,
        '',
        column_header,
        '|---|---:|---:|---:|---:|',
        *rows,
    ]


def render_key_metrics(scalar):
    rows = [markdown_metric_row(label, scalar[field_name], value_format) for label, field_name, value_format in KEY_METRIC_ROWS]
    return render_table_section('## Key Metrics', rows)


def should_render_payload_snapshot(scalar):
    return (
        scalar['payload_dense_masked_score_float32_bytes']['a'] is not None
        or scalar['payload_dense_masked_score_float32_bytes']['b'] is not None
    )


def render_payload_snapshot(scalar):
    rows = [markdown_metric_row(label, scalar[field_name], 'bytes') for label, field_name in PAYLOAD_METRIC_ROWS]
    return render_table_section('## Payload Snapshot', rows)


def should_render_accuracy_snapshot(scalar):
    return scalar['plaintext_argmax_accuracy']['a'] is not None or scalar['plaintext_argmax_accuracy']['b'] is not None


def render_accuracy_snapshot(scalar):
    rows = [markdown_accuracy_row(label, scalar[field_name]) for label, field_name in ACCURACY_METRIC_ROWS]
    return [
        '## Accuracy Snapshot',
        '',
        '| Metric | A | B | Delta (B-A) |',
        '|---|---:|---:|---:|',
        *rows,
    ]


def render_pipeline_steps(step_compare):
    lines = [
        '## Pipeline Steps',
        '',
        '| Step | A | B | Delta (B-A) | Ratio (B/A) |',
        '|---|---:|---:|---:|---:|',
    ]
    for step_name, payload in step_compare.items():
        lines.append(markdown_metric_row(step_name, payload, 'seconds'))
    return lines


def render_notes(summary_a, summary_b):
    lines = [
        '## Notes',
        '',
        f"- A communication source: `{summary_a['communication_source']}`",
        f"- B communication source: `{summary_b['communication_source']}`",
        (
            f"- A compact payload ratio: `{human_ratio(summary_a.get('payload_compact_ratio'))}` "
            f"(stage_count=`{human_float(summary_a.get('payload_stage_count'), 0)}`)"
        ),
        (
            f"- B compact payload ratio: `{human_ratio(summary_b.get('payload_compact_ratio'))}` "
            f"(stage_count=`{human_float(summary_b.get('payload_stage_count'), 0)}`)"
        ),
        f"- A RPC / compact payload ratio: `{human_ratio(summary_a.get('payload_rpc_total_over_compact_ratio'))}`",
        f"- B RPC / compact payload ratio: `{human_ratio(summary_b.get('payload_rpc_total_over_compact_ratio'))}`",
        f"- A make_shares / compact payload ratio: `{human_ratio(summary_a.get('payload_make_shares_over_compact_ratio'))}`",
        f"- B make_shares / compact payload ratio: `{human_ratio(summary_b.get('payload_make_shares_over_compact_ratio'))}`",
        f"- A mixed transport / compact float32 ratio: `{human_ratio(summary_a.get('payload_mixed_transport_ratio_vs_compact_float32'))}`",
        f"- B mixed transport / compact float32 ratio: `{human_ratio(summary_b.get('payload_mixed_transport_ratio_vs_compact_float32'))}`",
        f"- A RPC / mixed transport ratio: `{human_ratio(summary_a.get('payload_rpc_total_over_mixed_ratio'))}`",
        f"- B RPC / mixed transport ratio: `{human_ratio(summary_b.get('payload_rpc_total_over_mixed_ratio'))}`",
        f"- A make_shares / mixed transport ratio: `{human_ratio(summary_a.get('payload_make_shares_over_mixed_ratio'))}`",
        f"- B make_shares / mixed transport ratio: `{human_ratio(summary_b.get('payload_make_shares_over_mixed_ratio'))}`",
        '- When Python fastpath RPC bytes are available, they are the primary communication display.',
        '- This report only compares observed runtime/profile artifacts; it does not by itself prove protocol-level optimality.',
    ]
    if (
        summary_a.get('selection_mode_changes_secure_input_shape') is False
        and summary_b.get('selection_mode_changes_secure_input_shape') is False
        and summary_a.get('selection_mode_scope')
    ):
        lines.extend(
            [
                '',
                'Payload interpretation:',
                f"- {summary_a['selection_mode_scope']}",
            ]
        )
    if summary_a.get('payload_transport_scope') or summary_b.get('payload_transport_scope'):
        lines.extend(
            [
                '',
                'Transport interpretation:',
                f"- {summary_a.get('payload_transport_scope') or summary_b.get('payload_transport_scope')}",
            ]
        )
    return lines


def render_markdown(report):
    summary_a = report['summary_a']
    summary_b = report['summary_b']
    scalar = report['scalar_compare']
    lines = ['# Secure Pipeline Profile Compare', '']
    for section in (
        render_overview(summary_a, summary_b),
        render_correctness_guardrail(summary_a, summary_b),
        render_key_metrics(scalar),
    ):
        lines.extend(section)
        lines.append('')
    if should_render_payload_snapshot(scalar):
        lines.extend(render_payload_snapshot(scalar))
        lines.append('')
    if should_render_accuracy_snapshot(scalar):
        lines.extend(render_accuracy_snapshot(scalar))
        lines.append('')
    lines.extend(render_pipeline_steps(report['step_compare']))
    lines.append('')
    lines.extend(render_notes(summary_a, summary_b))
    lines.append('')
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Compare two secure profile runs across selection modes and/or payload schemes.'
    )
    parser.add_argument('--run-dir-a', required=True)
    parser.add_argument('--run-dir-b', required=True)
    parser.add_argument('--label-a', default='selection_a')
    parser.add_argument('--label-b', default='selection_b')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    summary_a = extract_run_summary(Path(args.run_dir_a).resolve(), args.label_a)
    summary_b = extract_run_summary(Path(args.run_dir_b).resolve(), args.label_b)
    report = build_output(summary_a, summary_b)
    write_json(Path(args.output_json).resolve(), report)
    write_text(Path(args.output_md).resolve(), render_markdown(report) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
