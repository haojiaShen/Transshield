import argparse
import json
import re
from pathlib import Path


LINK_RE = re.compile(
    r'(?P<label>Link details|ColocatedIo sync link details): total send bytes '
    r'(?P<send_bytes>\d+), recv bytes (?P<recv_bytes>\d+), '
    r'send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)'
)
OP_RE = re.compile(
    r'- (?P<name>[A-Za-z0-9_]+), executed (?P<count>\d+) times, duration (?P<duration>[0-9.eE+-]+)s'
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def safe_load_json(path: Path):
    return load_json(path) if path.exists() else None


def parse_link_metrics(log_path: Path):
    if not log_path.exists():
        return None
    matches = []
    nonzero_matches = []
    top_ops = {}
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        link_match = LINK_RE.search(line)
        if link_match:
            item = {
                'label': link_match.group('label'),
                'send_bytes': int(link_match.group('send_bytes')),
                'recv_bytes': int(link_match.group('recv_bytes')),
                'send_actions': int(link_match.group('send_actions')),
                'recv_actions': int(link_match.group('recv_actions')),
            }
            matches.append(item)
            if any(item[key] > 0 for key in ('send_bytes', 'recv_bytes', 'send_actions', 'recv_actions')):
                nonzero_matches.append(item)
        op_match = OP_RE.search(line)
        if op_match:
            name = op_match.group('name')
            item = {
                'count': int(op_match.group('count')),
                'duration_sec': float(op_match.group('duration')),
            }
            previous = top_ops.get(name)
            if previous is None or item['duration_sec'] > previous['duration_sec']:
                top_ops[name] = item

    latest = matches[-1] if matches else None
    top_ops_list = [
        {'name': name, **payload}
        for name, payload in sorted(top_ops.items(), key=lambda item: item[1]['duration_sec'], reverse=True)[:12]
    ]
    return {
        'log_path': str(log_path),
        'latest_link_details': latest,
        'latest_nonzero_link_details': nonzero_matches[-1] if nonzero_matches else None,
        'link_detail_count': len(matches),
        'nonzero_link_detail_count': len(nonzero_matches),
        'has_profile_ops': bool(top_ops),
        'latest_link_details_is_zero': bool(
            latest is not None
            and latest['send_bytes'] == 0
            and latest['recv_bytes'] == 0
            and latest['send_actions'] == 0
            and latest['recv_actions'] == 0
        ),
        'top_ops_by_duration': top_ops_list,
    }


def compact_fastpath_profile(profile):
    if not isinstance(profile, dict):
        return None
    return {
        'source': 'python_distributed_rpc_cloudpickle',
        'status': 'available' if int(profile.get('rpc_total_bytes') or 0) > 0 else 'missing',
        'rpc_request_total_bytes': profile.get('rpc_request_total_bytes'),
        'rpc_response_total_bytes': profile.get('rpc_response_total_bytes'),
        'rpc_total_bytes': profile.get('rpc_total_bytes'),
        'make_shares_total_input_bytes': profile.get('make_shares_total_input_bytes'),
        'matched_line_count': profile.get('matched_line_count'),
        'link_details_all_zero': profile.get('link_details_all_zero'),
        'top_rpc_requests': (profile.get('rpc_requests_by_fn_peer') or [])[:8],
        'top_rpc_responses': (profile.get('rpc_responses_by_fn_peer') or [])[:8],
        'fetch_objects': profile.get('fetch_objects') or [],
        'diagnosis': profile.get('diagnosis'),
    }


def safe_ratio(numerator, denominator):
    if numerator in (None, '') or denominator in (None, '', 0):
        return None
    return float(numerator) / float(denominator)


def build_payload_profile(kth_candidate_summary, fastpath_profile, pipeline_run_summary=None):
    if not isinstance(kth_candidate_summary, dict):
        return None
    payload = kth_candidate_summary.get('payload_estimate')
    if not isinstance(payload, dict):
        return None
    compact_bytes = payload.get('compact_masked_score_float32_bytes')
    dense_bytes = payload.get('dense_masked_score_float32_bytes')
    mixed_transport_total = payload.get('mixed_transport_total_bytes')
    rpc_total = fastpath_profile.get('rpc_total_bytes') if isinstance(fastpath_profile, dict) else None
    rpc_request = fastpath_profile.get('rpc_request_total_bytes') if isinstance(fastpath_profile, dict) else None
    rpc_response = fastpath_profile.get('rpc_response_total_bytes') if isinstance(fastpath_profile, dict) else None
    make_shares = fastpath_profile.get('make_shares_total_input_bytes') if isinstance(fastpath_profile, dict) else None
    payload_boundary_window = (
        pipeline_run_summary.get('payload_boundary_window')
        if isinstance(pipeline_run_summary, dict)
        else None
    )
    if payload_boundary_window is None:
        payload_boundary_window = payload.get('payload_boundary_window')

    diagnosis = (
        'Current run enables mixed payload transport before exact float32 reconstruction on P1. '
        'If RPC bytes do not fall roughly with mixed_transport_total_bytes, the remaining overhead is dominated '
        'by RPC/share framing or downstream fetch overhead rather than masked_score payload size itself.'
        if mixed_transport_total not in (None, '') and compact_bytes not in (None, '', 0)
        else 'Current selection-mode changes only the in-SPU kth schedule. '
        'If compact payload bytes are unchanged, RPC bytes will usually stay flat until the payload representation '
        'or RPC/share framing itself is reduced.'
    )
    return {
        'stage_count': payload.get('stage_count'),
        'dense_masked_score_float32_bytes': dense_bytes,
        'compact_masked_score_float32_bytes': compact_bytes,
        'saved_float32_bytes': payload.get('saved_float32_bytes'),
        'compact_ratio': payload.get('compact_ratio'),
        'payload_default_dtype': (
            (pipeline_run_summary.get('payload_dtype') if isinstance(pipeline_run_summary, dict) else None)
            or payload.get('payload_default_dtype')
        ),
        'payload_stage_dtypes': (
            (pipeline_run_summary.get('payload_stage_dtypes') if isinstance(pipeline_run_summary, dict) else None)
            or ''
        ),
        'payload_boundary_window': payload_boundary_window,
        'mixed_transport_total_bytes': mixed_transport_total,
        'mixed_transport_ratio_vs_compact_float32': payload.get('mixed_transport_ratio_vs_compact_float32'),
        'mixed_transport_ratio_vs_dense_float32': safe_ratio(mixed_transport_total, dense_bytes),
        'mixed_base_half_total_bytes': payload.get('mixed_base_half_total_bytes'),
        'mixed_boundary_index_total_bytes': payload.get('mixed_boundary_index_total_bytes'),
        'mixed_boundary_value_total_bytes': payload.get('mixed_boundary_value_total_bytes'),
        'selection_mode_changes_secure_input_shape': payload.get('selection_mode_changes_secure_input_shape'),
        'selection_mode_scope': payload.get('selection_mode_scope'),
        'payload_transport_scope': payload.get('payload_transport_scope'),
        'rpc_total_over_compact_payload_ratio': safe_ratio(rpc_total, compact_bytes),
        'rpc_request_over_compact_payload_ratio': safe_ratio(rpc_request, compact_bytes),
        'rpc_response_over_compact_payload_ratio': safe_ratio(rpc_response, compact_bytes),
        'make_shares_over_compact_payload_ratio': safe_ratio(make_shares, compact_bytes),
        'rpc_total_over_mixed_transport_ratio': safe_ratio(rpc_total, mixed_transport_total),
        'rpc_request_over_mixed_transport_ratio': safe_ratio(rpc_request, mixed_transport_total),
        'rpc_response_over_mixed_transport_ratio': safe_ratio(rpc_response, mixed_transport_total),
        'make_shares_over_mixed_transport_ratio': safe_ratio(make_shares, mixed_transport_total),
        'rpc_total_over_dense_payload_ratio': safe_ratio(rpc_total, dense_bytes),
        'diagnosis': diagnosis,
    }


def build_step_profile(pipeline_run_summary, replay_summary, kth_candidate_summary, tie_candidate_summary):
    step_durations = {}
    if pipeline_run_summary is not None:
        for step in pipeline_run_summary.get('steps', []):
            if step.get('duration_sec') is not None:
                step_durations[step['name']] = float(step['duration_sec'])
    return {
        'durations_sec': step_durations,
        'total_pipeline_duration_sec': float(sum(step_durations.values())) if step_durations else None,
        'replay_duration_sec': replay_summary.get('command_duration_sec') if replay_summary is not None else None,
        'network_kth_bridge_elapsed_sec': kth_candidate_summary.get('elapsed_sec') if kth_candidate_summary else None,
        'tie_bridge_elapsed_sec': tie_candidate_summary.get('elapsed_sec') if tie_candidate_summary else None,
    }


def collect_node_summaries(spu_log_dir: Path):
    if not spu_log_dir.exists():
        return []
    node_summaries = []
    for log_path in sorted(spu_log_dir.glob('node_*.log')):
        parsed = parse_link_metrics(log_path)
        if parsed is not None:
            node_summaries.append(parsed)
    return node_summaries


def summarize_link_observations(node_summaries):
    latest_metrics = [item['latest_link_details'] for item in node_summaries if item['latest_link_details'] is not None]
    nonzero_metric_counts = sum(int(item.get('nonzero_link_detail_count', 0)) for item in node_summaries)
    total_metric_counts = sum(int(item.get('link_detail_count', 0)) for item in node_summaries)
    nonzero_latest_metrics = [
        item
        for item in (summary.get('latest_nonzero_link_details') for summary in node_summaries)
        if item is not None
    ]
    return {
        'latest_metrics': latest_metrics,
        'nonzero_metric_counts': nonzero_metric_counts,
        'total_metric_counts': total_metric_counts,
        'nonzero_latest_metrics': nonzero_latest_metrics,
        'has_profile_ops': any(item.get('has_profile_ops') for item in node_summaries),
    }


def aggregate_link_metrics(nonzero_latest_metrics):
    if not nonzero_latest_metrics:
        return None
    return {
        'max_send_bytes': max(item['send_bytes'] for item in nonzero_latest_metrics),
        'max_recv_bytes': max(item['recv_bytes'] for item in nonzero_latest_metrics),
        'max_total_bytes': max(item['send_bytes'] + item['recv_bytes'] for item in nonzero_latest_metrics),
        'sum_send_bytes': sum(item['send_bytes'] for item in nonzero_latest_metrics),
        'sum_recv_bytes': sum(item['recv_bytes'] for item in nonzero_latest_metrics),
        'sum_total_bytes': sum(item['send_bytes'] + item['recv_bytes'] for item in nonzero_latest_metrics),
    }


def has_available_fastpath_profile(fastpath_profile):
    return (
        fastpath_profile is not None
        and fastpath_profile.get('status') == 'available'
        and int(fastpath_profile.get('rpc_total_bytes') or 0) > 0
    )


def build_communication_assessment(is_spu_runtime, link_stats, disable_colocated_optimization, has_python_fastpath_profile):
    total_metric_counts = link_stats['total_metric_counts']
    nonzero_metric_counts = link_stats['nonzero_metric_counts']
    latest_metrics = link_stats['latest_metrics']
    has_profile_ops = link_stats['has_profile_ops']

    communication_status = 'unsupported' if not is_spu_runtime else 'missing'
    communication_warning = None
    communication_diagnosis = None
    suspected_colocated_optimization_blind_spot = False

    if link_stats['nonzero_latest_metrics']:
        communication_status = 'available'
    elif (
        is_spu_runtime
        and total_metric_counts > 0
        and nonzero_metric_counts == 0
        and disable_colocated_optimization is False
    ):
        communication_status = 'colocated_private_path_no_link_counters'
        communication_warning = (
            'Link details counters remain zero under the colocated/private fast path; '
            'treat link-byte metrics as not applicable for this runtime rather than as true zero traffic.'
        )
    elif is_spu_runtime and has_profile_ops:
        communication_status = 'unreliable_zero_counters'
        communication_warning = (
            'SPU op profiling is present, but Link details counters are zero or missing; '
            'treat communication bytes as unavailable rather than true zero traffic.'
        )
    elif is_spu_runtime and latest_metrics:
        communication_status = 'zero_counters_no_ops'
        communication_warning = (
            'SPU node logs emitted Link details only with zero counters; '
            'communication bytes may be unavailable for this runtime/logging configuration.'
        )

    if is_spu_runtime:
        if total_metric_counts > 0 and nonzero_metric_counts == 0:
            if disable_colocated_optimization is False:
                suspected_colocated_optimization_blind_spot = True
                communication_diagnosis = (
                    'Link details lines are present, but every observed counter is zero while '
                    'experimental_enable_colocated_optimization remains enabled; this strongly suggests '
                    'the current fast runtime is taking a colocated/private execution path whose '
                    'traffic is not exposed through the inspected link-byte counters.'
                )
            else:
                communication_diagnosis = (
                    'Link details lines are present, but every observed counter is zero; '
                    'this indicates the current runtime/logging path is not exposing byte counters.'
                )
        elif total_metric_counts == 0 and has_profile_ops:
            communication_diagnosis = (
                'Op profiling exists, but no Link details lines were emitted in the inspected node logs.'
            )

    if is_spu_runtime and has_python_fastpath_profile:
        communication_status = (
            'available_python_fastpath'
            if communication_status in (
                'colocated_private_path_no_link_counters',
                'unreliable_zero_counters',
                'zero_counters_no_ops',
                'missing',
            )
            else communication_status
        )
        communication_warning = (
            'C++ Link details counters remain zero, but Python distributed RPC/cloudpickle '
            'fastpath traffic is available and should be used for the default fast runtime display.'
        )
        fastpath_diagnosis = (
            'Python fastpath RPC/cloudpickle traffic is nonzero while C++ Link details remain zero; '
            'default fast runtime communication is visible at the Python distributed layer, not in the inspected yacl link counters.'
        )
        communication_diagnosis = (
            f'{communication_diagnosis} {fastpath_diagnosis}'
            if communication_diagnosis
            else fastpath_diagnosis
        )

    return {
        'status': communication_status,
        'warning': communication_warning,
        'diagnosis': communication_diagnosis,
        'suspected_colocated_optimization_blind_spot': suspected_colocated_optimization_blind_spot,
    }


def build_communication_note(is_spu_runtime, has_python_fastpath_profile):
    if has_python_fastpath_profile:
        return (
            'Primary communication display uses Python distributed RPC/cloudpickle fastpath metrics; '
            'C++ LinkDetails remain available only as diagnostic counters.'
        )
    if is_spu_runtime:
        return 'Communication metrics are collected from SPU node logs.'
    return 'CPU reference runtime does not emit SPU communication logs; byte metrics are intentionally omitted.'


def build_aggregate_python_fastpath_metrics(fastpath_profile, has_python_fastpath_profile):
    if not has_python_fastpath_profile:
        return None
    return {
        'rpc_request_total_bytes': fastpath_profile.get('rpc_request_total_bytes'),
        'rpc_response_total_bytes': fastpath_profile.get('rpc_response_total_bytes'),
        'rpc_total_bytes': fastpath_profile.get('rpc_total_bytes'),
        'make_shares_total_input_bytes': fastpath_profile.get('make_shares_total_input_bytes'),
        'source': fastpath_profile.get('source'),
    }


def build_communication_profile(
    is_spu_runtime,
    link_stats,
    aggregate_link,
    fastpath_profile,
    fastpath_profile_raw,
    fastpath_profile_path,
    disable_colocated_optimization,
    node_summaries,
):
    has_python_fastpath_profile = has_available_fastpath_profile(fastpath_profile)
    assessment = build_communication_assessment(
        is_spu_runtime,
        link_stats,
        disable_colocated_optimization,
        has_python_fastpath_profile,
    )
    return {
        'supported': is_spu_runtime,
        'status': assessment['status'],
        'diagnosis': assessment['diagnosis'],
        'note': build_communication_note(is_spu_runtime, has_python_fastpath_profile),
        'warning': assessment['warning'],
        'link_detail_count': link_stats['total_metric_counts'],
        'nonzero_link_detail_count': link_stats['nonzero_metric_counts'],
        'aggregate_link_metrics': aggregate_link,
        'aggregate_python_fastpath_metrics': build_aggregate_python_fastpath_metrics(
            fastpath_profile,
            has_python_fastpath_profile,
        ),
        'python_fastpath_profile': fastpath_profile,
        'python_fastpath_profile_json': str(fastpath_profile_path) if fastpath_profile_raw is not None else None,
        'disable_colocated_optimization': disable_colocated_optimization,
        'suspected_colocated_optimization_blind_spot': assessment['suspected_colocated_optimization_blind_spot'],
        'node_logs': node_summaries,
    }


def build_parser():
    parser = argparse.ArgumentParser(description='Summarize time/communication/profile artifacts for a secure BumbleBee/SPU run.')
    parser.add_argument('--secure-run-dir', required=True)
    parser.add_argument('--spu-state-json', default='logs/spu_runtime_ports.json')
    parser.add_argument('--spu-log-dir', default='logs/spu_nodes')
    parser.add_argument(
        '--fastpath-profile-json',
        default='',
        help='Optional fastpath_profile_summary.json from tools/transshield_fastpath_profile_summary.py; defaults to <secure-run-dir>/fastpath_profile_summary.json if present.',
    )
    parser.add_argument('--output-json', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    secure_run_dir = Path(args.secure_run_dir).resolve()
    fastpath_profile_path = (
        Path(args.fastpath_profile_json).resolve()
        if args.fastpath_profile_json
        else secure_run_dir / 'fastpath_profile_summary.json'
    )
    fastpath_profile_raw = safe_load_json(fastpath_profile_path)
    fastpath_profile = compact_fastpath_profile(fastpath_profile_raw)
    pipeline_run_summary = safe_load_json(secure_run_dir / 'pipeline_run_summary.json')
    pipeline_verify_summary = safe_load_json(secure_run_dir / 'pipeline_verify_summary.json')
    replay_summary = safe_load_json(secure_run_dir / 'pipeline_inference_replay_summary.json')
    score_compare_summary = safe_load_json(secure_run_dir / 'plaintext_vs_secure_score_compare.json')
    kth_candidate_summary = safe_load_json(secure_run_dir / 'stage2_secure_network_kth_candidate_from_server.json')
    tie_candidate_summary = safe_load_json(secure_run_dir / 'stage2_secure_tie_candidate_from_server.json')

    runtime = pipeline_run_summary.get('runtime') if pipeline_run_summary is not None else None
    is_spu_runtime = runtime == 'spu'
    spu_state = safe_load_json(Path(args.spu_state_json).resolve()) if is_spu_runtime else None
    disable_colocated_optimization = (
        spu_state.get('disable_colocated_optimization')
        if isinstance(spu_state, dict)
        else None
    )

    node_summaries = collect_node_summaries(Path(args.spu_log_dir).resolve()) if is_spu_runtime else []
    link_stats = summarize_link_observations(node_summaries)
    output = {
        'secure_run_dir': str(secure_run_dir),
        'runtime': runtime,
        'overall_passed': (
            bool(pipeline_verify_summary.get('overall_passed')) if pipeline_verify_summary is not None else None
        ),
        'replay_overall_passed': (
            bool(replay_summary.get('overall_passed')) if replay_summary is not None else None
        ),
        'step_profile': build_step_profile(
            pipeline_run_summary,
            replay_summary,
            kth_candidate_summary,
            tie_candidate_summary,
        ),
        'communication_profile': build_communication_profile(
            is_spu_runtime,
            link_stats,
            aggregate_link_metrics(link_stats['nonzero_latest_metrics']),
            fastpath_profile,
            fastpath_profile_raw,
            fastpath_profile_path,
            disable_colocated_optimization,
            node_summaries,
        ),
        'spu_runtime': {
            'state_json': str(Path(args.spu_state_json).resolve()),
            'state': spu_state,
        },
        'payload_profile': build_payload_profile(
            kth_candidate_summary,
            fastpath_profile,
            pipeline_run_summary,
        ),
        'score_compare': score_compare_summary.get('comparison') if score_compare_summary is not None else None,
    }
    write_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
