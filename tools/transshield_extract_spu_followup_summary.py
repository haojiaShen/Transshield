#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def fmt(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def extract_summary(run_dir: Path):
    verify = load_json(run_dir / 'pipeline_verify_summary.json')
    replay = load_json(run_dir / 'pipeline_inference_replay_summary.json')
    compare = load_json(run_dir / 'plaintext_vs_secure_score_compare.json')
    profile = load_json(run_dir / 'secure_profile_summary.json')
    communication = profile.get('communication_profile') or {}
    aggregate = communication.get('aggregate_link_metrics')
    fastpath = communication.get('aggregate_python_fastpath_metrics') or {}

    return {
        'run_dir': str(run_dir),
        'pipeline_verify_overall_passed': verify.get('overall_passed'),
        'replay_overall_passed': replay.get('overall_passed'),
        'sample_count': compare.get('sample_count'),
        'secure_overall_passed': compare.get('source_status', {}).get('secure_overall_passed'),
        'secure_model_replay_status': compare.get('source_status', {}).get('secure_model_replay_status'),
        'argmax_match_ratio': compare.get('comparison', {}).get('argmax_predictions', {}).get('match_ratio'),
        'threshold_match_ratio': compare.get('comparison', {}).get('threshold_predictions', {}).get('match_ratio'),
        'plaintext_argmax_accuracy': compare.get('comparison', {}).get('plaintext_argmax_accuracy'),
        'secure_argmax_accuracy': compare.get('comparison', {}).get('secure_argmax_accuracy'),
        'plaintext_threshold_accuracy': compare.get('comparison', {}).get('plaintext_threshold_accuracy'),
        'secure_threshold_accuracy': compare.get('comparison', {}).get('secure_threshold_accuracy'),
        'logits_max_abs_error': compare.get('comparison', {}).get('logits', {}).get('max_abs_error'),
        'probabilities_max_abs_error': compare.get('comparison', {}).get('probabilities', {}).get('max_abs_error'),
        'runtime': profile.get('runtime'),
        'communication_status': communication.get('status'),
        'diagnosis': communication.get('diagnosis'),
        'link_detail_count': communication.get('link_detail_count'),
        'nonzero_link_detail_count': communication.get('nonzero_link_detail_count'),
        'aggregate_link_metrics': aggregate,
        'aggregate_python_fastpath_metrics': fastpath,
        'python_fastpath_rpc_total_bytes': fastpath.get('rpc_total_bytes'),
        'python_fastpath_rpc_request_total_bytes': fastpath.get('rpc_request_total_bytes'),
        'python_fastpath_rpc_response_total_bytes': fastpath.get('rpc_response_total_bytes'),
        'python_fastpath_make_shares_input_bytes': fastpath.get('make_shares_total_input_bytes'),
        'total_pipeline_duration_sec': profile.get('step_profile', {}).get('total_pipeline_duration_sec'),
        'replay_duration_sec': profile.get('step_profile', {}).get('replay_duration_sec'),
        'network_kth_bridge_elapsed_sec': profile.get('step_profile', {}).get('network_kth_bridge_elapsed_sec'),
        'tie_bridge_elapsed_sec': profile.get('step_profile', {}).get('tie_bridge_elapsed_sec'),
    }


def render_markdown(summary, title: str):
    agg = summary.get('aggregate_link_metrics') or {}
    fastpath = summary.get('aggregate_python_fastpath_metrics') or {}
    has_fastpath = fastpath.get('rpc_total_bytes') is not None
    lines = [
        f'# {title}',
        '',
        f"- run dir: `{summary.get('run_dir')}`",
        f"- runtime: `{summary.get('runtime')}`",
        f"- communication status: `{summary.get('communication_status')}`",
        '',
        '## Correctness',
        '',
        f"- pipeline verify passed: `{fmt(summary.get('pipeline_verify_overall_passed'))}`",
        f"- replay overall passed: `{fmt(summary.get('replay_overall_passed'))}`",
        f"- secure model replay status: `{summary.get('secure_model_replay_status')}`",
        f"- argmax match ratio: `{fmt(summary.get('argmax_match_ratio'))}`",
        f"- threshold match ratio: `{fmt(summary.get('threshold_match_ratio'))}`",
        '',
        '## Runtime',
        '',
        f"- total pipeline duration: `{fmt(summary.get('total_pipeline_duration_sec'))}s`",
        f"- replay duration: `{fmt(summary.get('replay_duration_sec'))}s`",
        f"- network_kth bridge duration: `{fmt(summary.get('network_kth_bridge_elapsed_sec'))}s`",
        f"- tie bridge duration: `{fmt(summary.get('tie_bridge_elapsed_sec'))}s`",
        '',
        '## Communication',
        '',
    ]
    if has_fastpath:
        lines.extend([
            '- communication source: `Python distributed RPC/cloudpickle fastpath`',
            f"- Python fastpath RPC total bytes: `{fmt(fastpath.get('rpc_total_bytes'), 0)}`",
            f"- Python fastpath RPC request bytes: `{fmt(fastpath.get('rpc_request_total_bytes'), 0)}`",
            f"- Python fastpath RPC response bytes: `{fmt(fastpath.get('rpc_response_total_bytes'), 0)}`",
            f"- Python fastpath make_shares input bytes: `{fmt(fastpath.get('make_shares_total_input_bytes'), 0)}`",
            '- C++ LinkDetails counters: `not shown in primary summary; not applicable to this fastpath communication layer`',
        ])
    else:
        lines.extend([
            '- communication source: `C++ yacl LinkDetails`',
            f"- max total bytes: `{fmt(agg.get('max_total_bytes'), 0)}`",
            f"- max send bytes: `{fmt(agg.get('max_send_bytes'), 0)}`",
            f"- max recv bytes: `{fmt(agg.get('max_recv_bytes'), 0)}`",
        ])
    if summary.get('diagnosis'):
        lines.extend(['', '## Diagnosis', '', f"- {summary.get('diagnosis')}"])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Extract a compact SPU follow-up summary from a secure run dir.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--title', default='SPU Follow-Up Summary')
    args = parser.parse_args()

    summary = extract_summary(Path(args.run_dir).resolve())
    write_text(Path(args.output_json).resolve(), json.dumps(summary, indent=2, sort_keys=True) + '\n')
    write_text(Path(args.output_md).resolve(), render_markdown(summary, args.title))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
