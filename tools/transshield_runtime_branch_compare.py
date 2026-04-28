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


def ratio(lhs, rhs):
    if lhs in (None, 0) or rhs is None:
        return None
    return float(rhs) / float(lhs)


def nested_get(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_summary(summary):
    return {
        'sample_count': summary.get('sample_count'),
        'argmax_match_ratio': summary.get('argmax_match_ratio'),
        'threshold_match_ratio': summary.get('threshold_match_ratio'),
        'secure_model_replay_status': summary.get('secure_model_replay_status'),
        'total_pipeline_duration_sec': summary.get('total_pipeline_duration_sec')
        or nested_get(summary, 'step_profile', 'total_pipeline_duration_sec'),
        'replay_duration_sec': summary.get('replay_duration_sec')
        or nested_get(summary, 'step_profile', 'replay_duration_sec'),
        'communication_status': summary.get('communication_status')
        or nested_get(summary, 'communication_profile', 'status'),
        'aggregate_link_metrics': summary.get('aggregate_link_metrics')
        or nested_get(summary, 'communication_profile', 'aggregate_link_metrics'),
    }


def build_correctness_compare(summary_a, summary_b):
    return {
        'a_argmax_match_ratio': summary_a.get('argmax_match_ratio'),
        'b_argmax_match_ratio': summary_b.get('argmax_match_ratio'),
        'a_threshold_match_ratio': summary_a.get('threshold_match_ratio'),
        'b_threshold_match_ratio': summary_b.get('threshold_match_ratio'),
        'a_secure_model_replay_status': summary_a.get('secure_model_replay_status'),
        'b_secure_model_replay_status': summary_b.get('secure_model_replay_status'),
    }


def build_runtime_compare(summary_a, summary_b):
    return {
        'a_total_pipeline_duration_sec': summary_a.get('total_pipeline_duration_sec'),
        'b_total_pipeline_duration_sec': summary_b.get('total_pipeline_duration_sec'),
        'b_over_a_total_ratio': ratio(summary_a.get('total_pipeline_duration_sec'), summary_b.get('total_pipeline_duration_sec')),
        'a_replay_duration_sec': summary_a.get('replay_duration_sec'),
        'b_replay_duration_sec': summary_b.get('replay_duration_sec'),
        'b_over_a_replay_ratio': ratio(summary_a.get('replay_duration_sec'), summary_b.get('replay_duration_sec')),
    }


def build_communication_compare(summary_a, summary_b):
    return {
        'a_status': summary_a.get('communication_status'),
        'b_status': summary_b.get('communication_status'),
        'a_aggregate_link_metrics': summary_a.get('aggregate_link_metrics'),
        'b_aggregate_link_metrics': summary_b.get('aggregate_link_metrics'),
    }


def build_interpretation(summary_b, label_a, label_b):
    return {
        'preferred_fast_runtime': label_a,
        'preferred_comm_runtime': label_b if summary_b.get('communication_status') == 'available' else None,
    }


def build_compare(summary_a, summary_b, label_a, label_b):
    normalized_a = normalize_summary(summary_a)
    normalized_b = normalize_summary(summary_b)
    return {
        'label_a': label_a,
        'label_b': label_b,
        'sample_count_match': normalized_a.get('sample_count') == normalized_b.get('sample_count'),
        'correctness_match': build_correctness_compare(normalized_a, normalized_b),
        'runtime_compare': build_runtime_compare(normalized_a, normalized_b),
        'communication_compare': build_communication_compare(normalized_a, normalized_b),
        'interpretation': build_interpretation(normalized_b, label_a, label_b),
    }


def render_correctness(report):
    correctness = report['correctness_match']
    return [
        '## Correctness',
        '',
        f"- argmax match ratio: `{fmt(correctness['a_argmax_match_ratio'])}` vs `{fmt(correctness['b_argmax_match_ratio'])}`",
        f"- threshold match ratio: `{fmt(correctness['a_threshold_match_ratio'])}` vs `{fmt(correctness['b_threshold_match_ratio'])}`",
        f"- secure replay status: `{correctness['a_secure_model_replay_status']}` vs `{correctness['b_secure_model_replay_status']}`",
    ]


def render_runtime(report):
    runtime = report['runtime_compare']
    return [
        '## Runtime',
        '',
        f"- total pipeline duration: `{fmt(runtime['a_total_pipeline_duration_sec'])}s` vs `{fmt(runtime['b_total_pipeline_duration_sec'])}s`",
        f"- total duration ratio (`b/a`): `{fmt(runtime['b_over_a_total_ratio'])}x`",
        f"- replay duration: `{fmt(runtime['a_replay_duration_sec'])}s` vs `{fmt(runtime['b_replay_duration_sec'])}s`",
        f"- replay ratio (`b/a`): `{fmt(runtime['b_over_a_replay_ratio'])}x`",
    ]


def render_communication(report):
    communication = report['communication_compare']
    lines = [
        '## Communication',
        '',
        f"- communication status: `{communication['a_status']}` vs `{communication['b_status']}`",
    ]
    if communication['b_aggregate_link_metrics'] is not None:
        aggregate = communication['b_aggregate_link_metrics']
        lines.extend(
            [
                f"- communication-visible max total bytes: `{fmt(aggregate.get('max_total_bytes'), 0)}`",
                f"- communication-visible max send bytes: `{fmt(aggregate.get('max_send_bytes'), 0)}`",
                f"- communication-visible max recv bytes: `{fmt(aggregate.get('max_recv_bytes'), 0)}`",
            ]
        )
    return lines


def render_recommendation(report):
    interpretation = report['interpretation']
    return [
        '## Recommendation',
        '',
        f"- default demonstration/runtime branch keeps `{interpretation['preferred_fast_runtime']}` because it is much faster",
        f"- communication citation branch uses `{interpretation['preferred_comm_runtime']}` because counters are available there",
    ]


def render_markdown(report):
    lines = [
        '# Runtime Branch Compare — Verified Candidate SPU',
        '',
        f"- fast branch: `{report['label_a']}`",
        f"- communication branch: `{report['label_b']}`",
        '',
    ]
    for section in (
        render_correctness(report),
        render_runtime(report),
        render_communication(report),
        render_recommendation(report),
    ):
        lines.extend(section)
        lines.append('')
    return '\n'.join(lines) + '\n'


def build_parser():
    parser = argparse.ArgumentParser(description='Compare two SPU runtime branches for the verified candidate.')
    parser.add_argument('--summary-a', required=True)
    parser.add_argument('--summary-b', required=True)
    parser.add_argument('--label-a', default='default_fast_runtime')
    parser.add_argument('--label-b', default='diagnostic_comm_runtime')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    summary_a = load_json(Path(args.summary_a).resolve())
    summary_b = load_json(Path(args.summary_b).resolve())
    report = build_compare(summary_a, summary_b, args.label_a, args.label_b)
    write_text(Path(args.output_json).resolve(), json.dumps(report, indent=2, sort_keys=True) + '\n')
    write_text(Path(args.output_md).resolve(), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
