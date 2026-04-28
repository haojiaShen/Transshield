import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def scalar_delta(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return float(rhs - lhs)


def scalar_ratio(lhs, rhs):
    if lhs in (None, 0) or rhs is None:
        return None
    return float(rhs / lhs)


def main():
    parser = argparse.ArgumentParser(description='Compare two secure profiling summaries.')
    parser.add_argument('--summary-a', required=True)
    parser.add_argument('--summary-b', required=True)
    parser.add_argument('--label-a', default='secure_a')
    parser.add_argument('--label-b', default='secure_b')
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    summary_a = load_json(Path(args.summary_a).resolve())
    summary_b = load_json(Path(args.summary_b).resolve())

    step_keys = sorted(
        set(summary_a.get('step_profile', {}).get('durations_sec', {}))
        | set(summary_b.get('step_profile', {}).get('durations_sec', {}))
    )
    step_compare = {}
    for key in step_keys:
        left = summary_a.get('step_profile', {}).get('durations_sec', {}).get(key)
        right = summary_b.get('step_profile', {}).get('durations_sec', {}).get(key)
        step_compare[key] = {
            'a_sec': left,
            'b_sec': right,
            'delta_sec': scalar_delta(left, right),
            'ratio_b_over_a': scalar_ratio(left, right),
        }

    link_a = summary_a.get('communication_profile', {}).get('aggregate_link_metrics', {}) or {}
    link_b = summary_b.get('communication_profile', {}).get('aggregate_link_metrics', {}) or {}
    output = {
        'summary_a': str(Path(args.summary_a).resolve()),
        'summary_b': str(Path(args.summary_b).resolve()),
        'label_a': args.label_a,
        'label_b': args.label_b,
        'time_compare': {
            'total_pipeline_duration_sec': {
                'a': summary_a.get('step_profile', {}).get('total_pipeline_duration_sec'),
                'b': summary_b.get('step_profile', {}).get('total_pipeline_duration_sec'),
                'delta_sec': scalar_delta(
                    summary_a.get('step_profile', {}).get('total_pipeline_duration_sec'),
                    summary_b.get('step_profile', {}).get('total_pipeline_duration_sec'),
                ),
                'ratio_b_over_a': scalar_ratio(
                    summary_a.get('step_profile', {}).get('total_pipeline_duration_sec'),
                    summary_b.get('step_profile', {}).get('total_pipeline_duration_sec'),
                ),
            },
            'steps': step_compare,
        },
        'communication_compare': {
            'max_total_bytes': {
                'a': link_a.get('max_total_bytes'),
                'b': link_b.get('max_total_bytes'),
                'delta': scalar_delta(link_a.get('max_total_bytes'), link_b.get('max_total_bytes')),
                'ratio_b_over_a': scalar_ratio(link_a.get('max_total_bytes'), link_b.get('max_total_bytes')),
            },
            'sum_total_bytes': {
                'a': link_a.get('sum_total_bytes'),
                'b': link_b.get('sum_total_bytes'),
                'delta': scalar_delta(link_a.get('sum_total_bytes'), link_b.get('sum_total_bytes')),
                'ratio_b_over_a': scalar_ratio(link_a.get('sum_total_bytes'), link_b.get('sum_total_bytes')),
            },
        },
    }
    write_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
