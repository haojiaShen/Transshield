import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def metric_delta(lhs, rhs, key):
    left = lhs['metrics'].get(key)
    right = rhs['metrics'].get(key)
    if left is None or right is None:
        return None
    return float(right - left)


def main():
    parser = argparse.ArgumentParser(description='Compare two plaintext model evaluation summaries.')
    parser.add_argument('--eval-a', required=True)
    parser.add_argument('--eval-b', required=True)
    parser.add_argument('--label-a', default='model_a')
    parser.add_argument('--label-b', default='model_b')
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    eval_a = load_json(Path(args.eval_a).resolve())
    eval_b = load_json(Path(args.eval_b).resolve())

    output = {
        'eval_a': str(Path(args.eval_a).resolve()),
        'eval_b': str(Path(args.eval_b).resolve()),
        'label_a': args.label_a,
        'label_b': args.label_b,
        'sample_count_match': bool(eval_a.get('sample_count') == eval_b.get('sample_count')),
        'sample_paths_match': bool(eval_a.get('sample_paths_sha256') == eval_b.get('sample_paths_sha256')),
        'metrics_a': eval_a.get('metrics', {}),
        'metrics_b': eval_b.get('metrics', {}),
        'delta_b_minus_a': {
            'auc': metric_delta(eval_a, eval_b, 'auc'),
            'argmax_accuracy': metric_delta(eval_a, eval_b, 'argmax_accuracy'),
            'argmax_f1': metric_delta(eval_a, eval_b, 'argmax_f1'),
            'threshold_accuracy': metric_delta(eval_a, eval_b, 'threshold_accuracy'),
            'threshold_f1': metric_delta(eval_a, eval_b, 'threshold_f1'),
        },
        'args_snapshot_summary': {
            'a': eval_a.get('args_snapshot_summary', {}),
            'b': eval_b.get('args_snapshot_summary', {}),
        },
    }
    write_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
