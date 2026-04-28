import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_class_names(raw_value: str):
    if not raw_value:
        return ['class_0', 'class_1']
    values = [item.strip() for item in raw_value.split(',') if item.strip()]
    if len(values) != 2:
        raise ValueError('--class-names must contain exactly two comma-separated names')
    return values


def label_name(class_names, index):
    if index is None:
        return None
    return class_names[index] if 0 <= index < len(class_names) else f'class_{index}'


def infer_target_from_path(sample_path: str):
    parent_name = Path(sample_path).parent.name
    return int(parent_name) if parent_name.isdigit() else None


def confidence_margin(probabilities):
    if len(probabilities) < 2:
        return None
    ordered = sorted((float(value) for value in probabilities), reverse=True)
    return float(ordered[0] - ordered[1])


def resolve_model_replay(payload):
    if 'model_replay' in payload:
        return payload['model_replay']
    return payload


def main():
    parser = argparse.ArgumentParser(
        description='Summarize secure BumbleBee/SPU replay output into an AI-friendly diagnosis report.'
    )
    parser.add_argument('--secure-replay-json', required=True)
    parser.add_argument('--class-names', default='class_0,class_1')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    class_names = parse_class_names(args.class_names)
    payload = load_json(Path(args.secure_replay_json).resolve())
    replay = resolve_model_replay(payload)

    if replay.get('status') != 'ok':
        raise ValueError(f'secure replay is not ready: status={replay.get("status")}')

    sample_paths = replay.get('sample_paths', [])
    probabilities = replay.get('probabilities', [])
    argmax_predictions = replay.get('argmax_predictions', [])
    threshold_predictions = replay.get('threshold_predictions')

    rows = []
    for index, sample_path in enumerate(sample_paths):
        target_index = infer_target_from_path(sample_path)
        argmax_class = int(argmax_predictions[index])
        threshold_class = int(threshold_predictions[index]) if threshold_predictions is not None else None
        probs = probabilities[index]
        row = {
            'sample_index': index,
            'sample_path': sample_path,
            'target_index': target_index,
            'target_label': label_name(class_names, target_index),
            'secure_argmax_class_index': argmax_class,
            'secure_argmax_label': label_name(class_names, argmax_class),
            'secure_threshold_class_index': threshold_class,
            'secure_threshold_label': label_name(class_names, threshold_class),
            'prob_class_0': float(probs[0]),
            'prob_class_1': float(probs[1]),
            'confidence_margin': confidence_margin(probs),
        }
        rows.append(row)

    summary = {
        'mode': 'secure_selected_image_diagnosis',
        'secure_replay_json': str(Path(args.secure_replay_json).resolve()),
        'class_names': class_names,
        'sample_count': len(rows),
        'class_threshold': replay.get('class_threshold'),
        'secure_runtime_status': replay.get('status'),
        'summary': {
            'secure_argmax_positive_count': sum(int(row['secure_argmax_class_index'] == 1) for row in rows),
            'secure_threshold_positive_count': (
                sum(int(row['secure_threshold_class_index'] == 1) for row in rows)
                if threshold_predictions is not None else None
            ),
        },
        'results': rows,
        'artifacts': {
            'output_csv': str(Path(args.output_csv).resolve()) if args.output_csv else None,
        },
    }
    write_json(Path(args.output_json).resolve(), summary)
    if args.output_csv:
        write_csv(Path(args.output_csv).resolve(), rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
