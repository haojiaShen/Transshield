import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_input_selection import (
    SelectedImageDataset,
    add_input_selection_args,
    resolve_selected_sample_paths,
)
from tools.transshield_plaintext_checkpoint_eval import sample_paths_sha256
from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold


def parse_class_names(raw_value: str):
    if not raw_value:
        return ['class_0', 'class_1']
    values = [item.strip() for item in raw_value.split(',') if item.strip()]
    if len(values) != 2:
        raise ValueError('--class-names must contain exactly two comma-separated names')
    return values


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


def label_name(class_names, index):
    if index is None:
        return None
    return class_names[index] if 0 <= index < len(class_names) else f'class_{index}'


def confidence_from_probs(probabilities):
    probs = probabilities.detach().cpu().tolist()
    if len(probs) < 2:
        return None
    ordered = sorted((float(value) for value in probs), reverse=True)
    return float(ordered[0] - ordered[1])


def main():
    parser = argparse.ArgumentParser(
        description='Run deployment-style diagnosis on selected images using the modified frozen Transshield bundle.'
    )
    parser.add_argument('--bundle-dir', required=True)
    add_input_selection_args(parser, include_data_path=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--class-names', default='class_0,class_1')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    if not args.image and not args.image_list and not args.input_dir:
        raise ValueError('one of --image / --image-list / --input-dir is required')

    class_names = parse_class_names(args.class_names)
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle = load_frozen_bundle(bundle_dir, args.device)
    threshold = resolve_threshold(bundle_dir, None)

    selection = resolve_selected_sample_paths(
        image_paths=args.image,
        image_list=args.image_list,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        max_samples=0,
    )
    dataset = SelectedImageDataset(selection['sample_paths'], bundle['transform'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    all_logits = []
    with torch.no_grad():
        for inputs, _paths, _targets in loader:
            inputs = inputs.to(args.device)
            logits = bundle['model'](inputs)
            all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(logits, dim=-1)
    argmax_predictions = logits.argmax(dim=1)
    threshold_predictions = None
    if threshold is not None and probabilities.shape[1] == 2:
        threshold_predictions = (probabilities[:, 1] >= threshold).long()

    rows = []
    threshold_positive_count = 0
    argmax_positive_count = 0
    for index, sample_path in enumerate(dataset.sample_paths):
        argmax_class = int(argmax_predictions[index].item())
        threshold_class = int(threshold_predictions[index].item()) if threshold_predictions is not None else None
        argmax_positive_count += int(argmax_class == 1)
        threshold_positive_count += int(threshold_class == 1) if threshold_class is not None else 0

        target_index = dataset.targets[index]
        row = {
            'sample_index': index,
            'sample_path': str(sample_path),
            'target_index': target_index,
            'target_label': label_name(class_names, target_index),
            'argmax_class_index': argmax_class,
            'argmax_label': label_name(class_names, argmax_class),
            'threshold_class_index': threshold_class,
            'threshold_label': label_name(class_names, threshold_class),
            'prob_class_0': float(probabilities[index, 0].item()),
            'prob_class_1': float(probabilities[index, 1].item()),
            'confidence_margin': confidence_from_probs(probabilities[index]),
        }
        rows.append(row)

    summary = {
        'mode': 'selected_image_diagnosis',
        'bundle_dir': str(bundle_dir),
        'selection_mode': selection['selection_mode'],
        'sample_count': len(rows),
        'sample_paths_sha256': sample_paths_sha256([row['sample_path'] for row in rows]),
        'class_names': class_names,
        'threshold': threshold,
        'summary': {
            'argmax_positive_count': argmax_positive_count,
            'threshold_positive_count': threshold_positive_count if threshold_predictions is not None else None,
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
