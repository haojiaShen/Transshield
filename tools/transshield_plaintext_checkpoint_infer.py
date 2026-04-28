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
from tools.transshield_plaintext_checkpoint_eval import (
    accuracy,
    binary_f1,
    build_eval_transform,
    build_model,
    checkpoint_args_to_dict,
    find_threshold_json,
    import_repo_modules,
    load_json,
    pairwise_binary_auc,
    sample_paths_sha256,
    write_json,
)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Run plaintext inference on an explicit image list or image directory for either the baseline or modified Transshield checkpoint.'
    )
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    add_input_selection_args(parser, include_data_path=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threshold-json', default='')
    parser.add_argument('--label', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    if not args.image and not args.image_list and not args.input_dir:
        raise ValueError('one of --image / --image-list / --input-dir is required')

    repo_root = Path(args.repo_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    threshold_json = find_threshold_json(checkpoint_path, args.threshold_json)

    datasets_mod, dyvit_mod = import_repo_modules(repo_root)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args_snapshot = checkpoint_args_to_dict(checkpoint.get('args'))
    model = build_model(args_snapshot, dyvit_mod.VisionTransformerDiffPruning).to(args.device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, 'missing_keys', None) or getattr(load_result, 'unexpected_keys', None):
        raise ValueError(
            f'non-strict load result: missing={getattr(load_result, "missing_keys", None)} '
            f'unexpected={getattr(load_result, "unexpected_keys", None)}'
        )
    model.eval()

    transform = build_eval_transform(args_snapshot, datasets_mod.build_transform)
    selection = resolve_selected_sample_paths(
        image_paths=args.image,
        image_list=args.image_list,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        max_samples=0,
    )
    dataset = SelectedImageDataset(selection['sample_paths'], transform)
    selection_mode = selection['selection_mode']

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    all_logits = []
    finite_logits = True
    with torch.no_grad():
        for inputs, _paths, _targets in loader:
            inputs = inputs.to(args.device)
            logits = model(inputs)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(logits, dim=-1)
    argmax_predictions = logits.argmax(dim=1)
    class1_prob = probabilities[:, 1]

    threshold = None
    threshold_predictions = None
    if threshold_json is not None and threshold_json.exists():
        threshold_payload = load_json(threshold_json)
        threshold = float(threshold_payload['eval_binary_threshold'])
        threshold_predictions = (class1_prob >= threshold).long()

    targets = None
    if all(target is not None for target in dataset.targets):
        targets = torch.tensor(dataset.targets, dtype=torch.long)

    rows = []
    sample_paths = [str(path) for path in dataset.sample_paths]
    for index, sample_path in enumerate(sample_paths):
        row = {
            'sample_index': index,
            'sample_path': sample_path,
            'target': None if targets is None else int(targets[index].item()),
            'logit_0': float(logits[index, 0].item()),
            'logit_1': float(logits[index, 1].item()),
            'prob_0': float(probabilities[index, 0].item()),
            'prob_1': float(probabilities[index, 1].item()),
            'argmax_prediction': int(argmax_predictions[index].item()),
        }
        if threshold_predictions is not None:
            row['threshold_prediction'] = int(threshold_predictions[index].item())
        rows.append(row)

    if args.output_csv:
        write_csv(Path(args.output_csv).resolve(), rows)

    auc = pairwise_binary_auc(class1_prob, targets) if targets is not None else None
    argmax_accuracy = accuracy(argmax_predictions, targets) if targets is not None else None
    argmax_f1 = binary_f1(argmax_predictions, targets) if targets is not None else None
    threshold_accuracy = accuracy(threshold_predictions, targets) if targets is not None and threshold_predictions is not None else None
    threshold_f1 = binary_f1(threshold_predictions, targets) if targets is not None and threshold_predictions is not None else None

    summary = {
        'label': args.label or checkpoint_path.stem,
        'mode': 'selected_image_inference',
        'selection_mode': selection_mode,
        'repo_root': str(repo_root),
        'checkpoint_path': str(checkpoint_path),
        'threshold_json': str(threshold_json) if threshold_json is not None else None,
        'sample_count': len(sample_paths),
        'sample_paths_sha256': sample_paths_sha256(sample_paths),
        'finite_logits': bool(finite_logits),
        'metrics': {
            'auc': auc,
            'argmax_accuracy': argmax_accuracy,
            'argmax_f1': argmax_f1,
            'threshold': threshold,
            'threshold_accuracy': threshold_accuracy,
            'threshold_f1': threshold_f1,
        },
        'args_snapshot_summary': {
            'model': args_snapshot.get('model'),
            'use_square_gelu': bool(args_snapshot.get('use_square_gelu', False)),
            'square_activation_mode': args_snapshot.get('square_activation_mode'),
            'use_approx_attn': bool(args_snapshot.get('use_approx_attn', False)),
            'approx_attn_mode': args_snapshot.get('approx_attn_mode'),
            'use_mask_pruning': bool(args_snapshot.get('use_mask_pruning', False)),
            'eval_pruning_mode': args_snapshot.get('eval_pruning_mode', 'topk_argsort'),
            'eval_tie_policy': args_snapshot.get('eval_tie_policy', 'lowest_index'),
        },
        'artifacts': {
            'output_csv': str(Path(args.output_csv).resolve()) if args.output_csv else None,
        },
        'per_sample_preview': rows[:8],
    }
    write_json(Path(args.output_json).resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
