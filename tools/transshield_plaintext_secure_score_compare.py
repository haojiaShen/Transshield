import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def resolve_sample_path(sample_path: str, sample_root_from: str, sample_root_to: str):
    path = Path(sample_path)
    if path.exists() or not sample_root_from or not sample_root_to:
        return path
    try:
        relative = path.relative_to(Path(sample_root_from))
    except ValueError:
        return path
    return Path(sample_root_to) / relative


def infer_target_from_path(path: Path):
    parent_name = path.parent.name
    return int(parent_name) if parent_name.isdigit() else None


class SamplePathDataset(Dataset):
    def __init__(self, sample_paths, transform):
        self.sample_paths = [Path(path) for path in sample_paths]
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        path = self.sample_paths[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        target = infer_target_from_path(path)
        return tensor, str(path), (-1 if target is None else target)


def load_secure_replay_payload(path: Path):
    payload = load_json(path)
    if 'model_replay' in payload:
        return payload, payload['model_replay']
    return payload, payload


@torch.no_grad()
def run_plaintext_inference(bundle_dir: Path, sample_paths, device: str, batch_size: int, num_workers: int):
    bundle = load_frozen_bundle(bundle_dir, device)
    class_threshold = resolve_threshold(bundle_dir, None)
    dataset = SamplePathDataset(sample_paths, bundle['transform'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    all_logits = []
    ordered_paths = []
    ordered_targets = []
    for inputs, paths, targets in loader:
        inputs = inputs.to(device)
        logits = bundle['model'](inputs)
        all_logits.append(logits.detach().cpu())
        ordered_paths.extend(paths)
        ordered_targets.extend(targets.tolist())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=-1)
    argmax_predictions = logits.argmax(dim=1)
    threshold_predictions = None
    if class_threshold is not None and probs.shape[1] == 2:
        threshold_predictions = (probs[:, 1] >= class_threshold).long()

    targets = None
    if all(target >= 0 for target in ordered_targets):
        targets = torch.tensor(ordered_targets, dtype=torch.long)

    return {
        'device': device,
        'class_threshold': class_threshold,
        'sample_paths': ordered_paths,
        'targets': targets,
        'logits': logits,
        'probabilities': probs,
        'argmax_predictions': argmax_predictions,
        'threshold_predictions': threshold_predictions,
    }


def tensor_diff_summary(lhs, rhs):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if lhs.shape != rhs.shape:
        return {
            'shape_match': False,
            'lhs_shape': list(lhs.shape),
            'rhs_shape': list(rhs.shape),
            'max_abs_error': None,
            'mean_abs_error': None,
        }
    diff = (lhs - rhs).abs()
    return {
        'shape_match': True,
        'shape': list(lhs.shape),
        'max_abs_error': float(diff.max().item()) if diff.numel() else 0.0,
        'mean_abs_error': float(diff.mean().item()) if diff.numel() else 0.0,
    }


def prediction_match_summary(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    lhs = lhs.detach().cpu().long()
    rhs = rhs.detach().cpu().long()
    if lhs.shape != rhs.shape:
        return {
            'shape_match': False,
            'lhs_shape': list(lhs.shape),
            'rhs_shape': list(rhs.shape),
            'match_ratio': None,
            'mismatch_count': None,
        }
    mismatch_count = int((lhs != rhs).sum().item())
    total = int(lhs.numel())
    return {
        'shape_match': True,
        'shape': list(lhs.shape),
        'mismatch_count': mismatch_count,
        'match_ratio': float((total - mismatch_count) / total) if total else 1.0,
    }


def accuracy_summary(predictions, targets):
    if predictions is None or targets is None:
        return None
    predictions = predictions.detach().cpu().long()
    targets = targets.detach().cpu().long()
    return float((predictions == targets).float().mean().item() * 100.0)


def build_per_sample_rows(sample_paths, targets, plaintext, secure):
    rows = []
    sample_count = len(sample_paths)
    for index in range(sample_count):
        target = None if targets is None else int(targets[index].item())
        plaintext_probs = plaintext['probabilities'][index]
        secure_probs = secure['probabilities'][index]
        row = {
            'sample_index': index,
            'sample_path': sample_paths[index],
            'target': target,
            'plaintext_logit_0': float(plaintext['logits'][index, 0].item()),
            'plaintext_logit_1': float(plaintext['logits'][index, 1].item()),
            'secure_logit_0': float(secure['logits'][index, 0].item()),
            'secure_logit_1': float(secure['logits'][index, 1].item()),
            'plaintext_prob_0': float(plaintext_probs[0].item()),
            'plaintext_prob_1': float(plaintext_probs[1].item()),
            'secure_prob_0': float(secure_probs[0].item()),
            'secure_prob_1': float(secure_probs[1].item()),
            'logit_0_abs_diff': float(abs(plaintext['logits'][index, 0].item() - secure['logits'][index, 0].item())),
            'logit_1_abs_diff': float(abs(plaintext['logits'][index, 1].item() - secure['logits'][index, 1].item())),
            'prob_0_abs_diff': float(abs(plaintext_probs[0].item() - secure_probs[0].item())),
            'prob_1_abs_diff': float(abs(plaintext_probs[1].item() - secure_probs[1].item())),
            'plaintext_argmax': int(plaintext['argmax_predictions'][index].item()),
            'secure_argmax': int(secure['argmax_predictions'][index].item()),
        }
        if plaintext['threshold_predictions'] is not None and secure['threshold_predictions'] is not None:
            row['plaintext_threshold_pred'] = int(plaintext['threshold_predictions'][index].item())
            row['secure_threshold_pred'] = int(secure['threshold_predictions'][index].item())
        rows.append(row)
    return rows


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
        description='Compare direct plaintext bundle inference against the secure BumbleBee/SPU replay output on the same samples.'
    )
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--secure-replay-json', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-root-from', default='')
    parser.add_argument('--sample-root-to', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    secure_summary, secure_model_replay = load_secure_replay_payload(Path(args.secure_replay_json).resolve())
    if secure_model_replay.get('status') != 'ok':
        raise ValueError(f'secure replay is not ready for comparison: status={secure_model_replay.get("status")}')

    secure_sample_paths = [
        str(resolve_sample_path(path, args.sample_root_from, args.sample_root_to).resolve())
        for path in secure_model_replay['sample_paths']
    ]
    missing = [path for path in secure_sample_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f'comparison sample paths are missing: {missing[:4]}')

    plaintext = run_plaintext_inference(
        Path(args.bundle_dir).resolve(),
        secure_sample_paths,
        args.device,
        args.batch_size,
        args.num_workers,
    )

    secure_logits = torch.tensor(secure_model_replay['logits'], dtype=torch.float32)
    secure_probabilities = torch.tensor(secure_model_replay['probabilities'], dtype=torch.float32)
    secure_argmax_predictions = torch.tensor(secure_model_replay['argmax_predictions'], dtype=torch.long)
    secure_threshold_predictions = None
    if secure_model_replay.get('threshold_predictions') is not None:
        secure_threshold_predictions = torch.tensor(secure_model_replay['threshold_predictions'], dtype=torch.long)

    secure_targets = None
    if secure_model_replay.get('targets') is not None:
        secure_targets = torch.tensor(secure_model_replay['targets'], dtype=torch.long)
    elif plaintext['targets'] is not None:
        secure_targets = plaintext['targets']

    if plaintext['sample_paths'] != secure_sample_paths:
        raise ValueError('plaintext sample order does not match secure replay sample order')

    secure = {
        'logits': secure_logits,
        'probabilities': secure_probabilities,
        'argmax_predictions': secure_argmax_predictions,
        'threshold_predictions': secure_threshold_predictions,
    }

    per_sample_rows = build_per_sample_rows(secure_sample_paths, secure_targets, plaintext, secure)
    output = {
        'bundle_dir': str(Path(args.bundle_dir).resolve()),
        'secure_replay_json': str(Path(args.secure_replay_json).resolve()),
        'sample_count': len(secure_sample_paths),
        'device': args.device,
        'class_threshold': plaintext['class_threshold'],
        'comparison': {
            'logits': tensor_diff_summary(plaintext['logits'], secure_logits),
            'probabilities': tensor_diff_summary(plaintext['probabilities'], secure_probabilities),
            'argmax_predictions': prediction_match_summary(plaintext['argmax_predictions'], secure_argmax_predictions),
            'threshold_predictions': prediction_match_summary(plaintext['threshold_predictions'], secure_threshold_predictions),
            'plaintext_argmax_accuracy': accuracy_summary(plaintext['argmax_predictions'], secure_targets),
            'secure_argmax_accuracy': accuracy_summary(secure_argmax_predictions, secure_targets),
            'plaintext_threshold_accuracy': accuracy_summary(plaintext['threshold_predictions'], secure_targets),
            'secure_threshold_accuracy': accuracy_summary(secure_threshold_predictions, secure_targets),
        },
        'source_status': {
            'secure_overall_passed': secure_summary.get('overall_passed'),
            'secure_model_replay_status': secure_model_replay.get('status'),
        },
        'per_sample_preview': per_sample_rows[:8],
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    if args.output_csv:
        write_csv(Path(args.output_csv).resolve(), per_sample_rows)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
