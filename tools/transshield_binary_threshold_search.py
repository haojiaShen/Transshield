import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_threshold_branch_eval import binary_auc
from tools.transshield_plaintext_checkpoint_eval import (
    build_eval_transform,
    build_model,
    checkpoint_args_to_dict,
    import_repo_modules,
)


def write_json(path: Path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def load_checkpoint_bundle(checkpoint_path: Path, device: str, repo_root: Path, checkpoint_model_key: str = 'model'):
    checkpoint = torch.load(checkpoint_path.resolve(), map_location='cpu', weights_only=False)
    args_snapshot = checkpoint_args_to_dict(checkpoint.get('args'))
    datasets_mod, dyvit_mod = import_repo_modules(repo_root)
    model = build_model(args_snapshot, dyvit_mod.VisionTransformerDiffPruning).to(device)
    if checkpoint_model_key not in checkpoint:
        raise KeyError(f'checkpoint model key {checkpoint_model_key!r} not found in {checkpoint_path}')
    state_dict = checkpoint[checkpoint_model_key]
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            f'non-strict load result: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}'
        )
    model.eval()
    transform = build_eval_transform(args_snapshot, datasets_mod.build_transform)
    return {
        'checkpoint': checkpoint,
        'checkpoint_model_key': checkpoint_model_key,
        'args_snapshot': args_snapshot,
        'model': model,
        'transform': transform,
    }


def collect_eval_outputs(model, data_loader, device: str):
    all_probs = []
    all_targets = []
    all_logits = []
    finite_logits = True

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            all_logits.append(logits.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    loss = F.cross_entropy(logits, targets).item()
    return {
        'logits': logits,
        'probs': probs,
        'targets': targets,
        'loss': float(loss),
        'finite_logits': finite_logits,
    }


def accuracy_at_threshold(class1_prob, targets, threshold: float):
    pred = (class1_prob >= threshold).long()
    return float((pred == targets).float().mean().item() * 100.0), pred


def candidate_thresholds(class1_prob):
    values = torch.unique(class1_prob.detach().cpu())
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return [0.5]
    values, _ = torch.sort(values)
    thresholds = [0.0]
    thresholds.extend(float(value.item()) for value in values)
    thresholds.append(1.0)
    return thresholds


def find_best_threshold(class1_prob, targets):
    best = None
    for threshold in candidate_thresholds(class1_prob):
        acc, pred = accuracy_at_threshold(class1_prob, targets, threshold)
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(acc),
            'pred_counts': [int(value) for value in torch.bincount(pred, minlength=2).tolist()],
        }
        if best is None:
            best = candidate
            continue
        better = candidate['accuracy'] > best['accuracy']
        same_acc = math.isclose(candidate['accuracy'], best['accuracy'], rel_tol=0.0, abs_tol=1e-9)
        closer_to_half = abs(candidate['threshold'] - 0.5) < abs(best['threshold'] - 0.5)
        if better or (same_acc and closer_to_half):
            best = candidate
    return best


def load_data_loader(data_path: Path, transform, batch_size: int, num_workers: int):
    dataset = ImageFolder(root=str(data_path.resolve()), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)


def build_search_report(checkpoint_path: Path, data_path: Path, device: str, batch_size: int, num_workers: int, repo_root: Path, checkpoint_model_key: str):
    bundle = load_checkpoint_bundle(checkpoint_path, device, repo_root, checkpoint_model_key)
    data_loader = load_data_loader(data_path, bundle['transform'], batch_size, num_workers)
    outputs = collect_eval_outputs(bundle['model'], data_loader, device)

    probs = outputs['probs']
    targets = outputs['targets']
    class1_prob = probs[:, 1]
    argmax_pred = probs.argmax(dim=1)
    default_acc = float((argmax_pred == targets).float().mean().item() * 100.0)
    best = find_best_threshold(class1_prob, targets)
    auc = binary_auc(class1_prob, targets)

    return {
        'checkpoint': str(checkpoint_path.resolve().name),
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_model_key': bundle['checkpoint_model_key'],
        'data_path': str(data_path.resolve()),
        'sample_count': int(targets.numel()),
        'finite_logits': bool(outputs['finite_logits']),
        'eval_binary_threshold': float(best['threshold']),
        'eval_acc1': float(best['accuracy']),
        'eval_loss': float(outputs['loss']),
        'auc': auc,
        'default_argmax_acc1': float(default_acc),
        'default_pred_counts': [int(value) for value in torch.bincount(argmax_pred, minlength=2).tolist()],
        'best_threshold': float(best['threshold']),
        'best_threshold_acc': float(best['accuracy']),
        'best_threshold_pred_counts': best['pred_counts'],
        'args_snapshot_summary': {
            'model': bundle['args_snapshot'].get('model'),
            'use_square_gelu': bool(bundle['args_snapshot'].get('use_square_gelu')),
            'square_activation_mode': bundle['args_snapshot'].get('square_activation_mode'),
            'use_approx_attn': bool(bundle['args_snapshot'].get('use_approx_attn')),
            'approx_attn_mode': bundle['args_snapshot'].get('approx_attn_mode'),
            'use_mask_pruning': bool(bundle['args_snapshot'].get('use_mask_pruning')),
            'eval_pruning_mode': bundle['args_snapshot'].get('eval_pruning_mode', 'topk_argsort'),
            'eval_tie_policy': bundle['args_snapshot'].get('eval_tie_policy', 'lowest_index'),
            'secure_static_train_depth': int(bundle['args_snapshot'].get('secure_static_train_depth', 0) or 0),
            'secure_static_skip_pruning': bool(bundle['args_snapshot'].get('secure_static_skip_pruning', True)),
        },
    }


def build_eval_report(checkpoint_path: Path, threshold_json: Path, data_path: Path, device: str, batch_size: int, num_workers: int, repo_root: Path, checkpoint_model_key: str):
    bundle = load_checkpoint_bundle(checkpoint_path, device, repo_root, checkpoint_model_key)
    threshold_payload = json.loads(threshold_json.resolve().read_text(encoding='utf-8'))
    threshold = float(threshold_payload['eval_binary_threshold'])
    data_loader = load_data_loader(data_path, bundle['transform'], batch_size, num_workers)
    outputs = collect_eval_outputs(bundle['model'], data_loader, device)

    probs = outputs['probs']
    targets = outputs['targets']
    class1_prob = probs[:, 1]
    threshold_acc, pred = accuracy_at_threshold(class1_prob, targets, threshold)
    auc = binary_auc(class1_prob, targets)

    return {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_model_key': bundle['checkpoint_model_key'],
        'threshold_json': str(threshold_json.resolve()),
        'data_path': str(data_path.resolve()),
        'sample_count': int(targets.numel()),
        'finite_logits': bool(outputs['finite_logits']),
        'eval_binary_threshold': float(threshold),
        'eval_acc1': float(threshold_acc),
        'eval_loss': float(outputs['loss']),
        'auc': auc,
        'pred_counts': [int(value) for value in torch.bincount(pred, minlength=2).tolist()],
    }


def main():
    parser = argparse.ArgumentParser(description='Search or evaluate the best binary threshold for a Transshield checkpoint.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_search = subparsers.add_parser('search', help='find and save the best validation threshold')
    parser_search.add_argument('--checkpoint', required=True)
    parser_search.add_argument('--checkpoint-model-key', default='model')
    parser_search.add_argument('--data-path', required=True)
    parser_search.add_argument('--repo-root', default=str(REPO_ROOT))
    parser_search.add_argument('--device', default='cpu')
    parser_search.add_argument('--batch-size', type=int, default=32)
    parser_search.add_argument('--num-workers', type=int, default=0)
    parser_search.add_argument('--output-json', required=True)

    parser_eval = subparsers.add_parser('eval', help='evaluate a checkpoint with a saved threshold json')
    parser_eval.add_argument('--checkpoint', required=True)
    parser_eval.add_argument('--checkpoint-model-key', default='model')
    parser_eval.add_argument('--threshold-json', required=True)
    parser_eval.add_argument('--data-path', required=True)
    parser_eval.add_argument('--repo-root', default=str(REPO_ROOT))
    parser_eval.add_argument('--device', default='cpu')
    parser_eval.add_argument('--batch-size', type=int, default=32)
    parser_eval.add_argument('--num-workers', type=int, default=0)
    parser_eval.add_argument('--output-json', required=True)

    args = parser.parse_args()

    if args.command == 'search':
        report = build_search_report(
            checkpoint_path=Path(args.checkpoint),
            data_path=Path(args.data_path),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repo_root=Path(args.repo_root).resolve(),
            checkpoint_model_key=args.checkpoint_model_key,
        )
    else:
        report = build_eval_report(
            checkpoint_path=Path(args.checkpoint),
            threshold_json=Path(args.threshold_json),
            data_path=Path(args.data_path),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repo_root=Path(args.repo_root).resolve(),
            checkpoint_model_key=args.checkpoint_model_key,
        )

    write_json(Path(args.output_json).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
