import argparse
import math
import csv
import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def checkpoint_args_to_dict(checkpoint_args):
    if hasattr(checkpoint_args, '__dict__'):
        return dict(vars(checkpoint_args))
    if isinstance(checkpoint_args, dict):
        return dict(checkpoint_args)
    return {}


def import_repo_modules(repo_root: Path):
    if 'torch._six' not in sys.modules:
        torch_six = types.ModuleType('torch._six')
        torch_six.inf = math.inf
        sys.modules['torch._six'] = torch_six
    sys.path.insert(0, str(repo_root))
    datasets_mod = importlib.import_module('datasets')
    dyvit_mod = importlib.import_module('models.dyvit')
    return datasets_mod, dyvit_mod


def infer_model_size(model_name: str):
    name = (model_name or '').lower()
    if any(token in name for token in ['deit-b', 'vit_deit_base', 'base']):
        return {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    if any(token in name for token in ['deit-256', 'tiny256']):
        return {'embed_dim': 256, 'depth': 12, 'num_heads': 4}
    return {'embed_dim': 384, 'depth': 12, 'num_heads': 6}


def normalized_args_snapshot(args_snapshot: dict):
    snapshot = dict(args_snapshot)
    snapshot.setdefault('model', 'deit-s')
    snapshot.setdefault('base_rate', 0.7)
    snapshot.setdefault('nb_classes', 2)
    snapshot.setdefault('input_size', 224)
    snapshot.setdefault('imagenet_default_mean_and_std', True)
    return snapshot


def build_model(args_snapshot: dict, model_cls):
    model_size = infer_model_size(args_snapshot.get('model', 'deit-s'))
    base_rate = float(args_snapshot.get('base_rate', 0.7))
    keep_rate = [base_rate, base_rate ** 2, base_rate ** 3]

    kwargs = {
        'patch_size': 16,
        'embed_dim': model_size['embed_dim'],
        'depth': model_size['depth'],
        'num_heads': model_size['num_heads'],
        'mlp_ratio': 4,
        'qkv_bias': True,
        'num_classes': int(args_snapshot.get('nb_classes', 2)),
        'pruning_loc': [3, 6, 9],
        'token_ratio': keep_rate,
        'distill': True,
    }

    parameters = inspect.signature(model_cls.__init__).parameters
    optional_args = {
        'use_mask_pruning': bool(args_snapshot.get('use_mask_pruning', False)),
        'use_approx_attn': bool(args_snapshot.get('use_approx_attn', False)),
        'approx_attn_mode': args_snapshot.get('approx_attn_mode', 'relu'),
        'fp32_attention': True,
        'eval_pruning_mode': args_snapshot.get('eval_pruning_mode', 'topk_argsort'),
        'eval_tie_policy': args_snapshot.get('eval_tie_policy', 'lowest_index'),
    }
    if 'act_layer' in parameters:
        optional_args['act_layer'] = (
            args_snapshot.get('square_activation_mode', 'gelu')
            if bool(args_snapshot.get('use_square_gelu', False))
            else 'gelu'
        )

    for key, value in optional_args.items():
        if key in parameters:
            kwargs[key] = value

    return model_cls(**kwargs)


def build_eval_transform(args_snapshot: dict, build_transform_fn):
    transform_args = SimpleNamespace(
        input_size=int(args_snapshot.get('input_size', 224)),
        imagenet_default_mean_and_std=bool(args_snapshot.get('imagenet_default_mean_and_std', True)),
        crop_pct=args_snapshot.get('crop_pct'),
        color_jitter=args_snapshot.get('color_jitter', 0.4),
        aa=args_snapshot.get('aa', 'rand-m9-mstd0.5-inc1'),
        train_interpolation=args_snapshot.get('train_interpolation', 'bicubic'),
        reprob=float(args_snapshot.get('reprob', 0.25)),
        remode=args_snapshot.get('remode', 'pixel'),
        recount=int(args_snapshot.get('recount', 1)),
    )
    return build_transform_fn(is_train=False, args=transform_args)


def binary_auc(class1_prob: torch.Tensor, targets: torch.Tensor):
    positive = class1_prob[targets == 1]
    negative = class1_prob[targets == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    positive = positive.view(-1, 1)
    negative = negative.view(1, -1)
    greater = (positive > negative).float().sum().item()
    equal = (positive == negative).float().sum().item()
    total = positive.numel() * negative.numel()
    return float((greater + 0.5 * equal) / total)


def binary_f1(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.long()
    targets = targets.long()
    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def sample_paths_sha256(sample_paths):
    digest = hashlib.sha256()
    for path in sample_paths:
        digest.update(path.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def candidate_thresholds(class1_prob: torch.Tensor):
    values = torch.unique(class1_prob.detach().cpu())
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return [0.5]
    values, _ = torch.sort(values)
    thresholds = [0.0]
    thresholds.extend(float(value.item()) for value in values)
    thresholds.append(1.0)
    return thresholds


def accuracy_at_threshold(class1_prob: torch.Tensor, targets: torch.Tensor, threshold: float):
    predictions = (class1_prob >= threshold).long()
    return float((predictions == targets).float().mean().item() * 100.0), predictions


def find_best_threshold(class1_prob: torch.Tensor, targets: torch.Tensor):
    best = None
    for threshold in candidate_thresholds(class1_prob):
        accuracy, predictions = accuracy_at_threshold(class1_prob, targets, threshold)
        f1 = binary_f1(predictions, targets)
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'pred_counts': [int(value) for value in torch.bincount(predictions, minlength=2).tolist()],
        }
        if best is None:
            best = candidate
            continue
        better = candidate['accuracy'] > best['accuracy']
        same_accuracy_better_f1 = (
            abs(candidate['accuracy'] - best['accuracy']) <= 1e-9
            and candidate['f1'] > best['f1']
        )
        same_accuracy_f1_closer_to_half = (
            abs(candidate['accuracy'] - best['accuracy']) <= 1e-9
            and abs(candidate['f1'] - best['f1']) <= 1e-12
            and abs(candidate['threshold'] - 0.5) < abs(best['threshold'] - 0.5)
        )
        if better or same_accuracy_better_f1 or same_accuracy_f1_closer_to_half:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description='Search a compatible binary threshold JSON for an external DynamicViT checkpoint.')
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    data_path = Path(args.data_path).resolve()

    datasets_mod, dyvit_mod = import_repo_modules(repo_root)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args_snapshot = normalized_args_snapshot(checkpoint_args_to_dict(checkpoint.get('args')))

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
    dataset = ImageFolder(root=str(data_path), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    all_logits = []
    all_probs = []
    all_targets = []
    finite_logits = True
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            all_logits.append(logits.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probabilities = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    class1_prob = probabilities[:, 1]
    argmax_predictions = logits.argmax(dim=1)
    argmax_accuracy = float((argmax_predictions == targets).float().mean().item() * 100.0)
    best = find_best_threshold(class1_prob, targets)
    threshold_accuracy, threshold_predictions = accuracy_at_threshold(class1_prob, targets, best['threshold'])

    rows = []
    sample_paths = [sample[0] for sample in dataset.samples]
    for index, sample_path in enumerate(sample_paths):
        rows.append(
            {
                'sample_index': index,
                'sample_path': sample_path,
                'target': int(targets[index].item()),
                'prob_1': float(class1_prob[index].item()),
                'argmax_prediction': int(argmax_predictions[index].item()),
                'threshold_prediction': int(threshold_predictions[index].item()),
            }
        )

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    output = {
        'checkpoint': checkpoint_path.name,
        'checkpoint_path': str(checkpoint_path),
        'repo_root': str(repo_root),
        'data_path': str(data_path),
        'sample_count': int(targets.numel()),
        'sample_paths_sha256': sample_paths_sha256(sample_paths),
        'finite_logits': bool(finite_logits),
        'eval_binary_threshold': float(best['threshold']),
        'eval_acc1': float(threshold_accuracy),
        'eval_loss': float(F.cross_entropy(logits, targets).item()),
        'auc': binary_auc(class1_prob, targets),
        'default_argmax_acc1': float(argmax_accuracy),
        'default_argmax_f1': binary_f1(argmax_predictions, targets),
        'best_threshold': float(best['threshold']),
        'best_threshold_acc': float(best['accuracy']),
        'best_threshold_f1': float(best['f1']),
        'best_threshold_pred_counts': best['pred_counts'],
        'args_snapshot_summary': {
            'model': args_snapshot.get('model'),
            'nb_classes': args_snapshot.get('nb_classes'),
            'base_rate': args_snapshot.get('base_rate'),
            'data_set': args_snapshot.get('data_set'),
            'use_square_gelu': bool(args_snapshot.get('use_square_gelu', False)),
            'square_activation_mode': args_snapshot.get('square_activation_mode'),
            'use_approx_attn': bool(args_snapshot.get('use_approx_attn', False)),
            'use_mask_pruning': bool(args_snapshot.get('use_mask_pruning', False)),
            'eval_pruning_mode': args_snapshot.get('eval_pruning_mode', 'topk_argsort'),
            'eval_tie_policy': args_snapshot.get('eval_tie_policy', 'lowest_index'),
        },
        'artifacts': {
            'output_csv': str(Path(args.output_csv).resolve()) if args.output_csv else None,
        },
    }
    write_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
