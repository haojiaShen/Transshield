import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


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
    return {'embed_dim': 384, 'depth': 12, 'num_heads': 6}


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

    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters
    if 'act_layer' in parameters:
        kwargs['act_layer'] = (
            args_snapshot.get('square_activation_mode', 'gelu')
            if bool(args_snapshot.get('use_square_gelu', False))
            else 'gelu'
        )
    if 'use_mask_pruning' in parameters:
        kwargs['use_mask_pruning'] = bool(args_snapshot.get('use_mask_pruning', False))
    if 'use_approx_attn' in parameters:
        kwargs['use_approx_attn'] = bool(args_snapshot.get('use_approx_attn', False))
    if 'approx_attn_mode' in parameters:
        kwargs['approx_attn_mode'] = args_snapshot.get('approx_attn_mode', 'relu')
    if 'fp32_attention' in parameters:
        kwargs['fp32_attention'] = True
    if 'eval_pruning_mode' in parameters:
        kwargs['eval_pruning_mode'] = args_snapshot.get('eval_pruning_mode', 'topk_argsort')
    if 'eval_tie_policy' in parameters:
        kwargs['eval_tie_policy'] = args_snapshot.get('eval_tie_policy', 'lowest_index')
    if 'secure_static_depth' in parameters:
        kwargs['secure_static_depth'] = int(args_snapshot.get('secure_static_train_depth', 0) or 0)
    if 'secure_static_skip_pruning' in parameters:
        kwargs['secure_static_skip_pruning'] = bool(args_snapshot.get('secure_static_skip_pruning', True))
    if 'nonempty_keep_guard' in parameters and 'nonempty_keep_guard' in args_snapshot:
        kwargs['nonempty_keep_guard'] = bool(args_snapshot.get('nonempty_keep_guard'))

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


def pairwise_binary_auc(class1_prob: torch.Tensor, targets: torch.Tensor):
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
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    return float((predictions == targets).float().mean().item() * 100.0)


def sample_paths_sha256(sample_paths):
    digest = hashlib.sha256()
    for path in sample_paths:
        digest.update(path.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def find_threshold_json(checkpoint_path: Path, threshold_json_arg: str):
    if threshold_json_arg:
        return Path(threshold_json_arg).resolve()
    candidate = checkpoint_path.resolve().parent / 'threshold_best.json'
    return candidate if candidate.exists() else None


def resolve_bundle_threshold(bundle_dir: Path, threshold_json_arg: str):
    from tools.transshield_stage2_bundle import resolve_threshold_payload

    if threshold_json_arg:
        threshold_path = Path(threshold_json_arg).resolve()
        if not threshold_path.exists():
            raise FileNotFoundError(f'threshold json does not exist: {threshold_path}')
        return threshold_path, load_json(threshold_path)

    threshold_payload = resolve_threshold_payload(bundle_dir)
    threshold_path = bundle_dir / 'threshold_best.json'
    return threshold_path if threshold_path.exists() or threshold_path.is_symlink() else None, threshold_payload


def main():
    parser = argparse.ArgumentParser(description='Evaluate a plaintext ViT checkpoint from either the baseline repo or the modified Transshield repo.')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--checkpoint')
    source_group.add_argument('--bundle-dir')
    parser.add_argument(
        '--checkpoint-model-key',
        default='model',
        help='State dict key to load when --checkpoint points to a training checkpoint.',
    )
    parser.add_argument('--repo-root', default='')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--threshold-json', default='')
    parser.add_argument('--label', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    repo_root: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    bundle_dir: Optional[Path] = None
    model_state_dict_path: Optional[Path] = None
    threshold_json: Optional[Path] = None
    threshold_payload = None

    if args.bundle_dir:
        from tools.transshield_stage2_bundle import load_frozen_bundle

        bundle_dir = Path(args.bundle_dir).resolve()
        bundle = load_frozen_bundle(bundle_dir, args.device)
        args_snapshot = dict(bundle['args_snapshot'])
        model = bundle['model']
        transform = bundle['transform']
        model_state_dict_path = Path(bundle['model_state_dict_path']).resolve()
        threshold_json, threshold_payload = resolve_bundle_threshold(bundle_dir, args.threshold_json)
        repo_root = Path(__file__).resolve().parents[1]
    else:
        repo_root = Path(args.repo_root).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        threshold_json = find_threshold_json(checkpoint_path, args.threshold_json)

        datasets_mod, dyvit_mod = import_repo_modules(repo_root)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        args_snapshot = checkpoint_args_to_dict(checkpoint.get('args'))
        model = build_model(args_snapshot, dyvit_mod.VisionTransformerDiffPruning).to(args.device)
        if isinstance(checkpoint, dict):
            if args.checkpoint_model_key not in checkpoint:
                raise KeyError(
                    f'checkpoint model key {args.checkpoint_model_key!r} not found in {checkpoint_path}'
                )
            state_dict = checkpoint[args.checkpoint_model_key]
        else:
            state_dict = checkpoint
        load_result = model.load_state_dict(state_dict, strict=True)
        if getattr(load_result, 'missing_keys', None) or getattr(load_result, 'unexpected_keys', None):
            raise ValueError(
                f'non-strict load result: missing={getattr(load_result, "missing_keys", None)} '
                f'unexpected={getattr(load_result, "unexpected_keys", None)}'
            )
        model.eval()

        transform = build_eval_transform(args_snapshot, datasets_mod.build_transform)
        if threshold_json is not None and threshold_json.exists():
            threshold_payload = load_json(threshold_json)

    dataset = ImageFolder(root=str(data_path), transform=transform)
    if args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.imgs = dataset.imgs[: args.max_samples]
        dataset.targets = dataset.targets[: args.max_samples]
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
    argmax_predictions = logits.argmax(dim=1)
    class1_prob = probabilities[:, 1]
    loss = float(F.cross_entropy(logits, targets).item())
    auc = pairwise_binary_auc(class1_prob, targets)

    threshold = None
    threshold_predictions = None
    threshold_accuracy = None
    threshold_f1 = None
    if threshold_payload is not None:
        threshold = float(threshold_payload['eval_binary_threshold'])
        threshold_predictions = (class1_prob >= threshold).long()
        threshold_accuracy = accuracy(threshold_predictions, targets)
        threshold_f1 = binary_f1(threshold_predictions, targets)

    rows = []
    sample_paths = [sample[0] for sample in dataset.samples]
    for index, sample_path in enumerate(sample_paths):
        row = {
            'sample_index': index,
            'sample_path': sample_path,
            'target': int(targets[index].item()),
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
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        'label': args.label or (bundle_dir.name if bundle_dir is not None else checkpoint_path.stem),
        'repo_root': str(repo_root) if repo_root is not None else None,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None,
        'checkpoint_model_key': args.checkpoint_model_key if checkpoint_path is not None else None,
        'bundle_dir': str(bundle_dir) if bundle_dir is not None else None,
        'model_state_dict_path': str(model_state_dict_path) if model_state_dict_path is not None else None,
        'threshold_json': str(threshold_json) if threshold_json is not None else None,
        'data_path': str(data_path),
        'sample_count': int(targets.numel()),
        'sample_paths_sha256': sample_paths_sha256(sample_paths),
        'finite_logits': bool(finite_logits),
        'metrics': {
            'eval_loss': loss,
            'auc': auc,
            'argmax_accuracy': accuracy(argmax_predictions, targets),
            'argmax_f1': binary_f1(argmax_predictions, targets),
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
            'secure_static_train_depth': int(args_snapshot.get('secure_static_train_depth', 0) or 0),
            'secure_static_skip_pruning': bool(args_snapshot.get('secure_static_skip_pruning', True)),
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
