import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dyvit import VisionTransformerDiffPruning
from tools.transshield_stage2_bundle import resolve_model_state_dict_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_model(args_snapshot: dict):
    base_rate = float(args_snapshot['base_rate'])
    keep_rate = [base_rate, base_rate ** 2, base_rate ** 3]
    return VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=int(args_snapshot['nb_classes']),
        pruning_loc=[3, 6, 9],
        token_ratio=keep_rate,
        distill=True,
        act_layer=args_snapshot['square_activation_mode'] if args_snapshot['use_square_gelu'] else 'gelu',
        use_mask_pruning=bool(args_snapshot['use_mask_pruning']),
        use_approx_attn=bool(args_snapshot['use_approx_attn']),
        approx_attn_mode=args_snapshot['approx_attn_mode'],
        fp32_attention=True,
        eval_pruning_mode=args_snapshot.get('eval_pruning_mode', 'topk_argsort'),
        eval_tie_policy=args_snapshot.get('eval_tie_policy', 'lowest_index'),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest_path = bundle_dir / 'manifest.json'
    args_snapshot_path = bundle_dir / 'args_snapshot.json'
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    checkpoint_path = bundle_dir / 'checkpoint-best.pth'

    manifest = load_json(manifest_path)
    args_snapshot = load_json(args_snapshot_path)

    required_paths = [
        manifest_path,
        args_snapshot_path,
        state_dict_path,
        checkpoint_path,
        bundle_dir / 'threshold_best.json',
        bundle_dir / 'commands.sh',
        bundle_dir / 'README.md',
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f'missing required bundle file: {path}')

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != manifest['primary']['checkpoint_sha256']:
        raise ValueError(
            f'checkpoint sha256 mismatch: expected {manifest["primary"]["checkpoint_sha256"]}, got {checkpoint_sha256}'
        )

    state_dict_sha256 = sha256_file(state_dict_path)
    expected_sha256 = manifest['export'].get('model_state_dict_sha256', manifest['export']['student_model_state_dict_sha256'])
    expected_tensor_count = manifest['export'].get('model_tensor_count', manifest['export']['student_model_tensor_count'])

    if state_dict_sha256 != expected_sha256:
        raise ValueError(
            f'state_dict sha256 mismatch: expected {expected_sha256}, got {state_dict_sha256}'
        )

    state_dict = torch.load(state_dict_path, map_location=args.device, weights_only=False)
    if len(state_dict) != expected_tensor_count:
        raise ValueError(
            f'state_dict tensor count mismatch: expected {expected_tensor_count}, got {len(state_dict)}'
        )

    model = build_model(args_snapshot).to(args.device)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            f'non-strict load result: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}'
        )

    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, int(args_snapshot['input_size']), int(args_snapshot['input_size']), device=args.device)
        output = model(sample)

    if not torch.isfinite(output).all():
        raise ValueError('non-finite output from frozen exported state dict')

    threshold_metrics = manifest['primary'].get('threshold_metrics') or {}
    summary = {
        'bundle_dir': str(bundle_dir),
        'checkpoint_sha256': checkpoint_sha256,
        'state_dict_sha256': state_dict_sha256,
        'output_shape': list(output.shape),
        'output_finite': True,
        'default_argmax_acc1': threshold_metrics.get('default_argmax_acc1'),
        'threshold_eval_acc1': threshold_metrics.get('eval_acc1'),
        'threshold': threshold_metrics.get('eval_binary_threshold'),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
