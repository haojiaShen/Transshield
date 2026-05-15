#!/usr/bin/env python3
"""Merge LRD decomposed weights back to standard format for SPU deployment."""
import torch, json, os, re, sys

src = '/home/yclcg/Transshield_final/artifacts/lrd_finetuned_rank192/lrd_rank192_finetuned_best.pth'
dst_dir = '/home/yclcg/Transshield_final/artifacts/frozen_bundle_lrd_rank192_merged_20260514'
os.makedirs(dst_dir, exist_ok=True)

sd = torch.load(src, map_location='cpu', weights_only=False)
if isinstance(sd, dict) and 'model' in sd:
    sd = sd['model']
elif isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']

merged = {}

# Find sequential groups
seq_groups = {}
for k in sorted(sd.keys()):
    m = re.match(r'^(.+)\.(\d+)\.(weight|bias)$', k)
    if m:
        prefix, idx, param = m.groups()
        if prefix not in seq_groups:
            seq_groups[prefix] = {}
        seq_groups[prefix][f'{idx}.{param}'] = k

for prefix, params in sorted(seq_groups.items()):
    has_0w = '0.weight' in params
    has_1w = '1.weight' in params
    has_1b = '1.bias' in params
    if has_0w and has_1w:
        V = sd[params['0.weight']]
        U = sd[params['1.weight']]
        W_merged = U @ V
        merged[f'{prefix}.weight'] = W_merged
        if has_1b:
            merged[f'{prefix}.bias'] = sd[params['1.bias']]
        print(f'Merged: {prefix} -> {W_merged.shape}')
    else:
        for pk, k in params.items():
            merged[k] = sd[k]

for k in sorted(sd.keys()):
    if k not in merged:
        m = re.match(r'^(.+)\.\d+\.(weight|bias)$', k)
        if m and m.group(1) in seq_groups:
            continue
        merged[k] = sd[k]

print(f'\nMerged keys: {len(merged)}, Original: {len(sd)}')
torch.save(merged, os.path.join(dst_dir, 'modified_plaintext_model_state_dict.pth'))

args_snapshot = {
    "model": "deit-s", "use_approx_attn": True, "approx_attn_mode": "uniform",
    "use_square_gelu": True, "square_activation_mode": "fixed_square",
    "use_mask_pruning": True, "eval_pruning_mode": "compare_network_tie",
    "eval_tie_policy": "lowest_index", "embed_dim": 384, "depth": 12,
    "num_heads": 6, "base_rate": 0.7, "input_size": 224,
    "imagenet_default_mean_and_std": True, "crop_pct": 0.875,
    "data_path": os.environ.get("VAL_DATA_PATH", str(REPO_ROOT / "data" / "val")),
    "nb_classes": 2, "lrd_rank": 192, "lrd_merged": True,
}
with open(os.path.join(dst_dir, 'args_snapshot.json'), 'w') as f:
    json.dump(args_snapshot, f, indent=2)

# Verify forward
sys.path.insert(0, '/home/yclcg/Transshield_final')
from models.dyvit import VisionTransformerDiffPruning as ViT
model = ViT(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
    mlp_ratio=4, qkv_bias=True, norm_layer=torch.nn.LayerNorm,
    pruning_loc=[3, 6, 9], token_ratio=[0.7, 0.49, 0.343],
    distill=False, act_layer='square', use_approx_attn=True,
    use_square_gelu=True, square_activation_mode='fixed_square',
    use_mask_pruning=True)
missing, unexpected = model.load_state_dict(merged, strict=False)
print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    print(f'Logits: {logits.tolist()}, Finite: {torch.isfinite(logits).all().item()}')
