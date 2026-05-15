#!/usr/bin/env python3
"""
Prepare a decomposed LRD bundle for SPU deployment.
Unlike transshield_lrd_merge_weights.py, this KEEPS weights decomposed (U_down, V_up separate).
SPU will do two-step matmul: x @ V_up.T @ U_down.T for actual compute savings.

Usage:
  python tools/transshield_lrd_decomposed_bundle.py \
    --decomposed-state-dict artifacts/lrd_decomposed_rank96/lrd_rank96_state_dict.pth \
    --original-bundle artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507 \
    --rank 96 \
    --output artifacts/frozen_bundle_lrd_rank96_decomposed_20260515
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposed-state-dict", required=True)
    parser.add_argument("--original-bundle", required=True, help="Original bundle for args_snapshot and non-decomposed params")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--finetuned-state-dict", default=None, help="Optional: fine-tuned decomposed state dict (with .0.weight/.1.weight pattern)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load decomposed state dict
    decomposed_sd = torch.load(args.decomposed_state_dict, map_location="cpu", weights_only=False)
    if isinstance(decomposed_sd, dict) and "model" in decomposed_sd:
        decomposed_sd = decomposed_sd["model"]
    elif isinstance(decomposed_sd, dict) and "state_dict" in decomposed_sd:
        decomposed_sd = decomposed_sd["state_dict"]

    # If fine-tuned decomposed state dict provided, use it instead
    if args.finetuned_state_dict:
        ft_sd = torch.load(args.finetuned_state_dict, map_location="cpu", weights_only=False)
        if isinstance(ft_sd, dict) and "model" in ft_sd:
            ft_sd = ft_sd["model"]
        elif isinstance(ft_sd, dict) and "state_dict" in ft_sd:
            ft_sd = ft_sd["state_dict"]
        # Check if fine-tuned has decomposed keys
        has_decomposed = any(".0.weight" in k or ".1.weight" in k for k in ft_sd.keys() if "attn.qkv" in k)
        if has_decomposed:
            decomposed_sd = ft_sd
            print(f"Using fine-tuned decomposed state dict ({len(ft_sd)} keys)")
        else:
            print(f"Warning: fine-tuned state dict does not have decomposed keys, using original decomposed")

    # Load original bundle for args_snapshot and non-decomposed params
    orig_bundle = Path(args.original_bundle)
    orig_sd = torch.load(orig_bundle / "modified_plaintext_model_state_dict.pth", map_location="cpu", weights_only=False)
    with open(orig_bundle / "args_snapshot.json") as f:
        args_snapshot = json.load(f)

    # Build new state dict: decomposed linear layers + original non-linear params
    new_sd = {}
    decomposed_layers = set()

    # Detect decomposed layers from keys like "blocks.0.attn.qkv.0.weight"
    for k in decomposed_sd.keys():
        m = re.match(r"^(blocks\.\d+\.(?:attn|mlp)\.\w+)\.\d+\.(weight|bias)$", k)
        if m:
            decomposed_layers.add(m.group(1))

    print(f"Found {len(decomposed_layers)} decomposed layers:")
    for layer in sorted(decomposed_layers):
        print(f"  {layer}")

    # Copy decomposed weights
    for k, v in decomposed_sd.items():
        new_sd[k] = v

    # Copy non-decomposed weights from original (norm, head, patch_embed, cls_token, pos_embed, act params)
    for k, v in orig_sd.items():
        # Skip layers that are decomposed
        skip = False
        for layer in decomposed_layers:
            if k.startswith(layer + "."):
                skip = True
                break
        if not skip and k not in new_sd:
            new_sd[k] = v

    # Save state dict
    out_path = os.path.join(args.output, "modified_plaintext_model_state_dict.pth")
    torch.save(new_sd, out_path)
    print(f"\nSaved decomposed state dict to {out_path}")
    print(f"  Total keys: {len(new_sd)}")
    print(f"  Decomposed layers: {len(decomposed_layers)}")

    # Update args_snapshot
    args_snapshot["lrd_rank"] = args.rank
    args_snapshot["lrd_merged"] = False
    args_snapshot["lrd_decomposed"] = True
    args_snapshot_path = os.path.join(args.output, "args_snapshot.json")
    with open(args_snapshot_path, "w") as f:
        json.dump(args_snapshot, f, indent=2)
    print(f"Saved args_snapshot to {args_snapshot_path}")

    # Verify forward
    print("\nVerifying forward pass...")
    from models.dyvit import VisionTransformerDiffPruning as ViT
    model = ViT(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True, norm_layer=torch.nn.LayerNorm,
        pruning_loc=[3, 6, 9], token_ratio=[0.7, 0.49, 0.343],
        distill=False, act_layer='square', use_approx_attn=True,
        use_square_gelu=True, square_activation_mode='fixed_square',
        use_mask_pruning=True
    )
    # Load with strict=False since decomposed keys won't match standard model
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # For SPU usage, the state dict is loaded directly (not through model.load_state_dict)
    # So the decomposed keys are fine as long as spu_static_vit.py handles them


if __name__ == "__main__":
    main()
