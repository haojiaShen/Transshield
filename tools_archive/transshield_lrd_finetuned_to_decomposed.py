#!/usr/bin/env python3
"""
Decompose fine-tuned merged LRD weights to decomposed form for SPU.
Takes the fine-tuned merged bundle and re-decomposes via SVD to get (U_down, V_up) pairs.
This avoids re-training: the fine-tuned weights are already adapted.
"""
import argparse, json, os, re, sys, time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def svd_decompose_weight(W, rank):
    """Decompose weight matrix W (out, in) into (down (rank, in), up (out, rank))."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    max_rank = min(rank, U.shape[1])
    U_r = U[:, :max_rank]
    S_r = S[:max_rank]
    Vh_r = Vh[:max_rank, :]
    sqrt_S = torch.sqrt(S_r)
    down = (Vh_r * sqrt_S[:, None]).contiguous()  # (rank, in)
    up = (U_r * sqrt_S[None, :]).contiguous()      # (out, rank)
    return down, up


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-bundle", required=True, help="Path to fine-tuned merged bundle")
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    bundle_dir = Path(args.merged_bundle)
    
    # Load merged state dict
    sd = torch.load(bundle_dir / "modified_plaintext_model_state_dict.pth", map_location="cpu", weights_only=False)
    with open(bundle_dir / "args_snapshot.json") as f:
        args_snapshot = json.load(f)
    
    # Layers to decompose
    target_suffixes = [".attn.qkv", ".attn.proj", ".mlp.fc1", ".mlp.fc2"]
    new_sd = {}
    decomposed_count = 0
    total_orig = 0
    total_new = 0
    
    for k, v in sd.items():
        matched = False
        for suffix in target_suffixes:
            if k.endswith(suffix + ".weight"):
                prefix = k[:-len(".weight")]
                bias_key = prefix + ".bias"
                W = v  # (out, in)
                bias = sd.get(bias_key, None)
                
                down, up = svd_decompose_weight(W, args.rank)
                new_sd[prefix + ".0.weight"] = down
                new_sd[prefix + ".1.weight"] = up
                if bias is not None:
                    new_sd[prefix + ".1.bias"] = bias
                
                recon = up @ down
                err = (W - recon).abs().max().item()
                print(f"  {prefix}: {W.shape} -> down{down.shape} + up{up.shape}, max_err={err:.6f}")
                total_orig += W.numel()
                total_new += down.numel() + up.numel()
                decomposed_count += 1
                matched = True
                break
        
        if not matched:
            # Non-decomposed layer (norm, head, patch_embed, etc.)
            # Skip bias of decomposed layers (already handled above)
            skip = False
            for suffix in target_suffixes:
                if k.endswith(suffix + ".bias"):
                    skip = True
                    break
            if not skip:
                new_sd[k] = v
    
    print(f"\nDecomposed {decomposed_count} layers")
    print(f"  Original params: {total_orig:,}")
    print(f"  Decomposed params: {total_new:,}")
    print(f"  Compression ratio: {total_new/total_orig:.4f}")
    
    # Save
    out_path = os.path.join(args.output, "modified_plaintext_model_state_dict.pth")
    torch.save(new_sd, out_path)
    print(f"Saved to {out_path} ({len(new_sd)} keys)")
    
    # Update args_snapshot
    args_snapshot["lrd_rank"] = args.rank
    args_snapshot["lrd_merged"] = False
    args_snapshot["lrd_decomposed"] = True
    with open(os.path.join(args.output, "args_snapshot.json"), "w") as f:
        json.dump(args_snapshot, f, indent=2)
    print(f"Saved args_snapshot")


if __name__ == "__main__":
    main()
