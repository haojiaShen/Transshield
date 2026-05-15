#!/usr/bin/env python3
"""
Transshield Low-Rank Decomposition (LRD)
==========================================
Apply SVD-based low-rank decomposition to DeiT-Small linear layers.

Based on: LRD-MPC (2026) - replacing one large matmul with two smaller ones.

Target layers per Transformer block:
  - Attention.qkv:   (384, 1152) → (384, r) × (r, 1152)
  - Attention.proj:   (384, 384)  → (384, r) × (r, 384)
  - MLP.fc1:          (384, 1536) → (384, r) × (r, 1536)
  - MLP.fc2:          (1536, 384) → (1536, r) × (r, 384)

Usage:
  python tools/transshield_low_rank_decompose.py \\
    --checkpoint artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507/modified_plaintext_model_state_dict.pth \\
    --rank 192 \\
    --output artifacts/lrd_decomposed_rank192/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add repo to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def svd_decompose_linear(linear_layer, rank):
    """Decompose a nn.Linear layer into two smaller layers using SVD.
    
    Original: y = xW^T + b  where W is (out_features, in_features)
    Decomposed: y = (x @ V_r^T) @ U_r^T + b  where:
        U_r: (out_features, r)  - left singular vectors
        V_r: (r, in_features)   - right singular vectors (transposed)
    
    Args:
        linear_layer: nn.Linear to decompose
        rank: target rank r
    
    Returns:
        nn.Sequential(linear_down, linear_up) where:
            linear_down: in_features → r
            linear_up: r → out_features
    """
    W = linear_layer.weight.data  # (out_features, in_features)
    has_bias = linear_layer.bias is not None
    
    out_features, in_features = W.shape
    
    # Clamp rank to valid range
    max_rank = min(out_features, in_features)
    rank = min(rank, max_rank)
    
    # SVD: W = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Truncate to rank r
    U_r = U[:, :rank]          # (out_features, r)
    S_r = S[:rank]             # (r,)
    Vh_r = Vh[:rank, :]        # (r, in_features)
    
    # Absorb sqrt(S) into both sides for balanced decomposition
    sqrt_S = torch.sqrt(S_r)
    W_down = (Vh_r * sqrt_S[:, None]).contiguous()  # (r, in_features)
    W_up = (U_r * sqrt_S[None, :]).contiguous()      # (out_features, r)
    
    # Create new layers
    linear_down = nn.Linear(in_features, rank, bias=False)
    linear_down.weight.data = W_down
    
    linear_up = nn.Linear(rank, out_features, bias=has_bias)
    linear_up.weight.data = W_up
    if has_bias:
        linear_up.bias.data = linear_layer.bias.data.clone()
    
    return nn.Sequential(linear_down, linear_up)


def get_decomposable_layers(model):
    """Find all linear layers that can be decomposed.
    
    Returns list of (name, module) pairs for:
      - blocks.*.attn.qkv
      - blocks.*.attn.proj
      - blocks.*.mlp.fc1
      - blocks.*.mlp.fc2
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip small layers (head, patch_embed)
            if module.weight.shape[0] <= 2 or module.weight.shape[1] <= 2:
                continue
            # Skip embedding layers
            if 'patch_embed' in name or 'head' in name or 'pos_embed' in name:
                continue
            targets.append((name, module))
    return targets


def decompose_model(model, rank, target_layers=None):
    """Apply SVD decomposition to specified layers of the model.
    
    Args:
        model: the model to decompose
        rank: target rank for all layers
        target_layers: list of layer name patterns to decompose (None = all linear in blocks)
    
    Returns:
        model with decomposed layers, decomposition stats
    """
    if target_layers is None:
        target_layers = ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
    
    stats = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        # Check if this layer matches any target pattern
        matched = any(pattern in name for pattern in target_layers)
        if not matched:
            continue
        
        orig_params = module.weight.numel()
        if module.bias is not None:
            orig_params += module.bias.numel()
        
        out_f, in_f = module.weight.shape
        max_rank = min(out_f, in_f)
        actual_rank = min(rank, max_rank)
        
        # Decompose
        decomposed = svd_decompose_linear(module, actual_rank)
        
        # Count new params
        new_params = sum(p.numel() for p in decomposed.parameters())
        
        # Replace in model
        # Navigate to parent module and replace the attribute
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], decomposed)
        
        compression_ratio = new_params / orig_params if orig_params > 0 else 1.0
        
        stats.append({
            'layer': name,
            'orig_shape': [out_f, in_f],
            'rank': actual_rank,
            'orig_params': orig_params,
            'new_params': new_params,
            'compression_ratio': round(compression_ratio, 4),
        })
        
        print(f"  {name}: ({out_f}, {in_f}) → rank {actual_rank} | "
              f"params {orig_params} → {new_params} ({compression_ratio:.2%})")
    
    return model, stats


def evaluate_accuracy(model, val_loader, device='cpu'):
    """Quick accuracy evaluation."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Transshield LRD: SVD-based low-rank decomposition')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model state dict (.pth)')
    parser.add_argument('--rank', type=int, default=192,
                        help='Target rank for decomposition (default: 192)')
    parser.add_argument('--ranks', type=str, default=None,
                        help='Comma-separated ranks to test (e.g., "96,128,192,256")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for decomposed model')
    parser.add_argument('--eval-data', type=str, default=None,
                        help='Validation data path for accuracy evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--layers', type=str, default='attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
                        help='Comma-separated layer patterns to decompose')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    from models.dyvit import VisionTransformerDiffPruning as create_model
    model = create_model(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=2, pruning_loc=[3,6,9], token_ratio=[0.7, 0.49, 0.3429], distill=True, act_layer="fixed_square", use_mask_pruning=False, use_approx_attn=True, approx_attn_mode="uniform", fp32_attention=True, eval_pruning_mode="topk_argsort", eval_tie_policy="lowest_index", secure_static_depth=12, secure_static_skip_pruning=False)
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Handle DataParallel prefix
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('student.', '')
        cleaned[k] = v
    
    model.load_state_dict(cleaned, strict=False)
    model.to(args.device)
    model.eval()
    
    # Count original params
    orig_total = sum(p.numel() for p in model.parameters())
    print(f"Original model params: {orig_total:,}")
    
    # Get target layers
    target_layers = [l.strip() for l in args.layers.split(',')]
    
    # If ranks specified, do sweep; otherwise just do single rank
    ranks_to_test = [args.rank]
    if args.ranks:
        ranks_to_test = [int(r) for r in args.ranks.split(',')]
    
    results = []
    
    for rank in ranks_to_test:
        print(f"\n{'='*60}")
        print(f"Testing rank = {rank}")
        print(f"{'='*60}")
        
        # Reload model for each rank test
        model_copy = create_model(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=2, pruning_loc=[3,6,9], token_ratio=[0.7, 0.49, 0.3429], distill=True, act_layer="fixed_square", use_mask_pruning=False, use_approx_attn=True, approx_attn_mode="uniform", fp32_attention=True, eval_pruning_mode="topk_argsort", eval_tie_policy="lowest_index", secure_static_depth=12, secure_static_skip_pruning=False)
        model_copy.load_state_dict(cleaned, strict=False)
        model_copy.to(args.device)
        model_copy.eval()
        
        # Decompose
        t0 = time.time()
        model_decomposed, stats = decompose_model(model_copy, rank, target_layers)
        decompose_time = time.time() - t0
        
        # Count decomposed params
        new_total = sum(p.numel() for p in model_decomposed.parameters())
        overall_ratio = new_total / orig_total
        
        print(f"\n  Rank {rank}: params {orig_total:,} → {new_total:,} ({overall_ratio:.2%})")
        print(f"  Decompose time: {decompose_time:.2f}s")
        
        # Evaluate accuracy if data available
        accuracy = None
        if args.eval_data:
            try:
                from torchvision import datasets, transforms
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                val_dataset = datasets.ImageFolder(args.eval_data, transform=transform)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
                accuracy = evaluate_accuracy(model_decomposed, val_loader, args.device)
                print(f"  Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"  Accuracy eval failed: {e}")
        
        # Save if this is the target rank
        if rank == args.rank and args.output:
            os.makedirs(args.output, exist_ok=True)
            save_path = os.path.join(args.output, f'lrd_rank{rank}_state_dict.pth')
            torch.save(model_decomposed.state_dict(), save_path)
            print(f"\n  Saved to: {save_path}")
            
            # Save stats
            stats_path = os.path.join(args.output, f'lrd_rank{rank}_stats.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'rank': rank,
                    'orig_params': orig_total,
                    'new_params': new_total,
                    'compression_ratio': round(overall_ratio, 4),
                    'decompose_time_sec': round(decompose_time, 2),
                    'accuracy': accuracy,
                    'layer_stats': stats,
                }, f, indent=2)
            print(f"  Stats saved to: {stats_path}")
        
        results.append({
            'rank': rank,
            'orig_params': orig_total,
            'new_params': new_total,
            'compression_ratio': round(overall_ratio, 4),
            'accuracy': accuracy,
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        print(f"  rank={r['rank']:4d} | params={r['new_params']:>10,} ({r['compression_ratio']:.2%}) | acc={acc_str}")


if __name__ == '__main__':
    main()
