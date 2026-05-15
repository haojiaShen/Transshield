#!/usr/bin/env python3
"""
将 LRD (低秩分解) 应用到 KD 学生模型
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import model definitions
from models.dyvit import VisionTransformerDiffPruning

def build_student_model():
    """创建学生模型"""
    student = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=2,
        act_layer='fixed_square',
        use_approx_attn=True,
        approx_attn_mode='uniform',
        fp32_attention=True,
        pruning_loc=[3, 6, 9],
        token_ratio=[1.0, 1.0, 1.0],  # No pruning
        distill=True,
        secure_static_depth=0,
        secure_static_skip_pruning=True,
    )
    return student

def load_student_model(bundle_dir):
    """加载学生模型"""
    model = build_student_model()
    state_dict_path = os.path.join(bundle_dir, 'modified_plaintext_model_state_dict.pth')
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def svd_decompose_linear(layer, rank):
    """对线性层进行 SVD 分解"""
    if not isinstance(layer, nn.Linear):
        return layer, None, None
    
    weight = layer.weight.data  # [out_features, in_features]
    bias = layer.bias.data if layer.bias is not None else None
    
    # SVD 分解
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    
    # 截断到指定 rank
    U_r = U[:, :rank]  # [out_features, rank]
    S_r = S[:rank]  # [rank]
    Vh_r = Vh[:rank, :]  # [rank, in_features]
    
    # 构建分解后的权重
    # W = U_r @ diag(S_r) @ Vh_r
    # 可以写成两个矩阵: U_r @ diag(sqrt(S_r)) 和 diag(sqrt(S_r)) @ Vh_r
    sqrt_S = torch.sqrt(S_r)
    U_new = U_r * sqrt_S.unsqueeze(0)  # [out_features, rank]
    V_new = sqrt_S.unsqueeze(1) * Vh_r  # [rank, in_features]
    
    return U_new, V_new, bias

def apply_lrd_to_model(model, rank):
    """对模型的所有线性层应用 LRD"""
    decomposed_layers = {}
    original_params = 0
    decomposed_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_params += module.weight.numel()
            if module.bias is not None:
                original_params += module.bias.numel()
            
            U, V, bias = svd_decompose_linear(module, rank)
            if U is not None and V is not None:
                decomposed_layers[name] = (U, V, bias)
                decomposed_params += U.numel() + V.numel()
                if bias is not None:
                    decomposed_params += bias.numel()
    
    print(f"Original parameters: {original_params}")
    print(f"Decomposed parameters: {decomposed_params}")
    print(f"Compression ratio: {decomposed_params / original_params:.2%}")
    
    return decomposed_layers

def save_decomposed_model(decomposed_layers, output_path):
    """保存分解后的模型"""
    # 转换为可序列化的格式
    save_dict = {}
    for name, (U, V, bias) in decomposed_layers.items():
        save_dict[f"{name}.U"] = U
        save_dict[f"{name}.V"] = V
        if bias is not None:
            save_dict[f"{name}.bias"] = bias
    
    torch.save(save_dict, output_path)
    print(f"Decomposed model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', default='artifacts/kd_deit_tiny_bundled')
    parser.add_argument('--output-dir', default='artifacts/kd_lrd_combined')
    parser.add_argument('--rank', type=int, default=96, help='LRD rank (default: 96 = embed_dim/2)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("Loading student model...")
    model = load_student_model(args.bundle_dir)
    
    # 应用 LRD
    print(f"Applying LRD with rank={args.rank}...")
    decomposed_layers = apply_lrd_to_model(model, args.rank)
    
    # 保存分解后的模型
    output_path = os.path.join(args.output_dir, 'kd_lrd_combined_state_dict.pth')
    save_decomposed_model(decomposed_layers, output_path)
    
    # 保存配置
    config = {
        'bundle_dir': args.bundle_dir,
        'rank': args.rank,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'decomposed_layers': list(decomposed_layers.keys()),
        'timestamp': '2026-05-14',
    }
    
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to: {config_path}")
    print("Done!")

if __name__ == "__main__":
    main()
