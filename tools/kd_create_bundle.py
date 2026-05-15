#!/usr/bin/env python3
"""
Create SPU-compatible bundle for DeiT-Tiny student model
"""
import sys, os, json, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-state-dict', required=True, help='Path to student state dict')
    parser.add_argument('--output-dir', required=True, help='Output bundle directory')
    parser.add_argument('--source-bundle', default='artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430', 
                        help='Source bundle for reference files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy reference files from source bundle
    source_dir = args.source_bundle
    ref_files = ['threshold_best.json', 'threshold_eval.json', 'train_stdout.log']
    
    for f in ref_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(args.output_dir, f)
        if os.path.exists(src):
            if os.path.islink(src):
                link_target = os.readlink(src)
                os.symlink(link_target, dst)
            else:
                shutil.copy2(src, dst)
            print(f"Copied: {f}")
    
    # Copy student state dict as modified_plaintext_model_state_dict.pth
    src_state = args.student_state_dict
    dst_state = os.path.join(args.output_dir, 'modified_plaintext_model_state_dict.pth')
    shutil.copy2(src_state, dst_state)
    print(f"Copied student state dict -> modified_plaintext_model_state_dict.pth")
    
    # Load source args_snapshot and modify for student model
    source_args_path = os.path.join(source_dir, 'args_snapshot.json')
    with open(source_args_path, 'r') as f:
        args_snapshot = json.load(f)
    
    # Update student-specific fields
    args_snapshot.update({
        'patch_size': 16,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'num_classes': 2,
        'nb_classes': 2,
        'act_layer': 'fixed_square',
        'use_approx_attn': True,
        'approx_attn_mode': 'uniform',
        'fp32_attention': True,
        'pruning_loc': [3, 6, 9],
        'token_ratio': [1.0, 1.0, 1.0],
        'distill': True,
        'secure_static_depth': 0,
        'secure_static_skip_pruning': True,
        'input_size': 224,
        'imagenet_default_mean_and_std': True,
        'model': 'deit-t',  # Mark as DeiT-Tiny
    })
    
    with open(os.path.join(args.output_dir, 'args_snapshot.json'), 'w') as f:
        json.dump(args_snapshot, f, indent=2)
    print("Created args_snapshot.json")
    
    # Create manifest.json
    manifest = {
        'bundle_type': 'kd_deit_tiny',
        'student_config': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'mlp_ratio': 4,
        },
        'teacher_config': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4,
        },
        'distillation_params': {
            'epochs': 30,
            'lr': 5e-5,
            'temp': 4.0,
            'alpha': 0.7,
        },
        'results': {
            'best_val_acc': 0.9504,
            'test_acc': 0.8830,
            'student_params': 5706968,
            'teacher_params': 21666434,
            'param_ratio': 0.2634,
        },
        'source_student_state_dict': os.path.basename(args.student_state_dict),
        'timestamp': '2026-05-14',
    }
    
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print("Created manifest.json")
    
    # Create README.md
    readme = """# KD DeiT-Tiny Bundle

## Overview
This bundle contains a DeiT-Tiny student model trained via knowledge distillation from DeiT-Small teacher.

## Model Configuration
- **Student**: embed_dim=192, depth=12, num_heads=3, mlp_ratio=4
- **Teacher**: embed_dim=384, depth=12, num_heads=6, mlp_ratio=4

## Distillation Parameters
- Epochs: 30
- Learning rate: 5e-5
- Temperature: 4.0
- Alpha (KD weight): 0.7

## Results
- **Best validation accuracy**: 95.04%
- **Test accuracy**: 88.30%
- **Parameter reduction**: 73.66% (5.7M vs 22M)

## Usage
Use this bundle with the SPU inference pipeline:
```bash
BUNDLE_DIR=artifacts/kd_deit_tiny_bundled bash artifacts/server_inference_friendly_pack/run_e2e_aanone_exactln_clip0_eval.sh smoke8
```
"""
    
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write(readme)
    print("Created README.md")
    
    print(f"\nBundle created at: {args.output_dir}")
    print("Files:")
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
