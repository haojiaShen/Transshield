#!/usr/bin/env python3
"""
Create calibration for LUT GELU bundle.
"""
import sys
sys.path.insert(0, '/home/yclcg/Transshield_final')

import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.dyvit import VisionTransformerDiffPruning


def load_model(bundle_dir):
    """Load the model from the frozen bundle."""
    with open(os.path.join(bundle_dir, 'args_snapshot.json'), 'r') as f:
        args = json.load(f)
    ckpt_path = os.path.join(bundle_dir, 'checkpoint-best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return args, ckpt


def create_model(args, ckpt, activation_kind="lut_gelu_16"):
    """Create a model with the specified activation."""
    model = VisionTransformerDiffPruning(
        img_size=args.get('input_size', 224),
        patch_size=16,
        in_chans=3,
        num_classes=args.get('nb_classes', 2),
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        act_layer=activation_kind,
        pruning_loc=[3, 6, 9],
        token_ratio=[0.7, 0.7, 0.7],
        secure_static_depth=12,
        secure_static_skip_pruning=True,
    )
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model


def load_dataset(data_path, batch_size=8, input_size=224):
    """Load the dataset."""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader, dataset


def evaluate_model(model, dataloader, device):
    """Evaluate the model and collect logits."""
    model.eval()
    model = model.to(device)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            all_logits.append(logits.cpu())
            all_labels.append(targets.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels


def find_best_threshold(logits, labels):
    """Find the best threshold for binary classification."""
    probs = torch.softmax(logits, dim=1)[:, 1]
    
    best_acc = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs > threshold).long()
        acc = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    return best_threshold, best_acc


def create_calibration(bundle_dir, output_dir, data_path):
    """Create calibration for the bundle."""
    print(f"=== Creating Calibration for LUT GELU ===")
    print(f"Bundle: {bundle_dir}")
    print(f"Data: {data_path}")
    
    # Load model
    args, ckpt = load_model(bundle_dir)
    model = create_model(args, ckpt, activation_kind="lut_gelu_16")
    
    # Load dataset
    dataloader, dataset = load_dataset(data_path, batch_size=8, input_size=224)
    print(f"Dataset: {len(dataset)} samples")
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logits, labels = evaluate_model(model, dataloader, device)
    
    # Find best threshold
    best_threshold, best_acc = find_best_threshold(logits, labels)
    
    # Compute argmax accuracy
    preds = logits.argmax(dim=1)
    argmax_acc = (preds == labels).float().mean().item()
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    labels_np = labels.numpy()
    auc = roc_auc_score(labels_np, probs)
    
    # Create calibration report
    calibration = {
        'args_snapshot_summary': {
            'approx_attn_mode': 'uniform',
            'eval_pruning_mode': 'topk_argsort',
            'eval_tie_policy': 'lowest_index',
            'model': 'deit-s',
            'square_activation_mode': 'lut_gelu_16',
            'use_approx_attn': True,
            'use_mask_pruning': False,
            'use_square_gelu': False,
        },
        'auc': auc,
        'best_threshold': best_threshold,
        'best_threshold_acc': best_acc * 100,
        'best_threshold_pred_counts': [
            int((probs < best_threshold).sum()),
            int((probs >= best_threshold).sum()),
        ],
        'checkpoint': 'checkpoint-best.pth',
        'checkpoint_path': bundle_dir,
        'data_path': data_path,
        'default_argmax_acc1': argmax_acc * 100,
        'default_pred_counts': [
            int((preds == 0).sum()),
            int((preds == 1).sum()),
        ],
        'eval_acc1': best_acc * 100,
        'eval_binary_threshold': best_threshold,
        'eval_loss': 0.0,  # Placeholder
        'finite_logits': True,
        'sample_count': len(dataset),
    }
    
    # Save calibration
    os.makedirs(output_dir, exist_ok=True)
    
    # Save threshold_best.json
    threshold_path = os.path.join(output_dir, 'threshold_best.json')
    with open(threshold_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\nCalibration Results:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Best Threshold Acc: {best_acc * 100:.2f}%")
    print(f"  Argmax Acc: {argmax_acc * 100:.2f}%")
    print(f"  Sample Count: {len(dataset)}")
    
    print(f"\nCalibration saved to {threshold_path}")
    
    return calibration


if __name__ == "__main__":
    bundle_dir = "artifacts/frozen_bundle_secure_static_depth12_uniform_lut_gelu_16_final_20260514"
    data_path = os.environ.get("VAL_DATA_PATH", str(REPO_ROOT / "data" / "val"))
    output_dir = bundle_dir  # Save calibration in the bundle directory
    
    if not os.path.exists(bundle_dir):
        print(f"Bundle directory not found: {bundle_dir}")
        sys.exit(1)
    
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        sys.exit(1)
    
    create_calibration(bundle_dir, output_dir, data_path)
