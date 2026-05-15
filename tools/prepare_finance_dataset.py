#!/usr/bin/env python3
"""
Prepare a synthetic credit card fraud detection dataset for Transshield.

Generates a binary classification dataset with realistic statistical properties:
- Class 0 (normal): majority, ~99.8%
- Class 1 (fraud): minority, ~0.2%

Each sample has 30 numerical features (similar to Kaggle Credit Card Fraud dataset).
Features are encoded as 224x224 grayscale PNG images in imagefolder format,
so the existing ViT training pipeline can consume them directly.

Feature-to-image encoding:
- 30 real features → spread across 14x14 = 196 patch positions
- Each patch (16x16 pixels) gets a constant value proportional to the feature
- Remaining 166 positions are zero (pad)
- This creates a sparse "image" where the first ~30 patches encode features
"""

import argparse
import os
import numpy as np
from PIL import Image
from pathlib import Path


def generate_synthetic_finance_data(n_normal=5000, n_fraud=100, n_features=30, seed=42):
    """Generate synthetic financial data mimicking credit card fraud patterns."""
    rng = np.random.RandomState(seed)
    
    # Normal transactions: centered around 0, moderate variance
    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    # Add some structure: features 0-9 are important, 10-19 moderate, 20-29 noisy
    X_normal[:, 0:10] *= 2.0
    X_normal[:, 10:20] *= 1.0
    X_normal[:, 20:30] *= 0.5
    
    # Fraud transactions: shifted distribution in key features
    X_fraud = rng.randn(n_fraud, n_features).astype(np.float32)
    X_fraud[:, 0:5] += 3.0  # Key fraud indicators are higher
    X_fraud[:, 5:10] -= 2.0
    X_fraud[:, 10:20] *= 1.5
    X_fraud[:, 20:30] *= 0.5
    
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0] * n_normal + [1] * n_fraud)
    
    # Shuffle
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    
    # Normalize to [-1, 1] range per feature
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 2 * (X - X_min) / (X_max - X_min + 1e-8) - 1
    
    return X, y


def features_to_image(features, img_size=224, patch_size=16):
    """Convert 30 features to a 224x224 grayscale image.
    
    Each feature maps to one 16x16 patch in the 14x14 grid.
    Remaining patches are zero.
    """
    n_patches = (img_size // patch_size) ** 2  # 196
    n_features = len(features)
    
    # Create patch-level values: features mapped to [0, 255]
    patch_values = np.zeros(n_patches, dtype=np.float32)
    patch_values[:n_features] = (features + 1) / 2 * 255  # [-1,1] -> [0,255]
    
    # Expand to full image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(n_patches):
        row = i // (img_size // patch_size)
        col = i % (img_size // patch_size)
        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        img[r0:r1, c0:c1] = np.clip(patch_values[i], 0, 255).astype(np.uint8)
    
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/finance_fraud_detection")
    parser.add_argument("--n-train-normal", type=int, default=5000)
    parser.add_argument("--n-train-fraud", type=int, default=100)
    parser.add_argument("--n-val-normal", type=int, default=1000)
    parser.add_argument("--n-val-fraud", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Generate training data
    print("Generating synthetic financial data...")
    X_train, y_train = generate_synthetic_finance_data(
        args.n_train_normal, args.n_train_fraud, args.n_features, args.seed
    )
    X_val, y_val = generate_synthetic_finance_data(
        args.n_val_normal, args.n_val_fraud, args.n_features, args.seed + 1
    )
    
    print(f"  Train: {len(y_train)} samples ({(y_train==0).sum()} normal, {(y_train==1).sum()} fraud)")
    print(f"  Val:   {len(y_val)} samples ({(y_val==0).sum()} normal, {(y_val==1).sum()} fraud)")
    
    # Create imagefolder structure
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        for cls_name in ["normal", "fraud"]:
            (output_dir / split / cls_name).mkdir(parents=True, exist_ok=True)
        
        for i in range(len(y)):
            cls_name = "fraud" if y[i] == 1 else "normal"
            img = features_to_image(X[i])
            img_path = output_dir / split / cls_name / f"sample_{i:06d}.png"
            Image.fromarray(img, mode='L').save(str(img_path))
    
    print(f"Saved to {output_dir}/")
    print(f"  train/normal: {(y_train==0).sum()} images")
    print(f"  train/fraud:  {(y_train==1).sum()} images")
    print(f"  val/normal:   {(y_val==0).sum()} images")
    print(f"  val/fraud:    {(y_val==1).sum()} images")
    
    # Also save raw features for later analysis
    np.save(str(output_dir / "train_features.npy"), X_train)
    np.save(str(output_dir / "train_labels.npy"), y_train)
    np.save(str(output_dir / "val_features.npy"), X_val)
    np.save(str(output_dir / "val_labels.npy"), y_val)
    print("Raw features saved as .npy files")


if __name__ == "__main__":
    main()
