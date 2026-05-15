#!/usr/bin/env python3
"""
Finance dataset v3: Generate images that look like real grayscale photos.

Key insight: ViT was pretrained on ImageNet. We need images with realistic
spatial structure (textures, edges, gradients) for the ViT to work.

Strategy:
- Class 0 (normal): Generate images with smooth gradients + mild noise
  (like a normal chest X-ray - uniform tissue)
- Class 1 (fraud): Generate images with sharp edges + high contrast patches
  (like abnormal findings - localized bright/dark spots)

Each image also embeds 30 financial features in pixel intensity patterns,
but the class signal comes from the overall texture/structure, not raw feature values.
"""
import os
import numpy as np
from PIL import Image

np.random.seed(42)
IMG_SIZE = 224

def make_normal_image(features, rng):
    """Generate image resembling normal tissue: smooth, moderate intensity."""
    # Base: smooth gradient
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    base = 80 + 40 * np.sin(2 * np.pi * x / IMG_SIZE) * np.cos(2 * np.pi * y / IMG_SIZE)
    
    # Add mild texture noise (Gaussian)
    noise = rng.randn(IMG_SIZE, IMG_SIZE) * 15
    
    # Modulate with features (subtle, spread across image)
    for i in range(min(len(features), 10)):
        freq = (i + 1) * 0.3
        base += features[i] * 5 * np.sin(2 * np.pi * freq * x / IMG_SIZE)
    
    img = base + noise
    return np.clip(img, 0, 255).astype(np.uint8)

def make_fraud_image(features, rng):
    """Generate image resembling abnormal findings: high contrast, sharp edges."""
    # Base: darker overall
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    base = 50 + 20 * np.sin(2 * np.pi * x / IMG_SIZE * 1.5)
    
    # Add bright anomaly patches (like calcifications or masses)
    n_patches = rng.randint(2, 5)
    for _ in range(n_patches):
        cx, cy = rng.randint(30, IMG_SIZE - 30, 2)
        r = rng.randint(10, 30)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        intensity = rng.uniform(120, 200)
        base += intensity * np.exp(-dist**2 / (2 * r**2))
    
    # Add structured noise
    noise = rng.randn(IMG_SIZE, IMG_SIZE) * 25
    
    # Modulate with features
    for i in range(min(len(features), 10)):
        freq = (i + 1) * 0.5
        base += features[i] * 8 * np.cos(2 * np.pi * freq * y / IMG_SIZE)
    
    img = base + noise
    return np.clip(img, 0, 255).astype(np.uint8)

def generate_dataset(n_normal, n_fraud, split_name, output_dir):
    """Generate a dataset split."""
    rng = np.random.RandomState(42 if split_name == 'train' else 123)
    
    for cls_idx, cls_name, n_samples, gen_func in [
        (0, 'normal', n_normal, make_normal_image),
        (1, 'fraud', n_fraud, make_fraud_image)
    ]:
        dst_dir = os.path.join(output_dir, split_name, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        for i in range(n_samples):
            features = rng.randn(30).astype(np.float32)
            if cls_idx == 1:  # fraud
                features[0:10] += 2.0  # fraud indicators
            
            img = gen_func(features, rng)
            Image.fromarray(img, mode='L').save(
                os.path.join(dst_dir, f"sample_{i:06d}.png")
            )
        
        print(f"  {split_name}/{cls_name}: {n_samples} images")

def main():
    output_dir = "data/finance_fraud_v3"
    
    print("Generating finance v3 dataset (realistic image-like encoding)...")
    generate_dataset(500, 500, 'train', output_dir)  # Balanced
    generate_dataset(100, 100, 'val', output_dir)     # Balanced
    
    print(f"\nSaved to {output_dir}/")
    print("Train: 500 normal + 500 fraud (balanced)")
    print("Val: 100 normal + 100 fraud (balanced)")

if __name__ == "__main__":
    main()
