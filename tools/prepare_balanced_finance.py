#!/usr/bin/env python3
"""
Prepare balanced finance fraud detection dataset.
Use balanced subset: equal number of fraud and normal samples.
Convert existing imagefolder to balanced imagefolder.
"""
import os
import shutil
import random
import numpy as np
from PIL import Image

random.seed(42)

SRC = "data/finance_fraud_detection"
BAL = "data/finance_fraud_balanced"

# Create balanced dataset with equal classes
for split in ['train', 'val']:
    src_normal = os.path.join(SRC, split, 'normal')
    src_fraud = os.path.join(SRC, split, 'fraud')
    
    normal_files = sorted([f for f in os.listdir(src_normal) if f.endswith('.png')])
    fraud_files = sorted([f for f in os.listdir(src_fraud) if f.endswith('.png')])
    
    print(f"{split}: normal={len(normal_files)}, fraud={len(fraud_files)}")
    
    # Use all fraud samples, subsample normal to match
    n_fraud = len(fraud_files)
    if split == 'train':
        # For train: use all fraud + sample same number of normal
        sampled_normal = random.sample(normal_files, min(n_fraud * 2, len(normal_files)))
    else:
        # For val: use all fraud + sample same number of normal
        sampled_normal = random.sample(normal_files, min(n_fraud * 2, len(normal_files)))
    
    for cls_name, files in [('normal', sampled_normal), ('fraud', fraud_files)]:
        dst_dir = os.path.join(BAL, split, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        src_dir = os.path.join(SRC, split, cls_name)
        for f in files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        
        print(f"  {cls_name}: {len(files)} samples")

print("\nBalanced dataset created at:", BAL)
print("Train: train/normal + train/fraud")
print("Val: val/normal + val/fraud")

# Count final
for split in ['train', 'val']:
    for cls in ['normal', 'fraud']:
        d = os.path.join(BAL, split, cls)
        n = len([f for f in os.listdir(d) if f.endswith('.png')])
        print(f"  {split}/{cls}: {n}")
