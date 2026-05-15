#!/usr/bin/env python3
"""
为 DeiT-Tiny 学生模型生成专门的校准参数
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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

def generate_calibration(model, data_root, output_dir, num_samples=100):
    """生成校准参数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 加载数据
    val_dataset = datasets.ImageFolder(f"{data_root}/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 收集 logits
    all_logits = []
    all_targets = []
    
    print(f"Collecting logits from {num_samples} samples...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i * 32 >= num_samples:
                break
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_logits.append(outputs.numpy())
            all_targets.append(targets.numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算统计信息
    logits_mean = all_logits.mean(axis=0)
    logits_std = all_logits.std(axis=0)
    
    # 计算 score (class1_score = logits @ [-1, 1])
    scores = all_logits[:, 1] - all_logits[:, 0]
    
    # 计算最优阈值
    from sklearn.metrics import accuracy_score
    best_accuracy = 0
    best_threshold = 0
    
    for threshold in np.linspace(scores.min(), scores.max(), 100):
        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(all_targets, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # 生成校准参数
    calibration = {
        'manifest_type': 'transshield_e2e_output_calibration_v0',
        'model_type': 'deit-t',
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'bias': float(best_threshold),
        'threshold': 0.5,
        'weights': [-1.0, 1.0],
        'score_rule': 'class1_score = logits @ [-1, 1] + public_class1_logit_bias',
        'note': 'Student model (DeiT-Tiny) dedicated calibration',
        'statistics': {
            'logits_mean': logits_mean.tolist(),
            'logits_std': logits_std.tolist(),
            'scores_mean': float(scores.mean()),
            'scores_std': float(scores.std()),
            'best_accuracy': float(best_accuracy),
            'best_threshold': float(best_threshold),
            'num_samples': len(all_logits),
        },
        'timestamp': '2026-05-14',
    }
    
    # 保存校准参数
    output_path = os.path.join(output_dir, 'student_calibration.json')
    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"Calibration saved to: {output_path}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Scores mean: {scores.mean():.4f}")
    print(f"Scores std: {scores.std():.4f}")
    
    return calibration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', default='artifacts/kd_deit_tiny_bundled')
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT', str(REPO_ROOT / 'data')))
    parser.add_argument('--output-dir', default='artifacts/kd_deit_tiny_calibration')
    parser.add_argument('--num-samples', type=int, default=100)
    args = parser.parse_args()
    
    # 加载模型
    print("Loading student model...")
    model = load_student_model(args.bundle_dir)
    
    # 生成校准
    print("Generating calibration...")
    calibration = generate_calibration(model, args.data_root, args.output_dir, args.num_samples)
    
    print("Done!")

if __name__ == "__main__":
    main()
