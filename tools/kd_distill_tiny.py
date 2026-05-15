#!/usr/bin/env python3
"""
知识蒸馏：DeiT-Small(teacher) → DeiT-Tiny(student)
Teacher: embed_dim=384, depth=12, heads=6, params=22M
Student: embed_dim=192, depth=12, heads=3, params=5.7M
数据集: PneumoniaMNIST (binary classification)
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

# -- Import model definitions from existing codebase --
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher

# -- Config --
REPO_ROOT = Path(__file__).resolve().parents[1]
TEACHER_BUNDLE = str(REPO_ROOT / "artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430")
DATA_ROOT = os.environ.get("DATA_ROOT", str(REPO_ROOT / "data"))
OUT_DIR = str(REPO_ROOT / "artifacts/kd_deit_tiny")

# Student config (DeiT-Tiny)
STUDENT_CFG = dict(
    patch_size=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4,
    qkv_bias=True,
    num_classes=2,
    act_layer="fixed_square",
    use_approx_attn=True,
    approx_attn_mode="uniform",
    fp32_attention=True,
)

# Teacher config (DeiT-Small)
TEACHER_CFG = dict(
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    qkv_bias=True,
    num_classes=2,
)


def build_dataloaders(data_root, batch_size=32, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=val_transform)
    test_ds = datasets.ImageFolder(f"{data_root}/test", transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def load_teacher(bundle_path):
    """Load the frozen teacher model"""
    teacher = VisionTransformerTeacher(**TEACHER_CFG)
    state = torch.load(f"{bundle_path}/modified_plaintext_model_state_dict.pth", map_location="cpu")
    
    # Map keys from saved state to VisionTransformerTeacher
    teacher_state = teacher.state_dict()
    mapped = {}
    for k, v in state.items():
        if k in teacher_state and teacher_state[k].shape == v.shape:
            mapped[k] = v
        # Skip predictor/predictorlg keys (teacher doesn't have them)
    
    teacher.load_state_dict(mapped, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters())} params")
    return teacher


def create_student():
    """Create DeiT-Tiny student with same custom ops as teacher"""
    # Use VisionTransformerDiffPruning with no pruning (for distillation)
    student = VisionTransformerDiffPruning(
        **STUDENT_CFG,
        pruning_loc=[3, 6, 9],
        token_ratio=[1.0, 1.0, 1.0],  # No pruning during distillation
        distill=True,
        secure_static_depth=0,
        secure_static_skip_pruning=True,
    )
    print(f"Student created: {sum(p.numel() for p in student.parameters())} params")
    return student


def kd_loss_fn(student_logits, teacher_logits, targets, temp=4.0, alpha=0.7):
    """Knowledge Distillation loss: KL(soft) + CE(hard)"""
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temp, dim=-1)
    soft_student = F.log_softmax(student_logits / temp, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temp ** 2)
    
    # Hard targets
    ce_loss = F.cross_entropy(student_logits, targets)
    
    return alpha * kd_loss + (1 - alpha) * ce_loss


def evaluate(model, loader, device):
    """Evaluate accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_epoch(student, teacher, loader, optimizer, device, temp=4.0, alpha=0.7):
    """Train one epoch with KD"""
    student.train()
    total_loss = 0
    n_batches = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(imgs)
            # Handle tuple output from VisionTransformerTeacher
            if isinstance(teacher_out, tuple):
                teacher_out = teacher_out[0]
        
        # Student forward
        student_out = student(imgs)
        if isinstance(student_out, tuple):
            student_out = student_out[0]
        
        # KD loss
        loss = kd_loss_fn(student_out, teacher_out, labels, temp, alpha)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(DATA_ROOT, batch_size=32)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Load teacher
    teacher = load_teacher(TEACHER_BUNDLE).to(device)
    
    # Create student
    student = create_student().to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    
    # Training
    best_val_acc = 0
    history = []
    
    print(f"\n{'='*60}")
    print("Knowledge Distillation: DeiT-Small -> DeiT-Tiny")
    print(f"{'='*60}")
    
    for epoch in range(30):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(student, teacher, train_loader, optimizer, device)
        
        # Evaluate
        val_acc = evaluate(student, val_loader, device)
        scheduler.step()
        
        dt = time.time() - t0
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # Save best
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_acc": val_acc,
            }, f"{OUT_DIR}/kd_deit_tiny_best.pth")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "time": dt,
        })
        
        print(f"Epoch {epoch+1:02d}/30 | loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s {'*** BEST ***' if is_best else ''}")
    
    # Final test evaluation
    checkpoint = torch.load(f"{OUT_DIR}/kd_deit_tiny_best.pth", map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    test_acc = evaluate(student, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Best val acc:  {best_val_acc:.4f}")
    print(f"  Test acc:      {test_acc:.4f}")
    print(f"  Student params: {sum(p.numel() for p in student.parameters())}")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters())}")
    
    # Save final model in SPU-compatible format
    torch.save(student.state_dict(), f"{OUT_DIR}/kd_deit_tiny_state_dict.pth")
    
    # Save report
    report = {
        "student_config": STUDENT_CFG,
        "teacher_config": TEACHER_CFG,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "student_params": sum(p.numel() for p in student.parameters()),
        "teacher_params": sum(p.numel() for p in teacher.parameters()),
        "param_ratio": sum(p.numel() for p in student.parameters()) / sum(p.numel() for p in teacher.parameters()),
        "epochs": 30,
        "history": history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{OUT_DIR}/kd_distill_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
