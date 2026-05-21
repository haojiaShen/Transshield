#!/usr/bin/env python3
"""Train and evaluate a plaintext CNN baseline on the ImageFolder chest X-ray dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--val-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "densenet121", "mobilenet_v3_small"])
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--class-weight-mode", default="inverse_freq", choices=["none", "inverse_freq"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        image = Image.open(handle)
        return image.convert("RGB")


def build_transforms(input_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_datasets(args: argparse.Namespace):
    train_transform, val_transform = build_transforms(args.input_size)
    train_dataset = datasets.ImageFolder(str(args.train_data), transform=train_transform, loader=rgb_loader)
    val_dataset = datasets.ImageFolder(str(args.val_data), transform=val_transform, loader=rgb_loader)
    return train_dataset, val_dataset


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        last_linear = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_linear.in_features, num_classes)
        return model
    raise ValueError(f"unsupported arch: {arch}")


def maybe_freeze_backbone(model: nn.Module, arch: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if arch.startswith("resnet"):
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif arch == "densenet121":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif arch == "mobilenet_v3_small":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def build_class_weights(dataset: datasets.ImageFolder, mode: str, device: torch.device):
    if mode == "none":
        return None
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for value_index, count in enumerate(counts):
            if count <= 1:
                continue
            mask = inverse == value_index
            ranks[mask] = ranks[mask].mean()
    pos_rank_sum = ranks[positives].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def search_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    unique_scores = np.unique(scores)
    candidates = np.concatenate(([0.0], unique_scores, [1.0]))
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        accuracy = float((predictions == labels).mean())
        if accuracy > best_accuracy or (math.isclose(accuracy, best_accuracy) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += int(images.size(0))
    return {
        "loss": total_loss / max(total_samples, 1),
        "argmax_accuracy": total_correct / max(total_samples, 1),
        "sample_count": total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_paths: list[str] = []
    sample_offset = 0
    for images, targets in loader:
        batch_size = images.size(0)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        total_samples += batch_size
        all_scores.append(probs.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        dataset_samples = loader.dataset.samples[sample_offset : sample_offset + batch_size]
        all_paths.extend(path for path, _ in dataset_samples)
        sample_offset += batch_size
    scores = np.concatenate(all_scores) if all_scores else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    preds = np.concatenate(all_preds) if all_preds else np.zeros((0,), dtype=np.int64)
    threshold, threshold_accuracy = search_best_threshold(scores, labels) if len(scores) else (0.5, 0.0)
    auc = binary_auc(labels, scores) if len(scores) else None
    return {
        "loss": total_loss / max(total_samples, 1),
        "argmax_accuracy": total_correct / max(total_samples, 1),
        "threshold_accuracy": threshold_accuracy,
        "best_threshold": threshold,
        "auc": auc,
        "sample_count": total_samples,
        "scores": scores,
        "labels": labels,
        "preds": preds,
        "paths": all_paths,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, args: argparse.Namespace):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(payload, path)


def write_predictions_csv(path: Path, eval_payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path", "label", "prob_positive", "pred_argmax", "pred_threshold"])
        threshold = eval_payload["best_threshold"]
        for index, (image_path, label, score, pred_argmax) in enumerate(
            zip(eval_payload["paths"], eval_payload["labels"], eval_payload["scores"], eval_payload["preds"]),
            start=1,
        ):
            pred_threshold = int(score >= threshold)
            writer.writerow([index, image_path, int(label), float(score), int(pred_argmax), pred_threshold])


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.arch, num_classes=len(train_dataset.classes), pretrained=args.pretrained)
    if args.freeze_backbone:
        maybe_freeze_backbone(model, args.arch)
    model.to(device)

    class_weights = build_class_weights(train_dataset, args.class_weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None

    history_path = args.output_dir / "metrics_history.jsonl"
    best_ckpt_path = args.output_dir / "checkpoint_best.pth"
    last_ckpt_path = args.output_dir / "checkpoint_last.pth"
    summary_path = args.output_dir / "cnn_baseline_summary.json"
    prediction_csv = args.output_dir / "val_predictions.csv"
    best_threshold_json = args.output_dir / "best_threshold.json"

    best_val_acc = -1.0
    best_epoch = -1
    best_eval_payload = None
    start_time = time.time()

    with history_path.open("w", encoding="utf-8") as history_handle:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp)
            val_metrics = evaluate(model, val_loader, criterion, device, amp)
            epoch_metrics = {
                "epoch": epoch,
                "elapsed_sec": time.time() - epoch_start,
                "train_loss": train_metrics["loss"],
                "train_argmax_accuracy": train_metrics["argmax_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_argmax_accuracy": val_metrics["argmax_accuracy"],
                "val_threshold_accuracy": val_metrics["threshold_accuracy"],
                "val_auc": val_metrics["auc"],
                "val_best_threshold": val_metrics["best_threshold"],
            }
            history_handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            history_handle.flush()

            save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_acc, args)
            if args.save_every_epoch:
                save_checkpoint(args.output_dir / f"checkpoint_epoch{epoch}.pth", model, optimizer, epoch, best_val_acc, args)

            if val_metrics["argmax_accuracy"] > best_val_acc:
                best_val_acc = float(val_metrics["argmax_accuracy"])
                best_epoch = epoch
                best_eval_payload = val_metrics
                save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_acc, args)

    total_elapsed = time.time() - start_time
    if best_eval_payload is None:
        best_eval_payload = evaluate(model, val_loader, criterion, device, amp)
        best_epoch = args.epochs
        best_val_acc = float(best_eval_payload["argmax_accuracy"])

    write_predictions_csv(prediction_csv, best_eval_payload)
    best_threshold_json.write_text(
        json.dumps(
            {
                "best_threshold": best_eval_payload["best_threshold"],
                "best_threshold_accuracy": best_eval_payload["threshold_accuracy"],
                "val_auc": best_eval_payload["auc"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "arch": args.arch,
        "pretrained": bool(args.pretrained),
        "freeze_backbone": bool(args.freeze_backbone),
        "class_weight_mode": args.class_weight_mode,
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "classes": train_dataset.classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "amp": amp,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_argmax_accuracy": best_val_acc,
        "best_val_threshold_accuracy": best_eval_payload["threshold_accuracy"],
        "best_threshold": best_eval_payload["best_threshold"],
        "best_val_auc": best_eval_payload["auc"],
        "trainable_parameter_count": count_trainable_parameters(model),
        "elapsed_sec_total": total_elapsed,
        "sec_per_epoch": total_elapsed / max(args.epochs, 1),
        "outputs": {
            "history_jsonl": str(history_path),
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "prediction_csv": str(prediction_csv),
            "best_threshold_json": str(best_threshold_json),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
