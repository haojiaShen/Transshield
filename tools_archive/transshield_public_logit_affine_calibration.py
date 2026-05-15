#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(path: Path) -> Tuple[List[float], List[int]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    scores = [float(row["logit_1"]) - float(row["logit_0"]) for row in rows]
    targets = [int(row["target"]) for row in rows]
    if not scores:
        raise ValueError(f"empty static-logits CSV: {path}")
    return scores, targets


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_cross_entropy(scores: List[float], targets: List[int], scale: float, bias: float) -> float:
    total = 0.0
    for score, target in zip(scores, targets):
        logit = scale * score + bias
        if logit >= 0:
            total += math.log1p(math.exp(-logit)) + (1 - target) * logit
        else:
            total += math.log1p(math.exp(logit)) - target * logit
    return float(total / len(targets))


def accuracy(scores: List[float], targets: List[int], scale: float, bias: float) -> float:
    correct = sum(int((scale * score + bias >= 0.0) == bool(target)) for score, target in zip(scores, targets))
    return float(correct / len(targets) * 100.0)


def binary_auc(scores: List[float], targets: List[int], scale: float) -> float:
    # Positive scale preserves ranking; negative scale would invert ranking and is intentionally unsupported.
    scaled_scores = [scale * score for score in scores]
    positives = [score for score, target in zip(scaled_scores, targets) if target == 1]
    negatives = [score for score, target in zip(scaled_scores, targets) if target == 0]
    if not positives or not negatives:
        return float("nan")
    greater = 0
    equal = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                greater += 1
            elif pos == neg:
                equal += 1
    return float((greater + 0.5 * equal) / (len(positives) * len(negatives)))


def fit_bias_for_scale(scores: List[float], targets: List[int], scale: float) -> float:
    bias = 0.0
    for _ in range(100):
        grad = 0.0
        hess = 0.0
        for score, target in zip(scores, targets):
            prob = sigmoid(scale * score + bias)
            grad += prob - target
            hess += prob * (1.0 - prob)
        if hess <= 1e-12:
            break
        step = grad / hess
        bias -= step
        if abs(step) < 1e-12:
            break
    return float(bias)


def log_space(min_value: float, max_value: float, count: int) -> List[float]:
    if min_value <= 0 or max_value <= 0:
        raise ValueError("scale range must be positive")
    if count < 2:
        return [float(min_value)]
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    return [10 ** (log_min + index * (log_max - log_min) / (count - 1)) for index in range(count)]


def load_min_accuracy(args: argparse.Namespace, scores: List[float], targets: List[int]) -> float:
    if args.min_accuracy is not None:
        return float(args.min_accuracy)
    if not args.baseline_calibration_json:
        return 0.0
    payload = json.loads(Path(args.baseline_calibration_json).expanduser().resolve().read_text(encoding="utf-8"))
    weights = payload.get("weights") or [-1.0, 1.0]
    if len(weights) != 2:
        raise ValueError("baseline calibration weights must have length 2")
    baseline_scale = float(weights[1])
    baseline_bias = float(payload.get("bias", 0.0))
    return accuracy(scores, targets, baseline_scale, baseline_bias)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit public affine calibration for binary E2E logits.")
    parser.add_argument("--static-logits-csv", required=True)
    parser.add_argument("--baseline-calibration-json", default="")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--scale-min", type=float, default=0.01)
    parser.add_argument("--scale-max", type=float, default=100.0)
    parser.add_argument("--scale-grid-count", type=int, default=3001)
    parser.add_argument("--label", default="public_logit_affine_calibration")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-calibration-json", required=True)
    args = parser.parse_args()

    csv_path = Path(args.static_logits_csv).expanduser().resolve()
    scores, targets = load_rows(csv_path)
    min_accuracy = load_min_accuracy(args, scores, targets)

    raw = {
        "scale": 1.0,
        "bias": 0.0,
        "accuracy": accuracy(scores, targets, 1.0, 0.0),
        "ce_loss": binary_cross_entropy(scores, targets, 1.0, 0.0),
        "auc": binary_auc(scores, targets, 1.0),
    }

    candidates = []
    for scale in log_space(args.scale_min, args.scale_max, args.scale_grid_count):
        bias = fit_bias_for_scale(scores, targets, scale)
        item = {
            "scale": float(scale),
            "bias": float(bias),
            "accuracy": accuracy(scores, targets, scale, bias),
            "ce_loss": binary_cross_entropy(scores, targets, scale, bias),
            "auc": binary_auc(scores, targets, scale),
        }
        if item["accuracy"] + 1e-9 >= min_accuracy:
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"no affine calibration candidate met min_accuracy={min_accuracy}")
    best = min(candidates, key=lambda item: (item["ce_loss"], -item["accuracy"]))

    report = {
        "manifest_type": "transshield_public_logit_affine_calibration_report_v0",
        "label": args.label,
        "inputs": {
            "static_logits_csv": str(csv_path),
            "baseline_calibration_json": str(Path(args.baseline_calibration_json).expanduser().resolve())
            if args.baseline_calibration_json
            else None,
        },
        "sample_count": len(targets),
        "min_accuracy": min_accuracy,
        "raw": raw,
        "best_accuracy_constrained_affine": best,
        "deployment_note": {
            "operation": "replace score = logit_1 - logit_0 + bias with score = scale * (logit_1 - logit_0) + bias",
            "secure_friendly": True,
            "requires_retraining": False,
            "changes_auc_ranking": False,
        },
    }
    calibration = {
        "manifest_type": "transshield_e2e_output_calibration_v0",
        "weights": [-float(best["scale"]), float(best["scale"])],
        "bias": float(best["bias"]),
        "threshold": 0.5,
        "score_rule": "class1_score = logits @ [-scale, scale] + public_bias",
        "note": "Public affine calibration fitted on E2E static logits with an accuracy-preserving constraint.",
        "source_public_logit_affine_report": {
            "label": args.label,
            "static_logits_csv": str(csv_path),
            "sample_count": len(targets),
            "min_accuracy": min_accuracy,
            "accuracy": best["accuracy"],
            "ce_loss": best["ce_loss"],
        },
    }

    write_json(Path(args.output_json).expanduser().resolve(), report)
    write_json(Path(args.output_calibration_json).expanduser().resolve(), calibration)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
