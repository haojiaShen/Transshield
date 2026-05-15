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
        logit = scale * (score + bias)
        if logit >= 0:
            total += math.log1p(math.exp(-logit)) + (1 - target) * logit
        else:
            total += math.log1p(math.exp(logit)) - target * logit
    return float(total / len(targets))


def accuracy(scores: List[float], targets: List[int], scale: float, bias: float) -> float:
    correct = sum(int((scale * (score + bias) >= 0.0) == bool(target)) for score, target in zip(scores, targets))
    return float(correct / len(targets) * 100.0)


def fit_temperature(scores: List[float], targets: List[int], bias: float, scale_min: float, scale_max: float, grid_count: int) -> Dict:
    best = None
    log_min = math.log10(scale_min)
    log_max = math.log10(scale_max)
    for index in range(grid_count):
        scale = 10 ** (log_min + index * (log_max - log_min) / max(grid_count - 1, 1))
        item = {
            "scale": float(scale),
            "bias": float(bias),
            "accuracy": accuracy(scores, targets, scale, bias),
            "ce_loss": binary_cross_entropy(scores, targets, scale, bias),
        }
        if best is None or item["ce_loss"] < best["ce_loss"] - 1e-12 or (
            abs(item["ce_loss"] - best["ce_loss"]) <= 1e-12 and item["accuracy"] > best["accuracy"]
        ):
            best = item
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a boundary-preserving public temperature calibration for binary E2E logits.")
    parser.add_argument("--static-logits-csv", required=True)
    parser.add_argument("--bias", type=float, required=True, help="Public class-1 logit bias to preserve exactly.")
    parser.add_argument("--scale-min", type=float, default=0.01)
    parser.add_argument("--scale-max", type=float, default=100.0)
    parser.add_argument("--scale-grid-count", type=int, default=5001)
    parser.add_argument("--label", default="public_logit_temperature_calibration")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-calibration-json", required=True)
    args = parser.parse_args()

    csv_path = Path(args.static_logits_csv).expanduser().resolve()
    scores, targets = load_rows(csv_path)
    raw = {
        "scale": 1.0,
        "bias": float(args.bias),
        "accuracy": accuracy(scores, targets, 1.0, float(args.bias)),
        "ce_loss": binary_cross_entropy(scores, targets, 1.0, float(args.bias)),
    }
    best = fit_temperature(scores, targets, float(args.bias), args.scale_min, args.scale_max, args.scale_grid_count)
    report = {
        "manifest_type": "transshield_public_logit_temperature_calibration_report_v0",
        "label": args.label,
        "inputs": {
            "static_logits_csv": str(csv_path),
            "bias": float(args.bias),
        },
        "sample_count": len(targets),
        "raw": raw,
        "best_temperature": best,
        "deployment_note": {
            "operation": "replace score = logit_1 - logit_0 + bias with scale * (logit_1 - logit_0 + bias)",
            "secure_friendly": True,
            "requires_retraining": False,
            "changes_auc_ranking": False,
            "preserves_decision_boundary": True,
        },
    }
    calibration = {
        "manifest_type": "transshield_e2e_output_calibration_v0",
        "weights": [-float(best["scale"]), float(best["scale"])],
        "bias": float(best["scale"] * float(args.bias)),
        "threshold": 0.5,
        "score_rule": "class1_score = scale * (logits @ [-1, 1] + public_bias)",
        "note": "Public temperature calibration that preserves the bias-only decision boundary exactly.",
        "source_public_logit_temperature_report": {
            "label": args.label,
            "static_logits_csv": str(csv_path),
            "sample_count": len(targets),
            "raw_accuracy": raw["accuracy"],
            "raw_ce_loss": raw["ce_loss"],
            "best_scale": best["scale"],
            "best_accuracy": best["accuracy"],
            "best_ce_loss": best["ce_loss"],
        },
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    write_json(Path(args.output_calibration_json).expanduser().resolve(), calibration)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
