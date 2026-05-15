#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import math


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_targets(reference_json: Path) -> List[int]:
    payload = load_json(reference_json)
    rows = payload.get("per_sample") or []
    if not rows:
        raise ValueError(f"plaintext reference has no per_sample rows: {reference_json}")
    return [int(row["target"]) for row in rows]


def load_images(image_list: Path, count: int) -> List[str]:
    if not image_list.exists():
        return [f"sample_{index:06d}" for index in range(count)]
    rows = image_list.read_text(encoding="utf-8").splitlines()
    if len(rows) != count:
        raise ValueError(f"image list count mismatch: {len(rows)} != {count}")
    return rows


def apply_calibration(raw_logits, calibration: Dict) -> Tuple[List[int], List[float]]:
    import torch

    weights = torch.tensor(calibration["weights"], dtype=raw_logits.dtype)
    bias = float(calibration.get("bias", 0.0))
    scores = raw_logits.matmul(weights) + bias
    predictions = (scores >= float(calibration.get("decision_boundary", 0.0))).long()
    return [int(value) for value in predictions.tolist()], [float(value) for value in scores.tolist()]


def accuracy(predictions: List[int], targets: List[int]) -> float:
    return float(sum(int(pred == target) for pred, target in zip(predictions, targets)) / len(targets) * 100.0)


def binary_cross_entropy(scores: List[float], targets: List[int]) -> float:
    total = 0.0
    for score, target in zip(scores, targets):
        if score >= 0:
            total += math.log1p(math.exp(-score)) + (1 - target) * score
        else:
            total += math.log1p(math.exp(score)) - target * score
    return float(total / len(targets))


def summarize_scores(scores: List[float], targets: List[int], predictions: List[int]) -> Dict:
    margins = [abs(score) for score in scores]
    wrong_margins = [margin for margin, pred, target in zip(margins, predictions, targets) if pred != target]
    correct_margins = [margin for margin, pred, target in zip(margins, predictions, targets) if pred == target]

    def mean(values: List[float]):
        return None if not values else float(sum(values) / len(values))

    return {
        "mean_abs_margin": mean(margins),
        "mean_abs_margin_correct": mean(correct_margins),
        "mean_abs_margin_wrong": mean(wrong_margins),
        "wrong_count": int(len(wrong_margins)),
        "low_margin_count_abs_lt_0_25": int(sum(1 for value in margins if value < 0.25)),
        "low_margin_count_abs_lt_0_5": int(sum(1 for value in margins if value < 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E2E raw logits under multiple public output calibrations.")
    parser.add_argument("--candidate-pt", required=True)
    parser.add_argument("--plaintext-reference-json", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--calibration", action="append", required=True, help="label=calibration_json")
    parser.add_argument("--label", default="e2e_calibration_drift")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    import torch

    candidate_pt = Path(args.candidate_pt).expanduser().resolve()
    payload = torch.load(candidate_pt, map_location="cpu")
    raw_logits = payload.get("raw_logits_before_output_calibration", payload.get("logits"))
    if raw_logits is None:
        raise ValueError(f"candidate PT has no logits: {candidate_pt}")
    raw_logits = raw_logits.detach().cpu().float()
    targets = load_targets(Path(args.plaintext_reference_json).expanduser().resolve())
    if len(targets) != int(raw_logits.shape[0]):
        raise ValueError(f"target/logit count mismatch: {len(targets)} != {int(raw_logits.shape[0])}")
    images = load_images(Path(args.image_list).expanduser().resolve(), len(targets))

    calibrations = {}
    for item in args.calibration:
        if "=" not in item:
            raise ValueError("--calibration must use label=path")
        label, raw_path = item.split("=", 1)
        calibrations[label] = load_json(Path(raw_path).expanduser().resolve())

    calibration_results = {}
    per_sample = []
    for label, calibration in calibrations.items():
        predictions, scores = apply_calibration(raw_logits, calibration)
        calibration_results[label] = {
            "accuracy": accuracy(predictions, targets),
            "binary_cross_entropy": binary_cross_entropy(scores, targets),
            "calibration": calibration,
            "score_summary": summarize_scores(scores, targets, predictions),
        }
        for index, (image, target, prediction, score) in enumerate(zip(images, targets, predictions, scores)):
            if len(per_sample) <= index:
                z0 = float(raw_logits[index, 0].item())
                z1 = float(raw_logits[index, 1].item())
                per_sample.append(
                    {
                        "index": index,
                        "image": image,
                        "target": target,
                        "raw_logit_0": z0,
                        "raw_logit_1": z1,
                        "raw_score_logit1_minus_logit0": z1 - z0,
                    }
                )
            per_sample[index][f"{label}_score"] = score
            per_sample[index][f"{label}_prediction"] = prediction
            per_sample[index][f"{label}_correct"] = prediction == target
            per_sample[index][f"{label}_abs_margin"] = abs(score)

    labels = list(calibrations)
    if labels:
        primary = labels[0]
        wrong_rows = [row for row in per_sample if not row.get(f"{primary}_correct", False)]
        hard_wrong = sorted(wrong_rows, key=lambda row: row.get(f"{primary}_abs_margin", 0.0), reverse=True)
        boundary_wrong = sorted(wrong_rows, key=lambda row: row.get(f"{primary}_abs_margin", 0.0))
    else:
        hard_wrong = []
        boundary_wrong = []

    report = {
        "manifest_type": "transshield_e2e_calibration_drift_report_v0",
        "label": args.label,
        "candidate_pt": str(candidate_pt),
        "sample_count": len(targets),
        "calibration_results": calibration_results,
        "image_list_sha_preview": {
            "first": images[:5],
            "last": images[-5:],
        },
        "wrong_sample_recommendations": {
            "primary_label": labels[0] if labels else None,
            "largest_wrong_margins": hard_wrong[:8],
            "smallest_wrong_margins": boundary_wrong[:8],
        },
        "per_sample_preview": per_sample[:12],
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in per_sample for key in row})
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
