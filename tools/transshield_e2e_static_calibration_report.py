#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binary_auc(scores, targets) -> float:
    positives = [score for score, target in zip(scores, targets) if target == 1]
    negatives = [score for score, target in zip(scores, targets) if target == 0]
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


def binary_f1(predictions, targets) -> float:
    tp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    fp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))


def accuracy(predictions, targets) -> float:
    return float(sum(int(pred == target) for pred, target in zip(predictions, targets)) / len(targets) * 100.0)


def cross_entropy(logit_pairs, targets) -> float:
    total = 0.0
    for logits, target in zip(logit_pairs, targets):
        z0, z1 = logits
        max_logit = max(z0, z1)
        total += -(z1 if target == 1 else z0) + (
            max_logit + math.log(math.exp(z0 - max_logit) + math.exp(z1 - max_logit))
        )
    return float(total / len(targets))


def candidate_thresholds(prob_1):
    values = sorted(set(float(value) for value in prob_1))
    thresholds = [0.0]
    thresholds.extend(values)
    thresholds.append(1.0)
    return thresholds


def best_threshold(prob_1, targets):
    best = None
    for threshold in candidate_thresholds(prob_1):
        predictions = [1 if value >= threshold else 0 for value in prob_1]
        acc = accuracy(predictions, targets)
        candidate = {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "f1": binary_f1(predictions, targets),
            "predicted_positive_count": int(sum(predictions)),
            "predicted_negative_count": int(len(predictions) - sum(predictions)),
        }
        if best is None:
            best = candidate
            continue
        if candidate["accuracy"] > best["accuracy"]:
            best = candidate
            continue
        if candidate["accuracy"] == best["accuracy"] and abs(candidate["threshold"] - 0.5) < abs(best["threshold"] - 0.5):
            best = candidate
    return best


def tensor_to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def load_targets_from_reference(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_sample") or []
    if not rows:
        raise ValueError(f"reference JSON has no per_sample rows: {path}")
    return [int(row["target"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a public calibration report for E2E static logits.")
    parser.add_argument("--candidate-pt", required=True)
    parser.add_argument("--reference-json", default="", help="optional plaintext reference JSON used when candidate PT omits targets")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-calibration-json", default="")
    parser.add_argument("--label", default="e2e_static")
    args = parser.parse_args()

    import torch

    payload = torch.load(Path(args.candidate_pt).expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("candidate PT must contain a dict payload")
    logits_tensor = payload.get("raw_logits_before_output_calibration", payload.get("logits"))
    targets_tensor = payload.get("targets")
    if logits_tensor is None:
        raise ValueError("candidate PT does not contain logits")
    if targets_tensor is None and not args.reference_json:
        raise ValueError("candidate PT does not contain targets; pass --reference-json or run preprocessing with --include-targets")

    logits = [[float(x) for x in row] for row in tensor_to_list(logits_tensor)]
    if targets_tensor is not None:
        targets = [int(value) for value in tensor_to_list(targets_tensor)]
    else:
        targets = load_targets_from_reference(Path(args.reference_json).expanduser().resolve())
    if len(targets) != len(logits):
        raise ValueError(f"logit/target count mismatch: {len(logits)} != {len(targets)}")
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    prob_1 = [float(value) for value in probabilities[:, 1].tolist()]
    scores = [row[1] - row[0] for row in logits]
    argmax_predictions = [1 if row[1] >= row[0] else 0 for row in logits]
    best = best_threshold(prob_1, targets)
    threshold = float(best["threshold"])
    threshold_predictions = [1 if value >= threshold else 0 for value in prob_1]
    class1_bias = math.log((1.0 - threshold) / threshold) if 0.0 < threshold < 1.0 else 0.0
    calibrated_logits = [[z0, z1 + class1_bias] for z0, z1 in logits]
    calibrated_argmax = [1 if z1 >= z0 else 0 for z0, z1 in calibrated_logits]

    rows = []
    sample_ids = tensor_to_list(payload.get("sample_ids")) or [f"sample_{index:06d}" for index in range(len(targets))]
    for index, (sample_id, logit_pair, prob, target) in enumerate(zip(sample_ids, logits, prob_1, targets)):
        rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "target": target,
                "logit_0": logit_pair[0],
                "logit_1": logit_pair[1],
                "prob_1": prob,
                "argmax_prediction": argmax_predictions[index],
                "threshold_prediction": threshold_predictions[index],
                "calibrated_argmax_prediction": calibrated_argmax[index],
            }
        )

    report = {
        "manifest_type": "transshield_e2e_static_calibration_report_v0",
        "label": args.label,
        "candidate_pt": str(Path(args.candidate_pt).expanduser().resolve()),
        "reference_json": str(Path(args.reference_json).expanduser().resolve()) if args.reference_json else None,
        "sample_count": len(targets),
        "metrics": {
            "argmax_accuracy": accuracy(argmax_predictions, targets),
            "argmax_f1": binary_f1(argmax_predictions, targets),
            "auc": binary_auc(scores, targets),
            "ce_loss": cross_entropy(logits, targets),
            "best_threshold": threshold,
            "best_threshold_accuracy": best["accuracy"],
            "best_threshold_f1": best["f1"],
            "calibrated_argmax_accuracy": accuracy(calibrated_argmax, targets),
            "calibrated_argmax_f1": binary_f1(calibrated_argmax, targets),
            "calibrated_ce_loss": cross_entropy(calibrated_logits, targets),
        },
        "public_logit_bias_calibration": {
            "effective_class1_logit_bias": float(class1_bias),
            "source_eval_binary_threshold": threshold,
            "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
        },
    }

    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_calibration_json:
        write_json(
            Path(args.output_calibration_json).expanduser().resolve(),
            {
                "manifest_type": "transshield_e2e_output_calibration_v0",
                "weights": [-1.0, 1.0],
                "bias": float(class1_bias),
                "threshold": 0.5,
                "source_eval_binary_threshold": threshold,
                "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
                "note": "Static-path public logit bias derived from E2E static logits.",
            },
        )
    if args.output_csv:
        csv_path = Path(args.output_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.output_md:
        md = [
            f"# {args.label} Calibration",
            "",
            f"- sample_count: `{len(targets)}`",
            f"- argmax_accuracy: `{report['metrics']['argmax_accuracy']}`",
            f"- best_threshold: `{threshold}`",
            f"- best_threshold_accuracy: `{report['metrics']['best_threshold_accuracy']}`",
            f"- auc: `{report['metrics']['auc']}`",
            f"- effective_class1_logit_bias: `{class1_bias}`",
        ]
        Path(args.output_md).expanduser().resolve().write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
