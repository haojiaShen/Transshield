#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def binary_f1(predictions: List[int], targets: List[int]) -> float:
    tp = sum(1 for p, y in zip(predictions, targets) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(predictions, targets) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(predictions, targets) if p == 0 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def binary_auc(scores: List[float], targets: List[int]) -> float:
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
    total = len(positives) * len(negatives)
    return float((greater + 0.5 * equal) / total)


def cross_entropy(logit_pairs: List[List[float]], targets: List[int]) -> float:
    total = 0.0
    for logits, target in zip(logit_pairs, targets):
        z0, z1 = logits
        m = max(z0, z1)
        total += -(z1 if target == 1 else z0) + (m + math.log(math.exp(z0 - m) + math.exp(z1 - m)))
    return float(total / len(targets))


def accuracy(predictions: List[int], targets: List[int]) -> float:
    return float(sum(1 for p, y in zip(predictions, targets) if p == y) / len(targets) * 100.0)


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(Path(args.plaintext_eval_csv).resolve())
    plaintext_eval = load_json(Path(args.plaintext_eval_json).resolve()) if args.plaintext_eval_json else {}
    threshold_payload = load_json(Path(args.threshold_json).resolve())
    threshold = float(threshold_payload["eval_binary_threshold"])
    class1_bias = math.log((1.0 - threshold) / threshold)
    effective_bias = class1_bias + float(args.epsilon)

    targets = [int(row["target"]) for row in rows]
    logits = [[float(row["logit_0"]), float(row["logit_1"])] for row in rows]
    original_scores = [float(row["logit_1"]) - float(row["logit_0"]) for row in rows]
    calibrated_logits = [[z0, z1 + effective_bias] for z0, z1 in logits]
    calibrated_scores = [z1 - z0 for z0, z1 in calibrated_logits]

    original_argmax = [1 if z1 >= z0 else 0 for z0, z1 in logits]
    threshold_predictions = [1 if float(row["prob_1"]) >= threshold else 0 for row in rows]
    calibrated_argmax = [1 if z1 >= z0 else 0 for z0, z1 in calibrated_logits]

    original_loss = cross_entropy(logits, targets)
    calibrated_loss = cross_entropy(calibrated_logits, targets)

    metrics = {
        "original_argmax_accuracy": accuracy(original_argmax, targets),
        "threshold_accuracy": accuracy(threshold_predictions, targets),
        "calibrated_argmax_accuracy": accuracy(calibrated_argmax, targets),
        "original_argmax_f1": binary_f1(original_argmax, targets),
        "threshold_f1": binary_f1(threshold_predictions, targets),
        "calibrated_argmax_f1": binary_f1(calibrated_argmax, targets),
        "original_auc": binary_auc(original_scores, targets),
        "calibrated_auc": binary_auc(calibrated_scores, targets),
        "original_ce_loss": original_loss,
        "calibrated_ce_loss": calibrated_loss,
        "calibrated_minus_original_argmax_accuracy": accuracy(calibrated_argmax, targets)
        - accuracy(original_argmax, targets),
        "calibrated_minus_original_ce_loss": calibrated_loss - original_loss,
    }

    status = (
        "public_bias_recovers_threshold_argmax"
        if metrics["calibrated_argmax_accuracy"] >= metrics["threshold_accuracy"] - 1e-9
        else "public_bias_nearly_recovers_threshold_argmax"
    )
    reason = (
        "公开 class-1 logit bias 将最优 threshold 等价搬到 argmax 边界；该操作只需一次公开加法，"
        "不改变 token pruning / SPU 主体算子。"
    )

    return {
        "status": status,
        "reason": reason,
        "inputs": {
            "plaintext_eval_json": str(Path(args.plaintext_eval_json).resolve()) if args.plaintext_eval_json else None,
            "plaintext_eval_csv": str(Path(args.plaintext_eval_csv).resolve()),
            "threshold_json": str(Path(args.threshold_json).resolve()),
        },
        "label": args.label or plaintext_eval.get("label"),
        "sample_count": len(rows),
        "sample_paths_sha256": plaintext_eval.get("sample_paths_sha256"),
        "threshold": threshold,
        "class1_logit_bias": class1_bias,
        "epsilon": float(args.epsilon),
        "effective_class1_logit_bias": effective_bias,
        "metrics": metrics,
        "deployment_note": {
            "operation": "add public scalar to class-1 logit before final argmax/threshold decision",
            "secure_friendly": True,
            "requires_retraining": False,
            "changes_auc_ranking": False,
        },
        "e2e_output_calibration": {
            "manifest_type": "transshield_e2e_output_calibration_v0",
            "weights": [-1.0, 1.0],
            "bias": effective_bias,
            "threshold": 0.5,
            "source_eval_binary_threshold": threshold,
            "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
            "note": (
                "Equivalent to adding the public scalar to class-1 logit before argmax; "
                "compatible with --output-calibration-json in the E2E OpenBumbleBee runner."
            ),
        },
    }


def build_e2e_output_calibration(report: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(report["e2e_output_calibration"])
    payload["source_public_logit_bias_report"] = {
        "label": report.get("label"),
        "status": report.get("status"),
        "plaintext_eval_json": report.get("inputs", {}).get("plaintext_eval_json"),
        "plaintext_eval_csv": report.get("inputs", {}).get("plaintext_eval_csv"),
        "threshold_json": report.get("inputs", {}).get("threshold_json"),
        "sample_count": report.get("sample_count"),
        "sample_paths_sha256": report.get("sample_paths_sha256"),
    }
    return payload


def build_markdown(report: Dict[str, Any]) -> str:
    m = report["metrics"]
    lines = [
        f"# Public Logit Bias Calibration: {report.get('label')}",
        "",
        "## 1. Conclusion",
        "",
        f"- status: `{report['status']}`",
        f"- reason: {report['reason']}",
        f"- threshold: `{report['threshold']}`",
        f"- class1_logit_bias: `{report['effective_class1_logit_bias']}`",
        "",
        "## 2. Metrics",
        "",
        f"- original_argmax_accuracy: `{m['original_argmax_accuracy']}`",
        f"- threshold_accuracy: `{m['threshold_accuracy']}`",
        f"- calibrated_argmax_accuracy: `{m['calibrated_argmax_accuracy']}`",
        f"- original_ce_loss: `{m['original_ce_loss']}`",
        f"- calibrated_ce_loss: `{m['calibrated_ce_loss']}`",
        f"- calibrated_minus_original_ce_loss: `{m['calibrated_minus_original_ce_loss']}`",
        f"- calibrated_auc: `{m['calibrated_auc']}`",
        "",
        "## 3. Deployment Note",
        "",
        "- Add the public scalar to the class-1 logit before final argmax.",
        "- This is a public post-processing add and does not change the secure ViT operator family.",
        "- AUC ranking is unchanged because the bias is a monotonic shift of the binary score.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public class-1 logit bias calibration from plaintext eval CSV.")
    parser.add_argument("--plaintext-eval-json", default="")
    parser.add_argument("--plaintext-eval-csv", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-e2e-calibration-json", default="")
    args = parser.parse_args()

    report = build_report(args)
    write_json(Path(args.output_json).resolve(), report)
    if args.output_md:
        write_text(Path(args.output_md).resolve(), build_markdown(report))
    if args.output_e2e_calibration_json:
        write_json(Path(args.output_e2e_calibration_json).resolve(), build_e2e_output_calibration(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
