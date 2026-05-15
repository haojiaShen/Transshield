#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import median


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_bridge_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--bridge must use label=plaintext_json,static_json")
    label, raw_paths = value.split("=", 1)
    paths = [Path(item).expanduser().resolve() for item in raw_paths.split(",")]
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("--bridge requires plaintext_json,static_json")
    for path in paths:
        if not path.exists():
            raise argparse.ArgumentTypeError(f"missing path: {path}")
    return label.strip(), paths[0], paths[1]


def parse_eval_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--eval must use label=candidate_pt,reference_json")
    label, raw_paths = value.split("=", 1)
    paths = [Path(item).expanduser().resolve() for item in raw_paths.split(",")]
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("--eval requires candidate_pt,reference_json")
    for path in paths:
        if not path.exists():
            raise argparse.ArgumentTypeError(f"missing path: {path}")
    return label.strip(), paths[0], paths[1]


def parse_compare_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--compare-calibration must use label=calibration_json")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"missing path: {path}")
    return label.strip(), path


def mean(values):
    return float(sum(values) / len(values))


def read_per_sample_json(path: Path):
    payload = load_json(path)
    rows = payload.get("per_sample") or []
    if not rows:
        raise ValueError(f"per_sample missing in {path}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    targets = [int(row["target"]) for row in rows]
    logits = [[float(item) for item in row["logits"]] for row in rows]
    scores = [float(row[1] - row[0]) for row in logits]
    return {
        "path": str(path),
        "sample_ids": sample_ids,
        "targets": targets,
        "scores": scores,
    }


def threshold_candidates(scores):
    ordered = sorted(set(float(item) for item in scores))
    if not ordered:
        return [0.0]
    candidates = [ordered[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    candidates.append(ordered[-1] + 1e-6)
    return candidates


def accuracy_from_scores(scores, targets, threshold):
    return float(sum(int((score >= threshold) == bool(target)) for score, target in zip(scores, targets)) / len(targets) * 100.0)


def best_threshold(scores, targets):
    best = None
    for threshold in threshold_candidates(scores):
        accuracy = accuracy_from_scores(scores, targets, threshold)
        item = {"threshold": float(threshold), "accuracy": accuracy}
        if best is None or item["accuracy"] > best["accuracy"] or (
            item["accuracy"] == best["accuracy"] and abs(item["threshold"]) < abs(best["threshold"])
        ):
            best = item
    return best


def fit_affine(x_values, y_values):
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_var = sum((item - x_mean) ** 2 for item in x_values)
    if x_var <= 0.0:
        raise ValueError("x variance is zero")
    cov = sum((x_item - x_mean) * (y_item - y_mean) for x_item, y_item in zip(x_values, y_values))
    slope = cov / x_var
    intercept = y_mean - slope * x_mean
    residuals = [abs((slope * x_item + intercept) - y_item) for x_item, y_item in zip(x_values, y_values)]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_abs_error": float(sum(residuals) / len(residuals)),
    }


def build_bridge_candidate(label, plaintext_json: Path, static_json: Path):
    plaintext = read_per_sample_json(plaintext_json)
    static = read_per_sample_json(static_json)
    if plaintext["sample_ids"] != static["sample_ids"] or plaintext["targets"] != static["targets"]:
        raise ValueError(f"sample mismatch in bridge dataset {label}")
    threshold = best_threshold(plaintext["scores"], plaintext["targets"])
    affine = fit_affine(plaintext["scores"], static["scores"])
    translated_static_threshold = affine["slope"] * threshold["threshold"] + affine["intercept"]
    bias = -translated_static_threshold
    return {
        "label": label,
        "bridge_plaintext_json": str(plaintext_json),
        "bridge_static_json": str(static_json),
        "sample_count": len(plaintext["targets"]),
        "plaintext_best_threshold": threshold,
        "affine_static_from_plaintext_score": affine,
        "translated_static_threshold": float(translated_static_threshold),
        "candidate_output_calibration": {
            "manifest_type": "transshield_e2e_output_calibration_v0",
            "weights": [-1.0, 1.0],
            "bias": float(bias),
            "threshold": 0.5,
            "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
            "note": "Bridge calibration translated from plaintext best-threshold into static/E2E raw-score space.",
        },
    }


def load_eval_dataset(label, candidate_pt: Path, reference_json: Path):
    import torch

    payload = torch.load(candidate_pt, map_location="cpu")
    raw_logits = payload.get("raw_logits_before_output_calibration")
    if raw_logits is None:
        raw_logits = payload.get("logits")
    if raw_logits is None:
        raise ValueError(f"candidate pt missing logits: {candidate_pt}")
    raw_logits = raw_logits.detach().cpu().float()
    targets = [int(row["target"]) for row in load_json(reference_json).get("per_sample") or []]
    if len(targets) != int(raw_logits.shape[0]):
        raise ValueError(f"target count mismatch for {label}")
    return {
        "label": label,
        "candidate_pt": str(candidate_pt),
        "reference_json": str(reference_json),
        "sample_count": len(targets),
        "targets": targets,
        "raw_scores": (raw_logits[:, 1] - raw_logits[:, 0]).tolist(),
    }


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_f1(predictions, targets):
    tp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    fp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))


def binary_auc(scores, targets):
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


def binary_cross_entropy(scores, targets):
    total = 0.0
    for score, target in zip(scores, targets):
        if score >= 0:
            total += math.log1p(math.exp(-score)) + (1 - target) * score
        else:
            total += math.log1p(math.exp(score)) - target * score
    return float(total / len(targets))


def evaluate_calibration_on_dataset(calibration, dataset):
    weights = calibration.get("weights") or [-1.0, 1.0]
    if len(weights) != 2:
        raise ValueError("calibration weights must have length 2")
    scale = float(weights[1])
    bias = float(calibration.get("bias", 0.0))
    scores = [scale * raw_score + bias for raw_score in dataset["raw_scores"]]
    predictions = [1 if score >= 0.0 else 0 for score in scores]
    return {
        "accuracy": float(sum(int(pred == target) for pred, target in zip(predictions, dataset["targets"])) / len(dataset["targets"]) * 100.0),
        "f1": binary_f1(predictions, dataset["targets"]),
        "auc": binary_auc(scores, dataset["targets"]),
        "binary_cross_entropy": binary_cross_entropy(scores, dataset["targets"]),
        "wrong_count": int(sum(int(pred != target) for pred, target in zip(predictions, dataset["targets"]))),
        "mean_abs_margin": float(mean([abs(score) for score in scores])),
    }


def weighted_accuracy(rows):
    total = 0.0
    count = 0.0
    for row in rows:
        accuracy = row.get("accuracy")
        sample_count = row.get("sample_count")
        if isinstance(accuracy, (int, float)) and isinstance(sample_count, (int, float)):
            total += float(accuracy) * float(sample_count)
            count += float(sample_count)
    return None if count <= 0.0 else total / count


def weighted_bce(rows):
    total = 0.0
    count = 0.0
    for row in rows:
        value = row.get("binary_cross_entropy")
        sample_count = row.get("sample_count")
        if isinstance(value, (int, float)) and isinstance(sample_count, (int, float)):
            total += float(value) * float(sample_count)
            count += float(sample_count)
    return None if count <= 0.0 else total / count


def bridge_aggregate_candidates(bridge_candidates):
    biases = [item["candidate_output_calibration"]["bias"] for item in bridge_candidates]
    return [
        {
            "label": "bridge_mean_bias",
            "source_bridge_labels": [item["label"] for item in bridge_candidates],
            "candidate_output_calibration": {
                "manifest_type": "transshield_e2e_output_calibration_v0",
                "weights": [-1.0, 1.0],
                "bias": float(mean(biases)),
                "threshold": 0.5,
                "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
                "note": "Mean bridge bias aggregated from multiple plaintext-static bridge datasets.",
            },
        },
        {
            "label": "bridge_median_bias",
            "source_bridge_labels": [item["label"] for item in bridge_candidates],
            "candidate_output_calibration": {
                "manifest_type": "transshield_e2e_output_calibration_v0",
                "weights": [-1.0, 1.0],
                "bias": float(median(biases)),
                "threshold": 0.5,
                "score_rule": "class1_score = logits @ [-1, 1] + public_class1_logit_bias",
                "note": "Median bridge bias aggregated from multiple plaintext-static bridge datasets.",
            },
        },
    ]


def build_markdown(report):
    lines = [
        "# E2E Plaintext Bridge Calibration",
        "",
        f"- label: `{report['label']}`",
        f"- status: `{report['judgement']['status']}`",
        f"- reason: {report['judgement']['reason']}",
        f"- best weighted-accuracy candidate: `{report['judgement']['best_accuracy_candidate']}`",
        f"- best weighted-BCE candidate: `{report['judgement']['best_bce_candidate']}`",
        "",
        "## Weighted Summary",
        "",
        "| candidate | weighted acc | weighted BCE |",
        "|---|---:|---:|",
    ]
    for item in report["candidate_summaries"]:
        lines.append(
            "| {label} | {acc:.6f} | {bce:.6f} |".format(
                label=item["label"],
                acc=item["weighted_accuracy"],
                bce=item["weighted_binary_cross_entropy"],
            )
        )
    lines.extend(["", "## Per Eval Dataset", "", "| candidate | eval dataset | acc | BCE | wrong |", "|---|---|---:|---:|---:|"])
    for item in report["candidate_summaries"]:
        for dataset in item["per_eval_dataset"]:
            lines.append(
                "| {label} | {dataset_label} | {acc:.6f} | {bce:.6f} | {wrong} |".format(
                    label=item["label"],
                    dataset_label=dataset["label"],
                    acc=dataset["accuracy"],
                    bce=dataset["binary_cross_entropy"],
                    wrong=dataset["wrong_count"],
                )
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Translate plaintext full-model boundary into static/E2E raw-score calibration and compare against existing public calibrations."
    )
    parser.add_argument("--bridge", action="append", type=parse_bridge_arg, required=True)
    parser.add_argument("--eval", action="append", type=parse_eval_arg, required=True)
    parser.add_argument("--compare-calibration", action="append", type=parse_compare_arg, default=[])
    parser.add_argument("--label", default="e2e_plaintext_bridge_calibration")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-best-calibration-json", default="")
    parser.add_argument("--output-best-bridge-calibration-json", default="")
    args = parser.parse_args()

    bridge_candidates = [build_bridge_candidate(label, plaintext_json, static_json) for label, plaintext_json, static_json in args.bridge]
    bridge_candidates.extend(bridge_aggregate_candidates(bridge_candidates))

    eval_datasets = [load_eval_dataset(label, candidate_pt, reference_json) for label, candidate_pt, reference_json in args.eval]

    all_candidates = []
    for item in bridge_candidates:
        all_candidates.append(
            {
                "label": item["label"],
                "kind": "bridge",
                "source": item,
                "calibration": item["candidate_output_calibration"],
            }
        )
    for label, path in args.compare_calibration:
        all_candidates.append(
            {
                "label": label,
                "kind": "compare",
                "source": {"path": str(path)},
                "calibration": load_json(path),
            }
        )

    candidate_summaries = []
    for candidate in all_candidates:
        per_eval_dataset = []
        for dataset in eval_datasets:
            metrics = evaluate_calibration_on_dataset(candidate["calibration"], dataset)
            per_eval_dataset.append(
                {
                    "label": dataset["label"],
                    "sample_count": dataset["sample_count"],
                    **metrics,
                }
            )
        candidate_summaries.append(
            {
                "label": candidate["label"],
                "kind": candidate["kind"],
                "source": candidate["source"],
                "calibration": candidate["calibration"],
                "per_eval_dataset": per_eval_dataset,
                "weighted_accuracy": weighted_accuracy(per_eval_dataset),
                "weighted_binary_cross_entropy": weighted_bce(per_eval_dataset),
            }
        )

    best_accuracy = max(candidate_summaries, key=lambda item: item["weighted_accuracy"])
    best_bce = min(candidate_summaries, key=lambda item: item["weighted_binary_cross_entropy"])
    spuaware_summary = next((item for item in candidate_summaries if item["label"] == "spuaware_bias"), None)
    best_bridge = max((item for item in candidate_summaries if item["kind"] == "bridge"), key=lambda item: item["weighted_accuracy"])

    if spuaware_summary is not None and best_bridge["weighted_accuracy"] <= spuaware_summary["weighted_accuracy"] + 1e-9:
        status = "bridge_candidate_not_better_than_current_spuaware_default"
        reason = (
            "Plaintext-static bridge calibration is numerically plausible, but on current held-out raw E2E logits "
            "it does not beat the existing spuaware_bias default in sample-weighted accuracy."
        )
    else:
        status = "bridge_candidate_is_competitive_for_accuracy"
        reason = "At least one bridge-derived calibration matches or exceeds the current comparison baseline on held-out raw E2E logits."

    report = {
        "manifest_type": "transshield_e2e_plaintext_bridge_calibration_report_v0",
        "label": args.label,
        "bridge_candidates": bridge_candidates,
        "candidate_summaries": candidate_summaries,
        "judgement": {
            "status": status,
            "reason": reason,
            "best_accuracy_candidate": best_accuracy["label"],
            "best_accuracy_weighted_accuracy": best_accuracy["weighted_accuracy"],
            "best_bce_candidate": best_bce["label"],
            "best_bce_weighted_binary_cross_entropy": best_bce["weighted_binary_cross_entropy"],
            "best_bridge_candidate": best_bridge["label"],
            "best_bridge_weighted_accuracy": best_bridge["weighted_accuracy"],
            "spuaware_bias_weighted_accuracy": None if spuaware_summary is None else spuaware_summary["weighted_accuracy"],
        },
    }
    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)
    if args.output_md:
        write_text(Path(args.output_md).expanduser().resolve(), build_markdown(report))
    if args.output_best_calibration_json:
        write_json(Path(args.output_best_calibration_json).expanduser().resolve(), best_accuracy["calibration"])
    if args.output_best_bridge_calibration_json:
        write_json(Path(args.output_best_bridge_calibration_json).expanduser().resolve(), best_bridge["calibration"])
    print(json.dumps(report["judgement"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
