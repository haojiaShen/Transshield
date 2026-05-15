#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def softmax_probabilities(logits):
    probabilities = []
    for row in logits:
        max_logit = max(row)
        exps = [math.exp(value - max_logit) for value in row]
        denom = sum(exps)
        probabilities.append([float(value / denom) for value in exps])
    return probabilities


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


def cross_entropy(logits, targets) -> float:
    total = 0.0
    for row, target in zip(logits, targets):
        z0, z1 = row
        max_logit = max(z0, z1)
        total += -(z1 if target == 1 else z0) + (
            max_logit + math.log(math.exp(z0 - max_logit) + math.exp(z1 - max_logit))
        )
    return float(total / len(targets))


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
    probabilities = [
        [float(item) for item in row.get("probabilities", [])] if row.get("probabilities") is not None else None
        for row in rows
    ]
    if any(item is None or len(item) != 2 for item in probabilities):
        probabilities = softmax_probabilities(logits)
    argmax_predictions = [int(row.get("argmax_prediction", 1 if logit[1] >= logit[0] else 0)) for row, logit in zip(rows, logits)]
    threshold_predictions = [
        int(row.get("threshold_prediction", 1 if prob[1] >= 0.5 else 0))
        for row, prob in zip(rows, probabilities)
    ]
    score_deltas = [float(logit[1] - logit[0]) for logit in logits]
    return {
        "path": str(path),
        "payload": payload,
        "sample_ids": sample_ids,
        "targets": targets,
        "logits": logits,
        "probabilities": probabilities,
        "argmax_predictions": argmax_predictions,
        "threshold_predictions": threshold_predictions,
        "score_deltas": score_deltas,
    }


def stage_summary(label: str, stage):
    targets = stage["targets"]
    return {
        "label": label,
        "path": stage["path"],
        "sample_count": len(stage["sample_ids"]),
        "metrics": {
            "argmax_accuracy": accuracy(stage["argmax_predictions"], targets),
            "argmax_f1": binary_f1(stage["argmax_predictions"], targets),
            "threshold_accuracy": accuracy(stage["threshold_predictions"], targets),
            "threshold_f1": binary_f1(stage["threshold_predictions"], targets),
            "auc": binary_auc(stage["score_deltas"], targets),
            "ce_loss": cross_entropy(stage["logits"], targets),
        },
        "score_stats": {
            "mean": mean(stage["score_deltas"]),
            "min": float(min(stage["score_deltas"])),
            "max": float(max(stage["score_deltas"])),
        },
    }


def abs_error(lhs, rhs):
    flat = []
    for lhs_row, rhs_row in zip(lhs, rhs):
        for lhs_item, rhs_item in zip(lhs_row, rhs_row):
            flat.append(abs(lhs_item - rhs_item))
    return {
        "max_abs_error": float(max(flat)),
        "mean_abs_error": float(sum(flat) / len(flat)),
    }


def match_ratio(lhs, rhs):
    return float(sum(int(a == b) for a, b in zip(lhs, rhs)) / len(lhs))


def pearson_correlation(lhs, rhs):
    lhs_mean = mean(lhs)
    rhs_mean = mean(rhs)
    lhs_var = sum((item - lhs_mean) ** 2 for item in lhs)
    rhs_var = sum((item - rhs_mean) ** 2 for item in rhs)
    if lhs_var <= 0.0 or rhs_var <= 0.0:
        return None
    cov = sum((a - lhs_mean) * (b - rhs_mean) for a, b in zip(lhs, rhs))
    return float(cov / math.sqrt(lhs_var * rhs_var))


def fit_affine(x_values, y_values):
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_var = sum((item - x_mean) ** 2 for item in x_values)
    if x_var <= 0.0:
        return {
            "slope": None,
            "intercept": None,
            "boundary_shift_x_at_y0": None,
            "mean_abs_error": None,
        }
    cov = sum((x_item - x_mean) * (y_item - y_mean) for x_item, y_item in zip(x_values, y_values))
    slope = cov / x_var
    intercept = y_mean - slope * x_mean
    residuals = [abs((slope * x_item + intercept) - y_item) for x_item, y_item in zip(x_values, y_values)]
    boundary_shift = None if abs(slope) < 1e-12 else float(-intercept / slope)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "boundary_shift_x_at_y0": boundary_shift,
        "mean_abs_error": float(sum(residuals) / len(residuals)),
    }


def threshold_candidates(scores):
    ordered = sorted(set(float(item) for item in scores))
    if not ordered:
        return [0.0]
    candidates = [ordered[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    candidates.append(ordered[-1] + 1e-6)
    return candidates


def sweep_threshold(scores, targets):
    best = None
    for threshold in threshold_candidates(scores):
        predictions = [1 if score >= threshold else 0 for score in scores]
        item = {
            "threshold": float(threshold),
            "accuracy": accuracy(predictions, targets),
            "f1": binary_f1(predictions, targets),
        }
        if best is None or item["accuracy"] > best["accuracy"] or (
            item["accuracy"] == best["accuracy"] and abs(item["threshold"]) < abs(best["threshold"])
        ):
            best = item
    return best


def sweep_threshold_to_match(scores, reference_predictions):
    best = None
    for threshold in threshold_candidates(scores):
        predictions = [1 if score >= threshold else 0 for score in scores]
        item = {
            "threshold": float(threshold),
            "match_ratio": match_ratio(predictions, reference_predictions),
        }
        if best is None or item["match_ratio"] > best["match_ratio"] or (
            item["match_ratio"] == best["match_ratio"] and abs(item["threshold"]) < abs(best["threshold"])
        ):
            best = item
    return best


def compare_stages(plaintext, static):
    if plaintext["sample_ids"] != static["sample_ids"]:
        raise ValueError("sample_ids mismatch between plaintext and static json")
    if plaintext["targets"] != static["targets"]:
        raise ValueError("targets mismatch between plaintext and static json")

    plaintext_scores = plaintext["score_deltas"]
    static_scores = static["score_deltas"]
    plaintext_targets = plaintext["targets"]

    return {
        "prediction_match": {
            "argmax_match_ratio": match_ratio(plaintext["argmax_predictions"], static["argmax_predictions"]),
            "threshold_match_ratio": match_ratio(plaintext["threshold_predictions"], static["threshold_predictions"]),
        },
        "error": {
            "logits": abs_error(plaintext["logits"], static["logits"]),
            "probabilities": abs_error(plaintext["probabilities"], static["probabilities"]),
            "score_delta": {
                "max_abs_error": float(max(abs(a - b) for a, b in zip(plaintext_scores, static_scores))),
                "mean_abs_error": float(mean([abs(a - b) for a, b in zip(plaintext_scores, static_scores)])),
            },
        },
        "score_alignment": {
            "pearson_correlation": pearson_correlation(plaintext_scores, static_scores),
            "same_sign_ratio": float(sum((a >= 0.0) == (b >= 0.0) for a, b in zip(plaintext_scores, static_scores)) / len(plaintext_scores)),
            "plaintext_score_mean": mean(plaintext_scores),
            "static_score_mean": mean(static_scores),
            "affine_static_from_plaintext_score": fit_affine(plaintext_scores, static_scores),
        },
        "plaintext_score_threshold_sweep_vs_targets": {
            "zero_threshold_accuracy": accuracy([1 if score >= 0.0 else 0 for score in plaintext_scores], plaintext_targets),
            "best_accuracy": sweep_threshold(plaintext_scores, plaintext_targets),
        },
        "plaintext_score_threshold_sweep_vs_static_argmax": sweep_threshold_to_match(
            plaintext_scores, static["argmax_predictions"]
        ),
        "metric_delta_static_minus_plaintext": {
            "argmax_accuracy": float(accuracy(static["argmax_predictions"], plaintext_targets) - accuracy(plaintext["argmax_predictions"], plaintext_targets)),
            "threshold_accuracy": float(accuracy(static["threshold_predictions"], plaintext_targets) - accuracy(plaintext["threshold_predictions"], plaintext_targets)),
            "auc": float(binary_auc(static_scores, plaintext_targets) - binary_auc(plaintext_scores, plaintext_targets)),
            "ce_loss": float(cross_entropy(static["logits"], plaintext_targets) - cross_entropy(plaintext["logits"], plaintext_targets)),
        },
    }


def infer_judgement(plaintext_summary, static_summary, compare_payload):
    score_alignment = compare_payload["score_alignment"]
    sweep = compare_payload["plaintext_score_threshold_sweep_vs_targets"]
    corr = score_alignment["pearson_correlation"]
    sign_ratio = score_alignment["same_sign_ratio"]
    best_accuracy = sweep["best_accuracy"]["accuracy"]
    zero_accuracy = sweep["zero_threshold_accuracy"]
    static_accuracy = static_summary["metrics"]["argmax_accuracy"]

    if corr is not None and corr >= 0.95 and sign_ratio <= 0.5 and best_accuracy >= static_accuracy - 2.0:
        status = "ranking_preserved_but_zero_boundary_misaligned"
        reason = (
            "Plaintext full-path class scores stay strongly rank-correlated with static scores, "
            "but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy."
        )
    elif corr is not None and corr >= 0.9:
        status = "ranking_related_but_boundary_and_scale_both_shifted"
        reason = (
            "Plaintext and static scores are still correlated, but the boundary/scale drift is large enough "
            "that a simple zero-threshold decision is not reliable."
        )
    else:
        status = "representation_gap_dominant"
        reason = "Plaintext and static scores are not well aligned; the gap is not explained by a simple decision-boundary shift."

    return {
        "status": status,
        "reason": reason,
        "plaintext_argmax_accuracy": plaintext_summary["metrics"]["argmax_accuracy"],
        "static_argmax_accuracy": static_summary["metrics"]["argmax_accuracy"],
        "plaintext_best_threshold_accuracy": best_accuracy,
        "plaintext_best_threshold": sweep["best_accuracy"]["threshold"],
        "plaintext_to_static_best_match_threshold": compare_payload["plaintext_score_threshold_sweep_vs_static_argmax"]["threshold"],
        "score_correlation": corr,
        "same_sign_ratio": sign_ratio,
        "affine_boundary_shift_x_at_y0": score_alignment["affine_static_from_plaintext_score"]["boundary_shift_x_at_y0"],
    }


def markdown_report(label, plaintext_summary, static_summary, compare_payload, judgement):
    lines = [
        f"# {label}",
        "",
        "## Stage Metrics",
        (
            f"- `plaintext_full_model`: argmax_acc=`{plaintext_summary['metrics']['argmax_accuracy']:.6f}`, "
            f"threshold_acc=`{plaintext_summary['metrics']['threshold_accuracy']:.6f}`, "
            f"auc=`{plaintext_summary['metrics']['auc']:.9f}`, ce_loss=`{plaintext_summary['metrics']['ce_loss']:.9f}`"
        ),
        (
            f"- `static_whole_forward`: argmax_acc=`{static_summary['metrics']['argmax_accuracy']:.6f}`, "
            f"threshold_acc=`{static_summary['metrics']['threshold_accuracy']:.6f}`, "
            f"auc=`{static_summary['metrics']['auc']:.9f}`, ce_loss=`{static_summary['metrics']['ce_loss']:.9f}`"
        ),
        "",
        "## Score Alignment",
        f"- `argmax_match_ratio = {compare_payload['prediction_match']['argmax_match_ratio']:.6f}`",
        f"- `threshold_match_ratio = {compare_payload['prediction_match']['threshold_match_ratio']:.6f}`",
        f"- `score_correlation = {compare_payload['score_alignment']['pearson_correlation']:.6f}`",
        f"- `same_sign_ratio = {compare_payload['score_alignment']['same_sign_ratio']:.6f}`",
        (
            f"- `affine static_score ~= a * plaintext_score + b`: "
            f"`a = {compare_payload['score_alignment']['affine_static_from_plaintext_score']['slope']:.6f}`, "
            f"`b = {compare_payload['score_alignment']['affine_static_from_plaintext_score']['intercept']:.6f}`, "
            f"`x_at_y0 = {compare_payload['score_alignment']['affine_static_from_plaintext_score']['boundary_shift_x_at_y0']:.6f}`"
        ),
        "",
        "## Threshold Sweep",
        (
            f"- `plaintext zero-threshold accuracy = "
            f"{compare_payload['plaintext_score_threshold_sweep_vs_targets']['zero_threshold_accuracy']:.6f}`"
        ),
        (
            f"- `plaintext best-threshold accuracy = "
            f"{compare_payload['plaintext_score_threshold_sweep_vs_targets']['best_accuracy']['accuracy']:.6f}` "
            f"at threshold `{compare_payload['plaintext_score_threshold_sweep_vs_targets']['best_accuracy']['threshold']:.6f}`"
        ),
        (
            f"- `best threshold to match static argmax = "
            f"{compare_payload['plaintext_score_threshold_sweep_vs_static_argmax']['threshold']:.6f}` "
            f"with match `{compare_payload['plaintext_score_threshold_sweep_vs_static_argmax']['match_ratio']:.6f}`"
        ),
        "",
        "## Judgement",
        f"- status: `{judgement['status']}`",
        f"- reason: `{judgement['reason']}`",
    ]
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare full-model plaintext per-sample logits against static whole-forward per-sample logits."
    )
    parser.add_argument("--plaintext-json", required=True)
    parser.add_argument("--static-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--label", default="e2e_plaintext_static_gap")
    return parser


def main():
    args = build_parser().parse_args()
    plaintext = read_per_sample_json(Path(args.plaintext_json).expanduser().resolve())
    static = read_per_sample_json(Path(args.static_json).expanduser().resolve())
    plaintext_summary = stage_summary("plaintext_full_model", plaintext)
    static_summary = stage_summary("static_whole_forward", static)
    compare_payload = compare_stages(plaintext, static)
    judgement = infer_judgement(plaintext_summary, static_summary, compare_payload)
    report = {
        "manifest_type": "transshield_e2e_plaintext_static_gap_report_v0",
        "label": args.label,
        "sample_count": len(plaintext["sample_ids"]),
        "stages": {
            "plaintext": plaintext_summary,
            "static": static_summary,
        },
        "compare": compare_payload,
        "judgement": judgement,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)
    if args.output_md:
        write_text(
            Path(args.output_md).expanduser().resolve(),
            markdown_report(args.label, plaintext_summary, static_summary, compare_payload, judgement),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
