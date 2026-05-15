#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tensor_to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def load_payload(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"expected dict payload in {path}")
    return payload


def load_targets_from_reference_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_sample") or []
    if not rows:
        raise ValueError(f"reference json has no per_sample rows: {path}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    targets = [int(row["target"]) for row in rows]
    return sample_ids, targets


def ensure_logits(payload, key: str, label: str):
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{label} payload missing logits key: {key}")
    rows = tensor_to_list(value)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} logits key {key} is empty")
    return [[float(item) for item in row] for row in rows]


def ensure_sample_ids(payload, expected_count: int, fallback_prefix: str):
    sample_ids = payload.get("sample_ids")
    if sample_ids is None:
        return [f"{fallback_prefix}_{index:06d}" for index in range(expected_count)]
    sample_ids = [str(item) for item in sample_ids]
    if len(sample_ids) != expected_count:
        raise ValueError(f"sample_id count mismatch: {len(sample_ids)} != {expected_count}")
    return sample_ids


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


def select_stage_threshold(payload, logits_key: str, fallback_threshold):
    if logits_key == "raw_logits_before_output_calibration":
        return fallback_threshold
    threshold = payload.get("threshold")
    if threshold is None:
        return fallback_threshold
    return float(threshold)


def build_stage_summary(name: str, path: Path, payload, logits_key: str, sample_ids, targets, fallback_threshold):
    logits = ensure_logits(payload, logits_key, name)
    if len(logits) != len(sample_ids):
        raise ValueError(f"{name} sample count mismatch: {len(logits)} != {len(sample_ids)}")
    payload_sample_ids = ensure_sample_ids(payload, len(logits), name)
    if payload_sample_ids != sample_ids:
        raise ValueError(f"{name} sample_ids mismatch against reference ordering")
    probabilities = softmax_probabilities(logits)
    threshold = select_stage_threshold(payload, logits_key, fallback_threshold)
    argmax_predictions = [1 if row[1] >= row[0] else 0 for row in logits]
    threshold_predictions = [1 if row[1] >= threshold else 0 for row in probabilities] if threshold is not None else None
    scores = [row[1] - row[0] for row in logits]
    return {
        "label": name,
        "path": str(path),
        "logits_key": logits_key,
        "sample_count": len(logits),
        "threshold": threshold,
        "metrics": {
            "argmax_accuracy": accuracy(argmax_predictions, targets),
            "argmax_f1": binary_f1(argmax_predictions, targets),
            "threshold_accuracy": None if threshold_predictions is None else accuracy(threshold_predictions, targets),
            "threshold_f1": None if threshold_predictions is None else binary_f1(threshold_predictions, targets),
            "auc": binary_auc(scores, targets),
            "ce_loss": cross_entropy(logits, targets),
        },
        "artifacts": {
            "logits": logits,
            "probabilities": probabilities,
            "argmax_predictions": argmax_predictions,
            "threshold_predictions": threshold_predictions,
        },
    }


def compare_match(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    if len(lhs) != len(rhs):
        raise ValueError("prediction length mismatch")
    return float(sum(int(a == b) for a, b in zip(lhs, rhs)) / len(lhs))


def compare_pair(lhs, rhs):
    logits_abs = []
    prob_abs = []
    for lhs_row, rhs_row in zip(lhs["artifacts"]["logits"], rhs["artifacts"]["logits"]):
        for lhs_value, rhs_value in zip(lhs_row, rhs_row):
            logits_abs.append(abs(lhs_value - rhs_value))
    for lhs_row, rhs_row in zip(lhs["artifacts"]["probabilities"], rhs["artifacts"]["probabilities"]):
        for lhs_value, rhs_value in zip(lhs_row, rhs_row):
            prob_abs.append(abs(lhs_value - rhs_value))
    return {
        "logits_error": {
            "max_abs_error": float(max(logits_abs)),
            "mean_abs_error": float(sum(logits_abs) / len(logits_abs)),
        },
        "probabilities_error": {
            "max_abs_error": float(max(prob_abs)),
            "mean_abs_error": float(sum(prob_abs) / len(prob_abs)),
        },
        "prediction_match": {
            "argmax_match_ratio": compare_match(
                lhs["artifacts"]["argmax_predictions"], rhs["artifacts"]["argmax_predictions"]
            ),
            "threshold_match_ratio": compare_match(
                lhs["artifacts"]["threshold_predictions"], rhs["artifacts"]["threshold_predictions"]
            ),
        },
        "metric_delta_rhs_minus_lhs": {
            "argmax_accuracy": float(rhs["metrics"]["argmax_accuracy"] - lhs["metrics"]["argmax_accuracy"]),
            "threshold_accuracy": (
                None
                if lhs["metrics"]["threshold_accuracy"] is None or rhs["metrics"]["threshold_accuracy"] is None
                else float(rhs["metrics"]["threshold_accuracy"] - lhs["metrics"]["threshold_accuracy"])
            ),
            "auc": float(rhs["metrics"]["auc"] - lhs["metrics"]["auc"]),
            "ce_loss": float(rhs["metrics"]["ce_loss"] - lhs["metrics"]["ce_loss"]),
        },
    }


def strip_artifacts(stage_summary):
    output = dict(stage_summary)
    output.pop("artifacts", None)
    return output


def infer_judgement(reference_stage, cpu_stage, spu_stage, ref_cpu_compare, cpu_spu_compare, ref_spu_compare):
    ref_cpu_argmax_delta = abs(cpu_stage["metrics"]["argmax_accuracy"] - reference_stage["metrics"]["argmax_accuracy"])
    cpu_spu_argmax_delta = abs(spu_stage["metrics"]["argmax_accuracy"] - cpu_stage["metrics"]["argmax_accuracy"])
    ref_spu_argmax_delta = abs(spu_stage["metrics"]["argmax_accuracy"] - reference_stage["metrics"]["argmax_accuracy"])
    ref_cpu_ce_delta = abs(cpu_stage["metrics"]["ce_loss"] - reference_stage["metrics"]["ce_loss"])
    cpu_spu_ce_delta = abs(spu_stage["metrics"]["ce_loss"] - cpu_stage["metrics"]["ce_loss"])
    ref_spu_logit_max = ref_spu_compare["logits_error"]["max_abs_error"]

    if ref_cpu_argmax_delta < 0.05 and cpu_spu_argmax_delta < 0.05 and ref_spu_argmax_delta < 0.05:
        if ref_spu_logit_max > 0.0:
            status = "spu_numeric_drift_present_but_decision_negligible"
            reason = (
                "SPU introduces measurable logit drift, but on this subset it does not change argmax or threshold decisions."
            )
        else:
            status = "no_meaningful_gap_detected"
            reason = "Reference, CPU static, and SPU outputs are identical on this subset."
    elif ref_cpu_argmax_delta < 0.05 and ref_cpu_ce_delta < 1e-6 and cpu_spu_argmax_delta > 0.05:
        status = "spu_specific_gap_dominant"
        reason = (
            "CPU static and plaintext static whole-forward are effectively identical; "
            "the remaining gap is introduced on the SPU-side output path."
        )
    elif ref_cpu_argmax_delta > 0.05 and cpu_spu_argmax_delta < 0.05:
        status = "pre_spu_static_approximation_gap_dominant"
        reason = "Most accuracy drift is already present before SPU; SPU adds comparatively little extra decision error."
    else:
        status = "mixed_gap_cpu_and_spu_both_contribute"
        reason = "Both CPU static approximation and SPU execution add measurable drift; neither stage is negligible."

    return {
        "status": status,
        "reason": reason,
        "reference_to_cpu_argmax_accuracy_delta": float(
            cpu_stage["metrics"]["argmax_accuracy"] - reference_stage["metrics"]["argmax_accuracy"]
        ),
        "cpu_to_spu_argmax_accuracy_delta": float(
            spu_stage["metrics"]["argmax_accuracy"] - cpu_stage["metrics"]["argmax_accuracy"]
        ),
        "reference_to_spu_argmax_accuracy_delta": float(
            spu_stage["metrics"]["argmax_accuracy"] - reference_stage["metrics"]["argmax_accuracy"]
        ),
        "reference_to_cpu_ce_loss_delta": float(cpu_stage["metrics"]["ce_loss"] - reference_stage["metrics"]["ce_loss"]),
        "cpu_to_spu_ce_loss_delta": float(spu_stage["metrics"]["ce_loss"] - cpu_stage["metrics"]["ce_loss"]),
        "reference_to_spu_ce_loss_delta": float(spu_stage["metrics"]["ce_loss"] - reference_stage["metrics"]["ce_loss"]),
        "key_compare_metrics": {
            "reference_vs_cpu_argmax_match_ratio": ref_cpu_compare["prediction_match"]["argmax_match_ratio"],
            "cpu_vs_spu_argmax_match_ratio": cpu_spu_compare["prediction_match"]["argmax_match_ratio"],
            "reference_vs_spu_argmax_match_ratio": ref_spu_compare["prediction_match"]["argmax_match_ratio"],
            "reference_vs_cpu_logit_max_abs_error": ref_cpu_compare["logits_error"]["max_abs_error"],
            "cpu_vs_spu_logit_max_abs_error": cpu_spu_compare["logits_error"]["max_abs_error"],
            "reference_vs_spu_logit_max_abs_error": ref_spu_compare["logits_error"]["max_abs_error"],
        },
    }


def build_markdown(label, reference_stage, cpu_stage, spu_stage, ref_cpu_compare, cpu_spu_compare, ref_spu_compare, judgement):
    def line_stage(stage):
        metrics = stage["metrics"]
        return (
            f"- `{stage['label']}` [{stage['logits_key']}]: "
            f"argmax_acc=`{metrics['argmax_accuracy']:.6f}`, "
            f"threshold_acc=`{metrics['threshold_accuracy']:.6f}`"
            if metrics["threshold_accuracy"] is not None
            else f"- `{stage['label']}` [{stage['logits_key']}]: argmax_acc=`{metrics['argmax_accuracy']:.6f}`"
        ) + f", auc=`{metrics['auc']:.9f}`, ce_loss=`{metrics['ce_loss']:.9f}`"

    def line_compare(name, payload):
        return (
            f"- `{name}`: "
            f"argmax_match=`{payload['prediction_match']['argmax_match_ratio']:.6f}`, "
            f"threshold_match=`{payload['prediction_match']['threshold_match_ratio']:.6f}`"
            if payload["prediction_match"]["threshold_match_ratio"] is not None
            else f"- `{name}`: argmax_match=`{payload['prediction_match']['argmax_match_ratio']:.6f}`"
        ) + (
            f", logit_max_abs=`{payload['logits_error']['max_abs_error']:.9f}`, "
            f"logit_mean_abs=`{payload['logits_error']['mean_abs_error']:.9f}`"
        )

    lines = [
        f"# {label}",
        "",
        "## Stage Metrics",
        line_stage(reference_stage),
        line_stage(cpu_stage),
        line_stage(spu_stage),
        "",
        "## Pairwise Compare",
        line_compare("reference_vs_cpu", ref_cpu_compare),
        line_compare("cpu_vs_spu", cpu_spu_compare),
        line_compare("reference_vs_spu", ref_spu_compare),
        "",
        "## Judgement",
        f"- status: `{judgement['status']}`",
        f"- reason: `{judgement['reason']}`",
        f"- reference_to_cpu_argmax_accuracy_delta: `{judgement['reference_to_cpu_argmax_accuracy_delta']}`",
        f"- cpu_to_spu_argmax_accuracy_delta: `{judgement['cpu_to_spu_argmax_accuracy_delta']}`",
        f"- reference_to_spu_argmax_accuracy_delta: `{judgement['reference_to_spu_argmax_accuracy_delta']}`",
        f"- reference_to_cpu_ce_loss_delta: `{judgement['reference_to_cpu_ce_loss_delta']}`",
        f"- cpu_to_spu_ce_loss_delta: `{judgement['cpu_to_spu_ce_loss_delta']}`",
        f"- reference_to_spu_ce_loss_delta: `{judgement['reference_to_spu_ce_loss_delta']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Attribution report for plaintext static -> CPU static -> SPU static E2E gaps.")
    parser.add_argument("--reference-pt", required=True)
    parser.add_argument("--cpu-pt", required=True)
    parser.add_argument("--spu-pt", required=True)
    parser.add_argument("--reference-logits-key", default="logits")
    parser.add_argument("--cpu-logits-key", default="logits")
    parser.add_argument("--spu-logits-key", default="logits")
    parser.add_argument("--reference-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--label", default="e2e_gap_attribution")
    args = parser.parse_args()

    reference_pt = Path(args.reference_pt).expanduser().resolve()
    cpu_pt = Path(args.cpu_pt).expanduser().resolve()
    spu_pt = Path(args.spu_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else None

    reference_payload = load_payload(reference_pt)
    cpu_payload = load_payload(cpu_pt)
    spu_payload = load_payload(spu_pt)

    reference_logits = ensure_logits(reference_payload, args.reference_logits_key, "reference")
    sample_ids = ensure_sample_ids(reference_payload, len(reference_logits), "reference")

    targets = tensor_to_list(reference_payload.get("targets"))
    if targets is not None:
        targets = [int(item) for item in targets]
    elif args.reference_json:
        ref_json_sample_ids, targets = load_targets_from_reference_json(Path(args.reference_json).expanduser().resolve())
        if ref_json_sample_ids != sample_ids:
            raise ValueError("reference json sample_ids mismatch against reference pt")
    else:
        raise ValueError("reference targets missing; pass --reference-json")

    threshold = reference_payload.get("threshold")
    if threshold is not None:
        threshold = float(threshold)

    reference_stage = build_stage_summary(
        "reference_static_plaintext", reference_pt, reference_payload, args.reference_logits_key, sample_ids, targets, threshold
    )
    cpu_stage = build_stage_summary(
        "cpu_static_candidate", cpu_pt, cpu_payload, args.cpu_logits_key, sample_ids, targets, threshold
    )
    spu_stage = build_stage_summary(
        "spu_static_candidate", spu_pt, spu_payload, args.spu_logits_key, sample_ids, targets, threshold
    )

    ref_cpu_compare = compare_pair(reference_stage, cpu_stage)
    cpu_spu_compare = compare_pair(cpu_stage, spu_stage)
    ref_spu_compare = compare_pair(reference_stage, spu_stage)
    judgement = infer_judgement(reference_stage, cpu_stage, spu_stage, ref_cpu_compare, cpu_spu_compare, ref_spu_compare)

    report = {
        "manifest_type": "transshield_e2e_gap_attribution_report_v0",
        "label": args.label,
        "reference_json": (
            str(Path(args.reference_json).expanduser().resolve()) if args.reference_json else None
        ),
        "sample_count": len(sample_ids),
        "threshold_used_for_metrics": threshold,
        "stages": {
            "reference": strip_artifacts(reference_stage),
            "cpu": strip_artifacts(cpu_stage),
            "spu": strip_artifacts(spu_stage),
        },
        "compares": {
            "reference_vs_cpu": ref_cpu_compare,
            "cpu_vs_spu": cpu_spu_compare,
            "reference_vs_spu": ref_spu_compare,
        },
        "judgement": judgement,
    }
    write_json(output_json, report)
    if output_md is not None:
        write_text(
            output_md,
            build_markdown(
                args.label, reference_stage, cpu_stage, spu_stage, ref_cpu_compare, cpu_spu_compare, ref_spu_compare, judgement
            ),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
