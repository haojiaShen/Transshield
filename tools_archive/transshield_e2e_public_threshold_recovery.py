#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_dataset_arg(value: str) -> Tuple[str, Path, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise argparse.ArgumentTypeError("--dataset must use label=candidate_json,reference_json,image_list")
    label = parts[0].strip()
    paths = [Path(item).expanduser().resolve() for item in parts[1].split(",")]
    if len(paths) != 3:
        raise argparse.ArgumentTypeError("--dataset must provide exactly three comma-separated paths")
    for path in paths:
        if not path.exists():
            raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return label, paths[0], paths[1], paths[2]


def load_images(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def calibration_score_scale(candidate: Dict) -> float:
    calibration = candidate.get("output_calibration") or {}
    weights = calibration.get("weights") or []
    if len(weights) == 2 and isinstance(weights[0], (int, float)) and isinstance(weights[1], (int, float)):
        scale = abs(float(weights[1]))
        if scale > 0.0:
            return scale
    return 1.0


def load_dataset(label: str, candidate_json: Path, reference_json: Path, image_list: Path) -> Dict:
    candidate = load_json(candidate_json)
    reference = load_json(reference_json)
    logits = candidate.get("prediction_preview", {}).get("logits") or []
    targets = [int(row["target"]) for row in reference.get("per_sample") or []]
    images = load_images(image_list)
    if not logits or len(logits) != len(targets) or len(images) != len(targets):
        raise ValueError(
            f"dataset {label} count mismatch: logits={len(logits)} targets={len(targets)} images={len(images)}"
        )
    scale = calibration_score_scale(candidate)
    rows = []
    for index, (logit_pair, target, image) in enumerate(zip(logits, targets, images)):
        score = float(logit_pair[1]) - float(logit_pair[0])
        canonical_score = score / scale
        rows.append(
            {
                "index": index,
                "image": image,
                "target": target,
                "score": score,
                "canonical_score": canonical_score,
                "default_prediction": int(score >= 0.0),
                "default_correct": int(score >= 0.0) == target,
            }
        )
    return {
        "label": label,
        "candidate_json": str(candidate_json),
        "reference_json": str(reference_json),
        "image_list": str(image_list),
        "sample_count": len(rows),
        "score_scale": scale,
        "output_calibration": candidate.get("output_calibration"),
        "rows": rows,
    }


def accuracy(rows: List[Dict], threshold: float, score_key: str = "canonical_score") -> float:
    if not rows:
        return 0.0
    correct = sum(int((row[score_key] >= threshold) == bool(row["target"])) for row in rows)
    return float(correct) / float(len(rows)) * 100.0


def threshold_candidates(rows: List[Dict], score_key: str = "canonical_score") -> List[float]:
    scores = sorted({float(row[score_key]) for row in rows})
    if not scores:
        return [0.0]
    candidates = [scores[0] - 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 1.0)
    candidates.append(0.0)
    return sorted(set(candidates))


def best_threshold(rows: List[Dict], score_key: str = "canonical_score") -> Dict:
    candidates = threshold_candidates(rows, score_key)
    scored = [(accuracy(rows, threshold, score_key), threshold) for threshold in candidates]
    scored.sort(key=lambda item: (-item[0], abs(item[1]), item[1]))
    best_acc, threshold = scored[0]
    default_acc = accuracy(rows, 0.0, score_key)
    return {
        "default_threshold": 0.0,
        "default_accuracy": default_acc,
        "best_threshold": threshold,
        "best_accuracy": best_acc,
        "best_delta_vs_default": best_acc - default_acc,
    }


def apply_threshold_summary(rows: List[Dict], threshold: float, score_key: str = "canonical_score") -> Dict:
    per_sample = []
    correct = 0
    for row in rows:
        prediction = int(float(row[score_key]) >= threshold)
        is_correct = prediction == int(row["target"])
        correct += int(is_correct)
        per_sample.append(
            {
                "index": row["index"],
                "image": row["image"],
                "target": row["target"],
                "canonical_score": row["canonical_score"],
                "default_prediction": row["default_prediction"],
                "default_correct": row["default_correct"],
                "threshold_prediction": prediction,
                "threshold_correct": is_correct,
            }
        )
    wrong = [row for row in per_sample if not row["threshold_correct"]]
    boundary_wrong = sorted(wrong, key=lambda row: abs(float(row["canonical_score"]) - threshold))
    return {
        "threshold": threshold,
        "accuracy": float(correct) / float(len(rows)) * 100.0 if rows else 0.0,
        "wrong_count": len(wrong),
        "nearest_wrong_samples": boundary_wrong[:8],
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E Public Threshold Recovery Report",
        "",
        f"- label: `{report['label']}`",
        f"- status: `{report['judgement']['status']}`",
        f"- reason: {report['judgement']['reason']}",
        "",
        "| dataset | samples | score scale | default acc | best acc | best threshold | delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["dataset_summaries"]:
        best = item["within_dataset_threshold_sweep"]
        lines.append(
            "| {label} | {count} | {scale} | {default} | {best_acc} | {threshold} | {delta} |".format(
                label=item["label"],
                count=item["sample_count"],
                scale=fmt(item["score_scale"]),
                default=fmt(best["default_accuracy"]),
                best_acc=fmt(best["best_accuracy"]),
                threshold=fmt(best["best_threshold"]),
                delta=fmt(best["best_delta_vs_default"]),
            )
        )
    if report["cross_dataset_eval"]:
        lines.extend(["", "## Cross Dataset Eval", ""])
        lines.append("| source threshold | eval dataset | threshold | accuracy | wrong count |")
        lines.append("|---|---|---:|---:|---:|")
        for item in report["cross_dataset_eval"]:
            lines.append(
                "| {source} | {target} | {threshold} | {accuracy} | {wrong} |".format(
                    source=item["source_dataset"],
                    target=item["eval_dataset"],
                    threshold=fmt(item["threshold"]),
                    accuracy=fmt(item["accuracy"]),
                    wrong=item["wrong_count"],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public threshold recovery on E2E logits.")
    parser.add_argument("--dataset", action="append", type=parse_dataset_arg, required=True)
    parser.add_argument("--label", default="e2e_public_threshold_recovery")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    datasets = [load_dataset(*dataset_arg) for dataset_arg in args.dataset]
    summaries = []
    for dataset in datasets:
        summaries.append(
            {
                "label": dataset["label"],
                "candidate_json": dataset["candidate_json"],
                "reference_json": dataset["reference_json"],
                "image_list": dataset["image_list"],
                "sample_count": dataset["sample_count"],
                "score_scale": dataset["score_scale"],
                "output_calibration": dataset["output_calibration"],
                "within_dataset_threshold_sweep": best_threshold(dataset["rows"]),
            }
        )

    cross = []
    for source, source_summary in zip(datasets, summaries):
        threshold = source_summary["within_dataset_threshold_sweep"]["best_threshold"]
        for target in datasets:
            if target["label"] == source["label"]:
                continue
            applied = apply_threshold_summary(target["rows"], threshold)
            applied["source_dataset"] = source["label"]
            applied["eval_dataset"] = target["label"]
            cross.append(applied)

    max_within_delta = max(
        (item["within_dataset_threshold_sweep"]["best_delta_vs_default"] for item in summaries),
        default=0.0,
    )
    cross_improved = [
        item
        for item in cross
        if item["accuracy"]
        > next(summary["within_dataset_threshold_sweep"]["default_accuracy"] for summary in summaries if summary["label"] == item["eval_dataset"])
    ]
    if cross_improved:
        status = "public_threshold_transfer_improves_eval_subset"
        reason = "A threshold fitted on one E2E subset improved another E2E subset; SPU-aware public threshold calibration is a viable lightweight recovery axis."
    elif max_within_delta > 0.0:
        status = "within_subset_threshold_can_improve_but_transfer_not_proven"
        reason = "Some subset can be improved by refitting the public threshold, but cross-subset transfer is not yet positive."
    else:
        status = "public_threshold_recovery_not_promising"
        reason = "Best threshold search did not improve over the current default boundary on the evaluated E2E subsets."

    report = {
        "manifest_type": "transshield_e2e_public_threshold_recovery_v0",
        "label": args.label,
        "dataset_summaries": summaries,
        "cross_dataset_eval": cross,
        "judgement": {
            "status": status,
            "reason": reason,
        },
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps(report["judgement"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
