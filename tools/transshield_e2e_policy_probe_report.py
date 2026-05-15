import argparse
import json
from pathlib import Path

import torch


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_from_logits(logits):
    if logits.shape[-1] != 2:
        raise ValueError(f"binary logits expected, got shape {tuple(logits.shape)}")
    return logits[:, 1] - logits[:, 0]


def load_variant(raw):
    if "=" not in raw:
        raise ValueError("--variant must use LABEL=PT")
    label, pt_path = raw.split("=", 1)
    payload = torch.load(pt_path, map_location="cpu")
    logits = payload["logits"].detach().cpu().float()
    raw_logits = payload.get("raw_logits_before_output_calibration")
    if raw_logits is not None:
        raw_logits = raw_logits.detach().cpu().float()
    predictions = logits.argmax(dim=1).long()
    return {
        "label": label,
        "pt": str(Path(pt_path)),
        "logits": logits,
        "score": score_from_logits(logits),
        "raw_logits": raw_logits,
        "raw_score": None if raw_logits is None else score_from_logits(raw_logits),
        "predictions": predictions,
    }


def fmt(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def command_report(args):
    manifest = load_json(args.share_manifest_json)
    targets = manifest.get("targets")
    if targets is None:
        raise ValueError("share manifest must include targets")
    targets_tensor = torch.tensor([int(item) for item in targets], dtype=torch.long)
    source_paths = manifest.get("source_paths") or [""] * len(targets)
    selected_indices = (manifest.get("source_selection") or {}).get("indices")
    if selected_indices is None:
        selected_indices = list(range(len(targets)))

    variants = [load_variant(item) for item in args.variant]
    if not variants:
        raise ValueError("at least one --variant LABEL=PT is required")
    sample_count = int(targets_tensor.numel())
    for variant in variants:
        if int(variant["logits"].shape[0]) < sample_count:
            raise ValueError(f"variant {variant['label']} has too few samples")
        for key in ("logits", "score", "predictions"):
            variant[key] = variant[key][:sample_count]
        if variant["raw_logits"] is not None:
            variant["raw_logits"] = variant["raw_logits"][:sample_count]
            variant["raw_score"] = variant["raw_score"][:sample_count]

    per_variant = {}
    for variant in variants:
        correct = variant["predictions"].eq(targets_tensor)
        per_variant[variant["label"]] = {
            "pt": variant["pt"],
            "accuracy": float(correct.float().mean().item() * 100.0),
            "correct_count": int(correct.sum().item()),
            "wrong_count": int((~correct).sum().item()),
            "mean_abs_score": float(variant["score"].abs().mean().item()),
            "finite_logits": bool(torch.isfinite(variant["logits"]).all().item()),
        }

    baseline = variants[0]
    rows = []
    recovered_by_any = 0
    regressed_by_any = 0
    for row_index in range(sample_count):
        target = int(targets_tensor[row_index].item())
        item = {
            "row_index": row_index,
            "source_index": int(selected_indices[row_index]),
            "target": target,
            "image": source_paths[row_index] if row_index < len(source_paths) else "",
            "variants": {},
        }
        baseline_correct = bool(baseline["predictions"][row_index].item() == target)
        any_recovered = False
        any_regressed = False
        for variant in variants:
            pred = int(variant["predictions"][row_index].item())
            correct = pred == target
            if (not baseline_correct) and correct:
                any_recovered = True
            if baseline_correct and (not correct):
                any_regressed = True
            item["variants"][variant["label"]] = {
                "prediction": pred,
                "correct": correct,
                "score": float(variant["score"][row_index].item()),
                "abs_score": float(abs(variant["score"][row_index].item())),
                "raw_score": (
                    None
                    if variant["raw_score"] is None
                    else float(variant["raw_score"][row_index].item())
                ),
            }
        if any_recovered:
            recovered_by_any += 1
        if any_regressed:
            regressed_by_any += 1
        item["recovered_by_any_nonbaseline_variant"] = any_recovered
        item["regressed_by_any_nonbaseline_variant"] = any_regressed
        rows.append(item)

    if recovered_by_any == 0 and regressed_by_any == 0:
        status = "no_policy_variant_recovery_or_regression_on_selected_samples"
    elif recovered_by_any == 0:
        status = "policy_variant_regresses_without_recovery"
    elif regressed_by_any == 0:
        status = "policy_variant_recovers_baseline_wrong_samples"
    elif regressed_by_any > recovered_by_any:
        status = "policy_variant_regression_dominates_recovery"
    else:
        status = "policy_variant_mixed_recovery_and_regression"
    payload = {
        "manifest_type": "transshield_e2e_policy_probe_report_v0",
        "label": args.label,
        "sample_count": sample_count,
        "selected_indices": selected_indices,
        "baseline_variant": baseline["label"],
        "status": status,
        "recovered_by_any_nonbaseline_variant": recovered_by_any,
        "regressed_by_any_nonbaseline_variant": regressed_by_any,
        "per_variant": per_variant,
        "per_sample": rows,
        "interpretation": {
            "argmax_recovery_axis": (
                "At least one secure-graph policy variant flips a baseline-wrong selected sample."
                if recovered_by_any > 0
                else "The tested secure-graph policy variants did not flip the selected baseline-wrong samples."
            ),
            "loss_recovery_axis": (
                "Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; "
                "post-reveal calibration can improve BCE/confidence without changing the secret SPU graph."
            ),
        },
    }
    write_json(args.output_json, payload)

    lines = [
        f"# E2E Policy Probe Report",
        "",
        f"- label: `{args.label}`",
        f"- sample_count: `{sample_count}`",
        f"- baseline_variant: `{baseline['label']}`",
        f"- status: `{status}`",
        f"- recovered_by_any_nonbaseline_variant: `{recovered_by_any}`",
        f"- regressed_by_any_nonbaseline_variant: `{regressed_by_any}`",
        "",
        "## Variant Summary",
        "",
        "| variant | accuracy | correct | wrong | mean_abs_score | finite |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, item in per_variant.items():
        lines.append(
            f"| {label} | {item['accuracy']:.4f} | {item['correct_count']} | "
            f"{item['wrong_count']} | {item['mean_abs_score']:.6g} | {item['finite_logits']} |"
        )
    lines += [
        "",
        "## Per-Sample Scores",
        "",
        "| source_index | target | image | " + " | ".join(f"{v['label']} pred/score" for v in variants) + " |",
        "|---:|---:|---|" + "|".join(["---:" for _ in variants]) + "|",
    ]
    for row in rows:
        cells = []
        for variant in variants:
            item = row["variants"][variant["label"]]
            marker = "ok" if item["correct"] else "wrong"
            cells.append(f"{item['prediction']}/{fmt(item['score'])} {marker}")
        lines.append(
            f"| {row['source_index']} | {row['target']} | {row['image']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- {payload['interpretation']['argmax_recovery_axis']}",
        f"- {payload['interpretation']['loss_recovery_axis']}",
    ]
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize E2E selected-sample policy probe candidates.")
    parser.add_argument("--label", default="e2e_policy_probe")
    parser.add_argument("--share-manifest-json", required=True)
    parser.add_argument("--variant", action="append", default=[], help="LABEL=path/to/candidate.pt")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.set_defaults(func=command_report)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
