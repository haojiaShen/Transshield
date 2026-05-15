#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_nonzero_link(path: Path, pattern):
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        item = {
            "log_path": str(path),
            "send_bytes": int(match.group("send")),
            "recv_bytes": int(match.group("recv")),
            "send_actions": int(match.group("send_actions")),
            "recv_actions": int(match.group("recv_actions")),
        }
        if item["send_bytes"] or item["recv_bytes"] or item["send_actions"] or item["recv_actions"]:
            latest = item
    return latest


def accuracy(predictions, targets):
    import torch

    return float((predictions == targets).float().mean().item() * 100.0)


def compute_threshold_predictions(probabilities, threshold):
    if threshold is None:
        return None
    return (probabilities[:, 1] >= float(threshold)).long()


def compare_match(lhs, rhs):
    import torch

    if lhs is None or rhs is None:
        return None
    lhs = lhs.detach().cpu().long()
    rhs = rhs.detach().cpu().long()
    if lhs.shape != rhs.shape:
        raise ValueError(f"prediction shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    return float((lhs == rhs).float().mean().item())


def compare_tensor_error(lhs, rhs):
    diff = (lhs - rhs).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Build E2E approximate-path metrics against plaintext and static whole-forward references.")
    parser.add_argument("--share-manifest-json", required=True)
    parser.add_argument("--plaintext-reference-json", required=True)
    parser.add_argument("--static-reference-pt", required=True)
    parser.add_argument("--candidate-pt", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--spu-log-dir", default="logs/spu_nodes")
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch

    share_manifest = load_json(Path(args.share_manifest_json).expanduser().resolve())
    plaintext_reference = load_json(Path(args.plaintext_reference_json).expanduser().resolve())
    static_reference = torch.load(Path(args.static_reference_pt).expanduser().resolve(), map_location="cpu")
    candidate_pt = Path(args.candidate_pt).expanduser().resolve()
    candidate_payload = torch.load(candidate_pt, map_location="cpu")
    candidate_json = load_json(Path(args.candidate_json).expanduser().resolve())
    output_json = Path(args.output_json).expanduser().resolve()
    spu_log_dir = Path(args.spu_log_dir).expanduser().resolve()

    targets = share_manifest.get("targets")
    if targets is None:
        raise SystemExit("share manifest does not include targets; rerun with --include-targets")
    targets = torch.tensor(targets, dtype=torch.long)

    logits = candidate_payload["logits"].detach().cpu().float()
    probabilities = candidate_payload.get("probabilities")
    if probabilities is None:
        probabilities = torch.softmax(logits, dim=-1)
    else:
        probabilities = probabilities.detach().cpu().float()

    count = min(
        int(targets.numel()),
        int(logits.shape[0]),
        int(static_reference["logits"].shape[0]),
        len(plaintext_reference.get("per_sample") or []),
    )
    targets = targets[:count]
    logits = logits[:count]
    probabilities = probabilities[:count]
    argmax_predictions = logits.argmax(dim=1)
    threshold = candidate_payload.get("threshold")
    threshold_predictions = candidate_payload.get("threshold_predictions")
    if threshold_predictions is None:
        threshold_predictions = compute_threshold_predictions(probabilities, threshold)
    if threshold_predictions is not None:
        threshold_predictions = threshold_predictions.detach().cpu()[:count]

    e2e_argmax_acc = accuracy(argmax_predictions, targets)
    e2e_threshold_acc = None if threshold_predictions is None else accuracy(threshold_predictions, targets)

    plaintext_rows = (plaintext_reference.get("per_sample") or [])[:count]
    plaintext_argmax_predictions = torch.tensor([int(row["argmax_prediction"]) for row in plaintext_rows], dtype=torch.long)
    plaintext_threshold_values = [row.get("threshold_prediction") for row in plaintext_rows]
    plaintext_threshold_predictions = None
    if all(value is not None for value in plaintext_threshold_values):
        plaintext_threshold_predictions = torch.tensor([int(value) for value in plaintext_threshold_values], dtype=torch.long)
    plaintext_argmax_acc = accuracy(plaintext_argmax_predictions, targets)
    plaintext_threshold_acc = None
    if plaintext_threshold_predictions is not None:
        plaintext_threshold_acc = accuracy(plaintext_threshold_predictions, targets)

    static_logits = static_reference["logits"].detach().cpu().float()[:count]
    static_probabilities = static_reference.get("probabilities")
    if static_probabilities is None:
        static_probabilities = torch.softmax(static_logits, dim=-1)
    else:
        static_probabilities = static_probabilities.detach().cpu().float()[:count]
    static_argmax_predictions = static_reference.get("argmax_predictions")
    if static_argmax_predictions is None:
        static_argmax_predictions = static_logits.argmax(dim=1)
    else:
        static_argmax_predictions = static_argmax_predictions.detach().cpu()[:count]
    static_threshold = static_reference.get("threshold")
    static_threshold_predictions = static_reference.get("threshold_predictions")
    if static_threshold_predictions is None:
        static_threshold_predictions = compute_threshold_predictions(static_probabilities, static_threshold)
    else:
        static_threshold_predictions = static_threshold_predictions.detach().cpu()[:count]
    static_argmax_acc = accuracy(static_argmax_predictions, targets)
    static_threshold_acc = None
    if static_threshold_predictions is not None:
        static_threshold_acc = accuracy(static_threshold_predictions, targets)

    raw_block = None
    raw_logits = candidate_payload.get("raw_logits_before_output_calibration")
    if raw_logits is not None:
        raw_logits = raw_logits.detach().cpu().float()[:count]
        raw_probabilities = torch.softmax(raw_logits, dim=-1)
        raw_argmax_predictions = raw_logits.argmax(dim=1)
        raw_threshold_predictions = compute_threshold_predictions(raw_probabilities, static_threshold)
        raw_block = {
            "present": True,
            "same_subset_argmax_accuracy": accuracy(raw_argmax_predictions, targets),
            "same_subset_threshold_accuracy": (
                None if raw_threshold_predictions is None else accuracy(raw_threshold_predictions, targets)
            ),
            "prediction_match_vs_static_whole_forward": {
                "argmax_match_ratio": compare_match(raw_argmax_predictions, static_argmax_predictions),
                "threshold_match_ratio": compare_match(raw_threshold_predictions, static_threshold_predictions),
            },
            "logits_error_vs_static_whole_forward": compare_tensor_error(raw_logits, static_logits),
            "probabilities_error_vs_static_whole_forward": compare_tensor_error(raw_probabilities, static_probabilities),
        }

    link_pattern = re.compile(
        r"Link details: total send bytes (?P<send>\d+), recv bytes (?P<recv>\d+), "
        r"send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)"
    )
    node_link_details = []
    for log_path in sorted(spu_log_dir.glob("node_*.log")):
        item = latest_nonzero_link(log_path, link_pattern)
        if item is not None:
            node_link_details.append(item)
    aggregate_send = sum(item["send_bytes"] for item in node_link_details)
    aggregate_recv = sum(item["recv_bytes"] for item in node_link_details)
    aggregate_total = aggregate_send + aggregate_recv

    result = {
        "manifest_type": "transshield_e2e_approx_eval_metrics_v1",
        "image_list": str(Path(share_manifest.get("source_image_list", ""))) if share_manifest.get("source_image_list") else None,
        "plaintext_reference_json": str(Path(args.plaintext_reference_json).expanduser().resolve()),
        "static_whole_forward_reference_pt": str(Path(args.static_reference_pt).expanduser().resolve()),
        "candidate_json": str(Path(args.candidate_json).expanduser().resolve()),
        "candidate_pt": str(candidate_pt),
        "sample_count": count,
        "target_count": int(targets.numel()),
        "comparison_scope": "same_image_list_same_targets_full_model_and_static_whole_forward_vs_e2e_output_path",
        "original_plaintext_same_subset_argmax_accuracy": plaintext_argmax_acc,
        "original_plaintext_same_subset_threshold_accuracy": plaintext_threshold_acc,
        "static_whole_forward_same_subset_argmax_accuracy": static_argmax_acc,
        "static_whole_forward_same_subset_threshold_accuracy": static_threshold_acc,
        "e2e_argmax_accuracy": e2e_argmax_acc,
        "e2e_threshold_accuracy": e2e_threshold_acc,
        "argmax_accuracy_gap_e2e_minus_plaintext_pp": e2e_argmax_acc - plaintext_argmax_acc,
        "threshold_accuracy_gap_e2e_minus_plaintext_pp": (
            None if e2e_threshold_acc is None or plaintext_threshold_acc is None else e2e_threshold_acc - plaintext_threshold_acc
        ),
        "argmax_accuracy_gap_e2e_minus_static_whole_forward_pp": e2e_argmax_acc - static_argmax_acc,
        "threshold_accuracy_gap_e2e_minus_static_whole_forward_pp": (
            None if e2e_threshold_acc is None or static_threshold_acc is None else e2e_threshold_acc - static_threshold_acc
        ),
        "prediction_match_vs_original_plaintext": {
            "argmax_match_ratio": compare_match(argmax_predictions, plaintext_argmax_predictions),
            "threshold_match_ratio": compare_match(threshold_predictions, plaintext_threshold_predictions),
        },
        "prediction_match_vs_static_whole_forward": {
            "argmax_match_ratio": compare_match(argmax_predictions, static_argmax_predictions),
            "threshold_match_ratio": compare_match(threshold_predictions, static_threshold_predictions),
        },
        "raw_secure_graph_before_output_calibration": raw_block,
        "original_plaintext_full_val_reference": {
            "argmax_accuracy": 93.70229244232178,
            "threshold_accuracy": 94.08397078514099,
            "note": "full-val reference is retained only for context; primary gap fields above use the same eval subset",
        },
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "output_calibration": candidate_json.get("output_calibration"),
        "e2e_elapsed_sec": candidate_json.get("elapsed_sec"),
        "e2e_communication_from_spu_node_logs": {
            "status": "available" if node_link_details else "missing",
            "node_latest_nonzero_link_details": node_link_details,
            "aggregate_send_bytes": aggregate_send if node_link_details else None,
            "aggregate_recv_bytes": aggregate_recv if node_link_details else None,
            "aggregate_total_bytes": aggregate_total if node_link_details else None,
            "scope_note": "Parsed from latest nonzero Link details in logs/spu_nodes/node_*.log after the e2e run.",
        },
        "original_plaintext_communication": {
            "status": "not_applicable",
            "total_bytes": 0,
            "note": "Plaintext local references have no 2PC/SPU communication.",
        },
        "privacy_fields": {
            "input_pt": candidate_json.get("input_pt"),
            "input_mode": candidate_json.get("input_mode", (candidate_json.get("spu") or {}).get("input_mode")),
            "host_plaintext_pixel_values_materialized": candidate_json.get(
                "host_plaintext_pixel_values_materialized",
                (candidate_json.get("spu") or {}).get("host_plaintext_pixel_values_materialized"),
            ),
            "host_private_share_tensors_loaded": candidate_json.get(
                "host_private_share_tensors_loaded",
                (candidate_json.get("spu") or {}).get("host_private_share_tensors_loaded"),
            ),
            "private_input_paths_redacted": candidate_json.get(
                "private_input_paths_redacted",
                (candidate_json.get("spu") or {}).get("private_input_paths_redacted"),
            ),
        },
        "scope_note": (
            "Use static_whole_forward_same_subset_* as the primary comparator for the secure-static whole-forward path. "
            "The original_plaintext_* fields are retained for context against the full bundle forward. "
            "If raw_secure_graph_before_output_calibration is present, it isolates secure-graph drift from public output calibration."
        ),
    }
    write_json(output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
