#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

LEGACY_CONSISTENCY_POLICY = {
    "argmax_match_ratio_min": 0.99,
    "threshold_match_ratio_min": 0.97,
    "logits_max_abs_error_max": 0.05,
    "probabilities_max_abs_error_max": 0.02,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path_value: str):
    if not path_value:
        return None, None
    path = Path(path_value).expanduser().resolve()
    return path, load_json(path)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fmt(value, digits=6):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def close_float(left, right, abs_tol=1e-6):
    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) <= abs_tol


def nested_get(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_plaintext_eval(payload):
    if not payload:
        return {"available": False}
    metrics = payload.get("metrics") or {}
    return {
        "available": True,
        "label": payload.get("label"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "threshold_json": payload.get("threshold_json"),
        "data_path": payload.get("data_path"),
        "sample_count": payload.get("sample_count"),
        "sample_paths_sha256": payload.get("sample_paths_sha256"),
        "finite_logits": payload.get("finite_logits"),
        "metrics": {
            "argmax_accuracy": metrics.get("argmax_accuracy"),
            "threshold_accuracy": metrics.get("threshold_accuracy"),
            "auc": metrics.get("auc"),
            "eval_loss": metrics.get("eval_loss"),
            "threshold": metrics.get("threshold"),
        },
        "args_snapshot_summary": payload.get("args_snapshot_summary"),
    }


def parse_fairness(payload):
    if not payload:
        return {"available": False}
    transshield = payload.get("transshield") or {}
    mpcvit = nested_get(payload, "external_baselines", "mpcvit") or {}
    return {
        "available": True,
        "title": payload.get("title"),
        "fairness_checks": {
            "accuracy_comparison_is_fair": nested_get(
                payload,
                "fairness_checks",
                "accuracy_comparison_is_fair",
            ),
            "accuracy_comparison_reason": nested_get(
                payload,
                "fairness_checks",
                "accuracy_comparison_reason",
            ),
            "requested_dataset": nested_get(payload, "fairness_checks", "requested_dataset"),
            "transshield_dataset_match": nested_get(
                payload,
                "fairness_checks",
                "transshield",
                "matches_requested_val_sample_paths_sha256",
            ),
            "mpcvit_dataset_match": nested_get(
                payload,
                "fairness_checks",
                "mpcvit",
                "matches_requested_val_sample_count",
            ),
        },
        "fairness_scope": {
            "dataset": nested_get(payload, "fairness_scope", "dataset"),
            "same_dataset_accuracy_comparison": nested_get(
                payload,
                "fairness_scope",
                "same_dataset_accuracy_comparison",
            ),
            "same_protocol_secure_communication_comparison": nested_get(
                payload,
                "fairness_scope",
                "same_protocol_secure_communication_comparison",
            ),
        },
        "comparison": payload.get("comparison"),
        "transshield": {
            "run_dir": transshield.get("run_dir"),
            "sample_count": transshield.get("sample_count"),
            "sample_paths_sha256": transshield.get("sample_paths_sha256"),
            "metrics": transshield.get("metrics"),
            "secure_consistency": transshield.get("secure_consistency"),
            "secure_runtime": transshield.get("secure_runtime"),
        },
        "external_baselines": {
            "mpcvit": {
                "type": mpcvit.get("type"),
                "summary_json": mpcvit.get("summary_json"),
                "sample_count": mpcvit.get("sample_count"),
                "metrics": mpcvit.get("metrics"),
                "aggregate": mpcvit.get("aggregate"),
            }
        },
    }


def parse_network_kth_check(payload):
    if not payload:
        return {"available": False}
    stage_reports = payload.get("stage_reports") or []
    available_reports = [item for item in stage_reports if item.get("available")]
    passed_reports = [item for item in available_reports if item.get("passed")]
    max_abs_error = None
    for item in available_reports:
        error = nested_get(item, "kth_threshold_compare", "max_abs_error")
        if error is None:
            continue
        max_abs_error = error if max_abs_error is None else max(max_abs_error, float(error))
    return {
        "available": True,
        "overall_passed": payload.get("overall_passed"),
        "stage_count": len(stage_reports),
        "available_stage_count": len(available_reports),
        "passed_stage_count": len(passed_reports),
        "stage_pass_ratio": (
            float(len(passed_reports) / len(available_reports)) if available_reports else None
        ),
        "max_threshold_abs_error": max_abs_error,
        "stage_reports": [
            {
                "stage_index": item.get("stage_index"),
                "pruning_layer": item.get("candidate_pruning_layer"),
                "keep_count": item.get("candidate_keep_count"),
                "passed": item.get("passed"),
                "kth_threshold_max_abs_error": nested_get(item, "kth_threshold_compare", "max_abs_error"),
            }
            for item in available_reports
        ],
    }


def parse_tie_check(payload):
    if not payload:
        return {"available": False}
    stage_reports = payload.get("stage_reports") or []
    available_reports = [item for item in stage_reports if item.get("available")]
    semantic_reports = [
        item for item in available_reports if nested_get(item, "semantic_check", "passed") is not None
    ]
    semantic_match_count = sum(
        1 for item in semantic_reports if nested_get(item, "semantic_check", "reconstructed_branch_matches_topk_reference")
    )
    threshold_snap_max_distance = None
    for item in semantic_reports:
        distance = nested_get(item, "semantic_check", "threshold_snap", "max_distance")
        if distance is None:
            continue
        threshold_snap_max_distance = (
            distance
            if threshold_snap_max_distance is None
            else max(threshold_snap_max_distance, float(distance))
        )
    return {
        "available": True,
        "overall_passed": payload.get("overall_passed"),
        "stage_count": len(stage_reports),
        "available_stage_count": len(available_reports),
        "semantic_stage_count": len(semantic_reports),
        "semantic_match_count": semantic_match_count,
        "stage_decision_match_ratio": (
            float(semantic_match_count / len(semantic_reports)) if semantic_reports else None
        ),
        "threshold_snap_max_distance": threshold_snap_max_distance,
        "stage_reports": [
            {
                "stage_index": item.get("stage_index"),
                "pruning_layer": item.get("pruning_layer"),
                "keep_count": item.get("keep_count"),
                "passed": item.get("passed"),
                "semantic_passed": nested_get(item, "semantic_check", "passed"),
                "reconstructed_branch_matches_topk_reference": nested_get(
                    item,
                    "semantic_check",
                    "reconstructed_branch_matches_topk_reference",
                ),
                "selected_equal_mask_exact_match": nested_get(
                    item,
                    "selected_equal_mask_compare",
                    "exact_match",
                ),
                "tie_keep_quota_exact_match": nested_get(
                    item,
                    "tie_keep_quota_compare",
                    "exact_match",
                ),
            }
            for item in available_reports
        ],
    }


def parse_legacy_replay_consistency(compare_payload, fairness_payload):
    if compare_payload:
        comparison = compare_payload.get("comparison") or {}
        return {
            "available": True,
            "source": "plaintext_vs_secure_score_compare.json",
            "argmax_match_ratio": nested_get(comparison, "argmax_predictions", "match_ratio"),
            "threshold_match_ratio": nested_get(comparison, "threshold_predictions", "match_ratio"),
            "logits_max_abs_error": nested_get(comparison, "logits", "max_abs_error"),
            "probabilities_max_abs_error": nested_get(comparison, "probabilities", "max_abs_error"),
            "plaintext_argmax_accuracy": comparison.get("plaintext_argmax_accuracy"),
            "plaintext_threshold_accuracy": comparison.get("plaintext_threshold_accuracy"),
            "secure_argmax_accuracy": comparison.get("secure_argmax_accuracy"),
            "secure_threshold_accuracy": comparison.get("secure_threshold_accuracy"),
            "secure_overall_passed": nested_get(compare_payload, "source_status", "secure_overall_passed"),
            "secure_model_replay_status": nested_get(compare_payload, "source_status", "secure_model_replay_status"),
        }
    if fairness_payload:
        transshield = fairness_payload.get("transshield") or {}
        consistency = transshield.get("secure_consistency") or {}
        return {
            "available": True,
            "source": "fair_external_comparison.json::transshield.secure_consistency",
            "argmax_match_ratio": consistency.get("argmax_match_ratio"),
            "threshold_match_ratio": consistency.get("threshold_match_ratio"),
            "logits_max_abs_error": consistency.get("logits_max_abs_error"),
            "probabilities_max_abs_error": consistency.get("probabilities_max_abs_error"),
            "plaintext_argmax_accuracy": nested_get(transshield, "metrics", "argmax_accuracy"),
            "plaintext_threshold_accuracy": nested_get(transshield, "metrics", "threshold_accuracy"),
            "secure_argmax_accuracy": None,
            "secure_threshold_accuracy": None,
            "secure_overall_passed": consistency.get("overall_passed"),
            "secure_model_replay_status": None,
        }
    return {"available": False}


def parse_e2e_verify(payload):
    if not payload:
        return {"available": False}
    match = payload.get("prediction_match") or {}
    logits_error = payload.get("logits_error") or {}
    probabilities_error = payload.get("probabilities_error") or {}
    return {
        "available": True,
        "source": payload.get("manifest_type"),
        "sample_count": payload.get("sample_count"),
        "slice_count": payload.get("slice_count"),
        "allow_prefix_candidate": payload.get("allow_prefix_candidate"),
        "argmax_match_ratio": match.get("argmax_match_ratio"),
        "threshold_match_ratio": match.get("threshold_match_ratio"),
        "logits_max_abs_error": logits_error.get("max_abs_error"),
        "probabilities_max_abs_error": probabilities_error.get("max_abs_error"),
        "reference_pt": payload.get("reference_pt"),
        "candidate_pt": payload.get("candidate_pt"),
    }


def parse_secret_runtime(payload):
    if not payload:
        return {"available": False}
    accepted_items = payload.get("accepted_items") or []
    mean_elapsed = None
    if accepted_items:
        mean_elapsed = sum(float(item.get("elapsed_sec") or 0.0) for item in accepted_items) / len(accepted_items)
    return {
        "available": True,
        "complete": payload.get("complete"),
        "sample_count": payload.get("sample_count"),
        "accepted_count": payload.get("accepted_count"),
        "unstable_count": payload.get("unstable_count"),
        "pending_count": payload.get("pending_count"),
        "accepted_accuracy": payload.get("accepted_accuracy"),
        "sum_accepted_elapsed_sec": payload.get("sum_accepted_elapsed_sec"),
        "mean_accepted_elapsed_sec": mean_elapsed,
        "outlier_rule": payload.get("outlier_rule"),
        "layer_norm_calibration_json": payload.get("layer_norm_calibration_json"),
        "output_calibration_json": payload.get("output_calibration_json"),
        "accepted_item_preview": accepted_items[:4],
        "unstable_items": (payload.get("unstable_items") or [])[:4],
    }


def build_plaintext_section(plaintext_eval, fairness):
    if plaintext_eval.get("available"):
        return plaintext_eval
    if fairness.get("available"):
        transshield = fairness.get("transshield") or {}
        metrics = transshield.get("metrics") or {}
        return {
            "available": True,
            "label": "transshield_modified_plaintext_from_fairness",
            "checkpoint_path": None,
            "threshold_json": None,
            "data_path": transshield.get("data_path"),
            "sample_count": transshield.get("sample_count"),
            "sample_paths_sha256": transshield.get("sample_paths_sha256"),
            "finite_logits": None,
            "metrics": {
                "argmax_accuracy": metrics.get("argmax_accuracy"),
                "threshold_accuracy": metrics.get("threshold_accuracy"),
                "auc": metrics.get("auc"),
                "eval_loss": metrics.get("eval_loss"),
                "threshold": metrics.get("threshold"),
            },
            "args_snapshot_summary": None,
            "derived_from_fairness_report": True,
        }
    return {"available": False}


def consistency_high(consistency, policy):
    if not consistency.get("available"):
        return None
    argmax_match_ratio = consistency.get("argmax_match_ratio")
    threshold_match_ratio = consistency.get("threshold_match_ratio")
    logits_max_abs_error = consistency.get("logits_max_abs_error")
    probabilities_max_abs_error = consistency.get("probabilities_max_abs_error")
    secure_overall_passed = consistency.get("secure_overall_passed")
    if any(
        value is None
        for value in (
            argmax_match_ratio,
            threshold_match_ratio,
            logits_max_abs_error,
            probabilities_max_abs_error,
        )
    ):
        return None
    return bool(
        float(argmax_match_ratio) >= policy["argmax_match_ratio_min"]
        and float(threshold_match_ratio) >= policy["threshold_match_ratio_min"]
        and float(logits_max_abs_error) <= policy["logits_max_abs_error_max"]
        and float(probabilities_max_abs_error) <= policy["probabilities_max_abs_error_max"]
        and (secure_overall_passed is None or bool(secure_overall_passed))
    )


def plaintext_fairness_metrics_match(plaintext_section, fairness):
    if not plaintext_section.get("available") or not fairness.get("available"):
        return None
    transshield_metrics = nested_get(fairness, "transshield", "metrics") or {}
    plaintext_metrics = plaintext_section.get("metrics") or {}
    metric_matches = [
        close_float(plaintext_metrics.get("argmax_accuracy"), transshield_metrics.get("argmax_accuracy")),
        close_float(plaintext_metrics.get("threshold_accuracy"), transshield_metrics.get("threshold_accuracy")),
        close_float(plaintext_metrics.get("auc"), transshield_metrics.get("auc")),
    ]
    if any(match is None for match in metric_matches):
        return None
    return all(metric_matches)


def build_gates(plaintext_section, fairness, kth_check, tie_check, legacy_consistency, e2e_verify, secret_runtime):
    legacy_exact = None
    if legacy_consistency.get("available"):
        legacy_exact = (
            legacy_consistency.get("argmax_match_ratio") == 1.0
            and legacy_consistency.get("threshold_match_ratio") == 1.0
        )
    legacy_high = consistency_high(legacy_consistency, LEGACY_CONSISTENCY_POLICY)
    e2e_exact = None
    if e2e_verify.get("available"):
        e2e_exact = (
            e2e_verify.get("argmax_match_ratio") == 1.0
            and e2e_verify.get("threshold_match_ratio") == 1.0
        )
    e2e_high = consistency_high(e2e_verify, LEGACY_CONSISTENCY_POLICY)
    return {
        "plaintext_fullval_available": plaintext_section.get("available"),
        "fairness_report_available": fairness.get("available"),
        "fairness_comparison_is_fair": nested_get(fairness, "fairness_checks", "accuracy_comparison_is_fair"),
        "fairness_transshield_matches_current_plaintext_fullval": plaintext_fairness_metrics_match(
            plaintext_section,
            fairness,
        ),
        "boundary_kth_check_passed": kth_check.get("overall_passed") if kth_check.get("available") else None,
        "boundary_tie_check_passed": tie_check.get("overall_passed") if tie_check.get("available") else None,
        "boundary_stage_decision_match_full": (
            tie_check.get("stage_decision_match_ratio") == 1.0 if tie_check.get("available") else None
        ),
        "legacy_replay_consistency_exact": legacy_exact,
        "legacy_replay_consistency_high": legacy_high,
        "e2e_same_policy_consistency_exact": e2e_exact,
        "e2e_same_policy_consistency_high": e2e_high,
        "secret_runtime_summary_available": secret_runtime.get("available"),
        "secret_runtime_complete": secret_runtime.get("complete") if secret_runtime.get("available") else None,
        "secret_runtime_no_unstable_items": (
            secret_runtime.get("unstable_count") == 0 if secret_runtime.get("available") else None
        ),
        "secret_runtime_no_pending_items": (
            secret_runtime.get("pending_count") == 0 if secret_runtime.get("available") else None
        ),
    }


def build_readiness(gates):
    p0_core = all(
        value is True
        for value in (
            gates.get("plaintext_fullval_available"),
            gates.get("fairness_comparison_is_fair"),
            gates.get("fairness_transshield_matches_current_plaintext_fullval"),
            gates.get("boundary_kth_check_passed"),
            gates.get("boundary_tie_check_passed"),
            gates.get("boundary_stage_decision_match_full"),
        )
    )
    consistency_ready = any(
        value is True
        for value in (
            gates.get("legacy_replay_consistency_exact"),
            gates.get("legacy_replay_consistency_high"),
            gates.get("e2e_same_policy_consistency_exact"),
            gates.get("e2e_same_policy_consistency_high"),
        )
    )
    secret_ready = all(
        value is True
        for value in (
            gates.get("secret_runtime_summary_available"),
            gates.get("secret_runtime_complete"),
            gates.get("secret_runtime_no_unstable_items"),
            gates.get("secret_runtime_no_pending_items"),
        )
    )
    if p0_core and consistency_ready and secret_ready:
        return {
            "status": "p0_delivery_closure_ready",
            "reason": "plaintext/fairness/boundary/consistency/secret-runtime 五个闭环都已有当前证据。",
        }
    if p0_core and consistency_ready:
        return {
            "status": "p0_core_ready_but_secret_runtime_incomplete",
            "reason": "方法、legacy consistency 与 fairness 已收口，但 guarded secret runtime 还缺完整稳定证据。",
        }
    return {
        "status": "p0_closure_incomplete",
        "reason": "至少一个核心闭环仍缺当前证据或当前输入文件未提供。",
    }


def build_markdown(report):
    plaintext = report["sections"]["plaintext_fullval"]
    fairness = report["sections"]["fairness"]
    boundary = report["sections"]["boundary"]
    legacy = report["sections"]["legacy_replay_consistency"]
    e2e_verify = report["sections"]["e2e_same_policy_consistency"]
    secret_runtime = report["sections"]["secret_runtime"]
    gates = report["gates"]
    readiness = report["readiness"]

    lines = [
        "# Transshield Delivery Acceptance Report",
        "",
        f"- bundle_dir: `{report.get('bundle_dir') or 'N/A'}`",
        f"- readiness: `{readiness.get('status')}`",
        f"- reason: {readiness.get('reason')}",
        "",
        "## Plaintext Full-Val",
        f"- available: `{plaintext.get('available')}`",
        f"- argmax_accuracy: `{fmt(nested_get(plaintext, 'metrics', 'argmax_accuracy'))}`",
        f"- threshold_accuracy: `{fmt(nested_get(plaintext, 'metrics', 'threshold_accuracy'))}`",
        f"- auc: `{fmt(nested_get(plaintext, 'metrics', 'auc'))}`",
        f"- sample_count: `{fmt(plaintext.get('sample_count'))}`",
        "",
        "## Fairness",
        f"- available: `{fairness.get('available')}`",
        f"- accuracy_comparison_is_fair: `{fmt(nested_get(fairness, 'fairness_checks', 'accuracy_comparison_is_fair'))}`",
        f"- fairness_reason: `{nested_get(fairness, 'fairness_checks', 'accuracy_comparison_reason') or 'N/A'}`",
        f"- gap_vs_mpcvit_argmax_pt: `{fmt(nested_get(fairness, 'comparison', 'transshield_minus_mpcvit_argmax_accuracy_pt'))}`",
        f"- gap_vs_mpcvit_threshold_pt: `{fmt(nested_get(fairness, 'comparison', 'transshield_minus_mpcvit_threshold_accuracy_pt'))}`",
        f"- gap_vs_mpcvit_auc: `{fmt(nested_get(fairness, 'comparison', 'transshield_minus_mpcvit_auc'))}`",
        "",
        "## Boundary Checks",
        f"- network_kth_overall_passed: `{fmt(nested_get(boundary, 'network_kth', 'overall_passed'))}`",
        f"- tie_policy_overall_passed: `{fmt(nested_get(boundary, 'tie_policy', 'overall_passed'))}`",
        f"- stage_decision_match_ratio: `{fmt(nested_get(boundary, 'tie_policy', 'stage_decision_match_ratio'))}`",
        f"- max_kth_threshold_abs_error: `{fmt(nested_get(boundary, 'network_kth', 'max_threshold_abs_error'))}`",
        f"- max_threshold_snap_distance: `{fmt(nested_get(boundary, 'tie_policy', 'threshold_snap_max_distance'))}`",
        "",
        "## Legacy Replay Consistency",
        f"- available: `{legacy.get('available')}`",
        f"- argmax_match_ratio: `{fmt(legacy.get('argmax_match_ratio'))}`",
        f"- threshold_match_ratio: `{fmt(legacy.get('threshold_match_ratio'))}`",
        f"- logits_max_abs_error: `{fmt(legacy.get('logits_max_abs_error'))}`",
        f"- probabilities_max_abs_error: `{fmt(legacy.get('probabilities_max_abs_error'))}`",
        "",
        "## E2E Same-Policy Consistency",
        f"- available: `{e2e_verify.get('available')}`",
        f"- argmax_match_ratio: `{fmt(e2e_verify.get('argmax_match_ratio'))}`",
        f"- threshold_match_ratio: `{fmt(e2e_verify.get('threshold_match_ratio'))}`",
        f"- logits_max_abs_error: `{fmt(e2e_verify.get('logits_max_abs_error'))}`",
        f"- probabilities_max_abs_error: `{fmt(e2e_verify.get('probabilities_max_abs_error'))}`",
        "",
        "## Secret Runtime",
        f"- available: `{secret_runtime.get('available')}`",
        f"- complete: `{fmt(secret_runtime.get('complete'))}`",
        f"- accepted_count: `{fmt(secret_runtime.get('accepted_count'))}`",
        f"- unstable_count: `{fmt(secret_runtime.get('unstable_count'))}`",
        f"- pending_count: `{fmt(secret_runtime.get('pending_count'))}`",
        f"- accepted_accuracy: `{fmt(secret_runtime.get('accepted_accuracy'))}`",
        f"- mean_accepted_elapsed_sec: `{fmt(secret_runtime.get('mean_accepted_elapsed_sec'))}`",
        "",
        "## Gates",
    ]
    for key, value in gates.items():
        lines.append(f"- {key}: `{fmt(value)}`")
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate current Transshield delivery-line evidence into one acceptance report."
    )
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--plaintext-eval-json", default="")
    parser.add_argument("--fair-comparison-json", default="")
    parser.add_argument("--network-kth-check-json", default="")
    parser.add_argument("--tie-check-json", default="")
    parser.add_argument("--plaintext-secure-compare-json", default="")
    parser.add_argument("--e2e-verify-json", default="")
    parser.add_argument("--secret-isolated-summary-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main():
    args = build_parser().parse_args()

    plaintext_eval_path, plaintext_eval_payload = load_optional_json(args.plaintext_eval_json)
    fairness_path, fairness_payload = load_optional_json(args.fair_comparison_json)
    kth_path, kth_payload = load_optional_json(args.network_kth_check_json)
    tie_path, tie_payload = load_optional_json(args.tie_check_json)
    legacy_compare_path, legacy_compare_payload = load_optional_json(args.plaintext_secure_compare_json)
    e2e_verify_path, e2e_verify_payload = load_optional_json(args.e2e_verify_json)
    secret_summary_path, secret_summary_payload = load_optional_json(args.secret_isolated_summary_json)

    plaintext_eval = parse_plaintext_eval(plaintext_eval_payload)
    fairness = parse_fairness(fairness_payload)
    plaintext_section = build_plaintext_section(plaintext_eval, fairness)
    network_kth = parse_network_kth_check(kth_payload)
    tie_policy = parse_tie_check(tie_payload)
    legacy_replay = parse_legacy_replay_consistency(legacy_compare_payload, fairness_payload)
    e2e_verify = parse_e2e_verify(e2e_verify_payload)
    secret_runtime = parse_secret_runtime(secret_summary_payload)
    gates = build_gates(
        plaintext_section,
        fairness,
        network_kth,
        tie_policy,
        legacy_replay,
        e2e_verify,
        secret_runtime,
    )
    readiness = build_readiness(gates)

    bundle_dir = str(Path(args.bundle_dir).expanduser().resolve()) if args.bundle_dir else None
    report = {
        "manifest_type": "transshield_delivery_acceptance_report_v0",
        "bundle_dir": bundle_dir,
        "inputs": {
            "plaintext_eval_json": str(plaintext_eval_path) if plaintext_eval_path else None,
            "fair_comparison_json": str(fairness_path) if fairness_path else None,
            "network_kth_check_json": str(kth_path) if kth_path else None,
            "tie_check_json": str(tie_path) if tie_path else None,
            "plaintext_secure_compare_json": str(legacy_compare_path) if legacy_compare_path else None,
            "e2e_verify_json": str(e2e_verify_path) if e2e_verify_path else None,
            "secret_isolated_summary_json": str(secret_summary_path) if secret_summary_path else None,
        },
        "sections": {
            "plaintext_fullval": plaintext_section,
            "fairness": fairness,
            "boundary": {
                "network_kth": network_kth,
                "tie_policy": tie_policy,
            },
            "legacy_replay_consistency": legacy_replay,
            "e2e_same_policy_consistency": e2e_verify,
            "secret_runtime": secret_runtime,
        },
        "gates": gates,
        "readiness": readiness,
        "official_line": {
            "method_core": "masking -> F_mux ; threshold compare -> F_less ; secure sidecar/replay",
            "default_plaintext_bundle": "artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430",
            "default_secret_profile": "secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1",
        },
        "acceptance_policy": {
            "same_policy_consistency_high": LEGACY_CONSISTENCY_POLICY,
        },
    }

    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)

    if args.output_md:
        output_md = Path(args.output_md).expanduser().resolve()
        write_text(output_md, build_markdown(report))

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
