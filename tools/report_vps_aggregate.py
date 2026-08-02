#!/usr/bin/env python3
"""Aggregate detailed VPS-only regression evidence without touching formal results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.report_vps_test import read_json, sha256_file, write_json


DEFAULT_MATRIX = REPO_ROOT / "configs" / "report_vps_test_matrix.json"


def evidence_file(path: Path) -> dict:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def reduction_fraction(baseline: float, candidate: float) -> float:
    return 1.0 - float(candidate) / float(baseline)


def measurement_value(path: Path, key: str) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            return int(line.split("=", 1)[1])
    raise ValueError(f"{key} missing from {path}")


def preview_probabilities(payload: dict) -> list[list[float]]:
    return (payload.get("prediction_preview") or {}).get("probabilities") or []


def probability_difference(left: list[list[float]], right: list[list[float]]) -> float | None:
    values = [abs(float(a) - float(b)) for x, y in zip(left, right) for a, b in zip(x, y)]
    return max(values) if values else None


def same_vps_ab(smoke_root: Path) -> dict:
    specs = {
        "single_sample_all_optimizations": {
            "baseline_json": smoke_root / "medical_00305_depth10_secret_compact_fm64_baseline.json",
            "candidate_json": smoke_root / "medical_00305_depth10_final_code_v4.json",
            "baseline_network_json": smoke_root / "medical_00305_depth10_secret_compact_fm64_baseline.metrics.json",
            "candidate_network_text": smoke_root / "medical_00305_depth10_final_code_v4.measurement.txt",
        },
        "four_sample_value_fusion_ablation": {
            "baseline_json": smoke_root / "representative4_depth10_secret_compact_fm64_cls_only.json",
            "candidate_json": smoke_root / "representative4_depth10_secret_compact_fm64_value_fusion_v4.json",
            "baseline_network_json": smoke_root / "representative4_depth10_secret_compact_fm64_cls_only.metrics.json",
            "candidate_network_text": smoke_root / "representative4_depth10_secret_compact_fm64_value_fusion_v4.measurement.txt",
        },
    }
    results = {}
    for name, paths in specs.items():
        baseline = read_json(paths["baseline_json"])
        candidate = read_json(paths["candidate_json"])
        baseline_tx = int(read_json(paths["baseline_network_json"])["lo_tx_bytes"])
        candidate_tx = measurement_value(paths["candidate_network_text"], "lo_tx_bytes")
        baseline_probs = preview_probabilities(baseline)
        candidate_probs = preview_probabilities(candidate)
        baseline_predictions = [max(range(len(row)), key=row.__getitem__) for row in baseline_probs]
        candidate_predictions = [max(range(len(row)), key=row.__getitem__) for row in candidate_probs]
        results[name] = {
            "sample_count": int(candidate["sample_count"]),
            "baseline_elapsed_sec": float(baseline["elapsed_sec"]),
            "candidate_elapsed_sec": float(candidate["elapsed_sec"]),
            "elapsed_reduction_fraction": reduction_fraction(baseline["elapsed_sec"], candidate["elapsed_sec"]),
            "baseline_loopback_tx_bytes": baseline_tx,
            "candidate_loopback_tx_bytes": candidate_tx,
            "loopback_tx_reduction_fraction": reduction_fraction(baseline_tx, candidate_tx),
            "argmax_match_ratio": sum(a == b for a, b in zip(baseline_predictions, candidate_predictions))
            / len(candidate_predictions),
            "probabilities_max_abs_error": probability_difference(baseline_probs, candidate_probs),
            "evidence": {key: evidence_file(value) for key, value in paths.items()},
        }
    return results


def medical_acceptance_checks(matrix: dict, medical: dict) -> list[dict]:
    acceptance = matrix["acceptance"]["medical_secure_candidate"]
    report = matrix["datasets"]["medical_secure_deployment_batch"]["report_expected"]
    accuracy_drop_points = (report["threshold_accuracy"] - medical["threshold_accuracy"]) * 100.0
    return [
        {"name": "finite_logits", "actual": medical["finite_logits"], "expected": True, "passed": medical["finite_logits"] is True},
        {
            "name": "argmax_match_ratio_vs_vps_cpu_reference",
            "actual": medical["reference_comparison"]["argmax_match_ratio"],
            "minimum": acceptance["minimum_argmax_match_ratio_vs_vps_cpu_reference"],
            "passed": medical["reference_comparison"]["argmax_match_ratio"]
            >= acceptance["minimum_argmax_match_ratio_vs_vps_cpu_reference"],
        },
        {
            "name": "threshold_match_ratio_vs_vps_cpu_reference",
            "actual": medical["reference_comparison"]["threshold_match_ratio"],
            "minimum": acceptance["minimum_threshold_match_ratio_vs_vps_cpu_reference"],
            "passed": medical["reference_comparison"]["threshold_match_ratio"]
            >= acceptance["minimum_threshold_match_ratio_vs_vps_cpu_reference"],
            "classification": "diagnostic_fixed_point_agreement_gate",
        },
        {
            "name": "threshold_accuracy",
            "actual": medical["threshold_accuracy"],
            "minimum": acceptance["minimum_threshold_accuracy"],
            "passed": medical["threshold_accuracy"] >= acceptance["minimum_threshold_accuracy"],
        },
        {
            "name": "accuracy_drop_vs_report_percentage_points",
            "actual": accuracy_drop_points,
            "maximum": acceptance["maximum_accuracy_drop_vs_report_points"],
            "passed": accuracy_drop_points <= acceptance["maximum_accuracy_drop_vs_report_points"],
        },
    ]


def finance_acceptance_checks(matrix: dict, finance: dict) -> list[dict]:
    acceptance = matrix["acceptance"]["finance_secure_candidate"]
    return [
        {"name": "finite_logits", "actual": finance["finite_logits"], "expected": True, "passed": finance["finite_logits"] is True},
        {
            "name": "argmax_match_ratio_vs_vps_cpu_reference",
            "actual": finance["reference_comparison"]["argmax_match_ratio"],
            "minimum": acceptance["minimum_argmax_match_ratio_vs_vps_cpu_reference"],
            "passed": finance["reference_comparison"]["argmax_match_ratio"]
            >= acceptance["minimum_argmax_match_ratio_vs_vps_cpu_reference"],
        },
        {
            "name": "threshold_match_ratio_vs_vps_cpu_reference",
            "actual": finance["reference_comparison"]["threshold_match_ratio"],
            "minimum": acceptance["minimum_threshold_match_ratio_vs_vps_cpu_reference"],
            "passed": finance["reference_comparison"]["threshold_match_ratio"]
            >= acceptance["minimum_threshold_match_ratio_vs_vps_cpu_reference"],
        },
    ]


def append_gate_check(
    checks: list[dict],
    name: str,
    actual,
    *,
    expected=None,
    minimum=None,
    maximum=None,
    passed: bool | None = None,
) -> None:
    if passed is None:
        if minimum is not None:
            passed = actual is not None and actual >= minimum
        elif maximum is not None:
            passed = actual is not None and actual <= maximum
        else:
            passed = actual == expected
    check = {"name": name, "actual": actual, "passed": bool(passed)}
    if expected is not None:
        check["expected"] = expected
    if minimum is not None:
        check["minimum"] = minimum
    if maximum is not None:
        check["maximum"] = maximum
    checks.append(check)


def summary_hostname(summary: dict) -> str | None:
    network = summary.get("network") or {}
    return (network.get("before") or {}).get("hostname")


def summary_network_interface(summary: dict) -> str | None:
    network = summary.get("network") or {}
    return network.get("interface") or (network.get("before") or {}).get("interface")


def report_section_accounting(
    matrix: dict,
    run_root: Path,
    change_scope: str,
) -> tuple[list[dict], list[dict]]:
    contract = matrix["report_update_contract"]
    frozen_groups = contract["frozen_when_unaffected"]
    conditional = contract["rerun_if_change_scope_includes"]
    sections = sorted({section for group in frozen_groups.values() for section in group})
    if change_scope == "full":
        required_reruns = {section for group in conditional.values() for section in group}
    else:
        required_reruns = set(conditional.get(change_scope, []))

    accounting = []
    checks = []
    evidence_root = run_root / "report_section_evidence"
    for section in sections:
        if section not in required_reruns:
            accounting.append(
                {
                    "section": section,
                    "disposition": "frozen_unaffected",
                    "basis": "report hash plus dataset and bundle inventory",
                }
            )
            continue
        evidence_path = evidence_root / f"{section}.json"
        evidence = read_json(evidence_path) if evidence_path.is_file() else None
        passed = bool(evidence and evidence.get("passed") is True)
        accounting.append(
            {
                "section": section,
                "disposition": "rerun" if passed else "missing_required_rerun",
                "evidence": evidence_file(evidence_path),
            }
        )
        append_gate_check(
            checks,
            f"report_section.{section}.rerun_passed",
            passed,
            expected=True,
        )
    return accounting, checks


def report_update_readiness(
    matrix: dict,
    payload: dict,
    run_root: Path,
    baseline_path: Path,
    change_scope: str,
) -> dict:
    contract = matrix["report_update_contract"]
    quality = contract["quality_gates"]
    checks = []

    inventory = payload["data_and_bundle_inventory"]
    append_gate_check(checks, "inventory.passed", inventory["passed"], expected=True)

    preprocessing = payload["preprocessing_reproduction"]
    for domain in ("medical", "finance"):
        append_gate_check(
            checks,
            f"preprocessing.{domain}.passed",
            preprocessing[domain].get("passed"),
            expected=True,
        )

    medical_full = payload["medical_full_validation"]["actual"]
    append_gate_check(checks, "medical524.sample_count", medical_full.get("sample_count"), expected=524)
    append_gate_check(checks, "medical524.per_sample_count", len(medical_full.get("per_sample") or []), expected=524)
    append_gate_check(checks, "medical524.finite_logits", medical_full.get("finite_logits"), expected=True)
    append_gate_check(
        checks,
        "medical524.threshold_accuracy",
        medical_full.get("threshold_accuracy"),
        minimum=quality["medical_524_minimum_threshold_accuracy"],
    )
    append_gate_check(
        checks,
        "medical524.auc",
        medical_full.get("auc"),
        minimum=quality["medical_524_minimum_auc"],
    )

    secure = payload["secure_inference"]
    medical = secure["medical_32"]["actual"]
    append_gate_check(checks, "medical32.sample_count", medical.get("sample_count"), expected=32)
    append_gate_check(checks, "medical32.per_sample_count", len(medical.get("per_sample") or []), expected=32)
    append_gate_check(checks, "medical32.finite_logits", medical.get("finite_logits"), expected=True)
    append_gate_check(checks, "medical32.finite_probabilities", medical.get("finite_probabilities"), expected=True)
    append_gate_check(
        checks,
        "medical32.elapsed_sec_recorded",
        medical.get("elapsed_sec"),
        passed=bool(medical.get("elapsed_sec") and medical["elapsed_sec"] > 0),
    )
    medical_network_bytes = (medical.get("network") or {}).get("total_bytes")
    append_gate_check(
        checks,
        "medical32.communication_recorded",
        medical_network_bytes,
        passed=bool(medical_network_bytes and medical_network_bytes > 0),
    )
    append_gate_check(
        checks,
        "medical32.threshold_accuracy",
        medical.get("threshold_accuracy"),
        minimum=quality["medical_32_minimum_threshold_accuracy"],
    )
    append_gate_check(
        checks,
        "medical32.auc",
        medical.get("auc"),
        minimum=quality["medical_32_minimum_auc"],
    )

    finance = secure["finance_8"]["actual"]
    append_gate_check(checks, "finance8.sample_count", finance.get("sample_count"), expected=8)
    append_gate_check(checks, "finance8.per_sample_count", len(finance.get("per_sample") or []), expected=8)
    append_gate_check(checks, "finance8.finite_logits", finance.get("finite_logits"), expected=True)
    append_gate_check(
        checks,
        "finance8.elapsed_sec_recorded",
        finance.get("elapsed_sec"),
        passed=bool(finance.get("elapsed_sec") and finance["elapsed_sec"] > 0),
    )
    finance_network_bytes = (finance.get("network") or {}).get("total_bytes")
    append_gate_check(
        checks,
        "finance8.communication_recorded",
        finance_network_bytes,
        passed=bool(finance_network_bytes and finance_network_bytes > 0),
    )
    finance_reference = finance.get("reference_comparison") or {}
    append_gate_check(
        checks,
        "finance8.argmax_match_ratio",
        finance_reference.get("argmax_match_ratio"),
        minimum=quality["finance_8_minimum_argmax_match_ratio"],
    )
    append_gate_check(
        checks,
        "finance8.threshold_match_ratio",
        finance_reference.get("threshold_match_ratio"),
        minimum=quality["finance_8_minimum_threshold_match_ratio"],
    )

    robustness = payload["robustness"]
    expected_robustness = matrix["robustness"]["expected"]
    append_gate_check(
        checks,
        "robustness.protocol_passed",
        robustness["protocol_passed"],
        expected=expected_robustness["protocol_passed"],
    )
    append_gate_check(
        checks,
        "robustness.guard_passed",
        robustness["guard_passed"],
        expected=expected_robustness["guard_passed"],
    )
    append_gate_check(
        checks,
        "robustness.fd_socket_no_leak",
        robustness["fd_and_socket_no_leak_count"],
        expected=expected_robustness["fd_socket_no_leak"],
    )
    append_gate_check(
        checks,
        "robustness.steady_state_recovered",
        robustness["steady_state_recovered_count"],
        expected=expected_robustness["steady_state_recovered"],
    )
    append_gate_check(
        checks,
        "robustness.guard_inflight_recovered",
        robustness["guard_inflight_recovered_count"],
        expected=expected_robustness["guard_inflight_recovered"],
    )
    append_gate_check(checks, "code_tests.passed", payload["code_tests"]["passed"], expected=True)
    append_gate_check(
        checks,
        "code_tests.test_count",
        payload["code_tests"].get("test_count"),
        minimum=quality["minimum_code_tests"],
    )

    missing_artifacts = [
        name
        for name, evidence in payload["evidence_artifacts"].items()
        if not evidence["exists"]
    ]
    append_gate_check(
        checks,
        "evidence_artifacts.complete",
        missing_artifacts,
        expected=[],
    )

    required_privacy = contract["required_privacy_facts"]
    for domain in ("medical", "finance"):
        actual_privacy = payload["privacy_boundary"][domain]
        for key, expected in required_privacy.items():
            append_gate_check(
                checks,
                f"privacy.{domain}.{key}",
                actual_privacy.get(key),
                expected=expected,
            )

    baseline_path = baseline_path.expanduser().resolve()
    baseline = read_json(baseline_path) if baseline_path.is_file() else None
    append_gate_check(checks, "same_vps_full_baseline.exists", baseline is not None, expected=True)
    if baseline is not None:
        append_gate_check(checks, "same_vps_full_baseline.sample_count", baseline.get("sample_count"), expected=32)
        append_gate_check(
            checks,
            "same_vps_full_baseline.per_sample_count",
            len(baseline.get("per_sample") or []),
            expected=32,
        )
        append_gate_check(
            checks,
            "same_vps_full_baseline.finite_logits",
            baseline.get("finite_logits"),
            expected=True,
        )
        append_gate_check(
            checks,
            "same_vps_full_baseline.dataset_key",
            baseline.get("dataset_key"),
            expected=medical.get("dataset_key"),
        )
        append_gate_check(
            checks,
            "same_vps_full_baseline.sample_list_sha256",
            baseline.get("sample_list_sha256"),
            expected=medical.get("sample_list_sha256"),
        )
        append_gate_check(
            checks,
            "same_vps_full_baseline.hostname",
            summary_hostname(baseline),
            expected=summary_hostname(medical),
        )
        append_gate_check(
            checks,
            "same_vps_full_baseline.network_interface",
            summary_network_interface(baseline),
            expected=summary_network_interface(medical),
        )
        if change_scope in {"runtime_only", "control_plane", "operator_proxy_harness"}:
            append_gate_check(
                checks,
                "same_vps_full_baseline.threshold",
                baseline.get("threshold"),
                expected=medical.get("threshold"),
            )
        baseline_elapsed = baseline.get("elapsed_sec")
        append_gate_check(
            checks,
            "medical32.time_improved_vs_same_vps_baseline",
            medical.get("elapsed_sec"),
            maximum=baseline_elapsed,
            passed=bool(
                baseline_elapsed
                and medical.get("elapsed_sec")
                and medical["elapsed_sec"] < baseline_elapsed
            ),
        )
        baseline_bytes = (baseline.get("network") or {}).get("total_bytes")
        append_gate_check(
            checks,
            "medical32.communication_not_worse_vs_same_vps_baseline",
            medical_network_bytes,
            maximum=baseline_bytes,
            passed=bool(
                baseline_bytes
                and medical_network_bytes
                and medical_network_bytes <= baseline_bytes
            ),
        )

    section_accounting, section_checks = report_section_accounting(
        matrix,
        run_root,
        change_scope,
    )
    checks.extend(section_checks)
    ready = all(check["passed"] for check in checks)
    return {
        "classification": "report_update_ready" if ready else "screening_only",
        "report_update_ready": ready,
        "change_scope": change_scope,
        "same_vps_full_baseline": evidence_file(baseline_path),
        "mandatory_reruns": contract["mandatory_reruns"],
        "report_section_accounting": section_accounting,
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
    }


def build_aggregate(args) -> dict:
    run_root = args.run_root.expanduser().resolve()
    matrix = read_json(args.matrix)
    inventory = read_json(run_root / "inventory.json")
    medical_full = read_json(run_root / "medical524_cpu_depth10_summary.json")
    medical = read_json(run_root / "medical32_spu_latest_summary.json")
    finance = read_json(run_root / "finance8_spu_latest_summary.json")
    protocol = read_json(run_root / "protocol_fuzz_vps.json")
    guard = read_json(run_root / "guard_stress_vps.json")
    protocol_cases = protocol.get("cases") or protocol.get("results") or []
    guard_checks = guard.get("checks") or []
    all_robustness = protocol_cases + guard_checks
    medical_checks = medical_acceptance_checks(matrix, medical)
    finance_checks = finance_acceptance_checks(matrix, finance)
    report_medical = matrix["datasets"]["medical_secure_deployment_batch"]["report_expected"]
    report_finance = matrix["datasets"]["finance_boundary_stress"]["report_expected"]

    unit_text = ""
    for name in ["unittest_vps.stdout.log", "unittest_vps.stderr.log"]:
        path = run_root / name
        if path.is_file():
            unit_text += path.read_text(encoding="utf-8", errors="replace") + "\n"
    unit_match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", unit_text)
    unit_exit = int((run_root / "unittest_vps.exit_code").read_text(encoding="utf-8").strip())

    artifact_names = [
        "inventory.json",
        "medical524_cpu_depth10_summary.json",
        "medical32_preprocess_compare.json",
        "medical32_cpu_depth10_summary.json",
        "medical32_spu_latest.json",
        "medical32_spu_latest_summary.json",
        "finance8_preprocess_compare.json",
        "finance8_cpu_depth12_summary.json",
        "finance8_spu_latest.json",
        "finance8_spu_latest_summary.json",
        "protocol_fuzz_vps.json",
        "guard_stress_vps.json",
        "unittest_vps.stdout.log",
        "unittest_vps.stderr.log",
        "unittest_vps.exit_code",
    ]
    source_files = [
        REPO_ROOT / "integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py",
        REPO_ROOT / "integrations/transshield_runtime/e2e_secure_vit/spu_static_vit.py",
        REPO_ROOT / "integrations/transshield_runtime/e2e_secure_vit/secure_pruning_ops.py",
        REPO_ROOT / "showcase_api/app.py",
        REPO_ROOT / "tools/showcase_protocol_fuzz.py",
        REPO_ROOT / "tools/showcase_guard_stress.py",
        REPO_ROOT / "tools/report_vps_test.py",
        DEFAULT_MATRIX,
        Path(__file__).resolve(),
    ]
    payload = {
        "manifest_type": "transshield_report_vps_regression_aggregate_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": matrix["report"],
        "execution_policy": matrix["execution_policy"],
        "vps_environment": inventory["environment"],
        "report_reference_environment": matrix["report_reference_environment"],
        "data_and_bundle_inventory": {
            "passed": inventory["passed"],
            "checks": inventory["checks"],
            "dataset_content_manifest_sha256": {
                key: value["dataset_content_manifest_sha256"] for key, value in inventory["datasets"].items()
            },
        },
        "preprocessing_reproduction": {
            "medical": read_json(run_root / "medical32_preprocess_compare.json"),
            "finance": read_json(run_root / "finance8_preprocess_compare.json"),
        },
        "medical_full_validation": {
            "actual": medical_full,
            "report_expected": matrix["datasets"]["medical_full_validation"]["expected"],
        },
        "secure_inference": {
            "medical_32": {
                "actual": medical,
                "report_expected": report_medical,
                "acceptance_checks": medical_checks,
                "primary_task_accuracy_gate_passed": next(x["passed"] for x in medical_checks if x["name"] == "threshold_accuracy"),
                "all_diagnostic_gates_passed": all(x["passed"] for x in medical_checks),
                "cross_environment_observation": {
                    "elapsed_reduction_fraction": reduction_fraction(report_medical["elapsed_sec"], medical["elapsed_sec"]),
                    "communication_reduction_fraction": reduction_fraction(report_medical["dual_total_gib"], medical["network"]["total_gib"]),
                    "comparison_status": "not_optimization_proof_different_hardware_and_communication_counter",
                },
            },
            "finance_8": {
                "actual": finance,
                "report_expected": report_finance,
                "acceptance_checks": finance_checks,
                "all_gates_passed": all(x["passed"] for x in finance_checks),
                "cross_environment_observation": {
                    "elapsed_reduction_fraction": reduction_fraction(report_finance["elapsed_sec"], finance["elapsed_sec"]),
                    "communication_reduction_fraction": reduction_fraction(report_finance["dual_total_gib"], finance["network"]["total_gib"]),
                    "comparison_status": "not_optimization_proof_different_hardware_and_communication_counter",
                },
            },
        },
        "same_vps_optimization_ab": same_vps_ab(args.vps_smoke_root.expanduser().resolve()),
        "robustness": {
            "protocol_passed": sum(x.get("passed") is True for x in protocol_cases),
            "protocol_total": len(protocol_cases),
            "guard_passed": sum(x.get("passed") is True for x in guard_checks),
            "guard_total": len(guard_checks),
            "all_passed": protocol.get("passed") is True and guard.get("passed") is True,
            "fd_and_socket_no_leak_count": sum(
                x.get("system_state", {}).get("delta", {}).get("fd_count") == 0
                and x.get("system_state", {}).get("delta", {}).get("socket_fd_count") == 0
                for x in all_robustness
            ),
            "fd_and_socket_observation_count": len(all_robustness),
            "steady_state_recovered_count": sum(
                x.get("system_state", {}).get("stable") is True for x in protocol_cases
            ),
            "guard_inflight_recovered_count": sum(x.get("inflight_recovered") is True for x in guard_checks),
            "truncated_body_transport_note": (
                "The VPS recorded streaming_body_reader/truncated_body, but the half-closed client received no HTTP "
                "response; this differs from the report environment's HTTP 400 disposition."
            ),
            "protocol_cases": protocol_cases,
            "guard_checks": guard_checks,
        },
        "code_tests": {
            "exit_code": unit_exit,
            "passed": unit_exit == 0,
            "test_count": int(unit_match.group(1)) if unit_match else None,
            "elapsed_sec": float(unit_match.group(2)) if unit_match else None,
        },
        "privacy_boundary": {
            "medical": medical["privacy_facts"],
            "finance": finance["privacy_facts"],
            "wording_rule": matrix["acceptance"]["privacy_wording_rule"],
            "production_limit": (
                "The runner loads the plaintext model bundle before secret SPU placement, and the input shares are "
                "debug float additive shares; this is not production-independent P1/P2 ingestion."
            ),
        },
        "evidence_artifacts": {name: evidence_file(run_root / name) for name in artifact_names},
        "source_file_hashes": {str(path.relative_to(REPO_ROOT)): evidence_file(path) for path in source_files},
    }
    baseline_path = args.medical_baseline_summary
    if baseline_path is None:
        baseline_path = run_root / "medical32_spu_baseline_summary.json"
    payload["report_update_readiness"] = report_update_readiness(
        matrix,
        payload,
        run_root,
        baseline_path,
        args.change_scope,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate detailed report-scope VPS regression evidence.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--vps-smoke-root", type=Path, default=REPO_ROOT / "results" / "vps_smoke")
    parser.add_argument("--medical-baseline-summary", type=Path)
    parser.add_argument(
        "--change-scope",
        choices=[
            "runtime_only",
            "model_graph",
            "training_or_weights",
            "operator_proxy_harness",
            "control_plane",
            "full",
        ],
        default="runtime_only",
    )
    parser.add_argument("--require-report-update-ready", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_aggregate(args)
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out.expanduser().resolve()),
                "inventory_passed": payload["data_and_bundle_inventory"]["passed"],
                "medical_primary_accuracy_gate_passed": payload["secure_inference"]["medical_32"]["primary_task_accuracy_gate_passed"],
                "medical_all_diagnostic_gates_passed": payload["secure_inference"]["medical_32"]["all_diagnostic_gates_passed"],
                "finance_all_gates_passed": payload["secure_inference"]["finance_8"]["all_gates_passed"],
                "robustness_passed": payload["robustness"]["all_passed"],
                "code_tests_passed": payload["code_tests"]["passed"],
                "result_classification": payload["report_update_readiness"]["classification"],
                "report_update_ready": payload["report_update_readiness"]["report_update_ready"],
                "failed_report_update_checks": payload["report_update_readiness"]["failed_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_report_update_ready and not payload["report_update_readiness"]["report_update_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
