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
        Path(__file__).resolve(),
    ]
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate detailed report-scope VPS regression evidence.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--vps-smoke-root", type=Path, default=REPO_ROOT / "results" / "vps_smoke")
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
