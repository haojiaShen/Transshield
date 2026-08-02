from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from showcase_api.admin_config import (
    REPO_ROOT,
    ShowcaseConfig,
    build_path_replacements,
    relative_to_repo_path,
    relativize_text,
    relativize_value,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_threshold_payload(bundle_dir: Path, manifest: dict) -> dict:
    threshold_path = bundle_dir / "threshold_best.json"
    if threshold_path.exists():
        return _load_json(threshold_path)
    return (((manifest.get("primary") or {}).get("threshold_metrics")) or {})


def list_model_artifacts(config: ShowcaseConfig) -> list[dict]:
    artifacts_root = REPO_ROOT / "artifacts"
    records: list[dict] = []
    if not artifacts_root.exists():
        return records
    path_replacements = build_path_replacements(config.default_train_data_path, config.default_eval_data_path)

    for bundle_dir in sorted(artifacts_root.glob("frozen_bundle_*")):
        args_path = bundle_dir / "args_snapshot.json"
        manifest_path = bundle_dir / "manifest.json"
        if not args_path.exists():
            continue
        args_snapshot = _load_json(args_path)
        manifest = _load_json(manifest_path) if manifest_path.exists() else {}
        threshold_payload = _read_threshold_payload(bundle_dir, manifest)
        primary = manifest.get("primary") or {}
        bundle_name = manifest.get("bundle_name") or bundle_dir.name
        dataset_path = args_snapshot.get("data_path") or ""
        lower_name = bundle_name.lower()
        domain = "finance" if "finance" in lower_name else "medical"
        source_run = primary.get("run_dir") or args_snapshot.get("output_dir") or ""

        records.append(
            {
                "id": bundle_dir.name,
                "name": bundle_name,
                "domain": domain,
                "bundle_name": bundle_name,
                "bundle_dir": relative_to_repo_path(bundle_dir),
                "source_run": relativize_text(str(source_run), path_replacements),
                "dataset_path": relativize_text(str(dataset_path), path_replacements),
                "status": "正式主线" if bundle_dir.resolve() == config.bundle_dir.resolve() else "历史版本",
                "base_rate": _safe_float(args_snapshot.get("base_rate")),
                "secure_static_train_depth": _safe_int(args_snapshot.get("secure_static_train_depth"), 0),
                "cls_distill_weight": _safe_float(args_snapshot.get("cls_distill_weight")),
                "token_distill_weight": _safe_float(args_snapshot.get("token_distill_weight")),
                "threshold_accuracy": _safe_float(
                    threshold_payload.get("best_threshold_accuracy", threshold_payload.get("eval_acc1"))
                ),
                "argmax_accuracy": _safe_float(
                    threshold_payload.get("default_argmax_acc1", threshold_payload.get("default_acc1"))
                ),
                "auc": _safe_float(threshold_payload.get("auc")),
                "teacher_checkpoint_path": relativize_text(
                    str(args_snapshot.get("teacher_checkpoint_path") or ""),
                    path_replacements,
                ),
                "manifest_path": relative_to_repo_path(manifest_path) if manifest_path.exists() else "",
                "args_snapshot_path": relative_to_repo_path(args_path),
                "threshold_path": (
                    relative_to_repo_path(bundle_dir / "threshold_best.json")
                    if (bundle_dir / "threshold_best.json").exists()
                    else ""
                ),
            }
        )

    return records


def build_results_catalog(config: ShowcaseConfig) -> dict:
    path_replacements = build_path_replacements(config.default_train_data_path, config.default_eval_data_path)
    result_files = [
        ("正式阈值校准", REPO_ROOT / "results" / "final" / "medical_dynamic_threshold_calibration_final.json"),
        ("正式 AUC 参考", REPO_ROOT / "results" / "final" / "medical_dynamic_auc_reference_final.json"),
        ("通信量画像", REPO_ROOT / "results" / "communication" / "mainline_communication_profile_final.json"),
        ("蒸馏补偿配对", REPO_ROOT / "results" / "distillation_compensation" / "codex_distill_pair_epoch1_20260611_230112" / "distill_compensation_pair_compare.json"),
        ("部署对齐控制", REPO_ROOT / "results" / "secure_static_train_depth_evidence" / "codex_depth_pair_epoch1_20260611_230112" / "secure_static_train_depth_pair_compare.json"),
        ("协议层 fuzz", REPO_ROOT / "results" / "fuzzing" / "protocol_fuzz_final.json"),
        ("控制面 guard", REPO_ROOT / "results" / "guard_stress" / "guard_stress_final.json"),
    ]

    sections = []
    for label, path in result_files:
        if not path.exists():
            continue
        payload = _load_json(path)
        sections.append(
            {
                "label": label,
                "path": relative_to_repo_path(path),
                "payload": relativize_value(payload, path_replacements),
            }
        )

    return {
        "formal_metrics": {
            "threshold_accuracy": config.formal_threshold_accuracy,
            "auc": config.formal_auc,
            "sec_per_sample": config.formal_sec_per_sample,
            "dual_total_gib": config.formal_dual_total_gib,
            "threshold": config.formal_threshold,
            "bundle_dir": relative_to_repo_path(config.bundle_dir),
        },
        "sections": sections,
    }


def detect_gpu_summary() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "message": f"nvidia-smi unavailable: {exc}", "devices": []}

    if result.returncode != 0:
        return {"available": False, "message": result.stderr.strip() or "nvidia-smi returned non-zero", "devices": []}

    devices = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        devices.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": parts[2],
                "utilization_gpu_percent": parts[3],
            }
        )
    return {"available": bool(devices), "message": "ok" if devices else "no GPU detected", "devices": devices}


def build_overview(config: ShowcaseConfig, jobs: list[dict], models: list[dict]) -> dict:
    counts = {name: 0 for name in ["queued", "starting", "running", "postprocessing", "completed", "failed", "cancelled"]}
    for job in jobs:
        status = str(job.get("status", "queued"))
        if status in counts:
            counts[status] += 1

    recent_completed = sorted(
        [job for job in jobs if job.get("status") == "completed"],
        key=lambda item: item.get("finished_at") or "",
        reverse=True,
    )[:5]
    formal_model = next((item for item in models if item.get("bundle_dir") == relative_to_repo_path(config.bundle_dir)), None)

    return {
        "queue": counts,
        "recent_completed": recent_completed,
        "formal_model": formal_model,
        "gpu": detect_gpu_summary(),
        "environment": {
            "python_bin": relative_to_repo_path(config.python_bin),
            "python_version": sys.version.split()[0],
            "repo_root": ".",
            "train_output_root": relative_to_repo_path(config.train_output_root),
            "bundle_output_root": relative_to_repo_path(config.bundle_output_root),
            "job_root": relative_to_repo_path(config.admin_job_root),
            "default_train_data_path": relative_to_repo_path(config.default_train_data_path),
            "default_eval_data_path": relative_to_repo_path(config.default_eval_data_path),
            "max_concurrent_train_jobs": config.max_concurrent_train_jobs,
            "runtime_mode": config.runtime_mode,
        },
    }
