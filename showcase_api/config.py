from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from showcase_api.preprocessing import DEFAULT_EVAL_CROP_PCT, eval_resize_shorter_side


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def getenv_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RunnerProfile:
    static_depth_limit: int
    spu_batch_size: int
    spu_params_mode: str
    spu_layer_norm_policy: str
    spu_attention_policy: str
    spu_activation_override: str
    spu_activation_clip_value: float
    spu_secure_pruning_mode: str
    spu_secure_pruning_network: str
    spu_final_block_cls_only: bool
    spu_uniform_attention_value_fusion: bool
    spu_compile_cache_dir: Path


@dataclass(frozen=True)
class ShowcaseConfig:
    bundle_dir: Path
    threshold_source_json: Path
    communication_json: Path
    demo_summary_json: Path
    health_bind_dist: Path
    audit_dir: Path
    run_dir: Path
    spu_config_path: Path
    python_bin: str
    runtime_mode: str
    accepted_sleep_sec: float
    allowed_mime_types: tuple[str, ...]
    max_file_size_bytes: int
    max_request_bytes: int
    max_image_dimension: int
    estimated_wait_seconds: int
    input_size: int
    channels: int
    class_names: tuple[str, str]
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    norm_clip_abs: float
    eval_crop_pct: float
    share_abs_guard: float
    quality_drift_tolerance: float
    per_ip_window_limit: int
    per_ip_window_seconds: int
    per_ip_inflight_limit: int
    global_inflight_limit: int
    replay_nonce_ttl_seconds: int
    replay_payload_ttl_seconds: int
    replay_nonce_capacity: int
    replay_payload_capacity: int
    request_manifest_type: str
    request_contract_version: str
    runner_profile: RunnerProfile
    formal_threshold: float
    formal_threshold_accuracy: float
    formal_auc: float
    formal_sec_per_sample: float
    formal_dual_total_gib: float

    @property
    def expected_shape(self) -> list[int]:
        return [1, self.channels, self.input_size, self.input_size]

    @property
    def float_count(self) -> int:
        return self.channels * self.input_size * self.input_size

    @property
    def share_byte_count(self) -> int:
        return self.float_count * 4

    @property
    def eval_resize_shorter_side(self) -> int:
        return eval_resize_shorter_side(self.input_size, self.eval_crop_pct)


def load_showcase_config() -> ShowcaseConfig:
    bundle_dir = getenv_path(
        "TRANSSHIELD_SHOWCASE_BUNDLE_DIR",
        REPO_ROOT / "artifacts" / "frozen_bundle_medical_dynamic_mainline",
    )
    threshold_source_json = getenv_path(
        "TRANSSHIELD_SHOWCASE_THRESHOLD_JSON",
        REPO_ROOT / "results" / "final" / "medical_dynamic_threshold_calibration_final.json",
    )
    communication_json = getenv_path(
        "TRANSSHIELD_SHOWCASE_COMMUNICATION_JSON",
        REPO_ROOT / "results" / "communication" / "mainline_communication_profile_final.json",
    )
    demo_summary_json = getenv_path(
        "TRANSSHIELD_SHOWCASE_DEMO_SUMMARY_JSON",
        REPO_ROOT / "results" / "final" / "demo_content_summary_final.json",
    )
    dist_dir = getenv_path("TRANSSHIELD_SHOWCASE_DIST_DIR", REPO_ROOT / "showcase" / "dist")
    audit_dir = getenv_path("TRANSSHIELD_SHOWCASE_AUDIT_DIR", REPO_ROOT / "artifacts" / "showcase_audit")
    run_dir = getenv_path("TRANSSHIELD_SHOWCASE_RUN_DIR", REPO_ROOT / "artifacts" / "showcase_runs")
    spu_config_path = getenv_path(
        "TRANSSHIELD_SHOWCASE_SPU_CONFIG",
        REPO_ROOT / "configs" / "transshield_runtime" / "2pc.json",
    )
    runtime_mode = os.environ.get("TRANSSHIELD_SHOWCASE_RUNTIME_MODE", "spu").strip().lower() or "spu"
    if runtime_mode not in {"spu", "mock"}:
        runtime_mode = "spu"
    eval_crop_pct = getenv_float("TRANSSHIELD_SHOWCASE_EVAL_CROP_PCT", DEFAULT_EVAL_CROP_PCT)
    if eval_crop_pct <= 0:
        eval_crop_pct = DEFAULT_EVAL_CROP_PCT
    spu_attention_policy = (
        os.environ.get("TRANSSHIELD_SHOWCASE_SPU_ATTENTION_POLICY", "uniform").strip()
        or "uniform"
    )

    threshold_payload = load_json(threshold_source_json)
    communication_payload = load_json(communication_json)

    return ShowcaseConfig(
        bundle_dir=bundle_dir,
        threshold_source_json=threshold_source_json,
        communication_json=communication_json,
        demo_summary_json=demo_summary_json,
        health_bind_dist=dist_dir,
        audit_dir=audit_dir,
        run_dir=run_dir,
        spu_config_path=spu_config_path,
        python_bin=os.environ.get("TRANSSHIELD_SHOWCASE_PYTHON_BIN", sys.executable).strip() or sys.executable,
        runtime_mode=runtime_mode,
        accepted_sleep_sec=getenv_float("TRANSSHIELD_SHOWCASE_ACCEPTED_SLEEP_SEC", 1.5),
        allowed_mime_types=("image/png", "image/jpeg"),
        max_file_size_bytes=getenv_int("TRANSSHIELD_SHOWCASE_MAX_FILE_BYTES", 10 * 1024 * 1024),
        max_request_bytes=getenv_int("TRANSSHIELD_SHOWCASE_MAX_REQUEST_BYTES", 5 * 1024 * 1024),
        max_image_dimension=getenv_int("TRANSSHIELD_SHOWCASE_MAX_IMAGE_DIMENSION", 8192),
        estimated_wait_seconds=getenv_int("TRANSSHIELD_SHOWCASE_ESTIMATED_WAIT_SECONDS", 90),
        input_size=getenv_int("TRANSSHIELD_SHOWCASE_INPUT_SIZE", 224),
        channels=getenv_int("TRANSSHIELD_SHOWCASE_CHANNELS", 3),
        class_names=("正常", "肺炎"),
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
        norm_clip_abs=getenv_float("TRANSSHIELD_SHOWCASE_NORM_CLIP_ABS", 0.0),
        eval_crop_pct=eval_crop_pct,
        share_abs_guard=getenv_float("TRANSSHIELD_SHOWCASE_SHARE_ABS_GUARD", 1e3),
        quality_drift_tolerance=getenv_float("TRANSSHIELD_SHOWCASE_QUALITY_DRIFT_TOLERANCE", 1e-4),
        per_ip_window_limit=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_LIMIT", 6),
        per_ip_window_seconds=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_SECONDS", 60),
        per_ip_inflight_limit=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_INFLIGHT_LIMIT", 1),
        global_inflight_limit=getenv_int("TRANSSHIELD_SHOWCASE_GLOBAL_INFLIGHT_LIMIT", 1),
        replay_nonce_ttl_seconds=getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_NONCE_TTL_SECONDS", 600),
        replay_payload_ttl_seconds=getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_PAYLOAD_TTL_SECONDS", 120),
        replay_nonce_capacity=max(1, getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_NONCE_CAPACITY", 4096)),
        replay_payload_capacity=max(1, getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_PAYLOAD_CAPACITY", 4096)),
        request_manifest_type="transshield_showcase_medical_live_request_v1",
        request_contract_version="medical_live_demo_v1",
        runner_profile=RunnerProfile(
            static_depth_limit=getenv_int("TRANSSHIELD_SHOWCASE_STATIC_DEPTH_LIMIT", 10),
            spu_batch_size=getenv_int("TRANSSHIELD_SHOWCASE_SPU_BATCH_SIZE", 1),
            spu_params_mode=os.environ.get(
                "TRANSSHIELD_SHOWCASE_SPU_PARAMS_MODE",
                "secret",
            ).strip()
            or "secret",
            spu_layer_norm_policy=os.environ.get(
                "TRANSSHIELD_SHOWCASE_SPU_LAYER_NORM_POLICY",
                "exact",
            ).strip()
            or "exact",
            spu_attention_policy=spu_attention_policy,
            spu_activation_override=os.environ.get(
                "TRANSSHIELD_SHOWCASE_SPU_ACTIVATION_OVERRIDE",
                "fixed_square",
            ).strip()
            or "fixed_square",
            spu_activation_clip_value=getenv_float("TRANSSHIELD_SHOWCASE_SPU_ACTIVATION_CLIP_VALUE", 0.0),
            spu_secure_pruning_mode=os.environ.get(
                "TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_MODE",
                "compact",
            ).strip()
            or "compact",
            spu_secure_pruning_network=os.environ.get(
                "TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_NETWORK",
                "unpadded_selection",
            ).strip()
            or "unpadded_selection",
            spu_final_block_cls_only=getenv_bool(
                "TRANSSHIELD_SHOWCASE_SPU_FINAL_BLOCK_CLS_ONLY",
                spu_attention_policy == "uniform",
            ),
            spu_uniform_attention_value_fusion=getenv_bool(
                "TRANSSHIELD_SHOWCASE_SPU_UNIFORM_ATTENTION_VALUE_FUSION",
                spu_attention_policy == "uniform",
            ),
            spu_compile_cache_dir=getenv_path(
                "TRANSSHIELD_SHOWCASE_SPU_COMPILE_CACHE_DIR",
                REPO_ROOT / "logs" / "showcase_runtime" / "spu_compile_cache",
            ),
        ),
        formal_threshold=float(threshold_payload["best_threshold"]),
        formal_threshold_accuracy=float(threshold_payload["best_threshold_accuracy"]),
        formal_auc=float(load_json(REPO_ROOT / "results" / "final" / "medical_dynamic_auc_reference_final.json")["auc"]),
        formal_sec_per_sample=float(communication_payload["medical"]["sec_per_sample"]),
        formal_dual_total_gib=float(communication_payload["medical"]["dual_total_gib"]),
    )
