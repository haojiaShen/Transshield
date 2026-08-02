from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

LEGACY_REPO_PATH_ALIASES = (
    (
        "/data/wyb/pneumoniamnist_imagefolder_subset/train",
        "data/pneumoniamnist_imagefolder_subset/train",
    ),
    (
        "/data/wyb/pneumoniamnist_imagefolder_subset/val",
        "data/pneumoniamnist_imagefolder_subset/val",
    ),
    (
        "/data/wyb/pneumoniamnist_imagefolder_subset/test",
        "data/pneumoniamnist_imagefolder_subset/test",
    ),
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_repo_path(path: Path | str | None) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""

    normalized = raw.replace("\\", "/")
    if re.search(r"(?:^|/)python(?:\d+)?\.exe$", normalized, flags=re.IGNORECASE) or re.search(
        r"(?:^|/)python(?:\d+(?:\.\d+)*)?$", normalized, flags=re.IGNORECASE
    ):
        return "python"

    for legacy_prefix, repo_relative_prefix in LEGACY_REPO_PATH_ALIASES:
        if normalized == legacy_prefix or normalized.startswith(legacy_prefix + "/"):
            suffix = normalized[len(legacy_prefix) :].lstrip("/")
            return "/".join(part for part in (repo_relative_prefix, suffix) if part)

    for marker in (
        "密捷_管理员控制台运行包",
        "密捷_客户演示界面运行包",
        "Transshield_final",
        "Transshield",
        "源代码·",
    ):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1].strip("/")
            return suffix or "."
    candidate = Path(raw)
    if not candidate.is_absolute():
        if re.match(r"^[A-Za-z]:/", normalized):
            return Path(normalized).name
        return normalized.removeprefix("./") or "."

    try:
        resolved = candidate.resolve()
    except Exception:
        return Path(normalized).name or "."

    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.name
    return str(relative) if str(relative) else "."


def resolve_repo_path(path: Path | str | None) -> Path:
    raw = str(path or "").strip()
    if not raw:
        return REPO_ROOT
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def build_path_replacements(default_train_data_path: Path, default_eval_data_path: Path) -> dict[str, str]:
    train_placeholder = relative_to_repo_path(default_train_data_path)
    eval_placeholder = relative_to_repo_path(default_eval_data_path)
    return {
        "/data/wyb/pneumoniamnist_imagefolder_subset/train/": train_placeholder + "/",
        "/data/wyb/pneumoniamnist_imagefolder_subset/val/": eval_placeholder + "/",
        "/data/wyb/pneumoniamnist_imagefolder_subset/train": train_placeholder,
        "/data/wyb/pneumoniamnist_imagefolder_subset/val": eval_placeholder,
    }


def relativize_text(value: str, replacements: dict[str, str] | None = None) -> str:
    result = value
    for old, new in (replacements or {}).items():
        result = result.replace(old, new)
    result = re.sub(
        r'(?:(?<=^)|(?<=\s))(?:"(?:[A-Za-z]:\\.*?python(?:\d+)?\.exe|/[^"]*?/python(?:\d+(?:\.\d+)*)?)"|(?:[A-Za-z]:\\.*?python(?:\d+)?\.exe|/[^"\'\s]*/python(?:\d+(?:\.\d+)*)?))',
        "python",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r'[A-Za-z]:\\[^"\r\n]*?site-packages\\', r'site-packages\\', result, flags=re.IGNORECASE)
    result = re.sub(r'/[^"\r\n]*?/site-packages/', 'site-packages/', result, flags=re.IGNORECASE)
    result = re.sub(r'[^"\'\s]*Transshield(?:_final)?(?=(?:"|\'|\s|$))', ".", result)
    result = re.sub(r'[^"\'\s]*Transshield(?:_final)?[\\/]', "", result)
    if not result and re.search(r"Transshield(?:_final)?", value):
        return "."
    return result


def relativize_value(value, replacements: dict[str, str] | None = None):
    if isinstance(value, dict):
        return {key: relativize_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_value(item, replacements) for item in value]
    if isinstance(value, str):
        return relativize_text(value, replacements)
    return value


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


def getenv_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class RunnerProfile:
    static_depth_limit: int
    spu_batch_size: int
    spu_params_mode: str
    spu_layer_norm_policy: str
    spu_attention_policy: str
    spu_activation_override: str
    spu_activation_clip_value: float


@dataclass(frozen=True)
class TrainingPreset:
    id: str
    name: str
    description: str
    parameters: dict[str, object]


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
    share_abs_guard: float
    quality_drift_tolerance: float
    per_ip_window_limit: int
    per_ip_window_seconds: int
    per_ip_inflight_limit: int
    global_inflight_limit: int
    replay_nonce_ttl_seconds: int
    replay_payload_ttl_seconds: int
    request_manifest_type: str
    request_contract_version: str
    runner_profile: RunnerProfile
    formal_threshold: float
    formal_threshold_accuracy: float
    formal_auc: float
    formal_sec_per_sample: float
    formal_dual_total_gib: float
    admin_username: str
    admin_display_name: str
    admin_initial_password: str
    admin_auth_file: Path
    admin_session_cookie_name: str
    admin_session_ttl_seconds: int
    admin_job_root: Path
    train_output_root: Path
    bundle_output_root: Path
    default_train_data_path: Path
    default_eval_data_path: Path
    default_device: str
    default_batch_size: int
    default_num_workers: int
    max_concurrent_train_jobs: int
    cors_allowed_origins: tuple[str, ...]
    cors_allowed_origin_regex: str
    training_presets: tuple[TrainingPreset, ...]

    @property
    def expected_shape(self) -> list[int]:
        return [1, self.channels, self.input_size, self.input_size]

    @property
    def float_count(self) -> int:
        return self.channels * self.input_size * self.input_size

    @property
    def share_byte_count(self) -> int:
        return self.float_count * 4


def _build_training_preset(
    *,
    preset_id: str,
    name: str,
    description: str,
    snapshot_path: Path,
    default_train_data_path: Path,
    default_eval_data_path: Path,
    overrides: dict[str, object] | None = None,
) -> TrainingPreset:
    payload = load_json(snapshot_path) if snapshot_path.exists() else {}
    path_replacements = build_path_replacements(default_train_data_path, default_eval_data_path)
    parameters: dict[str, object] = {
        "model": payload.get("model", "deit-s"),
        "data_set": payload.get("data_set", "image_folder"),
        "train_data_path": relativize_text(str(payload.get("data_path", default_train_data_path)), path_replacements),
        "eval_data_path": relativize_text(str(payload.get("eval_data_path", default_eval_data_path)), path_replacements),
        "nb_classes": int(payload.get("nb_classes", 2)),
        "input_size": int(payload.get("input_size", 224)),
        "batch_size": int(payload.get("batch_size", 32)),
        "epochs": int(payload.get("epochs", 8)),
        "num_workers": int(payload.get("num_workers", 4)),
        "device": payload.get("device", "cuda"),
        "base_rate": float(payload.get("base_rate", 0.7)),
        "ratio_weight": float(payload.get("ratio_weight", 2.0)),
        "lr": float(payload.get("lr", 1e-5)),
        "warmup_epochs": int(payload.get("warmup_epochs", 0)),
        "warmup_steps": int(payload.get("warmup_steps", 20)),
        "clip_grad": float(payload.get("clip_grad", 1.0)),
        "cls_distill_weight": float(payload.get("cls_distill_weight", 0.0) or 0.0),
        "token_distill_weight": float(payload.get("token_distill_weight", 0.0) or 0.0),
        "square_activation_mode": payload.get("square_activation_mode", "learnable_quadratic_gelu_init"),
        "approx_attn_mode": payload.get("approx_attn_mode", "relu"),
        "use_square_gelu": bool(payload.get("use_square_gelu", True)),
        "use_approx_attn": bool(payload.get("use_approx_attn", True)),
        "use_mask_pruning": bool(payload.get("use_mask_pruning", True)),
        "inference_friendly_ops": bool(payload.get("inference_friendly_ops", False)),
        "eval_pruning_mode": payload.get("eval_pruning_mode", "compare_network_tie"),
        "eval_tie_policy": payload.get("eval_tie_policy", "lowest_index"),
        "patch_embed_bias_init_mode": payload.get("patch_embed_bias_init_mode", "zero"),
        "freeze_patch_embed_proj": bool(payload.get("freeze_patch_embed_proj", True)),
        "pretrained_fix_step": int(payload.get("pretrained_fix_step", 0)),
        "secure_static_train_depth": int(payload.get("secure_static_train_depth", 0) or 0),
        "secure_static_skip_pruning": bool(payload.get("secure_static_skip_pruning", True)),
        "teacher_checkpoint_path": relativize_text(str(payload.get("teacher_checkpoint_path") or ""), path_replacements),
        "finetune": relativize_text(str(payload.get("finetune") or ""), path_replacements),
        "model_ema": bool(payload.get("model_ema", False)),
        "save_ckpt": bool(payload.get("save_ckpt", True)),
        "save_ckpt_freq": int(payload.get("save_ckpt_freq", 1)),
        "save_ckpt_num": int(payload.get("save_ckpt_num", 2)),
        "auto_resume": bool(payload.get("auto_resume", False)),
        "use_amp": bool(payload.get("use_amp", False)),
        "mixup": float(payload.get("mixup", 0.0)),
        "cutmix": float(payload.get("cutmix", 0.0)),
        "seed": int(payload.get("seed", 0)),
        "lr_scale": float(payload.get("lr_scale", 1.0)),
        "groupa_lr_scale": float(payload.get("groupa_lr_scale", 0.1)),
        "activation_lr_scale": float(payload.get("activation_lr_scale", 1.0)),
        "export_bundle": True,
    }
    if overrides:
        parameters.update(overrides)
    return TrainingPreset(
        id=preset_id,
        name=name,
        description=description,
        parameters=parameters,
    )


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
    default_train_data_path = getenv_path(
        "TRANSSHIELD_SHOWCASE_DEFAULT_TRAIN_DATA_PATH",
        REPO_ROOT / "data" / "pneumoniamnist_imagefolder_subset" / "train",
    )
    default_eval_data_path = getenv_path(
        "TRANSSHIELD_SHOWCASE_DEFAULT_EVAL_DATA_PATH",
        REPO_ROOT / "data" / "pneumoniamnist_imagefolder_subset" / "val",
    )
    admin_auth_file = getenv_path(
        "TRANSSHIELD_ADMIN_AUTH_FILE",
        REPO_ROOT / "artifacts" / "showcase_admin" / "admin_auth.json",
    )
    runtime_mode = os.environ.get("TRANSSHIELD_SHOWCASE_RUNTIME_MODE", "spu").strip().lower() or "spu"
    if runtime_mode not in {"spu", "mock"}:
        runtime_mode = "spu"

    threshold_payload = load_json(threshold_source_json)
    communication_payload = load_json(communication_json)

    training_presets = (
        _build_training_preset(
            preset_id="medical_dynamic_mainline",
            name="医疗动态主线",
            description="动态剪枝、蒸馏补偿与冻结导出主线。",
            snapshot_path=REPO_ROOT / "artifacts" / "frozen_bundle_medical_dynamic_mainline" / "args_snapshot.json",
            default_train_data_path=default_train_data_path,
            default_eval_data_path=default_eval_data_path,
            overrides={
                "run_name": "medical_dynamic_mainline",
                "bundle_name": "frozen_bundle_medical_dynamic_mainline_admin",
                "inference_friendly_ops": True,
                "use_mask_pruning": True,
                "eval_pruning_mode": "compare_network_tie",
                "patch_embed_bias_init_mode": "zero",
                "freeze_patch_embed_proj": True,
            },
        ),
        _build_training_preset(
            preset_id="secure_static_depth12_control",
            name="静态部署对齐控制线",
            description="secure_static_train_depth=12 的部署对齐控制训练。",
            snapshot_path=REPO_ROOT / "artifacts" / "frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430" / "args_snapshot.json",
            default_train_data_path=default_train_data_path,
            default_eval_data_path=default_eval_data_path,
            overrides={
                "run_name": "secure_static_depth12_control",
                "bundle_name": "frozen_bundle_secure_static_depth12_control_admin",
                "use_mask_pruning": False,
                "inference_friendly_ops": False,
                "secure_static_train_depth": 12,
                "secure_static_skip_pruning": True,
            },
        ),
        _build_training_preset(
            preset_id="secure_static_depth12_aanone",
            name="静态 uniform/fixed-square 复现实验",
            description="沿用 secure static 轻量复现实验口径。",
            snapshot_path=REPO_ROOT / "artifacts" / "frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507" / "args_snapshot.json",
            default_train_data_path=default_train_data_path,
            default_eval_data_path=default_eval_data_path,
            overrides={
                "run_name": "secure_static_depth12_aanone",
                "bundle_name": "frozen_bundle_secure_static_depth12_aanone_admin",
                "inference_friendly_ops": False,
            },
        ),
    )

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
        norm_clip_abs=getenv_float("TRANSSHIELD_SHOWCASE_NORM_CLIP_ABS", 2.0),
        share_abs_guard=getenv_float("TRANSSHIELD_SHOWCASE_SHARE_ABS_GUARD", 1e3),
        quality_drift_tolerance=getenv_float("TRANSSHIELD_SHOWCASE_QUALITY_DRIFT_TOLERANCE", 1e-4),
        per_ip_window_limit=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_LIMIT", 6),
        per_ip_window_seconds=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_SECONDS", 60),
        per_ip_inflight_limit=getenv_int("TRANSSHIELD_SHOWCASE_PER_IP_INFLIGHT_LIMIT", 1),
        global_inflight_limit=getenv_int("TRANSSHIELD_SHOWCASE_GLOBAL_INFLIGHT_LIMIT", 1),
        replay_nonce_ttl_seconds=getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_NONCE_TTL_SECONDS", 600),
        replay_payload_ttl_seconds=getenv_int("TRANSSHIELD_SHOWCASE_REPLAY_PAYLOAD_TTL_SECONDS", 120),
        request_manifest_type="transshield_showcase_medical_live_request_v1",
        request_contract_version="medical_live_demo_v1",
        runner_profile=RunnerProfile(
            static_depth_limit=getenv_int("TRANSSHIELD_SHOWCASE_STATIC_DEPTH_LIMIT", 10),
            spu_batch_size=getenv_int("TRANSSHIELD_SHOWCASE_SPU_BATCH_SIZE", 1),
            spu_params_mode=os.environ.get("TRANSSHIELD_SHOWCASE_SPU_PARAMS_MODE", "secret").strip() or "secret",
            spu_layer_norm_policy=os.environ.get("TRANSSHIELD_SHOWCASE_SPU_LAYER_NORM_POLICY", "exact").strip() or "exact",
            spu_attention_policy=os.environ.get("TRANSSHIELD_SHOWCASE_SPU_ATTENTION_POLICY", "uniform").strip() or "uniform",
            spu_activation_override=os.environ.get("TRANSSHIELD_SHOWCASE_SPU_ACTIVATION_OVERRIDE", "fixed_square").strip() or "fixed_square",
            spu_activation_clip_value=getenv_float("TRANSSHIELD_SHOWCASE_SPU_ACTIVATION_CLIP_VALUE", 0.0),
        ),
        formal_threshold=float(threshold_payload["best_threshold"]),
        formal_threshold_accuracy=float(threshold_payload["best_threshold_accuracy"]),
        formal_auc=float(load_json(REPO_ROOT / "results" / "final" / "medical_dynamic_auc_reference_final.json")["auc"]),
        formal_sec_per_sample=float(communication_payload["medical"]["sec_per_sample"]),
        formal_dual_total_gib=float(communication_payload["medical"]["dual_total_gib"]),
        admin_username=os.environ.get("TRANSSHIELD_ADMIN_USERNAME", "admin").strip() or "admin",
        admin_display_name=os.environ.get("TRANSSHIELD_ADMIN_DISPLAY_NAME", "系统管理员").strip() or "系统管理员",
        admin_initial_password=os.environ.get("TRANSSHIELD_ADMIN_PASSWORD", "admin123").strip() or "admin123",
        admin_auth_file=admin_auth_file,
        admin_session_cookie_name=os.environ.get("TRANSSHIELD_ADMIN_SESSION_COOKIE_NAME", "mijie_admin_session").strip() or "mijie_admin_session",
        admin_session_ttl_seconds=getenv_int("TRANSSHIELD_ADMIN_SESSION_TTL_SECONDS", 12 * 60 * 60),
        admin_job_root=getenv_path("TRANSSHIELD_SHOWCASE_ADMIN_JOB_ROOT", REPO_ROOT / "artifacts" / "showcase_train_jobs"),
        train_output_root=getenv_path("TRANSSHIELD_SHOWCASE_TRAIN_OUTPUT_ROOT", REPO_ROOT / "artifacts" / "train_runs"),
        bundle_output_root=getenv_path("TRANSSHIELD_SHOWCASE_BUNDLE_OUTPUT_ROOT", REPO_ROOT / "artifacts" / "showcase_bundles"),
        default_train_data_path=default_train_data_path,
        default_eval_data_path=default_eval_data_path,
        default_device=os.environ.get("TRANSSHIELD_SHOWCASE_DEFAULT_DEVICE", "cuda").strip() or "cuda",
        default_batch_size=getenv_int("TRANSSHIELD_SHOWCASE_DEFAULT_BATCH_SIZE", 32),
        default_num_workers=getenv_int("TRANSSHIELD_SHOWCASE_DEFAULT_NUM_WORKERS", 4),
        max_concurrent_train_jobs=getenv_int("TRANSSHIELD_SHOWCASE_MAX_CONCURRENT_JOBS", 1),
        cors_allowed_origins=getenv_csv(
            "TRANSSHIELD_CORS_ALLOWED_ORIGINS",
            (
                "http://127.0.0.1:5173",
                "http://localhost:5173",
                "http://127.0.0.1:4173",
                "http://localhost:4173",
                "http://127.0.0.1:4174",
                "http://localhost:4174",
                "http://127.0.0.1:7860",
                "http://localhost:7860",
                "http://127.0.0.1:7863",
                "http://localhost:7863",
            ),
        ),
        cors_allowed_origin_regex=os.environ.get(
            "TRANSSHIELD_CORS_ALLOWED_ORIGIN_REGEX",
            r"^https?://(?:localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?::\d+)?$",
        ).strip(),
        training_presets=training_presets,
    )
