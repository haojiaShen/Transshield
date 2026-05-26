from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from showcase_api.config import REPO_ROOT


TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ROLE_TO_PARTY = {"hospital": "P1", "ai": "P2"}
PARTY_TO_RANK = {"P1": 0, "P2": 1}
PUBLIC_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_public_manifest_v0"
PARTY_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_party_manifest_v0"
SHARE_SEMANTICS = "debug_float_additive_share_not_production_mpc_share"
RUNNERS: dict[str, subprocess.Popen] = {}
RUNNERS_LOCK = threading.Lock()


@dataclass(frozen=True)
class SplitGatewayConfig:
    role: str
    storage_dir: Path
    auth_token: str
    python_bin: str
    runner_script: Path
    default_bundle_dir: Path
    default_spu_config: Path
    default_runtime_mode: str
    run_timeout_sec: float
    max_share_bytes: int
    max_image_bytes: int
    max_image_dimension: int
    input_size: int
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    norm_clip_abs: float
    ai_gateway_url: str
    ai_gateway_auth_token: str
    gateway_forward_timeout_sec: float

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_token)


def getenv_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def getenv_float_tuple(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError:
        return default
    if len(values) != 3:
        return default
    return values


def load_split_gateway_config() -> SplitGatewayConfig:
    role = os.environ.get("TRANSSHIELD_SPLIT_ROLE", "coordinator").strip().lower() or "coordinator"
    if role not in {"hospital", "ai", "coordinator"}:
        role = "coordinator"
    runtime_mode = os.environ.get("TRANSSHIELD_SPLIT_RUNTIME_MODE", "mock").strip().lower() or "mock"
    if runtime_mode not in {"mock", "spu"}:
        runtime_mode = "mock"
    return SplitGatewayConfig(
        role=role,
        storage_dir=getenv_path("TRANSSHIELD_SPLIT_STORAGE_DIR", REPO_ROOT / "artifacts" / "split_gateway" / role),
        auth_token=os.environ.get("TRANSSHIELD_SPLIT_AUTH_TOKEN", "").strip(),
        python_bin=os.environ.get("TRANSSHIELD_SPLIT_PYTHON_BIN", os.environ.get("TRANSSHIELD_SHOWCASE_PYTHON_BIN", "python")).strip()
        or "python",
        runner_script=getenv_path(
            "TRANSSHIELD_SPLIT_RUNNER_SCRIPT",
            REPO_ROOT / "integrations" / "transshield_runtime" / "e2e_secure_vit" / "transshield_e2e_secure_vit.py",
        ),
        default_bundle_dir=getenv_path(
            "TRANSSHIELD_SPLIT_BUNDLE_DIR",
            REPO_ROOT / "artifacts" / "frozen_bundle_medical_dynamic_mainline",
        ),
        default_spu_config=getenv_path(
            "TRANSSHIELD_SPLIT_SPU_CONFIG",
            REPO_ROOT / "configs" / "transshield_runtime" / "2pc.remote.json",
        ),
        default_runtime_mode=runtime_mode,
        run_timeout_sec=getenv_float("TRANSSHIELD_SPLIT_RUN_TIMEOUT_SEC", 600.0),
        max_share_bytes=getenv_int("TRANSSHIELD_SPLIT_MAX_SHARE_BYTES", 20 * 1024 * 1024),
        max_image_bytes=getenv_int("TRANSSHIELD_SPLIT_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
        max_image_dimension=getenv_int("TRANSSHIELD_SPLIT_MAX_IMAGE_DIMENSION", 8192),
        input_size=getenv_int("TRANSSHIELD_SPLIT_INPUT_SIZE", 224),
        norm_mean=getenv_float_tuple("TRANSSHIELD_SPLIT_NORM_MEAN", (0.485, 0.456, 0.406)),
        norm_std=getenv_float_tuple("TRANSSHIELD_SPLIT_NORM_STD", (0.229, 0.224, 0.225)),
        norm_clip_abs=getenv_float("TRANSSHIELD_SPLIT_NORM_CLIP_ABS", 2.0),
        ai_gateway_url=os.environ.get("TRANSSHIELD_SPLIT_AI_GATEWAY_URL", "").strip().rstrip("/"),
        ai_gateway_auth_token=os.environ.get("TRANSSHIELD_SPLIT_AI_GATEWAY_AUTH_TOKEN", "").strip(),
        gateway_forward_timeout_sec=getenv_float("TRANSSHIELD_SPLIT_GATEWAY_FORWARD_TIMEOUT_SEC", 5.0),
    )


CONFIG = load_split_gateway_config()
app = FastAPI(title="TransShield Split Gateway", version="2026.05")


def now_ts() -> float:
    return round(time.time(), 6)


def response(status_code: int, payload: dict) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


def require_auth(authorization: Optional[str]):
    if not CONFIG.auth_enabled:
        return
    expected = f"Bearer {CONFIG.auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def require_role(*roles: str):
    if CONFIG.role not in roles:
        allowed = ", ".join(roles)
        raise HTTPException(status_code=403, detail=f"endpoint requires role: {allowed}")


def validate_task_id(task_id: str) -> str:
    if not TASK_ID_RE.fullmatch(task_id):
        raise HTTPException(status_code=400, detail="task_id must be 1-128 chars of letters, digits, dot, dash, or underscore")
    return task_id


def task_dir(task_id: str) -> Path:
    return CONFIG.storage_dir / "tasks" / validate_task_id(task_id)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_state(task_id: str) -> dict:
    state_path = task_dir(task_id) / "task_state.json"
    if not state_path.exists():
        return {"task_id": task_id, "role": CONFIG.role, "status": "missing"}
    return read_json(state_path)


def write_state(task_id: str, updates: dict) -> dict:
    current = read_state(task_id)
    current.update(updates)
    current["task_id"] = task_id
    current["role"] = CONFIG.role
    current["updated_at"] = now_ts()
    write_json(task_dir(task_id) / "task_state.json", current)
    return current


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pack_float32_le(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def decode_share_b64(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail="share_b64 must be a non-empty base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as error:
        raise HTTPException(status_code=400, detail=f"invalid share_b64: {error}") from error


def expected_share_bytes(public_manifest: dict) -> Optional[int]:
    shape = public_manifest.get("share_shape") or public_manifest.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    count = 1
    for value in shape:
        if not isinstance(value, int) or value <= 0 or value > 100000:
            return None
        count *= value
        if count > 100000000:
            return None
    return count * 4


def validate_public_manifest(public_manifest: dict, task_id: Optional[str] = None):
    if public_manifest.get("manifest_type") != PUBLIC_MANIFEST_TYPE:
        raise HTTPException(status_code=400, detail=f"public_manifest.manifest_type must be {PUBLIC_MANIFEST_TYPE}")
    if task_id is not None and public_manifest.get("task_id") != task_id:
        raise HTTPException(status_code=422, detail="public_manifest.task_id must match request task_id")
    if expected_share_bytes(public_manifest) is None:
        raise HTTPException(status_code=400, detail="public_manifest.share_shape is missing or invalid")


def check_share_hash(public_manifest: dict, party_id: str, share_digest: str, payload_digest: Optional[str]):
    declared = payload_digest
    if not declared:
        declared = public_manifest.get("share0_sha256" if party_id == "P1" else "share1_sha256")
    if declared and str(declared).lower() != share_digest:
        raise HTTPException(status_code=422, detail=f"{party_id} share sha256 mismatch")


def build_party_manifest(task_id: str, public_manifest_path: Path, share_path: Path, party_id: str, share_digest: str) -> dict:
    public_manifest = read_json(public_manifest_path)
    return {
        "manifest_type": PARTY_MANIFEST_TYPE,
        "task_id": task_id,
        "party_id": party_id,
        "share_rank": PARTY_TO_RANK[party_id],
        "share_count": 2,
        "share_path": str(share_path.resolve()),
        "share_storage_format": "raw_float32_le",
        "share_sha256": share_digest,
        "public_manifest_json": str(public_manifest_path.resolve()),
        "share_semantics": public_manifest.get("share_semantics", SHARE_SEMANTICS),
        "share_dtype": public_manifest.get("share_dtype", "torch.float32"),
        "share_shape": public_manifest.get("share_shape"),
        "sample_count": public_manifest.get("sample_count"),
        "sample_ids": public_manifest.get("sample_ids"),
        "privacy_status": "stored only by the owning split gateway role",
    }


def compute_quality_summary(rgb: np.ndarray) -> dict:
    luma = 0.299 * rgb[0, 0] + 0.587 * rgb[0, 1] + 0.114 * rgb[0, 2]
    p05, p95 = np.percentile(luma, [5, 95])
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return {
        "mean_luma": round(float(np.mean(luma)), 8),
        "std_luma": round(float(np.std(luma)), 8),
        "overexposed_ratio": round(float(np.mean(luma >= 0.95)), 8),
        "underexposed_ratio": round(float(np.mean(luma <= 0.05)), 8),
        "effective_luma_ratio": round(float(np.mean((luma >= 0.02) & (luma <= 0.98))), 8),
        "dynamic_range_p95_p05": round(float(p95 - p05), 8),
        "laplacian_variance": round(float(np.var(lap)), 8),
    }


def decode_and_normalize_image(image_bytes: bytes, content_type: str) -> tuple[np.ndarray, dict]:
    if len(image_bytes) > CONFIG.max_image_bytes:
        raise HTTPException(status_code=413, detail="image payload exceeds TRANSSHIELD_SPLIT_MAX_IMAGE_BYTES")
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="only image/png and image/jpeg are supported")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width <= 0 or height <= 0 or width > CONFIG.max_image_dimension or height > CONFIG.max_image_dimension:
                raise HTTPException(status_code=422, detail="image dimensions are invalid or too large")
            image = image.convert("RGB")
            crop_size = min(width, height)
            left = max(0, (width - crop_size) // 2)
            top = max(0, (height - crop_size) // 2)
            resample = getattr(Image, "Resampling", Image).BILINEAR
            image = image.crop((left, top, left + crop_size, top + crop_size)).resize(
                (CONFIG.input_size, CONFIG.input_size),
                resample,
            )
            rgb_hwc = np.asarray(image, dtype=np.float32) / 255.0
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"failed to decode image: {error}") from error

    rgb = np.transpose(rgb_hwc, (2, 0, 1))[None, ...].astype(np.float32)
    mean = np.asarray(CONFIG.norm_mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CONFIG.norm_std, dtype=np.float32).reshape(1, 3, 1, 1)
    normalized = np.clip((rgb - mean) / std, -CONFIG.norm_clip_abs, CONFIG.norm_clip_abs).astype(np.float32)
    return normalized, {
        "source_dimensions": {"width": int(width), "height": int(height)},
        "source_mime": mime,
        "quality_summary": compute_quality_summary(rgb),
    }


def split_normalized_tensor(task_id: str, normalized: np.ndarray, image_bytes: bytes, image_meta: dict, source_filename: str) -> dict:
    seed = int.from_bytes(os.urandom(16), byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    share0 = rng.uniform(-0.8, 0.8, size=normalized.shape).astype(np.float32)
    share1 = (normalized - share0).astype(np.float32)
    normalized_bytes = pack_float32_le(normalized)
    share0_bytes = pack_float32_le(share0)
    share1_bytes = pack_float32_le(share1)
    audit_nonce = os.urandom(16).hex()
    sample_id = f"{task_id}_sample_000000"
    public_manifest = {
        "manifest_type": PUBLIC_MANIFEST_TYPE,
        "task_id": task_id,
        "share_count": 2,
        "party_ids": ["P1", "P2"],
        "share_semantics": SHARE_SEMANTICS,
        "share_dtype": "torch.float32",
        "share_shape": [int(value) for value in normalized.shape],
        "sample_count": 1,
        "sample_ids": [sample_id],
        "targets_included": False,
        "source_paths_included": False,
        "private_share_paths_included": False,
        "source_file_name": source_filename,
        "source_mime": image_meta["source_mime"],
        "source_size_bytes": len(image_bytes),
        "source_dimensions": image_meta["source_dimensions"],
        "input_size": CONFIG.input_size,
        "dtype": "float32_le",
        "norm_mean": list(CONFIG.norm_mean),
        "norm_std": list(CONFIG.norm_std),
        "norm_clip_abs": CONFIG.norm_clip_abs,
        "audit_nonce": audit_nonce,
        "source_image_sha256": sha256_hex(image_bytes),
        "normalized_tensor_sha256": sha256_hex(normalized_bytes),
        "share0_sha256": sha256_hex(share0_bytes),
        "share1_sha256": sha256_hex(share1_bytes),
        "privacy_status": "hospital gateway generated split shares; original image stays on the hospital side",
    }
    return {
        "public_manifest": public_manifest,
        "share0_bytes": share0_bytes,
        "share1_bytes": share1_bytes,
        "quality_assurance": {
            "status": "pass",
            "client_quality_summary": image_meta["quality_summary"],
        },
    }


def store_party_share(task_id: str, public_manifest: dict, party_id: str, share_bytes: bytes) -> tuple[dict, dict]:
    validate_public_manifest(public_manifest, task_id)
    root = task_dir(task_id)
    root.mkdir(parents=True, exist_ok=True)
    public_manifest_path = root / "public_manifest.json"
    share_path = root / f"{party_id.lower()}_share.float32le"
    party_manifest_path = root / f"{party_id.lower()}_share_manifest.json"
    write_json(public_manifest_path, public_manifest)
    share_path.write_bytes(share_bytes)
    share_digest = sha256_hex(share_bytes)
    party_manifest = build_party_manifest(task_id, public_manifest_path, share_path, party_id, share_digest)
    write_json(party_manifest_path, party_manifest)
    state = write_state(
        task_id,
        {
            "status": "share_received",
            "party_id": party_id,
            "share_sha256": share_digest,
            "artifacts": {
                "public_manifest": str(public_manifest_path),
                "party_manifest": str(party_manifest_path),
                "share_path": str(share_path),
            },
        },
    )
    return state, party_manifest


async def request_json(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {error}") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return payload


def post_json(url: str, payload: dict, auth_token: str, timeout_sec: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if auth_token:
        request.add_header("Authorization", f"Bearer {auth_token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response_handle:
            response_body = response_handle.read()
            status_code = int(response_handle.status)
    except urllib.error.HTTPError as error:
        response_body = error.read()
        status_code = int(error.code)
    except Exception as error:
        return {"enabled": True, "status": "failed", "error": f"{type(error).__name__}: {error}"}

    parsed_body: Any
    try:
        parsed_body = json.loads(response_body.decode("utf-8")) if response_body else None
    except Exception:
        parsed_body = response_body.decode("utf-8", errors="replace")[:1000]
    return {
        "enabled": True,
        "status": "accepted" if 200 <= status_code < 300 else "failed",
        "http_status": status_code,
        "response": parsed_body,
    }


def forward_p2_share_delivery(task_id: str, p2_delivery: dict) -> dict:
    if not CONFIG.ai_gateway_url:
        return {"enabled": False, "status": "not_configured"}
    parsed = urllib.parse.urlparse(CONFIG.ai_gateway_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"enabled": True, "status": "failed", "error": "TRANSSHIELD_SPLIT_AI_GATEWAY_URL must be http(s)://host[:port]"}
    ai_url = f"{CONFIG.ai_gateway_url}/api/split/tasks/{validate_task_id(task_id)}/share"
    return post_json(ai_url, p2_delivery, CONFIG.ai_gateway_auth_token, CONFIG.gateway_forward_timeout_sec)


@app.get("/api/split/health")
async def get_health():
    CONFIG.storage_dir.mkdir(parents=True, exist_ok=True)
    task_count = 0
    task_root = CONFIG.storage_dir / "tasks"
    if task_root.exists():
        task_count = sum(1 for item in task_root.iterdir() if item.is_dir())
    return {
        "status": "ok",
        "role": CONFIG.role,
        "storage_dir": str(CONFIG.storage_dir),
        "auth_enabled": CONFIG.auth_enabled,
        "default_runtime_mode": CONFIG.default_runtime_mode,
        "runner_present": CONFIG.runner_script.exists(),
        "input_size": CONFIG.input_size,
        "max_image_bytes": CONFIG.max_image_bytes,
        "max_share_bytes": CONFIG.max_share_bytes,
        "ai_gateway_forwarding_enabled": bool(CONFIG.ai_gateway_url),
        "task_count": task_count,
    }


@app.post("/api/hospital/tasks/{task_id}/image")
async def ingest_hospital_image(
    task_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    content_type: Optional[str] = Header(default=None),
    x_source_filename: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    require_role("hospital")
    task_id = validate_task_id(task_id)
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image body is empty")
    normalized, image_meta = decode_and_normalize_image(image_bytes, content_type or "")
    source_filename = x_source_filename or f"{validate_task_id(task_id)}.image"
    split_payload = split_normalized_tensor(task_id, normalized, image_bytes, image_meta, source_filename)
    state, p1_manifest = store_party_share(
        task_id,
        split_payload["public_manifest"],
        "P1",
        split_payload["share0_bytes"],
    )
    p2_delivery = {
        "task_id": task_id,
        "public_manifest": split_payload["public_manifest"],
        "share_b64": base64.b64encode(split_payload["share1_bytes"]).decode("ascii"),
        "share_sha256": split_payload["public_manifest"]["share1_sha256"],
    }
    p2_forward_delivery = forward_p2_share_delivery(task_id, p2_delivery)
    return response(
        202,
        {
            "status": "accepted",
            "task": state,
            "quality_assurance": split_payload["quality_assurance"],
            "p1_party_manifest": p1_manifest,
            "p2_share_delivery": p2_delivery,
            "p2_forward_delivery": p2_forward_delivery,
            "next_step": "POST p2_share_delivery to the AI gateway /api/split/tasks/{task_id}/share endpoint.",
        },
    )


@app.post("/api/split/tasks/{task_id}/share")
async def ingest_party_share(task_id: str, request: Request, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    require_role("hospital", "ai")
    task_id = validate_task_id(task_id)
    party_id = ROLE_TO_PARTY[CONFIG.role]
    payload = await request_json(request)
    payload_task_id = payload.get("task_id")
    if payload_task_id is not None and payload_task_id != task_id:
        raise HTTPException(status_code=422, detail="payload.task_id must match request task_id")
    public_manifest = payload.get("public_manifest")
    if not isinstance(public_manifest, dict):
        raise HTTPException(status_code=400, detail="public_manifest must be an object")
    share_bytes = decode_share_b64(payload.get("share_b64"))
    if len(share_bytes) > CONFIG.max_share_bytes:
        raise HTTPException(status_code=413, detail="share payload exceeds TRANSSHIELD_SPLIT_MAX_SHARE_BYTES")
    expected_bytes = expected_share_bytes(public_manifest)
    if expected_bytes is not None and len(share_bytes) != expected_bytes:
        raise HTTPException(status_code=422, detail=f"{party_id} share byte length mismatch")
    share_digest = sha256_hex(share_bytes)
    check_share_hash(public_manifest, party_id, share_digest, payload.get("share_sha256"))
    state, party_manifest = store_party_share(task_id, public_manifest, party_id, share_bytes)
    return response(
        202,
        {
            "status": "accepted",
            "task": state,
            "party_manifest": party_manifest,
        },
    )


@app.get("/api/split/tasks/{task_id}")
async def get_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    task_id = validate_task_id(task_id)
    state = read_state(task_id)
    status_code = 404 if state["status"] == "missing" else 200
    return response(status_code, state)


@app.get("/api/split/tasks/{task_id}/public-manifest")
async def get_public_manifest(task_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    task_id = validate_task_id(task_id)
    path = task_dir(task_id) / "public_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="public manifest not found")
    return read_json(path)


@app.get("/api/split/tasks/{task_id}/party-manifest")
async def get_party_manifest(task_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    require_role("hospital", "ai")
    task_id = validate_task_id(task_id)
    party_id = ROLE_TO_PARTY[CONFIG.role]
    path = task_dir(task_id) / f"{party_id.lower()}_share_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="party manifest not found")
    return read_json(path)


@app.post("/api/ai/tasks/{task_id}/model-manifest")
async def put_model_manifest(task_id: str, request: Request, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    require_role("ai")
    task_id = validate_task_id(task_id)
    payload = await request_json(request)
    bundle_dir = str(payload.get("bundle_dir") or CONFIG.default_bundle_dir)
    model_manifest = {
        "manifest_type": "transshield_split_gateway_model_manifest_v0",
        "task_id": task_id,
        "owner_role": "ai",
        "bundle_dir": bundle_dir,
        "model_version": str(payload.get("model_version") or "site-defined"),
        "spu_params_mode": str(payload.get("spu_params_mode") or "secret"),
        "created_at": now_ts(),
        "privacy_status": "model path is stored by the AI/model-provider gateway",
    }
    path = task_dir(task_id) / "model_manifest.json"
    write_json(path, model_manifest)
    state = write_state(
        task_id,
        {
            "status": "model_manifest_received",
            "model_manifest": str(path),
        },
    )
    return response(202, {"status": "accepted", "task": state, "model_manifest": model_manifest})


def manifest_path_from_payload(task_id: str, payload: dict, path_key: str, object_key: str, filename: str) -> Path:
    raw_path = payload.get(path_key)
    if raw_path:
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"{path_key} does not exist: {path}")
        validate_manifest_task_id(read_json(path), task_id, path_key)
        return path
    manifest = payload.get(object_key)
    if isinstance(manifest, dict):
        validate_manifest_task_id(manifest, task_id, object_key)
        path = task_dir(task_id) / filename
        write_json(path, manifest)
        return path
    raise HTTPException(status_code=400, detail=f"missing {path_key} or {object_key}")


def validate_manifest_task_id(manifest: dict, task_id: str, label: str):
    manifest_task_id = manifest.get("task_id")
    if manifest_task_id is not None and manifest_task_id != task_id:
        raise HTTPException(status_code=422, detail=f"{label}.task_id must match request task_id")


def load_optional_model_manifest(task_id: str, payload: dict) -> dict:
    raw_path = payload.get("model_manifest_json")
    if raw_path:
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"model_manifest_json does not exist: {path}")
        manifest = read_json(path)
        validate_manifest_task_id(manifest, task_id, "model_manifest_json")
        return manifest
    manifest = payload.get("model_manifest")
    if isinstance(manifest, dict):
        validate_manifest_task_id(manifest, task_id, "model_manifest")
        path = task_dir(task_id) / "model_manifest.json"
        write_json(path, manifest)
        return manifest
    return {"bundle_dir": str(payload.get("bundle_dir") or CONFIG.default_bundle_dir)}


def build_runner_command(task_id: str, payload: dict, output_dir: Path) -> tuple[list[str], dict]:
    public_manifest = manifest_path_from_payload(
        task_id,
        payload,
        "public_manifest_json",
        "public_manifest",
        "public_manifest.json",
    )
    p1_manifest = manifest_path_from_payload(
        task_id,
        payload,
        "p1_share_manifest_json",
        "p1_share_manifest",
        "p1_share_manifest.json",
    )
    p2_manifest = manifest_path_from_payload(
        task_id,
        payload,
        "p2_share_manifest_json",
        "p2_share_manifest",
        "p2_share_manifest.json",
    )
    model_manifest = load_optional_model_manifest(task_id, payload)
    bundle_dir = Path(str(model_manifest.get("bundle_dir") or CONFIG.default_bundle_dir)).expanduser().resolve()
    spu_params_mode = str(model_manifest.get("spu_params_mode") or "secret")
    spu_config = Path(str(payload.get("spu_config_json") or CONFIG.default_spu_config)).expanduser().resolve()
    candidate_json = output_dir / "candidate.json"
    candidate_pt = output_dir / "candidate.pt"
    command = [
        CONFIG.python_bin,
        str(CONFIG.runner_script),
        "run",
        "--runtime",
        "spu",
        "--bundle-dir",
        str(bundle_dir),
        "--input-share-public-manifest-json",
        str(public_manifest),
        "--input-p1-share-manifest-json",
        str(p1_manifest),
        "--input-p2-share-manifest-json",
        str(p2_manifest),
        "--party-local-share-load",
        "--redact-private-input-paths",
        "--config",
        str(spu_config),
        "--output-json",
        str(candidate_json),
        "--output-pt",
        str(candidate_pt),
        "--max-samples",
        str(int(payload.get("max_samples") or 1)),
        "--spu-params-mode",
        spu_params_mode,
    ]
    return command, {
        "public_manifest_json": str(public_manifest),
        "p1_share_manifest_json": str(p1_manifest),
        "p2_share_manifest_json": str(p2_manifest),
        "model_manifest": model_manifest,
        "candidate_json": str(candidate_json),
        "candidate_pt": str(candidate_pt),
        "spu_config_json": str(spu_config),
        "spu_params_mode": spu_params_mode,
    }


def terminate_process(process: subprocess.Popen):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass


def log_tail(path: Path, line_count: int = 60) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def run_task_worker(task_id: str, runtime_mode: str, command: list[str], output_dir: Path, timeout_sec: float):
    log_path = output_dir / "runner.log"
    try:
        write_state(task_id, {"status": "running", "started_at": now_ts(), "runner_log": str(log_path)})
        if runtime_mode == "mock":
            result = {
                "status": "completed",
                "live_mode": "mock",
                "task_id": task_id,
                "note": "mock coordinator run; no SPU process was started",
            }
            write_json(output_dir / "candidate.json", result)
            write_state(task_id, {"status": "completed", "completed_at": now_ts(), "result": result})
            return

        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        with RUNNERS_LOCK:
            RUNNERS[task_id] = process
        try:
            returncode = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            terminate_process(process)
            write_state(task_id, {"status": "failed", "error_code": "runner_timeout", "log_tail": log_tail(log_path)})
            return
        if returncode == 0:
            result_path = output_dir / "candidate.json"
            result = read_json(result_path) if result_path.exists() else {"status": "completed"}
            write_state(task_id, {"status": "completed", "completed_at": now_ts(), "result": result})
        elif read_state(task_id).get("status") == "cancel_requested":
            write_state(task_id, {"status": "cancelled", "cancelled_at": now_ts(), "returncode": returncode})
        else:
            write_state(
                task_id,
                {
                    "status": "failed",
                    "error_code": "runner_nonzero_exit",
                    "returncode": returncode,
                    "log_tail": log_tail(log_path),
                },
            )
    finally:
        with RUNNERS_LOCK:
            RUNNERS.pop(task_id, None)


@app.post("/api/coordinator/tasks/{task_id}/runs")
async def start_coordinator_run(task_id: str, request: Request, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    require_role("coordinator")
    task_id = validate_task_id(task_id)
    payload = await request_json(request)
    output_dir = task_dir(task_id) / "coordinator_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_mode = str(payload.get("runtime_mode") or CONFIG.default_runtime_mode).lower()
    if runtime_mode not in {"mock", "spu"}:
        raise HTTPException(status_code=400, detail="runtime_mode must be mock or spu")
    command, artifacts = build_runner_command(task_id, payload, output_dir)
    state = write_state(
        task_id,
        {
            "status": "queued",
            "queued_at": now_ts(),
            "runtime_mode": runtime_mode,
            "command": command,
            "artifacts": artifacts,
        },
    )
    thread = threading.Thread(
        target=run_task_worker,
        args=(task_id, runtime_mode, command, output_dir, CONFIG.run_timeout_sec),
        daemon=True,
    )
    thread.start()
    return response(202, {"status": "queued", "task": state})


@app.post("/api/coordinator/tasks/{task_id}/cancel")
async def cancel_coordinator_run(task_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    require_role("coordinator")
    task_id = validate_task_id(task_id)
    with RUNNERS_LOCK:
        process = RUNNERS.get(task_id)
    if process is None:
        state = write_state(task_id, {"status": "cancel_requested", "cancelled_at": now_ts(), "cancel_effect": "no_running_process"})
        return response(202, {"status": "cancel_requested", "task": state})
    terminate_process(process)
    state = write_state(task_id, {"status": "cancel_requested", "cancelled_at": now_ts(), "cancel_effect": "terminated_process"})
    return response(202, {"status": "cancel_requested", "task": state})
