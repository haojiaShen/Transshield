from __future__ import annotations

import asyncio
import json
import threading
import time
from http import HTTPStatus
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from showcase_api.bundle_preflight import inspect_bundle
from showcase_api.config import REPO_ROOT, ShowcaseConfig, load_json, load_showcase_config
from showcase_api.control_plane import (
    ControlPlaneError,
    GuardState,
    append_jsonl,
    parse_raw_multipart,
    validate_medical_payload,
)
from showcase_api.runtime import run_live_demo


CONFIG = load_showcase_config()
GUARD_STATE = GuardState(CONFIG)
AUDIT_LOCK = threading.Lock()
AUDIT_EVENTS_PATH = CONFIG.audit_dir / "audit_events.jsonl"
AUDIT_REJECTIONS_PATH = CONFIG.audit_dir / "audit_rejections.jsonl"
MEDICAL_REQUEST_FIELDS = {
    "request_manifest",
    "quality_assurance",
    "audit",
    "control_plane_metrics",
    "share0",
    "share1",
}

app = FastAPI(title="TransShield Showcase API", version="2026.05")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_response_payload(
    *,
    status: str,
    result=None,
    quality_assurance=None,
    audit=None,
    control_plane_metrics=None,
    error_code: str | None = None,
    interception_layer: str | None = None,
    detail: str | None = None,
):
    payload = {
        "status": status,
        "result": result,
        "quality_assurance": quality_assurance,
        "audit": audit,
        "control_plane_metrics": control_plane_metrics,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    if interception_layer is not None:
        payload["interception_layer"] = interception_layer
    if detail is not None:
        payload["detail"] = detail
    return payload


def response_json(status_code: int, payload: dict):
    return JSONResponse(status_code=status_code, content=payload)


def parse_content_length(request: Request) -> int:
    raw = request.headers.get("content-length", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def record_event(path: Path, payload: dict):
    append_jsonl(path, payload, AUDIT_LOCK)


def read_demo_summary():
    return load_json(CONFIG.demo_summary_json)


@app.get("/api/medical/config")
async def get_medical_config():
    summary = read_demo_summary()
    medical_item = next(
        item for item in summary["showcase_domains"]["items"] if item["id"] == "medical"
    )
    return {
        "status": "ok",
        "bundle": {
            "bundle_dir": str(CONFIG.bundle_dir.relative_to(REPO_ROOT)),
            "display_name": summary["default_bundle"]["display_name"],
            "status": summary["default_bundle"]["status"],
        },
        "threshold": CONFIG.formal_threshold,
        "input_size": CONFIG.input_size,
        "shape": CONFIG.expected_shape,
        "dtype": "float32_le",
        "mean": list(CONFIG.norm_mean),
        "std": list(CONFIG.norm_std),
        "clip_abs": CONFIG.norm_clip_abs,
        "crop_pct": CONFIG.eval_crop_pct,
        "resize_shorter_side": CONFIG.eval_resize_shorter_side,
        "allowed_mime_types": list(CONFIG.allowed_mime_types),
        "max_file_size_bytes": CONFIG.max_file_size_bytes,
        "max_image_dimension": CONFIG.max_image_dimension,
        "estimated_wait_seconds": CONFIG.estimated_wait_seconds,
        "class_names": list(CONFIG.class_names),
        "formal_metrics": {
            "threshold_accuracy": CONFIG.formal_threshold_accuracy,
            "auc": CONFIG.formal_auc,
            "sec_per_sample": CONFIG.formal_sec_per_sample,
            "dual_total_gib": CONFIG.formal_dual_total_gib,
        },
        "demo_boundary_note": medical_item["summary"],
        "limitations": [
            "请求进入 SPU 后，当前 demo 原型不能保证中途断连即终止任务。",
            "当前协调服务会在单进程内短暂接收 share0/share1；生产边界应拆分为独立 P1/P2 服务。",
        ],
    }


@app.get("/api/health")
async def get_health():
    bundle_status = inspect_bundle(CONFIG.bundle_dir)
    return {
        "status": "ok" if CONFIG.runtime_mode != "spu" or bundle_status["ready"] else "degraded",
        "runtime_mode": CONFIG.runtime_mode,
        "bundle_present": bundle_status["bundle_dir_present"],
        "model_state_present": bundle_status["model_state_present"],
        "model_state_file": bundle_status["model_state_file"],
        "spu_config_present": CONFIG.spu_config_path.exists(),
        "runner_present": (REPO_ROOT / "integrations" / "transshield_runtime" / "e2e_secure_vit" / "transshield_e2e_secure_vit.py").exists(),
        "dist_present": CONFIG.health_bind_dist.exists(),
        "inflight": GUARD_STATE.snapshot(),
    }


@app.post("/api/medical/live-run")
async def medical_live_run(request: Request):
    request_started_at = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    allowed, rate_error = GUARD_STATE.check_rate_limit(client_ip)
    if not allowed:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code=rate_error,
            interception_layer="ip_rate_limit_guard",
            detail="当前来源请求过于频繁，请稍后再试。",
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {"ts": time.time(), "ip": client_ip, "error_code": rate_error, "interception_layer": "ip_rate_limit_guard"},
        )
        return response_json(int(HTTPStatus.TOO_MANY_REQUESTS), payload)

    if request.headers.get("transfer-encoding"):
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="transfer_encoding_not_supported",
            interception_layer="http_request_body_gate",
            detail="当前演示接口不接受 Transfer-Encoding 请求体。",
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {"ts": time.time(), "ip": client_ip, "error_code": "transfer_encoding_not_supported", "interception_layer": "http_request_body_gate"},
        )
        return response_json(int(HTTPStatus.BAD_REQUEST), payload)

    content_length = parse_content_length(request)
    if content_length <= 0:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="invalid_content_length",
            interception_layer="http_request_body_gate",
            detail="请求体为空或 Content-Length 非法。",
        )
        return response_json(int(HTTPStatus.BAD_REQUEST), payload)
    if content_length > CONFIG.max_request_bytes:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="payload_too_large",
            interception_layer="http_request_body_gate",
            detail="请求体超过当前演示接口允许的大小。",
        )
        return response_json(int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE), payload)

    raw_body = await request.body()
    if len(raw_body) > CONFIG.max_request_bytes:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="payload_too_large",
            interception_layer="http_request_body_gate",
            detail="请求体超过当前演示接口允许的大小。",
        )
        return response_json(int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE), payload)

    try:
        parts = parse_raw_multipart(request.headers.get("content-type", ""), raw_body, MEDICAL_REQUEST_FIELDS)
    except ValueError as exc:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="malformed_multipart_precheck_failed",
            interception_layer="raw_multipart_precheck",
            detail=str(exc),
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {"ts": time.time(), "ip": client_ip, "error_code": "malformed_multipart_precheck_failed", "interception_layer": "raw_multipart_precheck", "detail": str(exc)},
        )
        return response_json(int(HTTPStatus.BAD_REQUEST), payload)

    try:
        validated = validate_medical_payload(raw_body, parts, CONFIG)
    except ControlPlaneError as exc:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=exc.quality_assurance,
            audit=exc.audit,
            control_plane_metrics=exc.control_plane_metrics,
            error_code=exc.error_code,
            interception_layer=exc.interception_layer,
            detail=exc.detail or exc.message,
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {
                "ts": time.time(),
                "ip": client_ip,
                "audit_nonce": None,
                "error_code": exc.error_code,
                "interception_layer": exc.interception_layer,
                "detail": exc.detail or exc.message,
            },
        )
        return response_json(exc.status_code, payload)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=None,
            audit=None,
            control_plane_metrics=None,
            error_code="invalid_control_plane_payload",
            interception_layer="json_bytes_gate",
            detail=str(exc),
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {"ts": time.time(), "ip": client_ip, "error_code": "invalid_control_plane_payload", "interception_layer": "json_bytes_gate", "detail": str(exc)},
        )
        return response_json(int(HTTPStatus.BAD_REQUEST), payload)

    validated.control_plane_metrics_response["server_pre_spu_checks_ms"] = round(
        (time.perf_counter() - request_started_at) * 1000.0,
        3,
    )
    audit_nonce = validated.request_manifest["audit_nonce"]

    replay_ok, replay_error = GUARD_STATE.check_and_remember_replay(audit_nonce, validated.payload_fingerprint)
    if not replay_ok:
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=validated.quality_assurance_response,
            audit=validated.audit_response,
            control_plane_metrics=validated.control_plane_metrics_response,
            error_code=replay_error,
            interception_layer="replay_guard",
            detail="检测到重复或饱和的控制面请求。",
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {
                "ts": time.time(),
                "ip": client_ip,
                "audit_nonce": audit_nonce,
                "payload_fingerprint": validated.payload_fingerprint,
                "error_code": replay_error,
                "interception_layer": "replay_guard",
            },
        )
        return response_json(int(HTTPStatus.CONFLICT), payload)

    reserved, reserve_error = GUARD_STATE.reserve_inflight(client_ip)
    if not reserved:
        GUARD_STATE.forget_replay(audit_nonce, validated.payload_fingerprint)
        payload = build_response_payload(
            status="rejected",
            result=None,
            quality_assurance=validated.quality_assurance_response,
            audit=validated.audit_response,
            control_plane_metrics=validated.control_plane_metrics_response,
            error_code=reserve_error,
            interception_layer="inflight_guard",
            detail="当前演示环境繁忙，请稍后重试。",
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {
                "ts": time.time(),
                "ip": client_ip,
                "audit_nonce": audit_nonce,
                "payload_fingerprint": validated.payload_fingerprint,
                "error_code": reserve_error,
                "interception_layer": "inflight_guard",
            },
        )
        return response_json(int(HTTPStatus.TOO_MANY_REQUESTS), payload)

    try:
        result = await asyncio.to_thread(
            run_live_demo,
            CONFIG,
            validated.share0_bytes,
            validated.share1_bytes,
        )
    except Exception as exc:
        payload = build_response_payload(
            status="failed",
            result=None,
            quality_assurance=validated.quality_assurance_response,
            audit=validated.audit_response,
            control_plane_metrics=validated.control_plane_metrics_response,
            error_code="medical_secure_run_failed",
            interception_layer="spu_runtime",
            detail="安全推理运行失败，请查看服务端审计日志。",
        )
        record_event(
            AUDIT_REJECTIONS_PATH,
            {
                "ts": time.time(),
                "ip": client_ip,
                "audit_nonce": audit_nonce,
                "payload_fingerprint": validated.payload_fingerprint,
                "error_code": "medical_secure_run_failed",
                "interception_layer": "spu_runtime",
                "detail": str(exc),
            },
        )
        return response_json(int(HTTPStatus.INTERNAL_SERVER_ERROR), payload)
    finally:
        GUARD_STATE.release_inflight(client_ip)

    validated.control_plane_metrics_response["server_total_ms"] = round(
        (time.perf_counter() - request_started_at) * 1000.0,
        3,
    )
    response_payload = build_response_payload(
        status="completed",
        result={
            **result,
            "formal_metrics": {
                "threshold_accuracy": CONFIG.formal_threshold_accuracy,
                "auc": CONFIG.formal_auc,
                "sec_per_sample": CONFIG.formal_sec_per_sample,
                "dual_total_gib": CONFIG.formal_dual_total_gib,
            },
            "boundary_note": (
                "当前 live demo 为单协调服务原型：浏览器本地生成 share0/share1，服务端短暂接收两份 share 后进入单通道 SPU 运行。"
                "正式生产边界应拆分为独立 P1/P2 服务，并维持 final logits only 的 reveal 策略。"
            ),
            "limitation_note": (
                "请求进入 SPU 后，当前 demo 原型不能保证中途断连即终止任务。"
            ),
        },
        quality_assurance=validated.quality_assurance_response,
        audit=validated.audit_response,
        control_plane_metrics=validated.control_plane_metrics_response,
    )
    record_event(
        AUDIT_EVENTS_PATH,
        {
            "ts": time.time(),
            "ip": client_ip,
            "audit_nonce": audit_nonce,
            "payload_fingerprint": validated.payload_fingerprint,
            "request_total_ms": validated.control_plane_metrics_response["server_total_ms"],
            "quality_status": validated.quality_assurance_response["status"],
        },
    )
    return response_json(int(HTTPStatus.OK), response_payload)


if CONFIG.health_bind_dist.exists():
    assets_dir = CONFIG.health_bind_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="showcase-assets")
    report_assets_dir = CONFIG.health_bind_dist / "report-assets"
    if report_assets_dir.exists():
        app.mount("/report-assets", StaticFiles(directory=str(report_assets_dir)), name="showcase-report-assets")


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith("api/"):
        return response_json(int(HTTPStatus.NOT_FOUND), {"status": "failed", "detail": "Not found"})
    index_path = CONFIG.health_bind_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return response_json(
        int(HTTPStatus.SERVICE_UNAVAILABLE),
        {
            "status": "failed",
            "detail": "showcase dist is missing; run `cd showcase && npm install && npm run build` first.",
        },
    )
