from __future__ import annotations

import email.policy
import hashlib
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from email.parser import BytesParser
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import numpy as np

from showcase_api.config import ShowcaseConfig


JSON_PART_MAX_BYTES = 4096
REQUEST_MANIFEST_MAX_BYTES = 2048
QUALITY_ASSURANCE_MAX_BYTES = 2048
AUDIT_MAX_BYTES = 2048
CONTROL_PLANE_METRICS_MAX_BYTES = 1024
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JSON_INT_MAX_DIGITS = 32
JSON_FLOAT_MAX_CHARS = 64
JSON_FLOAT_MAX_EXPONENT = 6


@dataclass(frozen=True)
class RawPart:
    name: str
    headers: dict
    body_start: int
    body_end: int
    filename: Optional[str] = None
    content_type: Optional[str] = None


@dataclass
class ValidatedRequest:
    share0_bytes: bytes
    share1_bytes: bytes
    request_manifest: dict
    client_quality_assurance: dict
    audit_payload: dict
    client_control_plane_metrics: dict
    quality_assurance_response: dict
    audit_response: dict
    control_plane_metrics_response: dict
    payload_fingerprint: str


@dataclass
class ControlPlaneError(Exception):
    message: str
    error_code: str
    interception_layer: str
    status_code: int
    detail: str = ""
    quality_assurance: Optional[dict] = None
    audit: Optional[dict] = None
    control_plane_metrics: Optional[dict] = None


@dataclass
class IpWindowState:
    requests: deque = field(default_factory=deque)
    inflight: int = 0
    last_seen: float = 0.0


class GuardState:
    def __init__(self, config: ShowcaseConfig):
        self.config = config
        self._lock = threading.Lock()
        self._nonce_expiry: dict[str, float] = {}
        self._payload_expiry: dict[str, float] = {}
        self._ip_states: dict[str, IpWindowState] = defaultdict(IpWindowState)
        self._global_inflight = 0

    def _cleanup_locked(self, now: float):
        expired_nonces = [key for key, expiry in self._nonce_expiry.items() if expiry <= now]
        for key in expired_nonces:
            self._nonce_expiry.pop(key, None)
        expired_payloads = [key for key, expiry in self._payload_expiry.items() if expiry <= now]
        for key in expired_payloads:
            self._payload_expiry.pop(key, None)
        stale_ips = []
        for ip, state in self._ip_states.items():
            while state.requests and state.requests[0] <= now - self.config.per_ip_window_seconds:
                state.requests.popleft()
            if state.inflight == 0 and not state.requests and state.last_seen <= now - self.config.per_ip_window_seconds:
                stale_ips.append(ip)
        for ip in stale_ips:
            self._ip_states.pop(ip, None)

    def check_rate_limit(self, ip: str) -> tuple[bool, Optional[str]]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            state = self._ip_states[ip]
            while state.requests and state.requests[0] <= now - self.config.per_ip_window_seconds:
                state.requests.popleft()
            if len(state.requests) >= self.config.per_ip_window_limit:
                state.last_seen = now
                return False, "rate_limited_ip"
            state.requests.append(now)
            state.last_seen = now
            return True, None

    def check_and_remember_replay(self, nonce: str, payload_fingerprint: str) -> tuple[bool, Optional[str]]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            if nonce in self._nonce_expiry:
                return False, "duplicate_nonce"
            if payload_fingerprint in self._payload_expiry:
                return False, "duplicate_payload"
            if (
                len(self._nonce_expiry) >= self.config.replay_nonce_capacity
                or len(self._payload_expiry) >= self.config.replay_payload_capacity
            ):
                return False, "replay_cache_saturated"
            self._nonce_expiry[nonce] = now + self.config.replay_nonce_ttl_seconds
            self._payload_expiry[payload_fingerprint] = now + self.config.replay_payload_ttl_seconds
            return True, None

    def forget_replay(self, nonce: str, payload_fingerprint: str):
        """Roll back a replay reservation when the request never enters the run queue."""
        with self._lock:
            self._nonce_expiry.pop(nonce, None)
            self._payload_expiry.pop(payload_fingerprint, None)

    def reserve_inflight(self, ip: str) -> tuple[bool, Optional[str]]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            state = self._ip_states[ip]
            if state.inflight >= self.config.per_ip_inflight_limit:
                state.last_seen = now
                return False, "busy_retry_later"
            if self._global_inflight >= self.config.global_inflight_limit:
                state.last_seen = now
                return False, "busy_retry_later"
            state.inflight += 1
            state.last_seen = now
            self._global_inflight += 1
            return True, None

    def release_inflight(self, ip: str):
        with self._lock:
            state = self._ip_states.get(ip)
            if state is not None and state.inflight > 0:
                state.inflight -= 1
                state.last_seen = time.monotonic()
            if self._global_inflight > 0:
                self._global_inflight -= 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "global_inflight": self._global_inflight,
                "global_inflight_limit": self.config.global_inflight_limit,
                "per_ip_inflight_limit": self.config.per_ip_inflight_limit,
                "replay_nonce_cache_size": len(self._nonce_expiry),
                "replay_payload_cache_size": len(self._payload_expiry),
                "replay_nonce_capacity": self.config.replay_nonce_capacity,
                "replay_payload_capacity": self.config.replay_payload_capacity,
                "tracked_ip_count": len(self._ip_states),
            }


def build_mime_message(content_type: str, body: bytes) -> bytes:
    return f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body


def parse_content_type_boundary(content_type: str) -> Optional[bytes]:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    return raw.encode("utf-8") if raw else None


def parse_part_headers_bytes(header_bytes: bytes) -> dict:
    headers = {}
    for line in header_bytes.split(b"\r\n"):
        if not line:
            continue
        if b":" not in line:
            raise ValueError("invalid multipart header line")
        key, value = line.split(b":", 1)
        key_text = key.decode("ascii", "strict").strip().lower()
        if "\x00" in key_text:
            raise ValueError("multipart header contains null byte")
        headers[key_text] = value.decode("latin-1").strip()
    return headers


def parse_content_disposition(value: str) -> dict:
    parts = [item.strip() for item in value.split(";") if item.strip()]
    if not parts:
        return {}
    result = {"_kind": parts[0].lower()}
    for item in parts[1:]:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        result[key.strip().lower()] = raw.strip().strip('"')
    return result


def strict_json_int(value: str):
    if len(value.lstrip("+-")) > JSON_INT_MAX_DIGITS:
        raise ValueError("json integer token too large")
    return int(value)


def strict_json_float(value: str):
    if len(value) > JSON_FLOAT_MAX_CHARS:
        raise ValueError("json float token too large")
    lower = value.lower()
    if "e" in lower:
        _, exponent = lower.split("e", 1)
        if abs(int(exponent)) > JSON_FLOAT_MAX_EXPONENT:
            raise ValueError("json float exponent too large")
    return float(value)


def strict_json_constant(value: str):
    raise ValueError(f"json constant not allowed: {value}")


def ensure_sha256_hex(name: str, value) -> str:
    text = str(value or "").strip().lower()
    if not HEX_SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase 64-char sha256 hex string")
    return text


def parse_raw_multipart(content_type: str, raw_body: bytes, expected_fields: set[str]) -> dict[str, RawPart]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("content type is not multipart/form-data")
    boundary = parse_content_type_boundary(content_type)
    if not boundary:
        raise ValueError("multipart boundary missing")
    marker = b"--" + boundary
    if not raw_body.startswith(marker + b"\r\n"):
        raise ValueError("multipart body does not start with boundary")

    parts = {}
    position = len(marker) + 2
    boundary_count = 1
    while True:
        header_end = raw_body.find(b"\r\n\r\n", position)
        if header_end == -1:
            raise ValueError("multipart part headers not terminated")
        header_bytes = raw_body[position:header_end]
        if len(header_bytes) > 8 * 1024:
            raise ValueError("multipart part headers too large")
        headers = parse_part_headers_bytes(header_bytes)
        disposition = parse_content_disposition(headers.get("content-disposition", ""))
        if disposition.get("_kind") != "form-data":
            raise ValueError("invalid content-disposition")
        name = disposition.get("name")
        if not name:
            raise ValueError("multipart field name missing")
        if "\x00" in name:
            raise ValueError("multipart field contains null byte")
        if name in parts:
            raise ValueError(f"duplicate multipart field: {name}")
        if headers.get("content-type", "").lower().startswith("multipart/"):
            raise ValueError("nested multipart not allowed")

        body_start = header_end + 4
        next_boundary = raw_body.find(b"\r\n" + marker, body_start)
        if next_boundary == -1:
            raise ValueError("multipart closing boundary missing")
        parts[name] = RawPart(
            name=name,
            headers=headers,
            body_start=body_start,
            body_end=next_boundary,
            filename=disposition.get("filename"),
            content_type=headers.get("content-type"),
        )
        boundary_line_start = next_boundary + 2
        boundary_line_end = boundary_line_start + len(marker)
        boundary_count += 1
        if boundary_count > len(expected_fields) + 2:
            raise ValueError("too many multipart boundaries")
        if raw_body.startswith(b"--", boundary_line_end):
            if raw_body[boundary_line_end + 2 :] not in (b"", b"\r\n"):
                raise ValueError("multipart epilogue is not allowed")
            break
        if raw_body[boundary_line_end : boundary_line_end + 2] != b"\r\n":
            raise ValueError("invalid multipart boundary separator")
        position = boundary_line_end + 2

    if set(parts) != expected_fields:
        raise ValueError("unexpected multipart field set")

    message = BytesParser(policy=email.policy.default).parsebytes(build_mime_message(content_type, raw_body))
    if not message.is_multipart():
        raise ValueError("mime root is not multipart")
    payload = message.get_payload()
    if not isinstance(payload, list) or len(payload) != len(expected_fields):
        raise ValueError("unexpected multipart part count")
    for item in payload:
        if item.is_multipart():
            raise ValueError("nested multipart is not allowed")
    return parts


def extract_part_bytes(raw_body: bytes, part: RawPart) -> bytes:
    return raw_body[part.body_start : part.body_end]


def load_json_part(raw_body: bytes, part: RawPart) -> dict:
    payload = extract_part_bytes(raw_body, part)
    field_limit = {
        "request_manifest": REQUEST_MANIFEST_MAX_BYTES,
        "quality_assurance": QUALITY_ASSURANCE_MAX_BYTES,
        "audit": AUDIT_MAX_BYTES,
        "control_plane_metrics": CONTROL_PLANE_METRICS_MAX_BYTES,
    }.get(part.name, JSON_PART_MAX_BYTES)
    if len(payload) > field_limit:
        raise ValueError(f"{part.name} exceeds field size limit")
    decoded = payload.decode("utf-8", "strict")
    parsed = json.loads(
        decoded,
        parse_int=strict_json_int,
        parse_float=strict_json_float,
        parse_constant=strict_json_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{part.name} must be a JSON object")
    return parsed


def bytes_to_float32_aligned(payload: bytes) -> np.ndarray:
    if len(payload) % 4 != 0:
        raise ValueError("payload length is not aligned to 4 bytes")
    return np.frombuffer(payload, dtype="<u4").copy()


def flush_tiny_values_inplace(values: np.ndarray, threshold: float = 1e-30):
    mask = np.abs(values) < threshold
    if np.any(mask):
        values[mask] = np.float32(0.0)


def contains_subnormal_values(share_u32: np.ndarray) -> bool:
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponent_zero = (share_u32 & exponent_mask) == 0
    mantissa_nonzero = (share_u32 & mantissa_mask) != 0
    return bool(np.any(exponent_zero & mantissa_nonzero))


def server_sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_share_digest(label: str, payload: bytes) -> str:
    digest = hashlib.blake2s()
    digest.update(label.encode("ascii"))
    digest.update(b"|")
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)
    return digest.hexdigest()


def make_payload_fingerprint(share0_bytes: bytes, share1_bytes: bytes) -> str:
    payload = (
        f"medical-live|s0:{make_share_digest('share0', share0_bytes)}|"
        f"s1:{make_share_digest('share1', share1_bytes)}"
    )
    return hashlib.blake2s(payload.encode("utf-8")).hexdigest()


def compute_luma_metrics(rgb_tensor: np.ndarray) -> dict:
    rgb = np.clip(rgb_tensor.astype(np.float32, copy=False), 0.0, 1.0)
    r = rgb[0, 0]
    g = rgb[0, 1]
    b = rgb[0, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    p05, p95 = np.percentile(luma, [5, 95])
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return {
        "mean_luma": float(np.mean(luma)),
        "std_luma": float(np.std(luma)),
        "overexposed_ratio": float(np.mean(luma >= 0.95)),
        "underexposed_ratio": float(np.mean(luma <= 0.05)),
        "effective_luma_ratio": float(np.mean((luma >= 0.02) & (luma <= 0.98))),
        "dynamic_range_p95_p05": float(p95 - p05),
        "laplacian_variance": float(np.var(lap)),
    }


def validate_quality_summary_object(payload: dict):
    for key in (
        "mean_luma",
        "std_luma",
        "overexposed_ratio",
        "underexposed_ratio",
        "effective_luma_ratio",
        "dynamic_range_p95_p05",
        "laplacian_variance",
    ):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ValueError(f"quality summary field {key} must be a finite number")


def validate_control_plane_metrics_object(payload: dict):
    for key in ("decode_ms", "preprocess_ms", "dqa_ms", "hash_ms", "share_build_ms", "total_ms"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"control_plane_metrics.{key} must be a finite non-negative number")


def build_quality_assurance_response(
    server_summary: dict,
    client_summary: dict,
    client_status: str,
    tolerance: float,
) -> dict:
    severe_reasons = []
    warning_reasons = []
    if server_summary["overexposed_ratio"] > 0.40:
        severe_reasons.append("overexposed_ratio>0.40")
    elif server_summary["overexposed_ratio"] > 0.15:
        warning_reasons.append("overexposed_ratio>0.15")
    if server_summary["underexposed_ratio"] > 0.40:
        severe_reasons.append("underexposed_ratio>0.40")
    elif server_summary["underexposed_ratio"] > 0.15:
        warning_reasons.append("underexposed_ratio>0.15")
    if server_summary["laplacian_variance"] < 1e-4 and (
        server_summary["effective_luma_ratio"] < 0.10 or server_summary["dynamic_range_p95_p05"] < 0.02
    ):
        severe_reasons.append("degenerate_structure")
    elif server_summary["laplacian_variance"] < 5e-4:
        warning_reasons.append("laplacian_variance<5e-4")

    drift = {}
    significant = []
    for key in (
        "mean_luma",
        "std_luma",
        "overexposed_ratio",
        "underexposed_ratio",
        "effective_luma_ratio",
        "dynamic_range_p95_p05",
        "laplacian_variance",
    ):
        client_value = float(client_summary[key])
        server_value = float(server_summary[key])
        abs_diff = abs(client_value - server_value)
        drift[key] = {
            "client": client_value,
            "server": server_value,
            "abs_diff": abs_diff,
            "within_tolerance": abs_diff <= tolerance,
        }
        if abs_diff > tolerance:
            significant.append(key)

    if severe_reasons:
        status = "block"
    elif warning_reasons:
        status = "warn"
    else:
        status = "pass"
    return {
        "client_status": client_status,
        "status": status,
        "integrity_status": "consistent" if not significant else "client_summary_drifted",
        "integrity_reasons": significant,
        "client_quality_summary": client_summary,
        "server_quality_summary": server_summary,
        "client_vs_server_drift": drift,
        "blocking_reasons": severe_reasons,
        "warning_reasons": warning_reasons,
    }


def validate_request_manifest(manifest: dict, config: ShowcaseConfig):
    if manifest.get("manifest_type") != config.request_manifest_type:
        raise ValueError("unexpected request_manifest.manifest_type")
    if manifest.get("contract_version") != config.request_contract_version:
        raise ValueError("unexpected request_manifest.contract_version")
    if manifest.get("input_size") != config.input_size:
        raise ValueError("unexpected request_manifest.input_size")
    if manifest.get("shape") != config.expected_shape:
        raise ValueError("unexpected request_manifest.shape")
    if manifest.get("dtype") != "float32_le":
        raise ValueError("unexpected request_manifest.dtype")
    mime = str(manifest.get("source_mime") or "").strip().lower()
    if mime not in config.allowed_mime_types:
        raise ValueError("unexpected request_manifest.source_mime")
    dimensions = manifest.get("source_dimensions") or {}
    width = int(dimensions.get("width", 0) or 0)
    height = int(dimensions.get("height", 0) or 0)
    if width <= 0 or height <= 0 or width > config.max_image_dimension or height > config.max_image_dimension:
        raise ValueError("invalid request_manifest.source_dimensions")
    if int(manifest.get("source_size_bytes", 0) or 0) <= 0:
        raise ValueError("invalid request_manifest.source_size_bytes")
    ensure_sha256_hex("source_image_sha256", manifest.get("source_image_sha256"))
    ensure_sha256_hex("normalized_tensor_sha256", manifest.get("normalized_tensor_sha256"))
    ensure_sha256_hex("share0_sha256", manifest.get("share0_sha256"))
    ensure_sha256_hex("share1_sha256", manifest.get("share1_sha256"))
    ensure_sha256_hex("audit_chain_sha256", manifest.get("audit_chain_sha256"))
    audit_nonce = str(manifest.get("audit_nonce") or "").strip()
    if not audit_nonce or len(audit_nonce) > 128:
        raise ValueError("invalid request_manifest.audit_nonce")


def validate_medical_payload(
    raw_body: bytes,
    parts: dict[str, RawPart],
    config: ShowcaseConfig,
) -> ValidatedRequest:
    request_manifest = load_json_part(raw_body, parts["request_manifest"])
    quality_assurance = load_json_part(raw_body, parts["quality_assurance"])
    audit_payload = load_json_part(raw_body, parts["audit"])
    control_plane_metrics = load_json_part(raw_body, parts["control_plane_metrics"])

    validate_request_manifest(request_manifest, config)
    client_summary = quality_assurance.get("client_quality_summary")
    if not isinstance(client_summary, dict):
        raise ValueError("quality_assurance.client_quality_summary must be a JSON object")
    validate_quality_summary_object(client_summary)
    client_status = str(quality_assurance.get("status") or "pass")
    validate_control_plane_metrics_object(control_plane_metrics)

    if audit_payload.get("hash_chain_version") != config.request_contract_version:
        raise ValueError("unexpected audit.hash_chain_version")
    if audit_payload.get("browser_generated_shares") is not True:
        raise ValueError("audit.browser_generated_shares must be true")
    if audit_payload.get("server_should_receive_plain_image") is not False:
        raise ValueError("audit.server_should_receive_plain_image must be false")
    if audit_payload.get("server_should_receive_plain_pixel_values") is not False:
        raise ValueError("audit.server_should_receive_plain_pixel_values must be false")

    share0_bytes = extract_part_bytes(raw_body, parts["share0"])
    share1_bytes = extract_part_bytes(raw_body, parts["share1"])
    if len(share0_bytes) != config.share_byte_count or len(share1_bytes) != config.share_byte_count:
        raise ControlPlaneError(
            message="share 大小不正确。",
            error_code="invalid_share_length",
            interception_layer="share_tensor_gate",
            status_code=int(HTTPStatus.BAD_REQUEST),
            detail=f"expected {config.share_byte_count} bytes for each share",
        )

    share0_sha256 = server_sha256_hex(share0_bytes)
    share1_sha256 = server_sha256_hex(share1_bytes)
    client_share0_sha256 = ensure_sha256_hex("share0_sha256", request_manifest.get("share0_sha256"))
    client_share1_sha256 = ensure_sha256_hex("share1_sha256", request_manifest.get("share1_sha256"))
    if client_share0_sha256 != share0_sha256 or client_share1_sha256 != share1_sha256:
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="audit_share_hash_mismatch",
            interception_layer="share_hash_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="share sha256 mismatch",
        )

    audit_nonce = str(request_manifest["audit_nonce"]).strip()
    expected_audit_chain = hashlib.sha256(
        (
            f"{config.request_contract_version}|{audit_nonce}|{request_manifest['source_image_sha256']}|"
            f"{request_manifest['normalized_tensor_sha256']}|{share0_sha256}|{share1_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    if ensure_sha256_hex("audit_chain_sha256", request_manifest.get("audit_chain_sha256")) != expected_audit_chain:
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="audit_chain_mismatch",
            interception_layer="audit_chain_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="audit chain sha256 mismatch",
        )

    share0_u32 = bytes_to_float32_aligned(share0_bytes)
    share1_u32 = bytes_to_float32_aligned(share1_bytes)
    if contains_subnormal_values(share0_u32) or contains_subnormal_values(share1_u32):
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="invalid_subnormal_share",
            interception_layer="share_tensor_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="subnormal share values detected",
        )

    share0_f32 = share0_u32.view("<f4")
    share1_f32 = share1_u32.view("<f4")
    flush_tiny_values_inplace(share0_f32)
    flush_tiny_values_inplace(share1_f32)
    if not np.isfinite(share0_f32).all() or not np.isfinite(share1_f32).all():
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="non_finite_share",
            interception_layer="share_tensor_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="share contains non-finite values",
        )
    if float(np.max(np.abs(share0_f32))) > config.share_abs_guard or float(np.max(np.abs(share1_f32))) > config.share_abs_guard:
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="share_magnitude_out_of_range",
            interception_layer="share_tensor_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="share magnitude exceeds allowed bound",
        )

    reconstructed = np.add(share0_f32, share1_f32, dtype=np.float32).reshape(config.expected_shape)
    if not np.isfinite(reconstructed).all():
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="non_finite_tensor",
            interception_layer="tensor_reconstruction_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="reconstructed tensor contains non-finite values",
        )

    mean = np.asarray(config.norm_mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(config.norm_std, dtype=np.float32).reshape(1, 3, 1, 1)
    rgb = reconstructed * std + mean
    if float(np.min(rgb)) < -0.05 or float(np.max(rgb)) > 1.05:
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="invalid_tensor_rgb_range",
            interception_layer="tensor_reconstruction_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="reconstructed rgb range is invalid",
        )

    server_quality_summary = compute_luma_metrics(rgb)
    quality_assurance_response = build_quality_assurance_response(
        server_summary=server_quality_summary,
        client_summary=client_summary,
        client_status=client_status,
        tolerance=config.quality_drift_tolerance,
    )
    if quality_assurance_response["status"] == "block":
        raise ControlPlaneError(
            message="医疗控制面快检未通过。",
            error_code="quality_assurance_blocked",
            interception_layer="server_dqa_gate",
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            detail="quality assurance blocked request",
            quality_assurance=quality_assurance_response,
        )

    payload_fingerprint = make_payload_fingerprint(share0_bytes, share1_bytes)
    audit_response = {
        "audit_nonce": audit_nonce,
        "client_source_image_sha256": request_manifest["source_image_sha256"],
        "client_normalized_tensor_sha256": request_manifest["normalized_tensor_sha256"],
        "client_audit_chain_sha256": request_manifest["audit_chain_sha256"],
        "server_audit_chain_sha256": expected_audit_chain,
        "audit_chain_consistent": True,
        "server_share0_sha256": share0_sha256,
        "server_share1_sha256": share1_sha256,
        "server_payload_fingerprint": payload_fingerprint,
        "demo_boundary": {
            "browser_generated_shares": True,
            "server_received_plain_image": False,
            "server_received_plain_pixel_values": False,
            "server_reconstructed_normalized_tensor_for_dqa": True,
            "server_received_share0_and_share1_in_single_process": True,
        },
    }
    metrics_response = {
        "client": control_plane_metrics,
        "server_pre_spu_checks_ms": None,
        "request_size_bytes": len(raw_body),
    }
    return ValidatedRequest(
        share0_bytes=share0_bytes,
        share1_bytes=share1_bytes,
        request_manifest=request_manifest,
        client_quality_assurance=quality_assurance,
        audit_payload=audit_payload,
        client_control_plane_metrics=control_plane_metrics,
        quality_assurance_response=quality_assurance_response,
        audit_response=audit_response,
        control_plane_metrics_response=metrics_response,
        payload_fingerprint=payload_fingerprint,
    )


def append_jsonl(path: Path, payload: dict, lock: threading.Lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
