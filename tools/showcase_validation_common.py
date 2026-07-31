#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


SHOWCASE_REQUEST_MANIFEST_TYPE = "transshield_showcase_medical_live_request_v1"
SHOWCASE_REQUEST_CONTRACT_VERSION = "medical_live_demo_v1"


@dataclass
class HttpResult:
    status: Optional[int]
    headers: dict
    body: bytes
    json_body: Optional[dict]
    raw: bytes


@dataclass
class ProcessState:
    available: bool
    pid: Optional[int]
    timestamp: float
    fd_count: Optional[int]
    socket_fd_count: Optional[int]
    rss_kib: Optional[int]
    vmhwm_kib: Optional[int]
    thread_count: Optional[int]

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "pid": self.pid,
            "timestamp": round(self.timestamp, 6),
            "fd_count": self.fd_count,
            "socket_fd_count": self.socket_fd_count,
            "rss_kib": self.rss_kib,
            "vmhwm_kib": self.vmhwm_kib,
            "thread_count": self.thread_count,
        }


def parse_base_url(base_url: str):
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    return scheme, host, port, base_path


def build_endpoint(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def http_get_json(base_url: str, path: str, timeout: float) -> dict:
    endpoint = build_endpoint(base_url, path)
    with urlopen(endpoint, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_remote_medical_config(base_url: str, timeout: float) -> dict:
    return http_get_json(base_url, "/api/medical/config", timeout)


def pack_float32_le(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def compute_quality_summary(rgb_tensor: np.ndarray) -> dict:
    rgb = np.clip(np.asarray(rgb_tensor, dtype=np.float32), 0.0, 1.0)
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
        "mean_luma": round(float(np.mean(luma)), 8),
        "std_luma": round(float(np.std(luma)), 8),
        "overexposed_ratio": round(float(np.mean(luma >= 0.95)), 8),
        "underexposed_ratio": round(float(np.mean(luma <= 0.05)), 8),
        "effective_luma_ratio": round(float(np.mean((luma >= 0.02) & (luma <= 0.98))), 8),
        "dynamic_range_p95_p05": round(float(p95 - p05), 8),
        "laplacian_variance": round(float(np.var(lap)), 8),
    }


def make_synthetic_rgb(input_size: int, tensor_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(tensor_seed)
    xs = np.linspace(0.0, 2.0 * np.pi, input_size, dtype=np.float32)
    ys = np.linspace(0.0, 2.0 * np.pi, input_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base = 0.5 + 0.18 * np.sin(grid_x * 2.7) * np.cos(grid_y * 3.1)
    ripple = 0.08 * np.sin(grid_x * 7.0 + 0.3) + 0.05 * np.cos(grid_y * 5.0 - 0.7)
    noise = rng.normal(0.0, 0.01, size=(input_size, input_size)).astype(np.float32)
    red = np.clip(base + ripple + noise, 0.08, 0.92)
    green = np.clip(base - 0.6 * ripple + 0.5 * noise, 0.08, 0.92)
    blue = np.clip(base + 0.4 * np.sin(grid_x + grid_y) - 0.3 * noise, 0.08, 0.92)
    return np.stack([red, green, blue], axis=0)[None, ...].astype(np.float32)


def make_synthetic_tensor(config: dict, tensor_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    input_size = int(config["input_size"])
    rgb = make_synthetic_rgb(input_size, tensor_seed=tensor_seed)
    mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(config["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    clip_abs = float(config["clip_abs"])
    tensor = ((rgb - mean) / std).astype(np.float32)
    if clip_abs > 0:
        tensor = np.clip(tensor, -clip_abs, clip_abs).astype(np.float32)
    return tensor, rgb


def default_client_metrics() -> dict:
    return {
        "decode_ms": 1.25,
        "preprocess_ms": 4.75,
        "dqa_ms": 1.10,
        "hash_ms": 2.85,
        "share_build_ms": 3.65,
        "total_ms": 13.60,
    }


def build_medical_parts(
    config: dict,
    *,
    tensor_seed: int = 0,
    share_seed: int = 0,
    nonce: Optional[str] = None,
    quality_override: Optional[dict] = None,
    metrics_override: Optional[dict] = None,
    request_manifest_override: Optional[dict] = None,
    audit_override: Optional[dict] = None,
    share0_bytes: Optional[bytes] = None,
    share1_bytes: Optional[bytes] = None,
) -> list[dict]:
    tensor, _rgb = make_synthetic_tensor(config, tensor_seed=tensor_seed)
    mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(config["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    if share0_bytes is None or share1_bytes is None:
        rng = np.random.default_rng(share_seed)
        share0 = rng.uniform(-0.8, 0.8, size=tensor.shape).astype(np.float32)
        share1 = (tensor - share0).astype(np.float32)
        share0_bytes = pack_float32_le(share0)
        share1_bytes = pack_float32_le(share1)

    shape = tuple(int(value) for value in config["shape"])
    share0_tensor = np.frombuffer(share0_bytes, dtype="<f4").copy().reshape(shape)
    share1_tensor = np.frombuffer(share1_bytes, dtype="<f4").copy().reshape(shape)
    aligned_tensor = (share0_tensor + share1_tensor).astype(np.float32)
    aligned_rgb = aligned_tensor * std + mean

    normalized_tensor_sha256 = hashlib.sha256(pack_float32_le(tensor)).hexdigest()
    share0_sha256 = hashlib.sha256(share0_bytes).hexdigest()
    share1_sha256 = hashlib.sha256(share1_bytes).hexdigest()
    source_image_sha256 = hashlib.sha256(f"showcase-synthetic-source:{tensor_seed}".encode("utf-8")).hexdigest()
    audit_nonce = nonce or str(uuid.uuid4())
    audit_chain_sha256 = hashlib.sha256(
        (
            f"{SHOWCASE_REQUEST_CONTRACT_VERSION}|{audit_nonce}|{source_image_sha256}|"
            f"{normalized_tensor_sha256}|{share0_sha256}|{share1_sha256}"
        ).encode("utf-8")
    ).hexdigest()

    max_image_dimension = int(config["max_image_dimension"])
    source_width = min(max_image_dimension, max(int(config["input_size"]) * 2, int(config["input_size"])))
    source_height = min(max_image_dimension, max(int(config["input_size"]) * 2, int(config["input_size"])))
    request_manifest = {
        "manifest_type": SHOWCASE_REQUEST_MANIFEST_TYPE,
        "contract_version": SHOWCASE_REQUEST_CONTRACT_VERSION,
        "bundle_dir": config["bundle"]["bundle_dir"],
        "input_size": int(config["input_size"]),
        "shape": list(config["shape"]),
        "dtype": "float32_le",
        "source_file_name": f"synthetic_{tensor_seed}.png",
        "source_mime": config["allowed_mime_types"][0],
        "source_size_bytes": 131072,
        "source_dimensions": {"width": source_width, "height": source_height},
        "audit_nonce": audit_nonce,
        "source_image_sha256": source_image_sha256,
        "normalized_tensor_sha256": normalized_tensor_sha256,
        "share0_sha256": share0_sha256,
        "share1_sha256": share1_sha256,
        "audit_chain_sha256": audit_chain_sha256,
    }
    if request_manifest_override:
        request_manifest.update(request_manifest_override)

    quality_assurance = {
        "status": "pass",
        "client_quality_summary": quality_override or compute_quality_summary(aligned_rgb),
    }
    audit_payload = {
        "hash_chain_version": SHOWCASE_REQUEST_CONTRACT_VERSION,
        "browser_generated_shares": True,
        "server_should_receive_plain_image": False,
        "server_should_receive_plain_pixel_values": False,
        "centralized_demo_reconstructs_normalized_tensor_for_dqa": True,
        "production_target_should_not_co_locate_both_shares": True,
    }
    if audit_override:
        audit_payload.update(audit_override)
    control_plane_metrics = metrics_override or default_client_metrics()

    return [
        {
            "name": "request_manifest",
            "content_type": "application/json",
            "body": json.dumps(request_manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        },
        {
            "name": "quality_assurance",
            "content_type": "application/json",
            "body": json.dumps(quality_assurance, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        },
        {
            "name": "audit",
            "content_type": "application/json",
            "body": json.dumps(audit_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        },
        {
            "name": "control_plane_metrics",
            "content_type": "application/json",
            "body": json.dumps(control_plane_metrics, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        },
        {
            "name": "share0",
            "filename": "share0.bin",
            "content_type": "application/octet-stream",
            "body": share0_bytes,
        },
        {
            "name": "share1",
            "filename": "share1.bin",
            "content_type": "application/octet-stream",
            "body": share1_bytes,
        },
    ]


def encode_multipart(parts: Iterable[dict], boundary: str) -> bytes:
    chunks = []
    for part in parts:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        disposition = f'form-data; name="{part["name"]}"'
        if part.get("filename"):
            disposition += f'; filename="{part["filename"]}"'
        chunks.append(f"Content-Disposition: {disposition}\r\n".encode("utf-8"))
        if part.get("content_type"):
            chunks.append(f'Content-Type: {part["content_type"]}\r\n'.encode("utf-8"))
        chunks.append(b"\r\n")
        chunks.append(part["body"])
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def post_multipart(base_url: str, path: str, body: bytes, boundary: str, timeout: float) -> HttpResult:
    scheme, host, port, base_path = parse_base_url(base_url)
    if scheme != "http":
        raise ValueError("validation helpers currently only support http URLs")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    request_path = (base_path + path) or path
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    connection.request("POST", request_path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    header_dict = {key.lower(): value for key, value in response.getheaders()}
    try:
        json_body = json.loads(payload.decode("utf-8")) if payload else None
    except Exception:
        json_body = None
    finally:
        connection.close()
    raw = (
        f"HTTP/1.1 {response.status} {response.reason}\r\n".encode("utf-8")
        + b"".join(f"{key}: {value}\r\n".encode("utf-8") for key, value in header_dict.items())
        + b"\r\n"
        + payload
    )
    return HttpResult(
        status=response.status,
        headers=header_dict,
        body=payload,
        json_body=json_body,
        raw=raw,
    )


def send_raw_http(base_url: str, raw_request: bytes, timeout: float) -> HttpResult:
    scheme, host, port, _ = parse_base_url(base_url)
    if scheme != "http":
        raise ValueError("validation helpers currently only support http URLs")
    chunks = []
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(raw_request)
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        while True:
            try:
                chunk = connection.recv(65536)
            except (ConnectionResetError, socket.timeout):
                break
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw.startswith(b"HTTP/"):
        return HttpResult(status=None, headers={}, body=b"", json_body=None, raw=raw)
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    parts = lines[0].split()
    status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
    headers = {}
    for line in lines[1:]:
        if b":" not in line:
            continue
        key, value = line.split(b":", 1)
        headers[key.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    try:
        json_body = json.loads(body.decode("utf-8")) if body else None
    except Exception:
        json_body = None
    return HttpResult(status=status, headers=headers, body=body, json_body=json_body, raw=raw)


def sample_process_state(pid: Optional[int]) -> ProcessState:
    now = time.time()
    if not pid:
        return ProcessState(False, None, now, None, None, None, None, None)
    proc_root = Path("/proc") / str(pid)
    if not proc_root.exists():
        return ProcessState(False, pid, now, None, None, None, None, None)

    fd_count = None
    socket_fd_count = None
    try:
        fd_entries = list((proc_root / "fd").iterdir())
        fd_count = len(fd_entries)
        socket_fd_count = 0
        for entry in fd_entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith("socket:"):
                socket_fd_count += 1
    except OSError:
        pass

    rss_kib = None
    vmhwm_kib = None
    thread_count = None
    try:
        for line in (proc_root / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                vmhwm_kib = int(line.split()[1])
            elif line.startswith("Threads:"):
                thread_count = int(line.split()[1])
    except OSError:
        pass

    return ProcessState(True, pid, now, fd_count, socket_fd_count, rss_kib, vmhwm_kib, thread_count)


def describe_process_state_delta(before: ProcessState, after: ProcessState) -> dict:
    if not before.available or not after.available:
        return {
            "available": False,
            "stable": None,
            "summary": "server pid unavailable; resource-state observation skipped",
            "before": before.to_dict(),
            "after": after.to_dict(),
        }

    def delta(field: str) -> Optional[int]:
        before_value = getattr(before, field)
        after_value = getattr(after, field)
        if before_value is None or after_value is None:
            return None
        return int(after_value - before_value)

    fd_delta = delta("fd_count")
    socket_delta = delta("socket_fd_count")
    rss_delta = delta("rss_kib")
    thread_delta = delta("thread_count")
    stable = (
        (fd_delta is None or fd_delta <= 0)
        and (socket_delta is None or socket_delta <= 0)
        and (rss_delta is None or rss_delta <= 1024)
        and (thread_delta is None or thread_delta <= 0)
    )
    summary_bits = []
    if fd_delta is not None:
        summary_bits.append(f"ΔFD={fd_delta:+d}")
    if socket_delta is not None:
        summary_bits.append(f"ΔSock={socket_delta:+d}")
    if rss_delta is not None:
        summary_bits.append(f"ΔRSS={rss_delta:+d} KiB")
    if thread_delta is not None:
        summary_bits.append(f"ΔThr={thread_delta:+d}")
    if not summary_bits:
        summary_bits.append("no comparable resource counters")

    return {
        "available": True,
        "stable": stable,
        "summary": "；".join(summary_bits),
        "before": before.to_dict(),
        "after": after.to_dict(),
        "delta": {
            "fd_count": fd_delta,
            "socket_fd_count": socket_delta,
            "rss_kib": rss_delta,
            "thread_count": thread_delta,
        },
    }
