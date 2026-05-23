#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.showcase_validation_common import (
    HttpResult,
    build_medical_parts,
    describe_process_state_delta,
    encode_multipart,
    load_remote_medical_config,
    post_multipart,
    sample_process_state,
    send_raw_http,
)


@dataclass
class CaseResult:
    name: str
    passed: bool
    status: int | None
    error_code: str | None
    interception_layer: str
    details: str
    system_state: dict


def build_raw_request(path: str, boundary_param: str, body: bytes, *, content_length: int | None = None) -> bytes:
    headers = [
        f"POST {path} HTTP/1.1",
        "Host: 127.0.0.1",
        f"Content-Type: multipart/form-data; {boundary_param}",
        "Connection: close",
        f"Content-Length: {content_length if content_length is not None else len(body)}",
    ]
    return ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body


def build_custom_body(boundary: str, header_bytes: bytes, body_bytes: bytes, *, close: bool = True) -> bytes:
    marker = boundary.encode("utf-8")
    chunks = [b"--" + marker + b"\r\n", header_bytes, b"\r\n\r\n", body_bytes, b"\r\n"]
    if close:
        chunks.append(b"--" + marker + b"--\r\n")
    return b"".join(chunks)


def capture_case(
    *,
    name: str,
    server_pid: int | None,
    action: Callable[[], HttpResult],
    passed_when: Callable[[HttpResult], bool],
    interception_layer: str,
    details: str,
    settle_sec: float = 0.2,
) -> CaseResult:
    before = sample_process_state(server_pid)
    response = action()
    time.sleep(settle_sec)
    after = sample_process_state(server_pid)
    error_code = None
    if isinstance(response.json_body, dict):
        error_code = response.json_body.get("error_code")
    return CaseResult(
        name=name,
        passed=passed_when(response),
        status=response.status,
        error_code=error_code,
        interception_layer=interception_layer,
        details=details,
        system_state=describe_process_state_delta(before, after),
    )


def case_baseline_accepted(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"baseline-{uuid.uuid4().hex}"
    body = encode_multipart(build_medical_parts(config, tensor_seed=1, share_seed=101), boundary)
    return capture_case(
        name="baseline_valid_request",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 200 and (response.json_body or {}).get("status") == "completed",
        interception_layer="accepted_path",
        details="Valid request should pass end-to-end and return completed.",
    )


def case_duplicate_field(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"dupe-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=2, share_seed=202)
    body = encode_multipart([parts[0], *parts], boundary)
    return capture_case(
        name="duplicate_field_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        details="Duplicate multipart field must be rejected before JSON parsing.",
    )


def case_extra_field(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"extra-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=3, share_seed=303)
    body = encode_multipart([*parts, {"name": "rogue", "body": b"1"}], boundary)
    return capture_case(
        name="extra_field_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        details="Unexpected multipart field set must be rejected.",
    )


def case_nested_multipart(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"nested-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=4, share_seed=404)
    patched = []
    for part in parts:
        if part["name"] == "share0":
            patched.append(
                {
                    "name": "share0",
                    "filename": "share0.bin",
                    "content_type": "multipart/mixed",
                    "body": b"--inner\r\nContent-Type: text/plain\r\n\r\nnested\r\n--inner--\r\n",
                }
            )
        else:
            patched.append(part)
    body = encode_multipart(patched, boundary)
    return capture_case(
        name="nested_multipart_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        details="Nested multipart payloads must be rejected.",
    )


def case_invalid_header(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"header-{uuid.uuid4().hex}"
    body = build_custom_body(boundary, b"Broken-Header-Line", b"x")
    raw = build_raw_request("/api/medical/live-run", f"boundary={boundary}", body)
    return capture_case(
        name="invalid_header_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        details="Malformed multipart part headers must be rejected.",
    )


def case_truncated_body(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"trunc-{uuid.uuid4().hex}"
    full_body = encode_multipart(build_medical_parts(config, tensor_seed=5, share_seed=505), boundary)
    truncated = full_body[:-18]
    return capture_case(
        name="truncated_body_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", truncated, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        details="Truncated multipart body must be rejected before tensor parsing.",
    )


def case_share_hash_mismatch(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"hash-{uuid.uuid4().hex}"
    parts = build_medical_parts(
        config,
        tensor_seed=6,
        share_seed=606,
        request_manifest_override={"share0_sha256": "0" * 64},
    )
    body = encode_multipart(parts, boundary)
    return capture_case(
        name="share_hash_mismatch_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 422
        and (response.json_body or {}).get("error_code") == "audit_share_hash_mismatch",
        interception_layer="share_hash_gate",
        details="Client-declared share hash must match server-observed bytes.",
    )


def main():
    parser = argparse.ArgumentParser(description="Black-box protocol and multipart fuzz checks for the showcase live demo endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--cases",
        default="duplicate_field,extra_field,nested_multipart,invalid_header,truncated_body,share_hash_mismatch",
        help="Comma-separated subset: baseline, duplicate_field, extra_field, nested_multipart, invalid_header, truncated_body, share_hash_mismatch",
    )
    parser.add_argument("--server-pid", type=int, default=0, help="Optional local server pid for resource-state sampling.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_remote_medical_config(args.base_url, args.timeout)
    registry = {
        "baseline": lambda: case_baseline_accepted(args.base_url, args.timeout, args.server_pid or None, config),
        "duplicate_field": lambda: case_duplicate_field(args.base_url, args.timeout, args.server_pid or None, config),
        "extra_field": lambda: case_extra_field(args.base_url, args.timeout, args.server_pid or None, config),
        "nested_multipart": lambda: case_nested_multipart(args.base_url, args.timeout, args.server_pid or None, config),
        "invalid_header": lambda: case_invalid_header(args.base_url, args.timeout, args.server_pid or None),
        "truncated_body": lambda: case_truncated_body(args.base_url, args.timeout, args.server_pid or None, config),
        "share_hash_mismatch": lambda: case_share_hash_mismatch(args.base_url, args.timeout, args.server_pid or None, config),
    }
    requested = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = [item for item in requested if item not in registry]
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)}")

    cases = [registry[name]() for name in requested]
    payload = {
        "base_url": args.base_url,
        "server_pid": args.server_pid or None,
        "passed": all(item.passed for item in cases),
        "cases": [asdict(item) for item in cases],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
