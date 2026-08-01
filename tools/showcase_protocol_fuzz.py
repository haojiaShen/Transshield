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
    fallback_layer: str
    details: str
    system_state: dict


def build_raw_request(
    path: str,
    boundary_param: str,
    body: bytes,
    *,
    content_length: int | None = None,
    transfer_encoding: str | None = None,
) -> bytes:
    headers = [
        f"POST {path} HTTP/1.1",
        "Host: 127.0.0.1",
        f"Content-Type: multipart/form-data; {boundary_param}",
        "Connection: close",
    ]
    if transfer_encoding:
        headers.append(f"Transfer-Encoding: {transfer_encoding}")
    else:
        headers.append(f"Content-Length: {content_length if content_length is not None else len(body)}")
    return ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body


def build_custom_body(
    boundary: str,
    header_bytes: bytes,
    body_bytes: bytes,
    *,
    close: bool = True,
    epilogue: bytes = b"",
) -> bytes:
    marker = boundary.encode("utf-8")
    chunks = [b"--" + marker + b"\r\n", header_bytes, b"\r\n\r\n", body_bytes, b"\r\n"]
    if close:
        chunks.append(b"--" + marker + b"--\r\n")
        chunks.append(epilogue)
    return b"".join(chunks)


def capture_case(
    *,
    name: str,
    server_pid: int | None,
    action: Callable[[], HttpResult],
    passed_when: Callable[[HttpResult], bool],
    interception_layer: str,
    details: str,
    fallback_layer: str = "not_applicable",
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
        fallback_layer=fallback_layer,
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


def case_transfer_encoding_chunked(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    raw = build_raw_request(
        "/api/medical/live-run",
        "boundary=chunked-test",
        b"5\r\nhello\r\n0\r\n\r\n",
        transfer_encoding="chunked",
    )
    return capture_case(
        name="transfer_encoding_chunked_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "transfer_encoding_not_supported",
        interception_layer="http_request_body_gate",
        details="Transfer-Encoding: chunked must be rejected before body parsing.",
    )


def case_content_length_oversize(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    raw = build_raw_request(
        "/api/medical/live-run",
        "boundary=oversize-test",
        b"",
        content_length=5 * 1024 * 1024 + 1,
    )
    return capture_case(
        name="oversized_content_length_best_effort_413",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 413
        and (response.json_body or {}).get("error_code") == "payload_too_large",
        interception_layer="content_length_header_gate",
        fallback_layer="tcp_force_close_allowed",
        details="Oversized declared body must be rejected before the server drains it.",
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
        fallback_layer="mime_tree_validation",
        details="Duplicate multipart field must be rejected before JSON parsing.",
    )


def case_boundary_fanout(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"fanout-{uuid.uuid4().hex}"
    parts = [{"name": f"x{index}", "body": b"1"} for index in range(13)]
    body = encode_multipart(parts, boundary)
    return capture_case(
        name="boundary_fanout_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        fallback_layer="mime_tree_validation",
        details="Multipart boundary fanout must be rejected before MIME expansion becomes expensive.",
    )


def case_extra_field(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"extra-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=3, share_seed=303)
    body = encode_multipart([*parts, {"name": "rogue", "body": b"1"}], boundary)
    return capture_case(
        name="unexpected_field_set_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        fallback_layer="exact_field_set_gate",
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
        fallback_layer="mime_tree_validation",
        details="Nested multipart payloads must be rejected.",
    )


def case_oversized_json_part(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"json-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=5, share_seed=505)
    oversized = json.dumps({"x": "A" * 5000}, separators=(",", ":")).encode("utf-8")
    patched = [
        {**part, "body": oversized} if part["name"] == "quality_assurance" else part
        for part in parts
    ]
    body = encode_multipart(patched, boundary)
    return capture_case(
        name="oversized_json_part_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "invalid_control_plane_payload",
        interception_layer="json_bytes_gate",
        fallback_layer="strict_json_decoder",
        details="Per-field JSON byte limits must reject oversized JSON before decoding.",
    )


def case_boundary_param_whitespace(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"whitespace-{uuid.uuid4().hex}"
    body = build_custom_body(boundary, b'Content-Disposition: form-data; name="request_manifest"', b"{}")
    raw = build_raw_request(
        "/api/medical/live-run",
        f'boundary = "{boundary}"',
        body,
    )
    return capture_case(
        name="boundary_param_whitespace_rejected",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="content_type_boundary_parser",
        fallback_layer="raw_multipart_precheck",
        details="Non-canonical boundary parameter formatting must be rejected.",
    )


def case_invalid_header(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"header-{uuid.uuid4().hex}"
    body = build_custom_body(boundary, b"Broken-Header-Line", b"x")
    raw = build_raw_request("/api/medical/live-run", f"boundary={boundary}", body)
    return capture_case(
        name="malformed_part_header_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="multipart_header_parser",
        fallback_layer="mime_tree_validation",
        details="Malformed multipart part headers must be rejected.",
    )


def case_header_null_byte(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"nullbyte-{uuid.uuid4().hex}"
    body = build_custom_body(
        boundary,
        b'Content-Disposition: form-data; name="request_manif\x00est"',
        b"{}",
    )
    raw = build_raw_request("/api/medical/live-run", f"boundary={boundary}", body)
    return capture_case(
        name="multipart_header_null_byte_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="multipart_header_parser",
        fallback_layer="mime_tree_validation",
        details="Control characters inside multipart headers must be rejected.",
    )


def case_non_empty_epilogue(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f"epilogue-{uuid.uuid4().hex}"
    body = build_custom_body(
        boundary,
        b'Content-Disposition: form-data; name="request_manifest"',
        b"{}",
        epilogue=b"extra-junk-after-closing-boundary",
    )
    raw = build_raw_request("/api/medical/live-run", f"boundary={boundary}", body)
    return capture_case(
        name="non_empty_epilogue_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "malformed_multipart_precheck_failed",
        interception_layer="raw_multipart_precheck",
        fallback_layer="mime_tree_validation",
        details="Bytes after the closing multipart boundary must be rejected.",
    )


def case_utf16_json_charset(base_url: str, timeout: float, server_pid: int | None, config: dict) -> CaseResult:
    boundary = f"utf16-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=7, share_seed=707)
    patched = []
    for part in parts:
        if part["name"] == "quality_assurance":
            patched.append(
                {
                    **part,
                    "content_type": "application/json; charset=utf-16",
                    "body": json.dumps({"status": "pass"}).encode("utf-16"),
                }
            )
        else:
            patched.append(part)
    body = encode_multipart(patched, boundary)
    return capture_case(
        name="utf16_json_charset_blocked",
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout),
        passed_when=lambda response: response.status == 400
        and (response.json_body or {}).get("error_code") == "invalid_control_plane_payload",
        interception_layer="strict_utf8_json_decoder",
        fallback_layer="json_numeric_hooks",
        details="Non-UTF-8 JSON text must be rejected before semantic validation.",
    )


def case_truncated_body(
    base_url: str,
    timeout: float,
    server_pid: int | None,
    config: dict,
    audit_rejections_jsonl: Path | None,
) -> CaseResult:
    boundary = f"trunc-{uuid.uuid4().hex}"
    body = b"--" + boundary.encode("utf-8") + b'\r\nContent-Disposition: form-data; name="request_manifest"\r\n\r\n{"x":1}'
    audit_offset = (
        audit_rejections_jsonl.stat().st_size
        if audit_rejections_jsonl is not None and audit_rejections_jsonl.exists()
        else 0
    )

    def audit_confirms_server_rejection() -> bool:
        if audit_rejections_jsonl is None or not audit_rejections_jsonl.exists():
            return False
        with audit_rejections_jsonl.open("rb") as handle:
            handle.seek(audit_offset)
            new_records = handle.read().decode("utf-8", "replace").splitlines()
        for raw_record in new_records:
            try:
                record = json.loads(raw_record)
            except json.JSONDecodeError:
                continue
            if (
                record.get("error_code") == "truncated_body"
                and record.get("interception_layer") == "streaming_body_reader"
            ):
                return True
        return False

    raw = build_raw_request(
        "/api/medical/live-run",
        f"boundary={boundary}",
        body,
        content_length=len(body) + 32,
    )
    return capture_case(
        name="truncated_body_blocked",
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: (
            response.status == 400
            and (response.json_body or {}).get("error_code") == "truncated_body"
        )
        or (response.status is None and audit_confirms_server_rejection()),
        interception_layer="streaming_body_reader",
        fallback_layer="tcp_disconnect_no_response",
        details=(
            "Early disconnect must be rejected before multipart parsing. Depending on the ASGI server, "
            "the client may receive HTTP 400/truncated_body or no response after its own TCP half-close; "
            "the server rejection audit must be checked for streaming_body_reader/truncated_body."
        ),
        settle_sec=0.35,
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
        default=(
            "transfer_encoding_chunked,content_length_oversize,duplicate_field,boundary_fanout,"
            "nested_multipart,oversized_json_part,extra_field,boundary_param_whitespace,"
            "invalid_header,header_null_byte,non_empty_epilogue,utf16_json_charset,truncated_body"
        ),
        help=(
            "Comma-separated subset matching the 13-case report matrix. Additional checks: "
            "baseline, share_hash_mismatch."
        ),
    )
    parser.add_argument("--server-pid", type=int, default=0, help="Optional local server pid for resource-state sampling.")
    parser.add_argument(
        "--audit-rejections-jsonl",
        type=Path,
        default=None,
        help="Server rejection audit used to confirm truncated-body interception when TCP closes before HTTP 400 is delivered.",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_remote_medical_config(args.base_url, args.timeout)
    registry = {
        "transfer_encoding_chunked": lambda: case_transfer_encoding_chunked(args.base_url, args.timeout, args.server_pid or None),
        "content_length_oversize": lambda: case_content_length_oversize(args.base_url, args.timeout, args.server_pid or None),
        "baseline": lambda: case_baseline_accepted(args.base_url, args.timeout, args.server_pid or None, config),
        "duplicate_field": lambda: case_duplicate_field(args.base_url, args.timeout, args.server_pid or None, config),
        "boundary_fanout": lambda: case_boundary_fanout(args.base_url, args.timeout, args.server_pid or None),
        "extra_field": lambda: case_extra_field(args.base_url, args.timeout, args.server_pid or None, config),
        "nested_multipart": lambda: case_nested_multipart(args.base_url, args.timeout, args.server_pid or None, config),
        "oversized_json_part": lambda: case_oversized_json_part(args.base_url, args.timeout, args.server_pid or None, config),
        "boundary_param_whitespace": lambda: case_boundary_param_whitespace(args.base_url, args.timeout, args.server_pid or None),
        "invalid_header": lambda: case_invalid_header(args.base_url, args.timeout, args.server_pid or None),
        "header_null_byte": lambda: case_header_null_byte(args.base_url, args.timeout, args.server_pid or None),
        "non_empty_epilogue": lambda: case_non_empty_epilogue(args.base_url, args.timeout, args.server_pid or None),
        "utf16_json_charset": lambda: case_utf16_json_charset(args.base_url, args.timeout, args.server_pid or None, config),
        "truncated_body": lambda: case_truncated_body(
            args.base_url,
            args.timeout,
            args.server_pid or None,
            config,
            args.audit_rejections_jsonl,
        ),
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
