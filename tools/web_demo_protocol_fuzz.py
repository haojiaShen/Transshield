#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.web_demo_validation_common import (
    HttpResult,
    build_medical_parts,
    describe_process_state_delta,
    encode_multipart,
    parse_base_url,
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


def request_path(base_url: str, endpoint_path: str) -> str:
    _, _, _, base_path = parse_base_url(base_url)
    return (base_path + endpoint_path) or endpoint_path


def make_case_result(
    *,
    name: str,
    response: HttpResult,
    passed: bool,
    interception_layer: str,
    fallback_layer: str,
    details: str,
    system_state: dict,
) -> CaseResult:
    error_code = None
    if isinstance(response.json_body, dict):
        error_code = response.json_body.get('error_code')
    return CaseResult(
        name=name,
        passed=passed,
        status=response.status,
        error_code=error_code,
        interception_layer=interception_layer,
        fallback_layer=fallback_layer,
        details=details,
        system_state=system_state,
    )


def capture_case(
    *,
    name: str,
    server_pid: int | None,
    action: Callable[[], HttpResult],
    passed_when: Callable[[HttpResult], bool],
    interception_layer: str,
    fallback_layer: str,
    details: str,
    settle_sec: float = 0.20,
) -> CaseResult:
    before = sample_process_state(server_pid)
    response = action()
    time.sleep(settle_sec)
    after = sample_process_state(server_pid)
    system_state = describe_process_state_delta(before, after)
    return make_case_result(
        name=name,
        response=response,
        passed=passed_when(response),
        interception_layer=interception_layer,
        fallback_layer=fallback_layer,
        details=details,
        system_state=system_state,
    )


def build_raw_multipart_request(
    base_url: str,
    *,
    boundary_param: str,
    body: bytes,
    transfer_encoding: str | None = None,
    content_length: int | None = None,
) -> bytes:
    path = request_path(base_url, '/api/e2e/analyze_private_shares')
    headers = [
        f'POST {path} HTTP/1.1',
        'Host: 127.0.0.1',
        f'Content-Type: multipart/form-data; {boundary_param}',
        'Connection: close',
    ]
    if transfer_encoding:
        headers.append(f'Transfer-Encoding: {transfer_encoding}')
    elif content_length is not None:
        headers.append(f'Content-Length: {content_length}')
    else:
        headers.append(f'Content-Length: {len(body)}')
    return ('\r\n'.join(headers) + '\r\n\r\n').encode('utf-8') + body


def build_custom_body(boundary: str, header_bytes: bytes, body_bytes: bytes, *, epilogue: bytes = b'') -> bytes:
    marker = boundary.encode('utf-8')
    return b''.join(
        [
            b'--' + marker + b'\r\n',
            header_bytes,
            b'\r\n\r\n',
            body_bytes,
            b'\r\n',
            b'--' + marker + b'--\r\n',
            epilogue,
        ]
    )


def case_transfer_encoding_chunked(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    raw = build_raw_multipart_request(
        base_url,
        boundary_param='boundary=chunked-test',
        body=b'5\r\nhello\r\n0\r\n\r\n',
        transfer_encoding='chunked',
    )
    return capture_case(
        name='transfer_encoding_chunked_blocked',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'transfer_encoding_not_supported',
        interception_layer='http_request_body_gate',
        fallback_layer='not_applicable',
        details='Expect Transfer-Encoding: chunked to be rejected before any body read.',
    )


def case_content_length_oversize(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    raw = build_raw_multipart_request(
        base_url,
        boundary_param='boundary=oversize-test',
        body=b'',
        content_length=5 * 1024 * 1024 + 1,
    )
    return capture_case(
        name='oversized_content_length_best_effort_413',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 413,
        interception_layer='content_length_header_gate',
        fallback_layer='tcp_force_close_allowed',
        details='Expect best-effort 413 before the server attempts to drain an oversized body.',
    )


def case_duplicate_domain(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'dupe-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=1, share_seed=101)
    body = encode_multipart([{'name': 'domain', 'body': b'medical'}, *parts], boundary)
    return capture_case(
        name='duplicate_field_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='raw_multipart_precheck',
        fallback_layer='mime_tree_validation',
        details='Expect duplicate multipart field rejection before JSON parsing.',
    )


def case_boundary_fanout(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'fanout-{uuid.uuid4().hex}'
    parts = [{'name': 'domain', 'body': b'medical'}]
    for index in range(13):
        parts.append({'name': f'x{index}', 'body': b'1'})
    body = encode_multipart(parts, boundary)
    return capture_case(
        name='boundary_fanout_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='raw_multipart_precheck',
        fallback_layer='mime_tree_validation',
        details='Expect multipart fanout / extra fields rejection before MIME tree expansion becomes expensive.',
    )


def case_nested_multipart(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'nested-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=2, share_seed=202)
    patched = []
    for part in parts:
        if part['name'] == 'share0':
            patched.append(
                {
                    'name': 'share0',
                    'filename': 'share0.bin',
                    'content_type': 'multipart/mixed',
                    'body': b'--inner\r\nContent-Type: text/plain\r\n\r\nnested\r\n--inner--\r\n',
                }
            )
        else:
            patched.append(part)
    body = encode_multipart(patched, boundary)
    return capture_case(
        name='nested_multipart_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='raw_multipart_precheck',
        fallback_layer='mime_tree_validation',
        details='Expect nested multipart parts to be rejected during raw precheck.',
    )


def case_oversized_json_part(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'json-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=3, share_seed=303)
    oversized = json.dumps({'x': 'A' * 5000}, separators=(',', ':')).encode('utf-8')
    patched = []
    for part in parts:
        if part['name'] == 'client_quality_summary':
            patched.append(
                {
                    'name': 'client_quality_summary',
                    'content_type': 'application/json',
                    'body': oversized,
                }
            )
        else:
            patched.append(part)
    body = encode_multipart(patched, boundary)
    return capture_case(
        name='oversized_json_part_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'invalid_control_plane_payload',
        interception_layer='json_bytes_gate',
        fallback_layer='strict_json_decoder',
        details='Expect JSON byte gate to reject the part before json.loads touches oversized payloads.',
    )


def case_extra_field_set(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'rogue-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=4, share_seed=404)
    body = encode_multipart([*parts, {'name': 'rogue', 'body': b'1'}], boundary)
    return capture_case(
        name='unexpected_field_set_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='raw_multipart_precheck',
        fallback_layer='exact_field_set_gate',
        details='Expect exact medical field-set enforcement to reject rogue fields.',
    )


def case_boundary_param_whitespace(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'whitespace-{uuid.uuid4().hex}'
    body = build_custom_body(boundary, b'Content-Disposition: form-data; name="domain"', b'medical')
    raw = build_raw_multipart_request(
        base_url,
        boundary_param=f'boundary = "{boundary}"',
        body=body,
    )
    return capture_case(
        name='boundary_param_whitespace_rejected',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='content_type_boundary_parser',
        fallback_layer='raw_multipart_precheck',
        details='Expect non-canonical boundary parameter formatting to be rejected before body traversal.',
    )


def case_malformed_part_header(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'header-{uuid.uuid4().hex}'
    body = build_custom_body(boundary, b'Broken-Header-Line', b'medical')
    raw = build_raw_multipart_request(base_url, boundary_param=f'boundary={boundary}', body=body)
    return capture_case(
        name='malformed_part_header_blocked',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='multipart_header_parser',
        fallback_layer='mime_tree_validation',
        details='Expect header lines without colon separators to be rejected during raw multipart precheck.',
    )


def case_header_null_byte(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'nullbyte-{uuid.uuid4().hex}'
    header = b'Content-Disposition: form-data; name="domai\x00n"'
    body = build_custom_body(boundary, header, b'medical')
    raw = build_raw_multipart_request(base_url, boundary_param=f'boundary={boundary}', body=body)
    return capture_case(
        name='multipart_header_null_byte_blocked',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='multipart_header_parser',
        fallback_layer='mime_tree_validation',
        details='Expect control characters inside multipart headers to be rejected during precheck.',
    )


def case_non_empty_epilogue(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'epilogue-{uuid.uuid4().hex}'
    body = build_custom_body(
        boundary,
        b'Content-Disposition: form-data; name="domain"',
        b'medical',
        epilogue=b'extra-junk-after-closing-boundary',
    )
    raw = build_raw_multipart_request(base_url, boundary_param=f'boundary={boundary}', body=body)
    return capture_case(
        name='non_empty_epilogue_blocked',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'malformed_multipart_precheck_failed',
        interception_layer='raw_multipart_precheck',
        fallback_layer='mime_tree_validation',
        details='Expect multipart epilogue bytes after the closing boundary to be rejected.',
    )


def case_utf16_json_charset(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'utf16-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=5, share_seed=505)
    patched = []
    for part in parts:
        if part['name'] == 'client_quality_summary':
            payload = json.dumps({'mean_luma': 0.5}, ensure_ascii=False).encode('utf-16le')
            patched.append(
                {
                    'name': 'client_quality_summary',
                    'content_type': 'application/json; charset=utf-16le',
                    'body': payload,
                }
            )
        else:
            patched.append(part)
    body = encode_multipart(patched, boundary)
    return capture_case(
        name='utf16_json_charset_blocked',
        server_pid=server_pid,
        action=lambda: post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'invalid_control_plane_payload',
        interception_layer='strict_utf8_json_decoder',
        fallback_layer='json_numeric_hooks',
        details='Expect non-UTF-8 JSON text to be rejected before semantic validation or tensor reconstruction.',
    )


def case_truncated_body(base_url: str, timeout: float, server_pid: int | None) -> CaseResult:
    boundary = f'truncated-{uuid.uuid4().hex}'
    body = b'--' + boundary.encode('utf-8') + b'\r\nContent-Disposition: form-data; name="domain"\r\n\r\nmed'
    raw = build_raw_multipart_request(
        base_url,
        boundary_param=f'boundary={boundary}',
        body=body,
        content_length=len(body) + 32,
    )
    return capture_case(
        name='truncated_body_blocked',
        server_pid=server_pid,
        action=lambda: send_raw_http(base_url, raw, timeout),
        passed_when=lambda response: response.status == 400 and (response.json_body or {}).get('error_code') == 'truncated_body',
        interception_layer='streaming_body_reader',
        fallback_layer='not_applicable',
        details='Expect early client disconnect / truncated body to be rejected before multipart parsing begins.',
        settle_sec=0.35,
    )


def iter_cases(base_url: str, timeout: float, server_pid: int | None) -> Iterable[Callable[[], CaseResult]]:
    return {
        'transfer_encoding_chunked': lambda: case_transfer_encoding_chunked(base_url, timeout, server_pid),
        'content_length_oversize': lambda: case_content_length_oversize(base_url, timeout, server_pid),
        'duplicate_domain': lambda: case_duplicate_domain(base_url, timeout, server_pid),
        'boundary_fanout': lambda: case_boundary_fanout(base_url, timeout, server_pid),
        'nested_multipart': lambda: case_nested_multipart(base_url, timeout, server_pid),
        'oversized_json_part': lambda: case_oversized_json_part(base_url, timeout, server_pid),
        'extra_field_set': lambda: case_extra_field_set(base_url, timeout, server_pid),
        'boundary_param_whitespace': lambda: case_boundary_param_whitespace(base_url, timeout, server_pid),
        'malformed_part_header': lambda: case_malformed_part_header(base_url, timeout, server_pid),
        'header_null_byte': lambda: case_header_null_byte(base_url, timeout, server_pid),
        'non_empty_epilogue': lambda: case_non_empty_epilogue(base_url, timeout, server_pid),
        'utf16_json_charset': lambda: case_utf16_json_charset(base_url, timeout, server_pid),
        'truncated_body': lambda: case_truncated_body(base_url, timeout, server_pid),
    }


def main():
    parser = argparse.ArgumentParser(description='Black-box protocol fuzz checks for the web demo medical endpoint.')
    parser.add_argument('--base-url', default='http://127.0.0.1:7860')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--server-pid', type=int, default=0, help='Optional local server pid for FD/RSS/socket state sampling.')
    parser.add_argument('--out', default='', help='Optional JSON output path.')
    parser.add_argument('--case-reset-sec', type=float, default=0.0, help='Optional sleep between cases to avoid IP-window cross-contamination.')
    parser.add_argument('--cases', default='', help='Optional comma-separated subset of case keys.')
    args = parser.parse_args()

    results = []
    registry = iter_cases(args.base_url, args.timeout, args.server_pid or None)
    if args.cases:
        requested = [item.strip() for item in args.cases.split(',') if item.strip()]
    else:
        requested = list(registry.keys())
    unknown = [item for item in requested if item not in registry]
    if unknown:
        raise SystemExit(f'unknown cases: {", ".join(unknown)}')
    for index, name in enumerate(requested):
        results.append(registry[name]())
        if index != len(requested) - 1 and args.case_reset_sec > 0:
            time.sleep(args.case_reset_sec)
    payload = {
        'base_url': args.base_url,
        'server_pid': args.server_pid or None,
        'passed': all(item.passed for item in results),
        'results': [asdict(item) for item in results],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    if not payload['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
