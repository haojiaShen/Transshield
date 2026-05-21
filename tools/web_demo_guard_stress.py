#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.web_demo_validation_common import (
    build_medical_parts,
    describe_process_state_delta,
    encode_multipart,
    post_multipart,
    sample_process_state,
)


@dataclass
class GuardCheck:
    name: str
    passed: bool
    interception_layer: str
    fallback_layer: str
    summary: str
    details: dict
    system_state: dict


def run_medical_request(base_url: str, timeout: float, *, tensor_seed: int, share_seed: int, nonce: str):
    boundary = f'guard-{uuid.uuid4().hex}'
    parts = build_medical_parts(tensor_seed=tensor_seed, share_seed=share_seed, nonce=nonce)
    body = encode_multipart(parts, boundary)
    result = post_multipart(base_url, '/api/e2e/analyze_private_shares', body, boundary, timeout)
    return {
        'status': result.status,
        'json': result.json_body,
        'error_code': (result.json_body or {}).get('error_code') if isinstance(result.json_body, dict) else None,
    }


def run_empty_request(base_url: str, timeout: float):
    boundary = f'empty-{uuid.uuid4().hex}'
    result = post_multipart(base_url, '/api/e2e/analyze_private_shares', b'', boundary, timeout)
    return {
        'status': result.status,
        'json': result.json_body,
        'error_code': (result.json_body or {}).get('error_code') if isinstance(result.json_body, dict) else None,
    }


def capture_guard_check(
    *,
    server_pid: int | None,
    name: str,
    interception_layer: str,
    fallback_layer: str,
    summary: str,
    action: Callable[[], tuple[bool, dict]],
    settle_sec: float = 0.35,
) -> GuardCheck:
    before = sample_process_state(server_pid)
    passed, details = action()
    time.sleep(settle_sec)
    after = sample_process_state(server_pid)
    return GuardCheck(
        name=name,
        passed=passed,
        interception_layer=interception_layer,
        fallback_layer=fallback_layer,
        summary=summary,
        details=details,
        system_state=describe_process_state_delta(before, after),
    )


def check_duplicate_nonce(base_url: str, timeout: float, concurrency: int, server_pid: int | None) -> GuardCheck:
    def action():
        shared_nonce = str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    run_medical_request,
                    base_url,
                    timeout,
                    tensor_seed=11,
                    share_seed=11,
                    nonce=shared_nonce,
                )
                for _ in range(concurrency)
            ]
        results = [future.result() for future in futures]
        success = sum(item['status'] == 200 for item in results)
        duplicate = sum(item['error_code'] == 'duplicate_nonce' for item in results)
        passed = success == 1 and duplicate >= concurrency - 1
        return passed, {'results': results, 'success': success, 'duplicate_nonce': duplicate}

    return capture_guard_check(
        server_pid=server_pid,
        name='duplicate_nonce_concurrent',
        interception_layer='replay_guard_nonce_cache',
        fallback_layer='payload_fingerprint_guard',
        summary='One request should pass; concurrent requests with the same nonce should be rejected as duplicate_nonce.',
        action=action,
    )


def check_duplicate_payload(base_url: str, timeout: float, server_pid: int | None) -> GuardCheck:
    def action():
        first = run_medical_request(base_url, timeout, tensor_seed=21, share_seed=21, nonce=str(uuid.uuid4()))
        second = run_medical_request(base_url, timeout, tensor_seed=21, share_seed=21, nonce=str(uuid.uuid4()))
        passed = first['status'] == 200 and second['error_code'] == 'duplicate_payload'
        return passed, {'first': first, 'second': second}

    return capture_guard_check(
        server_pid=server_pid,
        name='duplicate_payload_different_nonce',
        interception_layer='replay_guard_payload_cache',
        fallback_layer='ip_inflight_limit',
        summary='Same share payload with a fresh nonce should still be rejected as duplicate_payload.',
        action=action,
    )


def check_inflight_limit(base_url: str, timeout: float, concurrency: int, server_pid: int | None) -> GuardCheck:
    def action():
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    run_medical_request,
                    base_url,
                    timeout,
                    tensor_seed=31 + index,
                    share_seed=131 + index,
                    nonce=str(uuid.uuid4()),
                )
                for index in range(concurrency)
            ]
        results = [future.result() for future in futures]
        busy = sum(item['error_code'] == 'busy_retry_later' for item in results)
        success = sum(item['status'] == 200 for item in results)
        passed = success <= 2 and busy >= max(1, concurrency - 2)
        return passed, {'results': results, 'success': success, 'busy_retry_later': busy}

    return capture_guard_check(
        server_pid=server_pid,
        name='per_ip_inflight_limit',
        interception_layer='per_ip_inflight_guard',
        fallback_layer='global_inflight_guard',
        summary='At most two requests from the same IP should enter the accepted path at once.',
        action=action,
        settle_sec=0.60,
    )


def check_rate_limit(base_url: str, timeout: float, count: int, server_pid: int | None) -> GuardCheck:
    def action():
        results = [run_empty_request(base_url, timeout) for _ in range(count)]
        limited = sum(item['error_code'] == 'rate_limited_ip' for item in results)
        passed = limited >= 1
        return passed, {'results': results, 'rate_limited_ip': limited}

    return capture_guard_check(
        server_pid=server_pid,
        name='ip_window_rate_limit',
        interception_layer='ip_sliding_window_guard',
        fallback_layer='not_applicable',
        summary='After a short burst from the same IP, the endpoint should return rate_limited_ip before body parsing.',
        action=action,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Black-box concurrency/replay guard checks for the web demo endpoint. '
            'Recommended server env for deterministic runs: WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC=1.5'
        )
    )
    parser.add_argument('--base-url', default='http://127.0.0.1:7860')
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument(
        '--checks',
        default='duplicate_nonce,duplicate_payload,inflight,rate_limit',
        help='Comma-separated subset: duplicate_nonce, duplicate_payload, inflight, rate_limit',
    )
    parser.add_argument('--replay-concurrency', type=int, default=4)
    parser.add_argument('--inflight-concurrency', type=int, default=4)
    parser.add_argument('--rate-limit-count', type=int, default=8)
    parser.add_argument('--server-pid', type=int, default=0, help='Optional local server pid for FD/RSS/socket state sampling.')
    parser.add_argument('--out', default='', help='Optional JSON output path.')
    parser.add_argument(
        '--window-reset-sec',
        type=float,
        default=61.0,
        help='Sleep between checks to avoid cross-test contamination from the 60s per-IP rate-limit window.',
    )
    args = parser.parse_args()

    requested = [item.strip() for item in args.checks.split(',') if item.strip()]
    registry = {
        'duplicate_nonce': lambda: check_duplicate_nonce(args.base_url, args.timeout, args.replay_concurrency, args.server_pid or None),
        'duplicate_payload': lambda: check_duplicate_payload(args.base_url, args.timeout, args.server_pid or None),
        'inflight': lambda: check_inflight_limit(args.base_url, args.timeout, args.inflight_concurrency, args.server_pid or None),
        'rate_limit': lambda: check_rate_limit(args.base_url, args.timeout, args.rate_limit_count, args.server_pid or None),
    }
    unknown = [item for item in requested if item not in registry]
    if unknown:
        raise SystemExit(f'unknown checks: {", ".join(unknown)}')

    checks = []
    for index, name in enumerate(requested):
        checks.append(registry[name]())
        if index != len(requested) - 1 and args.window_reset_sec > 0:
            time.sleep(args.window_reset_sec)

    payload = {
        'base_url': args.base_url,
        'server_pid': args.server_pid or None,
        'passed': all(item.passed for item in checks),
        'checks': [asdict(item) for item in checks],
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
