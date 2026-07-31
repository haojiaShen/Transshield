#!/usr/bin/env python3
from __future__ import annotations

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

from tools.showcase_validation_common import (
    build_medical_parts,
    describe_process_state_delta,
    encode_multipart,
    load_remote_medical_config,
    post_multipart,
    sample_process_state,
)


@dataclass
class GuardCheck:
    name: str
    passed: bool
    interception_layer: str
    summary: str
    details: dict
    system_state: dict


def run_medical_request(base_url: str, timeout: float, config: dict, *, tensor_seed: int, share_seed: int, nonce: str):
    boundary = f"guard-{uuid.uuid4().hex}"
    parts = build_medical_parts(config, tensor_seed=tensor_seed, share_seed=share_seed, nonce=nonce)
    body = encode_multipart(parts, boundary)
    result = post_multipart(base_url, "/api/medical/live-run", body, boundary, timeout)
    return {
        "status": result.status,
        "json": result.json_body,
        "error_code": (result.json_body or {}).get("error_code") if isinstance(result.json_body, dict) else None,
    }


def run_invalid_empty_request(base_url: str, timeout: float):
    boundary = f"empty-{uuid.uuid4().hex}"
    result = post_multipart(base_url, "/api/medical/live-run", b"", boundary, timeout)
    return {
        "status": result.status,
        "json": result.json_body,
        "error_code": (result.json_body or {}).get("error_code") if isinstance(result.json_body, dict) else None,
    }


def capture_guard_check(
    *,
    server_pid: int | None,
    name: str,
    interception_layer: str,
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
        summary=summary,
        details=details,
        system_state=describe_process_state_delta(before, after),
    )


def check_duplicate_nonce(base_url: str, timeout: float, config: dict, concurrency: int, server_pid: int | None) -> GuardCheck:
    def action():
        shared_nonce = str(uuid.uuid4())
        base_seed = uuid.uuid4().int % 1_000_000_000
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    run_medical_request,
                    base_url,
                    timeout,
                    config,
                    tensor_seed=base_seed,
                    share_seed=base_seed,
                    nonce=shared_nonce,
                )
                for _ in range(concurrency)
            ]
        results = [future.result() for future in futures]
        success = sum(item["status"] == 200 for item in results)
        duplicate = sum(item["error_code"] == "duplicate_nonce" for item in results)
        passed = success == 1 and duplicate >= concurrency - 1
        return passed, {"results": results, "success": success, "duplicate_nonce": duplicate}

    return capture_guard_check(
        server_pid=server_pid,
        name="duplicate_nonce_concurrent",
        interception_layer="replay_guard",
        summary="One request should pass; concurrent requests with the same nonce should be rejected as duplicate_nonce.",
        action=action,
    )


def check_duplicate_payload(base_url: str, timeout: float, config: dict, server_pid: int | None) -> GuardCheck:
    def action():
        base_seed = uuid.uuid4().int % 1_000_000_000
        first = run_medical_request(base_url, timeout, config, tensor_seed=base_seed, share_seed=base_seed, nonce=str(uuid.uuid4()))
        second = run_medical_request(base_url, timeout, config, tensor_seed=base_seed, share_seed=base_seed, nonce=str(uuid.uuid4()))
        passed = first["status"] == 200 and second["error_code"] == "duplicate_payload"
        return passed, {"first": first, "second": second}

    return capture_guard_check(
        server_pid=server_pid,
        name="duplicate_payload_different_nonce",
        interception_layer="replay_guard",
        summary="Same share payload with a fresh nonce should still be rejected as duplicate_payload.",
        action=action,
    )


def check_inflight_limit(base_url: str, timeout: float, config: dict, concurrency: int, server_pid: int | None) -> GuardCheck:
    def action():
        base_seed = uuid.uuid4().int % 1_000_000_000
        retry_nonce = str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=max(concurrency, 2)) as executor:
            primary = executor.submit(
                run_medical_request,
                base_url,
                timeout,
                config,
                tensor_seed=base_seed,
                share_seed=base_seed + 100,
                nonce=str(uuid.uuid4()),
            )
            time.sleep(0.25)
            challengers = [
                executor.submit(
                    run_medical_request,
                    base_url,
                    timeout,
                    config,
                    tensor_seed=base_seed + 1 + index,
                    share_seed=base_seed + 101 + index,
                    nonce=retry_nonce if index == 0 else str(uuid.uuid4()),
                )
                for index in range(max(1, concurrency - 1))
            ]
            results = [primary.result(), *[future.result() for future in challengers]]
        busy = sum(item["error_code"] == "busy_retry_later" for item in results)
        success = sum(item["status"] == 200 for item in results)
        retry = run_medical_request(
            base_url,
            timeout,
            config,
            tensor_seed=base_seed + 1,
            share_seed=base_seed + 101,
            nonce=retry_nonce,
        )
        passed = success >= 1 and busy >= 1 and retry["status"] == 200
        return passed, {
            "results": results,
            "success": success,
            "busy_retry_later": busy,
            "retry_after_busy": retry,
        }

    return capture_guard_check(
        server_pid=server_pid,
        name="inflight_limit",
        interception_layer="inflight_guard",
        summary=(
            "With single-channel limits, only one request should enter at a time and a busy request "
            "must remain retryable after the active request finishes."
        ),
        action=action,
        settle_sec=0.6,
    )


def check_rate_limit(base_url: str, timeout: float, count: int, server_pid: int | None) -> GuardCheck:
    def action():
        results = [run_invalid_empty_request(base_url, timeout) for _ in range(count)]
        limited = sum(item["error_code"] == "rate_limited_ip" for item in results)
        passed = limited >= 1
        return passed, {"results": results, "rate_limited_ip": limited}

    return capture_guard_check(
        server_pid=server_pid,
        name="ip_window_rate_limit",
        interception_layer="ip_rate_limit_guard",
        summary="After a short burst from the same IP, the endpoint should return rate_limited_ip before payload parsing.",
        action=action,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Black-box replay, inflight, and rate-limit guard checks for the showcase live demo endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--checks",
        default="duplicate_nonce,duplicate_payload,inflight,rate_limit",
        help="Comma-separated subset: duplicate_nonce, duplicate_payload, inflight, rate_limit",
    )
    parser.add_argument("--replay-concurrency", type=int, default=4)
    parser.add_argument("--inflight-concurrency", type=int, default=4)
    parser.add_argument("--rate-limit-count", type=int, default=8)
    parser.add_argument("--server-pid", type=int, default=0, help="Optional local server pid for resource-state sampling.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument(
        "--window-reset-sec",
        type=float,
        default=61.0,
        help="Sleep between checks to avoid cross-test contamination from the default per-IP rate window.",
    )
    args = parser.parse_args()

    config = load_remote_medical_config(args.base_url, args.timeout)
    registry = {
        "duplicate_nonce": lambda: check_duplicate_nonce(args.base_url, args.timeout, config, args.replay_concurrency, args.server_pid or None),
        "duplicate_payload": lambda: check_duplicate_payload(args.base_url, args.timeout, config, args.server_pid or None),
        "inflight": lambda: check_inflight_limit(args.base_url, args.timeout, config, args.inflight_concurrency, args.server_pid or None),
        "rate_limit": lambda: check_rate_limit(args.base_url, args.timeout, args.rate_limit_count, args.server_pid or None),
    }
    requested = [item.strip() for item in args.checks.split(",") if item.strip()]
    unknown = [item for item in requested if item not in registry]
    if unknown:
        raise SystemExit(f"unknown checks: {', '.join(unknown)}")

    checks = []
    for index, name in enumerate(requested):
        checks.append(registry[name]())
        if index != len(requested) - 1 and args.window_reset_sec > 0:
            time.sleep(args.window_reset_sec)

    payload = {
        "base_url": args.base_url,
        "server_pid": args.server_pid or None,
        "passed": all(item.passed for item in checks),
        "checks": [asdict(item) for item in checks],
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
