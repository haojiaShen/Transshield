#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, process: subprocess.Popen, timeout_sec: float = 20.0):
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"showcase test server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return
            last_error = f"health status: {payload}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.2)
    raise RuntimeError(f"showcase test server did not become healthy: {last_error}")


def stop_process(process: subprocess.Popen):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_check(
    *,
    script_name: str,
    script_args: list[str],
    server_env: dict[str, str],
    temp_dir: Path,
) -> dict:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(server_env)
    audit_dir = temp_dir / f"{script_name}_audit"
    env["TRANSSHIELD_SHOWCASE_AUDIT_DIR"] = str(audit_dir)
    env["TRANSSHIELD_SHOWCASE_RUN_DIR"] = str(temp_dir / f"{script_name}_runs")
    log_path = temp_dir / f"{script_name}_server.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "showcase_api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(base_url, process)
            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / script_name),
                "--base-url",
                base_url,
                "--server-pid",
                str(process.pid),
                *script_args,
            ]
            if script_name == "showcase_protocol_fuzz.py":
                command.extend(["--audit-rejections-jsonl", str(audit_dir / "audit_rejections.jsonl")])
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                timeout=60.0,
                check=False,
            )
        finally:
            stop_process(process)

    try:
        payload = json.loads(completed.stdout)
    except (UnboundLocalError, ValueError) as error:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{script_name} did not return JSON: {error}; stderr={getattr(completed, 'stderr', '')!r}; "
            f"server_log_tail={log_tail!r}"
        ) from error
    if completed.returncode != 0 or not payload.get("passed"):
        raise RuntimeError(f"{script_name} failed: {json.dumps(payload, ensure_ascii=False)}")
    return payload


def main():
    with tempfile.TemporaryDirectory(prefix="transshield_lightweight_checks_") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        fuzz = run_check(
            script_name="showcase_protocol_fuzz.py",
            script_args=[
                "--cases",
                "baseline,duplicate_field,extra_field,nested_multipart,invalid_header,truncated_body,share_hash_mismatch",
            ],
            server_env={
                "TRANSSHIELD_SHOWCASE_RUNTIME_MODE": "mock",
                "TRANSSHIELD_SHOWCASE_ACCEPTED_SLEEP_SEC": "0",
                "TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_LIMIT": "100",
            },
            temp_dir=temp_dir,
        )
        guard = run_check(
            script_name="showcase_guard_stress.py",
            script_args=[
                "--checks",
                "duplicate_nonce,duplicate_payload,inflight,rate_limit",
                "--window-reset-sec",
                "0",
            ],
            server_env={
                "TRANSSHIELD_SHOWCASE_RUNTIME_MODE": "mock",
                "TRANSSHIELD_SHOWCASE_ACCEPTED_SLEEP_SEC": "1.2",
                "TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_LIMIT": "16",
            },
            temp_dir=temp_dir,
        )

    print(
        json.dumps(
            {
                "passed": True,
                "model_or_spu_computation_started": False,
                "protocol_fuzz_cases": len(fuzz["cases"]),
                "guard_stress_checks": len(guard["checks"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
