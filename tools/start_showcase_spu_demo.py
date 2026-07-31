#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from showcase_api.bundle_preflight import require_bundle_ready


SPU_TEMPLATE = REPO_ROOT / "configs" / "transshield_runtime" / "2pc.template.json"
SPU_RUNTIME_DIR = REPO_ROOT / "logs" / "showcase_runtime"
SPU_CONFIG = SPU_RUNTIME_DIR / "2pc.runtime.json"
SPU_STATE = REPO_ROOT / "logs" / "spu_runtime_ports.json"
SPU_LOG_DIR = REPO_ROOT / "logs" / "spu_nodes"
SHOWCASE_DIST = REPO_ROOT / "showcase" / "dist"
DEFAULT_LOG = REPO_ROOT / "artifacts" / "showcase_server_logs" / "uvicorn_7862.log"
DEFAULT_BUNDLE = REPO_ROOT / "artifacts" / "frozen_bundle_medical_dynamic_mainline"


def python_bin(raw: str) -> str:
    if not raw:
        return sys.executable
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.absolute())
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return str(repo_candidate.absolute())
    return raw


def run(command: list[str]):
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def require_dist():
    if (SHOWCASE_DIST / "index.html").exists():
        return
    raise RuntimeError("showcase/dist is missing; run `cd showcase && npm install && npm run build` first.")


def require_spu_bundle():
    raw = os.environ.get("TRANSSHIELD_SHOWCASE_BUNDLE_DIR", "").strip()
    bundle_dir = Path(raw).expanduser().resolve() if raw else DEFAULT_BUNDLE
    status = require_bundle_ready(bundle_dir, verify_hash=True)
    print(
        json.dumps(
            {
                "bundle_preflight": "passed",
                "bundle_dir": status["bundle_dir"],
                "model_state_file": status["model_state_file"],
                "model_sha256_matches": status["model_sha256_matches"],
            },
            ensure_ascii=False,
        )
    )


def start_spu(python: str, timeout_sec: float, template: Path):
    SPU_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    template = template.expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(f"SPU runtime template not found: {template}")
    shutil.copy2(template, SPU_CONFIG)
    run(
        [
            python,
            str(REPO_ROOT / "tools" / "transshield_spu_runtime_setup.py"),
            "start",
            "--config",
            str(SPU_CONFIG),
            "--template",
            "",
            "--restart",
            "--remove-unsupported-cheetah-fields",
            "--log-dir",
            str(SPU_LOG_DIR),
            "--state-json",
            str(SPU_STATE),
            "--startup-timeout-sec",
            str(timeout_sec),
        ]
    )


def api_env(python: str, runtime_mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TRANSSHIELD_SHOWCASE_RUNTIME_MODE"] = runtime_mode
    env["TRANSSHIELD_SHOWCASE_PYTHON_BIN"] = python
    env["TRANSSHIELD_SHOWCASE_SPU_CONFIG"] = str(SPU_CONFIG)
    return env


def probe_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def wait_for_health(host: str, port: int, timeout_sec: float) -> dict:
    url = f"http://{probe_host(host)}:{port}/api/health"
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return payload
            last_error = f"unexpected payload: {payload}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.75)
    raise RuntimeError(f"showcase API did not become healthy: {last_error}")


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def start_api_daemon(args, python: str, env: dict[str, str]):
    tmux = shutil.which("tmux")
    if not tmux:
        raise RuntimeError("daemon mode requires tmux; run without `--daemon` for foreground mode.")

    session = args.api_session or f"transshield_showcase_{args.port}"
    subprocess.run([tmux, "kill-session", "-t", session], check=False)
    args.api_log.parent.mkdir(parents=True, exist_ok=True)

    exports = " ".join(
        f"{key}={quote(value)}" for key, value in sorted(env.items()) if key.startswith("TRANSSHIELD_")
    )
    command = (
        f"cd {quote(str(REPO_ROOT))} && export {exports} && "
        f"{quote(python)} -m uvicorn showcase_api.app:app --host {quote(args.host)} --port {args.port} "
        f"> {quote(str(args.api_log))} 2>&1"
    )
    run([tmux, "new-session", "-d", "-s", session, command])
    health = wait_for_health(args.host, args.port, args.api_startup_timeout_sec)
    print(
        json.dumps(
            {
                "status": "started",
                "url": f"http://{probe_host(args.host)}:{args.port}",
                "live_demo_url": f"http://{probe_host(args.host)}:{args.port}/live-demo",
                "runtime_mode": health.get("runtime_mode"),
                "runner_present": health.get("runner_present"),
                "spu_config_present": health.get("spu_config_present"),
                "tmux_session": session,
                "api_log": str(args.api_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def start_api_foreground(args, python: str, env: dict[str, str]):
    command = [
        python,
        "-m",
        "uvicorn",
        "showcase_api.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("$ " + " ".join(command))
    os.execvpe(command[0], command, env)


def build_parser():
    parser = argparse.ArgumentParser(description="Start the TransShield showcase SPU Live Demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--runtime-mode", choices=["spu", "mock"], default="spu")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--api-session", default="")
    parser.add_argument("--api-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--api-startup-timeout-sec", type=float, default=30.0)
    parser.add_argument("--spu-startup-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--spu-template",
        type=Path,
        default=SPU_TEMPLATE,
        help="Runtime template to copy before allocating local SPU ports.",
    )
    parser.add_argument(
        "--secure-pruning-mode",
        choices=["mask", "compact"],
        default="",
        help="Override TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_MODE for this launch.",
    )
    parser.add_argument(
        "--spu-compile-cache-dir",
        type=Path,
        default=None,
        help="Override the content-addressed SPU compile-cache directory for this launch.",
    )
    parser.add_argument("--skip-spu-start", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    python = python_bin(args.python_bin)
    if not args.api_log.is_absolute():
        args.api_log = REPO_ROOT / args.api_log
    if args.secure_pruning_mode:
        os.environ["TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_MODE"] = args.secure_pruning_mode
    if args.spu_compile_cache_dir is not None:
        os.environ["TRANSSHIELD_SHOWCASE_SPU_COMPILE_CACHE_DIR"] = str(
            args.spu_compile_cache_dir.expanduser().resolve()
        )
    require_dist()
    if args.runtime_mode == "spu":
        require_spu_bundle()
        if not args.skip_spu_start:
            start_spu(python, args.spu_startup_timeout_sec, args.spu_template)
        else:
            print("Skipping SPU runtime startup because --skip-spu-start was provided.")
    elif args.runtime_mode == "mock":
        print("Skipping SPU runtime startup because runtime mode is mock.")

    env = api_env(python, args.runtime_mode)
    if args.daemon:
        start_api_daemon(args, python, env)
    else:
        start_api_foreground(args, python, env)


if __name__ == "__main__":
    main()
