#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
ADMIN_DIR = REPO_ROOT / "showcase_admin"
ADMIN_LOG_DIR = REPO_ROOT / "artifacts" / "showcase_admin_launcher"
NPM_CACHE_DIR = ADMIN_DIR / ".npm-cache"
ADMIN_DIST_DIR = ADMIN_DIR / "dist"
ADMIN_URL = "http://127.0.0.1:4174"
API_URL = "http://127.0.0.1:7863"


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
  subprocess.run(command, cwd=str(cwd), env=env, check=True)


def ensure_binary(name: str) -> str:
  path = shutil.which(name)
  if not path:
    raise RuntimeError(f"{name} not found in PATH.")
  return path


def is_url_ready(url: str, timeout: float = 2.0) -> bool:
  try:
    with urlopen(url, timeout=timeout) as response:
      return response.status == 200
  except Exception:
    return False


def wait_for_url(url: str, timeout_sec: float) -> None:
  deadline = time.time() + timeout_sec
  while time.time() < deadline:
    if is_url_ready(url):
      return
    time.sleep(0.75)
  raise RuntimeError(f"Timed out waiting for {url}")


def ensure_frontend_deps() -> dict[str, str]:
  ensure_binary("npm")
  ensure_binary("node")
  NPM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env["npm_config_cache"] = str(NPM_CACHE_DIR)

  vite_cmd = ADMIN_DIR / "node_modules" / ".bin" / "vite.cmd"
  tsc_cmd = ADMIN_DIR / "node_modules" / ".bin" / "tsc.cmd"
  if not vite_cmd.exists() or not tsc_cmd.exists():
    run(["npm", "install", "--cache", str(NPM_CACHE_DIR)], cwd=ADMIN_DIR, env=env)
  return env


def build_frontend(env: dict[str, str]) -> None:
  node = ensure_binary("node")
  tsc_js = ADMIN_DIR / "node_modules" / "typescript" / "bin" / "tsc"
  vite_js = ADMIN_DIR / "node_modules" / "vite" / "bin" / "vite.js"
  run([node, str(tsc_js), "-b"], cwd=ADMIN_DIR, env=env)
  run([node, str(vite_js), "build"], cwd=ADMIN_DIR, env=env)


def launch_backend_if_needed() -> None:
  if is_url_ready(f"{API_URL}/api/health"):
    return

  ADMIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
  stdout_path = ADMIN_LOG_DIR / "admin_api.stdout.log"
  stderr_path = ADMIN_LOG_DIR / "admin_api.stderr.log"
  env = os.environ.copy()
  env["TRANSSHIELD_CORS_ALLOWED_ORIGINS"] = ",".join(
    [
      "http://127.0.0.1:4174",
      "http://localhost:4174",
      "http://127.0.0.1:4173",
      "http://localhost:4173",
      "http://127.0.0.1:7863",
      "http://localhost:7863"
    ]
  )
  env.setdefault("TRANSSHIELD_SHOWCASE_RUNTIME_MODE", "mock")

  creationflags = 0
  if os.name == "nt":
    creationflags = subprocess.CREATE_NO_WINDOW

  with open(stdout_path, "a", encoding="utf-8") as stdout_handle, open(
    stderr_path, "a", encoding="utf-8"
  ) as stderr_handle:
    subprocess.Popen(
      [
        sys.executable,
        "-m",
        "uvicorn",
        "showcase_api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7863"
      ],
      cwd=str(REPO_ROOT),
      env=env,
      stdout=stdout_handle,
      stderr=stderr_handle,
      creationflags=creationflags
    )

  wait_for_url(f"{API_URL}/api/health", timeout_sec=30.0)


def launch_preview_if_needed(env: dict[str, str]) -> None:
  if is_url_ready(ADMIN_URL):
    return

  ADMIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
  stdout_path = ADMIN_LOG_DIR / "showcase_admin.stdout.log"
  stderr_path = ADMIN_LOG_DIR / "showcase_admin.stderr.log"
  creationflags = 0
  if os.name == "nt":
    creationflags = subprocess.CREATE_NO_WINDOW

  with open(stdout_path, "a", encoding="utf-8") as stdout_handle, open(
    stderr_path, "a", encoding="utf-8"
  ) as stderr_handle:
    subprocess.Popen(
      [
        sys.executable,
        "-m",
        "http.server",
        "4174",
        "--bind",
        "127.0.0.1",
        "--directory",
        str(ADMIN_DIST_DIR)
      ],
      cwd=str(REPO_ROOT),
      env=env,
      stdout=stdout_handle,
      stderr=stderr_handle,
      creationflags=creationflags
    )

  wait_for_url(ADMIN_URL, timeout_sec=20.0)


def main() -> int:
  env = os.environ.copy()
  try:
    env = ensure_frontend_deps()
    build_frontend(env)
  except RuntimeError as exc:
    if ADMIN_DIST_DIR.exists() and (ADMIN_DIST_DIR / "index.html").exists():
      print(f"[WARN] {exc} Falling back to existing showcase_admin/dist.")
    else:
      raise
  launch_backend_if_needed()
  launch_preview_if_needed(env)
  webbrowser.open(f"{ADMIN_URL}/")
  print("密捷管理控制台已启动。")
  print(f"管理端地址: {ADMIN_URL}")
  print(f"训练后端地址: {API_URL}")
  print("默认账号: admin / admin123")
  print(f"日志目录: {ADMIN_LOG_DIR}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except subprocess.CalledProcessError as exc:
    print(f"[ERROR] command failed: {exc}", file=sys.stderr)
    raise
  except Exception as exc:  # pragma: no cover
    print(f"[ERROR] {exc}", file=sys.stderr)
    raise
