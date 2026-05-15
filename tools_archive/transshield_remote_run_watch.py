import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PTY = REPO_ROOT / "tools" / "transshield_remote_pty.py"
DEFAULT_REMOTE_TARGET = "wyb@10.204.248.175"
DEFAULT_REMOTE_PORT = 9001
DEFAULT_REMOTE_PYTHON = "/data/wyb/conda_envs/transshield/bin/python"
DEFAULT_REMOTE_REPO_ROOT = "/home/yclcg/Transshield_final"
DEFAULT_STATUS_TIMEOUT_SEC = 1200
DEFAULT_SYNC_TIMEOUT_SEC = 3600


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect or sync a long-running remote Transshield run directory."
    )
    parser.add_argument(
        "--password-env",
        default="TRANSSHIELD_REMOTE_PASSWORD",
        help="Environment variable consumed by tools/transshield_remote_pty.py",
    )
    parser.add_argument("--remote-target", default=DEFAULT_REMOTE_TARGET)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-repo-root", default=DEFAULT_REMOTE_REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print remote run status")
    add_common_run_args(status)
    status.add_argument("--json", action="store_true", help="Print raw JSON payload")
    status.add_argument("--timeout-sec", type=int, default=DEFAULT_STATUS_TIMEOUT_SEC)

    sync = subparsers.add_parser("sync", help="Sync a completed remote run directory to local")
    add_common_run_args(sync)
    sync.add_argument("--local-run-dir", help="Local output directory; inferred from repo roots if omitted")
    sync.add_argument("--post-sync-keepmask-summary", action="store_true")
    sync.add_argument("--post-sync-keepmask-doc-snippet", action="store_true")
    sync.add_argument("--timeout-sec", type=int, default=DEFAULT_SYNC_TIMEOUT_SEC)

    watch = subparsers.add_parser("watch-sync", help="Poll status until compare JSON exists, then sync")
    add_common_run_args(watch)
    watch.add_argument("--local-run-dir", help="Local output directory; inferred from repo roots if omitted")
    watch.add_argument("--post-sync-keepmask-summary", action="store_true")
    watch.add_argument("--post-sync-keepmask-doc-snippet", action="store_true")
    watch.add_argument("--interval-sec", type=int, default=60)
    watch.add_argument("--wait-timeout-sec", type=int, default=0, help="0 means wait forever")
    watch.add_argument("--status-timeout-sec", type=int, default=DEFAULT_STATUS_TIMEOUT_SEC)
    watch.add_argument("--sync-timeout-sec", type=int, default=DEFAULT_SYNC_TIMEOUT_SEC)
    return parser.parse_args()


def add_common_run_args(parser):
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument(
        "--compare-json-name",
        default="e2e_static_whole_forward_compare.json",
    )
    parser.add_argument(
        "--candidate-json-name",
        default="e2e_static_whole_forward_candidate_from_server.json",
    )


def run_remote_pty(command, password_env, timeout_sec):
    argv = [
        sys.executable,
        str(REMOTE_PTY),
        "--password-env",
        password_env,
        "--timeout-sec",
        str(timeout_sec),
        "--",
        *command,
    ]
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def extract_last_json(stdout_text):
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"failed to parse JSON from output:\n{stdout_text}")


def ssh_prefix(remote_target, remote_port):
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        str(remote_port),
        remote_target,
    ]


def remote_status(args, timeout_sec):
    remote_py = shlex.quote(args.remote_python)
    remote_run_dir = shlex.quote(args.remote_run_dir)
    compare_name = shlex.quote(args.compare_json_name)
    candidate_name = shlex.quote(args.candidate_json_name)
    remote_script = f"""
set -euo pipefail
REMOTE_RUN_DIR={remote_run_dir} \
REMOTE_COMPARE_NAME={compare_name} \
REMOTE_CANDIDATE_NAME={candidate_name} \
{remote_py} - <<'PY'
import json
import os
import subprocess
from pathlib import Path


def pick(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


run_dir = Path(os.environ["REMOTE_RUN_DIR"])
compare_path = run_dir / os.environ["REMOTE_COMPARE_NAME"]
candidate_path = run_dir / os.environ["REMOTE_CANDIDATE_NAME"]
payload = {{
    "run_dir": str(run_dir),
    "run_dir_exists": run_dir.is_dir(),
    "top_level_files": sorted(
        [entry.name for entry in run_dir.iterdir() if entry.is_file()]
    ) if run_dir.is_dir() else [],
    "candidate_exists": candidate_path.is_file(),
    "compare_exists": compare_path.is_file(),
}}
candidate_json = load_json(candidate_path) if candidate_path.is_file() else None
compare_json = load_json(compare_path) if compare_path.is_file() else None
if candidate_json is not None:
    payload["candidate_summary"] = {{
        "runtime": candidate_json.get("runtime"),
        "sample_count": candidate_json.get("sample_count"),
        "elapsed_sec": candidate_json.get("elapsed_sec"),
        "finite_logits": candidate_json.get("finite_logits"),
        "spu_params_mode": candidate_json.get("spu_params_mode", pick(candidate_json, "spu", "spu_params_mode")),
        "input_mode": candidate_json.get("input_mode", pick(candidate_json, "spu", "input_mode")),
        "host_plaintext_pixel_values_materialized": candidate_json.get(
            "host_plaintext_pixel_values_materialized",
            pick(candidate_json, "spu", "host_plaintext_pixel_values_materialized"),
        ),
        "host_private_share_tensors_loaded": candidate_json.get(
            "host_private_share_tensors_loaded",
            pick(candidate_json, "spu", "host_private_share_tensors_loaded"),
        ),
        "private_input_paths_redacted": candidate_json.get(
            "private_input_paths_redacted",
            pick(candidate_json, "spu", "private_input_paths_redacted"),
        ),
        "reveal_policy": candidate_json.get("reveal_policy", pick(candidate_json, "spu", "reveal_policy")),
        "runtime_pruning_keep_mask_stage_count": candidate_json.get("runtime_pruning_keep_mask_stage_count"),
    }}
if compare_json is not None:
    payload["compare_summary"] = {{
        "sample_count": compare_json.get("sample_count"),
        "argmax_match_ratio": pick(compare_json, "prediction_match", "argmax_match_ratio"),
        "threshold_match_ratio": pick(compare_json, "prediction_match", "threshold_match_ratio"),
        "logits_max_abs_error": pick(compare_json, "logits_error", "max_abs_error"),
        "probabilities_max_abs_error": pick(compare_json, "probabilities_error", "max_abs_error"),
    }}
try:
    ps_output = subprocess.check_output(["ps", "-ef"], text=True, errors="replace")
except Exception:
    ps_output = ""
needle = str(run_dir)
payload["matching_processes"] = [
    line.strip()
    for line in ps_output.splitlines()
    if needle in line and "grep" not in line
]
print(json.dumps(payload, ensure_ascii=False))
PY
"""
    result = run_remote_pty(
        [
            *ssh_prefix(args.remote_target, args.remote_port),
            remote_script,
        ],
        password_env=args.password_env,
        timeout_sec=timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return extract_last_json(result.stdout)


def infer_local_run_dir(remote_run_dir, remote_repo_root):
    remote_path = PurePosixPath(remote_run_dir)
    remote_root = PurePosixPath(remote_repo_root)
    try:
        rel = remote_path.relative_to(remote_root)
    except ValueError as exc:
        raise SystemExit(
            "--local-run-dir is required when --remote-run-dir is outside --remote-repo-root"
        ) from exc
    return REPO_ROOT / Path(*rel.parts)


def sync_remote_dir(args, local_run_dir, timeout_sec):
    local_run_dir = Path(local_run_dir).expanduser().resolve()
    local_run_dir.parent.mkdir(parents=True, exist_ok=True)
    remote_src = f"{args.remote_target}:{args.remote_run_dir.rstrip('/')}/"
    result = run_remote_pty(
        [
            "rsync",
            "-av",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {args.remote_port}",
            remote_src,
            str(local_run_dir) + "/",
        ],
        password_env=args.password_env,
        timeout_sec=timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print(result.stdout, end="")


def maybe_postprocess_keepmask_summary(local_run_dir: Path, enabled: bool):
    if not enabled:
        return
    summary_tool = REPO_ROOT / "tools" / "transshield_e2e_keepmask_result_summary.py"
    if not summary_tool.is_file():
        raise SystemExit(f"missing postprocess tool: {summary_tool}")
    output_json = local_run_dir / "keepmask_result_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(summary_tool),
            "--run-dir",
            str(local_run_dir),
            "--output-json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print(result.stdout, end="")
    print(f"postprocessed_keepmask_summary: {output_json}", flush=True)


def maybe_postprocess_keepmask_doc_snippet(local_run_dir: Path, enabled: bool):
    if not enabled:
        return
    snippet_tool = REPO_ROOT / "tools" / "transshield_e2e_keepmask_doc_snippet.py"
    if not snippet_tool.is_file():
        raise SystemExit(f"missing postprocess tool: {snippet_tool}")
    summary_json = local_run_dir / "keepmask_result_summary.json"
    output_md = local_run_dir / "keepmask_result_snippet.md"
    if not summary_json.is_file():
        raise SystemExit(f"missing keepmask summary JSON: {summary_json}")
    result = subprocess.run(
        [
            sys.executable,
            str(snippet_tool),
            "--summary-json",
            str(summary_json),
            "--output-md",
            str(output_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print(result.stdout, end="")
    print(f"postprocessed_keepmask_doc_snippet: {output_md}", flush=True)


def print_status(payload):
    print(f"run_dir: {payload['run_dir']}", flush=True)
    print(f"run_dir_exists: {payload['run_dir_exists']}", flush=True)
    print(f"candidate_exists: {payload['candidate_exists']}", flush=True)
    print(f"compare_exists: {payload['compare_exists']}", flush=True)
    files = payload.get("top_level_files") or []
    if files:
        print("top_level_files:", flush=True)
        for name in files:
            print(f"  - {name}", flush=True)
    candidate = payload.get("candidate_summary")
    if candidate:
        print("candidate_summary:", flush=True)
        for key in (
            "runtime",
            "sample_count",
            "elapsed_sec",
            "finite_logits",
            "spu_params_mode",
            "input_mode",
            "host_plaintext_pixel_values_materialized",
            "host_private_share_tensors_loaded",
            "private_input_paths_redacted",
            "reveal_policy",
            "runtime_pruning_keep_mask_stage_count",
        ):
            print(f"  - {key}: {candidate.get(key)}", flush=True)
    compare = payload.get("compare_summary")
    if compare:
        print("compare_summary:", flush=True)
        for key in (
            "sample_count",
            "argmax_match_ratio",
            "threshold_match_ratio",
            "logits_max_abs_error",
            "probabilities_max_abs_error",
        ):
            print(f"  - {key}: {compare.get(key)}", flush=True)
    processes = payload.get("matching_processes") or []
    print(f"matching_process_count: {len(processes)}", flush=True)
    for line in processes:
        print(f"  - {line}", flush=True)


def command_status(args):
    payload = remote_status(args, timeout_sec=args.timeout_sec)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return
    print_status(payload)


def command_sync(args):
    payload = remote_status(args, timeout_sec=args.timeout_sec)
    if not payload.get("compare_exists"):
        raise SystemExit("remote compare JSON does not exist yet; refusing to sync as completed result")
    local_run_dir = (
        Path(args.local_run_dir).expanduser().resolve()
        if args.local_run_dir
        else infer_local_run_dir(args.remote_run_dir, args.remote_repo_root)
    )
    sync_remote_dir(args, local_run_dir=local_run_dir, timeout_sec=args.timeout_sec)
    maybe_postprocess_keepmask_summary(local_run_dir=local_run_dir, enabled=args.post_sync_keepmask_summary)
    maybe_postprocess_keepmask_doc_snippet(local_run_dir=local_run_dir, enabled=args.post_sync_keepmask_doc_snippet)
    print(f"synced_to: {local_run_dir}")


def command_watch_sync(args):
    deadline = None if args.wait_timeout_sec <= 0 else time.time() + args.wait_timeout_sec
    last_fingerprint = None
    while True:
        payload = remote_status(args, timeout_sec=args.status_timeout_sec)
        fingerprint = json.dumps(
            {
                "candidate_exists": payload.get("candidate_exists"),
                "compare_exists": payload.get("compare_exists"),
                "top_level_files": payload.get("top_level_files"),
                "matching_processes": payload.get("matching_processes"),
                "candidate_summary": payload.get("candidate_summary"),
                "compare_summary": payload.get("compare_summary"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if fingerprint != last_fingerprint:
            print(f"[{time.strftime('%F %T')}] remote status changed")
            print_status(payload)
            last_fingerprint = fingerprint
        if payload.get("compare_exists"):
            local_run_dir = (
                Path(args.local_run_dir).expanduser().resolve()
                if args.local_run_dir
                else infer_local_run_dir(args.remote_run_dir, args.remote_repo_root)
            )
            sync_remote_dir(args, local_run_dir=local_run_dir, timeout_sec=args.sync_timeout_sec)
            maybe_postprocess_keepmask_summary(local_run_dir=local_run_dir, enabled=args.post_sync_keepmask_summary)
            maybe_postprocess_keepmask_doc_snippet(local_run_dir=local_run_dir, enabled=args.post_sync_keepmask_doc_snippet)
            print(f"synced_to: {local_run_dir}", flush=True)
            return
        if deadline is not None and time.time() >= deadline:
            raise SystemExit("watch timeout exceeded")
        time.sleep(max(args.interval_sec, 1))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.command == "status":
        command_status(args)
    elif args.command == "sync":
        command_sync(args)
    elif args.command == "watch-sync":
        command_watch_sync(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
