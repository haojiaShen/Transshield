from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from showcase_api.admin_config import (
    REPO_ROOT,
    ShowcaseConfig,
    build_path_replacements,
    relative_to_repo_path,
    relativize_text,
    relativize_value,
    resolve_repo_path,
)


ACTIVE_STATUSES = {"queued", "starting", "running", "postprocessing"}


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _fs_ts(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return _now_ts()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(path: Path, max_chars: int = 16000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _metrics_payload(*, threshold_best: dict | None, threshold_eval: dict | None) -> dict:
    best_threshold = None
    if threshold_best:
        best_threshold = threshold_best.get("best_threshold", threshold_best.get("eval_binary_threshold"))
    elif threshold_eval:
        best_threshold = threshold_eval.get("eval_binary_threshold")

    argmax_accuracy = None
    if threshold_best:
        argmax_accuracy = threshold_best.get("default_argmax_acc1", threshold_best.get("default_acc1"))

    best_threshold_accuracy = None
    if threshold_eval:
        best_threshold_accuracy = threshold_eval.get("eval_acc1")
    elif threshold_best:
        best_threshold_accuracy = threshold_best.get("best_threshold_acc")

    auc = None
    if threshold_eval:
        auc = threshold_eval.get("auc")
    elif threshold_best:
        auc = threshold_best.get("auc")

    sample_count = None
    if threshold_best:
        sample_count = threshold_best.get("sample_count")
    elif threshold_eval:
        sample_count = threshold_eval.get("sample_count")

    return {
        "argmaxAccuracy": _safe_float(argmax_accuracy),
        "bestThreshold": _safe_float(best_threshold),
        "bestThresholdAccuracy": _safe_float(best_threshold_accuracy),
        "auc": _safe_float(auc),
        "sampleCount": _safe_int(sample_count),
        "argmax_accuracy": _safe_float(argmax_accuracy),
        "best_threshold": _safe_float(best_threshold),
        "best_threshold_accuracy": _safe_float(best_threshold_accuracy),
        "sample_count": _safe_int(sample_count),
    }


def _infer_historical_name(run_name: str) -> str:
    if "distill_pair" in run_name and "official" in run_name:
        return "蒸馏补偿配对实验（蒸馏补偿）"
    if "distill_pair" in run_name and "nodistill" in run_name:
        return "蒸馏补偿配对实验（无蒸馏基线）"
    if "depth_pair" in run_name and "depth0" in run_name:
        return "部署对齐控制实验（depth=0）"
    if "depth_pair" in run_name and "depth12" in run_name:
        return "部署对齐控制实验（depth=12）"
    if "baserate_r030" in run_name:
        return "base_rate 扫描（0.3）"
    if "baserate_r040" in run_name:
        return "base_rate 扫描（0.4）"
    if "baserate_r050" in run_name:
        return "base_rate 扫描（0.5）"
    if "baserate_r060" in run_name:
        return "base_rate 扫描（0.6）"
    if "baserate_r080" in run_name:
        return "base_rate 扫描（0.8）"
    if "funcabl_joint" in run_name:
        return "安全友好函数联合消融"
    if "secure_static_accprof" in run_name:
        return "医疗动态主线正式训练"
    if "secure_static_depth12_uniform_fixed_square" in run_name:
        return "静态安全对照训练"
    if "cnn_plaintext" in run_name:
        return "CNN 明文基线训练"
    return f"历史训练任务：{run_name}"


class TrainingJobManager:
    def __init__(self, config: ShowcaseConfig):
        self.config = config
        self.job_root = config.admin_job_root
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._path_replacements = build_path_replacements(config.default_train_data_path, config.default_eval_data_path)
        self._path_replacements[str(config.python_bin)] = "python"
        self._path_replacements["/data/wyb/conda_envs/transshield/bin/python"] = "python"
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._runtime: dict[str, dict] = {}
        self._queue: list[str] = []
        self._active_job_id: str | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="showcase-admin-jobs", daemon=True)
        self._load_existing_jobs()
        self._load_historical_jobs()

    def _portable_payload(self, payload):
        return relativize_value(payload, self._path_replacements)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            items = [self._jobs[job_id].copy() for job_id in self._jobs]
        return sorted(items, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.copy() if job else None

    def get_job_log(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            stdout_path = resolve_repo_path(job["log_paths"]["stdout"])
            stderr_path = resolve_repo_path(job["log_paths"]["stderr"])
        return {
            "job_id": job_id,
            "stdout": relativize_text(_tail_text(stdout_path), self._path_replacements),
            "stderr": relativize_text(_tail_text(stderr_path), self._path_replacements),
        }

    def create_job(self, payload: dict[str, Any]) -> dict:
        request = self._normalize_request(payload)
        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.job_root / job_id
        run_dir = self.config.train_output_root / request["run_name"]
        bundle_dir = self.config.bundle_output_root / request["bundle_name"]
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        status = {
            "job_id": job_id,
            "name": request["name"],
            "preset_id": request.get("preset_id"),
            "mode": request["mode"],
            "status": "queued",
            "created_at": _now_ts(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "step_index": None,
            "command": None,
            "command_sequence": [],
            "output_dir": relative_to_repo_path(run_dir),
            "bundle_dir": relative_to_repo_path(bundle_dir),
            "log_paths": {
                "stdout": relative_to_repo_path(stdout_path),
                "stderr": relative_to_repo_path(stderr_path),
            },
            "metrics_summary": None,
            "artifacts": {
                "run_dir": relative_to_repo_path(run_dir),
                "bundle_dir": relative_to_repo_path(bundle_dir),
            },
            "error_message": None,
            "cancel_requested": False,
            "readonly": False,
            "source": "console",
            "request_path": relative_to_repo_path(job_dir / "request.json"),
            "status_path": relative_to_repo_path(job_dir / "status.json"),
            "summary_path": relative_to_repo_path(job_dir / "summary.json"),
        }
        job_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        _write_json(job_dir / "request.json", self._portable_payload(request))
        _write_json(job_dir / "status.json", self._portable_payload(status))
        with self._lock:
            self._jobs[job_id] = self._portable_payload(status)
            self._queue.append(job_id)
        return status.copy()

    def cancel_job(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False, "Task not found."
            if job.get("readonly"):
                return False, "Historical training records are read-only."
            if job["status"] in {"completed", "failed", "cancelled"}:
                return False, "Task has already finished."
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = _now_ts()
                job["cancel_requested"] = True
                self._queue = [item for item in self._queue if item != job_id]
                self._save_job_status(job_id)
                return True, "Queued task cancelled."
            runtime = self._runtime.get(job_id)
            if runtime and runtime.get("process"):
                job["cancel_requested"] = True
                process: subprocess.Popen = runtime["process"]
                self._terminate_process(process)
                self._save_job_status(job_id)
                return True, "Termination signal sent."
            return False, "Task cannot be cancelled right now."

    def _load_existing_jobs(self) -> None:
        for status_path in sorted(self.job_root.glob("job_*/status.json")):
            request_path = status_path.with_name("request.json")
            if request_path.exists():
                original_request = _read_json(request_path)
                request = self._portable_payload(original_request)
                if request != original_request:
                    _write_json(request_path, request)
            original_job = _read_json(status_path)
            job = self._portable_payload(original_job)
            if job != original_job:
                _write_json(status_path, job)
            job_id = job["job_id"]
            job.setdefault("readonly", False)
            job.setdefault("source", "console")
            if job.get("status") in ACTIVE_STATUSES:
                job["status"] = "failed"
                job["finished_at"] = _now_ts()
                job["error_message"] = "showcase_api restarted while task was active."
                _write_json(status_path, job)
            self._jobs[job_id] = job
            if job.get("status") == "queued":
                self._queue.append(job_id)

    def _load_historical_jobs(self) -> None:
        bundle_map = self._build_bundle_run_index()
        if not self.config.train_output_root.exists():
            return
        for run_dir in sorted(self.config.train_output_root.iterdir()):
            if not run_dir.is_dir():
                continue
            job = self._build_historical_job(run_dir, bundle_map)
            if not job:
                continue
            self._jobs[job["job_id"]] = job

    def _build_bundle_run_index(self) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        artifacts_root = REPO_ROOT / "artifacts"
        for bundle_dir in artifacts_root.glob("frozen_bundle_*"):
            manifest_path = bundle_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = _read_json(manifest_path)
            except Exception:
                continue
            primary = manifest.get("primary") or {}
            run_dir = str(primary.get("run_dir") or "").strip()
            if run_dir:
                mapping[Path(run_dir).name] = bundle_dir
        return mapping

    def _build_historical_job(self, run_dir: Path, bundle_map: dict[str, Path]) -> dict | None:
        checkpoint_path = run_dir / "checkpoint-best.pth"
        threshold_best_path = run_dir / "threshold_best.json"
        threshold_eval_path = run_dir / "threshold_eval.json"
        command_path = run_dir / "command.sh"
        if not checkpoint_path.exists() and not threshold_best_path.exists() and not threshold_eval_path.exists():
            return None

        threshold_best = _read_json(threshold_best_path) if threshold_best_path.exists() else {}
        threshold_eval = _read_json(threshold_eval_path) if threshold_eval_path.exists() else {}
        bundle_dir = bundle_map.get(run_dir.name)
        bundle_manifest = bundle_dir / "manifest.json" if bundle_dir else None
        bundle_args = bundle_dir / "args_snapshot.json" if bundle_dir else None
        stdout_path = run_dir / "train_stdout.log"
        fallback_log_path = run_dir / "log.txt"
        log_path = stdout_path if stdout_path.exists() else fallback_log_path
        created_at = _fs_ts(run_dir)
        command_text = (
            relativize_text(command_path.read_text(encoding="utf-8", errors="replace").strip(), self._path_replacements)
            if command_path.exists()
            else ""
        )

        return {
            "job_id": f"historical::{run_dir.name}",
            "name": _infer_historical_name(run_dir.name),
            "preset_id": "历史真实训练",
            "mode": "historical",
            "status": "completed",
            "created_at": created_at,
            "started_at": created_at,
            "finished_at": created_at,
            "current_step": "历史结果归档",
            "step_index": None,
            "command": command_text or None,
            "command_sequence": [command_text] if command_text else [],
            "output_dir": relative_to_repo_path(run_dir),
            "bundle_dir": relative_to_repo_path(bundle_dir) if bundle_dir else None,
            "log_paths": {
                "stdout": relative_to_repo_path(log_path),
                "stderr": relative_to_repo_path(run_dir / "_no_stderr.log"),
            },
            "metrics_summary": _metrics_payload(threshold_best=threshold_best, threshold_eval=threshold_eval),
            "artifacts": {
                "run_dir": relative_to_repo_path(run_dir),
                "checkpoint_best": relative_to_repo_path(checkpoint_path) if checkpoint_path.exists() else None,
                "threshold_best_json": relative_to_repo_path(threshold_best_path) if threshold_best_path.exists() else None,
                "threshold_eval_json": relative_to_repo_path(threshold_eval_path) if threshold_eval_path.exists() else None,
                "bundle_dir": relative_to_repo_path(bundle_dir) if bundle_dir else None,
                "bundle_manifest": (
                    relative_to_repo_path(bundle_manifest)
                    if bundle_manifest and bundle_manifest.exists()
                    else None
                ),
                "bundle_args_snapshot": (
                    relative_to_repo_path(bundle_args)
                    if bundle_args and bundle_args.exists()
                    else None
                ),
            },
            "error_message": None,
            "cancel_requested": False,
            "readonly": True,
            "source": "historical",
            "request_path": None,
            "status_path": None,
            "summary_path": None,
        }

    def _save_job_status(self, job_id: str) -> None:
        if self._jobs[job_id].get("readonly"):
            return
        _write_json(resolve_repo_path(self._jobs[job_id]["status_path"]), self._portable_payload(self._jobs[job_id]))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_active_job()
                self._dispatch_next_job()
            except Exception:
                pass
            time.sleep(1.0)

    def _dispatch_next_job(self) -> None:
        with self._lock:
            if self._active_job_id is not None:
                return
            if not self._queue:
                return
            next_job_id = self._queue.pop(0)
            job = self._jobs.get(next_job_id)
            if not job or job.get("status") != "queued":
                return
            self._active_job_id = next_job_id
        self._start_job(next_job_id)

    def _start_job(self, job_id: str) -> None:
        request = _read_json(self.job_root / job_id / "request.json")
        steps = self._build_steps(request)
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "starting"
            job["started_at"] = _now_ts()
            job["command_sequence"] = [step["command_display"] for step in steps]
            self._runtime[job_id] = {"request": request, "steps": steps, "step_index": 0, "process": None}
            self._save_job_status(job_id)
        self._start_step(job_id)

    def _start_step(self, job_id: str) -> None:
        with self._lock:
            runtime = self._runtime[job_id]
            step_index = int(runtime["step_index"])
            step = runtime["steps"][step_index]
            job = self._jobs[job_id]
            job["status"] = step["phase"]
            job["current_step"] = step["name"]
            job["step_index"] = step_index
            job["command"] = step["command_display"]
            stdout_path = resolve_repo_path(job["log_paths"]["stdout"])
            stderr_path = resolve_repo_path(job["log_paths"]["stderr"])
            self._append_log(stdout_path, f"\n[{_now_ts()}] >>> start step: {step['name']}\n{step['command_display']}\n")
            self._save_job_status(job_id)

        stdout_handle = open(stdout_path, "a", encoding="utf-8")
        stderr_handle = open(stderr_path, "a", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        process = subprocess.Popen(
            step["command"],
            cwd=str(REPO_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        with self._lock:
            runtime["process"] = process
            runtime["stdout_handle"] = stdout_handle
            runtime["stderr_handle"] = stderr_handle

    def _poll_active_job(self) -> None:
        with self._lock:
            job_id = self._active_job_id
            runtime = self._runtime.get(job_id or "")
            process = runtime.get("process") if runtime else None
        if not job_id or not runtime or process is None:
            return

        return_code = process.poll()
        if return_code is None:
            return

        with self._lock:
            stdout_handle = runtime.pop("stdout_handle", None)
            stderr_handle = runtime.pop("stderr_handle", None)
            if stdout_handle:
                stdout_handle.close()
            if stderr_handle:
                stderr_handle.close()
            job = self._jobs[job_id]
            cancel_requested = bool(job.get("cancel_requested"))
            step_index = int(runtime["step_index"])
            step = runtime["steps"][step_index]

        if cancel_requested:
            self._finalize_cancelled(job_id)
            return
        if return_code != 0:
            self._finalize_failed(job_id, f"step `{step['name']}` exited with code {return_code}")
            return

        with self._lock:
            runtime["process"] = None
            runtime["step_index"] = step_index + 1

        if runtime["step_index"] >= len(runtime["steps"]):
            self._finalize_completed(job_id)
            return
        self._start_step(job_id)

    def _finalize_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "cancelled"
            job["finished_at"] = _now_ts()
            job["error_message"] = "Task cancelled by administrator."
            self._save_job_status(job_id)
            self._runtime.pop(job_id, None)
            self._active_job_id = None

    def _finalize_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "failed"
            job["finished_at"] = _now_ts()
            job["error_message"] = message
            self._save_job_status(job_id)
            self._runtime.pop(job_id, None)
            self._active_job_id = None

    def _finalize_completed(self, job_id: str) -> None:
        summary = self._build_summary(job_id)
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "completed"
            job["finished_at"] = _now_ts()
            job["metrics_summary"] = summary.get("metrics_summary")
            job["artifacts"] = summary.get("artifacts")
            job["command"] = None
            self._save_job_status(job_id)
            _write_json(resolve_repo_path(job["summary_path"]), self._portable_payload(summary))
            self._runtime.pop(job_id, None)
            self._active_job_id = None

    def _append_log(self, path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def _terminate_process(self, process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
        except Exception:
            return

    def _normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset_id = str(payload.get("preset_id") or "").strip()
        mode = str(payload.get("mode") or ("preset" if preset_id else "custom"))
        overrides = dict(payload.get("parameters") or {})
        base: dict[str, Any] = {}
        if preset_id:
            matched = next((item for item in self.config.training_presets if item.id == preset_id), None)
            if not matched:
                raise ValueError(f"unknown preset_id: {preset_id}")
            base.update(matched.parameters)
        base.update(overrides)

        run_suffix = uuid.uuid4().hex[:6]
        name = str(payload.get("name") or base.get("name") or preset_id or "admin-train-job").strip()
        run_name = str(base.get("run_name") or f"{preset_id or 'custom'}_{time.strftime('%Y%m%d_%H%M%S')}_{run_suffix}")
        bundle_name = str(base.get("bundle_name") or f"{run_name}_bundle")

        return {
            "mode": mode,
            "name": name,
            "preset_id": preset_id or None,
            "run_name": run_name,
            "bundle_name": bundle_name,
            "model": str(base.get("model") or "deit-s"),
            "data_set": str(base.get("data_set") or "image_folder"),
            "train_data_path": relativize_text(
                str(base.get("train_data_path") or self.config.default_train_data_path),
                self._path_replacements,
            ),
            "eval_data_path": relativize_text(
                str(base.get("eval_data_path") or self.config.default_eval_data_path),
                self._path_replacements,
            ),
            "nb_classes": int(base.get("nb_classes") or 2),
            "input_size": int(base.get("input_size") or 224),
            "batch_size": int(base.get("batch_size") or self.config.default_batch_size),
            "epochs": int(base.get("epochs") or 8),
            "num_workers": int(base.get("num_workers") or self.config.default_num_workers),
            "device": str(base.get("device") or self.config.default_device),
            "base_rate": float(base.get("base_rate") or 0.7),
            "ratio_weight": float(base.get("ratio_weight") or 2.0),
            "lr": float(base.get("lr") or 1e-5),
            "warmup_epochs": int(base.get("warmup_epochs") or 0),
            "warmup_steps": int(base.get("warmup_steps") or 20),
            "clip_grad": float(base.get("clip_grad") or 1.0),
            "cls_distill_weight": float(base.get("cls_distill_weight") or 0.0),
            "token_distill_weight": float(base.get("token_distill_weight") or 0.0),
            "square_activation_mode": str(base.get("square_activation_mode") or "learnable_quadratic_gelu_init"),
            "approx_attn_mode": str(base.get("approx_attn_mode") or "relu"),
            "use_square_gelu": bool(base.get("use_square_gelu", True)),
            "use_approx_attn": bool(base.get("use_approx_attn", True)),
            "use_mask_pruning": bool(base.get("use_mask_pruning", True)),
            "inference_friendly_ops": bool(base.get("inference_friendly_ops", False)),
            "eval_pruning_mode": str(base.get("eval_pruning_mode") or "compare_network_tie"),
            "eval_tie_policy": str(base.get("eval_tie_policy") or "lowest_index"),
            "patch_embed_bias_init_mode": str(base.get("patch_embed_bias_init_mode") or "zero"),
            "freeze_patch_embed_proj": bool(base.get("freeze_patch_embed_proj", True)),
            "pretrained_fix_step": int(base.get("pretrained_fix_step") or 0),
            "secure_static_train_depth": int(base.get("secure_static_train_depth") or 0),
            "secure_static_skip_pruning": bool(base.get("secure_static_skip_pruning", True)),
            "teacher_checkpoint_path": relativize_text(
                str(base.get("teacher_checkpoint_path") or ""),
                self._path_replacements,
            ),
            "finetune": relativize_text(str(base.get("finetune") or ""), self._path_replacements),
            "model_ema": bool(base.get("model_ema", False)),
            "save_ckpt": bool(base.get("save_ckpt", True)),
            "save_ckpt_freq": int(base.get("save_ckpt_freq") or 1),
            "save_ckpt_num": int(base.get("save_ckpt_num") or 2),
            "auto_resume": bool(base.get("auto_resume", False)),
            "use_amp": bool(base.get("use_amp", False)),
            "mixup": float(base.get("mixup") or 0.0),
            "cutmix": float(base.get("cutmix") or 0.0),
            "seed": int(base.get("seed") or 0),
            "lr_scale": float(base.get("lr_scale") or 1.0),
            "groupa_lr_scale": float(base.get("groupa_lr_scale") or 0.1),
            "activation_lr_scale": float(base.get("activation_lr_scale") or 1.0),
            "export_bundle": bool(base.get("export_bundle", True)),
            "threshold_device": str(base.get("threshold_device") or base.get("device") or self.config.default_device),
            "threshold_batch_size": int(base.get("threshold_batch_size") or max(16, int(base.get("batch_size") or self.config.default_batch_size))),
            "threshold_num_workers": int(base.get("threshold_num_workers") or base.get("num_workers") or self.config.default_num_workers),
        }

    def _build_steps(self, request: dict[str, Any]) -> list[dict]:
        run_dir = self.config.train_output_root / request["run_name"]
        bundle_dir = self.config.bundle_output_root / request["bundle_name"]
        train_command = self._build_train_command(request, run_dir)
        threshold_search_command = self._build_threshold_search_command(request, run_dir)
        threshold_eval_command = self._build_threshold_eval_command(request, run_dir)
        steps = [
            {"name": "train", "phase": "running", "command": train_command, "command_display": " ".join(train_command)},
            {
                "name": "threshold_search",
                "phase": "postprocessing",
                "command": threshold_search_command,
                "command_display": " ".join(threshold_search_command),
            },
            {
                "name": "threshold_eval",
                "phase": "postprocessing",
                "command": threshold_eval_command,
                "command_display": " ".join(threshold_eval_command),
            },
        ]
        if request["export_bundle"]:
            export_command = self._build_export_command(
                request, run_dir, bundle_dir, train_command, threshold_search_command, threshold_eval_command
            )
            steps.append(
                {
                    "name": "freeze_export",
                    "phase": "postprocessing",
                    "command": export_command,
                    "command_display": " ".join(export_command),
                }
            )
        return steps

    def _build_train_command(self, request: dict[str, Any], run_dir: Path) -> list[str]:
        run_dir_arg = relative_to_repo_path(run_dir)
        log_dir_arg = relative_to_repo_path(run_dir / "tb")
        command = [
            self.config.python_bin,
            "main.py",
            "--model",
            request["model"],
            "--data_set",
            request["data_set"],
            "--data_path",
            request["train_data_path"],
            "--eval_data_path",
            request["eval_data_path"],
            "--nb_classes",
            str(request["nb_classes"]),
            "--output_dir",
            run_dir_arg,
            "--log_dir",
            log_dir_arg,
            "--input_size",
            str(request["input_size"]),
            "--batch_size",
            str(request["batch_size"]),
            "--epochs",
            str(request["epochs"]),
            "--num_workers",
            str(request["num_workers"]),
            "--base_rate",
            str(request["base_rate"]),
            "--ratio_weight",
            str(request["ratio_weight"]),
            "--lr",
            str(request["lr"]),
            "--warmup_epochs",
            str(request["warmup_epochs"]),
            "--warmup_steps",
            str(request["warmup_steps"]),
            "--clip_grad",
            str(request["clip_grad"]),
            "--device",
            request["device"],
            "--model_ema",
            _bool_str(request["model_ema"]),
            "--save_ckpt",
            _bool_str(request["save_ckpt"]),
            "--save_ckpt_freq",
            str(request["save_ckpt_freq"]),
            "--save_ckpt_num",
            str(request["save_ckpt_num"]),
            "--auto_resume",
            _bool_str(request["auto_resume"]),
            "--use_amp",
            _bool_str(request["use_amp"]),
            "--mixup",
            str(request["mixup"]),
            "--cutmix",
            str(request["cutmix"]),
            "--seed",
            str(request["seed"]),
            "--lr_scale",
            str(request["lr_scale"]),
            "--groupa_lr_scale",
            str(request["groupa_lr_scale"]),
            "--activation_lr_scale",
            str(request["activation_lr_scale"]),
            "--cls_distill_weight",
            str(request["cls_distill_weight"]),
            "--token_distill_weight",
            str(request["token_distill_weight"]),
            "--square_activation_mode",
            request["square_activation_mode"],
            "--approx_attn_mode",
            request["approx_attn_mode"],
            "--use_square_gelu",
            _bool_str(request["use_square_gelu"]),
            "--use_approx_attn",
            _bool_str(request["use_approx_attn"]),
            "--use_mask_pruning",
            _bool_str(request["use_mask_pruning"]),
            "--eval_pruning_mode",
            request["eval_pruning_mode"],
            "--eval_tie_policy",
            request["eval_tie_policy"],
            "--inference_friendly_ops",
            _bool_str(request["inference_friendly_ops"]),
            "--patch_embed_bias_init_mode",
            request["patch_embed_bias_init_mode"],
            "--freeze_patch_embed_proj",
            _bool_str(request["freeze_patch_embed_proj"]),
            "--pretrained_fix_step",
            str(request["pretrained_fix_step"]),
            "--secure_static_train_depth",
            str(request["secure_static_train_depth"]),
            "--secure_static_skip_pruning",
            _bool_str(request["secure_static_skip_pruning"]),
        ]
        if request["teacher_checkpoint_path"]:
            command.extend(["--teacher_checkpoint_path", request["teacher_checkpoint_path"]])
        if request["finetune"]:
            command.extend(["--finetune", request["finetune"]])
        return command

    def _build_threshold_search_command(self, request: dict[str, Any], run_dir: Path) -> list[str]:
        checkpoint_arg = relative_to_repo_path(run_dir / "checkpoint-best.pth")
        output_arg = relative_to_repo_path(run_dir / "threshold_best.json")
        return [
            self.config.python_bin,
            "tools/transshield_binary_threshold_search.py",
            "search",
            "--checkpoint",
            checkpoint_arg,
            "--data-path",
            request["eval_data_path"],
            "--device",
            request["threshold_device"],
            "--batch-size",
            str(request["threshold_batch_size"]),
            "--num-workers",
            str(request["threshold_num_workers"]),
            "--output-json",
            output_arg,
        ]

    def _build_threshold_eval_command(self, request: dict[str, Any], run_dir: Path) -> list[str]:
        checkpoint_arg = relative_to_repo_path(run_dir / "checkpoint-best.pth")
        threshold_arg = relative_to_repo_path(run_dir / "threshold_best.json")
        output_arg = relative_to_repo_path(run_dir / "threshold_eval.json")
        return [
            self.config.python_bin,
            "tools/transshield_binary_threshold_search.py",
            "eval",
            "--checkpoint",
            checkpoint_arg,
            "--threshold-json",
            threshold_arg,
            "--data-path",
            request["eval_data_path"],
            "--device",
            request["threshold_device"],
            "--batch-size",
            str(request["threshold_batch_size"]),
            "--num-workers",
            str(request["threshold_num_workers"]),
            "--output-json",
            output_arg,
        ]

    def _build_export_command(
        self,
        request: dict[str, Any],
        run_dir: Path,
        bundle_dir: Path,
        train_command: list[str],
        threshold_search_command: list[str],
        threshold_eval_command: list[str],
    ) -> list[str]:
        return [
            self.config.python_bin,
            "tools/freeze_export_candidate.py",
            "--source-dir",
            relative_to_repo_path(run_dir),
            "--output-dir",
            relative_to_repo_path(bundle_dir),
            "--train-command",
            " ".join(train_command),
            "--threshold-search-command",
            " ".join(threshold_search_command),
            "--eval-command",
            " ".join(threshold_eval_command),
        ]

    def _build_summary(self, job_id: str) -> dict:
        job = self._jobs[job_id]
        run_dir = resolve_repo_path(job["output_dir"])
        bundle_dir = resolve_repo_path(job["bundle_dir"])
        threshold_best_path = run_dir / "threshold_best.json"
        threshold_eval_path = run_dir / "threshold_eval.json"
        threshold_best = _read_json(threshold_best_path) if threshold_best_path.exists() else {}
        threshold_eval = _read_json(threshold_eval_path) if threshold_eval_path.exists() else {}
        return {
            "job_id": job_id,
            "completed_at": _now_ts(),
            "artifacts": {
                "run_dir": relative_to_repo_path(run_dir),
                "checkpoint_best": (
                    relative_to_repo_path(run_dir / "checkpoint-best.pth")
                    if (run_dir / "checkpoint-best.pth").exists()
                    else None
                ),
                "threshold_best_json": relative_to_repo_path(threshold_best_path) if threshold_best_path.exists() else None,
                "threshold_eval_json": relative_to_repo_path(threshold_eval_path) if threshold_eval_path.exists() else None,
                "bundle_dir": relative_to_repo_path(bundle_dir) if bundle_dir.exists() else None,
                "bundle_manifest": (
                    relative_to_repo_path(bundle_dir / "manifest.json")
                    if (bundle_dir / "manifest.json").exists()
                    else None
                ),
                "bundle_args_snapshot": (
                    relative_to_repo_path(bundle_dir / "args_snapshot.json")
                    if (bundle_dir / "args_snapshot.json").exists()
                    else None
                ),
            },
            "metrics_summary": _metrics_payload(threshold_best=threshold_best, threshold_eval=threshold_eval),
        }
