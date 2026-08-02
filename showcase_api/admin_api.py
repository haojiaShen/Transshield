from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from showcase_api.admin_auth import AdminAuthManager
from showcase_api.admin_catalog import build_overview, build_results_catalog, list_model_artifacts
from showcase_api.admin_config import (
    build_path_replacements,
    load_showcase_config,
    relative_to_repo_path,
    relativize_value,
)
from showcase_api.admin_jobs import TrainingJobManager


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class CreateTrainingJobRequest(BaseModel):
    name: str = Field(default="管理员训练任务")
    mode: str = Field(default="preset")
    preset_id: Optional[str] = None
    parameters: dict = Field(default_factory=dict)


CONFIG = load_showcase_config()
AUTH_MANAGER = AdminAuthManager(CONFIG)
JOB_MANAGER = TrainingJobManager(CONFIG)
router = APIRouter(prefix="/api/admin", tags=["admin"])


def start_admin_jobs() -> None:
    JOB_MANAGER.start()


def stop_admin_jobs() -> None:
    JOB_MANAGER.stop()


def get_admin_token(request: Request) -> str | None:
    header_token = request.headers.get("x-admin-session", "").strip()
    if header_token:
        return header_token
    return request.cookies.get(CONFIG.admin_session_cookie_name)


def require_admin(request: Request) -> dict:
    session = AUTH_MANAGER.session_payload(get_admin_token(request))
    if not session:
        raise HTTPException(status_code=401, detail="管理员登录状态已失效。")
    return session


@router.post("/login")
async def admin_login(payload: LoginRequest, response: Response):
    if not AUTH_MANAGER.verify_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="账号或密码不正确。")
    session = AUTH_MANAGER.create_session(payload.username)
    response.set_cookie(
        key=CONFIG.admin_session_cookie_name,
        value=str(session["token"]),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=CONFIG.admin_session_ttl_seconds,
    )
    return {
        "status": "ok",
        "token": str(session["token"]),
        "session": AUTH_MANAGER.session_payload(str(session["token"])),
    }


@router.post("/logout")
async def admin_logout(request: Request, response: Response):
    AUTH_MANAGER.destroy_session(get_admin_token(request))
    response.delete_cookie(CONFIG.admin_session_cookie_name)
    return {"status": "ok"}


@router.get("/session")
async def admin_session(request: Request):
    session = AUTH_MANAGER.session_payload(get_admin_token(request))
    if not session:
        raise HTTPException(status_code=401, detail="未登录。")
    return {"status": "ok", "session": session}


@router.post("/change-password")
async def admin_change_password(request: Request, payload: ChangePasswordRequest):
    session = require_admin(request)
    ok, message = AUTH_MANAGER.change_password(
        str(session["username"]),
        payload.old_password,
        payload.new_password,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@router.get("/overview")
async def admin_overview(request: Request):
    require_admin(request)
    jobs = JOB_MANAGER.list_jobs()
    models = list_model_artifacts(CONFIG)
    return {"status": "ok", "overview": build_overview(CONFIG, jobs, models)}


@router.get("/train/jobs")
async def admin_train_jobs(request: Request):
    require_admin(request)
    return {"status": "ok", "jobs": JOB_MANAGER.list_jobs()}


@router.post("/train/jobs")
async def admin_create_train_job(request: Request, payload: CreateTrainingJobRequest):
    require_admin(request)
    try:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        job = JOB_MANAGER.create_job(payload_dict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "job": job}


@router.get("/train/jobs/{job_id}")
async def admin_get_train_job(job_id: str, request: Request):
    require_admin(request)
    job = JOB_MANAGER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return {"status": "ok", "job": job}


@router.post("/train/jobs/{job_id}/cancel")
async def admin_cancel_train_job(job_id: str, request: Request):
    require_admin(request)
    ok, message = JOB_MANAGER.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@router.get("/train/jobs/{job_id}/log")
async def admin_get_train_job_log(job_id: str, request: Request):
    require_admin(request)
    payload = JOB_MANAGER.get_job_log(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return {"status": "ok", "log": payload}


@router.get("/models")
async def admin_models(request: Request):
    require_admin(request)
    return {"status": "ok", "models": list_model_artifacts(CONFIG)}


@router.get("/results")
async def admin_results(request: Request):
    require_admin(request)
    return {"status": "ok", "results": build_results_catalog(CONFIG)}


@router.get("/system/config")
async def admin_system_config(request: Request):
    require_admin(request)
    path_replacements = build_path_replacements(CONFIG.default_train_data_path, CONFIG.default_eval_data_path)
    return {
        "status": "ok",
        "config": {
            "repo_root": ".",
            "python_bin": relative_to_repo_path(CONFIG.python_bin),
            "job_root": relative_to_repo_path(CONFIG.admin_job_root),
            "train_output_root": relative_to_repo_path(CONFIG.train_output_root),
            "bundle_output_root": relative_to_repo_path(CONFIG.bundle_output_root),
            "default_train_data_path": relative_to_repo_path(CONFIG.default_train_data_path),
            "default_eval_data_path": relative_to_repo_path(CONFIG.default_eval_data_path),
            "default_device": CONFIG.default_device,
            "default_batch_size": CONFIG.default_batch_size,
            "default_num_workers": CONFIG.default_num_workers,
            "max_concurrent_train_jobs": CONFIG.max_concurrent_train_jobs,
            "runtime_mode": CONFIG.runtime_mode,
            "admin_display_name": CONFIG.admin_display_name,
            "training_presets": [
                {
                    "id": preset.id,
                    "name": preset.name,
                    "description": preset.description,
                    "parameters": relativize_value(preset.parameters, path_replacements),
                }
                for preset in CONFIG.training_presets
            ],
        },
    }
