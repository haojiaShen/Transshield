#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: str
    status_code: int | None = None


def make_sample_png() -> bytes:
    height, width = 80, 104
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = np.stack(
        [
            np.repeat(x, height, axis=0),
            np.repeat(y, width, axis=1),
            ((np.repeat(x, height, axis=0).astype(np.uint16) + np.repeat(y, width, axis=1).astype(np.uint16)) // 2).astype(
                np.uint8
            ),
        ],
        axis=2,
    )
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class ForwardCaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length)
        try:
            parsed_body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            parsed_body = None
        self.server.captured_requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "json": parsed_body,
            }
        )
        response = json.dumps({"status": "accepted"}, ensure_ascii=False).encode("utf-8")
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return


def start_forward_capture_server():
    server = HTTPServer(("127.0.0.1", 0), ForwardCaptureHandler)
    server.captured_requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def reset_split_gateway_module(
    role: str,
    storage_root: Path,
    auth_token: str,
    ai_gateway_url: str = "",
    ai_gateway_auth_token: str = "",
):
    os.environ.update(
        {
            "TRANSSHIELD_SPLIT_ROLE": role,
            "TRANSSHIELD_SPLIT_STORAGE_DIR": str(storage_root / role),
            "TRANSSHIELD_SPLIT_AUTH_TOKEN": auth_token,
            "TRANSSHIELD_SPLIT_RUNTIME_MODE": "mock",
            "TRANSSHIELD_SPLIT_PYTHON_BIN": sys.executable,
            "TRANSSHIELD_SPLIT_AI_GATEWAY_URL": ai_gateway_url,
            "TRANSSHIELD_SPLIT_AI_GATEWAY_AUTH_TOKEN": ai_gateway_auth_token,
            "TRANSSHIELD_SPLIT_GATEWAY_FORWARD_TIMEOUT_SEC": "1",
        }
    )
    module_name = "showcase_api.split_gateway"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def add_case(results: list[CaseResult], name: str, passed: bool, details: str, status_code: int | None = None):
    results.append(CaseResult(name=name, passed=passed, details=details, status_code=status_code))


def response_json(response):
    try:
        return response.json()
    except Exception:
        return None


def poll_task(client: TestClient, task_id: str, headers: dict[str, str], timeout_sec: float = 5.0) -> dict:
    deadline = time.time() + timeout_sec
    last_payload = {}
    while time.time() < deadline:
        response = client.get(f"/api/split/tasks/{task_id}", headers=headers)
        last_payload = response_json(response) or {}
        if last_payload.get("status") in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(0.05)
    return last_payload


def run_smoke(storage_root: Path, task_id: str, auth_token: str) -> dict:
    results: list[CaseResult] = []
    image_bytes = make_sample_png()
    headers = {"Authorization": f"Bearer {auth_token}"}
    image_headers = {**headers, "Content-Type": "image/png", "X-Source-Filename": "smoke.png"}

    forward_server, forward_url = start_forward_capture_server()
    try:
        hospital_module = reset_split_gateway_module(
            "hospital",
            storage_root,
            auth_token,
            ai_gateway_url=forward_url,
            ai_gateway_auth_token=auth_token,
        )
        with TestClient(hospital_module.app) as hospital_client:
            health = hospital_client.get("/api/split/health")
            add_case(results, "hospital_health", health.status_code == 200, "hospital role health endpoint is reachable", health.status_code)

            no_auth = hospital_client.post(
                f"/api/hospital/tasks/{task_id}/image",
                content=image_bytes,
                headers={"Content-Type": "image/png"},
            )
            add_case(results, "hospital_auth_required", no_auth.status_code == 401, "hospital image endpoint rejects missing auth", no_auth.status_code)

            hospital_image = hospital_client.post(
                f"/api/hospital/tasks/{task_id}/image",
                content=image_bytes,
                headers=image_headers,
            )
            hospital_payload = response_json(hospital_image) or {}
            public_manifest = hospital_payload.get("p2_share_delivery", {}).get("public_manifest", {})
            expected_shape = public_manifest.get("share_shape") == [1, 3, 224, 224]
            add_case(
                results,
                "hospital_image_split",
                hospital_image.status_code == 202 and expected_shape,
                "hospital image endpoint decodes PNG and returns P1/P2 split-share manifests",
                hospital_image.status_code,
            )

            forward_delivery = hospital_payload.get("p2_forward_delivery", {})
            captured = forward_server.captured_requests
            forwarded = (
                forward_delivery.get("status") == "accepted"
                and len(captured) == 1
                and captured[0]["path"] == f"/api/split/tasks/{task_id}/share"
                and captured[0]["authorization"] == f"Bearer {auth_token}"
                and isinstance(captured[0]["json"], dict)
                and captured[0]["json"].get("share_sha256") == hospital_payload["p2_share_delivery"]["share_sha256"]
            )
            add_case(
                results,
                "hospital_p2_forwarded",
                forwarded,
                "hospital image endpoint forwards P2 share when TRANSSHIELD_SPLIT_AI_GATEWAY_URL is configured",
                forward_delivery.get("http_status"),
            )
    finally:
        forward_server.shutdown()
        forward_server.server_close()

    p2_delivery = hospital_payload["p2_share_delivery"]
    p1_manifest = hospital_payload["p1_party_manifest"]

    ai_module = reset_split_gateway_module("ai", storage_root, auth_token)
    with TestClient(ai_module.app) as ai_client:
        wrong_role = ai_client.post(f"/api/hospital/tasks/{task_id}/image", content=image_bytes, headers=image_headers)
        add_case(results, "ai_wrong_role_blocked", wrong_role.status_code == 403, "AI role cannot call hospital image endpoint", wrong_role.status_code)

        wrong_task_payload = copy.deepcopy(p2_delivery)
        wrong_task_payload["public_manifest"]["task_id"] = f"{task_id}_other"
        wrong_task = ai_client.post(f"/api/split/tasks/{task_id}/share", json=wrong_task_payload, headers=headers)
        add_case(
            results,
            "manifest_task_id_mismatch_blocked",
            wrong_task.status_code == 422,
            "share endpoint rejects public manifest whose task_id differs from the URL",
            wrong_task.status_code,
        )

        wrong_payload_task = copy.deepcopy(p2_delivery)
        wrong_payload_task["task_id"] = f"{task_id}_other"
        wrong_payload_task_response = ai_client.post(f"/api/split/tasks/{task_id}/share", json=wrong_payload_task, headers=headers)
        add_case(
            results,
            "payload_task_id_mismatch_blocked",
            wrong_payload_task_response.status_code == 422,
            "share endpoint rejects payload task_id that differs from the URL",
            wrong_payload_task_response.status_code,
        )

        bad_hash_payload = copy.deepcopy(p2_delivery)
        bad_hash_payload["share_sha256"] = "0" * 64
        bad_hash = ai_client.post(f"/api/split/tasks/{task_id}/share", json=bad_hash_payload, headers=headers)
        add_case(results, "share_hash_mismatch_blocked", bad_hash.status_code == 422, "AI share endpoint rejects hash mismatch", bad_hash.status_code)

        ai_share = ai_client.post(f"/api/split/tasks/{task_id}/share", json=p2_delivery, headers=headers)
        ai_share_payload = response_json(ai_share) or {}
        add_case(results, "ai_share_received", ai_share.status_code == 202, "AI role stores only the P2 share manifest", ai_share.status_code)

        model_manifest_request = {
            "model_version": "smoke-model",
            "spu_params_mode": "secret",
        }
        model_response = ai_client.post(
            f"/api/ai/tasks/{task_id}/model-manifest",
            json=model_manifest_request,
            headers=headers,
        )
        model_payload = response_json(model_response) or {}
        add_case(
            results,
            "ai_model_manifest_received",
            model_response.status_code == 202,
            "AI role records a model manifest for the coordinator",
            model_response.status_code,
        )

    p2_manifest = ai_share_payload["party_manifest"]
    model_manifest = model_payload["model_manifest"]

    coordinator_module = reset_split_gateway_module("coordinator", storage_root, auth_token)
    with TestClient(coordinator_module.app) as coordinator_client:
        bad_run_manifest = copy.deepcopy(p2_manifest)
        bad_run_manifest["task_id"] = f"{task_id}_other"
        bad_run_request = {
            "runtime_mode": "mock",
            "public_manifest": p2_delivery["public_manifest"],
            "p1_share_manifest": p1_manifest,
            "p2_share_manifest": bad_run_manifest,
            "model_manifest": model_manifest,
            "max_samples": 1,
        }
        bad_run_response = coordinator_client.post(f"/api/coordinator/tasks/{task_id}/runs", json=bad_run_request, headers=headers)
        add_case(
            results,
            "coordinator_manifest_task_id_mismatch_blocked",
            bad_run_response.status_code == 422,
            "coordinator rejects split manifests whose task_id differs from the URL",
            bad_run_response.status_code,
        )

        run_request = {
            "runtime_mode": "mock",
            "public_manifest": p2_delivery["public_manifest"],
            "p1_share_manifest": p1_manifest,
            "p2_share_manifest": p2_manifest,
            "model_manifest": model_manifest,
            "max_samples": 1,
        }
        run_response = coordinator_client.post(f"/api/coordinator/tasks/{task_id}/runs", json=run_request, headers=headers)
        add_case(results, "coordinator_run_queued", run_response.status_code == 202, "coordinator accepts split manifests and queues a run", run_response.status_code)

        final_state = poll_task(coordinator_client, task_id, headers)
        completed = final_state.get("status") == "completed"
        params_mode = final_state.get("artifacts", {}).get("spu_params_mode")
        add_case(
            results,
            "coordinator_mock_completed",
            completed and params_mode == "secret",
            "coordinator mock run completes and preserves the AI model manifest params mode",
            None,
        )

    return {
        "task_id": task_id,
        "storage_root": str(storage_root),
        "passed": all(item.passed for item in results),
        "cases": [asdict(item) for item in results],
        "final_state": final_state,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a local split-gateway hospital/AI/coordinator smoke flow.")
    parser.add_argument("--storage-root", default="", help="Optional directory for smoke artifacts. Defaults to a temporary directory.")
    parser.add_argument("--task-id", default="", help="Optional task id. Defaults to a generated smoke id.")
    parser.add_argument("--auth-token", default="smoke-token")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--keep-state", action="store_true", help="Keep the temporary storage directory when --storage-root is not set.")
    args = parser.parse_args()

    task_id = args.task_id or f"smoke_{uuid.uuid4().hex[:12]}"
    if args.storage_root:
        storage_root = Path(args.storage_root).expanduser().resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        payload = run_smoke(storage_root, task_id, args.auth_token)
    elif args.keep_state:
        storage_root = Path(tempfile.mkdtemp(prefix="transshield_split_gateway_smoke_keep_")).resolve()
        payload = run_smoke(storage_root, task_id, args.auth_token)
    else:
        with tempfile.TemporaryDirectory(prefix="transshield_split_gateway_smoke_") as tmp_dir:
            storage_root = Path(tmp_dir).resolve()
            payload = run_smoke(storage_root, task_id, args.auth_token)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
