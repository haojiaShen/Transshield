from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time

from showcase_api.admin_config import ShowcaseConfig


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AdminAuthManager:
    def __init__(self, config: ShowcaseConfig):
        self.config = config
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, float | str]] = {}
        self._auth_file = config.admin_auth_file
        self._auth_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_auth_file()

    def _ensure_auth_file(self) -> None:
        if self._auth_file.exists():
            return
        payload = {
            "username": self.config.admin_username,
            "password_sha256": _sha256_hex(self.config.admin_initial_password),
            "updated_at": int(time.time()),
        }
        self._auth_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _read_auth_payload(self) -> dict[str, str | int]:
        self._ensure_auth_file()
        return json.loads(self._auth_file.read_text(encoding="utf-8"))

    def verify_credentials(self, username: str, password: str) -> bool:
        payload = self._read_auth_payload()
        return (
            username == payload.get("username")
            and _sha256_hex(password) == payload.get("password_sha256")
        )

    def create_session(self, username: str) -> dict[str, str | int]:
        token = secrets.token_hex(24)
        now = int(time.time())
        expires_at = now + self.config.admin_session_ttl_seconds
        with self._lock:
            self._sessions[token] = {
                "username": username,
                "created_at": now,
                "expires_at": expires_at,
            }
        return {
            "token": token,
            "username": username,
            "created_at": now,
            "expires_at": expires_at,
        }

    def get_session(self, token: str | None) -> dict[str, str | int] | None:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            now = int(time.time())
            expires_at = int(session["expires_at"])
            if expires_at <= now:
                self._sessions.pop(token, None)
                return None
            return {
                "token": token,
                "username": str(session["username"]),
                "created_at": int(session["created_at"]),
                "expires_at": expires_at,
            }

    def destroy_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def change_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        if len(new_password.strip()) < 8:
            return False, "新密码长度至少 8 位。"
        with self._lock:
            payload = self._read_auth_payload()
            if username != payload.get("username"):
                return False, "管理员账号不存在。"
            if _sha256_hex(old_password) != payload.get("password_sha256"):
                return False, "原密码不正确。"
            payload["password_sha256"] = _sha256_hex(new_password)
            payload["updated_at"] = int(time.time())
            self._auth_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return True, "管理员密码已更新。"

    def session_payload(self, token: str | None) -> dict[str, str | int] | None:
        session = self.get_session(token)
        if not session:
            return None
        return {
            "username": str(session["username"]),
            "display_name": self.config.admin_display_name,
            "created_at": int(session["created_at"]),
            "expires_at": int(session["expires_at"]),
        }
