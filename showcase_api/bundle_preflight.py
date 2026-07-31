from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODEL_STATE_NAMES = (
    "modified_plaintext_model_state_dict.pth",
    "modified_plaintext_model_state_dict_ema.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_bundle(bundle_dir: Path, *, verify_hash: bool = False) -> dict:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "manifest.json"
    manifest = None
    manifest_error = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            manifest_error = f"{type(error).__name__}: {error}"

    candidates: list[Path] = []
    expected_sha256 = None
    if isinstance(manifest, dict):
        export = manifest.get("export") or {}
        for key in ("model_state_dict_path", "student_model_state_dict_path"):
            raw_path = export.get(key)
            if raw_path:
                candidates.append(bundle_dir / Path(str(raw_path)).name)
        expected_sha256 = export.get("model_state_dict_sha256") or export.get("student_model_state_dict_sha256")
    candidates.extend(bundle_dir / name for name in MODEL_STATE_NAMES)

    model_state_path = next((path for path in candidates if path.is_file()), None)
    actual_sha256 = sha256_file(model_state_path) if verify_hash and model_state_path is not None else None
    hash_matches = None
    if actual_sha256 is not None and expected_sha256:
        hash_matches = actual_sha256.lower() == str(expected_sha256).lower()

    return {
        "bundle_dir": str(bundle_dir),
        "bundle_dir_present": bundle_dir.is_dir(),
        "manifest_present": manifest_path.is_file(),
        "manifest_error": manifest_error,
        "model_state_present": model_state_path is not None,
        "model_state_file": model_state_path.name if model_state_path is not None else None,
        "expected_model_sha256": expected_sha256,
        "actual_model_sha256": actual_sha256,
        "model_sha256_matches": hash_matches,
        "ready": bool(
            bundle_dir.is_dir()
            and manifest_error is None
            and model_state_path is not None
            and hash_matches is not False
        ),
    }


def require_bundle_ready(bundle_dir: Path, *, verify_hash: bool = True) -> dict:
    status = inspect_bundle(bundle_dir, verify_hash=verify_hash)
    if not status["bundle_dir_present"]:
        raise RuntimeError(f"bundle directory is missing: {status['bundle_dir']}")
    if status["manifest_error"]:
        raise RuntimeError(f"bundle manifest is invalid: {status['manifest_error']}")
    if not status["model_state_present"]:
        raise RuntimeError(
            "bundle model state is missing; provide modified_plaintext_model_state_dict.pth "
            f"under {status['bundle_dir']} before starting SPU mode"
        )
    if status["model_sha256_matches"] is False:
        raise RuntimeError(
            "bundle model state SHA-256 does not match manifest.json: "
            f"expected {status['expected_model_sha256']}, got {status['actual_model_sha256']}"
        )
    return status
