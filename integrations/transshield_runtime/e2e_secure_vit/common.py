import json
# =============================================================================
# Transshield E2E 公共工具
# =============================================================================
# 角色映射：
#   "client" → 数据使用方（如医院），提交影像数据，获取诊断结果
#   "server" → 模型提供方的推理服务（内部含 P0/P1 两台 MPC 服务器）
# =============================================================================

from pathlib import Path


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_existing_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(
            f"[e2e-secure-vit] missing {label}: {path}\n"
            "Set E2E_INPUT_PT to an existing client pixel package, or create one with "
            "`tools/transshield_e2e_secure_infer.py client-preprocess` before running."
        )


def path_for_output(path, *, redact: bool, redaction_label: str):
    if path is None:
        return None
    if redact:
        return f"[redacted:{redaction_label}]"
    return str(path)


def tensor_stats(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def debug_tensor_record(tensor):
    import torch

    value = torch.as_tensor(tensor).detach().cpu().float()
    record = tensor_stats(value)
    record["abs_mean"] = float(value.abs().mean().item())
    record["l2_norm"] = float(torch.linalg.vector_norm(value).item())
    record["values"] = value.tolist()
    return record


def record_to_float_tensor(record):
    import torch

    return torch.as_tensor(record["values"], dtype=torch.float32)


def compare_debug_records(reference_record, candidate_record):
    import torch

    reference_value = record_to_float_tensor(reference_record)
    candidate_value = record_to_float_tensor(candidate_record)
    if tuple(reference_value.shape) != tuple(candidate_value.shape):
        raise ValueError(
            f"debug tensor shape mismatch: {tuple(reference_value.shape)} vs {tuple(candidate_value.shape)}"
        )

    abs_error = (reference_value - candidate_value).abs()
    reference_flat = reference_value.reshape(-1)
    candidate_flat = candidate_value.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        reference_flat.unsqueeze(0),
        candidate_flat.unsqueeze(0),
        dim=1,
        eps=1e-12,
    )
    return {
        "shape": list(reference_value.shape),
        "max_abs_error": float(abs_error.max().item()),
        "mean_abs_error": float(abs_error.mean().item()),
        "l2_error": float(torch.linalg.vector_norm(reference_value - candidate_value).item()),
        "reference_l2_norm": float(torch.linalg.vector_norm(reference_value).item()),
        "candidate_l2_norm": float(torch.linalg.vector_norm(candidate_value).item()),
        "cosine_similarity": float(cosine.item()),
    }


def numpy_from_torch_tensor(tensor):
    import numpy as np

    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
