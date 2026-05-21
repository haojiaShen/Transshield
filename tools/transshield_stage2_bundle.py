import importlib
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BUNDLE_MODEL_STATE_NAME = "modified_plaintext_model_state_dict.pth"
DEFAULT_BUNDLE_EMA_MODEL_STATE_NAME = "modified_plaintext_model_state_dict_ema.pth"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_bundle_dir(bundle_dir) -> Path:
    return Path(bundle_dir).expanduser().resolve()


def resolve_model_state_dict_path(bundle_dir: Path) -> Path:
    bundle_dir = resolve_bundle_dir(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        export = manifest.get("export") or {}
        for key in ("model_state_dict_path", "student_model_state_dict_path"):
            raw = export.get(key)
            if raw:
                candidate = bundle_dir / Path(raw).name
                if candidate.is_file():
                    return candidate
    for candidate_name in (DEFAULT_BUNDLE_MODEL_STATE_NAME, DEFAULT_BUNDLE_EMA_MODEL_STATE_NAME):
        candidate = bundle_dir / candidate_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model state_dict found under {bundle_dir}")


def resolve_threshold_payload(bundle_dir: Path):
    bundle_dir = resolve_bundle_dir(bundle_dir)
    threshold_path = bundle_dir / "threshold_best.json"
    if threshold_path.is_file():
        return load_json(threshold_path)

    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        threshold_metrics = ((manifest.get("primary") or {}).get("threshold_metrics")) or {}
        threshold = threshold_metrics.get("eval_binary_threshold")
        if threshold is None:
            threshold = threshold_metrics.get("best_threshold")
        if threshold is not None:
            return {
                "eval_binary_threshold": float(threshold),
                "best_threshold": float(threshold),
                "eval_acc1": threshold_metrics.get("eval_acc1"),
                "best_threshold_acc": threshold_metrics.get("best_threshold_acc"),
                "auc": threshold_metrics.get("auc"),
                "sample_count": threshold_metrics.get("sample_count"),
                "source": "manifest.primary.threshold_metrics",
            }
    raise FileNotFoundError(f"no threshold payload found under {bundle_dir}")


def resolve_threshold(bundle_dir: Path, threshold_json: Optional[str]):
    if threshold_json:
        payload = load_json(Path(threshold_json).expanduser().resolve())
    else:
        payload = resolve_threshold_payload(bundle_dir)
    threshold = payload.get("eval_binary_threshold")
    if threshold is None:
        threshold = payload.get("best_threshold")
    if threshold is None:
        raise KeyError(f"threshold payload missing eval_binary_threshold/best_threshold: {payload}")
    return float(threshold)


def _ensure_torch_six():
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.inf = math.inf
        sys.modules["torch._six"] = torch_six


def import_repo_modules(repo_root: Path):
    _ensure_torch_six()
    sys.path.insert(0, str(repo_root))
    try:
        datasets_mod = importlib.import_module("datasets")
        dyvit_mod = importlib.import_module("models.dyvit")
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)
    return datasets_mod, dyvit_mod


def infer_model_size(model_name: str):
    name = (model_name or "").lower()
    if any(token in name for token in ["deit-b", "vit_deit_base", "base"]):
        return {"embed_dim": 768, "depth": 12, "num_heads": 12}
    return {"embed_dim": 384, "depth": 12, "num_heads": 6}


def build_model(args_snapshot: dict, model_cls):
    model_size = infer_model_size(args_snapshot.get("model", "deit-s"))
    base_rate = float(args_snapshot.get("base_rate", 0.7))
    keep_rate = [base_rate, base_rate**2, base_rate**3]

    kwargs = {
        "patch_size": 16,
        "embed_dim": model_size["embed_dim"],
        "depth": model_size["depth"],
        "num_heads": model_size["num_heads"],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "num_classes": int(args_snapshot.get("nb_classes", 2)),
        "pruning_loc": [3, 6, 9],
        "token_ratio": keep_rate,
        "distill": True,
    }

    signature = model_cls.__init__.__code__.co_varnames
    if "act_layer" in signature:
        kwargs["act_layer"] = (
            args_snapshot.get("square_activation_mode", "gelu")
            if bool(args_snapshot.get("use_square_gelu", False))
            else "gelu"
        )
    if "use_mask_pruning" in signature:
        kwargs["use_mask_pruning"] = bool(args_snapshot.get("use_mask_pruning", False))
    if "use_approx_attn" in signature:
        kwargs["use_approx_attn"] = bool(args_snapshot.get("use_approx_attn", False))
    if "approx_attn_mode" in signature:
        kwargs["approx_attn_mode"] = args_snapshot.get("approx_attn_mode", "relu")
    if "fp32_attention" in signature:
        kwargs["fp32_attention"] = True
    if "eval_pruning_mode" in signature:
        kwargs["eval_pruning_mode"] = args_snapshot.get("eval_pruning_mode", "topk_argsort")
    if "eval_tie_policy" in signature:
        kwargs["eval_tie_policy"] = args_snapshot.get("eval_tie_policy", "lowest_index")
    if "secure_static_depth" in signature:
        kwargs["secure_static_depth"] = int(args_snapshot.get("secure_static_train_depth", 0) or 0)
    if "secure_static_skip_pruning" in signature:
        kwargs["secure_static_skip_pruning"] = bool(args_snapshot.get("secure_static_skip_pruning", True))
    if "nonempty_keep_guard" in signature and "nonempty_keep_guard" in args_snapshot:
        kwargs["nonempty_keep_guard"] = bool(args_snapshot.get("nonempty_keep_guard"))

    return model_cls(**kwargs)


def build_eval_transform(args_snapshot: dict, build_transform_fn):
    transform_args = SimpleNamespace(
        input_size=int(args_snapshot.get("input_size", 224)),
        imagenet_default_mean_and_std=bool(args_snapshot.get("imagenet_default_mean_and_std", True)),
        crop_pct=args_snapshot.get("crop_pct"),
        color_jitter=args_snapshot.get("color_jitter", 0.4),
        aa=args_snapshot.get("aa", "rand-m9-mstd0.5-inc1"),
        train_interpolation=args_snapshot.get("train_interpolation", "bicubic"),
        reprob=float(args_snapshot.get("reprob", 0.25)),
        remode=args_snapshot.get("remode", "pixel"),
        recount=int(args_snapshot.get("recount", 1)),
    )
    return build_transform_fn(is_train=False, args=transform_args)


def load_frozen_bundle(bundle_dir: Path, device="cpu"):
    bundle_dir = resolve_bundle_dir(bundle_dir)
    args_snapshot = load_json(bundle_dir / "args_snapshot.json")
    datasets_mod, dyvit_mod = import_repo_modules(REPO_ROOT)
    model = build_model(args_snapshot, dyvit_mod.VisionTransformerDiffPruning).to(device)
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise ValueError(
            f"non-strict load result: missing={getattr(load_result, 'missing_keys', None)} "
            f"unexpected={getattr(load_result, 'unexpected_keys', None)}"
        )
    model.eval()
    transform = build_eval_transform(args_snapshot, datasets_mod.build_transform)
    threshold = resolve_threshold(bundle_dir, None)
    return {
        "bundle_dir": str(bundle_dir),
        "args_snapshot": args_snapshot,
        "model": model,
        "transform": transform,
        "threshold": threshold,
        "model_state_dict_path": str(state_dict_path),
    }


def preprocess_image(image_path, transform, device="cpu"):
    image_path = Path(image_path).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image_path, tensor


def postprocess_binary_output(probabilities, threshold=None):
    probs = probabilities.detach().cpu().float()
    if probs.ndim == 2:
        probs = probs[0]
    argmax_class = int(probs.argmax().item())
    class1_probability = float(probs[1].item()) if probs.numel() >= 2 else None
    threshold_class = None
    if threshold is not None and class1_probability is not None:
        threshold_class = int(class1_probability >= float(threshold))
    return {
        "argmax_class": argmax_class,
        "argmax_confidence": float(probs[argmax_class].item()),
        "class1_probability": class1_probability,
        "threshold": None if threshold is None else float(threshold),
        "threshold_class": threshold_class,
    }
