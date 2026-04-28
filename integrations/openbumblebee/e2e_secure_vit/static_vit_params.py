from pathlib import Path

from integrations.openbumblebee.e2e_secure_vit.common import numpy_from_torch_tensor


STATIC_FORWARD_SCOPE = "student_patch_embed_blocks_head_without_runtime_pruning_predictor_path"


def scalar_state_value(state_dict, key, default):
    import numpy as np

    value = state_dict.get(key)
    if value is None:
        return np.asarray(default, dtype=np.float32)
    return np.asarray(float(value.detach().cpu().item()), dtype=np.float32)


def softplus_scalar(value):
    import numpy as np

    return np.asarray(np.logaddexp(float(value), 0.0), dtype=np.float32)


def resolve_block_activation_params(state_dict, block_index: int, activation_kind: str):
    import numpy as np

    prefix = f"blocks.{block_index}.mlp.act"
    if activation_kind == "gelu":
        return np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)
    if activation_kind == "fixed_square":
        alpha = scalar_state_value(state_dict, f"{prefix}.fixed_alpha", 0.25)
        return alpha, np.asarray(0.0, dtype=np.float32)
    if activation_kind == "learnable_square":
        raw_alpha = scalar_state_value(state_dict, f"{prefix}.raw_alpha", 1.0)
        return softplus_scalar(raw_alpha), np.asarray(0.0, dtype=np.float32)
    if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
        alpha = scalar_state_value(state_dict, f"{prefix}.alpha", 0.0)
        beta = scalar_state_value(state_dict, f"{prefix}.beta", 1.0)
        return alpha, beta
    raise ValueError(f"unsupported activation kind for SPU backend: {activation_kind}")


def resolve_static_activation_kind(args_snapshot):
    if not bool(args_snapshot.get("use_square_gelu", False)):
        return "gelu"
    activation_kind = str(args_snapshot.get("square_activation_mode", "fixed_square"))
    if activation_kind in {"square", "fixed_square"}:
        return "fixed_square"
    return activation_kind


def normalize_depth_limit(raw_depth_limit: int, full_depth: int = 12) -> int:
    raw_depth_limit = int(raw_depth_limit)
    if raw_depth_limit < 0:
        return int(full_depth)
    return max(0, min(raw_depth_limit, int(full_depth)))


def resolve_spu_activation_kind(base_activation_kind: str, activation_override: str) -> str:
    activation_override = str(activation_override)
    if activation_override == "bundle":
        return base_activation_kind
    if activation_override in {
        "gelu",
        "fixed_square",
        "learnable_square",
        "learnable_quadratic",
        "learnable_quadratic_gelu_init",
    }:
        return activation_override
    raise ValueError(f"unsupported SPU activation override: {activation_override}")


def load_static_vit_spu_params(
    bundle_dir: Path,
    static_depth_limit: int = -1,
    attention_policy: str = "smoothed",
    activation_override: str = "bundle",
):
    import torch

    from tools.transshield_stage2_bundle import load_json as load_stage2_json
    from tools.transshield_stage2_bundle import resolve_model_state_dict_path

    args_snapshot = load_stage2_json(bundle_dir / "args_snapshot.json")
    if args_snapshot.get("model") != "deit-s":
        raise NotImplementedError(
            f"SPU whole-forward backend currently supports deit-s only, got {args_snapshot.get('model')}"
        )
    if bool(args_snapshot.get("use_approx_attn", False)):
        raise NotImplementedError("SPU whole-forward backend does not yet support approximate attention")

    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    base_activation_kind = resolve_static_activation_kind(args_snapshot)
    activation_kind = resolve_spu_activation_kind(base_activation_kind, activation_override)
    attention_policy = str(attention_policy)
    if attention_policy not in {"smoothed", "standard", "uniform", "identity"}:
        raise ValueError(f"unsupported SPU attention policy: {attention_policy}")

    full_depth = 12
    depth = normalize_depth_limit(static_depth_limit, full_depth=full_depth)
    num_heads = 6
    embed_dim = 384
    patch_size = 16
    head_dim = embed_dim // num_heads

    def required(key):
        if key not in state_dict:
            raise KeyError(f"missing state_dict key required by SPU static forward: {key}")
        return numpy_from_torch_tensor(state_dict[key])

    block_params = []
    for block_index in range(depth):
        act_alpha, act_beta = resolve_block_activation_params(state_dict, block_index, activation_kind)
        prefix = f"blocks.{block_index}"
        block_params.append(
            (
                required(f"{prefix}.norm1.weight"),
                required(f"{prefix}.norm1.bias"),
                required(f"{prefix}.attn.qkv.weight"),
                required(f"{prefix}.attn.qkv.bias"),
                required(f"{prefix}.attn.proj.weight"),
                required(f"{prefix}.attn.proj.bias"),
                required(f"{prefix}.norm2.weight"),
                required(f"{prefix}.norm2.bias"),
                required(f"{prefix}.mlp.fc1.weight"),
                required(f"{prefix}.mlp.fc1.bias"),
                act_alpha,
                act_beta,
                required(f"{prefix}.mlp.fc2.weight"),
                required(f"{prefix}.mlp.fc2.bias"),
            )
        )

    params = (
        required("patch_embed.proj.weight"),
        required("patch_embed.proj.bias"),
        required("cls_token"),
        required("pos_embed"),
        tuple(block_params),
        required("norm.weight"),
        required("norm.bias"),
        required("head.weight"),
        required("head.bias"),
    )
    metadata = {
        "state_dict_path": str(state_dict_path),
        "base_activation_kind": base_activation_kind,
        "activation_kind": activation_kind,
        "activation_override": str(activation_override),
        "full_depth": full_depth,
        "depth": depth,
        "static_depth_limit": int(static_depth_limit),
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "head_dim": head_dim,
        "patch_size": patch_size,
        "layer_norm_eps": 1e-6,
        "attention_policy": attention_policy,
        "attention_policy_eps": 1e-6,
        "forward_scope": STATIC_FORWARD_SCOPE,
        "unsupported_currently_bypassed": [
            "runtime pruning predictor path",
            "intermediate feature reveal",
            "dynamic masking-pruning inside secure forward",
        ],
    }
    return params, metadata
