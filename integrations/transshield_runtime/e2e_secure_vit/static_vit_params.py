from pathlib import Path

from integrations.transshield_runtime.e2e_secure_vit.common import numpy_from_torch_tensor


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
        # Fallback: compute from scratch
        import numpy as np_lib
        x_min, x_max = -8.0, 8.0
        bp = np_lib.linspace(x_min, x_max, num_segments + 1).astype(np.float32)
        vals = (0.5 * bp * (1 + np_lib.tanh(np_lib.sqrt(2 / np_lib.pi) * (bp + 0.044715 * bp**3)))).astype(np.float32)
        return bp, vals
    raise ValueError(f"unsupported activation kind for SPU backend: {activation_kind}")


def resolve_static_activation_kind(args_snapshot):
    square_activation_mode = str(args_snapshot.get("square_activation_mode", ""))
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
    token_ratio_base_override: float = 0.0,
):
    import torch

    from tools.transshield_stage2_bundle import load_json as load_stage2_json
    from tools.transshield_stage2_bundle import resolve_model_state_dict_path

    args_snapshot = load_stage2_json(bundle_dir / "args_snapshot.json")
    # Support both deit-s (teacher) and deit-t (student) models
    model_type = args_snapshot.get("model", "deit-s")
    if model_type not in ("deit-s", "deit-t"):
        raise NotImplementedError(
            f"SPU whole-forward backend currently supports deit-s and deit-t only, got {model_type}"
        )
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    base_activation_kind = resolve_static_activation_kind(args_snapshot)
    activation_kind = resolve_spu_activation_kind(base_activation_kind, activation_override)
    bundle_base_rate = float(args_snapshot["base_rate"])
    if token_ratio_base_override > 0.0:
        base_rate = token_ratio_base_override
    else:
        base_rate = bundle_base_rate
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    pruning_loc = [3, 6, 9]
    attention_policy = str(attention_policy)
    if attention_policy not in {"smoothed", "standard", "uniform", "identity"}:
        raise ValueError(f"unsupported SPU attention policy: {attention_policy}")
    if bool(args_snapshot.get("use_approx_attn", False)):
        approx_attn_mode = str(args_snapshot.get("approx_attn_mode", "relu"))
        if approx_attn_mode != "uniform" or attention_policy != "uniform":
            raise NotImplementedError(
                "SPU whole-forward only supports trained approximate attention when "
                "args_snapshot.approx_attn_mode=uniform and runtime attention_policy=uniform"
            )

    # Read architecture parameters from args_snapshot (support both teacher and student models)
    full_depth = int(args_snapshot.get("depth", 12))
    depth = normalize_depth_limit(static_depth_limit, full_depth=full_depth)
    num_heads = int(args_snapshot.get("num_heads", 6))
    embed_dim = int(args_snapshot.get("embed_dim", 384))
    patch_size = int(args_snapshot.get("patch_size", 16))
    head_dim = embed_dim // num_heads

    def required(key):
        if key not in state_dict:
            raise KeyError(f"missing state_dict key required by SPU static forward: {key}")
        return numpy_from_torch_tensor(state_dict[key])

    # Detect decomposed LRD mode
    lrd_decomposed = bool(args_snapshot.get("lrd_decomposed", False))

    block_params = []
    for block_index in range(depth):
        act_alpha, act_beta = resolve_block_activation_params(state_dict, block_index, activation_kind)
        prefix = f"blocks.{block_index}"

        if lrd_decomposed:
            # Decomposed weights: pack (down_weight, up_weight) as tuple
            def _decomp_w(layer_prefix):
                return (required(f"{layer_prefix}.0.weight"), required(f"{layer_prefix}.1.weight"))

            def _decomp_b(layer_prefix):
                bk = f"{layer_prefix}.1.bias"
                return required(bk) if bk in state_dict else numpy_from_torch_tensor(torch.zeros(1))

            block_params.append(
                (
                    required(f"{prefix}.norm1.weight"),
                    required(f"{prefix}.norm1.bias"),
                    _decomp_w(f"{prefix}.attn.qkv"),
                    _decomp_b(f"{prefix}.attn.qkv"),
                    _decomp_w(f"{prefix}.attn.proj"),
                    _decomp_b(f"{prefix}.attn.proj"),
                    required(f"{prefix}.norm2.weight"),
                    required(f"{prefix}.norm2.bias"),
                    _decomp_w(f"{prefix}.mlp.fc1"),
                    _decomp_b(f"{prefix}.mlp.fc1"),
                    act_alpha,
                    act_beta,
                    _decomp_w(f"{prefix}.mlp.fc2"),
                    _decomp_b(f"{prefix}.mlp.fc2"),
                )
            )
        else:
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
        "use_mask_pruning": bool(args_snapshot.get("use_mask_pruning", False)),
        "pruning_loc": pruning_loc,
        "token_ratio": token_ratio,
        "forward_scope": STATIC_FORWARD_SCOPE,
        "unsupported_currently_bypassed": [
            "intermediate feature reveal",
            "dynamic masking-pruning inside secure forward",
        ],
        "secure_pruning_note": (
            "PredictorLG + kth_threshold + tie_resolution execute inside SPU "
            "(jax_spu_secure_pruning_forward_backend_v0), but the current runner host first loads "
            "the plaintext model bundle before secret SPU injection."
        ),
        "base_rate": float(base_rate),
        "bundle_base_rate": float(bundle_base_rate),
    }
    return params, metadata


# ---------------------------------------------------------------------------
# PredictorLG parameter extraction for secure in-SPU pruning
# ---------------------------------------------------------------------------

SECURE_PRUNING_FORWARD_SCOPE = "student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path"


def resolve_predictor_activation_alpha(state_dict, predictor_index: int, sub_module: str):
    """Resolve the fixed_alpha for a PredictorLG sub-activation.

    sub_module is one of:
      - "in_conv.2"   (first square activation in in_conv)
      - "out_conv.1"  (second square activation in out_conv, after first linear)
      - "out_conv.3"  (third square activation in out_conv, after second linear)
    """
    import numpy as np

    key = f"score_predictor.{predictor_index}.{sub_module}.fixed_alpha"
    value = state_dict.get(key)
    if value is None:
        return np.asarray(0.25, dtype=np.float32)
    return np.asarray(float(value.detach().cpu().item()), dtype=np.float32)


def load_static_vit_spu_predictor_params(state_dict, pruning_loc):
    """Extract PredictorLG weights as numpy tuples for SPU secure pruning.

    Returns a tuple of predictor stages, each stage being:
      (in_norm_weight, in_norm_bias, in_linear_weight, in_linear_bias, in_act_alpha,
       out_linear0_weight, out_linear0_bias, out_act0_alpha,
       out_linear1_weight, out_linear1_bias, out_act1_alpha,
       out_proj_weight, out_proj_bias)
    """
    predictor_params = []
    for stage_index, _ in enumerate(pruning_loc):
        prefix = f"score_predictor.{stage_index}"
        predictor_params.append((
            numpy_from_torch_tensor(state_dict[f"{prefix}.in_conv.0.weight"]),    # LayerNorm weight [384]
            numpy_from_torch_tensor(state_dict[f"{prefix}.in_conv.0.bias"]),      # LayerNorm bias [384]
            numpy_from_torch_tensor(state_dict[f"{prefix}.in_conv.1.weight"]),    # Linear weight [384, 384]
            numpy_from_torch_tensor(state_dict[f"{prefix}.in_conv.1.bias"]),      # Linear bias [384]
            resolve_predictor_activation_alpha(state_dict, stage_index, "in_conv.2"),  # square alpha scalar
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_conv.0.weight"]),   # Linear weight [192, 384]
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_conv.0.bias"]),     # Linear bias [192]
            resolve_predictor_activation_alpha(state_dict, stage_index, "out_conv.1"),  # square alpha scalar
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_conv.2.weight"]),   # Linear weight [96, 192]
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_conv.2.bias"]),     # Linear bias [96]
            resolve_predictor_activation_alpha(state_dict, stage_index, "out_conv.3"),  # square alpha scalar
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_proj.weight"]),     # Linear weight [2, 96]
            numpy_from_torch_tensor(state_dict[f"{prefix}.out_proj.bias"]),       # Linear bias [2]
        ))
    return tuple(predictor_params)


def load_static_vit_spu_params_with_predictor(
    bundle_dir: Path,
    static_depth_limit: int = -1,
    attention_policy: str = "smoothed",
    activation_override: str = "bundle",
    token_ratio_base_override: float = 0.0,
):
    """Load SPU params including PredictorLG weights for secure in-SPU pruning.

    Returns (params, predictor_params, metadata) where predictor_params is the
    tuple from load_static_vit_spu_predictor_params.
    """
    import torch

    from tools.transshield_stage2_bundle import load_json as load_stage2_json
    from tools.transshield_stage2_bundle import resolve_model_state_dict_path

    args_snapshot = load_stage2_json(bundle_dir / "args_snapshot.json")
    # Support both deit-s (teacher) and deit-t (student) models
    model_type = args_snapshot.get("model", "deit-s")
    if model_type not in ("deit-s", "deit-t"):
        raise NotImplementedError(
            f"SPU secure pruning currently supports deit-s and deit-t only, got {model_type}"
        )
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)

    # Re-use existing loader for the base params
    params, metadata = load_static_vit_spu_params(
        bundle_dir, static_depth_limit, attention_policy, activation_override,
        token_ratio_base_override=token_ratio_base_override,
    )

    # Extract predictor params
    pruning_loc = metadata["pruning_loc"]
    predictor_params = load_static_vit_spu_predictor_params(state_dict, pruning_loc)

    # Compute token keep counts (use metadata which respects token_ratio_base_override)
    base_rate = float(metadata.get("base_rate", args_snapshot["base_rate"]))
    embed_dim = int(metadata["embed_dim"])
    token_ratio = metadata["token_ratio"]
    init_n = (224 // int(metadata["patch_size"])) ** 2  # 196 for 224/16
    token_keep_counts = tuple(int(init_n * r) for r in token_ratio)

    # Update metadata
    metadata["forward_scope"] = SECURE_PRUNING_FORWARD_SCOPE
    metadata["has_predictor_params"] = True
    metadata["token_keep_counts"] = list(token_keep_counts)
    metadata["eval_pruning_mode"] = str(args_snapshot.get("eval_pruning_mode", "topk_argsort"))
    metadata["eval_tie_policy"] = str(args_snapshot.get("eval_tie_policy", "lowest_index"))
    if "unsupported_currently_bypassed" in metadata:
        metadata["unsupported_currently_bypassed"] = [
            item for item in metadata["unsupported_currently_bypassed"]
            if item != "runtime pruning predictor path"
        ]

    return params, predictor_params, metadata
