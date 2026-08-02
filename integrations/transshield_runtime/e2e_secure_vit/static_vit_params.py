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


def resolve_uniform_fixed_square_scale(params, predictor_params=None):
    """Return the common fixed-square scale after validating every activation.

    Publishing this value is only safe as a graph optimization when it is an
    architecture constant shared by all executed MLP and PredictorLG square
    activations.  Keep the validation next to parameter loading so a different
    bundle cannot silently inherit the optimization.
    """
    import math

    values = [float(block_params[10]) for block_params in params[4]]
    if predictor_params is not None:
        for stage_params in predictor_params:
            values.extend(float(stage_params[index]) for index in (4, 7, 10))
    if not values:
        raise ValueError("public fixed-square scale requires at least one activation")
    scale = values[0]
    if not math.isfinite(scale):
        raise ValueError(f"fixed-square scale must be finite, got {scale}")
    if any(value != scale for value in values[1:]):
        raise ValueError(
            "public fixed-square scale requires one identical architecture constant; "
            f"observed={values}"
        )
    return scale


def fold_input_scale_into_linear_weight(weight, scale):
    """Fold a scalar input multiplier into the following linear weight.

    For a regular linear layer, ``linear(scale * x, weight, bias)`` equals
    ``linear(x, scale * weight, bias)`` over real arithmetic.  For a
    decomposed linear layer, scaling the first (down) matrix has the same
    effect and leaves the second matrix and bias unchanged.
    """
    import numpy as np

    scale = np.asarray(scale, dtype=np.float32)
    if scale.size != 1:
        raise ValueError(f"linear input scale must be scalar, got shape={scale.shape}")
    if isinstance(weight, tuple):
        if len(weight) != 2:
            raise ValueError("decomposed linear weight must contain down and up matrices")
        down_weight, up_weight = weight
        return (
            np.asarray(np.asarray(down_weight, dtype=np.float32) * scale, dtype=np.float32),
            up_weight,
        )
    return np.asarray(np.asarray(weight, dtype=np.float32) * scale, dtype=np.float32)


def fold_layer_norm_affine_into_linear(
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
):
    """Fold LayerNorm's affine transform into its following linear layer.

    For ``linear(normalized * gamma + beta, W, b)``, the real-valued
    equivalent is ``linear(normalized, W * gamma, b + W @ beta)``.  For a
    decomposed linear, scale the input columns of the down matrix and propagate
    beta through both matrices.  The returned LayerNorm affine parameters are
    value-shaped ones and zeros; the secure graph uses metadata to skip the
    affine multiplication and addition entirely.
    """
    import numpy as np

    gamma = np.asarray(norm_weight, dtype=np.float32)
    beta = np.asarray(norm_bias, dtype=np.float32)
    if gamma.ndim != 1 or beta.shape != gamma.shape:
        raise ValueError(
            "LayerNorm affine parameters must be matching vectors, got "
            f"gamma={gamma.shape} beta={beta.shape}"
        )

    if isinstance(linear_weight, tuple):
        if len(linear_weight) != 2:
            raise ValueError("decomposed linear weight must contain down and up matrices")
        down_weight = np.asarray(linear_weight[0], dtype=np.float32)
        up_weight = np.asarray(linear_weight[1], dtype=np.float32)
        if down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError("decomposed linear weights must be matrices")
        if down_weight.shape[1] != gamma.size or up_weight.shape[1] != down_weight.shape[0]:
            raise ValueError(
                "decomposed linear dimensions do not match LayerNorm width: "
                f"down={down_weight.shape} up={up_weight.shape} gamma={gamma.shape}"
            )
        beta_projection = up_weight @ (down_weight @ beta)
        fused_weight = (
            np.asarray(down_weight * gamma[None, :], dtype=np.float32),
            up_weight,
        )
    else:
        weight = np.asarray(linear_weight, dtype=np.float32)
        if weight.ndim != 2 or weight.shape[1] != gamma.size:
            raise ValueError(
                "linear dimensions do not match LayerNorm width: "
                f"weight={weight.shape} gamma={gamma.shape}"
            )
        beta_projection = weight @ beta
        fused_weight = np.asarray(weight * gamma[None, :], dtype=np.float32)

    if linear_bias is None:
        fused_bias = np.asarray(beta_projection, dtype=np.float32)
    else:
        bias = np.asarray(linear_bias, dtype=np.float32)
        if bias.size == 1:
            fused_bias = np.asarray(
                beta_projection + bias.reshape(-1)[0],
                dtype=np.float32,
            )
        elif bias.shape != beta_projection.shape:
            raise ValueError(
                "linear bias does not match projected LayerNorm bias: "
                f"bias={bias.shape} projected={beta_projection.shape}"
            )
        else:
            fused_bias = np.asarray(bias + beta_projection, dtype=np.float32)

    return (
        np.ones_like(gamma, dtype=np.float32),
        np.zeros_like(beta, dtype=np.float32),
        fused_weight,
        fused_bias,
    )


def fold_block_layer_norm_affines(block_params):
    """Fold a ViT block's norm1→QKV and norm2→FC1 affine transforms."""
    values = list(block_params)
    values[0], values[1], values[2], values[3] = fold_layer_norm_affine_into_linear(
        values[0], values[1], values[2], values[3]
    )
    values[6], values[7], values[8], values[9] = fold_layer_norm_affine_into_linear(
        values[6], values[7], values[8], values[9]
    )
    return tuple(values)


def fold_predictor_layer_norm_affine(stage_params):
    """Fold PredictorLG's input LayerNorm affine into its first linear."""
    values = list(stage_params)
    values[0], values[1], values[2], values[3] = fold_layer_norm_affine_into_linear(
        values[0], values[1], values[2], values[3]
    )
    return tuple(values)


def fold_predictor_square_activation_scales(stage_params):
    """Fold PredictorLG's three square scales into their following linears."""
    import numpy as np

    (
        in_norm_w,
        in_norm_b,
        in_lin_w,
        in_lin_b,
        in_act_alpha,
        out0_w,
        out0_b,
        out0_alpha,
        out1_w,
        out1_b,
        out1_alpha,
        out_proj_w,
        out_proj_b,
    ) = stage_params
    out0_w = fold_input_scale_into_linear_weight(out0_w, in_act_alpha)
    out1_w = fold_input_scale_into_linear_weight(out1_w, out0_alpha)
    out_proj_w = fold_input_scale_into_linear_weight(out_proj_w, out1_alpha)
    one = np.asarray(1.0, dtype=np.float32)
    return (
        in_norm_w,
        in_norm_b,
        in_lin_w,
        in_lin_b,
        one,
        out0_w,
        out0_b,
        one,
        out1_w,
        out1_b,
        one,
        out_proj_w,
        out_proj_b,
    )


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


def pruning_schedule_for_depth(base_rate: float, depth: int):
    """Return only pruning stages that execute within the selected prefix depth."""
    base_rate = float(base_rate)
    if not 0.0 < base_rate <= 1.0:
        raise ValueError(f"base_rate must be within (0, 1], got {base_rate}")
    depth = int(depth)
    full_locations = (3, 6, 9)
    full_ratios = (
        base_rate,
        base_rate**2,
        base_rate**3,
    )
    active = [
        (location, ratio)
        for location, ratio in zip(full_locations, full_ratios)
        if location < depth
    ]
    return [location for location, _ in active], [ratio for _, ratio in active]


def secure_pruning_unsupported_items(items):
    implemented = {
        "runtime pruning predictor path",
        "dynamic masking-pruning inside secure forward",
    }
    return [item for item in items if item not in implemented]


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
    fold_square_activation_scale: bool = False,
    fold_layer_norm_affine: bool = False,
    *,
    preloaded_args_snapshot=None,
    preloaded_state_dict=None,
):
    import numpy as np
    import torch

    from tools.transshield_stage2_bundle import load_json as load_stage2_json
    from tools.transshield_stage2_bundle import resolve_model_state_dict_path

    args_snapshot = (
        load_stage2_json(bundle_dir / "args_snapshot.json")
        if preloaded_args_snapshot is None
        else preloaded_args_snapshot
    )
    # Support both deit-s (teacher) and deit-t (student) models
    model_type = args_snapshot.get("model", "deit-s")
    if model_type not in ("deit-s", "deit-t"):
        raise NotImplementedError(
            f"SPU whole-forward backend currently supports deit-s and deit-t only, got {model_type}"
        )
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = (
        torch.load(state_dict_path, map_location="cpu", weights_only=False)
        if preloaded_state_dict is None
        else preloaded_state_dict
    )
    base_activation_kind = resolve_static_activation_kind(args_snapshot)
    activation_kind = resolve_spu_activation_kind(base_activation_kind, activation_override)
    fold_square_activation_scale = bool(fold_square_activation_scale)
    fold_layer_norm_affine = bool(fold_layer_norm_affine)
    if fold_square_activation_scale and activation_kind not in {
        "fixed_square",
        "learnable_square",
    }:
        raise ValueError(
            "square activation scale fusion requires fixed_square or learnable_square"
        )
    bundle_base_rate = float(args_snapshot["base_rate"])
    if token_ratio_base_override > 0.0:
        base_rate = token_ratio_base_override
    else:
        base_rate = bundle_base_rate
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
    pruning_loc, token_ratio = pruning_schedule_for_depth(base_rate, depth)
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

            fc2_weight = _decomp_w(f"{prefix}.mlp.fc2")
            if fold_square_activation_scale:
                fc2_weight = fold_input_scale_into_linear_weight(fc2_weight, act_alpha)
                act_alpha = np.asarray(1.0, dtype=np.float32)
            block_param = (
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
                fc2_weight,
                _decomp_b(f"{prefix}.mlp.fc2"),
            )
            if fold_layer_norm_affine:
                block_param = fold_block_layer_norm_affines(block_param)
            block_params.append(block_param)
        else:
            fc2_weight = required(f"{prefix}.mlp.fc2.weight")
            if fold_square_activation_scale:
                fc2_weight = fold_input_scale_into_linear_weight(fc2_weight, act_alpha)
                act_alpha = np.asarray(1.0, dtype=np.float32)
            block_param = (
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
                fc2_weight,
                required(f"{prefix}.mlp.fc2.bias"),
            )
            if fold_layer_norm_affine:
                block_param = fold_block_layer_norm_affines(block_param)
            block_params.append(block_param)

    final_norm_weight = required("norm.weight")
    final_norm_bias = required("norm.bias")
    head_weight = required("head.weight")
    head_bias = required("head.bias")
    if fold_layer_norm_affine:
        (
            final_norm_weight,
            final_norm_bias,
            head_weight,
            head_bias,
        ) = fold_layer_norm_affine_into_linear(
            final_norm_weight,
            final_norm_bias,
            head_weight,
            head_bias,
        )

    params = (
        required("patch_embed.proj.weight"),
        required("patch_embed.proj.bias"),
        required("cls_token"),
        required("pos_embed"),
        tuple(block_params),
        final_norm_weight,
        final_norm_bias,
        head_weight,
        head_bias,
    )
    metadata = {
        "state_dict_path": str(state_dict_path),
        "base_activation_kind": base_activation_kind,
        "activation_kind": activation_kind,
        "activation_override": str(activation_override),
        "square_activation_scale_fused": fold_square_activation_scale,
        "layer_norm_affine_fused": fold_layer_norm_affine,
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


def load_static_vit_spu_predictor_params(
    state_dict,
    pruning_loc,
    *,
    fold_square_activation_scale: bool = False,
    fold_layer_norm_affine: bool = False,
):
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
        stage_params = (
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
        )
        if fold_square_activation_scale:
            stage_params = fold_predictor_square_activation_scales(stage_params)
        if fold_layer_norm_affine:
            stage_params = fold_predictor_layer_norm_affine(stage_params)
        predictor_params.append(stage_params)
    return tuple(predictor_params)


def load_static_vit_spu_params_with_predictor(
    bundle_dir: Path,
    static_depth_limit: int = -1,
    attention_policy: str = "smoothed",
    activation_override: str = "bundle",
    token_ratio_base_override: float = 0.0,
    fold_square_activation_scale: bool = False,
    fold_predictor_square_activation_scale: bool = False,
    fold_layer_norm_affine: bool = False,
    fold_predictor_layer_norm_affine: bool = False,
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
        fold_square_activation_scale=fold_square_activation_scale,
        fold_layer_norm_affine=fold_layer_norm_affine,
        preloaded_args_snapshot=args_snapshot,
        preloaded_state_dict=state_dict,
    )

    # Extract predictor params
    pruning_loc = metadata["pruning_loc"]
    predictor_params = load_static_vit_spu_predictor_params(
        state_dict,
        pruning_loc,
        fold_square_activation_scale=fold_predictor_square_activation_scale,
        fold_layer_norm_affine=fold_predictor_layer_norm_affine,
    )

    # Compute token keep counts (use metadata which respects token_ratio_base_override)
    base_rate = float(metadata.get("base_rate", args_snapshot["base_rate"]))
    embed_dim = int(metadata["embed_dim"])
    token_ratio = metadata["token_ratio"]
    init_n = (224 // int(metadata["patch_size"])) ** 2  # 196 for 224/16
    token_keep_counts = tuple(int(init_n * r) for r in token_ratio)

    # Update metadata
    metadata["forward_scope"] = SECURE_PRUNING_FORWARD_SCOPE
    metadata["has_predictor_params"] = True
    metadata["predictor_square_activation_scale_fused"] = bool(
        fold_predictor_square_activation_scale
    )
    metadata["predictor_layer_norm_affine_fused"] = bool(
        fold_predictor_layer_norm_affine
    )
    metadata["token_keep_counts"] = list(token_keep_counts)
    metadata["eval_pruning_mode"] = str(args_snapshot.get("eval_pruning_mode", "topk_argsort"))
    metadata["eval_tie_policy"] = str(args_snapshot.get("eval_tie_policy", "lowest_index"))
    if "unsupported_currently_bypassed" in metadata:
        metadata["unsupported_currently_bypassed"] = secure_pruning_unsupported_items(
            metadata["unsupported_currently_bypassed"]
        )

    return params, predictor_params, metadata
