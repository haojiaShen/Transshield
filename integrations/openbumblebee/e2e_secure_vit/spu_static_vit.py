from pathlib import Path

from integrations.openbumblebee.e2e_secure_vit.common import load_json


def run_static_vit_forward_spu(
    pixel_values_np,
    params,
    config_path: Path,
    metadata,
    batch_size: int,
    params_mode: str,
    probe_block_index=None,
    pixel_value_shares_np=None,
    pixel_value_share_manifest_paths=None,
    share_sample_count=None,
    block_chunk_size: int = 0,
    layer_norm_chunk_size: int = 0,
    layer_norm_policy: str = "exact",
    layer_norm_calibration=None,
    activation_clip_value: float = 0.0,
    external_keep_masks_np=None,
    predictor_params_np=None,
    pruning_metadata=None,
    token_recycle_scale: float = 0.0,
):
    import numpy as np
    import jax.numpy as jnp
    import jax.scipy.special as jsp_special
    import spu.utils.distributed as ppd

    config = load_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    patch_size = int(metadata["patch_size"])
    layer_norm_eps = float(metadata["layer_norm_eps"])
    attention_policy = str(metadata.get("attention_policy", "smoothed"))
    attention_policy_eps = float(metadata["attention_policy_eps"])
    activation_kind = str(metadata["activation_kind"])
    use_mask_pruning = bool(metadata.get("use_mask_pruning", False))
    pruning_loc = tuple(int(value) for value in metadata.get("pruning_loc", ()))
    token_keep_counts_raw = list(metadata.get("token_keep_counts", ()))
    activation_clip_value = float(activation_clip_value)
    attn_scale = head_dim ** -0.5
    if attention_policy not in {"smoothed", "standard", "uniform", "identity"}:
        raise ValueError(f"unsupported attention policy: {attention_policy}")
    if probe_block_index is not None:
        probe_block_index = int(probe_block_index)
        if probe_block_index < 0 or probe_block_index >= int(metadata["depth"]):
            raise ValueError(
                f"probe_block_index={probe_block_index} must be within executed depth={metadata['depth']}"
            )
    block_chunk_size = int(block_chunk_size)
    layer_norm_chunk_size = int(layer_norm_chunk_size)
    layer_norm_policy = str(layer_norm_policy)
    if layer_norm_policy not in {"exact", "affine", "public_calibrated"}:
        raise ValueError(f"unsupported layer_norm_policy: {layer_norm_policy}")
    if layer_norm_policy == "public_calibrated" and layer_norm_calibration is None:
        raise ValueError("layer_norm_policy=public_calibrated requires layer_norm_calibration")
    supported_params_modes = {
        "secret",
        "public",
        "secret_patch_head_public_blocks",
        "public_patch_head_secret_blocks",
        "secret_patch_head_secret_blocks_split",
        "secret_patch_public_head_secret_blocks",
        "public_patch_secret_head_secret_blocks",
        "secret_three_stage",
        "secret_blockwise_stage",
        "secret_block_group_stage",
    }
    if params_mode not in supported_params_modes:
        raise ValueError(f"unsupported SPU params mode: {params_mode}")
    if params_mode == "secret_block_group_stage" and block_chunk_size <= 0:
        block_chunk_size = 2
    using_chunked_forward = block_chunk_size > 0
    if using_chunked_forward and probe_block_index is not None:
        raise ValueError("block_chunk_size is not supported with probe_block_index")
    if using_chunked_forward and params_mode not in {"public", "secret_block_group_stage"}:
        raise ValueError("block_chunk_size currently requires public or secret_block_group_stage SPU params")
    if external_keep_masks_np is not None:
        if probe_block_index is not None:
            raise ValueError("external_keep_masks_np is not supported with probe_block_index")
        if using_chunked_forward:
            raise ValueError("external_keep_masks_np is not supported with block_chunk_size")
        if attention_policy != "uniform":
            raise ValueError("external_keep_masks_np currently requires attention_policy=uniform")
        if params_mode not in {"public", "secret"}:
            raise ValueError("external_keep_masks_np currently supports only public or secret SPU params mode")

    def linear(x, weight, bias):
        return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias

    def decomposed_linear(x, down_weight, up_weight, bias):
        """Two-step matmul for decomposed LRD weights.
        down_weight: (rank, in_features), up_weight: (out_features, rank)
        y = x @ down_weight.T @ up_weight.T + bias
        """
        mid = jnp.matmul(x, jnp.swapaxes(down_weight, -1, -2))
        result = jnp.matmul(mid, jnp.swapaxes(up_weight, -1, -2))
        if bias is not None:
            result = result + bias
        return result

    def feature_sum(x):
        feature_dim = int(x.shape[-1])
        if layer_norm_chunk_size <= 0 or layer_norm_chunk_size >= feature_dim:
            return jnp.sum(x, axis=-1, keepdims=True)
        total = None
        for start in range(0, feature_dim, layer_norm_chunk_size):
            end = min(start + layer_norm_chunk_size, feature_dim)
            partial = jnp.sum(x[..., start:end], axis=-1, keepdims=True)
            total = partial if total is None else total + partial
        return total

    def layer_norm(x, weight, bias, calibration=None):
        if layer_norm_policy == "affine":
            return x * weight + bias
        if layer_norm_policy == "public_calibrated":
            if calibration is None:
                raise ValueError("missing public calibration stats for layer_norm")
            mean, variance = calibration
            inverse_std = 1.0 / jnp.sqrt(variance + layer_norm_eps)
            return (x - mean) * inverse_std * weight + bias
        feature_dim = int(x.shape[-1])
        mean = feature_sum(x) / feature_dim
        centered = x - mean
        variance = feature_sum(centered * centered) / feature_dim
        inverse_std = 1.0 / jnp.sqrt(variance + layer_norm_eps)
        return centered * inverse_std * weight + bias

    def patch_embed(pixel_values, patch_weight, patch_bias):
        batch, channels, height, width = pixel_values.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = jnp.reshape(
            pixel_values,
            (batch, channels, grid_h, patch_size, grid_w, patch_size),
        )
        patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))
        patches = jnp.reshape(patches, (batch, grid_h * grid_w, channels * patch_size * patch_size))
        patch_weight_flat = jnp.reshape(patch_weight, (patch_weight.shape[0], -1))
        return linear(patches, patch_weight_flat, patch_bias)

    def gelu_exact(x):
        return 0.5 * x * (1.0 + jsp_special.erf(x / jnp.sqrt(2.0)))


    def patch_embed(pixel_values, patch_weight, patch_bias):
        batch, channels, height, width = pixel_values.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = jnp.reshape(
            pixel_values,
            (batch, channels, grid_h, patch_size, grid_w, patch_size),
        )
        patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))
        patches = jnp.reshape(patches, (batch, grid_h * grid_w, channels * patch_size * patch_size))
        patch_weight_flat = jnp.reshape(patch_weight, (patch_weight.shape[0], -1))
        return linear(patches, patch_weight_flat, patch_bias)

    def audit_fn(share0, share1, forward_params):
        patch_weight, patch_bias, cls_token, pos_embed = forward_params[:4]
        reconstructed = share0 + share1
        patch_tokens = patch_embed(reconstructed, patch_weight, patch_bias)
        batch = patch_tokens.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        tokens_with_pos = jnp.concatenate([cls_tokens, patch_tokens], axis=1) + pos_embed
        return reconstructed, patch_tokens, tokens_with_pos

    p1 = ppd.device("P1")
    p2 = ppd.device("P2")
    spu = ppd.device("SPU")

    def identity(value):
        return value

    secret_identity = p1(identity)
    p2_secret_identity = p2(identity)
    if params_mode == "secret":
        params_ref = secret_identity(params)
    elif params_mode == "public":
        params_ref = params
    else:
        raise ValueError(f"unsupported SPU params mode: {params_mode}")

    share0_np = np.asarray(pixel_value_shares_np[0], dtype=np.float32)
    share1_np = np.asarray(pixel_value_shares_np[1], dtype=np.float32)
    result_ref = spu(audit_fn)(secret_identity(share0_np), p2_secret_identity(share1_np), params_ref)
    reconstructed, patch_tokens, tokens_with_pos = ppd.get(result_ref)
    return {
        "reconstructed_pixel_values": np.asarray(reconstructed, dtype=np.float32),
        "patch_tokens": np.asarray(patch_tokens, dtype=np.float32),
        "tokens_with_pos": np.asarray(tokens_with_pos, dtype=np.float32),
    }
