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

    def lut_gelu_interp(x, breakpoints, values):
        """Binary-search LUT GELU: O(log N) secure comparisons vs O(N) linear scan.
        
        Uses nested jnp.where to binary-search the segment index,
        then linear interpolation within the found segment.
        For 16 segments: 4 comparisons instead of 15 (3.75x fewer).
        For 8 segments: 3 comparisons instead of 7 (2.3x fewer).
        For 4 segments: 2 comparisons instead of 3.
        
        Key SPU benefit: eliminates the broadcasting to (..., N-1) shape
        which multiplies memory and communication volume.
        """
        x_c = jnp.clip(x, breakpoints[0], breakpoints[-1])
        n_seg = len(breakpoints) - 1
        
        # Precompute slopes and intercepts for all segments
        # slopes[i] = (values[i+1] - values[i]) / (breakpoints[i+1] - breakpoints[i])
        slopes = (values[1:] - values[:-1]) / (breakpoints[1:] - breakpoints[:-1])
        intercepts = values[:-1] - slopes * breakpoints[:-1]
        
        def _interp_segment(seg_idx):
            return slopes[seg_idx] * x_c + intercepts[seg_idx]
        
        def _binary_search(lo, hi):
            if lo == hi:
                return _interp_segment(lo)
            mid = (lo + hi) // 2
            return jnp.where(
                x_c <= breakpoints[mid + 1],
                _binary_search(lo, mid),
                _binary_search(mid + 1, hi)
            )
        
        return _binary_search(0, n_seg - 1)

    def activate(x, alpha, beta):
        if activation_clip_value > 0.0:
            x = jnp.clip(x, -activation_clip_value, activation_clip_value)
        if activation_kind == "gelu":
            return gelu_exact(x)
        if activation_kind in {"fixed_square", "learnable_square"}:
            return alpha * (x * x)
        if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
            return alpha * (x * x) + beta * x
        if activation_kind.startswith("lut_gelu"):
            # alpha contains breakpoints, beta contains values
            return lut_gelu_interp(x, alpha, beta)
        raise ValueError(f"unsupported activation kind: {activation_kind}")

    def attention_softmax(attn):
        if attention_policy == "uniform":
            return jnp.ones_like(attn) / attn.shape[-1]
        if attention_policy == "identity":
            raise ValueError("identity attention does not materialize an attention matrix")
        shifted = attn - jnp.max(attn, axis=-1, keepdims=True)
        exp_values = jnp.exp(shifted)
        if attention_policy == "standard":
            return exp_values / jnp.sum(exp_values, axis=-1, keepdims=True)
        token_count = attn.shape[-1]
        return (exp_values + attention_policy_eps / token_count) / (
            jnp.sum(exp_values, axis=-1, keepdims=True) + attention_policy_eps
        )

    def normalize_keep_mask(keep_mask, dtype):
        keep_mask = jnp.asarray(keep_mask, dtype=dtype)
        if keep_mask.ndim == 2:
            keep_mask = jnp.expand_dims(keep_mask, axis=-1)
        return keep_mask

    def apply_spatial_mask(x, keep_mask):
        keep_mask = normalize_keep_mask(keep_mask, x.dtype)
        return jnp.concatenate([x[:, :1], x[:, 1:] * keep_mask], axis=1)

    def _manual_sigmoid(x):
        """SPU-compatible sigmoid: 1 / (1 + exp(-x))."""
        return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -20.0, 20.0)))

    def _apply_spatial_mask_with_recycle(x, keep_mask, keep_score):
        """Apply spatial mask with Dropped-Token Context Recycling.

        Before zeroing dropped tokens, compute a soft-weighted summary of tokens
        that are about to be dropped (weight = 1 − sigmoid(keep_score)) and
        inject it into the CLS token.  All ops are multiply-accumulate — fully
        MPC-friendly, zero extra communication rounds.

        When token_recycle_scale == 0 this degrades to the original
        apply_spatial_mask exactly.
        """
        km = normalize_keep_mask(keep_mask, x.dtype)
        if token_recycle_scale <= 0.0:
            return jnp.concatenate([x[:, :1], x[:, 1:] * km], axis=1)

        # Drop weight: higher for tokens more likely to be dropped
        drop_w = (1.0 - _manual_sigmoid(keep_score)) * (1.0 - km.squeeze(-1))  # [B, N]
        drop_w_3d = jnp.expand_dims(drop_w, -1)                                 # [B, N, 1]

        spatial = x[:, 1:]                                                       # [B, N, C]
        num = jnp.sum(spatial * drop_w_3d, axis=1, keepdims=True)               # [B, 1, C]
        den = jnp.maximum(jnp.sum(drop_w_3d, axis=1, keepdims=True), 1e-8)      # [B, 1, 1]
        dropped_summary = num / den                                              # [B, 1, C]

        cls_part = x[:, 0:1, :] + dropped_summary * token_recycle_scale
        masked_spatial = spatial * km
        return jnp.concatenate([cls_part, masked_spatial], axis=1)

    def block_forward(x, block_param, block_calibration=None, capture_probe=False):
        (
            norm1_weight,
            norm1_bias,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            norm2_weight,
            norm2_bias,
            fc1_weight,
            fc1_bias,
            act_alpha,
            act_beta,
            fc2_weight,
            fc2_bias,
        ) = block_param
        # Detect decomposed LRD weights (tuple of down_weight, up_weight)
        _use_decomposed = isinstance(qkv_weight, tuple)
        block_input = x
        residual = x
        norm1_calibration = None if block_calibration is None else block_calibration[:2]
        norm2_calibration = None if block_calibration is None else block_calibration[2:]
        norm1_out = layer_norm(x, norm1_weight, norm1_bias, norm1_calibration)
        batch, token_count, channels = norm1_out.shape
        if attention_policy == "uniform":
            if _use_decomposed:
                # Slice up_weight to extract only V portion
                up_weight_v = qkv_weight[1][2 * channels :, :]
                down_weight_v = qkv_weight[0]
                value_bias_v = qkv_bias[2 * channels : 3 * channels]
                value = decomposed_linear(norm1_out, down_weight_v, up_weight_v, value_bias_v)
            else:
                value_weight = qkv_weight[2 * channels : 3 * channels, :]
                value_bias = qkv_bias[2 * channels : 3 * channels]
                value = linear(norm1_out, value_weight, value_bias)
            value = jnp.reshape(value, (batch, token_count, num_heads, head_dim))
            value = jnp.transpose(value, (0, 2, 1, 3))
            mean_value = jnp.mean(value, axis=2, keepdims=True)
            attn_out = jnp.broadcast_to(mean_value, (batch, num_heads, token_count, head_dim))
        else:
            if _use_decomposed:
                qkv = decomposed_linear(norm1_out, qkv_weight[0], qkv_weight[1], qkv_bias)
            else:
                qkv = linear(norm1_out, qkv_weight, qkv_bias)
            qkv = jnp.reshape(qkv, (batch, token_count, 3, num_heads, head_dim))
            qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
            query, key, value = qkv[0], qkv[1], qkv[2]
            if attention_policy == "identity":
                attn_out = value
            else:
                attn = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * attn_scale
                attn = attention_softmax(attn)
                attn_out = jnp.matmul(attn, value)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = jnp.reshape(attn_out, (batch, token_count, channels))
        if _use_decomposed:
            projected_attn_out = decomposed_linear(attn_out, proj_weight[0], proj_weight[1], proj_bias)
        else:
            projected_attn_out = linear(attn_out, proj_weight, proj_bias)
        x = residual + projected_attn_out

        residual = x
        norm2_out = layer_norm(x, norm2_weight, norm2_bias, norm2_calibration)
        if _use_decomposed:
            mlp_hidden = activate(decomposed_linear(norm2_out, fc1_weight[0], fc1_weight[1], fc1_bias), act_alpha, act_beta)
        else:
            mlp_hidden = activate(linear(norm2_out, fc1_weight, fc1_bias), act_alpha, act_beta)
        if _use_decomposed:
            mlp_out = decomposed_linear(mlp_hidden, fc2_weight[0], fc2_weight[1], fc2_bias)
        else:
            mlp_out = linear(mlp_hidden, fc2_weight, fc2_bias)
        block_output = residual + mlp_out
        if capture_probe:
            return block_output, (
                block_input[:, 0, :],
                norm1_out[:, 0, :],
                projected_attn_out[:, 0, :],
                x[:, 0, :],
                norm2_out[:, 0, :],
                mlp_out[:, 0, :],
                block_output[:, 0, :],
            )
        return block_output

    def forward_fn(pixel_values, forward_params):
        (
            patch_weight,
            patch_bias,
            cls_token,
            pos_embed,
            block_params,
            norm_weight,
            norm_bias,
            head_weight,
            head_bias,
        ) = forward_params
        block_calibrations = () if layer_norm_calibration is None else layer_norm_calibration["blocks"]
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        probe_payload = None
        for block_index, block_param in enumerate(block_params):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            if probe_block_index is not None and block_index == probe_block_index:
                x, probe_payload = block_forward(x, block_param, block_calibration, capture_probe=True)
            else:
                x = block_forward(x, block_param, block_calibration)
        final_calibration = None if layer_norm_calibration is None else layer_norm_calibration["final_norm"]
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        logits = linear(cls_features, head_weight, head_bias)
        if probe_payload is None:
            return logits
        return (logits, *probe_payload, cls_features, cls_features)

    def forward_from_shares_fn(share0, share1, forward_params):
        return forward_fn(share0 + share1, forward_params)

    def forward_with_external_keep_masks_fn(pixel_values, forward_params, keep_masks):
        (
            patch_weight,
            patch_bias,
            cls_token,
            pos_embed,
            block_params,
            norm_weight,
            norm_bias,
            head_weight,
            head_bias,
        ) = forward_params
        block_calibrations = () if layer_norm_calibration is None else layer_norm_calibration["blocks"]
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        spatial_token_count = int(x.shape[1] - 1)
        prev_decision = jnp.ones((batch, spatial_token_count, 1), dtype=x.dtype)
        stage_index = 0
        for block_index, block_param in enumerate(block_params):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            if block_index in pruning_loc:
                if use_mask_pruning:
                    x = apply_spatial_mask(x, prev_decision)
                keep_mask = normalize_keep_mask(keep_masks[stage_index], x.dtype)
                prev_decision = prev_decision * keep_mask
                x = apply_spatial_mask(x, prev_decision)
                x = block_forward(x, block_param, block_calibration)
                x = apply_spatial_mask(x, prev_decision)
                stage_index += 1
            else:
                x = block_forward(x, block_param, block_calibration)
                x = apply_spatial_mask(x, prev_decision)
        final_calibration = None if layer_norm_calibration is None else layer_norm_calibration["final_norm"]
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    def forward_with_external_keep_masks_from_shares_fn(share0, share1, forward_params, keep_masks):
        return forward_with_external_keep_masks_fn(share0 + share1, forward_params, keep_masks)

    def forward_split_params_fn(pixel_values, patch_head_params, split_block_params):
        (
            patch_weight,
            patch_bias,
            cls_token,
            pos_embed,
            norm_weight,
            norm_bias,
            head_weight,
            head_bias,
        ) = patch_head_params
        block_calibrations = () if layer_norm_calibration is None else layer_norm_calibration["blocks"]
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        for block_index, block_param in enumerate(split_block_params):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            x = block_forward(x, block_param, block_calibration)
        final_calibration = None if layer_norm_calibration is None else layer_norm_calibration["final_norm"]
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    def forward_split_params_from_shares_fn(share0, share1, patch_head_params, split_block_params):
        return forward_split_params_fn(share0 + share1, patch_head_params, split_block_params)

    def forward_three_part_params_fn(pixel_values, patch_params, split_block_params, final_head_params):
        patch_weight, patch_bias, cls_token, pos_embed = patch_params
        norm_weight, norm_bias, head_weight, head_bias = final_head_params
        block_calibrations = () if layer_norm_calibration is None else layer_norm_calibration["blocks"]
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        for block_index, block_param in enumerate(split_block_params):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            x = block_forward(x, block_param, block_calibration)
        final_calibration = None if layer_norm_calibration is None else layer_norm_calibration["final_norm"]
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    def forward_three_part_params_from_shares_fn(share0, share1, patch_params, split_block_params, final_head_params):
        return forward_three_part_params_fn(share0 + share1, patch_params, split_block_params, final_head_params)

    def embed_only_fn(pixel_values, patch_params):
        patch_weight, patch_bias, cls_token, pos_embed = patch_params
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        return x + pos_embed

    def embed_only_from_shares_fn(share0, share1, patch_params):
        return embed_only_fn(share0 + share1, patch_params)

    def embed_and_blocks_fn(pixel_values, prefix_params):
        (
            patch_weight,
            patch_bias,
            cls_token,
            pos_embed,
            segment_block_params,
            segment_block_calibrations,
        ) = prefix_params
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        for block_index, block_param in enumerate(segment_block_params):
            block_calibration = None if not segment_block_calibrations else segment_block_calibrations[block_index]
            x = block_forward(x, block_param, block_calibration)
        return x

    def embed_and_blocks_from_shares_fn(share0, share1, prefix_params):
        return embed_and_blocks_fn(share0 + share1, prefix_params)

    def blocks_only_fn(x, segment_params):
        segment_block_params, segment_block_calibrations = segment_params
        for block_index, block_param in enumerate(segment_block_params):
            block_calibration = None if not segment_block_calibrations else segment_block_calibrations[block_index]
            x = block_forward(x, block_param, block_calibration)
        return x

    def single_block_fn(x, block_param, block_calibration):
        return block_forward(x, block_param, block_calibration)

    def norm_head_fn(x, head_params):
        norm_weight, norm_bias, head_weight, head_bias, final_calibration = head_params
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    def norm_head_split_fn(x, final_head_params, final_calibration):
        norm_weight, norm_bias, head_weight, head_bias = final_head_params
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)



    # =========================================================================
    # Secure in-SPU pruning: PredictorLG + kth-threshold + tie-resolution
    # =========================================================================

    def _predictorlg_forward(spatial_x, prev_decision_2d, stage_predictor_params):
        """Run PredictorLG forward inside SPU.

        Args:
            spatial_x: [B, N, 384] spatial tokens (no CLS).
            prev_decision_2d: [B, N, 1] keep decision from previous stage.
            stage_predictor_params: tuple of 13 numpy arrays for one stage.

        Returns:
            keep_score: [B, N] log-probability of keeping each token.
        """
        (
            in_norm_w, in_norm_b, in_lin_w, in_lin_b, in_act_alpha,
            out0_w, out0_b, out0_alpha,
            out1_w, out1_b, out1_alpha,
            out_proj_w, out_proj_b,
        ) = stage_predictor_params

        # in_conv: LayerNorm -> Linear -> fixed_square(alpha*x)
        x = layer_norm(spatial_x, in_norm_w, in_norm_b)
        x = linear(x, in_lin_w, in_lin_b)
        x = in_act_alpha * (x * x)

        feat_dim = int(x.shape[-1])  # 384
        half = feat_dim // 2          # 192
        local_feat = x[:, :, :half]
        global_feat = x[:, :, half:]

        # Apply previous keep-decision as policy mask on global features
        policy_float = prev_decision_2d.astype(x.dtype)
        global_input = global_feat * policy_float

        # Global average pooling over active tokens
        active_count = jnp.sum(policy_float, axis=1, keepdims=True)
        active_count = jnp.maximum(active_count, 1.0)
        global_x = jnp.sum(global_input, axis=1, keepdims=True) / active_count

        # Concatenate: [local_feat, broadcast(global_x)]
        batch_size = int(x.shape[0])
        x = jnp.concatenate(
            [local_feat, jnp.broadcast_to(global_x, (batch_size, int(x.shape[1]), half))],
            axis=-1,
        )

        # out_conv: Linear -> square -> Linear -> square
        x = linear(x, out0_w, out0_b)
        x = out0_alpha * (x * x)
        x = linear(x, out1_w, out1_b)
        x = out1_alpha * (x * x)

        # out_proj: Linear -> clamp -> log_softmax
        logits = linear(x, out_proj_w, out_proj_b)
        logits = jnp.clip(logits, -10.0, 10.0)
        # Manual logsumexp: avoids stablehlo.is_finite which SPU doesn't support
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        log_sum = max_logits + jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=-1, keepdims=True))
        log_probs = logits - log_sum

        # Return keep-score: log_prob of class-0 (keep)
        return log_probs[:, :, 0]

    def _bitonic_sort_desc(values):
        """Bitonic sort descending.  O(N log^2 N) compare-and-swap; SPU-friendly fixed pattern.

        values: [B, N] where N must be a power of 2 (caller pads if needed).
        Fully JAX-tracer-safe: no boolean fancy indexing, only jnp.where operations.
        """
        N = int(values.shape[1])
        x = values
        k = 2
        while k <= N:
            j = k // 2
            while j >= 1:
                # For each position p, compute its partner = p XOR j
                p_arr = jnp.arange(N, dtype=jnp.int32)
                p_partner = p_arr ^ j

                # Safe partner index for gathering (clip out-of-range to 0)
                p_partner_safe = jnp.clip(p_partner, 0, N - 1)

                # Gather values from current state
                x_at_p = x                                       # [B, N]
                x_at_partner = x[:, p_partner_safe]              # [B, N]

                # Determine: is p the left element of its pair?
                is_left = (p_arr < p_partner)                    # [N] bool
                has_partner = (p_partner < N)                    # [N] bool

                # Direction is defined at the LEFT element of the pair
                left_index = jnp.where(is_left, p_arr, p_partner)
                is_desc = (left_index & k) == 0                  # [N] bool

                # Descending: left gets max, right gets min
                # Ascending:   left gets min, right gets max
                high = jnp.maximum(x_at_p, x_at_partner)
                low  = jnp.minimum(x_at_p, x_at_partner)

                left_val  = jnp.where(is_desc, high, low)
                right_val = jnp.where(is_desc, low, high)

                # Position p gets left_val if it is the left element, else right_val
                new_val = jnp.where(is_left, left_val, right_val)

                # If no partner (out of range), keep original value
                x = jnp.where(has_partner, new_val, x_at_p)

                j //= 2
            k *= 2
        return x


    def _bitonic_sort_desc_with_indices(values):
        """Bitonic sort descending while tracking original indices.

        Returns (sorted_values, sorted_indices) where sorted_indices[b, i] is the
        original position of the element now at position i in sorted order.
        Eliminates the need for a separate threshold comparison after sorting.

        values: [B, N] where N must be a power of 2.
        """
        N = int(values.shape[1])
        x = values
        indices = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32), values.shape)
        k = 2
        while k <= N:
            j = k // 2
            while j >= 1:
                p_arr = jnp.arange(N, dtype=jnp.int32)
                p_partner = p_arr ^ j
                p_partner_safe = jnp.clip(p_partner, 0, N - 1)

                x_at_p = x
                x_at_partner = x[:, p_partner_safe]
                idx_at_p = indices
                idx_at_partner = indices[:, p_partner_safe]

                is_left = (p_arr < p_partner)
                has_partner = (p_partner < N)
                left_index = jnp.where(is_left, p_arr, p_partner)
                is_desc = (left_index & k) == 0

                high = jnp.maximum(x_at_p, x_at_partner)
                low = jnp.minimum(x_at_p, x_at_partner)
                left_val = jnp.where(is_desc, high, low)
                right_val = jnp.where(is_desc, low, high)
                new_val = jnp.where(is_left, left_val, right_val)
                x = jnp.where(has_partner, new_val, x_at_p)

                # Swap indices in the same pattern
                idx_high = jnp.where(is_desc, idx_at_p, idx_at_partner)
                idx_low = jnp.where(is_desc, idx_at_partner, idx_at_p)
                left_idx = jnp.where(is_left, idx_high, idx_low)
                right_idx = jnp.where(is_left, idx_low, idx_high)
                new_idx = jnp.where(is_left, left_idx, right_idx)
                indices = jnp.where(has_partner, new_idx, indices)

                j //= 2
            k *= 2
        return x, indices


    def _secure_build_keep_decision(score, prev_decision_2d, keep_count):
        """Build keep decision via index-tracking bitonic sort.

        Instead of: sort -> extract threshold -> compare all >= threshold (2 MPC ops),
        we now: sort with index tracking -> directly build mask from sorted indices (1 MPC op).
        Eliminates the separate (encoded_key >= threshold) comparison step.
        """
        N = int(score.shape[1])
        active_before = prev_decision_2d.squeeze(-1) > 0

        epsilon = 1e-6
        index_vals = jnp.arange(N, dtype=score.dtype) * epsilon
        encoded_key = score - index_vals
        encoded_masked = jnp.where(active_before, encoded_key, float('-inf'))

        # Pad to next power of 2
        padded_count = 1
        while padded_count < N:
            padded_count *= 2
        if padded_count > N:
            pad_width = padded_count - N
            neg_inf = jnp.full((int(score.shape[0]), pad_width), float('-inf'), dtype=score.dtype)
            sortable = jnp.concatenate([encoded_masked, neg_inf], axis=1)
        else:
            sortable = encoded_masked

        # Bitonic sort with index tracking: one MPC op instead of two
        sorted_keys, sorted_indices = _bitonic_sort_desc_with_indices(sortable)

        # Build keep mask directly from sorted indices (vectorized).
        # The first keep_count elements in sorted order are the top-k tokens.
        # Compare each position against sorted_indices[:keep_count] to find kept tokens.
        top_k_indices = sorted_indices[:, :keep_count]  # [B, keep_count]
        pos = jnp.arange(N, dtype=jnp.int32)  # [N]
        # keep_mask[b, n] = 1 iff position n appears in top_k_indices[b]
        keep_mask = jnp.any(top_k_indices[:, :, None] == pos[None, None, :], axis=1)  # [B, N]
        keep_mask = keep_mask[:, :N] & active_before
        return keep_mask[:, :, None]


    def forward_with_secure_pruning_fn(pixel_values, forward_params, predictor_params):
        """Full forward with PredictorLG pruning running inside SPU.

        Args:
            pixel_values: [B, C, H, W] input image.
            forward_params: standard (patch_weight, ..., head_bias) tuple.
            predictor_params: tuple of per-stage PredictorLG param tuples.
        Note:
            pruning_loc and token_keep_counts are captured from the enclosing scope.
        """
        (
            patch_weight,
            patch_bias,
            cls_token,
            pos_embed,
            block_params,
            norm_weight,
            norm_bias,
            head_weight,
            head_bias,
        ) = forward_params

        loc_set = frozenset(pruning_loc)
        keep_counts = tuple(int(x) for x in token_keep_counts_raw)

        block_calibrations = () if layer_norm_calibration is None else layer_norm_calibration["blocks"]
        x = patch_embed(pixel_values, patch_weight, patch_bias)
        batch = x.shape[0]
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed

        spatial_token_count = int(x.shape[1] - 1)
        prev_decision = jnp.ones((batch, spatial_token_count, 1), dtype=x.dtype)
        stage_index = 0

        for block_index, block_param in enumerate(block_params):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            if block_index in loc_set:
                # Run PredictorLG on spatial tokens
                spatial_x = x[:, 1:]
                keep_score = _predictorlg_forward(
                    spatial_x, prev_decision, predictor_params[stage_index]
                )
                # Build keep decision
                prev_decision = _secure_build_keep_decision(
                    keep_score, prev_decision, int(keep_counts[stage_index])
                )
                # Apply spatial mask with Dropped-Token Context Recycling
                x = _apply_spatial_mask_with_recycle(x, prev_decision, keep_score)
                # Run block
                x = block_forward(x, block_param, block_calibration)
                # Apply spatial mask again (post-block) — no recycling needed here
                x = apply_spatial_mask(x, prev_decision)
                stage_index += 1
            else:
                x = block_forward(x, block_param, block_calibration)
                x = apply_spatial_mask(x, prev_decision)

        final_calibration = None if layer_norm_calibration is None else layer_norm_calibration["final_norm"]
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    def forward_with_secure_pruning_from_shares_fn(share0, share1, forward_params, predictor_params):
        return forward_with_secure_pruning_fn(
            share0 + share1, forward_params, predictor_params
        )



# =========================================================================
# Secure in-SPU pruning: PredictorLG + kth-threshold + tie-resolution
# =========================================================================


    def load_share_chunk_from_party_manifest(manifest_path, start_index, end_index, padded_batch_size):
        import json
        from pathlib import Path

        import numpy as np
        import torch

        manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
        share_path = Path(manifest["share_path"]).expanduser()
        if manifest.get("share_storage_format") == "raw_float32_le":
            raw = np.fromfile(share_path, dtype="<f4")
            expected_shape = tuple(int(dim) for dim in manifest.get("share_shape", []))
            expected_size = int(np.prod(expected_shape))
            if raw.size != expected_size:
                raise ValueError(f"raw share size mismatch for {share_path}: {raw.size} vs {expected_size}")
            full_share = raw.reshape(expected_shape).astype(np.float32, copy=False)
            chunk = full_share[int(start_index) : int(end_index)]
        else:
            payload = torch.load(share_path, map_location="cpu")
            if payload.get("manifest_type") != "transshield_e2e_debug_float_additive_share_v0":
                raise ValueError(f"unsupported debug share payload in {share_path}: {payload.get('manifest_type')}")
            if int(payload.get("share_rank")) != int(manifest["share_rank"]):
                raise ValueError(
                    f"share rank mismatch for {share_path}: payload={payload.get('share_rank')} "
                    f"manifest={manifest['share_rank']}"
                )
            chunk_tensor = payload["share_tensor"][int(start_index) : int(end_index)].detach().cpu().float()
            chunk = chunk_tensor.numpy().astype(np.float32, copy=False)
        actual_count = int(chunk.shape[0])
        padded_batch_size = int(padded_batch_size)
        if actual_count < padded_batch_size:
            pad_shape = (padded_batch_size - actual_count,) + tuple(chunk.shape[1:])
            chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        return chunk

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
    elif params_mode in {
        "secret_patch_head_public_blocks",
        "public_patch_head_secret_blocks",
        "secret_patch_head_secret_blocks_split",
        "secret_patch_public_head_secret_blocks",
        "public_patch_secret_head_secret_blocks",
        "secret_three_stage",
        "secret_blockwise_stage",
        "secret_block_group_stage",
    }:
        params_ref = None
    else:
        raise ValueError(f"unsupported SPU params mode: {params_mode}")

    using_party_local_share_load = pixel_value_share_manifest_paths is not None
    using_input_shares = pixel_value_shares_np is not None
    if using_party_local_share_load and using_input_shares:
        raise ValueError("pixel_value_shares_np and pixel_value_share_manifest_paths are mutually exclusive")
    if using_input_shares:
        share0_np = np.asarray(pixel_value_shares_np[0], dtype=np.float32)
        share1_np = np.asarray(pixel_value_shares_np[1], dtype=np.float32)
        if tuple(share0_np.shape) != tuple(share1_np.shape):
            raise ValueError(f"input share shape mismatch: {share0_np.shape} vs {share1_np.shape}")
        pixel_values_np = share0_np
    elif using_party_local_share_load:
        if share_sample_count is None:
            raise ValueError("share_sample_count is required with pixel_value_share_manifest_paths")
        if len(pixel_value_share_manifest_paths) != 2:
            raise ValueError("expected exactly two party share manifest paths")
        pixel_values_np = np.zeros((int(share_sample_count), 1, 1, 1), dtype=np.float32)
    else:
        pixel_values_np = np.asarray(pixel_values_np, dtype=np.float32)
    if pixel_values_np.ndim != 4:
        raise ValueError(f"expected NCHW pixel_values, got shape={pixel_values_np.shape}")
    sample_count = int(pixel_values_np.shape[0])
    if sample_count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    external_keep_masks_np = None if external_keep_masks_np is None else tuple(external_keep_masks_np)
    using_secure_pruning = predictor_params_np is not None
    if using_secure_pruning:
        if external_keep_masks_np is not None:
            raise ValueError("predictor_params_np and external_keep_masks_np are mutually exclusive")
        if probe_block_index is not None:
            raise ValueError("predictor_params_np is not supported with probe_block_index")
        if using_chunked_forward:
            raise ValueError("predictor_params_np is not supported with block_chunk_size")
        if attention_policy != "uniform":
            raise ValueError("predictor_params_np currently requires attention_policy=uniform")
        if params_mode not in {"public", "secret"}:
            raise ValueError("predictor_params_np currently supports only public or secret SPU params mode")
        predictor_params_np = tuple(predictor_params_np)
        if pruning_metadata is None:
            pruning_metadata = {
                "pruning_loc": list(pruning_loc),
                "token_keep_counts": [
                    int(int(metadata.get('pruning_loc', pruning_loc)[i] is not None) * 1)
                    for i in range(len(pruning_loc))
                ],
            }
        # Sync closure-level mutable list so SPU closures see concrete Python ints
        token_keep_counts_raw[:] = [int(x) for x in pruning_metadata.get("token_keep_counts", [])]

    effective_batch_size = max(1, min(int(batch_size), sample_count))
    (
        patch_weight,
        patch_bias,
        cls_token,
        pos_embed,
        block_params,
        norm_weight,
        norm_bias,
        head_weight,
        head_bias,
    ) = params
    patch_params = (
        patch_weight,
        patch_bias,
        cls_token,
        pos_embed,
    )
    final_head_params = (
        norm_weight,
        norm_bias,
        head_weight,
        head_bias,
    )
    if external_keep_masks_np is not None:
        expected_stage_count = sum(1 for block_index in range(int(metadata["depth"])) if block_index in pruning_loc)
        if len(external_keep_masks_np) != expected_stage_count:
            raise ValueError(
                f"external_keep_masks_np stage count mismatch: {len(external_keep_masks_np)} vs {expected_stage_count}"
            )
        normalized_keep_masks = []
        for stage_index, keep_mask in enumerate(external_keep_masks_np):
            keep_mask = np.asarray(keep_mask, dtype=np.float32)
            if keep_mask.ndim == 3 and keep_mask.shape[-1] == 1:
                keep_mask = keep_mask[..., 0]
            if keep_mask.ndim != 2:
                raise ValueError(
                    f"external keep mask stage {stage_index} must have shape [N,T] or [N,T,1], got {keep_mask.shape}"
                )
            if int(keep_mask.shape[0]) != sample_count:
                raise ValueError(
                    f"external keep mask sample count mismatch at stage {stage_index}: {keep_mask.shape[0]} vs {sample_count}"
                )
            normalized_keep_masks.append(keep_mask)
        external_keep_masks_np = tuple(normalized_keep_masks)
    patch_head_params = patch_params + final_head_params
    split_patch_head_ref = patch_head_params
    split_patch_ref = patch_params
    split_head_ref = final_head_params
    split_block_params_ref = block_params
    using_split_params = params_mode in {
        "secret_patch_head_public_blocks",
        "public_patch_head_secret_blocks",
        "secret_patch_head_secret_blocks_split",
        "secret_patch_public_head_secret_blocks",
        "public_patch_secret_head_secret_blocks",
        "secret_three_stage",
        "secret_blockwise_stage",
        "secret_block_group_stage",
    }
    using_three_part_params = params_mode in {
        "secret_patch_public_head_secret_blocks",
        "public_patch_secret_head_secret_blocks",
    }
    if params_mode in {"secret_patch_head_public_blocks", "secret_patch_head_secret_blocks_split"}:
        split_patch_head_ref = secret_identity(patch_head_params)
    if params_mode in {"secret_patch_public_head_secret_blocks"}:
        split_patch_ref = secret_identity(patch_params)
    if params_mode in {"public_patch_secret_head_secret_blocks"}:
        split_head_ref = secret_identity(final_head_params)
    if params_mode in {"public_patch_head_secret_blocks", "secret_patch_head_secret_blocks_split"}:
        split_block_params_ref = secret_identity(block_params)
    if params_mode in {"secret_patch_public_head_secret_blocks", "public_patch_secret_head_secret_blocks"}:
        split_block_params_ref = secret_identity(block_params)
    if params_mode == "secret_three_stage":
        split_patch_ref = secret_identity(patch_params)
        split_block_params_ref = secret_identity(block_params)
        split_head_ref = secret_identity(final_head_params)
    secret_block_param_refs = None
    if params_mode in {"secret_blockwise_stage", "secret_block_group_stage"}:
        # These refs are intentionally rebuilt per input chunk below. Reusing the
        # same secret PYU object refs across many staged SPU calls has shown
        # unstable behavior in long party-local sample loops.
        secret_block_param_refs = ()
    block_calibrations = () if layer_norm_calibration is None else tuple(layer_norm_calibration["blocks"])
    final_calibration = (
        (np.zeros_like(norm_weight, dtype=np.float32), np.ones_like(norm_weight, dtype=np.float32))
        if layer_norm_calibration is None
        else layer_norm_calibration["final_norm"]
    )
    chunked_segments = []
    chunked_calibration_segments = []
    if using_chunked_forward:
        for segment_start in range(0, len(block_params), block_chunk_size):
            segment_end = min(segment_start + block_chunk_size, len(block_params))
            chunked_segments.append(tuple(block_params[segment_start:segment_end]))
            chunked_calibration_segments.append(tuple(block_calibrations[segment_start:segment_end]))
        if not chunked_segments:
            chunked_segments.append(tuple())
            chunked_calibration_segments.append(tuple())
    prefix_params = (
        patch_weight,
        patch_bias,
        cls_token,
        pos_embed,
        chunked_segments[0] if using_chunked_forward else tuple(),
        chunked_calibration_segments[0] if using_chunked_forward else tuple(),
    )
    head_params = (norm_weight, norm_bias, head_weight, head_bias, final_calibration)
    outputs = []
    probe_outputs = None
    if probe_block_index is not None:
        probe_outputs = {
            "block_input_cls": [],
            "norm1_out_cls": [],
            "attn_out_cls": [],
            "attn_residual_out_cls": [],
            "norm2_out_cls": [],
            "mlp_out_cls": [],
            "block_output_cls": [],
            "final_norm_cls": [],
            "head_input_cls": [],
        }

    def slice_keep_mask_chunk(start_index, end_index, padded_batch_size):
        if external_keep_masks_np is None:
            return None
        chunk_keep_masks = []
        for keep_mask in external_keep_masks_np:
            chunk = keep_mask[int(start_index) : int(end_index)]
            actual_count = int(chunk.shape[0])
            padded_batch_size = int(padded_batch_size)
            if actual_count < padded_batch_size:
                pad_shape = (padded_batch_size - actual_count, int(chunk.shape[1]))
                chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)
            chunk_keep_masks.append(chunk.astype(np.float32, copy=False))
        return tuple(chunk_keep_masks)

    def slice_predictor_params_chunk():
        """Slice predictor params for the current batch (they are full-batch numpy)."""
        return predictor_params_np

    def make_secret_blockwise_refs():
        if secret_block_param_refs is None:
            raise ValueError("secret_blockwise refs are required for secret_blockwise_stage")
        return (
            secret_identity(patch_params),
            [secret_identity(block_param) for block_param in block_params],
            secret_identity(final_head_params),
        )

    def make_secret_block_group_refs():
        if secret_block_param_refs is None:
            raise ValueError("secret block-group refs are required for secret_block_group_stage")
        return (
            secret_identity(patch_params),
            [secret_identity(segment_block_params) for segment_block_params in chunked_segments],
            secret_identity(final_head_params),
        )

    def run_secret_blockwise_stage(x_ref, block_param_refs):
        for block_index, block_param_ref in enumerate(block_param_refs):
            block_calibration = None if not block_calibrations else block_calibrations[block_index]
            x_ref = spu(single_block_fn)(x_ref, block_param_ref, block_calibration)
        return x_ref

    def run_secret_block_group_stage(x_ref, segment_param_refs):
        for segment_index, segment_param_ref in enumerate(segment_param_refs):
            x_ref = spu(blocks_only_fn)(
                x_ref,
                (segment_param_ref, chunked_calibration_segments[segment_index]),
            )
        return x_ref

    for start in range(0, sample_count, effective_batch_size):
        end = min(start + effective_batch_size, sample_count)
        chunk = pixel_values_np[start:end]
        if using_input_shares:
            chunk_share1 = share1_np[start:end]
        actual_count = int(end - start) if using_party_local_share_load else int(chunk.shape[0])
        if actual_count < effective_batch_size:
            pad_shape = (effective_batch_size - actual_count,) + tuple(chunk.shape[1:])
            chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)
            if using_input_shares:
                chunk_share1 = np.concatenate(
                    [chunk_share1, np.zeros(pad_shape, dtype=np.float32)],
                    axis=0,
                )
        chunk_keep_masks = slice_keep_mask_chunk(start, end, effective_batch_size)
        if using_secure_pruning:
            # Build pruning metadata dict for the SPU closure
            spu_pruning_metadata = dict(pruning_metadata)
            predictor_stage_params = slice_predictor_params_chunk()
            if using_input_shares:
                result_ref = spu(forward_with_secure_pruning_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    params_ref,
                    predictor_stage_params,
                )
            elif using_party_local_share_load:
                chunk_share0_ref = p1(load_share_chunk_from_party_manifest)(
                    str(pixel_value_share_manifest_paths[0]),
                    start,
                    end,
                    effective_batch_size,
                )
                chunk_share1_ref = p2(load_share_chunk_from_party_manifest)(
                    str(pixel_value_share_manifest_paths[1]),
                    start,
                    end,
                    effective_batch_size,
                )
                result_ref = spu(forward_with_secure_pruning_from_shares_fn)(
                    chunk_share0_ref,
                    chunk_share1_ref,
                    params_ref,
                    predictor_stage_params,
                )
            elif params_mode == "secret":
                result_ref = spu(forward_with_secure_pruning_fn)(
                    secret_identity(chunk),
                    params_ref,
                    predictor_stage_params,
                )
            else:
                result_ref = spu(forward_with_secure_pruning_fn)(
                    secret_identity(chunk),
                    params,
                    predictor_stage_params,
                )
        elif external_keep_masks_np is not None:
            if using_input_shares:
                result_ref = spu(forward_with_external_keep_masks_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    params_ref,
                    chunk_keep_masks,
                )
            elif using_party_local_share_load:
                chunk_share0_ref = p1(load_share_chunk_from_party_manifest)(
                    str(pixel_value_share_manifest_paths[0]),
                    start,
                    end,
                    effective_batch_size,
                )
                chunk_share1_ref = p2(load_share_chunk_from_party_manifest)(
                    str(pixel_value_share_manifest_paths[1]),
                    start,
                    end,
                    effective_batch_size,
                )
                result_ref = spu(forward_with_external_keep_masks_from_shares_fn)(
                    chunk_share0_ref,
                    chunk_share1_ref,
                    params_ref,
                    chunk_keep_masks,
                )
            else:
                result_ref = spu(forward_with_external_keep_masks_fn)(
                    secret_identity(chunk),
                    params_ref,
                    chunk_keep_masks,
                )
        elif using_input_shares:
            if params_mode == "secret_three_stage":
                x_ref = spu(embed_only_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    split_patch_ref,
                )
                x_ref = spu(blocks_only_fn)(x_ref, (split_block_params_ref, block_calibrations))
                result_ref = spu(norm_head_split_fn)(x_ref, split_head_ref, final_calibration)
            elif params_mode == "secret_blockwise_stage":
                chunk_patch_ref, chunk_block_param_refs, chunk_head_ref = make_secret_blockwise_refs()
                x_ref = spu(embed_only_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    chunk_patch_ref,
                )
                x_ref = run_secret_blockwise_stage(x_ref, chunk_block_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif params_mode == "secret_block_group_stage":
                chunk_patch_ref, chunk_segment_param_refs, chunk_head_ref = make_secret_block_group_refs()
                x_ref = spu(embed_only_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    chunk_patch_ref,
                )
                x_ref = run_secret_block_group_stage(x_ref, chunk_segment_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif using_chunked_forward:
                x_ref = spu(embed_and_blocks_from_shares_fn)(
                    secret_identity(chunk),
                    p2_secret_identity(chunk_share1),
                    prefix_params,
                )
                for segment_index, segment_block_params in enumerate(chunked_segments[1:], start=1):
                    x_ref = spu(blocks_only_fn)(
                        x_ref,
                        (segment_block_params, chunked_calibration_segments[segment_index]),
                    )
                result_ref = spu(norm_head_fn)(x_ref, head_params)
            else:
                if using_three_part_params:
                    result_ref = spu(forward_three_part_params_from_shares_fn)(
                        secret_identity(chunk),
                        p2_secret_identity(chunk_share1),
                        split_patch_ref,
                        split_block_params_ref,
                        split_head_ref,
                    )
                elif using_split_params:
                    result_ref = spu(forward_split_params_from_shares_fn)(
                        secret_identity(chunk),
                        p2_secret_identity(chunk_share1),
                        split_patch_head_ref,
                        split_block_params_ref,
                    )
                else:
                    result_ref = spu(forward_from_shares_fn)(
                        secret_identity(chunk),
                        p2_secret_identity(chunk_share1),
                        params_ref,
                    )
        elif using_party_local_share_load:
            chunk_share0_ref = p1(load_share_chunk_from_party_manifest)(
                str(pixel_value_share_manifest_paths[0]),
                start,
                end,
                effective_batch_size,
            )
            chunk_share1_ref = p2(load_share_chunk_from_party_manifest)(
                str(pixel_value_share_manifest_paths[1]),
                start,
                end,
                effective_batch_size,
            )
            if params_mode == "secret_three_stage":
                x_ref = spu(embed_only_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, split_patch_ref)
                x_ref = spu(blocks_only_fn)(x_ref, (split_block_params_ref, block_calibrations))
                result_ref = spu(norm_head_split_fn)(x_ref, split_head_ref, final_calibration)
            elif params_mode == "secret_blockwise_stage":
                chunk_patch_ref, chunk_block_param_refs, chunk_head_ref = make_secret_blockwise_refs()
                x_ref = spu(embed_only_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, chunk_patch_ref)
                x_ref = run_secret_blockwise_stage(x_ref, chunk_block_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif params_mode == "secret_block_group_stage":
                chunk_patch_ref, chunk_segment_param_refs, chunk_head_ref = make_secret_block_group_refs()
                x_ref = spu(embed_only_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, chunk_patch_ref)
                x_ref = run_secret_block_group_stage(x_ref, chunk_segment_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif using_chunked_forward:
                x_ref = spu(embed_and_blocks_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, prefix_params)
                for segment_index, segment_block_params in enumerate(chunked_segments[1:], start=1):
                    x_ref = spu(blocks_only_fn)(
                        x_ref,
                        (segment_block_params, chunked_calibration_segments[segment_index]),
                    )
                result_ref = spu(norm_head_fn)(x_ref, head_params)
            else:
                if using_three_part_params:
                    result_ref = spu(forward_three_part_params_from_shares_fn)(
                        chunk_share0_ref,
                        chunk_share1_ref,
                        split_patch_ref,
                        split_block_params_ref,
                        split_head_ref,
                    )
                elif using_split_params:
                    result_ref = spu(forward_split_params_from_shares_fn)(
                        chunk_share0_ref,
                        chunk_share1_ref,
                        split_patch_head_ref,
                        split_block_params_ref,
                    )
                else:
                    result_ref = spu(forward_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, params_ref)
        else:
            if params_mode == "secret_three_stage":
                x_ref = spu(embed_only_fn)(secret_identity(chunk), split_patch_ref)
                x_ref = spu(blocks_only_fn)(x_ref, (split_block_params_ref, block_calibrations))
                result_ref = spu(norm_head_split_fn)(x_ref, split_head_ref, final_calibration)
            elif params_mode == "secret_blockwise_stage":
                chunk_patch_ref, chunk_block_param_refs, chunk_head_ref = make_secret_blockwise_refs()
                x_ref = spu(embed_only_fn)(secret_identity(chunk), chunk_patch_ref)
                x_ref = run_secret_blockwise_stage(x_ref, chunk_block_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif params_mode == "secret_block_group_stage":
                chunk_patch_ref, chunk_segment_param_refs, chunk_head_ref = make_secret_block_group_refs()
                x_ref = spu(embed_only_fn)(secret_identity(chunk), chunk_patch_ref)
                x_ref = run_secret_block_group_stage(x_ref, chunk_segment_param_refs)
                result_ref = spu(norm_head_split_fn)(x_ref, chunk_head_ref, final_calibration)
            elif using_chunked_forward:
                x_ref = spu(embed_and_blocks_fn)(secret_identity(chunk), prefix_params)
                for segment_index, segment_block_params in enumerate(chunked_segments[1:], start=1):
                    x_ref = spu(blocks_only_fn)(
                        x_ref,
                        (segment_block_params, chunked_calibration_segments[segment_index]),
                    )
                result_ref = spu(norm_head_fn)(x_ref, head_params)
            else:
                if using_three_part_params:
                    result_ref = spu(forward_three_part_params_fn)(
                        secret_identity(chunk),
                        split_patch_ref,
                        split_block_params_ref,
                        split_head_ref,
                    )
                elif using_split_params:
                    result_ref = spu(forward_split_params_fn)(
                        secret_identity(chunk),
                        split_patch_head_ref,
                        split_block_params_ref,
                    )
                else:
                    result_ref = spu(forward_fn)(secret_identity(chunk), params_ref)
        result = ppd.get(result_ref)
        if probe_block_index is None:
            logits = np.asarray(result, dtype=np.float32)
            outputs.append(logits[:actual_count])
            continue

        logits = np.asarray(result[0], dtype=np.float32)
        outputs.append(logits[:actual_count])
        for name, value in zip(probe_outputs, result[1:]):
            probe_outputs[name].append(np.asarray(value, dtype=np.float32)[:actual_count])

    logits_output = np.concatenate(outputs, axis=0)
    if probe_outputs is None:
        return logits_output

    for name in probe_outputs:
        probe_outputs[name] = np.concatenate(probe_outputs[name], axis=0)
    return logits_output, probe_outputs


def compare_array_payload(reference, candidate):
    import numpy as np

    reference_np = np.asarray(reference, dtype=np.float32)
    candidate_np = np.asarray(candidate, dtype=np.float32)
    if tuple(reference_np.shape) != tuple(candidate_np.shape):
        raise ValueError(f"array shape mismatch: {reference_np.shape} vs {candidate_np.shape}")
    diff = reference_np - candidate_np
    abs_diff = np.abs(diff)
    reference_flat = reference_np.reshape(-1)
    candidate_flat = candidate_np.reshape(-1)
    reference_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    denom = max(reference_norm * candidate_norm, 1e-12)
    return {
        "shape": list(reference_np.shape),
        "reference_min": float(reference_np.min()),
        "reference_max": float(reference_np.max()),
        "candidate_min": float(candidate_np.min()),
        "candidate_max": float(candidate_np.max()),
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "l2_error": float(np.linalg.norm(diff.reshape(-1))),
        "reference_l2_norm": reference_norm,
        "candidate_l2_norm": candidate_norm,
        "cosine_similarity": float(np.dot(reference_flat, candidate_flat) / denom),
        "finite_candidate": bool(np.isfinite(candidate_np).all()),
    }


def static_patch_embed_numpy(pixel_values_np, params, metadata):
    import numpy as np

    patch_weight, patch_bias, cls_token, pos_embed = params[:4]
    patch_size = int(metadata["patch_size"])
    batch, channels, height, width = pixel_values_np.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = np.reshape(
        pixel_values_np,
        (batch, channels, grid_h, patch_size, grid_w, patch_size),
    )
    patches = np.transpose(patches, (0, 2, 4, 1, 3, 5))
    patches = np.reshape(patches, (batch, grid_h * grid_w, channels * patch_size * patch_size))
    patch_weight_flat = np.reshape(patch_weight, (patch_weight.shape[0], -1))
    patch_tokens = np.matmul(patches, np.swapaxes(patch_weight_flat, -1, -2)) + patch_bias
    cls_tokens = np.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
    tokens_with_pos = np.concatenate([cls_tokens, patch_tokens], axis=1) + pos_embed
    return patch_tokens.astype(np.float32), tokens_with_pos.astype(np.float32)


def run_share_recomposition_audit_spu(
    params,
    config_path: Path,
    metadata,
    params_mode: str,
    pixel_value_shares_np,
):
    import numpy as np
    import jax.numpy as jnp
    import spu.utils.distributed as ppd

    config = load_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    patch_size = int(metadata["patch_size"])

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
