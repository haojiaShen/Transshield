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
    using_chunked_forward = block_chunk_size > 0
    if using_chunked_forward and probe_block_index is not None:
        raise ValueError("block_chunk_size is not supported with probe_block_index")
    if using_chunked_forward and params_mode != "public":
        raise ValueError("block_chunk_size currently requires public SPU params")

    def linear(x, weight, bias):
        return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias

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

    def activate(x, alpha, beta):
        if activation_clip_value > 0.0:
            x = jnp.clip(x, -activation_clip_value, activation_clip_value)
        if activation_kind == "gelu":
            return gelu_exact(x)
        if activation_kind in {"fixed_square", "learnable_square"}:
            return alpha * (x * x)
        if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
            return alpha * (x * x) + beta * x
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
        block_input = x
        residual = x
        norm1_calibration = None if block_calibration is None else block_calibration[:2]
        norm2_calibration = None if block_calibration is None else block_calibration[2:]
        norm1_out = layer_norm(x, norm1_weight, norm1_bias, norm1_calibration)
        batch, token_count, channels = norm1_out.shape
        qkv = linear(norm1_out, qkv_weight, qkv_bias)
        qkv = jnp.reshape(qkv, (batch, token_count, 3, num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * attn_scale
        if attention_policy == "uniform":
            mean_value = jnp.mean(value, axis=2, keepdims=True)
            attn_out = jnp.broadcast_to(mean_value, (batch, num_heads, token_count, head_dim))
        elif attention_policy == "identity":
            attn_out = value
        else:
            attn = attention_softmax(attn)
            attn_out = jnp.matmul(attn, value)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = jnp.reshape(attn_out, (batch, token_count, channels))
        x = residual + linear(attn_out, proj_weight, proj_bias)

        residual = x
        norm2_out = layer_norm(x, norm2_weight, norm2_bias, norm2_calibration)
        mlp_hidden = activate(linear(norm2_out, fc1_weight, fc1_bias), act_alpha, act_beta)
        mlp_out = linear(mlp_hidden, fc2_weight, fc2_bias)
        block_output = residual + mlp_out
        if capture_probe:
            return block_output, (
                block_input[:, 0, :],
                norm1_out[:, 0, :],
                attn_out[:, 0, :],
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

    def norm_head_fn(x, head_params):
        norm_weight, norm_bias, head_weight, head_bias, final_calibration = head_params
        x = layer_norm(x, norm_weight, norm_bias, final_calibration)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

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
        if using_input_shares:
            if using_chunked_forward:
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
            if using_chunked_forward:
                x_ref = spu(embed_and_blocks_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, prefix_params)
                for segment_index, segment_block_params in enumerate(chunked_segments[1:], start=1):
                    x_ref = spu(blocks_only_fn)(
                        x_ref,
                        (segment_block_params, chunked_calibration_segments[segment_index]),
                    )
                result_ref = spu(norm_head_fn)(x_ref, head_params)
            else:
                result_ref = spu(forward_from_shares_fn)(chunk_share0_ref, chunk_share1_ref, params_ref)
        else:
            if using_chunked_forward:
                x_ref = spu(embed_and_blocks_fn)(secret_identity(chunk), prefix_params)
                for segment_index, segment_block_params in enumerate(chunked_segments[1:], start=1):
                    x_ref = spu(blocks_only_fn)(
                        x_ref,
                        (segment_block_params, chunked_calibration_segments[segment_index]),
                    )
                result_ref = spu(norm_head_fn)(x_ref, head_params)
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
