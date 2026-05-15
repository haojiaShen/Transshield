from pathlib import Path

from integrations.openbumblebee.e2e_secure_vit.common import load_json, write_json
from integrations.openbumblebee.e2e_secure_vit.static_vit_params import load_static_vit_spu_params


CALIBRATION_MANIFEST_TYPE = "transshield_e2e_public_layer_norm_calibration_v0"


def _torch_from_numpy(value, torch):
    return torch.as_tensor(value, dtype=torch.float32)


def _linear(x, weight, bias, torch):
    return torch.matmul(x, torch.swapaxes(weight, -1, -2)) + bias


def _record_feature_stats(x):
    variance, mean = torch_var_mean(x)
    return {
        "mean": mean.detach().cpu().tolist(),
        "variance": variance.detach().cpu().tolist(),
    }


def torch_var_mean(x):
    import torch

    reduce_dims = tuple(range(x.ndim - 1))
    variance, mean = torch.var_mean(x.float(), dim=reduce_dims, unbiased=False)
    variance = torch.clamp(variance, min=1e-8)
    return variance, mean


def compute_public_layer_norm_calibration(
    *,
    bundle_dir: Path,
    input_pt: Path,
    output_json: Path,
    static_depth_limit: int,
    max_samples: int = 0,
    attention_policy: str = "uniform",
    activation_override: str = "bundle",
    activation_clip_value: float = 0.0,
):
    import torch

    from integrations.openbumblebee.e2e_secure_vit.common import require_existing_file

    require_existing_file(input_pt, "public calibration pixel package")
    payload = torch.load(input_pt, map_location="cpu", weights_only=False)
    pixel_values = payload["pixel_values"].detach().cpu().float()
    if int(max_samples) > 0:
        pixel_values = pixel_values[: int(max_samples)]
    if pixel_values.ndim != 4 or int(pixel_values.shape[0]) <= 0:
        raise ValueError(f"expected non-empty NCHW calibration pixels, got shape={tuple(pixel_values.shape)}")

    params, metadata = load_static_vit_spu_params(
        bundle_dir,
        static_depth_limit=static_depth_limit,
        attention_policy=attention_policy,
        activation_override=activation_override,
    )
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
    ) = tuple(_torch_from_numpy(value, torch) if index not in {4} else value for index, value in enumerate(params))
    block_params = tuple(tuple(_torch_from_numpy(item, torch) for item in block_param) for block_param in block_params)

    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    patch_size = int(metadata["patch_size"])
    layer_norm_eps = float(metadata["layer_norm_eps"])
    attention_policy_eps = float(metadata["attention_policy_eps"])
    activation_kind = str(metadata["activation_kind"])
    activation_clip_value = float(activation_clip_value)
    attn_scale = head_dim ** -0.5

    def patch_embed(x):
        batch, channels, height, width = x.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = torch.reshape(x, (batch, channels, grid_h, patch_size, grid_w, patch_size))
        patches = torch.permute(patches, (0, 2, 4, 1, 3, 5))
        patches = torch.reshape(patches, (batch, grid_h * grid_w, channels * patch_size * patch_size))
        patch_weight_flat = torch.reshape(patch_weight, (patch_weight.shape[0], -1))
        return _linear(patches, patch_weight_flat, patch_bias, torch)

    def calibrated_layer_norm(x, weight, bias, stats_record):
        variance, mean = torch_var_mean(x)
        stats_record["mean"] = mean.detach().cpu().tolist()
        stats_record["variance"] = variance.detach().cpu().tolist()
        return (x - mean) * torch.rsqrt(variance + layer_norm_eps) * weight + bias

    def activate(x, alpha, beta):
        if activation_clip_value > 0.0:
            x = torch.clamp(x, min=-activation_clip_value, max=activation_clip_value)
        if activation_kind == "gelu":
            return torch.nn.functional.gelu(x, approximate="none")
        if activation_kind in {"fixed_square", "learnable_square"}:
            return alpha * (x * x)
        if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
            return alpha * (x * x) + beta * x
        raise ValueError(f"unsupported activation kind: {activation_kind}")

    def attention_softmax(attn):
        if attention_policy == "uniform":
            return torch.ones_like(attn) / attn.shape[-1]
        if attention_policy == "identity":
            raise ValueError("identity attention does not materialize an attention matrix")
        shifted = attn - torch.max(attn, dim=-1, keepdim=True).values
        exp_values = torch.exp(shifted)
        if attention_policy == "standard":
            return exp_values / torch.sum(exp_values, dim=-1, keepdim=True)
        token_count = attn.shape[-1]
        return (exp_values + attention_policy_eps / token_count) / (
            torch.sum(exp_values, dim=-1, keepdim=True) + attention_policy_eps
        )

    x = patch_embed(pixel_values)
    batch = int(x.shape[0])
    x = torch.cat([cls_token.expand(batch, -1, -1), x], dim=1)
    x = x + pos_embed

    block_stats = []
    with torch.no_grad():
        for block_param in block_params:
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
            stats = {"norm1": {}, "norm2": {}}
            residual = x
            norm1_out = calibrated_layer_norm(x, norm1_weight, norm1_bias, stats["norm1"])
            batch, token_count, channels = norm1_out.shape
            if attention_policy == "uniform":
                value_weight = qkv_weight[2 * channels : 3 * channels, :]
                value_bias = qkv_bias[2 * channels : 3 * channels]
                value = _linear(norm1_out, value_weight, value_bias, torch)
                value = torch.reshape(value, (batch, token_count, num_heads, head_dim))
                value = torch.permute(value, (0, 2, 1, 3))
                mean_value = torch.mean(value, dim=2, keepdim=True)
                attn_out = torch.broadcast_to(mean_value, (batch, num_heads, token_count, head_dim))
            else:
                qkv = _linear(norm1_out, qkv_weight, qkv_bias, torch)
                qkv = torch.reshape(qkv, (batch, token_count, 3, num_heads, head_dim))
                qkv = torch.permute(qkv, (2, 0, 3, 1, 4))
                query, key, value = qkv[0], qkv[1], qkv[2]
                if attention_policy == "identity":
                    attn_out = value
                else:
                    attn = torch.matmul(query, torch.swapaxes(key, -1, -2)) * attn_scale
                    attn_out = torch.matmul(attention_softmax(attn), value)
            attn_out = torch.permute(attn_out, (0, 2, 1, 3))
            attn_out = torch.reshape(attn_out, (batch, token_count, channels))
            x = residual + _linear(attn_out, proj_weight, proj_bias, torch)

            residual = x
            norm2_out = calibrated_layer_norm(x, norm2_weight, norm2_bias, stats["norm2"])
            mlp_hidden = activate(_linear(norm2_out, fc1_weight, fc1_bias, torch), act_alpha, act_beta)
            x = residual + _linear(mlp_hidden, fc2_weight, fc2_bias, torch)
            block_stats.append(stats)

        final_stats = {}
        final_norm = calibrated_layer_norm(x, norm_weight, norm_bias, final_stats)
        logits = _linear(final_norm[:, 0], head_weight, head_bias, torch)

    output = {
        "manifest_type": CALIBRATION_MANIFEST_TYPE,
        "bundle_dir": str(Path(bundle_dir).expanduser().resolve()),
        "input_pt": str(Path(input_pt).expanduser().resolve()),
        "sample_count": int(pixel_values.shape[0]),
        "static_depth_limit": int(static_depth_limit),
        "metadata": metadata,
        "policy": "public_calibrated",
        "stats_kind": "per_feature_public_activation",
        "activation_clip_value": float(activation_clip_value),
        "blocks": block_stats,
        "final_norm": final_stats,
        "calibration_logits": {
            "shape": list(logits.shape),
            "min": float(logits.min().item()),
            "max": float(logits.max().item()),
            "mean": float(logits.mean().item()),
            "std": float(logits.std(unbiased=False).item()),
        },
    }
    write_json(output_json, output)
    return output


def load_public_layer_norm_calibration(path: Path, *, expected_depth: int):
    import numpy as np

    payload = load_json(Path(path).expanduser())
    if payload.get("manifest_type") != CALIBRATION_MANIFEST_TYPE:
        raise ValueError(f"unsupported layer-norm calibration manifest: {payload.get('manifest_type')}")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != int(expected_depth):
        raise ValueError(f"calibration depth mismatch: expected {expected_depth}, got {len(blocks) if isinstance(blocks, list) else 'invalid'}")

    def pair(record):
        mean = np.asarray(record["mean"], dtype=np.float32)
        variance = np.asarray(record["variance"], dtype=np.float32)
        if mean.shape != variance.shape:
            raise ValueError(f"calibration mean/variance shape mismatch: {mean.shape} vs {variance.shape}")
        return mean, variance

    block_pairs = []
    for block in blocks:
        norm1_mean, norm1_var = pair(block["norm1"])
        norm2_mean, norm2_var = pair(block["norm2"])
        block_pairs.append((norm1_mean, norm1_var, norm2_mean, norm2_var))
    final_mean, final_var = pair(payload["final_norm"])
    return {
        "path": str(Path(path).expanduser().resolve()),
        "manifest": payload,
        "blocks": tuple(block_pairs),
        "final_norm": (final_mean, final_var),
    }
