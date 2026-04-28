import json
from pathlib import Path

from integrations.openbumblebee.e2e_secure_vit.common import (
    load_json,
    numpy_from_torch_tensor,
    require_existing_file,
    write_json,
)
from integrations.openbumblebee.e2e_secure_vit.static_vit_params import load_static_vit_spu_params


def run_runtime_primitive_smoke(args):
    import traceback

    import numpy as np
    import jax.numpy as jnp
    import spu.utils.distributed as ppd

    output_json = Path(args.output_json).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    token_count = int(args.token_count)
    embed_dim = int(args.embed_dim)
    num_heads = int(args.num_heads)
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
    head_dim = embed_dim // num_heads
    mlp_dim = int(embed_dim * float(args.mlp_ratio))
    layer_norm_chunk_size = int(args.layer_norm_chunk_size)
    layer_norm_policy = str(args.layer_norm_policy)
    if layer_norm_policy not in {"exact", "affine"}:
        raise ValueError(f"unsupported layer_norm_policy: {layer_norm_policy}")
    attention_policy = str(args.attention_policy)
    if attention_policy not in {"standard", "uniform"}:
        raise ValueError(f"unsupported attention_policy: {attention_policy}")
    rng = np.random.default_rng(int(args.seed))

    x_np = rng.normal(0.0, 0.25, size=(1, token_count, embed_dim)).astype(np.float32)
    qkv_weight = rng.normal(0.0, 0.02, size=(3 * embed_dim, embed_dim)).astype(np.float32)
    qkv_bias = rng.normal(0.0, 0.02, size=(3 * embed_dim,)).astype(np.float32)
    proj_weight = rng.normal(0.0, 0.02, size=(embed_dim, embed_dim)).astype(np.float32)
    proj_bias = rng.normal(0.0, 0.02, size=(embed_dim,)).astype(np.float32)
    fc1_weight = rng.normal(0.0, 0.02, size=(mlp_dim, embed_dim)).astype(np.float32)
    fc1_bias = rng.normal(0.0, 0.02, size=(mlp_dim,)).astype(np.float32)
    fc2_weight = rng.normal(0.0, 0.02, size=(embed_dim, mlp_dim)).astype(np.float32)
    fc2_bias = rng.normal(0.0, 0.02, size=(embed_dim,)).astype(np.float32)

    attn_scale = head_dim ** -0.5

    def linear(x, weight, bias):
        return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias

    def stage_scalar_add(value):
        return value + 1.0

    def feature_sum(value):
        feature_dim = int(value.shape[-1])
        if layer_norm_chunk_size <= 0 or layer_norm_chunk_size >= feature_dim:
            return jnp.sum(value, axis=-1, keepdims=True)
        total = None
        for start in range(0, feature_dim, layer_norm_chunk_size):
            end = min(start + layer_norm_chunk_size, feature_dim)
            partial = jnp.sum(value[..., start:end], axis=-1, keepdims=True)
            total = partial if total is None else total + partial
        return total

    def stage_layer_norm(x):
        if layer_norm_policy == "affine":
            return x
        feature_dim = int(x.shape[-1])
        mean = feature_sum(x) / feature_dim
        centered = x - mean
        variance = feature_sum(centered * centered) / feature_dim
        return centered / jnp.sqrt(variance + 1e-6)

    def stage_qkv(normed):
        batch, tokens, _channels = normed.shape
        qkv = linear(normed, qkv_weight, qkv_bias)
        qkv = jnp.reshape(qkv, (batch, tokens, 3, num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        return qkv[0], qkv[1], qkv[2]

    def stage_attention_scores(qkv):
        query, key, _value = qkv
        return jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * attn_scale

    def stage_attention_probs(scores):
        if attention_policy == "uniform":
            return jnp.ones_like(scores) / token_count
        shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
        exp_values = jnp.exp(shifted)
        return exp_values / jnp.sum(exp_values, axis=-1, keepdims=True)

    def stage_attention_context(qkv, probs):
        _query, _key, value = qkv
        if attention_policy == "uniform":
            mean_value = jnp.mean(value, axis=2, keepdims=True)
            out = jnp.broadcast_to(mean_value, (1, num_heads, token_count, head_dim))
        else:
            out = jnp.matmul(probs, value)
        out = jnp.transpose(out, (0, 2, 1, 3))
        return jnp.reshape(out, (1, token_count, embed_dim))

    def stage_projection_residual(x, context):
        return x + linear(context, proj_weight, proj_bias)

    def stage_mlp_hidden(x):
        hidden = linear(x, fc1_weight, fc1_bias)
        return hidden * hidden

    def stage_mlp_residual(x, hidden):
        return x + linear(hidden, fc2_weight, fc2_bias)

    p1 = ppd.device("P1")
    spu = ppd.device("SPU")

    def identity(value):
        return value

    secret_identity = p1(identity)
    stage_records = []

    def record_array(name, value):
        array = np.asarray(value, dtype=np.float32)
        stage_records.append(
            {
                "name": name,
                "status": "passed",
                "shape": list(array.shape),
                "finite": bool(np.isfinite(array).all()),
                "max": float(array.max()),
                "min": float(array.min()),
                "mean": float(array.mean()),
                "std": float(array.std()),
                "l2_norm": float(np.linalg.norm(array.reshape(-1))),
            }
        )

    def write_summary(failed_stage=None, exception=None):
        summary = {
            "manifest_type": "transshield_e2e_spu_runtime_primitive_smoke_v0",
            "config": str(config_path),
            "output_json": str(output_json),
            "token_count": token_count,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mlp_dim": mlp_dim,
            "layer_norm_chunk_size": layer_norm_chunk_size,
            "layer_norm_policy": layer_norm_policy,
            "attention_policy": attention_policy,
            "seed": int(args.seed),
            "failed_stage": failed_stage,
            "stage_records": stage_records,
            "privacy_note": (
                "Debug-only synthetic SPU primitive smoke. It reveals synthetic stage outputs and "
                "does not use private images, share manifests, or model weights."
            ),
        }
        if exception is not None:
            summary["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
        write_json(output_json, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    def run_stage(name, fn, *fn_args):
        try:
            ref = spu(fn)(*fn_args)
            value = ppd.get(ref)
            if isinstance(value, tuple):
                for index, item in enumerate(value):
                    record_array(f"{name}[{index}]", item)
            else:
                record_array(name, value)
            return ref
        except Exception as exc:
            write_summary(failed_stage=name, exception=exc)
            raise

    scalar_ref = secret_identity(np.asarray([1.0], dtype=np.float32))
    run_stage("scalar_add", stage_scalar_add, scalar_ref)

    x_ref = secret_identity(x_np)
    norm_ref = run_stage("layer_norm", stage_layer_norm, x_ref)
    qkv_ref = run_stage("qkv_linear", stage_qkv, norm_ref)
    scores_ref = run_stage("attention_scores", stage_attention_scores, qkv_ref)
    probs_ref = run_stage("attention_probs", stage_attention_probs, scores_ref)
    context_ref = run_stage("attention_context", stage_attention_context, qkv_ref, probs_ref)
    residual_ref = run_stage("projection_residual", stage_projection_residual, x_ref, context_ref)
    hidden_ref = run_stage("mlp_hidden_square", stage_mlp_hidden, residual_ref)
    run_stage("mlp_residual", stage_mlp_residual, residual_ref, hidden_ref)
    write_summary()


def run_block1_subgraph_smoke(args, helpers):
    import traceback

    import numpy as np
    import torch
    import jax.numpy as jnp
    import jax.scipy.special as jsp_special
    import spu.utils.distributed as ppd

    bundle_dir = helpers["resolve_bundle_dir"](args.bundle_dir)
    input_pt = Path(args.input_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    require_existing_file(input_pt, "input pixel package")

    client_payload = helpers["load_client_pixel_package"](input_pt)
    pixel_values_cpu = client_payload["pixel_values"].detach().cpu()
    if args.max_samples > 0:
        pixel_values_cpu = pixel_values_cpu[: args.max_samples]
    pixel_values_np = numpy_from_torch_tensor(pixel_values_cpu)

    params, metadata = load_static_vit_spu_params(
        bundle_dir,
        1,
        attention_policy=args.spu_attention_policy,
        activation_override=args.spu_activation_override,
    )
    if args.spu_params_mode != "public":
        raise ValueError("block1-subgraph-smoke currently supports --spu-params-mode public only")

    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    ppd.init(config["nodes"], config["devices"])

    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    patch_size = int(metadata["patch_size"])
    layer_norm_eps = float(metadata["layer_norm_eps"])
    layer_norm_chunk_size = int(getattr(args, "layer_norm_chunk_size", 0))
    layer_norm_policy = str(getattr(args, "layer_norm_policy", "exact"))
    if layer_norm_policy not in {"exact", "affine"}:
        raise ValueError(f"unsupported layer_norm_policy: {layer_norm_policy}")
    attention_policy = str(metadata.get("attention_policy", "standard"))
    attention_policy_eps = float(metadata["attention_policy_eps"])
    activation_kind = str(metadata["activation_kind"])
    attn_scale = head_dim ** -0.5

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
    block0 = block_params[0]
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
    ) = block0

    def linear(x, weight, bias):
        return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias

    def feature_sum(value):
        feature_dim = int(value.shape[-1])
        if layer_norm_chunk_size <= 0 or layer_norm_chunk_size >= feature_dim:
            return jnp.sum(value, axis=-1, keepdims=True)
        total = None
        for start in range(0, feature_dim, layer_norm_chunk_size):
            end = min(start + layer_norm_chunk_size, feature_dim)
            partial = jnp.sum(value[..., start:end], axis=-1, keepdims=True)
            total = partial if total is None else total + partial
        return total

    def layer_norm(x, weight, bias):
        if layer_norm_policy == "affine":
            return x * weight + bias
        feature_dim = int(x.shape[-1])
        mean = feature_sum(x) / feature_dim
        centered = x - mean
        variance = feature_sum(centered * centered) / feature_dim
        inverse_std = 1.0 / jnp.sqrt(variance + layer_norm_eps)
        return centered * inverse_std * weight + bias

    def patch_embed(pixel_values):
        batch, channels, height, width = pixel_values.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = jnp.reshape(pixel_values, (batch, channels, grid_h, patch_size, grid_w, patch_size))
        patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))
        patches = jnp.reshape(patches, (batch, grid_h * grid_w, channels * patch_size * patch_size))
        patch_weight_flat = jnp.reshape(patch_weight, (patch_weight.shape[0], -1))
        x = linear(patches, patch_weight_flat, patch_bias)
        cls_tokens = jnp.broadcast_to(cls_token, (batch, cls_token.shape[1], cls_token.shape[2]))
        return jnp.concatenate([cls_tokens, x], axis=1) + pos_embed

    def gelu_exact(x):
        return 0.5 * x * (1.0 + jsp_special.erf(x / jnp.sqrt(2.0)))

    def activate(x):
        if activation_kind == "gelu":
            return gelu_exact(x)
        if activation_kind in {"fixed_square", "learnable_square"}:
            return act_alpha * (x * x)
        if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
            return act_alpha * (x * x) + act_beta * x
        raise ValueError(f"unsupported activation kind: {activation_kind}")

    def attention_softmax(attn):
        shifted = attn - jnp.max(attn, axis=-1, keepdims=True)
        exp_values = jnp.exp(shifted)
        if attention_policy == "standard":
            return exp_values / jnp.sum(exp_values, axis=-1, keepdims=True)
        token_count = attn.shape[-1]
        return (exp_values + attention_policy_eps / token_count) / (
            jnp.sum(exp_values, axis=-1, keepdims=True) + attention_policy_eps
        )

    def qkv_from_norm1(norm1_out):
        batch, token_count, _channels = norm1_out.shape
        qkv = linear(norm1_out, qkv_weight, qkv_bias)
        qkv = jnp.reshape(qkv, (batch, token_count, 3, num_heads, head_dim))
        return jnp.transpose(qkv, (2, 0, 3, 1, 4))

    def attention_context(qkv):
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * attn_scale
        attn = attention_softmax(attn)
        out = jnp.matmul(attn, value)
        out = jnp.transpose(out, (0, 2, 1, 3))
        batch, _heads, token_count, _head_dim = query.shape
        return jnp.reshape(out, (batch, token_count, num_heads * head_dim))

    def stage_patch(pixel_values):
        return patch_embed(pixel_values)

    def stage_norm1(x):
        return layer_norm(x, norm1_weight, norm1_bias)

    def stage_qkv(norm1_out):
        query, key, value = qkv_from_norm1(norm1_out)
        return query, key, value

    def stage_attention(qkv):
        return attention_context(qkv)

    def stage_attn_residual(x, attn_out):
        return x + linear(attn_out, proj_weight, proj_bias)

    def stage_norm2(x):
        return layer_norm(x, norm2_weight, norm2_bias)

    def stage_mlp_hidden(norm2_out):
        return activate(linear(norm2_out, fc1_weight, fc1_bias))

    def stage_block_output(attn_residual, mlp_hidden):
        return attn_residual + linear(mlp_hidden, fc2_weight, fc2_bias)

    def stage_head(x):
        x = layer_norm(x, norm_weight, norm_bias)
        cls_features = x[:, 0]
        return linear(cls_features, head_weight, head_bias)

    p1 = ppd.device("P1")
    spu = ppd.device("SPU")

    def identity(value):
        return value

    secret_identity = p1(identity)
    stage_records = []
    refs = {}

    def record_array(name, value):
        array = np.asarray(value, dtype=np.float32)
        stage_records.append(
            {
                "name": name,
                "status": "passed",
                "shape": list(array.shape),
                "finite": bool(np.isfinite(array).all()),
                "max": float(array.max()),
                "min": float(array.min()),
                "mean": float(array.mean()),
                "std": float(array.std()),
            }
        )

    def run_stage(name, fn, *args):
        try:
            ref = spu(fn)(*args)
            value = ppd.get(ref)
            if isinstance(value, tuple):
                for index, item in enumerate(value):
                    record_array(f"{name}[{index}]", item)
            else:
                record_array(name, value)
            return ref, value
        except Exception as exc:
            summary = {
                "manifest_type": "transshield_e2e_block1_subgraph_smoke_v0",
                "bundle_dir": str(bundle_dir),
                "input_pt": str(input_pt),
                "output_json": str(output_json),
                "config": str(config_path),
                "max_samples": int(args_namespace.max_samples),
                "spu_params_mode": args_namespace.spu_params_mode,
                "spu_attention_policy": args_namespace.spu_attention_policy,
                "spu_activation_override": args_namespace.spu_activation_override,
                "layer_norm_chunk_size": layer_norm_chunk_size,
                "layer_norm_policy": layer_norm_policy,
                "failed_stage": name,
                "stage_records": stage_records,
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "privacy_note": (
                    "Debug-only subgraph smoke. It reveals stage outputs to localize the first SPU runtime/link failure."
                ),
            }
            write_json(output_json, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise

    args_namespace = args
    pixel_ref = secret_identity(pixel_values_np)
    refs["patch"], _ = run_stage("patch_pos", stage_patch, pixel_ref)
    refs["norm1"], _ = run_stage("norm1", stage_norm1, refs["patch"])
    refs["qkv"], _ = run_stage("qkv", stage_qkv, refs["norm1"])
    refs["attn"], _ = run_stage("attention_context", stage_attention, refs["qkv"])
    refs["attn_residual"], _ = run_stage("attention_residual", stage_attn_residual, refs["patch"], refs["attn"])
    refs["norm2"], _ = run_stage("norm2", stage_norm2, refs["attn_residual"])
    refs["mlp_hidden"], _ = run_stage("mlp_hidden", stage_mlp_hidden, refs["norm2"])
    refs["block_output"], _ = run_stage("block_output", stage_block_output, refs["attn_residual"], refs["mlp_hidden"])
    refs["logits"], _ = run_stage("head_logits", stage_head, refs["block_output"])

    summary = {
        "manifest_type": "transshield_e2e_block1_subgraph_smoke_v0",
        "bundle_dir": str(bundle_dir),
        "input_pt": str(input_pt),
        "output_json": str(output_json),
        "config": str(config_path),
        "max_samples": int(args.max_samples),
        "spu_params_mode": args.spu_params_mode,
        "spu_attention_policy": args.spu_attention_policy,
        "spu_activation_override": args.spu_activation_override,
        "layer_norm_chunk_size": layer_norm_chunk_size,
        "layer_norm_policy": layer_norm_policy,
        "failed_stage": None,
        "stage_records": stage_records,
        "privacy_note": (
            "Debug-only subgraph smoke. It reveals stage outputs to localize the first SPU runtime/link failure."
        ),
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
