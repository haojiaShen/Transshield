import math

import torch

from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import normalize_depth_limit


def resolve_runtime_pruning_token_ratios(model_token_ratios, token_ratio_base_override: float):
    """Resolve an optional cumulative pruning-rate override without mutating the model."""
    base_rate = float(token_ratio_base_override)
    if base_rate == 0.0:
        return tuple(float(value) for value in model_token_ratios)
    if not math.isfinite(base_rate) or not 0.0 < base_rate <= 1.0:
        raise ValueError(
            "token_ratio_base_override must be 0 (bundle default) or within (0, 1], "
            f"got {base_rate}"
        )
    return tuple(base_rate**stage for stage in range(1, 4))


def _prepare_student_tokens(model, pixel_values):
    batch_size = int(pixel_values.shape[0])
    x = model.patch_embed(pixel_values)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    return x


def _init_full_keep_policy(x):
    batch_size = int(x.shape[0])
    spatial_token_count = int(x.shape[1] - 1)
    spatial_keep = torch.ones(
        batch_size,
        spatial_token_count,
        1,
        dtype=x.dtype,
        device=x.device,
    )
    cls_keep = torch.ones(batch_size, 1, 1, dtype=x.dtype, device=x.device)
    policy = torch.cat([cls_keep, spatial_keep], dim=1)
    return spatial_keep, policy


def run_static_student_whole_forward_limited(model, pixel_values, static_depth_limit: int):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    x = _prepare_student_tokens(model, pixel_values)
    spatial_keep, policy = _init_full_keep_policy(x)

    for block in list(model.blocks)[:depth]:
        x = block(x, policy=policy)
        if getattr(model, "use_mask_pruning", False):
            x = model._apply_spatial_mask(x, spatial_keep)

    x = model.norm(x)
    token_features = x[:, 1:]
    cls_features = model.pre_logits(x[:, 0])
    logits = model.head(cls_features)
    return {
        "logits": logits,
        "cls_features": cls_features,
        "token_features": token_features,
        "policy": policy,
        "static_depth": depth,
    }


def run_runtime_pruning_student_whole_forward_limited(
    model,
    pixel_values,
    static_depth_limit: int,
    token_ratio_base_override: float = 0.0,
):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    token_ratios = resolve_runtime_pruning_token_ratios(
        model.token_ratio,
        token_ratio_base_override,
    )
    x = _prepare_student_tokens(model, pixel_values)
    prev_decision, policy = _init_full_keep_policy(x)
    pruning_trace = []
    stage_keep_masks = []
    pruning_stage_index = 0
    init_n = int(x.shape[1] - 1)

    for block_index, block in enumerate(list(model.blocks)[:depth]):
        if block_index in getattr(model, "pruning_loc", []):
            if getattr(model, "use_mask_pruning", False):
                x = model._apply_spatial_mask(x, prev_decision)
            spatial_x = x[:, 1:]
            pred_score = model.score_predictor[pruning_stage_index](spatial_x, prev_decision).reshape(
                pixel_values.shape[0], -1, 2
            )
            score = pred_score[:, :, 0]
            keep_count = int(init_n * token_ratios[pruning_stage_index])
            active_before = prev_decision.squeeze(-1).sum(dim=1)
            prev_decision = model._build_eval_keep_decision(score, prev_decision, keep_count)
            active_after = prev_decision.squeeze(-1).sum(dim=1)
            stage_keep_masks.append(prev_decision.clone())
            x = model._apply_spatial_mask(x, prev_decision)
            cls_policy = torch.ones(
                int(pixel_values.shape[0]),
                1,
                1,
                dtype=prev_decision.dtype,
                device=prev_decision.device,
            )
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = block(x, policy=policy)
            x = model._apply_spatial_mask(x, prev_decision)
            pruning_trace.append(
                {
                    "stage_index": int(pruning_stage_index),
                    "pruning_layer": int(block_index),
                    "keep_count_target": int(keep_count),
                    "active_count_before_min": int(active_before.min().item()),
                    "active_count_before_max": int(active_before.max().item()),
                    "active_count_before_mean": float(active_before.float().mean().item()),
                    "active_count_after_min": int(active_after.min().item()),
                    "active_count_after_max": int(active_after.max().item()),
                    "active_count_after_mean": float(active_after.float().mean().item()),
                }
            )
            pruning_stage_index += 1
        else:
            x = block(x, policy=policy)
            x = model._apply_spatial_mask(x, prev_decision)

    x = model.norm(x)
    token_features = x[:, 1:]
    cls_features = model.pre_logits(x[:, 0])
    logits = model.head(cls_features)
    return {
        "logits": logits,
        "cls_features": cls_features,
        "token_features": token_features,
        "policy": policy,
        "prev_decision": prev_decision,
        "pruning_trace": pruning_trace,
        "stage_keep_masks": stage_keep_masks,
        "static_depth": depth,
    }


def run_external_keep_mask_student_whole_forward_limited(
    model,
    pixel_values,
    stage_keep_masks,
    static_depth_limit: int,
):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    x = _prepare_student_tokens(model, pixel_values)
    prev_decision, policy = _init_full_keep_policy(x)
    pruning_trace = []
    pruning_stage_index = 0

    for block_index, block in enumerate(list(model.blocks)[:depth]):
        if block_index in getattr(model, "pruning_loc", []):
            if pruning_stage_index >= len(stage_keep_masks):
                raise ValueError(
                    f"missing keep mask for pruning stage {pruning_stage_index} at block {block_index}"
                )
            if getattr(model, "use_mask_pruning", False):
                x = model._apply_spatial_mask(x, prev_decision)
            active_before = prev_decision.squeeze(-1).sum(dim=1)
            keep_mask = stage_keep_masks[pruning_stage_index].to(device=x.device)
            if keep_mask.ndim == 2:
                keep_mask = keep_mask.unsqueeze(-1)
            prev_decision = keep_mask.to(dtype=x.dtype) * prev_decision
            active_after = prev_decision.squeeze(-1).sum(dim=1)
            x = model._apply_spatial_mask(x, prev_decision)
            cls_policy = torch.ones(
                int(pixel_values.shape[0]),
                1,
                1,
                dtype=prev_decision.dtype,
                device=prev_decision.device,
            )
            policy = torch.cat([cls_policy, prev_decision], dim=1)
            x = block(x, policy=policy)
            x = model._apply_spatial_mask(x, prev_decision)
            pruning_trace.append(
                {
                    "stage_index": int(pruning_stage_index),
                    "pruning_layer": int(block_index),
                    "active_count_before_min": int(active_before.min().item()),
                    "active_count_before_max": int(active_before.max().item()),
                    "active_count_before_mean": float(active_before.float().mean().item()),
                    "active_count_after_min": int(active_after.min().item()),
                    "active_count_after_max": int(active_after.max().item()),
                    "active_count_after_mean": float(active_after.float().mean().item()),
                }
            )
            pruning_stage_index += 1
        else:
            x = block(x, policy=policy)
            x = model._apply_spatial_mask(x, prev_decision)

    x = model.norm(x)
    token_features = x[:, 1:]
    cls_features = model.pre_logits(x[:, 0])
    logits = model.head(cls_features)
    return {
        "logits": logits,
        "cls_features": cls_features,
        "token_features": token_features,
        "policy": policy,
        "prev_decision": prev_decision,
        "pruning_trace": pruning_trace,
        "static_depth": depth,
    }


def run_static_student_whole_forward_probe(model, pixel_values, static_depth_limit: int, probe_block_index: int):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    probe_block_index = int(probe_block_index)
    if probe_block_index < 0 or probe_block_index >= depth:
        raise ValueError(f"probe_block_index={probe_block_index} must be within executed depth={depth}")

    x = _prepare_student_tokens(model, pixel_values)
    spatial_keep, policy = _init_full_keep_policy(x)

    probe_tensors = {}
    for block_index, block in enumerate(list(model.blocks)[:depth]):
        if block_index == probe_block_index:
            probe_tensors["block_input_cls"] = x[:, 0, :].detach().cpu()
            x, debug_payload = block(x, policy=policy, return_debug=True)
            if getattr(model, "use_mask_pruning", False):
                x = model._apply_spatial_mask(x, spatial_keep)
            probe_tensors["norm1_out_cls"] = debug_payload["norm1_out"][:, 0, :].detach().cpu()
            probe_tensors["attn_out_cls"] = debug_payload["attn_out"][:, 0, :].detach().cpu()
            probe_tensors["attn_residual_out_cls"] = debug_payload["attn_residual_out"][:, 0, :].detach().cpu()
            probe_tensors["norm2_out_cls"] = debug_payload["norm2_out"][:, 0, :].detach().cpu()
            probe_tensors["mlp_out_cls"] = debug_payload["mlp_out"][:, 0, :].detach().cpu()
            probe_tensors["block_output_cls"] = x[:, 0, :].detach().cpu()
        else:
            x = block(x, policy=policy)
            if getattr(model, "use_mask_pruning", False):
                x = model._apply_spatial_mask(x, spatial_keep)

    x = model.norm(x)
    final_norm_cls = x[:, 0]
    head_input_cls = model.pre_logits(final_norm_cls)
    logits = model.head(head_input_cls)
    probabilities = torch.softmax(logits, dim=-1)
    probe_tensors["final_norm_cls"] = final_norm_cls.detach().cpu()
    probe_tensors["head_input_cls"] = head_input_cls.detach().cpu()
    probe_tensors["final_logits"] = logits.detach().cpu()
    probe_tensors["final_probabilities"] = probabilities.detach().cpu()
    return {
        "logits": logits.detach().cpu(),
        "probabilities": probabilities.detach().cpu(),
        "probe_tensors": probe_tensors,
        "static_depth": depth,
    }
