from integrations.openbumblebee.e2e_secure_vit.static_vit_params import normalize_depth_limit


def run_static_student_whole_forward_limited(model, pixel_values, static_depth_limit: int):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    batch_size = int(pixel_values.shape[0])
    x = model.patch_embed(pixel_values)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

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


def run_static_student_whole_forward_probe(model, pixel_values, static_depth_limit: int, probe_block_index: int):
    import torch

    depth = normalize_depth_limit(static_depth_limit, full_depth=len(model.blocks))
    probe_block_index = int(probe_block_index)
    if probe_block_index < 0 or probe_block_index >= depth:
        raise ValueError(f"probe_block_index={probe_block_index} must be within executed depth={depth}")

    batch_size = int(pixel_values.shape[0])
    x = model.patch_embed(pixel_values)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

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
