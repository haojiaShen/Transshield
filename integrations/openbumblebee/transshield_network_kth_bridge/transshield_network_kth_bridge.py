import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


FLOAT32_NBYTES = 4
FLOAT16_NBYTES = 2


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    network_definition = manifest.get("network_definition", {})
    if network_definition.get("network_type") != "fixed_odd_even_compare_swap_desc":
        raise ValueError(f"unsupported network type: {network_definition.get('network_type')}")
    return manifest


def load_phase3_selection_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_type") != "phase3_lower_tail_partial_selection":
        raise ValueError(f"unsupported phase3 manifest type: {manifest.get('manifest_type')}")
    return manifest


def load_blockwise_selection_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_type") != "blockwise_exact_kth_selection":
        raise ValueError(f"unsupported blockwise manifest type: {manifest.get('manifest_type')}")
    return manifest


def load_selection_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_type = manifest.get("manifest_type")
    if manifest_type == "phase3_lower_tail_partial_selection":
        for stage in manifest.get("stage_plan", []):
            stage.setdefault("stage_selection_kind", "blockwise_lower_tail_exact")
        return manifest
    if manifest_type == "blockwise_exact_kth_selection":
        return manifest
    raise ValueError(f"unsupported selection manifest type: {manifest_type}")


def load_input_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "stages" not in payload:
        raise ValueError(f"missing stages in input payload: {path}")
    return payload


def build_odd_even_pass_specs(token_count):
    pass_specs = []
    for pass_index in range(token_count):
        start_index = pass_index % 2
        pair_count = len(range(start_index, token_count - 1, 2))
        if pair_count <= 0:
            continue
        pass_specs.append((start_index, pair_count, pair_count * 2))
    return pass_specs


def prepare_network_schedule(token_count, padded_token_count=None):
    token_count = int(token_count)
    if padded_token_count is None:
        padded_token_count = token_count
    padded_token_count = int(padded_token_count)
    pass_specs = build_odd_even_pass_specs(padded_token_count)
    return token_count, padded_token_count, pass_specs


def pad_masked_score_numpy(masked_score, padded_token_count):
    token_count = masked_score.shape[1]
    if padded_token_count == token_count:
        return masked_score
    pad = np.full((masked_score.shape[0], padded_token_count - token_count), -np.inf, dtype=masked_score.dtype)
    return np.concatenate([masked_score, pad], axis=1)


def run_compare_network_numpy(masked_score, token_count, padded_token_count, pass_specs, descending=True):
    if int(masked_score.shape[1]) != token_count:
        raise ValueError(f"masked_score token count mismatch: expected {token_count}, got {masked_score.shape[1]}")

    sortable = pad_masked_score_numpy(masked_score, padded_token_count)
    for start_index, pair_count, sortable_width in pass_specs:
        prefix = sortable[:, :start_index]
        sortable_body = sortable[:, start_index : start_index + sortable_width].reshape(sortable.shape[0], pair_count, 2)
        left_values = sortable_body[:, :, 0]
        right_values = sortable_body[:, :, 1]
        high_values = np.maximum(left_values, right_values)
        low_values = np.minimum(left_values, right_values)
        if descending:
            merged_body = np.stack([high_values, low_values], axis=2).reshape(sortable.shape[0], sortable_width)
        else:
            merged_body = np.stack([low_values, high_values], axis=2).reshape(sortable.shape[0], sortable_width)
        suffix = sortable[:, start_index + sortable_width :]
        sortable = np.concatenate([prefix, merged_body, suffix], axis=1)
    return sortable


def run_stage_cpu(masked_score_np, token_count, padded_token_count, pass_specs, keep_count):
    masked_score = np.asarray(masked_score_np, dtype=np.float32)
    sorted_values = run_compare_network_numpy(masked_score, token_count, padded_token_count, pass_specs)
    kth_threshold = sorted_values[:, keep_count - 1]
    return np.asarray(kth_threshold, dtype=np.float32)


def run_phase3_selection_network_numpy(masked_score_np, selection_plan):
    masked_score = np.asarray(masked_score_np, dtype=np.float32)
    token_count = int(selection_plan["active_token_count"])
    if int(masked_score.shape[1]) != token_count:
        raise ValueError(f"masked_score token count mismatch: expected {token_count}, got {masked_score.shape[1]}")

    candidate_side = selection_plan["candidate_side"]
    candidate_rank = int(selection_plan["candidate_rank"])
    descending = candidate_side != "lower_tail"
    candidate_parts = []
    for block in selection_plan["blocks"]:
        start = int(block["start_index"])
        end = int(block["end_index_exclusive"])
        block_values = masked_score[:, start:end]
        block_token_count, block_padded_token_count, block_pass_specs = prepare_network_schedule(block_values.shape[1])
        sorted_block = run_compare_network_numpy(
            block_values,
            block_token_count,
            block_padded_token_count,
            block_pass_specs,
            descending=descending,
        )
        take_count = min(int(block_values.shape[1]), candidate_rank)
        candidate_parts.append(sorted_block[:, :take_count])

    candidate_values = np.concatenate(candidate_parts, axis=1) if candidate_parts else masked_score[:, :0]
    candidate_token_count, candidate_padded_token_count, candidate_pass_specs = prepare_network_schedule(
        candidate_values.shape[1]
    )
    sorted_candidates = run_compare_network_numpy(
        candidate_values,
        candidate_token_count,
        candidate_padded_token_count,
        candidate_pass_specs,
        descending=descending,
    )
    return np.asarray(sorted_candidates[:, candidate_rank - 1], dtype=np.float32)


def run_stages_cpu(masked_scores_np, token_count, padded_token_count, pass_specs, keep_counts):
    if not masked_scores_np:
        return []
    batch_size = int(masked_scores_np[0].shape[0])
    stacked_scores = np.concatenate([np.asarray(masked_score_np, dtype=np.float32) for masked_score_np in masked_scores_np], axis=0)
    sorted_values = run_compare_network_numpy(stacked_scores, token_count, padded_token_count, pass_specs)
    outputs = []
    start = 0
    for keep_count in keep_counts:
        end = start + batch_size
        outputs.append(np.asarray(sorted_values[start:end, int(keep_count) - 1], dtype=np.float32))
        start = end
    return outputs


def compact_stage_inputs(masked_score_np, active_before_np):
    masked_score_np = np.asarray(masked_score_np, dtype=np.float32)
    active_before_np = np.asarray(active_before_np, dtype=bool)
    if active_before_np.shape != masked_score_np.shape:
        raise ValueError(
            f"active_before shape mismatch: expected {masked_score_np.shape}, got {active_before_np.shape}"
        )
    active_counts = active_before_np.sum(axis=1)
    if active_counts.size == 0:
        return masked_score_np, int(masked_score_np.shape[1]), False
    if not np.all(active_counts == active_counts[0]):
        return masked_score_np, int(masked_score_np.shape[1]), False
    compact_token_count = int(active_counts[0])
    if compact_token_count <= 0 or compact_token_count >= masked_score_np.shape[1]:
        return masked_score_np, int(masked_score_np.shape[1]), False
    compact_masked_score = masked_score_np[active_before_np].reshape(masked_score_np.shape[0], compact_token_count)
    return compact_masked_score, compact_token_count, True


def parse_stage_dtype_overrides(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return {}
    mapping = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid stage dtype override: {item}")
        stage_raw, dtype_name = item.split(":", 1)
        stage_index = int(stage_raw.strip())
        dtype_name = dtype_name.strip()
        if dtype_name not in {"float32", "float16"}:
            raise ValueError(f"unsupported payload dtype: {dtype_name}")
        mapping[stage_index] = dtype_name
    return mapping


def build_boundary_window_indices(masked_score_np, keep_count: int, boundary_window: int):
    masked_score_np = np.asarray(masked_score_np, dtype=np.float32)
    batch_size, token_count = masked_score_np.shape
    if boundary_window <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32)
    sorted_indices = np.argsort(-masked_score_np, axis=1)
    left_rank = max(int(keep_count) - 1 - int(boundary_window), 0)
    right_rank = min(int(keep_count) - 1 + int(boundary_window), token_count - 1)
    return np.asarray(sorted_indices[:, left_rank : right_rank + 1], dtype=np.int32)


def build_mixed_precision_stage_payload(masked_score_np, keep_count: int, stage_dtype: str, boundary_window: int):
    masked_score_np = np.asarray(masked_score_np, dtype=np.float32)
    if stage_dtype == "float32":
        return {
            "transport_mode": "float32_dense",
            "masked_score": masked_score_np,
            "masked_score_base_half": None,
            "boundary_indices": None,
            "boundary_values": None,
            "payload_dtype": "float32",
            "payload_boundary_window": int(boundary_window),
        }

    boundary_indices = build_boundary_window_indices(masked_score_np, keep_count, boundary_window)
    boundary_values = np.take_along_axis(masked_score_np, boundary_indices, axis=1).astype(np.float32, copy=False)
    base_half = masked_score_np.astype(np.float16)
    return {
        "transport_mode": "float16_with_boundary_fp32",
        "masked_score": masked_score_np,
        "masked_score_base_half": base_half,
        "boundary_indices": boundary_indices,
        "boundary_values": boundary_values,
        "payload_dtype": "float16",
        "payload_boundary_window": int(boundary_window),
    }


def estimate_mixed_transport_bytes(stage_payload):
    masked_score = np.asarray(stage_payload["masked_score"], dtype=np.float32)
    float32_total = int(masked_score.size * FLOAT32_NBYTES)
    if stage_payload["transport_mode"] == "float32_dense":
        return {
            "transport_total_bytes": float32_total,
            "transport_ratio_vs_float32": 1.0,
            "base_half_bytes": 0,
            "boundary_index_bytes": 0,
            "boundary_value_bytes": 0,
        }
    base_half = np.asarray(stage_payload["masked_score_base_half"], dtype=np.float16)
    boundary_indices = np.asarray(stage_payload["boundary_indices"], dtype=np.int32)
    boundary_values = np.asarray(stage_payload["boundary_values"], dtype=np.float32)
    transport_total = (
        int(base_half.size * FLOAT16_NBYTES)
        + int(boundary_indices.size * 4)
        + int(boundary_values.size * FLOAT32_NBYTES)
    )
    return {
        "transport_total_bytes": transport_total,
        "transport_ratio_vs_float32": float(transport_total / float32_total) if float32_total > 0 else 1.0,
        "base_half_bytes": int(base_half.size * FLOAT16_NBYTES),
        "boundary_index_bytes": int(boundary_indices.size * 4),
        "boundary_value_bytes": int(boundary_values.size * FLOAT32_NBYTES),
    }


def run_stages_spu(stage_inputs, config_path):
    import jax.nn as jnn
    import jax.numpy as jnp
    import spu.utils.distributed as ppd

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ppd.init(config["nodes"], config["devices"])

    def pad_masked_score_jax(masked_score, padded_token_count):
        token_count = int(masked_score.shape[1])
        if padded_token_count == token_count:
            return masked_score
        pad = jnp.full((masked_score.shape[0], padded_token_count - token_count), -jnp.inf, dtype=masked_score.dtype)
        return jnp.concatenate([masked_score, pad], axis=1)

    def run_compare_network_jax(masked_score, stage_token_count, stage_padded_token_count, stage_pass_specs, descending=True):
        if int(masked_score.shape[1]) != stage_token_count:
            raise ValueError(f"masked_score token count mismatch: expected {stage_token_count}, got {masked_score.shape[1]}")

        sortable = pad_masked_score_jax(masked_score, stage_padded_token_count)
        for start_index, pair_count, sortable_width in stage_pass_specs:
            prefix = sortable[:, :start_index]
            sortable_body = jnp.reshape(
                sortable[:, start_index : start_index + sortable_width],
                (sortable.shape[0], pair_count, 2),
            )
            left_values = sortable_body[:, :, 0]
            right_values = sortable_body[:, :, 1]
            high_values = jnp.maximum(left_values, right_values)
            low_values = jnp.minimum(left_values, right_values)
            if descending:
                merged_pair = jnp.stack([high_values, low_values], axis=2)
            else:
                merged_pair = jnp.stack([low_values, high_values], axis=2)
            merged_body = jnp.reshape(merged_pair, (sortable.shape[0], sortable_width))
            suffix = sortable[:, start_index + sortable_width :]
            sortable = jnp.concatenate([prefix, merged_body, suffix], axis=1)
        return sortable

    def run_phase3_selection_network_jax(masked_score, selection_plan):
        token_count = int(selection_plan["active_token_count"])
        if int(masked_score.shape[1]) != token_count:
            raise ValueError(f"masked_score token count mismatch: expected {token_count}, got {masked_score.shape[1]}")
        candidate_side = selection_plan["candidate_side"]
        candidate_rank = int(selection_plan["candidate_rank"])
        descending = candidate_side != "lower_tail"
        candidate_parts = []
        for block in selection_plan["blocks"]:
            start = int(block["start_index"])
            end = int(block["end_index_exclusive"])
            block_values = masked_score[:, start:end]
            block_token_count, block_padded_token_count, block_pass_specs = prepare_network_schedule(block_values.shape[1])
            sorted_block = run_compare_network_jax(
                block_values,
                block_token_count,
                block_padded_token_count,
                block_pass_specs,
                descending=descending,
            )
            take_count = min(int(block_values.shape[1]), candidate_rank)
            candidate_parts.append(sorted_block[:, :take_count])
        candidate_values = jnp.concatenate(candidate_parts, axis=1) if candidate_parts else masked_score[:, :0]
        candidate_token_count, candidate_padded_token_count, candidate_pass_specs = prepare_network_schedule(
            candidate_values.shape[1]
        )
        sorted_candidates = run_compare_network_jax(
            candidate_values,
            candidate_token_count,
            candidate_padded_token_count,
            candidate_pass_specs,
            descending=descending,
        )
        return sorted_candidates[:, candidate_rank - 1]

    stage_count = len(stage_inputs)
    keep_counts = tuple(int(stage_input["keep_count"]) for stage_input in stage_inputs)
    stage_token_counts = tuple(int(stage_input["token_count"]) for stage_input in stage_inputs)
    stage_padded_token_counts = tuple(int(stage_input["padded_token_count"]) for stage_input in stage_inputs)
    stage_pass_specs = tuple(stage_input["pass_specs"] for stage_input in stage_inputs)
    stage_selection_plans = tuple(stage_input.get("selection_plan") for stage_input in stage_inputs)
    stage_selection_kinds = tuple(stage_input.get("selection_kind", "flat_odd_even") for stage_input in stage_inputs)
    stage_transport_modes = tuple(stage_input.get("transport_mode", "float32_dense") for stage_input in stage_inputs)

    def reconstruct_mixed_precision_stage_input_jax(base_half, boundary_indices, boundary_values):
        dense = jnp.asarray(base_half, dtype=jnp.float32)
        if boundary_indices.shape[1] > 0:
            token_count = dense.shape[1]
            boundary_onehot = jnn.one_hot(boundary_indices, token_count, dtype=jnp.float32)
            boundary_mask = jnp.clip(jnp.sum(boundary_onehot, axis=1), 0.0, 1.0)
            boundary_update = jnp.sum(
                boundary_onehot * jnp.asarray(boundary_values, dtype=jnp.float32)[:, :, None],
                axis=1,
            )
            dense = dense * (1.0 - boundary_mask) + boundary_update
        return dense

    def stage_fn(secret_stage_inputs, public_boundary_indices_bundle):
        outputs = []
        for stage_index, stage_spec in enumerate(
            zip(
                keep_counts,
                stage_token_counts,
                stage_padded_token_counts,
                stage_pass_specs,
                stage_selection_plans,
                stage_selection_kinds,
                stage_transport_modes,
            )
        ):
            (
                keep_count,
                token_count,
                padded_token_count,
                pass_specs,
                selection_plan,
                selection_kind,
                transport_mode,
            ) = stage_spec
            stage_secret_input = secret_stage_inputs[stage_index]
            if transport_mode == "float16_with_boundary_fp32":
                base_half, boundary_values = stage_secret_input
                boundary_indices = public_boundary_indices_bundle[stage_index]
                masked_score = reconstruct_mixed_precision_stage_input_jax(
                    base_half,
                    boundary_indices,
                    boundary_values,
                )
            else:
                masked_score = stage_secret_input
            if selection_kind == "blockwise_lower_tail_exact":
                outputs.append(run_phase3_selection_network_jax(masked_score, selection_plan))
            else:
                sorted_values = run_compare_network_jax(masked_score, token_count, padded_token_count, pass_specs)
                outputs.append(sorted_values[:, keep_count - 1])
        return jnp.stack(outputs, axis=0)

    p1 = ppd.device("P1")
    spu = ppd.device("SPU")

    def identity(x):
        return x

    secret_identity = p1(identity)
    legacy_io_mode = os.environ.get("TRANSSHIELD_SPU_IO_MODE", "").strip().lower() == "per_stage"
    if legacy_io_mode:
        secure_inputs = []

        def legacy_stage_fn(*transport_args):
            outputs = []
            arg_offset = 0
            for keep_count, token_count, padded_token_count, pass_specs, selection_plan, selection_kind, transport_mode in zip(
                keep_counts,
                stage_token_counts,
                stage_padded_token_counts,
                stage_pass_specs,
                stage_selection_plans,
                stage_selection_kinds,
                stage_transport_modes,
            ):
                if transport_mode == "float16_with_boundary_fp32":
                    base_half = transport_args[arg_offset]
                    boundary_indices = transport_args[arg_offset + 1]
                    boundary_values = transport_args[arg_offset + 2]
                    arg_offset += 3
                    masked_score = reconstruct_mixed_precision_stage_input_jax(
                        base_half,
                        boundary_indices,
                        boundary_values,
                    )
                else:
                    masked_score = transport_args[arg_offset]
                    arg_offset += 1
                if selection_kind == "blockwise_lower_tail_exact":
                    outputs.append(run_phase3_selection_network_jax(masked_score, selection_plan))
                else:
                    sorted_values = run_compare_network_jax(masked_score, token_count, padded_token_count, pass_specs)
                    outputs.append(sorted_values[:, keep_count - 1])
            return jnp.stack(outputs, axis=0)

        for stage_input in stage_inputs:
            if stage_input.get("transport_mode") == "float16_with_boundary_fp32":
                secure_inputs.extend(
                    [
                        secret_identity(np.asarray(stage_input["masked_score_base_half"], dtype=np.float16)),
                        np.asarray(stage_input["boundary_indices"], dtype=np.int32),
                        secret_identity(np.asarray(stage_input["boundary_values"], dtype=np.float32)),
                    ]
                )
            else:
                secure_inputs.append(secret_identity(np.asarray(stage_input["masked_score"], dtype=np.float32)))
        kth_thresholds = spu(legacy_stage_fn)(*secure_inputs)
    else:
        secret_stage_bundle = []
        public_boundary_indices_bundle = []
        for stage_input in stage_inputs:
            if stage_input.get("transport_mode") == "float16_with_boundary_fp32":
                secret_stage_bundle.append(
                    (
                        np.asarray(stage_input["masked_score_base_half"], dtype=np.float16),
                        np.asarray(stage_input["boundary_values"], dtype=np.float32),
                    )
                )
                public_boundary_indices_bundle.append(np.asarray(stage_input["boundary_indices"], dtype=np.int32))
            else:
                secret_stage_bundle.append(np.asarray(stage_input["masked_score"], dtype=np.float32))
                public_boundary_indices_bundle.append(
                    np.empty((int(stage_input["masked_score"].shape[0]), 0), dtype=np.int32)
                )
        kth_thresholds = spu(stage_fn)(
            secret_identity(tuple(secret_stage_bundle)),
            tuple(public_boundary_indices_bundle),
        )

    kth_thresholds = np.asarray(ppd.get(kth_thresholds), dtype=np.float32)
    if kth_thresholds.ndim == 1:
        kth_thresholds = kth_thresholds[None, :]
    return [np.asarray(kth_thresholds[stage_index], dtype=np.float32) for stage_index in range(stage_count)]


def summarize_float_array(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else 0.0,
    }


def estimate_float32_bytes(shape):
    element_count = 1
    for dim in shape:
        element_count *= int(dim)
    return int(element_count * FLOAT32_NBYTES)


def summarize_stage_payload_estimate(full_masked_score_np, compact_masked_score_np):
    dense_shape = list(np.asarray(full_masked_score_np, dtype=np.float32).shape)
    compact_shape = list(np.asarray(compact_masked_score_np, dtype=np.float32).shape)
    dense_bytes = estimate_float32_bytes(dense_shape)
    compact_bytes = estimate_float32_bytes(compact_shape)
    saved_bytes = max(dense_bytes - compact_bytes, 0)
    return {
        "dense_masked_score_shape": dense_shape,
        "compact_masked_score_shape": compact_shape,
        "dense_masked_score_float32_bytes": dense_bytes,
        "compact_masked_score_float32_bytes": compact_bytes,
        "saved_float32_bytes": saved_bytes,
        "compact_ratio": float(compact_bytes / dense_bytes) if dense_bytes > 0 else 1.0,
    }


def aggregate_payload_estimates(stage_payload_estimates, payload_default_dtype: str, payload_boundary_window: int):
    if not stage_payload_estimates:
        return {
            "stage_count": 0,
            "dense_masked_score_float32_bytes": 0,
            "compact_masked_score_float32_bytes": 0,
            "saved_float32_bytes": 0,
            "compact_ratio": 1.0,
        }
    dense_total = int(sum(int(item["dense_masked_score_float32_bytes"]) for item in stage_payload_estimates))
    compact_total = int(sum(int(item["compact_masked_score_float32_bytes"]) for item in stage_payload_estimates))
    saved_total = int(sum(int(item["saved_float32_bytes"]) for item in stage_payload_estimates))
    mixed_transport_total = int(sum(int(item.get("mixed_transport_total_bytes") or item["compact_masked_score_float32_bytes"]) for item in stage_payload_estimates))
    mixed_base_half_total = int(sum(int(item.get("mixed_base_half_bytes") or 0) for item in stage_payload_estimates))
    mixed_boundary_index_total = int(sum(int(item.get("mixed_boundary_index_bytes") or 0) for item in stage_payload_estimates))
    mixed_boundary_value_total = int(sum(int(item.get("mixed_boundary_value_bytes") or 0) for item in stage_payload_estimates))
    return {
        "stage_count": len(stage_payload_estimates),
        "dense_masked_score_float32_bytes": dense_total,
        "compact_masked_score_float32_bytes": compact_total,
        "saved_float32_bytes": saved_total,
        "compact_ratio": float(compact_total / dense_total) if dense_total > 0 else 1.0,
        "mixed_transport_total_bytes": mixed_transport_total,
        "mixed_transport_ratio_vs_compact_float32": (
            float(mixed_transport_total / compact_total) if compact_total > 0 else 1.0
        ),
        "mixed_base_half_total_bytes": mixed_base_half_total,
        "mixed_boundary_index_total_bytes": mixed_boundary_index_total,
        "mixed_boundary_value_total_bytes": mixed_boundary_value_total,
        "payload_default_dtype": payload_default_dtype,
        "payload_boundary_window": int(payload_boundary_window),
        "selection_mode_changes_secure_input_shape": False,
        "selection_mode_scope": (
            "Selection mode only changes the in-SPU kth selection schedule; "
            "active-token compaction happens before selection-mode dispatch, "
            "so flat_odd_even and blockwise_exact_kth currently send the same compacted masked_score shapes."
        ),
        "payload_transport_scope": (
            "Mixed payload transport changes what is serialized to P1 before exact float32 reconstruction; "
            "it can reduce RPC bytes even when selection mode is unchanged."
        ),
    }


def build_candidate_payload(
    manifest,
    input_payload,
    manifest_path: Path,
    input_path: Path,
    runtime,
    config_path,
    selection_mode,
    selection_manifest,
    selection_manifest_path,
    payload_default_dtype,
    payload_stage_dtypes,
    payload_boundary_window,
):
    network_definition = manifest["network_definition"]
    manifest_token_count = int(network_definition["input_token_count"])
    manifest_padded_token_count = int(network_definition.get("padded_token_count", manifest_token_count))
    manifest_stages = {int(stage["stage_index"]): stage for stage in manifest["stage_plan"]}
    input_stages = {int(stage["stage_index"]): stage for stage in input_payload["stages"]}
    selection_stages = {}
    if selection_mode != "flat_odd_even":
        if not selection_manifest:
            raise ValueError("--phase3-selection-manifest is required when selection_mode is not flat_odd_even")
        selection_stages = {int(stage["stage_index"]): stage for stage in selection_manifest["stage_plan"]}

    candidate_stages = []
    summary_stages = []
    started_at = time.time()
    stage_indices = sorted(manifest_stages)
    stage_inputs = []
    stage_payload_estimates = []
    spu_io_mode = "per_stage_legacy" if os.environ.get("TRANSSHIELD_SPU_IO_MODE", "").strip().lower() == "per_stage" else "batched_pyu_bundle"

    for stage_index in stage_indices:
        manifest_stage = manifest_stages[stage_index]
        input_stage = input_stages.get(stage_index)
        if input_stage is None:
            raise ValueError(f"missing input stage {stage_index}")
        keep_count = int(manifest_stage["keep_count"])
        full_masked_score_np = np.asarray(input_stage["masked_score"], dtype=np.float32)
        active_before_np = np.asarray(input_stage["active_before"], dtype=bool)
        compact_masked_score_np, compact_token_count, used_compaction = compact_stage_inputs(full_masked_score_np, active_before_np)
        stage_payload_estimate = summarize_stage_payload_estimate(full_masked_score_np, compact_masked_score_np)
        stage_payload_dtype = payload_stage_dtypes.get(stage_index, payload_default_dtype)
        mixed_stage_payload = build_mixed_precision_stage_payload(
            compact_masked_score_np,
            keep_count,
            stage_payload_dtype,
            payload_boundary_window,
        )
        mixed_transport = estimate_mixed_transport_bytes(mixed_stage_payload)
        stage_payload_estimate.update(
            {
                "payload_dtype": stage_payload_dtype,
                "payload_boundary_window": int(payload_boundary_window),
                "mixed_transport_mode": mixed_stage_payload["transport_mode"],
                "mixed_transport_total_bytes": mixed_transport["transport_total_bytes"],
                "mixed_transport_ratio_vs_float32": mixed_transport["transport_ratio_vs_float32"],
                "mixed_transport_ratio_vs_compact_float32": (
                    float(mixed_transport["transport_total_bytes"] / stage_payload_estimate["compact_masked_score_float32_bytes"])
                    if stage_payload_estimate["compact_masked_score_float32_bytes"] > 0
                    else 1.0
                ),
                "mixed_base_half_bytes": mixed_transport["base_half_bytes"],
                "mixed_boundary_index_bytes": mixed_transport["boundary_index_bytes"],
                "mixed_boundary_value_bytes": mixed_transport["boundary_value_bytes"],
            }
        )
        stage_token_count = int(manifest_stage.get("effective_input_token_count", compact_token_count))
        stage_padded_token_count = int(manifest_stage.get("effective_padded_token_count", stage_token_count))
        if stage_token_count != compact_token_count:
            stage_token_count = compact_token_count
            stage_padded_token_count = compact_token_count
        _token_count, _padded_token_count, pass_specs = prepare_network_schedule(stage_token_count, stage_padded_token_count)
        stage_selection_kind = "flat_odd_even"
        selection_plan = None
        if selection_mode != "flat_odd_even":
            selection_plan = selection_stages.get(stage_index)
            if selection_plan is None:
                raise ValueError(f"missing selection plan for stage {stage_index}")
            stage_selection_kind = selection_plan.get("stage_selection_kind", "blockwise_lower_tail_exact")
            if stage_selection_kind not in {"flat_odd_even", "blockwise_lower_tail_exact"}:
                raise ValueError(f"unsupported stage selection kind: {stage_selection_kind}")
            if stage_selection_kind == "blockwise_lower_tail_exact":
                if int(selection_plan["active_token_count"]) != int(stage_token_count):
                    raise ValueError(
                        f"selection active token count mismatch for stage {stage_index}: "
                        f"expected {stage_token_count}, got {selection_plan['active_token_count']}"
                    )
        stage_inputs.append(
            {
                "stage_index": stage_index,
                "pruning_layer": int(manifest_stage["pruning_layer"]),
                "keep_count": keep_count,
                "masked_score": compact_masked_score_np,
                "masked_score_base_half": mixed_stage_payload.get("masked_score_base_half"),
                "boundary_indices": mixed_stage_payload.get("boundary_indices"),
                "boundary_values": mixed_stage_payload.get("boundary_values"),
                "transport_mode": mixed_stage_payload["transport_mode"],
                "payload_dtype": stage_payload_dtype,
                "payload_boundary_window": int(payload_boundary_window),
                "token_count": stage_token_count,
                "padded_token_count": stage_padded_token_count,
                "pass_specs": pass_specs,
                "used_compaction": bool(used_compaction),
                "original_token_count": int(full_masked_score_np.shape[1]),
                "selection_plan": selection_plan,
                "selection_kind": stage_selection_kind,
                "payload_estimate": stage_payload_estimate,
            }
        )
        stage_payload_estimates.append(stage_payload_estimate)

    kth_thresholds_by_stage = {}
    if runtime == "cpu":
        for stage_input in stage_inputs:
            stage_index = int(stage_input["stage_index"])
            if stage_input["selection_kind"] == "blockwise_lower_tail_exact":
                kth_thresholds_by_stage[stage_index] = run_phase3_selection_network_numpy(
                    stage_input["masked_score"],
                    stage_input["selection_plan"],
                )
            else:
                kth_thresholds_by_stage[stage_index] = run_stage_cpu(
                    stage_input["masked_score"],
                    stage_input["token_count"],
                    stage_input["padded_token_count"],
                    stage_input["pass_specs"],
                    stage_input["keep_count"],
                )
    else:
        if not config_path:
            raise ValueError("--config is required when --runtime spu")
        spu_outputs = run_stages_spu(stage_inputs, config_path)
        kth_thresholds_by_stage = {stage_index: output for stage_index, output in zip(stage_indices, spu_outputs)}

    for stage_index in stage_indices:
        stage_input = next(item for item in stage_inputs if int(item["stage_index"]) == int(stage_index))
        pruning_layer = int(stage_input["pruning_layer"])
        keep_count = int(stage_input["keep_count"])

        kth_threshold_np = kth_thresholds_by_stage[stage_index]

        candidate_stages.append(
            {
                "stage_index": stage_index,
                "pruning_layer": pruning_layer,
                "keep_count": keep_count,
                "kth_threshold": torch.from_numpy(kth_threshold_np.copy()),
            }
        )
        summary_stages.append(
            {
                "stage_index": stage_index,
                "pruning_layer": pruning_layer,
                "keep_count": keep_count,
                "kth_threshold": summarize_float_array(kth_threshold_np),
                "effective_input_token_count": int(stage_input["token_count"]),
                "effective_padded_token_count": int(stage_input["padded_token_count"]),
                "used_active_token_compaction": bool(stage_input["used_compaction"]),
                "selection_mode": selection_mode,
                "stage_selection_kind": stage_input["selection_kind"],
                "payload_estimate": stage_input["payload_estimate"],
                "payload_dtype": stage_input["payload_dtype"],
                "payload_boundary_window": stage_input["payload_boundary_window"],
                "blockwise_selection": {
                    "candidate_side": stage_input["selection_plan"]["candidate_side"],
                    "candidate_rank": int(stage_input["selection_plan"]["candidate_rank"]),
                    "candidate_count": int(stage_input["selection_plan"]["candidate_count"]),
                    "block_size": int(stage_input["selection_plan"]["block_size"]),
                    "block_count": int(stage_input["selection_plan"]["block_count"]),
                }
                if stage_input["selection_kind"] == "blockwise_lower_tail_exact"
                else None,
            }
        )

    elapsed = time.time() - started_at
    candidate_payload = {
        "bundle_dir": input_payload.get("bundle_dir"),
        "data_path": input_payload.get("data_path"),
        "sample_paths": input_payload.get("sample_paths"),
        "stages": candidate_stages,
        "candidate_metadata": {
            "source_manifest_json": str(manifest_path.resolve()),
            "source_input_pt": str(input_path.resolve()),
            "runtime": runtime,
            "config_path": str(Path(config_path).resolve()) if config_path else "",
            "selection_mode": selection_mode,
            "spu_io_mode": spu_io_mode if runtime == "spu" else "cpu_local",
            "selection_manifest_json": str(selection_manifest_path.resolve())
            if selection_manifest_path
            else "",
            "source_compare_network_manifest": True,
            "source_reference_topk": False,
            "model_semantics_changed": False,
            "format_purpose": "open_bumblebee_transshield_network_kth_candidate",
        },
    }
    summary = {
        "bundle_dir": candidate_payload.get("bundle_dir"),
        "data_path": candidate_payload.get("data_path"),
        "input_pt": str(input_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "runtime": runtime,
        "selection_mode": selection_mode,
        "spu_io_mode": spu_io_mode if runtime == "spu" else "cpu_local",
        "elapsed_sec": elapsed,
        "stage_count": len(candidate_stages),
        "payload_estimate": aggregate_payload_estimates(
            stage_payload_estimates,
            payload_default_dtype,
            payload_boundary_window,
        ),
        "stages": summary_stages,
        "notes": [
            "This bridge consumes Transshield masked_score inputs and the compare-network manifest.",
            "The output is checker-compatible with tools/transshield_secure_network_kth.py check.",
            (
                f"Compact odd-even schedule is used with manifest input_token_count={manifest_token_count} "
                f"and manifest padded_token_count={manifest_padded_token_count}."
            ),
            "Later pruning stages compact masked_score to active tokens only before compare-network execution.",
            "Stage-selective blockwise mode is opt-in; default flat odd-even bridge semantics are unchanged.",
            "If selection modes show lower runtime but identical RPC bytes, first check payload_estimate: in the current implementation selection mode does not change compacted secure input tensor shapes.",
            "Mixed payload transport can reduce serialized input bytes before exact float32 reconstruction on P1.",
            "SPU runtime now defaults to batched PYU bundle transfer for network-kth inputs/outputs; set TRANSSHIELD_SPU_IO_MODE=per_stage to force the legacy per-stage object path.",
        ],
    }
    return candidate_payload, summary


def main():
    parser = argparse.ArgumentParser(description="Run the Transshield compare-network kth bridge inside OpenBumbleBee.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--input-pt", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--runtime", choices=["cpu", "spu"], default="cpu")
    parser.add_argument("--selection-mode", choices=["flat_odd_even", "phase3_lower_tail", "blockwise_exact_kth"], default="flat_odd_even")
    parser.add_argument("--phase3-selection-manifest", default="")
    parser.add_argument("--payload-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--payload-stage-dtypes", default="")
    parser.add_argument("--payload-boundary-window", type=int, default=0)
    parser.add_argument("-c", "--config", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_json).resolve()
    input_path = Path(args.input_pt).resolve()
    manifest = load_manifest(manifest_path)
    selection_manifest_path = (
        Path(args.phase3_selection_manifest).resolve() if args.phase3_selection_manifest else None
    )
    selection_manifest = (
        load_selection_manifest(selection_manifest_path) if selection_manifest_path else None
    )
    payload_stage_dtypes = parse_stage_dtype_overrides(args.payload_stage_dtypes)
    input_payload = load_input_payload(input_path)
    candidate_payload, summary = build_candidate_payload(
        manifest,
        input_payload,
        manifest_path,
        input_path,
        args.runtime,
        args.config,
        args.selection_mode,
        selection_manifest,
        selection_manifest_path,
        args.payload_dtype,
        payload_stage_dtypes,
        args.payload_boundary_window,
    )

    output_pt = Path(args.output_pt).resolve()
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate_payload, output_pt)

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
