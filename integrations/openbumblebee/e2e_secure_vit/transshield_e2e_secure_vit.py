# =============================================================================
# Transshield E2E Secure ViT — 安全推理端到端执行器
# =============================================================================
# 角色映射：
#   "client" / "client-side"  → 数据使用方（如医院），提交影像数据，获取诊断结果
#   "server" / "server-side"  → 模型提供方的推理服务（内部含 P0/P1 两台 MPC 服务器）
#   "client_pixel_package"    → 数据使用方提交的预处理后数据包
#
# 变量命名保持历史兼容，新增注释统一使用"数据使用方" / "模型提供方"术语。
# =============================================================================

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.openbumblebee.e2e_secure_vit.common import (  # noqa: E402
    compare_debug_records,
    debug_tensor_record,
    load_json,
    numpy_from_torch_tensor,
    path_for_output,
    record_to_float_tensor,
    require_existing_file,
    tensor_stats,
    write_json,
)
from integrations.openbumblebee.e2e_secure_vit.input_shares import (  # noqa: E402
    load_debug_input_share_pair as load_debug_input_share_pair_impl,
    load_debug_input_share_pair_from_party_manifests as load_debug_input_share_pair_from_party_manifests_impl,
    load_debug_party_share_metadata as load_debug_party_share_metadata_impl,
)
from integrations.openbumblebee.e2e_secure_vit.cpu_static_vit import (  # noqa: E402
    run_external_keep_mask_student_whole_forward_limited,
    run_runtime_pruning_student_whole_forward_limited,
    run_static_student_whole_forward_limited,
    run_static_student_whole_forward_probe,
)
from integrations.openbumblebee.e2e_secure_vit.spu_static_vit import (  # noqa: E402
    compare_array_payload,
    run_share_recomposition_audit_spu,
    run_static_vit_forward_spu,
    static_patch_embed_numpy,
)
from integrations.openbumblebee.e2e_secure_vit.static_vit_params import (  # noqa: E402
    load_static_vit_spu_params_with_predictor,
)
from integrations.openbumblebee.e2e_secure_vit.debug_probe import (  # noqa: E402
    run_runtime_primitive_smoke,
    run_block1_subgraph_smoke,
)
from integrations.openbumblebee.e2e_secure_vit.calibrated_ln import (  # noqa: E402
    compute_public_layer_norm_calibration,
    load_public_layer_norm_calibration,
)
from integrations.openbumblebee.e2e_secure_vit.static_vit_params import (  # noqa: E402
    STATIC_FORWARD_SCOPE,
    load_static_vit_spu_params as load_static_vit_spu_params_impl,
    normalize_depth_limit as normalize_depth_limit_impl,
)


DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts" / "frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430"
DEFAULT_SPU_CONFIG = REPO_ROOT / "configs" / "openbumblebee" / "2pc.json"
DEFAULT_E2E_SPU_TEMPLATE = REPO_ROOT / "configs" / "openbumblebee" / "2pc_e2e.template.json"
DEFAULT_OUTPUT_PT_NAME = "e2e_static_whole_forward_candidate_from_server.pt"
DEFAULT_OUTPUT_JSON_NAME = "e2e_static_whole_forward_candidate_from_server.json"
DEFAULT_COMPARE_JSON_NAME = "e2e_static_whole_forward_compare.json"
DEFAULT_INPUT_PT_NAME = "client_pixel_values.pt"
DEFAULT_REFERENCE_PT_NAME = "static_whole_forward_reference.pt"
DEFAULT_CONTRACT_JSON_NAME = "e2e_secure_contract.json"
CANDIDATE_MANIFEST_TYPE = "transshield_e2e_static_whole_forward_candidate_pt_v0"
SUMMARY_MANIFEST_TYPE = "transshield_e2e_static_whole_forward_candidate_summary_v0"
SHARE_AUDIT_MANIFEST_TYPE = "transshield_e2e_static_whole_forward_share_audit_v0"
BLOCK_PROBE_MANIFEST_TYPE = "transshield_e2e_static_whole_forward_block_probe_v0"
BLOCK_PROBE_COMPARE_MANIFEST_TYPE = "transshield_e2e_static_whole_forward_block_probe_compare_v0"
BLOCK_PROBE_TENSOR_NAMES = [
    "block_input_cls",
    "norm1_out_cls",
    "attn_out_cls",
    "attn_residual_out_cls",
    "norm2_out_cls",
    "mlp_out_cls",
    "block_output_cls",
    "final_norm_cls",
    "head_input_cls",
    "final_logits",
    "final_probabilities",
]
BLOCK_PROBE_FINAL_OUTPUT_TENSOR_NAMES = {
    "final_logits",
    "final_probabilities",
}
BLOCK_PROBE_INTERMEDIATE_TENSOR_NAMES = [
    name for name in BLOCK_PROBE_TENSOR_NAMES if name not in BLOCK_PROBE_FINAL_OUTPUT_TENSOR_NAMES
]

def candidate_payload_paths(output_dir: Path):
    return {
        "candidate_pt": output_dir / DEFAULT_OUTPUT_PT_NAME,
        "candidate_json": output_dir / DEFAULT_OUTPUT_JSON_NAME,
        "compare_json": output_dir / DEFAULT_COMPARE_JSON_NAME,
    }


def build_prepare_manifest(
    output_dir: Path,
    bundle_dir: Path,
    input_pt: Path,
    reference_pt: Path,
    contract_json: Path,
    spu_config: Path,
    spu_template: Path,
):
    paths = candidate_payload_paths(output_dir)
    commands = {
        "cpu_run": (
            f"python {Path(__file__).resolve()} run "
            f"--runtime cpu "
            f"--bundle-dir {bundle_dir} "
            f"--input-pt {input_pt} "
            f"--output-pt {paths['candidate_pt']} "
            f"--output-json {paths['candidate_json']}"
        ),
        "spu_run_template": (
            f"python {Path(__file__).resolve()} run "
            f"--runtime spu "
            f"--bundle-dir {bundle_dir} "
            f"--input-pt {input_pt} "
            f"--output-pt {paths['candidate_pt']} "
            f"--output-json {paths['candidate_json']} "
            f"--config {spu_config}"
        ),
        "spu_smoke_run_template": (
            f"python {Path(__file__).resolve()} run "
            f"--runtime spu "
            f"--bundle-dir {bundle_dir} "
            f"--input-pt {input_pt} "
            f"--output-pt {paths['candidate_pt']} "
            f"--output-json {paths['candidate_json']} "
            f"--config {spu_config} "
            f"--spu-params-mode public "
            f"--max-samples 1 "
            f"--spu-batch-size 1"
        ),
        "verify": (
            f"python {Path(__file__).resolve()} verify "
            f"--reference-pt {reference_pt} "
            f"--candidate-pt {paths['candidate_pt']} "
            f"--output-json {paths['compare_json']}"
        ),
        "verify_prefix_candidate": (
            f"python {Path(__file__).resolve()} verify "
            f"--reference-pt {reference_pt} "
            f"--candidate-pt {paths['candidate_pt']} "
            f"--output-json {paths['compare_json']} "
            f"--allow-prefix-candidate"
        ),
    }
    return {
        "manifest_type": "transshield_e2e_secure_vit_prepare_pack_v0",
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "bundle_dir": str(bundle_dir),
        "input_pt": str(input_pt),
        "reference_pt": str(reference_pt),
        "contract_json": str(contract_json) if contract_json.exists() else None,
        "spu_config": str(spu_config),
        "spu_template": str(spu_template),
        "expected_outputs": {key: str(value) for key, value in paths.items()},
        "commands": commands,
        "notes": [
            "CPU backend is the current local reference backend for the whole-forward contract.",
            "SPU whole-forward execution now has an experimental static JAX backend; run the smoke command first on the model provider server.",
            "Use the verify command to compare a future secure candidate against the static whole-forward plaintext reference.",
        ],
    }


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def command_prepare(args):
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    spu_config = Path(args.spu_config).expanduser().resolve()
    spu_template = Path(args.spu_template).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_pt = Path(args.input_pt).expanduser().resolve()
    reference_pt = Path(args.reference_pt).expanduser().resolve()
    contract_json = (
        Path(args.contract_json).expanduser().resolve()
        if args.contract_json
        else input_pt.parent / DEFAULT_CONTRACT_JSON_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_pt.exists():
        raise FileNotFoundError(f"missing input_pt: {input_pt}")
    if not reference_pt.exists():
        raise FileNotFoundError(f"missing reference_pt: {reference_pt}")

    copied = {}
    copied["input_pt"] = copy_if_exists(input_pt, output_dir / DEFAULT_INPUT_PT_NAME)
    input_json = input_pt.with_suffix(".json")
    copied["input_json"] = copy_if_exists(input_json, output_dir / input_json.name)
    copied["reference_pt"] = copy_if_exists(reference_pt, output_dir / DEFAULT_REFERENCE_PT_NAME)
    reference_json = reference_pt.with_suffix(".json")
    copied["reference_json"] = copy_if_exists(reference_json, output_dir / reference_json.name)
    copied["contract_json"] = copy_if_exists(contract_json, output_dir / DEFAULT_CONTRACT_JSON_NAME)

    manifest = build_prepare_manifest(
        output_dir=output_dir,
        bundle_dir=bundle_dir,
        input_pt=output_dir / DEFAULT_INPUT_PT_NAME,
        reference_pt=output_dir / DEFAULT_REFERENCE_PT_NAME,
        contract_json=output_dir / DEFAULT_CONTRACT_JSON_NAME,
        spu_config=spu_config,
        spu_template=spu_template,
    )
    manifest["copied"] = copied
    write_json(output_dir / "commands.json", manifest)
    (output_dir / "README.md").write_text(
        "# Transshield e2e secure whole-forward pack\n\n"
        "This pack freezes the current whole-forward contract for the new e2e track.\n\n"
        "Included artifacts:\n"
        "- 数据使用方数据包（client pixel package，历史变量名兼容）\n"
        "- static whole-forward plaintext reference\n"
        "- optional e2e contract JSON\n"
        "- command templates for CPU run / experimental SPU smoke run / verify\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_runtime_helpers():
    from tools.transshield_e2e_secure_infer import (
        compare_prediction_match,
        load_client_pixel_package,
        load_debug_share_party_manifest,
        load_debug_share_manifest,
        load_debug_share_public_manifest,
        load_tensor_payload,
        resolve_bundle_dir,
        run_static_student_whole_forward,
    )
    from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold

    return {
        "compare_prediction_match": compare_prediction_match,
        "load_client_pixel_package": load_client_pixel_package,
        "load_debug_share_party_manifest": load_debug_share_party_manifest,
        "load_debug_share_manifest": load_debug_share_manifest,
        "load_debug_share_public_manifest": load_debug_share_public_manifest,
        "load_tensor_payload": load_tensor_payload,
        "resolve_bundle_dir": resolve_bundle_dir,
        "run_static_student_whole_forward": run_static_student_whole_forward,
        "load_frozen_bundle": load_frozen_bundle,
        "resolve_threshold": resolve_threshold,
    }


def load_debug_input_share_pair(share_manifest_json: Path):
    return load_debug_input_share_pair_impl(share_manifest_json, load_runtime_helpers())


def load_debug_input_share_pair_from_party_manifests(
    public_manifest_json: Path,
    p1_share_manifest_json: Path,
    p2_share_manifest_json: Path,
):
    return load_debug_input_share_pair_from_party_manifests_impl(
        public_manifest_json,
        p1_share_manifest_json,
        p2_share_manifest_json,
        load_runtime_helpers(),
    )


def load_debug_party_share_metadata(
    public_manifest_json: Path,
    p1_share_manifest_json: Path,
    p2_share_manifest_json: Path,
):
    return load_debug_party_share_metadata_impl(
        public_manifest_json,
        p1_share_manifest_json,
        p2_share_manifest_json,
        load_runtime_helpers(),
    )


def normalize_depth_limit(raw_depth_limit: int, full_depth: int = 12) -> int:
    return normalize_depth_limit_impl(raw_depth_limit, full_depth)


def load_static_vit_spu_params(
    bundle_dir: Path,
    static_depth_limit: int = -1,
    attention_policy: str = "smoothed",
    activation_override: str = "bundle",
    token_ratio_base_override: float = 0.0,
):
    return load_static_vit_spu_params_impl(
        bundle_dir,
        static_depth_limit=static_depth_limit,
        attention_policy=attention_policy,
        activation_override=activation_override,
        token_ratio_base_override=token_ratio_base_override,
    )


def apply_output_calibration(logits_cpu, calibration_json: Path):
    import torch

    payload = load_json(calibration_json)
    weights = torch.tensor(payload["weights"], dtype=logits_cpu.dtype)
    bias = torch.tensor(float(payload.get("bias", 0.0)), dtype=logits_cpu.dtype)
    if weights.numel() != int(logits_cpu.shape[-1]):
        raise ValueError(
            f"output calibration weight size mismatch: weights={weights.numel()} logits_dim={logits_cpu.shape[-1]}"
        )
    score = logits_cpu.matmul(weights) + bias
    calibrated_logits = torch.stack([-0.5 * score, 0.5 * score], dim=-1)
    return calibrated_logits, {
        "calibration_json": str(calibration_json),
        "manifest_type": payload.get("manifest_type", "transshield_e2e_output_calibration_v0"),
        "weights": [float(value) for value in weights.tolist()],
        "bias": float(bias.item()),
        "threshold": float(payload.get("threshold", 0.5)),
        "score_rule": payload.get("score_rule", "class1_score = logits @ weights + bias"),
        "note": payload.get("note"),
    }


def command_run(args):
    import torch

    helpers = load_runtime_helpers()
    runtime = args.runtime

    bundle_dir = helpers["resolve_bundle_dir"](args.bundle_dir)
    input_pt = Path(args.input_pt).expanduser().resolve() if args.input_pt else None
    input_share_manifest_json = (
        Path(args.input_share_manifest_json).expanduser().resolve()
        if args.input_share_manifest_json
        else None
    )
    input_share_public_manifest_json = (
        Path(args.input_share_public_manifest_json).expanduser().resolve()
        if args.input_share_public_manifest_json
        else None
    )
    input_p1_share_manifest_json = (
        Path(args.input_p1_share_manifest_json).expanduser().resolve()
        if args.input_p1_share_manifest_json
        else None
    )
    input_p2_share_manifest_json = (
        Path(args.input_p2_share_manifest_json).expanduser().resolve()
        if args.input_p2_share_manifest_json
        else None
    )
    output_pt = Path(args.output_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    redact_private_input_paths = bool(args.redact_private_input_paths)
    layer_norm_calibration_json = (
        Path(args.spu_layer_norm_calibration_json).expanduser().resolve()
        if getattr(args, "spu_layer_norm_calibration_json", "")
        else None
    )
    runtime_pruning_keep_mask_pt = (
        Path(args.runtime_pruning_keep_mask_pt).expanduser().resolve()
        if getattr(args, "runtime_pruning_keep_mask_pt", "")
        else None
    )

    threshold = helpers["resolve_threshold"](bundle_dir, None)
    share_pair_cpu = None
    share_manifest = None
    party_share_manifests = None
    party_local_share_manifest_paths = None
    party_local_share_sample_count = None
    using_legacy_share_manifest = input_share_manifest_json is not None
    using_party_share_manifests = any(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    )
    if using_legacy_share_manifest and using_party_share_manifests:
        raise ValueError(
            "--input-share-manifest-json cannot be combined with "
            "--input-share-public-manifest-json/--input-p*-share-manifest-json"
        )
    if using_party_share_manifests and not all(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    ):
        raise ValueError(
            "--input-share-public-manifest-json, --input-p1-share-manifest-json, and "
            "--input-p2-share-manifest-json must be set together"
        )
    if args.party_local_share_load and not using_party_share_manifests:
        raise ValueError("--party-local-share-load requires split public/P1/P2 share manifests")
    if input_share_manifest_json is not None:
        if runtime != "spu":
            raise ValueError("--input-share-manifest-json is currently supported only with --runtime spu")
        share_payload = load_debug_input_share_pair(input_share_manifest_json)
        share_pair_cpu = share_payload["share_tensors"]
        share_manifest = share_payload["manifest"]
        pixel_values_cpu = share_pair_cpu[0]
        client_payload = {
            "sample_ids": share_payload.get("sample_ids"),
            "targets": share_payload.get("targets"),
        }
        input_source = str(input_share_manifest_json)
    elif using_party_share_manifests:
        if runtime != "spu":
            raise ValueError("--input-share-public-manifest-json is currently supported only with --runtime spu")
        if args.party_local_share_load:
            share_payload = load_debug_party_share_metadata(
                input_share_public_manifest_json,
                input_p1_share_manifest_json,
                input_p2_share_manifest_json,
            )
            share_manifest = share_payload["manifest"]
            party_share_manifests = share_payload["party_manifests"]
            party_local_share_manifest_paths = share_payload["party_manifest_paths"]
            party_local_share_sample_count = int(share_manifest["sample_count"])
            pixel_values_cpu = None
            client_payload = {
                "sample_ids": share_payload.get("sample_ids"),
                "targets": None,
            }
        else:
            share_payload = load_debug_input_share_pair_from_party_manifests(
                input_share_public_manifest_json,
                input_p1_share_manifest_json,
                input_p2_share_manifest_json,
            )
            share_pair_cpu = share_payload["share_tensors"]
            share_manifest = share_payload["manifest"]
            party_share_manifests = share_payload["party_manifests"]
            pixel_values_cpu = share_pair_cpu[0]
            client_payload = {
                "sample_ids": share_payload.get("sample_ids"),
                "targets": share_payload.get("targets"),
            }
        input_source = str(input_share_public_manifest_json)
    else:
        if input_pt is None:
            raise ValueError(
                "either --input-pt, --input-share-manifest-json, or split public/P1/P2 share manifests is required"
            )
        require_existing_file(input_pt, "input pixel package")
        client_payload = helpers["load_client_pixel_package"](input_pt)
        pixel_values_cpu = client_payload["pixel_values"].detach().cpu()
        input_source = str(input_pt)
    if args.max_samples > 0 and pixel_values_cpu is not None:
        pixel_values_cpu = pixel_values_cpu[: args.max_samples]
        if share_pair_cpu is not None:
            share_pair_cpu = [share[: args.max_samples] for share in share_pair_cpu]
    if party_local_share_sample_count is not None and args.max_samples > 0:
        party_local_share_sample_count = min(party_local_share_sample_count, int(args.max_samples))

    targets = client_payload.get("targets")
    if targets is not None:
        targets = targets.detach().cpu()
        if args.max_samples > 0:
            targets = targets[: args.max_samples]
    sample_ids = client_payload.get("sample_ids")
    if sample_ids is not None and args.max_samples > 0:
        sample_ids = list(sample_ids)[: args.max_samples]

    runtime_pruning_keep_masks = None
    if runtime_pruning_keep_mask_pt is not None:
        require_existing_file(runtime_pruning_keep_mask_pt, "runtime pruning keep-mask payload")
        keep_mask_payload = helpers["load_tensor_payload"](runtime_pruning_keep_mask_pt)
        runtime_pruning_keep_masks = keep_mask_payload.get("stage_keep_masks")
        if not isinstance(runtime_pruning_keep_masks, list) or not runtime_pruning_keep_masks:
            raise ValueError("runtime pruning keep-mask payload missing non-empty stage_keep_masks")
        payload_sample_ids = keep_mask_payload.get("sample_ids")
        if payload_sample_ids is not None and sample_ids is not None:
            if list(payload_sample_ids)[: len(sample_ids)] != list(sample_ids):
                raise ValueError("runtime pruning keep-mask payload sample_ids do not match current input ordering")
        if args.max_samples > 0:
            runtime_pruning_keep_masks = [mask[: args.max_samples] for mask in runtime_pruning_keep_masks]

    start = time.time()
    cls_features_cpu = None
    token_features_cpu = None
    cpu_forward_mode = str(getattr(args, "cpu_forward_mode", "static_no_pruning"))
    forward_scope = STATIC_FORWARD_SCOPE
    if runtime == "spu":
        if args.include_intermediates:
            raise ValueError(
                "--include-intermediates is intentionally disabled for runtime=spu; "
                "the e2e reveal policy only returns final logits."
            )
        predictor_params_np = None
        if runtime_pruning_keep_masks is None:
            # Secure pruning: load predictor params so pruning happens inside SPU
            params, predictor_params_np, spu_metadata = load_static_vit_spu_params_with_predictor(
                bundle_dir,
                args.static_depth_limit,
                attention_policy=args.spu_attention_policy,
                activation_override=args.spu_activation_override,
                token_ratio_base_override=getattr(args, 'spu_token_ratio_base_override', 0.0),
            )
            forward_scope = spu_metadata.get("forward_scope", forward_scope)
        else:
            params, spu_metadata = load_static_vit_spu_params(
                bundle_dir,
                args.static_depth_limit,
                attention_policy=args.spu_attention_policy,
                activation_override=args.spu_activation_override,
                token_ratio_base_override=getattr(args, 'spu_token_ratio_base_override', 0.0),
            )
        layer_norm_calibration = None
        if args.spu_layer_norm_policy == "public_calibrated":
            if layer_norm_calibration_json is None:
                raise ValueError("--spu-layer-norm-calibration-json is required for public_calibrated policy")
            layer_norm_calibration = load_public_layer_norm_calibration(
                layer_norm_calibration_json,
                expected_depth=int(spu_metadata["depth"]),
            )
        logits_np = run_static_vit_forward_spu(
            (
                None
                if share_pair_cpu is not None or party_local_share_manifest_paths is not None
                else numpy_from_torch_tensor(pixel_values_cpu)
            ),
            params,
            Path(args.config).expanduser().resolve(),
            spu_metadata,
            args.spu_batch_size,
            args.spu_params_mode,
            pixel_value_shares_np=(
                None
                if share_pair_cpu is None
                else [numpy_from_torch_tensor(share) for share in share_pair_cpu]
            ),
            pixel_value_share_manifest_paths=party_local_share_manifest_paths,
            share_sample_count=party_local_share_sample_count,
            block_chunk_size=args.spu_block_chunk_size,
            layer_norm_chunk_size=args.spu_layer_norm_chunk_size,
            layer_norm_policy=args.spu_layer_norm_policy,
            layer_norm_calibration=layer_norm_calibration,
            activation_clip_value=args.spu_activation_clip_value,
            external_keep_masks_np=(
                None
                if runtime_pruning_keep_masks is None
                else [numpy_from_torch_tensor(mask.float()) for mask in runtime_pruning_keep_masks]
            ),
            predictor_params_np=predictor_params_np,
            pruning_metadata=(
                {
                    "pruning_loc": list(spu_metadata.get("pruning_loc", [])),
                    "token_keep_counts": list(spu_metadata.get("token_keep_counts", [])),
                }
                if predictor_params_np is not None
                else None
            ),
            token_recycle_scale=args.spu_token_recycle_scale,
        )
        logits_cpu = torch.from_numpy(logits_np).float()
        probabilities_cpu = torch.softmax(logits_cpu, dim=-1)
        if runtime_pruning_keep_masks is None:
            if predictor_params_np is not None:
                backend = "jax_spu_secure_pruning_forward_backend_v0"
            else:
                backend = "jax_spu_static_whole_forward_backend_v0"
        else:
            backend = "jax_spu_external_keep_mask_whole_forward_backend_v0"
            forward_scope = "student_patch_embed_blocks_head_with_external_runtime_pruning_keep_masks"
    else:
        bundle = helpers["load_frozen_bundle"](bundle_dir, device=args.device)
        model = bundle["model"]
        pixel_values = pixel_values_cpu.to(args.device)
        with torch.no_grad():
            if runtime_pruning_keep_masks is not None:
                outputs = run_external_keep_mask_student_whole_forward_limited(
                    model,
                    pixel_values,
                    runtime_pruning_keep_masks,
                    args.static_depth_limit,
                )
                forward_scope = "student_patch_embed_blocks_head_with_external_runtime_pruning_keep_masks"
            elif cpu_forward_mode == "runtime_pruning_reference":
                outputs = run_runtime_pruning_student_whole_forward_limited(
                    model,
                    pixel_values,
                    args.static_depth_limit,
                )
                forward_scope = "student_patch_embed_blocks_head_with_runtime_pruning_predictor_path"
            elif args.static_depth_limit >= 0:
                outputs = run_static_student_whole_forward_limited(model, pixel_values, args.static_depth_limit)
            else:
                outputs = helpers["run_static_student_whole_forward"](model, pixel_values)
            logits = outputs["logits"]
            cls_features = outputs["cls_features"]
            token_features = outputs["token_features"]
            probabilities = torch.softmax(logits, dim=-1)
        logits_cpu = logits.detach().cpu()
        probabilities_cpu = probabilities.detach().cpu()
        cls_features_cpu = cls_features.detach().cpu()
        token_features_cpu = token_features.detach().cpu()
        if runtime_pruning_keep_masks is not None:
            backend = "cpu_plaintext_external_keep_mask_whole_forward_reference_backend"
        elif cpu_forward_mode == "runtime_pruning_reference":
            backend = "cpu_plaintext_runtime_pruning_whole_forward_reference_backend"
        else:
            backend = "cpu_plaintext_whole_forward_reference_backend"
        spu_metadata = None
    elapsed_sec = time.time() - start

    raw_logits_before_output_calibration = None
    output_calibration = None
    output_calibration_json = (
        Path(args.output_calibration_json).expanduser().resolve()
        if getattr(args, "output_calibration_json", "")
        else None
    )
    if output_calibration_json is not None:
        require_existing_file(output_calibration_json, "E2E output calibration JSON")
        raw_logits_before_output_calibration = logits_cpu.clone()
        logits_cpu, output_calibration = apply_output_calibration(logits_cpu, output_calibration_json)
        probabilities_cpu = torch.softmax(logits_cpu, dim=-1)
        threshold = float(output_calibration["threshold"])

    argmax_predictions = logits_cpu.argmax(dim=1)
    threshold_predictions = None
    if threshold is not None and probabilities_cpu.shape[-1] == 2:
        threshold_predictions = (probabilities_cpu[:, 1] >= float(threshold)).long()

    payload = {
        "manifest_type": CANDIDATE_MANIFEST_TYPE,
        "runtime": runtime,
        "backend": backend,
        "bundle_dir": str(bundle_dir),
        "input_pt": None if input_pt is None else str(input_pt),
        "input_share_manifest_json": path_for_output(
            input_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="legacy_share_manifest",
        ),
        "input_share_public_manifest_json": (
            None if input_share_public_manifest_json is None else str(input_share_public_manifest_json)
        ),
        "input_p1_share_manifest_json": path_for_output(
            input_p1_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="p1_share_manifest",
        ),
        "input_p2_share_manifest_json": path_for_output(
            input_p2_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="p2_share_manifest",
        ),
        "input_source": input_source,
        "runtime_pruning_keep_mask_pt": (
            None if runtime_pruning_keep_mask_pt is None else str(runtime_pruning_keep_mask_pt)
        ),
        "sample_ids": sample_ids,
        "targets": targets,
        "threshold": threshold,
        "logits": logits_cpu,
        "probabilities": probabilities_cpu,
        "argmax_predictions": argmax_predictions,
        "threshold_predictions": threshold_predictions,
    }
    if runtime == "cpu":
        payload["cpu_forward_mode"] = cpu_forward_mode
    if raw_logits_before_output_calibration is not None:
        payload["raw_logits_before_output_calibration"] = raw_logits_before_output_calibration
        payload["output_calibration"] = output_calibration
    if args.include_intermediates and cls_features_cpu is not None and token_features_cpu is not None:
        payload["cls_features"] = cls_features_cpu
        payload["token_features"] = token_features_cpu
    if runtime == "spu":
        payload["spu_metadata"] = {
            "config": str(Path(args.config).expanduser().resolve()),
            "spu_batch_size": int(args.spu_batch_size),
            "spu_params_mode": args.spu_params_mode,
            "spu_block_chunk_size": int(args.spu_block_chunk_size),
            "spu_layer_norm_chunk_size": int(args.spu_layer_norm_chunk_size),
            "spu_layer_norm_policy": args.spu_layer_norm_policy,
            "spu_layer_norm_calibration_json": (
                str(layer_norm_calibration_json) if layer_norm_calibration_json is not None else None
            ),
            "spu_activation_clip_value": float(args.spu_activation_clip_value),
            "spu_token_recycle_scale": float(args.spu_token_recycle_scale),
            "spu_token_ratio_base_override": float(getattr(args, "spu_token_ratio_base_override", 0.0)),
            "static_forward_metadata": spu_metadata,
            "reveal_policy": "final_logits_only",
            "private_input_paths_redacted": redact_private_input_paths,
        }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)

    summary = {
        "manifest_type": SUMMARY_MANIFEST_TYPE,
        "runtime": runtime,
        "backend": backend,
        "bundle_dir": str(bundle_dir),
        "input_pt": None if input_pt is None else str(input_pt),
        "input_share_manifest_json": path_for_output(
            input_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="legacy_share_manifest",
        ),
        "input_share_public_manifest_json": (
            None if input_share_public_manifest_json is None else str(input_share_public_manifest_json)
        ),
        "input_p1_share_manifest_json": path_for_output(
            input_p1_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="p1_share_manifest",
        ),
        "input_p2_share_manifest_json": path_for_output(
            input_p2_share_manifest_json,
            redact=redact_private_input_paths,
            redaction_label="p2_share_manifest",
        ),
        "input_source": input_source,
        "runtime_pruning_keep_mask_pt": (
            None if runtime_pruning_keep_mask_pt is None else str(runtime_pruning_keep_mask_pt)
        ),
        "output_pt": str(output_pt),
        "sample_count": int(logits_cpu.shape[0]),
        "elapsed_sec": float(elapsed_sec),
        "threshold": threshold,
        "finite_logits": bool(torch.isfinite(logits_cpu).all().item()),
        "include_intermediates": bool(args.include_intermediates),
        "max_samples": int(args.max_samples),
        "cpu_forward_mode": cpu_forward_mode if runtime == "cpu" else None,
        "runtime_pruning_keep_mask_stage_count": (
            None if runtime_pruning_keep_masks is None else len(runtime_pruning_keep_masks)
        ),
        "static_depth_limit": int(args.static_depth_limit),
        "effective_static_depth": (
            int(spu_metadata["depth"])
            if runtime == "spu"
            else normalize_depth_limit(args.static_depth_limit, full_depth=12)
        ),
        "logits": tensor_stats(logits_cpu),
        "probabilities": tensor_stats(probabilities_cpu),
        "raw_logits_before_output_calibration": (
            None
            if raw_logits_before_output_calibration is None
            else tensor_stats(raw_logits_before_output_calibration)
        ),
        "output_calibration": output_calibration,
        "prediction_preview": {
            "logits": [[float(value) for value in row] for row in logits_cpu.tolist()],
            "probabilities": [[float(value) for value in row] for row in probabilities_cpu.tolist()],
            "argmax_predictions": [int(value) for value in argmax_predictions.tolist()],
            "threshold_predictions": (
                None if threshold_predictions is None else [int(value) for value in threshold_predictions.tolist()]
            ),
        },
        "cls_features": tensor_stats(cls_features_cpu) if cls_features_cpu is not None else None,
        "token_features": tensor_stats(token_features_cpu) if token_features_cpu is not None else None,
        "forward_scope": forward_scope,
    }
    if runtime == "spu":
        summary["spu"] = {
            "config": str(Path(args.config).expanduser().resolve()),
            "spu_batch_size": int(args.spu_batch_size),
            "spu_params_mode": args.spu_params_mode,
            "spu_block_chunk_size": int(args.spu_block_chunk_size),
            "spu_layer_norm_chunk_size": int(args.spu_layer_norm_chunk_size),
            "spu_layer_norm_policy": args.spu_layer_norm_policy,
            "spu_layer_norm_calibration_json": (
                str(layer_norm_calibration_json) if layer_norm_calibration_json is not None else None
            ),
            "spu_activation_clip_value": float(args.spu_activation_clip_value),
            "spu_token_recycle_scale": float(args.spu_token_recycle_scale),
            "spu_token_ratio_base_override": float(getattr(args, "spu_token_ratio_base_override", 0.0)),
            "runtime_pruning_keep_mask_pt": (
                None if runtime_pruning_keep_mask_pt is None else str(runtime_pruning_keep_mask_pt)
            ),
            "spu_forward_graph_mode": (
                "reveal_less_block_chunked" if int(args.spu_block_chunk_size) > 0 else "monolithic"
            ),
            "reveal_policy": "final_logits_only",
            "input_mode": (
                "party_local_debug_share_load"
                if party_local_share_manifest_paths is not None
                else "debug_per_party_additive_share_manifests"
                if party_share_manifests is not None
                else "debug_additive_share_manifest"
                if share_pair_cpu is not None
                else "plaintext_pixel_package"
            ),
            "host_plaintext_pixel_values_materialized": (
                False if share_pair_cpu is not None or party_local_share_manifest_paths is not None else True
            ),
            "host_private_share_tensors_loaded": False if party_local_share_manifest_paths is not None else share_pair_cpu is not None,
            "private_input_paths_redacted": redact_private_input_paths,
            "driver_private_share_manifest_paths_recorded": (
                bool(party_local_share_manifest_paths is not None) and not redact_private_input_paths
            ),
            "static_forward_metadata": spu_metadata,
        }
        if share_manifest is not None:
            summary["spu"]["share_manifest_type"] = share_manifest.get("manifest_type")
            summary["spu"]["share_semantics"] = share_manifest.get("share_semantics")
        if party_share_manifests is not None:
            summary["spu"]["party_manifest_types"] = [
                party_manifest.get("manifest_type") for party_manifest in party_share_manifests
            ]
        if party_local_share_manifest_paths is not None:
            summary["privacy_note"] = (
                "This candidate runs the static whole-forward function on SPU/JAX and reveals final logits only. "
                "In party-local share-load mode, the driver does not materialize plaintext pixel_values or private "
                "share tensors; P1/P2 device functions load their own share files before SPU recomposition. "
                "It is still a debug bridge until P1/P2 are launched as independent party processes."
            )
        else:
            summary["privacy_note"] = (
                "This candidate runs the static whole-forward function on SPU/JAX and reveals final logits only. "
                "When share manifests are used without party-local share loading, the runner no longer feeds "
                "plaintext pixel_values as its input, but the driver may still materialize private share tensors."
            )
        summary["spu_params_mode"] = summary["spu"]["spu_params_mode"]
        summary["spu_forward_graph_mode"] = summary["spu"]["spu_forward_graph_mode"]
        summary["reveal_policy"] = summary["spu"]["reveal_policy"]
        summary["input_mode"] = summary["spu"]["input_mode"]
        summary["spu_token_recycle_scale"] = summary["spu"]["spu_token_recycle_scale"]
        summary["host_plaintext_pixel_values_materialized"] = summary["spu"]["host_plaintext_pixel_values_materialized"]
        summary["host_private_share_tensors_loaded"] = summary["spu"]["host_private_share_tensors_loaded"]
        summary["private_input_paths_redacted"] = summary["spu"]["private_input_paths_redacted"]
        summary["driver_private_share_manifest_paths_recorded"] = summary["spu"]["driver_private_share_manifest_paths_recorded"]
    else:
        summary["privacy_note"] = (
            "This candidate uses the CPU reference backend to freeze the output contract. "
            "It is not a secure SPU execution result."
        )
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_calibrate_layer_norm(args):
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    input_pt = Path(args.input_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    payload = compute_public_layer_norm_calibration(
        bundle_dir=bundle_dir,
        input_pt=input_pt,
        output_json=output_json,
        static_depth_limit=int(args.static_depth_limit),
        max_samples=int(args.max_samples),
        attention_policy=str(args.spu_attention_policy),
        activation_override=str(args.spu_activation_override),
        activation_clip_value=float(args.spu_activation_clip_value),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_audit_input_shares(args):
    helpers = load_runtime_helpers()

    bundle_dir = helpers["resolve_bundle_dir"](args.bundle_dir)
    output_json = Path(args.output_json).expanduser().resolve()
    input_share_manifest_json = (
        Path(args.input_share_manifest_json).expanduser().resolve()
        if args.input_share_manifest_json
        else None
    )
    input_share_public_manifest_json = (
        Path(args.input_share_public_manifest_json).expanduser().resolve()
        if args.input_share_public_manifest_json
        else None
    )
    input_p1_share_manifest_json = (
        Path(args.input_p1_share_manifest_json).expanduser().resolve()
        if args.input_p1_share_manifest_json
        else None
    )
    input_p2_share_manifest_json = (
        Path(args.input_p2_share_manifest_json).expanduser().resolve()
        if args.input_p2_share_manifest_json
        else None
    )

    using_legacy_share_manifest = input_share_manifest_json is not None
    using_party_share_manifests = any(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    )
    if using_legacy_share_manifest and using_party_share_manifests:
        raise ValueError(
            "--input-share-manifest-json cannot be combined with split public/P1/P2 share manifests"
        )
    if using_party_share_manifests and not all(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    ):
        raise ValueError(
            "--input-share-public-manifest-json, --input-p1-share-manifest-json, and "
            "--input-p2-share-manifest-json must be set together"
        )
    if input_share_manifest_json is not None:
        share_payload = load_debug_input_share_pair(input_share_manifest_json)
        share_manifest = share_payload["manifest"]
        party_share_manifests = None
        input_source = str(input_share_manifest_json)
    elif using_party_share_manifests:
        share_payload = load_debug_input_share_pair_from_party_manifests(
            input_share_public_manifest_json,
            input_p1_share_manifest_json,
            input_p2_share_manifest_json,
        )
        share_manifest = share_payload["manifest"]
        party_share_manifests = share_payload["party_manifests"]
        input_source = str(input_share_public_manifest_json)
    else:
        raise ValueError("audit-input-shares requires legacy or split share manifests")

    share_pair_cpu = share_payload["share_tensors"]
    if args.max_samples > 0:
        share_pair_cpu = [share[: args.max_samples] for share in share_pair_cpu]

    share_pair_np = [numpy_from_torch_tensor(share) for share in share_pair_cpu]
    reconstructed_cpu_np = share_pair_np[0] + share_pair_np[1]
    params, spu_metadata = load_static_vit_spu_params(
        bundle_dir,
        0,
        attention_policy=args.spu_attention_policy,
        activation_override=args.spu_activation_override,
    )
    patch_tokens_cpu_np, tokens_with_pos_cpu_np = static_patch_embed_numpy(
        reconstructed_cpu_np,
        params,
        spu_metadata,
    )

    plaintext_compare = None
    if args.input_pt:
        input_pt = Path(args.input_pt).expanduser().resolve()
        require_existing_file(input_pt, "input pixel package")
        client_payload = helpers["load_client_pixel_package"](input_pt)
        pixel_values_cpu = client_payload["pixel_values"].detach().cpu()
        if args.max_samples > 0:
            pixel_values_cpu = pixel_values_cpu[: args.max_samples]
        plaintext_compare = compare_array_payload(numpy_from_torch_tensor(pixel_values_cpu), reconstructed_cpu_np)
    else:
        input_pt = None

    start = time.time()
    spu_outputs = run_share_recomposition_audit_spu(
        params,
        Path(args.config).expanduser().resolve(),
        spu_metadata,
        args.spu_params_mode,
        pixel_value_shares_np=share_pair_np,
    )
    elapsed_sec = time.time() - start

    summary = {
        "manifest_type": SHARE_AUDIT_MANIFEST_TYPE,
        "bundle_dir": str(bundle_dir),
        "input_source": input_source,
        "input_pt": None if input_pt is None else str(input_pt),
        "input_share_manifest_json": None if input_share_manifest_json is None else str(input_share_manifest_json),
        "input_share_public_manifest_json": (
            None if input_share_public_manifest_json is None else str(input_share_public_manifest_json)
        ),
        "input_p1_share_manifest_json": (
            None if input_p1_share_manifest_json is None else str(input_p1_share_manifest_json)
        ),
        "input_p2_share_manifest_json": (
            None if input_p2_share_manifest_json is None else str(input_p2_share_manifest_json)
        ),
        "sample_count": int(reconstructed_cpu_np.shape[0]),
        "max_samples": int(args.max_samples),
        "elapsed_sec": float(elapsed_sec),
        "share_manifest_type": share_manifest.get("manifest_type"),
        "party_manifest_types": (
            None
            if party_share_manifests is None
            else [party_manifest.get("manifest_type") for party_manifest in party_share_manifests]
        ),
        "spu": {
            "config": str(Path(args.config).expanduser().resolve()),
            "spu_params_mode": args.spu_params_mode,
            "input_mode": (
                "debug_per_party_additive_share_manifests"
                if party_share_manifests is not None
                else "debug_additive_share_manifest"
            ),
            "static_forward_metadata": spu_metadata,
        },
        "plaintext_vs_cpu_reconstructed_pixel_values": plaintext_compare,
        "cpu_vs_spu_reconstructed_pixel_values": compare_array_payload(
            reconstructed_cpu_np,
            spu_outputs["reconstructed_pixel_values"],
        ),
        "cpu_vs_spu_patch_tokens": compare_array_payload(
            patch_tokens_cpu_np,
            spu_outputs["patch_tokens"],
        ),
        "cpu_vs_spu_tokens_with_pos": compare_array_payload(
            tokens_with_pos_cpu_np,
            spu_outputs["tokens_with_pos"],
        ),
        "privacy_note": (
            "This is an explicit debug audit. It reveals reconstructed pixels and patch embeddings to localize "
            "the split-share failure boundary and must not be used as the production reveal policy."
        ),
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_probe_block(args):
    import torch

    helpers = load_runtime_helpers()
    runtime = args.runtime
    bundle_dir = helpers["resolve_bundle_dir"](args.bundle_dir)
    input_pt = Path(args.input_pt).expanduser().resolve() if args.input_pt else None
    input_share_manifest_json = (
        Path(args.input_share_manifest_json).expanduser().resolve()
        if args.input_share_manifest_json
        else None
    )
    input_share_public_manifest_json = (
        Path(args.input_share_public_manifest_json).expanduser().resolve()
        if args.input_share_public_manifest_json
        else None
    )
    input_p1_share_manifest_json = (
        Path(args.input_p1_share_manifest_json).expanduser().resolve()
        if args.input_p1_share_manifest_json
        else None
    )
    input_p2_share_manifest_json = (
        Path(args.input_p2_share_manifest_json).expanduser().resolve()
        if args.input_p2_share_manifest_json
        else None
    )
    output_json = Path(args.output_json).expanduser().resolve()

    share_pair_cpu = None
    share_manifest = None
    party_share_manifests = None
    using_legacy_share_manifest = input_share_manifest_json is not None
    using_party_share_manifests = any(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    )
    if using_legacy_share_manifest and using_party_share_manifests:
        raise ValueError(
            "--input-share-manifest-json cannot be combined with split public/P1/P2 share manifests"
        )
    if using_party_share_manifests and not all(
        item is not None
        for item in [input_share_public_manifest_json, input_p1_share_manifest_json, input_p2_share_manifest_json]
    ):
        raise ValueError(
            "--input-share-public-manifest-json, --input-p1-share-manifest-json, and "
            "--input-p2-share-manifest-json must be set together"
        )
    if runtime == "spu" and input_share_manifest_json is not None:
        share_payload = load_debug_input_share_pair(input_share_manifest_json)
        share_pair_cpu = share_payload["share_tensors"]
        share_manifest = share_payload["manifest"]
        sample_ids = share_payload.get("sample_ids")
        pixel_values_cpu = share_pair_cpu[0]
        input_source = str(input_share_manifest_json)
    elif runtime == "spu" and using_party_share_manifests:
        share_payload = load_debug_input_share_pair_from_party_manifests(
            input_share_public_manifest_json,
            input_p1_share_manifest_json,
            input_p2_share_manifest_json,
        )
        share_pair_cpu = share_payload["share_tensors"]
        share_manifest = share_payload["manifest"]
        party_share_manifests = share_payload["party_manifests"]
        sample_ids = share_payload.get("sample_ids")
        pixel_values_cpu = share_pair_cpu[0]
        input_source = str(input_share_public_manifest_json)
    else:
        if input_pt is None:
            raise ValueError("probe-block requires --input-pt unless runtime=spu uses share manifests")
        require_existing_file(input_pt, "input pixel package")
        client_payload = helpers["load_client_pixel_package"](input_pt)
        pixel_values_cpu = client_payload["pixel_values"].detach().cpu()
        sample_ids = client_payload.get("sample_ids")
        input_source = str(input_pt)
    threshold = helpers["resolve_threshold"](bundle_dir, None)
    if args.max_samples > 0:
        pixel_values_cpu = pixel_values_cpu[: args.max_samples]
        if share_pair_cpu is not None:
            share_pair_cpu = [share[: args.max_samples] for share in share_pair_cpu]

    if sample_ids is not None and args.max_samples > 0:
        sample_ids = list(sample_ids)[: args.max_samples]

    start = time.time()
    if runtime == "spu":
        params, spu_metadata = load_static_vit_spu_params(
            bundle_dir,
            args.static_depth_limit,
            attention_policy=args.spu_attention_policy,
            activation_override=args.spu_activation_override,
        )
        layer_norm_calibration = None
        layer_norm_calibration_json = (
            Path(args.spu_layer_norm_calibration_json).expanduser().resolve()
            if getattr(args, "spu_layer_norm_calibration_json", "")
            else None
        )
        if args.spu_layer_norm_policy == "public_calibrated":
            if layer_norm_calibration_json is None:
                raise ValueError("--spu-layer-norm-calibration-json is required for public_calibrated policy")
            layer_norm_calibration = load_public_layer_norm_calibration(
                layer_norm_calibration_json,
                expected_depth=int(spu_metadata["depth"]),
            )
        logits_np, probe_tensors = run_static_vit_forward_spu(
            None if share_pair_cpu is not None else numpy_from_torch_tensor(pixel_values_cpu),
            params,
            Path(args.config).expanduser().resolve(),
            spu_metadata,
            args.spu_batch_size,
            args.spu_params_mode,
            probe_block_index=args.probe_block_index,
            pixel_value_shares_np=(
                None
                if share_pair_cpu is None
                else [numpy_from_torch_tensor(share) for share in share_pair_cpu]
            ),
            layer_norm_chunk_size=args.spu_layer_norm_chunk_size,
            layer_norm_policy=args.spu_layer_norm_policy,
            layer_norm_calibration=layer_norm_calibration,
            activation_clip_value=args.spu_activation_clip_value,
            token_recycle_scale=args.spu_token_recycle_scale,
        )
        logits_cpu = torch.from_numpy(logits_np).float()
        probabilities_cpu = torch.softmax(logits_cpu, dim=-1)
        probe_tensors["final_logits"] = logits_cpu
        probe_tensors["final_probabilities"] = probabilities_cpu
        backend = "jax_spu_static_whole_forward_backend_v0"
        effective_static_depth = int(spu_metadata["depth"])
    else:
        bundle = helpers["load_frozen_bundle"](bundle_dir, device=args.device)
        model = bundle["model"]
        outputs = run_static_student_whole_forward_probe(
            model,
            pixel_values_cpu.to(args.device),
            args.static_depth_limit,
            args.probe_block_index,
        )
        logits_cpu = outputs["logits"].detach().cpu().float()
        probabilities_cpu = outputs["probabilities"].detach().cpu().float()
        probe_tensors = outputs["probe_tensors"]
        backend = "cpu_plaintext_whole_forward_reference_backend"
        spu_metadata = None
        effective_static_depth = int(outputs["static_depth"])
    elapsed_sec = time.time() - start

    argmax_predictions = logits_cpu.argmax(dim=1)
    threshold_predictions = None
    if threshold is not None and probabilities_cpu.shape[-1] == 2:
        threshold_predictions = (probabilities_cpu[:, 1] >= float(threshold)).long()

    probe_records = {}
    for name in BLOCK_PROBE_TENSOR_NAMES:
        if name in probe_tensors:
            probe_records[name] = debug_tensor_record(probe_tensors[name])

    summary = {
        "manifest_type": BLOCK_PROBE_MANIFEST_TYPE,
        "runtime": runtime,
        "backend": backend,
        "bundle_dir": str(bundle_dir),
        "input_pt": None if input_pt is None else str(input_pt),
        "input_share_manifest_json": None if input_share_manifest_json is None else str(input_share_manifest_json),
        "input_share_public_manifest_json": (
            None if input_share_public_manifest_json is None else str(input_share_public_manifest_json)
        ),
        "input_p1_share_manifest_json": (
            None if input_p1_share_manifest_json is None else str(input_p1_share_manifest_json)
        ),
        "input_p2_share_manifest_json": (
            None if input_p2_share_manifest_json is None else str(input_p2_share_manifest_json)
        ),
        "input_source": input_source,
        "output_json": str(output_json),
        "sample_count": int(logits_cpu.shape[0]),
        "max_samples": int(args.max_samples),
        "threshold": threshold,
        "static_depth_limit": int(args.static_depth_limit),
        "effective_static_depth": int(effective_static_depth),
        "probe_block_index": int(args.probe_block_index),
        "probe_block_ordinal": int(args.probe_block_index) + 1,
        "spu_token_recycle_scale": float(getattr(args, "spu_token_recycle_scale", 0.0)),
        "elapsed_sec": float(elapsed_sec),
        "finite_logits": bool(torch.isfinite(logits_cpu).all().item()),
        "sample_ids": sample_ids,
        "argmax_predictions": [int(value) for value in argmax_predictions.tolist()],
        "threshold_predictions": (
            None if threshold_predictions is None else [int(value) for value in threshold_predictions.tolist()]
        ),
        "probe_tensors": probe_records,
        "probe_final_output_safety": {
            "full_candidate_decision_safe": False,
            "reason": (
                "probe-block reveals intermediate tensors and may alter the SPU/JAX debug graph. "
                "Use final_logits/final_probabilities here only as debug-graph outputs; "
                "use non-probe run+verify for full candidate decisions."
            ),
        },
        "privacy_note": (
            "This is an explicit debug probe for same-depth CPU/SPU drift attribution. "
            "It reveals intermediate CLS-token summaries for the selected block and must not be treated as the default e2e reveal policy."
        ),
    }
    if runtime == "spu":
        summary["spu"] = {
            "config": str(Path(args.config).expanduser().resolve()),
            "spu_batch_size": int(args.spu_batch_size),
            "spu_layer_norm_policy": args.spu_layer_norm_policy,
            "spu_layer_norm_calibration_json": (
                None if layer_norm_calibration_json is None else str(layer_norm_calibration_json)
            ),
            "spu_params_mode": args.spu_params_mode,
            "spu_attention_policy": args.spu_attention_policy,
            "spu_activation_override": args.spu_activation_override,
            "spu_activation_clip_value": float(args.spu_activation_clip_value),
            "spu_token_recycle_scale": float(args.spu_token_recycle_scale),
            "spu_token_ratio_base_override": float(getattr(args, "spu_token_ratio_base_override", 0.0)),
            "input_mode": (
                "debug_per_party_additive_share_manifests"
                if party_share_manifests is not None
                else "debug_additive_share_manifest"
                if share_pair_cpu is not None
                else "plaintext_pixel_package"
            ),
            "static_forward_metadata": spu_metadata,
        }
        if share_manifest is not None:
            summary["spu"]["share_manifest_type"] = share_manifest.get("manifest_type")
        if party_share_manifests is not None:
            summary["spu"]["party_manifest_types"] = [
                party_manifest.get("manifest_type") for party_manifest in party_share_manifests
            ]

    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_compare_block_probe(args):
    import torch

    reference_json = Path(args.reference_json).expanduser().resolve()
    candidate_json = Path(args.candidate_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    reference = load_json(reference_json)
    candidate = load_json(candidate_json)

    reference_probe = reference.get("probe_tensors", {})
    candidate_probe = candidate.get("probe_tensors", {})
    stage_errors = {}
    for name in BLOCK_PROBE_TENSOR_NAMES:
        if name in reference_probe and name in candidate_probe:
            stage_errors[name] = compare_debug_records(reference_probe[name], candidate_probe[name])

    if not stage_errors:
        raise ValueError("no shared probe tensors found between reference and candidate probe JSONs")

    stage_ranking_including_probe_final_outputs = [
        {"name": name, **metrics}
        for name, metrics in sorted(
            stage_errors.items(),
            key=lambda item: item[1]["max_abs_error"],
            reverse=True,
        )
    ]
    intermediate_stage_errors = {
        name: stage_errors[name]
        for name in BLOCK_PROBE_INTERMEDIATE_TENSOR_NAMES
        if name in stage_errors
    }
    stage_ranking = [
        {"name": name, **metrics}
        for name, metrics in sorted(
            intermediate_stage_errors.items(),
            key=lambda item: item[1]["max_abs_error"],
            reverse=True,
        )
    ]

    reference_logits = record_to_float_tensor(reference_probe["final_logits"])
    candidate_logits = record_to_float_tensor(candidate_probe["final_logits"])
    reference_probabilities = record_to_float_tensor(reference_probe["final_probabilities"])
    candidate_probabilities = record_to_float_tensor(candidate_probe["final_probabilities"])
    reference_argmax = reference_logits.argmax(dim=1)
    candidate_argmax = candidate_logits.argmax(dim=1)
    threshold = reference.get("threshold")
    reference_threshold_predictions = None
    candidate_threshold_predictions = None
    if threshold is not None and reference_probabilities.shape[-1] == 2:
        reference_threshold_predictions = (reference_probabilities[:, 1] >= float(threshold)).long()
        candidate_threshold_predictions = (candidate_probabilities[:, 1] >= float(threshold)).long()

    logits_abs_error = (reference_logits - candidate_logits).abs()
    probabilities_abs_error = (reference_probabilities - candidate_probabilities).abs()
    debug_graph_prediction_match = {
        "argmax_match_ratio": float((reference_argmax == candidate_argmax).float().mean().item()),
        "threshold_match_ratio": (
            None
            if reference_threshold_predictions is None or candidate_threshold_predictions is None
            else float(
                (reference_threshold_predictions == candidate_threshold_predictions)
                .float()
                .mean()
                .item()
            )
        ),
    }

    summary = {
        "manifest_type": BLOCK_PROBE_COMPARE_MANIFEST_TYPE,
        "reference_json": str(reference_json),
        "candidate_json": str(candidate_json),
        "reference_runtime": reference.get("runtime"),
        "candidate_runtime": candidate.get("runtime"),
        "static_depth_limit": reference.get("static_depth_limit"),
        "probe_block_index": reference.get("probe_block_index"),
        "probe_block_ordinal": reference.get("probe_block_ordinal"),
        "stage_errors": stage_errors,
        "intermediate_stage_errors": intermediate_stage_errors,
        "stage_ranking_by_max_abs_error": stage_ranking,
        "stage_ranking_including_probe_final_outputs": stage_ranking_including_probe_final_outputs,
        "debug_graph_final_output": {
            "safety": "debug_graph_only_not_full_candidate_decision",
            "logits_error": {
                "max_abs_error": float(logits_abs_error.max().item()),
                "mean_abs_error": float(logits_abs_error.mean().item()),
            },
            "probabilities_error": {
                "max_abs_error": float(probabilities_abs_error.max().item()),
                "mean_abs_error": float(probabilities_abs_error.mean().item()),
            },
            "prediction_match": debug_graph_prediction_match,
            "note": (
                "probe-block may reveal intermediate tensors and can alter the SPU/JAX debug graph. "
                "Use non-probe run+verify outputs for full candidate decisions."
            ),
        },
        "prediction_match": debug_graph_prediction_match,
        "prediction_match_is_full_candidate_safe": False,
        "prediction_match_safety_note": (
            "This value is retained for backward compatibility and describes only the probe debug graph. "
            "It must not be used as a full candidate decision result."
        ),
        "largest_stage": stage_ranking[0],
        "threshold": threshold,
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_verify(args):
    import torch

    helpers = load_runtime_helpers()
    reference_pt = Path(args.reference_pt).expanduser().resolve()
    candidate_pt = Path(args.candidate_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    reference = helpers["load_tensor_payload"](reference_pt)
    candidate = helpers["load_tensor_payload"](candidate_pt)

    reference_logits = reference.get("logits")
    candidate_logits = candidate.get("logits")
    if reference_logits is None or candidate_logits is None:
        raise ValueError("both reference and candidate payloads must contain logits")

    reference_logits = reference_logits.detach().cpu().float()
    candidate_logits = candidate_logits.detach().cpu().float()
    slice_count = int(args.max_samples) if int(args.max_samples) > 0 else None
    if slice_count is None and args.allow_prefix_candidate:
        if (
            candidate_logits.ndim == reference_logits.ndim
            and tuple(candidate_logits.shape[1:]) == tuple(reference_logits.shape[1:])
            and 0 < int(candidate_logits.shape[0]) <= int(reference_logits.shape[0])
        ):
            slice_count = int(candidate_logits.shape[0])
    if slice_count is not None:
        reference_logits = reference_logits[:slice_count]
        candidate_logits = candidate_logits[:slice_count]
    if tuple(reference_logits.shape) != tuple(candidate_logits.shape):
        raise ValueError(
            f"logits shape mismatch: {tuple(reference_logits.shape)} vs {tuple(candidate_logits.shape)}"
        )

    def slice_first_dim(value):
        if value is None or slice_count is None:
            return value
        if torch.is_tensor(value):
            return value[:slice_count]
        if isinstance(value, (list, tuple)):
            return value[:slice_count]
        return value

    reference_probabilities = reference.get("probabilities")
    candidate_probabilities = candidate.get("probabilities")
    if reference_probabilities is None:
        reference_probabilities = torch.softmax(reference_logits, dim=-1)
    else:
        reference_probabilities = slice_first_dim(reference_probabilities).detach().cpu().float()
    if candidate_probabilities is None:
        candidate_probabilities = torch.softmax(candidate_logits, dim=-1)
    else:
        candidate_probabilities = slice_first_dim(candidate_probabilities).detach().cpu().float()

    reference_argmax = reference.get("argmax_predictions")
    if reference_argmax is None:
        reference_argmax = reference_logits.argmax(dim=1)
    else:
        reference_argmax = slice_first_dim(reference_argmax)
    candidate_argmax = candidate.get("argmax_predictions")
    if candidate_argmax is None:
        candidate_argmax = candidate_logits.argmax(dim=1)
    else:
        candidate_argmax = slice_first_dim(candidate_argmax)

    reference_threshold = reference.get("threshold")
    candidate_threshold = candidate.get("threshold")
    if candidate_threshold is None:
        candidate_threshold = reference_threshold

    reference_threshold_predictions = reference.get("threshold_predictions")
    reference_threshold_predictions = slice_first_dim(reference_threshold_predictions)
    if reference_threshold_predictions is None and reference_threshold is not None and reference_probabilities.shape[-1] == 2:
        reference_threshold_predictions = (reference_probabilities[:, 1] >= float(reference_threshold)).long()
    candidate_threshold_predictions = candidate.get("threshold_predictions")
    candidate_threshold_predictions = slice_first_dim(candidate_threshold_predictions)
    if candidate_threshold_predictions is None and candidate_threshold is not None and candidate_probabilities.shape[-1] == 2:
        candidate_threshold_predictions = (candidate_probabilities[:, 1] >= float(candidate_threshold)).long()

    logits_abs_error = (reference_logits - candidate_logits).abs()
    probabilities_abs_error = (reference_probabilities - candidate_probabilities).abs()

    summary = {
        "manifest_type": "transshield_e2e_secure_vit_verify_v0",
        "reference_pt": str(reference_pt),
        "candidate_pt": str(candidate_pt),
        "reference_manifest_type": reference.get("manifest_type"),
        "candidate_manifest_type": candidate.get("manifest_type"),
        "sample_count": int(reference_logits.shape[0]),
        "slice_count": slice_count,
        "allow_prefix_candidate": bool(args.allow_prefix_candidate),
        "logits_shape": list(reference_logits.shape),
        "logits_error": {
            "max_abs_error": float(logits_abs_error.max().item()),
            "mean_abs_error": float(logits_abs_error.mean().item()),
        },
        "probabilities_error": {
            "max_abs_error": float(probabilities_abs_error.max().item()),
            "mean_abs_error": float(probabilities_abs_error.mean().item()),
        },
        "prediction_match": {
            "argmax_match_ratio": helpers["compare_prediction_match"](reference_argmax, candidate_argmax),
            "threshold_match_ratio": helpers["compare_prediction_match"](
                reference_threshold_predictions,
                candidate_threshold_predictions,
            ),
        },
        "threshold": {
            "reference_threshold": reference_threshold,
            "candidate_threshold": candidate_threshold,
        },
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))



def command_block1_subgraph_smoke(args):
    return run_block1_subgraph_smoke(args, load_runtime_helpers())


def command_runtime_primitive_smoke(args):
    return run_runtime_primitive_smoke(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Transshield e2e whole-forward integration entry. "
            "CPU reference backend and experimental static JAX/SPU backend are available."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    primitive_parser = subparsers.add_parser(
        "runtime-primitive-smoke",
        help="debug-only synthetic SPU primitive workload smoke",
    )
    primitive_parser.add_argument("--output-json", required=True)
    primitive_parser.add_argument("--config", default=str(DEFAULT_SPU_CONFIG))
    primitive_parser.add_argument("--token-count", type=int, default=197)
    primitive_parser.add_argument("--embed-dim", type=int, default=384)
    primitive_parser.add_argument("--num-heads", type=int, default=6)
    primitive_parser.add_argument("--mlp-ratio", type=float, default=4.0)
    primitive_parser.add_argument(
        "--layer-norm-chunk-size",
        type=int,
        default=0,
        help="Experimental SPU layer-norm feature chunk size; 0 keeps the original reduction graph.",
    )
    primitive_parser.add_argument(
        "--layer-norm-policy",
        choices=["exact", "affine"],
        default="exact",
        help="Experimental primitive smoke layer-norm policy; affine skips secret mean/variance reduction.",
    )
    primitive_parser.add_argument(
        "--attention-policy",
        choices=["standard", "uniform"],
        default="standard",
        help="Experimental primitive smoke attention policy; uniform skips secret softmax.",
    )
    primitive_parser.add_argument("--seed", type=int, default=0)
    primitive_parser.set_defaults(func=command_runtime_primitive_smoke)

    prepare_parser = subparsers.add_parser("prepare", help="prepare a server-friendly whole-forward pack")
    prepare_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--input-pt", required=True)
    prepare_parser.add_argument("--reference-pt", required=True)
    prepare_parser.add_argument("--contract-json", default="")
    prepare_parser.add_argument("--spu-config", default=str(DEFAULT_SPU_CONFIG))
    prepare_parser.add_argument("--spu-template", default=str(DEFAULT_E2E_SPU_TEMPLATE))
    prepare_parser.set_defaults(func=command_prepare)

    calibrate_ln_parser = subparsers.add_parser(
        "calibrate-layer-norm",
        help="build public calibration stats for SPU public_calibrated layer norm",
    )
    calibrate_ln_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    calibrate_ln_parser.add_argument("--input-pt", required=True)
    calibrate_ln_parser.add_argument("--output-json", required=True)
    calibrate_ln_parser.add_argument("--max-samples", type=int, default=0)
    calibrate_ln_parser.add_argument(
        "--static-depth-limit",
        type=int,
        default=-1,
        help="Build calibration stats for the first N transformer blocks, -1 means all blocks.",
    )
    calibrate_ln_parser.add_argument(
        "--spu-attention-policy",
        choices=["smoothed", "standard", "uniform", "identity"],
        default="uniform",
        help="Attention policy used while collecting public calibration activations.",
    )
    calibrate_ln_parser.add_argument(
        "--spu-activation-override",
        choices=[
            "bundle",
            "gelu",
            "fixed_square",
            "learnable_square",
            "learnable_quadratic",
            "learnable_quadratic_gelu_init",
            "pade_gelu",
            "lut_gelu_4",
            "lut_gelu_8",
            "lut_gelu_16",
            "lut_gelu_32",
        ],
        default="bundle",
    )
    calibrate_ln_parser.add_argument(
        "--spu-activation-clip-value",
        type=float,
        default=0.0,
        help="Clip MLP pre-activation to [-value, value] before the SPU-friendly activation; 0 disables clipping.",
    )
    calibrate_ln_parser.set_defaults(func=command_calibrate_layer_norm)

    run_parser = subparsers.add_parser("run", help="run the whole-forward contract backend")
    run_parser.add_argument("--runtime", required=True, choices=["cpu", "spu"])
    run_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    run_parser.add_argument("--input-pt", default="")
    run_parser.add_argument(
        "--input-share-manifest-json",
        default="",
        help=(
            "Experimental e2e input mode for runtime=spu: load debug additive share files and send "
            "P1/P2 shares separately into the SPU graph. This is not production MPC ingestion yet."
        ),
    )
    run_parser.add_argument(
        "--input-share-public-manifest-json",
        default="",
        help=(
            "Experimental split-share input mode for runtime=spu: public manifest without private share paths. "
            "Must be used with --input-p1-share-manifest-json and --input-p2-share-manifest-json."
        ),
    )
    run_parser.add_argument(
        "--input-p1-share-manifest-json",
        default="",
        help="Experimental split-share input mode: P1 party manifest containing only the P1 share path.",
    )
    run_parser.add_argument(
        "--input-p2-share-manifest-json",
        default="",
        help="Experimental split-share input mode: P2 party manifest containing only the P2 share path.",
    )
    run_parser.add_argument(
        "--party-local-share-load",
        action="store_true",
        help=(
            "Experimental privacy-forward mode for split-share input: P1/P2 load their own share files "
            "inside party devices, so the driver does not torch.load private share tensors."
        ),
    )
    run_parser.add_argument(
        "--redact-private-input-paths",
        action="store_true",
        help=(
            "Do not persist legacy/P1/P2 private share manifest paths into the candidate .pt or summary JSON. "
            "The public manifest path is still recorded for provenance."
        ),
    )
    run_parser.add_argument("--output-pt", required=True)
    run_parser.add_argument("--output-json", required=True)
    run_parser.add_argument(
        "--output-calibration-json",
        default="",
        help=(
            "Optional public post-reveal calibration for e2e approximate logits. "
            "JSON schema: weights=[w0,w1], bias=b, threshold=0.5."
        ),
    )
    run_parser.add_argument("--config", default=str(DEFAULT_SPU_CONFIG))
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument(
        "--cpu-forward-mode",
        choices=["static_no_pruning", "runtime_pruning_reference"],
        default="static_no_pruning",
        help=(
            "CPU-only reference mode. static_no_pruning keeps the existing static whole-forward contract; "
            "runtime_pruning_reference replays DyViT eval-time pruning semantics inside the CPU whole-forward path."
        ),
    )
    run_parser.add_argument(
        "--runtime-pruning-keep-mask-pt",
        default="",
        help=(
            "Optional explicit keep-mask payload exported from "
            "`tools/transshield_e2e_secure_infer.py export-runtime-pruning-keep-mask-payload`. "
            "When set, the runner replays whole-forward using external stage keep masks."
        ),
    )
    run_parser.add_argument("--include-intermediates", action="store_true")
    run_parser.add_argument("--max-samples", type=int, default=0)
    run_parser.add_argument(
        "--static-depth-limit",
        type=int,
        default=-1,
        help="For SPU smoke only: run first N transformer blocks, -1 means all blocks.",
    )
    run_parser.add_argument("--spu-batch-size", type=int, default=1)
    run_parser.add_argument(
        "--spu-block-chunk-size",
        type=int,
        default=0,
        help=(
            "Experimental reveal-less SPU graph split: execute transformer blocks in chunks of this size "
            "and reveal only final logits. 0 keeps the monolithic graph."
        ),
    )
    run_parser.add_argument(
        "--spu-layer-norm-chunk-size",
        type=int,
        default=0,
        help=(
            "Experimental SPU layer-norm feature chunk size. 0 keeps the original reduction graph; "
            "values such as 64 or 128 split mean/variance reductions inside the SPU graph."
        ),
    )
    run_parser.add_argument(
        "--spu-layer-norm-policy",
        choices=["exact", "affine", "public_calibrated"],
        default="exact",
        help=(
            "Experimental SPU layer-norm policy. exact preserves secret mean/variance; "
            "affine skips secret reduction and applies only public affine weights; "
            "public_calibrated uses public calibration stats instead of private sample stats."
        ),
    )
    run_parser.add_argument(
        "--spu-layer-norm-calibration-json",
        default="",
        help="Public calibration JSON required when --spu-layer-norm-policy public_calibrated is used.",
    )
    spu_params_mode_choices = [
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
    ]
    run_parser.add_argument("--spu-params-mode", choices=spu_params_mode_choices, default="public")
    run_parser.add_argument(
        "--spu-attention-policy",
        choices=["smoothed", "standard", "uniform", "identity"],
        default="smoothed",
        help="Experimental SPU attention softmax policy; default preserves the existing smoothed policy.",
    )
    run_parser.add_argument(
        "--spu-activation-override",
        choices=[
            "bundle",
            "gelu",
            "fixed_square",
            "learnable_square",
            "learnable_quadratic",
            "learnable_quadratic_gelu_init",
            "pade_gelu",
            "lut_gelu_4",
            "lut_gelu_8",
            "lut_gelu_16",
            "lut_gelu_32",
        ],
        default="bundle",
        help="Experimental SPU-only MLP activation override for e2e drift ablation; bundle preserves current behavior.",
    )
    run_parser.add_argument(
        "--spu-activation-clip-value",
        type=float,
        default=0.0,
        help="Clip MLP pre-activation to [-value, value] before the SPU-friendly activation; 0 disables clipping.",
    )
    run_parser.add_argument(
        "--spu-token-recycle-scale",
        type=float,
        default=0.0,
        help="Scale factor for Dropped-Token Context Recycling: inject weighted summary of dropped tokens into CLS before masking. 0 disables (original behavior).",
    )
    run_parser.add_argument(
        "--spu-token-ratio-base-override",
        type=float,
        default=0.0,
        help="Override the base_rate for token pruning ratio computation (token_ratio=[r, r^2, r^3]). 0 uses bundle default (0.7 -> 137/96/67 tokens). Lower values = more aggressive pruning = fewer tokens = faster but potentially less accurate.",
    )
    run_parser.set_defaults(func=command_run)

    audit_parser = subparsers.add_parser(
        "audit-input-shares",
        help="debug-only audit for split-share reconstruction and patch embedding inside SPU",
    )
    audit_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    audit_parser.add_argument("--input-pt", default="")
    audit_parser.add_argument("--input-share-manifest-json", default="")
    audit_parser.add_argument("--input-share-public-manifest-json", default="")
    audit_parser.add_argument("--input-p1-share-manifest-json", default="")
    audit_parser.add_argument("--input-p2-share-manifest-json", default="")
    audit_parser.add_argument("--output-json", required=True)
    audit_parser.add_argument("--config", default=str(DEFAULT_SPU_CONFIG))
    audit_parser.add_argument("--max-samples", type=int, default=0)
    audit_parser.add_argument("--spu-params-mode", choices=spu_params_mode_choices, default="public")
    audit_parser.add_argument(
        "--spu-attention-policy",
        choices=["smoothed", "standard"],
        default="smoothed",
        help="Kept for metadata consistency; audit only runs reconstruction and patch embedding.",
    )
    audit_parser.add_argument(
        "--spu-activation-override",
        choices=[
            "bundle",
            "gelu",
            "fixed_square",
            "learnable_square",
            "learnable_quadratic",
            "learnable_quadratic_gelu_init",
            "pade_gelu",
            "lut_gelu_4",
            "lut_gelu_8",
            "lut_gelu_16",
            "lut_gelu_32",
        ],
        default="bundle",
        help="Kept for metadata consistency; audit only runs reconstruction and patch embedding.",
    )
    audit_parser.set_defaults(func=command_audit_input_shares)

    probe_parser = subparsers.add_parser(
        "probe-block",
        help="run explicit block-level CLS debug for CPU/SPU drift attribution",
    )
    probe_parser.add_argument("--runtime", required=True, choices=["cpu", "spu"])
    probe_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    probe_parser.add_argument("--input-pt", default="")
    probe_parser.add_argument("--input-share-manifest-json", default="")
    probe_parser.add_argument("--input-share-public-manifest-json", default="")
    probe_parser.add_argument("--input-p1-share-manifest-json", default="")
    probe_parser.add_argument("--input-p2-share-manifest-json", default="")
    probe_parser.add_argument("--output-json", required=True)
    probe_parser.add_argument("--config", default=str(DEFAULT_SPU_CONFIG))
    probe_parser.add_argument("--device", default="cpu")
    probe_parser.add_argument("--max-samples", type=int, default=0)
    probe_parser.add_argument("--static-depth-limit", type=int, required=True)
    probe_parser.add_argument("--probe-block-index", type=int, required=True)
    probe_parser.add_argument("--spu-batch-size", type=int, default=1)
    probe_parser.add_argument(
        "--spu-layer-norm-chunk-size",
        type=int,
        default=0,
        help="Experimental SPU layer-norm feature chunk size for probe-block.",
    )
    probe_parser.add_argument(
        "--spu-layer-norm-policy",
        choices=["exact", "affine", "public_calibrated"],
        default="exact",
        help="Experimental SPU layer-norm policy for probe-block.",
    )
    probe_parser.add_argument(
        "--spu-layer-norm-calibration-json",
        default="",
        help="Public calibration JSON required when --spu-layer-norm-policy public_calibrated is used.",
    )
    probe_parser.add_argument("--spu-params-mode", choices=spu_params_mode_choices, default="public")
    probe_parser.add_argument(
        "--spu-attention-policy",
        choices=["smoothed", "standard", "uniform", "identity"],
        default="smoothed",
        help="Experimental SPU attention softmax policy; default preserves the existing smoothed policy.",
    )
    probe_parser.add_argument(
        "--spu-activation-override",
        choices=[
            "bundle",
            "gelu",
            "fixed_square",
            "learnable_square",
            "learnable_quadratic",
            "learnable_quadratic_gelu_init",
            "pade_gelu",
            "lut_gelu_4",
            "lut_gelu_8",
            "lut_gelu_16",
            "lut_gelu_32",
        ],
        default="bundle",
        help="Experimental SPU-only MLP activation override for e2e drift ablation; bundle preserves current behavior.",
    )
    probe_parser.add_argument(
        "--spu-activation-clip-value",
        type=float,
        default=0.0,
        help="Optional activation clipping value for SPU probe-block; 0 disables clipping.",
    )
    probe_parser.add_argument(
        "--spu-token-recycle-scale",
        type=float,
        default=0.0,
        help="Scale factor for Dropped-Token Context Recycling during SPU probe-block runs.",
    )
    probe_parser.set_defaults(func=command_probe_block)

    compare_probe_parser = subparsers.add_parser(
        "compare-block-probe",
        help="compare CPU/SPU block-probe JSON outputs",
    )
    compare_probe_parser.add_argument("--reference-json", required=True)
    compare_probe_parser.add_argument("--candidate-json", required=True)
    compare_probe_parser.add_argument("--output-json", required=True)
    compare_probe_parser.set_defaults(func=command_compare_block_probe)

    verify_parser = subparsers.add_parser("verify", help="verify a whole-forward candidate against the plaintext reference")
    verify_parser.add_argument("--reference-pt", required=True)
    verify_parser.add_argument("--candidate-pt", required=True)
    verify_parser.add_argument("--output-json", required=True)
    verify_parser.add_argument("--max-samples", type=int, default=0)
    verify_parser.add_argument("--allow-prefix-candidate", action="store_true")
    verify_parser.set_defaults(func=command_verify)

    subgraph_parser = subparsers.add_parser(
        "block1-subgraph-smoke",
        help="debug-only SPU smoke for first transformer block subgraphs",
    )
    subgraph_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    subgraph_parser.add_argument("--input-pt", required=True)
    subgraph_parser.add_argument("--output-json", required=True)
    subgraph_parser.add_argument("--config", default=str(DEFAULT_SPU_CONFIG))
    subgraph_parser.add_argument("--max-samples", type=int, default=1)
    subgraph_parser.add_argument("--spu-params-mode", choices=["public"], default="public")
    subgraph_parser.add_argument(
        "--layer-norm-chunk-size",
        type=int,
        default=0,
        help="Experimental SPU layer-norm feature chunk size for block1 subgraph smoke.",
    )
    subgraph_parser.add_argument(
        "--layer-norm-policy",
        choices=["exact", "affine"],
        default="exact",
        help="Experimental SPU layer-norm policy for block1 subgraph smoke.",
    )
    subgraph_parser.add_argument(
        "--spu-attention-policy",
        choices=["smoothed", "standard", "uniform", "identity"],
        default="standard",
    )
    subgraph_parser.add_argument(
        "--spu-activation-override",
        choices=[
            "bundle",
            "gelu",
            "fixed_square",
            "learnable_square",
            "learnable_quadratic",
            "learnable_quadratic_gelu_init",
            "pade_gelu",
            "lut_gelu_4",
            "lut_gelu_8",
            "lut_gelu_16",
            "lut_gelu_32",
        ],
        default="bundle",
    )
    subgraph_parser.set_defaults(func=command_block1_subgraph_smoke)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
