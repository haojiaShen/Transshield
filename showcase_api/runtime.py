from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from showcase_api.config import REPO_ROOT, ShowcaseConfig


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def build_share_manifests(run_dir: Path, config: ShowcaseConfig, share0_bytes: bytes, share1_bytes: bytes) -> dict[str, Path]:
    e2e_dir = run_dir / "e2e_secure_live_demo"
    share_dir = e2e_dir / "party_manifests"
    share_dir.mkdir(parents=True, exist_ok=True)

    share0_path = e2e_dir / "share0.float32le"
    share1_path = e2e_dir / "share1.float32le"
    share0_path.write_bytes(share0_bytes)
    share1_path.write_bytes(share1_bytes)

    public_manifest = {
        "manifest_type": "transshield_e2e_debug_float_additive_share_public_manifest_v0",
        "share_count": 2,
        "party_ids": ["P1", "P2"],
        "share_semantics": "debug_float_additive_share_not_production_mpc_share",
        "share_dtype": "torch.float32",
        "share_shape": config.expected_shape,
        "sample_count": 1,
        "sample_ids": ["live_demo_sample_000000"],
        "targets_included": False,
        "source_paths_included": False,
        "private_share_paths_included": False,
        "privacy_status": (
            "browser-generated split shares; the original image is not uploaded, but the centralized "
            "showcase coordinator receives both shares and reconstructs the normalized tensor for DQA"
        ),
    }
    public_manifest_path = e2e_dir / "share_public_manifest.json"
    public_manifest_path.write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    p1_manifest = {
        "manifest_type": "transshield_e2e_debug_float_additive_share_party_manifest_v0",
        "party_id": "P1",
        "share_rank": 0,
        "share_count": 2,
        "share_path": str(share0_path.resolve()),
        "share_storage_format": "raw_float32_le",
        "public_manifest_json": str(public_manifest_path.resolve()),
        "share_semantics": public_manifest["share_semantics"],
        "share_dtype": public_manifest["share_dtype"],
        "share_shape": public_manifest["share_shape"],
        "sample_count": 1,
        "sample_ids": public_manifest["sample_ids"],
        "privacy_status": "single-process demo upload; production should keep this only on its owning party",
    }
    p2_manifest = {
        **p1_manifest,
        "party_id": "P2",
        "share_rank": 1,
        "share_path": str(share1_path.resolve()),
    }
    p1_manifest_path = share_dir / "p1_share_manifest.json"
    p2_manifest_path = share_dir / "p2_share_manifest.json"
    p1_manifest_path.write_text(json.dumps(p1_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p2_manifest_path.write_text(json.dumps(p2_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "e2e_dir": e2e_dir,
        "share0_path": share0_path,
        "share1_path": share1_path,
        "public_manifest_path": public_manifest_path,
        "p1_manifest_path": p1_manifest_path,
        "p2_manifest_path": p2_manifest_path,
        "candidate_pt_path": e2e_dir / "candidate.pt",
        "candidate_json_path": e2e_dir / "candidate.json",
        "runner_log_path": run_dir / "spu_live_demo.log",
    }


def cleanup_private_share_artifacts(artifacts: dict[str, Path]):
    for key in ("share0_path", "share1_path", "public_manifest_path", "p1_manifest_path", "p2_manifest_path"):
        path = artifacts.get(key)
        if path and path.exists():
            path.unlink()
    party_dir = artifacts["e2e_dir"] / "party_manifests"
    if party_dir.exists():
        shutil.rmtree(party_dir, ignore_errors=True)


def run_mock_live_demo(config: ShowcaseConfig) -> dict:
    time.sleep(max(config.accepted_sleep_sec, 0.0))
    probability_class_1 = 0.7124
    probability_class_0 = 1.0 - probability_class_1
    threshold_label = 1 if probability_class_1 >= config.formal_threshold else 0
    return {
        "live_mode": "mock",
        "prediction": {
            "argmax_index": 1,
            "argmax_label": config.class_names[1],
            "threshold_index": threshold_label,
            "threshold_label": config.class_names[threshold_label],
            "prob_class_0": probability_class_0,
            "prob_class_1": probability_class_1,
            "decision_threshold": config.formal_threshold,
        },
        "logits": [0.3128, 0.9074],
        "probabilities": [probability_class_0, probability_class_1],
        "runtime": {
            "actual_elapsed_sec": config.accepted_sleep_sec,
            "formal_reference_sec_per_sample": config.formal_sec_per_sample,
            "formal_reference_dual_total_gib": config.formal_dual_total_gib,
        },
        "artifacts": {
            "candidate_json": None,
            "candidate_pt": None,
            "run_dir": None,
        },
    }


def parse_candidate_result(config: ShowcaseConfig, artifacts: dict[str, Path], elapsed_sec: float) -> dict:
    candidate = json.loads(artifacts["candidate_json_path"].read_text(encoding="utf-8"))
    preview = candidate.get("prediction_preview") or {}
    probabilities = preview.get("probabilities") or []
    logits = preview.get("logits") or []
    if not probabilities or not logits:
        raise RuntimeError("candidate JSON missing prediction_preview probabilities/logits")
    probs = probabilities[0]
    raw_logits = logits[0]
    prob_class_0 = float(probs[0])
    prob_class_1 = float(probs[1])
    argmax_index = 0 if prob_class_0 >= prob_class_1 else 1
    threshold_index = 1 if prob_class_1 >= config.formal_threshold else 0
    return {
        "live_mode": "spu",
        "prediction": {
            "argmax_index": argmax_index,
            "argmax_label": config.class_names[argmax_index],
            "threshold_index": threshold_index,
            "threshold_label": config.class_names[threshold_index],
            "prob_class_0": prob_class_0,
            "prob_class_1": prob_class_1,
            "decision_threshold": config.formal_threshold,
        },
        "logits": [float(raw_logits[0]), float(raw_logits[1])],
        "probabilities": [prob_class_0, prob_class_1],
        "runner_summary": candidate,
        "runtime": {
            "actual_elapsed_sec": elapsed_sec,
            "formal_reference_sec_per_sample": config.formal_sec_per_sample,
            "formal_reference_dual_total_gib": config.formal_dual_total_gib,
            "profile": {
                "static_depth_limit": config.runner_profile.static_depth_limit,
                "spu_params_mode": config.runner_profile.spu_params_mode,
                "spu_layer_norm_policy": config.runner_profile.spu_layer_norm_policy,
                "spu_attention_policy": config.runner_profile.spu_attention_policy,
                "spu_activation_override": config.runner_profile.spu_activation_override,
                "spu_activation_clip_value": config.runner_profile.spu_activation_clip_value,
                "spu_secure_pruning_mode": config.runner_profile.spu_secure_pruning_mode,
                "spu_secure_pruning_network": config.runner_profile.spu_secure_pruning_network,
                "spu_token_ratio_base_override": (
                    config.runner_profile.spu_token_ratio_base_override
                ),
                "spu_final_block_cls_only": config.runner_profile.spu_final_block_cls_only,
                "spu_uniform_attention_value_fusion": (
                    config.runner_profile.spu_uniform_attention_value_fusion
                ),
            },
        },
        "artifacts": {
            "candidate_json": relative_to_repo(artifacts["candidate_json_path"]),
            "candidate_pt": relative_to_repo(artifacts["candidate_pt_path"]),
            "run_dir": relative_to_repo(artifacts["run_dir"]),
            "runner_log": relative_to_repo(artifacts["runner_log_path"]),
        },
    }


def run_spu_live_demo(config: ShowcaseConfig, share0_bytes: bytes, share1_bytes: bytes) -> dict:
    run_id = uuid.uuid4().hex
    run_dir = config.run_dir / f"medical_live_demo_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_share_manifests(run_dir, config, share0_bytes, share1_bytes)
    command = [
        config.python_bin,
        str(REPO_ROOT / "integrations" / "transshield_runtime" / "e2e_secure_vit" / "transshield_e2e_secure_vit.py"),
        "run",
        "--runtime",
        "spu",
        "--bundle-dir",
        str(config.bundle_dir),
        "--input-share-public-manifest-json",
        str(artifacts["public_manifest_path"]),
        "--input-p1-share-manifest-json",
        str(artifacts["p1_manifest_path"]),
        "--input-p2-share-manifest-json",
        str(artifacts["p2_manifest_path"]),
        "--party-local-share-load",
        "--redact-private-input-paths",
        "--output-pt",
        str(artifacts["candidate_pt_path"]),
        "--output-json",
        str(artifacts["candidate_json_path"]),
        "--config",
        str(config.spu_config_path),
        "--device",
        "cpu",
        "--max-samples",
        "1",
        "--static-depth-limit",
        str(config.runner_profile.static_depth_limit),
        "--spu-batch-size",
        str(config.runner_profile.spu_batch_size),
        "--spu-params-mode",
        config.runner_profile.spu_params_mode,
        "--spu-layer-norm-policy",
        config.runner_profile.spu_layer_norm_policy,
        "--spu-attention-policy",
        config.runner_profile.spu_attention_policy,
        "--spu-activation-override",
        config.runner_profile.spu_activation_override,
        "--spu-activation-clip-value",
        str(config.runner_profile.spu_activation_clip_value),
        "--spu-secure-pruning-mode",
        config.runner_profile.spu_secure_pruning_mode,
        "--spu-secure-pruning-network",
        config.runner_profile.spu_secure_pruning_network,
        "--token-ratio-base-override",
        str(config.runner_profile.spu_token_ratio_base_override),
        "--spu-compile-cache-dir",
        str(config.runner_profile.spu_compile_cache_dir),
    ]
    if config.runner_profile.spu_final_block_cls_only:
        command.append("--spu-final-block-cls-only")
    if config.runner_profile.spu_uniform_attention_value_fusion:
        command.append("--spu-uniform-attention-value-fusion")
    started = time.perf_counter()
    try:
        with artifacts["runner_log_path"].open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError("SPU live demo runner failed; inspect the server-side runner log.")
        elapsed = time.perf_counter() - started
        result = parse_candidate_result(config, artifacts, elapsed)
        redacted_summary_path = run_dir / "live_run_response.json"
        redacted_summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        cleanup_private_share_artifacts(artifacts)


def run_live_demo(config: ShowcaseConfig, share0_bytes: bytes, share1_bytes: bytes) -> dict:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.runtime_mode == "mock":
        return run_mock_live_demo(config)
    return run_spu_live_demo(config, share0_bytes, share1_bytes)
