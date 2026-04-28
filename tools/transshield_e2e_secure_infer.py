import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts/frozen_bundle_verified_tracka_lr3e5_20260414"
MANIFEST_TYPE = "transshield_e2e_client_pixel_values_v0"
CONTRACT_TYPE = "transshield_e2e_secure_inference_contract_v0"
DEBUG_SHARE_TYPE = "transshield_e2e_debug_float_additive_share_v0"
DEBUG_SHARE_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_manifest_v0"
DEBUG_SHARE_PUBLIC_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_public_manifest_v0"
DEBUG_SHARE_PARTY_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_party_manifest_v0"


def ensure_repo_import_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_bundle_dir(raw_bundle_dir: str) -> Path:
    return Path(raw_bundle_dir).expanduser().resolve()


def tensor_stats(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def build_contract_payload(bundle_dir: Path):
    args_snapshot_path = bundle_dir / "args_snapshot.json"
    threshold_path = bundle_dir / "threshold_best.json"
    args_snapshot = load_json(args_snapshot_path)
    threshold_payload = load_json(threshold_path) if threshold_path.exists() else {}
    return {
        "manifest_type": CONTRACT_TYPE,
        "bundle_dir": str(bundle_dir),
        "status": "design_contract_only",
        "current_default_pipeline": "secure_sidecar_replay_compare",
        "e2e_track_status": "new_parallel_research_track",
        "model_scope": {
            "first_poc": "static_deit_s_or_vit_without_pruning",
            "later_scope": "dyvit_masking_pruning_inside_secure_forward",
            "current_bundle_model": args_snapshot.get("model"),
            "input_size": args_snapshot.get("input_size"),
            "num_classes": args_snapshot.get("nb_classes"),
            "use_mask_pruning": bool(args_snapshot.get("use_mask_pruning", False)),
        },
        "privacy_boundary": {
            "client_side": [
                "read_original_image",
                "run_public_eval_preprocess",
                "create secret shares for pixel_values",
            ],
            "secure_runtime": [
                "consume secret-shared pixel_values",
                "consume secret-shared or protected model parameters",
                "run patch_embed/blocks/head in secure forward",
            ],
            "reveal_policy": [
                "default reveal only final logits or final label",
                "do not reveal intermediate features",
                "do not reveal masks, thresholds, active-token counts, or patch embeddings outside explicit debug mode",
            ],
        },
        "non_goals_for_current_step": [
            "does not replace the current Web demo secure sidecar path",
            "does not claim current Transshield is already end-to-end encrypted X-ray inference",
            "does not directly copy OpenBumbleBee flax_vit as the final implementation",
        ],
        "threshold": {
            "threshold_json": str(threshold_path) if threshold_path.exists() else None,
            "eval_binary_threshold": threshold_payload.get("eval_binary_threshold"),
        },
    }


def command_contract(args):
    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    payload = build_contract_payload(bundle_dir)
    write_json(Path(args.output_json).resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def resolve_selected_paths(args):
    ensure_repo_import_path()
    from tools.transshield_input_selection import resolve_selected_sample_paths

    return resolve_selected_sample_paths(
        data_path=args.data_path,
        image_paths=args.image,
        image_list=args.image_list,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        max_samples=args.max_samples,
    )


def preprocess_selected_images(bundle_dir: Path, selection):
    ensure_repo_import_path()
    from PIL import Image
    from tools.transshield_stage2_bundle import build_eval_transform_from_args_snapshot

    args_snapshot = load_json(bundle_dir / "args_snapshot.json")
    transform = build_eval_transform_from_args_snapshot(args_snapshot)

    tensors = []
    for sample_path in selection["sample_paths"]:
        image = Image.open(sample_path).convert("RGB")
        tensors.append(transform(image))

    import torch

    pixel_values = torch.stack(tensors, dim=0).contiguous()
    return args_snapshot, pixel_values


def save_debug_additive_shares(
    pixel_values,
    output_prefix: Path,
    share_seed: int,
    *,
    sample_ids=None,
    targets=None,
    bundle_dir=None,
    output_manifest_json=None,
):
    import torch

    target_values = None if targets is None else [int(value) for value in targets.detach().cpu().tolist()]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(share_seed))
    share0 = torch.empty_like(pixel_values).uniform_(-2.0, 2.0, generator=generator)
    share1 = pixel_values - share0
    reconstruct_error = (share0 + share1 - pixel_values).abs().max().item()
    share_paths = [
        output_prefix.with_name(output_prefix.name + "_p1_share.pt"),
        output_prefix.with_name(output_prefix.name + "_p2_share.pt"),
    ]
    for rank, (share_path, share_tensor) in enumerate(zip(share_paths, [share0, share1])):
        share_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "manifest_type": DEBUG_SHARE_TYPE,
                "share_rank": rank,
                "share_count": 2,
                "share_semantics": "debug_float_additive_share_not_production_mpc_share",
                "share_tensor": share_tensor,
                "share_seed": int(share_seed) if rank == 0 else None,
                "source_manifest_type": MANIFEST_TYPE,
                "sample_ids": sample_ids,
                "targets": targets,
            },
            share_path,
        )
    manifest = {
        "manifest_type": DEBUG_SHARE_MANIFEST_TYPE,
        "bundle_dir": None if bundle_dir is None else str(bundle_dir),
        "share_paths": [str(path) for path in share_paths],
        "share_count": len(share_paths),
        "share_semantics": "debug_float_additive_share_not_production_mpc_share",
        "share_dtype": str(pixel_values.dtype),
        "share_shape": list(pixel_values.shape),
        "sample_count": int(pixel_values.shape[0]),
        "sample_ids": sample_ids,
        "targets_included": targets is not None,
        "targets": target_values,
        "max_reconstruct_abs_error": float(reconstruct_error),
        "privacy_status": (
            "debug_additive_shares_only_not_production_mpc; each share file must be held by a different party"
        ),
        "server_boundary_note": (
            "A production e2e path should provide each share to its owning party and must not reconstruct "
            "plaintext pixel_values on the server host."
        ),
    }
    if output_manifest_json is not None:
        write_json(output_manifest_json, manifest)
    return manifest


def build_debug_share_public_manifest(manifest):
    return {
        "manifest_type": DEBUG_SHARE_PUBLIC_MANIFEST_TYPE,
        "bundle_dir": manifest.get("bundle_dir"),
        "share_count": manifest.get("share_count"),
        "share_semantics": manifest.get("share_semantics"),
        "share_dtype": manifest.get("share_dtype"),
        "share_shape": manifest.get("share_shape"),
        "sample_count": manifest.get("sample_count"),
        "sample_ids": manifest.get("sample_ids"),
        "targets_included": bool(manifest.get("targets_included", False)),
        "selection_mode": manifest.get("selection_mode"),
        "input_size": manifest.get("input_size"),
        "imagenet_default_mean_and_std": manifest.get("imagenet_default_mean_and_std"),
        "crop_pct": manifest.get("crop_pct"),
        "source_paths_included": bool(manifest.get("source_paths_included", False)),
        "private_share_paths_included": False,
        "party_ids": ["P1", "P2"],
        "privacy_status": (
            "debug_public_share_manifest_only; private share file paths are intentionally kept in party manifests"
        ),
        "server_boundary_note": (
            "The server-side coordinator may read this public manifest, but each private share path should be "
            "provided only to its owning party in a production deployment."
        ),
    }


def build_debug_share_party_manifests(manifest, public_manifest_json: Path):
    share_paths = manifest.get("share_paths") or []
    if len(share_paths) != 2:
        raise ValueError(f"expected exactly 2 debug share paths, got {len(share_paths)}")
    party_ids = ["P1", "P2"]
    party_manifests = []
    for rank, (party_id, share_path) in enumerate(zip(party_ids, share_paths)):
        party_manifests.append(
            {
                "manifest_type": DEBUG_SHARE_PARTY_MANIFEST_TYPE,
                "party_id": party_id,
                "share_rank": rank,
                "share_count": 2,
                "share_path": str(share_path),
                "public_manifest_json": str(public_manifest_json),
                "share_semantics": manifest.get("share_semantics"),
                "share_dtype": manifest.get("share_dtype"),
                "share_shape": manifest.get("share_shape"),
                "sample_count": manifest.get("sample_count"),
                "sample_ids": manifest.get("sample_ids"),
                "privacy_status": (
                    "debug_party_share_manifest_only; this manifest must not include the other party share path"
                ),
            }
        )
    return party_manifests


def write_debug_share_public_and_party_manifests(
    manifest,
    public_manifest_json: Path,
    party_manifest_dir: Path,
):
    public_manifest_json = public_manifest_json.expanduser().resolve()
    party_manifest_dir = party_manifest_dir.expanduser().resolve()
    public_manifest = build_debug_share_public_manifest(manifest)
    party_manifests = build_debug_share_party_manifests(manifest, public_manifest_json)
    write_json(public_manifest_json, public_manifest)
    party_manifest_dir.mkdir(parents=True, exist_ok=True)
    party_manifest_paths = []
    for party_manifest in party_manifests:
        party_path = party_manifest_dir / f"{party_manifest['party_id'].lower()}_share_manifest.json"
        write_json(party_path, party_manifest)
        party_manifest_paths.append(str(party_path))
    return {
        "public_manifest_json": str(public_manifest_json),
        "party_manifest_jsons": party_manifest_paths,
    }


def command_client_share_preprocess(args):
    import torch

    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    selection = resolve_selected_paths(args)
    args_snapshot, pixel_values = preprocess_selected_images(bundle_dir, selection)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    sample_count = int(pixel_values.shape[0])
    sample_ids = [f"sample_{index:06d}" for index in range(sample_count)]

    targets = selection.get("targets") if args.include_targets else None
    target_tensor = None
    if targets is not None and all(target is not None for target in targets):
        target_tensor = torch.tensor(targets, dtype=torch.long)

    manifest = save_debug_additive_shares(
        pixel_values,
        output_prefix,
        args.share_seed,
        sample_ids=sample_ids,
        targets=target_tensor,
        bundle_dir=bundle_dir,
        output_manifest_json=output_json,
    )
    manifest["selection_mode"] = selection["selection_mode"]
    manifest["input_size"] = int(args_snapshot["input_size"])
    manifest["imagenet_default_mean_and_std"] = bool(args_snapshot["imagenet_default_mean_and_std"])
    manifest["crop_pct"] = args_snapshot.get("crop_pct")
    manifest["plaintext_pixel_package_written"] = False
    manifest["source_paths_included"] = bool(args.include_source_paths)
    if args.include_source_paths:
        manifest["source_paths"] = selection["sample_paths"]
    if args.output_public_json and args.output_party_manifest_dir:
        manifest["split_public_and_party_manifests"] = write_debug_share_public_and_party_manifests(
            manifest,
            Path(args.output_public_json),
            Path(args.output_party_manifest_dir),
        )
    elif args.output_public_json or args.output_party_manifest_dir:
        raise ValueError("--output-public-json and --output-party-manifest-dir must be set together")
    write_json(output_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_debug_share_manifest(manifest_json: Path):
    manifest = load_json(manifest_json)
    if manifest.get("manifest_type") != DEBUG_SHARE_MANIFEST_TYPE:
        raise ValueError(f"unsupported share manifest type: {manifest.get('manifest_type')}")
    share_paths = manifest.get("share_paths") or []
    if len(share_paths) != 2:
        raise ValueError(f"expected exactly 2 debug share paths, got {len(share_paths)}")
    return manifest


def load_debug_share_public_manifest(manifest_json: Path):
    manifest = load_json(manifest_json)
    if manifest.get("manifest_type") != DEBUG_SHARE_PUBLIC_MANIFEST_TYPE:
        raise ValueError(f"unsupported public share manifest type: {manifest.get('manifest_type')}")
    if manifest.get("share_count") != 2:
        raise ValueError(f"expected share_count=2 in public manifest, got {manifest.get('share_count')}")
    if manifest.get("private_share_paths_included") is not False:
        raise ValueError("public share manifest must not include private share paths")
    return manifest


def load_debug_share_party_manifest(manifest_json: Path):
    manifest = load_json(manifest_json)
    if manifest.get("manifest_type") != DEBUG_SHARE_PARTY_MANIFEST_TYPE:
        raise ValueError(f"unsupported party share manifest type: {manifest.get('manifest_type')}")
    if manifest.get("share_count") != 2:
        raise ValueError(f"expected share_count=2 in party manifest, got {manifest.get('share_count')}")
    if manifest.get("share_rank") not in (0, 1):
        raise ValueError(f"unsupported share_rank in party manifest: {manifest.get('share_rank')}")
    if not manifest.get("share_path"):
        raise ValueError(f"missing share_path in party manifest: {manifest_json}")
    return manifest


def command_split_debug_share_manifest(args):
    manifest_json = Path(args.share_manifest_json).expanduser().resolve()
    manifest = load_debug_share_manifest(manifest_json)
    output = write_debug_share_public_and_party_manifests(
        manifest,
        Path(args.output_public_json),
        Path(args.output_party_manifest_dir),
    )
    summary = {
        "manifest_type": DEBUG_SHARE_MANIFEST_TYPE + "_split_summary",
        "source_share_manifest_json": str(manifest_json),
        **output,
        "privacy_status": (
            "debug split only; production still needs separate party processes so the coordinator cannot see both shares"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_reconstruct_debug_shares(args):
    import torch

    manifest_json = Path(args.share_manifest_json).expanduser().resolve()
    output_pt = Path(args.output_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    manifest = load_debug_share_manifest(manifest_json)

    share_payloads = []
    for raw_path in manifest["share_paths"]:
        share_path = Path(raw_path).expanduser().resolve()
        payload = torch.load(share_path, map_location="cpu")
        if payload.get("manifest_type") != DEBUG_SHARE_TYPE:
            raise ValueError(f"unsupported share payload type in {share_path}: {payload.get('manifest_type')}")
        share_payloads.append(payload)
    share_payloads = sorted(share_payloads, key=lambda item: int(item["share_rank"]))
    pixel_values = share_payloads[0]["share_tensor"] + share_payloads[1]["share_tensor"]

    target_tensor = share_payloads[0].get("targets")
    payload = {
        "manifest_type": MANIFEST_TYPE,
        "bundle_dir": manifest.get("bundle_dir"),
        "client_boundary": "debug_reconstructed_from_additive_shares_for_current_backend",
        "pixel_values": pixel_values,
        "sample_ids": manifest.get("sample_ids"),
        "targets": target_tensor,
        "metadata": {
            "source_share_manifest_json": str(manifest_json),
            "sample_count": int(pixel_values.shape[0]),
            "targets_included": target_tensor is not None,
            "privacy_note": (
                "This file reconstructs plaintext pixel_values for current local/SPU backend compatibility. "
                "It is a debug bridge and must not be used as the final all-private server path."
            ),
        },
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    summary = {
        "manifest_type": MANIFEST_TYPE + "_debug_reconstruct_summary",
        "share_manifest_json": str(manifest_json),
        "output_pt": str(output_pt),
        "pixel_values": tensor_stats(pixel_values),
        "sample_count": int(pixel_values.shape[0]),
        "privacy_status": "plaintext_reconstructed_debug_bridge_not_production",
        "next_secure_step": "replace host reconstruction with per-party share ingestion by the SPU runtime",
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_client_preprocess(args):
    import torch

    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    selection = resolve_selected_paths(args)
    args_snapshot, pixel_values = preprocess_selected_images(bundle_dir, selection)
    output_pt = Path(args.output_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    include_source_paths = bool(args.include_source_paths)
    include_targets = bool(args.include_targets)
    sample_count = int(pixel_values.shape[0])

    sample_ids = [f"sample_{index:06d}" for index in range(sample_count)]
    targets = selection.get("targets") if include_targets else None
    target_tensor = None
    if targets is not None and all(target is not None for target in targets):
        target_tensor = torch.tensor(targets, dtype=torch.long)

    payload = {
        "manifest_type": MANIFEST_TYPE,
        "bundle_dir": str(bundle_dir),
        "client_boundary": "plaintext_preprocess_output_before_secret_sharing",
        "pixel_values": pixel_values,
        "sample_ids": sample_ids,
        "targets": target_tensor,
        "metadata": {
            "selection_mode": selection["selection_mode"],
            "sample_count": sample_count,
            "source_paths": selection["sample_paths"] if include_source_paths else None,
            "targets_included": target_tensor is not None,
            "input_size": int(args_snapshot["input_size"]),
            "imagenet_default_mean_and_std": bool(args_snapshot["imagenet_default_mean_and_std"]),
            "crop_pct": args_snapshot.get("crop_pct"),
            "privacy_note": (
                "This package still contains plaintext pixel_values. "
                "A real e2e secure deployment must create MPC/SPU shares on the client side "
                "and avoid sending this plaintext package to an untrusted server."
            ),
        },
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)

    summary = {
        "manifest_type": MANIFEST_TYPE + "_summary",
        "output_pt": str(output_pt),
        "bundle_dir": str(bundle_dir),
        "selection_mode": selection["selection_mode"],
        "sample_count": sample_count,
        "sample_ids": sample_ids,
        "source_paths_included": include_source_paths,
        "targets_included": target_tensor is not None,
        "pixel_values": tensor_stats(pixel_values),
        "privacy_status": "plaintext_client_tensor_not_yet_mpc_share",
        "next_secure_step": "replace plaintext package transport with real client-side secret sharing",
    }
    if args.debug_share_prefix:
        summary["debug_additive_shares"] = save_debug_additive_shares(
            pixel_values,
            Path(args.debug_share_prefix).expanduser().resolve(),
            args.share_seed,
            sample_ids=sample_ids,
            targets=target_tensor,
            bundle_dir=bundle_dir,
        )
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_client_pixel_package(input_pt: Path):
    import torch

    payload = torch.load(input_pt, map_location="cpu")
    if payload.get("manifest_type") != MANIFEST_TYPE:
        raise ValueError(f"unsupported input manifest type: {payload.get('manifest_type')}")
    if "pixel_values" not in payload:
        raise ValueError(f"missing pixel_values in {input_pt}")
    return payload


def command_plaintext_reference(args):
    import torch

    ensure_repo_import_path()
    from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold

    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    input_pt = Path(args.input_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    client_payload = load_client_pixel_package(input_pt)
    pixel_values = client_payload["pixel_values"].to(args.device)
    bundle = load_frozen_bundle(bundle_dir, device=args.device)
    model = bundle["model"]
    threshold = resolve_threshold(bundle_dir, None)

    with torch.no_grad():
        logits = model(pixel_values)
        probabilities = torch.softmax(logits, dim=-1)

    argmax_predictions = logits.argmax(dim=1).detach().cpu()
    threshold_predictions = None
    if threshold is not None and probabilities.shape[-1] == 2:
        threshold_predictions = (probabilities[:, 1] >= float(threshold)).long().detach().cpu()

    targets = client_payload.get("targets")
    if targets is not None:
        targets = targets.detach().cpu()

    per_sample = []
    sample_ids = client_payload.get("sample_ids") or [f"sample_{index:06d}" for index in range(logits.shape[0])]
    logits_cpu = logits.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu()
    for index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            "target": None if targets is None else int(targets[index].item()),
            "logits": [float(value) for value in logits_cpu[index].tolist()],
            "probabilities": [float(value) for value in probabilities_cpu[index].tolist()],
            "argmax_prediction": int(argmax_predictions[index].item()),
        }
        if threshold_predictions is not None:
            row["threshold_prediction"] = int(threshold_predictions[index].item())
        per_sample.append(row)

    summary = {
        "manifest_type": "transshield_e2e_plaintext_reference_v0",
        "status": "plaintext_reference_not_secure",
        "bundle_dir": str(bundle_dir),
        "input_pt": str(input_pt),
        "device": args.device,
        "sample_count": len(per_sample),
        "threshold": threshold,
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "logits": tensor_stats(logits_cpu),
        "privacy_note": (
            "This is only a plaintext reference for e2e checker development. "
            "It is not an MPC/SPU secure execution result."
        ),
        "per_sample": per_sample,
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_static_student_whole_forward(model, pixel_values):
    import torch

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

    for block in model.blocks:
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
    }


def command_static_whole_forward_reference(args):
    import torch

    ensure_repo_import_path()
    from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold

    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    input_pt = Path(args.input_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_pt = Path(args.output_pt).expanduser().resolve() if args.output_pt else None

    client_payload = load_client_pixel_package(input_pt)
    pixel_values = client_payload["pixel_values"].to(args.device)
    bundle = load_frozen_bundle(bundle_dir, device=args.device)
    model = bundle["model"]
    threshold = resolve_threshold(bundle_dir, None)

    with torch.no_grad():
        outputs = run_static_student_whole_forward(model, pixel_values)
        logits = outputs["logits"]
        cls_features = outputs["cls_features"]
        token_features = outputs["token_features"]
        probabilities = torch.softmax(logits, dim=-1)

    argmax_predictions = logits.argmax(dim=1).detach().cpu()
    threshold_predictions = None
    if threshold is not None and probabilities.shape[-1] == 2:
        threshold_predictions = (probabilities[:, 1] >= float(threshold)).long().detach().cpu()

    targets = client_payload.get("targets")
    if targets is not None:
        targets = targets.detach().cpu()

    sample_ids = client_payload.get("sample_ids") or [f"sample_{index:06d}" for index in range(logits.shape[0])]
    logits_cpu = logits.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu()
    cls_features_cpu = cls_features.detach().cpu()
    token_features_cpu = token_features.detach().cpu()

    per_sample = []
    for index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            "target": None if targets is None else int(targets[index].item()),
            "logits": [float(value) for value in logits_cpu[index].tolist()],
            "probabilities": [float(value) for value in probabilities_cpu[index].tolist()],
            "argmax_prediction": int(argmax_predictions[index].item()),
        }
        if threshold_predictions is not None:
            row["threshold_prediction"] = int(threshold_predictions[index].item())
        per_sample.append(row)

    if output_pt is not None:
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "manifest_type": "transshield_e2e_static_whole_forward_reference_pt_v0",
                "input_pt": str(input_pt),
                "bundle_dir": str(bundle_dir),
                "sample_ids": sample_ids,
                "targets": targets,
                "logits": logits_cpu,
                "probabilities": probabilities_cpu,
                "cls_features": cls_features_cpu,
                "token_features": token_features_cpu,
                "threshold": threshold,
                "threshold_predictions": threshold_predictions,
                "argmax_predictions": argmax_predictions,
            },
            output_pt,
        )

    summary = {
        "manifest_type": "transshield_e2e_static_whole_forward_reference_v0",
        "status": "plaintext_static_whole_forward_reference_not_secure",
        "bundle_dir": str(bundle_dir),
        "input_pt": str(input_pt),
        "output_pt": str(output_pt) if output_pt is not None else None,
        "device": args.device,
        "sample_count": len(per_sample),
        "threshold": threshold,
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "forward_scope": (
            "student_patch_embed_blocks_head_without_runtime_pruning_predictor_path"
        ),
        "logits": tensor_stats(logits_cpu),
        "cls_features": tensor_stats(cls_features_cpu),
        "token_features": tensor_stats(token_features_cpu),
        "privacy_note": (
            "This is a plaintext static whole-forward reference for the future e2e SPU path. "
            "It keeps the student blocks/head but bypasses runtime pruning decisions."
        ),
        "per_sample": per_sample,
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_tensor_payload(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"expected dict payload in {path}")
    return payload


def compare_prediction_match(lhs, rhs):
    import torch

    if lhs is None or rhs is None:
        return None
    lhs = lhs.detach().cpu().long()
    rhs = rhs.detach().cpu().long()
    if lhs.shape != rhs.shape:
        raise ValueError(f"prediction shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    return float((lhs == rhs).float().mean().item())


def command_compare_static_whole_forward(args):
    import torch

    reference_pt = Path(args.reference_pt).expanduser().resolve()
    candidate_pt = Path(args.candidate_pt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    reference = load_tensor_payload(reference_pt)
    candidate = load_tensor_payload(candidate_pt)

    reference_logits = reference.get("logits")
    candidate_logits = candidate.get("logits")
    if reference_logits is None or candidate_logits is None:
        raise ValueError("both reference and candidate payloads must contain logits")

    reference_logits = reference_logits.detach().cpu().float()
    candidate_logits = candidate_logits.detach().cpu().float()
    if tuple(reference_logits.shape) != tuple(candidate_logits.shape):
        raise ValueError(
            f"logits shape mismatch: {tuple(reference_logits.shape)} vs {tuple(candidate_logits.shape)}"
        )

    reference_sample_ids = reference.get("sample_ids")
    candidate_sample_ids = candidate.get("sample_ids")
    if reference_sample_ids is not None and candidate_sample_ids is not None:
        if list(reference_sample_ids) != list(candidate_sample_ids):
            raise ValueError("sample_ids mismatch between reference and candidate")

    reference_probabilities = reference.get("probabilities")
    candidate_probabilities = candidate.get("probabilities")
    if reference_probabilities is None:
        reference_probabilities = torch.softmax(reference_logits, dim=-1)
    else:
        reference_probabilities = reference_probabilities.detach().cpu().float()
    if candidate_probabilities is None:
        candidate_probabilities = torch.softmax(candidate_logits, dim=-1)
    else:
        candidate_probabilities = candidate_probabilities.detach().cpu().float()

    reference_argmax = reference.get("argmax_predictions")
    if reference_argmax is None:
        reference_argmax = reference_logits.argmax(dim=1)
    candidate_argmax = candidate.get("argmax_predictions")
    if candidate_argmax is None:
        candidate_argmax = candidate_logits.argmax(dim=1)

    reference_threshold = reference.get("threshold")
    candidate_threshold = candidate.get("threshold")
    if candidate_threshold is None:
        candidate_threshold = reference_threshold

    reference_threshold_predictions = reference.get("threshold_predictions")
    if reference_threshold_predictions is None and reference_threshold is not None and reference_probabilities.shape[-1] == 2:
        reference_threshold_predictions = (reference_probabilities[:, 1] >= float(reference_threshold)).long()
    candidate_threshold_predictions = candidate.get("threshold_predictions")
    if candidate_threshold_predictions is None and candidate_threshold is not None and candidate_probabilities.shape[-1] == 2:
        candidate_threshold_predictions = (candidate_probabilities[:, 1] >= float(candidate_threshold)).long()

    logits_abs_error = (reference_logits - candidate_logits).abs()
    probabilities_abs_error = (reference_probabilities - candidate_probabilities).abs()

    summary = {
        "manifest_type": "transshield_e2e_static_whole_forward_compare_v0",
        "reference_pt": str(reference_pt),
        "candidate_pt": str(candidate_pt),
        "sample_count": int(reference_logits.shape[0]),
        "logits_shape": list(reference_logits.shape),
        "candidate_manifest_type": candidate.get("manifest_type"),
        "reference_manifest_type": reference.get("manifest_type"),
        "logits_error": {
            "max_abs_error": float(logits_abs_error.max().item()),
            "mean_abs_error": float(logits_abs_error.mean().item()),
        },
        "probabilities_error": {
            "max_abs_error": float(probabilities_abs_error.max().item()),
            "mean_abs_error": float(probabilities_abs_error.mean().item()),
        },
        "prediction_match": {
            "argmax_match_ratio": compare_prediction_match(reference_argmax, candidate_argmax),
            "threshold_match_ratio": compare_prediction_match(
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


def command_spu_plan(args):
    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    payload = build_contract_payload(bundle_dir)
    payload["status"] = "spu_static_whole_forward_backend_scaffolded_server_validation_pending"
    payload["next_implementation_steps"] = [
        "run the minimal static whole-forward SPU smoke with --max-samples 1 and --spu-batch-size 1",
        "verify the SPU candidate against static_whole_forward_reference with --allow-prefix-candidate",
        "if the smoke passes, increase sample count gradually before attempting full-val",
        "then merge dyvit masking-pruning semantics into the secure forward after static whole-forward is stable",
    ]
    payload["current_spu_entry"] = (
        "integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py run --runtime spu"
    )
    payload["current_spu_scope"] = (
        "experimental static deit-s whole-forward JAX/SPU backend; bypasses runtime pruning predictors"
    )
    payload["current_limitations"] = [
        "server validation is still pending",
        "current POC still loads plaintext client pixel_values on the host before SPU secret sharing",
        "first smoke defaults to public model parameters; use --spu-params-mode secret only after public smoke is stable",
        "intermediate features are not revealed in SPU mode",
        "dynamic masking-pruning inside secure forward is not implemented yet",
    ]
    if args.output_json:
        write_json(Path(args.output_json).resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def add_input_selection_args(parser):
    parser.add_argument("--data-path", default="")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-list", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--glob-pattern", default="*")
    parser.add_argument("--max-samples", type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Early e2e secure inference scaffolding for Transshield. "
            "Current commands define the privacy boundary and plaintext reference; "
            "they do not replace the secure sidecar default path."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="write the e2e privacy-boundary contract")
    contract_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    contract_parser.add_argument("--output-json", required=True)
    contract_parser.set_defaults(func=command_contract)

    preprocess_parser = subparsers.add_parser(
        "client-preprocess",
        help="simulate client-side image preprocessing into pixel_values for the first e2e POC",
    )
    preprocess_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    add_input_selection_args(preprocess_parser)
    preprocess_parser.add_argument("--output-pt", required=True)
    preprocess_parser.add_argument("--output-json", required=True)
    preprocess_parser.add_argument("--include-source-paths", action="store_true")
    preprocess_parser.add_argument("--include-targets", action="store_true")
    preprocess_parser.add_argument("--debug-share-prefix", default="")
    preprocess_parser.add_argument("--share-seed", type=int, default=0)
    preprocess_parser.set_defaults(func=command_client_preprocess)

    share_preprocess_parser = subparsers.add_parser(
        "client-share-preprocess",
        help="simulate client-side preprocessing directly into debug additive share files without writing plaintext pixels",
    )
    share_preprocess_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    add_input_selection_args(share_preprocess_parser)
    share_preprocess_parser.add_argument("--output-prefix", required=True)
    share_preprocess_parser.add_argument("--output-json", required=True)
    share_preprocess_parser.add_argument(
        "--output-public-json",
        default="",
        help="Optional public manifest without private share paths; must be used with --output-party-manifest-dir.",
    )
    share_preprocess_parser.add_argument(
        "--output-party-manifest-dir",
        default="",
        help="Optional directory for P1/P2 party manifests; must be used with --output-public-json.",
    )
    share_preprocess_parser.add_argument("--include-source-paths", action="store_true")
    share_preprocess_parser.add_argument("--include-targets", action="store_true")
    share_preprocess_parser.add_argument("--share-seed", type=int, default=0)
    share_preprocess_parser.set_defaults(func=command_client_share_preprocess)

    split_share_parser = subparsers.add_parser(
        "split-debug-share-manifest",
        help="write public plus per-party manifests from an existing legacy debug share manifest",
    )
    split_share_parser.add_argument("--share-manifest-json", required=True)
    split_share_parser.add_argument("--output-public-json", required=True)
    split_share_parser.add_argument("--output-party-manifest-dir", required=True)
    split_share_parser.set_defaults(func=command_split_debug_share_manifest)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct-debug-shares",
        help="debug bridge: reconstruct a plaintext client pixel package from additive shares for current backend compatibility",
    )
    reconstruct_parser.add_argument("--share-manifest-json", required=True)
    reconstruct_parser.add_argument("--output-pt", required=True)
    reconstruct_parser.add_argument("--output-json", required=True)
    reconstruct_parser.set_defaults(func=command_reconstruct_debug_shares)

    reference_parser = subparsers.add_parser(
        "plaintext-reference",
        help="run plaintext reference logits from a client pixel package",
    )
    reference_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    reference_parser.add_argument("--input-pt", required=True)
    reference_parser.add_argument("--device", default="cpu")
    reference_parser.add_argument("--output-json", required=True)
    reference_parser.set_defaults(func=command_plaintext_reference)

    static_reference_parser = subparsers.add_parser(
        "static-whole-forward-reference",
        help="run a plaintext static whole-forward reference using student blocks/head without runtime pruning",
    )
    static_reference_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    static_reference_parser.add_argument("--input-pt", required=True)
    static_reference_parser.add_argument("--device", default="cpu")
    static_reference_parser.add_argument("--output-json", required=True)
    static_reference_parser.add_argument("--output-pt", default="")
    static_reference_parser.set_defaults(func=command_static_whole_forward_reference)

    compare_static_parser = subparsers.add_parser(
        "compare-static-whole-forward",
        help="compare a future secure/static-whole-forward candidate against the plaintext reference tensors",
    )
    compare_static_parser.add_argument("--reference-pt", required=True)
    compare_static_parser.add_argument("--candidate-pt", required=True)
    compare_static_parser.add_argument("--output-json", required=True)
    compare_static_parser.set_defaults(func=command_compare_static_whole_forward)

    spu_plan_parser = subparsers.add_parser(
        "spu-plan",
        help="write the current plan for SPU whole-forward smoke validation",
    )
    spu_plan_parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    spu_plan_parser.add_argument("--output-json", default="")
    spu_plan_parser.set_defaults(func=command_spu_plan)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
