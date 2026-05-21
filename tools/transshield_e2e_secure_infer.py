import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.openbumblebee.e2e_secure_vit.cpu_static_vit import (
    run_static_student_whole_forward_limited,
)
from tools.transshield_stage2_bundle import (
    load_frozen_bundle,
    load_json,
    preprocess_image,
    resolve_bundle_dir,
    write_json,
)


CLIENT_PIXEL_MANIFEST_TYPE = "transshield_e2e_client_pixel_values_v0"
DEBUG_SHARE_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_manifest_v0"
DEBUG_SHARE_PUBLIC_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_public_manifest_v0"
DEBUG_SHARE_PARTY_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_party_manifest_v0"
DEBUG_SHARE_PAYLOAD_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_v0"
DEBUG_SHARE_SEMANTICS = "debug_float_additive_share_not_production_mpc_share"


def load_tensor_payload(path):
    return torch.load(Path(path).expanduser().resolve(), map_location="cpu")


def load_client_pixel_package(path):
    payload = load_tensor_payload(path)
    if payload.get("manifest_type") != CLIENT_PIXEL_MANIFEST_TYPE:
        raise ValueError(f"unsupported client pixel package: {payload.get('manifest_type')}")
    return payload


def load_debug_share_manifest(path):
    payload = load_json(Path(path).expanduser().resolve())
    if payload.get("manifest_type") != DEBUG_SHARE_MANIFEST_TYPE:
        raise ValueError(f"unsupported debug share manifest: {payload.get('manifest_type')}")
    return payload


def load_debug_share_public_manifest(path):
    payload = load_json(Path(path).expanduser().resolve())
    if payload.get("manifest_type") != DEBUG_SHARE_PUBLIC_MANIFEST_TYPE:
        raise ValueError(f"unsupported debug share public manifest: {payload.get('manifest_type')}")
    return payload


def load_debug_share_party_manifest(path):
    payload = load_json(Path(path).expanduser().resolve())
    if payload.get("manifest_type") != DEBUG_SHARE_PARTY_MANIFEST_TYPE:
        raise ValueError(f"unsupported debug share party manifest: {payload.get('manifest_type')}")
    return payload


def compare_prediction_match(reference, candidate):
    if reference is None or candidate is None:
        return None
    ref = torch.as_tensor(reference).detach().cpu()
    cand = torch.as_tensor(candidate).detach().cpu()
    if ref.shape != cand.shape:
        raise ValueError(f"prediction shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    if ref.numel() == 0:
        return None
    return float((ref == cand).float().mean().item())


def run_static_student_whole_forward(model, pixel_values):
    return run_static_student_whole_forward_limited(model, pixel_values, static_depth_limit=-1)


def infer_binary_target_from_path(path: Path):
    parent = path.parent.name.lower()
    if parent in {"0", "fraud"}:
        return 0
    if parent in {"1", "normal"}:
        return 1
    return None


def collect_selected_paths(args):
    if args.image:
        return [Path(args.image).expanduser().resolve()], "single_image"
    if args.image_list:
        paths = [
            Path(line.strip()).expanduser().resolve()
            for line in Path(args.image_list).expanduser().resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return paths, "explicit_paths"
    if args.data_path:
        root = Path(args.data_path).expanduser().resolve()
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        return paths, "data_path_recursive"
    raise ValueError("one of --image / --image-list / --data-path is required")


def build_client_pixel_payload(bundle, args):
    selected_paths, selection_mode = collect_selected_paths(args)
    if args.max_samples > 0:
        selected_paths = selected_paths[: int(args.max_samples)]
    if not selected_paths:
        raise ValueError("no images selected")

    tensors = []
    sample_ids = []
    targets = []
    resolved_source_paths = []
    for index, image_path in enumerate(selected_paths):
        resolved_path, input_tensor = preprocess_image(image_path, bundle["transform"], device="cpu")
        tensors.append(input_tensor.detach().cpu())
        sample_ids.append(f"sample_{index:06d}")
        resolved_source_paths.append(str(resolved_path))
        targets.append(infer_binary_target_from_path(resolved_path))

    pixel_values = torch.cat(tensors, dim=0).contiguous()
    targets_tensor = None
    if args.include_targets and all(target is not None for target in targets):
        targets_tensor = torch.tensor(targets, dtype=torch.long)

    payload = {
        "manifest_type": CLIENT_PIXEL_MANIFEST_TYPE,
        "bundle_dir": str(resolve_bundle_dir(args.bundle_dir)),
        "input_size": int(bundle["args_snapshot"].get("input_size", 224)),
        "crop_pct": bundle["args_snapshot"].get("crop_pct"),
        "imagenet_default_mean_and_std": bool(bundle["args_snapshot"].get("imagenet_default_mean_and_std", True)),
        "selection_mode": selection_mode,
        "sample_count": int(pixel_values.shape[0]),
        "sample_ids": sample_ids,
        "pixel_values": pixel_values,
        "targets": targets_tensor,
        "source_paths": resolved_source_paths if args.include_source_paths else None,
        "source_paths_included": bool(args.include_source_paths),
        "targets_included": bool(targets_tensor is not None),
    }
    return payload


def write_client_preprocess_outputs(payload, output_pt: Path, output_json: Path):
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    summary = {
        "manifest_type": CLIENT_PIXEL_MANIFEST_TYPE,
        "bundle_dir": payload["bundle_dir"],
        "input_size": payload["input_size"],
        "crop_pct": payload["crop_pct"],
        "imagenet_default_mean_and_std": payload["imagenet_default_mean_and_std"],
        "selection_mode": payload["selection_mode"],
        "sample_count": payload["sample_count"],
        "sample_ids": payload["sample_ids"],
        "source_paths": payload["source_paths"],
        "source_paths_included": payload["source_paths_included"],
        "targets_included": payload["targets_included"],
        "output_pt": str(output_pt),
        "pixel_values_shape": list(payload["pixel_values"].shape),
        "pixel_values_dtype": str(payload["pixel_values"].dtype),
    }
    if payload["targets"] is not None:
        summary["targets"] = [int(value) for value in payload["targets"].tolist()]
    write_json(output_json, summary)


def command_client_preprocess(args):
    bundle = load_frozen_bundle(args.bundle_dir, device="cpu")
    payload = build_client_pixel_payload(bundle, args)
    write_client_preprocess_outputs(payload, Path(args.output_pt).expanduser().resolve(), Path(args.output_json).expanduser().resolve())


def command_client_share_preprocess(args):
    bundle = load_frozen_bundle(args.bundle_dir, device="cpu")
    payload = build_client_pixel_payload(bundle, args)

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_public_json = Path(args.output_public_json).expanduser().resolve()
    output_party_manifest_dir = Path(args.output_party_manifest_dir).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_party_manifest_dir.mkdir(parents=True, exist_ok=True)

    pixel_values = payload["pixel_values"].detach().cpu().float().contiguous()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.share_seed))
    share0 = torch.empty_like(pixel_values).uniform_(-0.5, 0.5, generator=generator)
    share1 = pixel_values - share0

    share_paths = [
        output_prefix.parent / f"{output_prefix.name}_p1_share.pt",
        output_prefix.parent / f"{output_prefix.name}_p2_share.pt",
    ]
    share_payloads = [
        {
            "manifest_type": DEBUG_SHARE_PAYLOAD_MANIFEST_TYPE,
            "source_manifest_type": CLIENT_PIXEL_MANIFEST_TYPE,
            "share_rank": 0,
            "share_count": 2,
            "share_seed": int(args.share_seed),
            "share_semantics": DEBUG_SHARE_SEMANTICS,
            "sample_ids": payload["sample_ids"],
            "targets": payload["targets"],
            "share_tensor": share0,
        },
        {
            "manifest_type": DEBUG_SHARE_PAYLOAD_MANIFEST_TYPE,
            "source_manifest_type": CLIENT_PIXEL_MANIFEST_TYPE,
            "share_rank": 1,
            "share_count": 2,
            "share_seed": None,
            "share_semantics": DEBUG_SHARE_SEMANTICS,
            "sample_ids": payload["sample_ids"],
            "targets": payload["targets"],
            "share_tensor": share1,
        },
    ]
    for share_path, share_payload in zip(share_paths, share_payloads):
        torch.save(share_payload, share_path)

    share_manifest = {
        "manifest_type": DEBUG_SHARE_MANIFEST_TYPE,
        "bundle_dir": payload["bundle_dir"],
        "input_size": payload["input_size"],
        "crop_pct": payload["crop_pct"],
        "imagenet_default_mean_and_std": payload["imagenet_default_mean_and_std"],
        "selection_mode": payload["selection_mode"],
        "sample_count": payload["sample_count"],
        "sample_ids": payload["sample_ids"],
        "share_count": 2,
        "share_dtype": "torch.float32",
        "share_shape": list(pixel_values.shape),
        "share_semantics": DEBUG_SHARE_SEMANTICS,
        "share_paths": [str(path) for path in share_paths],
        "plaintext_pixel_package_written": False,
        "source_paths": payload["source_paths"] if args.include_source_paths else None,
        "source_paths_included": bool(args.include_source_paths),
        "targets": None if payload["targets"] is None else [int(value) for value in payload["targets"].tolist()],
        "targets_included": bool(payload["targets"] is not None),
        "privacy_status": "debug_additive_shares_only_not_production_mpc; each share file must be held by a different party",
        "server_boundary_note": "A production e2e path should provide each share to its owning party and must not reconstruct plaintext pixel_values on the server host.",
        "split_public_and_party_manifests": {
            "public_manifest_json": str(output_public_json),
            "party_manifest_jsons": [
                str(output_party_manifest_dir / "p1_share_manifest.json"),
                str(output_party_manifest_dir / "p2_share_manifest.json"),
            ],
        },
    }
    write_json(output_json, share_manifest)

    public_manifest = {
        "manifest_type": DEBUG_SHARE_PUBLIC_MANIFEST_TYPE,
        "bundle_dir": payload["bundle_dir"],
        "input_size": payload["input_size"],
        "crop_pct": payload["crop_pct"],
        "imagenet_default_mean_and_std": payload["imagenet_default_mean_and_std"],
        "party_ids": ["P1", "P2"],
        "privacy_status": "debug_public_share_manifest_only; private share file paths are intentionally kept in party manifests",
        "private_share_paths_included": False,
        "sample_count": payload["sample_count"],
        "sample_ids": payload["sample_ids"],
        "selection_mode": payload["selection_mode"],
        "server_boundary_note": "The server-side coordinator may read this public manifest, but each private share path should be provided only to its owning party in a production deployment.",
        "share_count": 2,
        "share_dtype": "torch.float32",
        "share_semantics": DEBUG_SHARE_SEMANTICS,
        "share_shape": list(pixel_values.shape),
        "source_paths_included": bool(args.include_source_paths),
        "targets_included": False,
    }
    write_json(output_public_json, public_manifest)

    for rank, party_id in enumerate(["P1", "P2"]):
        party_manifest = {
            "manifest_type": DEBUG_SHARE_PARTY_MANIFEST_TYPE,
            "party_id": party_id,
            "privacy_status": "debug_party_share_manifest_only; this manifest must not include the other party share path",
            "public_manifest_json": str(output_public_json),
            "sample_count": payload["sample_count"],
            "sample_ids": payload["sample_ids"],
            "share_count": 2,
            "share_dtype": "torch.float32",
            "share_path": str(share_paths[rank]),
            "share_rank": rank,
            "share_semantics": DEBUG_SHARE_SEMANTICS,
            "share_shape": list(pixel_values.shape),
        }
        write_json(output_party_manifest_dir / f"p{rank + 1}_share_manifest.json", party_manifest)


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal Transshield E2E input/share preprocessing helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_selection_args(target):
        target.add_argument("--bundle-dir", required=True)
        target.add_argument("--image", default="")
        target.add_argument("--image-list", default="")
        target.add_argument("--data-path", default="")
        target.add_argument("--max-samples", type=int, default=0)
        target.add_argument("--include-source-paths", action="store_true")
        target.add_argument("--include-targets", action="store_true")

    client_parser = subparsers.add_parser("client-preprocess")
    add_selection_args(client_parser)
    client_parser.add_argument("--output-pt", required=True)
    client_parser.add_argument("--output-json", required=True)
    client_parser.set_defaults(func=command_client_preprocess)

    share_parser = subparsers.add_parser("client-share-preprocess")
    add_selection_args(share_parser)
    share_parser.add_argument("--output-prefix", required=True)
    share_parser.add_argument("--output-json", required=True)
    share_parser.add_argument("--output-public-json", required=True)
    share_parser.add_argument("--output-party-manifest-dir", required=True)
    share_parser.add_argument("--share-seed", type=int, default=0)
    share_parser.set_defaults(func=command_client_share_preprocess)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
