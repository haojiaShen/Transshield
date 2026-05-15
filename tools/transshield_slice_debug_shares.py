import argparse
import json
from pathlib import Path


DEBUG_SHARE_TYPE = "transshield_e2e_debug_float_additive_share_v0"
DEBUG_SHARE_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_manifest_v0"
DEBUG_SHARE_PUBLIC_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_public_manifest_v0"
DEBUG_SHARE_PARTY_MANIFEST_TYPE = "transshield_e2e_debug_float_additive_share_party_manifest_v0"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_path_maps(raw_items):
    mappings = []
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError("--path-map must use FROM=TO")
        source, target = item.split("=", 1)
        mappings.append((source, target))
    return mappings


def apply_path_maps(raw_path, mappings):
    value = str(raw_path)
    for source, target in mappings:
        if value.startswith(source):
            value = target + value[len(source) :]
            break
    return Path(value).expanduser().resolve()


def slice_optional_sequence(value, start, end):
    if value is None:
        return None
    return list(value)[start:end]


def select_optional_sequence(value, indices):
    if value is None:
        return None
    items = list(value)
    return [items[index] for index in indices]


def tensor_to_target_list(targets):
    if targets is None:
        return None
    return [int(item) for item in targets.detach().cpu().tolist()]


def build_public_manifest(manifest, output_json: Path, sample_count: int, sample_ids, targets):
    public_manifest = {
        "manifest_type": DEBUG_SHARE_PUBLIC_MANIFEST_TYPE,
        "bundle_dir": manifest.get("bundle_dir"),
        "share_count": manifest.get("share_count"),
        "share_semantics": manifest.get("share_semantics"),
        "share_dtype": manifest.get("share_dtype"),
        "share_shape": [sample_count, *list(manifest.get("share_shape", [])[1:])],
        "sample_count": sample_count,
        "sample_ids": sample_ids,
        "targets_included": targets is not None,
        "targets": tensor_to_target_list(targets),
        "selection_mode": f"slice:{manifest.get('selection_mode', 'unknown')}",
        "source_paths_included": bool(manifest.get("source_paths_included", False)),
        "source_paths": slice_optional_sequence(manifest.get("source_paths"), 0, sample_count)
        if manifest.get("source_paths_included")
        else None,
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
    write_json(output_json, public_manifest)
    return public_manifest


def write_party_manifest(path: Path, rank: int, share_path: Path, public_json: Path, manifest, sample_count, sample_ids):
    party_id = "P1" if rank == 0 else "P2"
    payload = {
        "manifest_type": DEBUG_SHARE_PARTY_MANIFEST_TYPE,
        "party_id": party_id,
        "share_rank": rank,
        "share_count": 2,
        "share_path": str(share_path),
        "public_manifest_json": str(public_json),
        "share_semantics": manifest.get("share_semantics"),
        "share_dtype": manifest.get("share_dtype"),
        "share_shape": [sample_count, *list(manifest.get("share_shape", [])[1:])],
        "sample_count": sample_count,
        "sample_ids": sample_ids,
        "privacy_status": "debug_party_share_manifest_only; this manifest must not include the other party share path",
    }
    write_json(path, payload)
    return payload


def command_slice(args):
    import torch

    manifest_path = Path(args.share_manifest_json).expanduser().resolve()
    manifest = load_json(manifest_path)
    if manifest.get("manifest_type") != DEBUG_SHARE_MANIFEST_TYPE:
        raise ValueError(f"unsupported share manifest type: {manifest.get('manifest_type')}")

    if args.indices and args.indices_file:
        raise ValueError("--indices and --indices-file cannot be used together")
    if args.indices_file:
        raw_indices = Path(args.indices_file).read_text(encoding="utf-8").replace(",", " ").split()
        selected_indices = [int(item) for item in raw_indices]
    elif args.indices:
        selected_indices = [int(item) for item in args.indices.replace(",", " ").split()]
    else:
        if args.start_index is None or args.end_index is None:
            raise ValueError("--start-index/--end-index are required unless --indices or --indices-file is set")
        start = int(args.start_index)
        end = int(args.end_index)
        if start < 0 or end <= start:
            raise ValueError(f"invalid slice [{start}, {end})")
        selected_indices = list(range(start, end))
    if not selected_indices:
        raise ValueError("empty selected index list")
    if min(selected_indices) < 0:
        raise ValueError(f"negative index in selected indices: {selected_indices}")

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_public_json = Path(args.output_public_json).expanduser().resolve()
    output_party_dir = Path(args.output_party_manifest_dir).expanduser().resolve()
    output_input_pt = Path(args.output_input_pt).expanduser().resolve() if args.output_input_pt else None
    source_paths_override = None
    if args.source_paths_file:
        source_paths_override = Path(args.source_paths_file).read_text(encoding="utf-8").splitlines()

    path_maps = parse_path_maps(args.path_map)
    share_paths = [apply_path_maps(raw, path_maps) for raw in manifest.get("share_paths", [])]
    if len(share_paths) != 2:
        raise ValueError(f"expected exactly 2 share paths, got {len(share_paths)}")

    output_share_paths = []
    sliced_targets = None
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    for rank, share_path in enumerate(share_paths):
        payload = torch.load(share_path, map_location="cpu")
        if payload.get("manifest_type") != DEBUG_SHARE_TYPE:
            raise ValueError(f"unsupported share payload type in {share_path}: {payload.get('manifest_type')}")
        if int(payload.get("share_rank")) != rank:
            raise ValueError(f"share rank mismatch in {share_path}: {payload.get('share_rank')} vs {rank}")
        full_share_tensor = payload["share_tensor"]
        if max(selected_indices) >= int(full_share_tensor.shape[0]):
            raise ValueError(
                f"selected index {max(selected_indices)} exceeds share tensor shape {tuple(full_share_tensor.shape)}"
            )
        share_tensor = full_share_tensor.index_select(0, index_tensor).detach().cpu().contiguous()
        targets = payload.get("targets")
        rank_targets = targets.index_select(0, index_tensor).detach().cpu().contiguous() if targets is not None else None
        if rank == 0:
            sliced_targets = rank_targets

        output_share_path = output_prefix.with_name(output_prefix.name + f"_p{rank + 1}_share.pt")
        output_share_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **payload,
                "share_tensor": share_tensor,
                "sample_ids": select_optional_sequence(payload.get("sample_ids"), selected_indices),
                "targets": rank_targets,
                "source_manifest_type": DEBUG_SHARE_MANIFEST_TYPE,
                "source_selection": {
                    "share_manifest_json": str(manifest_path),
                    "indices": selected_indices,
                },
            },
            output_share_path,
        )
        output_share_paths.append(output_share_path)

    sample_count = len(selected_indices)
    sample_ids = select_optional_sequence(manifest.get("sample_ids"), selected_indices)
    if source_paths_override is not None:
        if max(selected_indices) >= len(source_paths_override):
            raise ValueError(
                f"selected index {max(selected_indices)} exceeds source path list length {len(source_paths_override)}"
            )
        selected_source_paths = select_optional_sequence(source_paths_override, selected_indices)
        source_paths_included = True
    else:
        selected_source_paths = (
            select_optional_sequence(manifest.get("source_paths"), selected_indices)
            if manifest.get("source_paths_included")
            else None
        )
        source_paths_included = bool(manifest.get("source_paths_included", False))
    sliced_manifest = {
        **manifest,
        "share_paths": [str(path) for path in output_share_paths],
        "share_shape": [sample_count, *list(manifest.get("share_shape", [])[1:])],
        "sample_count": sample_count,
        "sample_ids": sample_ids,
        "targets_included": sliced_targets is not None,
        "targets": tensor_to_target_list(sliced_targets),
        "source_paths": selected_source_paths,
        "source_paths_included": source_paths_included,
        "source_selection": {
            "share_manifest_json": str(manifest_path),
            "indices": selected_indices,
        },
    }
    write_json(output_json, sliced_manifest)
    build_public_manifest(sliced_manifest, output_public_json, sample_count, sample_ids, sliced_targets)

    output_party_dir.mkdir(parents=True, exist_ok=True)
    party_paths = []
    for rank, share_path in enumerate(output_share_paths):
        party_path = output_party_dir / f"p{rank + 1}_share_manifest.json"
        write_party_manifest(party_path, rank, share_path, output_public_json, sliced_manifest, sample_count, sample_ids)
        party_paths.append(str(party_path))

    result = {
        "manifest_type": "transshield_e2e_debug_share_slice_result_v0",
        "output_json": str(output_json),
        "output_input_pt": None if output_input_pt is None else str(output_input_pt),
        "output_public_json": str(output_public_json),
        "output_party_manifest_dir": str(output_party_dir),
        "party_manifest_jsons": party_paths,
        "sample_count": sample_count,
        "sample_ids": sample_ids,
        "targets": tensor_to_target_list(sliced_targets),
        "selected_indices": selected_indices,
    }
    if args.input_pt:
        input_pt = Path(args.input_pt).expanduser().resolve()
        if output_input_pt is None:
            raise ValueError("--output-input-pt is required when --input-pt is set")
        input_payload = torch.load(input_pt, map_location="cpu")
        if not isinstance(input_payload, dict) or "pixel_values" not in input_payload:
            raise ValueError(f"unsupported input PT payload: {input_pt}")
        sliced_input = dict(input_payload)
        full_pixel_values = input_payload["pixel_values"]
        if max(selected_indices) >= int(full_pixel_values.shape[0]):
            raise ValueError(
                f"selected index {max(selected_indices)} exceeds input tensor shape {tuple(full_pixel_values.shape)}"
            )
        sliced_input["pixel_values"] = full_pixel_values.index_select(0, index_tensor).detach().cpu().contiguous()
        if input_payload.get("targets") is not None:
            sliced_input["targets"] = input_payload["targets"].index_select(0, index_tensor).detach().cpu().contiguous()
        if input_payload.get("sample_ids") is not None:
            sliced_input["sample_ids"] = select_optional_sequence(input_payload.get("sample_ids"), selected_indices)
        metadata = dict(input_payload.get("metadata") or {})
        if metadata.get("source_paths") is not None:
            metadata["source_paths"] = select_optional_sequence(metadata.get("source_paths"), selected_indices)
        metadata["sample_count"] = sample_count
        metadata["source_selection"] = {
            "input_pt": str(input_pt),
            "indices": selected_indices,
        }
        sliced_input["metadata"] = metadata
        output_input_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sliced_input, output_input_pt)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description="Slice a debug additive-share manifest into a smaller manifest.")
    parser.add_argument("--share-manifest-json", required=True)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--indices", default="", help="Comma/space-separated non-contiguous sample indices.")
    parser.add_argument("--indices-file", default="", help="File containing comma/space/newline-separated sample indices.")
    parser.add_argument("--source-paths-file", default="", help="Optional original image-list file for source paths.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-public-json", required=True)
    parser.add_argument("--output-party-manifest-dir", required=True)
    parser.add_argument("--input-pt", default="", help="Optional plaintext pixel package to slice with the same range.")
    parser.add_argument("--output-input-pt", default="", help="Output path for the sliced plaintext pixel package.")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Rewrite input paths recorded in the manifest, e.g. /home/yclcg/Transshield_final=/home/yclcg/Transshield_final",
    )
    parser.set_defaults(func=command_slice)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
