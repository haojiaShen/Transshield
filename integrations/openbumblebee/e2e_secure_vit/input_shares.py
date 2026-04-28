from pathlib import Path


def load_debug_input_share_pair(share_manifest_json: Path, helpers):
    import torch

    manifest = helpers["load_debug_share_manifest"](share_manifest_json)
    share_payloads = []
    for raw_path in manifest["share_paths"]:
        share_path = Path(raw_path).expanduser().resolve()
        payload = torch.load(share_path, map_location="cpu")
        if payload.get("manifest_type") != "transshield_e2e_debug_float_additive_share_v0":
            raise ValueError(f"unsupported debug share payload in {share_path}: {payload.get('manifest_type')}")
        share_payloads.append(payload)
    share_payloads = sorted(share_payloads, key=lambda item: int(item["share_rank"]))
    share_tensors = [item["share_tensor"].detach().cpu().float() for item in share_payloads]
    if tuple(share_tensors[0].shape) != tuple(share_tensors[1].shape):
        raise ValueError(
            f"share shape mismatch: {tuple(share_tensors[0].shape)} vs {tuple(share_tensors[1].shape)}"
        )
    return {
        "manifest": manifest,
        "share_tensors": share_tensors,
        "sample_ids": manifest.get("sample_ids"),
        "targets": share_payloads[0].get("targets"),
    }


def load_debug_input_share_pair_from_party_manifests(
    public_manifest_json: Path,
    p1_share_manifest_json: Path,
    p2_share_manifest_json: Path,
    helpers,
):
    import torch

    public_manifest = helpers["load_debug_share_public_manifest"](public_manifest_json)
    party_manifest_paths = [p1_share_manifest_json, p2_share_manifest_json]
    share_payloads = []
    party_manifests = []
    for expected_rank, party_manifest_path in enumerate(party_manifest_paths):
        party_manifest = helpers["load_debug_share_party_manifest"](party_manifest_path)
        party_manifests.append(party_manifest)
        if int(party_manifest["share_rank"]) != expected_rank:
            raise ValueError(
                f"expected share_rank={expected_rank} in {party_manifest_path}, "
                f"got {party_manifest['share_rank']}"
            )
        if Path(party_manifest["public_manifest_json"]).expanduser().resolve() != public_manifest_json:
            raise ValueError(
                f"party manifest {party_manifest_path} points to a different public manifest: "
                f"{party_manifest['public_manifest_json']}"
            )
        share_path = Path(party_manifest["share_path"]).expanduser().resolve()
        payload = torch.load(share_path, map_location="cpu")
        if payload.get("manifest_type") != "transshield_e2e_debug_float_additive_share_v0":
            raise ValueError(f"unsupported debug share payload in {share_path}: {payload.get('manifest_type')}")
        if int(payload.get("share_rank")) != expected_rank:
            raise ValueError(f"share payload rank mismatch in {share_path}: {payload.get('share_rank')}")
        share_payloads.append(payload)
    share_tensors = [item["share_tensor"].detach().cpu().float() for item in share_payloads]
    if tuple(share_tensors[0].shape) != tuple(share_tensors[1].shape):
        raise ValueError(
            f"share shape mismatch: {tuple(share_tensors[0].shape)} vs {tuple(share_tensors[1].shape)}"
        )
    if list(share_tensors[0].shape) != list(public_manifest.get("share_shape", [])):
        raise ValueError(
            f"share shape does not match public manifest: {list(share_tensors[0].shape)} "
            f"vs {public_manifest.get('share_shape')}"
        )
    return {
        "manifest": public_manifest,
        "party_manifests": party_manifests,
        "share_tensors": share_tensors,
        "sample_ids": public_manifest.get("sample_ids"),
        "targets": share_payloads[0].get("targets"),
    }


def load_debug_party_share_metadata(
    public_manifest_json: Path,
    p1_share_manifest_json: Path,
    p2_share_manifest_json: Path,
    helpers,
):
    public_manifest = helpers["load_debug_share_public_manifest"](public_manifest_json)
    party_manifest_paths = [p1_share_manifest_json, p2_share_manifest_json]
    party_manifests = []
    for expected_rank, party_manifest_path in enumerate(party_manifest_paths):
        party_manifest = helpers["load_debug_share_party_manifest"](party_manifest_path)
        if int(party_manifest["share_rank"]) != expected_rank:
            raise ValueError(
                f"expected share_rank={expected_rank} in {party_manifest_path}, "
                f"got {party_manifest['share_rank']}"
            )
        if Path(party_manifest["public_manifest_json"]).expanduser().resolve() != public_manifest_json:
            raise ValueError(
                f"party manifest {party_manifest_path} points to a different public manifest: "
                f"{party_manifest['public_manifest_json']}"
            )
        party_manifests.append(party_manifest)
    return {
        "manifest": public_manifest,
        "party_manifests": party_manifests,
        "party_manifest_paths": [str(path) for path in party_manifest_paths],
        "sample_ids": public_manifest.get("sample_ids"),
        "targets": None,
    }
