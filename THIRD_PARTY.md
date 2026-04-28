# Third-party provenance

This repository is an extracted Transshield-focused derivative assembled from two working repositories:

## DynamicViT side

- Source workspace: `DynamicViT_exp_square`
- Main borrowed areas:
  - `main.py`, `engine.py`, `infer.py`
  - `datasets.py`, `losses.py`, `optim_factory.py`, `samplers.py`, `utils.py`
  - `models/`
  - `tools/transshield_*`
- Preserved license text: `licenses/DynamicViT-MIT.txt`

## OpenBumbleBee side

- Source workspace: `OpenBumbleBee`
- Main borrowed area:
  - `examples/python/ml/transshield_network_kth_bridge/`
- Preserved license text: `licenses/OpenBumbleBee-Apache-2.0.txt`

## Notes

- This repo intentionally keeps the extracted code under a single Transshield-owned submission surface.
- Historical JSON artifacts may still contain absolute paths from the source workspaces for provenance purposes.

