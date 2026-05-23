# Third-party provenance

本文件说明当前 `Transshield` 最终交付仓内保留的第三方来源、主要落点和许可证映射。

## DynamicViT side

- Source workspace: `DynamicViT_exp_square`
- Main retained areas:
  - `main.py`
  - `training_core/`
  - `models/dyvit.py`
  - `tools/transshield_*`
- Preserved license text:
  - `licenses/DynamicViT-MIT.txt`

## OpenBumbleBee integration side

- Source workspace: `OpenBumbleBee`
- Main retained areas:
  - `integrations/transshield_runtime/`
  - historical bridge lineage noted in `docs/evidence/spu_bumblebee_local_modifications.md`
- Preserved license text:
  - `licenses/OpenBumbleBee-Apache-2.0.txt`

## SPU vendored side

- Retained subtree:
  - `spu_vendored/`
- Usage:
  - retained as in-repo vendored protocol/runtime reference and provenance carrier for the final competition delivery
  - local adaptation notes are kept in `spu_vendored/MODIFICATIONS.md`
- Preserved license text:
  - `spu_vendored/LICENSE`
  - `licenses/SPU-Apache-2.0.txt`

## Notes

- `licenses/README.md` is the central index for preserved third-party license texts.
- `spu_vendored/LICENSE` remains the authoritative in-place license file for the vendored subtree; the mirrored file under `licenses/` is for centralized review convenience.
- Historical JSON artifacts and metadata files may still contain original absolute paths for provenance purposes.
- This repository currently does not add a root-level outbound `LICENSE`, because the final assembled submission contains mixed upstream provenance plus team-authored glue code, and no single repo-wide relicensing decision has been explicitly declared.
