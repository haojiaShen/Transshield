# frozen_bundle_verified_tracka_lr3e5_20260414

Safe promotion-ready bundle generated from a verified candidate without overwriting the current official bundle.

## Source
- source candidate bundle: `/home/yclcg/Transshield_final/artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414`
- source manifest: `source_manifest.json`

## Bundled assets
- `modified_plaintext_model_state_dict.pth`
- `modified_plaintext_eval_checkpoint_light.pth`
- `threshold_best.json`
- `args_snapshot.json`
- `promotion_manifest.json`

## Verified metrics
- plaintext / secure argmax accuracy: `93.70229244232178`
- plaintext / secure threshold accuracy: `94.08397078514099`
- argmax match ratio: `1.0`
- threshold match ratio: `1.0`
- communication status: `available_python_fastpath`

## Communication display
- source: `Python distributed RPC/cloudpickle fastpath`
- Python fastpath RPC total bytes: `1,993,399`
- C++ `LinkDetails` zero counters are diagnostic-only and are not the primary communication metric.

## Note
- This bundle is promotion-ready and self-contained for repo-side evaluation.
- It does not overwrite `artifacts/frozen_bundle_full/`; promotion remains an explicit human decision.
