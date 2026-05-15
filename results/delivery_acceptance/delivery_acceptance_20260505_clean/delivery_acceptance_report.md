# Transshield Delivery Acceptance Report

- bundle_dir: `/data/wyb/Transshield_final/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- readiness: `p0_delivery_closure_ready`
- reason: plaintext/fairness/boundary/consistency/secret-runtime 五个闭环都已有当前证据。

## Plaintext Full-Val
- available: `True`
- argmax_accuracy: `75.190842`
- threshold_accuracy: `90.076333`
- auc: `0.958431`
- sample_count: `524`

## Fairness
- available: `True`
- accuracy_comparison_is_fair: `true`
- fairness_reason: `Transshield 与 MPCViT 都指向同一组 train/val 路径，样本量也一致，可做同数据集效果对比。`
- gap_vs_mpcvit_argmax_pt: `-20.992364`
- gap_vs_mpcvit_threshold_pt: `-6.106873`
- gap_vs_mpcvit_auc: `-0.034133`

## Boundary Checks
- network_kth_overall_passed: `true`
- tie_policy_overall_passed: `true`
- stage_decision_match_ratio: `1.000000`
- max_kth_threshold_abs_error: `0.000013`
- max_threshold_snap_distance: `0.000013`

## Legacy Replay Consistency
- available: `True`
- argmax_match_ratio: `0.994275`
- threshold_match_ratio: `0.975191`
- logits_max_abs_error: `0.040701`
- probabilities_max_abs_error: `0.019504`

## E2E Same-Policy Consistency
- available: `False`
- argmax_match_ratio: `N/A`
- threshold_match_ratio: `N/A`
- logits_max_abs_error: `N/A`
- probabilities_max_abs_error: `N/A`

## Secret Runtime
- available: `True`
- complete: `true`
- accepted_count: `8`
- unstable_count: `0`
- pending_count: `0`
- accepted_accuracy: `1.000000`
- mean_accepted_elapsed_sec: `93.895938`

## Gates
- plaintext_fullval_available: `true`
- fairness_report_available: `true`
- fairness_comparison_is_fair: `true`
- fairness_transshield_matches_current_plaintext_fullval: `true`
- boundary_kth_check_passed: `true`
- boundary_tie_check_passed: `true`
- boundary_stage_decision_match_full: `true`
- legacy_replay_consistency_exact: `false`
- legacy_replay_consistency_high: `true`
- e2e_same_policy_consistency_exact: `N/A`
- e2e_same_policy_consistency_high: `N/A`
- secret_runtime_summary_available: `true`
- secret_runtime_complete: `true`
- secret_runtime_no_unstable_items: `true`
- secret_runtime_no_pending_items: `true`
