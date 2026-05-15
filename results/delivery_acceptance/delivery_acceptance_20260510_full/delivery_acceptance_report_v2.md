# Transshield Delivery Acceptance Report

- bundle_dir: `/home/yclcg/Transshield_final/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`
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
- argmax_match_ratio: `1.000000`
- threshold_match_ratio: `1.000000`
- logits_max_abs_error: `0.000000`
- probabilities_max_abs_error: `0.000000`

## E2E Same-Policy Consistency
- available: `True`
- argmax_match_ratio: `1.000000`
- threshold_match_ratio: `1.000000`
- logits_max_abs_error: `0.003555`
- probabilities_max_abs_error: `0.001773`

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
- legacy_replay_consistency_exact: `true`
- legacy_replay_consistency_high: `true`
- e2e_same_policy_consistency_exact: `true`
- e2e_same_policy_consistency_high: `true`
- secret_runtime_summary_available: `true`
- secret_runtime_complete: `true`
- secret_runtime_no_unstable_items: `true`
- secret_runtime_no_pending_items: `true`
