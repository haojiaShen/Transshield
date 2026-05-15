# Secure Static Train Depth Evidence

## 1. 结论

- 当前最稳妥的表述：`secure_static_train_depth` 当前已经能被表述为官方 secure-friendly 主线中的训练-部署对齐设计选择，但还不能被表述为已经完成单因子因果归因的收益来源。

## 2. 当前 bundle 训练口径

- secure_static_train_depth: `12`
- secure_static_skip_pruning: `true`
- approx_attn_mode: `uniform`
- square_activation_mode: `fixed_square`
- use_square_gelu: `true`
- epochs: `8`
- lr: `0.00001000`
- batch_size: `8`

## 3. 当前可验证证据

| Evidence | Baseline / Control | Current Line | Delta |
|---|---:|---:|---:|
| secure_static_train_depth | 0 | 12 | N/A |
| threshold_accuracy | 75.3817 | 90.0763 | 14.6947 |
| argmax_accuracy | 74.8092 | 75.1908 | 0.3817 |
| auc | 0.663525 | 0.958431 | 0.294906 |

## 4. 当前验收口径

- readiness: `p0_delivery_closure_ready`
- reason: plaintext/fairness/boundary/consistency/secret-runtime 五个闭环都已有当前证据。
- boundary_kth_check_passed: `true`
- boundary_tie_check_passed: `true`
- e2e_same_policy_consistency_exact: `true`
- secret_runtime_complete: `true`

## 5. 当前 paired control

| Pair | Epochs | Depth 0 -> 12 | Status | Threshold Delta | AUC Delta | Argmax Delta |
|---|---:|---:|---|---:|---:|---:|
| secure_static_depth_pair_epoch1_20260506_depth12a | 1 | 0 -> 12 | no_clear_depth_benefit_yet | -1.5267 | -0.011673 | 4.0076 |
| secure_static_depth_pair_epoch3_20260506_depth12b | 3 | 0 -> 12 | no_clear_depth_benefit_yet | -0.9542 | -0.009750 | 5.5344 |

## 6. Secret 路径对齐

- deployable secret profile: `secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1`
- secret_runtime_complete: `true`
- pending_count: `0`
- unstable_count: `0`
- argmax_match_ratio: `99.43%`
- threshold_match_ratio: `97.52%`
- logits_max_abs_error: `0.04070145`
- probabilities_max_abs_error: `0.01950449`

## 7. 证据分级

- training_semantics_match_static_secret_scope: `high`
  reason: current official bundle uses secure_static_train_depth>0, secure_static_skip_pruning=true, uniform attention and fixed_square activation.
- current_bundle_supports_deployable_secret_line: `medium`
  reason: depth6 clip0 guarded secret runtime is complete and stable on the current official line, but it is still shallower than the training depth12 setting.
- isolated_causal_attribution_of_secure_static_train_depth: `medium`
  reason: the repo already retains paired controls that change only secure_static_train_depth, but current epoch1/epoch3 results still show no_clear_depth_benefit_yet.

## 8. 当前支持什么

- 当前 official bundle 明确训练在 static-scope-compatible 配置上
- 当前 modified line 在 threshold accuracy 和 AUC 上显著优于 retained baseline
- 同一条 official line 可以被当前 depth6 clip0 guarded secret runtime 以较高一致性稳定承接
- 仓库中已经存在只改变 secure_static_train_depth 的 paired control，当前 epoch1/epoch3 compare 都给出 no_clear_depth_benefit_yet。

## 9. 当前还不支持什么

- 更深 secure_static_train_depth 单独带来 threshold accuracy / AUC 收益
- 一个 train-depth 与 deploy-depth 精确对齐的保留式最小 ablation

## 10. 下一步最小 ablation

- goal: retain one paired control bundle under the same uniform/fixed_square/static-skip-pruning stack, changing only secure_static_train_depth.
- suggested_runner: `bash artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh suite`

