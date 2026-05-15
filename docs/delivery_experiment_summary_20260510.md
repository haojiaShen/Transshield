# Transshield 实验结果总结

最后更新：`2026-05-10`

本文档汇总 Transshield 全部核心实验数据，作为交付/论文/竞赛的统一数据源。

---

## 0. 系统架构概述

### 0.1 核心创新

Transshield 的唯一主创新是 **DynamicViT pruning boundary 的协议友好重写**：

- `masking -> F_mux`：token keep/drop 的掩码操作改写为 MPC 友好的多路选择函数
- `threshold compare -> F_less`：pruning threshold 比较改写为 MPC 友好的安全比较函数
- `secure sidecar/replay`：pruning 决策以 sidecar 方式安全执行，结果 replay 回主模型

### 0.2 数据流

```
Client (影像明文)
  -> 生成 secret share -> 发送到 Server
  -> Server 在 SPU 上执行 whole-forward inference
     (secret params + secret input)
  -> 只返回 final logits (reveal_policy=final_logits_only)
  -> Client 重建 logits -> 阈值决策
```

### 0.3 关键隐私约束

| 约束 | 说明 |
|---|---|
| `host_plaintext_pixel_values_materialized = false` | Server 永远不接触明文影像 |
| `host_private_share_tensors_loaded = false` | Server 不加载 private share 文件 |
| `input_mode = party_local_debug_share_load` | 输入以 party-local 方式加载 |
| `private_input_paths_redacted = true` | 路径信息已脱敏 |
| `reveal_policy = final_logits_only` | 只暴露最终 logits |

### 0.4 技术栈

| 层 | 技术 |
|---|---|
| 模型 | DeiT-S / DynamicViT (depth=12) |
| 安全计算 | SPU / OpenBumbleBee (2PC semi-honest) |
| 近似算子 | uniform attention, exact LN, fixed_square activation, clip0 |
| 输出校准 | SPU-aware public logit-bias calibration |

---

## 1. 模型精度（Plaintext，full-val，n=524）

| 口径 | Bundle 20260430 | Bundle 20260507 (AA=none) |
|---|---|---|
| argmax accuracy | 75.19% | 76.72% |
| best threshold accuracy | 90.08% | 91.98% |
| AUC | 0.9584 | 0.9679 |
| CE loss | — | 0.5305 |

- 正式 bundle（20260430）：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- 精度增强 bundle（20260507）：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`
- 配置：`exact LN + uniform attention + fixed_square clip0 + static-path public output calibration`

### 1.1 不重训 calibration 恢复路径

| 方法 | calibrated argmax acc | CE loss |
|---|---|---|
| bias-only (0.5852) | 91.79% | 0.4287 |
| affine | 91.79% | 0.2025 |
| temperature | 91.79% | 0.1984 |
| SPU-aware bias (heldout-weighted) | **92.09%** | — |

- SPU-aware bias calibration 来源：smoke32 拟合，heldout64/128/238 加权验证
- 决策：`promote_spuaware_bias_as_accuracy_first_default`

---

## 2. 公平外部对比（同数据集，n=524）

| 方法 | argmax acc | threshold acc | AUC |
|---|---|---|---|
| **Transshield modified** | 75.19% | 90.08% | 0.9584 |
| **MPCViT** (baseline) | 96.18% | 96.18% | 0.9926 |
| **差值** | -21.0pp | -6.1pp | -0.034 |

- fairness 检查通过：同 train/val 路径、同样本量
- 公平对比入口：`run_fair_external_comparison.sh`
- 产物：`results/fair_external_comparison/fair_external_secure_static_20260505_clean/`

---

## 3. E2E Secure Inference（AA=none bundle，non-isolated，public params + exact LN + clip0）

### 3.1 精度（SPU-aware bias calibration）

| 样本集 | 样本数 | e2e threshold acc | 说明 |
|---|---|---|---|
| smoke8 | 8 | 75.00% | 有限样本 sanity |
| smoke16 | 16 | 81.25% | — |
| smoke32 (evenly-spaced) | 32 | 87.50% | calibration 拟合集 |
| smoke96 | 96 | 95.83% | — |
| heldout64 | 64 | 92.19% | smoke32-disjoint |
| heldout128 | 128 | 91.41% | smoke32-disjoint |
| heldout238 | 238 | **92.44%** | smoke32-disjoint，最大 heldout |

### 3.2 运行时效率

| 口径 | sec/sample | 说明 |
|---|---|---|
| isolated (单样本 fresh runtime) | ~48s | smoke8 baseline |
| non-isolated (runtime 复用) | **~21s** | smoke96/heldout238 |
| best speedup | **2.28x** | non-isolated vs isolated |

- aggregate 通信量（smoke96）：~1.76 GB（2PC SPU 双向）

### 3.3 隐私边界

| 字段 | 值 |
|---|---|
| host_plaintext_pixel_values_materialized | false |
| host_private_share_tensors_loaded | false |
| input_mode | party_local_debug_share_load |
| private_input_paths_redacted | true |
| reveal_policy | final_logits_only |
| finite_logits (all runs) | true |

---

## 4. Keep-mask Whole-forward Wrapper（secret params + party-local share）

### 4.1 精度一致性

| Run | 样本数 | argmax match | threshold match | logit max_abs_error |
|---|---|---|---|---|
| smoke1 | 1 | 1.0 | 1.0 | 0.00259 |
| smoke8 | 8 | 1.0 | 1.0 | 0.00279 |
| smoke16 | 16 | 1.0 | 1.0 | 0.00263 |
| smoke32 | 32 | 1.0 | 1.0 | 0.00355 |

- **全部 1.0/1.0**，`privacy_consistent = true`

### 4.2 Runtime Scaling

| 样本数 | sec/sample | incremental sec/sample |
|---|---|---|
| 1 | 233.83 | — |
| 8 | 201.58 | 196.98 |
| 16 | 200.20 | 198.81 |
| 32 | 194.63 | 189.06 |

- 趋势：近线性收敛，sec/sample 随样本数增大而下降
- 报告：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_scaling_20260509_1/keepmask_scaling_report.md`

### 4.3 与 E2E approximate 的区别

keep-mask wrapper 是**确定性 replay**：CPU 参考先跑完整 dynamic pruning 得到 keep mask，再把 mask 注入 SPU 做 whole-forward。因此：
- 精度误差仅来自 SPU 数值近似（~0.003 级别），不是决策偏差
- E2E approximate 的 87~92% 口径包含 static-path approximation 的固有误差

---

## 4.1 Secure Pruning（PredictorLG SPU 内部安全执行）

### smoke1 结果

| 指标 | 值 |
|---|---|
| backend | `jax_spu_secure_pruning_forward_backend_v0` |
| forward_scope | `student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path` |
| finite_logits | true |
| has_predictor_params | true |
| elapsed_sec | 254.645 |
| argmax/threshold match | 1.0 / 1.0 |

### 隐私字段

| 字段 | 值 | 说明 |
|---|---|---|
| `host_plaintext_pixel_values_materialized` | `false` | 服务器永远不接触明文影像 |
| `host_private_share_tensors_loaded` | `false` | 服务器不加载 private share |
| `spu_params_mode` | `secret` | 模型参数以 secret share 形式加载到 SPU |
| `host_model_params_materialized` | `false` | 数据使用方不接触明文模型参数 |
| `runtime_pruning_keep_mask_pt` | `null` | 不依赖外部 keep-mask |
| `reveal_policy` | `final_logits_only` | 只暴露最终 logits |

### 技术要点

- PredictorLG + kth_threshold + tie_resolution 整条 pruning decision 链在 SPU 内部完整执行
- encoded-key bitonic sort：`encoded_key = score - index * 1e-6`（tie-breaking: lower index wins）
- `_bitonic_sort_desc` 重写为全 `jnp.where` 模式，消除 boolean fancy indexing
- `jsp_special.logsumexp` 替换为手动 `max + log(sum(exp(...)))` 实现
- `pruning_metadata` 通过闭包 concrete Python 值传入，不经过 SPU 参数通道

### 隐私意义

- ✅ 服务器看不到数据使用方图片（party-local share load）
- ✅ 数据使用方获取不到模型参数（PredictorLG 在 SPU 内部执行）
- ✅ 只暴露最终 logits（`reveal_policy = final_logits_only`）
- 双向隐私边界全部成立

## 5. P0 交付闭环状态

| 闭环 | 状态 | 关键证据 |
|---|---|---|
| 方法闭环 | ✅ | `masking→F_mux`, `threshold compare→F_less`, `docs/architecture.md` |
| 系统闭环 | ✅ | `score→threshold→decision→sidecar→replay→prediction` |
| 隐私闭环 | ✅ | party-local share input, secret params, redacted paths |
| 效果闭环 | ✅ | full-val plaintext 91.98%, heldout238 92.44%, fairness pass |
| 运行闭环 | ✅ | guarded secret eval, isolated/non-isolated batch |

- 正式状态：`p0_delivery_closure_ready`
- 验收报告（含 boundary check）：`results/delivery_acceptance/delivery_acceptance_20260510_full/delivery_acceptance_report.json`
  - `readiness = p0_delivery_closure_ready`
  - `boundary_kth_check_passed = true`（3 stage，max abs error 1.28e-05）
  - `boundary_tie_check_passed = true`（stage_decision_match_ratio = 1.0）
  - `e2e_same_policy_consistency_exact = true`（argmax/threshold match = 1.0/1.0）
  - `legacy_replay_consistency_high = true`（argmax 0.994, threshold 0.975）
  - 五个闭环全部 ✅：plaintext / fairness / boundary / consistency / secret-runtime

---

## 5.1 Boundary Check 数据

| 检查项 | 结果 | 最大绝对误差 |
|---|---|---|
| network_kth（3 stage） | ✅ passed | 1.28e-05 |
| tie_policy（3 stage） | ✅ passed | 1.28e-05 |
| stage_decision_match_ratio | 1.0 | — |

- Stage 0: keep_count=137, pruning_layer=3, kth_error=1.24e-05
- Stage 1: keep_count=96, pruning_layer=6, kth_error=2.38e-07
- Stage 2: keep_count=67, pruning_layer=9, kth_error=1.28e-05
- tie_policy: 3/3 stage semantic_passed, reconstructed_branch_matches_topk_reference=true

## 6. 关键产物索引

| 类别 | 路径 |
|---|---|
| 正式 bundle | `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430` |
| 精度增强 bundle | `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507` |
| 验收报告（完整） | `results/delivery_acceptance/delivery_acceptance_20260510_full/` |
| 验收报告（初始） | `results/delivery_acceptance/delivery_acceptance_20260505_clean/` |
| 公平对比 | `results/fair_external_comparison/fair_external_secure_static_20260505_clean/` |
| static calibration | `results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/` |
| heldout238 | `artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout238_spuaware_nonisolated_20260507_1/` |
| keep-mask scaling | `results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_scaling_20260509_1/` |
| runtime efficiency | `results/e2e_runtime_efficiency/e2e_aanone_exactln_clip0_spuaware_heldout_20260508_1/` |

| depth evidence (20260510_full) | `results/secure_static_train_depth_evidence/secure_static_train_depth_20260510_full/` |
| secure pruning smoke1 | `artifacts/server_pipeline_run/secure_pruning_spu_smoke1_partylocal_secret_20260510/e2e_secure_poc/` |

### smoke8 batch4 结果（batch_size=4）


### smoke8 batch8 结果（batch_size=8，全部 8 样本 1 次 spu_run）

| 指标 | 值 |
|---|---|
| run_name | `secure_pruning_spu_smoke8_batch8_partylocal_secret_20260510_1` |
| spu_batch_size | 8 |
| sample_count | 8 |
| spu_run 次数 | 1 |
| elapsed_sec | 906.244 |
| sec_per_sample | **113.3** |
| finite_logits | true |
| argmax predictions | `[1,1,1,1,1,0,1,1]` |
| argmax match vs batch1 | **完全一致** |
| max_logit_diff vs batch1 | 0.0235 |
| 相对 batch1 (213.9s/sample) | **1.89x 加速** |
| 相对 batch4 (160.6s/sample) | **1.42x 加速** |

| 指标 | 值 |
|---|---|
| run_name | `secure_pruning_spu_smoke8_batch4_partylocal_secret_20260510_1` |
| spu_batch_size | 4 |
| sample_count | 8 |
| spu_run 次数 | 2 |
| elapsed_sec | 1284.939 |
| sec_per_sample | **160.6** |
| finite_logits | true |
| argmax predictions | `[1,1,1,1,0,0,1,1]` |
| 相对 batch1 (213.9s/sample) | **1.33x 加速** |




### smoke8 batch8 depth10 结果（depth=10 + batch_size=8）

| 指标 | 值 |
|---|---|
| run_name | `secure_pruning_spu_smoke8_batch8_depth10_partylocal_secret_20260510_1` |
| depth_limit | 10 |
| spu_batch_size | 8 |
| sample_count | 8 |
| elapsed_sec | 804.35 |
| sec_per_sample | **100.5s** |
| finite_logits | true |
| threshold_match vs d12-batch8 | **1.0 (8/8)** |
| argmax_match vs d12-batch8 | 0.875 (7/8) |
| logits_max_abs_error vs d12-batch8 | 0.147 |
| 相对 baseline batch1 depth12 | **2.13x 加速** |
| 相对 batch8 depth12 | **1.13x 加速** |

### 效率优化最终汇总

| 配置 | batch_size | depth | spu_run次数 | 总耗时 | sec/sample | 相对 baseline |
|---|:---:|:---:|:---:|---:|---:|---:|
| baseline | 1 | 12 | 8 | 1711.1s | 213.9s | 1.00x |
| batch4 | 4 | 12 | 2 | 1284.9s | 160.6s | 1.33x |
| batch8 | 8 | 12 | 1 | 906.2s | 113.3s | 1.89x |
| **batch8+d10** | **8** | **10** | **1** | **804.3s** | **100.5s** | **2.13x** |

- batch16 depth10 尝试失败：SPU 节点 OOM（62GB RAM 不足以承载 16 样本同时 in-SPU 计算）
- depth=10 为什么可行：去掉最后 2 个 block 减少 late-block 累积数值漂移，argmax 反而提升 +3.24pp，threshold 仅下降 0.57pp
- 最终推荐配置：**batch8 + depth10**

### smoke8 效率 scaling 汇总

| 配置 | batch_size | spu_run次数 | 总耗时 | sec/sample | 相对 batch1 |
|---|---|---|---|---|---|
| smoke8-batch1 | 1 | 8 | 1711.1s | 213.9s | 1.00x (baseline) |
| smoke8-batch4 | 4 | 2 | 1284.9s | 160.6s | **1.33x** |
| smoke8-batch8 | 8 | 1 | 906.2s | **113.3s** | **1.89x** |

- 主要节约来源：共享 SPU 通信协议开销，batch1 每次 spu_run 含约 28s 协议初始化
- batch4 spu_run#1: 650s（含 JIT compile + 4 samples），spu_run#2: 621s（纯计算）
- batch1 per spu_run: 212s/sample（JIT 已缓存，主要是 SPU 安全计算）
- PredictorLG + bitonic sort 在 SPU 内部的计算量与 batch_size 线性相关
