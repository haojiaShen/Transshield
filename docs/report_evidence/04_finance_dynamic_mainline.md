# 金融正式主线：LRD rank192 动态安全剪枝与完全隐私推理

## 1. 正式结论

- 金融正式主线现锁定为：`LRD rank192 merged bundle + dynamic secure pruning + full privacy`
- 当前正式 bundle 仍是：`artifacts/frozen_bundle_finance_lrd_rank192_20260515`
- `true static no-pruning` 已被重新跑通，但不再作为正式默认主线

## 2. 为什么今天要改口径

此前“金融是 static whole-forward 主线”的判断，主要来自：

- `args_snapshot.json` 中的
  - `use_mask_pruning = false`
  - `secure_static_skip_pruning = true`
- 以及旧文档中“金融主展示线 = LRD rank192 merged”的写法

但 2026-05-19 重新核对在线运行结果与结果 JSON 后已经确认：

- 当前 finance 在线安全运行的 JSON 明确写着：
  - `backend = jax_spu_secure_pruning_forward_backend_v0`
  - `forward_scope = student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path`
- 当前 SPU 主入口默认行为是：
  - 没有外部保留掩码（keep-mask）时，默认走 `secure_internal_pruning`
- 因此，仅凭 `args_snapshot.json.use_mask_pruning = false` 不能再断言“当前 full-privacy 金融主线一定是静态”

## 3. 正式主线固定配置

| 项目 | 固定口径 |
|---|---|
| bundle | `artifacts/frozen_bundle_finance_lrd_rank192_20260515` |
| 运行入口 | `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh` |
| 运行配置 | `configs/openbumblebee/2pc.json` |
| `spu_pruning_mode` | `secure_internal_pruning` |
| 参数模式 | `secret` |
| 输入边界 | `party_local_debug_share_load` |
| 模型压缩 | `lrd_rank = 192`, `lrd_merged = true` |
| 算子族 | `uniform attention + fixed_square + exact LN` |
| reveal policy | `final_logits_only` |

## 4. 2026-05-19 同条件复跑结果

### 4.1 `8` 样本压力验证集：动态臂正式主线

- 内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`
- 结果：
  - `elapsed_sec = 841.28`
  - `sec_per_sample = 105.16`
  - `argmax_accuracy_vs_targets = 100%`
  - `threshold_accuracy_vs_targets = 100%`
  - 对 CPU dynamic reference：`argmax/threshold match = 1.0 / 1.0`
  - `backend = jax_spu_secure_pruning_forward_backend_v0`
  - `forward_scope = student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path`
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`

### 4.2 `8` 样本压力验证集：固定结构对照线

- 内部运行标识：`finance_lrd_rank192_true_static_partylocal_secret_smoke8_20260519_1`
- 结果：
  - `elapsed_sec = 829.09`
  - `sec_per_sample = 103.64`
  - `argmax_accuracy_vs_targets = 100%`
  - `threshold_accuracy_vs_targets = 100%`
  - 对 CPU static reference：`argmax/threshold match = 1.0 / 1.0`
  - `backend = jax_spu_static_whole_forward_backend_v0`
  - `forward_scope = student_patch_embed_blocks_head_without_runtime_pruning_predictor_path`
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`

### 4.3 dynamic vs static 同条件比较

- 同一 bundle
- 同一 `finance_smoke8_balanced_evenly_spaced.txt`
- 同一 `party_local_debug_share_load`
- 同一 `spu_params_mode = secret`
- 同一 `reveal_policy = final_logits_only`

结果：

- dynamic vs static 候选预测完全一致：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
- 静态臂只比动态臂快：
  - `105.16 - 103.64 = 1.52s/sample`

### 4.4 动态主线同配置通信复核

- 证据：`results/report_evidence/mainline_communication_profile.json`
- 通信复核内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_commprofile_20260519_1`
- 结果：
  - `elapsed_sec = 849.26`
  - `sec_per_sample = 106.16`
  - `dual_total_bytes = 27168265566`
  - `dual_total_gib = 25.30`
  - `per_sample_gib = 3.16`
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`
- 说明：
  - 这批通信量来自同配置复核运行后立即解析的 `Link details` 计数器。
  - 运行环境仍是 colocated `2PC` 原型链，因此适合做当前报告主表的同口径通信补充，不代表真实异地部署流量。

## 5. 为什么正式默认改成动态主线

本次改口径，不是因为静态臂跑不通；相反，静态臂已经被显式跑通。

正式默认改成动态 secure pruning 的原因有三条：

1. **符合当前最终用户要求**
   - 两个领域最终使用的模型都要能满足“动态剪枝 + 完全隐私”

2. **当前代码真实运行口径已经能支持金融 dynamic secure pruning**
   - 不是“理论支持”，而是 2026-05-19 同条件重跑已拿到正式结果

3. **静态臂相对动态臂没有形成足够强的默认化理由**
   - 预测完全一致
   - 速度只快 `1.52s/sample`
   - 不足以支撑“再维护第二条正式主线”

## 6. 静态臂现在的正确角色

- `true static no-pruning` 仍然有价值
- 但价值是：
  - fixed-shape fallback
  - 对照线
  - 极端部署场景下的保守选项
- 它不再与动态金融主线并列为正式默认方案

## 7. 当前 repo 内可直接打开的 live 证据

- `artifacts/web_demo_runs/web_demo_finance_fraud-sample_000000_1541ae9a_finance_secret_uniform_live/e2e_secure_poc/e2e_static_whole_forward_candidate_from_server.json`

该文件已经明确显示：

- `backend = jax_spu_secure_pruning_forward_backend_v0`
- `forward_scope = student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path`
- `spu_pruning_mode` 当时还未显式落盘，但行为上就是动态 secure pruning

## 8. 当前正式写法

- 金融正式主指标：
  - `argmax_accuracy / threshold_accuracy`
  - `sec/sample`
  - 双向总通信量
  - 参数压缩比例
  - 三个隐私字段是否通过
- 正式主线：
  - `LRD rank192 merged + dynamic secure pruning + full privacy`
- fallback / 对照线：
  - `LRD rank192 merged + true static no-pruning + full privacy`
