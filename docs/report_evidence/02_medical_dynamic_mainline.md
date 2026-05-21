# 医疗正式主线：动态安全剪枝与完全隐私推理

## 1. 正式结论

- 医疗正式主线继续锁定为：`DynamicViT + dynamic secure pruning + PredictorLG in-SPU + full privacy`
- 2026-05-19 新闭环后，正式主线必须同时带上：
  - `party_local_debug_share_load`
  - `spu_params_mode = secret`
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`
  - `dynamic-path public threshold calibration`
- 对外报告时，医疗正式主线不再只写 `69.57s/sample`；必须同时写清：
  - 小样本最快工程配置：`12` 样本工程复核集、批次规模 `12`、深度 `10`，`69.57s/sample`
  - 当前服务器上已补齐的大样本正式复核配置：`32` 样本复核集、批次规模 `8`、深度 `10`，`86.91s/sample`

## 2. 正式主线固定配置

| 项目 | 固定口径 |
|---|---|
| bundle 主来源 | `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507` |
| 运行入口 | `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh` |
| 集成入口 | `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` |
| SPU forward 实现 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` |
| 动态剪枝 | 开启 |
| PredictorLG 位置 | SPU 内部 |
| 输入边界 | `party_local_debug_share_load` |
| 参数模式 | `secret` |
| 算子族 | `exact LN + uniform attention + fixed_square clip0` |
| reveal policy | `final_logits_only` |

## 3. 为什么这条线现在算正式闭环

### 3.1 隐私闭环

- `runtime_pruning_keep_mask_pt = null`
- `host_plaintext_pixel_values_materialized = false`
- `host_model_params_materialized = false`
- `reveal_policy = final_logits_only`

说明：

- 这表示医疗最终系统不再依赖外部保留掩码（keep-mask）注入。
- PredictorLG / kth-threshold / tie-resolution 全都在 SPU 内部完成。

### 3.2 语义闭环

- `32` 样本复核集、批次规模 `8`、深度 `12` 的运行结果，对 CPU dynamic reference：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
- `32` 样本复核集、批次规模 `8`、深度 `10` 的运行结果，对 `32` 样本复核集、批次规模 `8`、深度 `12` 基线：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`

说明：

- 这证明 2026-05-19 新补的 `32` 样本复核配置已经把“动态 secure pruning 语义是否仍成立”补齐。

### 3.3 指标闭环

医疗动态主线的正式指标，必须分两层写：

1. **语义 / 隐私运行指标**
   - `sec/sample`
   - `argmax/threshold match`
   - 隐私字段是否通过
2. **任务精度指标**
   - 不能再直接套用 static bundle 里的 `0.3577311039`
   - 必须使用 dynamic-path 的公开阈值校准结果

## 4. 2026-05-19 新增关键证据

### 4.1 `32` 样本复核集、批次规模 `12`、深度 `12`：当前服务器 OOM

- 内部运行标识：`med_secure_pruning_smoke32_batch12_depth12_20260519_1`
- 结果：
  - SPU 进程被 OOM kill
  - gRPC 报错：`Socket closed`
- 结论：
  - 当前服务器上，`32` 样本复核集 + 批次规模 `12` + 深度 `12` 不可复现
  - 正式大样本复核配置需要降到批次规模 `8`

### 4.2 `32` 样本复核集、批次规模 `8`、深度 `12`：大样本语义基线

- 内部运行标识：`med_secure_pruning_smoke32_batch8_depth12_20260519_1`
- 结果：
  - `elapsed_sec = 3288.79`
  - `sec_per_sample = 102.77`
  - 对 CPU dynamic reference：`argmax/threshold match = 1.0 / 1.0`
  - 隐私字段全通过
- 说明：
  - 这是当前服务器上可复现的 `32` 样本 dynamic secure pruning 语义基线

### 4.3 `32` 样本复核集、批次规模 `8`、深度 `10`：当前可复现正式复核配置

- 内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_20260519_1`
- 结果：
  - `elapsed_sec = 2781.02`
  - `sec_per_sample = 86.91`
  - 对 `32` 样本复核集、批次规模 `8`、深度 `12` 基线：`argmax/threshold match = 1.0 / 1.0`
  - 隐私字段全通过
- 说明：
  - 这是 2026-05-19 之后，医疗 adopted dynamic mainline 在当前服务器上的正式大样本复核配置

### 4.4 `32` 样本复核集、批次规模 `8`、深度 `10` 的同配置通信复核

- 证据：`results/report_evidence/mainline_communication_profile.json`
- 通信复核内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_commprofile_20260519_1`
- 结果：
  - `elapsed_sec = 2849.94`
  - `sec_per_sample = 89.06`
  - `dual_total_bytes = 90694658258`
  - `dual_total_gib = 84.47`
  - `per_sample_gib = 2.64`
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`
- 说明：
  - 这批通信量来自同配置复核运行后立即解析的 `Link details` 计数器。
  - 运行环境仍是当前 `127.0.0.1` colocated 原型链，因此它适合用作报告主表中的同口径通信补充，不应直接外推为真实广域网部署时延。

## 5. dynamic-path public threshold calibration

### 5.1 为什么必须单独校准

- 当前 bundle 内置阈值 `0.3577311039` 来自 static control line。
- 直接把这个阈值套到 dynamic secure pruning 输出上，会导致：
  - `32` 样本复核集、批次规模 `8`、深度 `12`：`32/32` 全部判成 class1，`threshold_accuracy = 50%`
  - `32` 样本复核集、批次规模 `8`、深度 `10`：同样 `threshold_accuracy = 50%`
- 这不是 secure runtime 语义错误，而是“动态路径与静态路径的输出边界不同”。

### 5.2 全量验证集 depth12 dynamic threshold

- 证据：`results/report_evidence/medical_dynamic_threshold_calibration/dynamic_fullval_threshold_summary.json`
- 结果：
  - `sample_count = 524`
  - `default_threshold_in_bundle = 0.3577311039`
  - `argmax_accuracy = 74.6183%`
  - `best_threshold = 0.6226428151`
  - `best_threshold_accuracy = 92.7481%`

### 5.3 全量验证集 depth10 dynamic threshold

- 证据：`results/report_evidence/medical_dynamic_threshold_calibration/dynamic_fullval_depth10_threshold_summary.json`
- 结果：
  - `sample_count = 524`
  - `default_threshold_in_bundle = 0.3577311039`
  - `argmax_accuracy = 74.2366%`
  - `best_threshold = 0.6619606018`
  - `best_threshold_accuracy = 92.7481%`

### 5.4 把全量验证集 dynamic threshold 回代到 `32` 样本复核集

| 配置 | 使用阈值 | `32` 样本复核集阈值准确率 |
|---|---:|---:|
| depth12 secure pruning | `0.6226428151` | `93.75%` |
| depth10 secure pruning | `0.6619606018` | `93.75%` |

结论：

- 医疗 dynamic secure pruning 主线在 `32` 样本复核配置上并不是“50% 精度”。
- 正确口径应当是：
  - dynamic secure pruning 语义与隐私闭环成立
  - 再配套 dynamic-path public threshold calibration
  - 则全量验证集和 `32` 样本复核集都能回到 `~92.75% / 93.75%`

## 6. 当前正式写法

### 6.1 方法主线

- DynamicViT 原始 pruning 语义来自 `models/dyvit.py`
- secure 语义改写通过 `masking -> F_mux`、`threshold compare -> F_less`
- 最终正式运行链通过 `spu_static_vit.py` 在 SPU 内部执行：
  - `PredictorLG`
  - `kth_threshold`
  - `tie-resolution`

### 6.2 工程主线

- **正式 adopted dynamic line**：`32` 样本复核集、批次规模 `8`、深度 `10` + 动态路径全量验证集阈值 `0.6619606018`
- **小样本最快工程配置**：`12` 样本工程复核集、批次规模 `12`、深度 `10`，`69.57s/sample`
- **当前大样本语义基线**：`32` 样本复核集、批次规模 `8`、深度 `12`
- **当前同配置通信补测**：双向总通信量约 `84.47 GiB`

## 7. 当前不再混写的点

- 不再把 `91.9847%` 的全量验证集阈值准确率直接写成医疗动态主线指标。
  - 它属于 static control line。
- 不再把 `0.3577311039` 当作 dynamic secure pruning 默认阈值。
  - 2026-05-19 已明确证明它是 static-path threshold。
- 不再把保留掩码回放链（keep-mask replay）写成医疗最终系统。
  - 它是 exact 语义对照，不是最终双向隐私主线。
