# 重跑验收闭环（2026-05-19）

## 1. 当前状态

- **必须重跑项**：已完成
- **建议重跑项**：已完成
- **当前仍存在的约束**：已记录，不再阻塞正式口径

当前正式主线现锁定为：

- 医疗：`dynamic secure pruning + full privacy + dynamic-path threshold calibration`
- 金融：`LRD rank192 merged + dynamic secure pruning + full privacy`

## 2. 本轮完成项

### 2.1 医疗 `32` 样本复核集：动态全隐私正式复核

已完成：

- 内部准备标识：`med_secure_pruning_smoke32_prepare_20260519_1`
- 内部运行标识：`med_secure_pruning_smoke32_batch8_depth12_20260519_1`
- 内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_20260519_1`

结果：

- 批次规模 `12`、深度 `12` 的配置在当前服务器上 OOM，已明确定性为资源约束，不再继续作为本轮正式复核前提
- 批次规模 `8`、深度 `12` 的配置成功建立 `32` 样本 dynamic 语义基线
- 批次规模 `8`、深度 `10` 的配置成功建立 adopted dynamic mainline 的 `32` 样本正式复核配置
- `全量验证集 dynamic threshold calibration` 已额外补齐：
  - depth12：`best_threshold = 0.6226428151`
  - depth10：`best_threshold = 0.6619606018`

### 2.2 金融动态-静态同条件配对复核

已完成：

- 内部准备标识：`finance_dynamic_static_pair_prepare_20260519_1`
- 内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`
- 内部运行标识：`finance_lrd_rank192_true_static_partylocal_secret_smoke8_20260519_1`

结果：

- dynamic 臂：`105.16s/sample`，`100%`，隐私全过
- true static 臂：`103.64s/sample`，`100%`，隐私全过
- dynamic vs static 候选预测完全一致：`argmax/threshold match = 1.0 / 1.0`

## 3. 本轮关键结论

### 3.1 医疗

- dynamic secure pruning 的 SPU 语义闭环已经在 `32` 样本复核集上补齐
- 当前服务器上正式可复现的复核配置应写成：
  - 批次规模 `8`、深度 `10`
  - `86.91s/sample`
  - `32` 样本复核集阈值准确率 `93.75%`（使用 depth10 dynamic 全量验证集阈值）
- `69.57s/sample` 仍保留为小样本最快工程配置，不再单独承担大样本正式复核配置的角色

### 3.2 金融

- 当前 full-privacy 运行口径已被澄清：dynamic secure pruning 与 true static no-pruning 都能显式跑通
- 由于 dynamic full-privacy 已成立，且 static 相对 dynamic 只有极小速度优势，因此金融正式默认主线改为 dynamic secure pruning
- static no-pruning 改记为 fallback / 对照线

## 4. 当前仍存在的非阻塞约束

| 约束 | 当前判断 |
|---|---|
| 医疗 `32` 样本复核集、批次规模 `12`、深度 `12` OOM | 已记录为当前服务器内存约束，不阻塞正式口径 |
| 医疗 dynamic path 不能直接沿用 static threshold | 已通过全量验证集 dynamic threshold calibration 解决 |
| 金融 static vs dynamic 差距很小 | 已记录，这正是 static 不再作为正式默认主线的原因 |

## 5. 当前不再需要继续补跑的项

- 医疗 `32` 样本复核集：
  - 已有 `depth12` 语义基线 + `depth10` adopted 正式复核配置 + 全量验证集 dynamic threshold
- 金融 dynamic-vs-static：
  - 已有同条件 pair 结果，不再缺“为什么默认选 dynamic”的正式证据

## 6. 后续若再加预算，优先级如何排

本轮已经满足“正式口径闭环”。若后续还有额外预算，优先级建议为：

1. 金融更大样本 dynamic 正式复核（增强论证硬度，不阻塞当前结论）
2. 医疗更大批量 / 更高并发 profile（工程扩展，不改变正式主线）
3. 网页展示数据与最终报告口径同步
