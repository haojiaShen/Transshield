# 2026-05-19 重建总记录：方法清单

## 1. 当前锁定口径

- 医疗正式主线：`DynamicViT + dynamic secure pruning + PredictorLG in-SPU + full privacy`
- 金融正式主线：`LRD rank192 merged bundle + dynamic secure pruning + full privacy`
- 两域共同隐私底线：
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`
- OpenBumbleBee 口径：
  - 当前正式系统使用的是 `OpenBumbleBee / SPU 2PC runtime + e2e_secure_vit integration`
  - `network_kth_bridge` / `tie_policy_bridge` / `transshield_openbumblebee_pipeline.py` 只保留为历史对比链

## 2. 2026-05-19 新增运行模式澄清

- 当前 `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py run --runtime spu` 的默认行为，不是简单读取 `args_snapshot.json.use_mask_pruning`。
- 当没有外部保留掩码（keep-mask）载荷时，SPU 主入口默认走：
  - `spu_pruning_mode = secure_internal_pruning`
  - 即 `PredictorLG + kth_threshold + tie-resolution` 在 SPU 内部执行。
- 为了把“真正静态 no-pruning”与“动态 secure pruning”彻底分开，本次已新增显式运行开关：
  - `--spu-pruning-mode secure_internal_pruning`
  - `--spu-pruning-mode static_no_pruning`
- 因此，不能再仅凭 bundle 里的 `use_mask_pruning = false` 推断“当前 full-privacy 运行一定是静态线”。

## 3. 证据优先级

1. 本目录 `docs/report_evidence/`
2. `results/report_evidence/`
3. `docs/current_work_status.md`
4. `docs/delivery_experiment_summary_20260510.md`
5. `docs/handoff-next.md`
6. `artifacts/` 下仍保留的 bundle 元数据与 `artifacts/web_demo_assets/demo_content_summary.json`

说明：

- 当前 Git 工作树中，多数历史 `artifacts/server_pipeline_run/...` 目录原本已清理；本次重跑后，只把关键 JSON / PT / log 拉回本仓。
- 后续正式长报告，优先引用本目录和 `results/report_evidence/` 中的新闭环证据。

## 4. 方法总表

| 轨道 | 当前角色 | 动态剪枝 | 完全隐私 | 代码入口 | 运行入口 | 直接证据 |
|---|---|---:|---:|---|---|---|
| 医疗原始动态语义参考线 | 语义基线 | 是 | 否 | `models/dyvit.py` | 训练 / CPU 参考流程 | `PredictorLG`、`_kth_threshold_compare_network()`、`_select_equal_by_index()` |
| 医疗保留掩码回放线（keep-mask replay） | 精确一致性对照线 | 是 | 否 | `integrations/openbumblebee/e2e_secure_vit/cpu_static_vit.py` | `run_e2e_secure_whole_forward.sh` | `run_runtime_pruning_student_whole_forward_limited()` / `run_external_keep_mask_student_whole_forward_limited()` |
| 医疗 secure pruning 线 | 正式主线 | 是 | 是 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` | `run_e2e_secure_whole_forward.sh` | 医疗动态安全推理正式主线（内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_20260519_1`）与阈值校准目录 |
| 医疗固定结构 exact-LN 控制线 | 工程控制线 | 否 | 部分 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` | `run_e2e_aanone_exactln_clip0_eval.sh` | `91.9847%` 的 static 全量验证集指标 |
| 金融 dynamic secure pruning + LRD rank192 | 正式主线 | 是 | 是 | `transshield_e2e_secure_vit.py`、`spu_static_vit.py` | `run_e2e_secure_whole_forward.sh` | 金融动态安全推理正式主线（内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`） |
| 金融 true static no-pruning + LRD rank192 | 已验证 fallback / 对照线 | 否 | 是 | `transshield_e2e_secure_vit.py --spu-pruning-mode static_no_pruning` | `run_e2e_secure_whole_forward.sh` | 固定结构对照线（内部运行标识：`finance_lrd_rank192_true_static_partylocal_secret_smoke8_20260519_1`） |
| 金融保留掩码历史线（keep-mask） | 跨域历史证据 | 是 | 不作为正式口径 | 通用 `e2e_secure_vit` 路径 | 历史内部运行标识：`finance_keepmask_smoke8_20260511_131750` | `artifacts/frozen_bundle_finance_fraud_v3_20260511/README.md` |
| OpenBumbleBee bridge compare 链 | 历史/对比链 | 是 | 不作为正式口径 | `tools/transshield_openbumblebee_pipeline.py`、`integrations/openbumblebee/transshield_*_bridge/` | pipeline prepare / replay / check | 历史 bridge / replay 文档与工具链 |

## 5. 医疗线边界

### 5.1 原始动态语义参考线

- 作用：保留 DynamicViT 原始 `score -> threshold -> keep/drop` 语义。
- 结论：它是语义来源，不是最终完全隐私推理路径。

### 5.2 保留掩码回放线（keep-mask replay）

- 作用：先在参考路径得到按样本变化的保留掩码（keep-mask），再把它注入 whole-forward 合同执行。
- 结论：它解决“语义精确保回”，但 PredictorLG 不在 SPU 内，因此不能替代医疗正式主线。

### 5.3 secure pruning 线

- 作用：把 `PredictorLG + kth_threshold + tie-resolution` 全部搬进 SPU。
- 2026-05-19 新证据：
  - `32` 样本复核集、批次规模 `8`、深度 `12`：`102.77s/sample`，对 CPU dynamic reference `argmax/threshold match = 1.0 / 1.0`
  - `32` 样本复核集、批次规模 `8`、深度 `10`：`86.91s/sample`，对 `depth12` secure pruning `argmax/threshold match = 1.0 / 1.0`
  - `524` 条全量验证集动态阈值校准：
    - depth12：`best_threshold = 0.6226428151`，`best_threshold_accuracy = 92.7481%`
    - depth10：`best_threshold = 0.6619606018`，`best_threshold_accuracy = 92.7481%`
- 结论：医疗正式主线继续锁定为 dynamic secure pruning；但正式指标必须配套“dynamic-path public threshold calibration”，不能再直接套用 static bundle 阈值。

## 6. 金融线边界

- 当前正式金融 bundle：`artifacts/frozen_bundle_finance_lrd_rank192_20260515`
- 训练元数据仍保留：
  - `use_mask_pruning = false`
  - `secure_static_skip_pruning = true`
  - `lrd_rank = 192`
  - `lrd_merged = true`
- 但 2026-05-19 同条件复跑已经明确证明：
  - `secure_internal_pruning` 动态臂真实可跑且完整隐私成立
  - `static_no_pruning` 静态臂也可跑且完整隐私成立
  - 两臂在同一 `8` 样本平衡压力验证集上都达到 `100%`，预测完全一致
  - 静态臂仅比动态臂快 `1.52s/sample`
- 结论：
  - 金融域现在不再写成“只有静态主线”
  - 正式默认口径改为：`LRD rank192 merged + dynamic secure pruning + full privacy`
  - `true static no-pruning` 降级为“已验证 fallback / 对照线”

## 7. 旧展示口径与本次重建口径的关系

- 2026-05-19 之前，`docs/transshield_innovation.md` 与 `artifacts/web_demo_assets/demo_content_summary.json` 曾经承载过旧展示叙事：
  - 医疗强调 `69.57s/sample`
  - 金融强调 `117.80s/sample static`
- 截至本次重建收口，这两份文件已经同步到新口径：
  - 医疗首页正式指标：`92.7481%` + `86.91s/sample`
  - 金融首页正式指标：`100%` + `105.16s/sample` + `68.39%`
- 今后正式长报告统一按以下规则取证：
  - 医疗：先写 dynamic secure pruning + public threshold calibration + full privacy
  - 金融：先写 dynamic secure pruning + LRD rank192 + full privacy
  - static no-pruning 只作为 fallback / 对照线解释，不再与正式主线并列
