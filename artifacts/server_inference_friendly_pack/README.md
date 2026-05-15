
# Transshield 推理服务部署包（Server Inference Friendly Pack）

> **角色映射说明**（2026-05-12 更新）：
>
> 本目录中的脚本和配置对应"模型即服务"（Model-as-a-Service）架构：
> - **模型提供方**：拥有并训练模型，将其部署为安全推理服务（内部含 P0/P1 两台 MPC 服务器）
> - **数据使用方**：向推理服务提交待分析数据（如医院提交患者影像），获取分析结果
>
> 文件名/变量名中的 `client` 对应"数据使用方"，`server` 对应"模型提供方内部推理服务"。
> 历史变量名保持不变，以确保运行中的管线不受影响。


`artifacts/server_inference_friendly_pack/` 是当前比赛展示版的默认运行入口。

如果仓库根目录存在同名脚本（例如 `run_secure_selection_mode_profile_compare.sh`），默认只把它当作**兼容 wrapper**；权威实现仍以本目录为准。

当前目录下的大部分 `run_*.sh` 会自动加载：

- `final_compare_env.template.sh`

`final_compare_env.local.sh` 只在显式设置 `TRANSSHIELD_USE_LOCAL_ENV=1` 时加载。默认不加载本机路径，避免把 `/home/...` 的开发环境同步到服务器后污染运行配置。

## 当前主线应如何理解

在使用本目录脚本前，先固定当前口径：

- 当前主线仍是 `ViT / DynamicViT`，不是因为 CNN 对胸片分类无效，而是因为 token-level pruning boundary 更适合作为当前 `F_less / F_mux` 主创新的载体。
- 当前并没有放弃明文 pruning；正式 secure-facing 语义只是把“直接删 token”改写成 masking-friendly `keep/zero` 表达。
- 当前 “动态 pruning” 的动态性来自样本级、stage 级 `kth` 边界，而不是一个全局固定阈值；它也不是最终二分类评测阈值。
- 当前 `OpenBumbleBee / SPU` 同时承接两条 related 路线：
  - 当前正式交付线：pruning boundary `secure sidecar + replay`
  - 后续扩展线：whole-forward secure ViT，再迁入当前 masking-pruning 语义
- “利用稀疏 token 压低开销” 应被理解为上述 secure backend 下的优化子项，而不是另一条平行主方法。
- `CNN + ViT` hybrid 不属于当前主线；`embedding / position encoding` secure 优化只属于后续 `P2` 候选。

## 当前默认入口差异

当前目录下的默认运行 bundle 已统一到 current delivery line：

- `run_web_demo.sh` 默认 bundle：
  - `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- `final_compare_env.template.sh`
- `run_full_final_comparison_suite.sh`
- `run_fair_external_comparison.sh`

都默认指向同一个 current delivery bundle。

历史 `verified_tracka` 只保留为：

- benchmark / provenance
- 旧静态成绩板来源
- 不再作为当前 wrapper 默认值

如果你只想做一件事，优先使用：

- 小样本链路验证：`run_full_final_comparison_smoke.sh`
- 完整对比链运行：`run_full_final_comparison_suite.sh`
- 前后端交互演示：`run_web_demo.sh`
- `e2e secure` 新线最小骨架：`run_e2e_secure_poc.sh`
- `e2e whole-forward` 集成入口：`run_e2e_secure_whole_forward.sh`
- 当前 full-model plaintext vs static residual-gap 诊断：`run_fullval_plaintext_static_gap.sh`
- 当前 reference sidecar replay 语义闭环：`run_fullval_reference_sidecar_replay.sh`
- 当前 runtime-pruning keep-mask 注入 SPU 验证：`run_e2e_runtime_pruning_keepmask_bridge.sh`
- 当前 delivery line 验收汇总：`run_delivery_acceptance_report.sh`
- 当前 `P1` 第一项证据：`run_stage_cost_risk_model.sh`
- 当前 `P1` 第二项证据：`run_secure_static_depth_evidence.sh`
- 当前 `P1` 第二项 paired control：`run_secure_static_depth_pair_study.sh`
- 当前 `P1` 第三项入口：`run_protocol_aware_pruning_train.sh`
- 当前 `P1` 第三项长配对证据：`run_protocol_aware_pruning_pair_study.sh`
- 当前 `P1` 第四项长配对证据：`run_distill_compensation_pair_study.sh`
- 当前 accuracy-profile 长配对入口：`run_accuracy_profile_pair_study.sh`
- 统一 secure benchmark 外部 proxy 对比：`run_standardized_secure_external_benchmark.sh`

---

## 1. 推荐运行顺序

### 明文主链路

1. `run_plaintext_eval.sh baseline`
2. `run_plaintext_eval.sh modified`
3. `run_plaintext_model_compare.sh`

### secure 闭环链路

4. `run_secure_export_inputs.sh`
5. `run_secure_pipeline.sh cpu|spu`
6. `run_secure_replay.sh`
7. `run_secure_score_compare.sh`
8. `run_final_comparison_report.sh`
9. `run_delivery_acceptance_report.sh`
10. `run_stage_cost_risk_model.sh`
11. `run_secure_static_depth_evidence.sh`
12. `run_secure_static_depth_pair_study.sh suite`
13. `run_protocol_aware_pruning_train.sh recipe`
14. `run_protocol_aware_pruning_train.sh epoch1`
15. `run_protocol_aware_pruning_report.sh`
16. `run_protocol_aware_pruning_pair_study.sh suite`
17. `run_distill_compensation_pair_study.sh suite`
18. `run_accuracy_profile_pair_study.sh suite`

### 一键快捷方式

- `run_full_final_comparison_smoke.sh`
- `run_full_final_comparison_suite.sh`
- `run_selected_image_secure_suite.sh`

### `P1` 证据补齐

- `run_stage_cost_risk_model.sh`
  - 作用：基于当前 delivery suite 与 secret guarded eval，生成阶段级 `cost / risk` 报告；
  - 重点输出：
    - 各 pruning stage 的 active-token 负载占比；
    - boundary tie 压力；
    - strict margin 与 secure kth 数值误差的关系；
    - stage-level 近似 sidecar 成本分解；
  - 它不会改模型，只会把当前主线的 secure boundary 证据补齐。

- `run_secure_static_depth_evidence.sh`
  - 作用：把当前 `secure_static_train_depth` 的证据收束成正式报告；
  - 重点输出：
    - 当前官方 bundle 的 static-depth 训练口径；
    - retained baseline 与 current official line 的 full-val 对照；
    - current official line 与 deployable secret path 的承接关系；
    - 当前 paired control 历史上为什么缺失，以及现在是否已经补上；
  - 它会明确区分 “deployment-aligned evidence” 和 “single-factor causal proof”，避免误写结论。

- `run_secure_static_depth_pair_study.sh`
  - 作用：把 `secure_static_train_depth` 的单因子 paired control 收束成正式配对流程；
  - baseline / candidate 保持同一：
    - base bundle
    - teacher
    - 数据
    - `uniform + fixed_square`
    - `secure_static_skip_pruning=true`
    - distill 配置
  - 默认只改：
    - baseline：`secure_static_train_depth=0`
    - candidate：`secure_static_train_depth=12`
  - 然后自动完成：
    - threshold search / eval
    - plaintext checkpoint eval
    - paired compare report
  - 当前已经有第一条正式 paired result：
    - pair：`secure_static_depth_pair_epoch1_20260506_depth12a`
    - judgement：`no_clear_depth_benefit_yet`
    - `threshold_accuracy delta = -1.5267 pt`
    - `auc delta = -0.0116729`
    - `argmax_accuracy delta = +4.0076 pt`
  - 当前也已经有第一条更长一点的 follow-up：
    - pair：`secure_static_depth_pair_epoch3_20260506_depth12b`
    - judgement：`no_clear_depth_benefit_yet`
    - `threshold_accuracy delta = -0.9542 pt`
    - `auc delta = -0.0097496`
    - `argmax_accuracy delta = +5.5344 pt`
  - 因此这条 wrapper 现在的角色已经从“补入口”转成“给出正式单因子结论”：当前 paired control 已补齐，但更深 train-depth 仍未形成明确收益。

- `run_protocol_aware_pruning_train.sh`
  - 作用：把现有 `pruning_margin_*` 训练接口正式收束成 protocol-aware 训练入口；
  - `recipe` 模式会先根据 `stage_cost_risk_report.json` 生成当前 clean recipe；
  - `print-env` 模式会只解析并打印当前真正生效的 `PRUNING_MARGIN_*`，不启动训练；
  - `debug80` 用于接线与稳定性检查；
  - `epoch1` 是当前数据规模下第一条应当产出 `margin_stats` 的最短 run；
  - 若设置 `PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1`，会强制用 recipe 覆写现有 `PRUNING_MARGIN_*`；
  - 若不强制覆写且现有环境变量与 recipe 不一致，脚本会打印显式告警，避免静默误跑；
  - 它不会自动宣称收益提升，只负责把 objective 接入当前正式训练线。

- `run_protocol_aware_pruning_report.sh`
  - 作用：解析训练日志里的 `pruning_margin=... margin_stats=[...]`；
  - 用来判断当前 objective 是否真的生效；
  - 如果 `debug80` 没有任何 `margin_stats`，report 会明确指出这通常只是因为当前日志打印 cadence 是 100 step。

- `run_protocol_aware_pruning_pair_study.sh`
  - 作用：把 `protocol-aware pruning objective` 的下一阶段证据收束成一套正式配对流程；
  - 它会固定同一 base bundle / teacher / 数据 / static-depth 训练口径，
    分别跑：
    - baseline：`pruning_margin_weight=0`
    - candidate：指定 `PROTOCOL_AWARE_PROFILE`
  - candidate 分支会固定开启 `PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1`，防止旧 shell / local env 把 candidate 静默压回 baseline 参数；
  - 然后自动完成：
    - threshold search / eval
    - plaintext checkpoint eval
    - pruning margin log report
    - paired compare report
  - 这条 wrapper 的目标不是直接宣称收益，而是把“更长配对训练证据”变成标准可复现流程。

- `run_distill_compensation_pair_study.sh`
  - 作用：把蒸馏补偿的长期收益验证收束成一套正式配对流程；
  - baseline / candidate 共享同一 base bundle / teacher / 数据 / static-depth 训练口径，
    分别跑：
    - baseline：`CLS_DISTILL_WEIGHT=0.0`、`TOKEN_DISTILL_WEIGHT=0.0`
    - candidate：默认沿用当前 official distill 配置 `1.0 / 0.02`
  - 然后自动完成：
    - threshold search / eval
    - plaintext checkpoint eval
    - distill log report
    - paired compare report
- 这条 wrapper 的目标是直接回答：
  - “当前 official distill 相对 no-distill 参考，到底有没有稳定补偿收益”。

- `run_accuracy_profile_pair_study.sh`
  - 作用：把“不改主模型语义”的 accuracy 修正收束成正式 paired compare；
  - baseline / candidate 共享同一 base bundle、teacher、数据、`uniform + fixed_square`、`secure_static_skip_pruning=true` 与 distill 配置；
  - 支持两类入口：
    - 直接复用 `ACCURACY_PROFILE`
    - 显式设置 baseline/candidate 各自的 `SEED`
    - 显式设置 baseline/candidate 各自的 `AUGMENTATION_PROFILE`
    - 显式设置 baseline/candidate 各自的 `BATCH_SIZE`
    - 显式设置 baseline/candidate 各自的 `CLIP_GRAD`
    - 显式设置 `class_weight_mode / class_weight_power / train_sampler_mode / model_ema`
    - 显式设置 baseline/candidate 各自的 `SMOOTHING`
    - 显式设置 baseline/candidate 各自的 `WEIGHT_DECAY`
    - 显式设置 baseline/candidate 各自的 `LR / MIN_LR`
    - 显式设置 baseline/candidate 各自的 `WARMUP_STEPS`
    - 显式设置 baseline/candidate 各自的 `GROUPA_LR_SCALE`
    - 显式设置 baseline/candidate 各自的 `CLS_TOKEN_FULL_LR`
    - 显式设置 baseline/candidate 各自的 `TRAIN_POS_EMBED`
    - 显式设置 baseline/candidate 各自的 `FREEZE_PATCH_EMBED_PROJ`
    - 显式设置 baseline/candidate 各自的 `FREEZE_PATCH_EMBED_WEIGHT / FREEZE_PATCH_EMBED_BIAS`
    - 显式设置 baseline/candidate 各自的 `PATCH_EMBED_BIAS_INIT_MODE / SKIP_PATCH_EMBED_BIAS_PRETRAINED`
    - 显式设置 baseline/candidate 各自的 `PRETRAINED_FIX_STEP`
    - 显式设置 baseline/candidate 各自的 `CHECKPOINT_NAME` 与 `CHECKPOINT_MODEL_KEY`，用于评估 `checkpoint-best-ema.pth:model_ema` 这类非默认 checkpoint
  - 当前这条轴已经拿到的结论是：
    - `weighted_sqrt_sampler`：`epoch1` 判负；
    - `sqrt_class_weight`：`epoch1` 勉强不差，但 `epoch3` 仍未形成明确收益；
    - `power_inverse_freq=0.15`：`epoch5` 已转负；
    - `power_inverse_freq=0.18`：`epoch5` 已转负；
    - `power_inverse_freq=0.20`：`epoch1 / epoch3 / epoch5` 全部保持 `candidate_eval_not_worse`；
    - `power_inverse_freq=0.22`：`epoch5` 非劣，但不优于 `0.20`；
    - `power_inverse_freq=0.20`：`epoch8` 已转负；
    - `power_inverse_freq=0.22`：`epoch8` 也已转负；
    - `power_inverse_freq=0.25`：`epoch1 + epoch3` 非劣，但 `epoch5` 转负；
    - `MODEL_EMA=true`：`epoch5` 能明显提升 argmax，但 `threshold_accuracy / AUC` 没有提升；
    - `SMOOTHING=0.05`：`epoch1` 能改善 argmax/loss，但 `threshold_accuracy` 持平、`AUC` 略降；
    - `GROUPA_LR_SCALE=1.0`：`epoch1` 保持 `threshold_accuracy` 持平且 `AUC` 小幅改善，但 `epoch3` 转负；
    - `PRETRAINED_FIX_STEP=1`：`epoch1` 有 argmax/AUC 短信号，但 `epoch3` 的 `threshold_accuracy / AUC` 转负；
    - `LR=1e-6`：`epoch1` 只改善 argmax，`AUC / eval_loss` 变差；
    - `WARMUP_STEPS=0`：`epoch1` 只改善 loss，`AUC` 变差；
    - `AUGMENTATION_PROFILE=mpcvit_like`：`epoch1` 只改善 argmax/loss，`threshold_accuracy / AUC` 变差；
    - `FREEZE_PATCH_EMBED_PROJ=true`：`epoch1` 有 threshold/AUC 短正信号，但 `epoch3` 未延续，AUC、argmax 和 loss 转差；
    - `FREEZE_PATCH_EMBED_WEIGHT=true`：复现 projection freeze 的短正信号，但 `epoch3` 未延续；
    - `FREEZE_PATCH_EMBED_BIAS=true`：`epoch1` 基本等同 baseline；
    - `PATCH_EMBED_BIAS_INIT_MODE=zero`：`epoch1` 只极小改善 loss，不改善正式主指标；
    - `BATCH_SIZE=16`：`epoch1` 改善 loss/argmax，但 AUC 明显转负；
    - `WEIGHT_DECAY=0.01`：`epoch1` 基本等同 baseline；
    - `CLIP_GRAD=2.0`：`epoch1` 只改善 argmax，AUC/loss 转差；
    - `GROUPA_LR_SCALE=0.0`：`epoch1/3/5/8` 均保持 threshold 非劣且 AUC 小幅正向，但 loss/argmax 不一致；
    - `GROUPA_LR_SCALE=0.0 seed1 epoch3`：没有复现原先的小正 AUC 信号；
  - 因此这条 wrapper 当前不应再默认继续沿 `0.18 / 0.20 / 0.22` 这一近邻区间、EMA 轴、`PRETRAINED_FIX_STEP=1`、`LR=1e-6`、`WARMUP_STEPS=0`、`AUGMENTATION_PROFILE=mpcvit_like`、`FREEZE_PATCH_EMBED_PROJ=true`、`FREEZE_PATCH_EMBED_WEIGHT=true`、`FREEZE_PATCH_EMBED_BIAS=true`、`PATCH_EMBED_BIAS_INIT_MODE=zero`、`BATCH_SIZE=16`、`WEIGHT_DECAY=0.01` 或 `CLIP_GRAD=2.0` 追加预算；`GROUPA_LR_SCALE=0.0` 已显示 seed 敏感，也不能直接升级默认值。
  - `SMOOTHING=0.05` 已经验证为“不改善正式主指标”，后续不要默认沿这条继续追加预算。

---

## 2. `smoke` 与 `suite` 的区别

### `run_full_final_comparison_smoke.sh`

- 只取很少样本；
- 默认同时截断 plaintext 与 secure 输入；
- 用来验证脚本、bridge、checker、replay、compare 是否跑通；
- 不适合拿来判断模型最终性能。

### `run_full_final_comparison_suite.sh`

- 使用完整验证集；
- 用来生成正式展示结果；
- 是答辩时应优先引用的结果来源。
- 当前 `run_plaintext_eval.sh modified` 默认直接评估 `BUNDLE_DIR` 内冻结的 `modified_plaintext_model_state_dict.pth`，
  不再依赖 source-run 的 `checkpoint-best.pth`，从而与 `secure replay / fairness` 保持同一 bundle 口径。

当前默认运行入口已经切到：

- `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/`

历史 benchmark bundle 已从当前最终仓移除，不再作为任何默认运行或展示入口。

---

## 3. CPU 与 SPU 的区别

### `run_secure_pipeline.sh cpu`

- 运行 secure sidecar 的本地明文参考执行；
- 主要用于开发调试、reference check 与快速验证；
- **不是我们的最终安全路径**；
- 不是真正的 2PC。

### `run_secure_pipeline.sh spu`

- 运行同一逻辑在 `SPU / OpenBumbleBee` 上的真实安全执行；
- **这是 legacy sidecar 链路中的真实安全执行路径**；
- 它和 whole-forward secure ViT 不是两套互斥系统，而是同一 `OpenBumbleBee / SPU` backend 下的两条不同成熟度路线；
- 最终 Web 主路径已切到浏览器本地分片 + E2E approximate SPU；
- 涉及 secret sharing、协议执行、节点通信与额外性能开销。
- 当前也支持和 `e2e` wrapper 同样的 runtime 稳定性开关：
  - `SPU_RUNTIME_REUSE=1`
  - `SPU_DISABLE_COLOCATED_OPTIMIZATION=1`
- 若服务器上出现 `Socket closed`、`Not connected` 一类 internal link 异常，优先先保留当前命令入口，只额外加 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 复验，不要回退到手动 `tools/transshield_spu_runtime_setup.py start`。

两者结果一致，表示它们实现的是同一函数语义；不表示 CPU 模式本身完成了真实 2PC。
更准确地说：`CPU secure` 不是最终安全路径，它只是 sidecar 函数的本地参考实现；`SPU secure` 是 legacy sidecar 的真实安全执行。当前最终 Web 主路径使用浏览器本地分片 + `run_e2e_secure_whole_forward.sh spu`，默认 profile 是 `secret_depth6_clip0_showcase`。因此后续如果讨论 sparse-token、chunked forward 或 whole-forward secure 优化，应把它们写成 `SPU / OpenBumbleBee` 集成下的子优化，而不是脱离当前主线的独立方法。

---

## 4. 重要环境变量

优先参考：`final_compare_env.template.sh`

常用变量包括：

- `TRAIN_DATA_PATH`
- `VAL_DATA_PATH`
- `SECURE_RUNTIME=cpu` 或 `SECURE_RUNTIME=spu`
- `KTH_SELECTION_MODE=blockwise_exact_kth`、`KTH_SELECTION_MODE=flat_odd_even` 或 `KTH_SELECTION_MODE=phase3_lower_tail`
- 当前默认值是 `blockwise_exact_kth`
- 默认 manifest 是 `results/blockwise_exact_kth_selection_manifest_default.json`
- `flat_odd_even` 保留为旧 reference fallback
- `phase3_lower_tail` 保留为旧实验开关，不再作为默认展示 / 运行口径
- `PHASE3_SELECTION_MANIFEST`
- `PLAINTEXT_MAX_SAMPLES`
- `SECURE_MAX_SAMPLES`
- `SPU_RUNTIME_REUSE`
- `SPU_DISABLE_COLOCATED_OPTIMIZATION`

其中：

- `SECURE_MAX_SAMPLES=8` 与 `PLAINTEXT_MAX_SAMPLES=8` 适合 smoke；
- 完整展示时建议不要截断样本。
- `SPU_RUNTIME_REUSE=1` 适合复用已启动的 SPU runtime；
- 如果遇到 runtime internal link 不稳定，可先加 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 做单变量复验；

---

## 5. 结果文件说明

运行结束后建议优先查看：

- `artifacts/server_pipeline_run/<RUN_NAME>/comparison_report_summary.txt`
- `artifacts/server_pipeline_run/<RUN_NAME>/comparison_report_summary.json`
- `artifacts/server_pipeline_run/<RUN_NAME>/plaintext_vs_secure_score_compare.json`

如果要看更简洁的中文整理，请直接查看：

- `docs/transshield_master_plan_20260505.md`
- `results/fair_external_comparison/fair_external_secure_static_20260505_clean/fair_external_comparison.md`
- `docs/data_source_policy.md`

---

## 6. 其他辅助脚本

### 明文辅助

- `run_single_image_comparison.sh`
- `run_plaintext_predict.sh baseline`
- `run_plaintext_predict.sh modified`
- `run_selected_image_diagnosis.sh`
- `run_web_demo.sh`

### secure 辅助

- `run_cpu_spu_profile.sh`
- `run_e2e_secure_poc.sh`
- `run_e2e_secure_whole_forward.sh`
- `run_e2e_secure_isolated_parallel2.sh`
- `run_secure_static_distill_train.sh`
- `run_secure_selection_mode_profile_compare.sh`
- `run_secure_profile_summary.sh`
- `run_secure_profile_compare.sh`
- `run_standardized_secure_external_benchmark.sh`
- `run_token_pruning_visualization.sh`
- `run_selected_image_secure_diagnosis.sh`

这些脚本主要用于：

- 单图或指定图片列表诊断；
- 运行剖析；
- 演示时做更细粒度的截图与说明。

### `run_e2e_secure_poc.sh`

- 启动当前 `e2e secure inference` 并行研究线的最小骨架；
- 先写出 e2e 边界 contract；
- 再生成数据使用方预处理 `pixel_values` 包；
- 再跑一条 plaintext reference，给后续整网 SPU 对齐做基准；
- 另外会补一条 `static whole-forward` plaintext reference，专门对应“先不做 pruning、先做整网 secure forward”的下一步；
- 同时仓内已提供 whole-forward compare 子命令，后续一旦产出 SPU 候选 `logits`，即可直接对齐；
- 这是早期 POC 入口；当前 Web 主路径已切到浏览器本地分片 + `run_e2e_secure_whole_forward.sh spu` 的展示 profile，不要再把这个 POC 的早期边界当作最终状态。

### `run_e2e_secure_whole_forward.sh`

- 是 `e2e secure inference` 新线的集成 wrapper；
- 当前支持：
  - `prepare`
  - `cpu`
  - `spu`
  - `verify`
  - `audit-shares`
  - `probe-cpu`
  - `probe-spu`
  - `probe-compare`
- `spu` 模式默认仍是实验性 `static whole-forward` JAX/SPU 后端；但现在已经可以通过 `E2E_RUNTIME_PRUNING_KEEP_MASK_PT=/path/to/runtime_pruning_keep_mask_payload.pt`，或直接设置 `E2E_RUNTIME_PRUNING_KEEP_MASK_AUTO_EXPORT=1`，把 external keep-mask 注入这条主入口，从而 replay 当前 runtime pruning 语义；
- keep-mask 主入口当前约束：
  - `E2E_SPU_ATTENTION_POLICY=uniform`
  - `E2E_SPU_PARAMS_MODE=public|secret`
  - 不支持 `E2E_SPU_BLOCK_CHUNK_SIZE>0`
  - `E2E_RUNTIME_PRUNING_KEEP_MASK_AUTO_EXPORT=1` 时必须有可用的 `E2E_INPUT_PT`，因为 keep-mask payload 目前仍先从 plaintext client package 导出
- 若目标是“双向隐私”，不要使用本地明文 backbone / encoder + secure head 的快路径；应使用当前 ViT whole-forward 路径，并从小样本、小深度开始跑 secret-param 模式。普通 chunked forward 仍要求 public 参数；secret 参数下只有实验性 `secret_block_group_stage` 支持 `E2E_SPU_BLOCK_CHUNK_SIZE>0`。
- 当前另有实验性性能模式 `E2E_SPU_PARAMS_MODE=secret_block_group_stage`：patch/head 仍作为 secret，blocks 按 `E2E_SPU_BLOCK_CHUNK_SIZE` 分组作为 secret 参数执行；若未设置 chunk size，默认按 2 个 block 一组。它用于测试减少 staged SPU call 次数是否能提速，尚不是默认稳定路径。
- 为支持 `block9` 数值漂移归因，wrapper 现还暴露两个实验性 ablation 开关：
  - `E2E_SPU_ATTENTION_POLICY=smoothed|standard`，默认 `smoothed` 保持既有行为，`standard` 用普通 softmax 去掉 policy smoothing；
  - `E2E_SPU_ACTIVATION_OVERRIDE=bundle|gelu|fixed_square|learnable_square|learnable_quadratic|learnable_quadratic_gelu_init`，默认 `bundle` 保持当前 bundle 激活；其它值只用于 SPU-only 诊断，不应写成正式同口径 compare；
- `E2E_SPU_BLOCK_CHUNK_SIZE=N` 是 depth11/12 party-local runtime 边界后的实验性无 reveal 图拆分开关：按 N 个 transformer blocks 分段执行 SPU 图，中间 token state 仍保留为 SPU value，只 reveal final logits；默认 `0` 保持原 monolithic graph；
- `block1-smoke` 是当前 depth0 通过、depth1 断链后的 debug-only 子图定位入口，会逐段运行 `patch_pos / norm1 / qkv / attention / mlp / head` 并 reveal 阶段输出，只用于定位第一个断链子图，不属于生产 e2e reveal policy；
- `E2E_REDACT_PRIVATE_INPUT_PATHS=1` 默认开启，会在 `run` 的 `.pt` 与 summary JSON 中隐藏 legacy/P1/P2 私有 share manifest 路径；share 输入模式下 wrapper 也不再传 `--input-pt`，避免 candidate metadata 指回 plaintext client pixel package；只有显式本地 debug 才建议设为 `0`；
- `SPU_RUNTIME_TEMPLATE_PATH` 可指定 runtime setup 使用的 template；并行 isolated worker 必须为每个 worker 使用独立 `CONFIG_PATH`、`SPU_RUNTIME_TEMPLATE_PATH`、`SPU_RUNTIME_STATE_JSON` 和 `SPU_RUNTIME_LOG_DIR`，避免互相改端口导致 `Socket closed`；
- `probe-spu` 会沿用同一个 wrapper 内置的 runtime 自启动逻辑，因此在 block-level drift attribution 时不需要额外手动执行 `tools/transshield_spu_runtime_setup.py start`；
- wrapper 现已额外暴露 runtime 稳定性开关：可用 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 让 `spu/probe-spu` 自动以 `--disable-colocated-optimization` 拉起 runtime；如果同时设置 `SPU_RUNTIME_REUSE=1`，wrapper 也会先检查已存在 runtime 的 colocated 配置是否匹配，不匹配就自动重启，而不是误复用旧节点；
- 推荐顺序：
  1. 先跑 `run_e2e_secure_poc.sh`
  2. 再跑 `run_e2e_secure_whole_forward.sh prepare`
  3. 再跑 `run_e2e_secure_whole_forward.sh cpu`
  4. 再用 `E2E_RUN_MAX_SAMPLES=1 E2E_SPU_BATCH_SIZE=1 run_e2e_secure_whole_forward.sh spu` 做服务器 smoke
  5. 最后用 `E2E_VERIFY_ALLOW_PREFIX=1 run_e2e_secure_whole_forward.sh verify` 对齐 reference 前缀
- 若当前目标是把 runtime pruning 语义迁进 whole-forward / SPU 主入口，推荐最小 smoke：
  - 先准备 `E2E_INPUT_PT`、split share manifests，以及 runtime-pruning plaintext reference
  - 再跑：
    - `E2E_RUNTIME_PRUNING_KEEP_MASK_AUTO_EXPORT=1 E2E_PARTY_LOCAL_SHARE_LOAD=1 E2E_SPU_PARAMS_MODE=secret E2E_SPU_ATTENTION_POLICY=uniform run_e2e_secure_whole_forward.sh spu`
  - 最后用：
    - `E2E_REFERENCE_PT=/path/to/runtime_pruning_reference.pt run_e2e_secure_whole_forward.sh verify`
- 截至 `2026-05-09`，新服务器 `10.204.248.175:9001` 上这条主 wrapper keep-mask 注入线已验证到 `smoke16`：
  - 共同边界：`input_mode=party_local_debug_share_load`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`spu_params_mode=secret`、`runtime_pruning_keep_mask_stage_count=3`
  - `smoke1`：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke1_partylocal_secret_20260509_2/`
    - `elapsed_sec = 233.8283`
    - `logits/probabilities max_abs_error = 0.0025852 / 0.0011970`
    - `argmax / threshold match = 1.0 / 1.0`
  - `smoke8`：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke8_partylocal_secret_20260509_1/`
    - `elapsed_sec = 1612.6744`
    - `logits/probabilities max_abs_error = 0.0027894 / 0.0013530`
    - `argmax / threshold match = 1.0 / 1.0`
  - `smoke16`：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke16_partylocal_secret_20260509_1/`
    - `elapsed_sec = 3203.1877`
    - `logits/probabilities max_abs_error = 0.0026325 / 0.0012865`
    - `argmax / threshold match = 1.0 / 1.0`
  - 汇总报告入口：`bash artifacts/server_inference_friendly_pack/run_e2e_keepmask_scaling_report.sh <output_dir>`
    - 若只给 `output_dir`，脚本会自动搜本地 `keepmask_wholeforward_wrapper_spu_smoke*_partylocal_secret_*` 的 `keepmask_result_summary.json`
    - 也可以手工追加 `summary_json` 列表，强制只汇总指定 runs
    - 当前 `smoke1/8/16/32` 聚合报告：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_scaling_20260509_1/keepmask_scaling_report.md`
    - 当前结论：`privacy_consistent=true`，`all_finite_logits=true`，`all_argmax/threshold_match_ratio_one=true`，`status=scaling_observed_but_needs_more_points`，`sec/sample mean/min/max = 207.56 / 194.63 / 233.83`，`smoke16→smoke32 incremental_sec_per_new_sample = 189.06`
  - 当前解释：
    - keep-mask replay 语义已经并入更正式的 `run_e2e_secure_whole_forward.sh` 主入口；
    - 但这仍是“外部 keep-mask 注入 replay 当前 runtime pruning 语义”，还不是 secure 图内原生 predictor/kth/tie 动态决策；
    - 下一步可沿主 wrapper 继续推进 `smoke64` 或更大样本。
- 当前若要在服务器检查 / 抓取这条线的结果，先固定：
  - `export REPO_ROOT=/data/wyb/Transshield_final`
  - `export RUN_NAME=tracka_e2e_secure_poc_cpu`
  - `export E2E_DIR="$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc"`
  - `export PACK_DIR="$E2E_DIR/whole_forward_pack"`
- 截至 `2026-04-22`，服务器 run `tracka_e2e_secure_poc_cpu` 已验证：
  - `sample_count = 524`
  - `cpu candidate elapsed_sec ≈ 20.73`
  - `logits/probabilities max_abs_error = 0.0`
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
- 截至 `2026-04-23`，这条线的阶段结论已前进到：
  - `depth=0..5 / sample=1 / public params` 的 same-depth smoke 已在服务器通过；
  - 默认 colocated runtime 配置下的一个历史 `depth=6` full run 曾失配并伴随 node/link 异常；

### `run_e2e_secure_isolated_parallel2.sh`

- 是 `secret_blockwise_stage` / no-lsb / public-calibrated whole-forward 的 2-worker isolated runner；
- 用于把 8 个单样本 fresh-runtime 任务拆成两个独立 worker 并行执行，降低墙钟时间；
- 每个 worker 会生成独立 `2pc.json`、`2pc.template.json`、`logs/spu_runtime_ports_*.json`、`logs/spu_nodes_*` 与 `/tmp/transshield_spu_*`，避免两个 worker 互相改端口；
- 默认 worker 分配是 `0 2 4 6` 与 `1 3 5 7`，可通过 `WORKER0_INDICES` / `WORKER1_INDICES` 覆盖；
- 默认仍启用 `enable_mul_lsb_error=false`、`E2E_SPU_PARAMS_MODE=secret_blockwise_stage`、`E2E_SPU_LAYER_NORM_POLICY=public_calibrated`、`E2E_SPU_ATTENTION_POLICY=uniform`、`E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`；
- 输出汇总为 `$RUN_ROOT/sample8_parallel2_no_lsb_summary.json`，判定重点是 `missing=[]`、`outlier_count=0`、`sample_count=8`。

### `run_e2e_secure_secret_isolated_eval.sh`

- 是当前更稳妥的 `secret_blockwise_stage` 单 worker guarded eval 入口；
- 每个样本单独 slice、单独 fresh-runtime 推理，并对每个 attempt 设置 `timeout -k`，避免单个 SPU run 半死后拖住整轮；
- 默认 `MAX_ATTEMPTS=3`、`RUN_TIMEOUT_SEC=900`、`OUTLIER_ABS_THRESHOLD=10`，可按服务器负载调整；
- 对每个 attempt 检查 raw logits（若存在）或最终 logits 的有限性和绝对值范围，异常 attempt 会被 reject 并继续重试；
- 超过重试次数仍没有非 outlier 结果的样本会进入 `unstable_items`，不会混进 `accepted_accuracy`；

### `run_delivery_acceptance_report.sh`

- 是当前 delivery line 的统一验收汇总入口；
- 不负责重新跑训练或 secure runtime，只负责收口现有产物；
- 推荐输入：
  - `plaintext_modified_eval.json`
  - `fair_external_comparison.json`
  - `stage2_secure_network_kth_candidate_check.json`
  - `stage2_secure_tie_candidate_check.json`
  - `plaintext_vs_secure_score_compare.json`
  - `secret_isolated_eval_summary.json`
- 输出：
  - `results/delivery_acceptance/<run>/delivery_acceptance_report.json`
  - `results/delivery_acceptance/<run>/delivery_acceptance_report.md`
- 作用：
  - 把 full-val plaintext、fairness、boundary checks、legacy replay consistency、same-policy E2E verify 和 guarded secret runtime summary 收到同一份报告里。
- 输出汇总为 `$RUN_ROOT/secret_isolated_eval_summary.json`，判定重点是 `accepted_count`、`unstable_count`、`accepted_accuracy` 和 `unstable_items`。

### `run_secure_static_distill_train.sh`

- 是为了从算法上降低 SPU 卡死概率的 secure-native distill 入口；
- 训练/验证时使用 `secure_static_train_depth=N`，跳过 runtime pruning predictor，只跑静态 whole-forward 前 N 个 blocks，再接 final norm/head；
- 默认使用 `fixed_square` activation 和 `approx_attn_mode=uniform`，与当前 SPU `uniform + fixed_square + public_calibrated LN` 路径对齐；
- 默认从当前新 bundle `frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430` 继续微调，并用同一 checkpoint 作为 teacher；如需回到旧 verified bundle，再显式覆盖 `BASE_BUNDLE_DIR`；
- 当前也支持把“准确率修正但不改模型结构”的单变量实验直接写进 wrapper：
  - `ACCURACY_PROFILE=weighted_sqrt_sampler`：自动设置 `TRAIN_SAMPLER_MODE=weighted_sqrt_inverse_freq`
  - `ACCURACY_PROFILE=sqrt_class_weight`：自动设置 `CLASS_WEIGHT_MODE=sqrt_inverse_freq`
  - 也可显式覆盖 `TRAIN_SAMPLER_MODE`、`CLASS_WEIGHT_MODE`、`MODEL_EMA`、`SMOOTHING`、`EVAL_BINARY_THRESHOLD`
- 背景是当前服务器 train / val 分布不平衡（`train 0:1214, 1:3494`；`val 0:135, 1:389`），而新 bundle 的同数据集公平外部对比表现为 `argmax < threshold`；
- 这条不平衡修正轴现已完成最小验证：
  - `weighted_sqrt_sampler` 在 `epoch1` 下已给出明确负信号；
  - `sqrt_class_weight` 在 `epoch1` 仅勉强不差，但到 `epoch3` 仍未形成明确收益；
  - `power_inverse_freq=0.15` 在 `epoch5` 已转负；
  - `power_inverse_freq=0.18` 在 `epoch5` 已转负；
  - `power_inverse_freq=0.20` 已成为第一条在 `epoch1 / epoch3 / epoch5` 都保持 `candidate_eval_not_worse` 的稳定候选；
  - `power_inverse_freq=0.22` 在 `epoch5` 仍非劣，但没有超过 `0.20`；
  - `power_inverse_freq=0.20` 与 `power_inverse_freq=0.22` 到 `epoch8` 都已重新转负，因此当前都不能算稳定长预算候选；
  - `power_inverse_freq=0.25` 现在只保留为“更强但不稳定”的近邻对照；
  - `MODEL_EMA=true` 在 `epoch5` 下只改善 argmax，不改善 `threshold_accuracy / AUC`；
  - `SMOOTHING=0.05` 在 `epoch1` 下只改善 argmax/loss，不改善 `threshold_accuracy / AUC`；
- 因此当前不要默认继续沿当前 `power_inverse_freq` 邻域、EMA 轴或 `SMOOTHING=0.05` 追加预算；如果还要继续做 accuracy 修正，应切到新的单变量假设。
- `GROUPA_LR_SCALE=1.0` 已经验证为 `epoch1` 非劣但 `epoch3` 转负，后续不要默认沿这条继续追加预算。
- `CLS_TOKEN_FULL_LR=true` 的测试用于拆分 `GROUPA_LR_SCALE=1.0` 里“cls token vs patch embedding”哪一部分可能有用；
- `CLS_TOKEN_FULL_LR=true` 已验证为 `epoch1` 指标完全持平，后续不要默认沿这条继续追加预算。
- `TRAIN_POS_EMBED=true` 已验证为 `epoch1` 指标完全持平，后续不要默认沿这条继续追加预算；这条结果可用于回应“位置编码是否有直接优化空间”：当前训练口径下没有观察到收益。
- `PRETRAINED_FIX_STEP=1` 已验证为 `epoch1` 有 argmax/AUC 短信号，但 `epoch3` 的 `threshold_accuracy / AUC` 转负，后续不要默认沿这条继续追加预算。
- `LR=1e-6` 已验证为 `epoch1` 只改善 argmax，但 `AUC / eval_loss` 变差，后续不要默认沿这条继续追加预算。
- `WARMUP_STEPS=0` 已验证为 `epoch1` 只改善 loss，但 `AUC` 变差，后续不要默认沿这条继续追加预算。
- `AUGMENTATION_PROFILE=mpcvit_like` 已验证为 `epoch1` 只改善 argmax/loss，但 `threshold_accuracy / AUC` 变差，后续不要默认沿这条继续追加预算。
- `FREEZE_PATCH_EMBED_PROJ=true` 已验证为 `epoch1` 有 threshold/AUC 短正信号，但 `epoch3` 没有延续；这条结果可用于回应“patch embedding 是否有直接训练优化空间”：当前没有稳定收益，后续不要默认沿这条继续追加预算。
- `FREEZE_PATCH_EMBED_WEIGHT / FREEZE_PATCH_EMBED_BIAS` 已完成拆分验证：短正信号来自 weight，但 `epoch3` 不稳定；bias 基本无效。因此 patch embedding freeze 组当前不再继续加预算。
- `PATCH_EMBED_BIAS_INIT_MODE=zero` 已完成 `epoch1` 验证：loss 极小改善，但 `threshold_accuracy` 持平、`AUC` 略降；当前仍保留预训练 patch embedding bias，不继续加预算。
- `BATCH_SIZE=16` 已完成 `epoch1` 验证：loss/argmax 改善，但 `AUC` 明显下降；当前不继续加预算。
- `WEIGHT_DECAY=0.01` 已完成 `epoch1` 验证：正式主指标完全持平，loss 只有 `2e-6` 量级变化；当前不继续加预算。
- `CLIP_GRAD=2.0` 已完成 `epoch1` 验证：argmax 小幅改善，但 `AUC / loss` 转差；当前不继续加预算。
- `GROUPA_LR_SCALE=0.0` 已完成 `epoch1/3/5/8` 验证：threshold 非劣且 AUC 持续小幅正向，但 loss/argmax 不一致；当前仅作为 AUC/calibration 候选。
- `GROUPA_LR_SCALE=0.0 seed1 epoch3` 已完成验证：没有复现小正 AUC，因此当前不升级默认值。
- accuracy pair wrapper 现在支持增强单因子：`BASELINE_/CANDIDATE_COLOR_JITTER`、`BASELINE_/CANDIDATE_AA`、`BASELINE_/CANDIDATE_REPROB`。
- `AA=none` 会在训练 wrapper 内转换为空 AutoAugment 策略，用于关闭 timm RandAugment。
- `AA=none` 当前是最强 threshold/AUC 候选：`epoch8` candidate `threshold_accuracy = 91.9847%`、`auc = 0.96787584`，相对默认 `+2.2901 pt / +0.02096544`；但 argmax/loss 变差，不能写成全面默认。
- `AA=none + MODEL_EMA=true` 的 EMA 权重评估会改善 argmax/loss，但压低 threshold/AUC；当前不要作为正式主线。
- `tools/transshield_public_logit_bias_calibration.py` 可把 best threshold 转成公开 class-1 logit bias；`AA=none epoch8` 对应 bias `0.5852264595359804`，calibrated argmax `91.9847%`，calibrated CE loss `0.42866483`，AUC 不变。
- `run_accuracy_profile_pair_study.sh compare` 已默认启用 public logit-bias calibration，并在 `accuracy_profile_compare.json` 中写入 `public_logit_bias_calibration_compare`；`accprof_epoch8_20260507_default_vs_aanone_1` 校准口径相对 default 为 `argmax +2.2901 pt`、`AUC +0.02096544`、`CE loss -0.01501771`。
- `tools/transshield_public_logit_bias_calibration.py --output-e2e-calibration-json` 可直接生成 E2E `--output-calibration-json`；当前文件为 `results/accuracy_profile_imbalance/accprof_epoch8_20260507_default_vs_aanone_1/e2e_output_calibration_public_logit_bias.json`。
- 该 output calibration 已通过 E2E smoke：
  - `e2e_approx_eval_public_bias_smoke2_20260507_1`：`sample_count=2`、`finite_logits=true`、`threshold_match_ratio=1.0`；
  - `e2e_approx_eval_public_bias_smoke4_20260507_1`：`sample_count=4`、`output_calibration` 正确落盘、`e2e_threshold_accuracy=100%`、same-subset plaintext threshold accuracy `100%`、`threshold_accuracy_gap=0.0pp`；
  - `e2e_approx_eval_public_bias_smoke8_20260507_1`：`sample_count=8`、`finite_logits=true`、`output_calibration` 正确落盘、`e2e_threshold_accuracy=100%`、same-subset plaintext threshold accuracy `100%`、`threshold_accuracy_gap=0.0pp`；
  - `e2e_approx_eval_public_bias_smoke16_20260507_1`：`sample_count=16`、`finite_logits=true`，但 `threshold_match_ratio=0.6875`、`threshold_accuracy_gap=-18.75pp`；
  - `e2e_approx_eval_public_bias_smoke16_chunk3_20260507_1`：`E2E_SPU_BLOCK_CHUNK_SIZE=3` 未改善；
  - 这验证的是 calibrated decision 接入路径，不是 raw argmax 逐样本一致；16 样本结果说明当前 E2E approximate whole-forward 仍有数值/排序漂移。
- 推荐顺序：
  - `debug80`：80 step finite/grad gate；
  - `epoch1`：1 epoch train+eval，检查准确率和 loss；
  - freeze/export bundle；
  - 用 `E2E_STATIC_DEPTH_LIMIT=N` 跑 public raw calibration 和 guarded secret eval。

### `run_e2e_secure_approx_deploy.sh`

- 是当前可实际使用的 e2e 全隐私输入近似推理入口；
- 固定服务器已验证的近似配置：
  - `E2E_STATIC_DEPTH_LIMIT=12`
  - `E2E_SPU_BATCH_SIZE=1`
  - `E2E_PARTY_LOCAL_SHARE_LOAD=1`
  - `E2E_REDACT_PRIVATE_INPUT_PATHS=1`
  - `E2E_SPU_LAYER_NORM_POLICY=public_calibrated`
  - `E2E_SPU_ATTENTION_POLICY=uniform`
  - `E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`
  - `E2E_SPU_ACTIVATION_CLIP_VALUE=3.0`
  - 可选但当前 smoke-stable 配置需要 `E2E_OUTPUT_CALIBRATION_JSON`
- 支持模式：
  - `make-calib-pixels`：从公开校准图片目录生成 public calibration pixel package；
  - `calibrate`：生成 public-calibrated layer norm JSON；
  - `infer`：对 split public/P1/P2 share manifests 执行 party-local SPU 推理；
  - `all`：按上述三步一次执行；
- 该入口会拒绝 `E2E_SPU_BATCH_SIZE != 1`，因为服务器已定位 `bsz=2` 会触发 full-depth batched graph 数值爆炸；多样本部署应按 `bsz=1` 逐样本 chunk 顺序处理；
- 默认公开校准目录为 `/data/wyb/pneumoniamnist_imagefolder_subset`，可用 `PUBLIC_CALIB_DATASET_DIR` 或 `PUBLIC_CALIB_IMAGE_LIST` 覆盖；
- 默认 reveal policy 仍是 final logits only；candidate JSON 会 redacted P1/P2 私有 manifest path；
- 这条路径是 deployable approximation，不是原始 exact ViT：exact secret LayerNorm、secret softmax attention、dynamic pruning、独立 P1/P2 进程仍是后续工作。
- `2026-04-26` 服务器已验证的当前部署基线是：
  - `depth=12 / public_calibrated LN / uniform attention / fixed_square / party-local share load / bsz=1`；
  - `sample=2` 按 `bsz=1` 顺序执行时 logits 有限、概率非饱和，两个样本输出均为 class 1；
  - `bsz=2` full-depth batched graph 会出现数值爆炸，因此不要作为部署默认值。
- `2026-04-27` 服务器 smoke-stable 配置在上述基础上进一步固定为：
  - `depth=12 / public_calibrated LN / uniform attention / fixed_square / clip3.0 / output calibration / party-local share load / bsz=1 / isolate samples`；
  - output calibration JSON：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json`；
  - same-image-list / same-targets 对比下，`class0_4`、`class1_4`、balanced8 三组 smoke 均达到 e2e 100%、original plaintext same subset 100%、prediction match 1.0；
  - balanced8 metrics：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced8_uniform_clip3_calibrated_20260427_201413/e2e_secure_poc/e2e_approx_eval_metrics.json`；
  - 这不是 full-val 证明，扩大样本前必须保持 `E2E_SPU_BATCH_SIZE=1`、`E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`，并确认 candidate 文件名包含 `uniform_fixed_square_clip3p0_eval`。
- balanced16 诊断结果：
  - smoke8 output calibration 在 balanced16 上退化到 E2E `81.25%` / match `0.75`；
  - balanced16 诊断版 output calibration：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_balanced16_diag.json`；
  - 一次性 run 曾在 `i=15` 出现 raw logits 偶发爆炸，fresh-runtime 单样本复跑恢复；
  - patched 诊断汇总：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_20260427_211653/e2e_secure_poc/e2e_approx_eval_metrics_patched_i15.json`，same-subset plaintext `93.75%`、E2E `93.75%`、gap `0.0pp`；
  - 这说明后续扩大样本必须启用 per-sample logits guard/retry，不能只看未 guard 的一次性 aggregate。
- balanced16 chunk3 guarded 结果：
  - `E2E_SPU_BLOCK_CHUNK_SIZE=3` 单样本 smoke 已确认 `spu_forward_graph_mode=reveal_less_block_chunked`，最大 request 从约 `86MB` monolithic 降到约 `22.8MB/21.3MB`；
  - 原始一次性 guarded balanced16 metrics：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_chunk3_guarded_20260427_235545/e2e_secure_poc/e2e_approx_eval_metrics.json`；
  - same-subset plaintext `93.75%`、E2E `93.75%`、gap `0.0pp`、match `0.875`、`finite_logits=true`、隐私字段通过；
  - 代价是慢：`e2e_elapsed_sec≈1474.46s`。后续需要优化 chunk size、runtime 启停/复用和通信统计，不能把该耗时作为最终 demo 性能。

### `client_private_prepare_image.sh`

- 用于“遇到一张新图时，在本地完成隐私输入准备”；
- 输入原图只在数据使用方本地机器出现，脚本会本地完成 image preprocess，并输出 split share manifests；
- 输出目录默认在 `artifacts/client_private_inputs/<timestamp>/`；
- 主要输出：
  - `client_pixel_values_debug_share_public_manifest.json`
  - `client_pixel_values_debug_share_party_manifests/p1_share_manifest.json`
  - `client_pixel_values_debug_share_party_manifests/p2_share_manifest.json`
  - `server_e2e_infer_env.sh`
- 本地准备命令：
  - `CLIENT_INPUT_IMAGE=/path/to/new_image.png bash artifacts/server_inference_friendly_pack/client_private_prepare_image.sh`
- 服务器推理命令：
  - `source artifacts/client_private_inputs/<run>/server_e2e_infer_env.sh`
  - `bash artifacts/server_inference_friendly_pack/run_e2e_secure_approx_deploy.sh infer`
- 注意：当前 share 语义仍是 `debug_float_additive_share_not_production_mpc_share`，生产部署时必须保证 P1/P2 share 文件分别只发送给各自 party，并通过 TLS / 独立主机 / 访问控制隔离；服务器后端不能接收原图。

### Web demo 的浏览器端隐私分析

- `run_web_demo.sh` 启动的页面现在主按钮走浏览器本地分片流程：
  - 浏览器读取图片；
  - Canvas 在本地完成 resize / center crop / normalize；
  - 浏览器生成 `share0/share1`；
  - 调用 `/api/e2e/analyze_private_shares` 上传二进制 share；
  - 后端只把 share 落成 party manifest，并调用 `run_e2e_secure_whole_forward.sh spu`；
  - 页面显示最终类别、概率、candidate JSON 和隐私字段。
- 该页面主流程不再调用 `/api/upload` 上传原图；旧 CPU/SPU sidecar endpoint 默认禁用，只有设置 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1` 时才作为调试路径开放。
- 当前 web demo 仍是单进程演示接口，会同时接收两份 share 并写入本机目录；它证明“网页端不上传原图 / 不上传完整 pixel_values”的产品交互，但生产环境还应拆成独立 P1/P2 上传端点和主机。
- 本地 Python 缺 `jax` / `spu.utils.distributed` 时，不要用默认 `WEB_DEMO_E2E_EXECUTION_MODE=local` 跑 live full-depth SPU；可切到远程执行代理：
  - `WEB_DEMO_E2E_EXECUTION_MODE=ssh`
  - `WEB_DEMO_REMOTE_SSH_TARGET=wyb@10.204.248.175`
  - `WEB_DEMO_REMOTE_SSH_PORT=9001`
  - `WEB_DEMO_REMOTE_REPO_ROOT=/data/wyb/Transshield_final`
  - `WEB_DEMO_REMOTE_PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python`
  - `WEB_DEMO_REMOTE_SSH_PASSWORD='<password>'`
  - 该模式保留本地页面和浏览器分片，把当前 `artifacts/web_demo_runs/<run>/` 同步到服务器同名相对目录执行 SPU，再把 candidate JSON/PT 拉回本地。
- Web 后端默认要求 public layer norm calibration JSON 和 output calibration JSON 都已存在；缺失时会直接报错，不会再隐式挑“最新文件”。
- `2026-05-02` 起，`run_web_demo.sh` 的默认 profile 已切到当前可展示 secret 线：
  - `WEB_DEMO_E2E_PROFILE=secret_depth6_clip0_showcase`
  - `E2E_SPU_PARAMS_MODE=secret_blockwise_stage`
  - `E2E_STATIC_DEPTH_LIMIT=6`
  - `E2E_SPU_LAYER_NORM_POLICY=public_calibrated`
  - `E2E_SPU_ATTENTION_POLICY=uniform`
  - `E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`
  - `E2E_SPU_ACTIVATION_CLIP_VALUE=0`
  - `E2E_OUTPUT_CALIBRATION_JSON=/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_secret_depth6_clip0_balanced8_20260502.json`
  - 默认 `WEB_DEMO_REUSE_SPU_RUNTIME=0`，每次请求 fresh runtime，优先保证展示稳定性。
- 如需回退旧的 public full-depth same-policy 路径，可显式设置 `WEB_DEMO_E2E_PROFILE=public_depth12_clip3_showcase`，并按需覆盖 `WEB_DEMO_E2E_LN_CALIBRATION_JSON` / `WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON`。
- 新版 E2E candidate JSON 会写入 `prediction_preview`，Web 后端读取结果时不再额外 `torch.load(.pt)`，避免在页面后处理阶段再次触发 `import torch`。
- 若服务器刚经历过 torch/native import 卡死，先只运行 `$PYTHON_BIN tools/transshield_chat_demo.py --help` 验证 Web 后端轻量启动链路；确认不卡后再启动 `run_web_demo.sh`，不要直接跑 `run_e2e_secure_approx_eval.sh`。

### `run_e2e_secure_approx_eval.sh`

- 用于计算 e2e 近似路径相对原始明文路径的准确率差，不再只看 1-2 张 smoke 样本；
- 默认评测数据目录：`/data/wyb/pneumoniamnist_imagefolder_subset/val`；
- 默认样本数：`E2E_EVAL_MAX_SAMPLES=8`，建议先用 8/16 做 smoke，再逐步扩大；
- 当前稳定 smoke 建议显式设置 `E2E_SPU_ACTIVATION_CLIP_VALUE=3.0`、`E2E_OUTPUT_CALIBRATION_JSON`、`E2E_SPU_BATCH_SIZE=1`、`E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`；
- isolated per-sample eval 会检查 raw/current logits 是否有限且绝对值不超过 `E2E_APPROX_EVAL_LOGIT_ABS_GUARD`，异常时按 `E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES` fresh-runtime 重试该样本；
- 每个 isolated SPU infer attempt 还应设置 `E2E_ISOLATED_INFER_TIMEOUT_SEC`，避免单个样本 runtime 半死导致整轮 eval 卡住；超时后同样按 retry 逻辑 fresh-runtime 重试。
- 如果日志卡在 `builtin_spu_run req_bytes≈86MB`，说明仍在跑 monolithic full-depth SPU graph；先设置 `E2E_SPU_BLOCK_CHUNK_SIZE=3` 做单样本 smoke，确认 candidate JSON 里 `spu_forward_graph_mode=reveal_less_block_chunked` 后再扩大样本。必要时把 chunk size 降到 `2` 或 `1`。
- 当前 chunk3 能稳定 balanced16，但耗时较高。效率优化应先小步比较 `E2E_SPU_BLOCK_CHUNK_SIZE=4/6`，同时记录 max `req_bytes`、elapsed 和是否卡死；不要直接扩大到 long-run full-val。
- 该脚本会在同一份 `e2e_eval_images.txt` 上同时运行：
  - 原始明文 reference；
  - static whole-forward plaintext reference；
  - e2e approximate SPU；
  - 同 target 的准确率差、预测一致率与 e2e SPU LinkDetails 通信量解析。
- 输出：`$E2E_RUN_DIR/e2e_approx_eval_metrics.json`，包含：
  - `original_plaintext_same_subset_argmax_accuracy`
  - `original_plaintext_same_subset_threshold_accuracy`
  - `static_whole_forward_same_subset_argmax_accuracy`
  - `static_whole_forward_same_subset_threshold_accuracy`
  - `e2e_argmax_accuracy`
  - `e2e_threshold_accuracy`
  - `argmax_accuracy_gap_e2e_minus_plaintext_pp`
  - `threshold_accuracy_gap_e2e_minus_plaintext_pp`
  - `argmax_accuracy_gap_e2e_minus_static_whole_forward_pp`
  - `threshold_accuracy_gap_e2e_minus_static_whole_forward_pp`
  - `prediction_match_vs_original_plaintext`
  - `prediction_match_vs_static_whole_forward`
  - `raw_secure_graph_before_output_calibration`
  - `e2e_communication_from_spu_node_logs`
- 同时额外输出：
  - `"$E2E_RUN_DIR/plaintext_static_gap.json"`
  - `"$E2E_RUN_DIR/plaintext_static_gap.md"`
  - 这份报告会直接比较 `plaintext full-model` 与 `static whole-forward` 的逐样本 score 对齐情况，并给出 `score_correlation`、`same_sign_ratio`、`affine boundary shift`、以及公开 threshold sweep 能恢复到什么准确率。
- 解释口径：
  - `original_plaintext_*` 只保留为 full-model context；
  - 当前 secure-static whole-forward 路线的主对照应看 `static_whole_forward_*`；
  - 若存在 `raw_secure_graph_before_output_calibration`，表示该 run 已把“raw secure graph drift”和“public output calibration gain”拆开记录。
  - 若 `original_plaintext_same_subset_*` 接近 `50%`，不要直接把它解释成 full-model ranking 坏掉；先看 `plaintext_static_gap.*`，确认是不是 `class1-class0` 零边界错位。
- 运行示例：
  - `E2E_EVAL_DATASET_DIR=/data/wyb/pneumoniamnist_imagefolder_subset/val E2E_EVAL_MAX_SAMPLES=8 bash artifacts/server_inference_friendly_pack/run_e2e_secure_approx_eval.sh`
- Python 预处理、明文 reference、share 生成和 metrics 写出步骤都带 timeout；服务器若在 `import torch` 等 native import 阶段卡死，会返回 `124` 并打印具体卡住的 step。

### `run_fullval_plaintext_static_gap.sh`

- 作用：
  - 不跑 SPU，只比较**同一 bundle** 下的 `plaintext full-model` 与 `static whole-forward`；
  - 用于判断当前剩余 gap 是不是还能靠公开 calibration 吃回来，还是已经进入语义层差异。
- 默认配置：
  - bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`
  - dataset：`/data/wyb/pneumoniamnist_imagefolder_subset/val`
- 支持：
  - `bash artifacts/server_inference_friendly_pack/run_fullval_plaintext_static_gap.sh fullval`
  - `bash artifacts/server_inference_friendly_pack/run_fullval_plaintext_static_gap.sh smoke8`
  - `E2E_GAP_MAX_SAMPLES=32 bash artifacts/server_inference_friendly_pack/run_fullval_plaintext_static_gap.sh custom`
- 输出：
  - `fullval_pixel_values.pt/json`
  - `fullval_plaintext_reference.json`
  - `fullval_static_reference.json`
  - `fullval_plaintext_static_gap_report.json/md`
- 当前已完成的正式 full-val 结果：
  - `results/e2e_gap_attribution/fullval_plaintext_static_gap_20260508_1/fullval_plaintext_static_gap_report.md`
  - 关键数：
    - `score_correlation = 0.962371`
    - `same_sign_ratio = 0.513359`
    - `x_at_y0 = 0.778917`
    - `plaintext best-threshold accuracy = 92.7481%`
    - `static threshold accuracy = 91.9847%`
  - 当前解释：
    - `plaintext full-model` 与 `static whole-forward` 的 score 排序仍强相关；
    - 但当前已经不是单纯 zero-boundary 偏移，而是 `boundary + scale` 同时漂移；
    - 因此继续在 `bias / affine / temperature` 之间切换，更像是在做公开后处理 tradeoff，而不是恢复 secure/static 主体语义。

### `run_fullval_reference_sidecar_replay.sh`

- 作用：
  - 把当前 bundle 的 `input + kth + tie payload` 导出后，直接用 `transshield_openbumblebee_inference_replay.py` 做 CPU reference replay；
  - 用来验证“runtime pruning boundary 一旦能正确外部化并回放，是否还能精确复现当前 full-model plaintext”。
- 默认配置：
  - bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`
  - dataset：`/data/wyb/pneumoniamnist_imagefolder_subset/val`
- 支持：
  - `bash artifacts/server_inference_friendly_pack/run_fullval_reference_sidecar_replay.sh fullval`
  - `bash artifacts/server_inference_friendly_pack/run_fullval_reference_sidecar_replay.sh smoke8`
  - `REFERENCE_REPLAY_MAX_SAMPLES=32 bash artifacts/server_inference_friendly_pack/run_fullval_reference_sidecar_replay.sh custom`
- 输出：
  - `stage2_secure_network_kth_input_smoke8.pt/json`
  - `stage2_secure_network_kth_reference_smoke8.pt/json`
  - `stage2_secure_tie_policy_lowest_smoke8.pt/json`
  - `pipeline_inference_replay_reference_summary.json`
  - `plaintext_vs_reference_replay_score_compare.json/csv`
- 当前已完成的正式 full-val 结果：
  - `results/e2e_gap_attribution/fullval_sidecar_replay_20260508_1/plaintext_vs_reference_replay_score_compare.json`
  - 关键数：
    - `logits max_abs_error = 0.0`
    - `probabilities max_abs_error = 0.0`
    - `argmax match ratio = 1.0`
    - `threshold match ratio = 1.0`
- 当前解释：
    - 当前 pruning boundary 只要能正确 externalize + replay，就可以**精确复现** `plaintext full-model`；
    - 因此剩余精度 gap 的主问题不是 sidecar replay，而是 `whole-forward static / SPU` 仍未接入 runtime pruning semantics；
    - stage 级 tie/mask 报告不一定逐位等于 argsort top-k，但在当前大量 tie-equivalent 情况下，final logits 仍可保持完全一致。

### `run_e2e_runtime_pruning_keepmask_bridge.sh`

- 作用：
  - 把 `runtime pruning keep-mask payload` 注入当前 `SPU whole-forward` 路径；
  - 同时输出 plaintext `runtime-pruning whole-forward reference`、SPU candidate 和 verify 对照；
  - 用来复现 `external keep-mask` 这条迁移线在 `plaintext/host_share/party_local` 三种输入边界下的行为。
- 支持：
  - `bash artifacts/server_inference_friendly_pack/run_e2e_runtime_pruning_keepmask_bridge.sh smoke1`
  - `KEEPMASK_INPUT_MODE=party_local KEEPMASK_SPU_PARAMS_MODE=public bash artifacts/server_inference_friendly_pack/run_e2e_runtime_pruning_keepmask_bridge.sh smoke4`
  - `KEEPMASK_INPUT_MODE=party_local KEEPMASK_SPU_PARAMS_MODE=secret bash artifacts/server_inference_friendly_pack/run_e2e_runtime_pruning_keepmask_bridge.sh smoke4`
  - `KEEPMASK_INPUT_MODE=party_local KEEPMASK_SPU_PARAMS_MODE=secret bash artifacts/server_inference_friendly_pack/run_e2e_runtime_pruning_keepmask_bridge.sh smoke16`
- 当前支持的输入边界：
  - `KEEPMASK_INPUT_MODE=plaintext`
  - `KEEPMASK_INPUT_MODE=host_share`
  - `KEEPMASK_INPUT_MODE=party_local`
- 当前 external keep-mask SPU 约束：
  - `KEEPMASK_SPU_ATTENTION_POLICY=uniform`
  - `KEEPMASK_SPU_PARAMS_MODE=public|secret`
- 主要输出：
  - `share_input_pixel_values.pt/json`
  - `keep_mask_payload.pt/json`
  - `runtime_pruning_reference.pt/json`
  - `candidate.pt/json`
  - `verify.json`

### `run_e2e_aanone_exactln_clip0_eval.sh`

- 是当前 `AA=none epoch8` 高精度 E2E 路线的固定 wrapper；
- 默认配置：
  - `AA_NONE_BUNDLE_DIR=artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`
  - `E2E_SPU_LAYER_NORM_POLICY=exact`
  - `E2E_SPU_ATTENTION_POLICY=uniform`
  - `E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`
  - `AA_NONE_OUTPUT_PROFILE=accuracy_first`
  - 支持 profile：`accuracy_first`、`loss_first_affine`、`loss_first_temperature`、`static_bias`、`bridge_best`
  - 其中 `accuracy_first` 对应 `spuaware_bias`，`loss_first_affine/temperature` 对应当前 held-out BCE 最优分支，`bridge_best` 只用于复核 bridge 候选，不是默认推荐
  - `E2E_SPU_ACTIVATION_CLIP_VALUE=0`
  - `AA_NONE_OUTPUT_CALIBRATION_JSON=results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias.json`
  - `E2E_APPROX_EVAL_ISOLATE_SAMPLES=0`
- 该 wrapper 默认强制使用 `AA=none epoch8` bundle 与 heldout-confirmed SPU-aware public logit-bias calibration，避免 `final_compare_env.template.sh` 中旧 `BUNDLE_DIR` 串扰；只有显式设置 `ALLOW_E2E_AANONE_OVERRIDE=1` 时才允许外部 `BUNDLE_DIR` / `E2E_OUTPUT_CALIBRATION_JSON` 覆盖。
- 支持 `smoke4|smoke8|smoke16|smoke32|smoke64|custom`：
  - `bash artifacts/server_inference_friendly_pack/run_e2e_aanone_exactln_clip0_eval.sh smoke16`
  - `bash artifacts/server_inference_friendly_pack/run_e2e_aanone_exactln_clip0_eval.sh smoke32`
  - `bash artifacts/server_inference_friendly_pack/run_e2e_aanone_exactln_clip0_eval.sh smoke64`
- 默认 `E2E_EVAL_LIST_STRATEGY=balanced_evenly_spaced`，避免按文件名取每类前缀样本导致 smoke 子集偏置；如需复现旧结果，可显式设置 `E2E_EVAL_LIST_STRATEGY=balanced_head`。
- 当前已验证：
  - isolated `smoke32`：`sample_count=32`，`finite_logits=true`，`e2e_threshold_accuracy=90.625%`，`elapsed=1522.97s`
  - non-isolated `smoke16`：`sample_count=16`，`finite_logits=true`，`e2e_threshold_accuracy=81.25%`，`elapsed=352.30s`
  - non-isolated `smoke16_affine`：`sample_count=16`，`finite_logits=true`，`e2e_threshold_accuracy=87.5%`，`elapsed=343.96s`
  - non-isolated `smoke32_legacy_bias`：`sample_count=32`，`finite_logits=true`，`e2e_threshold_accuracy=90.625%`，`elapsed=689.41s`
  - non-isolated `smoke32_affine_even`：`sample_count=32`，`finite_logits=true`，`e2e_threshold_accuracy=87.5%`，`elapsed=687.09s`
  - non-isolated `smoke32_temperature_even`：`sample_count=32`，`finite_logits=true`，`e2e_threshold_accuracy=87.5%`，`elapsed=668.73s`
  - non-isolated `smoke64_head`：`sample_count=64`，`finite_logits=true`，`e2e_threshold_accuracy=64.0625%`，`elapsed=1345.32s`
  - non-isolated `smoke64_even`：`sample_count=64`，`finite_logits=true`，`e2e_threshold_accuracy=87.5%`，`elapsed=1352.91s`
- `smoke32_legacy_bias` 与 current evenly-spaced `smoke32_*_even` 不是同一 image list；当前不要把 `90.625% -> 87.5%` 解释成 calibration 退化。
- `smoke64_head` 暴露了按文件名前缀取样的子集偏置；当前默认 `balanced_evenly_spaced` 的 `smoke64_even` 更适合作为大样本 smoke 精度口径，但 full-val static 仍是主精度口径。
- 如果目标是恢复 argmax/loss，可用 public affine output calibration：
  - `AA_NONE_OUTPUT_CALIBRATION_JSON=results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_affine.json`
  - 这是公开后处理 `score = scale * (logit_1 - logit_0) + bias`，不改变 secure ViT 主体算子，也不需要重训。
- 如果目标是保留 bias-only 决策边界并主要改善 CE/loss，可用 public temperature output calibration：
  - `AA_NONE_OUTPUT_CALIBRATION_JSON=results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_temperature.json`
  - 这是公开后处理 `score = scale * (logit_1 - logit_0 + bias)`，正 scale 不改变最终二分类边界。
- 该 wrapper 固定的是当前精度/部署候选，不替代旧 secret-depth6 showcase；如果需要旧展示线，继续使用对应 Web/demo profile。
- fixed block9 probe 与 wrong_idx13 block sweep 已修正旧归因：
  - fixed block9 summary：`results/e2e_block_probe/e2e_aanone_block9_probe_smoke32_even_fixed_20260507_1/block9_probe_summary.json`
  - wrong_idx13 sweep：`results/e2e_block_probe/e2e_aanone_block_sweep_wrong_idx13_20260507_1/block_sweep_summary.json`
  - heldout238 idx121 sample diagnosis：`results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_heldout238_spuaware_sample_diagnosis.md`
  - heldout238 idx121 block probe：`results/e2e_block_probe/e2e_aanone_heldout238_idx121_blocks_20260508/block_probe_summary.md`
  - heldout238 idx220 block probe：`results/e2e_block_probe/e2e_aanone_heldout238_idx220_blocks_20260508/block_probe_summary.md`
  - heldout238 idx167 block probe：`results/e2e_block_probe/e2e_aanone_heldout238_idx167_blocks_20260508_1/block_probe_summary.md`
  - heldout238 idx21 block probe：`results/e2e_block_probe/e2e_aanone_heldout238_idx21_blocks_20260508_1/block_probe_summary.md`
  - heldout238 high-margin batch report：`results/e2e_block_probe/e2e_aanone_heldout238_high_margin_batch_20260508_1.md`
  - heldout238 raw gap attribution：`results/e2e_gap_attribution/e2e_aanone_heldout238_20260508_1/e2e_gap_attribution_raw.md`
  - heldout238 calibrated gap attribution：`results/e2e_gap_attribution/e2e_aanone_heldout238_20260508_1/e2e_gap_attribution_calibrated.md`
  - heldout238 idx121 chunked runtime-axis：`results/e2e_block_probe/e2e_aanone_heldout238_idx121_blocks_20260508/idx121_chunk3_runtime_axis_report.md`
  - 修正后 `attn_out_cls` 仍高 cosine 对齐，当前错样本更符合“低决策 margin + late-block 累积数值 offset/amplitude drift”；不要再沿用旧的 attention-direction-drift 解释。
  - sweep 关键数：`block_output_cls` max-abs drift 从 block1 到 block12 增长约 `8.74x`，`min_attn_out_cls_cosine = 0.999995`。
  - heldout238 idx121 关键数：`block_output_cls` max-abs drift 从 block1 到 block12 增长约 `6.03x`，`min_attn_out_cls_cosine = 0.99999785`。
  - heldout238 idx220 关键数：`block_output_cls` max-abs drift 从 block1 到 block12 增长约 `7.00x`，`min_attn_out_cls_cosine = 0.99999738`。
  - heldout238 idx167 关键数：`block_output_cls` max-abs drift 从 block1 到 block12 增长约 `7.40x`，`min_attn_out_cls_cosine = 0.99999630`。
  - heldout238 idx21 关键数：`block_output_cls` max-abs drift 从 block1 到 block12 增长约 `7.54x`，`min_attn_out_cls_cosine = 0.99999428`。
  - 最新 aggregate 归因：`static whole-forward reference == cpu candidate`，`raw SPU` 只有 `logit_max_abs_error = 0.004115` 且 `argmax/threshold match = 1.0 / 1.0`；当前 heldout238 `92.437%` 提升来自 `SPU-side public output bias`，不是 raw secure graph 自身翻回错误样本。
  - 四样本 batch 结论：`consistent_late_block_cumulative_drift_pattern_observed`；当前 high-margin residual wrong 已经表现出稳定的 late-block cumulative drift 模式。
  - `E2E_SPU_BLOCK_CHUNK_SIZE=3` 在 idx121 上没有恢复预测，score 从 monolithic `-0.692276` 到 chunk3 `-0.69191`，仍预测 class 0；chunking 当前应归为通信/图大小优化，不应写成精度恢复方法。
- SPU-aware public threshold calibration 已给出轻量正信号：
  - report：`results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_public_threshold_recovery_smoke32_to_smoke64.json`
  - smoke32 threshold sweep：`87.5% -> 96.875%`
  - smoke32 threshold transfer to smoke64：`87.5% -> 92.1875%`
  - candidate calibration JSON：`results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json`
  - use with: `AA_NONE_OUTPUT_CALIBRATION_JSON=results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json`
  - stability report：`results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_spuaware_calibration_stability_report.json`
  - actual smoke96 E2E：`sample_count=96`，`finite_logits=true`，`e2e_threshold_accuracy=95.8333%`，`elapsed=2028.84s`
  - same smoke96 raw logits compare：static bias / temperature / affine 均为 `92.7083%`，SPU-aware bias 为 `95.8333%`
  - smoke32-disjoint heldout64：`sample_count=64`，与 smoke32 拟合集重叠数 `0`，`finite_logits=true`，`e2e_threshold_accuracy=92.1875%`
  - same heldout64 raw logits compare：static bias / temperature / affine / SPU-aware bias 均为 `92.1875%`
  - smoke32-disjoint heldout128：`sample_count=128`，与 smoke32 拟合集重叠数 `0`，`finite_logits=true`，`e2e_threshold_accuracy=91.40625%`
  - same heldout128 raw logits compare：static bias / temperature 为 `87.5%`，affine 为 `88.28125%`，SPU-aware bias 为 `91.40625%`
  - smoke32-disjoint heldout238：`sample_count=238`，与 smoke32 拟合集重叠数 `0`，`finite_logits=true`，`e2e_threshold_accuracy=92.43698%`，`elapsed=4983.14s`，`aggregate_total_bytes=1765262983`
  - same heldout238 raw logits compare：static bias `90.7563%`，SPU-aware bias / affine / temperature 均为 `92.4370%`
  - decision report：`results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_output_calibration_decision_report.md`
  - 当前 decision：accuracy-first default 使用 `spuaware_bias`；loss-first 使用 E2E-smoke32 affine / temperature。
  - sample-weighted heldout64/128/238 accuracy：SPU-aware bias `92.0930%`，static bias `90.0000%`。
  - loss-first calibration JSON：
    - `results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_output_calibration_public_affine_fit_on_spu_smoke32.json`
    - `results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_output_calibration_public_temperature_fit_on_spu_smoke32.json`
  - finished E2E run postprocess:
    - `RUN_DIR=artifacts/server_pipeline_run/<run>/e2e_secure_poc SPLIT_LABEL=<heldout_label> PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python bash artifacts/server_inference_friendly_pack/run_e2e_calibration_transfer_report.sh`
  - one-shot E2E calibration postprocess:
    - `RUN_DIR=artifacts/server_pipeline_run/<run>/e2e_secure_poc SPLIT_LABEL=<heldout_label> PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python bash artifacts/server_inference_friendly_pack/run_e2e_calibration_postprocess.sh`
    - 该入口会先生成当前 split 的 transfer report，再把已有 `heldout64/heldout128/heldout238` 汇总进 decision report。
  - unified output-calibration suite:
    - `PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python bash artifacts/server_inference_friendly_pack/run_e2e_output_calibration_suite.sh`
    - 该入口会先复用现有 `run_e2e_calibration_postprocess.sh` 生成 `transfer + decision report`，再额外跑 `plaintext -> static -> raw E2E` 的 bridge calibration 评测。
    - 额外输出位于：`$CALIBRATION_ROOT/e2e_plaintext_bridge_calibration_suite/`
    - 其中 `e2e_plaintext_bridge_best_bridge_calibration.json` 只表示“bridge 分支里最好的候选”，不代表它会替换当前 `spuaware_bias` 默认。
  - decision promotion gate：
    - 未包含 `heldout238` 时，`spuaware_bias` 只能保持 accuracy-first candidate；
    - 包含 `heldout238` 后，只有 sample-weighted accuracy gain >= `0.5` percentage point、`heldout238` 非劣、且无 held-out split regression，才升级为 accuracy-first default。
  - `heldout238` 已确认 aggregate gain；当前 accuracy-first 默认可切到 SPU-aware bias。若目标是降低 BCE/loss/confidence calibration，则仍应选择 E2E-smoke32 affine / temperature，而不是把它们写成 accuracy-first 方案。
- 服务器端快速查看关键 JSON：
  - `cat "$E2E_DIR/e2e_secure_contract.json"`
  - `cat "$E2E_DIR/client_pixel_values.json"`
  - `cat "$E2E_DIR/plaintext_reference.json"`
  - `cat "$E2E_DIR/static_whole_forward_reference.json"`
  - `cat "$E2E_DIR/e2e_static_whole_forward_candidate_from_server.json"`
  - `cat "$E2E_DIR/e2e_static_whole_forward_compare.json"`
  - `cat "$PACK_DIR/commands.json"`
- 若当前阶段已经进入 `depth=5 -> 6` 归因，不要继续手工 `cat` 散落 JSON；固定：
  - `export E2E_RUN_MAX_SAMPLES=1`
  - `export E2E_STATIC_DEPTH_LIMIT=6`
  - `export E2E_PROBE_BLOCK_INDEX=5`
  - `export E2E_SPU_BATCH_SIZE=1`
  - `export E2E_SPU_PARAMS_MODE=public`
  - 若 full run 还有 `Socket closed` / `Not connected`，再额外加：
    - `export SPU_DISABLE_COLOCATED_OPTIMIZATION=1`
    - `export SPU_RUNTIME_REUSE=0`
  - 然后依次运行：
    - `run_e2e_secure_whole_forward.sh probe-cpu`
    - `run_e2e_secure_whole_forward.sh probe-spu`
    - `run_e2e_secure_whole_forward.sh probe-compare`
  - 最终查看：
    - `cat "$E2E_DIR/block6_probe_compare_cpu_vs_spu_depth6.json"`
- 若当前阶段已经转到 heldout238 residual wrong 的单样本 block probe，优先直接使用：
  - `SOURCE_INDEX=<heldout238 source index> PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python bash artifacts/server_inference_friendly_pack/run_e2e_block_probe_sample.sh`
  - 默认输入：
    - `AA=none 20260507` bundle
    - heldout238 SPU-aware non-isolated source share manifest
    - heldout238 calibration drift CSV
  - 默认输出：
    - `results/e2e_block_probe/e2e_aanone_heldout238_idx<SOURCE_INDEX>_blocks_<timestamp>/`
  - 它会自动完成：
    - slice selected debug shares
    - `probe-cpu`
    - `probe-spu`
    - `probe-compare`
    - `block_probe_summary.json/.md`
  - 批量汇总可用：
    - `PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python python tools/transshield_e2e_block_probe_batch_report.py --probe-summary label=.../block_probe_summary.json ...`
- 如果只想抓这一个 run 回本地，不要同步整个 `artifacts/` 根；使用：
  - `mkdir -p /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME`
  - `rsync -avP -e "ssh -p 9001" --prune-empty-dirs --include='*/' --include='*.json' --include='*.md' --include='*.txt' --include='*.log' --exclude='*' wyb@10.204.248.175:/data/wyb/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/ /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/`
- 当前 `spu` 模式默认只 reveal final logits；不要加 `--include-intermediates`。当前 POC 仍会在 host 侧加载 plaintext `client_pixel_values.pt` 后再送入 SPU secret sharing，所以还不能表述为生产级“服务器从未接触明文输入”。

### `run_token_pruning_visualization.sh`

- 针对单张输入图片生成 token pruning 可视化；
- 输出 stage 级 overlay 图、trace JSON 和 Markdown 说明；
- 默认输出目录：
  - `artifacts/server_pipeline_run/<RUN_NAME>/token_pruning_visualization/`
- 主要文件：
  - `token_pruning_summary.png`
  - `stage_1_overlay.png`
  - `stage_2_overlay.png`
  - `stage_3_overlay.png`
  - `token_pruning_trace.json`
  - `token_pruning_trace_report.md`
- 适合答辩时解释 `masking` 如何替代直接裁剪。

### `run_single_image_comparison.sh`

- 针对同一张图片同时生成 baseline 与 modified 的单图对照；
- 输出摘要图、JSON 与 Markdown；
- 默认输出目录：
  - `artifacts/server_pipeline_run/<RUN_NAME>/single_image_comparison/`
- 主要文件：
  - `baseline_vs_modified_summary.png`
  - `baseline_vs_modified_comparison.json`
  - `baseline_vs_modified_comparison.md`
- 适合直接做答辩里的“baseline vs modified”案例页。

### `run_web_demo.sh`

- 启动一个最小可用的前后端一体化 Web demo；
- 支持前端上传图片、后端推理、最佳 bundle 摘要与 secure 结果展示；
- 适合做交互式流程展示与答辩演示界面。

### `run_cpu_spu_profile.sh`

- 顺序运行一套 `CPU secure` 与一套 `SPU secure`；
- 分别生成各自的 `secure_profile_summary.json`；
- 额外输出一份 `cpu_vs_spu_profile_report.json` 与 `cpu_vs_spu_profile_report.md`；
- 适合直接补答辩所需的时间 / 通信 profiling。

### `run_standardized_secure_external_benchmark.sh`

- 调用 `external_baselines/MPCFormer/tools/run_transformer_local2pc_server.sh`
- 把 `Transshield` 当前最终模型 proxy 与外部模型 proxy 放进同一个 `local 2PC configurable transformer benchmark`
- 输出：
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.json`
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.md`
- 适合回答：
  - “如果都放进同一个 secure transformer benchmark harness，外部模型 proxy 和本项目 proxy 的通信 / 时间差别怎样？”
- 不适合回答：
  - “full-val 医学图像 pipeline 总通信量谁更低？”
  - “网页单图 live run 和外部 benchmark 数字谁更低？”

### `run_secure_selection_mode_profile_compare.sh`

- 顺序运行两套 `SPU secure`，默认比较：
  - `flat_odd_even`
  - `blockwise_exact_kth`
- 也支持新的：
  - `blockwise_exact_kth`
- 其中 `blockwise_exact_kth` 适合配合 `tools/transshield_blockwise_kth_selection_manifest.py` 生成的 manifest 使用；
- 会优先使用仓库内收口好的 runtime inputs：
  - `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/`
- 如果该目录不存在，再自动寻找可复用的 runtime inputs 来源目录；
- 也支持显式指定：
  - `RUNTIME_INPUT_SOURCE_DIR=<old_run_dir>`
- 每个模式各自产生：
  - `fastpath_profile_summary.json`
  - `secure_profile_summary.json`
- 最后额外输出一份：
  - `selection_mode_profile_compare.json`
  - `selection_mode_profile_compare.md`
- 适合回答：
  - “`phase3_lower_tail` 到底有没有比旧模式更快 / 更省通信”
  - “新的 `blockwise_exact_kth` manifest 是否比旧实验 manifest 更合理”
- 当前服务器结果表明：`blockwise_exact_kth` 已通过 checker / replay，并在同口径 SPU profile 中降低 `network_kth_bridge` 时间；旧 `phase3_lower_tail` 只保留为历史实验开关。
- 仓库根目录的 `run_secure_selection_mode_profile_compare.sh` 现在只是转发到本脚本，避免根目录再维护一份过期默认值。

### `run_e2e_selected_policy_probe.sh`

- 从已完成的 E2E debug share manifest 中切出显式样本索引，重跑一个或多个 SPU policy variant；
- 默认用于 heldout238 的 high-margin / low-margin wrong sample probe；
- 输出：
  - `artifacts/server_pipeline_run/<probe_name>/...`
  - `results/e2e_policy_probe/<probe_name>/e2e_policy_probe_report.json`
  - `results/e2e_policy_probe/<probe_name>/e2e_policy_probe_report.md`
- 典型变量：
  - `SELECTED_INDICES="121,220,167"`
  - `VARIANT_SPECS="exact_uniform_clip0:exact:uniform:fixed_square:0:0:0 exact_uniform_clip3:exact:uniform:fixed_square:3:0:0"`
- 注意：该 probe 会重编译/重跑 selected-window graph；若要判断相对原 heldout aggregate 的真实变化，应额外生成 anchored report，把原 heldout candidate logits 作为 baseline。

---

## 7. 权威入口说明

- 当前目录下的 `.sh` 脚本是权威运行入口；
- `commands.json` 是保留的打包快照；
- 如果 `commands.json` 与 shell 包装脚本不一致，以 `.sh` 脚本为准。

---

## 8. 数据集要求

- `TRAIN_DATA_PATH` 应指向 `pneumoniamnist_imagefolder_subset/train`
- `VAL_DATA_PATH` 应指向 `pneumoniamnist_imagefolder_subset/val`
- 目录结构需保持 `ImageFolder` 兼容

---

## 9. 权重命名说明

### baseline

- 默认轻量权重：`artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth`

### modified

- 默认展示 / 运行 bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/`
- modified full-val / fairness 默认直接使用 bundle 内 pure `state_dict`：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/modified_plaintext_model_state_dict.pth`
- 如需显式覆写到 checkpoint 路径，再设置 `MODIFIED_CHECKPOINT=...`

默认比赛流程优先依赖当前 frozen bundle 的 pure `state_dict`；历史正式 bundle 与完整 checkpoint 已不再保留在当前最终仓。

---

## 10. Secure Pruning（PredictorLG SPU 内部安全执行）

### 入口脚本

```bash
# smoke1（快速验证）
bash artifacts/server_inference_friendly_pack/run_secure_pruning_smoke1.sh

# smoke8（8 样本精度验证）
bash artifacts/server_inference_friendly_pack/run_secure_pruning_smoke8.sh

# 实验性：Dropped-Token Context Recycling
E2E_SPU_TOKEN_RECYCLE_SCALE=0.1 \
  bash artifacts/server_inference_friendly_pack/run_secure_pruning_smoke8_batch8.sh
```

- `E2E_SPU_TOKEN_RECYCLE_SCALE` 默认为 `0`，表示关闭 recycling，保持原始行为。
- 当 `E2E_SPU_TOKEN_RECYCLE_SCALE>0` 时，secure pruning 会在裁剪前把被丢弃 token 的加权摘要注入 `CLS`；该路径只增加 multiply-accumulate，不增加新的通信轮次。

### 已完成 run

| Run | 样本数 | elapsed_sec | per-sample | argmax match | finite_logits |
|---|---|---|---|---|---|
| smoke1 | 1 | 254.6s | 254.6s | 1.0 | true |
| smoke8 | 8 | 1711.1s | 213.9s | 1.0 | true |

### 隐私字段

| 字段 | 值 | 说明 |
|---|---|---|
| `host_plaintext_pixel_values_materialized` | `false` | 服务器不接触明文影像 |
| `host_model_params_materialized` | `false` | 数据使用方不接触明文模型参数 |
| `runtime_pruning_keep_mask_pt` | `null` | 不依赖外部 keep-mask |
| `reveal_policy` | `final_logits_only` | 只暴露最终 logits |
| `spu_params_mode` | `secret` | 模型参数以 secret share 加载 |

### 技术要点

- PredictorLG + kth_threshold + tie_resolution 整条 pruning decision 链在 SPU 内部完整执行
- `backend = jax_spu_secure_pruning_forward_backend_v0`
- `forward_scope = student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path`
- SPU JAX tracer 修复：frozenset concrete 传参、全 jnp.where bitonic sort、手动 logsumexp
- 与 keep-mask wrapper 的差异：SPU 内部动态 pruning 产生不同 mask（threshold match ~0.375），但 argmax 预测完全正确（1.0/1.0）
