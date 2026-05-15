# 当前正式主线与交付闭环

最后更新：`2026-05-10`

注意：本文件现在是**执行版压缩摘要**。若与 `docs/transshield_master_plan_20260505.md` 冲突，以后者为准。

2026-05-10 更新：完整验收报告（含 boundary check 五闭环全覆盖）已生成于 `results/delivery_acceptance/delivery_acceptance_20260510_full/`，`readiness = p0_delivery_closure_ready`。五个闭环全部通过：plaintext / fairness / boundary / consistency / secret-runtime。

这份文档把当前仓库的正式主线收束成一个单一口径，避免继续在 `verified_tracka` 历史展示包、`secure-static` 训练推进线、以及 `secret depth6` live secret 路径之间来回混写。

## 1. 单一句话口径

当前项目的正式主线是：

**以 pruning boundary 的协议友好重写为唯一主创新轴，用 `masking -> F_mux`、`threshold compare -> F_less` 和 `secure sidecar / replay` 形成方法闭环；同时以 `secure_static` 训练底座、`secret_blockwise_stage` 最小 secret runtime、same-policy/plaintext 对齐、fairness report 和 guarded secret eval 组成当前可运行、可验证、可交付的系统闭环。**

## 2. 当前只认这三层

### 2.1 方法主线

- `masking -> F_mux`
- `threshold compare -> F_less`
- pruning boundary `score -> threshold -> decision`
- `secure sidecar -> replay`

方法主线的角色是：

- 不再把原始 DynamicViT 的“直接删 token”当作正式 secure 语义；
- 明确当前正式语义是 **masking-friendly DynamicViT**；
- 明确 `secure sidecar / replay` 是 pruning boundary 的系统化承载链路，而不是替代 `F_less / F_mux` 的另一套方法本体。
- 明确当前选择 `ViT / DynamicViT` 作为主模型，是因为它天然暴露 token-level pruning boundary，更适合作为当前主创新载体，而不是在一般图像分类意义上宣称 CNN 无价值。
- 明确当前仍然存在真实的明文 pruning；变化的是 secure-facing 语义从“直接删 token”改写成“masking keep/zero”。
- 明确当前 pruning threshold 是随样本、stage、active token set 变化的动态 `kth` 边界，而不是一个全局固定常数，也不是最终二分类评测阈值。

对应代码 / 文档：

- `models/dyvit.py`
- `models/dylvvit.py`
- `docs/architecture.md`
- `tools/transshield_pruning_trace.py`
- `tools/transshield_stage2_report.py`
- `integrations/openbumblebee/transshield_network_kth_bridge/`
- `integrations/openbumblebee/transshield_tie_policy_bridge/`

### 2.2 当前 deployable plaintext / fairness 主线

当前默认 deployable plaintext bundle：

- `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`

它承担的是：

- full-val modified plaintext 评测；
- same-policy secure replay / compare 的明文参照；
- fairness report 的默认 `Transshield modified` 口径；
- Web live secure 路径的 bundle 底座。
- 当前正式口径下，modified plaintext full-val 默认直接评估该 bundle 冻结的 pure `state_dict`，
  不再把 source-run `checkpoint-best.pth` 当作 delivery line 的优先入口。

旧 `verified_tracka` 展示包已经从当前最终仓移除，不再作为当前 active delivery line 说明的一部分。

### 2.3 当前 deployable secret runtime 主线

当前最诚实、最小、正式的 secret runtime 口径是：

- `E2E_SPU_PARAMS_MODE=secret_blockwise_stage`
- `E2E_SPU_LAYER_NORM_POLICY=public_calibrated`
- `E2E_SPU_ATTENTION_POLICY=uniform`
- `E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`
- `E2E_SPU_ACTIVATION_CLIP_VALUE=0`
- `STATIC_DEPTH_LIMIT=6`
- `E2E_SPU_BATCH_SIZE=1`
- single-worker fresh runtime per sample
- guarded isolated eval

这条线的角色是：

- 它是当前正式 secret deployable 路径；
- 它不是 full-depth exact ViT secure 复现；
- `depth8+` 和 `clip3` 不再进入默认主线。

对应入口：

- `artifacts/server_inference_friendly_pack/run_e2e_secure_secret_isolated_eval.sh`
- `artifacts/server_inference_friendly_pack/run_web_demo.sh`
- `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py`

这条线与当前 `OpenBumbleBee / SPU` 承载的 `secure sidecar` 主线并不矛盾：

- 当前正式交付线：pruning boundary `secure sidecar + replay`
- 后续扩展线：whole-forward secure ViT，再迁入 masking-pruning 语义

“利用稀疏 token 压低开销” 应被理解为上述 `OpenBumbleBee / SPU` 集成下的优化子项，而不是另一条平行主方法。

### 2.4 当前不纳入交付承诺的方向

下面这些方向目前不写入当前交付承诺：

- `CNN + ViT` hybrid / adaptive selection / cross-check；
- 围绕双模型协议复用或转换的新系统主线；
- 把 embedding / position encoding secure 优化提前升级为当前主线硬目标。

其中：

- `embedding / position encoding` 优化可以作为后续 `P2`；
- `CNN + ViT` hybrid 会改变主模型与评测口径，应单列为独立研究分支。

## 3. P0 必须闭合的五个环

### 3.1 方法闭环

必须同时成立：

- `masking -> F_mux`
- `threshold compare -> F_less`
- pruning boundary 不再依赖“直接删 token”的旧叙事

当前证据位置：

- `docs/architecture.md`
- `tools/transshield_pruning_trace.py`
- `tools/transshield_stage2_report.py`

### 3.2 系统闭环

必须至少能说明：

- `score -> threshold -> decision -> secure sidecar -> replay -> final prediction`

当前承载入口：

- `artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`
- `tools/transshield_openbumblebee_pipeline.py`
- `tools/transshield_openbumblebee_inference_replay.py`

### 3.3 隐私闭环

必须至少能说明：

- 浏览器 / 数据使用方本地预处理并分片；
- 服务器正常 live 路径不 materialize 原图明文；
- 数据使用方不获取服务器明文参数；
- 中间 boundary / features / masks 不作为明文回流；
- 只 reveal 最终分类所需最小输出。

当前承载入口：

- `artifacts/server_inference_friendly_pack/run_web_demo.sh`
- `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh`
- `artifacts/server_inference_friendly_pack/run_e2e_secure_secret_isolated_eval.sh`

### 3.4 效果闭环

必须至少有：

- full-val modified plaintext 指标；
- same-policy / secure consistency 指标；
- fairness report；
- secret runtime 的稳定性 / accepted / unstable 统计。

当前对应产物：

- `plaintext_modified_eval.json`
- `plaintext_vs_secure_score_compare.json`
- `pipeline_verify_summary.json`
- `results/fair_external_comparison/.../fair_external_comparison.json`
- `secret_isolated_eval_summary.json`

### 3.5 运行闭环

必须至少满足：

- 可单样本稳定运行；
- 可 guarded batch / isolated eval；
- 中途中断时有 partial summary；
- 可继续复盘失败样本与 unstable items。

当前承载入口：

- `artifacts/server_inference_friendly_pack/run_e2e_secure_secret_isolated_eval.sh`

## 4. 当前默认运行入口

### 4.1 legacy pruning-boundary secure sidecar

- 环境模板：`artifacts/server_inference_friendly_pack/final_compare_env.template.sh`
- 完整运行：`artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`
- 公平对比：`artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh`

### 4.2 live secret 展示线

- Web：`artifacts/server_inference_friendly_pack/run_web_demo.sh`
- guarded secret eval：`artifacts/server_inference_friendly_pack/run_e2e_secure_secret_isolated_eval.sh`

### 4.3 统一验收汇总

- wrapper：`artifacts/server_inference_friendly_pack/run_delivery_acceptance_report.sh`
- tool：`tools/transshield_delivery_acceptance_report.py`

它负责把下面几类已有产物收成一份统一验收报告：

- plaintext full-val
- fairness
- boundary stage checks
- legacy replay consistency
- e2e same-policy verify
- guarded secret runtime summary

## 5. 当前正式验收顺序

推荐顺序：

1. `run_plaintext_eval.sh modified`
2. `run_full_final_comparison_suite.sh` 或已有 legacy secure run 的 compare / verify
3. `run_fair_external_comparison.sh`
4. `run_e2e_secure_secret_isolated_eval.sh`
5. `run_delivery_acceptance_report.sh`

推荐最少输入：

- `plaintext_modified_eval.json`
- `fair_external_comparison.json`
- `stage2_secure_network_kth_candidate_check.json`
- `stage2_secure_tie_candidate_check.json`
- `plaintext_vs_secure_score_compare.json`
- `secret_isolated_eval_summary.json`

如果补了 same-policy E2E verify，再额外纳入：

- `transshield_e2e_secure_vit_verify_v0`

## 6. 历史资产如何处理

下面这些已经完成更新或清理，不再作为当前默认交付线的历史残留入口：

- `artifacts/web_demo_assets/best_demo_content.json` 已切到当前 `secure_static` 交付线
- `docs/transshield_master_plan_20260505.md`
- `results/fair_external_comparison/fair_external_secure_static_20260505_clean/fair_external_comparison.md`

它们现在的角色应固定为：

- 当前正式主线说明；
- 或当前正式结果证据；
- 不再替历史 bundle 兜底。

## 7. 当前最重要的禁止事项

- 不再把 `depth8/9/12 secret` 当作当前 secret 主线。
- 不再把 full-depth public path 的成功说成 secret deployable。
- 不再把 Web live run、静态成绩板、fairness 报告混成同一组数字。
- 不再把 `exact ViT secure 复现` 当作当前交付目标。

## 8. 结论

当前仓库应被理解成：

- 方法核心：**pruning boundary 的协议友好重写**
- 明文主模型底座：**`secure_static` bundle**
- secret 最小交付 runtime：**`secret_blockwise_stage + depth6 + clip0`**
- 正式评测门：**full-val plaintext + same-policy consistency + fairness + guarded secret eval**

后续新增实验只能作为增强项进入，不能再反向改变这条主线。
