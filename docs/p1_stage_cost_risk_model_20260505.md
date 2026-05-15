# P1 第一项：Stage-Level Secure Cost / Risk Model

最后更新：`2026-05-05`

## 1. 当前状态

`P1` 的第一项增强已经落地为正式可复用工具与报告：

- 工具：`tools/transshield_stage_cost_risk_report.py`
- server wrapper：`artifacts/server_inference_friendly_pack/run_stage_cost_risk_model.sh`
- 当前 clean 结果报告：
  - `results/stage_cost_risk_model/stage_cost_risk_20260505_clean/stage_cost_risk_report.json`
  - `results/stage_cost_risk_model/stage_cost_risk_20260505_clean/stage_cost_risk_report.md`

这一步不引入新训练分支，而是直接复用当前 `20260505_clean` 产物，把 `active token / tie / strict margin / secure bridge cost` 之间的关系显式化。

## 2. 模型口径

当前报告只服务当前正式主线：

- 方法核心：`masking -> F_mux ; threshold compare -> F_less ; secure sidecar/replay`
- plaintext bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- secret profile：`secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1`

## 3. 采用的 cost / risk 代理

### cost proxy

- `network_kth_bridge` 按各 stage 的 `active token` 总量占比分摊。
- `tie_policy_bridge` 按各 stage 的 `boundary equal token` 总量占比分摊。
- 这是 **stage-level 近似成本模型**，不是 gate-level 精确 profiling。

### risk proxy

- `boundary_tie_sample_ratio`
- `all_active_equal_ratio`
- `mean_tie_excess_count`
- `semantic_zero_margin_ratio`
- `kth_threshold_max_abs_error` 与 `strict_below_margin` 的对比

这组信号的目的，是解释：

- 为什么 `threshold compare` 不能只停留在“取一个阈值然后比大小”；
- 为什么 `tie sidecar` 是当前主线方法的一部分；
- 为什么 `masking -> F_mux` 的 secure replay 结构是必要承载链。

## 4. 当前 clean 结果的关键发现

1. `stage 0`
   - active token 负载最大，估计占 boundary sidecar 成本约 `42.56%`。
   - `93.13%` 样本出现 boundary tie。
   - `strict_below_margin p50 ≈ 5.96e-08`，但 secure kth max abs error 已到 `1.24e-05`，说明“只看 strict margin”远不足以解释可执行语义，必须依赖 tie 语义闭环。

2. `stage 1`
   - 当前 clean run 中语义风险最高。
   - `all_active_equal_ratio = 1.0`，意味着所有 active token 在进入 tie disambiguation 之前都落在同一条 boundary 上。
   - `mean_tie_excess_count = 41`，说明没有 tie sidecar 时，直接阈值比较会严重欠定。

3. `stage 2`
   - 成本占比最小，但不是“安全无关”阶段。
   - `86.64%` 样本仍有 boundary tie。
   - `strict_below_margin p50` 仍在 `5.96e-08` 量级，同样远小于当前 secure kth 数值误差量级。

## 5. 这项 P1 的意义

这份报告把此前零散的 profiling 和 boundary 语义说明收束成一个更清晰的结论：

- pruning boundary 的协议友好重写，不只是表达替换；
- 它把 secure 执行成本拆成了可解释的 stage-level 负载；
- 同时把风险集中点明确地落在 `tie-dominated boundary` 上；
- 这进一步支撑了当前主线：`F_less + tie sidecar + F_mux` 是正式方法闭环，不是补丁。

## 6. 复现实验命令

本地或服务器都可以直接用：

```bash
bash artifacts/server_inference_friendly_pack/run_stage_cost_risk_model.sh
```

若要显式指定当前 clean 结果：

```bash
cd /data/wyb/Transshield_final
export REPO_ROOT=/data/wyb/Transshield_final
export RUN_NAME=delivery_line_suite_20260505_clean
export SECURE_RUN_DIR=$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME
export SECRET_RUN_DIR=$REPO_ROOT/artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean
export OUTPUT_DIR=$REPO_ROOT/results/stage_cost_risk_model/stage_cost_risk_20260505_clean
bash artifacts/server_inference_friendly_pack/run_stage_cost_risk_model.sh
```
