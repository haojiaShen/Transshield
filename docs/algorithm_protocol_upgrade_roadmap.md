# 算法 / 协议升级路线图

> 目标：把 `Transshield` 从“已经跑通 secure sidecar”继续推进到“这个 sidecar 本身就是为 MPC/secure execution 专门重设计过的”。

## 总体判断

当前仓库已经把最关键的 secure sidecar 闭环打通，但下一阶段若想让作品更像参赛成品，重点不应再停留在结构整理，而应推进四件事：

1. 让 pruning 分数分布更适合 secure 比较协议
2. 让 `network-kth` 本身更像协议友好的选择器，而不是明文 `topk` 的直接替身
3. 让 sidecar 输入 / 输出更轻、更稳、更容易展示通信收益
4. 在不破坏现有正式展示路径的前提下，把安全边界继续前移到客户端输入，新增 `e2e secure inference` 并行系统线

## 推荐推进顺序

### Phase 1 — Margin / tie 风险诊断（最低风险，立即可做）

目标：

- 先量化当前 `kth` 边界到底有多“挤”
- 识别哪些 stage 的 tie 比例高、margin 小、payload 大
- 为后续训练改动和协议改动提供读数基线

当前已接入的入口：

- `tools/transshield_stagewise_threshold_report.py`

这一阶段最该看的新增指标：

- `topk_boundary_margin_percentiles`
- `topk_boundary_small_margin_ratio_abs`
- `topk_boundary_small_margin_ratio_rel`
- `protocol_risk_signals.boundary_tie_sample_ratio`
- `protocol_risk_signals.mean_tie_excess_count`
- `payload_estimate.estimated_active_score_bytes_float32`

判断标准：

- 如果某个 stage 的 `boundary_tie_sample_ratio` 高、`small_margin_ratio` 高，则优先对该 stage 做 margin-aware 改造
- 如果某个 stage 的 payload 明显更大，则优先对该 stage 做分块/压缩

### Phase 2 — Margin-aware pruning 训练（最值得先做的算法升级）

目标：

- 不是只让模型分对，而是让“保留 token”和“边界外 token”的分数间隔更大
- 降低 tie 和近阈值样本比例
- 让 `F_less` / `network-kth` 更稳、更便宜

建议做法：

- 在训练损失中新增 boundary margin regularizer
- 只对 pruning stage 生效，不改主分类头语义
- 默认作为可开关项接入，先做 ablation，不直接替换现有 best 配置

第一批建议改动入口：

- `main.py`
  - 新增参数：
    - `--pruning_margin_weight`
    - `--pruning_margin_target`
    - `--pruning_margin_mode`
- `losses.py`
  - 在 `DistillDiffPruningLoss_dynamic` 中新增 margin loss
  - 优先复用当前已有 `ratio_loss` / `token_distill_loss` 的训练框架
- `models/dylvvit.py`
  - 在训练态暴露每个 pruning stage 的 `score`
  - 供 loss 读取 boundary keep / next score 或近边界 token 信息
- `engine.py`
  - 记录新的 margin loss 指标，便于看是否稳定

验收信号：

- `threshold_accuracy` 不明显下降
- `topk_boundary_margin_mean` / `p50` 上升
- `boundary_tie_sample_ratio` 下降
- secure replay 仍保持 `match_ratio = 1.0`

当前已经拿到一条明确证据：

- 服务器 `w10` 候选表明，这个方向本身是对的
- 代表性结果：
  - Argmax Acc：`88.9313%`
  - Threshold Acc：`90.2672%`
  - AUC：`0.956508`
  - secure replay：`argmax/threshold match = 100%`
- 最关键的是 Stage 2：
  - margin mean 提升 `243.532x`
  - `<=1e-4` near-boundary 比例从 `98.66%` 降到 `5.92%`
  - tie 风险没有变坏

这说明 `margin-aware pruning` 已经能提供“协议友好”的分数分布，但当前的**全局统一 margin 权重**会明显伤精度，因此下一步不该继续粗暴扫全局权重，而应改成：

- 只压最关键的 pruning stage
- 在训练后半程再启用
- 必要时配合更强的 token distillation 补回分类性能

推荐的下一轮最小改造：

- `--pruning_margin_stage_weights`
  - 例如：`0,1,0`
- `--pruning_margin_start_epoch`
  - 例如：`8`

这样能把 `w10` 这条研究证据，进一步转成“更稳、更接近正式模型精度”的可落地方向。

截至 `2026-04-18`，这条训练线已经拿到足够结论，当前建议先收口：

- 保留 `w10` 作为**最强研究性正结果**
- 保留 `w3 + formal hparams + tok0.02` 作为**当前最佳折中证据**
  - Threshold Acc：`91.6031%`
  - AUC：`0.967476`
  - Stage 2 margin：`20.032x`
- 负结果也已经说明问题边界：
  - `Stage2-only + delayed start`：精度失败
  - `w2`：Stage 2 更强，但 Stage 3 变坏
  - `w3 + tok0.04`：Argmax 上升，但 Threshold / AUC 和 Stage 3 分布变差

因此当前不再建议把主要精力继续放在 `margin-aware` 训练超参搜索，而是把这部分作为 **Phase 2 已经成立的研究证据**，转去推进：

- `Phase 3 — Hierarchical / block network-kth`
- `Phase 4 — Sidecar payload 压缩`

### Phase 3 — Hierarchical / block `network-kth`（最关键的协议升级）

目标：

- 减少 compare-network 的在线比较压力
- 避免直接对整段 token 做一次性排序式处理
- 把 `network_kth_bridge` 从当前主要瓶颈继续压下去

建议做法：

- 先做 block-wise `kth`
- 再做 hierarchical merge
- 允许 stage 按 token 数选择不同 pass 计划

第一批建议改动入口：

- `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
  - 新增 block-wise / hierarchical 路径
  - 保留当前路径作为 reference fallback
- `tools/transshield_secure_network_kth.py manifest`
  - 给 manifest 增加 block 配置 / merge 配置
- `tools/transshield_secure_network_kth.py export`
  - 输出对新协议仍兼容的参考 `kth_threshold`
- `tools/transshield_openbumblebee_inference_replay.py`
  - 验证新 `kth` 输出与 replay 语义一致

验收信号：

- `network_kth_bridge_elapsed_sec` 下降
- `kth_threshold` 与 reference 的误差保持可控
- tie payload / replay checker 不报错

### Phase 4 — Sidecar payload 压缩（通信和展示都受益）

目标：

- 降低 default fast runtime 的 Python fastpath RPC 负担
- 让展示里的通信数字更好看，也更真实反映“协议友好化”成果

建议做法：

- 对 `masked_score` 先做更稳的紧凑化
- 评估定点化/低比特量化
- 减少 stage 间重复元数据
- 尽量批量传输而非碎片化 RPC

第一批建议改动入口：

- `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
  - 扩展 `compact_stage_inputs(...)`
- `tools/transshield_secure_network_kth.py input-export`
  - 或直接通过 `tools/transshield_secure_sidecar_export_suite.py` 一次性导出并补充统计
- `tools/transshield_fastpath_profile_summary.py`
  - 对压缩前后通信做同口径对比

验收信号：

- `python_fastpath_rpc_total_bytes` 下降
- replay / compare 语义不变
- `communication_status` 仍保持 `available_python_fastpath`

### Phase 5 — Protocol-aware pruning policy（作品层面的升级）

目标：

- 不再只是“把原始 DynamicViT 适配进 secure”
- 而是显式设计出更适合协议执行的 pruning policy

建议做法：

- 控制各 stage 的 keep ratio 变化斜率
- 限制容易引起 tie 的 score 分布
- 让不同样本上的 active token 规模更平滑
- 在训练阶段就为 secure replay / `F_less` / `F_mux` 设计分布

建议改动入口：

- `main.py`
  - 增加 protocol-aware 配置开关
- `losses.py`
  - 增加针对边界拥挤度 / active-count 波动的 regularizer
- `tools/transshield_stagewise_threshold_report.py`
  - 继续作为读数基线
- `docs/architecture.md`
  - 后续同步更新为“协议共设计”叙事

### Parallel Track — `E2E secure inference`（并行系统线）

目标：

- 让服务器侧正常路径不再看到原始 X 光图或其明文 `pixel_values`
- 把隐私边界从当前 pruning sidecar 前移到输入端
- 最终形成“只 reveal 最终分类结果”的端到端 secure inference POC

建议做法：

- 保留当前 `secure sidecar + replay + compare` 作为正式默认路径，不直接覆盖
- 先做**不含 pruning** 的最小 `deit-s / ViT` 整网 secure inference：
  - 客户端本地读取 / 预处理图像
  - secret-share `pixel_values`
  - `patch_embed -> blocks -> head` 全放进 secure 前向
- 再迁入当前 `masking` 语义，把
  - `pred_score -> prev_decision -> policy -> block forward`
  - 这段也一起 secure 化
- 最后再复用 `Phase 2/3/4` 的 margin-aware / blockwise-kth / payload 优化结果，继续压时间与通信

第一批建议改动入口：

- `docs/architecture.md`
  - 补一版“当前 sidecar 边界 vs 未来 e2e 边界”的结构说明
- `integrations/openbumblebee/`
  - 新增独立 `e2e` 子目录，而不是直接改坏现有 bridge
- `tools/`
  - 新增独立 e2e 入口与 plaintext 对齐 checker
- `configs/openbumblebee/`
  - 补独立 e2e runtime 模板，不和当前 live sidecar 配置混写

验收信号：

- 非调试模式下，服务器正常路径不出现原始图片或明文 `pixel_values`
- 中间 `mask / threshold / features` 不回明文
- `e2e secure` 输出与 plaintext reference 保持语义一致
- 通信 / 时延作为独立口径统计，不与当前 sidecar live run 混写

截至 `2026-04-22`，这条并行线已经拿到第一轮明确的阶段性正结果：

- 服务器 run `tracka_e2e_secure_poc_cpu` 已完成 `client-preprocess -> static whole-forward reference -> cpu candidate -> verify` 闭环；
- `sample_count = 524`
- `static whole-forward reference` 输出形状：
  - `logits = [524, 2]`
  - `cls_features = [524, 384]`
  - `token_features = [524, 196, 384]`
- `cpu candidate elapsed_sec ≈ 20.73`
- `verify` 结果：
  - `logits max_abs_error = 0.0`
  - `probabilities max_abs_error = 0.0`
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`

这说明当前已经可以把 “static whole-forward contract + checker” 视作稳定基座；下一步不该继续重复 CPU 参考验证，而应直接进入 **whole-forward SPU backend** 的实现与服务器验证。

截至 `2026-04-23`，本地已经补上第一版实验性 `runtime=spu` 后端：

- 静态 DeiT-S whole-forward 已翻成 JAX/SPU 子集；
- 输入 `pixel_values` 会进入 SPU secret sharing；
- 模型参数首轮 smoke 默认按 public 参数进入 SPU，后续可切 `--spu-params-mode secret`；
- SPU 模式只 reveal final logits，不 reveal `cls_features/token_features`；
- 仍未包含动态 `masking-pruning` 决策路径。

因此路线图上的下一步已经从“写 backend”推进为“服务器小样本 smoke + verify”，验收后再逐步扩大样本数。

## 当前建议的实际执行顺序

1. 先跑 `Phase 1` 的 margin/tie 诊断
2. 基于读数做 `Phase 2` 的 margin-aware training
3. 再进入 `Phase 3` 的 hierarchical `network-kth`
4. 然后做 `Phase 4` 的 payload 压缩
5. 最后再做 `Phase 5` 的 pruning policy 重设计
6. 在不打断当前正式展示路径的前提下，并行开启 `E2E secure inference` 的最小 POC（先静态 ViT，再接 masking-pruning）

## 为什么这个顺序更合适

- 先诊断，再改训练，避免盲改
- 先改分数分布，再改 `kth` 协议，效果更容易叠加
- 先解决真正瓶颈 `network-kth`，再抠次要部件
- 保持每一步都能用现有 replay / compare 闭环验证，不会把系统打散

## 当前最推荐的下一步

当前 sidecar 主线仍建议直接从 `Phase 1` 开始：

- 用增强后的 `tools/transshield_stagewise_threshold_report.py`
- 对当前 verified bundle 重新导出一版 margin / tie / payload 风险报告
- 然后再决定 `Phase 2` 里 margin loss 的具体形式

如果当前工作重点改成“尝试把 `Transshield_final` 推到更强隐私边界”，那么最小正确起步不是重写整套 `dyvit`，而是：

- 先定义 e2e 路线的 reveal 策略与输入边界；
- 先做 `client preprocess + pixel_values secret share + static ViT secure inference`；
- 再决定何时把 `masking-pruning` 语义并入 e2e secure 前向。
