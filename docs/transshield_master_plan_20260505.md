# Transshield 最终总路线主文档

最后更新：`2026-05-10`

优先级说明：**这是当前仓库的最高优先级主文档。**

当下列文档之间出现冲突时，以本文件为准：

- `README.md`
- `docs/final_delivery_mainline_20260505.md`
- `docs/current_work_status.md`
- `docs/handoff-next.md`
- 任何旧展示摘要、旧公平对比摘要、旧审计摘要

## Summary

最终总路线应收束为一条清晰主线：

以 **pruning boundary 的协议友好重写** 作为唯一主创新轴，把 `masking -> F_mux`、`threshold compare -> F_less`、以及 `secure sidecar/replay` 闭环做成项目的核心方法；同时围绕这条主线，把 DynamicViT 主模型、MPC-friendly 算子替换、蒸馏补偿、secure-static 对齐训练、SPU/OpenBumbleBee 运行链和双向隐私输入边界收束成一个**可运行、可验证、可交付**的医疗影像隐私推理系统。

换句话说，项目不再追求“原始 ViT 精确 secure 复现”，而是追求：

- 客户图片不出明文
- 服务器模型参数不出明文
- pruning 决策边界以协议友好形式安全执行
- 模型精度对 baseline 保持可接受接近
- 系统能稳定运行、可批量评测、可复现实验与结果

## 为什么当前主线选择 ViT / DynamicViT，而不是 CNN

这里的模型选择，不是在泛化意义上宣称 “ViT 一定优于 CNN”，而是在回答：

- 哪一种主模型更适合作为当前项目主创新的载体；
- 哪一种表达更容易把 pruning 决策边界映射到 `F_less / F_mux`；
- 哪一种结构更方便围绕 `active token / tie risk / boundary ambiguity` 建立 secure 语义。

当前选择 `ViT / DynamicViT` 作为主模型的原因是：

- token 化表达天然暴露出“token score -> keep/drop decision”这一层边界；
- DynamicViT 的 pruning 过程可以被拆成 `score -> threshold -> decision`，更容易显式映射到协议友好接口；
- 后续围绕 `masked_score`、`kth_threshold`、`tie policy`、`active set` 的 secure replay / compare / risk 分析，都直接依赖这种 token-level 决策结构；
- 当前主创新讨论的是 pruning boundary 的协议友好重写，而不是单纯追求一个收敛更快的明文分类器。

同时，也必须明确当前路线的代价：

- ViT 的 token 交互和 attention 近似在 secure 环境下更昂贵；
- whole-forward secure inference 的工程复杂度显著高于 CNN；
- 因此当前需要 `uniform` attention、`fixed_square` activation、`secure_static_train_depth` 等部署对齐设计。

CNN 在胸片分类里仍然是有价值的基线或对照：

- 它有更强的局部归纳偏置；
- 训练与收敛通常更直接；
- 在不强调 token-level pruning boundary 的场景里，工程实现可能更轻。

但在当前项目中，CNN 不是主线载体，因为它不能自然承载当前这条 “token pruning boundary -> secure protocol mapping” 的核心叙事。后续如果要做 `CNN + ViT` 自适应混合，应被视为**独立研究分支**，而不是当前主线的顺手扩展。

## 明文 pruning 与 secure-facing masking 的统一语义

当前仓库并没有放弃明文剪枝。

- `main.py` / `models/dyvit.py` 中仍然保留了真实的 DynamicViT pruning 训练与推理逻辑；
- 当前变化的不是“是否剪枝”，而是“正式 secure 语义如何表达剪枝”。

原始 DynamicViT 更接近：

- 先为 token 打分；
- 再直接删除一部分 token；
- 后续 block 在变长 token 序列上继续前向。

这类“直接删除 token”的表达，在 secure 后端里会带来动态 shape 和更脆弱的执行边界。因此当前主线把它改写成：

- token 仍保留在张量里；
- 但通过 `masking` 表达“保留 / 置零”；
- 把 `score -> threshold -> decision` 这段 pruning boundary 显式拿出来；
- 再把 “compare + mux” 这部分映射到 `F_less / F_mux`。

因此，当前主线的正确说法是：

- 有明文 pruning；
- 有 secure-facing pruning 语义改写；
- 不是放弃 pruning，而是把 pruning 从“删除式表达”重写成“masking-friendly 表达”。

## “动态 pruning” 的动态性来自哪里

当前主线里的 “动态 Token 剪枝” 不是指“先拍脑袋设一个全局固定阈值”，而是指：

- 每个 pruning stage 都会基于当前样本、当前 active token 集、当前 predictor score 分布来形成边界；
- 训练时，predictor 会基于当前 stage 的 token 表征产出 `pred_score`，并通过硬决策形成 keep mask；
- 评估时，会对当前 `masked_score` 计算当前样本、当前 stage 的 `kth_threshold`，再结合 tie policy 决定 keep set。

也就是说，动态性来自：

- 样本不同，score 分布不同；
- stage 不同，active token 集不同；
- keep count 不同，`kth` 边界不同；
- 因而 pruning threshold 不是一个全局静态常数，而是一个**随当前输入和当前 stage 变化的边界值**。

这里还必须与二分类评测阈值严格区分：

- pruning threshold：服务于 token keep/drop 决策；
- binary classification threshold：服务于最终类别评测；
- 二者不是同一个阈值，不应混写。

## 分层收束表

说明：以下“必做落地项”属于正式可部署系统所必需的组成部分，并不等同于独立主创新；其作用是保证主创新能够在实际系统中训练、执行、验证和交付。

| 模块/方向 | 角色 | 当前状态 | 下一步任务 | 优先级 |
|---|---|---|---|---|
| `masking → F_mux` 表达重写 | 主路线 | **已完成**：正式方法定义已固化，语义映射证据已在 architecture.md | — | P0 ✅ |
| `threshold compare → F_less` 边界显式化 | 主路线 | **已完成**：protocol mapping 已固化，boundary check 通过 | — | P0 ✅ |
| pruning boundary `secure sidecar/replay` 闭环 | 主路线 | **已完成**：五闭环验收通过，boundary check + consistency 证据齐全 | — | P0 ✅ |
| DynamicViT 主模型 + masking 剪枝语义 | 必做落地项 | **已完成**：正式语义为 masking-friendly DynamicViT | — | P0 ✅ |
| party-local 输入边界 + secret model params | 必做落地项 | **已完成**：privacy constraints 全部验证通过，PredictorLG 在 SPU 内部执行（`host_model_params_materialized=false`） | — | P0 ✅ |
| `secret_blockwise_stage` secret runtime 路径 | 必做落地项 | **已完成**：8/8 accepted, 0 unstable, 0 pending | — | P0 ✅ |
| `public_calibrated + uniform + fixed_square` 默认 secure-friendly 路径 | 必做落地项 | **已完成**：正式 deployable approximation，full-val 已评测 | — | P0 ✅ |
| secure/plaintext 一致性验证链 | 必做落地项 | **已完成**：argmax/threshold match=1.0/1.0, logits error=0.0036 | — | P0 ✅ |
| fairness pipeline（同数据集公平对比） | 必做落地项 | **已完成**：fairness_comparison_is_fair=true | — | P0 ✅ |
| secure-static 训练主线 | 必做落地项 | **已完成**：91.98% threshold accuracy, 0.9679 AUC | — | P0 ✅ |
| 蒸馏补偿 | 强增强项 | **已完成**：official + cls-only paired result 已拿到，均 no_clear_distill_benefit_yet | 默认暂停，若重开需另起新的轻量单变量假设 | P1 ✅ (暂停) |
| `secure_static_train_depth` 训练-部署对齐 | 强增强项 | **已完成**：deployment-aligned + paired control（epoch1/epoch3 均 no_clear_depth_benefit_yet），聚合报告 20260510_full 已刷新 | 默认收口，收益未证明 | P1 ✅ |
| margin-aware / protocol-aware pruning loss | 强增强项 | **已完成**：focused5 epoch5 pair-study，no_boundary_relief_yet，violation_ratio=1.0 | 默认暂停，不继续沿当前路径追加预算 | P1 ✅ (暂停) |
| secure-friendly operator family 抽象 | 强增强项 | **已完成**：uniform/fixed_square/public_calibrated 已固化为默认设计族，轻量抽象文档已完成 | 可后续抽象成方法论论文 | P1 ✅ |
| sidecar 成本模型 / stage-level 风险模型 | 强增强项 | **已完成**：stage_cost_risk_report 已生成，3-stage cost/risk 分析完成 | 可后续补 active token/margin/tie/cost 更细关系分析 | P1 ✅ |
| embedding / position encoding 的 secure 优化 | 可兼容后续优化 | 当前未进入正式主线 | 仅在 whole-forward secure ViT 稳定后评估，不改变当前交付主线 | P2 |
| 更深 secret 路径（超出当前最小交付边界） | 可选新创新点 | 当前不是稳定主线 | 仅在不影响正式交付的前提下推进；目标是扩 secret 覆盖，不是替代当前最小可交付路线 | P2 |
| 更接近 exact ViT 的 attention/activation 近似 | 可选新创新点 | 当前与正式目标不一致 | 降级为长期研究方向；只有在不破坏稳定性和可交付性的情况下才投入 | P2 |
| 更完整的 secure-friendly operator family 方法命名与泛化 | 可选新创新点 | 还停留在潜力层 | 适合作为后续论文化/方法化抽象，不应阻塞系统交付 | P2 |

## 主创新主线与项目落地主线统一

### 必须完成的核心闭环

1. **方法闭环**
   - DynamicViT 的 pruning 决策边界被正式重写为：
     - `masking -> F_mux`
     - `threshold compare -> F_less`
   - 不是局部 patch，而是正式方法定义

2. **系统闭环**
   - `score -> threshold -> decision -> secure sidecar -> replay -> final prediction`
   - 其中，`secure sidecar/replay` 的定位是：**主创新的系统化承载链路，而非替代 `masking + F_less/F_mux` 的方法本体。**
   - 能在实际推理流程中稳定运行
   - 有 secret/plaintext 对齐验证

3. **隐私闭环**
   - 客户图片不以明文进入服务器正常推理路径
   - 服务器模型参数不以明文暴露给数据使用方
   - 中间 boundary / features / masks 不回明文
   - 最终只 reveal 结果所需最小输出

4. **效果闭环**
   - full-val plaintext 主模型可评测
   - secret 路径至少对同策略 plaintext reference 保持高一致性
   - 以 `threshold accuracy` / `AUC` 作为正式效果主指标
   - 结果与 baseline 的差距处于“可接受可交付”区间

5. **运行闭环**
   - 可单样本稳定运行
   - 可批量 isolated eval
   - 有失败恢复与部分结果落盘
   - 有可复现 fairness report

### 为真正可交付必须补齐的部分

- 正式的 full-val 评测与候选筛选流程
- 稳定的 secret runtime 批量执行与失败恢复
- same-policy plaintext reference
- secure/plaintext 一致性统一报告
- fairness pipeline 作为正式评估门
- 明确的最小部署边界与输入/输出约束

### 可以逐步加入的增强项

- 蒸馏补偿的进一步系统化
- `secure_static_train_depth` 的证据补齐
- margin-aware / protocol-aware pruning objective
- operator family 抽象
- 更细的 cost/risk model
- embedding / position encoding 的 secure 优化
- 更深或更完整的 secret 覆盖

## 最终项目至少要交付到什么程度

1. **模型层**
   - 有一条明确的 MPC-friendly DynamicViT 主模型训练线
   - 该模型不是原始 ViT 的临时近似，而是正式部署模型

2. **隐私层**
   - 双向隐私边界成立：
     - 图片不明文上传到服务器正常推理路径
     - 参数不明文暴露到数据使用方
   - secret runtime 是正式运行路径的一部分，不只是 demo 支线

3. **推理层**
   - pruning boundary secure sidecar/replay 是正式推理结构，而不是实验脚本
   - 同策略 plaintext reference 与 secret 路径可对齐、可验证

4. **评测层**
   - 有 full-val plaintext 指标
   - 有 same-policy 一致性指标
   - 有 fairness report
   - 有 secret runtime 稳定性统计

5. **系统层**
   - 可稳定运行、可重试、可批量评估
   - 关键结果可自动落盘
   - 失败不会导致整轮结果丢失

## 可加入的新创新点

| 新增点 | 与主路线关系 | 实现成本 | 是否值得现在投入 | 价值判断 |
|---|---|---:|---:|---|
| protocol-aware pruning objective（围绕 margin / tie / active-set 稳定性） | 直接增强 `masking + F_less/F_mux` 主创新 | 中 | 是，作为 P1 | 它能把主创新从“表达改写”进一步推到“训练阶段直接为 secure 执行服务” |
| `secure_static_train_depth` 的系统化证据 | 增强训练-部署闭环 | 中 | 是，作为 P1 | 它可能把当前训练线从经验配置提升为正式方法增强点 |
| stage-level secure cost / risk model | 直接强化主创新证据 | 中 | 是，作为 P1 | 它能说明为什么 pruning boundary 重写确实对 secure execution 有机理收益 |
| secure-friendly operator family 抽象 | 作为 deployable approximation 的统一理论化 | 中到高 | 可以，但不应阻塞主线 | 有助于后续方法化和论文化，但现在更适合作为增强层 |
| embedding / position encoding 的 secure 优化 | 与 whole-forward secure ViT 路线兼容 | 中到高 | 可以，作为 P2 | 它能补 whole-forward secure 路线前段开销，但不改变当前主创新轴 |
| 更接近 exact ViT 的 secret attention/activation | 与当前主路线弱耦合 | 高 | 否，暂不作为现在重点 | 会显著分散资源，且容易破坏当前可交付路线 |
| `CNN + ViT` hybrid / adaptive selection / cross-check | 会改变主模型、评测口径与协议目标 | 高 | 否，不并入当前主线 | 若要做，应单列为独立研究分支，而不是当前路线的增强项 |

## 当前不纳入主线的独立研究分支

下面这些方向不是“不能做”，而是**不应与当前主线混写成同一条交付承诺**：

- `CNN + ViT` hybrid / adaptive selection；
- 围绕双模型复用或 `ABY` 风格协议转换的新系统设计；
- 以多模型交叉验证为核心卖点的全新方法叙事。

原因是这类方向会同时改变：

- 主模型定义；
- 推理入口与模型选择策略；
- 效果评测口径；
- 协议设计目标。

因此，如果后续真的启动该方向，必须单列为新的研究分支，而不是把它直接并入当前 “DynamicViT pruning boundary secure rewrite” 主线。

## 明确结论

### 现在最应该集中资源完成什么

- 把 **pruning boundary 的协议友好重写** 真正做成正式系统核心
- 同时把它接上：
  - 可训练的 MPC-friendly DynamicViT 主模型
  - 稳定的 secret runtime
  - same-policy reference
  - full-val 效果评测
  - fairness pipeline
- 最终形成一条“模型训练 -> boundary secure execution -> replay -> 验证 -> full-val 评估”的完整交付链

### 哪些新创新点值得顺手加入

- protocol-aware pruning objective
- `secure_static_train_depth` 的证据化
- stage-level secure cost/risk model
- secure-friendly operator family 的轻量抽象
- embedding / position encoding 的 secure 优化（作为 P2）

这些都应被定义为：**主创新的增强项**。

### 哪些点先不要再发散

- exact ViT secure 复现
- 深度更大的 secret 路径探索取代当前最小交付线
- 只为“更像论文”而重新开辟新的算子/近似体系
- 把 `CNN + ViT` hybrid 误写成当前主线增强项
- 与主创新弱耦合的额外新名词路线

在 `P0` 主线闭环完成之前，不再新增会显著改变模型主干、协议主干或运行主干的研究分支；所有新增创新点仅允许作为主创新增强项并行验证，不能阻塞正式交付线。

## 当前执行结论

截至 `2026-05-10`，当前仓库已完成以下正式收束：

### P0 主线闭环

- `results/delivery_acceptance/delivery_acceptance_20260510_full/` 给出 `readiness.status = p0_delivery_closure_ready`
- 验收 gate 全部通过 **15/15 ✅**（`legacy_replay_consistency_exact` 已用 full-val 524 样本 keep-mask wrapper compare 修复）
- 五闭环全部通过：plaintext / fairness / boundary / consistency / secret-runtime
- 当前正式 plaintext bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- 当前精度增强 bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`（91.98% threshold accuracy）
- 当前正式 secret runtime：`secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1`

### P1 增强项全部完成

- **P1-1 stage cost/risk model**：完成。`results/stage_cost_risk_model/stage_cost_risk_20260505_clean/`
- **P1-2 secure_static_train_depth evidence**：完成。聚合报告 `secure_static_train_depth_20260510_full` 已含 paired control + acceptance gates，结论为"实现已完成，收益未证明"
- **P1-3 protocol-aware pruning**：完成。focused5 epoch5 pair-study 已拿到，no_boundary_relief_yet，默认暂停
- **P1-4 distillation compensation**：完成。official + cls-only paired result 均 no_clear_distill_benefit_yet，默认暂停
- **P1-5 secure-friendly operator family**：完成。uniform/fixed_square/public_calibrated 已固化为正式设计族

### keep-mask whole-forward wrapper

- smoke1/8/16/32 全部完成，argmax/threshold match = 1.0/1.0
- scaling 近线性收敛：sec/sample 从 233.83（smoke1）收敛至 194.63（smoke32）
- 验收报告：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_scaling_20260509_1/keepmask_scaling_report.md`

### 当前默认方向

- P0 + P1 全部子项已完成，无遗留阻塞项
- P2 方向（embedding/position encoding、更深 secret 路径、exact ViT 近似）为长期可选方向，不阻塞交付
- 如果继续推进，默认方向：runtime 效率优化（~194s/sample → 降低 SPU session 重启开销）或 E2E approximate 漂移诊断

### 2026-05-10 追加：PredictorLG SPU 内部安全执行

- smoke1 已验证：`artifacts/server_pipeline_run/secure_pruning_spu_smoke1_partylocal_secret_20260510/e2e_secure_poc/`
- PredictorLG + kth_threshold + tie_resolution 整条 pruning decision 链已在 SPU 内部完整执行
- 双向隐私边界全部达成：
  - ✅ 服务器看不到数据使用方图片（`host_plaintext_pixel_values_materialized = false`）
  - ✅ 数据使用方获取不到模型参数（`host_model_params_materialized = false`）
  - ✅ 只暴露最终 logits（`reveal_policy = final_logits_only`）
- `runtime_pruning_keep_mask_pt = null`：不再依赖外部 keep-mask，pruning decision 完全在 SPU 内部生成
- SPU JAX tracer 修复清单：frozenset concrete 传参、全 jnp.where bitonic sort、手动 logsumexp、去掉 pruning_metadata 参数
- smoke1 结果：`elapsed_sec = 254.6s/sample`
- smoke8 结果：`elapsed_sec = 1711.1s`，`per_sample_sec = 213.9s`，JIT 开销已摊薄
  - `argmax_match = 1.0`（vs plaintext reference，8/8 全部正确）
  - `threshold_match = 0.375`（SPU 内部动态 pruning 产生不同 mask，属预期差异）
  - 汇总报告：`results/e2e_gap_attribution/secure_pruning_spu_smoke8_20260510/secure_pruning_smoke8_summary.json`
- 当前状态：smoke1/8 均已通过，隐私边界全部成立

因此，后续所有 README、handoff、当前状态、答辩摘要都必须围绕本文件展开，而不是重新回到旧展示包叙事。
