# P1 第三项：Protocol-Aware Pruning Objective

最后更新：`2026-05-05`

## 1. 当前收口方式

这一项当前不再包装成“已经证明提升了最终指标”。

当前正式收口方式是：

- 把仓里已经存在的 `pruning_margin_*` 训练接口变成正式 server-pack 入口；
- 基于当前 clean `stage_cost_risk_report` 生成一份 **protocol-aware recipe**；
- 提供一份 **margin log report** 工具，用来解析短跑训练是否真的把该 objective 接上线；
- 明确区分：
  - “objective 已正式接入”
  - “objective 已经证明带来收益”

目前只完成前者，后者仍需要新的训练 run 和同口径 compare。

## 2. 对应工具与 wrapper

- 工具：
  - `tools/transshield_protocol_aware_pruning_recipe.py`
  - `tools/transshield_pruning_margin_log_report.py`
- server wrapper：
  - `artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh`
  - `artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_report.sh`
- 训练入口透传：
  - `artifacts/server_inference_friendly_pack/run_secure_static_distill_train.sh`

## 3. 这一步解决了什么

当前 `P1-1` 已经给出：

- stage 0 成本占比最高；
- stage 1 风险最高；
- stages 0/2 的 strict margin 远小于当前 secure kth 数值误差；
- stage 1 甚至出现 `all_active_equal_boundary`。

因此，`protocol-aware pruning objective` 的合理目标不是另开一条新模型主线，而是：

- 在训练时直接对 pruning boundary 加入 margin pressure；
- 优先压 `stage 1` 的 tie-dominated boundary ambiguity；
- 同时保留 `stage 0` 的成本敏感性；
- 让后续 secure `F_less + tie sidecar + F_mux` 执行链得到更稳定的边界输入。

## 4. 当前 recipe 的口径

当前 recipe 是 **基于 `20260505_clean` 证据的启发式入口**，不是经验最优超参。

它当前默认给出两个 profile：

1. `conservative`
   - 用于第一轮 wiring / stability 检查；
   - 默认使用较平滑的 `softplus`；
   - 立即激活 margin objective，但权重保守。

2. `focused`
   - 用于 conservative 短跑确认稳定后；
   - 对 dominant risk stage 给更强压力；
   - 使用更明确的 `hinge` 最小 margin 目标。

当前推荐先用：

- `conservative -> debug80`
- `conservative -> epoch1`

## 5. pair-study 注入 bug 与当前修复

在补做更长配对训练时，已经发现一类必须明确排除的无效证据：

- `protocol_aware_pair_epoch3_20260505_focused2`
- `protocol_aware_pair_epoch3_20260505_focused3`

这两组 pair-study 不能作为当前 `P1-3` 的有效收益证据，原因不是“focused profile 已被证明无效”，而是：

- candidate 训练命令被旧环境变量污染；
- `pruning_margin_stage_weights` 虽然进入了命令，
- 但 `pruning_margin_weight` 仍停在 `0.0`，
- `pruning_margin_target` 仍停在 baseline 的 `1e-4`，
- 所以 candidate 实际上没有按 recipe 真正启用目标 profile。

为避免这类静默退化继续发生，当前已经把脚本链修成：

- `run_protocol_aware_pruning_train.sh`
  - 新增 `PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1`
  - candidate 模式下会强制用 recipe 覆写 `PRUNING_MARGIN_*`
  - 若未强制覆写但现有环境变量与 recipe 不一致，会显式打印告警
  - 新增 `print-env` 模式，可在不开训练的情况下先核对实际注入值
- `run_protocol_aware_pruning_pair_study.sh`
  - baseline 分支显式清掉强制覆写标志
  - candidate 分支固定开启 `PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1`

因此，后续用于正式比较的 pair-study 应重新起新名字重跑，不再引用 `focused2/focused3`。

## 6. 一个关键注意点

当前训练日志里的：

- `loss info: ... pruning_margin=... margin_stats=[...]`

是 **每 100 个训练 step** 才打印一次。

因此：

- `debug80` 的意义是：
  - 检查命令接线是否正确；
  - 检查非有限值与早期稳定性；
  - 不保证产出 `margin_stats`
- `epoch1` 才是当前数据规模下第一条应当产出 `margin_stats` 的最短 run

这也是为什么 `run_protocol_aware_pruning_report.sh` 需要对“没有 margin 日志”给出语义化解释，而不是直接判失败。

## 7. `fix4`：第一条真正生效的短跑证据

当前真正可作为 `P1-3` 证据引用的不是 `debug80`，也不是早期误接线 run，而是：

- 训练 run：
  - `artifacts/train_runs/protocol_aware_pruning_epoch1_20260505_fix4/`
- 报告目录：
  - `results/protocol_aware_pruning_objective/protocol_aware_pruning_epoch1_20260505_fix4/`

这条 run 的关键事实是：

- `secure_static_train_depth=12`
- `secure_static_skip_pruning=false`
- `pruning_margin_weight=1.0`
- `pruning_margin_target=2.6e-05`
- `pruning_margin_mode=softplus`
- `pruning_margin_stage_weights=1.50,1.65,1.00`

并且日志中已经出现了真正的 stage-wise margin 信息：

- `loss info: ... pruning_margin=6.931602e-01 margin_stats=[s0:...; s1:...; s2:...]`

对应报告也明确给出：

- `interpretation.status = "protocol_margin_stats_available"`
- `max_pruning_margin = 0.6931602`
- `nonzero_pruning_margin_line_count = 1`

当前这说明的不是“效果已经提升”，而是：

- pruning margin objective 已经真实接入当前 official line；
- loss 不再只是配置层面存在，而是已经进入训练日志与解析报告；
- `run_protocol_aware_pruning_report.sh` 已能对当前 run 给出结构化解释。

## 8. `fix4` 当前暴露出的真实问题

`fix4` 的价值不在于它已经证明收益，而在于它第一次把问题显式量化出来。

当前三层 pruning stage 的短跑结果都是：

- `mean_violation_ratio = 1.0`
- `max_violation_ratio = 1.0`

并且：

- stage 0 `mean_margin_mean ~= 9.313e-09`
- stage 1 `mean_margin_mean = 0.0`
- stage 2 `mean_margin_mean ~= 2.049e-08`

这表示当前 conservative profile 虽然已经接上线，但面对现有 boundary ambiguity：

- 三个 stage 在首轮短跑里都处于“全量违反目标 margin”的状态；
- stage 1 仍然是最明显的高风险层；
- 现在还不能把它表述成“已改善 secure decision stability”。

同时，这条 run 的基本训练结果是：

- `train_loss = 1.3382698046106871`
- `train_class_acc = 0.7680697278911565`
- `test_acc1 = 78.05343514362364`
- `test_loss = 0.4972830739888278`

相较于未真正激活 margin 的同类短跑，当前可以说：

- accuracy 没有出现明显崩坏；
- train loss 明显抬升，符合新增 margin penalty 已真正参与优化的预期；
- 但还没有足够证据宣称它已经带来最终指标收益。

## 9. `focused1`：更强 profile 已验证，但 1-epoch 下未见缓解

在 `fix4` 之后，已经补做：

- 训练 run：
  - `artifacts/train_runs/protocol_aware_pruning_epoch1_20260505_focused1/`
- 报告目录：
  - `results/protocol_aware_pruning_objective/protocol_aware_pruning_epoch1_20260505_focused1/`

它使用的是更强的 `focused` profile：

- `pruning_margin_weight=3.0`
- `pruning_margin_target=3.9e-05`
- `pruning_margin_mode=hinge`
- `pruning_margin_stage_weights=1.40,1.90,0.90`

并且报告同样确认：

- `interpretation.status = "protocol_margin_stats_available"`
- `recipe_comparison.profile_name = "focused"`
- `nonzero_pruning_margin_line_count = 1`

所以这一步可以支持：

- `focused` 不是纸面 profile，而是已经被当前训练链真实执行；
- `hinge + stronger stage-1 pressure` 已经在当前 official line 上跑通。

但当前 1-epoch 结果同样表明：

- stage 0 `mean_violation_ratio = 1.0`
- stage 1 `mean_violation_ratio = 1.0`
- stage 2 `mean_violation_ratio = 1.0`

也就是：

- 更强 profile 并没有在当前短跑里把三层 pruning boundary 从“全量违反目标 margin”拉下来；
- stage 1 依然停留在 `mean_margin_mean = 0.0`；
- 当前不能把 `focused` 解释成已经缓解了 boundary ambiguity。

从训练指标看：

- `focused1 test_acc1 = 77.48091614701366`
- `fix4 test_acc1 = 78.05343514362364`

当前只能保守表述为：

- 更强 profile 没有在 1-epoch 下带来可见 boundary 改善；
- 同时 `test_acc1` 还有约 `0.57` 个点的小幅回落；
- 因此现在不宜继续做“更强权重 + 同样 1-epoch”的盲目加压。

## 10. `focused4`：第一条有效的 3-epoch pair-study 证据

在修复 candidate profile 注入 bug 之后，当前已经拿到第一条真正有效的更长配对结果：

- pair summary：
  - `results/protocol_aware_pruning_objective/protocol_aware_pair_epoch3_20260505_focused4/protocol_aware_pair_compare.json`
- baseline run：
  - `artifacts/train_runs/protocol_aware_pair_epoch3_20260505_focused4_baseline/`
- candidate run：
  - `artifacts/train_runs/protocol_aware_pair_epoch3_20260505_focused4_focused/`

这条 pair-study 现在可以支持两个关键事实：

1. candidate profile 终于被真实注入了
   - baseline：
     - `pruning_margin_weight=0.0`
     - `pruning_margin_target=1e-4`
   - candidate：
     - `pruning_margin_weight=3.0`
     - `pruning_margin_target=3.9e-05`
     - `pruning_margin_stage_weights=1.40,1.90,0.90`

2. 当前 objective 虽然已稳定接线，但还没有出现 boundary relief
   - candidate `interpretation_status = "protocol_margin_stats_available"`
   - `nonzero_pruning_margin_line_count = 4`
   - stage 1：
     - `focus_stage_margin_mean = 0.0`
     - `focus_stage_violation_ratio = 1.0`
   - pair judgement：
     - `status = "no_boundary_relief_yet"`

当前 3-epoch 对照的效果读数是：

- `threshold_accuracy delta = 0.0`
- `auc delta = -9.52e-05`
- `argmax_accuracy delta = +0.3817 pt`

因此，`focused4` 当前最稳妥的解释是：

- `protocol-aware pruning objective` 已经从“短跑激活证据”走到“更长 pair-study 也确实在跑”；
- 它在当前 3-epoch 配置下没有造成 threshold 指标恶化；
- 但 stage 1 仍完全处于 `violation_ratio = 1.0`，所以当前还不能把它升级成“已经缓解 secure boundary ambiguity”的收益证据。

换句话说，`focused4` 把这一项正式收口到：

- objective 已接线；
- candidate 注入 bug 已修复；
- 当前收益判断是“中性偏稳”，不是“已证明改善边界”。

## 10.1 `focused5`：训练预算延长到 5 epoch 后的结果

在 `focused4` 之后，已经补做：

- pair summary：
  - `results/protocol_aware_pruning_objective/protocol_aware_pair_epoch5_20260506_focused5/protocol_aware_pair_compare.json`
- baseline run：
  - `artifacts/train_runs/protocol_aware_pair_epoch5_20260506_focused5_baseline/`
- candidate run：
  - `artifacts/train_runs/protocol_aware_pair_epoch5_20260506_focused5_focused/`

当前可以确认：

- candidate `interpretation_status = "protocol_margin_stats_available"`
- `nonzero_pruning_margin_line_count = 7`
- stage 1：
  - `focus_stage_margin_mean = 0.0`
  - `focus_stage_violation_ratio = 1.0`
- pair judgement：
  - `status = "no_boundary_relief_yet"`

当前 5-epoch 对照的效果读数是：

- `threshold_accuracy delta = -0.3817 pt`
- `auc delta = +9.33e-04`

因此，`focused5` 当前最稳妥的解释是：

- 训练预算从 3 epoch 拉到 5 epoch 后，candidate profile 仍然真实生效；
- 但 stage 1 依然没有从 `violation_ratio = 1.0` 掉下来；
- 同时 `threshold_accuracy` 已出现负向回落；
- 所以下一步不宜继续沿同一条 `focused` 配置盲目加长训练预算。

## 10.2 `conservative5`：当前回传结果不能直接引用

在 `focused5` 之后，已经补做：

- pair summary：
  - `results/protocol_aware_pruning_objective/protocol_aware_pair_epoch5_20260506_conservative5/protocol_aware_pair_compare.json`
- baseline run：
  - `artifacts/train_runs/protocol_aware_pair_epoch5_20260506_conservative5_baseline/`
- candidate run：
  - `artifacts/train_runs/protocol_aware_pair_epoch5_20260506_conservative5_conservative/`

当前回传 JSON 暴露出两个关键矛盾：

- run 名为 `conservative5`，但 candidate 实际参数匹配的是 `focused`
- 同时 `secure_static_train_depth = 6`

这说明：

- 当前 shell 中的 `PAIR_NAME`、`PAIR_CANDIDATE_PROFILE`、`PAIR_SECURE_STATIC_DEPTH` 出现了串扰；
- 这条 compare 不能被当作“conservative profile 的正式证据”引用；
- 如需验证 `conservative`，必须在干净 shell 中重跑。

## 10.3 `depth6 focused_clean1`：deployment-aligned 口径下的结果

在清空旧环境变量后，已经补做：

- pair summary：
  - `results/protocol_aware_pruning_objective/protocol_aware_pair_depth6_epoch5_20260506_focused_clean1/protocol_aware_pair_compare.json`
- baseline run：
  - `artifacts/train_runs/protocol_aware_pair_depth6_epoch5_20260506_focused_clean1_baseline/`
- candidate run：
  - `artifacts/train_runs/protocol_aware_pair_depth6_epoch5_20260506_focused_clean1_focused/`

这条 run 的意义在于：

- 它对齐的是当前正式 secret runtime 的 `secure_static_train_depth = 6`
- 并且 compare 时显式把 `focus_stage_index` 固定到了当前实际产生日志的 stage 0

当前可以确认：

- candidate `interpretation_status = "protocol_margin_stats_available"`
- `nonzero_pruning_margin_line_count = 7`
- focus stage 0：
  - `focus_stage_margin_mean = 5.588e-09`
  - `focus_stage_violation_ratio = 1.0`
- pair judgement：
  - `status = "no_boundary_relief_yet"`

当前对照的效果读数是：

- `threshold_accuracy delta = -0.1908 pt`
- `auc delta = -0.00318004`

因此，`depth6 focused_clean1` 当前最稳妥的解释是：

- 即便把训练口径对齐到当前正式 secret runtime 的 `depth6`，objective 仍然只是“已接线”；
- 它仍没有把 focus stage 的 `violation_ratio` 从 `1.0` 拉下来；
- 同时 `threshold/AUC` 仍是负向；
- 因此当前 `P1-3` 可以暂时收口，不建议继续沿现有 objective/recipe 追加训练预算。

## 11. 当前已经能支持的说法

当前已经能支持：

- `protocol-aware pruning objective` 已经成为当前最终仓的正式可运行增强入口；
- 它的超参入口不再散落在训练源码里，而是被当前 server-pack 封装；
- 它的 recipe 直接由当前 `P1-1` 成本/风险证据驱动；
- 它的短跑日志已经有专门解析工具，可用于判断 objective 是否真的生效；
- `fix4` 已经证明当前 official line 上的 pruning margin objective 不是空接线，而是实际生效；
- `focused1` 已经证明更强的 protocol-aware profile 也可稳定执行，但其短跑收益目前未被看到。
- `focused4` 已经证明 3-epoch pair-study 下 candidate profile 真实生效，但当前仍未出现 boundary relief。
- `focused5` 已经证明把同一 `focused` 配置继续拉到 5 epoch，当前仍未出现 boundary relief。
- `conservative5` 当前因环境串扰，不能作为 `conservative` profile 的正式证据引用。
- `depth6 focused_clean1` 已经证明 deployment-aligned `depth6` 口径下，当前仍未出现 boundary relief。

## 12. 当前还不能支持的说法

当前还不能支持：

- “它已经明确提升了当前 official line 的 threshold/AUC”
- “它已经显著降低了 secure replay mismatch”
- “当前 recipe 已经是最优 protocol-aware 超参”
- “conservative profile 已经足够解决当前三层 stage 的 boundary ambiguity”
- “focused profile 在当前 1-epoch 配置下已经缓解 stage 1 边界风险”
- “focused profile 仅靠继续加长 epoch 就能解决当前 stage 1 边界风险”
- “当前 conservative profile 已经足够解决当前 stage 1 边界风险”
- “只要对齐到 depth6，当前 objective 就会自然带来 boundary relief”

这些都需要新的 short run / full run 结果来支持。

## 13. 推荐命令

先生成当前 clean recipe：

```bash
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh recipe
```

若要先确认当前 shell / local env 不会污染 recipe，可先执行：

```bash
export PROTOCOL_AWARE_PROFILE=focused
export PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh print-env
```

再做第一轮短跑：

```bash
export PROTOCOL_AWARE_PROFILE=conservative
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh debug80
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh epoch1
```

跑完之后生成日志报告：

```bash
export RUN_NAME=<你的训练run名>
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_report.sh
```

如果已经完成 `fix4` 这一步，下一条更有价值的短跑是：

```bash
export PROTOCOL_AWARE_PROFILE=focused
export RUN_NAME=protocol_aware_pruning_epoch1_20260505_focused1
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh epoch1
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_report.sh
```

但截至当前，这条 `focused1` 已经完成，且不建议继续用同类 1-epoch 强化短跑反复横跳。

如果继续做正式 pair-study，建议直接起新名字，例如：

```bash
export PAIR_NAME=protocol_aware_pair_epoch3_20260505_focused4
export PAIR_CANDIDATE_PROFILE=focused
export PAIR_EPOCHS=3
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_pair_study.sh suite
```

## 14. 当前结论

截至 `2026-05-05`，`P1-3` 的合理收口是：

- **先把 protocol-aware pruning objective 做成当前官方主线的正式增强入口**
- **再用 `fix4`、`focused1` 与 `focused4` 判断它是否值得继续投入更多训练预算**
- 当前新增的 `focused5` 也已经说明：训练预算不是唯一瓶颈，profile 选择本身仍需调整

也就是说，当前它已经从“源码里藏着一组 margin 参数”升级成了：

**有 recipe、有 wrapper、有日志解析、有真实激活证据、有主线文档位置的正式 P1 增强项。**

但同时，当前短证据也已经足够说明：

- 它还没有进入“已验证收益”的阶段；
- `focused` 1-epoch 没有把核心 violation 拉下来；
- `focused` 5-epoch 也没有把核心 violation 拉下来；
- 下一步更合理的是转向：
  - 暂停当前 `P1-3` 训练预算扩展；
  - 或为该目标服务的 `secure-friendly operator family` / 表征稳定性增强，
  - 而不是继续盲目抬高 margin 权重、拉长 epoch，切 profile，或重复 depth6 对齐训练。

## 15. 当前正式下一步入口

当前仓里已经补上用于“更长配对训练证据”的正式 wrapper：

- `artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_pair_study.sh`

它的定位是：

- 固定同一 base bundle / teacher / data / static-depth 训练口径；
- 以 `baseline vs protocol-aware candidate` 的方式做 paired study；
- 自动补齐：
  - checkpoint threshold search
  - plaintext eval
  - pruning margin report
  - paired compare report

因此，后续这条线不应再靠手工拼接 run 名和 compare 命令，而应优先通过这条 wrapper 产生正式证据链。
