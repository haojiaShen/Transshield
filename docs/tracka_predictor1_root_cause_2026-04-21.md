# TrackA `predictor_1` 过度 pruning 根因诊断（`2026-04-21`）

## 1. 任务完成状态

本轮最初被安排的任务已经完成。

任务目标是判断：`epoch=2 step=146` 附近，`predictor_1` 被推向过度 pruning，首发驱动到底是不是 `ratio loss`。

当前已经可以给出明确因果判断，而不是只停留在现象描述。

对应 Multica issue `aeb4bb6d-1d97-49a9-8e13-a1557af80b13` 已完成：本 issue 的交付范围是完成根因判断、归档证据链、同步当前状态与下一步 handoff，不包含改前端展示、修改 `best_demo_content.json`、启动 `full20` 或大范围超参搜索。

## 2. 最终结论

- `ratio loss` **不是**把 `predictor_1` 首先推向过度 pruning 的首发驱动。
- 更可能的顺序是：
  1. `cls_loss / cls_kl / token_kl` 等非 `ratio` 梯度链路先把 `predictor_1` 往 `drop` 方向推偏；
  2. `predictor_1` 的 keep ratio 持续下滑，margin 持续朝 `drop` 侧扩大；
  3. `ratio loss` 由于实际 keep 已明显低于 target，开始反向拉 `keep`；
  4. 多条梯度在 `epoch=2 step=146` 左右形成强烈拉扯，表现为 `score_predictor.1.out_proj.weight` 的大梯度爆发。

一句话收口：**先把 `predictor_1` 推偏的不是 `ratio loss`，而是非 `ratio` 路径；`ratio loss` 更像是在塌缩后尝试把 keep 拉回。**

## 3. 证据来源

本次判断基于服务器 `epoch3 guard-on` 诊断 run 的日志摘录，核心证据来自用户回传的 `grep` 输出，关注窗口为 `epoch=2 step=140~146`，日志类型包括：

- `predictor_1/2_keep_diag`
- `ratio_stage_0/1/2`
- `grad_watch parameter=score_predictor.1.out_proj.weight`

对应 run 名为：

- `tracka_source_epoch3_sched20_guardon_diag2_seed0`

## 4. 关键证据

### 4.1 `predictor_1` 先持续塌向 `drop`

`predictor_1_keep_diag` 在 `step 140 -> 146` 的趋势非常清楚：

- `raw_keep_ratio_mean`: `0.2974086 -> 0.1548948`
- `active_margin_mean`: `-0.8394073 -> -1.722985`

这说明两件事：

1. `predictor_1` 的保留比例在持续下降；
2. 它不是在“犹豫”地掉，而是在越来越自信地朝 `drop` 方向走。

### 4.2 `predictor_2` 没有先失控

同一窗口里，`predictor_2_keep_diag` 的 `active_margin_mean` 基本稳定在 `0.065` 左右，没有出现与 `predictor_1` 同等级别的持续恶化。

这更像是：

- `predictor_1` 先出问题；
- `predictor_2` 是后续链式症状，而不是首发原因。

### 4.3 `ratio_stage_i` 的方向与“继续压 keep”相反

如果是 `ratio loss` 在主动推动过度 pruning，那么更合理的现象应该是：实际 keep 已高于 target，或至少 `ratio loss` 的方向是在继续压 keep。

但当前日志恰好相反。

`step 140`：

- `ratio_stage_0`: `target=0.700000`, `pos_ratio_mean=0.4418048`, `gap_mean=-0.2581952`
- `ratio_stage_1`: `target=0.490000`, `pos_ratio_mean=0.1321747`, `gap_mean=-0.3578253`
- `ratio_stage_2`: `target=0.343000`, `pos_ratio_mean=0.06568877`, `gap_mean=-0.2773112`

`step 146`：

- `ratio_stage_0`: `target=0.700000`, `pos_ratio_mean=0.3871173`, `gap_mean=-0.3128827`
- `ratio_stage_1`: `target=0.490000`, `pos_ratio_mean=0.06042730`, `gap_mean=-0.4295727`
- `ratio_stage_2`: `target=0.343000`, `pos_ratio_mean=0.02822066`, `gap_mean=-0.3147793`

这说明三个 pruning stage 的实际 keep 都显著低于目标 keep，而且到 `step 146` 时偏离目标更严重。

因此从方向上看，`ratio loss` 此时应该是在**拉高 keep**，而不是继续把 `predictor_1` 往过度 pruning 方向推。

### 4.4 首个大梯度仍出现在 `score_predictor.1.out_proj.weight`

`step 146` 的 `grad_watch` 给出的关键异常是：

- parameter: `score_predictor.1.out_proj.weight`
- `min=-1.057964e+03`
- `max=1.057964e+03`

这个梯度峰值本身说明 `predictor_1` 路径确实在这里出现了强烈冲突，但结合上面的 `ratio_stage_i` 方向判断，它更像是：

- 非 `ratio` 路径先把 `predictor_1` 推向 `drop`
- `ratio loss` 在 keep 已过低后强力反拉
- 最终在该点表现为大梯度对冲

而不是“`ratio loss` 单方面把它往 drop 推爆”。

## 5. 因果判断为何已经足够明确

当前证据链已经同时满足三点：

1. **现象定位明确**  
   `predictor_1` 的 keep ratio 与 margin 在 `step 140~146` 单调恶化。

2. **方向证据明确**  
   `ratio_stage_0/1/2` 全部显示实际 keep 低于 target，因此 `ratio loss` 的方向应是增加 keep。

3. **参数级异常明确**  
   首个大梯度爆发点落在 `score_predictor.1.out_proj.weight`，与 `predictor_1` 的异常轨迹一致。

因此，本次原始任务“判断首发驱动是谁”已经可以收口，不需要再把结论停留在“还不确定是谁推动了过度 pruning”。

## 6. 对正式展示口径的影响

这次结论只属于 **TrackA 明文重训根因诊断**，不改变以下正式口径：

- Web demo 当前展示逻辑
- `artifacts/web_demo_assets/best_demo_content.json`
- 当前冻结 bundle 的正式展示指标

正式展示成绩仍以当前冻结 bundle 为准：

- Argmax：`93.702292%`
- Threshold：`94.083971%`
- AUC：`0.972332`

## 7. 本次文档落地范围

本次只做结论归档与状态同步，未修改前端展示口径，未改 `best_demo_content.json`，也未在本仓新增长训或大范围调参动作。

截至 `2026-04-21`，该 issue 可视为已完成。后续若继续推进，应另起“非 `ratio` 路径最小单变量修复验证”任务，而不是继续在本 issue 内重复判断 `ratio loss` 是否为首发驱动。
