# 金融消融矩阵（dynamic secure pruning 主线）

## 1. 比较合同

金融域默认比较固定为：

- 同 `data/finance_fraud_v3`
- 同 `party_local_debug_share_load`
- 同 `spu_params_mode = secret`
- 同 `reveal_policy = final_logits_only`
- 同 `LRD rank192 merged` bundle
- 同样本集

## 2. 主矩阵

| 技术点 | 当前证据 | 结论 | 原因 |
|---|---|---|---|
| LRD rank192 merged | 参数量 `22,390,184 -> 15,312,296`，当前 dynamic / static 两臂都基于它重跑成功 | 采用 | 精度保持或更高 |
| 动态安全推理正式主线 | `105.16s/sample`，`100%`，隐私全过（内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`） | 采用 | 隐私边界更完整 |
| 固定结构对照线 | `103.64s/sample`，`100%`，与动态路径预测完全一致（内部运行标识：`finance_lrd_rank192_true_static_partylocal_secret_smoke8_20260519_1`） | 不采用为默认主线 | 同条件下无收益 |
| 历史 keep-mask 对照线 | 内部运行标识：`finance_keepmask_smoke8_20260511_131750` | 不采用 | 只是历史链/对比链，不适合作为最终主线 |
| 明文输入对照线 | `196.39s`，且不是完整隐私 | 不采用 | 破坏完全隐私 |
| distillation | README 已明确会把 accuracy 压到 `57%~66%` | 不采用 | 精度下降不可接受 |
| uniform attention | 当前 dynamic / static 两臂都使用 uniform | 采用 | 更稳定 |
| fixed_square | 当前 dynamic / static 两臂都使用 fixed_square | 采用 | 更稳定 |

## 3. “是否启用动态剪枝”专门结论表

| 问题 | 当前结论 |
|---|---|
| 代码是否支持 | 支持，且 2026-05-19 已通过显式 `spu_pruning_mode` 开关区分动态 / 静态 |
| 金融域是否跑过正式动态全隐私路径 | 是，内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1` |
| 金融域是否跑过固定结构全隐私对照路径 | 是，内部运行标识：`finance_lrd_rank192_true_static_partylocal_secret_smoke8_20260519_1` |
| 同条件下 dynamic vs static 结果如何 | 两臂都是 `100%`，预测完全一致；静态只快 `1.52s/sample` |
| 当前正式默认为什么选 dynamic | 满足最终用户“动态剪枝 + 完全隐私”要求，且静态优势不足以支撑第二条正式主线 |

## 4. 当前静态线的正确位置

静态 no-pruning 线现在保留，但位置已经变成：

- fixed-shape fallback
- 审计 / 对照线
- 极端部署约束下的保守选项

它不再代表“金融最终默认模型”。

## 5. 本轮不升级项

- depth truncation
  - 当前 repo 仍没有一条被正式 adopted 的金融 depth 截断证据
  - 本轮不升级

- token recycle
  - 当前 repo 没有金融域 accepted dynamic token recycle 证据
  - 不升级

- token_ratio speedup
  - 当前 repo 没有金融域 accepted 同条件 speedup 证据
  - 不升级
