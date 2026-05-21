# 医疗消融矩阵（动态安全剪枝主线）

## 1. 比较合同

除非某一项本身就是被测变量，医疗域比较默认固定为：

- 同 bundle 主来源
- 同 `party_local_debug_share_load`
- 同 `spu_params_mode = secret`
- 同 `reveal_policy = final_logits_only`
- 同样本集
- 同 `uniform attention + fixed_square + exact LN`

## 2. 语义核心项

| 技术点 | 代码入口 | 当前证据 | 结论 | 原因 |
|---|---|---|---|---|
| pruning boundary rewrite（`masking -> F_mux`, `threshold compare -> F_less`） | `models/dyvit.py`、`spu_static_vit.py` | 主文档与当前 secure pruning 运行链长期固定 | 采用 | 隐私边界更完整 |
| encoded-key bitonic sort | `spu_static_vit.py` | 从 `1` 样本复核开始，secure pruning 全链稳定使用 | 采用 | 更稳定 |
| PredictorLG in-SPU | `spu_static_vit.py` | `runtime_pruning_keep_mask_pt = null`、`host_model_params_materialized = false` | 采用 | 隐私边界更完整 |
| dynamic-path public threshold calibration | `results/report_evidence/medical_dynamic_threshold_calibration/` | 全量验证集 `best_threshold_accuracy = 92.7481%`；`32` 样本复核集回代后 `93.75%` | 采用 | 精度保持或更高 |

## 3. 运行时与数值项

| 技术点 | 对照 | 当前证据 | 结论 | 原因 |
|---|---|---|---|---|
| batch4 | batch1 depth12 | `160.6s/sample` vs `213.9s/sample` | 不作为最终配置 | 同条件下无收益 |
| 批次规模 `8` | 批次规模 `1`、深度 `12` | `113.3s/sample` vs `213.9s/sample` | 不作为最终配置 | 同条件下无收益 |
| 批次规模 `12` | 批次规模 `8`、深度 `10` | `69.57s/sample` vs `100.5s/sample` | 采用为小样本最快配置 | 提速 |
| 批次规模 `8` 的 `32` 样本复核配置 | 批次规模 `12` 的 `32` 样本复核配置 | 批次规模 `12` OOM；批次规模 `8` 可复现 `depth12/depth10` | 采用为当前服务器正式复核配置 | 更稳定 |
| depth10 | depth12，同批次规模 `12` / `8` 同口径 | `69.57s/sample`（`12` 样本工程复核集）与 `86.91s/sample`（`32` 样本复核集） | 采用 | 提速 |
| `fxp16` | `fxp3` | `fxp3` 速度几乎不变，但 `threshold_match = 66.67%` | 采用 `fxp16` | 更稳定 |
| token recycle `0.1` | 批次规模 `12` + 深度 `10` | `70.85s` vs `69.57s` | 不采用 | 同条件下无收益 |
| token_ratio speedup | 同实现基线 | 文档已明确“full-shape masking 下不提速” | 不采用 | 同条件下无收益 |

## 4. 算子族与控制线

| 技术点 | 对照 | 当前证据 | 结论 | 原因 |
|---|---|---|---|---|
| uniform attention | 非 uniform 诊断线 | secure pruning 与控制线都固定采用 uniform | 采用 | 更稳定 |
| fixed_square | 非 fixed_square 诊断线 | 当前 secure pruning 主线持续依赖 fixed_square | 采用 | 更稳定 |
| exact LN | public-calibrated LN + clip0 | exact LN 能稳定作为 secure pruning 与 static control 的共同基础 | 采用 | 更稳定 |
| public-calibrated LN + clip0 | `exact_uniform_clip0` | 自然均匀 `32` 样本复核配置仅 `46.875%`，raw logits 量级失稳 | 不采用 | 精度下降不可接受 |
| clip3 路线 | `exact_uniform_clip0` | 自然均匀 `32` 样本复核配置仅 `50.0%`，`regressed = 14` | 不采用 | 精度下降不可接受 |

## 5. 结构 / 压缩 / 训练项

| 技术点 | 对照 | 当前证据 | 结论 | 原因 |
|---|---|---|---|---|
| LRD rank192 merged（医疗） | depth10 dynamic 主线 | 仓内旧记录明确“与 dynamic depth10 相比无额外速度收益” | 不采用 | 同条件下无收益 |
| LRD rank96 decomposed | dynamic 主线 | `96.55s/sample`，比主线慢 `38.8%` | 不采用 | 同条件下无收益 |
| distillation | 无蒸馏同口径 | official1 / cls-only1 都是 `no_clear_distill_benefit_yet` | 不采用 | 同条件下无收益 |

## 6. 本轮新增解释

- `32` 样本复核集上，批次规模 `8` 的 depth12 / depth10 默认阈值都只有 `50%`，并不代表 dynamic secure pruning 失效。
- 根因是：
  - dynamic path 的概率边界整体上移
  - static bundle threshold `0.3577311039` 不能直接复用
- 所以当前医疗正式主线的正确部署写法必须是：
  - `dynamic secure pruning + full privacy`
  - `+ dynamic 路径全量验证集公开阈值校准`

## 7. 未升级项说明

- `LUT GELU`
  - 当前 `spu_static_vit.py` 仍保留 hook
  - 但本轮没有新的 accepted run
  - 继续只记为代码候选，不升级成 adopted innovation

- `32` 样本复核集、批次规模 `12`
  - 当前服务器上已明确 OOM
  - 因此本轮不再把它当正式复核配置前提
