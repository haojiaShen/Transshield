# 最终采用决策（固定格式）

| 技术点 | 领域 | 是否采用 | 证据 | 原因 |
|---|---|---|---|---|
| pruning boundary rewrite（`masking -> F_mux`, `threshold compare -> F_less`） | 医疗 / 金融 | 采用 | `docs/transshield_master_plan_20260505.md`、`models/dyvit.py`、`spu_static_vit.py` | 隐私边界更完整 |
| encoded-key bitonic sort | 医疗 / 金融 | 采用 | `spu_static_vit.py` secure pruning 路径 | 更稳定 |
| PredictorLG in-SPU | 医疗 / 金融 | 采用 | 医疗 `32` 样本复核与金融动态 `8` 样本压力验证都满足 `runtime_pruning_keep_mask_pt = null` 且 `host_model_params_materialized = false` | 隐私边界更完整 |
| dynamic-path public threshold calibration | 医疗 | 采用 | `results/report_evidence/medical_dynamic_threshold_calibration/`：全量验证集 `92.7481%`，`32` 样本复核集回代 `93.75%` | 精度保持或更高 |
| batch12 + depth10 | 医疗 | 采用为小样本最快配置 | `69.57s/sample`，相对 baseline `3.07x` | 提速 |
| batch8 + depth10 | 医疗 | 采用为当前服务器正式复核配置 | `32` 样本复核集可复现，`86.91s/sample`，隐私全过 | 更稳定 |
| `fxp16` | 医疗 | 采用 | `fxp3` 虽略快但 `threshold_match = 66.67%` | 更稳定 |
| Cheetah packed matmul（保持默认开启） | 医疗 / 金融 | 采用 | `results/report_evidence/protocol_variant_assessment/protocol_variant_assessment_summary.json`：`disable_matmul_pack` 在批次规模 `1` 下虽更快，但双向通信约放大到 `3x`，且批次规模 `8` 的代表路径仍会失稳 | 更稳定 |
| token_ratio speedup | 医疗 | 不采用 | 文档已明确“full-shape masking 下不提速” | 同条件下无收益 |
| token recycle `0.1` | 医疗 | 不采用 | `70.85s` vs `69.57s` | 同条件下无收益 |
| public-calibrated LN + clip0 | 医疗 | 不采用 | 自然均匀 `32` 样本复核配置仅 `46.875%`，且 raw logits 失稳 | 精度下降不可接受 |
| clip3 路线 | 医疗 | 不采用 | 自然均匀 `32` 样本复核配置仅 `50.0%`，`regressed = 14` | 精度下降不可接受 |
| ABY2.0 mixed compare reactivation | 医疗 / 金融 | 不采用 | 历史服务器完整记录仅 `1.033x`；当前官方 packaged runtime 也不再直接暴露 `mixed_compare_mode` | 同条件下无收益 |
| positive-domain truncation scheduling | 医疗 | 不采用 | `docs/report_evidence/14_protocol_variant_completion_validation.md`：`1/2` 样本 + 批次规模 `1` 已完整跑通，但 `199.12 > 193.08`、`179.86 > 178.66`，通信几乎不变 | 同条件下无收益 |
| distillation | 医疗 / 金融 | 不采用 | 医疗 official1 / cls-only1 无收益；金融 README 明确 `57%~66%` 崩塌 | 精度下降不可接受 |
| LRD rank192 merged（医疗默认主线） | 医疗 | 不采用 | 与 dynamic depth10 相比无额外速度收益 | 同条件下无收益 |
| LRD rank96 decomposed | 医疗 | 不采用 | `96.55s/sample`，比主线慢 `38.8%` | 同条件下无收益 |
| LRD rank192 merged | 金融 | 采用 | 当前 dynamic / static 两臂都建立在它上面，参数压缩到 `68.39%` | 精度保持或更高 |
| 动态安全推理 + LRD rank192 | 金融 | 采用 | `105.16s/sample`，`100%`，隐私全过（内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`） | 隐私边界更完整 |
| 固定结构 no-pruning + LRD rank192（作为默认正式主线的对照线） | 金融 | 不采用 | `103.64s/sample`，与动态路径预测完全一致，仅快 `1.52s/sample` | 同条件下无收益 |
| 明文输入 LRD 对照线 | 金融 | 不采用 | 旧 8 样本对照线不是完整隐私口径 | 破坏完全隐私 |
| 历史保留掩码金融对照线（keep-mask） | 金融 | 不采用 | 只作跨域历史证据，不是当前正式合同下的默认线 | 只是历史链/对比链，不适合作为最终主线 |

注：

- `disable_matmul_pack` 现已补跑完成；它不再是“没跑完”的失败项，而是“仅在批次规模 `1` 的低批量档位下出现时间正信号，但当前正式批次规模 `8` 路径不稳定”的研究候选。
