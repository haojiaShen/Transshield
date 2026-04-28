# Margin-aware pruning ablation 对比

## 口径

- Baseline risk JSON：`/data/wyb/Transshield_final/results/stagewise_protocol_risk_tracka_lr3e5_verified_20260415.json`
- Candidate 数量：`1`
- 主要观察目标：Stage 2 boundary margin 是否变大、near-boundary 比例是否下降、tie 风险是否不变坏。
- 这份报告只用于算法 ablation，不会自动替换 Web demo 默认 bundle。

## Candidate：`margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4_bundle`

- Bundle：`/data/wyb/Transshield_final/artifacts/frozen_candidates/margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4_bundle`
- 样本数：`524`
- Argmax Acc：`85.1145`
- Threshold Acc：`91.6031`
- AUC：`0.967476`
- 建议：先不要替换默认 bundle；继续调小/调大 margin 权重或只保留为实验记录

| Stage | Layer | Baseline margin mean | Candidate margin mean | Margin ratio | Baseline <=1e-4 | Candidate <=1e-4 | Delta <=1e-4 | Tie delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.000696791 | 0.000571526 | 0.820x | 20.80% | 13.55% | -7.25% | 0.00% |
| 2 | 6 | 1.8524e-05 | 0.000371079 | 20.032x | 98.66% | 42.56% | -56.11% | -0.19% |
| 3 | 9 | 0.00019854 | 0.000595782 | 3.001x | 44.66% | 34.16% | -10.50% | 0.00% |

