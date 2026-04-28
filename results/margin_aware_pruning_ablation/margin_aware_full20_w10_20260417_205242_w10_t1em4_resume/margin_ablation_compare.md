# Margin-aware pruning ablation 对比

## 口径

- Baseline risk JSON：`/data/wyb/Transshield_final/results/stagewise_protocol_risk_tracka_lr3e5_verified_20260415.json`
- Candidate 数量：`1`
- 主要观察目标：Stage 2 boundary margin 是否变大、near-boundary 比例是否下降、tie 风险是否不变坏。
- 这份报告只用于算法 ablation，不会自动替换 Web demo 默认 bundle。

## Candidate：`margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`

- Bundle：`/data/wyb/Transshield_final/artifacts/frozen_candidates/margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`
- 样本数：`524`
- Argmax Acc：`88.9313`
- Threshold Acc：`90.2672`
- AUC：`0.956508`
- 建议：可以进入 secure replay / SPU 一致性检查

| Stage | Layer | Baseline margin mean | Candidate margin mean | Margin ratio | Baseline <=1e-4 | Candidate <=1e-4 | Delta <=1e-4 | Tie delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.000696791 | 0.000953475 | 1.368x | 20.80% | 19.27% | -1.53% | 0.00% |
| 2 | 6 | 1.8524e-05 | 0.00451118 | 243.532x | 98.66% | 5.92% | -92.75% | -0.19% |
| 3 | 9 | 0.00019854 | 0.000592817 | 2.986x | 44.66% | 24.81% | -19.85% | 0.00% |

