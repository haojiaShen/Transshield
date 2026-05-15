# Distill Log Report

## 1. 结论

- status: `distill_terms_observed`
- reason: distill loss lines were parsed successfully and the configured weights produce non-zero effective distill terms.

## 2. 当前配置

- cls_distill_weight: `1.0`
- token_distill_weight: `0.02`
- ratio_weight: `0.0`
- debug_max_steps: `0`

## 3. 日志摘要

- loss_info_line_count: `4`
- mean_cls_kl: `0.06755000`
- max_cls_kl: `0.07020000`
- mean_token_kl: `0.45610000`
- max_token_kl: `0.47230000`
- mean_effective_cls_term: `0.06755000`
- mean_effective_token_term: `0.00912200`
- nonzero_effective_distill_line_count: `4`
