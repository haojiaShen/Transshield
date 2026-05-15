# Distill Log Report

## 1. 结论

- status: `distill_terms_observed`
- reason: distill loss lines were parsed successfully and the configured weights produce non-zero effective distill terms.

## 2. 当前配置

- cls_distill_weight: `1.0`
- token_distill_weight: `0.0`
- ratio_weight: `0.0`
- debug_max_steps: `0`

## 3. 日志摘要

- loss_info_line_count: `4`
- mean_cls_kl: `0.06735000`
- max_cls_kl: `0.07000000`
- mean_token_kl: `0.48580000`
- max_token_kl: `0.48780000`
- mean_effective_cls_term: `0.06735000`
- mean_effective_token_term: `0.00000000`
- nonzero_effective_distill_line_count: `4`
