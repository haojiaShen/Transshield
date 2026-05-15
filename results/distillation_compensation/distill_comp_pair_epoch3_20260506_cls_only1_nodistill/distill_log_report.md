# Distill Log Report

## 1. 结论

- status: `distill_disabled_reference`
- reason: command.sh keeps both cls/token distill weights at 0, so this run acts as the no-distill reference.

## 2. 当前配置

- cls_distill_weight: `0.0`
- token_distill_weight: `0.0`
- ratio_weight: `0.0`
- debug_max_steps: `0`

## 3. 日志摘要

- loss_info_line_count: `4`
- mean_cls_kl: `0.16347500`
- max_cls_kl: `0.18640000`
- mean_token_kl: `0.47135000`
- max_token_kl: `0.47430000`
- mean_effective_cls_term: `0.00000000`
- mean_effective_token_term: `0.00000000`
- nonzero_effective_distill_line_count: `0`
