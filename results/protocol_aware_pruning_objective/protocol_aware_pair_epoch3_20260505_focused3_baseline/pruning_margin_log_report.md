# Pruning Margin Log Report

## 1. 结论

- status: `loss_info_present_but_no_stage_margin_stats`
- reason: loss info lines exist, but margin_stats is empty/none in the parsed window.

## 2. 当前配置

- pruning_margin_weight: `0.0`
- pruning_margin_target: `1e-4`
- pruning_margin_mode: `hinge`
- pruning_margin_stage_weights: `None`
- pruning_margin_start_epoch: `0`
- debug_max_steps: `0`

## 3. 日志摘要

- loss_info_line_count: `4`
- stage_margin_line_count: `0`
- mean_pruning_margin: `0.00000000`
- max_pruning_margin: `0.00000000`
- nonzero_pruning_margin_line_count: `0`

## 4. Stage 汇总

| Stage | Entries | OK | Mean Weight | Mean Margin | Mean Viol | Max Viol | Mean Stage Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N/A | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
