# Pruning Margin Log Report

## 1. 结论

- status: `protocol_margin_stats_available`
- reason: stage-wise pruning margin stats were parsed successfully from train_stdout.log.

## 2. 当前配置

- pruning_margin_weight: `1.0`
- pruning_margin_target: `2.6e-05`
- pruning_margin_mode: `softplus`
- pruning_margin_stage_weights: `1.50,1.65,1.00`
- pruning_margin_start_epoch: `0`
- debug_max_steps: `0`

## 3. 日志摘要

- loss_info_line_count: `1`
- stage_margin_line_count: `1`
- mean_pruning_margin: `0.69316020`
- max_pruning_margin: `0.69316020`
- nonzero_pruning_margin_line_count: `1`

## 4. Stage 汇总

| Stage | Entries | OK | Mean Weight | Mean Margin | Mean Viol | Max Viol | Mean Stage Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1.5000 | 0.00000001 | 1.0000 | 1.0000 | 0.69320000 |
| 1 | 1 | 1 | 1.6500 | 0.00000000 | 1.0000 | 1.0000 | 0.69320000 |
| 2 | 1 | 1 | 1.0000 | 0.00000002 | 1.0000 | 1.0000 | 0.69320000 |

## 5. Recipe 对照

- profile_name: `conservative`
- matches_weight: `true`
- matches_target: `true`
- matches_mode: `true`
- matches_start_epoch: `true`
- matches_stage_weights_csv: `true`
