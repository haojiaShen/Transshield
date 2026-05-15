# Protocol-Aware Pruning Recipe

## 1. 结论

- 当前 `P1-3` 的收口不是“已经证明更高精度”，而是把现有 margin-aware 接口变成正式可运行、可报告、可复验的 protocol-aware 训练入口。
- 推荐起点 profile：`conservative`

## 2. 当前 clean 证据为什么指向 protocol-aware pruning objective

- dominant cost stage: `stage 0`
- dominant risk stage: `stage 1`

| Stage | Layer | Cost Share | Tie Ratio | All-Equal Ratio | Risk Tier | Driver |
|---|---:|---:|---:|---:|---|---|
| 0 | 3 | 42.56% | 93.13% | 0.00% | medium | tie_dominated_and_numeric_margin_smaller_than_kth_error |
| 1 | 6 | 36.45% | 100.00% | 100.00% | high | all_active_equal_boundary |
| 2 | 9 | 20.99% | 86.64% | 0.00% | medium | numeric_margin_smaller_than_kth_error |

## 3. 推荐 profile

### conservative

- pruning_margin_weight: `1.00`
- pruning_margin_target: `0.00002600`
- pruning_margin_mode: `softplus`
- pruning_margin_stage_weights: `1.50,1.65,1.00`
- pruning_margin_start_epoch: `0`

### focused

- pruning_margin_weight: `3.00`
- pruning_margin_target: `0.00003900`
- pruning_margin_mode: `hinge`
- pruning_margin_stage_weights: `1.40,1.90,0.90`
- pruning_margin_start_epoch: `0`

## 4. 运行建议

- `debug80` 只检查命令接线、非有限值和早期稳定性；默认 80 step 不会触发 100-step 的 `loss info` 打印。
- `epoch1` 是当前数据规模下第一条应当产出 `pruning_margin` / `margin_stats` 日志的短跑模式。
- 如果 conservative `epoch1` 能稳定出日志，再考虑切换 `focused` 做更强约束。

## 5. 推荐命令

```bash
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh recipe
export PROTOCOL_AWARE_PROFILE=conservative
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh debug80
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_train.sh epoch1
bash artifacts/server_inference_friendly_pack/run_protocol_aware_pruning_report.sh
```
