# Stage-Level Secure Cost / Risk Model

## 1. 概览

- run_dir: `/data/wyb/Transshield_final/artifacts/server_pipeline_run/delivery_line_suite_20260505_clean`
- secret_run_dir: `/data/wyb/Transshield_final/artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean`
- sample_count: `524`
- stage_count: `3`
- network_kth_bridge_sec: `12.4121`
- tie_policy_bridge_sec: `1.2389`
- total_boundary_sidecar_bridge_sec: `13.6510`

## 2. 关键结论

- stage 0 contributes the largest estimated sidecar bridge share: 42.56%.
- stage 1 has the highest heuristic risk tier: high (0.9500).
- stages 0/2 use strict below-threshold margins on the order of 1e-8 while secure kth max abs error stays around 1e-5, so semantic tie handling is part of the method, not an optional patch.
- stage 1 is fully tie-dominated in the current clean run: all active tokens lie on the kth boundary before tie disambiguation.

## 3. Stage 明细

| Stage | Layer | Keep | Active Before | Tie Ratio | Tie Excess Mean | Strict Margin P50 | KTH Error Max | Est. Sidecar Sec | Est. Cost Share | Risk Tier | Driver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 3 | 137 | 196.0 | 93.13% | 8.050 | 0.00000006 | 0.00001240 | 5.8102 | 42.56% | medium | tie_dominated_and_numeric_margin_smaller_than_kth_error |
| 1 | 6 | 96 | 137.0 | 100.00% | 41.000 | N/A | 0.00000024 | 4.9758 | 36.45% | high | all_active_equal_boundary |
| 2 | 9 | 67 | 96.0 | 86.64% | 5.141 | 0.00000006 | 0.00001276 | 2.8650 | 20.99% | medium | numeric_margin_smaller_than_kth_error |

## 4. Secret Runtime 旁证

- accepted_count: `8` / `8`
- complete: `True`
- pending_count: `0`
- unstable_count: `0`
- mean accepted elapsed_sec: `93.8959`
- max raw-logit abs before output calibration: `0.264542`

## 5. 一致性旁证

- argmax_match_ratio: `99.43%`
- threshold_match_ratio: `97.52%`
- logits_max_abs_error: `0.04070145`
- probabilities_max_abs_error: `0.01950449`

## 6. 解释口径

- 这里的 cost 是 stage-level 近似分摊模型，不是假装拿到了协议内部每个 gate 的精确耗时。
- 这里的 risk 重点描述 boundary tie、strict margin、threshold snap 误差三者的关系，用来解释为什么 `F_less + tie sidecar + F_mux` 是主线方法的一部分。

