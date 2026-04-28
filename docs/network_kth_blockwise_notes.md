# Blockwise exact-kth 研究记录

最后更新：`2026-04-18`

本页记录 `Phase 3 network-kth` 中 `blockwise_exact_kth` 的当前有效结论。

## 当前定位

`blockwise_exact_kth` 是对原始 `flat_odd_even` compare-network 的**协议侧改造**：

- 不改变当前模型语义
- 不改变 tie policy
- 不改变最终 `kth_threshold` 正确性要求
- 只改变 `network-kth` 的内部选择路径

当前它已经升级为正式 secure pipeline / Web demo secure run 的默认选择模式；旧 `flat_odd_even` 只保留为 reference fallback。

## 当前实现状态

- manifest 生成器：
  - `tools/transshield_blockwise_kth_selection_manifest.py`
- 运行 bridge：
  - `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
- pipeline 入口：
  - `tools/transshield_openbumblebee_pipeline.py`
- 服务器 profile 入口：
  - `artifacts/server_inference_friendly_pack/run_secure_selection_mode_profile_compare.sh`

## 理论侧估计

来源：`/data/wyb/Transshield_final/results/blockwise_exact_kth_manifest_20260418_004103.md`

| Stage | Layer | Active | Keep | Candidate | Total comparator ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 196 | 137 | 120 | `0.8711x` |
| 1 | 6 | 137 | 96 | 84 | `0.8705x` |
| 2 | 9 | 96 | 67 | 60 | `0.8829x` |

一句话理解：如果只看 compare-network comparator 数量，当前设计预估能带来约 `11% ~ 13%` 的压缩。

## 本地语义正确性验证

在仓库内 `smoke8` runtime inputs 上，`blockwise_exact_kth` 已通过：

- `tools/transshield_secure_network_kth.py check`
- 逐 stage `max_abs_error = 0.0`
- `overall_passed = true`

这说明当前 blockwise manifest 在 **kth_threshold 语义** 上已经和 reference 完全一致。

## 服务器 replay / branch 语义验证已通过

来源：

- `blockwise_exact_kth` replay 目录：
  - `/data/wyb/Transshield_final/artifacts/server_pipeline_run/blockwise_vs_flat_20260418_004346_blockwise_exact_kth`
- 关键文件：
  - `pipeline_inference_replay_summary.json`

当前已经补齐 full replay / branch reconstruction 这一步，结果为：

- `overall_passed = true`
- 三个 pruning stage 都是 `overall_passed = true`
- 每个 stage 都满足：
  - `exact_count_match_ratio = 1.0`
  - `exact_mask_match_ratio = 1.0`
  - `mean_jaccard_vs_topk = 1.0`
  - `reconstructed_branch_matches_topk_reference = true`
  - `max_abs_error_after_snap_vs_argsort_reference = 0.0`
- `threshold_snap` 也稳定：
  - `snapped_count = 524`
  - `sample_count = 524`
  - `max_distance` 约 `1.52e-05`
  - 容差仍是 `5e-05`

从 stage 细节看：

| Stage | Layer | Keep | Boundary tie sample ratio | 结果 |
|---:|---:|---:|---:|---|
| 0 | 3 | 137 | `0.00%` | 通过 |
| 1 | 6 | 96 | `0.19%` | 通过 |
| 2 | 9 | 67 | `0.00%` | 通过 |

这说明 `blockwise_exact_kth` 不只是 profile 更快，它在 **full replay 语义** 上也已经和原始 top-k reference 对齐。

## 服务器 SPU profile 正结果

来源：

- 对比目录：`/data/wyb/Transshield_final/artifacts/server_profile_reports/blockwise_vs_flat_20260418_004346_selection_mode_compare`
- 对比报告：`/data/wyb/Transshield_final/artifacts/server_profile_reports/blockwise_vs_flat_20260418_004346_selection_mode_compare/selection_mode_profile_compare.md`

对比口径：

- A：`flat_odd_even`
- B：`blockwise_exact_kth`
- 两边都通过 `verify`
- 通信源都来自 `python_distributed_rpc_cloudpickle`

关键结果：

| 指标 | flat_odd_even | blockwise_exact_kth | 变化 |
|---|---:|---:|---:|
| Total pipeline duration | `16.9257s` | `15.6245s` | `-1.3012s` |
| Network-kth bridge | `11.6141s` | `10.3066s` | `-1.3075s` |
| Tie bridge | `0.0411s` | `0.0174s` | `-0.0237s` |
| Communication total bytes | `1.72 MB` | `1.72 MB` | `0.00 B` |

比例视角：

- total pipeline：`0.923x`
- network-kth bridge：`0.887x`
- tie bridge：`0.423x`
- communication bytes：`1.000x`

## 当前结论

### 1. 这是第一轮真正有意义的 Phase 3 正结果

和之前旧的 `phase3_lower_tail` 不同，这次 `blockwise_exact_kth` 在**同口径 SPU profile** 下已经出现了明确加速：

- `network_kth_bridge` 下降约 `11.3%`
- `total pipeline` 下降约 `7.7%`

### 2. 当前收益主要来自时间，不来自通信

这轮结果里：

- 时间下降是明显的
- 通信总字节数基本不变

这意味着当前 `Phase 3` 已经开始减少在线 compare-network 执行负担，但 `Phase 4 payload` 仍然是必要的，因为通信还没有跟着降下来。

这里有一个当前实现层面的**准确解释**：

- `flat_odd_even` 和 `blockwise_exact_kth` 在进入 SPU 之前，都先执行同一套 `active token compaction`
- 也就是说，两边送进 SPU 的 `masked_score` 紧凑张量形状当前是一样的
- `blockwise_exact_kth` 改掉的是 **SPU 内部如何求 exact kth**
- 它减少了 compare-network 的在线执行负担，所以时间下降
- 但它没有减少进入 SPU 前的输入 payload 体积，所以 Python fastpath RPC bytes 暂时不降

因此当前这组结果的正确解读是：

- **Phase 3 先拿到了算子 / 调度层面的加速**
- **Phase 4 才负责把输入 payload / RPC bytes 压下来**

### 3. 当前已经升级为正式 secure 默认模式

当前已经有：

- checker 通过
- tie checker 通过
- SPU profile 正向
- full replay / pipeline 语义再确认通过

所以当前收口为：

- `blockwise_exact_kth` 已经作为正式 secure pipeline / Web demo secure run 的默认选择模式
- 默认仍保持 `payload_dtype=float32`，不启用 Phase 4 mixed payload
- 如需复现实验对照，可显式设置 `KTH_SELECTION_MODE=flat_odd_even` 回退旧 reference
- 下一步不再是补 replay，而是继续记录 / 解释 Phase 4 payload 的负结论

## 推荐下一步

1. 保留 `blockwise_exact_kth + float32` 作为当前正式 secure 默认路径
2. 将 Phase 4 mixed payload 保留为研究 / 诊断证据
   - 重点不是把它设为默认
   - 而是说明 host 侧压缩生效但当前 SPU 重构代价过高
3. 后续如果要继续展示收益：
   - 优先展示“同口径 profile 时间下降”
   - 通信量要等 payload 真降了再更新

## 明确禁止的写法

- 不要把这组 profile 数字写成“单图 live run 通信量”
- 不要把它写成“full-val sidecar 总通信量”
- 不要把 `blockwise_exact_kth` 写成 mixed payload；当前正式默认是 `blockwise_exact_kth + float32`
