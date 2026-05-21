# 协议变体服务器验证（2026-05-19）

## 1. 验证目的

本节只回答一个问题：

> 在不改变当前正式模型语义的前提下，ABY2.0 / Cheetah 相关候选有没有一个可以进入当前“密捷”项目正式部署链？

## 2. 代表路径与环境

### 2.1 代表路径

本轮选用的代表路径为：

- 领域：医疗
- 模型语义：`depth10 dynamic secure pruning`
- 隐私边界：`party_local_debug_share_load + secret params + final_logits_only`
- 样本：内部准备标识 `med_secure_pruning_smoke32_prepare_20260519_1` 对应 share manifest 的前 `8` 个样本

之所以先用这条路径，而不是先扩到金融域，原因是：

- 这条路径同时包含了 secure pruning、attention、matmul、fixed-square activation；
- 它对 compare、matmul、truncation 三类协议候选都更敏感；
- 如果在这条代表路径上都不能形成稳定收益，就没有必要再扩到金融域复跑。

### 2.2 服务器环境

- 日期：`2026-05-19`
- 服务器：`10.204.248.175:9001`
- 仓库：`/data/wyb/Transshield_final`
- Python：`/data/wyb/conda_envs/transshield/bin/python`
- 运行入口：协议变体验证脚本 `artifacts/server_inference_friendly_pack/run_protocol_variant_medical_depth10.sh`

### 2.3 本轮新增实验入口

本轮为协议 triage 新增了以下入口：

- 协议变体验证脚本：`artifacts/server_inference_friendly_pack/run_protocol_variant_medical_depth10.sh`
- `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh`
- `tools/transshield_spu_runtime_setup.py`

其中新增支持的运行时开关包括：

- `SPU_CHEETAH_DISABLE_MATMUL_PACK`
- `SPU_CHEETAH_MIXED_COMPARE_MODE`
- `SPU_BB_TRUNC_SIGN_HINT=positive_when_unknown`（仅实验钩子，不是默认项）

## 3. baseline

### 3.1 baseline run

- 运行标识：`transshield_protocol_variant_medical_baseline`

### 3.2 baseline 结果

- `elapsed_sec = 719.26`
- `sec_per_sample = 89.91`
- `finite_logits = true`
- `backend = jax_spu_secure_pruning_forward_backend_v0`
- `forward_scope = student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path`

隐私字段：

- `host_plaintext_pixel_values_materialized = false`
- `host_model_params_materialized = false`
- `reveal_policy = final_logits_only`

### 3.3 baseline 通信证据

baseline 同步回本地的 `fastpath_profile_summary.json` 记录如下：

- `link_send_total_bytes = 11499064547`
- `link_recv_total_bytes = 11230315433`
- `dual_total = 22729379980 bytes`

折合约：

- `21.17 GiB` 双向总量

需要说明：

- 当前这份 fastpath 摘要来自 `logs/spu_nodes/node_0.log` 的已匹配条目；
- 因此它更适合作为**同入口候选之间的相对比较参考**，而不是对外报告中的最终通信主表。

## 4. 候选一：mixed compare reactivation

### 4.1 启用方式

- `SPU_CHEETAH_MIXED_COMPARE_MODE=1`

### 4.2 实际结果

- 运行标识：`protocol_candidate_medical_depth10_smoke8_mixed_compare_20260519_1`
- 状态：**启动失败**

关键报错：

- 当前 packaged `spu` Python runtime 的 `CheetahConfig` 不包含 `mixed_compare_mode`
- warmup 阶段即报：
  - `Failed to parse cheetah_2pc_config field`
  - `Message type "spu.CheetahConfig" has no field named "mixed_compare_mode"`

### 4.3 解释

这次失败不是算法错误，而是说明：

- 当前服务器真正执行时使用的是 `site-packages/spu/` 中打包好的 Python proto 与 native runtime；
- 该打包运行时已经支持 `disableMatmulPack`，但**不再暴露** `mixed_compare_mode`；
- 因而 mixed compare 现在不是一个“可直接复用的开关”，而是一个**需要重打包协议栈才能重新进入部署链**的历史分支。

### 4.4 本轮结论

- 对当前正式部署链：**不具备直接启用条件**
- 对当前主线价值：**不值得为它重新打开重编/重打包链**

## 5. 候选二：disable matmul pack

### 5.1 启用方式

- `SPU_CHEETAH_DISABLE_MATMUL_PACK=1`

### 5.2 实际结果

- 运行标识：`protocol_candidate_medical_depth10_smoke8_pack_off_20260519_1`
- 状态：**运行失败**

主进程报错：

- gRPC `StatusCode.UNAVAILABLE`
- `details = "Socket closed"`

### 5.3 节点侧证据

同步回本地的节点日志显示：

- `logs/protocol_candidate_packoff_node_logs_20260519/node_0.log`
  - `cheetah_dot.cc:475`
  - `1@1568x768x384 => 1x256x32 Recv 6.351 MiB, Response 2294.591 MiB Pack 0 ms (none)`
- `logs/protocol_candidate_packoff_node_logs_20260519/node_1.log`
  - 多次出现：
    - `SendImpl error`
    - `[E112]Not connected to 127.0.0.1:43269 yet`

### 5.4 解释

这说明在当前代表路径下：

- 关闭 matmul packing 后，CheetahDot 已经进入 `Pack 0 ms (none)` 路径；
- 但该路径在当前 `depth10 + batch8` 代表配置下没有形成稳定执行；
- 最终表现不是“慢一些但还能跑完”，而是直接把节点通信打崩。

### 5.5 本轮结论

- 对当前正式部署链：**不能采用**
- 原因不是精度问题，而是**稳定性直接不成立**

## 6. 候选三：positive-domain truncation scheduling

### 6.1 实验钩子

本轮已经在代码中加入实验钩子：

- `SPU_BB_TRUNC_SIGN_HINT=positive_when_unknown`

对应修改位置：

- `spu_vendored/libspu/mpc/cheetah/arithmetic.cc`

### 6.2 本轮阻塞

为了让该钩子真正进入服务器执行，需要重编 `libspu.so`。

本轮在服务器尝试过以下步骤：

- 把 `arithmetic.cc` 同步到 `/data/wyb/spu_src/src/libspu/mpc/cheetah/arithmetic.cc`
- 尝试恢复 Python `3.9` 编译条件
- 尝试使用 `bazel 6.5.0 + distdir + repository_cache + proxy` 重编

但当前 `/data/wyb/spu_src` 的构建链没有恢复到可用状态，先后卡在：

- `@rules_cuda` 未解析
- `@pybind11_bazel` 未解析
- `spulib~override` 对应的 local override 目录未形成有效 workspace/module

对应日志留在服务器：

- `/data/wyb/bazel_clean/logs/protocol_trunc_positive_build_20260519_*.log`

### 6.3 本轮结论

- 该候选**没有形成有效的端到端服务器结果**
- 因此它当前只能保留为：
  - 候选研究点
  - 未来协议优化方向
- 不能在本轮进入采用统计，更不能进入正式主线

## 7. 本轮服务器验证结论表

| 候选 | 启用方式 | 结果 | 结论 |
|---|---|---|---|
| baseline | 默认 packed path | `89.91s/sample`，隐私字段通过 | 作为对照基线保留 |
| mixed compare reactivation | `SPU_CHEETAH_MIXED_COMPARE_MODE=1` | warmup 前启动失败 | 不采用 |
| disable matmul pack | `SPU_CHEETAH_DISABLE_MATMUL_PACK=1` | 运行期节点失稳，gRPC socket closed | 不采用 |
| positive-domain truncation scheduling | `SPU_BB_TRUNC_SIGN_HINT=positive_when_unknown` | 依赖 `libspu.so` 重编；本轮 build blocked | 暂不采用 |

## 8. 补跑完成更新（当日追加）

上面第 4～6 节记录的是本轮**第一次 triage**的即时状态；在此之后，又补做了一轮“把没跑完的方向真正跑完”的收尾验证，结论如下。

### 8.1 mixed compare

- 本轮没有再重开；
- 原因不是继续阻塞，而是它已经有 `2026-05-12` 的完整历史服务器证据；
- 历史 `1` 样本低批量复核只有约 `1.033x` 轻微加速，因此直接移出当前候选。

### 8.2 disable matmul pack

为了把它从“batch8 下崩掉”推进到“至少完整跑通一次”，本轮改用 `SPU_BATCH_SIZE=1` 做同语义补跑：

| run | `sample_count` | `elapsed_sec` | `sec/sample` | 双向通信总量 |
|---|---:|---:|---:|---:|
| `protocol_candidate_medical_depth10_smoke8_baseline_20260519_smoke1_baseline_newlib` | `1` | `193.08` | `193.08` | `3.32 GB` |
| `protocol_candidate_medical_depth10_smoke8_pack_off_20260519_smoke1_packoff_newlib` | `1` | `158.02` | `158.02` | `10.00 GB` |
| `protocol_candidate_medical_depth10_smoke8_baseline_20260519_smoke2_baseline_newlib` | `2` | `357.32` | `178.66` | `6.56 GB` |
| `protocol_candidate_medical_depth10_smoke8_pack_off_20260519_smoke2_packoff_newlib` | `2` | `281.37` | `140.69` | `19.93 GB` |

补跑后可以确认：

- `pack-off` 已经不是“跑不完”；
- 在批次规模 `1` 下存在时间收益；
- 但通信量显著放大；
- 回到当前正式 `batch8` 代表路径时仍然失稳。

### 8.3 positive-domain truncation scheduling

本轮先修通实验用 `libspu.so` 重编链，再完成批次规模 `1` 的补跑：

| run | `sample_count` | `elapsed_sec` | `sec/sample` | 双向通信总量 |
|---|---:|---:|---:|---:|
| `protocol_candidate_medical_depth10_smoke8_baseline_20260519_smoke1_baseline_newlib` | `1` | `193.08` | `193.08` | `3.32 GB` |
| `protocol_candidate_medical_depth10_smoke8_trunc_positive_20260519_smoke1_truncpos_newlib` | `1` | `199.12` | `199.12` | `3.32 GB` |
| `protocol_candidate_medical_depth10_smoke8_baseline_20260519_smoke2_baseline_newlib` | `2` | `357.32` | `178.66` | `6.56 GB` |
| `protocol_candidate_medical_depth10_smoke8_trunc_positive_20260519_smoke2_truncpos_newlib` | `2` | `359.72` | `179.86` | `6.56 GB` |

补跑后可以确认：

- `trunc-positive` 已经完整跑通；
- 隐私字段保持通过；
- 但没有形成时间或通信收益。

### 8.4 补跑后的口径

补跑完成后，三个协议候选的最终状态统一为：

- `mixed_compare`：**已完整验证，收益不足，关闭**
- `trunc-positive`：**已完整验证，无收益，关闭**
- `pack-off`：**已完整验证；低批量下有时间收益，但当前正式 `batch8` 路径不稳定，因此不进入正式默认主线**
