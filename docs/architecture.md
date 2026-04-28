# 系统结构说明

## 1. 总体结构

当前 `Transshield_final` 将三条原本分散的链路收束到同一个比赛展示仓库中：

- `baseline` 明文对照链路；
- `modified` 明文主模型链路；
- `secure sidecar + replay + compare` 安全推理链路。

这样做的目的，是让评委可以在一个仓库里同时看到：

1. 原始对照模型效果；
2. 当前改造后主模型效果；
3. 安全推理是否真正与主模型闭环对接。

---

## 2. 三层主线

### 2.1 Baseline 明文对照层

这一层的角色是“原始参考系”。

- 参考运行代码：`references/original_plaintext_runtime/`
- 默认轻量评估权重：`artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth`
- 完整训练 checkpoint 归档：`artifacts/archive/baselines/baseline_plaintext_training_checkpoint_full.pth`

它的主要用途不是继续扩展，而是：

- 作为 `modified` 的对照组；
- 说明当前改造并不是“原模型照搬 secure”；
- 为答辩时回答“你们到底比原始模型改了什么、好在哪里”提供依据。

### 2.2 Modified 明文主模型层

这一层是当前比赛版主模型。

- 训练 / 评估主入口：`main.py`、`engine.py`、`infer.py`
- 模型定义：`models/`
- 默认轻量评估权重：`artifacts/frozen_bundle_full/modified_plaintext_eval_checkpoint_light.pth`
- secure replay 所需 pure `state_dict`：`artifacts/frozen_bundle_full/modified_plaintext_model_state_dict.pth`
- 完整训练 checkpoint 归档：`artifacts/archive/frozen_bundle_full/modified_plaintext_training_checkpoint_full.pth`

这一层既承担模型效果提升，也承担 secure 对接的明文源模型职责。

补充边界：

- 根仓 `main.py` / `models/` 是当前 final-repo 的 live 训练与评估实现；
- `training_source_tracka/` 是 TrackA source/provenance 控制路径；
- `training_compat/` 是当前服务器侧 plaintext compatibility runner；
- `scripts/run_tracka_train.sh source|compat` 走的是后两者，不是根仓 `main.py`。

### 2.3 Secure 推理闭环层

这一层承接的是最关键的 pruning 决策边界，而不是整网 Transformer。

核心组件包括：

- secure 输入导出；
- `network-kth` secure bridge；
- `tie-policy` secure bridge；
- checker；
- replay；
- compare；
- final comparison report。

这一层的主入口位于：`artifacts/server_inference_friendly_pack/`。

---

## 3. 为什么不是“整网 secure”

当前项目采取的是“关键 pruning 决策边界 secure sidecar 化”的路线，而不是把整张 ViT 一次性全部搬到 secure 环境。

原因是：

- DynamicViT 的直接 token 裁剪会带来动态 shape；
- 这类表达不利于安全计算后端稳定承接；
- 比赛阶段更重要的是把最关键、最有代表性的技术点闭环打通。

因此，本项目的结构重点是：

- 明文主模型保留大部分前向；
- 将 `masked_score -> kth_threshold -> tie payload` 这段关键边界安全外部化；
- 再通过 replay 把 secure 输出接回模型剩余前向。

---

## 4. 算法-密码学协同优化在结构中的体现

### 4.1 `masking` 对齐 `F_mux`

原始 DynamicViT 更接近“直接裁剪 token”的思路。当前版本将其改写为 `masking` 表达：

- token 不再直接从张量中删除；
- 而是通过掩码表达“保留 / 置零”；
- 这样更容易对齐安全多路复用 `F_mux` 的语义。

### 4.2 threshold compare 对齐 `F_less`

pruning 决策中最关键的一步，是分数与阈值的比较。当前结构将这部分边界显式化，使之可以被解释为：

- 安全比较 `F_less`；
- 进而生成布尔掩码；
- 再通过 `F_mux` 将冗余 token 置零。

换句话说，本项目不是先有 secure 框架再硬套模型，而是先把模型表达改造成更适合密码学执行语义的形式。

---

## 5. 关键代码与目录对应关系

### 5.1 训练与模型定义

- `main.py`
- `engine.py`
- `models/`
- `datasets.py`
- `infer.py`

### 5.2 secure sidecar 导出与检查

- `tools/transshield_secure_sidecar_export_suite.py`
- `tools/transshield_secure_network_kth.py`
- `tools/transshield_secure_tie_payload.py`

### 5.3 secure 执行与 replay

- `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
- `integrations/openbumblebee/transshield_tie_policy_bridge/transshield_tie_policy_bridge.py`
- `tools/transshield_openbumblebee_pipeline.py`
- `tools/transshield_openbumblebee_inference_replay.py`
- `tools/transshield_spu_runtime_setup.py`

当前 secure 代码导航以本文和 `tools/README.md` 为准，不再额外保留单独的 `secure_infer/` 导航目录。

### 5.4 最终展示主入口

- `artifacts/server_inference_friendly_pack/run_full_final_comparison_smoke.sh`
- `artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`

---

## 6. CPU 与 SPU 在结构中的位置

### `SECURE_RUNTIME=cpu`

- 是 secure sidecar 的本地明文参考执行；
- 用于开发调试、链路验证与快速 smoke 演示；
- **不是项目最终的安全执行路径**；
- 不是真正的 2PC。

### `SECURE_RUNTIME=spu`

- 是相同 secure sidecar 逻辑在 `SPU / OpenBumbleBee` 上的真实安全执行；
- **这是当前项目真正的 secure 运行路径**；
- 这里才涉及 secret sharing、协议执行、通信与额外开销。

因此，CPU 与 SPU 的差别不在“算的函数是否相同”，而在“后端执行机制是否真正安全”。
更准确地说：`CPU secure` 不是我们的最终安全路径，它只是 sidecar 函数的本地参考实现，用来验证 `SPU` 输出是否符合预期；真正的安全执行路径是 `SPU secure`。

---

## 7. 当前结构的比赛价值

当前仓库结构已经能够清楚支撑答辩中的三条主线：

1. `baseline` 与 `modified` 谁更好；
2. `modified` 是否能与 secure 推理闭环对接；
3. `secure` 是否与 `modified` 在逐样本预测语义上一致。

这也是当前 `Transshield_final` 作为比赛展示版最重要的结构价值。
