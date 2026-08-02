# SPU LayerNorm 与运行时后续优化

## 结论

本轮继续验证三类候选：LayerNorm 仿射参数折叠、SPU 调度开关、SPU 0.9.5
同图运行。所有模型测试均在 VPS `47.120.31.48` 执行，本地没有运行模型计算。

LayerNorm 仿射折叠在独立算子上有效：真实 FC1 形状的热运行时间减少 10.82%，
通信减少 50.16%。但 medical4 端到端测试中，折叠主干与预测器的 `all` 档比
baseline 慢 0.87%；只折叠主干的 `backbone` 档慢 0.93%。两档虽然分别减少
8.41% 和 7.14% 通信，且固定阈值下 4/4 预测一致，但没有达到“端到端时间必须
下降”的门槛，因此未继续运行 medical32。

调度器和运行时升级也没有产生可接入收益：关闭矩阵拆分慢 1.60%，开启 intra-op
并行慢 0.11%，inter-op 并行在 warm-up 阶段进入未实现的 `Object::fork` 路径；
SPU 0.9.5 在完全相同的 MLP 图上比当前 0.9.3b0 慢 1.75%。生产配置和正式展示
口径均保持不变。

## LayerNorm 仿射折叠

对标准 LayerNorm 输出 `normalized * gamma + beta`，后续线性层可按下式改写：

```text
linear(normalized * gamma + beta, W, b)
= linear(normalized, W * gamma, b + W @ beta)
```

该变换不公开学习参数，也不改变真实数算术下的模型函数。`gamma` 被吸收到秘密
线性权重的列中，`beta` 被吸收到秘密线性偏置中，安全图不再执行逐元素仿射乘加。
低秩分解线性层也有对应实现：缩放 down 矩阵的输入列，再把 `beta` 依次通过 down
和 up 矩阵传播到最终偏置。

FM64/fxp16 下运算顺序变化会改变截断位置，因此该路径是研究开关：

```text
--spu-layer-norm-affine-fusion none      # 默认与回退
--spu-layer-norm-affine-fusion backbone  # 主干 block 与最终 head
--spu-layer-norm-affine-fusion all       # backbone + PredictorLG
```

## VPS 结果

环境：16 vCPU、8 个物理核、约 61.4 GiB 内存、SPU 0.9.3b0、JAX 0.4.30、
CHEETAH + YACL Ferret、FM64/fxp16。端到端配置保持 medical32 当前批量档的模型、
`r=0.655`、10 层、秘密参数、精确 LayerNorm、uniform attention、完整秘密剪枝、
`unpadded_selection` 和 final-logits-only reveal；medical4 固定阈值为
`0.688152923150007`。

| medical4 指标 | baseline | `all` | `backbone` |
|---|---:|---:|---:|
| SPU 安全前向 | 195.035 s | 196.726 s | 196.841 s |
| 时间变化 | — | +0.87% | +0.93% |
| loopback 通信 | 6.591 GiB | 6.037 GiB | 6.120 GiB |
| 通信变化 | — | -8.41% | -7.14% |
| 阈值准确率 | 4/4 | 4/4 | 4/4 |
| 与 baseline 阈值预测一致率 | — | 100% | 100% |
| 最大概率绝对误差 | — | 0.03395 | 0.02697 |

独立 `LayerNorm -> 384×1536 Linear` 热态结果如下：

| 指标 | 原图 | 仿射折叠 | 变化 |
|---|---:|---:|---:|
| 耗时 | 21.740 s | 19.388 s | -10.82% |
| loopback 发送量 | 534,530,699 B | 266,402,439 B | -50.16% |
| 最大参考误差 | 0.003607 | 0.003652 | +0.000045 |

局部收益没有转化为端到端时间收益，说明当前整图耗时主要仍由矩阵、定点非线性和
秘密剪枝等部分共同决定，不能用单算子结果替代完整推理结论。

## 被否决的后置归一化

对于输出维度小于输入维度的线性层，还测试了先计算 centered-input 线性投影，再在
较小输出上乘逆标准差。它在真实数上等价，但 CHEETAH 的矩阵乘后定点截断使协议
代价反而增加：

- PredictorLG `384 -> 192`：热运行慢 7.18%，通信增加 180.14%。
- 分类头 `384 -> 2`：热运行慢 18.57%，通信增加 218.00%。

该改写只保留在微基准工具中，不接入生产模型图。

## 调度与版本筛选

调度测试以此前 `max_concurrency=8` 的真实 MLP 热态 `44.206 s` 为参照：

- `experimental_disable_mmul_split=true`：`44.915 s`，慢 1.60%。
- `experimental_enable_intra_op_par=true`：`44.253 s`，慢 0.11%。
- inter-op concurrency=2：SPU warm-up 失败，不进入模型测试。

SPU 0.9.5 使用与 0.9.3b0 相同的 MLP 图、CHEETAH、FM64/fxp16 和输入种子。热态
从 `47.653 s` 变为 `48.485 s`，慢 1.75%，通信仅减少 0.004%，因此不进行
medical4 或 medical32，也不升级默认 runtime。

## 验证、证据与回退

VPS 语法检查和单元测试日志位于
`results/vps_optimization/layernorm_affine_fusion_20260802_v1/test_logs/`。结构化结果：

- `results/vps_optimization/layernorm_affine_fusion_20260802_v1/optimization_summary.json`
- `results/vps_optimization/runtime_scheduler_20260802_v1/optimization_summary.json`
- `results/vps_optimization/spu095_unpadded_20260802_v1/optimization_summary.json`

正式 `results/final/`、`results/communication/`、`results/fuzzing/`、
`results/guard_stress/` 均未修改。LayerNorm 开关默认值为 `none`；不传新参数即可
回到进入本轮之前的安全图。
