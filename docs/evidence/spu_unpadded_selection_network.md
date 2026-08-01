# SPU 任意长度 Top-K 选择网络优化

## 结论

正式医疗安全推理链路新增 `unpadded_selection`，并将其设为 runner 与展示 API 的默认安全剪枝网络。旧实现仍可通过 `--spu-secure-pruning-network full_sort` 一键回退。

它不改变模型权重、depth、137/96/67 token 保留计划、FM64/fxp16 数值口径、精确 LayerNorm、输入分片边界或最终 logits-only reveal。`results/final/`、`results/communication/`、`results/fuzzing/` 和 `results/guard_stress/` 均未改写。

## 算法变化

旧图把 196 和 137 个 token 都补齐到 256，再执行完整 Bitonic 排序；最后 96 个 token 也补齐到 128。新图做两件事：

1. 用任意长度 Bitonic merge 递归直接构造 196、137、96 条 wire 的固定比较网络，不再生成虚拟 token。
2. 从所需输出 wire 反向遍历网络，只保留会影响前 K 个 token 或第 K 个阈值的 compare-and-swap。

控制流和索引仍完全公开、固定，不根据秘密 score 分支。网络变换是精确的，不属于近似 Top-K。

| 阶段 | 公开输入/保留数 | 旧比较器 | 新比较器 |
|---|---:|---:|---:|
| block 3 compact | 196 → 137 | 4,608 | 2,921 |
| block 6 compact | 137 → 96 | 4,608 | 1,799 |
| block 9 threshold only | 96 → 67 | 1,792 | 831 |
| 合计 | — | 11,008 | 5,551 |

局部比较器减少 49.57%。这不等同于端到端减少 49.57%，因为 Transformer 线性层、LayerNorm 和其他 SPU 算子没有改变。

## VPS 验证

环境：16 vCPU Intel Xeon Platinum、约 61.4 GiB、Python 3.9.25、SPU 0.9.3b0、JAX/JAXLIB 0.4.30、CHEETAH + YACL Ferret、FM64/fxp16。模型测试全部在 VPS 执行。

- 单样本热缓存：91.623 → 86.641 秒，减少 5.44%；通信减少 2.50%。
- batch=4 热缓存：253.002 → 235.663 秒，减少 6.85%；通信减少 3.01%。
- medical32、batch=8：1620.888 → 1463.214 秒，减少 9.73%；通信 58.786 → 56.947 GB，减少 3.13%。
- medical32 正式阈值准确率：93.75% → 96.875%；AUC：0.9805 → 0.9844；正式阈值预测与基线 31/32 一致。
- 37/37 VPS 单元测试通过，包括 2–10 wires 的所有 0/1 输入穷举、196/137/96 实际尺寸随机对照，以及 runner/showcase 默认接线检查。

精度小幅上升不能解释为模型质量提升。排序网络在数学上等价，跨运行概率差异来自 SPU 概率定点截断；本次结论只应表述为“未观察到精度退化”。

结构化数据与逐次原始 candidate JSON 位于 `results/vps_optimization/selection_network_20260801_v1/`。

## 被否决的更底层路线

FM32/fxp12 能直接缩短 share 和比较协议位宽，但 stock SPU 0.9.3b0 的精确 LayerNorm `rsqrt` 会尝试编码 FM32 无法表示的 I64 常量，完整图因此失败。没有替换并验证 `rsqrt` 或重编译 SPU 之前，不应启用 FM32。

进一步的高潜力方向是为裁剪 score 增加有界位宽的专用秘密比较：当前 Cheetah 比较默认按完整 ring 位宽处理，而 packed pruning key 的已知范围远小于 64 位。该方向需要定制 SPU intrinsic/runtime 并重新做范围证明、溢出 guard 和完整回归，暂未合入默认路径。

## 上游依据

- SecretFlow 的 [SPU 架构说明](https://secretflow.readthedocs.io/en/stable/developer/design/spu.html) 明确区分 JAX/XLA/MLIR 编译层与底层 MPC runtime，因此本次先在公开计算图层减少比较器，不需要改变密码协议。
- SPU 当前 [Cheetah `MsbA2B`](https://github.com/secretflow/spu/blob/main/src/libspu/mpc/cheetah/arithmetic.cc) 已有内部 `nbits` 概念；[`CompareProtocol`](https://github.com/secretflow/spu/blob/main/src/libspu/mpc/cheetah/nonlinear/compare_prot.cc) 的 digit 数按 `ceil(bitwidth / compare_radix)` 计算。这是后续有界位宽比较可能降低 OT 工作量的源码依据。
- SPU 当前 [`CompilerOptions`](https://github.com/secretflow/spu/blob/main/src/libspu/spu.proto) 暴露 partial-sort 优化开关。未来可审计 JAX 原生 Top-K 的 lowering，但必须先验证 payload 搬运和 tie 规则，不能仅凭存在该开关就替换现路径。
- 选择网络只计算所需 Top-K 输出、减少比较器的思路也可见 [Petersen 等人的 selection-network 讨论](https://proceedings.mlr.press/v162/petersen22a/petersen22a.pdf)。本仓实现是硬比较、固定拓扑的安全推理网络，不使用该论文的可微近似。
