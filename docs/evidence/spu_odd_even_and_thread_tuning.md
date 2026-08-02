# SPU 精确剪枝网络与线程并发实验

## 结论

本轮验证了两个不改变模型训练参数的底层候选：减少比较器的 odd-even 精确选择网络，以及 SPU `max_concurrency` 线程并发调优。两者均未通过当前严格验收门槛，因此没有替换正式默认路径，也没有改写 `results/final/`、`results/communication/` 或展示口径。

- `odd_even_selection` 的比较器总数从 5287 降至 4822，下降 8.80%，但 medical4 的 SPU forward 时间增加 2.08%，通信增加 0.54%。该候选在小样本筛选阶段即被否决，没有继续运行 medical32。
- `max_concurrency=8` 在 MLP 热缓存微基准中缩短 7.98%，medical32 完整图缩短 3.01%；但通信增加 0.06%，阈值准确率由 0.9375 降至 0.90625，AUC 由 0.98046875 降至 0.96484375，因此不接入默认配置。

## 实验边界

两组模型测试均在 VPS `47.120.31.48` 上执行。完整图沿用 medical32 的固定样本、`r=0.655` 配套阈值、batch16、10 层静态深度、CHEETAH、FM64/fxp16、秘密模型参数和仅揭示最终 logits 的设置。正式可接受基线仍是 `results/vps_optimization/batch_scaling_20260802_v1/optimization_summary.json`。

验收门槛为：耗时必须下降、通信不能回退、medical32 阈值准确率不得低于 0.9375、AUC 不得低于 0.98046875。首轮完整测试未通过时不重复消耗一轮完整推理。

## odd-even 精确选择网络

`odd_even_selection` 使用 Batcher odd-even merge sort 的公开比较器拓扑，并做两层静态化处理：

1. 对补齐到 2 的幂所产生的公开 `-inf` 哨兵线消除秘密比较，将其转换为公开路由。
2. 从实际需要的 Top-K 输出反向切片，只保留影响这些输出的比较器。

该路径仍执行精确 Top-K，并维持“分数优先、原始 token 下标用于边界同分裁决”的既有语义。VPS 上的 28 项相关测试全部通过，包括选择值、keep mask、token payload 和原始下标与现有精确路径的一致性。

| 指标 | `unpadded_selection` 基线 | `odd_even_selection` | 变化 |
|---|---:|---:|---:|
| 三阶段比较器总数 | 5287 | 4822 | -8.80% |
| medical4 SPU forward | 195.035 秒 | 199.088 秒 | +2.08% |
| medical4 通信 | 6.591 GiB | 6.627 GiB | +0.54% |
| medical4 阈值准确率 | 1.0 | 1.0 | 0 |

比较器减少却变慢，说明在当前 SPU/CHEETAH 实现中，公开路由层数、张量重排和交互轮次的开销超过了 465 个比较器的节省。这个结果不能外推为所有后端都更慢，所以代码只保留显式研究入口；默认值仍为 `unpadded_selection`。

原始结果和判定见 `results/vps_optimization/odd_even_selection_20260802_v1/optimization_summary.json`。

## SPU 线程并发

VPS 有 16 个逻辑 CPU，对应 8 个物理核和 SMT。默认节点日志显示每个参与方创建 15 个 worker；两个同机参与方会竞争同一组物理核。本轮测试 `max_concurrency=7/8/9/10`，其中 `8` 在固定 MLP 形状的热缓存微基准中最快。

| `max_concurrency` | MLP 热缓存均值 | 相对默认基线 |
|---:|---:|---:|
| 默认 | 48.037 秒 | — |
| 7 | 50.603 秒 | +5.34% |
| 8 | 44.206 秒 | -7.98% |
| 9 | 54.769 秒 | +14.01% |
| 10 | 55.099 秒 | +14.70% |

进一步测试 `max_concurrency=8`：

| 指标 | 已接受基线 | `max_concurrency=8` | 变化/判定 |
|---|---:|---:|---:|
| medical32 SPU forward | 1060.968 秒 | 1028.993 秒 | -3.01%，通过 |
| medical32 通信 | 49.504 GiB | 49.535 GiB | +0.06%，未通过 |
| 阈值准确率 | 0.9375 | 0.90625 | -3.125 个百分点，未通过 |
| AUC | 0.98046875 | 0.96484375 | -0.015625，未通过 |

线程参数没有改变数学图，但该次秘密定点完整运行仍出现概率最大绝对偏差 0.02837。现有证据只能确认发生了运行间定点漂移，不能证明精度差异完全由线程数本身造成。由于正式门槛要求完整结果同时通过，不以局部耗时收益覆盖精度证据，也不进行事后挑选性复跑。

原始结果和判定见 `results/vps_optimization/thread_tuning_20260802_v1/optimization_summary.json`。

## 当前使用规则

- 正式默认继续使用 `unpadded_selection` 和现有 SPU runtime 配置。
- `odd_even_selection` 仅用于研究复验，必须显式指定，不能写入正式展示启动配置。
- `max_concurrency=8` 的配置文件只保存在实验结果目录，不复制为生产配置。
- 后续若更换 SPU 版本、部署拓扑或 CPU 架构，应重新从微基准开始，并再次通过同一套 medical32 严格门槛。
