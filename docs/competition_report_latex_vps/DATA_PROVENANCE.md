# 当前 VPS 数据来源与口径

本文件只记录数据来源，不改变算法定义。报告正文中的替换均在原 PDF 坐标内完成，未重新排版表格或段落。

## 1. 当前 VPS 环境

环境于 2026-08-02 复核：

| 项目 | 记录 |
|---|---|
| 云平台/虚拟化 | 阿里云，KVM 全虚拟化 |
| CPU | Intel Xeon Platinum，16 vCPU，1 socket，8 cores/socket，2 threads/core |
| 内存 | 61 GiB，Swap 0 |
| 操作系统 | Ubuntu 24.04.4 LTS |
| 内核 | Linux 6.8.0-136-generic |
| Python | 3.9.25 |
| SPU | 0.9.3b0 |
| JAX | 0.4.30 |
| NumPy | 1.26.4 |
| PyTorch | 1.13.1+cpu |
| GPU | 本轮未配置、未使用 |

## 2. 当前 VPS 医疗完整运行

本轮使用固定 32 条医疗部署验证样本，运行参数为：完整 SPU 运行、两方 colocated localhost、batch size 16、depth 10、secret params、exact LayerNorm、uniform attention、fixed_square 激活、仅揭示最终 logits。

| 指标 | 结果 |
|---|---:|
| 样本数 | 32 |
| 总时长 | 913.363945 秒 |
| 平均时延 | 28.542623 秒/样本 |
| 32 条样本阈值精度 | 93.75% |
| 32 条样本 AUC | 0.984375 |
| 环回接口 TX 增量单计 | 40.485295 GiB |
| 每样本通信量 | 1.265165 GiB |
| 输出有限性 | 通过 |

对应证据为 `artifacts/vps_release_backup_20260803/results/vps_optimization/rlwe_equivalence_followup_20260802_v1/hybrid32_summary.json`。该运行保持模型权重、模型维度、剪枝结构、协议、定点精度和最终输出揭示策略不变；本地展示代码未随报告数据更新。

## 3. 当前 VPS 金融完整运行

金融边界压力验证使用固定 8 条样本、两方 colocated localhost、batch size 8，并采用与医疗结果相同的环回接口 TX 增量单计口径。

| 指标 | 结果 |
|---|---:|
| 样本数 | 8 |
| 总时长 | 377.854918 秒 |
| 平均时延 | 47.231865 秒/样本 |
| 逐样本一致性 | 8/8 |
| 环回接口 TX 增量单计 | 15.454028 GiB |
| 每样本通信量 | 1.931753 GiB |
| 输出有限性 | 通过 |

对应证据为 `artifacts/vps_release_backup_20260803/results/vps_report_tests/report_sumdiff_full_20260802_v1/finance8_spu_latest_summary.json`。

## 4. 计数口径

医疗与金融通信量均来自 Linux 环回接口 TX 增量单计。环回接口的 TX 与 RX 会镜像同一批传输字节，因此只计一次 TX。报告正文直接呈现实测绝对值，不展开底层优化过程，也不修改算法、公式和模型结构叙述。
