# 当前 VPS 数据来源与口径

本文件只记录数据来源，不改变算法定义。报告正文中的替换均在原 PDF 坐标内完成，未重新排版表格或段落。报告生成与审计直接读取已纳入版本控制的 `vps_report_data.json`；该文件只摘录报告实际使用的环境、运行配置与结果字段，并记录原始 VPS JSON 的路径和 SHA-256，避免构建依赖本机未跟踪备份目录。

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
| 总时长 | 964.207541 秒 |
| 平均时延 | 30.131486 秒/样本 |
| 32 条样本阈值精度 | 93.75% |
| 32 条样本 AUC | 0.964844 |
| 环回接口 TX 增量单计 | 42.574552 GiB |
| 每样本通信量 | 1.330455 GiB |
| 输出有限性 | 通过 |

对应原始证据为 `artifacts/vps_release_backup_20260803/results/vps_report_tests/report_sumdiff_full_20260802_v1/medical32_spu_latest_summary.json`，SHA-256 为 `9330e2dbe3564c3a5118a2bfd41d240421fdbf3f6fee300b8fb1a1e589f9ac06`。运行时实际使用 `base_rate=0.7`，三阶段保留率为 `0.7/0.49/0.343`，空间 token 保留数为 `137/96/67`，与报告算法口径一致；实际 SPU batch size 为 16。本地展示代码未随报告数据更新。

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

对应原始证据为 `artifacts/vps_release_backup_20260803/results/vps_report_tests/report_sumdiff_full_20260802_v1/finance8_spu_latest_summary.json`，SHA-256 为 `babc00f6d6897402b2dea8e81c0ef8df423a8f7c7d7a9ca7a55069f118b28a6c`。

## 4. 计数口径

医疗与金融通信量均来自 Linux 环回接口 TX 增量单计。环回接口的 TX 与 RX 会镜像同一批传输字节，因此只计一次 TX。报告正文直接呈现实测绝对值，不展开底层优化过程，也不修改算法、公式和模型结构叙述。
