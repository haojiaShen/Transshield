# SPU packing sum/difference 改写与完整报告回归

## 结论

本轮在隔离 SPU 运行时中改写 RLWE packing 蝶形合并：常见的非空分支不再先
复制整个密文，而是把和写回原密文、把差写入独立密文，再执行 Galois automorphism
并相加。改写前后计算的环表达式相同，不改变模型、权重、token 保留计划、FM64 /
fxp16、2PC 协议、通信消息或 reveal 策略。

完整报告门禁结果为 `report_update_ready=true`，没有失败项。但底层补丁相对同机
完整 baseline 只把 medical32 从 970.085 秒降到 964.208 秒，下降 0.61%；两批
16 张分别都略快，通信基本不变。该结果只能定性为边际改善，不能写成新的主要
性能突破。补丁保留为显式研究候选，不进入默认构建。

## 改写内容

原 common path 先执行 `tmp = E`，再分别计算 `E - O` 和 `tmp + O`。新路径执行：

```text
difference = E - O
E          = E + O
E          = E + Auto(difference)
```

零填充导致 odd 分支为空时仍保留原副本，因为该分支需要同时使用 `E` 与
`Auto(E)`。分支条件只由公开 packing 形状决定，不依赖秘密值。

补丁位于 `spu_vendored/patches/cheetah_packing_sumdiff.patch`。VPS 上从现有 hybrid
runtime 复制出 `/opt/transshield-spu-pack-sumdiff`，只替换新构建的 `libspu.so`；
原 `/opt/transshield-spu-mlp-pack-hybrid` 和 `/opt/transshield-spu` 均未覆盖。

## 完整报告测试范围

本次没有使用 1/4/16 张模型筛选结果替代报告数据。一次流水线完整重跑：

- 医疗全量验证 524 张；
- 固定医疗部署批次 32 张，同 VPS、同输入、同配置 baseline/candidate A/B；
- 固定金融边界压力 8 条；
- 13 类协议异常和 4 类 replay/inflight/rate-limit 守卫；
- 全部 50 项 Python 测试和 2 项 SPU 原生测试。

医疗使用正式 `base_rate=0.7`、depth 10、137/96/67 token、正式阈值
`0.6619606018066406`。金融使用 `base_rate=0.9`、depth 12、176/158/142 token。
安全执行保持 exact LayerNorm、uniform attention、fixed_square、secret 参数、
unpadded compact pruning、final logits only；医疗 A/B 均为 batch16。

## 完整结果

| 范围 | 结果 | 判定 |
|---|---:|---|
| 医疗 524 阈值精度 | 92.7480936% | 与报告冻结值一致 |
| 医疗 524 AUC | 0.9639150719 | 通过 |
| medical32 baseline | 970.085 s；42.574750 GiB | 同机完整基线 |
| medical32 candidate | 964.208 s；42.574552 GiB | 时间 -0.61%；通信近似不变 |
| medical32 阈值精度 / AUC | 93.75% / 0.96484375 | 通过 |
| medical32 argmax / 阈值 CPU 一致率 | 100% / 93.75% | 通过 |
| finance8 | 377.855 s；15.454028 GiB | 8/8，AUC 1.0 |
| finance8 argmax / 阈值 CPU 一致率 | 100% / 100% | 通过 |
| 协议与 guard | 13/13 + 4/4 | 17/17 FD/Socket 无净增长 |
| Python / SPU 原生测试 | 50/50 + 2/2 | 通过 |

medical32 两个 chunk 的 baseline 为 489.943 / 479.446 秒，candidate 为
488.116 / 475.396 秒；两批都略快，但总差值只有 5.878 秒，仍可能受到系统波动
影响。若要把 0.61% 单独写成稳定的底层收益，需要再做至少一次完整同机复跑；
本轮不把单次结果扩大解释。

## 与原报告数字的关系

最新完整 candidate 相对报告原始医疗记录 2849.945 秒、84.465983 GiB，表面上时间
下降 66.17%、loopback 计数下降 49.60%；金融相对 849.261 秒、25.302419 GiB，
时间下降 55.51%、通信下降 38.92%。其中包含 batch16、unpadded selection、公开固定
平方常数和 hybrid RLWE packing 等已验证优化，不是 sum/difference 补丁单独贡献。

报告原记录与 VPS 使用的硬件和通信计数来源不同，因此跨环境百分比只能作为新部署
观测；sum/difference 补丁本身只能引用本轮同 VPS baseline/candidate 的 0.61%。

## 隐私事实与限制

- 输入在 runner host 上未恢复为明文，runner host 也未读取两方私有 share 张量；
- 模型参数和 PredictorLG 参数以 secret 进入 SPU；
- 只 reveal 最终 logits；
- 当前 runner host 仍会加载 bundle 明文参数，再执行 secret placement；
- 输入 share 仍是 debug float additive share，不等于生产级独立 P1/P2 ingestion。

## 证据与回退

机器可读总入口：
`results/vps_report_tests/report_sumdiff_full_20260802_v1/report_regression_aggregate.json`。
同目录保存 inventory、逐样本 524/32/8 结果、网络快照、节点日志、构建日志、
13+4 明细、测试日志和源码哈希；大体积 `.pt` 与编译缓存只保留在 VPS。

回退时使用 `/opt/transshield-spu-mlp-pack-hybrid`，不应用
`cheetah_packing_sumdiff.patch`。模型 bundle、Python 图、正式结果目录和展示口径均
不需要回滚。本轮未修改 `results/final/`、`results/communication/`、
`results/fuzzing/` 或 `results/guard_stress/`。
