# Transshield 统一 secure transformer benchmark

## 口径说明

- Benchmark harness：`MPCFormer local 2PC configurable transformer benchmark`
- 是否 full-val 医学图像 pipeline：`false`
- 是否网页单图 live run：`false`
- 说明：这些结果只表示同一 secure transformer benchmark harness 下的通信 / 时间 profile，不能和当前网页单图 live run 或 full-val Transshield SPU sidecar 通信量混为一谈。

## Profile 结果

| Profile | 角色 | 结构 / 算子 | 平均总时间 | 平均模块通信 | 来源 |
|---|---|---|---:|---:|---|
| External baseline ops same-shape proxy | external_baseline | L=12, H=384, I=1536, seq=197, heads=6, act=relu, softmax=softmax | 15.0993s | 5.78 GiB | `/data/wyb/Transshield_final/results/standardized_secure_benchmark/standardized_secure_ops_20260417_181805/baseline_ops_same_shape_proxy/summary.json` |
| Transshield secure-friendly ops same-shape proxy | current_project | L=12, H=384, I=1536, seq=197, heads=6, act=quad, softmax=softmax_2QUAD | 8.2935s | 881.05 MiB | `/data/wyb/Transshield_final/results/standardized_secure_benchmark/standardized_secure_ops_20260417_181805/transshield_ops_same_shape_proxy/summary.json` |

## 对比结果

| 分组 | 左侧 | 右侧 | 通信差值 | 通信比例 | 时间差值 | 说明 |
|---|---|---|---:|---:|---:|---|
| same_shape_operator_proxy | Transshield secure-friendly ops same-shape proxy | External baseline ops same-shape proxy | -4.92 GiB | 0.1489x | -6.8059s | 同一模型形状、同一 2PC benchmark harness；主要用于观察算子配置导致的安全开销差异。 |

## 使用限制

- 可以用于说明外部模型 / 算子在同一 secure benchmark harness 下的开销差异。
- 不能写成 Transshield full-val SPU sidecar 与外部模型 full pipeline 的严格通信量对比。
- 如果要做 full-val 通信量公平对比，外部模型也必须接入同输入、同样本量、同协议路径后重新统计。
