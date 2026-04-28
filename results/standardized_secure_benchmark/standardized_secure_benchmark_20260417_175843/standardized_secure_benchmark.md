# Transshield 统一 secure transformer benchmark

## 口径说明

- Benchmark harness：`MPCFormer local 2PC configurable transformer benchmark`
- 是否 full-val 医学图像 pipeline：`false`
- 是否网页单图 live run：`false`
- 说明：这些结果只表示同一 secure transformer benchmark harness 下的通信 / 时间 profile，不能和当前网页单图 live run 或 full-val Transshield SPU sidecar 通信量混为一谈。

## Profile 结果

| Profile | 角色 | 结构 / 算子 | 平均总时间 | 平均模块通信 | 来源 |
|---|---|---|---:|---:|---|
| MPCViT vit_7_4_32 proxy | external_baseline | L=7, H=256, I=512, seq=65, heads=4, act=relu, softmax=softmax_2RELU | 4.6830s | 262.56 MiB | `/data/wyb/Transshield_final/results/standardized_secure_benchmark/standardized_secure_benchmark_20260417_175843/mpcvit_vit_7_4_32_arch_proxy/summary.json` |
| Transshield 当前最终模型 proxy | current_project | L=12, H=384, I=1536, seq=197, heads=6, act=quad, softmax=softmax | 14.6669s | 4.32 GiB | `/data/wyb/Transshield_final/results/standardized_secure_benchmark/standardized_secure_benchmark_20260417_175843/transshield_final_arch_proxy/summary.json` |

## 对比结果

| 分组 | 左侧 | 右侧 | 通信差值 | 通信比例 | 时间差值 | 说明 |
|---|---|---|---:|---:|---:|---|
| architecture_proxy | Transshield 当前最终模型 proxy | MPCViT vit_7_4_32 proxy | 4.06 GiB | 16.8447x | 9.9838s | 同一 MPCFormer local 2PC benchmark harness；各 profile 使用各自模型结构参数。这不是 full-val 医学图像 pipeline 通信量。 |

## 使用限制

- 可以用于说明外部模型 / 算子在同一 secure benchmark harness 下的开销差异。
- 不能写成 Transshield full-val SPU sidecar 与外部模型 full pipeline 的严格通信量对比。
- 如果要做 full-val 通信量公平对比，外部模型也必须接入同输入、同样本量、同协议路径后重新统计。
