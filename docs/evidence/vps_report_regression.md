# 正式报告口径的 VPS 回归测试

本文只定义“最终报告中的数字怎样在 VPS 上重新核对”。所有计算、预处理、
代码测试、精度测试、SPU 测试和通信量采样都必须在 VPS 执行；本地只负责代码
编辑、只读审阅和结果归档。

详细机器可读口径见 `configs/report_vps_test_matrix.json`。新结果统一写入
`results/vps_report_tests/<run-name>/`，不得覆盖 `results/final/`、
`results/communication/`、`results/fuzzing/` 或 `results/guard_stress/`。

## 1. 报告中的四个正式测试范围

| 范围 | 固定输入 | 必须记录的结果 |
|---|---|---|
| 医疗全量验证 | PneumoniaMNIST 验证集 524 张，0 类 135 张、1 类 389 张 | 每样本概率与预测、argmax 精度、最佳阈值、阈值精度、AUC |
| 医疗安全部署批次 | 固定 32 张，两类各 16 张、每类排序后等距抽样 | 每样本输出、CPU/SPU 一致性、总时延、单样本时延、双向通信量、运行时隐私事实 |
| 金融边界压力 | 固定 8 条，fraud/normal 各 4 条，编号 000000/000033/000066/000099 | 每样本一致性、总时延、单样本时延、双向通信量、参数保留比例 |
| 协议与控制面 | 13 类协议异常 + 4 类重放/并发/限流守卫 | 状态码、错误码、首个拦截层、兜底层、FD/Socket/RSS/线程变化、inflight 回落 |

医疗 32 张和金融 8 条不能临时换样本。精确清单仍使用仓内历史原始证据：

- `archive/old_runs/artifacts/server_pipeline_run/medical_dynamic_prepare_final/e2e_secure_poc/smoke32_balanced_evenly_spaced.txt`
- `archive/old_runs/artifacts/server_pipeline_run/finance_dynamic_static_prepare_final/e2e_secure_poc/finance_smoke8_balanced_evenly_spaced.txt`

医疗 524 张使用：

- `results/final/medical_threshold_calibration_raw/fullval_524_image_list.txt`

VPS inventory 会将清单中的历史绝对路径映射到当前 VPS 数据根目录，并为每个
文件记录相对路径、类别、字节数、SHA-256、图片尺寸和颜色模式。

## 2. 固定预处理和安全执行配置

- 图像：RGB，bicubic resize 到短边 256，再中心裁剪为 224×224。
- 张量：Float32、CHW、除以 255，按 ImageNet mean/std 归一化。
- 浏览器 live 路径：归一化值再裁剪到 `[-2, 2]`，以 little-endian Float32 生成两份加法 share。
- 医疗：DeiT-S/DynamicViT，depth 10，base rate 0.7，剪枝位置 3/6/9。
- 金融：DeiT-S/DynamicViT，depth 12，base rate 0.9，低秩参数保留比例 68.39%。
- 安全函数：exact LayerNorm、uniform attention、fixed_square。
- 运行边界：colocated localhost 2PC，参数在 SPU 中按 secret 放置，动态预测器、排序和 keep-mask 在 SPU 图内执行，只 reveal 最终 logits。

隐私记录必须拆成不同事实，不能再压成一个含混布尔值：

1. runner host 是否读取过 bundle 明文参数；
2. 参数和 PredictorLG 是否以 secret 进入 SPU；
3. runner host 是否恢复过明文输入或读取过两方私有 share；
4. 对端是否得到模型参数；
5. SPU 最终 reveal 的对象是否只有 logits。

当前 Python runner 会在模型提供方/协调 runner host 上读取 state dict 后再 secret
注入 SPU，因此新证据会如实记录 `runner_host_model_params_materialized=true`。
这不等于参数向另一参与方公开，但也不能写成 runner host 从未接触明文参数。

## 3. VPS 数据与资产预检

下面命令全部在 VPS 执行：

```bash
export REPO_ROOT=/opt/transshield-project
export PYTHON_BIN=/opt/transshield-spu/bin/python
export RUN_NAME=report_regression_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT="$REPO_ROOT/results/vps_report_tests/$RUN_NAME"
mkdir -p "$RUN_ROOT/lists"
cd "$REPO_ROOT"

"$PYTHON_BIN" tools/report_vps_test.py inventory \
  --medical-data-root "$REPO_ROOT/data/pneumoniamnist_imagefolder_subset/val" \
  --finance-data-root "$REPO_ROOT/data/finance_boundary_stress_imagefolder/val" \
  --runtime-config /opt/transshield-smoke/configs/transshield_runtime/2pc.runtime.json \
  --materialized-list-dir "$RUN_ROOT/lists" \
  --out "$RUN_ROOT/inventory.json"
```

只有 `inventory.json.passed=true` 才能继续长测。它会同时核对 524/32/8 数量、
类别数、缺失与重复、清单哈希、逐文件哈希、bundle 权重哈希、VPS 硬件和软件版本。

## 4. VPS 重新生成正式输入

医疗 32 张：

```bash
"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-preprocess \
  --bundle-dir artifacts/frozen_bundle_medical_dynamic_mainline \
  --image-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --include-source-paths --include-targets \
  --output-pt "$RUN_ROOT/medical32_plain.pt" \
  --output-json "$RUN_ROOT/medical32_plain.json"

"$PYTHON_BIN" tools/report_vps_test.py compare-preprocessed \
  --current-pt "$RUN_ROOT/medical32_plain.pt" \
  --frozen-pt archive/old_runs/artifacts/server_pipeline_run/medical_dynamic_prepare_final/e2e_secure_poc/plaintext_same_images_pixel_values.pt \
  --out "$RUN_ROOT/medical32_preprocess_compare.json"
```

金融 8 条同理，输入清单换为
`$RUN_ROOT/lists/finance_boundary_stress.txt`，bundle 换为金融 bundle。预处理比较
必须达到张量逐元素完全一致；否则不得把新运行和报告数字直接比较。

## 5. 精度、SPU 和通信量结果要求

医疗全量精度必须使用 524 张完整输入；32 张只用于端到端性能与一致性，不能替代
full-val 精度。SPU 长测前后分别执行：

```bash
"$PYTHON_BIN" tools/report_vps_test.py network-snapshot \
  --interface lo --out "$RUN_ROOT/medical32_network_before.json"

# 在这里执行最新代码的 medical32 SPU runner。

"$PYTHON_BIN" tools/report_vps_test.py network-snapshot \
  --interface lo --out "$RUN_ROOT/medical32_network_after.json"
```

随后用 `summarize` 将 candidate、同 VPS CPU reference、固定样本清单和前后网络
计数合并。输出必须包含每一条样本的路径、target、logits、class-1 概率、argmax
预测、阈值预测、正确性、与 CPU reference 的误差，以及整批精度、AUC、预测计数、
时延和通信量。

报告原机器为 80 逻辑 CPU，当前 VPS 为另一硬件环境，因此跨机器时延只能作为
迁移结果，不能单独证明优化。优化结论必须引用同一 VPS 上、同一清单、同一配置
的 baseline/candidate A/B；此前 4 张同 VPS A/B 可作为快速趋势证据，32 张结果
用于报告规模的最终验收。

## 6. 需要与报告分开的历史实验

报告表 4-2、4-3、4-10 属于训练/消融的冻结结果，表 4-6、4-7 和图 4-5、4-6
属于旧 benchmark harness 的算子/网络/底层原语证据。这些项目已经完整写入矩阵，
但在匹配的训练 checkpoint 或 legacy benchmark harness 未恢复到 VPS 前，不得用
新写的小脚本伪造“同口径复跑”。当前优先验收的是可运行最终代码对应的 524、
32、8 和 17 类四个范围。

## 7. 2026-08-01 VPS 实测结果

本轮 candidate 证据位于
`results/vps_report_tests/report_regression_20260801_v1/`，总入口为
`report_regression_aggregate.json`。该目录保留逐文件 inventory、逐样本 logits / 概率 /
预测、CPU 参考误差、网络快照、13+4 项控制面明细、运行日志与文件哈希；大体积中间
`.pt` 只留在 VPS，不进入仓库。

| 项目 | VPS candidate | 报告冻结值 | 判定 |
|---|---:|---:|---|
| 医疗全量 524 阈值精度 | 92.7481% | 92.7481% | 完全复现 |
| 医疗全量 524 AUC | 0.96391507 | 0.96391505 | 浮点统计尾差 |
| 医疗 32 阈值精度 | 90.625% | 93.750% | 少 1 个正确样本，满足主任务最低门槛 |
| 医疗 32 argmax / 阈值 CPU 一致率 | 100% / 84.375% | 未冻结该对照 | 阈值一致率低于预设 87.5% 诊断门槛，明确保留告警 |
| 医疗 32 时延 | 1589.62 秒；49.68 秒/样本 | 2849.94 秒；89.06 秒/样本 | 跨硬件观察，不作为优化因果证据 |
| 医疗 32 loopback 单向计数 | 58.259 GiB；1.821 GiB/样本 | SPU LinkDetails 84.466 GiB；2.640 GiB/样本 | 计数方法不同，只作迁移观察 |
| 金融 8 精度 / AUC | 100% / 1.0 | 8/8 语义一致 | 通过 |
| 金融 8 CPU 一致率 | argmax 100%；阈值 100% | 100% | 通过 |
| 金融 8 时延 | 544.53 秒；68.07 秒/样本 | 849.26 秒；106.16 秒/样本 | 跨硬件观察，不作为优化因果证据 |
| 13+4 黑盒验证 | 17/17 | 17/17 | 通过；FD/Socket 17/17 无净增长 |
| VPS 单元测试 | 24/24 | — | 通过 |

可用于“优化有效”结论的是同一 VPS A/B：单样本组合优化将 119.65 秒降到
93.48 秒（下降 21.88%），loopback TX 下降 7.18%，argmax 不变；4 样本 value
fusion 消融将 292.29 秒降到 251.32 秒（下降 14.02%），loopback TX 下降 3.51%，
argmax 4/4 不变。完整原始数值、SHA-256 和概率最大误差均在 aggregate 中。

截断请求体有一处环境差异：VPS 服务端审计确实记录
`streaming_body_reader/truncated_body`，但主动半关闭 TCP 的客户端没有收到 HTTP
响应；报告旧环境记录为 HTTP 400。新结果按“服务端命中审计 + TCP 无响应关闭”
如实保存，没有改写成 400。

生成聚合证据的命令同样只在 VPS 执行：

```bash
"$PYTHON_BIN" tools/report_vps_aggregate.py \
  --run-root "$RUN_ROOT" \
  --out "$RUN_ROOT/report_regression_aggregate.json"
```

## 8. 2026-08-01 bitonic 成对比较优化

在不修改模型、token 保留数、隐私边界和 reveal 策略的前提下，compact top-k 的
bitonic 网络由“每个 pair 的左右位置各做一次秘密比较”改为“每个 pair 只比较一次，
同时产生两侧结果，再按公开排列恢复 token 顺序”。两次 256 位 payload sort 加一次
128 位阈值 sort 的计划比较数由每样本 22016 降到 11008；这是排序网络层面的精确
计数，不把共享的阈值后处理算进降幅。

同 VPS、同模型、同固定输入和同 FM64/Cheetah 配置的结果如下：

| 规模 | 旧版时延 | 新版时延 | 旧版 loopback TX | 新版 loopback TX | 判定 |
|---|---:|---:|---:|---:|---|
| 单图 | 93.48 秒 | 91.85 秒 | 2.405 GB | 2.288 GB | 通信 -4.83%；时延属于运行波动 |
| 4 张，batch=4 | 251.32 秒 | 257.78 秒 | 8.134 GB | 7.669 GB | 通信 -5.72%；时延 +2.57% |
| 正式医疗 32 张，batch=8 | 1589.62 秒 | 1620.95 秒 | 62.556 GB | 58.786 GB | 通信 -6.03%；时延 +1.97% |

32 张使用报告正式阈值 `0.6619606018066406`：阈值精度 93.75%，AUC
0.98046875，argmax 对 CPU 一致率 100%，阈值对 CPU 一致率 93.75%，概率最大
绝对误差 0.06705。模型数学图等价，但 Cheetah 固定点截断带随机性，因此精度变化
只作为本次观测保存，不宣称该排序重写能确定性提升精度。

新增的 pair schedule 与含 ties 排序测试连同原回归在 VPS 上为 26/26 通过。机器
可读总表见
`results/vps_optimization/pairwise_bitonic_20260801_v1/optimization_summary.json`，
32 张逐样本 logits、概率、预测、CPU 误差和网络快照见同目录
`medical32_pairwise_bitonic_summary.json`。本改动不覆盖正式展示数字。

## 9. 2026-08-02 完整 report-ready 回归

在后续优化中，RLWE packing 的常见蝶形路径由“复制 `E` 后分别原地计算和/差”
改为“和分支原地写回、差分支单独输出”，减少一次整密文复制。补丁只应用于隔离
SPU runtime，完整结果位于
`results/vps_report_tests/report_sumdiff_full_20260802_v1/`。

本轮重新执行了 524/32/8 全部固定样本、同机 medical32 baseline/candidate、13+4
黑盒验证、50 项 Python 测试与 2 项 SPU 原生测试。聚合门禁结果为
`report_update_ready=true`，无失败项。

| 项目 | 完整结果 | 判定 |
|---|---:|---|
| 医疗 524 阈值精度 / AUC | 92.7481% / 0.96391507 | 与正式口径一致 |
| medical32 同机 baseline | 970.085 s；42.574750 GiB | `r=0.7`、batch16 |
| medical32 candidate | 964.208 s；42.574552 GiB | 时间 -0.61%；通信基本不变 |
| medical32 阈值精度 / AUC | 93.75% / 0.96484375 | 通过 |
| finance8 | 377.855 s；15.454028 GiB；8/8 | 通过 |
| 13+4 / Python / SPU 原生测试 | 17/17；50/50；2/2 | 通过 |

medical32 两个 chunk 均略快，但合计只减少 5.878 秒，仍属于边际改善。该底层
补丁保留为研究候选，不作为主要性能突破，也不自动替换默认 runtime。完整说明见
`docs/evidence/spu_packing_sumdiff_full_regression.md`；本轮仍未改写任何正式结果目录。
