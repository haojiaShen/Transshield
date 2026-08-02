# 证据链索引

本目录提供正式交付所依赖的语义化证据入口。

| 证据类型 | 文件位置 | 支撑报告内容 |
|---|---|---|
| 协议 fuzz 最终结果 | `results/fuzzing/protocol_fuzz_final.json` | 鲁棒性矩阵、协议层异常输入阻断 |
| 协议 fuzz 元数据 | `results/fuzzing/protocol_fuzz_final.meta.md` | 原始来源、保留理由、引用位置 |
| 控制面 guard 最终结果 | `results/guard_stress/guard_stress_final.json` | 控制面防护、重放与并发守卫 |
| 控制面 guard 元数据 | `results/guard_stress/guard_stress_final.meta.md` | 原始来源、保留理由、引用位置 |
| 通信量正式结果 | `results/communication/mainline_communication_profile_final.json` | 性能分析、通信量统计 |
| 通信量说明 | `results/communication/mainline_communication_profile_final.meta.md` | 原始来源、重命名说明 |
| 医疗阈值正式结果 | `results/final/medical_dynamic_threshold_calibration_final.json` | 医疗主线精度、阈值校准 |
| 医疗阈值元数据 | `results/final/medical_dynamic_threshold_calibration_final.meta.md` | 原始来源、引用位置 |
| 医疗 AUC 正式结果 | `results/final/medical_dynamic_auc_reference_final.json` | 医疗主线 AUC 证据 |
| 医疗 AUC 元数据 | `results/final/medical_dynamic_auc_reference_final.meta.md` | 原始来源、引用位置 |
| 演示摘要正式结果 | `results/final/demo_content_summary_final.json` | 展示页摘要、评委演示数据 |
| 演示摘要元数据 | `results/final/demo_content_summary_final.meta.md` | 原始来源、引用位置 |
| 控制面代码审计 | `docs/evidence/web_demo_control_plane_audit.md` | 工程安全特色、控制面约束 |
| SPU 本地修改说明 | `docs/evidence/spu_bumblebee_local_modifications.md` | 第三方许可、vendored 修改说明 |
| 第三方许可与局部改动 | `spu_vendored/MODIFICATIONS.md` | vendored 文件改动入口 |
| 第三方许可文本索引 | `licenses/README.md` | 许可分发完整性、报告知识产权说明 |
| VPS 报告口径回归说明 | `docs/evidence/vps_report_regression.md` | 固定 524/32/8 样本、VPS-only 执行、逐文件与逐样本新证据 |
| VPS 报告测试矩阵 | `configs/report_vps_test_matrix.json` | 报告环境、数据、预处理、模型配置、指标、验收阈值与历史数字 |
| 2026-08-01 VPS candidate 总证据 | `results/vps_report_tests/report_regression_20260801_v1/report_regression_aggregate.json` | 524/32/8、同机 A/B、13+4、隐私事实、代码测试与逐门槛判定 |
| SPU 任意长度选择网络说明 | `docs/evidence/spu_unpadded_selection_network.md` | 精确 Top-K 网络构造、比较器预算、回退路径与 VPS 验证 |
| SPU 选择网络结构化结果 | `results/vps_optimization/selection_network_20260801_v1/optimization_summary.json` | 单样本、batch=4、medical32 的耗时、通信、精度和环境证据 |
| SPU 低延迟 r=0.655 档说明 | `docs/evidence/spu_low_latency_r0655.md` | 可选 token 保留率、配套阈值、性能收益、精度代价与回退方式 |
| SPU 低延迟档结构化结果 | `results/vps_optimization/unpadded_r0655_20260802_v1/optimization_summary.json` | medical32 耗时、通信、精度、源码哈希及 524 样本校准证据 |
| SPU 0.9.5 原生排序否决证据 | `results/vps_optimization/native_sort_spu095_20260802_v1/optimization_summary.json` | 局部微基准收益与整图性能/通信回退的对照 |
| SPU batch16 批量吞吐说明 | `docs/evidence/spu_batch16_throughput.md` | 批量适用边界、medical32 性能/精度、复现命令和单图禁用条件 |
| SPU batch16 结构化结果 | `results/vps_optimization/batch_scaling_20260802_v1/optimization_summary.json` | 两个 chunk 耗时、通信、固定点漂移、源码哈希和回退约束 |
| SPU 微优化否决证据 | `results/vps_optimization/pruning_clip_elision_20260802_v1/optimization_summary.json` | predictor clip 与低精度 rsqrt 的同机热缓存 A/B |
| SPU MLP 底层并行实验说明 | `docs/evidence/spu_mlp_parallel_packing.md` | 真实 MLP 形状、RLWE packing 并行、两次 medical32 及精度稳定性否决 |
| SPU MLP 底层并行结构化结果 | `results/vps_optimization/mlp_kernel_20260802_v2_pack/optimization_summary.json` | 构建/单测、微基准、medical32 性能与不接入判定 |
| SPU MLP fxp20 复验结果 | `results/vps_optimization/mlp_kernel_fxp20_20260802_v1/optimization_summary.json` | 局部误差改善、medical32 耗时/通信/AUC 与不接入判定 |
| SPU 平方激活常数实验说明 | `docs/evidence/spu_square_scale_optimization.md` | 权重折叠、公开架构常数、定点截断风险和安全剪枝后续审查 |
| SPU 平方激活常数结构化结果 | `results/vps_optimization/square_alpha_fusion_20260802_v1/optimization_summary.json` | MLP、medical4、medical32 的耗时/通信/精度和不接入判定 |
| SPU 精确剪枝与线程并发实验说明 | `docs/evidence/spu_odd_even_and_thread_tuning.md` | odd-even 精确选择网络、线程并发扫描、完整门槛和不接入判定 |
| SPU odd-even 结构化结果 | `results/vps_optimization/odd_even_selection_20260802_v1/optimization_summary.json` | 比较器预算、medical4 耗时/通信、单测和提前终止原因 |
| SPU 线程并发结构化结果 | `results/vps_optimization/thread_tuning_20260802_v1/optimization_summary.json` | MLP 扫描、medical4/medical32、精度与 AUC 否决证据 |
| SPU LayerNorm 与运行时后续优化说明 | `docs/evidence/spu_layernorm_and_runtime_followup.md` | LayerNorm 仿射折叠、后置归一化、调度开关和 SPU 0.9.5 同图筛选 |
| SPU LayerNorm 折叠结构化结果 | `results/vps_optimization/layernorm_affine_fusion_20260802_v1/optimization_summary.json` | 微基准、medical4 时间/通信/一致率与不接入判定 |
| SPU 调度后续结构化结果 | `results/vps_optimization/runtime_scheduler_20260802_v1/optimization_summary.json` | 矩阵拆分、intra/inter-op 调度结果与失败证据 |
| SPU 0.9.5 同图 MLP 结果 | `results/vps_optimization/spu095_unpadded_20260802_v1/optimization_summary.json` | 与 0.9.3b0 的真实 MLP 热态筛选和停止条件 |
| SPU 混合 RLWE packing 与公开平方常数说明 | `docs/evidence/spu_hybrid_rlwe_public_alpha.md` | 单/多分组调度、基线概率波动归因、两次完整门槛与回退边界 |
| SPU 混合 RLWE packing 结构化结果 | `results/vps_optimization/rlwe_equivalence_followup_20260802_v1/optimization_summary.json` | medical4、medical32、矩阵微基准、构建测试、源码哈希和正式口径隔离 |
| SPU batch32 停止证据 | `results/vps_optimization/batch32_screen_20260802_v1/optimization_summary.json` | 单位样本耗时回退、内存余量和未启动完整图的停止条件 |

## 说明

- `docs/密捷竞赛作品报告.docx` 是当前正式报告主文件。
- 当前正式证据代码入口以 `tools/fuzzing/` 和 `results/` 为准。
- `docs/evidence/web_demo_control_plane_audit.md` 记录的是上一版正式前端实现的审计快照；该实现路径已移除，仅作为报告证据保留。
- 当前新增的 `showcase/` 与 `showcase_api/` 是对既有控制面证据的**可运行重建**，不是新增第二套正式结果口径。
- `tools/showcase_protocol_fuzz.py` 与 `tools/showcase_guard_stress.py` 面向新展示站接口做运行时验收；正式报告中的最终鲁棒性数字仍以 `results/fuzzing/` 与 `results/guard_stress/` 为准。
- 后续模型测试只在 VPS 执行：完整报告回归写入 `results/vps_report_tests/`，优化候选写入 `results/vps_optimization/`；除非另行审核批准，不回写或覆盖本页列出的正式结果。
- `r=0.655` 是显式启用的低延迟候选档，不替换正式 `r=0.7` 展示口径；启用时必须同时切换到其独立校准阈值。
- 最初的全量外层 MLP RLWE packing 候选曾因完整模型运行间波动被否决；后续未修改 runtime 复跑证明 CHEETAH 基线本身也存在同类概率截断波动。单/多分组混合调度与公开固定平方常数组合已通过两次干净节点完整门槛，接受为 `r=0.655 + batch16` 的显式吞吐档；默认 runtime 和正式展示仍不自动应用。
- fxp20 复验没有降低模型精度或改变 MLP 图，但 32 样本 AUC 和通信门槛仍未通过，默认 FM64/fxp16 配置不变。
- 早期单独评估平方激活 α 权重折叠和公开常数时，重复 medical32 门槛未全部通过；公开常数后来与混合 RLWE packing 组合完成两次完整验收，但仍保持显式吞吐开关，不替换正式结果。
- odd-even 精确选择网络虽减少 8.80% 比较器，但 medical4 的耗时和通信均回退；它只保留显式研究入口，默认仍为 `unpadded_selection`。
- `max_concurrency=8` 虽缩短 medical32 耗时，但通信、阈值准确率和 AUC 未同时过门槛；生产 SPU runtime 配置保持不变。
- LayerNorm 仿射折叠虽减少 medical4 通信，但端到端时间略有回退；开关默认 `none`，未运行 medical32，也未修改正式结果。
- 后置归一化、额外调度开关和 SPU 0.9.5 同图 MLP 均未通过前置性能筛选，不进入正式运行时。
- batch32 的真实 MLP 单位样本热态耗时比 batch16 慢 0.63%，节点内存约占 51 GiB，因此未花费完整 medical32。
