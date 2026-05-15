# 最终保留清单

## 1. 保留原则

当前 `Transshield_final` 面向比赛展示、答辩与最终提交，因此保留内容遵循以下原则：

- 能直接支撑 `baseline / modified / secure` 三条主线；
- 能支持最小必要复现；
- 能作为结果证据或答辩材料；
- 当前主线优先，能删掉的旧展示资产、旧候选资产、旧完整 checkpoint 就不继续留在仓内；
- `source/provenance` 快照只保留到不会反向干扰当前主入口为止。

另外，当前目录角色已分离：

- `Transshield_final`：权威源码、结果、文档、provenance 仓
- `Transshield`：服务器整仓替换使用的 clean deploy mirror

因此本仓的清理原则不是“无限保留历史”，而是：

- 删除已经被正式 clean 结果完整覆盖的旧 run / 重复件
- 只保留当前交付线仍会直接用到的源码、结果和最小说明材料

---

## 2. 训练核心与 provenance

| 路径 | 保留原因 | 类别 |
|---|---|---|
| `main.py`、`engine.py`、`infer.py` | 当前 modified 明文训练 / 评估主入口 | 训练核心 |
| `models/` | 当前最终仓的 ViT / DynamicViT 模型定义 | 训练核心 |
| `datasets.py`、`samplers.py`、`losses.py`、`optim_factory.py`、`utils.py`、`calc_flops.py` | 训练与评估基础模块 | 训练核心 |
| `pretrained/deit_small_patch16_224-cd65a155.pth` | 当前训练与复现实验使用的初始化权重 | 训练核心 |
| `training_source_tracka/` | TrackA source/provenance 快照；用于对齐 source 侧 runner 与历史 issue | provenance 保留 |
| `training_compat/` | 当前 server 侧 plaintext compatibility runner；与 `source` 做 parity / provenance 对照时必留 | provenance 保留 |
| `references/original_plaintext_runtime/` | baseline runtime 快照；用于说明“原始明文口径” | provenance 保留 |
| `artifacts/baselines/` | baseline 明文评估默认资产与阈值配置 | 训练资产 |
| `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/` | 当前正式 modified plaintext bundle | 训练资产 |
| `artifacts/train_runs/` | 当前仍需追溯的 server 训练日志 / stdout / tb 产物 | 训练证据 |

---

## 3. 安全推理核心

| 路径 | 保留原因 | 类别 |
|---|---|---|
| `integrations/openbumblebee/` | 当前只有两条 live bridge：`network-kth` 与 `tie-policy`；是 OpenBumbleBee 侧权威实现 | 安全推理核心 |
| `configs/openbumblebee/` | SPU / OpenBumbleBee live 配置与模板 | 安全推理核心 |
| `tools/transshield_openbumblebee_pipeline.py` | secure pipeline 统一入口 | 安全推理核心 |
| `tools/transshield_openbumblebee_inference_replay.py` | secure replay 与语义一致性校验入口 | 安全推理核心 |
| `tools/transshield_secure_sidecar_export_suite.py` | 高频 secure sidecar 一次性导出入口 | 安全推理核心 |
| `tools/transshield_secure_network_kth.py` | 统一收口 `network-kth` 输入导出、manifest、reference、checker 与 branch eval | 安全推理核心 |
| `tools/transshield_secure_tie_payload.py` | 统一收口 tie payload 导出、checker 与 branch eval | 安全推理核心 |
| `tools/transshield_spu_runtime_setup.py` | SPU runtime 启停、端口与兼容处理 | 安全推理核心 |
| `tools/transshield_stage2_report.py`、`tools/transshield_forward_trace.py`、`tools/transshield_pruning_trace.py`、`tools/transshield_reference_checker.py` | Stage-2 trace / contract / checker 工具 | 安全推理核心 |
| `tools/transshield_kth_threshold_report.py`、`tools/transshield_secure_kth_checker.py` | secure pruning 边界设计说明与复核工具 | 安全推理核心 |
| `artifacts/inference_ready_config/` | 已验证的 sidecar / selection-mode runtime inputs | 安全推理资产 |
| `artifacts/server_inference_friendly_pack/` | 最终服务器运行入口与权威 wrapper 集 | 安全推理资产 |
| `artifacts/server_pipeline_run/` | 当前 secure pipeline 运行证据与回放输入输出；只保留 current clean run、必要 calibration 与仍有引用价值的 evidence | 运行证据 |
| `artifacts/server_profile_reports/` | 当前 secure profile / selection-mode profile 报告 | 运行证据 |
| `logs/` | 当前仍被 secure 证据链引用的最小日志保留集 | 运行证据 |
| `artifacts/web_demo_assets/` | 当前前端统一摘要与离线最佳成绩数据源 | 展示资产 |

---

## 4. 展示、交接与结果材料

| 路径 | 保留原因 | 类别 |
|---|---|---|
| `README.md` | 比赛展示主入口 | 展示材料 |
| `docs/transshield_master_plan_20260505.md` | 当前最高优先级总路线主文档 | 展示材料 |
| `results/fair_external_comparison/fair_external_secure_static_20260505_clean/` | 当前公平外部对比主结果目录 | 展示材料 |
| `results/delivery_acceptance/delivery_acceptance_20260510_full/` | **当前交付闭环完整验收主结果目录（含 boundary check）** | 展示材料 |
| `results/delivery_acceptance/delivery_acceptance_20260505_clean/` | 初始验收结果目录 | 历史参考 |
| `docs/web_chat_demo.md` | Web demo 展示口径与使用说明 | 展示材料 |
| `docs/architecture.md` | 当前系统结构总览；同时承担 secure 代码导航 | 展示材料 |
| `docs/current_work_status.md` | 当前进度、最近收口与 repo 级改动记录 | 交接材料 |
| `docs/handoff-next.md` | 下次接手先读文档与权威命令入口 | 交接材料 |
| `docs/tracka_predictor1_root_cause_2026-04-21.md` | TrackA `ratio loss` 非首发驱动的归档结论 | 交接材料 |
| `docs/network_kth_blockwise_notes.md` | 当前 secure boundary 说明 | 交接材料 |
| `docs/data_source_policy.md` | 数据来源、默认数据集与展示口径主文档 | 展示材料 |
| `docs/retention_list.md` | 当前保留清单本身 | 展示材料 |
| `results/` | 已整理的最终结果 / 对比 / ablation 结果入口 | 结果材料 |
| `scripts/` | 高频训练 / SPU 入口与权威同步命令说明 | 运行说明 |
| `tools/README.md` | 当前支持的 final-repo 工具导航 | 运行说明 |
| `web_demo/` | 当前单页前端展示实现 | 展示材料 |
| `licenses/`、`THIRD_PARTY.md` | 第三方来源与许可证说明 | 交付材料 |

---

## 5. 当前主入口建议

比赛展示、交接或复现建议按以下顺序阅读或运行：

1. `README.md`
2. `docs/transshield_master_plan_20260505.md`
3. `docs/current_work_status.md`
4. `docs/handoff-next.md`
5. `docs/architecture.md`
7. `tools/README.md`
8. `scripts/README.md`
9. `artifacts/server_inference_friendly_pack/final_compare_env.template.sh`
10. `artifacts/server_inference_friendly_pack/run_full_final_comparison_smoke.sh`
11. `artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`

---

## 6. 说明

- 本仓库不再依赖外部旧副本作为正式代码来源。
- `DynamicViT_exp_square` 只作为研发来源档案，不再当作最终仓的 live 入口。
- baseline 与 modified 的默认对比流程均使用当前仓内轻量资产或 pure `state_dict`。
- secure 一致性展示以 `Modified Plaintext vs Secure` 为主，因为这是当前比赛版最重要的技术闭环。
- 安全推理代码导航现在统一由 `docs/architecture.md`、`tools/README.md` 与 `docs/handoff-next.md` 负责，不再单独保留 `secure_infer/` 目录。
- 历史 `results/` / `artifacts/` JSON、log、summary 中仍可能出现旧脚本名或旧绝对路径；这些属于 provenance 证据，不应反向覆盖当前 live 入口。
- `2026-05-05` 当前仓又执行了一轮主线化清理：旧 `archive/`、`frozen_candidates/`、`frozen_bundle_full/` 与历史展示 bundle 已全部移除，只保留当前 active delivery line 资产。
