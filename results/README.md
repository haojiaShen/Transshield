# `results/` 目录说明

本目录只保留当前仍有复现价值的阶段性摘要和后续运行输出，但**当前主展示口径不直接从这里抓数字**。

## 先看这些权威文档

1. `../docs/data_source_policy.md`
2. `../docs/result_summary.md`
3. `../docs/external_baseline_comparison.md`
4. `../docs/web_chat_demo.md`

## 使用原则

- 如果你要引用**当前权威指标**，先回到 `docs/` 中的权威文档确认；
- 如果你要追溯某次运行细节，优先查看新生成的子目录，例如 `fair_external_comparison/<run>/` 或 `standardized_secure_benchmark/<run>/`；
- 根目录只保留少量 phase3 / protocol 说明文件，不再保留旧 followup、dry-run、audit 快照；
- 不要把本目录中的任何阶段性摘要直接当成当前前端展示数字。

## 当前分类

### 1. 当前仍有 live 作用或仍被文档直接引用

- `blockwise_exact_kth_selection_manifest_default.json`
  - 当前 secure wrapper 默认使用的 Phase-3 manifest
- `blockwise_exact_kth_manifest_20260418_004103.json`
- `blockwise_exact_kth_manifest_20260418_004103.md`
  - 当前 `blockwise_exact_kth` 设计说明证据
- `fair_external_comparison/fair_external_20260417_143051/`
  - 当前公平外部对比的已落盘报告
- `standardized_secure_benchmark/standardized_secure_benchmark_20260417_175843/`
- `standardized_secure_benchmark/standardized_secure_ops_20260417_181805/`
  - 当前 standardized secure benchmark 的两条已完成结果
- `margin_aware_pruning_ablation/margin_aware_full20_w10_20260417_205242_w10_t1em4_resume/`
- `margin_aware_pruning_ablation/margin_formal_hparams_soft_stage2_20260417_231946/`
- `margin_aware_pruning_ablation/margin_w10_secure_check_20260417_212831/`
  - 当前 margin-aware 研究线仍有引用价值的主要证据
- `stagewise_protocol_risk_tracka_lr3e5_verified_20260415.json`
  - margin ablation 仍在使用的 baseline risk 基准

### 2. 小体量设计空间佐证

以下目录当前几乎没有 live wrapper / 主文档直接引用，主要用于保留 mixed payload 设计搜索证据：

- `payload_precision_ablation_20260418_012922/`
- `payload_precision_mix_20260418_013239/`
- `payload_boundary_window_formal_20260418_014147/`
- `payload_boundary_window_w10_20260418_014656/`
- `payload_formal_stage1_fp32_20260418_015623/`
- `w10_payload_precision_20260418_013507/`

这批目录合计只有约 `248K`，因此**不是当前优先清理对象**；若后续继续瘦身，更适合先把它们汇总成单份 payload 结论文档，再决定是否删原始小报告。

### 3. 本地清理原则

- 本地 `results/` 现在只保留**有实质内容**的结果目录；
- 之前误同步进来的空占位目录，以及未形成完整 benchmark 汇总的半成品目录，已从本地删除；
- 如果文档中仍出现某些 `/data/wyb/Transshield_final/results/...` 路径，应按**服务器 provenance 路径**理解，不等于本地仓库必须保留同名空目录。
