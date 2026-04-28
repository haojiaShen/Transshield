# 当前工作状态

最后更新：`2026-04-22`

这个根目录文件现在只保留为**兼容入口**，避免再维护一份会过期的状态副本。

## 当前使用规则

- 权威状态总览：`docs/current_work_status.md`
- 权威 handoff：`docs/handoff-next.md`
- 权威路线图：`docs/algorithm_protocol_upgrade_roadmap.md`

## 这次为什么改成薄入口

- 之前根目录 `current_work_status.md` 已明显落后于 `docs/current_work_status.md`
- 用户和后续 agent 会直接从根目录打开状态文件，继续保留 stale duplicate 风险太高
- 现在统一为：
  - 根目录只做跳转提示
  - 真实内容只维护 `docs/current_work_status.md`

## 当前最该记住的补充

- TrackA 的 server env provenance 已闭合，当前先排除 `/data/wyb/conda_envs/transshield` 独立漂移
- `docs/history_best_repro_drift_audit_2026-04-21.md` 是唯一主审计文档
- `training_source_tracka/` 与 `references/original_plaintext_runtime/` 仍是故意保留的 provenance / baseline runtime 快照，不是误删候选
- 根目录 `algorithm_protocol_upgrade_roadmap.md`、`transshield_blockwise_kth_selection_manifest.py`、`transshield_stagewise_threshold_report.py`、`transshield_network_kth_bridge.py` 现在也统一只保留兼容入口

如需继续工作，请直接打开 `docs/current_work_status.md`。
