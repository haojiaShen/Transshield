# 下一次接手前先看这里

最后更新：`2026-04-22`

这个根目录文件现在只保留为**兼容入口**，避免再维护一份会过期的 handoff 副本。

## 当前使用规则

- 权威 handoff：`docs/handoff-next.md`
- 当前状态总览：`docs/current_work_status.md`
- 运行入口说明：`artifacts/server_inference_friendly_pack/README.md`

## 这次为什么改成薄入口

- 之前根目录 `handoff-next.md` 已明显落后于 `docs/handoff-next.md`
- 用户会频繁从 handoff 里抄运行信息，所以不能继续保留 stale duplicate
- 现在统一为：
  - 根目录只做跳转提示
  - 真实内容只维护 `docs/handoff-next.md`

## 当前最该记住的补充

- TrackA 的 server env provenance 已闭合，当前先排除 `/data/wyb/conda_envs/transshield` 独立漂移
- `docs/history_best_repro_drift_audit_2026-04-21.md` 是唯一主审计文档
- `run_secure_selection_mode_profile_compare.sh` 的权威入口在：
  - `artifacts/server_inference_friendly_pack/run_secure_selection_mode_profile_compare.sh`
- 根目录同名脚本现在也只是兼容 wrapper，避免旧默认值继续漂移

如需继续工作，请直接打开 `docs/handoff-next.md`。
