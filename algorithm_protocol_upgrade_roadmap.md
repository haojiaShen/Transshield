# 算法 / 协议升级路线图

最后更新：`2026-04-22`

这个根目录文件现在只保留为**兼容入口**，避免再维护一份会过期的路线图副本。

## 当前使用规则

- 权威路线图：`docs/algorithm_protocol_upgrade_roadmap.md`
- 当前状态总览：`docs/current_work_status.md`
- 下一次接手说明：`docs/handoff-next.md`

## 这次为什么改成薄入口

- 之前根目录 `algorithm_protocol_upgrade_roadmap.md` 已落后于 `docs/algorithm_protocol_upgrade_roadmap.md`
- 这类文件经常被当作“当前计划”直接引用，保留双份正文容易继续漂移
- 现在统一为：
  - 根目录只做跳转提示
  - 真实内容只维护 `docs/algorithm_protocol_upgrade_roadmap.md`

## 当前最该记住的补充

- `tools/transshield_stagewise_threshold_report.py` 是 Phase 1 的权威入口
- `tools/transshield_blockwise_kth_selection_manifest.py` 是 blockwise manifest 的权威入口
- `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py` 是 `network-kth bridge` 的权威实现
- 根目录同名 Python 入口现在也统一只保留兼容 wrapper，避免根目录继续藏一份 stale 实现

如需继续工作，请直接打开 `docs/algorithm_protocol_upgrade_roadmap.md`。
