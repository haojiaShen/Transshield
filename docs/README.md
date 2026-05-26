# docs

本目录只保留正式交付所需的文档资产，不再保留过程型工作日志。

## 当前结构

- `docs/密捷竞赛作品报告.docx`
  - 当前正式报告主文件
- `docs/report/`
  - 报告成品说明
- `docs/evidence/`
  - 证据链索引、审计说明、第三方本地修改说明
- `docs/party_split_2pc.md`
  - 从单机 colocated 演示迁移到医院侧 P1 / AI 侧 P2 的边界说明
- `docs/host_server_2pc_validation.md`
  - 用本地主机 + 远端服务器验证远程 2PC 部署包、SSH 隧道和 SPU warmup 的操作说明

## 当前口径

- `docs/` 以正式报告、最终图件、最终证据为核心。
- 早期 `current_work_status`、`handoff-next`、`master_plan` 一类过程文档已不再作为当前仓内主入口保留。
- 若需要最终指标、最终结果或报告映射，优先读取：
  - `docs/evidence/README.md`
  - `docs/report/README.md`
  - `results/final/`
  - `results/communication/`
  - `results/fuzzing/`
  - `results/guard_stress/`
  - `docs/party_split_2pc.md`
  - `docs/host_server_2pc_validation.md`

## 说明

- 当前可运行展示链位于仓库根目录的 `showcase/` 与 `showcase_api/`；`docs/` 本身只负责正式报告与证据说明，不承载运行入口。
- 旧版 `demo_app/` 已从当前最终仓移除，不再属于现行交付主链路。
- `docs/evidence/web_demo_control_plane_audit.md` 保留为历史正式前端的审计证据，不代表当前仍存在同路径实现。
- 本地验证医院/AI/协调三角色网关 API 闭环时，使用 `tools/split_gateway_smoke.py --keep-state`；说明见 `docs/party_split_2pc.md`。
