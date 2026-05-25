# Tools

当前仓库保留两类工具：

1. 正式结果 / 正式证据导出工具
2. 新 `showcase/ + showcase_api/` 展示站的构建、抽取与黑盒验收工具

| 脚本 | 用途 | 输入 | 输出 | 是否为正式入口 |
|---|---|---|---|---|
| `tools/start_showcase_spu_demo.py` | 一键启动当前展示站与单通道 SPU Live Demo | 运行端口、Python 环境、SPU 配置 | 前端页面、FastAPI 服务、SPU 节点与健康检查结果 | 是 |
| `tools/transshield_spu_runtime_setup.py` | 配置并启动 colocated 2PC / SPU 节点 | `configs/transshield_runtime/2pc*.json` | `logs/spu_runtime_ports.json`、`logs/spu_nodes/` | 运行依赖 |
| `tools/fuzzing/protocol_fuzz.py` | 导出已保留的协议 fuzz 最终证据 | `--out` | `results/fuzzing/protocol_fuzz_final.json` | 否 |
| `tools/fuzzing/guard_stress.py` | 导出已保留的 guard stress 最终证据 | `--out` | `results/guard_stress/guard_stress_final.json` | 否 |
| `tools/showcase_extract_report_assets.py` | 从正式 `docx` 提取展示站图件与章节 JSON | `docs/密捷竞赛作品报告.docx` | `showcase/public/report-assets/`、`showcase/src/generated/report_content.json` | 否 |
| `tools/showcase_protocol_fuzz.py` | 对 `/api/medical/live-run` 做 black-box multipart / 协议拒绝测试 | `--base-url` | JSON 报告（可自定义 `--out`） | 否 |
| `tools/showcase_guard_stress.py` | 对 `/api/medical/live-run` 做 replay / inflight / rate-limit 守卫验证 | `--base-url` | JSON 报告（可自定义 `--out`） | 否 |
| `tools/transshield_e2e_secure_infer.py` | E2E share / pixel package 工具 | bundle、图像、manifest | 中间 `.pt/.json` 与推理结果 | 运行依赖 |
| `tools/transshield_stage2_bundle.py` | bundle 加载与阈值解析 | bundle 目录 | 运行时模型 / 阈值对象 | 运行依赖 |

## 当前精简原则

- `tools/fuzzing/` 目录保留正式证据导出入口
- `tools/showcase_*` 目录下脚本服务于当前评委展示站，不直接改写正式 `results/fuzzing/` / `results/guard_stress/` 终版数字
- `tools/start_showcase_spu_demo.py` 是当前推荐的展示站启动入口；它会先启动 SPU 节点，再启动 `showcase_api`
- 图件生成链已移除，当前仓库不再提供图件重建脚本
- 报告源码重建链和旧快照机制已移除，当前仓只保留正式 `docx` 成品
- 已删除的历史 `web_demo/` wrapper、重复报告 wrapper 和 `__pycache__` 不再进入最终提交面
