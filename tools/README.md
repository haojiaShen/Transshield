# Tools

当前仓库保留两类工具：

1. 正式结果 / 正式证据导出工具
2. 新 `showcase/ + showcase_api/` 展示站的构建、抽取与黑盒验收工具

| 脚本 | 用途 | 输入 | 输出 | 是否为正式入口 |
|---|---|---|---|---|
| `tools/start_showcase_spu_demo.py` | 一键启动当前展示站与单通道 SPU Live Demo | 运行端口、Python 环境、`configs/transshield_runtime/2pc.template.json` | `logs/showcase_runtime/2pc.runtime.json`、前端页面、FastAPI 服务、SPU 节点与健康检查结果 | 是 |
| `tools/transshield_spu_runtime_setup.py` | 配置并启动单机 colocated 2PC / SPU 节点；也可用 `render-deployment` 生成医院/算力方/协调方部署包 | `configs/transshield_runtime/2pc*.json` | `logs/spu_runtime_ports.json`、`logs/spu_nodes/`、`logs/spu_party_nodes/`、`deploy/transshield_remote_2pc/` | 运行依赖 |
| `showcase_api.split_gateway` | 医院/AI/协调三方最小网关框架，由部署包中的 `start_split_gateway.sh` 启动 | `.gateway.env`、PNG/JPEG、share/model/manifest JSON | 医院侧图片分片、单方 share 存储、AI model manifest、协调侧异步任务状态 | 迁移骨架 |
| `tools/split_gateway_smoke.py` | 本地验证 split gateway 三角色 API 闭环和拒绝路径 | 合成 PNG、临时 gateway state | JSON smoke 报告 | 验收工具 |
| `tools/fuzzing/protocol_fuzz.py` | 导出已保留的协议 fuzz 最终证据 | `--out` | `results/fuzzing/protocol_fuzz_final.json` | 否 |
| `tools/fuzzing/guard_stress.py` | 导出已保留的 guard stress 最终证据 | `--out` | `results/guard_stress/guard_stress_final.json` | 否 |
| `tools/showcase_extract_report_assets.py` | 从正式 `docx` 提取展示站图件与章节 JSON | `docs/密捷竞赛作品报告.docx` | `showcase/public/report-assets/`、`showcase/src/generated/report_content.json` | 否 |
| `tools/showcase_protocol_fuzz.py` | 对 `/api/medical/live-run` 做 black-box multipart / 协议拒绝测试 | `--base-url` | JSON 报告（可自定义 `--out`） | 否 |
| `tools/showcase_guard_stress.py` | 对 `/api/medical/live-run` 做 replay / inflight / rate-limit 守卫验证 | `--base-url` | JSON 报告（可自定义 `--out`） | 否 |
| `tools/transshield_e2e_secure_infer.py` | E2E share / pixel package 工具 | bundle、图像、manifest | 中间 `.pt/.json` 与推理结果 | 运行依赖 |
| `tools/report_vps_test.py` | 按最终报告的 524/32/8 固定样本口径生成 VPS-only 数据清单、环境、逐样本推理与通信证据 | `configs/report_vps_test_matrix.json`、正式样本清单、VPS 数据根目录和 candidate/reference | `results/vps_report_tests/<run>/` 下的详细 JSON | 验收工具 |
| `tools/report_vps_aggregate.py` | 汇总同 VPS A/B、524/32/8、13+4、隐私边界与代码测试，并给出逐门槛判定 | VPS candidate run 目录、报告测试矩阵 | `report_regression_aggregate.json` | 验收工具 |
| `tools/transshield_stage2_bundle.py` | bundle 加载与阈值解析 | bundle 目录 | 运行时模型 / 阈值对象 | 运行依赖 |

## 当前精简原则

- `tools/fuzzing/` 目录保留正式证据导出入口
- `tools/showcase_*` 目录下脚本服务于当前评委展示站，不直接改写正式 `results/fuzzing/` / `results/guard_stress/` 终版数字
- `tools/start_showcase_spu_demo.py` 是当前推荐的展示站启动入口；它会先启动 SPU 节点，再启动 `showcase_api`
- `tools/transshield_spu_runtime_setup.py render-deployment` 生成医院/算力方/协调方部署包；医院/AI 各自用 `start-party` 启动本方节点，也可用 `start_split_gateway.sh` 启动最小三方网关框架，协调方可用 `warmup` 验证最小 SPU runtime；真两方迁移说明见 `docs/party_split_2pc.md`
- `tools/split_gateway_smoke.py --keep-state` 可在没有真实医院/AI 系统时先验证医院图片入口、AI P2 share 接收、模型 manifest、协调方 mock run 以及常见拒绝路径
- 正式报告规模的回归只在 VPS 执行，使用 `tools/report_vps_test.py` 和 `docs/evidence/vps_report_regression.md`；新结果不得覆盖既有正式 JSON
- 图件生成链已移除，当前仓库不再提供图件重建脚本
- 报告源码重建链和旧快照机制已移除，当前仓只保留正式 `docx` 成品
- 已删除的历史 `web_demo/` wrapper、重复报告 wrapper 和 `__pycache__` 不再进入最终提交面
