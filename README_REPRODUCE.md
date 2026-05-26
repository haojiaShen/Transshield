# 密捷 TransShield 结果核对说明

本文档面向评委与仓库维护者。当前最终仓除了正式结果、正式报告和冻结模型资产外，还包含一套可运行的**评委展示站 + 单通道 SPU Live Demo**。旧 live demo / runtime rerun wrapper 已归档，不再作为当前主入口。

## 1. 环境依赖

- Python：`3.9`
- 基础依赖：`python3 -m pip install -r requirements.txt`
- 建议系统：Linux / WSL，支持 `fc-match` 时中文图件字体效果更稳定

## 2. 最小核对路径

如果只核对最终交付物，按以下顺序即可：

1. 查看正式报告 `docs/密捷竞赛作品报告.docx`
2. 查看正式指标 JSON
3. 查看证据链索引

如果要启动展示站并验证可运行演示，按第 4 节执行。

## 3. 当前可直接核对的结果文件

| 类型 | 位置 |
|---|---|
| 正式报告 | `docs/密捷竞赛作品报告.docx` |
| 证据链索引 | `docs/evidence/README.md` |
| 医疗阈值 | `results/final/medical_dynamic_threshold_calibration_final.json` |
| 医疗 AUC | `results/final/medical_dynamic_auc_reference_final.json` |
| 通信量 | `results/communication/mainline_communication_profile_final.json` |
| 协议 fuzz 终版结果 | `results/fuzzing/protocol_fuzz_final.json` |
| guard 终版结果 | `results/guard_stress/guard_stress_final.json` |

## 4. 展示站 / Live Demo 最小启动步骤

### 4.1 Python 环境

建议使用独立虚拟环境：

```bash
cd /path/to/Transshield
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 4.2 构建前端展示站

```bash
cd /path/to/Transshield/showcase
npm install
npm run build
```

### 4.3 一键启动完整 SPU 演示链路

完整启动命令如下：

```bash
cd /path/to/Transshield
python3 tools/start_showcase_spu_demo.py --host 0.0.0.0 --port 7860
```

该命令会完成以下动作：

- 检查 `showcase/dist` 是否存在
- 从 `configs/transshield_runtime/2pc.template.json` 复制一份运行时配置到 `logs/showcase_runtime/2pc.runtime.json`，并只在这份 `logs/` 下的临时配置里写入当前空闲端口
- 启动 colocated 2PC / SPU 节点并执行 warmup
- 启动 `showcase_api.app:app` 并托管 `showcase/dist`
- 轮询 `GET /api/health`，确认运行模式为 `spu`

若本机暂时没有完整 SPU/JAX 运行栈，可先用 mock 验证前后端闭环：

```bash
cd /path/to/Transshield
python3 tools/start_showcase_spu_demo.py --runtime-mode mock --host 0.0.0.0 --port 7860
```

说明：

- `showcase_api` 会托管 `showcase/dist` 静态站点
- `GET /api/medical/config` 返回 bundle、阈值、均值/方差、输入尺寸和等待时间提示
- `POST /api/medical/live-run` 是唯一 medical live upload 入口
- live upload 只针对医疗；金融只在 `/results` 页面做结果与压力验证展示
- 当前展示站启动的是 colocated 2PC / SPU 原型，不是真实跨机器医院方 + AI 方部署。真两方迁移边界见 `docs/party_split_2pc.md`。

### 4.4 可选黑盒验证脚本

```bash
python3 tools/showcase_protocol_fuzz.py --base-url http://127.0.0.1:7860
python3 tools/showcase_guard_stress.py --base-url http://127.0.0.1:7860 --timeout 140
```

如果只想先验证一次 accepted path：

```bash
python3 tools/showcase_protocol_fuzz.py --base-url http://127.0.0.1:7860 --cases baseline --timeout 140
```

`baseline` 会构造一份“合成医疗图像张量 + share0/share1 + 结构化摘要”的 multipart 请求，用来验证 `/api/medical/live-run` 的 accepted path 能进入 SPU 并返回 `completed`。它不读取本地 PNG/JPEG 文件；如果要验证“浏览器选择真实图片文件”的交互路径，应在 `/live-demo` 页面手动选择图片，由前端 Worker 完成解码、预处理和分片生成。

### 4.5 真两方部署迁移备注

当前仓库已经提供基础的“一方医院、一方 AI 公司”SPU 节点启动命令，但它还不是可直接用于生产的完整医院/AI 双网关部署。已有能力是：

- split public/P1/P2 manifest：public manifest 不包含私有路径，P1/P2 party manifest 分别描述各自 share
- `--party-local-share-load`：让 P1/P2 device function 各自加载自己的 share，避免 runner 直接 `torch.load` 两份私有 share
- `--redact-private-input-paths`：避免 candidate 输出持久化 P1/P2 私有路径
- `tools/transshield_spu_runtime_setup.py render-remote`：生成跨机器 `2pc.remote.json` 与双方启动命令
- `tools/transshield_spu_runtime_setup.py start-party --party hospital|ai`：在每一方机器上后台启动自己的 SPU 节点，并写入日志和状态文件
- `tools/transshield_spu_runtime_setup.py render-deployment`：生成医院侧、算力方、协调方和展示站 API base URL 配置示例的部署包框架
- `showcase_api.split_gateway`：提供医院侧、AI 侧、协调侧三个最小网关 role，支持医院侧 raw PNG/JPEG 图片入口、单方 share 接收、AI 模型 manifest、协调侧异步任务状态与取消入口

本地可先验证 split gateway 框架：

```bash
python3 tools/split_gateway_smoke.py --keep-state
```

该脚本不会调用真实 SPU 长任务，而是用 `runtime_mode=mock` 跑完“医院上传合成 PNG -> 医院生成 P1/P2 share -> AI 接收 P2 share -> AI 提交 model manifest -> 协调方完成 mock run”的 API 闭环，同时检查鉴权、角色、share/协调阶段的 payload 或 manifest task_id，以及 share hash 拒绝路径。

仍需补齐的生产化工作包括：生产级双网关接入、单方私有 manifest 保管策略、模型参数私有加载、链路认证加密、任务队列接入和跨服务版 fuzz/guard 验收。详见 `docs/party_split_2pc.md`。

如果要用本地主机和远端服务器做一次跨机器验证，见 `docs/host_server_2pc_validation.md`。该文档区分直连模式与 SSH 隧道模式，并使用 `warmup_all_parties.sh` 作为 SPU runtime 是否真正跑通的判断标准。

### 4.6 当前远端演示部署备注

以下是内部评委演示环境备注，不是公开复现的必要条件。如需复用当前评委演示服务器，可按最近一次有效部署口径执行：

- 服务器：`10.204.248.175:9001`
- 远端仓库：`/data/wyb/Transshield_final`
- 远端 Python：`/data/wyb/conda_envs/transshield/bin/python`
- 当前常用服务端口：`7862`
- 前端构建完成后，由 `showcase_api` 直接托管 `showcase/dist`

若 `7862` 未对外放行，默认通过 SSH 隧道访问：

```bash
ssh -p 9001 -L 7862:127.0.0.1:7862 wyb@10.204.248.175
```

远端启动示例：

```bash
cd /data/wyb/Transshield_final
/data/wyb/conda_envs/transshield/bin/python tools/start_showcase_spu_demo.py --host 127.0.0.1 --port 7862 --daemon
```

后台启动后可通过 `artifacts/showcase_server_logs/uvicorn_7862.log` 查看展示站日志，通过 `logs/spu_runtime_ports.json` 查看当前 SPU 节点端口与 warmup 状态。

## 5. 已归档内容

- 旧 runtime / live demo 运行包：`archive/deprecated/artifacts/server_inference_friendly_pack/`
- 中间准备目录与静态参考目录：`archive/old_runs/artifacts/server_pipeline_run/`

## 6. 常见问题

- 为什么 README 里同时提到“旧 demo 已归档”和“当前有 Live Demo”：旧 `web_demo/` 与历史 wrapper 已归档；当前 `showcase/ + showcase_api/` 是按正式报告口径重建的新展示链
- 为什么不建议重建报告：当前仓库已保留正式 `.docx` 成品，报告源码重建链与相关额外依赖已从当前主口径移除
- 为什么没有单独图件目录：展示站图件由 `tools/showcase_extract_report_assets.py` 从正式 `docx` 提取到 `showcase/public/report-assets/`
- 为什么 mock 模式也有价值：它用于验证“浏览器分片 → 后端快检 → 结果返回”的控制面闭环；正式 SPU 长等待路径仍保留在默认运行模式里
