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

### 4.3 启动后端控制面

#### 先做本地闭环验证（mock）

```bash
cd /path/to/Transshield
TRANSSHIELD_SHOWCASE_RUNTIME_MODE=mock uvicorn showcase_api.app:app --host 0.0.0.0 --port 7860
```

#### 切到单通道 SPU 实跑

```bash
cd /path/to/Transshield
uvicorn showcase_api.app:app --host 0.0.0.0 --port 7860
```

说明：

- `showcase_api` 会托管 `showcase/dist` 静态站点
- `GET /api/medical/config` 返回 bundle、阈值、均值/方差、输入尺寸和等待时间提示
- `POST /api/medical/live-run` 是唯一 medical live upload 入口
- live upload 只针对医疗；金融只在 `/results` 页面做结果与压力验证展示

### 4.4 可选黑盒验证脚本

```bash
python3 tools/showcase_protocol_fuzz.py --base-url http://127.0.0.1:7860
python3 tools/showcase_guard_stress.py --base-url http://127.0.0.1:7860
```

如果只想先验证一次 accepted path：

```bash
python3 tools/showcase_protocol_fuzz.py --base-url http://127.0.0.1:7860 --cases baseline
```

### 4.5 当前远端演示部署备注

如需复用当前评委演示服务器，可按最近一次有效部署口径执行：

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
/data/wyb/conda_envs/transshield/bin/python -m uvicorn showcase_api.app:app --host 127.0.0.1 --port 7862
```

## 5. 已归档内容

- 旧 runtime / live demo 运行包：`archive/deprecated/artifacts/server_inference_friendly_pack/`
- 中间准备目录与静态参考目录：`archive/old_runs/artifacts/server_pipeline_run/`

## 6. 常见问题

- 为什么 README 里同时提到“旧 demo 已归档”和“当前有 Live Demo”：旧 `web_demo/` 与历史 wrapper 已归档；当前 `showcase/ + showcase_api/` 是按正式报告口径重建的新展示链
- 为什么不建议重建报告：当前仓库已保留正式 `.docx` 成品，报告源码重建链与相关额外依赖已从当前主口径移除
- 为什么没有单独图件目录：展示站图件由 `tools/showcase_extract_report_assets.py` 从正式 `docx` 提取到 `showcase/public/report-assets/`
- 为什么 mock 模式也有价值：它用于验证“浏览器分片 → 后端快检 → 结果返回”的控制面闭环；正式 SPU 长等待路径仍保留在默认运行模式里
