# 密捷 TransShield

密捷（TransShield）是面向医疗影像隐私推理的双向隐私安全推理系统。当前仓库保留正式作品报告、最终证据、最终冻结模型资产，以及一套可运行的**评委展示站 + 单通道 SPU Live Demo**。旧 live demo / runtime wrapper 已归档；当前可运行展示链路位于 `showcase/` 和 `showcase_api/`。

## 当前正式指标

| 指标 | 数值 | 正式证据 |
|---|---:|---|
| 医疗阈值精度 | 92.7481% | `results/final/medical_dynamic_threshold_calibration_final.json` |
| 医疗 AUC | 0.9639 | `results/final/medical_dynamic_auc_reference_final.json` |
| 医疗端到端时延 | 89.06 秒/样本 | `results/communication/mainline_communication_profile_final.json` |
| 医疗双向通信量 | 84.47 GiB | `results/communication/mainline_communication_profile_final.json` |
| 鲁棒性验证 | 17 / 17 | `results/fuzzing/protocol_fuzz_final.json`、`results/guard_stress/guard_stress_final.json` |

## 当前目录

```text
Transshield/
├── main.py                    # 明文训练主入口
├── training_core/             # 训练/评估辅助模块
├── showcase/                  # Vite + React 展示站前端
├── showcase_api/              # FastAPI 控制面与静态托管
├── tools/                     # 证据导出与展示站验证工具
│   └── fuzzing/               # 导出最终协议 fuzz / guard 证据
├── docs/
│   ├── evidence/              # 证据索引与审计说明
│   └── report/                # 报告补充说明
├── archive/                   # 已归档的旧 runtime 包与历史运行目录
├── results/
│   ├── final/                 # 正式摘要与原始校准记录
│   ├── communication/         # 通信量最终记录
│   ├── fuzzing/               # 协议 fuzz 最终记录
│   └── guard_stress/          # guard 最终记录
├── artifacts/                 # 最终 bundle 与最终运行证据
├── integrations/              # SPU / secure runtime 相关实现
├── models/                    # 正式保留的 DynamicViT / PredictorLG 核心模型代码
├── configs/transshield_runtime/ # 2PC 运行配置
├── spu_vendored/              # vendored SPU 与修改说明
└── README_REPRODUCE.md        # 评委复现说明
```

## 快速查看

```bash
cd /path/to/Transshield
python3 -m pip install -r requirements.txt
```

- 正式报告：`docs/密捷竞赛作品报告.docx`
- 证据索引：`docs/evidence/README.md`

## 评委展示站与 Live Demo

- 前端源码：`showcase/`
- 后端入口：`showcase_api.app:app`
- 固定路由：
  - `/`
  - `/overview`
  - `/design`
  - `/implementation`
  - `/results`
  - `/innovation`
  - `/reproduce`
  - `/live-demo`
- 演示边界：
  - **医疗**：唯一 live upload + live run 场景
  - **金融**：仅保留结果与压力验证展示，不提供现场上传运行
  - 浏览器只上传 `share0/share1 + 结构化摘要`，不上传原图与明文像素包
  - 正式参考时延约 `89.06 秒/样本`，进入 SPU 后当前 demo 原型不能保证断连即终止

完整 SPU Live Demo 启动：

```bash
cd /path/to/Transshield
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
cd showcase && npm install && npm run build && cd ..
python3 tools/start_showcase_spu_demo.py --host 0.0.0.0 --port 7860
```

`tools/start_showcase_spu_demo.py` 会先启动并 warmup colocated 2PC / SPU 节点，再启动 `showcase_api` 托管 `showcase/dist`。若本机暂时没有完整 SPU/JAX 运行栈，可先用 mock 验证闭环：

```bash
python3 tools/start_showcase_spu_demo.py --runtime-mode mock --host 0.0.0.0 --port 7860
```

服务器后台启动示例：

```bash
cd /data/wyb/Transshield_final
/data/wyb/conda_envs/transshield/bin/python tools/start_showcase_spu_demo.py --host 127.0.0.1 --port 7862 --daemon
```

## 当前保留入口

| 入口 | 作用 |
|---|---|
| `showcase/` | 评委展示站前端工程，承载章节展示与医疗 Live Demo 页面 |
| `showcase_api.app:app` | FastAPI 控制面、静态托管和 `/api/medical/live-run` 入口 |
| `tools/start_showcase_spu_demo.py` | 一键启动 SPU 节点、展示站 API 与健康检查 |
| `tools/transshield_spu_runtime_setup.py` | 重写 2PC 端口、启动 SPU 节点并执行 warmup |
| `tools/transshield_stage2_bundle.py` | 读取冻结 bundle 与阈值 |
| `tools/transshield_e2e_secure_infer.py` | E2E share / pixel package 工具 |
| `tools/fuzzing/protocol_fuzz.py` | 导出最终协议 fuzz 证据 |
| `tools/fuzzing/guard_stress.py` | 导出最终 guard 证据 |
| `tools/showcase_protocol_fuzz.py` | 对新 showcase live demo 接口做黑盒 multipart / 协议拒绝测试 |
| `tools/showcase_guard_stress.py` | 对新 showcase live demo 接口做 replay / inflight / rate-limit 验证 |

## 正式交付物

| 类型 | 位置 |
|---|---|
| 正式报告 | `docs/密捷竞赛作品报告.docx` |
| 证据索引 | `docs/evidence/README.md` |
| 复现说明 | `README_REPRODUCE.md` |
| 工具说明 | `tools/README.md` |
| 归档说明 | `archive/README.md` |

## 许可证与第三方说明

- 第三方许可汇总：`THIRD_PARTY.md`
- 许可证文本索引：`licenses/README.md`
- SPU vendored 原位许可证：`spu_vendored/LICENSE`
- SPU vendored 修改说明：`spu_vendored/MODIFICATIONS.md`
