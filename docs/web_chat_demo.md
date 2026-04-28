# Web Demo 说明

最后更新：`2026-04-27`

## 页面现在怎么分区

### 1. 顶部总览

- 只显示**当前浏览器本地选择图片**触发的即时结果摘要
- 不显示离线验证集准确率

### 2. 交互演示

- 浏览器本地读取图片，不上传原图文件
- 浏览器端完成 resize / center crop / normalize
- 浏览器端生成 `share0/share1`
- 调用 `/api/e2e/analyze_private_shares`，后端只接收两份 share 并启动 E2E approximate SPU
- 在同一视区查看：
  - 当前图片的 E2E 推理结果
  - `finite_logits`、输入模式、host 明文 materialize 状态等隐私字段
  - 本次 E2E SPU live run 的耗时与通信证据

旧 `/api/upload` 与 `/api/run_secure` 只作为 legacy CPU/SPU sidecar 调试接口保留；默认需要显式设置 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1` 才能使用，不再是最终页面主路径。

当前为避免服务器 native/torch import 卡死影响演示，Web 后端还有两条保护：

- 启动阶段不再顶层导入明文分析 / 剪枝可视化所需 torch 模块；
- E2E 结果读取优先使用 candidate JSON 的 `prediction_preview`，不再对 candidate `.pt` 做 `torch.load`。

public layer norm calibration 默认要求提前生成；网页不会自动校准，除非显式设置 `WEB_DEMO_AUTO_CALIBRATE_E2E=1`。

`2026-04-27` 后的 E2E approximate Web 路径还应带上 output calibration：优先设置 `WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON`，否则后端会尝试从 `artifacts/server_pipeline_run/e2e_output_calibration*.json` 自动选择最新文件。服务器当前 smoke-stable 文件是 `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json`；该结论只覆盖 `class0_4 / class1_4 / balanced8` smoke，不是 full-val 证明。

### 3. 统一对比区

- 显示离线验证集最佳成绩
- 显示与 `MPCViT` 的外部基线对比
- 明确说明“这不是当前浏览器选择图片的单样本结果”
- 当前已写入统一 secure benchmark，会额外显示 benchmark 口径下的外部 secure proxy 对比；该数字仍不能和网页单图 live run 或 full-val pipeline 混用

### 4. 动态剪枝可视化

- `Original -> Stage 1 -> Stage 2 -> Stage 3`
- 绿色：继续参与后续计算的 token
- 灰色：在当前阶段被安全置零的 token
- 这不是病灶分割图

### 5. 附录

- 只保留必要工程细节
- 不再保留历史 fastpath 8 样本通信
- 不再保留 archived SPU profile

## 当前页面的数据规则

- 顶部与交互区：只认 `/api/e2e/analyze_private_shares` 返回的当前图片 E2E 结果
- 统一对比区：只认 `artifacts/web_demo_assets/best_demo_content.json`
- 外部基线：只认当前保留的 `MPCViT` 同数据集明文对比
- 外部 secure benchmark：只认 `results/standardized_secure_benchmark/<run>/` 中由统一脚本生成的报告

## 启动方式

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=/path/to/python
export WEB_DEMO_HOST=127.0.0.1
export WEB_DEMO_PORT=7860
export WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON=/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

服务器上如需外部访问：

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=/path/to/python
export WEB_DEMO_HOST=0.0.0.0
export WEB_DEMO_PORT=7860
export WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON=/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

## 当前页面的设计目标

- 先讲结论，再讲证据，再讲细节
- 让评委一眼看懂“当前图片不上传原图、E2E 隐私路径可运行、离线效果和通信量有同口径评测入口”
