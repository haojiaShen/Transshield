# Transshield Final

`Transshield_final` 是当前对外展示、答辩和交付使用的最终作品仓库。项目核心目标是：在保护医疗影像隐私的前提下，用**浏览器本地分片的 E2E 隐私输入推理**、**安全掩码动态剪枝**与**安全友好算子替换**降低安全推理开销，并保持接近明文模型的效果。

## 当前仓库只认这几类数据

- **当前浏览器选择图片的即时结果**：来自 Web demo 的浏览器本地分片 + E2E approximate SPU 运行，只能用于展示这张图的安全结果和本次 live run 开销。
- **离线验证集最佳成绩**：来自 `artifacts/web_demo_assets/best_demo_content.json`，只能用于统一对比区，不是当前图片结果。
- **外部基线对比**：当前主对比对象是同数据集明文基线 `MPCViT`。
- **禁止复用的旧口径**：历史 fastpath 8 样本通信、旧 archived SPU profile、旧正式展示模型收益、dated handoff / request 文档中的数字。

详细规则见 `docs/data_source_policy.md`。

## 快速开始

### 启动 Web Demo

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=/path/to/python
export WEB_DEMO_HOST=127.0.0.1
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

如果是在服务器上启动，并希望本机浏览器访问：

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=/path/to/python
export WEB_DEMO_HOST=0.0.0.0
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

然后访问 `http://<server-ip>:7860`。

### 运行 legacy secure sidecar 对比链

```bash
cd /path/to/Transshield_final
source artifacts/server_inference_friendly_pack/final_compare_env.template.sh
export TRAIN_DATA_PATH=/path/to/pneumoniamnist_imagefolder_subset/train
export VAL_DATA_PATH=/path/to/pneumoniamnist_imagefolder_subset/val
export SECURE_RUNTIME=spu
bash artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh
```

## 仓库结构

- `artifacts/`：冻结展示包、Web demo 资源、server pipeline 运行产物
- `docs/`：当前保留的权威说明文档
- `models/`：明文模型与结构改造实现
- `integrations/`：安全执行与外部后端集成
- `tools/`：导出、比对、可视化、Web demo 后端等工具
- `web_demo/`：前端展示页
- `configs/`：OpenBumbleBee / SPU 运行配置
- `references/`：baseline 最小运行快照
- `licenses/`、`THIRD_PARTY.md`：第三方来源与许可证说明

## 当前建议阅读顺序

1. `docs/project_overview_newcomer_defense.md`
2. `docs/data_source_policy.md`
3. `docs/result_summary.md`
4. `docs/external_baseline_comparison.md`
5. `docs/web_chat_demo.md`
6. `docs/current_work_status.md`
7. `docs/handoff-next.md`

## 当前展示原则

- 页面顶部和交互区只显示**当前浏览器选择图片**触发的 E2E 隐私推理即时结果。
- 准确率、AUC、最佳轮次、外部基线差值只放在统一对比区。
- 通信量默认只显示**本次 E2E SPU live run**，不再使用固定历史字节数。
- 附录只保留必要工程细节，不再保留旧 fastpath / archived profile 数字。

## 说明

- 当前仓库已经删除大量 dated 文档，避免旧数字继续回流到前端或答辩材料。
- 如果后续需要补新的比较数据，必须先在 `docs/data_source_policy.md` 中确认口径是否允许。
