# Transshield Final

`Transshield_final` 是当前对外展示、答辩和交付使用的最终作品仓库。

当前仓库的**最高优先级主文档**是：

- `docs/transshield_master_plan_20260505.md`

如果 `README.md`、`docs/current_work_status.md`、`docs/handoff-next.md`、`docs/transshield_innovation.md` 与其他旧摘要之间存在冲突，**一律以 `docs/transshield_master_plan_20260505.md` 为准**。

当前正式主线已经收束为：

- 以 **pruning boundary 的协议友好重写** 作为唯一主创新轴；
- 用 `masking -> F_mux`、`threshold compare -> F_less`、`secure sidecar / replay` 形成方法闭环；
- 用 `secure_static` 训练底座、`secret_blockwise_stage` 最小 secret runtime、same-policy/plaintext 对齐、fairness report 和 guarded secret eval 形成当前交付闭环。

当前主线还应按下面四点理解：

- 当前选择 `ViT / DynamicViT` 作为主模型，不是因为 “CNN 对胸片分类没价值”，而是因为 token-level pruning boundary 更适合作为当前 `F_less / F_mux` 主创新的载体。
- 当前仍然存在真实的明文 pruning；变化的是 secure-facing 语义从“直接删 token”改写成了 masking-friendly `keep/zero` 表达。
- 当前 “动态 pruning” 的动态性来自样本级、stage 级 `kth` 边界，而不是一个全局固定阈值；它也不是最终二分类评测阈值。
- `CNN + ViT` hybrid 不属于当前主线；`embedding / position encoding` 的 secure 优化只作为后续 `P2` 候选，不写入当前交付承诺。

当前主线目标按“双向隐私”理解：服务器不应看到用户原图 / plaintext pixels，客户端也不应看到服务器模型参数，只 reveal 最终分类结果。只把本地 backbone 或本地 encoder 明文执行、再对小 head 做 2PC 的路线不再作为主线，因为它会让客户端持有模型参数，不能满足双向隐私要求。

## 当前入口差异

当前仓库仍保留两类资产，但默认运行主线已经统一：

- 本地权威源码/结果仓：
  - `Transshield_final`
- 服务器替换使用的 clean mirror：
  - `Transshield`

- 当前默认 delivery bundle：
  - `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- 当前默认 secret live profile：
  - `secret_depth6_clip0_showcase`
- 当前仓库已不再保留旧展示 bundle、旧候选 bundle 与完整历史 checkpoint 归档。
- 目前只保留当前交付线所需的 bundle、结果和运行证据。

## 当前仓库只认这几类数据

- **当前浏览器选择图片的即时结果**：来自 Web demo 的浏览器本地分片 + E2E approximate SPU 运行，只能用于展示这张图的安全结果和本次 live run 开销；它是隐私输入演示口径，不等同于最终双向隐私生产部署。
- **离线验证集最佳成绩**：来自 `artifacts/web_demo_assets/best_demo_content.json`，只能用于统一对比区，不是当前图片结果。
- **外部基线对比**：当前主对比对象是同数据集明文基线 `MPCViT`。
- **禁止复用的旧口径**：历史 fastpath 8 样本通信、旧 archived SPU profile、旧正式展示模型收益、dated handoff / request 文档中的数字。

详细规则见 `docs/data_source_policy.md`。



## 模型文件说明

本仓库的模型权重文件（`.pth`、`.pt`）已通过 `.gitignore` 排除，不会推送到 GitHub。如需获取模型文件，请联系项目维护者。

### 必需的模型文件

| 模型 | 路径 | 说明 |
|------|------|------|
| 医疗模型 | `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507/` | 医疗主展示 bundle 元数据（权重本地保留，86M） |
| 金融模型 | `artifacts/frozen_bundle_finance_lrd_rank192_20260515/` | 金融主展示 bundle 元数据（LRD rank192，权重本地保留，171M） |
| 基线模型 | `artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth` | 明文基线模型（86M） |

### 数据文件

训练和验证数据（`data/` 目录）同样通过 `.gitignore` 排除。如需获取数据，请联系项目维护者。

## 快速开始

### 运行 Secure Pruning（PredictorLG SPU 内部执行）

```bash
cd /path/to/Transshield_final
source artifacts/server_inference_friendly_pack/final_compare_env.template.sh
bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh \
  --runtime spu \
  --spu-params-mode secret \
  --party-local-share-load \
  --secure-pruning
```

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

`final_compare_env.template.sh` 现在默认就是当前 active delivery line；一般不需要再手工切 bundle。

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

1. `docs/transshield_竞赛作品报告_最终版.docx` ← **最新作品报告**
2. `docs/transshield_master_plan_20260505.md`
3. `docs/delivery_experiment_summary_20260510.md` ← **实验数据汇总**
4. `docs/transshield_innovation.md` ← **创新点说明**
5. `docs/data_source_policy.md`
6. `docs/current_work_status.md`
7. `docs/handoff-next.md`

## 当前展示原则

- 页面顶部和交互区只显示**当前浏览器选择图片**触发的 E2E 隐私推理即时结果。
- 准确率、AUC、最佳轮次、外部基线差值只放在统一对比区。
- 通信量默认只显示**本次 E2E SPU live run**，不再使用固定历史字节数。
- 附录只保留必要工程细节，不再保留旧 fastpath / archived profile 数字。
- 双向隐私 secret 交付线优先使用 `run_e2e_secure_secret_isolated_eval.sh` 的 `secret_blockwise_stage + depth6 + clip0` guarded path；不要再把 `depth8+` 或 `clip3` 当作默认 secret 主线。
- 历史展示资产已经从当前仓库移除，避免旧口径再次回流。

## 说明

- 当前仓库已经删除一批与主线冲突的旧摘要文档，避免旧数字继续回流到前端或答辩材料。
- 如果后续需要补新的比较数据，必须先在 `docs/data_source_policy.md` 中确认口径是否允许。
