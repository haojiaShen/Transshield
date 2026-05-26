# Transshield Final Repository Guidelines

## 仓库定位

这里的 `Transshield_final/` 是用户的最终作品仓库，也是默认主工作目录。

凡是涉及以下内容，默认都应在本仓完成：

- 最终结果整理
- 外部模型对比
- secure pipeline / sidecar / replay / compare
- 演示、答辩、展示材料
- 最终代码结构与资产组织

除非用户明确要求，否则不要把 `DynamicViT_exp_square/` 当作默认主仓。

## 与其他仓库的关系

- `DynamicViT_exp_square/`：历史训练 / 实验 / 冻结候选来源仓
- `external_baselines/`：外部参照方法仓
- `OpenBumbleBee/`：安全后端依赖与集成仓

本仓负责“最后作品”的组织与表达；其他仓只提供来源证据、参考实现或外部对比。

## 当前最终主线事实

- 当前可运行展示主链路位于 `showcase/` 与 `showcase_api/`。
- 医疗域是唯一 live upload / live run 场景；金融域只保留结果与压力验证展示。
- 当前正式 bundle 默认口径：
  - `artifacts/frozen_bundle_medical_dynamic_mainline`
  - `artifacts/frozen_bundle_finance_boundary_stress`
- 当前正式数字与证据默认以以下目录为准：
  - `results/final/`
  - `results/communication/`
  - `results/fuzzing/`
  - `results/guard_stress/`
- 旧 `web_demo/`、旧 `server_inference_friendly_pack/` 与早期过程型状态文档不再属于当前默认主入口；只在 provenance / 归档追踪时回看。

## 默认事实来源

涉及最终结果、最终说明、最终对比时，优先读取：

1. `/home/yclcg/.codex/memories/transshield_final_handoff_2026-05-23.md`
2. `Transshield_final/docs/README.md`
3. `Transshield_final/docs/evidence/README.md`
4. `Transshield_final/docs/report/README.md`
5. `Transshield_final/README.md`
6. `Transshield_final/README_REPRODUCE.md`
7. `Transshield_final/results/`
8. `Transshield_final/artifacts/`

若这些文件与实验仓中的旧记录有差异，默认以本仓为准；只有在做 provenance 追踪时才回看实验仓。
历史过程型状态文档若已从最终仓清理，不要假定其仍存在；改以 `docs/` 索引、`results/*.meta.md` 和正式报告为准。

## 开工前记忆读取

开始任何 `Transshield_final` 相关的非简单任务前，先读取：

1. `/home/yclcg/.codex/memories/transshield_final_handoff_2026-05-23.md`
2. `docs/README.md`
3. `docs/evidence/README.md`
4. `docs/report/README.md`
5. `README.md`
6. `README_REPRODUCE.md`

读取后再判断当前应从哪个阶段继续，不要重复已经完成的排障或验证。

## 对比任务要求

当用户要求“和外部模型对比”时：

- 主表、主叙事、结论写在 `Transshield_final/`
- `MPCViT` 作为 same-dataset plaintext external baseline
- `MPCFormer` 作为 secure latency / communication reference
- 不要把外部 baseline 结果写成用户作品结果
- 不要把实验仓的中间 checkpoint 当作最终作品指标，除非是在解释来源链

## 修改原则

- 优先做小而聚焦的改动
- 优先修改 `docs/`、`artifacts/`、`tools/`、集成脚本和最终流程相关文件
- 若确需改训练或模型代码，要明确说明这是在最终作品仓修改，而不是实验仓
- 修改后要说明：
  - 改了哪些文件
  - 这些文件为什么属于最终作品仓
  - 是否会影响最终展示口径

## 服务器命令原则

- 不要在本地直接执行 `/data/wyb/...` 服务器路径命令；需要服务器运行时，只给用户可复制命令和判定标准。
- **例外**：如果用户在当前对话中明确授权“直接同步到服务器并直接执行”，则允许 agent 直接通过 SSH / rsync / tmux 等方式操作 `10.204.248.175:9001` 上的 `/data/wyb/Transshield_final`，并在最终回复中如实说明执行内容与结果。
- 如果服务器命令后续还要复用路径或 run 名，必须使用同一个 shell / tmux 会话内可持续存在的变量：
  - `export REPO_ROOT=/data/wyb/Transshield_final`
  - `export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"`
  - `cd "$REPO_ROOT"`
- 每个训练 run 用 `run_xxx=...` 保存名称，再用 `export RUN_NAME="$run_xxx"` 传给 runner。
- 后续日志提取统一用 `LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"`，不要在多处重复硬编码完整日志路径。
- 不要用一次性的 `env VAR=... bash ...`、`bash -c` 或子 shell setup 承载后续还要复用的变量。
- 给日志提取命令时优先按明确 `epoch/step/run` 窗口精确 grep，不要默认给大范围 `tail -n 220` 这类粗抓取。

## 历史排障与 Provenance 原则

- 当前最终仓已不再保留早期 `TrackA` 过程文档；若用户明确追问历史漂移归因或服务器侧过程记录，再回看 memory、归档目录或请求用户指定旧证据来源。
- 如果 issue 需要服务器结果但用户还没回贴日志，只能交付服务器命令、精确 grep 字段和判定标准，不能伪造结论。
- 若用户要求在当前最终仓新增历史结论说明，应优先回写到 `docs/evidence/` 或对应 `results/*.meta.md`，而不是恢复大批过程型日志文档。
- 默认不要跑 `full20`，不要扫大范围超参，不要在 attribution 证据出来前直接改变训练语义。
- 不要修改 `results/final/demo_content_summary_final.json` 或正式展示口径，除非用户明确要求更新正式展示数据。

## 不要默认做的事

- 不要默认回到 `DynamicViT_exp_square/` 做任务
- 不要默认引用实验仓的中间输出作为最终结果
- 不要默认重开长时间训练
- 不要把“外部 baseline 更强”误写成“用户作品失败”；要区分系统目标与比较维度
