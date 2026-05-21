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

## 默认事实来源

涉及最终结果、最终说明、最终对比时，优先读取：

1. `/home/yclcg/.codex/memories/transshield_final_handoff_2026-04-12.md`
2. `Transshield_final/docs/transshield_master_plan_20260505.md`
3. `Transshield_final/docs/data_source_policy.md`
4. `Transshield_final/docs/current_work_status.md`
5. `Transshield_final/docs/handoff-next.md`
6. `Transshield_final/docs/transshield_innovation.md`
7. `Transshield_final/docs/delivery_experiment_summary_20260510.md`
8. `Transshield_final/artifacts/`

若这些文件与实验仓中的旧记录有差异，默认以本仓为准；只有在做 provenance 追踪时才回看实验仓。

## 开工前记忆读取

开始任何 `Transshield_final` 相关的非简单任务前，先读取：

1. `/home/yclcg/.codex/memories/transshield_final_handoff_2026-04-12.md`
2. `docs/transshield_master_plan_20260505.md`
3. `docs/data_source_policy.md`
4. `docs/current_work_status.md`
5. `docs/handoff-next.md`
6. `docs/transshield_innovation.md`

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

## TrackA 排障原则

- TrackA 相关任务优先读取 `docs/current_work_status.md`、`docs/handoff-next.md` 和 `docs/delivery_experiment_summary_20260510.md`。
- 如果 issue 需要服务器结果但用户还没回贴日志，只能交付服务器命令、精确 grep 字段和判定标准，不能伪造结论。
- TrackA issue 只有在结论已回写到 `docs/history_best_repro_drift_audit_2026-04-21.md`、`docs/current_work_status.md`、`docs/handoff-next.md` 后，才算完成。
- 默认不要跑 `full20`，不要扫大范围超参，不要在 attribution 证据出来前直接改变训练语义。
- 不要修改 `artifacts/web_demo_assets/demo_content_summary.json` 或前端正式展示口径，除非用户明确要求更新正式展示数据。

## 不要默认做的事

- 不要默认回到 `DynamicViT_exp_square/` 做任务
- 不要默认引用实验仓的中间输出作为最终结果
- 不要默认重开长时间训练
- 不要把“外部 baseline 更强”误写成“用户作品失败”；要区分系统目标与比较维度
