# 历史 best 复现漂移审计（2026-04-21）

## 审计范围

本次只审计“历史 `tracka_lr3e5_timm_best_20260414` 为什么在当前 vendored source / runner 下没有稳定复现”，不改前端正式口径，不更新正式 best 指标，不跑 `full20`。

审计输入：

- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/source_commands.sh`
- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/source_manifest.json`
- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/args_snapshot.json`
- `artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414/train_stdout.log`
- `training_source_tracka/`
- `training_compat/`
- `scripts/run_tracka_train.sh source`
- `scripts/run_tracka_train.sh compat`

## 结论先行

1. **大部分 recipe / transform / sampler 代码路径已经对齐**，当前没有证据支持“`crop_pct`、train/eval transform、sampler 选择逻辑”本身是主因。
2. **历史 provenance 的环境冻结仍然缺失，但 `2026-04-22` 已确认当前 server runner 本身没有独立版本漂移**：用户回传的 `/data/wyb/conda_envs/transshield/bin/python` 实际版本与仓库 `requirements.txt` 完全一致，因此“server env 自己漂到 `torch 2.x`”这条怀疑可以先排除。
3. **当前 74.236643% 的 source guard-on 结果不是严格官方 recipe 对照**：`NONEMPTY_KEEP_GUARD=true` 会改变训练语义，虽然它更像“防崩溃保险丝”而不是首个分叉根因。
4. `2026-04-21` 新完成的 `predictor_1` 根因诊断已经进一步说明：**`ratio loss` 不是 `predictor_1` 过度 pruning 的首发驱动**，因此本 issue 不再把“`ratio_weight` / ratio loss 语义漂移”列为主怀疑项。
5. `2026-04-22` 用户回传的 `debug80` 对照日志已进一步确认：**`training_source_tracka` vs `training_compat` 默认路径 parity 已闭合**。在 `NONEMPTY_KEEP_GUARD=false`、同 seed 下，两侧 `Transform` / `Sampler_train` / `Use Cosine LR scheduler` / `Max WD` / `Averaged stats` 都重合；`Namespace(...)` 的差异仅剩 source 侧保留 `crop_pct=None`、`weight_decay_end=None` 的运行时补全写法，以及 compat 侧额外打印默认关闭的 `pruning_margin_*` 参数。
6. `2026-04-22` 的 `LOSS_GRAD_ATTRIB=true` 最小归因已经完成：在 strict source、`NONEMPTY_KEEP_GUARD=false`、`epoch3`、run `tracka_source_epoch3_lossgradattrib_guardoff_seed0_20260422_195751` 中，`score_predictor.1.out_proj.weight` 的关键 spike 主要由 `cls_kl` 梯度链路解释；`ratio_loss` 量级仍远小于 total，不是首发驱动。
7. `2026-04-22` 的首个单变量验证 issue 也已完成：`cls_distill_weight=0.0` 的 `resync1` run 是**有效但负向**的结果，虽然成功拿掉了目标参数上的 `cls_kl` attribution，但会把 `predictor_1 empty keep -> zero_active -> predictor_2 non-finite` 风险提前到 `epoch=2 step=34`。因此当前 issue 已可收成“首个单变量已拿到有效负结果”；若后续继续，应另开新 issue 专门决定“下一条最小单变量是什么”，而不是继续补跑 `clsdw0=0.0` 或在本 issue 内扩题。
8. `2026-04-22` 的“下一条最小单变量” issue 也已完成：唯一改动 `cls_distill_weight: 1.0 -> 0.5` 的 strict source / `epoch3` run 没有复现 `clsdw0=0.0` 的 `epoch=2 step=34` 早期失稳，并在 `epoch=2 step=146` 将 `score_predictor.1.out_proj.weight` 的 total `grad_l2` 从 `4.198726e+03` 压到 `2.400378e-01`、将 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `5.022926e-01`；但 terminal accuracy 仍是 `74.24%`。因此这条结果应记作**明确缓解的诊断性正结果**，不是正式修复或新正式成绩。
9. `2026-04-22` post-`clsdw05` 的下一条最小单变量 `cls_distill_weight: 1.0 -> 0.75` 也已完成验证：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true` 下，它没有命中 `step=34` early-fail gate，并在 `epoch=2 step=146` 将 `score_predictor.1.out_proj.weight` 的 total `grad_l2` 从 `4.198726e+03` 压到 `3.748181e+01`、将 `cls_kl grad_l2` 从 `4.041559e+03` 压到 `2.105142e+01`、将 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `4.730873e-01`；但 terminal 仍是 `74.24%`。因此这条结果应记作**明确缓解、但弱于 `clsdw05=0.5` 的诊断性正结果**，不是正式修复或新正式成绩。
10. `2026-04-22` 的 post-`clsdw075` 新阻塞点选择与服务器回贴分析已完成：不再继续扫 `cls_distill_weight`，而是在固定当前更强缓解底座 `cls_distill_weight=0.5` 的前提下，只把 `token_distill_weight: 0.02 -> 0.04` 作为下一条且仅一条最小单变量。服务器结果显示这条配置有效、未早崩，terminal 从 `74.24%` 松动到 `79.01%`，但 `epoch=2 step=146` 的 `score_predictor.1.out_proj.weight` 稳定性窗口明显变差，因此应记作**有效但混合/非 clean 稳定性正结果**，不能推进为正式修复或直接进入 `full20`。
11. `2026-04-23` 的 terminal-稳定性解耦最小单变量也已完成：固定 `cls_distill_weight=0.5` 后只把 `token_distill_weight: 0.02 -> 0.03`，配置有效且未早崩，但 `epoch=2 step=146` 的 total `grad_l2` 从 `2.400378e-01` 增至 `1.165204e+00`，`active_margin_mean` 从 `-3.137406e-02` 变为 `-1.206955e-01`，terminal 仅从 `74.24%` 轻微到 `74.43%`，远未接近 `79.01%`。因此 `0.03` 只能记作**剂量降低后仍未 clean 解耦**：稳定性仍较 control 恶化，terminal 提升基本被削弱，不能进入 `full20` 或写成正式修复。
12. `2026-04-23` 的 post-`tdw003` 新单变量 `ratio_weight: 2.0 -> 3.0` 已完成服务器回贴分析：配置有效且未早崩；在 `cls_distill_weight=0.5 / token_distill_weight=0.04` 底座上，提高 `ratio_weight` 确实把 `predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 抬到 `5.245984e-01`、把 `active_margin_mean` 从 `-1.643516e-01` 拉到 `4.946269e-02`，并降低了 `cls_kl` 分项 `grad_l2`，但目标参数 total `grad_l2` 却从 `4.365869e+00` 放大到 `5.206851e+01`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果应记作**负结果**：它没有完成 clean 解耦，也不能写成修复或推进 `full20`。
13. `2026-04-23` 的 post-`rw3` 最小单变量 `cls_distill_weight: 0.5 -> 0.4` 也已完成服务器回贴分析：配置有效且未早崩；它确实把 `cls_kl grad_l2` 从 `2.991327e+01` 降到 `4.482888e+00`，但 `epoch=2 step=146` 的 total `grad_l2` 从 `4.365869e+00` 升到 `7.772295e+00`，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 降到 `2.575499e-01`，`active_margin_mean` 从 `-1.643516e-01` 恶化到 `-1.041453e+00`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果应记作**有效但负向的非 clean 解耦**：减轻 `cls_kl` 分项压力并不足以保住 `tdw004` terminal-positive 信号，也不能写成修复或推进 `full20`。
14. `2026-04-23` 的 post-`clsdw04` midpoint 单变量 `token_distill_weight: 0.04 -> 0.035` 已完成服务器回贴分析：配置有效且未早崩；它把 `epoch=2 step=146` 的 total `grad_l2` 从 `4.365869e+00` 降到 `2.089920e+00`、`cls_kl grad_l2` 从 `2.991327e+01` 降到 `9.532135e-01`、`token_kl grad_l2` 从 `1.172528e+00` 降到 `2.397546e-03`，同时把 `predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 回升到 `4.878815e-01`、`active_margin_mean` 从 `-1.643516e-01` 回升到 `-8.497284e-02`；但 terminal 从 `79.01%` 回落到 `74.24%`。因此这条结果应记作**稳定性缓解、terminal 丢失的负向 midpoint 结果**：当前仍没有 clean 解耦迹象，不进入 `full20`，也不能写成正式修复。
15. `2026-04-23` 的 post-`tdw0035` 近端 midpoint 选择现已完成服务器回贴分析：固定 `cls_distill_weight=0.5 / ratio_weight=2.0` 后，把 `token_distill_weight` 从 `0.04` 降到唯一近端候选 `0.0375`，配置有效、未早崩，并在 `epoch=2 step=146` 把 `predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 提到 `5.085371e-01`、把 `active_margin_mean` 从 `-1.643516e-01` 拉到 `-2.972232e-03`、把 total `grad_l2` 从 `4.365869e+00` 压到 `1.294038e-01`、把 `cls_kl grad_l2` 从 `2.991327e+01` 压到 `1.165144e-01`、把 `token_kl grad_l2` 从 `1.172528e+00` 压到 `1.486021e-03`；但 terminal 仍从 `79.01%` 直接回落到 `74.24%`。因此这条结果应记作**更强稳定性缓解、但 terminal 完全丢失的负结果**：`0.03 / 0.035 / 0.0375` 三个低于 `0.04` 的点都未保住 `79.01%`，说明当前 token 轴在 `0.04` 附近的信息增益已基本耗尽，应正式收口为 `stop_token_midpoint`，不要继续把近端 token 轴扩成剂量搜索，也不能推进 `full20` 或写成修复。

## 0. `LOSS_GRAD_ATTRIB=true` 进入闸门判断（`2026-04-22`）

| 检查项 | Yes / No | 当前结论 |
|---|---|---|
| server env provenance 是否已闭合 | **Yes** | `/data/wyb/conda_envs/transshield` 当前版本已与 `requirements.txt` 核对一致，不再优先怀疑 server env 独立漂移。 |
| strict source `guard-off` vs `guard-on` 是否已有同口径证据 | **Yes** | `epoch3` 同 seed 对照已闭合；关键窗口 `epoch=2 step=140~146` 两侧同轨，guard 不是首发分叉点。 |
| source vs compat `debug80` parity 是否已有同口径证据 | **Yes** | 两侧 `Transform` / `Sampler_train` / scheduler / WD / `Averaged stats` 均对齐，`training_compat` 默认路径仍可视作 source-compatible。 |
| 当前是否仍可能被 runner / 默认路径差异干扰 | **No** | 现有证据不足以继续把默认 runner / 默认路径差异当主阻塞；后续更应看共享训练路径内部的非 `ratio` 归因。 |

本轮 gate 结论：

- **建议进入** `LOSS_GRAD_ATTRIB=true` 的非 `ratio` 分项归因；
- 但应当**另开 issue** 承接，并把范围限制在共享训练路径内部的最小归因；
- 当前不建议再把 server env、guard 开关或 source/compat 默认路径 parity 当成进入前阻塞；
- 当前也**不建议**在本 issue 内直接修改 `cls_loss / cls_kl / token_kl` 等训练语义。

### 0.1 `LOSS_GRAD_ATTRIB=true` 最小归因结果（`2026-04-22`）

服务器回传的 run：

- `tracka_source_epoch3_lossgradattrib_guardoff_seed0_20260422_195751`
- strict source：`scripts/run_tracka_train.sh source epoch3 0`
- `NONEMPTY_KEEP_GUARD=false`
- `LOSS_GRAD_ATTRIB=true`
- `LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight`
- 完成性：`epoch=0/1/2` 均跑完，并在 `Early stop after epoch 2 due to stop_after_epoch=3` 正常退出；`epoch=2` 的 `Averaged stats` 已复现 `grad_norm: 924.6571 (239.6665)`。

关键窗口 `epoch=2 step=140~146` 的 attribution 结论：

| step | total grad_l2 | cls_loss grad_l2 | ratio_loss grad_l2 | cls_kl grad_l2 | token_kl grad_l2 | 判读 |
|---:|---:|---:|---:|---:|---:|---|
| 140 | `7.982402e+02` | `7.254251e+02` | `2.109620e-01` | `1.483992e+03` | `3.985242e+01` | `cls_kl` 已是最大分项，但与 `cls_loss` 存在抵消。 |
| 141 | `6.918467e+02` | `1.395739e+03` | `2.393740e-01` | `2.039596e+03` | `4.807734e+01` | `cls_kl` 与 `cls_loss` 都大，total 被方向抵消压低。 |
| 142 | `1.021877e+03` | `1.237232e+03` | `2.652559e-01` | `2.215716e+03` | `4.365830e+01` | `cls_kl` 持续最大。 |
| 143 | `2.569316e+03` | `2.235381e+02` | `3.029379e-01` | `2.747210e+03` | `4.568792e+01` | total spike 主要由 `cls_kl` 解释。 |
| 144 | `3.021918e+02` | `2.536529e+03` | `3.398838e-01` | `2.788352e+03` | `5.067101e+01` | `cls_kl` 与 `cls_loss` 高量级抵消，total 暂低。 |
| 145 | `2.182495e+02` | `2.744612e+03` | `3.677114e-01` | `2.913292e+03` | `4.946731e+01` | 同样以高量级抵消为主。 |
| 146 | `4.198726e+03` | `1.036839e+02` | `3.899971e-01` | `4.041559e+03` | `5.404455e+01` | 与既有 `grad_watch absmax=1.057964e+03` 对齐，首发 spike 由 `cls_kl` 主导。 |

结论：

- `ratio_loss` 在关键窗口始终只有 `O(1e-1)` 的 grad_l2，继续排除为 `predictor_1` 首发驱动；
- `token_kl` 约为 `O(4e1~5e1)`，不是本次 total spike 主因；
- `cls_loss` 在部分 step 也有 `O(1e3)` 梯度，但 `epoch=2 step=146` 的首个明确 spike 处只剩 `1.036839e+02`，不是最终 spike 主解释；
- `cls_kl` 从 `step=140` 开始持续处于 `O(1e3)`，并在 `step=146` 达到 `grad_l2=4.041559e+03`、`grad_absmax=1.017747e+03`，与 total `grad_l2=4.198726e+03`、`grad_absmax=1.057964e+03` 同步放大。

下一步建议：

- 不在本 issue 内修改训练语义；
- 首个单变量验证 issue 已经完成，并已拿到 `cls_distill_weight=0.0` 的有效负结果；
- 若继续 TrackA，应另开新 issue 专门选择下一条最小单变量，仍保持 strict source、`NONEMPTY_KEEP_GUARD=false`、`epoch3`，不跑 `full20`，并把结果标注为低精度诊断而非正式成绩。

### 0.2 首个单变量 ablation 命令包与结果（`2026-04-22`）

当前首个且仅一个单变量 ablation 方案定为：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 唯一改动：`cls_distill_weight: 1.0 -> 0.0`
- 其余有效参数保持官方 recipe 不变，尤其保持：
  - `token_distill_weight=0.02`
  - `ratio_weight=2.0`
  - `activation_lr_scale=10.0`
  - `model_ema=false`
  - `stop_after_epoch=3`

选择 `cls_distill_weight=0.0` 而不是先扫 `0.5/0.25` 的原因是：

- 本轮目标是做**首个因果隔离验证**，不是直接找最优修复值；
- `epoch=2 step=146` 的 total spike 已与 `cls_kl` 同步放大，而不是由 `ratio_loss` 主导；
- 先把 `cls_kl` 这条链路单独拿掉，最容易判断：
  - `step=146` total grad 是否明显回落；
  - `predictor_1` keep ratio 是否不再在 `step=140~146` 提前塌缩；
  - 若无改善，再说明仅靠这条链路还不足以解释当前坏轨迹。

为避免直接绕过 TrackA provenance wrapper，`scripts/run_tracka_train.sh` 已新增 `CLS_DISTILL_WEIGHT` / `TOKEN_DISTILL_WEIGHT` 透传；服务器侧仍统一走 `scripts/run_tracka_train.sh source epoch3 0`。

建议用户执行以下命令：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_control=tracka_source_epoch3_lossgradattrib_clsdw1_seed0_20260422
run_ablation=tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export TOKEN_DISTILL_WEIGHT=0.02

export CLS_DISTILL_WEIGHT=1.0
export RUN_NAME="$run_control"
bash scripts/run_tracka_train.sh source epoch3 0

export CLS_DISTILL_WEIGHT=0.0
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0

unset RUN_NAME CLS_DISTILL_WEIGHT TOKEN_DISTILL_WEIGHT
unset LOSS_GRAD_ATTRIB LOSS_GRAD_ATTRIB_PARAM NONEMPTY_KEEP_GUARD

for run in "$run_control" "$run_ablation"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"

  echo "===== $run :: strict-source header ====="
  for key in mode seed train_entry stop_after_epoch nonempty_keep_guard loss_grad_attrib loss_grad_attrib_param cls_distill_weight token_distill_weight
  do
    grep -nF "[tracka-source] ${key}=" "$LOG" || true
  done
  grep -nF 'Namespace(' "$LOG" | head -n 1 || true

  echo "===== $run :: epoch2 step140-146 keep+attrib ====="
  grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true
  for step in 140 141 142 143 144 145 146
  do
    grep -nF "[LossGradAttrib][epoch=2 step=${step}]" "$LOG" || true
  done

  echo "===== $run :: terminal ====="
  grep -nE 'Early stop after epoch|Accuracy of the model on the 524 test images|Max accuracy' "$LOG" || true
done
```

判定标准：

1. `step=146` total grad 是否明显回落
   - 当前控制线参考值为：
     - `grad_l2=4.198726e+03`
     - `grad_absmax=1.057964e+03`
   - 若 ablation 在 `step=146` 同时满足：
     - `total grad_l2 <= 2.1e+03`
     - `total grad_absmax <= 5.3e+02`
     则记为“明显回落”；
   - 若下降幅度 `<20%`，则记为“无明显改善”；
   - 介于两者之间，记为“部分缓解，但不足以下结论”。

2. `predictor_1` keep ratio 是否不再提前塌缩
   - 当前控制线在 `step=146` 的参考值：
     - `final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
   - 若 ablation 在 `step=140~146` 窗口内不再持续单调下滑到 `0.16` 左右，且 `step=146` 同时满足：
     - `final_keep_ratio_mean >= 2.0e-01`
     - `active_margin_mean > -1.4`
     则记为“未提前塌缩”；
   - 若仍落回 `0.16` 左右或更低，并伴随 `active_margin_mean <= -1.5`，则记为“提前塌缩仍在”。

3. 是否仍保持 strict source 控制语义
   - `header + Namespace` 应共同满足：
     - `train_entry=/data/wyb/Transshield_final/training_source_tracka/main.py`
     - `mode=epoch3`
     - `nonempty_keep_guard=false`
     - `loss_grad_attrib=true`
     - `token_distill_weight=0.02`
   - control 与 ablation 之间，除 `RUN_NAME / output_dir / log_dir / cls_distill_weight` 外，不应出现其它有效参数差异；
   - 若出现额外参数漂移，则本轮结果不计入单变量结论。

当前状态：

- 本 issue 尚未收到新的 control / ablation 服务器日志；
- 因此这里只交付**首个单变量命令包与判定标准**，不宣称“修复已完成”。

### 0.3 首轮回贴结果与当前阻塞（`2026-04-22`）

用户随后补回 corrected grep 后，当前可把这轮结果正式收成：

1. 这轮 `clsdw0` run **不是有效单变量 ablation**
   - `tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422` 的 `Namespace(...)` 里仍然是：
     - `cls_distill_weight=1.0`
     - `token_distill_weight=0.02`
   - 这说明当前所谓 ablation run 的**有效配置并没有切到 `cls_distill_weight=0.0`**
2. 与之对应，control / ablation 两边的关键窗口完全同轨：
   - `predictor_1_keep_diag` 在 `epoch=2 step=140~146` 的 7 组数值逐项相同；
   - `step=146 final_keep_ratio_mean` 都是 `1.548948e-01`
   - `step=146 active_margin_mean` 都是 `-1.722985e+00`
   - `LossGradAttrib` 的 `total / cls_loss / ratio_loss / cls_kl / token_kl` 也逐项相同；
   - terminal 指标都仍是 `74.24%`
3. 因此当前能正式下的结论不是“`cls_distill_weight=0.0` 无效”，而是：
   - **这轮 run 没有形成有效的 `cls_distill_weight: 1.0 -> 0.0` 对照**
   - 所以它**不能**用于判断 `cls_kl / cls_distill_weight` 这条链路是否真的无效

当前定位到的直接问题有两个：

- 服务器这次实际执行到的有效参数仍是 `cls_distill_weight=1.0`；
- 旧版 `train_stdout.log` 里也不会保留 wrapper 的 `[tracka-source] key=value` header，因为此前只有 python 子进程输出被 `tee` 到日志。

本地现已补上 runner 日志修正：

- `scripts/_tracka_training_common.sh`
- `scripts/run_tracka_train.sh`

修正后，后续新的 `train_stdout.log` 会同时保留：

- wrapper header
- python `Namespace(...)`
- 训练期 `LossGradAttrib` / `NaNDebug`

所以下一步不建议继续引用这次 `clsdw0` run 做结论，而是：

1. 先把本地修正过的 runner 同步到服务器；
2. 只重跑 **ablation** 一条最小验证；
3. 再和当前已有效的 `clsdw1` control 对照。

建议用户执行以下命令：

```bash
rsync -avP -e "ssh -p 9001" \
  /home/yclcg/Transshield_final/scripts/_tracka_training_common.sh \
  /home/yclcg/Transshield_final/scripts/run_tracka_train.sh \
  wyb@10.204.244.1:/data/wyb/Transshield_final/scripts/

export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_ablation=tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422_resync1

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export CLS_DISTILL_WEIGHT=0.0
export TOKEN_DISTILL_WEIGHT=0.02
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0

LOG="$TRAIN_RUN_ROOT/$run_ablation/train_stdout.log"
echo "===== $run_ablation :: header ====="
grep -nF '[tracka-source] cls_distill_weight=' "$LOG" || true
grep -nF '[tracka-source] token_distill_weight=' "$LOG" || true
grep -nF 'Namespace(' "$LOG" | head -n 1 || true

echo "===== $run_ablation :: epoch2 step140-146 keep ====="
grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true

echo "===== $run_ablation :: epoch2 step146 attrib ====="
grep -nF '[LossGradAttrib][epoch=2 step=146]' "$LOG" || true

echo "===== $run_ablation :: terminal ====="
grep -nE 'Early stop after epoch|Accuracy of the model on the 524 test images|Max accuracy' "$LOG" || true
```

判定标准更新为：

1. 先看有效配置是否真的切到：
   - wrapper header 要出现 `cls_distill_weight=0.0`
   - `Namespace(...)` 里也要出现 `cls_distill_weight=0.0`
   - 两者任一不是 `0.0`，本轮仍按“无效对照”处理
2. 只有在第 1 条满足后，才继续比较：
   - `step=146 total grad_l2 / grad_absmax`
   - `step=146 predictor_1 final_keep_ratio_mean / active_margin_mean`
   - terminal `74.24%` 是否松动

### 0.4 `resync1` 新阻塞：forward non-finite（`2026-04-22`）

用户随后回贴的最新服务器结果，不再是旧 `clsdw0` 那条“无效对照”日志，而是新的 traceback：

- 运行入口：
  - `/data/wyb/Transshield_final/training_source_tracka/main.py`
- 报错位置：
  - `training_source_tracka/models/dyvit.py`
  - `_check_finite('predictor_2_pred_score', pred_score)`
- 异常类型：
  - `RuntimeError: Non-finite tensor detected in VisionTransformerDiffPruning: predictor_2_pred_score`

用户随后补回 `resync1` 的 header、`Namespace(...)`、`step=0` 局部窗口与 traceback 后，当前可把结论更新为：

1. 这条 `resync1` 已经是**有效单变量配置**
   - wrapper header 明确为：
     - `cls_distill_weight=0.0`
     - `token_distill_weight=0.02`
     - `nonempty_keep_guard=false`
     - `loss_grad_attrib=true`
   - `Namespace(...)` 也明确为：
     - `cls_distill_weight=0.0`
     - `token_distill_weight=0.02`
2. `epoch=0 step=0` 的 attribution 也证明这次 ablation 确实生效：
   - `[LossGradAttrib] component=cls_kl weight=0.000000e+00`
   - `scaled_loss=0.000000e+00`
   - `grad_l2=0.000000e+00`
   - 即：在目标参数 `score_predictor.1.out_proj.weight` 上，`cls_kl` 这条链路在 step0 已被真正拿掉
3. 但这条有效 ablation run **仍然没有给出缓解证据**
   - 用户后续贴回的真正 crash 邻域已经显示：
     - 崩溃点在 `epoch=2 step=34`
     - `predictor_1_keep_diag` 已出现：
       - `raw_empty=1`
       - `final_empty=1`
       - `raw_le1=2`
       - `final_le1=2`
       - `final_keep_ratio_mean=1.946639e-01`
     - 紧接着 `PredictorLG` 打印：
       - `zero_active_policy_samples=1`
       - `global_x isfinite=False`
       - `post_agg isfinite=False`
       - `out_conv_out isfinite=False`
     - 最后在 `predictor_2_pred_score` 处触发 `_check_finite`
   - 因而当前这条 `cls_distill_weight=0.0` 单变量尝试，已经可以更精确地定性为：
     - **有效，但会在 `epoch=2 step=34` 提前触发 zero-active → predictor_2 non-finite 的失稳链**

同时，这次也把两个窗口区分清楚了：

- `epoch=0 step=0` 的 `predictor_2_pred_score` 日志只是**首次出现**，且仍是 finite；
- 真正有判定价值的是基于 traceback 行号回贴出来的 crash 邻域；
- 当前 crash 邻域已足以说明：
  - 不是 `predictor_2_pred_score` 自己无缘无故先炸；
  - 而是 `predictor_1` 先出现 empty keep / zero-active，再把 `PredictorLG global_x` 链路推成 non-finite。

所以当前最小结论已经足够完整，不再需要本 issue 内追加新的服务器补抓命令：

1. `cls_distill_weight=0.0` 已经是**有效单变量 ablation**；
2. 它不是 clean comparator，而是明确的**负结果**；
3. 负结果的具体形式是：
   - 在 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` 下，
   - run 会在 `epoch=2 step=34` 提前出现 `predictor_1 empty keep`
   - 随后触发 `zero_active_policy_samples=1`
   - 再把 `predictor_2` 路径推成 non-finite

因此，本 issue 当前最合理的收口是：

- 不继续重复 `cls_distill_weight=0.0`
- 不把它写成“修复完成”
- 当前 issue 自身已经没有缺失证据；
- 如果后续继续，应另开新的最小单变量 issue，由新 issue 决定下一条最小单变量，而不是继续在本 issue 内扩展第二个 loss 项或复跑 `clsdw0=0.0`。

### 0.5 下一条最小单变量：`cls_distill_weight=0.5`（`2026-04-22`）

用户随后在新 issue 中，基于已有证据只选择了一条且仅一条后续最小验证：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 唯一改动：`cls_distill_weight: 1.0 -> 0.5`
- run：
  - `tracka_source_epoch3_lossgradattrib_clsdw1_control_seed0_20260422_next1`
  - `tracka_source_epoch3_lossgradattrib_clsdw05_seed0_20260422_next1`

当前回贴的服务器日志已经足以把这条 issue 收口：

1. 这次 `clsdw05` 是**有效单变量配置**
   - header 与 `Namespace(...)` 同时显示：
     - control：`cls_distill_weight=1.0`
     - ablation：`cls_distill_weight=0.5`
   - `epoch=0 step=0` 的 `LossGradAttrib` 进一步确认目标链路确实减半：
     - `scaled_loss: 4.530853e-02 -> 2.265427e-02`
     - `grad_l2: 2.592915e+01 -> 1.296458e+01`

2. 它没有复现 `clsdw0=0.0` 的 `step=34` 提前失稳链
   - 当前用户回贴的 `early-fail gate` 对 `clsdw05` 没有命中：
     - `raw_empty / final_empty > 0`
     - `zero_active_policy_samples > 0`
     - `RuntimeError: Non-finite tensor`
   - 因此当前没有证据表明 `clsdw05` 会像 `clsdw0=0.0` 一样在 `epoch=2 step=34` 提前崩溃。

3. 在真正关键窗口 `epoch=2 step=146`，它给出**强缓解**
   - control：
     - `predictor_1 final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
     - `total grad_l2=4.198726e+03`
     - `total grad_absmax=1.057964e+03`
     - `cls_kl grad_l2=4.041559e+03`
     - `cls_kl grad_absmax=1.017747e+03`
   - `clsdw05`：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - `total grad_l2=2.400378e-01`
     - `total grad_absmax=4.624692e-02`
     - `cls_kl grad_l2=1.123101e-01`
     - `cls_kl grad_absmax=2.186546e-02`
   - 因而按本 issue 预设标准，这条 `clsdw05` 应记为：
     - `step=146` 的 `cls_kl` 主导 spike 被显著压低；
     - `predictor_1` 不再在关键窗口提前塌缩。

4. 但它还不是正式修复
   - 两边 terminal 仍然一致：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 说明当前结果仍停留在 `epoch3` 诊断窗口内，还不能写成“正式精度恢复”。

因此，这条新 issue 的正式结论应收成：

- 已成功选出并验证“下一条且仅一条最小单变量”：`cls_distill_weight=0.5`
- 该变量在 strict source / `epoch3` 下给出了**明确缓解的诊断性正结果**
- 但当前不能把它写成正式修复、更不能写成新正式成绩
- 这一 issue 的目标已经完成；若后续还要继续推进，应另开新的单变量 issue，而不是在本 issue 内继续扩题

### 0.6 post-`clsdw05` 下一条最小单变量结果：`cls_distill_weight=0.75`（`2026-04-22`）

在 `clsdw05=0.5` 已确认是明确缓解的诊断性正结果后，本轮新 issue 只验证了一条且仅一条下一变量：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 唯一改动：`cls_distill_weight: 1.0 -> 0.75`
- run：
  - `tracka_source_epoch3_lossgradattrib_clsdw1_control_seed0_20260422_postclsdw05_next2`
  - `tracka_source_epoch3_lossgradattrib_clsdw075_seed0_20260422_postclsdw05_next2`

当前用户回贴的服务器日志已经足以把这条 issue 正式收口：

1. 这次 `clsdw075` 是**有效单变量配置**
   - control 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `cls_distill_weight=1.0`
     - `token_distill_weight=0.02`
   - ablation 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `cls_distill_weight=0.75`
     - `token_distill_weight=0.02`
   - 两边都保持：
     - `train_entry=/data/wyb/Transshield_final/training_source_tracka/main.py`
     - `nonempty_keep_guard=false`
     - `loss_grad_attrib=true`
     - `stop_after_epoch=3`
   - 因此这轮对照满足“只改一个变量”的前提。

2. 它没有命中 `step=34` 早崩 gate
   - 用户回贴的 `epoch=2 step=30~36` early-fail grep 里，没有出现：
     - `raw_empty / final_empty > 0`
     - `zero_active_policy_samples > 0`
     - `RuntimeError: Non-finite tensor`
   - 当前唯一命中的 `predictor_1_keep_diag` 仍为：
     - control：`final_keep_ratio_mean=5.012926e-01`
     - `clsdw075`：`final_keep_ratio_mean=5.022135e-01`
   - 因而当前没有证据表明 `0.75` 会走 `clsdw0=0.0` 的 `epoch=2 step=34` 早期失稳链。

3. 在真正关键窗口 `epoch=2 step=146`，它给出**明确缓解**
   - control：
     - `predictor_1 final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
     - `total grad_l2=4.198726e+03`
     - `total grad_absmax=1.057964e+03`
     - `cls_kl grad_l2=4.041559e+03`
     - `cls_kl grad_absmax=1.017747e+03`
   - `clsdw075`：
     - `predictor_1 final_keep_ratio_mean=4.730873e-01`
     - `active_margin_mean=-1.644128e-01`
     - `total grad_l2=3.748181e+01`
     - `total grad_absmax=7.467246e+00`
     - `cls_kl grad_l2=2.105142e+01`
     - `cls_kl grad_absmax=4.196131e+00`
   - 这对应：
     - total `grad_l2` 下降约 `99.11%`
     - `cls_kl grad_l2` 下降约 `99.48%`
     - `predictor_1 final_keep_ratio_mean` 从 `0.1549` 抬到 `0.4731`
   - 因而按本 issue 预设标准，这条 `clsdw075` 应记为：
     - **step146 明确缓解**
     - **predictor_1 未沿 control 轨迹提前塌缩**

4. 但它**弱于** `clsdw05=0.5`
   - 已知 `clsdw05` 在同一 `step=146` 的结果是：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - `total grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
   - 相比之下，`clsdw075` 虽然仍属强缓解，但：
     - total `grad_l2` 更高：`3.748181e+01`
     - `cls_kl grad_l2` 更高：`2.105142e+01`
     - keep ratio 略低：`4.730873e-01`
     - margin 更负：`-1.644128e-01`
   - 因此 `0.75` **没有优于** `0.5`；它更像是在更接近 control 的一侧，重复证明“削弱 `cls_distill_weight` 确实能缓解 spike”，但缓解强度弱于 `0.5`。

5. terminal 仍然不动
   - control 与 `clsdw075` 的 terminal 都仍是：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 因此这条 `0.75` 结果仍然只是 `epoch3` 低精度诊断窗口里的稳定性正信号，不能写成正式精度恢复。

因此，这条 post-`clsdw05` issue 的正式结论应收成：

- `cls_distill_weight=0.75` 是**有效单变量配置**
- 它相对 control 给出了**明确缓解**
- 但它**弱于** `cls_distill_weight=0.5`
- 它也**没有**把 terminal 从 `74.24%` 推开
- 因此当前不建议继续在同一 issue 内扩展更多 `cls_distill_weight` 剂量点；若后续还要继续推进，应另开新的单变量 issue

### 0.7 post-`clsdw075` 新阻塞点选择：固定 `clsdw05` 后上调 `token_distill_weight`（`2026-04-22`）

当前 issue 的目标不是继续证明 `clsdw075`，而是在 `cls_distill_weight=0.75` 已确认“有效但弱于 `0.5`”之后，选择新的阻塞点对应的下一条且仅一条最小单变量。

`clsdw075` 这条旧 issue 已经可以关闭，原因是：

- 它已经满足有效配置检查：header 与 `Namespace(...)` 同时显示 `cls_distill_weight=0.75`；
- 它没有命中 `epoch=2 step=34` 的 early-fail gate；
- 它在 `epoch=2 step=146` 对 `cls_kl` 主导 spike 给出明确缓解；
- 它弱于 `clsdw05=0.5`，且 terminal 仍停在 `74.24%`；
- 因此继续在同一 issue 内追加更多 `cls_distill_weight` 点只会变成剂量搜索，边际信息低，且违反“一条 issue 只做一个最小单变量”的约束。

下一条且仅一条最小单变量选择为：

- 固定当前更强缓解底座：`CLS_DISTILL_WEIGHT=0.5`
- 唯一改动：`TOKEN_DISTILL_WEIGHT=0.02 -> 0.04`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 仍固定 `ratio_weight=2.0`、`activation_lr_scale=10.0`、`model_ema=false`、`base_rate=0.7`

选择理由：

1. 它比继续扫 `cls_distill_weight` 更合理：`0.5` 已经比 `0.75` 更强，而 `0.0` 已是有效负结果；继续补 `0.625/0.25` 这类点只是在同一剂量线上细化，不会回答“为何稳定性缓解后 terminal 仍是 `74.24%`”这个新阻塞。
2. 它比先改 `ratio_weight` 更值得做：既有根因诊断已经排除 `ratio_loss` 是 `predictor_1` 首发驱动；改 `ratio_weight` 会直接改变 pruning target pressure，更容易掩盖症状，而不是验证新的非 `cls_kl` 信息保持链路。
3. `token_distill_weight` 是现有 runner 已暴露的最小参数，不需要改训练语义；`losses.py` 中 token distill 已通过 `_resolve_token_mask()` 走 mask-aware token loss，符合后续 F_mux/F_less 语义保留原则。
4. 从量级上看，原 control 的 `token_kl` 在 `step=146` 不是主 spike；把 `0.02` 小幅翻倍到 `0.04`，是在已稳定的 `clsdw05` 底座上测试“token-level feature alignment 是否不足”，而不是重新引入 `cls_kl` 主导爆炸。

建议用户在服务器同一个 shell / tmux 会话中执行：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

mkdir -p /data/wyb/tmp
export TMPDIR=/data/wyb/tmp
export TMP=/data/wyb/tmp
export TEMP=/data/wyb/tmp

run_control=tracka_source_epoch3_lossgradattrib_clsdw05_tdw002_control_seed0_20260422_next3
run_ablation=tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_seed0_20260422_next3

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export CLS_DISTILL_WEIGHT=0.5

export TOKEN_DISTILL_WEIGHT=0.02
export RUN_NAME="$run_control"
bash scripts/run_tracka_train.sh source epoch3 0

export TOKEN_DISTILL_WEIGHT=0.04
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0
```

训练后提取字段：

```bash
for run in "$run_control" "$run_ablation"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"

  echo "===== $run :: strict-source header ====="
  for key in mode seed train_entry stop_after_epoch nonempty_keep_guard loss_grad_attrib loss_grad_attrib_param cls_distill_weight token_distill_weight
  do
    grep -nF "[tracka-source] ${key}=" "$LOG" || true
  done
  grep -nE '^Namespace\(' "$LOG" | head -n 1 || true

  echo "===== $run :: epoch2 step30-36 early-fail gate ====="
  grep -nE 'epoch=2 step=3[0-6].*(predictor_1_keep_diag|zero_active_policy_samples|RuntimeError|Non-finite tensor|predictor_2_pred_score|isfinite=False)' "$LOG" || true

  echo "===== $run :: epoch2 step140-146 keep+attrib ====="
  grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true
  for step in 140 141 142 143 144 145 146
  do
    grep -nF "[LossGradAttrib][epoch=2 step=${step}]" "$LOG" || true
  done

  echo "===== $run :: terminal ====="
  grep -nE 'Early stop after epoch|Accuracy of the model on the 524 test images|Max accuracy|RuntimeError|Non-finite tensor' "$LOG" || true
done
```

判定标准：

1. 目标参数配置必须真正进入：
   - control：header + `Namespace(...)` 都应显示 `cls_distill_weight=0.5`、`token_distill_weight=0.02`；
   - ablation：header + `Namespace(...)` 都应显示 `cls_distill_weight=0.5`、`token_distill_weight=0.04`；
   - 两边除 `RUN_NAME / output_dir / log_dir / token_distill_weight` 外，不应出现其它有效参数差异。
2. `step=34` early-fail gate 必须不早崩：
   - 若 `epoch=2 step=30~36` 出现 `raw_empty/final_empty > 0`、`zero_active_policy_samples > 0`、`predictor_2_pred_score` 非有限或 `RuntimeError: Non-finite tensor`，则本轮记为**更早失稳**。
3. `epoch=2 step=146` 相比 control 的窗口判读：
   - 若无早崩，且 ablation 的 `total grad_l2` 相对 fixed-`clsdw05` control 下降至少 `20%`，同时 `final_keep_ratio_mean` 没有下降超过 `0.03`，记为**缓解**；
   - 若 `total grad_l2` 与 `final_keep_ratio_mean` 都只在约 `±20% / ±0.03` 内波动，记为**无变化**；
   - 若未早崩但 `total grad_l2` 明显变大或 keep 明显下降，记为**恶化但未早崩**，不推进为正结果。
4. terminal 必须单独记录：
   - 若仍是 `Accuracy ... 74.2%` / `Max accuracy: 74.24%`，说明 `token_distill_weight=0.04` 未撬动当前 `epoch3` terminal 阻塞；
   - 若 terminal 有提升，也只能写成低精度诊断信号，不能写成正式修复或新正式成绩。

用户随后回贴了两条服务器日志：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw002_control_seed0_20260422_next3`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_seed0_20260422_next3`

回贴日志足以把本 issue 的实验结论收口：

1. 新 ablation 确实进入目标参数配置
   - control 的 header + `Namespace(...)` 均显示：
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.02`
   - ablation 的 header + `Namespace(...)` 均显示：
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.04`
   - 两边仍保持 strict source、`NONEMPTY_KEEP_GUARD=false`、`epoch3`、`LOSS_GRAD_ATTRIB=true`、`ratio_weight=2.0`、`activation_lr_scale=10.0`、`model_ema=false`。

2. `step=34` early-fail gate 未命中
   - control 与 ablation 的 `epoch=2 step=30` 均显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`；
   - 回贴的 `step=30~36` grep 没有出现 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限或 `RuntimeError: Non-finite tensor`；
   - 因此 `token_distill_weight=0.04` 没有走 `clsdw0=0.0` 的早崩路径。

3. `epoch=2 step=146` 不是稳定性缓解，而是**恶化但未早崩**
   - fixed-`clsdw05` control：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - total `grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
     - `token_kl grad_l2=2.975943e-04`
   - `token_distill_weight=0.04`：
     - `predictor_1 final_keep_ratio_mean=4.537168e-01`
     - `active_margin_mean=-1.643516e-01`
     - total `grad_l2=4.365869e+00`
     - `cls_kl grad_l2=2.991327e+01`
     - `token_kl grad_l2=1.172528e+00`
   - 对照预设标准，ablation 的 keep ratio 下降约 `0.0486`，超过 `0.03` 容忍线；total `grad_l2` 从 `2.400378e-01` 增到 `4.365869e+00`，约为 control 的 `18.2x`；因此不能记为“缓解”。

4. terminal 发生正向松动，但只能记作低精度诊断信号
   - control terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - `token_distill_weight=0.04` terminal：
     - `Accuracy of the model on the 524 test images: 79.0%`
     - `Max accuracy: 79.01%`
   - 这说明 `token_distill_weight=0.04` 确实撬动了此前停在 `74.24%` 的 `epoch3` terminal 阻塞；但由于关键稳定性窗口同步恶化，它不是 clean 修复，更不能写成正式成绩。

本轮正式结论：

- `token_distill_weight=0.04` 是**有效单变量配置**；
- 它**没有**早崩；
- 它让 `epoch3` terminal 从 `74.24%` 提到 `79.01%`，是有价值的诊断信号；
- 但它在 `epoch=2 step=146` 明显放大 `score_predictor.1.out_proj.weight` 的 total / `cls_kl` / `token_kl` 梯度，并使 `predictor_1` keep 与 margin 变差；
- 因此本轮应记作**有效但混合：terminal 正向、稳定性负向**，不建议直接进入 `full20`，也不建议写成正式修复。

当前阻塞点已经从“`cls_distill_weight` 剂量是否有效”转为：

> 如何解释并解耦 `token_distill_weight=0.04` 带来的 terminal 改善与 `score_predictor.1` 稳定性恶化。

若后续继续，应另开新 issue 单独处理这个“terminal 改善但稳定性变差”的冲突，不在本 issue 内继续扩题。

### 0.8 terminal-稳定性解耦单变量结果：`token_distill_weight=0.03`（`2026-04-23`）

在 `token_distill_weight=0.04` 已给出“terminal 正向、稳定性负向”的 mixed diagnostic 后，新 issue 只选择一条最小解耦验证：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 固定底座：`cls_distill_weight=0.5`
- 唯一改动：`token_distill_weight: 0.02 -> 0.03`
- 不改：`ratio_weight=2.0`、`activation_lr_scale=10.0`、`model_ema=false`、`base_rate=0.7`

选择 `0.03` 的原因是：

- `0.04` 的旧 issue 已可关闭：它已经证明配置有效、未早崩、terminal 松动但稳定性变差；继续复跑只会重复 mixed result。
- 相比直接 `full20`，`0.03` 仍在 `epoch3` 诊断窗口内回答更小的问题：是否存在更温和 token distill 剂量，能保留 terminal 松动但不放大 `score_predictor.1` 稳定性风险。
- 相比改 `ratio_weight`、EMA 或 LR，`0.03` 只改同一 loss family 内的一个标量，不引入新机制。

本轮服务器 run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw002_control_seed0_20260422_next4`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw003_seed0_20260422_next4`

回贴日志足以收口：

1. 配置有效
   - control 的 header + `Namespace(...)` 均显示：
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.02`
   - ablation 的 header + `Namespace(...)` 均显示：
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.03`
   - `epoch=0 step=0` 的 attribution 也确认：
     - control：`token_kl weight=2.000000e-02`
     - ablation：`token_kl weight=3.000000e-02`
     - 两边 `cls_kl weight` 都保持 `5.000000e-01`

2. `step=34` early-fail gate 未命中
   - control 在 `epoch=2 step=30` 显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`；
   - `tdw003` 在 `epoch=2 step=30` 也显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`；
   - 回贴 grep 没有出现 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`；
   - 因此 `tdw003` 没有走 `clsdw0=0.0` 的早崩路径。

3. `epoch=2 step=146` 仍不是 clean 稳定性缓解
   - fixed-`clsdw05` control：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - total `grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
     - `token_kl grad_l2=2.975943e-04`
   - `token_distill_weight=0.03`：
     - `predictor_1 final_keep_ratio_mean=4.792178e-01`
     - `active_margin_mean=-1.206955e-01`
     - total `grad_l2=1.165204e+00`
     - `cls_kl grad_l2=1.053805e+00`
     - `token_kl grad_l2=1.148510e-04`
   - keep ratio 下降约 `0.0231`，仍在预设 `0.03` 容忍线内；
   - 但 total `grad_l2` 约为 control 的 `4.85x`，且 `active_margin_mean` 比 control 更负约 `0.0893`，超过预设 `0.05` 容忍线；
   - 因此按照判定标准，`tdw003` 应记为**稳定性恶化但弱于 `tdw004`**，不是“无变化 / 近似保稳”。

4. terminal 只有极弱松动，远未接近 `tdw004`
   - control terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - `tdw003` terminal：
     - `Accuracy of the model on the 524 test images: 74.4%`
     - `Max accuracy: 74.43%`
   - 它相对 control 只提升约 `0.19pp`，不接近 `tdw004` 的 `79.01%`。

本轮正式结论：

- `token_distill_weight=0.03` 是**有效单变量配置**；
- 它**没有**早崩；
- 它相比 `0.04` 降低了稳定性伤害：`step=146` total `grad_l2` 从 `4.365869e+00` 降到 `1.165204e+00`，`final_keep_ratio_mean` 从 `4.537168e-01` 回升到 `4.792178e-01`；
- 但它仍明显差于 fixed-`clsdw05` control，且 terminal 只到 `74.43%`；
- 因此这条解耦尝试没有拿到“terminal 提升 + 稳定性无变化/缓解”的 clean 迹象；
- 当前不建议直接进入 `full20`，也不能把 `74.43%` 或 `79.01%` 写成正式成绩。

当前阻塞点更新为：

> 更温和的 token distill 剂量会削弱 `0.04` 的稳定性恶化，但也几乎带走 terminal 提升；目前仍没有证明 token distill 轴上存在 clean 解耦点。

若后续继续，应另开新 issue，单独决定下一条最小单变量；不要在本 issue 内继续扫 `token_distill_weight`，也不要直接进入 `full20`。

### 0.9 post-`tdw003` 新单变量结果：`ratio_weight=3.0`（`2026-04-23`）

在 `token_distill_weight=0.03` 已确认没有 clean 解耦后，新 issue 只验证了一条且仅一条下一变量：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=3.0`
- 唯一改动：`ratio_weight: 2.0 -> 3.0`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_next5`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw3_seed0_20260423_next5`

用户回贴日志已足够收口：

1. 配置有效
   - control 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=2.0`
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.04`
   - ablation 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=3.0`
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.04`
   - 除 `RUN_NAME / output_dir / log_dir / ratio_weight` 外，没有新的有效参数漂移。

2. `step=34` early-fail gate 未命中
   - control 与 ablation 的 `epoch=2 step=30` 都显示：
     - `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
   - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`；
   - 因此 `ratio_weight=3.0` 没有走 `clsdw0=0.0` 的早崩路径。

3. `step=146` 不是稳定性缓解，而是**split 代理改善但 total grad 明显恶化**
   - paired control（`rw2`）：
     - `predictor_1 final_keep_ratio_mean=4.537168e-01`
     - `active_margin_mean=-1.643516e-01`
     - total `grad_l2=4.365869e+00`
     - `cls_kl grad_l2=2.991327e+01`
   - ablation（`rw3`）：
     - `predictor_1 final_keep_ratio_mean=5.245984e-01`
     - `active_margin_mean=4.946269e-02`
     - total `grad_l2=5.206851e+01`
     - `cls_kl grad_l2=1.911049e+01`
   - keep ratio 回升约 `+0.0709`，`active_margin_mean` 回升约 `+0.2138`，说明额外 keep-pressure 的确把 pruning 分布往“更保守 keep”方向推回；
   - 但 total `grad_l2` 却升到 control 的约 `11.93x`，远超预设容忍线，因此按本 issue 判定标准仍应记为**稳定性恶化**，不能因为 keep/margin 回升就写成缓解。
   - 从分项看，`cls_kl grad_l2` 虽降到 control 的约 `63.9%`，`ratio_loss grad_l2` 也进一步变小，但 `cls_loss grad_l2` 与 `token_kl grad_l2` 反而更高；因此更像是梯度构成与抵消关系被改坏，而不是拿到了 clean 稳定性修复。

4. terminal 提升被直接带走
   - paired control terminal：
     - `Accuracy of the model on the 524 test images: 79.0%`
     - `Max accuracy: 79.01%`
   - ablation terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 这说明 `ratio_weight=3.0` 没有保住 `tdw004` 的 terminal-positive 信号，而是直接把 terminal 拉回了原先停滞点。

本轮正式结论：

- `ratio_weight=3.0` 是**有效单变量配置**；
- 它**没有**早崩；
- 它在 keep/margin 代理上出现正向变化，但目标参数 total `grad_l2` 明显恶化；
- 同时 terminal 从 `79.01%` 回落到 `74.24%`；
- 因此这条结果应记为**负结果**：没有完成 terminal 提升与稳定性修复的 clean 解耦，也不能写成 mixed 修复、更不能推进 `full20`。

### 0.10 post-`rw3` 新单变量结果：`cls_distill_weight=0.4`（`2026-04-23`）

在 `ratio_weight=3.0` 已确认不能保住 `tdw004` 的 terminal-positive 信号后，新 issue 只验证了一条且仅一条下一变量：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.4 / token_distill_weight=0.04 / ratio_weight=2.0`
- 唯一改动：`cls_distill_weight: 0.5 -> 0.4`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_next5`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw04_tdw004_rw2_seed0_20260423_next6`

用户回贴日志已足够收口：

1. 配置有效
   - control 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=2.0`
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.04`
   - ablation 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=2.0`
     - `cls_distill_weight=0.4`
     - `token_distill_weight=0.04`
   - `epoch=0 step=0` 的 `LossGradAttrib` 进一步确认：
     - ablation `cls_kl weight=4.000000e-01`
     - ablation `token_kl weight=4.000000e-02`
   - 除 `RUN_NAME / output_dir / log_dir / cls_distill_weight` 外，没有新的有效参数漂移。

2. `step=34` early-fail gate 未命中
   - ablation 的 `epoch=2 step=30` 显示：
     - `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
     - `final_keep_ratio_mean=5.168403e-01`
     - `active_margin_mean=6.718890e-02`
   - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`；
   - 因此 `cls_distill_weight=0.4` 没有走 `clsdw0=0.0` 的 `epoch=2 step=34` 早崩路径。

3. `step=146` 不是稳定性缓解，而是**`cls_kl` 分项下降但 total / keep / margin 同时恶化**
   - paired control（`clsdw05_tdw004_rw2`）：
     - `predictor_1 final_keep_ratio_mean=4.537168e-01`
     - `active_margin_mean=-1.643516e-01`
     - total `grad_l2=4.365869e+00`
     - `cls_kl grad_l2=2.991327e+01`
     - `token_kl grad_l2=1.172528e+00`
   - ablation（`clsdw04_tdw004_rw2`）：
     - `predictor_1 final_keep_ratio_mean=2.575499e-01`
     - `active_margin_mean=-1.041453e+00`
     - total `grad_l2=7.772295e+00`
     - `cls_kl grad_l2=4.482888e+00`
     - `token_kl grad_l2=3.589258e+00`
   - `cls_kl grad_l2` 明显降低，说明轻降 `cls_distill_weight` 确实打到了目标分项；
   - 但 total `grad_l2` 升到 control 的约 `1.78x`，`final_keep_ratio_mean` 下降约 `0.1962`，`active_margin_mean` 进一步负移约 `0.8771`，均超过预设容忍线；
   - 因此按本 issue 判定标准，`cls_distill_weight=0.4` 应记为**稳定性恶化**，不能因为 `cls_kl` 分项下降就写成 clean 缓解。

4. terminal 提升也被带走
   - paired control terminal：
     - `Accuracy of the model on the 524 test images: 79.0%`
     - `Max accuracy: 79.01%`
   - ablation terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 这说明 `cls_distill_weight=0.4` 没有保住 `tdw004` 的 terminal-positive 信号，而是回落到原先停滞点。

本轮正式结论：

- `cls_distill_weight=0.4` 是**有效单变量配置**；
- 它**没有**早崩；
- 它确实降低了目标参数上的 `cls_kl grad_l2`；
- 但它同步恶化 total `grad_l2`、`predictor_1` keep / margin 与 terminal；
- 因此这条结果应记为**有效但负向的非 clean 解耦**：不是修复，不是可推进的 mixed 结果，也不能进入 `full20`。

### 0.11 本地交接维护记录（`2026-04-23`）

本轮已完成 `ratio_weight=3.0` 回贴分析与文档同步：

- 已将该负结果同步写入 `docs/history_best_repro_drift_audit_2026-04-21.md`、`docs/current_work_status.md`、`docs/handoff-next.md`；
- `scripts/run_tracka_train.sh` 与 `scripts/_tracka_training_common.sh` 已保留 `RATIO_WEIGHT` 透传与 header 打印，确保后续同类 issue 可继续复用；
- 当前 issue 的目标已经完成：`tdw003` 后选择的唯一下一变量已被实际验证并定性；
- 若后续继续推进，应另开新的单变量 issue，不要在当前 issue 内继续扩题，也不要把本轮 `79.01%` / `74.24%` 写成正式成绩或正式修复。

本轮也已完成 post-`rw3` 的 `cls_distill_weight=0.4` 回贴分析与文档同步：

- 已将该负结果同步写入 `docs/history_best_repro_drift_audit_2026-04-21.md`、`docs/current_work_status.md`、`docs/handoff-next.md`；
- 当前 issue 的目标已经完成：`rw3` 后选择的唯一下一变量已被实际验证并定性；
- 当前仍没有拿到“terminal 提升 + 稳定性无变化 / 缓解”的 clean 解耦迹象；
- 不要把本轮 `79.01%` / `74.24%` 写成正式成绩或正式修复，也不要直接推进 `full20`。

因此当时的最小下一步曾更新为（后续实际结果见 0.12）：

- fixed control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- single ablation：`token_distill_weight=0.035`
- 目标：检查唯一真正撬动 terminal 的 token 轴，是否存在比 `0.04` 更轻、但仍能保住 `79.01%` 附近 terminal-positive 信号的 midpoint
- 不再继续扫 `cls_distill_weight` 或 `ratio_weight`
- 对应的新 issue 结论已同步回写到本文、`docs/current_work_status.md` 和 `docs/handoff-next.md`

### 0.12 post-`clsdw04` midpoint 结果：`token_distill_weight=0.035`（`2026-04-23`）

在 `cls_distill_weight=0.4` 已确认不能保住 `tdw004` 的 terminal-positive 信号后，新 issue 只验证了一条且仅一条 midpoint 单变量：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.5 / token_distill_weight=0.035 / ratio_weight=2.0`
- 唯一改动：`token_distill_weight: 0.04 -> 0.035`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_tdw0035`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw0035_rw2_seed0_20260423_next7`

用户回贴日志已足够收口：

1. 配置有效
   - control 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=2.0`
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.04`
   - ablation 的 wrapper header 与 `Namespace(...)` 同时显示：
     - `ratio_weight=2.0`
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.035`
   - 除 `RUN_NAME / output_dir / log_dir / token_distill_weight` 外，没有新的有效参数漂移。

2. `step=34` early-fail gate 未命中
   - control 与 ablation 的 `epoch=2 step=30/40` 都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`；
   - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False`、`RuntimeError`、`Non-finite` 或 `skip optimizer step`；
   - 因此 `token_distill_weight=0.035` 没有走 `clsdw0=0.0` 的 `epoch=2 step=34` 早崩路径。

3. `step=146` 相比 `tdw004` control 是**稳定性明显缓解**
   - paired control（`tdw004`）：
     - `predictor_1 final_keep_ratio_mean=4.537168e-01`
     - `active_margin_mean=-1.643516e-01`
     - total `grad_l2=4.365869e+00`
     - `cls_kl grad_l2=2.991327e+01`
     - `token_kl grad_l2=1.172528e+00`
   - ablation（`tdw0035`）：
     - `predictor_1 final_keep_ratio_mean=4.878815e-01`
     - `active_margin_mean=-8.497284e-02`
     - total `grad_l2=2.089920e+00`
     - `cls_kl grad_l2=9.532135e-01`
     - `token_kl grad_l2=2.397546e-03`
   - total `grad_l2` 下降约 `52.1%`，`cls_kl grad_l2` 下降约 `96.8%`，`token_kl grad_l2` 下降约 `99.8%`；
   - `final_keep_ratio_mean` 回升约 `+0.0342`，`active_margin_mean` 回升约 `+0.0794`，说明 `0.035` 确实把 `0.04` 的坏窗口往更温和区间拉回。

4. terminal-positive 信号没有保住
   - paired control terminal：
     - `Accuracy of the model on the 524 test images: 79.0%`
     - `Max accuracy: 79.01%`
   - ablation terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 因此 `0.035` 虽然缓解稳定性，但没有保住 `tdw004` 的 terminal-positive 信号；terminal 直接回到 `74.24%`，甚至没有达到此前 `tdw003` 的 `74.43%` 弱松动。

本轮正式结论：

- `token_distill_weight=0.035` 是**有效单变量配置**；
- 它**没有**早崩；
- 它在 `epoch=2 step=146` 明显缓解 `tdw004` 的 total / `cls_kl` / `token_kl` 梯度放大，并改善 `predictor_1` keep / margin；
- 但它没有保住 `tdw004` 的 `79.01%` terminal-positive 信号，而是回落到 `74.24%`；
- 因此这条结果应记为**稳定性缓解、terminal 丢失的负向 midpoint 结果**：不是 clean 解耦，不是修复，也不能进入 `full20`。

当前 TrackA 结论更新为：

- `token_distill_weight` 仍是唯一实际撬动过 terminal 的轴，但 `0.035` 已证明简单 midpoint 会把 terminal-positive 信号带走；
- 当前仍没有拿到“terminal 提升 + 稳定性无变化 / 缓解”的 clean 解耦迹象；
- 本 issue 目标已经完成：`tdw0035` 已被实际验证并定性；
- 不要把 `74.24%`、`74.43%` 或 `79.01%` 写成正式成绩或正式修复，也不要直接推进 `full20`；
- 若后续继续，应另开新单变量 issue 决定是否还要在 `0.04` 附近做更贴近的 token midpoint；不要在本 issue 内继续扩题。

### 0.13 post-`tdw0035` 近端 midpoint 结果：`token_distill_weight=0.0375`（`2026-04-23`）

本轮 issue 的目标是验证：在 `0.035` 已确认“稳定性缓解、terminal 丢失”之后，`0.04` 附近是否还值得继续做一条且仅一条更贴近 `0.04` 的 token midpoint。

本地已执行的命令只包括读取与回写：

- `sed -n` / `rg -n` 读取：
  - `docs/history_best_repro_drift_audit_2026-04-21.md`
  - `docs/current_work_status.md`
  - `docs/handoff-next.md`
  - `docs/tracka_predictor1_root_cause_2026-04-21.md`
  - `scripts/run_tracka_train.sh`
  - `training_source_tracka/main.py`
  - `training_source_tracka/engine.py`
  - `training_source_tracka/losses.py`
- `apply_patch` 回写三份权威文档
- **未执行任何** `/data/wyb/...` 服务器训练、grep、评估或其它命令

用户回贴的 `tracka_source_epoch3_lossgradattrib_clsdw05_tdw00375_rw2_seed0_20260423_next8` 日志已足够收口：

1. 配置有效，且核心权重已进入目标值
   - 从回贴的 `LossGradAttrib` 可直接读到：
     - `component=cls_kl weight=5.000000e-01`
     - `component=token_kl weight=3.750000e-02`
     - `component=ratio_loss weight=6.666667e-01`
   - 由此可**推断** ablation 至少已正确进入：
     - `cls_distill_weight=0.5`
     - `token_distill_weight=0.0375`
     - `ratio_weight=2.0`
   - 同时日志含有 `Early stop after epoch 2 due to stop_after_epoch=3`，与既定 `epoch3` 口径一致。

2. `step=34` early-fail gate 未命中
   - 该 run 已正常到达 `epoch=2 step=145/146`，并完整给出 `epoch=2` 的 `Averaged stats` 与后续 eval；
   - 因此它显然没有走 `clsdw0=0.0` 的 `epoch=2 step=34` 早崩路径。

3. 相比 `tdw004` control，`step=146` 是**更强的稳定性缓解**
   - paired control（`token_distill_weight=0.04`）参考值：
     - `predictor_1 final_keep_ratio_mean=4.537168e-01`
     - `active_margin_mean=-1.643516e-01`
     - total `grad_l2=4.365869e+00`
     - `cls_kl grad_l2=2.991327e+01`
     - `token_kl grad_l2=1.172528e+00`
   - ablation（`token_distill_weight=0.0375`）：
     - `predictor_1 final_keep_ratio_mean=5.085371e-01`
     - `active_margin_mean=-2.972232e-03`
     - total `grad_l2=1.294038e-01`
     - `cls_loss grad_l2=2.109074e-02`
     - `ratio_loss grad_l2=3.086725e-03`
     - `cls_kl grad_l2=1.165144e-01`
     - `token_kl grad_l2=1.486021e-03`
   - 与 `tdw004` control 相比：
     - total `grad_l2` 再降约 `97.0%`
     - `cls_kl grad_l2` 再降约 `99.6%`
     - `token_kl grad_l2` 再降约 `99.9%`
     - `predictor_1 final_keep_ratio_mean` 再升约 `+0.0548`
     - `active_margin_mean` 从明显负值几乎回到 `0`
   - 这说明 `0.0375` 不只是“比 `0.04` 更稳”，而是已经把坏窗口拉回到**比 `0.035` 更强、且接近甚至优于 `clsdw05/tdw002` control** 的稳定区间。

4. terminal-positive 信号仍然完全丢失
   - paired control terminal：
     - `Max accuracy: 79.01%`
   - ablation terminal：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 因此 `0.0375` 虽然更接近 `0.04`，但 terminal 仍没有保住，直接回到 `74.24%`。

本轮最终结论：

- `token_distill_weight=0.0375` 是**有效单变量配置**；
- 它**没有**早崩；
- 它在 `epoch=2 step=146` 给出了**比 `0.035` 更强的稳定性缓解**；
- 但它依然**完全没有保住** `tdw004` 的 `79.01%` terminal-positive 信号，terminal 仍是 `74.24%`；
- 因此当前应正式从 `continue_near_004` 收口为 **`stop_token_midpoint`**。

为什么现在应停，而不是继续把 token 轴扩成剂量搜索：

1. 现在已经有三个 `<0.04` 的点：
   - `0.03 -> 74.43%`
   - `0.035 -> 74.24%`
   - `0.0375 -> 74.24%`
2. 这三个点都没有保住 `79.01%`，但稳定性却随着剂量降低持续改善；
3. 尤其 `0.0375` 已经证明：**即使把稳定性窗口拉回得比 `0.035` 更强，terminal 也仍然不会回来**；
4. 因此当前更合理的判断是：`tdw004` 的 terminal-positive 信号在现 recipe 下更像是**贴在 `0.04` 的阈值型现象**，而不是可以通过继续做近端 token midpoint 平滑保住的连续剂量曲线。

所以这条线的正式结论应是：

- `tdw0035` 原 issue 早已可以关闭；
- post-`tdw0035` 的唯一近端 midpoint `0.0375` 也已完成；
- 近端 token midpoint 线到此信息增益已不足，应**停在这里**；
- 当前不要再继续做 `0.038 / 0.0385 / 0.039` 这类近端 token 剂量搜索；
- 当前也不能把 `74.24%` 或 `79.01%` 写成正式成绩或修复；
- 当前不推进 `full20`。

## 1. 已对齐项

### 1.1 历史 provenance 内部一致

`source_commands.sh`、`source_manifest.json`、`args_snapshot.json`、`train_stdout.log` 对以下关键项是一致的：

- `model = deit-s`
- `epochs = 20`
- `seed = 0`
- `lr = 3e-5`
- `warmup_steps = 50`
- `model_ema = false`
- `use_amp = false`
- `base_rate = 0.7`
- `ratio_weight = 2.0`
- `cls_distill_weight = 1.0`
- `token_distill_weight = 0.02`
- `activation_lr_scale = 10.0`
- `use_square_gelu = true`
- `square_activation_mode = learnable_quadratic_gelu_init`
- `use_mask_pruning = true`
- `patch_embed_bias_init_mode = zero`
- `freeze_patch_embed_proj = true`
- `pretrained_fix_step = 0`

### 1.2 `crop_pct` / eval transform 没发现隐藏漂移

- 历史 `train_stdout.log` 中 `Namespace(...)` 里 `crop_pct=None`。
- 但 `datasets.py` 在 eval 分支里会把 `None` 解析为 `224 / 256 = 0.875`。
- 历史 `args_snapshot.json` 最终记录的正是 `crop_pct = 0.875`。
- 历史与当前 source log 打印出的 eval transform 都是：
  - `Resize(size=256, interpolation=bicubic)`
  - `CenterCrop(size=(224, 224))`
  - `ToTensor()`
  - `Normalize(...)`

结论：**“log 里是 `None`、快照里是 `0.875`”是代码正常归一化，不是隐藏漂移。**

### 1.3 train transform 没发现隐藏漂移

`DynamicViT_exp_square/datasets.py`、`training_source_tracka/datasets.py`、`training_compat/datasets.py` 三者一致；历史与当前 source log 打印出的 train transform 也一致：

- `RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)`
- `RandomHorizontalFlip(p=0.5)`
- `RandAugment`
- `ToTensor()`
- `Normalize(...)`
- `RandomErasing`

### 1.4 sampler / distributed 代码路径已对齐

- `DynamicViT_exp_square/samplers.py` 与 `training_source_tracka/samplers.py` 完全一致。
- `train_sampler_mode` 默认都是 `distributed`。
- 历史与当前 source log 都显示：
  - `Not using distributed mode`
  - `distributed=False`
  - `world_size=1`
  - `Sampler_train = torch.utils.data.distributed.DistributedSampler`

这说明当前 source runner 仍走 **单进程 + `DistributedSampler(num_replicas=1)`** 的历史路径。

### 1.5 `weight_decay_end` 的“None vs 0.05”也不是隐藏漂移

- 历史 `Namespace(...)` 里 `weight_decay_end=None`。
- 但 `main.py` 会在 scheduler 创建前执行：
  - `if args.weight_decay_end is None: args.weight_decay_end = args.weight_decay`
- 因此历史 `args_snapshot.json` 里的最终有效值是 `0.05`。
- 当前 source runner 代码同样保留这段逻辑。

结论：**`weight_decay_end` 也是“显式值缺省、有效值一致”的情况。**

### 1.6 数据口径基本对齐

本地历史 provenance 数据路径 `/home/yclcg/DynamicViT_exp_square/data/pneumoniamnist_imagefolder_subset` 统计结果为：

- train：`0=1214`, `1=3494`
- val：`0=135`, `1=389`

这与仓内 handoff / status 文档记录一致。当前 server `/data/wyb/...` 路径在本工作区不可直接核验，但仓内文档已记录其计数一致。

### 1.7 `args_snapshot.json` 记录的是“有效参数视图”，不是原始 CLI 文本

- 当前需要区分：
  - `train_stdout.log` 首行 `Namespace(...)`
  - `source_manifest.json`
  - `args_snapshot.json`
- 后两者来自训练运行后的 `checkpoint['args']` 视图，因此会包含：
  - 运行中被代码补全后的默认值
  - 解析后的有效参数
- 这正是为什么历史 provenance 中会出现：
  - `crop_pct`：日志首行是 `None`，但 `args_snapshot.json` 最终是 `0.875`
  - `weight_decay_end`：日志首行是 `None`，但 `args_snapshot.json` 最终是 `0.05`

结论：**`args_snapshot.json` 更接近“有效配置快照”，但不能把它逐字等同于原始 CLI 命令文本。**

### 1.8 `source_commands.sh` 与 translated rerun `commands.sh` 不是同一层级

- 原始 provenance 命令入口是：
  - `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/source_commands.sh`
- 当前最终仓里给人复跑用的映射命令是：
  - `artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414/commands.sh`

两者含义不同：

- `source_commands.sh` = 历史 best 当时是怎么跑出来的
- `commands.sh` = 当前最终仓为了复跑/映射而整理的命令入口

结论：**后续如果要讨论 provenance，必须优先引用 `source_commands.sh`，不要把 `commands.sh` 误写成原始历史命令。**

### 1.9 根仓训练入口不是这条 provenance 的复现入口

当前这条历史 best 复现链，应该优先使用：

- `training_source_tracka/main.py`
- 或在明确默认值后使用 `training_compat/main.py`

不应把根仓训练入口直接当作 provenance 复现入口：

- 根仓 `main.py`
- 根仓 `datasets.py`

因为根仓训练栈已经继续向最终作品/推理友好方向演化，语义边界与这条历史 provenance 复现任务不同。

## 2. 可疑漂移项

### 2.1 运行环境 provenance 没被冻结，但 server 现状已闭合

历史训练命令写的是：

- `"/home/yclcg/miniconda3/envs/transshield/bin/python" /home/yclcg/DynamicViT_exp_square/main.py ...`

但 `source_manifest.json` **没有**记录：

- `torch` 版本
- `torchvision` 版本
- `CUDA` 版本
- `cudnn` 版本
- `numpy/Pillow` 版本

当前本地同一路径 Python 实测为：

- `python = 3.9.25`
- `torch = 2.7.1+cu118`
- `torchvision = 0.22.1+cu118`
- `timm = 0.3.2`
- `numpy = 1.26.4`

而当前仓库 `requirements.txt` 固定的是：

- `torch==1.13.1+cu117`
- `torchvision==0.14.1+cu117`
- `timm==0.3.2`
- `numpy==1.26.4`

但 `2026-04-22` 用户已在服务器核对：

- `python = 3.9.25`
- `torch = 1.13.1+cu117`
- `torchvision = 0.14.1+cu117`
- `timm = 0.3.2`
- `numpy = 1.26.4`
- `cuda = 11.7`
- `cudnn = 8500`

并且与当前仓库 `requirements.txt` 完全一致。

结论：

- 历史 provenance 仍有“当年环境未冻结”的结构性缺口；
- **本地** `/home/yclcg/miniconda3/envs/transshield` 已偏离 repo pin，因此它不能继续拿来替代 server provenance；
- 运行环境 provenance 的 **server 现状核验** 已闭合；
- 当前 `74.236643%` 的 guard-on 结果**不能**再优先归因到“server env 独立漂移”。

因此，运行环境不再是当前 server 复现失败的首发优先怀疑项；在 `2026-04-22` 当前证据下，strict source vs guard-on 与 source vs compat parity 这两条 runner 级 closure 也都已经补齐。

### 2.2 当前 guard-on 失败 run 不是严格官方 recipe

当前 source 失败结论引用的是：

- `NONEMPTY_KEEP_GUARD=true`
- `scripts/run_tracka_train.sh source epoch5 0`

但历史 best provenance 并没有这个开关；`training_source_tracka/models/dyvit.py` 新增的 guard 会在训练时：

- 对 `PredictorLG` 的 zero-active 分母做 `clamp_min(1.0)`；
- 对空 keep 决策做 `single_token fallback`。

这会改变训练语义。

判断：

- **它不是首个分叉点最强嫌疑**，因为仓内已有结论：guard-off 也会出 zero-active / non-finite；
- 但 **它确实让当前 74.236643% 结果不再是“严格官方配方”**；
- 所以 guard-on run 只能用来做“防崩溃诊断”，不能直接拿来断言“官方配方必然失败”。

### 2.3 `training_source_tracka` 并非字节级原样 vendored

`diff -rq` 表明，相对历史 `DynamicViT_exp_square`，`training_source_tracka` 真正改过的源码文件只有：

- `main.py`
- `losses.py`
- `models/dyvit.py`

具体分类如下：

| 文件 | 改动 | 是否影响训练语义 |
| --- | --- | --- |
| `training_source_tracka/main.py` | 新增 `--nonempty_keep_guard`、`--stop_after_epoch`，并把 guard 传入模型 | `stop_after_epoch` 只影响早停诊断；`nonempty_keep_guard` **开启时会影响语义** |
| `training_source_tracka/losses.py` | 新增 `ratio_stage_i` debug 打印 | **否**，仅诊断 |
| `training_source_tracka/models/dyvit.py` | 新增 keep 诊断、zero-active clamp、single-token fallback | debug 打印 **否**；guard **开启时是语义改动** |

结论：`training_source_tracka` 可以视为“**接近历史 source 的带诊断快照**”，但不能说是 100% 原样。

### 2.4 `ratio loss` 已可排除为当前主漂移解释

根据最新归档：

- `docs/tracka_predictor1_root_cause_2026-04-21.md`

当前已经可以确认：

- `ratio loss` 不是把 `predictor_1` 首先推向过度 pruning 的首发驱动；
- 在 `epoch=2 step=140~146` 的关键窗口里，`ratio_stage_0/1/2` 的 `pos_ratio_mean` 都明显低于 target；
- 这意味着 `ratio loss` 的方向更像是**反向拉回 keep**，而不是继续把 `predictor_1` 推向 `drop`。

对本 issue 的含义是：

- **不能**再把“`ratio` 相关实现漂移”当作当前最优先解释；
- **也不能**仅凭这一结论，就在本 issue 内直接修改 `cls_loss / cls_kl / token_kl` 等非 `ratio` 训练语义；
- 如果后续继续推进，应另开“非 `ratio` 路径最小单变量验证”任务，而不是在本 issue 内扩大改动面。

### 2.5 source runner 仍有少量“隐式默认 + 运行时补全”特征

即使当前已能确认大部分有效值对齐，source runner 仍不是“所有关键值都在 shell 命令里显式钉死”的风格。

这意味着：

- 单看命令字符串，不足以完整判断是否与历史有效配置同口径；
- 需要结合：
  - `train_stdout.log`
  - `args_snapshot.json`
  - 代码中的运行时补全逻辑

结论：**后续若做最小单变量对照，应优先对齐“有效参数视图”，不要只对齐表面命令文本。**

## 3. `training_compat` 与 `training_source_tracka` 的差异分类

相对 `training_source_tracka`，`training_compat` 改过的文件是：

- `main.py`
- `engine.py`
- `losses.py`
- `models/dyvit.py`

分类如下：

| 文件 | 差异 | 默认是否影响语义 |
| --- | --- | --- |
| `training_compat/main.py` | 新增 `pruning_margin_*` 参数；当 `pruning_margin_weight>0` 时开启 `collect_pruning_diagnostics` 并给 criterion 传 epoch | **默认否**，因为默认 `pruning_margin_weight=0.0` |
| `training_compat/engine.py` | loss slot 从 6 扩成 7，用于记录 `pruning_margin_loss` | **默认否** |
| `training_compat/losses.py` | 增加 pruning margin regularization 路径 | **默认否**，只有 margin weight > 0 才生效 |
| `training_compat/models/dyvit.py` | 保留 `nonempty_keep_guard`，但去掉 source 版的 keep_diag 打印 | **默认否**，guard 默认 `false`；去日志不改语义 |

结论：

- `training_compat` 与 `training_source_tracka` 的**默认路径**已经比较接近；
- 真正会改训练语义的，主要是：
  - `nonempty_keep_guard=true`
  - `pruning_margin_weight>0`
- 其它大部分差异是“诊断或记录能力增强”，不是默认行为漂移。

## 4. Yes / No Checklist

### 4.1 “是否已对齐” checklist

- **Yes**：历史 provenance 里的主配方参数彼此一致。
- **Yes**：`crop_pct` 的有效值仍是 `0.875`，没有 hidden drift。
- **Yes**：train transform 代码与日志都对齐。
- **Yes**：eval transform 代码与日志都对齐。
- **Yes**：sampler 选择逻辑与 `distributed=False, world_size=1` 的运行方式对齐。
- **Yes**：`weight_decay_end` 的有效值仍是 `0.05`。
- **Yes**：`epoch1/epoch5` 旧版“压缩总 epochs”漂移已经通过 `stop_after_epoch` 修正。
- **Yes**：当前 server runner 的 package versions 已与 repo pin 核对一致。
- **No**：历史运行时 package versions 没有被冻结，`torch/torchvision` 版本对齐无法证明。
- **No**：当前 74.236643% 的 guard-on run 不是严格官方 recipe。
- **No**：`training_source_tracka` 不是字节级原样 vendored，只能算“带诊断补丁的 source 快照”。
- **Yes**：最新根因文档已确认 `ratio loss` 不是 `predictor_1` 首发驱动，因此这一项可以从“主怀疑项”里移出。

### 4.2 “是否仍可能解释复现失败” checklist

- **No**：`crop_pct` / eval transform 隐藏漂移。
- **No**：train transform 隐藏漂移。
- **No**：sampler 选择逻辑本身换路。
- **No**：`weight_decay_end` 缺省值导致 scheduler 变化。
- **No**：当前 server runner 存在独立 package / CUDA 漂移。
- **Yes**：current guard-on run 的 `nonempty_keep_guard` 语义差异。
- **No**：`ratio loss` / `ratio_weight` 本身是当前最优先的首发漂移解释。
- **Conditional**：`training_compat` 的 margin regularization；默认不影响，但一旦开权重就会影响语义。

## 5. 优先级排序

### P0：运行环境 provenance 已完成现状核验

原因：当前 server runner 与 repo pin 已闭合，因此这一项不再作为首发阻塞，但需要在文档里保留这个结论，避免后续再次误把本地 env 当作 server provenance。

### P1：把 guard-on 与 strict source recipe 拆开

原因：当前 guard-on run 只能说明“guard 能防崩”，不能直接说明“官方 recipe 也会走到同一坏轨迹”。

### P2：`training_source_tracka` vs `training_compat` 的默认路径 parity smoke 已完成

原因：用户回传的同机同 seed `debug80` 日志已经表明，两侧关键 header / transform / sampler / scheduler / stats 全部重合，`training_compat` 默认路径仍可继续视作 source-compatible。

### P3：把“非 `ratio` 路径最小单变量验证”留到新 issue

原因：

- 最新根因文档已经说明 `ratio loss` 不是首发驱动；
- 但本 issue 的范围是**复现漂移审计**，不是继续拆 `cls_loss / cls_kl / token_kl` 的训练根因；
- 当前 issue 内的 runner / parity 前置条件已经收敛，因此如果后续还要推进，更适合另开 issue 做共享训练路径内部的最小非 `ratio` 归因，而不是继续在本 issue 里扩大改动面。

### P4：其余显式参数项只做收尾证明，不再当主因

包括：

- `crop_pct`
- `weight_decay_end`
- `train/eval transform`
- `train_sampler_mode`

这些项目前已经有足够代码和日志证据，不值得优先继续投入。

## 6. 每个高概率漂移点的最小验证命令

### 6.0 服务器命令的持久路径 / 变量写法

后续 TrackA 服务器命令块不要再写成一次性的 `env VAR=... bash ...`。这类写法只影响当前进程，训练进程结束后变量上下文就丢失，后续 grep / 对照容易重新手写错路径。

统一使用同一个 shell / tmux 会话内可持续复用的 setup：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"
```

每个 run 再单独设置 run 名，并显式 `export RUN_NAME`：

```bash
run_example=tracka_source_epoch3_example_seed0

export RUN_NAME="$run_example"
export NONEMPTY_KEEP_GUARD=false
bash scripts/run_tracka_train.sh source epoch3 0

LOG="$TRAIN_RUN_ROOT/$run_example/train_stdout.log"
grep -nE '^Namespace\(' "$LOG" | head -n 1 || true
```

如果同一个命令包后面还要继续 grep / 比对，不要把上述 setup 包进 `bash -c`、heredoc 子 shell 或 `env VAR=... command`。

### 6.1 漂移点 A：server 运行环境现状核验（已完成）

目标：归档已完成的 server provenance 核验结果，后续不再把它当首发阻塞。

已核对结果：

- `python = 3.9.25`
- `torch = 1.13.1+cu117`
- `torchvision = 0.14.1+cu117`
- `timm = 0.3.2`
- `numpy = 1.26.4`
- `cuda = 11.7`
- `cudnn = 8500`

结论：

- 当前 server runner 与 `requirements.txt` 完全一致；
- 当前 `74.236643%` 的 guard-on 结果**不能**再归因到“server Python env 与 repo pin 不一致”；
- 但历史 best 当时环境没有被冻结，这个 provenance 结构性缺口仍需在叙事中明确标注。

如未来 server env 再发生变化，可复用以下命令重新核对：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
cd "$REPO_ROOT"

/data/wyb/conda_envs/transshield/bin/python - <<'PY'
import sys, torch, torchvision, timm, numpy
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("numpy", numpy.__version__)
PY

grep -nE '^(torch|torchvision|timm|numpy)==' requirements.txt
```

### 6.2 漂移点 B：`nonempty_keep_guard` 改变了当前失败 run 的语义

目标：把 strict source recipe 与 guard-on 诊断 recipe 分开。

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_guardoff=tracka_source_epoch3_sched20_guardoff_seed0
run_guardon=tracka_source_epoch3_sched20_guardon_seed0

export RUN_NAME="$run_guardoff"
export NONEMPTY_KEEP_GUARD=false
bash scripts/run_tracka_train.sh source epoch3 0

export RUN_NAME="$run_guardon"
export NONEMPTY_KEEP_GUARD=true
bash scripts/run_tracka_train.sh source epoch3 0

unset RUN_NAME NONEMPTY_KEEP_GUARD

for run in "$run_guardoff" "$run_guardon"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"
  echo "===== $run :: header ====="
  grep -nE '^Namespace\(' "$LOG" | head -n 1 || true

  echo "===== $run :: epoch2 step140-146 key-window ====="
  grep -nE "epoch=2 step=14[0-6].*(predictor_[12]_keep_diag|ratio_stage_[012]|grad_watch parameter=score_predictor\\.1\\.out_proj\\.weight)" "$LOG" || true

  echo "===== $run :: empty-keep edge ====="
  grep -nE "epoch=3 step=[0-5].*(predictor_[12]_empty_keep_samples|predictor_[12]_keep_diag)" "$LOG" || true

  echo "===== $run :: terminal metrics ====="
  grep -nE "Accuracy of the model on the 524 test images|Max accuracy|Early stop after epoch" "$LOG" || true
done
```

判定标准：

- 如果 guard-off 在关键窗口前就明显分叉，说明 root cause 在 guard 之前；
- 如果 guard-on 只是把崩溃改成多数类塌缩，guard 就是“症状缓冲层”，不是首因。

当前 issue 已回传 `tracka_source_epoch3_sched20_guardon_seed0_gpu3` 与 `tracka_source_epoch3_sched20_guardoff_seed0_gpu3` 两侧摘录，现可确认：

- 两侧 `Namespace(...)` 唯一有效差异是：
  - `nonempty_keep_guard=True` vs `False`
- 在 `epoch=2 step=140~146` 的关键窗口，以下字段按数值归一化后完全一致：
  - `predictor_1/2_keep_diag`
  - `ratio_stage_0/1/2`
  - `grad_watch parameter=score_predictor.1.out_proj.weight`
- 两侧都在 `step=145~146` 表现为：
  - `predictor_2 raw_le1=1`、`final_le1=1`
  - `predictor_1 final_keep_ratio_mean` 下滑到 `1.548948e-01`
  - `ratio_stage_1 pos_ratio_mean=6.042730e-02`
  - `ratio_stage_2 pos_ratio_mean=2.822066e-02`
  - `grad_watch parameter=score_predictor.1.out_proj.weight = ±1.057964e+03`
- 两侧三次 eval 都是：
  - `Accuracy of the model on the 524 test images: 74.2%`
  - `Max accuracy: 74.24%`
- 两侧 `epoch=3 step=0~5` grep 都为空，这同样是预期行为，因为 run 会在：
  - `Early stop after epoch 2 due to stop_after_epoch=3`
  - 后结束，并不会真正进入 `epoch=3` 训练 step

因此，strict source `guard-off` vs `guard-on` 的 `epoch3` closure 已经完成，结论是：

- guard 不是 `epoch=2 step=146` 之前的首发分叉点；
- 在这条 `epoch3` 关键窗口线上，guard 尚未真正改变轨迹；
- guard 更像后续防崩的缓冲层，而不是当前短窗口失败的首因。

### 6.3 漂移点 C：`training_compat` 默认路径是否真的与 source 快照一致

目标：验证默认 `training_compat` 是否仍可视为 source-compatible。

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_source=tracka_source_debug80_seed0
run_compat=tracka_compat_debug80_seed0

export RUN_NAME="$run_source"
export NONEMPTY_KEEP_GUARD=false
bash scripts/run_tracka_train.sh source debug80 0

export RUN_NAME="$run_compat"
export NONEMPTY_KEEP_GUARD=false
export MODEL_EMA=false
export ACTIVATION_LR_SCALE=10.0
export CROP_PCT=0.875
export WEIGHT_DECAY_END=0.05
export TRAIN_SAMPLER_MODE=distributed
bash scripts/run_tracka_train.sh compat debug80 0

unset RUN_NAME NONEMPTY_KEEP_GUARD MODEL_EMA ACTIVATION_LR_SCALE CROP_PCT WEIGHT_DECAY_END TRAIN_SAMPLER_MODE

for run in "$run_source" "$run_compat"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"
  echo "===== $run :: header ====="
  grep -nE '^Namespace\(' "$LOG" | head -n 1 || true

  echo "===== $run :: transforms+scheduler ====="
  grep -nE "^Transform =|^Sampler_train =|^Use Cosine LR scheduler|^Max WD =" "$LOG" || true

  echo "===== $run :: epoch0 grad smoke ====="
  grep -nE "epoch=0 step=(0|10|20|30|40).*(grad_watch parameter=score_predictor\\.1\\.out_proj\\.weight)" "$LOG" || true

  echo "===== $run :: epoch0 summary ====="
  grep -nE "Debug max steps reached at epoch=0 step=79|Averaged stats: lr:|Accuracy of the model on the 524 test images|Max accuracy" "$LOG" | head -n 4 || true
done
```

判定标准：

- 如果 80-step 内 header / transform / scheduler / stats 都基本重合，则 `training_compat` 默认路径可继续视作“默认语义兼容”；
- 如果这里就分叉，再回头追 `training_compat` 自身的默认逻辑。

`2026-04-22` 用户已回传两侧日志，当前可直接下结论：

- `Transform` 两侧完全一致：
  - train 都是 `RandomResizedCropAndInterpolation` + `RandomHorizontalFlip` + `RandAugment` + `ToTensor` + `Normalize` + `RandomErasing`
  - eval 都是 `Resize(256)` + `CenterCrop(224)` + `ToTensor` + `Normalize`
- `Sampler_train` 两侧都打印：
  - `torch.utils.data.distributed.DistributedSampler`
- scheduler / WD 两侧完全一致：
  - 都打印 `Use Cosine LR scheduler`
  - 都打印 `Max WD = 0.0500000, Min WD = 0.0500000`
- `Averaged stats: lr:` 两侧一共 20 条，数值逐项重合；差别只在 compat 去掉了 source 版若干额外 debug 打印，因此日志行号更短，但指标本身没有分叉。
- `Namespace(...)` 的差异也符合预期：
  - source 侧保留 `crop_pct=None`、`weight_decay_end=None`
  - compat 侧显式固定 `crop_pct=0.875`、`weight_decay_end=0.05`
  - compat 侧多出 `pruning_margin_*`，但当前是 `pruning_margin_weight=0.0`，属于默认关闭的兼容钩子

因此，当前正式结论是：

- `training_compat` 默认路径可继续视作 source-compatible；
- 当前问题**不是** `training_compat` 默认 runner 自己先分叉；
- source / compat 这条 runner parity closure 现已完成。

## 7. 本轮审计建议

建议下一步顺序更新为：

1. **6.2 已完成**：strict source `guard-off` vs `guard-on` 已闭合；
2. **6.3 已完成**：`training_compat` 默认路径仍可作为 source-compatible runner；
3. **首轮归因已完成**：`LOSS_GRAD_ATTRIB=true` 指向 `cls_kl` 主导 `predictor_1` 的关键 spike；
4. **首个单变量已完成**：`cls_distill_weight=0.0` 已拿到有效负结果，不再继续补跑同一路径；
5. **如后续继续推进**：另开新 issue 专门决定下一条最小单变量，不在本 issue 内继续扩题。

在这三步完成前，不建议：

- 继续把 server env 独立漂移当作首发主怀疑项；
- 继续把 `74.236643%` guard-on 结果表述成“官方 recipe 已证伪”；
- 继续把 `crop_pct` / transform / sampler 作为主怀疑项；
- 直接进入 `full20`。

## 8. 审计结论

当前可以先排除的主因：

- `crop_pct`
- train/eval transform
- `weight_decay_end`
- sampler 选择代码路径
- 旧版 `epoch5=5 epochs` 调度压缩问题
- 当前 server runner 的独立 package / CUDA 漂移
- `ratio loss` 作为 `predictor_1` 首发驱动

当前仍不能排除、且优先级最高的漂移：

- **current guard-on run 与官方 recipe 的语义差异**
- **历史 best 当时环境未冻结带来的 provenance 结构性缺口**
- **共享训练路径内部仍未归因的后续梯度 / 优化动力学问题**

因此，本轮审计的核心判断是：

> 当前“复现失败”已经**不再像是** `source` / `compat` 默认 runner 分叉，  
> 也不再像是当前 server env 独立漂移，  
> 更像是共享训练路径内部仍未归因的后续训练动力学问题；  
> 而 `crop_pct` / transform / sampler 这类显式训练配置在当前证据下也已基本可排除。
