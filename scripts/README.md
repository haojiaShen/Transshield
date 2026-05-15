# scripts 目录说明

本目录用于给比赛展示提供“脚本入口导航”。

当前仓库的正式可运行脚本集合位于：

- `artifacts/server_inference_friendly_pack/`

如果你要**直接全量替换服务器仓库并清空旧运行产物**，不要再用黑名单式整仓同步。
现在先本地构建一份 clean deploy repo，再 `rsync --delete` 到服务器：

```bash
cd /home/yclcg/Transshield_final
export PYTHON_BIN=/home/yclcg/miniconda3/envs/transshield/bin/python
bash scripts/build_clean_server_repo.sh /home/yclcg/Transshield_final_server_clean
```

说明文档见：

- `docs/server_clean_deploy_20260505.md`

推荐入口：

- 完整对比链：`artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`
- 快速 smoke：`artifacts/server_inference_friendly_pack/run_full_final_comparison_smoke.sh`
- 环境模板：`artifacts/server_inference_friendly_pack/final_compare_env.template.sh`

保留本目录的原因：

- 便于答辩与交接时快速说明“从哪里开始运行”
- 不改动现有稳定脚本路径，避免破坏复现链

当前额外保留两类服务器辅助脚本：

- 传关键环境变量给子进程时，统一使用 `env VAR=... bash ...` 或先 `export`，不要单独写一行 `VAR=...` 后再起 `bash/python`
- 根目录兼容入口如果与 `artifacts/server_inference_friendly_pack/` 同名，默认只当 wrapper；权威实现以后者为准

## 权威传输命令

首次执行前，建议先临时加上 `--dry-run` 确认文件列表。

### 本地 → 服务器（黑名单）

这组旧命令仍可用于“同步本地完整仓的增量代码”，但它**不是**当前推荐的服务器 clean replace 方案。
如果你要清掉服务器旧产物、旧 bundle 和旧结果目录，改用上面的 clean deploy 方案。

```bash
rsync -avP -e "ssh -p 9001" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '*.pth' \
  --exclude '*.pth.tar' \
  --exclude 'artifacts/train_runs/' \
  --exclude 'artifacts/server_runs/' \
  --exclude 'artifacts/server_pipeline_run/' \
  --exclude 'artifacts/server_profile_reports/' \
  --exclude 'artifacts/spu_build_reports/' \
  /home/yclcg/Transshield_final/ \
  wyb@10.204.248.175:/data/wyb/Transshield_final/
```

### 服务器 → 本地（白名单，只回传日志 / 报告）

```bash
rsync -avP -e "ssh -p 9001" --prune-empty-dirs \
  --include='/train_runs/' \
  --include='/train_runs/**/' \
  --include='/server_pipeline_run/' \
  --include='/server_pipeline_run/**/' \
  --include='/server_profile_reports/' \
  --include='/server_profile_reports/**/' \
  --include='*.log' \
  --include='*.txt' \
  --include='*.json' \
  --include='*.md' \
  --include='*.csv' \
  --exclude='*' \
  wyb@10.204.248.175:/data/wyb/Transshield_final/artifacts/ \
  /home/yclcg/Transshield_final/artifacts/
```

注意：

- 回传时必须是 `.../artifacts/ -> .../artifacts/`，不要再把服务器 `artifacts/` 同步到本地仓库根目录。
- 第二条命令默认不会拉回 `*.pth`、`*.pt`、`tb/`、`__pycache__/` 之类大文件和缓存。
- 仓内其余旧版同步命令已全部废弃；后续如果看到别的 rsync/scp 旧写法，不再沿用，统一以本节两条命令为准。

- `run_tracka_train.sh`
  - 统一收口 `source|compat` 两条 TrackA 训练入口
  - `source` 走 `training_source_tracka/main.py`
  - `compat` 走 `training_compat/main.py`
  - 根仓 `main.py` 仍保留为 final-repo live 训练 / ablation 入口，不参与 TrackA `source vs compat` provenance 对照
  - 统一约束服务器路径只使用 `/data/wyb/*`
  - 自动设置 `TMPDIR=/data/wyb/tmp`

- `run_tracka_spu.sh`
  - `followup`：跑当前 delivery bundle 的 SPU follow-up / replay / compare
  - 默认 bundle：`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/`
  - 默认 temp root：`/data/wyb/bazel_clean/tmp`，避免服务器侧 `/tmp`
  - 运行后自动汇总 `[fastpath-profile]`，主通信展示使用 Python distributed RPC/cloudpickle bytes
  - `dual-profile`：一次性跑默认快分支与 diagnostic communication 分支
- `run_spu_patch_build_probe.sh`
  - 给 `SPU` 源码重打 `ColocatedIo::sync()` patch，并把 Bazel 重编日志与当前 compile blockers 摘要化
  - 当前已按“预检 / wrapper+patch / build / 错误提取”分段整理，方便后续继续改
