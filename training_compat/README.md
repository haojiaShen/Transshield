# Transshield Plaintext Training Compatibility Stack

本目录把当前已验证 `tracka_lr3e5_timm` 明文训练链路内置到最终作品仓中。

## Purpose

- 让服务器侧明文稳定性复验从 `Transshield_final/` 发起并执行；
- 不再要求用户手动进入 `DynamicViT_exp_square/` 执行训练命令；
- 保持与 `tracka_lr3e5_timm_best_20260414` 成功候选一致的训练语义。

## Execution Contract

服务器侧默认入口：

```bash
cd /data/wyb/Transshield_final
bash scripts/run_tracka_train.sh compat debug80 1
```

实际运行文件：

- `training_compat/main.py`
- `training_compat/engine.py`
- `training_compat/losses.py`
- `training_compat/datasets.py`
- `training_compat/models/`

当前默认兼容语义说明：

- `nonempty_keep_guard=false`
- 即默认不启用“空 keep 后强制保底单 token”的训练时保护；
- 这样更接近 `tracka_lr3e5_timm_best_20260414` 的原始训练语义；
- 若只是做鲁棒性排障，而不是严格 provenance 复现，可显式传：
  - `NONEMPTY_KEEP_GUARD=true`

短验证运行说明：

- `epoch1` / `epoch5` 模式默认保持：
  - `epochs=20`
  - 原始 20-epoch 的 LR / WD scheduler
- 仅通过：
  - `stop_after_epoch=1` / `stop_after_epoch=5`
  - 提前结束训练
- 这样才能与官方 provenance 的“前几轮走势”做公平比较

默认输出：

- `/data/wyb/Transshield_final/artifacts/train_runs/`

## Boundary

本目录只用于明文训练 / 稳定性复验。secure sidecar、SPU pipeline、replay、compare 仍使用最终仓现有 `tools/`、`integrations/` 与 `artifacts/server_inference_friendly_pack/`。
