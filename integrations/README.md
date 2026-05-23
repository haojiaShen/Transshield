# integrations

本目录只保留当前正式作品仍有价值的安全后端集成实现。

## 当前结构

- `integrations/transshield_runtime/e2e_secure_vit/`
  - 当前仍需保留的主集成代码
  - 负责 E2E 安全推理执行器、SPU 静态前向、share 输入、调试探针、公共 layer norm 校准

## 当前主链路

当前正式保留的代码级主链路是：

- `tools/transshield_e2e_secure_infer.py`
- `integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py`

旧 runtime wrapper 已移入：

- `archive/deprecated/artifacts/server_inference_friendly_pack/`

## 命名说明

- 集成目录现统一使用 `transshield_runtime`。
- 这样可与 `configs/transshield_runtime/` 保持一致，并避免最终交付仓继续暴露历史目录命名。

## 当前判断

- `e2e_secure_vit/` 属于正式保留资产。
- 当前 `integrations/` 已删除不再使用的历史 bridge，只保留正式主链路所需实现。
