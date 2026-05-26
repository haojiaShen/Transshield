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

## 当前部署边界

`e2e_secure_vit/` 已支持 split public/P1/P2 manifest 和 `--party-local-share-load` 调试桥，但当前展示站仍通过单机 colocated SPU 节点运行。若要落到医院侧 P1 与 AI 公司侧 P2，可先用 `tools/transshield_spu_runtime_setup.py render-deployment` 生成部署包，并用 `start-party` 在双方机器分别启动 SPU 节点；`showcase_api.split_gateway` 提供了最小医院/AI/协调网关框架。完整生产部署仍需要把该最小网关接入现场认证、TLS/VPN、审计、任务队列和模型参数私有加载流程。迁移说明见 `docs/party_split_2pc.md`。

旧 runtime wrapper 已移入：

- `archive/deprecated/artifacts/server_inference_friendly_pack/`

## 命名说明

- 集成目录现统一使用 `transshield_runtime`。
- 这样可与 `configs/transshield_runtime/` 保持一致，并避免最终交付仓继续暴露历史目录命名。

## 当前判断

- `e2e_secure_vit/` 属于正式保留资产。
- 当前 `integrations/` 已删除不再使用的历史 bridge，只保留正式主链路所需实现。
