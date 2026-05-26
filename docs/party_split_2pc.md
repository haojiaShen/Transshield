# 真两方 2PC 迁移说明

本文档说明如何把当前单机 colocated 演示迁移到“医院侧一方 + AI 公司侧一方”的真实 2PC 部署边界。当前仓库已经保留了 split manifest、party-local share load、SPU 运行桥接代码、基础跨机器单方节点启动器，以及最小三方网关框架；但它还不是生产级双网关部署，仍需补齐真实证书、认证体系、审计系统和模型私有加载策略。

## 当前做到哪里

当前可运行链路是：

- 浏览器本地完成图片解码、裁剪、归一化、质量摘要、哈希链和 `share0/share1` 生成。
- `showcase_api` 在单进程内短暂接收 `share0/share1 + 结构化摘要`，完成控制面快检后调用 SPU runner。
- `tools/start_showcase_spu_demo.py` 会启动 colocated 2PC/SPU 节点，节点地址由 `tools/transshield_spu_runtime_setup.py` 重写为 `127.0.0.1:*`。
- `integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py run --runtime spu` 支持 split public/P1/P2 manifest，并可用 `--party-local-share-load` 让 P1/P2 device function 各自加载自己的 share 文件。
- `showcase_api.split_gateway` 提供医院侧、AI 侧、协调侧三个 role 的最小 API 框架：医院侧可接收 PNG/JPEG 原始图片并生成 P1/P2 share，也可只接收 P1 share；AI 侧只接收 P2 share 和模型 manifest；协调侧负责异步任务状态、取消和 runner 命令。医院侧还可通过 `TRANSSHIELD_SPLIT_AI_GATEWAY_URL` 把 P2 share 自动转发给 AI 网关，便于本地或内网联调。

因此，当前仓库证明的是“浏览器侧分片 + 控制面校验 + 单机 colocated 2PC/SPU 原型”可以跑通，并且已经有了拆分医院/AI/协调服务的脚手架；它仍不是“一方医院、一方 AI 公司、互相不可见私有输入/模型”的生产部署。

## 推荐角色映射

| 角色 | 建议映射 | 应该持有 | 不应该看到 |
|---|---|---|---|
| P1 | 医院侧服务器 | 原始影像、预处理逻辑、P1 输入 share、本方审计日志 | AI 方模型明文、P2 输入 share |
| P2 | AI 公司侧服务器 | 模型资产或模型参数 share、P2 输入 share、本方审计日志 | 医院原始影像、P1 输入 share |
| 协调层 | 可由一方兼任，也可独立 | 公开任务 ID、public manifest、运行状态、最终可 reveal 结果 | 两份私有 share、原始影像、未授权模型明文 |

如果项目要求“医院保护输入，AI 保护模型”，仅拆分输入 share 还不够；模型参数也需要由 AI 侧加载或以 secret 参数模式注入，不能由一个中央 driver 同时读取完整模型和两份输入 share。

## 医院拿到仓库后的最短运行命令

下面命令只负责启动医院侧的 SPU party 节点。前提是医院侧和 AI 侧已经拿到同一份 `configs/transshield_runtime/2pc.remote.json`，其中 `node:0` / `spu_internal_addrs[0]` 是医院机器可被 AI 侧访问的地址，`node:1` / `spu_internal_addrs[1]` 是 AI 机器可被医院侧访问的地址。

医院侧运行：

```bash
cd /path/to/Transshield
python tools/transshield_spu_runtime_setup.py start-party \
  --config configs/transshield_runtime/2pc.remote.json \
  --party hospital \
  --restart
```

脚本会在后台启动 `node:0`，把日志写到 `logs/spu_party_nodes/node_0.log`，把 PID 和节点地址写到 `logs/spu_party_node_0.json`。医院侧只检查自己的节点是否可监听：

```bash
python tools/transshield_spu_runtime_setup.py check-party \
  --config configs/transshield_runtime/2pc.remote.json \
  --party hospital
```

停止医院侧节点：

```bash
python tools/transshield_spu_runtime_setup.py stop-party \
  --config configs/transshield_runtime/2pc.remote.json \
  --party hospital
```

AI 侧对应把 `--party hospital` 换成 `--party ai`。两侧都启动后，任意能访问两侧端口的协调机器可运行：

```bash
python tools/transshield_spu_runtime_setup.py check \
  --config configs/transshield_runtime/2pc.remote.json
```

`start-party` 默认拒绝 `127.0.0.1`、`localhost`、`0.0.0.0` 等地址，避免医院误把本机 dry-run 当成真两方部署；只有本地测试才使用 `--allow-localhost`。

## 推荐生成部署包

如果希望把大部分固定框架一次性准备好，推荐先生成部署包：

```bash
python tools/transshield_spu_runtime_setup.py render-deployment \
  --output-dir deploy/transshield_remote_2pc \
  --repo-root /path/to/Transshield \
  --node0-addr HOSPITAL_NODE_IP:44869 \
  --node1-addr MODEL_PROVIDER_IP:44689 \
  --spu-internal0-addr HOSPITAL_NODE_IP:39889 \
  --spu-internal1-addr MODEL_PROVIDER_IP:38579
```

生成目录包含：

- `configs/2pc.remote.json`：双方共用的远程 SPU 配置。
- `hospital/`：医院侧 `.env.example`、启动、检查、停止脚本。
- `ai_provider/`：AI 公司或内部模型算力方 `.env.example`、启动、检查、停止脚本。
- `coordinator/`：检查双方节点与运行 secure ViT 的示例脚本。
- `start_split_gateway.sh` 与 `.gateway.env.example`：三方网关框架启动脚本和配置示例。
- `frontend/`：`VITE_TRANSSHIELD_API_BASE_URL` 示例，用于把当前展示站前端指向兼容 `/api/medical/*` 的展示 API 或反向代理；它不能直接替代为 split gateway 的 `/api/hospital/*`、`/api/split/*`。
- `systemd/`：可选的 service 示例，适合再交给现场运维调整。

这一步把“仓库怎么落到两台机器”固定下来；剩下应由不同医院或算力方现场决定的内容包括真实 IP/域名、VPN/TLS、证书、密钥、Python/SPU 环境路径、数据落盘目录、审计系统和 manifest 交换方式。

如果只是在同一台机器做演练，生成配置和运行脚本时都必须显式打开 dry-run 开关：生成时加 `--allow-localhost`，运行生成脚本时设置 `TRANSSHIELD_ALLOW_LOCALHOST=1`。正式部署不要打开这个开关。

如果要用“本地主机 + 远端服务器”做验证，尤其本地主机在 WSL/NAT 后面时，按 `docs/host_server_2pc_validation.md` 做 SSH 隧道和 `warmup_all_parties.sh` 验收。

## 最小三方网关框架

部署包会在 `hospital/`、`ai_provider/` 和 `coordinator/` 下生成 `start_split_gateway.sh`。复制 `.gateway.env.example` 为 `.gateway.env` 后即可启动对应 role：

```bash
cp .gateway.env.example .gateway.env
./start_split_gateway.sh
```

三个 role 的职责如下：

- 医院侧 `TRANSSHIELD_SPLIT_ROLE=hospital`：`POST /api/split/tasks/{task_id}/share` 只接受 P1 share，写入医院侧本地存储和 P1 party manifest。
- 医院侧图片入口：`POST /api/hospital/tasks/{task_id}/image` 接收 raw PNG/JPEG 请求体，医院侧生成 P1/P2 share，P1 share 留存在医院本地，并返回一份可转交给 AI 网关的 `p2_share_delivery` JSON。
- AI 侧 `TRANSSHIELD_SPLIT_ROLE=ai`：`POST /api/split/tasks/{task_id}/share` 只接受 P2 share，`POST /api/ai/tasks/{task_id}/model-manifest` 记录模型 bundle 或模型版本信息。
- 协调侧 `TRANSSHIELD_SPLIT_ROLE=coordinator`：`POST /api/coordinator/tasks/{task_id}/runs` 创建异步 runner 任务，`GET /api/split/tasks/{task_id}` 查询状态，`POST /api/coordinator/tasks/{task_id}/cancel` 请求取消。

如果 `.gateway.env` 中设置 `TRANSSHIELD_SPLIT_AUTH_TOKEN`，上述接口需要 `Authorization: Bearer <token>`。正式部署必须把网关放在 TLS/VPN、机构认证、审计落盘和运维守护之后；当前框架只提供代码边界、图片预处理、分片生成和任务状态骨架。

如果医院侧 `.gateway.env` 配置了 `TRANSSHIELD_SPLIT_AI_GATEWAY_URL=http://AI_GATEWAY_HOST:8702`，医院图片入口会在生成 P1/P2 share 后尝试把 `p2_share_delivery` 自动 POST 到 AI 网关。该功能只适合联调和受控内网演练：它没有生产级重试队列、消息签名、持久化投递状态或机构级审计，失败时仍会在响应里返回 `p2_share_delivery`，方便人工或上层系统重试。

医院侧 raw image 最小调用示例：

```bash
curl -X POST \
  -H 'Content-Type: image/png' \
  -H 'X-Source-Filename: sample.png' \
  --data-binary @sample.png \
  http://127.0.0.1:8701/api/hospital/tasks/demo001/image
```

响应中的 `p2_share_delivery` 可原样作为 AI 侧 share 接收接口的 JSON body：

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  --data @p2_share_delivery.json \
  http://127.0.0.1:8702/api/split/tasks/demo001/share
```

如果只想在本地验证三方 API 框架和图片分片流程是否可用，可运行仓内 smoke：

```bash
python tools/split_gateway_smoke.py --keep-state
```

该脚本会用一张合成 PNG 跑完“医院上传图片 -> 生成 P1/P2 share -> AI 接收 P2 share -> AI 提交 model manifest -> 协调方 mock run 完成”，并额外验证鉴权缺失、角色错误、share 接收和协调运行阶段的 payload/manifest `task_id` 不一致、share hash 不一致会被拒绝。

## 迁移步骤

1. 拆分上传入口。
   生产目标是由医院侧影像入口接收原始图片并生成两份输入 share，`share0` 留在医院侧，`share1` 通过认证通道发给 AI 侧；不能再使用当前 `showcase_api` 单进程同时接收两份 share 的方式。当前 `showcase_api.split_gateway` 已提供 raw PNG/JPEG 图片入口和单方 share 接收框架；后续只需把医院 HIS/PACS/上传系统按现场规范接到该入口或等价内部接口。

2. 拆分 manifest。
   public manifest 只包含样本数量、shape、dtype、party id、哈希和公开元数据，不包含私有路径。P1 party manifest 只保存在医院侧，只指向 P1 share；P2 party manifest 只保存在 AI 侧，只指向 P2 share。

3. 拆分 SPU 节点配置。
   使用 `render-remote` 生成类似 `configs/transshield_runtime/2pc.remote.json` 的配置，把 `nodes.node:0` 和 `nodes.node:1` 从 `127.0.0.1:*` 改成医院侧与 AI 侧可互通地址。当前 `start` 子命令仍会自动分配 localhost 端口并启动两个本地节点，不能直接用于跨机器生产部署。

   ```bash
   python tools/transshield_spu_runtime_setup.py render-remote \
     --template configs/transshield_runtime/2pc.template.json \
     --output configs/transshield_runtime/2pc.remote.json \
     --node0-addr HOSPITAL_NODE_IP:44869 \
     --node1-addr AI_NODE_IP:44689 \
     --spu-internal0-addr HOSPITAL_NODE_IP:39889 \
     --spu-internal1-addr AI_NODE_IP:38579
   ```

   该命令默认拒绝 `127.0.0.1` / `localhost`，避免误把单机配置当作真两方配置；只有本地 dry-run 才应加 `--allow-localhost`。

4. 分别启动两侧节点。
   `render-remote` 会同时生成 `configs/transshield_runtime/2pc.remote.commands.md`。两台机器应分别运行自己的 SPU 节点。推荐使用仓库封装好的单方启动器：

   ```bash
   python tools/transshield_spu_runtime_setup.py start-party --config configs/transshield_runtime/2pc.remote.json --party hospital --restart
   python tools/transshield_spu_runtime_setup.py start-party --config configs/transshield_runtime/2pc.remote.json --party ai --restart
   ```

   实际部署时仍需要配合防火墙、VPN/TLS、端口放行、进程守护和日志归档。`start-party` 只解决“单方节点如何便捷启动与检查”的问题，不负责上传网关、任务队列或身份认证。

5. 使用 party-local share load。
   runner 调用应保留 split manifest 参数，并打开隐私前向选项：

   ```bash
   python integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py run \
     --runtime spu \
     --bundle-dir artifacts/frozen_bundle_medical_dynamic_mainline \
     --input-share-public-manifest-json public_manifest.json \
     --input-p1-share-manifest-json p1_share_manifest.json \
     --input-p2-share-manifest-json p2_share_manifest.json \
     --party-local-share-load \
     --redact-private-input-paths \
     --config configs/transshield_runtime/2pc.remote.json \
     --max-samples 1
   ```

   这条命令体现当前代码支持的目标形态，但在生产中还需要把 runner/driver 改成不能读取另一方私有 manifest 或私有路径。

6. 补齐模型私有边界。
   当前展示站默认从本地 bundle 读取模型资产。若 AI 公司模型也必须保密，需要把模型参数加载迁移到 AI 侧，或按 SPU 参数模式注入 secret 参数，并确保医院侧和协调层不能读取完整明文模型。

7. 补齐服务化能力。
   `showcase_api.split_gateway` 已提供异步任务状态、取消入口和超时配置骨架。真实部署还需要把任务队列接入医院或 AI 公司现有队列系统，补齐认证授权、重放窗口持久化、审计落盘、结果最小 reveal 和异常告警。

## 迁移验收清单

- 除医院侧分片生成瞬间外，任何长期运行/落盘组件不能同时持有 `share0` 与 `share1`；医院侧生成后只持久化 P1 share，并把 P2 share 投递给 AI 侧。
- 医院侧日志不出现 AI 模型明文路径或完整参数。
- AI 侧日志不出现原始影像、明文像素张量或 P1 share。
- public manifest 不包含私有 share 路径。
- candidate `.pt/.json` 使用 `--redact-private-input-paths` 后不持久化 P1/P2 私有路径。
- `configs/transshield_runtime/2pc.remote.json` 中的 `nodes` 和 `spu_internal_addrs` 不再是 `127.0.0.1` / `localhost`。
- `runtime_config.experimental_enable_colocated_optimization=false`，除非只是本地 dry-run。
- 最终输出只 reveal 允许公开的 logits、概率或诊断标签。
- 重新运行 `tools/showcase_protocol_fuzz.py` 和 `tools/showcase_guard_stress.py` 的等价跨服务版本，并重新采集通信量。

## 不能直接宣传成已完成的部分

- 当前 live demo 是 colocated 2PC/SPU 原型，不是真正跨组织两方部署。
- 当前 debug additive shares 是工程桥接数据格式，不等价于完整生产 MPC 输入管线。
- 当前 `start-party` 可以把 P1/P2 SPU 节点放到两台机器上分别启动，`showcase_api.split_gateway` 可以启动三方 API 框架并提供医院侧 raw PNG/JPEG 图片入口；但还没有完成医院官方系统对接、生产认证、长期审计和模型参数私有加载边界。
- 当前展示站控制面验证可以作为迁移基础，但不能替代生产级身份认证、链路加密、任务调度和拒绝服务防护。
