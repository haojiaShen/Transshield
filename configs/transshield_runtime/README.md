# TransShield runtime configs

这些文件是当前 `TransShield` 交付仓内置的 2PC / SPU 运行配置。

## 文件

- `2pc.json`：默认 colocated 双方运行配置
- `2pc.template.json`：在线展示模板；默认关闭逐算子 HAL/PPHLO profiling，避免热路径统计与日志开销
- `2pc_e2e.template.json`：whole-forward E2E 路径使用的模板
- `2pc_fm32.template.json`：FM32 变体模板

`tools/start_showcase_spu_demo.py` 不会直接改写上述被仓库跟踪的配置；它会把 `2pc.template.json` 复制到 `logs/showcase_runtime/2pc.runtime.json`，再把自动分配的端口写入这份运行时配置。`logs/` 已被 `.gitignore` 忽略，适合作为本机展示时的临时运行目录。

`2pc.json` 与 `2pc_e2e.template.json` 继续保留 profiling，用于通信量复核和算子归因；在线时延测试使用 `2pc.template.json`。两类结果不要直接混用。

## 推荐入口

展示站与 SPU 节点使用同一条启动命令：

```bash
python tools/start_showcase_spu_demo.py --host 0.0.0.0 --port 7860
```

该命令只面向单机演示：它会使用 `127.0.0.1` 端口启动两个 colocated SPU 节点，不会把 P1/P2 分别部署到医院侧和 AI 公司侧。

服务器后台启动示例：

```bash
/data/wyb/conda_envs/transshield/bin/python tools/start_showcase_spu_demo.py --host 127.0.0.1 --port 7862 --daemon
```

## 底层 SPU 节点入口

如需只重启展示用 colocated 2PC / SPU 节点，推荐先生成一份 `logs/` 下的运行时配置，再单独运行：

```bash
mkdir -p logs/showcase_runtime
cp configs/transshield_runtime/2pc.template.json logs/showcase_runtime/2pc.runtime.json
python tools/transshield_spu_runtime_setup.py start \
  --config logs/showcase_runtime/2pc.runtime.json \
  --template "" \
  --restart \
  --remove-unsupported-cheetah-fields \
  --log-dir logs/spu_nodes \
  --state-json logs/spu_runtime_ports.json
```

## 真两方配置边界

如果要迁移到“一方医院、一方 AI 公司”，需要生成远端配置，例如 `2pc.remote.json`，把 `nodes.node:0` / `nodes.node:1` 和 `spu_internal_addrs` 改成双方机器可互通的地址。

```bash
python tools/transshield_spu_runtime_setup.py render-remote \
  --node0-addr HOSPITAL_NODE_IP:44869 \
  --node1-addr AI_NODE_IP:44689 \
  --spu-internal0-addr HOSPITAL_NODE_IP:39889 \
  --spu-internal1-addr AI_NODE_IP:38579
```

`render-remote` 会写出 `configs/transshield_runtime/2pc.remote.json` 和对应的 `.commands.md` 启动命令，并默认关闭 colocated optimization。当前 `start` 子命令仍只适合本机演示：它会自动分配 localhost 端口并在本机启动两个节点，因此不能直接当作跨机器部署器使用。

医院侧拿到同一份 `2pc.remote.json` 后，推荐用单方启动器启动自己的节点：

```bash
python tools/transshield_spu_runtime_setup.py start-party \
  --config configs/transshield_runtime/2pc.remote.json \
  --party hospital \
  --restart
```

AI 侧把 `--party hospital` 换成 `--party ai`。`start-party` 会后台启动本方节点、写入 `logs/spu_party_nodes/` 日志和 `logs/spu_party_node_*.json` 状态文件；它不会启动双网关服务，也不会处理生产认证、TLS 或任务队列。

如果希望把医院侧、算力方和协调方的脚本一次性生成出来，可使用部署包入口：

```bash
python tools/transshield_spu_runtime_setup.py render-deployment \
  --output-dir deploy/transshield_remote_2pc \
  --repo-root /path/to/Transshield \
  --node0-addr HOSPITAL_NODE_IP:44869 \
  --node1-addr MODEL_PROVIDER_IP:44689 \
  --spu-internal0-addr HOSPITAL_NODE_IP:39889 \
  --spu-internal1-addr MODEL_PROVIDER_IP:38579
```

生成目录会包含 `hospital/`、`ai_provider/`、`coordinator/`、`frontend/` 和 `systemd/` 示例。实际项目只需要按现场替换 IP/域名、Python 环境、证书/认证、manifest 路径和机构名称。

部署包中的三个角色目录还会生成 `start_split_gateway.sh` 和 `.gateway.env.example`，用于启动 `showcase_api.split_gateway` 最小三方网关框架。该框架提供医院侧 raw PNG/JPEG 图片入口，并把医院侧 P1 share、AI 侧 P2 share / model manifest、协调侧任务状态和取消入口分开；它不替代生产 TLS、认证、审计或模型私有化验收。

医院侧 `.gateway.env` 可配置 `TRANSSHIELD_SPLIT_AI_GATEWAY_URL`，让医院图片入口在生成 P2 share 后自动转发到 AI 网关；这只是联调便利功能，不包含生产级消息队列、重试和机构审计。若只想本地验证三角色 API 框架，可运行：

```bash
python tools/split_gateway_smoke.py --keep-state
```

推荐迁移说明见：`docs/party_split_2pc.md`。

## 说明

- 目录名已经统一为 `transshield_runtime`
- 这表示这些配置现在按**当前项目交付口径**管理
- `integrations/transshield_runtime/` 也已同步改名，配置层与集成层现已统一使用当前项目命名
