# 主机 + 服务器远程 2PC 验证说明

本文档说明如何判断当前远程 2PC 框架能不能用于“本地主机 + 远端服务器”两台机器，并给出可执行的验证步骤。它验证的是 SPU 节点跨机器启动、连通和最小 warmup，不等价于生产级医院/AI 双网关上线。

## 先看结论

要让主机和服务器各跑一方，必须同时满足三件事：

- 两边都有同一版本的 TransShield 仓库。
- 两边的 Python 环境都能导入 `jax` 和 `spu.utils.distributed`。
- 两边能互相访问配置里的四个端口：`node:0`、`node:1`、`spu_internal_addrs[0]`、`spu_internal_addrs[1]`。

如果本地主机在 WSL、NAT 或校园网内，服务器通常不能直接连回本机端口。这种情况下可以用 SSH 正向/反向隧道验证，但隧道模式必须显式使用 `--allow-localhost` 和 `TRANSSHIELD_ALLOW_LOCALHOST=1`，避免把 localhost 演练误写成真实跨组织部署。

## 当前环境检查记录

截至 2026-05-26，本次环境检查结果如下：

- 远端服务器 `/data/wyb/conda_envs/transshield/bin/python` 可以导入 `jax` 和 `spu.utils.distributed`。
- 本地主机 `/home/yclcg/miniconda3/envs/transshield/bin/python` 可以导入 `jax` 和 `spu.utils.distributed`。
- 服务器不能直接访问本地主机 WSL 地址 `172.22.177.174:55299`，直连测试超时。
- SSH 反向隧道可以让服务器访问本地主机服务，`-R 55300:127.0.0.1:55299` 测试通过。
- SSH 正向隧道可以让本地主机访问服务器本地服务，`-L 55301:127.0.0.1:7862` 访问 `/api/health` 测试通过。
- 使用 SSH 正向/反向隧道跑通了“本地主机 P1 + 服务器 P2”的最小 SPU warmup，返回 `warmup_result: [2.0]`。

因此，当前机器组合可以验证“部署包生成、SSH 隧道连通、双方 SPU 节点启动、最小 SPU runtime warmup”。它仍不代表生产级医院/AI 双网关已完成。

## 1. 检查两边 Python 环境

在本地主机运行：

```bash
cd /path/to/Transshield
/home/yclcg/miniconda3/envs/transshield/bin/python - <<'PY'
import importlib
for name in ["jax", "spu.utils.distributed"]:
    importlib.import_module(name)
    print(f"{name}: ok")
PY
```

在服务器运行：

```bash
cd /data/wyb/Transshield_final
/data/wyb/conda_envs/transshield/bin/python - <<'PY'
import importlib
for name in ["jax", "spu.utils.distributed"]:
    importlib.import_module(name)
    print(f"{name}: ok")
PY
```

两边都输出 `ok` 后，才继续做跨机器 SPU 验证。

## 2. 选择直连模式或隧道模式

### 直连模式

如果服务器能直接访问本地主机 IP，优先使用真实 IP：

```bash
/home/yclcg/miniconda3/envs/transshield/bin/python tools/transshield_spu_runtime_setup.py render-deployment \
  --output-dir deploy/host_server_2pc \
  --repo-root /path/to/Transshield \
  --node0-addr HOST_REACHABLE_IP:55269 \
  --node1-addr SERVER_REACHABLE_IP:55270 \
  --spu-internal0-addr HOST_REACHABLE_IP:55271 \
  --spu-internal1-addr SERVER_REACHABLE_IP:55272
```

直连模式不要加 `--allow-localhost`。

### SSH 隧道模式

如果服务器不能直接连本地主机端口，可在本地主机开一个保持不断的 SSH 隧道：

```bash
ssh -p 9001 \
  -L 55270:127.0.0.1:55270 \
  -L 55272:127.0.0.1:55272 \
  -R 55269:127.0.0.1:55269 \
  -R 55271:127.0.0.1:55271 \
  wyb@10.204.248.175
```

另开一个本地主机终端，生成本机侧部署包：

```bash
cd /path/to/Transshield
/home/yclcg/miniconda3/envs/transshield/bin/python tools/transshield_spu_runtime_setup.py render-deployment \
  --allow-localhost \
  --output-dir deploy/host_server_tunnel_2pc \
  --repo-root /path/to/Transshield \
  --node0-addr 127.0.0.1:55269 \
  --node1-addr 127.0.0.1:55270 \
  --spu-internal0-addr 127.0.0.1:55271 \
  --spu-internal1-addr 127.0.0.1:55272
```

在服务器也生成同端口部署包，但 `--repo-root` 改成服务器仓库路径：

```bash
cd /data/wyb/Transshield_final
/data/wyb/conda_envs/transshield/bin/python tools/transshield_spu_runtime_setup.py render-deployment \
  --allow-localhost \
  --output-dir deploy/host_server_tunnel_2pc \
  --repo-root /data/wyb/Transshield_final \
  --node0-addr 127.0.0.1:55269 \
  --node1-addr 127.0.0.1:55270 \
  --spu-internal0-addr 127.0.0.1:55271 \
  --spu-internal1-addr 127.0.0.1:55272
```

隧道模式下，本地主机看到服务器节点是 `127.0.0.1:55270/55272`，服务器看到本地主机节点是 `127.0.0.1:55269/55271`。

## 3. 启动双方节点

本地主机作为医院/P1：

```bash
cd /path/to/Transshield/deploy/host_server_tunnel_2pc/hospital
cp .env.example .env
TRANSSHIELD_ALLOW_LOCALHOST=1 ./start_spu_party.sh
```

服务器作为 AI 或模型算力方/P2：

```bash
cd /data/wyb/Transshield_final/deploy/host_server_tunnel_2pc/ai_provider
cp .env.example .env
TRANSSHIELD_ALLOW_LOCALHOST=1 ./start_spu_party.sh
```

如果使用直连模式，不要设置 `TRANSSHIELD_ALLOW_LOCALHOST=1`。

## 4. 判断是否成功

第一层：单方端口检查。

```bash
./check_spu_party.sh
```

第二层：协调方检查两个节点端口。

```bash
cd /data/wyb/Transshield_final/deploy/host_server_tunnel_2pc/coordinator
cp .env.example .env
./check_all_parties.sh
```

第三层：SPU 最小 warmup。该命令会初始化 SPU runtime，发送一个 P1 输入到 SPU 做 `+1.0`，成功时返回 `warmup_result`。

```bash
./warmup_all_parties.sh
```

看到类似输出即可认为远程 2PC runtime 已经跑通：

```json
{
  "status": "ok",
  "warmup_result": [2.0]
}
```

如果 warmup 失败，优先检查：

- SSH 隧道窗口是否仍然保持连接。
- 两边是否都使用同一组端口。
- 两边 `.env` 中 `TRANSSHIELD_REPO_ROOT`、`TRANSSHIELD_REMOTE_CONFIG` 是否指向本机真实路径。
- 本地主机和服务器是否都使用能导入 `jax` / `spu.utils.distributed` 的 Python。

## 5. 停止节点

本地主机：

```bash
cd /path/to/Transshield/deploy/host_server_tunnel_2pc/hospital
./stop_spu_party.sh
```

服务器：

```bash
cd /data/wyb/Transshield_final/deploy/host_server_tunnel_2pc/ai_provider
./stop_spu_party.sh
```

## 6. 不能由本验证覆盖的部分

上述流程只验证跨机器 SPU 节点与最小 runtime warmup。它不验证：

- 医院网关和 AI 网关是否已经拆分。
- 认证、TLS、证书、密钥轮换是否合规。
- 模型参数是否只由 AI 或模型算力方持有。
- 真实 public/P1/P2 manifest 的生成、交换和审计流程。
- 长任务队列、取消、超时回收和跨服务 fuzz/guard。

这些仍需要结合实际医院网络、安全策略和算力方部署方式单独验收。
