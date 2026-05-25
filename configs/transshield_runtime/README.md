# TransShield runtime configs

这些文件是当前 `TransShield` 交付仓内置的 2PC / SPU 运行配置。

## 文件

- `2pc.json`：默认 colocated 双方运行配置
- `2pc.template.json`：端口或临时目录需要重写时使用的模板
- `2pc_e2e.template.json`：whole-forward E2E 路径使用的模板
- `2pc_fm32.template.json`：FM32 变体模板

## 推荐入口

展示站与 SPU 节点使用同一条启动命令：

```bash
python tools/start_showcase_spu_demo.py --host 0.0.0.0 --port 7860
```

服务器后台启动示例：

```bash
/data/wyb/conda_envs/transshield/bin/python tools/start_showcase_spu_demo.py --host 127.0.0.1 --port 7862 --daemon
```

## 底层 SPU 节点入口

如需只重启 colocated 2PC / SPU 节点，可单独运行：

```bash
python tools/transshield_spu_runtime_setup.py start \
  --config configs/transshield_runtime/2pc.json \
  --template configs/transshield_runtime/2pc.template.json \
  --backup \
  --restart \
  --remove-unsupported-cheetah-fields \
  --log-dir logs/spu_nodes \
  --state-json logs/spu_runtime_ports.json
```

## 说明

- 目录名已经统一为 `transshield_runtime`
- 这表示这些配置现在按**当前项目交付口径**管理
- `integrations/transshield_runtime/` 也已同步改名，配置层与集成层现已统一使用当前项目命名
