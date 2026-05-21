# OpenBumbleBee 使用说明（2026-05-19 重建版）

## 1. 当前主线在用

| 模块 | 当前是否在用 | 代码入口 | 结果证据 | 在最终系统中的角色 |
|---|---|---|---|---|
| `configs/openbumblebee/2pc.json` 与模板 | 是 | `configs/openbumblebee/` | `configs/openbumblebee/README.md` | 当前 SPU 2PC 运行配置 |
| `tools/transshield_spu_runtime_setup.py` | 是 | `tools/transshield_spu_runtime_setup.py` | 当前服务器重跑全都通过它起 SPU | 统一改写端口、启动 colocated SPU 节点、记录状态 |
| `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` | 是 | `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` | 医疗 / 金融 2026-05-19 重跑 | whole-forward / secure pruning CLI 主入口 |
| `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` | 是 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` | 医疗 secure pruning、金融 dynamic / static 对照 | SPU/JAX 前向、party-local share load、secure pruning、true static no-pruning |
| `integrations/openbumblebee/e2e_secure_vit/cpu_static_vit.py` | 是 | `integrations/openbumblebee/e2e_secure_vit/cpu_static_vit.py` | 医疗全量验证集 dynamic threshold 校准、保留掩码 exact 对照 | CPU dynamic / static reference、keep-mask replay、阈值校准基线 |
| `integrations/openbumblebee/e2e_secure_vit/input_shares.py` | 是 | `integrations/openbumblebee/e2e_secure_vit/input_shares.py` | 所有 party-local share load 重跑 | public / P1 / P2 manifest 解析与 share ingestion |
| `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh` | 是 | `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh` | 医疗 `32` 样本复核、金融配对复核 | 统一服务器可执行包装器 |

## 2. 当前主线真实边界

- 当前主线实际使用的是：`OpenBumbleBee / SPU 2PC runtime + e2e_secure_vit integration`
- 当前正式会被报告引用的能力有五类：
  - whole-forward SPU contract
  - party-local share ingestion
  - secret/public parameter mode control
  - secure internal pruning（PredictorLG in-SPU）
  - true static no-pruning fallback（通过显式开关触发）
- 当前正式部署角色固定为两方：
  - 业务使用方侧服务器（医疗可对应医院侧服务器，金融可对应银行侧服务器）
  - AI 公司侧服务器（持有模型参数控制权）
- OpenBumbleBee / SPU 在这里承担的是两方之间的 2PC 运行层，不要求额外第三方可信服务器作为正式参与方

## 3. 2026-05-19 新增模式切换

本次新增并验证的显式开关：

- `--spu-pruning-mode secure_internal_pruning`
  - 默认值
  - 作用：把 `PredictorLG + kth_threshold + tie-resolution` 放在 SPU 里执行
  - 当前正式医疗主线、正式金融主线都使用这条

- `--spu-pruning-mode static_no_pruning`
  - 作用：关闭内部 dynamic pruning，强制固定结构 SPU forward
  - 当前只作为金融对照 / fallback 线保留

这一步的意义是把“当前到底在跑动态还是静态”写死，避免再把训练元数据里的 `use_mask_pruning = false` 误读成当前 runtime 必然静态。

## 4. 历史/对比链在用

| 模块 | 当前是否保留 | 代码入口 | 结果证据 | 在最终系统中的角色 |
|---|---|---|---|---|
| `tools/transshield_openbumblebee_pipeline.py` | 保留 | `tools/transshield_openbumblebee_pipeline.py` | 旧 pipeline step graph | 历史 `prepare -> bridge -> check -> replay` 对比链 |
| `integrations/openbumblebee/transshield_network_kth_bridge/` | 保留 | `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py` | 历史 bridge compare 文档 | `masked_score -> kth_threshold` 的旧 bridge compare |
| `integrations/openbumblebee/transshield_tie_policy_bridge/` | 保留 | `integrations/openbumblebee/transshield_tie_policy_bridge/transshield_tie_policy_bridge.py` | 历史 tie compare 文档 | tie-resolution 的旧 bridge compare |
| `tools/transshield_openbumblebee_inference_replay.py` | 保留 | `tools/transshield_openbumblebee_inference_replay.py` | 历史 replay / compare 分析链 | sidecar / replay / compare 的历史分析器 |

## 5. 不再作为正式主线

| 项目 | 当前判断 | 原因 |
|---|---|---|
| 只跑 `network_kth_bridge` / `tie_policy_bridge` 的桥接路径 | 不作为正式主线 | 只覆盖边界 compare，不是完整 whole-forward 系统 |
| 只保留 sidecar / replay、不开 whole-forward / secure pruning | 不作为正式主线 | 当前两域都已经有更完整的 full-privacy whole-forward 证据 |
| ABY / mixed-protocol 叙事 | 不再作为最终主线 | 本次重建不再把它升级为正式系统能力 |

## 6. 最终写作规则

- 报告里提到 OpenBumbleBee 时，统一写成：
  - “项目使用了 OpenBumbleBee / SPU 的 2PC 运行链与集成层”
- 然后明确拆成两层：
  - 当前主线：`configs/openbumblebee/` + `integrations/openbumblebee/e2e_secure_vit/` + `tools/transshield_spu_runtime_setup.py` + `run_e2e_secure_whole_forward.sh`
  - 历史/对比链：`tools/transshield_openbumblebee_pipeline.py` + `transshield_network_kth_bridge/` + `transshield_tie_policy_bridge/`
- 若报告要解释“为什么现在能把金融动态 / 静态分清”，必须引用本次新增的 `spu_pruning_mode` 明确说明。
