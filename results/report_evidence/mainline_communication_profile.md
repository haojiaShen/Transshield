# 主线通信量同口径补测（2026-05-20）

## 说明

- 这批数字来自 `2026-05-20` 对两条最终推荐路径做的同配置复核运行。
- 运行环境仍是当前仓库默认的 colocated `2PC` 原型环境，即两方节点通过 `127.0.0.1` 本地端口通信。
- 通信量来源是 fresh run 之后立即解析 `logs/spu_nodes/node_0.log` 中的 `Link details` 计数器，因此适合做当前报告主表中的同口径补充。
- 这些数字不应外推为医院—企业真实广域网部署时的最终 WAN 流量与时延。

## 医疗最终推荐路径

- 参考主线：医疗动态安全推理正式主线（内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_20260519_1`）
- 通信复核：同配置通信量复核运行（内部运行标识：`med_secure_pruning_smoke32_batch8_depth10_commprofile_20260519_1`）
- 样本数：`32`
- 实测耗时：`2849.94s`，即 `89.06s/sample`
- 双向总通信量：`90694658258 bytes`，约 `84.47 GiB`
- 单样本平均通信量：约 `2.64 GiB/sample`
- 隐私字段：`host_plaintext_pixel_values_materialized=false`、`host_model_params_materialized=false`、`reveal_policy=final_logits_only`

## 金融最终推荐路径

- 参考主线：金融动态安全推理正式主线（内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_20260519_1`）
- 通信复核：同配置通信量复核运行（内部运行标识：`finance_lrd_rank192_dynamic_partylocal_secret_smoke8_commprofile_20260519_1`）
- 样本数：`8`
- 实测耗时：`849.26s`，即 `106.16s/sample`
- 双向总通信量：`27168265566 bytes`，约 `25.30 GiB`
- 单样本平均通信量：约 `3.16 GiB/sample`
- 隐私字段：`host_plaintext_pixel_values_materialized=false`、`host_model_params_materialized=false`、`reveal_policy=final_logits_only`
