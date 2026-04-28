# 下一次接手前先看这里

最后更新：`2026-04-27`

## 先读哪些文件

1. `docs/project_overview_newcomer_defense.md`
2. `docs/data_source_policy.md`
3. `docs/result_summary.md`
4. `docs/external_baseline_comparison.md`
5. `docs/web_chat_demo.md`
6. `docs/current_work_status.md`
7. `docs/margin_aware_pruning_notes.md`
8. `docs/network_kth_blockwise_notes.md`
9. `artifacts/server_inference_friendly_pack/README.md`

## 最新 Web / E2E 主线口径（2026-04-26）

- `2026-04-27` 最新接手重点：先不要继续扩大 E2E eval 或 full-depth SPU。服务器当前曾出现 `import torch` 后卡死、Ctrl+C 不易中断、新终端也受影响的状态；代码层面已降低 Web 默认路径对 torch 的额外依赖，但已经卡住的会话更像 native/CUDA/SPU runtime 层问题。
- 下次服务器恢复后，第一步只跑轻量 Web 后端 sanity：

```bash
cd /data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
$PYTHON_BIN tools/transshield_chat_demo.py --help
```

- 如果上面不卡，再启动 Web demo：

```bash
export WEB_DEMO_HOST=0.0.0.0
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

- 不要默认设置 `WEB_DEMO_AUTO_CALIBRATE_E2E=1`。Web 现在要求 public layer norm calibration JSON 预先存在；缺失就明确报错，避免网页点击后隐式进入慢校准或 `import torch` 卡死。
- 新版 E2E candidate JSON 写入 `prediction_preview`；Web 后端读取 E2E 结果时不再 `torch.load(candidate.pt)`，所以若看到 “candidate JSON missing prediction_preview”，说明服务器还没同步新代码或使用了旧 candidate，需要重跑一次 `run_e2e_secure_approx_deploy.sh infer` 生成新 JSON。
- 最终 Web demo 主按钮已经收敛到浏览器本地分片：原图只在浏览器内存中读取，Canvas 预处理后生成 `share0/share1`，后端接口是 `/api/e2e/analyze_private_shares`。
- 旧 `/api/upload` 与 `/api/run_secure` 只作为 legacy CPU/SPU sidecar 调试接口保留，默认禁用；只有设置 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1` 时才开放。
- 当前可展示结论是：网页端不上传原图、不上传完整 plaintext `pixel_values`，E2E runner 的 candidate JSON 应满足 `input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`input_mode=party_local_debug_share_load`。
- 当前仍不能写成已经完成生产级全隐私部署，因为 demo 后端是单进程同时接收两份 debug share；生产环境需要独立 P1/P2 上传端点、独立信任域与访问隔离。
- 旧 SPU sidecar 的准确率/通信对比不再作为最终主口径；使用 `artifacts/server_inference_friendly_pack/run_e2e_secure_approx_eval.sh` 生成 E2E vs 原始明文同图片列表评测。

## 权威同步命令

首次执行前，建议先临时加上 `--dry-run` 确认文件列表。

### 本地 → 服务器（黑名单，默认同步代码 / 文档 / 脚本）

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
  wyb@10.204.244.1:/data/wyb/Transshield_final/
```

### 服务器 → 本地（白名单，只回传运行日志 / 报告）

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
  wyb@10.204.244.1:/data/wyb/Transshield_final/artifacts/ \
  /home/yclcg/Transshield_final/artifacts/
```

注意：

- 服务器回传时，必须保持 `artifacts/ -> artifacts/`，不要把 `/data/wyb/Transshield_final/artifacts/` 同步到 `/home/yclcg/Transshield_final/` 根目录。
- 上面第二条命令默认不会回传 `*.pth`、`*.pt`、`tb/`、`__pycache__/` 这类大文件或缓存。
- 仓内旧版同步命令已统一作废；如果在历史聊天记录或旧截图里再看到别的 rsync/scp 写法，默认按过期处理，以本节两条命令为唯一权威版本。

## 当前 `e2e secure whole-forward` 抓取命令

先在服务器同一个 shell / tmux 会话里固定：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export RUN_NAME=tracka_e2e_secure_poc_cpu
export E2E_DIR="$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc"
export PACK_DIR="$E2E_DIR/whole_forward_pack"
```

服务器端快速查看当前关键结果：

```bash
cat "$E2E_DIR/e2e_secure_contract.json"
cat "$E2E_DIR/client_pixel_values.json"
cat "$E2E_DIR/plaintext_reference.json"
cat "$E2E_DIR/static_whole_forward_reference.json"
cat "$E2E_DIR/e2e_static_whole_forward_candidate_from_server.json"
cat "$E2E_DIR/e2e_static_whole_forward_compare.json"
cat "$PACK_DIR/commands.json"
```

截至 `2026-04-23`，服务器 `runtime=spu` 渐进 smoke 已经从 `depth=0` 一路推进到 `depth=6`，而且 `depth=6` 现在已经分成“默认 colocated 配置的历史异常 run”与“`nocoloc` 缓解后的稳定 run”两条证据。当前环境仍会提示 `CUDA-enabled jaxlib is not installed. Falling back to cpu.`，因此整网 12 层 static ViT 仍然很慢；但在 `sample=1 / public params` 条件下：

- `depth=0..5` 都已通过“`SPU depth=k` vs `CPU depth=k`”对齐检查；
- 默认 colocated 配置下的首个 `depth=6` full run 曾出现 `0.0/0.0` 决策失配，并伴随 node/link 不稳定；
- `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 后，`depth=6`、`depth=7`、`depth=8`、`depth=9`、`depth=10`、`depth=11`、`depth=12` 的 **single-sample** full run 都已恢复为 `1.0/1.0` 决策一致。

因此下一位接手者**不要再从 `depth=0` 重新开始**；当前最合理的起点是把 `tracka_e2e_secure_spu_depth6_smoke1_20260423` 当作历史异常证据，把 `tracka_e2e_secure_spu_depth12_smoke1_nocoloc_20260423` 当作当前 single-sample full-depth 基线，再把 `tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423` 当作新的 multi-sample 失败边界，并在此基础上继续做 `sample=2` 的最小归因：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
E2E_DIR_5="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth5_smoke1_20260423/e2e_secure_poc"
E2E_DIR_6_BAD="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth6_smoke1_20260423/e2e_secure_poc"
E2E_DIR_6_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth6_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_7_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth7_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_8_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth8_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_9_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth9_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_10_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth10_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_11_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth11_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_12_NOCOLOC="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth12_smoke1_nocoloc_20260423/e2e_secure_poc"
E2E_DIR_12_SMOKE2="$REPO_ROOT/artifacts/server_pipeline_run/tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423/e2e_secure_poc"

echo "===== depth=5 :: cpu ====="
cat "$E2E_DIR_5/e2e_static_whole_forward_candidate_cpu_depth5.json"
echo "===== depth=5 :: spu ====="
cat "$E2E_DIR_5/e2e_static_whole_forward_candidate_from_server.json"
echo "===== depth=5 :: compare ====="
cat "$E2E_DIR_5/e2e_static_whole_forward_compare_spu_vs_cpu_depth5.json"

echo "===== depth=6 bad :: compare ====="
cat "$E2E_DIR_6_BAD/e2e_static_whole_forward_compare_spu_vs_cpu_depth6.json"
echo "===== depth=6 nocoloc :: cpu ====="
cat "$E2E_DIR_6_NOCOLOC/e2e_static_whole_forward_candidate_cpu_depth6_nocoloc.json"
echo "===== depth=6 nocoloc :: spu ====="
cat "$E2E_DIR_6_NOCOLOC/e2e_static_whole_forward_candidate_from_server.json"
echo "===== depth=6 nocoloc :: compare ====="
cat "$E2E_DIR_6_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth6_nocoloc.json"
echo "===== depth=6 nocoloc :: runtime ====="
cat "$REPO_ROOT/logs/spu_runtime_ports.json"
echo "===== depth=7 nocoloc :: compare ====="
cat "$E2E_DIR_7_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth7_nocoloc.json"
echo "===== depth=8 nocoloc :: compare ====="
cat "$E2E_DIR_8_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth8_nocoloc.json"
echo "===== depth=9 nocoloc :: compare ====="
cat "$E2E_DIR_9_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth9_nocoloc.json"
echo "===== depth=10 nocoloc :: compare ====="
cat "$E2E_DIR_10_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth10_nocoloc.json"
echo "===== depth=11 nocoloc :: compare ====="
cat "$E2E_DIR_11_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth11_nocoloc.json"
echo "===== depth=12 nocoloc :: compare ====="
cat "$E2E_DIR_12_NOCOLOC/e2e_static_whole_forward_compare_spu_vs_cpu_depth12_nocoloc.json"
echo "===== depth=12 smoke2 nocoloc :: compare ====="
cat "$E2E_DIR_12_SMOKE2/e2e_static_whole_forward_compare_spu_vs_cpu_depth12_smoke2_nocoloc.json"
```

判定标准：

- `depth=5` 当前是“最后一个仍保持决策一致”的边界点：`argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，但 `logits/probabilities max_abs_error` 已升到约 `1.20e-1 / 4.01e-2`；
- 历史 `depth=6` bad run 要作为**系统异常证据**看：`argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，并伴随 `grpc / Socket closed / Not connected`；
- `depth=6 nocoloc` 才是当前最新可复用结论：`argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，`logits/probabilities max_abs_error ≈ 1.43e-1 / 4.34e-2`，且 `disable_colocated_optimization = true` 时无 node 错误；
- `depth=7..12 nocoloc` 的 **single-sample** run 也都已保持 `1.0/1.0` 决策一致，其中 `depth=12` 的 `logits/probabilities max_abs_error` 约为 `1.54e-1 / 6.06e-2`；
- 但 `depth=12, sample=2, nocoloc` 已变成新的失败边界：`argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 6.92e-1 / 3.00e-1`，同时 runtime 依然稳定；
- 因此当前下一步优先级已更新为：**不再继续扩大样本量，而是固定 `depth=12, sample=2, nocoloc`，优先比较 `spu_batch_size=1` 与 `spu_batch_size=2`，判断问题到底出在“逐 chunk 多次调用/拼接”还是“真正 batched whole-forward”**。

`2026-04-25` 继续推进后，上面这条 `sample=2` issue 已进一步收口，下一位接手者应优先使用以下新结论，而不是再重复旧的 bsz 对照：

- `tracka_e2e_secure_depth12_smoke2_bsz2_direct_20260425_001632`：`depth=12 / sample=2 / spu_batch_size=2 / nocoloc` 可以稳定跑完，`spu_status=0`，无 node error，说明 batched 形状本身不会直接把 runtime 搞崩；但 compare 只有 `argmax_match_ratio=0.5`、`threshold_match_ratio=0.5`，`logits/probabilities max_abs_error≈3.74e-1 / 1.65e-1`。
- 同 run 逐样本拆解显示：`sample0` 虽误差大但决策仍对；`sample1` 决策翻转，CPU 为 `argmax=1 / threshold=1`，SPU 为 `argmax=0 / threshold=0`。
- `tracka_e2e_secure_depth12_sample1_single_fix_20260425_010810`：把 `sample1` 单独拿出来跑 `depth=12 / sample=1 / bsz=1 / nocoloc` 仍然 runtime 成功但决策翻转，compare 为 `0.0 / 0.0`，`logits/probabilities max_abs_error≈1.40e-1 / 5.93e-2`；因此 `sample2 bsz2` 的失败不是简单 batched 语义错误，而是 `sample1` 本身靠近决策边界、SPU 数值漂移后跨界。
- `sample1` 深度边界已定位：`tracka_e2e_secure_depth6_sample1_single_20260425_012825` 与 `tracka_e2e_secure_depth8_sample1_single_20260425_013826` 仍保持 `1.0 / 1.0`；`tracka_e2e_secure_depth9_sample1_single_20260425_014637` 开始变成 `0.0 / 0.0`；`depth10` 与 `depth12` 继续失败。因此翻转边界是 `depth8 -> depth9`。
- `tracka_e2e_secure_block9_probe_sample1_depth9_20260425_015446`：`probe_block_index=8` 的 block9 probe 成功复现最终 `0.0 / 0.0`。误差排名第一为 `mlp_out_cls`（`max_abs_error≈23.10`、`l2_error≈92.67`），但 `attn_out_cls` 已出现明显方向漂移（`cosine≈0.0908`、`max_abs_error≈6.31`）。当前归因口径应写成：**sample1 在 block9 attention 后方向漂移明显，随后 MLP 输出把误差幅度显著放大，最终跨过分类/阈值边界**。
- `tracka_e2e_secure_block9_probe_sample0_depth9_20260425_020754`：同口径 sample0 对照已完成，最终仍是 `1.0 / 1.0`。sample0 也有 `attn_out_cls` 方向漂移（`cosine≈0.0289`、`max_abs_error≈10.57`）和较大 MLP 误差（`max_abs_error≈11.50`），但最终 logits/probabilities 误差与方向仍可被分类 margin 吸收（`final_logits/probabilities max_abs_error≈1.96e-1 / 7.37e-2`，cosine `≈0.9845 / 0.9930`）。因此不要把 block9 attention 漂移直接等同于必然翻转；当前关键差异是 sample1 更靠近边界且 MLP 放大后 final head 方向劣化更明显。
- `tracka_e2e_secure_block9_probe_sample1_depth9_attn_standard_20260425_023618`：第一条 SPU-only ablation 已完成。将 `E2E_SPU_ATTENTION_POLICY=standard` 后，`sample1 / depth9 / block9 probe` 的最终 `argmax/threshold` 从默认 smoothed 的 `0.0 / 0.0` 恢复到 `1.0 / 1.0`。但它不是干净的数值误差改善：`mlp_out_cls max_abs_error≈47.07`，比默认 smoothed 的 `≈23.10` 更大；`final_logits/probabilities max_abs_error≈4.50e-1 / 1.95e-1`，`attn_out_cls cosine≈-0.1056`。因此它只能记作“决策方向诊断性正结果”，不能写成稳定修复。
- `tracka_e2e_secure_depth9_sample1_attn_standard_full_rerun_20260425_164037`：同一 `standard attention` 开关的 full `depth9 / sample1 / bsz=1 / nocoloc` 已补出有效 compare。该 run 从 `tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423` 的两样本输入包切出原始 `sample1`，CPU reference 与 SPU candidate 的 `.pt` 已正确分开；compare 仍为 `argmax_match_ratio=0.0`、`threshold_match_ratio=0.0`，`logits/probabilities max_abs_error≈3.68499e-1 / 1.77430e-1`。因此 block9 probe 的 `standard attention` 正结果不能外推到 full depth9，只能记作 probe-only / 决策方向诊断信号。
- `tracka_e2e_secure_depth9_sample1_gelu_full_20260425_171045`：`gelu` activation ablation 也已完成。固定 `depth9 / sample1 / bsz=1 / nocoloc`，保持默认 `E2E_SPU_ATTENTION_POLICY=smoothed`，只把 `E2E_SPU_ACTIVATION_OVERRIDE=gelu` 后，CPU reference 与 SPU candidate 的 `.pt` 已正确分开；compare 仍为 `argmax_match_ratio=0.0`、`threshold_match_ratio=0.0`，`logits/probabilities max_abs_error≈7.13191e-1 / 1.41736e-1`。因此 full depth9 失配不能简单归因成 activation override 选择错误。
- `tracka_e2e_secure_block9_probe_sample1_depth9_smoothed_gelu_20260425_172711`：`smoothed+gelu` 的 block9 probe 已完成，最终仍为 `argmax/threshold=0.0/0.0`。更重要的是最大误差不是 block9 的局部 attention / MLP 输出，而是进入 block9 前的 `block_input_cls` 已经爆炸：SPU `block_input_cls`、`attn_residual_out_cls`、`block_output_cls` 范数约 `9.95e9`，`max_abs_error≈1.737e9`。因此 `gelu` override 会在上游引入灾难性数值放大，不应继续作为修复方向。
- `tracka_e2e_secure_block9_probe_sample1_depth9_smoothed_bundle_20260425_181915`：`bundle activation` 下的 block9 同口径对照已完成。`smoothed+bundle` 最终保持 `argmax/threshold=1.0/1.0`，但 `mlp_out_cls` 最大误差仍较大（`max_abs_error≈60.40`、`l2_error≈253.54`），final logits/probabilities `max_abs_error≈4.892e-1 / 2.116e-1`。同一输入同一 probe 口径下切到 `standard+bundle` 后，整体数值误差明显更小（largest stage `block_output_cls max_abs_error≈14.63`，final logits/probabilities `max_abs_error≈8.044e-2 / 3.862e-2`），但决策反而变成 `0.0/0.0`。这说明当前关键不是“误差越小越好”，而是误差方向是否跨过该样本的分类/阈值 margin。
- `tracka_e2e_secure_depth9_sample1_smoothed_bundle_full_20260425_190634`：full `smoothed+bundle` control 已补出有效 compare。固定同一 `sample1` 输入、`depth9 / bsz=1 / nocoloc / E2E_SPU_ATTENTION_POLICY=smoothed / E2E_SPU_ACTIVATION_OVERRIDE=bundle`，CPU reference 与 SPU candidate 的 `.pt` 已正确分开；compare 仍为 `argmax_match_ratio=0.0`、`threshold_match_ratio=0.0`，`logits/probabilities max_abs_error≈3.70071e-1 / 1.78151e-1`。因此 block9 probe 的 `smoothed+bundle 1.0/1.0` 不能外推到 full depth9，probe 只能用于局部漂移定位，不能替代 full candidate 决策验证。
- 同一组文件级对比进一步确认：probe 的 SPU `final_logits=[-0.0151, 0.9563]`、`argmax=1`，full candidate 的 SPU `logits=[0.7820, 0.0992]`、`argmax=0`，logits/probabilities 最大差约 `0.8571 / 0.3898`。因此 `probe-block` 的 `final_logits/final_probabilities` 当前不能被当作 full candidate 的最终输出；后续要先查 `probe-block` 与 `run` 两条实现路径在 `static_depth_limit=9` 后的 head/norm/return 语义差异。
- `cpu_probe_vs_full_depth9_sample1_20260425_193830`：CPU-only 对照已完成。full CPU `run` 与 CPU `probe-block` 在同一输入、同一 `depth9` 下 logits/probabilities 完全一致（`max_abs_diff=0.0`），argmax 均为 `1`。因此 Python 侧 full/probe 语义一致，当前不一致来自 SPU/JAX 路径：一旦 `probe-block` 额外 reveal 中间张量，SPU 图 / 数值行为会改变，导致 probe 的 `final_logits` 不能代表 full SPU candidate。
- `spu_probe_reveal_perturb_depth9_sample1_20260425_194304`：SPU reveal 扰动审计已完成第一轮。block1 probe 与 full logits 最大差约 `1.34e-3`，block6 probe 与 full 最大差约 `1.68e-4`，二者 argmax 都保持 `0`；但 block9 probe 的 logits 变成约 `[7169.68, 8253.86]`，相对 full 最大差约 `8.25e3`，argmax 翻到 `1`。因此问题不是“所有 probe reveal 都会污染 full 输出”，而更像是 `depth9` 图里 probe 最后一个实际执行 block 时触发了 SPU/JAX 编译/数值扰动。
- 同一目录继续补了 block7 / block8 probe：block7 相对 full logits 最大差约 `2.62e-2`，block8 最大差约 `1.21e-3`，二者 argmax 都保持 `0`；只有 block9 probe 出现 `8.25e3` 量级差异并翻到 `1`。因此当前结论应写成 **last-executed-block probe perturbation**：`static_depth_limit=9` 下 reveal 最后一个执行 block 的中间输出会污染最终 logits，而 reveal 更早 block 基本稳定。
- `spu_last_block_probe_depth8_sample1_fixed_20260425_212620`：`depth8` 泛化验证已完成。固定同一 `sample1` 输入，在 `static_depth_limit=8 / probe_block_index=7` 下，probe logits 与 full logits 最大差仅约 `3.36e-4`，probabilities 最大差约 `6.38e-6`，argmax 均为 `0`。因此“所有最后执行 block probe 都会爆炸”的泛化假设不成立，当前异常应收窄为 **depth9/block9-specific SPU probe perturbation**。
- `spu_probe_pair_depth10_sample1_20260425_214432`：`depth10` probe pair 已完成。full depth10 logits 为 `[0.3971, -0.1138]`、argmax `0`；probe block9（index 8，不是最后执行 block）与 full logits 最大差仅约 `1.11e-3`；probe block10（index 9，最后执行 block）logits 变成约 `[6456.94, -6793.01]`，最大差约 `6.79e3`，probabilities 饱和到 `[1.0, 0.0]`，但 argmax 仍为 `0`。因此更准确的结论是：SPU probe reveal 最后执行 block 时会显著污染 probe 图的 final logits/probabilities；是否翻转取决于污染方向与 margin。
- e2e 隐私边界前移已从 legacy 双 share manifest 推进到 split-manifest 调试桥：`client-share-preprocess` 可同时产出 public manifest 与 `P1/P2` party manifests，`split-debug-share-manifest` 可把旧 manifest 补拆；`run --runtime spu` 现在支持 `--input-share-public-manifest-json`、`--input-p1-share-manifest-json`、`--input-p2-share-manifest-json`，会把 P1/P2 share 分别送入 SPU 图内相加，且不在 runner 里重构 plaintext `pixel_values`。这仍不是最终生产 MPC，因为当前单 launcher 仍能拿到两个 party manifest；下一步应拆成各 party 只加载本方 share 的独立进程 / 启动入口。
- split-share 当前服务器边界：depth0、depth1、depth2、depth3 已通过 same-depth CPU vs SPU 决策一致；depth4 split-share 数值爆炸（SPU logits 约 `[1310.53, -16.06]`，CPU logits 约 `[0.538, 0.047]`），但 plaintext-input depth4 SPU 对照正常（compare logits/probabilities max_abs_error 约 `0.143 / 0.0606`）。下一步不要继续盲跑 depth5+，优先跑新增 `audit-shares`，检查 SPU 内 `share0+share1` 重组与 `patch_embed+pos` 是否已经偏离。
- `audit-shares` 服务器结果已排除输入重组与 patch embedding：plaintext vs CPU reconstructed pixel `max_abs_error=0.0`，CPU vs SPU reconstructed pixel `max_abs_error≈3.02e-5`，CPU vs SPU patch tokens / tokens+pos `max_abs_error≈2.49e-3 / 2.48e-3`，cosine 均约 `0.9999998+`。因此 split depth4 爆炸不在 share 重组或 patch_embed，而应继续用 split-share `probe-block` 定位 block1-4 内部的 attention/MLP stage。
- split-share block4 probe 已回贴：`attn_out_cls` 是最大中间漂移（`max_abs_error≈5.44`、`l2_error≈27.70`、`cosine≈0.0438`），而 `norm1_out_cls` 与 `mlp_out_cls` 仍相对接近；但 probe debug graph 的 final logits 没有复现 non-probe split depth4 的 `≈1.31e3` 爆炸（probe final logits max_abs_error≈0.076）。因此不要用这个 probe 的 final logits 判断 full candidate；下一步优先跑 full split depth4 的 `E2E_SPU_ATTENTION_POLICY=standard` 对照，判断爆炸是否由 smoothed attention/compare 图触发。
- full split-share depth4 的 `standard attention` 对照已回贴并形成正结果：non-probe full run 的 CPU vs SPU compare 为 `logits/probabilities max_abs_error≈0.0762 / 0.02894`，`argmax/threshold=1.0/1.0`，不再出现 smoothed attention 下的 `≈1.31e3` logits 爆炸。因此下一步不要继续 smoothed depth5+；应固定 split-share + `E2E_SPU_ATTENTION_POLICY=standard`，向 depth6/8 做 same-depth smoke，验证该缓解是否能跨过原 depth4 边界。
- split-share + `standard attention` 后续 depth6/8/10/12 已全部回贴通过：depth6 `logits/probabilities max_abs_error≈0.1437 / 0.0435`，depth8 `≈0.2101 / 0.0594`，depth10 `≈0.1964 / 0.0717`，depth12 `≈0.1608 / 0.0630`，四者 `argmax/threshold` 均为 `1.0/1.0`。当前可写成：`sample=1 / public params / static full-depth ViT` 下，split public/P1/P2 debug-share 输入 + SPU whole-forward + final logits reveal 已闭合；仍需明确不是最终生产全隐私，因为 launcher 仍可同时访问两个 party manifest，且 runtime pruning 尚未进入 secure forward。
- 生产全隐私推进的下一步已开始落地：新增 `--party-local-share-load` / `E2E_PARTY_LOCAL_SHARE_LOAD=1`。该模式不再由 driver 读取两个 private share tensor；driver 只调度 public/P1/P2 manifests，P1/P2 device 函数各自读取本方 party manifest 指向的 share 文件，然后在 SPU 内组合。下一步服务器应先用 depth0 和 depth12 standard attention 复验 party-local 模式是否与原 split-share tensor-load 模式一致。
- party-local 输出侧已补隐私边界硬化：`run --runtime spu` 新增 `--redact-private-input-paths`，server wrapper 默认 `E2E_REDACT_PRIVATE_INPUT_PATHS=1`。这样 candidate `.pt` 与 summary JSON 只保留 public manifest provenance，不再持久化 legacy/P1/P2 私有 share manifest 路径；share 输入模式下 wrapper 也不再传 `--input-pt`，避免 candidate metadata 指回 plaintext client pixel package；如果确需本地 debug，才显式设为 `0`。
- party-local depth8 compare 已补齐：`logits/probabilities max_abs_error≈0.2095 / 0.0593`，`argmax/threshold=1.0/1.0`。因此 party-local + standard attention 的稳定边界已明确到 depth8，depth10/12 翻转；下次不要再补 depth8，应直接从 depth10 做定位。推荐先比较 `e2e_static_whole_forward_candidate_spu_depth10_split_share_standard_attn.pt` 与 `e2e_static_whole_forward_candidate_spu_depth10_party_local_standard_attn.pt` 的 logits/probs，再决定是否做 depth10 party-local block probe。
- 用户已回贴新生成 `sample=2` share input 的 depth10 split-share vs party-local 直接对照：`logits_max_abs_diff=0.002685546875`、`probabilities_max_abs_diff=0.0011300444602966309`，两边 `argmax=[0,0]`。因此当前不要再把旧 depth10 party-local 翻转当作必然边界；下一步优先在同一输入上推进 `depth12` split vs party-local，并同时检查 verify JSON 与 redaction metadata。
- 用户继续回贴后，`party-local + standard attention` 的更深边界已经变成 runtime/link 失败：`depth11 / sample=2` 在 `builtin_spu_run` 后长时间卡住，`depth12 / sample=1` 触发 `grpc UNAVAILABLE / Socket closed`，node 日志出现 `SendImpl error ... Not connected to 127.0.0.1:34357`。因此下一位接手者不要继续盲跑 depth12 party-local；当前应先把结论写成：**party-local 全隐私输入路径稳定到 depth10，depth11/12 在当前 CPU jaxlib + SPU runtime 下触发系统稳定性边界**。下一步应保存 depth10 正结果和 depth12 断链证据，再设计 runtime 图拆分 / 分块执行方案。
- 无 reveal 图拆分方案已落地为实验开关：`run --runtime spu --spu-block-chunk-size N` / wrapper 环境变量 `E2E_SPU_BLOCK_CHUNK_SIZE=N`。它按 N 个 transformer blocks 分段执行 SPU 图，中间 token state 仍作为 SPU value 传递，不 reveal 中间张量，只 reveal final logits。下一步优先跑 `depth12 / sample=1 / party-local / standard attention / chunk_size=4` smoke；通过后再试 `sample=2`，不要直接回到 monolithic depth12。
- `chunk_size=4` 已由用户回贴失败：`depth12 / sample=1 / party-local / standard attention` 在首个 `spu(embed_and_blocks_from_shares_fn)` 分段调用即报 `grpc UNAVAILABLE / Socket closed`，node 日志继续出现 `SendImpl error ... Not connected to 127.0.0.1:46019`。下一步不要继续 chunk4，应先试 `chunk_size=1`；如果仍失败，再跑 `depth4/chunk1` sanity，判断 chunked 实现路径是否本身可用。
- 用户继续回贴 chunk1 日志，仍是 `SendImpl error ... Not connected to 127.0.0.1:34253`，且 node1 记录 `builtin_spu_run ... inputs=20 wrapped_shares=2 public_values=18`，说明 `embed + 1 block` 的首段调用也会触发 runtime/link 失败。下一步不要继续 depth12 chunk 重试；应先做 `depth4` paired sanity：同一 party-local 输入下先跑 monolithic depth4，再跑 chunk1 depth4，用来区分基础 party-local 链路不稳还是 chunked 多 SPU call 图形态不稳。
- 用户继续反馈 `depth4 / sample1 / party-local / monolithic` 的 `spu` 命令仍会卡死。因此当前不要继续加深或继续试 chunked；下一步只做 `depth0 / party-local` sanity。如果 depth0 也卡死，就把结论收成“当前 session/runtime 下 party-local 基础链路不稳”，转向 SPU runtime / party device 文件加载 / localhost port-link 诊断。
- `depth0 / sample1 / party-local / standard` sanity 已跑完但 verify 决策不一致：`logits max_abs_error=0.6297486424446106`、`probabilities max_abs_error=0.2999129593372345`、`argmax/threshold_match_ratio=0.0/0.0`。下一步不要再直接加深；应先跑同一输入的 `depth0 split-share tensor-load`，再与 party-local candidate 做直接张量对照，区分是 split-share 输入漂移还是 party-local device-side 文件加载引入偏差。
- 用户回贴 `depth0 split-share tensor-load` vs `party-local` 直接对照：`logits_max_abs_diff=4.57763671875e-05`、`probabilities_max_abs_diff=1.0967254638671875e-05`，两边 `argmax=[1]`、`threshold=[1]`。因此 party-local device-side 文件加载不是首发问题；当前应定位 split-share/SPU depth0 相对 CPU reference 的基础漂移。下一步优先跑同一输入的 `depth0 plaintext-input SPU` 和 `audit-shares`。
- 更正：`depth0 plaintext-input SPU` 回贴里的 verify 仍以 full-depth `static_whole_forward_reference.pt` 为 reference，而不是 same-depth CPU depth0 candidate，因此不能把 `0.6297 / 0.2999` 误差写成 depth0 语义失败。下一步必须先生成 `depth0 CPU candidate`，再用它作为 reference 重新 verify plaintext/split/party-local 三条 SPU depth0 输出。
- same-depth CPU0 reference 已补出第一条正结果：`depth0 party-local SPU` 的 `logits/probabilities max_abs_error≈2.68e-4 / 9.51e-5`，`argmax/threshold_match_ratio=1.0/1.0`。因此 party-local depth0 基础语义正常；下一步补 split-share 和 plaintext-input depth0 的 same-depth verify 即可。
- 同一 CPU0 reference 下，`depth0 split-share tensor-load` 与 `depth0 plaintext-input SPU` 也已通过：split-share `logits/probabilities max_abs_error≈3.14e-4 / 1.06e-4`，plaintext-input `≈2.53e-4 / 8.41e-5`，二者 `argmax/threshold_match_ratio=1.0/1.0`。下一步回到 depth4，但必须先生成 same-depth CPU4 reference，再 verify monolithic depth4 party-local。
- `depth4 CPU candidate` 已生成，但 `depth4 / sample1 / party-local / monolithic` 重跑仍卡死。下一步不要继续 party-local depth4 重试；应跑同一 CPU4 reference 下的 `depth4 split-share tensor-load` 和 `depth4 plaintext-input SPU`，判断 depth4 图本身是否可通过。
- 用户继续反馈 `depth4 split-share tensor-load` 也已经卡死。下一步不要再跑 share 输入 depth4；只跑 `depth4 plaintext-input SPU` 隔离。如果 plaintext depth4 通过，问题集中在双 share 输入路径；如果也卡死，则是 depth4 SPU 图或 runtime 状态本身。
- 用户继续反馈 `depth4 plaintext-input SPU` 也卡死，错误为 `Socket closed`。因此当前不是 share 输入特有问题，而是当前服务器 session/runtime 下 depth4 SPU 图本身触发断链。下一步不要再重试 depth4，回退跑 same-depth `depth3 plaintext-input SPU`，确认当前真实边界。
- `depth3 CPU candidate` 已生成，但用户回贴 `depth3 plaintext-input SPU` 也卡死。当前 session 真实边界已退回为 depth0 可用、depth3+ 断链。下一步不要继续跑 depth3/4/12；只补 `depth1`、`depth2` plaintext-input same-depth smoke，建立最小断点并保存 runtime/link 证据。
- 用户继续回贴 `depth1 plaintext-input SPU` 也会卡死。当前最小断点已明确：depth0 plaintext/split/party-local same-depth 全部通过；进入第一个 transformer block（depth1）即触发 SPU runtime/link 卡死。下一步不要继续 depth sweep；应保存证据，转向 block1 子图拆分或底层 SPU link 稳定性诊断。
- block1 子图定位入口已落地：`transshield_e2e_secure_vit.py block1-subgraph-smoke`，wrapper 模式为 `run_e2e_secure_whole_forward.sh block1-smoke`。它逐段执行并 reveal debug 阶段输出，定位 `patch_pos / norm1 / qkv / attention_context / attention_residual / norm2 / mlp_hidden / block_output / head_logits` 哪一段首次断链；这不是生产 reveal policy。
- 用户继续反馈 `block1-smoke` 也会卡死，错误仍为 `grpc_status:14 / Socket closed`。下一步不要继续往这个 monolithic 文件里追加一次性调试代码；应先做 e2e 文件瘦身，把 stable runtime、debug-only probe、CLI/wrapper 分层，再决定是否继续补更底层 SPU link 诊断。
- e2e 文件瘦身已完成第一轮：主入口 `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` 现在只负责 CLI / command 编排，CPU static reference 在 `cpu_static_vit.py`，SPU/JAX whole-forward 与 share audit 在 `spu_static_vit.py`，block1 debug-only smoke 在 `debug_probe.py`，通用 helper 在 `common.py` / `input_shares.py` / `static_vit_params.py`。CLI 与 wrapper 不变，已通过静态编译和 help 检查。下一步不要再跑 `depth1+` depth sweep；应先做 SPU runtime/link 层的最小健康检查，确认是否是当前 session 的 localhost node/link 稳定性问题。
- 服务器 runtime warmup/check 正常，且拆分后的 `depth0 plaintext-input SPU` 回归已通过 same-depth CPU0 verify：`logits/probabilities max_abs_error≈3.75e-4 / 1.21e-4`，`argmax/threshold=1.0/1.0`。因此当前不是入口拆坏，也不是 SPU 节点起不来。
- 已新增下一阶段最小诊断入口：`runtime-primitive-smoke` / wrapper `runtime-smoke`。它用合成 transformer 形状张量逐段测试 `layer_norm -> qkv -> attention -> projection -> MLP`，不依赖真实图片、share manifest 或模型权重。下一步优先跑它；如果合成 primitive 都在某段断链，问题应转向 SPU/JAX/runtime/link 层，而不是继续改 e2e 输入隐私边界。
- runtime primitive smoke 已回贴：tiny shape `16 tokens / 64 dim / 4 heads / mlp_dim 128` 全阶段通过；ViT shape `197 tokens / 384 dim / 6 heads / mlp_dim 1536` 在 `layer_norm` 阶段直接 `grpc UNAVAILABLE / Socket closed`。因此下一步不要继续 depth1/block1/e2e 输入路径，应只做合成 `layer_norm` 形状阈值扫描，判断断链主要由 token 数、embed 维度还是总元素数触发。
- 后续形状阈值 loop 本身也出现长时间卡死：日志停在 `builtin_spu_run` RPC request 后并出现 `bvar is busy at sampling for 2 seconds`。因此阈值扫描不能复用同一个 runtime 连续跑；下一步如果继续做，只能“一次一个 shape、`SPU_RUNTIME_REUSE=0` 强制重启、`timeout -k` 硬杀、失败后立刻抓日志”。不要再执行多 shape loop。
- 单 shape 诊断继续暴露 runtime/link 半死状态：node 日志出现 `E1008 Reached timeout=20000ms`、`Duplicate ACK`、`bvar is busy`、`UpdateDerivedVars is too busy`，随后 `E112 Not connected`。这已经足够作为停止条件：不要再继续跑 e2e / primitive SPU 实验；当前应清理残留 SPU 进程，把结论写成“当前 CPU jaxlib + localhost SPU runtime 无法稳定承载 ViT 形状 layer_norm/attention primitive”，后续若继续只能换 runtime 环境、调 SPU/yacl/brpc 底层参数，或改成更小 secure 子图/算子替代路线。
- 若继续朝全程隐私推理推进，当前唯一仍值得试的代码侧缓解是新加的 layer-norm feature chunking：`--spu-layer-norm-chunk-size` / `E2E_SPU_LAYER_NORM_CHUNK_SIZE`，runtime-smoke 对应 `--layer-norm-chunk-size` / `E2E_RUNTIME_SMOKE_LAYER_NORM_CHUNK_SIZE`。它不改变 reveal policy，仍在 SPU 内计算 mean/variance，只把 feature 维 sum/sumsq 分块；默认 `0` 保持原图。先只用 runtime-smoke 测 `197x384` + chunk `64`，通过后再考虑 depth1。
- `layer_norm_chunk=64` 仍在 `layer_norm` 断链后，新增下一步隔离开关：`--spu-layer-norm-policy affine` / `E2E_SPU_LAYER_NORM_POLICY=affine`，runtime-smoke 对应 `--layer-norm-policy affine` / `E2E_RUNTIME_SMOKE_LAYER_NORM_POLICY=affine`。它跳过 secret mean/variance，只做 public affine/identity，不 reveal 中间值；用途是判断去掉 secret layer_norm reduction 后，ViT shape 的 qkv/attention/MLP 是否还能跑。它不是最终等价模型，只是继续推进全程隐私路径的 runtime 可承载性验证。
- `layer_norm_policy=affine` 已把断点推进到 `attention_probs`：`layer_norm / qkv_linear / attention_scores` 已能通过，secret softmax 的 `max/exp/sum` 图断链。新增下一步隔离开关：runtime-smoke `--attention-policy uniform` / `E2E_RUNTIME_SMOKE_ATTENTION_POLICY=uniform`，以及 e2e SPU `--spu-attention-policy uniform`。它跳过 secret softmax、使用 public uniform attention，仅用于验证后续 context matmul / projection / MLP 是否可承载；不是最终等价模型。
- `affine LN + uniform attention` 继续在约 `1.86MB` attention/probs fetch 后的下一次 `builtin_spu_run` 卡死，按阶段顺序对应 `attention_context`。已把 uniform context 改成等价 `mean(V)` broadcast，避免 `197x197 @ V` 大 matmul；下一步只复跑同一 runtime-smoke case，判断是否能进入 projection / MLP。
- `uniform attention => mean(V) broadcast` 复跑后，从用户回贴日志看，ViT shape synthetic runtime-smoke 已完成 `projection_residual / mlp_hidden / mlp_residual`，没有再断链。下一步可以尝试真实 e2e `depth1`，但必须使用诊断组合 `E2E_SPU_LAYER_NORM_POLICY=affine` + `E2E_SPU_ATTENTION_POLICY=uniform`，并仍只作为 runtime 可承载性验证，不作为数值等价结果。
- 真实 e2e `depth1` 诊断组合已从日志看跑通：`forward_fn completed`，耗时约 `24.95s`，没有 node/link 断链。下一步先 `cat $E2E_CANDIDATE_JSON` 确认 `finite_logits=true`，然后不要直接加深到 depth12；应先用同一 `affine LN + uniform attention` 组合跑 split/party-local share 输入的 depth1，验证隐私输入路径与 block1 runtime 能同时成立。
- 用户已确认 `depth1`、`depth2`、`depth3`、`depth4` 和 `depth5 / party-local share load / affine LN / uniform attention` 的 candidate metadata 全部满足隐私边界检查：`finite_logits=true`、`input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`input_mode=party_local_debug_share_load`。因此当前最强结论是：party-local 全隐私输入边界已经跨过真实前 5 个 block 的诊断图；driver 不再 materialize 明文像素或私有 share tensor。注意 `depth4` logits 已放大到约 `1e8`，`depth5` 到约 `1e9`，probabilities 均饱和，所以它们是 runtime/privacy-boundary 里程碑，不是有效数值推理结果。用户已明确目标是“实际可以使用的全隐私”，所以下一位接手者不要继续 depth6+ 诊断加深；应转为构建可用全隐私路径：优先实现 public-calibrated layer norm policy（用公开校准集统计替代裸 affine，避免激活爆炸），再替换/近似 secret softmax 与 full attention context，并且先用 CPU same-policy 验证非饱和和决策可用，再回到 party-local SPU。
- 已落地 public-calibrated layer norm 的第一版代码路径：新增 `calibrate-layer-norm` CLI / wrapper `calibrate-ln`，以及 `--spu-layer-norm-policy public_calibrated` / `E2E_SPU_LAYER_NORM_CALIBRATION_JSON`。该路径从非私有/public calibration pixel package 生成每个 LN 位置的 per-feature public activation mean/variance，再作为 public 参数供 SPU forward 使用；它不改变 party-local P1/P2 share 输入边界。下一步服务器先用公开校准输入生成 JSON，再跑 `depth5 / party-local / public_calibrated LN / uniform attention`，检查 logits 是否从 `1e9` 饱和回到可用量级。
- `depth5 / party-local / public_calibrated LN / uniform attention` 已由用户回贴通过：隐私字段全部继续成立，`spu_layer_norm_policy=public_calibrated`，logits 约 `[0.2576, 0.4537]`，probabilities 约 `[0.4511, 0.5489]`，不再出现 affine LN 下的 `1e8/1e9` 爆炸和 `0/1` 饱和。下一步优先做两件事之一：如果要推进可部署近似路径，先用同一 public calibration 尝试 `depth12 / party-local / public_calibrated LN / uniform attention`；如果要补严谨验证，先实现/运行 CPU same-policy reference，与 SPU same-policy 做 logits/probability compare。不要再继续 affine LN 路线。
- `depth12 / party-local / public_calibrated LN / uniform attention` 已由用户回贴：runtime/link 稳定跑完，隐私字段继续全部成立，但 logits 又到约 `2e5..1.5e6`，probabilities 饱和为 `0/1`。因此 full-depth 近似路径还不能作为实际可用分类器。下一步不要继续加样本或改回 affine；应先补 CPU same-policy 对照，判断 full-depth 饱和是否由 uniform attention + public calibration 策略本身导致。如果 CPU same-policy 也饱和，就需要换 attention/activation 近似或先把 depth5 public-calibrated 路径作为可用 prefix baseline。
- 用户随后尝试 `activation_override=gelu` 的 full-depth public-calibrated party-local 路线，SPU 在 `forward_from_shares_fn` 处 `grpc UNAVAILABLE / Socket closed`。因此不要继续 exact GELU；它会把当前 runtime 带回断链边界。若继续推进 full-depth 近似，下一步应试 `fixed_square`（SPU 友好、只有乘加）并重新生成对应 public LN calibration；若仍饱和或断链，则把 `depth5 / public_calibrated / uniform / bundle` 收作当前可用 prefix baseline。
- `depth12 / party-local / public_calibrated LN / uniform attention / fixed_square activation` 已由用户回贴成功：runtime 完整跑完，隐私边界字段全部继续成立，logits 约 `[0.0315, 0.1317]`，probabilities 约 `[0.4750, 0.5250]`，非饱和。当前最强可用近似基线应写为：full-depth static ViT scope、party-local debug share load、public-calibrated LN、uniform attention、fixed-square activation、final-logits-only reveal。注意这不是原始 exact ViT，且 launcher 仍是 colocated debug bridge；下一步应优先做 CPU same-policy reference + SPU candidate compare，或固定该配置做 sample=2/更多样本 smoke。
- 同配置 `sample_count=2 / spu_batch_size=2` 已回贴：隐私字段和 runtime 成功，但 logits 里有样本爆到 `-1.6e6` 量级，probabilities 饱和。因此当前 full-depth 近似只证明 single-sample 可用，不能声称 batched sample2 可部署。下一步优先跑 `sample_count=2 / spu_batch_size=1`，如果 bsz1 通过则问题是 batched graph；如果仍爆，则问题是第二个样本对当前 public calibration/近似策略不稳，需扩大校准集或加 robust variance floor。
- 同配置 `sample_count=2 / spu_batch_size=1` 已回贴通过：两条样本 logits 都是小量级，probabilities 非饱和，`argmax/threshold=[1,1]`。因此 sample2 失稳来自 `spu_batch_size=2` batched graph，而不是第二个样本本身。当前展示/部署命令必须固定 `E2E_SPU_BATCH_SIZE=1`，多样本按逐样本 chunk 顺序处理。
- 当前近似部署基线已集成为 `artifacts/server_inference_friendly_pack/run_e2e_secure_approx_deploy.sh`。下一位接手者应优先使用该脚本，而不是手写长串 env：`make-calib-pixels` 生成公开校准 pixel package，`calibrate` 生成 public LN JSON，`infer` 跑 party-local SPU 推理，`all` 一次执行全流程。脚本内固定并校验 `depth12 / public_calibrated LN / uniform attention / fixed_square / bsz1`，不允许 `bsz2+`。
- `2026-04-27` 已把 e2e 近似评测推进到同条件明文对比：`run_e2e_secure_approx_eval.sh` 会在同一份 image list / targets 上跑 original plaintext reference 与 e2e approximate SPU，并输出 accuracy gap、prediction match、privacy fields 与 SPU link bytes。注意旧脚本曾因 `find | sort | head` 在 `set -euo pipefail` 下触发 exit `141`，已改为先写 temp list 再 `head`；多样本 eval 应设置 `E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`，避免 batched/连续样本图污染。
- 当前 smoke-stable 配置已经固定为 `depth12 / party-local share load / public_calibrated LN / uniform attention / fixed_square / E2E_SPU_ACTIVATION_CLIP_VALUE=3.0 / E2E_OUTPUT_CALIBRATION_JSON / E2E_SPU_BATCH_SIZE=1 / E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`。其中 output calibration 是 public post-reveal calibration，不改变隐私输入边界；服务器 JSON 路径为 `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json`，规则是 `class1_score = logits @ [-3.75, 4.0] - 0.62`，再写成 `[-score/2, score/2]`。
- 已回贴通过的服务器 smoke 结果：`class0_4` 路径 `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_class0_4_uniform_clip3_calibrated_retry_20260427_195218/e2e_secure_poc/e2e_approx_eval_metrics.json`，`class1_4` 路径 `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_class1_4_uniform_clip3_calibrated_retry_20260427_200009/e2e_secure_poc/e2e_approx_eval_metrics.json`，balanced8 路径 `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced8_uniform_clip3_calibrated_20260427_201413/e2e_secure_poc/e2e_approx_eval_metrics.json`。三者均为 same-subset plaintext 100%、e2e 100%、prediction match 1.0、finite logits、隐私字段通过；balanced8 用时约 `736.99s`、SPU link total 约 `1.4526GB`。这是 smoke 闭合，不是 full-val 证明；下一步优先 balanced16，再根据结果决定是否扩大。
- balanced16 已完成并暴露两个后续约束：`e2e_output_calibration_uniform_clip3_smoke8.json` 在 balanced16 上退化到 E2E `81.25%` / match `0.75`；诊断版 calibration `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_balanced16_diag.json` 可把 match 提到 `0.9375`，但一次性 run 中 `i=15` 出现 SPU raw logits 偶发爆炸。fresh-runtime 单样本复跑 `i=15` 后，patched metrics `/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_20260427_211653/e2e_secure_poc/e2e_approx_eval_metrics_patched_i15.json` 为 plaintext same subset `93.75%`、E2E `93.75%`、gap `0.0pp`、match `0.875`。只剩 `i=6` 是稳定近似残差。后续 balanced32 前必须使用 per-sample logits guard/retry，不能把未 guard 的一次性 balanced16 当成稳定结论。

当前下一步不要再试 `identity attention` 或 `clip2.5`：前者 class1 全挂，后者 class0/class1 都退化；也不要继续 `gelu` 线。全隐私主线下一步应在 guarded eval 脚本下复跑 balanced16 或小步到 balanced32，并显式设置 `E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES>=2`、`E2E_APPROX_EVAL_LOGIT_ABS_GUARD=1000`。为避免单样本 SPU runtime 半死卡住整轮 eval，脚本还应使用 `E2E_ISOLATED_INFER_TIMEOUT_SEC` 和 `E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC` 给每个 isolated infer attempt 加 timeout，超时后 fresh-runtime 重试该样本。
- `2026-04-27 23:38` guarded balanced16 仍在 `builtin_spu_run req_bytes≈86MB` 后卡住，说明 monolithic full-depth graph 仍过大，不能只靠 retry/timeout 解决。仓内已有 `E2E_SPU_BLOCK_CHUNK_SIZE` 的 reveal-less block-chunk graph split，下一步应先用单样本 smoke 测 `E2E_SPU_BLOCK_CHUNK_SIZE=3`（必要时降到 `2` 或 `1`），确认 candidate JSON 里 `spu_forward_graph_mode=reveal_less_block_chunked` 且不再卡在 86MB monolithic request，再回到 balanced16/32。
- `E2E_SPU_BLOCK_CHUNK_SIZE=3` 已由用户回贴验证：单样本 `i=15` chunk smoke 成功，candidate 为 `reveal_less_block_chunked`，最大 request 降到约 `22.8MB/21.3MB`，raw/logits 正常。随后 guarded balanced16 chunk3 一次性通过：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_chunk3_guarded_20260427_235545/e2e_secure_poc/e2e_approx_eval_metrics.json`，plaintext same subset `93.75%`、E2E `93.75%`、gap `0.0pp`、match `0.875`、隐私字段通过。这是当前最强非 patched balanced16 结果。
- 当前主要阻塞已转为效率：chunk3 guarded balanced16 `e2e_elapsed_sec≈1474.46s`，用户反馈太慢。下一步不要直接长期跑大样本；应先做效率小实验：比较 chunk size `4` 或 `6` 的单样本和 balanced16 稳定性/req_bytes/elapsed，评估是否能在不回到 86MB monolithic 卡死的前提下降低分段次数；再考虑是否允许有限 `SPU_RUNTIME_REUSE=1` 或批量复用 runtime。另需修复通信统计，当前 metrics 的 `aggregate_total_bytes≈161KB` 只是 latest node log，不是 isolated chunked 全流程总通信。

如果要直接做 block-6 最小归因，优先使用已经接进 `run_e2e_secure_whole_forward.sh` 的 `probe-cpu / probe-spu / probe-compare`，这样 `probe-spu` 会自动拉起 / 复用 SPU runtime，不需要再手动执行 `tools/transshield_spu_runtime_setup.py start`：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
export RUN_NAME=tracka_e2e_secure_spu_depth6_smoke1_20260423
export E2E_DIR="$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc"
cd "$REPO_ROOT"

export E2E_RUN_MAX_SAMPLES=1
export E2E_STATIC_DEPTH_LIMIT=6
export E2E_PROBE_BLOCK_INDEX=5
export E2E_SPU_BATCH_SIZE=1
export E2E_SPU_PARAMS_MODE=public
export SPU_RUNTIME_REUSE=0
export SPU_DISABLE_COLOCATED_OPTIMIZATION=1

bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh probe-cpu
bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh probe-spu
bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh probe-compare

cat "$E2E_DIR/block6_probe_compare_cpu_vs_spu_depth6.json"
```

当前这份 `block6_probe_compare_cpu_vs_spu_depth6.json` 已经能支撑一个更精确的阶段判断：

- `norm1_out_cls` 还比较接近：`cosine≈0.9982`、`max_abs_error≈0.1825`；
- `attn_out_cls` 已是首个明显失真阶段：`cosine≈0.0159`、`l2_error≈35.21`、`max_abs_error≈5.37`；
- `attn_residual_out_cls / block_output_cls` 会继续放大误差；
- 因此当前最可信的工作假设是：**block 6 的首发大漂移来自 attention 输出，不是 norm1 首发，也不像 MLP 首发**。

如果后续 `probe-spu` 或 `spu` 在运行中途再次报 `Socket closed` / `failed to connect to all addresses`，不要回到手动 `start`；先继续用同一个 wrapper，并保持这条单变量稳定性开关：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
export RUN_NAME=tracka_e2e_secure_spu_depth6_smoke1_nocoloc_20260423
export E2E_DIR="$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc"
cd "$REPO_ROOT"

export E2E_RUN_MAX_SAMPLES=1
export E2E_STATIC_DEPTH_LIMIT=6
export E2E_SPU_BATCH_SIZE=1
export E2E_SPU_PARAMS_MODE=public
export SPU_RUNTIME_REUSE=0
export SPU_DISABLE_COLOCATED_OPTIMIZATION=1

bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh spu
```

如果 `nocoloc` 复验仍然中途崩掉，优先抓系统线证据，不要直接下算法结论：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
cd "$REPO_ROOT"

export SPU_NODE_PIDS="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("logs/spu_runtime_ports.json").read_text(encoding="utf-8"))
print(",".join(str(item["pid"]) for item in payload.get("node_processes", []) if item.get("pid")))
PY
)"

ps -p "$SPU_NODE_PIDS" -o pid,ppid,stat,etime,cmd
grep -nE 'Not connected|Socket closed|SendImpl error|ERROR|FATAL|Aborted|Killed|Traceback' logs/spu_nodes/node_*.log
grep -nE 'Not connected|Socket closed|SendImpl error|ERROR|FATAL|Aborted|Killed|Traceback' logs/spu_nodes/node_1.log
```

判定标准：

- `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 后，`tracka_e2e_secure_spu_depth6_smoke1_nocoloc_20260423` 已证明 full run 可以稳定且恢复 `1.0/1.0` 决策一致，因此当前系统线的首个可用缓解就是关闭 colocated optimization；
- 如果后续 `depth=7+` 在相同 `nocoloc` 配置下再次崩溃，则继续沿当前抓日志方式定位新的 runtime 稳定性问题；
- 如果 `nocoloc` 下 full run 继续稳定，但 `probe` 仍稳定复现 attention 首发漂移，则系统线与算法线继续分开记账：**full run 边界推进走 `nocoloc`，内部数值归因继续看 `probe-block`**；
- 如果 `node_1.log` 出现 `Killed / Aborted / Traceback`，优先回贴 `node_1.log` 证据，不要只看 `node_0.log` 的 `Not connected`。

如果只抓这一条 run 回本地，使用：

```bash
export RUN_NAME=tracka_e2e_secure_spu_depth6_smoke1_20260423
mkdir -p /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME
rsync -avP -e "ssh -p 9001" --prune-empty-dirs \
  --include='*/' \
  --include='*.json' \
  --include='*.md' \
  --include='*.txt' \
  --include='*.log' \
  --exclude='*' \
  wyb@10.204.244.1:/data/wyb/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/ \
  /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/
```

说明：

- 这条定向 `rsync` 只回传当前 `RUN_NAME` 下的文本结果，不会把 `*.pt`、`*.pth`、`tb/` 之类大文件拉回本地。
- 当前 `run_e2e_secure_whole_forward.sh spu` 已在服务器完成一轮真实 smoke：`depth=0..5 / sample=1 / public params` 均通过 same-depth CPU 对齐，而 `depth=6..12` 的 **single-sample** run 现已证明在 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 下也可以通过；因此当前不再需要回头补 `depth0`，而应直接从 `tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423` 继续。
- 当前最准确的阶段结论已更新为：`SPU backend scaffolded` 已升级为“**服务器 same-depth smoke 已在 `depth=0..12 / sample=1` 跑通，其中 `depth=6..12` 需要 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 才能稳定通过；但 `depth=12 / sample=2` 仍未闭环**”；但它仍不能写成“整网 SPU 已验证通过”，也仍不包含动态 `masking-pruning` secure forward。
- 当前接手优先级也要同步更新成：**系统线先固定 `nocoloc` 作为 depth6+ 的默认 smoke 配置，然后围绕 `depth=12 / sample=2` 做最小 batch-semantics 归因；算法线继续保留 block-6 attention 漂移的内部归因，但不再把它直接解释成 depth6 final mismatch**。

## 当前已经完成的关键收口

- Web demo 顶部不再误用离线准确率 / AUC 充当当前图片结果。
- 前端已移除：
  - 历史 fastpath 8 样本通信
  - archived SPU profile
  - 相对旧正式展示模型收益
- `docs/` 已删除大量 dated 文档，避免旧数字继续被引用。
- 仓库第二轮瘦身已完成，旧训练产物、旧 server run、旧中间 checkpoint 和 stale results 已清掉。
- `logs/` 也已收口到最小保留集：当前只保留 `logs/spu_nodes/node_*.log`、`logs/spu_runtime_ports.json`，以及仍被当前 blockwise 证据链引用的 `logs/selection_mode_profile/blockwise_vs_flat_20260418_004346/`；clean probe、web demo nohup、SPU 轮转备份和未再引用的旧 selection-mode profile 目录都已清掉。
- `tools/` 也已去掉一批退役脚本：旧 `phase3_lower_tail` 原型/manifest/planner、一次性 `SPU` fastpath logging patch，以及本地 repo audit/cleanup helper 已删除；当前可继续使用的工具名单以 `tools/README.md` 为准。
- `tools/` 中原来分散的 Stage-2 说明 / contract 报告入口也已并到 `tools/transshield_stage2_report.py`：以后通过 `pruning-semantics`、`f-mux-spec`、`policy-spec`、`forward-dataflow`、`tensor-contract`、`secure-kth-contract` 这些子命令生成对应报告，不再维护一串近似同模板的小脚本。
- secure sidecar 相关工具也已继续收口：原来的 `transshield_secure_network_kth_input_export.py`、`transshield_secure_network_kth_manifest.py`、`transshield_secure_network_kth_export.py`、`transshield_secure_network_kth_checker.py`、`transshield_secure_network_kth_branch_eval.py` 已并到 `tools/transshield_secure_network_kth.py`；原来的 `transshield_secure_tie_payload_export.py`、`transshield_secure_tie_payload_checker.py`、`transshield_secure_tie_payload_branch_eval.py` 已并到 `tools/transshield_secure_tie_payload.py`；高频导出入口优先用 `tools/transshield_secure_sidecar_export_suite.py`。
- `configs/openbumblebee/` 现在也只保留 live 配置：`2pc.json`、`2pc.template.json` 与 `README.md`；仓里的时间戳 `.bak.*` 备份快照已清掉。
- 最新一轮全仓审计已继续推进到 `integrations/`：当前这里只保留两条 live bridge；`network-kth bridge` 虽然仍偏长，但复杂度主要来自 CPU/SPU 双 runtime、selection-mode 与 mixed payload 传输的叠加，不宜当成“纯重复脚本”直接拆没。
- `secure_infer/` 已确认只是历史导航 README；现在 secure 代码导航统一看 `docs/architecture.md`、`tools/README.md` 与本文件，不再单独保留该目录。`docs/retention_list.md` 也已同步刷新到当前真实保留集。
- selection-mode profile 需要的最小 runtime inputs 已移到：
  - `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/`
- `margin-aware pruning` 的 `w10` 候选已经形成一条可复用的研究证据链：Stage 2 边界显著拉开，secure 一致性保持 `100%`，但精度暂时不足以替换正式模型。
- `margin-aware pruning` 这轮搜索已基本收口：`w10` 和 `w3 + tok0.02` 作为保留证据，`Stage2-only delayed`、`w2`、`w3 + tok0.04` 作为已验证负结果，不再继续扫这条线。
- TrackA 明文训练根因定位的原始 issue 已完成：现已确认 `ratio loss` 不是 `predictor_1` 过度 pruning 的首发驱动，归档见 `docs/tracka_predictor1_root_cause_2026-04-21.md`；后续不要在该 issue 内继续重复追 `ratio loss`，如要推进应另开非 `ratio` 路径最小单变量验证。
- TrackA 文档与 wrapper 已做一轮清理：后续只把 `docs/history_best_repro_drift_audit_2026-04-21.md` 当作主审计文档；两份 superseded 草稿已删除。
- Web demo 后端与 profile / threshold / single-image / scorecard 工具也已做结构收口：`tools/transshield_chat_demo.py` 现按摘要构造 / state helper / handler parsing 分层，`tools/transshield_selection_mode_profile_report.py` 现按运行产物解析 / compare / Markdown section builder 分层，`tools/transshield_secure_profile_summary.py` 现按日志提取 / communication diagnosis / payload summary 分层，`tools/transshield_threshold_branch_eval.py` 现按 tie 统计 / kth mask 构造 / eval 主循环 分层，`tools/transshield_single_image_comparison.py` 现按 trace report builder / stage panel builder / summary board 组装 分层，`tools/transshield_competition_scorecard.py` 现按输入解析 / 证据拼装 / checklist+outlook / Markdown section builder 分层，`tools/transshield_fastpath_profile_summary.py` 现按日志匹配 / bucket 更新 / 汇总收口 / Markdown 分段分层，`tools/transshield_runtime_branch_compare.py` 现按 summary 归一化 / compare section / recommendation builder 分层；当前改动不改 CLI，只有 token pruning 可视化的 Markdown 输出文件名统一改为 `token_pruning_trace_report.md`，避免重新生成已退役低频文档名。
- 新一轮根目录重复入口审计也已完成：过时的 `PLANS.md` 已删除；`current_work_status.md`、`handoff-next.md`、`algorithm_protocol_upgrade_roadmap.md`、`transshield_blockwise_kth_selection_manifest.py`、`transshield_stagewise_threshold_report.py`、`transshield_network_kth_bridge.py`，以及旧模型镜像 `dyvit.py`、`dylvvit.py` 现在按兼容入口或权威实现边界统一收口到 `docs/`、`tools/`、`integrations/`、`models/`。`training_source_tracka/` 是故意保留的 source/provenance 快照，`training_compat/` 是当前 server 侧 plaintext compatibility runner，`references/original_plaintext_runtime/` 是 baseline runtime 快照；这三处都不要按“重复文件”误删。
- 按最近 train run 产物时间和 handoff 引用频率看，当前最高频脚本主要是：`scripts/run_tracka_train.sh`、`scripts/run_tracka_spu.sh`、`scripts/run_spu_patch_build_probe.sh`，以及 `artifacts/server_inference_friendly_pack/` 里的 server pack wrappers。
- 当前两条高频 TrackA 入口已经进一步收口：`scripts/run_tracka_train.sh` 用 `source|compat` 子命令统一承接原来的两条训练路径；`scripts/run_tracka_spu.sh` 用 `followup|dual-profile` 子命令统一承接原来的两条 SPU 路径；以后优先使用这两个短名字入口。
- 最新一轮 `models/` / `training_*` 审计也已把边界核清：`training_source_tracka/` 不是随手拷贝的废副本，而是 source/provenance 控制路径；`training_compat/` 是当前 server 侧 plaintext compatibility runner；根仓 `main.py` / `models/` 则继续作为 final-repo live 训练 / ablation 实现。后续看训练入口时，先分清自己是在走 TrackA provenance runner，还是在走根仓 live training。
- 三套训练入口现已继续收口为 `deit-s` 单模型训练面：`main.py`、`training_compat/main.py`、`training_source_tracka/main.py` 都不再保留 `convnext/lvvit/swin/deit-b` 的 live 训练分支；deit-s 之外旧模型只继续作为历史资产 / 评估兼容保留。
- 一批低频说明文档也已并回权威入口：数据集说明、正式 bundle provenance、答辩问答 / 展示建议、单图对照 / token 可视化使用说明，以及临时 TrackA agent task pack，现统一看 `docs/data_source_policy.md`、`docs/result_summary.md`、`docs/project_overview_newcomer_defense.md`、`artifacts/server_inference_friendly_pack/README.md`、`docs/current_work_status.md`、`docs/handoff-next.md`。
- `references/original_plaintext_runtime/` 也不是“旧仓残留”：它现在仍被 baseline eval / predict / single-image compare wrapper 直接引用，是 baseline runtime 的最小快照。
- `results/` 这一轮也已完成分层：当前主线仍在用的主要是 `results/blockwise_exact_kth_selection_manifest_default.json`、`results/blockwise_exact_kth_manifest_20260418_004103.*`、最新公平外部对比 `results/fair_external_comparison/fair_external_20260423_113217/`、已完成的 `results/standardized_secure_benchmark/*` 运行，以及少数 margin-aware 主证据目录；payload 相关一串目录虽然名字多，但总体只有约 `248K`，现在更像“小型设计搜索证据”而不是主要清理负担。
- `results/` 这轮也已继续清理：本地误同步进来的空占位 margin-aware 目录、未形成完整 benchmark 汇总的 `standardized_secure_ops_20260417_181138/`，以及未被主文档再引用的本地 `margin_aware_full20_w3w5_20260417_213047/` 已删除；如果文档里仍出现对应 `/data/wyb/...` 路径，应按服务器 provenance 路径理解，而不是要求本地继续保留同名空目录。
- `artifacts/` 顶层当前也已核清：live runtime 核心主要是 `artifacts/baselines/`、`artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/`、`artifacts/inference_ready_config/`、`artifacts/server_inference_friendly_pack/` 与 `artifacts/web_demo_assets/`；`artifacts/server_pipeline_run/`、`artifacts/server_profile_reports/`、`artifacts/train_runs/` 是当前证据链；最大的体量则集中在 `artifacts/archive/` 和 `artifacts/frozen_candidates/` 这种 provenance/候选资产。
- 如果后续继续清 `artifacts/`，优先意识到“大头是少数大文件而不是碎文件”：`archive/` 里基本就是两份完整训练 checkpoint，`frozen_candidates/tracka_lr3e5_timm_best_20260414/` 则主要由 `checkpoint-best.pth` 撑大；但它们目前都还被 provenance / drift audit 文档引用，所以当前先保留。
- `artifacts/` 这轮也已继续做“零外部引用”清理：`train_runs/` 下两条仅本地误同步、且未被仓内文档 / 脚本 / 结果再次引用的 guard-off parity run —— `tracka_compat_debug80_seed0_guardoff_parity_20260422/` 与 `tracka_source_debug80_seed0_guardoff_parity_20260422/` —— 已删除；剩余当前保留目录都还能在交接文档、结果报告或 wrapper 中找到用途。
- `artifacts/frozen_bundle_full/` 与 `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/` 也不是简单重复：两边 key 文件 hash 不同，verified bundle 仍承担当前正式 bundle 的 provenance 收口角色，不要因为看起来“同样 172M”就误删其一。
- `web_demo/`、`configs/`、`references/`、`licenses/` 这批小目录也已核完：`web_demo/index.html` 是当前前端单页实现；`configs/openbumblebee/` 是 live SPU 配置；`references/original_plaintext_runtime/` 是 baseline 最小运行快照；`licenses/` 和 `THIRD_PARTY.md` 是交付许可证材料。它们体量都很小，本轮不删。
- 这类 cleanup / refactor 变更不再单独新开专门变更文档；以后直接看 `docs/current_work_status.md`、`docs/handoff-next.md`、`tools/README.md` 三处即可。
- `Phase 3 network-kth` 已开始推进：仓内新增 `blockwise_exact_kth` 选择模式与 manifest 生成器，CPU smoke checker 已验证通过。
- `Phase 3 network-kth` 已拿到第一轮服务器正结果：`blockwise_exact_kth` 相对 `flat_odd_even` 让 `network_kth_bridge` 从 `11.6141s` 降到 `10.3066s`，总 pipeline 从 `16.9257s` 降到 `15.6245s`，但通信量仍是 `1.72 MB`。
- `Phase 3 network-kth` 的 full replay 也已经通过：`pipeline_inference_replay_summary.json` 中三个 stage 全部 `overall_passed = true`，`exact_count_match_ratio / exact_mask_match_ratio / mean_jaccard_vs_topk` 全为 `1.0`。
- 当前已经确认：`blockwise_exact_kth` 降的是 **SPU 内部 compare-network 时间**，不是进入 SPU 前的 payload 大小；所以 bytes 不降是符合当前实现的。
- `Phase 4 payload` 的第一批诊断已接入：
  - bridge candidate summary 会写 dense/compact payload bytes
  - secure profile 会写 `rpc_total_over_compact_payload_ratio`
  - selection-mode compare 会直接显示 compact payload 与 RPC 放大量
- 正式模型的 mixed payload 已经找到一条当前最佳已知可落地路线：
  - `stage0=float16`
  - `stage1=float32`
  - `stage2=float16`
  - `boundary_window=4`
  - `all_exact_semantics_preserved = true`
  - `total_byte_ratio_vs_float32 = 0.6807`
- `w10` 在线下 payload ablation 中也有新的作用：
  - `all_float16 + boundary_window=4`
  - `all_exact_semantics_preserved = true`
  - `total_byte_ratio_vs_float32 = 0.5315`
  - 它现在主要作为“边界拉开后 mixed payload 更容易成立”的研究支撑线
- 当前正式 secure pipeline / Web demo 默认路径已经切到：
  - `KTH_SELECTION_MODE=blockwise_exact_kth`
  - `PHASE3_SELECTION_MANIFEST=results/blockwise_exact_kth_selection_manifest_default.json`
  - `payload_dtype=float32`
- Web demo 新增：
  - `WEB_DEMO_REUSE_SPU_RUNTIME=1`
  - 默认优先复用已启动的 SPU runtime，只有检查失败时才 fallback 到 restart
  - 这只影响演示时延，不影响离线准确率或 secure 语义
- `network_kth bridge` 新增：
  - 默认 `batched_pyu_bundle`
  - 目标是减少 Python fastpath 下的逐 stage object fetch / response 开销
  - 如需强制回退旧对象路径，可设置：
    - `TRANSSHIELD_SPU_IO_MODE=per_stage`
- `Web demo` 单图 fast path 新增：
  - 默认 `WEB_DEMO_SKIP_PIPELINE_VERIFY=1`
  - 即单图网页运行时跳过逐次 `pipeline verify`
  - replay / diagnosis / profile summary 仍保留，所以页面结果语义不变
- `mixed payload` 负结果不会默认启用，只能通过显式传 `--payload-dtype float16` 触发
- `e2e secure inference` 新线的最小骨架也已进仓：`tools/transshield_e2e_secure_infer.py`、`artifacts/server_inference_friendly_pack/run_e2e_secure_poc.sh`、`integrations/openbumblebee/e2e_secure_vit/README.md` 与 `configs/openbumblebee/2pc_e2e.template.json` 现已存在；它当前只负责边界合同、客户端像素包与 plaintext reference，不代表整网 SPU 已完成。
- `run_e2e_secure_poc.sh` 现还会额外产出 `static whole-forward` plaintext reference：它保留当前 student 的 `patch_embed + blocks + head`，但绕开 runtime pruning 决策路径，作为后续 e2e 整网 SPU 对齐的直接基准。
- 仓内也已预放 whole-forward compare 入口：`tools/transshield_e2e_secure_infer.py compare-static-whole-forward`。后续若有 SPU whole-forward 候选 `pt/json` 产物，可直接对齐 `logits / probabilities / argmax / threshold`。
- `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` 与 `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh` 也已进仓；当前建议顺序是 `prepare -> cpu -> verify` 确认 contract，再用 `spu` 做小样本服务器 smoke。
- `2026-04-22` 的服务器 run `tracka_e2e_secure_poc_cpu` 已完成这条线的第一轮闭环验证：`sample_count=524`，`static_whole_forward_reference` 的 `logits.shape=[524,2]`、`cls_features.shape=[524,384]`、`token_features.shape=[524,196,384]`，`cpu candidate elapsed_sec≈20.73`，`verify` 中 `logits/probabilities max_abs_error=0.0`，`argmax_match_ratio=1.0`，`threshold_match_ratio=1.0`。这条记录的历史意义是：**截至 `2026-04-22`，CPU reference contract 已闭合，而当日 SPU whole-forward backend 仍处于“本地实现、服务器待验证”阶段**。
- `2026-04-23` 本地已补上第一版实验性 `run --runtime spu`；而同一天的服务器回贴也已把这条线推进到真正的 same-depth smoke：在 `sample=1 / public params` 条件下，`depth=0..5` 都已通过“`SPU depth=k` vs `CPU depth=k`”对齐验证，说明静态 DeiT-S whole-forward 的 `patch_embed + 前 5 个 block + head` 在当前 JAX/SPU 子集里可以保持决策一致。
- 但同一轮回贴也明确给出第一处失配边界：`depth=6` 仍然 `finite_logits=true`，却首次出现 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，且 `logits/probabilities max_abs_error` 升到约 `9.29e-1 / 2.64e-1`。因此当前这条线的正确阶段结论是：**`SPU backend` 已完成到 `depth=5` 的服务器 smoke 验证，并在 `depth=6` 首次失配；下一步应做 block-6 附近的数值漂移归因，而不是继续盲目加深或切 `secret params`**。
- TrackA 的 server 运行环境 provenance 已闭合：`/data/wyb/conda_envs/transshield` 当前版本与 `requirements.txt` 完全一致，因此当前不要再把 server env 独立漂移当首发主因。
- TrackA 的 `source` vs `compat` `debug80` parity 也已闭合：`NONEMPTY_KEEP_GUARD=false`、同 seed 下，两侧 `Transform` / `Sampler_train` / scheduler / WD / `Averaged stats` 都一致；`compat` 仅多出默认关闭的 `pruning_margin_*` 参数，所以当前不要再把默认 runner parity 当首发怀疑项。
- TrackA 的 `LOSS_GRAD_ATTRIB=true` 首轮归因也已在 `2026-04-22` 收口：server env provenance、strict source `guard-off` vs `guard-on`、source vs compat `debug80` parity 三项都已闭合；run `tracka_source_epoch3_lossgradattrib_guardoff_seed0_20260422_195751` 显示 `score_predictor.1.out_proj.weight` 的关键 spike 由 `cls_kl` 主导，而不是 `ratio_loss`。下一步若继续推进，不要再追 runner / 默认路径差异，而应另开新 issue 专门决定下一条最小单变量。
- TrackA 的首个单变量 issue 也已在 `2026-04-22` 完成收口：`cls_distill_weight=0.0` 的 `resync1` 已确认是**有效但负向**结果，会把 `predictor_1 empty keep -> zero_active -> predictor_2 non-finite` 提前到 `epoch=2 step=34`；因此不要再继续跑 `clsdw0=0.0`，后续若继续推进应另开新 issue 设计下一条最小单变量。
- TrackA 的“下一条最小单变量” issue 现也已收口：唯一改动 `cls_distill_weight=0.5` 的 strict source / `epoch3` run 没有复现 `clsdw0=0.0` 的 `step=34` 早崩，并在 `epoch=2 step=146` 将 `score_predictor.1.out_proj.weight` 的 total `grad_l2` 从 `4.198726e+03` 压到 `2.400378e-01`、将 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `5.022926e-01`；但 terminal accuracy 仍为 `74.24%`，因此这条结果只记作低精度诊断上的明确缓解，不写成正式修复或新正式成绩。
- TrackA post-`clsdw05` 的下一条最小单变量 `cls_distill_weight=0.75` 现也已收口：它是有效单变量配置，没有命中 `step=34` early-fail gate，并在 `epoch=2 step=146` 将 total `grad_l2` 从 `4.198726e+03` 压到 `3.748181e+01`、将 `cls_kl grad_l2` 从 `4.041559e+03` 压到 `2.105142e+01`、将 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `4.730873e-01`；但 terminal 仍为 `74.24%`，且缓解强度弱于 `clsdw05=0.5`。因此这条结果也只记作低精度诊断上的正结果，不写成正式修复或新正式成绩。
- TrackA post-`clsdw075` 的新阻塞点单变量已完成服务器回贴分析：固定 `cls_distill_weight=0.5` 后把 `token_distill_weight` 从 `0.02` 上调到 `0.04` 是有效配置且未早崩；它让 `epoch3` terminal 从 `74.24%` 松动到 `79.01%`，但在 `epoch=2 step=146` 明显放大 `score_predictor.1.out_proj.weight` 梯度并压低 `predictor_1` keep。因此这条结果只能记作**terminal 正向、稳定性负向的混合诊断结果**，不能直接推进 `full20` 或写成正式修复。
- TrackA terminal-稳定性解耦单变量 `token_distill_weight=0.03` 已在 `2026-04-23` 收口：它是有效配置且未早崩，但 `step=146` total `grad_l2` 从 `2.400378e-01` 增至 `1.165204e+00`，`active_margin_mean` 从 `-3.137406e-02` 变为 `-1.206955e-01`，terminal 只从 `74.24%` 到 `74.43%`。因此 `0.03` 没有解耦成功：降低剂量会削弱 `0.04` 的稳定性伤害，但也几乎带走 terminal 提升；当前仍不建议直接进入 `full20`。
- TrackA post-`tdw003` 的下一条最小单变量 `ratio_weight: 2.0 -> 3.0` 也已在 `2026-04-23` 收口：它是有效配置且未早崩；在 `cls_distill_weight=0.5 / token_distill_weight=0.04` 底座上，keep ratio 与 active margin 确实回升，但目标参数 total `grad_l2` 从 `4.365869e+00` 放大到 `5.206851e+01`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果是**负结果**，不是 clean 解耦，也不能写成 mixed 修复或推进 `full20`。
- TrackA post-`rw3` 的最小单变量 `cls_distill_weight: 0.5 -> 0.4` 也已在 `2026-04-23` 收口：它是有效配置且未早崩；`cls_kl grad_l2` 从 `2.991327e+01` 降到 `4.482888e+00`，但 total `grad_l2` 从 `4.365869e+00` 升到 `7.772295e+00`，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 降到 `2.575499e-01`，`active_margin_mean` 从 `-1.643516e-01` 恶化到 `-1.041453e+00`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果是**有效但负向的非 clean 解耦**，不是修复，也不能推进 `full20`。
- TrackA post-`clsdw04` 的 midpoint 单变量 `token_distill_weight: 0.04 -> 0.035` 也已在 `2026-04-23` 收口：它是有效配置且未早崩；`step=146` total `grad_l2` 从 `4.365869e+00` 降到 `2.089920e+00`，`cls_kl grad_l2` 从 `2.991327e+01` 降到 `9.532135e-01`，`token_kl grad_l2` 从 `1.172528e+00` 降到 `2.397546e-03`，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 回升到 `4.878815e-01`，`active_margin_mean` 从 `-1.643516e-01` 回升到 `-8.497284e-02`；但 terminal 从 `79.01%` 回落到 `74.24%`。因此它是**稳定性缓解、terminal 丢失的负向 midpoint 结果**，不是 clean 解耦，也不能推进 `full20`。
- TrackA post-`tdw0035` 的近端 midpoint 选择已在 `2026-04-23` 完成服务器回贴分析：唯一近端候选 `token_distill_weight=0.0375` 是有效配置、未早崩，并在 `epoch=2 step=146` 把 `predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 提到 `5.085371e-01`、把 `active_margin_mean` 从 `-1.643516e-01` 拉到 `-2.972232e-03`、把 total `grad_l2` 从 `4.365869e+00` 压到 `1.294038e-01`、把 `cls_kl grad_l2` 从 `2.991327e+01` 压到 `1.165144e-01`、把 `token_kl grad_l2` 从 `1.172528e+00` 压到 `1.486021e-03`；但 terminal 仍从 `79.01%` 直接回落到 `74.24%`。因此当前应正式收口为 **`stop_token_midpoint`**：`0.03 / 0.035 / 0.0375` 三个 `<0.04` 的点都没有保住 `79.01%`，说明近端 token 轴的信息增益已不足，不再继续做 `0.04` 附近的剂量搜索，也不推进 `full20`。
- 前一轮本地交接维护已完成：`rw3`、`clsdw04` 与 `tdw0035` 后续最小单变量均已完成回贴分析并同步到权威文档；这个交接已被 post-`tdw0035` 近端 midpoint issue 接住，并已由用户回贴的 `token_distill_weight=0.0375` 服务器结果收口。当前仍没有“terminal 提升 + 稳定性无变化 / 缓解”的 clean 解耦迹象；这个维护动作不改变 e2e secure inference 线，e2e 仍按上方独立段落继续。
- 第二轮脚本整理已继续推进：`scripts/run_tracka_spu.sh`、`scripts/run_spu_patch_build_probe.sh`、`artifacts/server_inference_friendly_pack/run_margin_aware_pruning_ablation.sh` 现在按阶段函数化；`tools/transshield_inference_friendly_server_pack.py` 已拆成命令构造 / shortcut 生成 / manifest 写出三层，`tools/transshield_openbumblebee_pipeline.py` 已拆成 step builder / replay helper / CLI parser 三层，`tools/transshield_openbumblebee_inference_replay.py` 与 `tools/transshield_fair_external_comparison.py` 也已拆成分层 helper；根目录 `handoff-next.md` 与 `run_secure_selection_mode_profile_compare.sh` 改为兼容入口，真正内容与实现统一收口到 `docs/` 和 `artifacts/server_inference_friendly_pack/`。

## 当前必须遵守的规则

- **当前图片结果** 只能来自运行时 API。
- **离线最佳成绩** 只能来自 `artifacts/web_demo_assets/best_demo_content.json`。
- **外部比较** 当前只默认使用 `MPCViT` 同数据集明文基线。
- **统一 secure benchmark** 现在有独立脚本入口，但必须单独标注 benchmark 口径，不能写成单图 live run 或 full-val sidecar 通信量。
- **margin-aware ablation** 只能写成研究性证据，不能写进当前主展示成绩表，也不能替换 Web demo 默认 bundle，除非后续重新跑出更高精度版本。
- **`e2e secure inference`** 当前仍是并行研究线；在最小 POC 跑通前，禁止把当前系统表述为“原始 X 光输入从进入系统起即全程加密”。

## 明确不要再用的数字

- `1.90 MB`
- `979.9903s`
- `975.1174s`
- `3.21 GB`

如果在别的文件里再次看到这些数字，默认视为过期错误口径。

## 当前主要路径

- 前端页面：`web_demo/index.html`
- 前端摘要数据：`artifacts/web_demo_assets/best_demo_content.json`
- Web demo 后端：`tools/transshield_chat_demo.py`
- 摘要生成器：`tools/update_web_demo_summary.py`
- 当前安全 pipeline 正式入口：`tools/transshield_openbumblebee_pipeline.py`
- 根目录兼容入口：`transshield_openbumblebee_pipeline.py`
- margin ablation 入口：`artifacts/server_inference_friendly_pack/run_margin_aware_pruning_ablation.sh`
- margin ablation 汇总器：`tools/transshield_margin_ablation_report.py`

## `w10` 这条线已经确认的结论

- 候选 bundle：`artifacts/frozen_candidates/margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`
- 服务器报告目录：`/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_aware_full20_w10_20260417_205242`
- 效果：
  - Argmax 准确率：`88.9313%`
  - Threshold 准确率：`90.2672%`
  - AUC：`0.956508`
- Stage-wise 风险变化：
  - Stage 1 margin：`1.368x`
  - Stage 2 margin：`243.532x`
  - Stage 3 margin：`2.986x`
  - Stage 2 `<=1e-4`：`98.66% -> 5.92%`
- secure 检查：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
  - `logits_max_abs_error = 0.0`
  - `probabilities_max_abs_error = 0.0`

一句话判断：**这条线证明“把 pruning boundary 拉开”是对的，但当前全局统一权重太伤精度，下一步必须改成更局部、更晚启用。**

## 当前还要一起记住的另一条证据

- 候选 bundle：`artifacts/frozen_candidates/margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4_bundle`
- 服务器报告目录：`/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_formal_hparams_soft_stage2_20260417_231946`
- 效果：
  - Argmax Acc：`85.1145%`
  - Threshold Acc：`91.6031%`
  - AUC：`0.967476`
- Stage-wise 风险变化：
  - Stage 2 margin：`20.032x`
  - Stage 2 `<=1e-4`：`98.66% -> 42.56%`
  - Stage 3 margin：`3.001x`

一句话判断：**这条不是最强协议证据，但它是当前最好的精度 / 协议友好性折中证据。**

## 如果下一步要继续做什么

### 想继续改前端

- 先看 `docs/web_chat_demo.md`
- 再看 `docs/data_source_policy.md`

### 想更新离线成绩

- 先更新生成成绩的结果文件
- 再更新 `artifacts/web_demo_assets/best_demo_content.json`
- 最后检查前端文案有没有把离线成绩写成当前图片结果

### 想补新的 secure 对比

- 必须保证和对比对象同输入、同样本量、同协议口径
- 否则不要把数字放到主页面
- 如果只是想做“同一个 secure benchmark harness 下”的外部 proxy 对比，运行：
  - `artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh`
  - 输出在：`results/standardized_secure_benchmark/<run>/`

### 想重跑公平外部对比

- 运行入口：`artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh`
- 报告输出：`results/fair_external_comparison/<run>/fair_external_comparison.json`
- 使用前先确认：
  - `TRAIN_DATA_PATH`
  - `VAL_DATA_PATH`
  - `BUNDLE_DIR`
  - `MPCVIT_SEEDS`
- 只有当报告中的 `accuracy_comparison_is_fair = true` 时，才允许把这组结果写进外部对比主表。

### 想继续引用 `margin-aware pruning`

现在不要再默认继续扫这条训练线，而是按下面方式引用：

1. `w10`
   - 作为最强研究性正结果
2. `w3 + tok0.02`
   - 作为当前最佳折中证据
3. 负结果
   - `Stage2-only delayed`
   - `w2`
   - `w3 + tok0.04`
   - 这些说明继续靠扫训练超参，短期内不太像能逼近正式默认模型

想看完整记录，直接读：

- `docs/margin_aware_pruning_notes.md`

### 当前非 `e2e` 线最新状态

本轮已完成：

1. 重跑 `公平外部对比`
2. 修正 `MPCVIT_SEEDS` 必须使用空格分隔的问题
3. 生成最新公平报告：
   - `results/fair_external_comparison/fair_external_20260423_113217/fair_external_comparison.json`
   - `results/fair_external_comparison/fair_external_20260423_113217/fair_external_comparison.md`
4. 补跑最新 `统一 secure benchmark`：
   - `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_130435/standardized_secure_benchmark.json`
   - `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_130435/standardized_secure_benchmark.md`
5. 补跑 `same_shape_operator_proxy`：
   - `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_132121_same_shape/standardized_secure_benchmark.json`
   - `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_132121_same_shape/standardized_secure_benchmark.md`

当前结论：

- `accuracy_comparison_is_fair = true`
- `Transshield`: Argmax `93.702292%`，Threshold `94.083971%`，AUC `0.972313`
- `MPCViT` seed `1 2` mean: Argmax `96.660305%`，Threshold `96.946565%`，AUC `0.993449`
- 差距保持：Argmax `-2.958013 pt`，Threshold `-2.862594 pt`，AUC `-0.021137`
- `Transshield` 报告里的 `rpc_total_bytes = 10020639`
- `standardized_secure_benchmark_20260423_130435` 中，`architecture_proxy` 对比得到：
  - `Transshield 当前最终模型 proxy`: `4422.72 MiB / 13.4821s`
  - `MPCViT vit_7_4_32 proxy`: `262.56 MiB / 4.5622s`
  - 比值约：通信 `16.8447x`，时间 `2.9552x`
- `standardized_secure_benchmark_20260423_132121_same_shape` 中，`same_shape_operator_proxy` 对比得到：
  - `Transshield secure-friendly ops same-shape proxy`: `881.05 MiB / 8.1045s`
  - `External baseline ops same-shape proxy`: `5918.69 MiB / 15.3365s`
  - 比值约：通信 `0.1489x`，时间 `0.5284x`
- 两条 benchmark 合起来说明：
  - **结构尺度**会让 `Transshield` proxy 在 `architecture_proxy` 口径下明显更重；
  - 但**secure-friendly 算子替换本身**在固定形状下是明显正向的，不是额外负担；
- 这条线的实质价值是验证本项目的算法/算子替换在 secure transformer benchmark 里确实有效，而不是重新宣称正式 pipeline 通信下降；
- 它和当前历史最优正式模型的关系是：
  - `architecture_proxy` 使用当前正式 bundle 的结构口径（`DeiT-S / 224px / 197 tokens / 12 layers / hidden=384`）做 proxy；
  - `same_shape_operator_proxy` 固定同一 DeiT-S 形状，只替换 secure-friendly ops；
  - 两者都不加载历史最优权重、不跑验证集、不代表 full-val image pipeline 或 Web demo live run；
- 这组 benchmark 数字只代表**同一 MPCFormer local 2PC harness** 下的 proxy 开销，不是 full-val 医学图像 pipeline，也不是网页单图 live run
- 这些数值已经写入 `artifacts/web_demo_assets/best_demo_content.json`：公平外部对比的 `source_run` 已更新为 `fair_external_20260423_113217`，统一 secure benchmark 区也从 `not_run` 改为 `available`

如果还要继续非 `e2e`，下一步更推荐：

1. 如需继续补更细粒度协议证据，可再看是否值得新增模块级说明，但当前主结论已经足够闭合
2. 如需改前端展示文案，只改 `artifacts/web_demo_assets/best_demo_content.json` 和相关说明，不再追加新 benchmark 变体

前端静态 JSON 已更新，因此这条非 `e2e` 外部对比 / benchmark 线已经可以视为正式完成；后续不要继续默认追加 benchmark 变体。

不要做：

- 不要继续在原 TrackA issue 内扩题；
- 不要继续扩 `Phase 4 payload` mixed 配方搜索；
- 不要把旧 profile 数字混进外部对比主表。

### 如需重跑公平外部对比

建议直接在服务器同一 shell / tmux 会话里运行：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
mkdir -p /data/wyb/tmp
export TMPDIR=/data/wyb/tmp
export TMP=/data/wyb/tmp
export TEMP=/data/wyb/tmp

export TRAIN_DATA_PATH=/data/wyb/pneumoniamnist_imagefolder_subset/train
export VAL_DATA_PATH=/data/wyb/pneumoniamnist_imagefolder_subset/val
export BUNDLE_DIR="$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414"

run_fair=fair_external_$(date +%Y%m%d_%H%M%S)
export RUN_NAME="$run_fair"
export FAIR_OUTPUT_DIR="$REPO_ROOT/results/fair_external_comparison/$RUN_NAME"
export SECURE_RUN_DIR="$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}_transshield"

export MPCVIT_SEEDS="1 2"
export MPCVIT_DEVICE=cuda

cd "$REPO_ROOT"
bash artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh

grep -nE '"accuracy_comparison_is_fair"|\"argmax_accuracy\"|\"threshold_accuracy\"|\"auc\"|\"rpc_total_bytes\"' \
  "$FAIR_OUTPUT_DIR/fair_external_comparison.json" || true
```

回贴时优先给：

- `results/fair_external_comparison/<run>/fair_external_comparison.json`
- `accuracy_comparison_is_fair`
- `argmax_accuracy`
- `threshold_accuracy`
- `auc`

只有当 `accuracy_comparison_is_fair = true` 时，才允许把新结果写回外部对比主表或前端静态展示。

### `Phase 3 network-kth` 当前新增入口

- manifest 生成器：
  - `tools/transshield_blockwise_kth_selection_manifest.py`
- 新模式：
  - `blockwise_exact_kth`
- 已接入：
  - `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
  - `tools/transshield_openbumblebee_pipeline.py`
  - `artifacts/server_inference_friendly_pack/run_secure_selection_mode_profile_compare.sh`

当前已经验证：

- 在本地 `smoke8` 输入上，`blockwise_exact_kth` 的 checker 结果 `overall_passed = true`
- 逐 stage `kth_threshold` 与 reference `max_abs_error = 0.0`

现在已经补到服务器 profile 结果：

- `network_kth_bridge`
  - `11.6141s -> 10.3066s`
  - `0.887x`
- `total pipeline`
  - `16.9257s -> 15.6245s`
  - `0.923x`
- `communication total bytes`
  - `1.72 MB -> 1.72 MB`

因此当前最小下一步不是继续改 `blockwise` 代码，而是：

1. `blockwise_exact_kth + float32` 已经作为正式 secure pipeline / Web demo secure run 默认路径；
2. `Phase 4 mixed payload` 只保留为诊断证据，因为 host 侧压缩有效，但当前真实 SPU pipeline 没有带来端到端通信 / 时间收益。

### 如果下一步要开 `e2e secure inference` 新线

当前建议把它当作**并行研究线**，而不是直接替换当前 `blockwise_exact_kth + float32` 正式路径。

先记住三条硬边界：

1. 当前项目还**不能**写成“从加密 X 光输入开始的端到端密文推理已经完成”；
2. 当前正式展示默认路径仍然是 `secure sidecar + replay + compare`；
3. 新线启动时，不要直接改坏 Web demo / 正式 secure pipeline 的现口径。

如果下次要正式开工，这条线建议按下面顺序做：

1. **先做最小 POC**
   - 客户端本地预处理图像；
   - secret-share `pixel_values`；
   - 先做**不含 pruning** 的整网 `deit-s / ViT` secure inference；
   - 非调试模式默认只 reveal 最终 `logits / label`。
2. **再迁入当前 `dyvit` 的 masking-pruning 语义**
   - 核心是把 `pred_score -> prev_decision -> policy -> block forward` 迁进 secure 前向；
   - 继续坚持 fixed-shape / `masking` 表达，不回退到动态删 token。
3. **最后再接当前协议优化成果**
   - 把 `margin-aware`、`blockwise_exact_kth`、payload 诊断的有效结论接进 e2e 路线；
   - 再单独统计 e2e 口径下的时延 / 通信。

这条线还有两条明确提醒：

- 不要直接照搬 `OpenBumbleBee/flax_vit` 当最终实现；它更适合借鉴“整段前向进 SPU”的边界设计，而当前仓库主线仍是 `PyTorch / DynamicViT / masking`。
- 如果真开始做，优先补的是独立入口和独立 checker，而不是直接重写现有 `tools/transshield_openbumblebee_pipeline.py` 默认路径。

### `w10` 在下一阶段还有没有用

有，而且仍然很有用，但**用途已经变了**：

1. 不是拿来替换正式展示模型
2. 而是保留为：
   - `Phase 2` 最强协议友好性正结果
   - Stage 2 boundary crowding 的最强反例
   - 后续若做“局部化 pruning policy / payload 局部压缩 / stage-aware protocol design”时的目标参考

一句话说：

- `Phase 4` 的主线现在用当前正式模型继续推进；
- `w10` 留作“如果把 Stage 2 边界继续拉开，协议侧还能吃到多少红利”的研究标尺。

### 明文训练这条线下一步别再跑偏

刚补齐的 provenance 结论：

- 官方 best：
  - `artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414/manifest.json`
  - 明确记录：
    - `model_ema=false`
    - `activation_lr_scale=10.0`
    - `crop_pct=0.875`
    - `weight_decay_end=0.05`
- 最近服务器侧 `ema_only_epoch1/5/20` 复现实验：
  - `model_ema=true`
  - `activation_lr_scale=1.0`

因此：

- 这批 `EMA-only` 结果不能当作“相对官方 best 的单变量 EMA 结论”；
- 它们只能说明当前那套**非官方配方**下，精度上不去；
- 下一步必须先恢复官方 recipe，再做单变量比较。

### 明文重训当前最新排障点

最新日志确认：

- `training_source_tracka` 原始 source 路径 guard-off 也会在 `predictor_2_pred_score` 处出现 non-finite；
- 异常前 `PredictorLG global_x / post_agg / out_conv_out` 已经含 non-finite；
- 根因是上一阶段 `hard_keep_decision` 对部分样本变成全 0，导致下一阶段聚合时除 0；
- 这说明问题不是单纯 `training_compat` 代码漂移，而是当前重训路径存在 zero-active token 风险。

已补齐：

- `training_source_tracka/main.py` 新增 `--nonempty_keep_guard`；
- `training_source_tracka/models/dyvit.py` 新增同 `training_compat` 一致的可选 non-empty keep guard；
- `scripts/run_tracka_train.sh source` 新增 `NONEMPTY_KEEP_GUARD` 环境变量。

source guard-on `epoch5` 已验证：

- 不再 non-finite 崩溃；
- 但 argmax 仍为 `74.236643%`，threshold 仅 `75.381678%`；
- 首个关键分叉是 `score_predictor.1.out_proj.weight` 在 `epoch=2 step=146` 梯度爆到 `1.057964e+03`；
- `empty_keep` 从 `epoch=3 step=0` 开始出现，主要集中在 `predictor_2`；
- 这说明 guard 只是防崩溃，不能自动恢复官方好轨迹。

已新增诊断：

- `training_source_tracka/models/dyvit.py` 会打印 `predictor_1/2_keep_diag`；
- `training_source_tracka/losses.py` 会打印 `ratio_stage_i`；
- `scripts/run_tracka_train.sh source` 新增 `epoch3` 模式。

这三项 closure 现在已经收口：

1. strict source `guard-off` vs `guard-on` 对照已完成；
2. `training_source_tracka` vs `training_compat` 的 `debug80` parity 已完成；
3. `LOSS_GRAD_ATTRIB=true` 首轮非 `ratio` 分项归因已完成，结论是 `cls_kl` 主导 `epoch=2 step=146` 的 `predictor_1` 关键 spike。

下一步不要再重复追 runner / 默认路径差异；如果继续 TrackA，应另开新 issue 专门决定下一条最小单变量，而不是继续复跑 `clsdw0=0.0` 或在当前 issue 内扩题。

后来继续对比代码后，又确认一个真正的语义漂移点：

- `training_compat/models/dyvit.py`
  - 曾默认启用训练时 `single_token fallback`
  - 以及 `PredictorLG` 的 zero-active clamp guard
- 这两处不在历史 `DynamicViT_exp_square` 的原始 best 训练路径里

这个问题现在已经处理为：

- 新增显式参数：
  - `nonempty_keep_guard`
- 默认：
  - `false`
- 服务器运行脚本也已显式暴露：
  - `NONEMPTY_KEEP_GUARD=false`

所以当前如果目标是**严格复现官方 best provenance**，默认就不要开 guard。

后来还确认一个会误导短验证的调度问题：

- 旧版 `epoch1` / `epoch5` 模式会直接改：
  - `epochs=1` / `epochs=5`
- 这样 Cosine LR / WD 调度也会一起被压缩
- 所以那种跑法不能拿来和官方 20-epoch 配方的前几轮直接对比

现在这个问题已经修正为：

- `epoch1` / `epoch5` 默认仍然：
  - `epochs=20`
- 但通过：
  - `stop_after_epoch=1` / `stop_after_epoch=5`
  - 提前结束训练

如果下次看到前几轮的学习率明显比官方快掉很多，优先怀疑是不是又用了“压缩总 epoch”的旧跑法。

当前已经把服务器入口收口到：

- `scripts/run_tracka_train.sh compat`

这个脚本现在已经支持：

- `debug80`
- `epoch1`
- `epoch5`
- `full20`
- 并且会显式传：
  - `activation_lr_scale`
  - `crop_pct`
  - `weight_decay_end`
  - `model_ema`

所以下次如果要继续这条线，顺序固定为：

1. 先跑官方 recipe：
   - `MODEL_EMA=false`
2. 再跑真正单变量 EMA：
   - 其他参数不变
   - 只切 `MODEL_EMA=true`

最新补充：

- 官方 frozen bundle 在当前 `/data/wyb/pneumoniamnist_imagefolder_subset/val` 上仍能复现：
  - `93.702292 / 94.083971 / 0.972332`
- `/data` 侧 train / val 类别计数也与历史记录一致：
  - train：`1214 / 3494`
  - val：`135 / 389`

因此下一步不要再怀疑前端展示、验证集或 evaluator；重点转为隔离训练栈。

当前已经新增：

- `training_source_tracka/`
- `scripts/run_tracka_train.sh source`

用途：

- 在 `Transshield_final` 内运行一份接近 `DynamicViT_exp_square` 原始训练入口的快照；
- 与 `training_compat` 分开，专门用于同机同环境判断“训练栈漂移还是运行环境漂移”。

### TrackA server env 这条怀疑先放下

`2026-04-22` 已核对 `/data/wyb/conda_envs/transshield/bin/python` 与 `requirements.txt` 完全一致：

- `python 3.9.25`
- `torch 1.13.1+cu117`
- `torchvision 0.14.1+cu117`
- `timm 0.3.2`
- `numpy 1.26.4`
- `cuda 11.7`
- `cudnn 8500`

因此：

- 当前不要再把 server env 独立漂移当作首发主因；
- strict source `guard-off` vs `guard-on`、source vs compat `debug80` parity、以及 `LOSS_GRAD_ATTRIB=true` 首轮非 `ratio` 分项归因均已闭合；
- 下一步应另开新 issue 专门决定下一条最小单变量，而不是继续追 runner / 默认路径差异或补跑 `clsdw0=0.0`。

当前 issue 已贴回 `tracka_source_epoch3_sched20_guardon_seed0_gpu3` 与 `tracka_source_epoch3_sched20_guardoff_seed0_gpu3` 两侧摘录，现已确认：

- 两侧 `Namespace(...)` 唯一有效差异仅是：
  - `nonempty_keep_guard=True` vs `False`
- 关键窗口 `epoch=2 step=140~146` 的提取字段在两边同轨：
  - `predictor_1/2_keep_diag`
  - `ratio_stage_0/1/2`
  - `grad_watch parameter=score_predictor.1.out_proj.weight`
- 两侧都在 `step=146` 出现：
  - `predictor_1 final_keep_ratio_mean=1.548948e-01`
  - `ratio_stage_1 pos_ratio_mean=6.042730e-02`
  - `ratio_stage_2 pos_ratio_mean=2.822066e-02`
  - `grad_watch parameter=score_predictor.1.out_proj.weight = ±1.057964e+03`
- 两侧三次 eval 的 `Accuracy / Max accuracy` 都停在 `74.24%`
- 两侧 `epoch=3 step=0~5` grep 都为空，这不是异常，因为 run 在：
  - `Early stop after epoch 2 due to stop_after_epoch=3`
  - 后结束，本来就不会产生 `epoch=3` 训练步

这说明：

1. strict source `guard-off` vs `guard-on` 的 `epoch3` 对照现在已经闭合；
2. guard 不是 `epoch=2 step=146` 之前的首发分叉点；
3. 下一步可以把注意力继续放在非 `ratio` 梯度归因，而不是继续怀疑 guard 本身。

另外，`training_source_tracka` vs `training_compat` 的 `debug80` parity 也已闭合：

1. 两侧 `Transform` / `Sampler_train` / `Use Cosine LR scheduler` / `Max WD` 完全一致；
2. 两侧 20 条 `Averaged stats: lr:` 数值逐项重合；
3. `Namespace(...)` 的文本差异仅剩：
   - source 侧 `crop_pct=None`、`weight_decay_end=None`
   - compat 侧显式写成 `crop_pct=0.875`、`weight_decay_end=0.05`
   - compat 侧额外打印默认关闭的 `pruning_margin_*`

这说明当前问题不再像是默认 runner 分叉；如果后续继续追 TrackA，更合理的入口是另开 issue 做共享训练路径内部的非 `ratio` 归因，而不是继续追 source/compat parity。

### TrackA `LOSS_GRAD_ATTRIB=true` 首轮归因结论

`2026-04-22` 当前 gate checklist：

- server env provenance：`Yes`
- strict source `guard-off` vs `guard-on` 同口径证据：`Yes`
- source vs compat `debug80` parity 同口径证据：`Yes`
- runner / 默认路径差异仍是主干扰：`No`
- `LOSS_GRAD_ATTRIB=true` attribution：`Yes`

本轮 strict source、`NONEMPTY_KEEP_GUARD=false`、`epoch3` 的 attribution run 为：

- `tracka_source_epoch3_lossgradattrib_guardoff_seed0_20260422_195751`
- `epoch=2` 的 `Averaged stats` 已复现 `grad_norm: 924.6571 (239.6665)`
- `epoch=2 step=146`：total `grad_l2=4.198726e+03`、`grad_absmax=1.057964e+03`
- 同步分项：`cls_kl grad_l2=4.041559e+03`、`grad_absmax=1.017747e+03`
- 对照分项：`ratio_loss grad_l2=3.899971e-01`、`cls_loss grad_l2=1.036839e+02`、`token_kl grad_l2=5.404455e+01`

因此当前建议：

1. `ratio_loss` 继续排除为 `predictor_1` 首发驱动；
2. 首个单变量 `cls_distill_weight=0.0` issue 已完成，并已形成有效负结果；
3. 不在本 issue 内直接改训练语义；后续若另开 issue，仍保持 strict source、`NONEMPTY_KEEP_GUARD=false`、`epoch3`，不跑 `full20`，不把低精度诊断写成正式成绩；
4. 下一步不是继续复跑 `clsdw0=0.0`，而是另开新 issue 专门决定下一条最小单变量。

### TrackA 首个单变量命令包与结果（`2026-04-22`）

本轮首个且仅一个单变量验证的历史选择为：

- `cls_distill_weight: 1.0 -> 0.0`

原因：

- `epoch=2 step=146` 的 total spike 已由 `cls_kl` 主导；
- 这一步的目标是先做**因果隔离**，不是马上找最佳修复值；
- 因此先完全拿掉 `cls_kl` 这条链路，判断：
  - total grad spike 是否同步回落；
  - `predictor_1` keep ratio 是否停止在 `step=140~146` 提前塌缩。

为保持 TrackA provenance 口径，服务器侧仍统一走 strict source wrapper；不要改用根仓 `main.py`。

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_control=tracka_source_epoch3_lossgradattrib_clsdw1_seed0_20260422
run_ablation=tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export TOKEN_DISTILL_WEIGHT=0.02

export CLS_DISTILL_WEIGHT=1.0
export RUN_NAME="$run_control"
bash scripts/run_tracka_train.sh source epoch3 0

export CLS_DISTILL_WEIGHT=0.0
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0

unset RUN_NAME CLS_DISTILL_WEIGHT TOKEN_DISTILL_WEIGHT
unset LOSS_GRAD_ATTRIB LOSS_GRAD_ATTRIB_PARAM NONEMPTY_KEEP_GUARD

for run in "$run_control" "$run_ablation"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"

  echo "===== $run :: strict-source header ====="
  for key in mode seed train_entry stop_after_epoch nonempty_keep_guard loss_grad_attrib loss_grad_attrib_param cls_distill_weight token_distill_weight
  do
    grep -nF "[tracka-source] ${key}=" "$LOG" || true
  done
  grep -nF 'Namespace(' "$LOG" | head -n 1 || true

  echo "===== $run :: epoch2 step140-146 keep+attrib ====="
  grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true
  for step in 140 141 142 143 144 145 146
  do
    grep -nF "[LossGradAttrib][epoch=2 step=${step}]" "$LOG" || true
  done

  echo "===== $run :: terminal ====="
  grep -nE 'Early stop after epoch|Accuracy of the model on the 524 test images|Max accuracy' "$LOG" || true
done
```

判定标准：

1. `step=146` total grad 明显回落
   - control 参考值：
     - `grad_l2=4.198726e+03`
     - `grad_absmax=1.057964e+03`
   - 若 ablation 同时降到：
     - `grad_l2 <= 2.1e+03`
     - `grad_absmax <= 5.3e+02`
     则记为“明显回落”；
   - `<20%` 的降幅不算有效改善。

2. `predictor_1` keep ratio 不再提前塌缩
   - control 的 `step=146` 参考值：
     - `final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
   - 若 ablation 在 `step=146` 至少达到：
     - `final_keep_ratio_mean >= 2.0e-01`
     - `active_margin_mean > -1.4`
     且 `step=140~146` 不再持续单调下滑到 `0.16` 左右，则记为“未提前塌缩”。

3. strict source 控制语义保持不变
   - 必须继续满足：
     - `training_source_tracka/main.py`
     - `NONEMPTY_KEEP_GUARD=false`
     - `LOSS_GRAD_ATTRIB=true`
     - `TOKEN_DISTILL_WEIGHT=0.02`
     - `stop_after_epoch=3`
   - control 与 ablation 之间，除了 `RUN_NAME / output_dir / log_dir / cls_distill_weight` 外，不应再出现别的有效参数差异。

当前状态：

- 命令包已给出；
- 本 issue 尚未贴回新日志；
- 因此当前不要写“修复已完成”，只把它视为下一步最小验证入口。

`2026-04-22` 用户后续已补回 corrected header / `LossGradAttrib`，当前 handoff 要记住的是：

1. `tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422` 的 `Namespace(...)` 里仍是：
   - `cls_distill_weight=1.0`
   - `token_distill_weight=0.02`
   - 所以这轮 `clsdw0` **不是有效 ablation run**
2. control / ablation 的关键窗口与 attribution 也因此完全同轨：
   - `predictor_1_keep_diag` 在 `epoch=2 step=140~146` 逐项相同；
   - `LossGradAttrib` 的 `total / cls_loss / ratio_loss / cls_kl / token_kl` 逐项相同；
   - terminal 都停在 `74.24%`
3. 因而当前不能写“`cls_distill_weight=0.0` 无效”，只能写：
   - **这轮 run 没真正切到 `cls_distill_weight=0.0`**
   - 所以暂时还没有拿到有效的 `cls_kl / cls_distill_weight` 单变量证据

另外，这次顺手确认了一个 runner 日志问题：

- 旧版 `train_stdout.log` 里没有 wrapper `[tracka-source] key=value` header；
- 原因是此前只有 python 子进程输出被 `tee` 到日志；
- 本地现在已经修正：
  - `scripts/_tracka_training_common.sh`
  - `scripts/run_tracka_train.sh`
- 之后新的 `train_stdout.log` 会同时保留 wrapper header 与 python `Namespace(...)`

所以下一位接手者不要再分析旧的 `clsdw0` run 本身，而是先：

1. 把更新后的两份 runner 脚本同步到服务器；
2. 只重跑一条新的 `clsdw0` ablation；
3. 先确认 header + `Namespace` 都真正显示 `cls_distill_weight=0.0`。

推荐最小命令：

```bash
rsync -avP -e "ssh -p 9001" \
  /home/yclcg/Transshield_final/scripts/_tracka_training_common.sh \
  /home/yclcg/Transshield_final/scripts/run_tracka_train.sh \
  wyb@10.204.244.1:/data/wyb/Transshield_final/scripts/

export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

run_ablation=tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422_resync1

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export CLS_DISTILL_WEIGHT=0.0
export TOKEN_DISTILL_WEIGHT=0.02
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0

LOG="$TRAIN_RUN_ROOT/$run_ablation/train_stdout.log"
grep -nF '[tracka-source] cls_distill_weight=' "$LOG" || true
grep -nF '[tracka-source] token_distill_weight=' "$LOG" || true
grep -nF 'Namespace(' "$LOG" | head -n 1 || true
grep -nF '[LossGradAttrib][epoch=2 step=146]' "$LOG" || true
grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true
```

`2026-04-22` 最新用户回贴又出现了新情况，下一位接手者要记住：

1. 新的 `resync1` ablation 尝试没有正常跑完 `epoch3` 对照窗口；
2. 用户当前贴回的是 traceback，而不是完整 crash window；
3. traceback 明确显示：
   - `training_source_tracka/models/dyvit.py`
   - `_check_finite('predictor_2_pred_score', pred_score)`
   - `RuntimeError: Non-finite tensor detected in VisionTransformerDiffPruning: predictor_2_pred_score`

所以当前不要把这条线写成“已完成修复”。但和前一版 handoff 不同的是：现在已经足够确认这条 `resync1` 是**有效配置**，因为用户贴回了：

- wrapper header：`cls_distill_weight=0.0`
- `Namespace(...)`：`cls_distill_weight=0.0`
- `epoch=0 step=0` 的 `LossGradAttrib`：
  - `component=cls_kl`
  - `weight=0.000000e+00`
  - `grad_l2=0.000000e+00`

这表示：

- `cls_kl / cls_distill_weight` 这条链路在目标参数上已经被真正拿掉；
- 但 run 后续仍然会触发 `predictor_2_pred_score` non-finite；
- 因此这条首个单变量尝试当前应记作：
  - **有效、但负向的结果**
  - 即：不是“无效对照”，而是“生效后仍未稳定完成比较窗口”

用户现已贴回真正的 crash 邻域，因此 handoff 口径更新为：

- 这条 `resync1` 已经不是“无效对照”，而是**有效的 `cls_distill_weight=0.0` 单变量尝试**；
- 但它不是 clean comparator，而是明确的负结果；
- 真正的失稳链条已补齐为：
  1. `epoch=2 step=34`
  2. `predictor_1_keep_diag` 出现 `raw_empty=1 / final_empty=1`
  3. `PredictorLG zero_active_policy_samples=1`
  4. `global_x / post_agg / out_conv_out isfinite=False`
  5. `predictor_2_pred_score` 触发 `_check_finite`

因此下一位接手者要记住：

- 不要再把 `clsdw0=0.0` 当成可以继续补跑的 clean baseline；
- 也不要继续在当前 issue 里扩展到第二个 loss 项；
- 当前这个 issue 的实验性结论已经够完整：
  - `cls_distill_weight=0.0` 这条 strict source / epoch3 首个单变量尝试，虽然在目标参数上成功拿掉了 `cls_kl` 链路，但会把 zero-active / predictor_2 non-finite 风险提前到 `epoch=2 step=34`
- 当前 issue 自身已经没有缺失证据；

如果后续还要继续，建议另开新的最小单变量 issue，由新 issue 决定下一条最小单变量，而不是继续复用这条 `clsdw0=0.0` 路径。

### TrackA 下一条最小单变量结果（`cls_distill_weight=0.5`）

首个单变量 `clsdw0=0.0` 被确认是有效负结果后，新 issue 只做了一条最小验证：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 唯一改动：`cls_distill_weight: 1.0 -> 0.5`
- run：
  - `tracka_source_epoch3_lossgradattrib_clsdw1_control_seed0_20260422_next1`
  - `tracka_source_epoch3_lossgradattrib_clsdw05_seed0_20260422_next1`

当前必须记住的结论：

1. 这次 `clsdw05` 是**有效单变量配置**
   - header / `Namespace(...)` 明确显示 `cls_distill_weight=0.5`
   - `epoch=0 step=0` 的 `LossGradAttrib` 也精确反映了目标链路减半：
     - `scaled_loss: 4.530853e-02 -> 2.265427e-02`
     - `grad_l2: 2.592915e+01 -> 1.296458e+01`

2. 它没有走 `clsdw0=0.0` 的那条负路径
   - 用户回贴的 `early-fail gate` 没有命中 `epoch=2 step=34` 的：
     - `raw_empty / final_empty`
     - `zero_active_policy_samples`
     - `RuntimeError: Non-finite tensor`
   - 所以当前没有证据表明 `clsdw05` 会像 `clsdw0=0.0` 一样提前触发 empty keep / zero-active / predictor_2 non-finite。

3. 在真正关键窗口 `epoch=2 step=146`，它给出了**强烈缓解**
   - control：
     - `final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
     - `total grad_l2=4.198726e+03`
     - `cls_kl grad_l2=4.041559e+03`
   - `clsdw05`：
     - `final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - `total grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
   - 这说明 `predictor_1` 不再沿 control 那条轨迹提前塌缩，且 `step=146` 的 `cls_kl` 主导 spike 基本被压掉。

4. 但它不是正式修复
   - terminal 仍是：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 这只是 `epoch3` 低精度诊断窗口里的正结果，不能写成正式成绩恢复。

所以后续接手时请按下面口径理解：

- `cls_distill_weight=0.5` 是当前 issue 选出的**下一条且仅一条最小单变量**
- 它已经给出**明确缓解**
- 但本 issue 到这里就应收住，不要在同一 issue 内继续扩展第二条变量
- 若后续还要继续，应另开新的单变量 issue

### TrackA post-`clsdw05` 下一条单变量结果（`cls_distill_weight=0.75`）

本轮接续 issue 已完成；需要记住的最终结论如下：

- 唯一改动是 `cls_distill_weight: 1.0 -> 0.75`
- strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true` 保持不变
- `token_distill_weight=0.02`、`ratio_weight=2.0`、`activation_lr_scale=10.0`、`model_ema=false` 都未漂移

用户回贴日志后，本轮已可定性为：

1. `clsdw075` 是**有效单变量配置**
   - control 与 ablation 的 header / `Namespace(...)` 已分别对齐到：
     - control：`cls_distill_weight=1.0`
     - ablation：`cls_distill_weight=0.75`

2. 它没有命中 `step=34` 早崩 gate
   - `epoch=2 step=30~36` 没有出现：
     - `raw_empty / final_empty > 0`
     - `zero_active_policy_samples > 0`
     - `RuntimeError: Non-finite tensor`
   - 因此它没有走 `clsdw0=0.0` 那条负路径。

3. 它在 `step=146` 给出**明确缓解**
   - control：
     - `final_keep_ratio_mean=1.548948e-01`
     - `total grad_l2=4.198726e+03`
     - `cls_kl grad_l2=4.041559e+03`
   - `clsdw075`：
     - `final_keep_ratio_mean=4.730873e-01`
     - `total grad_l2=3.748181e+01`
     - `cls_kl grad_l2=2.105142e+01`
   - 所以这条 run 应按“缓解”收口，而不是“无变化”或“更早失稳”。

4. 但它**弱于** `clsdw05=0.5`
   - `clsdw05` 在同一窗口更强：
     - `final_keep_ratio_mean=5.022926e-01`
     - `total grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
   - 因此 `0.75` 并没有超过 `0.5`，只是再次证明削弱 `cls_distill_weight` 这条链路确实能缓解 spike。

5. terminal 仍不动
   - control 与 `clsdw075` 的 terminal 都仍是：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 所以它仍只是 `epoch3` 的低精度诊断正结果，不能写成正式修复。

接手时的推荐口径：

- 当前 `cls_distill_weight` 剂量线上已知：
  - `0.0`：有效负结果
  - `0.5`：更强的明确缓解
  - `0.75`：明确缓解，但弱于 `0.5`
- 因此若后续继续推进，不建议在同一 issue 内继续扫更多 `cls_distill_weight` 值；
- 应另开新的单变量 issue，转向新的阻塞，而不是继续扩展这条 issue。

后续给服务器的 TrackA 命令必须使用同一个 shell / tmux 会话内可复用的路径变量 setup，而不是一次性 `env VAR=... bash ...`：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"
```

run 名和日志路径也要复用变量：

```bash
run_example=tracka_source_epoch3_example_seed0

export RUN_NAME="$run_example"
export NONEMPTY_KEEP_GUARD=false
bash scripts/run_tracka_train.sh source epoch3 0

LOG="$TRAIN_RUN_ROOT/$run_example/train_stdout.log"
grep -nE '^Namespace\(' "$LOG" | head -n 1 || true
```

如果后续还要 grep / 比对，不要把 setup 放进 `bash -c`、heredoc 子 shell 或 `env VAR=... command`，否则路径变量会随子进程结束而消失。

### TrackA post-`clsdw075` 新阻塞点命令包（`token_distill_weight=0.04`）

当前下一条且仅一条最小单变量：

- 固定 `CLS_DISTILL_WEIGHT=0.5`
- 只改 `TOKEN_DISTILL_WEIGHT=0.02 -> 0.04`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

选择理由：

- `clsdw075` 已有效但弱于 `clsdw05=0.5`，继续扫 `cls_distill_weight` 只是在同一剂量线细化；
- `ratio_loss` 已被排除为 `predictor_1` 首发驱动，先改 `ratio_weight` 更像改变 pruning pressure，不如先测 mask-aware token alignment；
- `token_distill_weight` 已由 runner 透传，不需要改训练语义，也不影响正式前端展示口径。

服务器命令：

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"
cd "$REPO_ROOT"

mkdir -p /data/wyb/tmp
export TMPDIR=/data/wyb/tmp
export TMP=/data/wyb/tmp
export TEMP=/data/wyb/tmp

run_control=tracka_source_epoch3_lossgradattrib_clsdw05_tdw002_control_seed0_20260422_next3
run_ablation=tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_seed0_20260422_next3

export NONEMPTY_KEEP_GUARD=false
export LOSS_GRAD_ATTRIB=true
export LOSS_GRAD_ATTRIB_PARAM=score_predictor.1.out_proj.weight
export CLS_DISTILL_WEIGHT=0.5

export TOKEN_DISTILL_WEIGHT=0.02
export RUN_NAME="$run_control"
bash scripts/run_tracka_train.sh source epoch3 0

export TOKEN_DISTILL_WEIGHT=0.04
export RUN_NAME="$run_ablation"
bash scripts/run_tracka_train.sh source epoch3 0

for run in "$run_control" "$run_ablation"
do
  LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"

  echo "===== $run :: strict-source header ====="
  for key in mode seed train_entry stop_after_epoch nonempty_keep_guard loss_grad_attrib loss_grad_attrib_param cls_distill_weight token_distill_weight
  do
    grep -nF "[tracka-source] ${key}=" "$LOG" || true
  done
  grep -nE '^Namespace\(' "$LOG" | head -n 1 || true

  echo "===== $run :: epoch2 step30-36 early-fail gate ====="
  grep -nE 'epoch=2 step=3[0-6].*(predictor_1_keep_diag|zero_active_policy_samples|RuntimeError|Non-finite tensor|predictor_2_pred_score|isfinite=False)' "$LOG" || true

  echo "===== $run :: epoch2 step140-146 keep+attrib ====="
  grep -nE 'epoch=2 step=14[0-6].*predictor_1_keep_diag' "$LOG" || true
  for step in 140 141 142 143 144 145 146
  do
    grep -nF "[LossGradAttrib][epoch=2 step=${step}]" "$LOG" || true
  done

  echo "===== $run :: terminal ====="
  grep -nE 'Early stop after epoch|Accuracy of the model on the 524 test images|Max accuracy|RuntimeError|Non-finite tensor' "$LOG" || true
done
```

判定标准：

- 配置有效：control 必须是 `cls_distill_weight=0.5 / token_distill_weight=0.02`，ablation 必须是 `cls_distill_weight=0.5 / token_distill_weight=0.04`，且两边仍是 strict source、guard-off、`epoch3`。
- 不早崩：`epoch=2 step=30~36` 不应出现 `raw_empty/final_empty > 0`、`zero_active_policy_samples > 0`、`predictor_2_pred_score` 非有限或 `RuntimeError`。
- `step=146`：若 ablation 的 total `grad_l2` 较 fixed-`clsdw05` control 降低至少 `20%` 且 keep 不明显变差，记为缓解；若只在小范围波动，记为无变化；若早崩，记为更早失稳。
- terminal：必须单独检查是否仍为 `Accuracy ... 74.2%` / `Max accuracy: 74.24%`；即使有改善也只能记作低精度诊断信号，不能写成正式成绩。

用户已回贴日志，本轮实际判定如下：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.02`
  - ablation：`cls_distill_weight=0.5 / token_distill_weight=0.04`
- `step=34` 不早崩：
  - `epoch=2 step=30~36` 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限或 `RuntimeError`
- `step=146` 稳定性变差：
  - control：`final_keep_ratio_mean=5.022926e-01`，total `grad_l2=2.400378e-01`，`cls_kl grad_l2=1.123101e-01`，`token_kl grad_l2=2.975943e-04`
  - ablation：`final_keep_ratio_mean=4.537168e-01`，total `grad_l2=4.365869e+00`，`cls_kl grad_l2=2.991327e+01`，`token_kl grad_l2=1.172528e+00`
- terminal 正向松动：
  - control：`Max accuracy: 74.24%`
  - ablation：`Max accuracy: 79.01%`

接手口径：

- `token_distill_weight=0.04` 是有效单变量，但不是 clean 稳定性缓解；
- 它的价值在于暴露了一个新阻塞：terminal 改善与 `score_predictor.1` 稳定性恶化同时发生；
- 不建议直接跑 `full20`，也不要把 `79.01%` 写成正式成绩；
- 如果继续推进，请另开新 issue，专门解耦“terminal 提升 vs 稳定性恶化”的冲突，不要在本 issue 内继续扩题。

### TrackA terminal-稳定性解耦结果（`token_distill_weight=0.03`）

用户已回贴 `2026-04-23` 的下一条单变量日志，本轮只验证：

- 固定 `cls_distill_weight=0.5`
- 唯一改动 `token_distill_weight: 0.02 -> 0.03`
- 保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw002_control_seed0_20260422_next4`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw003_seed0_20260422_next4`

实际判定：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.02`
  - ablation：`cls_distill_weight=0.5 / token_distill_weight=0.03`
  - `epoch=0 step=0` 的 `LossGradAttrib` 确认 `token_kl weight=3.000000e-02`，`cls_kl weight=5.000000e-01`
- `step=34` 不早崩：
  - `epoch=2 step=30` 两边都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 没有 `zero_active_policy_samples`、`isfinite=False`、`predictor_2_pred_score` 非有限或 `RuntimeError`
- `step=146` 稳定性仍较 control 恶化：
  - control：`final_keep_ratio_mean=5.022926e-01`，`active_margin_mean=-3.137406e-02`，total `grad_l2=2.400378e-01`
  - `tdw003`：`final_keep_ratio_mean=4.792178e-01`，`active_margin_mean=-1.206955e-01`，total `grad_l2=1.165204e+00`
  - keep ratio 下降约 `0.0231` 仍在容忍线内，但 total `grad_l2` 约 `4.85x` control，margin 更负约 `0.0893`，超过容忍线
- terminal 仅微弱松动：
  - control：`Max accuracy: 74.24%`
  - `tdw003`：`Max accuracy: 74.43%`
  - 明显不接近 `tdw004` 的 `79.01%`

接手口径：

- `token_distill_weight=0.03` 是有效单变量，但没有拿到 clean 解耦；
- 它相比 `0.04` 降低了稳定性伤害，却也基本失去 terminal 提升；
- 当前结论是：token distill 剂量轴上尚未证明存在“terminal 提升 + 稳定性无变化/缓解”的点；
- 不建议直接跑 `full20`，也不要把 `74.43%` 或 `79.01%` 写成正式成绩；
- 如果继续推进，另开新 issue 选择下一条且仅一条最小单变量，不要在当前 issue 内继续扫 token 权重。

### TrackA post-`tdw003` 结果（`ratio_weight=3.0`）

用户已回贴 `2026-04-23` 的 control / ablation 日志，本轮只验证：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=3.0`
- 保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_next5`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw3_seed0_20260423_next5`

实际判定：

- 配置有效：
  - control / ablation 的 wrapper header 与 `Namespace(...)` 都正确进入；
  - 只有 `ratio_weight=2.0` vs `3.0` 不同。
- `step=34` 不早崩：
  - `epoch=2 step=30` 两边都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 没有 `zero_active_policy_samples`、`isfinite=False`、`predictor_2_pred_score` 非有限或 `RuntimeError`
- `step=146` 是 split 代理改善，但不是稳定性修复：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`
  - ablation：`final_keep_ratio_mean=5.245984e-01`，`active_margin_mean=4.946269e-02`，total `grad_l2=5.206851e+01`
  - keep ratio 回升约 `+0.0709`，margin 回升约 `+0.2138`；
  - 但 total `grad_l2` 升到 control 的约 `11.93x`，因此不能按当前规则记为稳定性缓解。
- terminal 直接丢失：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`

接手口径：

- `ratio_weight=3.0` 是有效单变量，但它是**负结果**；
- 它把 keep/margin 代理往“更保守 keep”方向推回，却没有修复目标参数 total grad，反而把 terminal 从 `79.01%` 拉回 `74.24%`；
- 因此这条结果不是 clean 解耦，也不是 mixed 修复，当前不要推进 `full20`；

### TrackA post-`rw3` 结果（`cls_distill_weight=0.4`）

用户已回贴 `2026-04-23` 的 control / ablation 日志，本轮只验证：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.4 / token_distill_weight=0.04 / ratio_weight=2.0`
- 保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_next5`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw04_tdw004_rw2_seed0_20260423_next6`

实际判定：

- 配置有效：
  - control / ablation 的 wrapper header 与 `Namespace(...)` 都正确进入；
  - `epoch=0 step=0` 的 attribution 权重确认 ablation 为 `cls_kl weight=4.000000e-01`、`token_kl weight=4.000000e-02`；
  - 只有 `cls_distill_weight=0.5` vs `0.4` 不同。
- `step=34` 不早崩：
  - ablation `epoch=2 step=30` 显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 没有 `zero_active_policy_samples`、`isfinite=False`、`predictor_2_pred_score` 非有限或 `RuntimeError`
- `step=146` 是 `cls_kl` 分项下降，但不是整体稳定性修复：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`，`cls_kl grad_l2=2.991327e+01`
  - ablation：`final_keep_ratio_mean=2.575499e-01`，`active_margin_mean=-1.041453e+00`，total `grad_l2=7.772295e+00`，`cls_kl grad_l2=4.482888e+00`
  - `cls_kl grad_l2` 明显下降，但 total `grad_l2` 升到 control 的约 `1.78x`，keep ratio 下降约 `0.1962`，margin 负移约 `0.8771`
- terminal 直接丢失：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`

接手口径：

- `cls_distill_weight=0.4` 是有效单变量，但它是**有效负结果**；
- 它只降低了 `cls_kl` 分项压力，没有保住 `tdw004` 的 terminal-positive 信号；
- 它还使目标参数 total grad、`predictor_1` keep / margin 和 terminal 同时变差；
- 因此这条结果不是 clean 解耦，也不是 mixed 修复，当前不要推进 `full20`；
- 当前仍没有拿到“terminal 提升 + 稳定性无变化 / 缓解”的 clean 解耦迹象。
- post-`clsdw04` 的下一条 midpoint `token_distill_weight=0.035` 已另行验证并定性，见下一节。

### TrackA post-`clsdw04` 结果（`token_distill_weight=0.035`）

用户已回贴 `2026-04-23` 的 control / ablation 日志，本轮只验证：

- paired control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
- ablation：`cls_distill_weight=0.5 / token_distill_weight=0.035 / ratio_weight=2.0`
- 保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`

run：

- control：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw004_rw2_control_seed0_20260423_tdw0035`
- ablation：`tracka_source_epoch3_lossgradattrib_clsdw05_tdw0035_rw2_seed0_20260423_next7`

实际判定：

- 配置有效：
  - control / ablation 的 wrapper header 与 `Namespace(...)` 都正确进入；
  - 只有 `token_distill_weight=0.04` vs `0.035` 不同。
- `step=34` 不早崩：
  - `epoch=2 step=30/40` 两边都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 没有 `zero_active_policy_samples`、`isfinite=False`、`predictor_2_pred_score` 非有限、`RuntimeError`、`Non-finite` 或 `skip optimizer step`
- `step=146` 是稳定性缓解：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`，`cls_kl grad_l2=2.991327e+01`，`token_kl grad_l2=1.172528e+00`
  - ablation：`final_keep_ratio_mean=4.878815e-01`，`active_margin_mean=-8.497284e-02`，total `grad_l2=2.089920e+00`，`cls_kl grad_l2=9.532135e-01`，`token_kl grad_l2=2.397546e-03`
  - total / `cls_kl` / `token_kl` 梯度明显下降，keep / margin 也从 `tdw004` 坏窗口回温。
- terminal 直接丢失：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`

接手口径：

- `token_distill_weight=0.035` 是有效单变量，且稳定性明显缓解；
- 但它没有保住 `tdw004` 的 terminal-positive 信号，terminal 直接回到 `74.24%`；
- 因此这条结果是**稳定性缓解、terminal 丢失的负向 midpoint 结果**，不是 clean 解耦，也不是修复；
- 当前不要推进 `full20`，也不要把 `74.24%`、`74.43%` 或 `79.01%` 写成正式成绩；
- 如果继续 TrackA，应另开新单变量 issue 决定是否还要在 `0.04` 附近做更贴近的 token midpoint，不要在本 issue 内扩题。

### TrackA post-`tdw0035` 近端 midpoint 结果：`token_distill_weight=0.0375`

当前 issue 的目标：

- 只决定在 `token_distill_weight=0.035` 已确认“稳定性缓解、terminal 丢失”之后，是否还值得沿 token 轴继续做**一条且仅一条**更贴近 `0.04` 的 midpoint。

本地已执行：

- 只执行 `sed -n` / `rg -n` 读取 `docs/history_best_repro_drift_audit_2026-04-21.md`、`docs/current_work_status.md`、`docs/handoff-next.md`、`docs/tracka_predictor1_root_cause_2026-04-21.md`、`scripts/run_tracka_train.sh`、`training_source_tracka/main.py`、`training_source_tracka/engine.py`、`training_source_tracka/losses.py`
- 未执行任何 `/data/wyb/...` 命令

当前唯一结论：

- `tdw0035` 这个 issue 已可关闭：`0.035` 的配置有效、未早崩、稳定性缓解、但 terminal 已回落到 `74.24%`；因此它自己的问题已经回答完，继续试新点已经属于新 issue，不应在 `tdw0035` 原 issue 内扩题。
- 唯一近端候选 `token_distill_weight=0.0375` 现也已完成：
  - 从回贴的 `LossGradAttrib` 权重可直接读出：
    - `cls_kl weight=5.000000e-01`
    - `token_kl weight=3.750000e-02`
    - `ratio_loss weight=6.666667e-01`
  - 这足以**推断** ablation 已正确进入 `cls_distill_weight=0.5 / token_distill_weight=0.0375 / ratio_weight=2.0`
  - run 已正常到达 `epoch=2 step=146` 并完成 eval，因此 `step=34` early-fail gate 未命中
  - `step=146` 相比 `tdw004` control 给出了更强稳定性缓解：
    - `predictor_1 final_keep_ratio_mean: 4.537168e-01 -> 5.085371e-01`
    - `active_margin_mean: -1.643516e-01 -> -2.972232e-03`
    - total `grad_l2: 4.365869e+00 -> 1.294038e-01`
    - `cls_kl grad_l2: 2.991327e+01 -> 1.165144e-01`
    - `token_kl grad_l2: 1.172528e+00 -> 1.486021e-03`
  - 但 terminal 仍从 `79.01%` 直接回落到 `74.24%`
- 因此本 issue 的最终选择应收口为 **`stop_token_midpoint`**。

为什么现在应停：

- 当前已经有三个 `<0.04` 的点：
  - `0.03 -> 74.43%`
  - `0.035 -> 74.24%`
  - `0.0375 -> 74.24%`
- 它们都没有保住 `79.01%`，但稳定性却持续改善；
- `0.0375` 已经证明：即使把坏窗口拉回到几乎与稳定 control 对齐，terminal 仍不会回来；
- 所以继续做 `0.038 / 0.0385 / 0.039` 这类近端 token 剂量搜索，信息增益已经不足。

当前状态：

- 本 issue 已经完成；
- 当前正式结论是 `stop_token_midpoint`；
- 当前不要继续扩近端 token 轴，不推进 `full20`，也不把 `74.24%` 或 `79.01%` 写成正式成绩或修复。

### `Phase 4 payload` formal-vs-mixed 工程复核结果

这一步已经按正式 fp32 路径 vs 唯一 mixed 候选做完单次工程复核，服务器 run 为：

- `phase4_payload_formal_vs_mixed_20260423_104207`
- A：`formal_fp32`
- B：`mixed_stage1fp32_bw4`

本轮固定条件：

1. 两边都保持：
   - `selection_mode=blockwise_exact_kth`
2. A 侧：
   - `payload_dtype=float32`
3. B 侧：
   - `payload_dtype=float16`
   - `payload_stage_dtypes="1:float32"`
   - `payload_boundary_window=4`

本轮结果：

- A / B 两边都通过：
  - `overall_passed = true`
  - `pipeline_verify_overall_passed = true`
- mixed 侧 host 输入压缩**确实成立**：
  - `payload_mixed_transport_total_bytes: 899184 -> 668624`
  - 比例 `0.7436x`
  - `python_fastpath_make_shares_total_input_bytes: 899188 -> 630900`
  - 比例 `0.7016x`
- 但最终工程指标明显更差：
  - `communication total bytes: 1802657 -> 1880245`
  - 比例 `1.0430x`
  - `network_kth_bridge: 11.7105s -> 197.2977s`
  - 比例 `16.8479x`
  - `total_pipeline_duration_sec: 18.1556s -> 203.4937s`
  - 比例 `11.2083x`

因此本轮 formal-vs-mixed 复核的正式结论是：

1. mixed payload 的 host 侧压缩方向**没有错**；
2. 但在当前 OpenBumbleBee / SPU fastpath 栈里，它**不能转化成更好的端到端通信 / 时延指标**；
3. 当前正式默认路径必须继续保持：
   - `blockwise_exact_kth + float32`
4. 当前不要再在这条 issue 内继续给第二个 mixed payload 配方，也不要把它扩成 payload 搜索。

### `Phase 4 mixed payload` 目前为什么先收住

现在已经有 formal-vs-mixed 的真实 SPU profile 结果，结论是：

- `payload_dtype=float16`
- `payload_stage_dtypes="1:float32"`
- `payload_boundary_window=4`

虽然：

- `payload_mixed_transport_total_bytes` 已经明显下降
- `make_shares_total_input_bytes` 也明显下降
- 说明 host/P1 侧 mixed payload transport 生效

但最终没有拿到想要的展示收益：

- `communication total bytes` 没降，反而从 `1802657` 升到 `1880245`
- `network_kth_bridge` 从 `11.7105s` 飙到 `197.2977s`

当前判断：

1. 这不是模型问题，也不是 ablation 方向错了；
2. 而是当前 OpenBumbleBee / SPU fastpath 的主显示通信量主要被：
   - `builtin_fetch_object`
   - share object 返回
   主导；
3. mixed payload 只能压缩 `make_shares` 输入，不会自然压缩 share object 本身；
4. 若在 SPU 内部精确重构 dense float32，再跑 compare-network，会引入过重的协议算子成本。

所以这条线当前的定位应改成：

- **保留代码与报告**
- **作为 Phase 4 的明确负结果 / 边界结论**
- **不要再继续作为正式展示优化主线**
- **不要在当前展示范围内继续给第二个 mixed payload 配方**

如果以后真要继续，只值得走两类更深改动：

1. 改底层 `distributed_impl` / share transport
   - 目标不是压缩 host 输入
   - 而是压缩/批量化 share object 的跨节点传输
2. 改协议设计
   - 不再在 SPU 内先重构 dense full tensor
   - 而是直接设计能消费紧凑 mixed 表示的 `kth` 路径

这两条都已经超出当前比赛展示所需范围，因此默认先不继续。
