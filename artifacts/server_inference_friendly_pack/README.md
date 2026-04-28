# 推理友好运行包说明

`artifacts/server_inference_friendly_pack/` 是当前比赛展示版的默认运行入口。

如果仓库根目录存在同名脚本（例如 `run_secure_selection_mode_profile_compare.sh`），默认只把它当作**兼容 wrapper**；权威实现仍以本目录为准。

当前目录下的大部分 `run_*.sh` 会自动加载：

- `final_compare_env.template.sh`

`final_compare_env.local.sh` 只在显式设置 `TRANSSHIELD_USE_LOCAL_ENV=1` 时加载。默认不加载本机路径，避免把 `/home/...` 的开发环境同步到服务器后污染运行配置。

如果你只想做一件事，优先使用：

- 小样本链路验证：`run_full_final_comparison_smoke.sh`
- 完整对比链运行：`run_full_final_comparison_suite.sh`
- 前后端交互演示：`run_web_demo.sh`
- `e2e secure` 新线最小骨架：`run_e2e_secure_poc.sh`
- `e2e whole-forward` 集成入口：`run_e2e_secure_whole_forward.sh`
- Margin-aware pruning ablation：`run_margin_aware_pruning_ablation.sh`
- 统一 secure benchmark 外部 proxy 对比：`run_standardized_secure_external_benchmark.sh`

---

## 1. 推荐运行顺序

### 明文主链路

1. `run_plaintext_eval.sh baseline`
2. `run_plaintext_eval.sh modified`
3. `run_plaintext_model_compare.sh`

### secure 闭环链路

4. `run_secure_export_inputs.sh`
5. `run_secure_pipeline.sh cpu|spu`
6. `run_secure_replay.sh`
7. `run_secure_score_compare.sh`
8. `run_final_comparison_report.sh`

### 一键快捷方式

- `run_full_final_comparison_smoke.sh`
- `run_full_final_comparison_suite.sh`
- `run_selected_image_secure_suite.sh`

### 算法 ablation

- `run_margin_aware_pruning_ablation.sh`
  - 作用：只改变 `pruning_margin_weight`，观察 pruning 边界 margin 是否变大；
  - 默认 `ABLATION_MODE=debug80`，只做 80 step 数值 smoke，不替换当前展示模型；
  - 设置 `ABLATION_MODE=full20` 后会训练、搜索阈值、冻结候选 bundle，并生成 `results/margin_aware_pruning_ablation/<run>/margin_ablation_compare.md`；
  - 设置 `RUN_SECURE_REPLAY=1` 后可对候选 bundle 追加 secure replay / compare 检查。

---

## 2. `smoke` 与 `suite` 的区别

### `run_full_final_comparison_smoke.sh`

- 只取很少样本；
- 默认同时截断 plaintext 与 secure 输入；
- 用来验证脚本、bridge、checker、replay、compare 是否跑通；
- 不适合拿来判断模型最终性能。

### `run_full_final_comparison_suite.sh`

- 使用完整验证集；
- 用来生成正式展示结果；
- 是答辩时应优先引用的结果来源。

当前默认运行入口已经切到：

- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/`

旧正式 bundle（只保留 provenance，不再作为主展示 / 主对比入口）：

- `artifacts/frozen_bundle_full/`

仍保留用于 provenance，不会被删除。

---

## 3. CPU 与 SPU 的区别

### `run_secure_pipeline.sh cpu`

- 运行 secure sidecar 的本地明文参考执行；
- 主要用于开发调试、reference check 与快速验证；
- **不是我们的最终安全路径**；
- 不是真正的 2PC。

### `run_secure_pipeline.sh spu`

- 运行同一逻辑在 `SPU / OpenBumbleBee` 上的真实安全执行；
- **这是 legacy sidecar 链路中的真实安全执行路径**；最终 Web 主路径已切到浏览器本地分片 + E2E approximate SPU；
- 涉及 secret sharing、协议执行、节点通信与额外性能开销。
- 当前也支持和 `e2e` wrapper 同样的 runtime 稳定性开关：
  - `SPU_RUNTIME_REUSE=1`
  - `SPU_DISABLE_COLOCATED_OPTIMIZATION=1`
- 若服务器上出现 `Socket closed`、`Not connected` 一类 internal link 异常，优先先保留当前命令入口，只额外加 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 复验，不要回退到手动 `tools/transshield_spu_runtime_setup.py start`。

两者结果一致，表示它们实现的是同一函数语义；不表示 CPU 模式本身完成了真实 2PC。
更准确地说：`CPU secure` 不是最终安全路径，它只是 sidecar 函数的本地参考实现；`SPU secure` 是 legacy sidecar 的真实安全执行。当前最终 Web 主路径使用 `run_e2e_secure_approx_deploy.sh`。

---

## 4. 重要环境变量

优先参考：`final_compare_env.template.sh`

常用变量包括：

- `TRAIN_DATA_PATH`
- `VAL_DATA_PATH`
- `SECURE_RUNTIME=cpu` 或 `SECURE_RUNTIME=spu`
- `KTH_SELECTION_MODE=blockwise_exact_kth`、`KTH_SELECTION_MODE=flat_odd_even` 或 `KTH_SELECTION_MODE=phase3_lower_tail`
- 当前默认值是 `blockwise_exact_kth`
- 默认 manifest 是 `results/blockwise_exact_kth_selection_manifest_default.json`
- `flat_odd_even` 保留为旧 reference fallback
- `phase3_lower_tail` 保留为旧实验开关，不再作为默认展示 / 运行口径
- `PHASE3_SELECTION_MANIFEST`
- `PLAINTEXT_MAX_SAMPLES`
- `SECURE_MAX_SAMPLES`
- `SPU_RUNTIME_REUSE`
- `SPU_DISABLE_COLOCATED_OPTIMIZATION`

其中：

- `SECURE_MAX_SAMPLES=8` 与 `PLAINTEXT_MAX_SAMPLES=8` 适合 smoke；
- 完整展示时建议不要截断样本。
- `SPU_RUNTIME_REUSE=1` 适合复用已启动的 SPU runtime；
- 如果遇到 runtime internal link 不稳定，可先加 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 做单变量复验；

---

## 5. 结果文件说明

运行结束后建议优先查看：

- `artifacts/server_pipeline_run/<RUN_NAME>/comparison_report_summary.txt`
- `artifacts/server_pipeline_run/<RUN_NAME>/comparison_report_summary.json`
- `artifacts/server_pipeline_run/<RUN_NAME>/plaintext_vs_secure_score_compare.json`

如果要看更简洁的中文整理，请直接查看：

- `docs/result_summary.md`
- `docs/external_baseline_comparison.md`
- `docs/data_source_policy.md`

---

## 6. 其他辅助脚本

### 明文辅助

- `run_single_image_comparison.sh`
- `run_plaintext_predict.sh baseline`
- `run_plaintext_predict.sh modified`
- `run_selected_image_diagnosis.sh`
- `run_web_demo.sh`

### secure 辅助

- `run_cpu_spu_profile.sh`
- `run_e2e_secure_poc.sh`
- `run_e2e_secure_whole_forward.sh`
- `run_secure_selection_mode_profile_compare.sh`
- `run_secure_profile_summary.sh`
- `run_secure_profile_compare.sh`
- `run_standardized_secure_external_benchmark.sh`
- `run_token_pruning_visualization.sh`
- `run_selected_image_secure_diagnosis.sh`

这些脚本主要用于：

- 单图或指定图片列表诊断；
- 运行剖析；
- 演示时做更细粒度的截图与说明。

### `run_e2e_secure_poc.sh`

- 启动当前 `e2e secure inference` 并行研究线的最小骨架；
- 先写出 e2e 边界 contract；
- 再生成客户端预处理 `pixel_values` 包；
- 再跑一条 plaintext reference，给后续整网 SPU 对齐做基准；
- 另外会补一条 `static whole-forward` plaintext reference，专门对应“先不做 pruning、先做整网 secure forward”的下一步；
- 同时仓内已提供 whole-forward compare 子命令，后续一旦产出 SPU 候选 `logits`，即可直接对齐；
- 这是早期 POC 入口；当前 Web 主路径已使用后续的 `run_e2e_secure_approx_deploy.sh`，不要再把这个 POC 的早期边界当作最终状态。

### `run_e2e_secure_whole_forward.sh`

- 是 `e2e secure inference` 新线的集成 wrapper；
- 当前支持：
  - `prepare`
  - `cpu`
  - `spu`
  - `verify`
  - `probe-cpu`
  - `probe-spu`
  - `probe-compare`
- `spu` 模式当前是实验性 `static whole-forward` JAX/SPU 后端，只覆盖“不含 runtime pruning 决策”的静态 ViT 前向；它默认 secret-share 输入、public 模型参数，后续可用 `E2E_SPU_PARAMS_MODE=secret` 进一步验证 secret 参数模式，不代表动态 `masking-pruning` 已进入 secure forward；
- 为支持 `block9` 数值漂移归因，wrapper 现还暴露两个实验性 ablation 开关：
  - `E2E_SPU_ATTENTION_POLICY=smoothed|standard`，默认 `smoothed` 保持既有行为，`standard` 用普通 softmax 去掉 policy smoothing；
  - `E2E_SPU_ACTIVATION_OVERRIDE=bundle|gelu|fixed_square|learnable_square|learnable_quadratic|learnable_quadratic_gelu_init`，默认 `bundle` 保持当前 bundle 激活；其它值只用于 SPU-only 诊断，不应写成正式同口径 compare；
- `E2E_SPU_BLOCK_CHUNK_SIZE=N` 是 depth11/12 party-local runtime 边界后的实验性无 reveal 图拆分开关：按 N 个 transformer blocks 分段执行 SPU 图，中间 token state 仍保留为 SPU value，只 reveal final logits；默认 `0` 保持原 monolithic graph；
- `block1-smoke` 是当前 depth0 通过、depth1 断链后的 debug-only 子图定位入口，会逐段运行 `patch_pos / norm1 / qkv / attention / mlp / head` 并 reveal 阶段输出，只用于定位第一个断链子图，不属于生产 e2e reveal policy；
- `E2E_REDACT_PRIVATE_INPUT_PATHS=1` 默认开启，会在 `run` 的 `.pt` 与 summary JSON 中隐藏 legacy/P1/P2 私有 share manifest 路径；share 输入模式下 wrapper 也不再传 `--input-pt`，避免 candidate metadata 指回 plaintext client pixel package；只有显式本地 debug 才建议设为 `0`；
- `probe-spu` 会沿用同一个 wrapper 内置的 runtime 自启动逻辑，因此在 block-level drift attribution 时不需要额外手动执行 `tools/transshield_spu_runtime_setup.py start`；
- wrapper 现已额外暴露 runtime 稳定性开关：可用 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 让 `spu/probe-spu` 自动以 `--disable-colocated-optimization` 拉起 runtime；如果同时设置 `SPU_RUNTIME_REUSE=1`，wrapper 也会先检查已存在 runtime 的 colocated 配置是否匹配，不匹配就自动重启，而不是误复用旧节点；
- 推荐顺序：
  1. 先跑 `run_e2e_secure_poc.sh`
  2. 再跑 `run_e2e_secure_whole_forward.sh prepare`
  3. 再跑 `run_e2e_secure_whole_forward.sh cpu`
  4. 再用 `E2E_RUN_MAX_SAMPLES=1 E2E_SPU_BATCH_SIZE=1 run_e2e_secure_whole_forward.sh spu` 做服务器 smoke
  5. 最后用 `E2E_VERIFY_ALLOW_PREFIX=1 run_e2e_secure_whole_forward.sh verify` 对齐 reference 前缀
- 当前若要在服务器检查 / 抓取这条线的结果，先固定：
  - `export REPO_ROOT=/data/wyb/Transshield_final`
  - `export RUN_NAME=tracka_e2e_secure_poc_cpu`
  - `export E2E_DIR="$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc"`
  - `export PACK_DIR="$E2E_DIR/whole_forward_pack"`
- 截至 `2026-04-22`，服务器 run `tracka_e2e_secure_poc_cpu` 已验证：
  - `sample_count = 524`
  - `cpu candidate elapsed_sec ≈ 20.73`
  - `logits/probabilities max_abs_error = 0.0`
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
- 截至 `2026-04-23`，这条线的阶段结论已前进到：
  - `depth=0..5 / sample=1 / public params` 的 same-depth smoke 已在服务器通过；
  - 默认 colocated runtime 配置下的一个历史 `depth=6` full run 曾失配并伴随 node/link 异常；

### `run_e2e_secure_approx_deploy.sh`

- 是当前可实际使用的 e2e 全隐私输入近似推理入口；
- 固定服务器已验证的近似配置：
  - `E2E_STATIC_DEPTH_LIMIT=12`
  - `E2E_SPU_BATCH_SIZE=1`
  - `E2E_PARTY_LOCAL_SHARE_LOAD=1`
  - `E2E_REDACT_PRIVATE_INPUT_PATHS=1`
  - `E2E_SPU_LAYER_NORM_POLICY=public_calibrated`
  - `E2E_SPU_ATTENTION_POLICY=uniform`
  - `E2E_SPU_ACTIVATION_OVERRIDE=fixed_square`
  - `E2E_SPU_ACTIVATION_CLIP_VALUE=3.0`
  - 可选但当前 smoke-stable 配置需要 `E2E_OUTPUT_CALIBRATION_JSON`
- 支持模式：
  - `make-calib-pixels`：从公开校准图片目录生成 public calibration pixel package；
  - `calibrate`：生成 public-calibrated layer norm JSON；
  - `infer`：对 split public/P1/P2 share manifests 执行 party-local SPU 推理；
  - `all`：按上述三步一次执行；
- 该入口会拒绝 `E2E_SPU_BATCH_SIZE != 1`，因为服务器已定位 `bsz=2` 会触发 full-depth batched graph 数值爆炸；多样本部署应按 `bsz=1` 逐样本 chunk 顺序处理；
- 默认公开校准目录为 `/data/wyb/pneumoniamnist_imagefolder_subset`，可用 `PUBLIC_CALIB_DATASET_DIR` 或 `PUBLIC_CALIB_IMAGE_LIST` 覆盖；
- 默认 reveal policy 仍是 final logits only；candidate JSON 会 redacted P1/P2 私有 manifest path；
- 这条路径是 deployable approximation，不是原始 exact ViT：exact secret LayerNorm、secret softmax attention、dynamic pruning、独立 P1/P2 进程仍是后续工作。
- `2026-04-26` 服务器已验证的当前部署基线是：
  - `depth=12 / public_calibrated LN / uniform attention / fixed_square / party-local share load / bsz=1`；
  - `sample=2` 按 `bsz=1` 顺序执行时 logits 有限、概率非饱和，两个样本输出均为 class 1；
  - `bsz=2` full-depth batched graph 会出现数值爆炸，因此不要作为部署默认值。
- `2026-04-27` 服务器 smoke-stable 配置在上述基础上进一步固定为：
  - `depth=12 / public_calibrated LN / uniform attention / fixed_square / clip3.0 / output calibration / party-local share load / bsz=1 / isolate samples`；
  - output calibration JSON：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json`；
  - same-image-list / same-targets 对比下，`class0_4`、`class1_4`、balanced8 三组 smoke 均达到 e2e 100%、original plaintext same subset 100%、prediction match 1.0；
  - balanced8 metrics：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced8_uniform_clip3_calibrated_20260427_201413/e2e_secure_poc/e2e_approx_eval_metrics.json`；
  - 这不是 full-val 证明，扩大样本前必须保持 `E2E_SPU_BATCH_SIZE=1`、`E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`，并确认 candidate 文件名包含 `uniform_fixed_square_clip3p0_eval`。
- balanced16 诊断结果：
  - smoke8 output calibration 在 balanced16 上退化到 E2E `81.25%` / match `0.75`；
  - balanced16 诊断版 output calibration：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_balanced16_diag.json`；
  - 一次性 run 曾在 `i=15` 出现 raw logits 偶发爆炸，fresh-runtime 单样本复跑恢复；
  - patched 诊断汇总：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_20260427_211653/e2e_secure_poc/e2e_approx_eval_metrics_patched_i15.json`，same-subset plaintext `93.75%`、E2E `93.75%`、gap `0.0pp`；
  - 这说明后续扩大样本必须启用 per-sample logits guard/retry，不能只看未 guard 的一次性 aggregate。
- balanced16 chunk3 guarded 结果：
  - `E2E_SPU_BLOCK_CHUNK_SIZE=3` 单样本 smoke 已确认 `spu_forward_graph_mode=reveal_less_block_chunked`，最大 request 从约 `86MB` monolithic 降到约 `22.8MB/21.3MB`；
  - 原始一次性 guarded balanced16 metrics：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_chunk3_guarded_20260427_235545/e2e_secure_poc/e2e_approx_eval_metrics.json`；
  - same-subset plaintext `93.75%`、E2E `93.75%`、gap `0.0pp`、match `0.875`、`finite_logits=true`、隐私字段通过；
  - 代价是慢：`e2e_elapsed_sec≈1474.46s`。后续需要优化 chunk size、runtime 启停/复用和通信统计，不能把该耗时作为最终 demo 性能。

### `client_private_prepare_image.sh`

- 用于“遇到一张新图时，在本地完成隐私输入准备”；
- 输入原图只在客户端/本地机器出现，脚本会本地完成 image preprocess，并输出 split share manifests；
- 输出目录默认在 `artifacts/client_private_inputs/<timestamp>/`；
- 主要输出：
  - `client_pixel_values_debug_share_public_manifest.json`
  - `client_pixel_values_debug_share_party_manifests/p1_share_manifest.json`
  - `client_pixel_values_debug_share_party_manifests/p2_share_manifest.json`
  - `server_e2e_infer_env.sh`
- 本地准备命令：
  - `CLIENT_INPUT_IMAGE=/path/to/new_image.png bash artifacts/server_inference_friendly_pack/client_private_prepare_image.sh`
- 服务器推理命令：
  - `source artifacts/client_private_inputs/<run>/server_e2e_infer_env.sh`
  - `bash artifacts/server_inference_friendly_pack/run_e2e_secure_approx_deploy.sh infer`
- 注意：当前 share 语义仍是 `debug_float_additive_share_not_production_mpc_share`，生产部署时必须保证 P1/P2 share 文件分别只发送给各自 party，并通过 TLS / 独立主机 / 访问控制隔离；服务器后端不能接收原图。

### Web demo 的浏览器端隐私分析

- `run_web_demo.sh` 启动的页面现在主按钮走浏览器本地分片流程：
  - 浏览器读取图片；
  - Canvas 在本地完成 resize / center crop / normalize；
  - 浏览器生成 `share0/share1`；
  - 调用 `/api/e2e/analyze_private_shares` 上传二进制 share；
  - 后端只把 share 落成 party manifest，并调用 `run_e2e_secure_approx_deploy.sh infer`；
  - 页面显示最终类别、概率、candidate JSON 和隐私字段。
- 该页面主流程不再调用 `/api/upload` 上传原图；旧 CPU/SPU sidecar endpoint 默认禁用，只有设置 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1` 时才作为调试路径开放。
- 当前 web demo 仍是单进程演示接口，会同时接收两份 share 并写入本机目录；它证明“网页端不上传原图 / 不上传完整 pixel_values”的产品交互，但生产环境还应拆成独立 P1/P2 上传端点和主机。
- Web 后端默认要求 public layer norm calibration JSON 已存在；缺失时会直接报错，不会隐式启动可能很慢的校准流程。只有调试时才设置 `WEB_DEMO_AUTO_CALIBRATE_E2E=1` 让页面自动执行 `make-calib-pixels` 和 `calibrate`。
- Web 后端会优先读取 `WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON` / `E2E_OUTPUT_CALIBRATION_JSON`；未显式设置时会在 `artifacts/server_pipeline_run/e2e_output_calibration*.json` 中选择最新文件。当前服务器演示应使用 `e2e_output_calibration_uniform_clip3_smoke8.json`，并保持 `WEB_DEMO_E2E_ACTIVATION_CLIP_VALUE=3.0` 默认值。
- 新版 E2E candidate JSON 会写入 `prediction_preview`，Web 后端读取结果时不再额外 `torch.load(.pt)`，避免在页面后处理阶段再次触发 `import torch`。
- 若服务器刚经历过 torch/native import 卡死，先只运行 `$PYTHON_BIN tools/transshield_chat_demo.py --help` 验证 Web 后端轻量启动链路；确认不卡后再启动 `run_web_demo.sh`，不要直接跑 `run_e2e_secure_approx_eval.sh`。

### `run_e2e_secure_approx_eval.sh`

- 用于计算 e2e 近似路径相对原始明文路径的准确率差，不再只看 1-2 张 smoke 样本；
- 默认评测数据目录：`/data/wyb/pneumoniamnist_imagefolder_subset/val`；
- 默认样本数：`E2E_EVAL_MAX_SAMPLES=8`，建议先用 8/16 做 smoke，再逐步扩大；
- 当前稳定 smoke 建议显式设置 `E2E_SPU_ACTIVATION_CLIP_VALUE=3.0`、`E2E_OUTPUT_CALIBRATION_JSON`、`E2E_SPU_BATCH_SIZE=1`、`E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`；
- isolated per-sample eval 会检查 raw/current logits 是否有限且绝对值不超过 `E2E_APPROX_EVAL_LOGIT_ABS_GUARD`，异常时按 `E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES` fresh-runtime 重试该样本；
- 每个 isolated SPU infer attempt 还应设置 `E2E_ISOLATED_INFER_TIMEOUT_SEC`，避免单个样本 runtime 半死导致整轮 eval 卡住；超时后同样按 retry 逻辑 fresh-runtime 重试。
- 如果日志卡在 `builtin_spu_run req_bytes≈86MB`，说明仍在跑 monolithic full-depth SPU graph；先设置 `E2E_SPU_BLOCK_CHUNK_SIZE=3` 做单样本 smoke，确认 candidate JSON 里 `spu_forward_graph_mode=reveal_less_block_chunked` 后再扩大样本。必要时把 chunk size 降到 `2` 或 `1`。
- 当前 chunk3 能稳定 balanced16，但耗时较高。效率优化应先小步比较 `E2E_SPU_BLOCK_CHUNK_SIZE=4/6`，同时记录 max `req_bytes`、elapsed 和是否卡死；不要直接扩大到 long-run full-val。
- 该脚本会在同一份 `e2e_eval_images.txt` 上同时运行：
  - 原始明文 reference；
  - e2e approximate SPU；
  - 同 target 的准确率差、预测一致率与 e2e SPU LinkDetails 通信量解析。
- 输出：`$E2E_RUN_DIR/e2e_approx_eval_metrics.json`，包含：
  - `original_plaintext_same_subset_argmax_accuracy`
  - `original_plaintext_same_subset_threshold_accuracy`
  - `e2e_argmax_accuracy`
  - `e2e_threshold_accuracy`
  - `argmax_accuracy_gap_e2e_minus_plaintext_pp`
  - `threshold_accuracy_gap_e2e_minus_plaintext_pp`
  - `prediction_match_vs_original_plaintext`
  - `e2e_communication_from_spu_node_logs`
- 运行示例：
  - `E2E_EVAL_DATASET_DIR=/data/wyb/pneumoniamnist_imagefolder_subset/val E2E_EVAL_MAX_SAMPLES=8 bash artifacts/server_inference_friendly_pack/run_e2e_secure_approx_eval.sh`
- Python 预处理、明文 reference、share 生成和 metrics 写出步骤都带 timeout；服务器若在 `import torch` 等 native import 阶段卡死，会返回 `124` 并打印具体卡住的 step。
  - `block9` probe 显示 `attn_out_cls` 已有明显方向漂移，`mlp_out_cls` 是最大幅度误差阶段；后续 sample0 对照也已完成，说明 sample0 虽有 attention 漂移但仍决策一致，而 sample1 因更接近边界并在 MLP/head 后进一步劣化而翻转；
  - 当前下一步更适合生成 sample0 vs sample1 的 block9 对照小报告，而不是继续扩大样本数。
- 服务器端快速查看关键 JSON：
  - `cat "$E2E_DIR/e2e_secure_contract.json"`
  - `cat "$E2E_DIR/client_pixel_values.json"`
  - `cat "$E2E_DIR/plaintext_reference.json"`
  - `cat "$E2E_DIR/static_whole_forward_reference.json"`
  - `cat "$E2E_DIR/e2e_static_whole_forward_candidate_from_server.json"`
  - `cat "$E2E_DIR/e2e_static_whole_forward_compare.json"`
  - `cat "$PACK_DIR/commands.json"`
- 若当前阶段已经进入 `depth=5 -> 6` 归因，不要继续手工 `cat` 散落 JSON；固定：
  - `export E2E_RUN_MAX_SAMPLES=1`
  - `export E2E_STATIC_DEPTH_LIMIT=6`
  - `export E2E_PROBE_BLOCK_INDEX=5`
  - `export E2E_SPU_BATCH_SIZE=1`
  - `export E2E_SPU_PARAMS_MODE=public`
  - 若 full run 还有 `Socket closed` / `Not connected`，再额外加：
    - `export SPU_DISABLE_COLOCATED_OPTIMIZATION=1`
    - `export SPU_RUNTIME_REUSE=0`
  - 然后依次运行：
    - `run_e2e_secure_whole_forward.sh probe-cpu`
    - `run_e2e_secure_whole_forward.sh probe-spu`
    - `run_e2e_secure_whole_forward.sh probe-compare`
  - 最终查看：
    - `cat "$E2E_DIR/block6_probe_compare_cpu_vs_spu_depth6.json"`
- 如果只想抓这一个 run 回本地，不要同步整个 `artifacts/` 根；使用：
  - `mkdir -p /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME`
  - `rsync -avP -e "ssh -p 9001" --prune-empty-dirs --include='*/' --include='*.json' --include='*.md' --include='*.txt' --include='*.log' --exclude='*' wyb@10.204.244.1:/data/wyb/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/ /home/yclcg/Transshield_final/artifacts/server_pipeline_run/$RUN_NAME/`
- 当前 `spu` 模式默认只 reveal final logits；不要加 `--include-intermediates`。当前 POC 仍会在 host 侧加载 plaintext `client_pixel_values.pt` 后再送入 SPU secret sharing，所以还不能表述为生产级“服务器从未接触明文输入”。

### `run_token_pruning_visualization.sh`

- 针对单张输入图片生成 token pruning 可视化；
- 输出 stage 级 overlay 图、trace JSON 和 Markdown 说明；
- 默认输出目录：
  - `artifacts/server_pipeline_run/<RUN_NAME>/token_pruning_visualization/`
- 主要文件：
  - `token_pruning_summary.png`
  - `stage_1_overlay.png`
  - `stage_2_overlay.png`
  - `stage_3_overlay.png`
  - `token_pruning_trace.json`
  - `token_pruning_trace_report.md`
- 适合答辩时解释 `masking` 如何替代直接裁剪。

### `run_single_image_comparison.sh`

- 针对同一张图片同时生成 baseline 与 modified 的单图对照；
- 输出摘要图、JSON 与 Markdown；
- 默认输出目录：
  - `artifacts/server_pipeline_run/<RUN_NAME>/single_image_comparison/`
- 主要文件：
  - `baseline_vs_modified_summary.png`
  - `baseline_vs_modified_comparison.json`
  - `baseline_vs_modified_comparison.md`
- 适合直接做答辩里的“baseline vs modified”案例页。

### `run_web_demo.sh`

- 启动一个最小可用的前后端一体化 Web demo；
- 支持前端上传图片、后端推理、最佳 bundle 摘要与 secure 结果展示；
- 适合做交互式流程展示与答辩演示界面。

### `run_cpu_spu_profile.sh`

- 顺序运行一套 `CPU secure` 与一套 `SPU secure`；
- 分别生成各自的 `secure_profile_summary.json`；
- 额外输出一份 `cpu_vs_spu_profile_report.json` 与 `cpu_vs_spu_profile_report.md`；
- 适合直接补答辩所需的时间 / 通信 profiling。

### `run_standardized_secure_external_benchmark.sh`

- 调用 `external_baselines/MPCFormer/tools/run_transformer_local2pc_server.sh`
- 把 `Transshield` 当前最终模型 proxy 与外部模型 proxy 放进同一个 `local 2PC configurable transformer benchmark`
- 输出：
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.json`
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.md`
- 适合回答：
  - “如果都放进同一个 secure transformer benchmark harness，外部模型 proxy 和本项目 proxy 的通信 / 时间差别怎样？”
- 不适合回答：
  - “full-val 医学图像 pipeline 总通信量谁更低？”
  - “网页单图 live run 和外部 benchmark 数字谁更低？”

### `run_secure_selection_mode_profile_compare.sh`

- 顺序运行两套 `SPU secure`，默认比较：
  - `flat_odd_even`
  - `blockwise_exact_kth`
- 也支持新的：
  - `blockwise_exact_kth`
- 其中 `blockwise_exact_kth` 适合配合 `tools/transshield_blockwise_kth_selection_manifest.py` 生成的 manifest 使用；
- 会优先使用仓库内收口好的 runtime inputs：
  - `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/`
- 如果该目录不存在，再自动寻找可复用的 runtime inputs 来源目录；
- 也支持显式指定：
  - `RUNTIME_INPUT_SOURCE_DIR=<old_run_dir>`
- 每个模式各自产生：
  - `fastpath_profile_summary.json`
  - `secure_profile_summary.json`
- 最后额外输出一份：
  - `selection_mode_profile_compare.json`
  - `selection_mode_profile_compare.md`
- 适合回答：
  - “`phase3_lower_tail` 到底有没有比旧模式更快 / 更省通信”
  - “新的 `blockwise_exact_kth` manifest 是否比旧实验 manifest 更合理”
- 当前服务器结果表明：`blockwise_exact_kth` 已通过 checker / replay，并在同口径 SPU profile 中降低 `network_kth_bridge` 时间；旧 `phase3_lower_tail` 只保留为历史实验开关。
- 仓库根目录的 `run_secure_selection_mode_profile_compare.sh` 现在只是转发到本脚本，避免根目录再维护一份过期默认值。

---

## 7. 权威入口说明

- 当前目录下的 `.sh` 脚本是权威运行入口；
- `commands.json` 是保留的打包快照；
- 如果 `commands.json` 与 shell 包装脚本不一致，以 `.sh` 脚本为准。

---

## 8. 数据集要求

- `TRAIN_DATA_PATH` 应指向 `pneumoniamnist_imagefolder_subset/train`
- `VAL_DATA_PATH` 应指向 `pneumoniamnist_imagefolder_subset/val`
- 目录结构需保持 `ImageFolder` 兼容

---

## 9. 权重命名说明

### baseline

- 默认轻量权重：`artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth`
- 归档完整 checkpoint：`artifacts/archive/baselines/baseline_plaintext_training_checkpoint_full.pth`

### modified

- 默认展示 / 运行 bundle：`artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/`
- 默认轻量权重：`artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/modified_plaintext_eval_checkpoint_light.pth`
- secure replay 所需 pure `state_dict`：`artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/modified_plaintext_model_state_dict.pth`
- 保留的旧正式 bundle：`artifacts/frozen_bundle_full/`
- 归档完整 checkpoint：`artifacts/archive/frozen_bundle_full/modified_plaintext_training_checkpoint_full.pth`

默认比赛流程只依赖轻量权重；完整 checkpoint 主要用于恢复训练与来源追溯。
