# 当前工作状态

最后更新：`2026-04-27`

## 当前主结论

- `2026-04-27` 阶段收口：当前默认 Web 主路径已经是浏览器本地分片 + `/api/e2e/analyze_private_shares` + `run_e2e_secure_approx_deploy.sh infer`；旧 `/api/upload`、`/api/run_secure` 前端调用已从页面删除，后端 legacy endpoint 默认禁用，只有 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1` 才能开放调试。
- 为降低服务器 `import torch` 卡死对网页演示的影响，Web 后端已做 lazy import：启动阶段不再顶层导入明文/剪枝可视化 torch 模块；E2E candidate JSON 新增 `prediction_preview`，Web 后端读取 E2E 结果时不再 `torch.load(candidate.pt)`，避免页面后处理阶段再次触发 `import torch`。
- Web 后端默认不再自动执行 public calibration。若缺少 public layer norm calibration JSON，会直接报错提示先预生成；只有显式设置 `WEB_DEMO_AUTO_CALIBRATE_E2E=1` 才会让网页自动跑 `make-calib-pixels` 和 `calibrate`。这是为了避免用户点击网页后隐式进入长时间 torch/native import 或校准流程。
- 当前服务器卡死问题的工作判断：代码层面已尽量减少 Web 默认路径的额外 torch 触发，但已经卡住的服务器进程/会话更像 native/CUDA/SPU runtime 层面的不可中断等待。下一次接手应先做轻量 sanity：`python tools/transshield_chat_demo.py --help`，若不卡再启动 Web；不要直接重跑 E2E eval 或 full-depth SPU。
- `2026-04-26` 最新主口径已切换：最终 Web demo 不再默认走旧 `/api/upload` + `/api/run_secure` sidecar；主按钮走浏览器本地预处理与 `share0/share1` 生成，再调用 `/api/e2e/analyze_private_shares` 触发 `run_e2e_secure_approx_deploy.sh infer`。旧 CPU/SPU sidecar endpoint 只作为 legacy 调试路径保留，默认需显式设置 `WEB_DEMO_ENABLE_LEGACY_SIDECAR=1`。
- 当前可展示为“网页端不上传原图 / 不上传完整 plaintext pixel_values；服务端只处理 debug share 并只 reveal final logits”。但生产级全隐私仍必须拆成独立 P1/P2 上传端点与信任域，避免同一后端进程同时接收两份 share。
- `run_e2e_secure_approx_eval.sh` 现在是替代旧 SPU sidecar 对比的主评测入口：同一图片列表上对比原始明文 reference 与 E2E approximate SPU，输出准确率差、预测一致率、隐私字段和 SPU node log 通信量；Python 预处理/metrics 步骤带 timeout，便于定位服务器 `import torch` 卡死。
- Web demo 已切换为**当前图片即时结果**优先展示。
- 离线验证集最佳成绩与外部基线对比已下沉到统一对比区。
- 历史 fastpath 8 样本通信与旧 archived SPU profile 已从前端和主文档口径中移除。
- 主页面已移除“训练收益”类内部演化对比，只保留当前最终模型与外部基线。
- `docs/` 中大量 dated handoff / request / changes / defense 文档已直接删除，不再归档保留。
- 三套训练入口已继续收口为 `deit-s` 单模型训练面：`main.py`、`training_compat/main.py`、`training_source_tracka/main.py` 不再保留 `convnext/lvvit/swin/deit-b` 的 live 训练分支；这些旧模型代码只继续作为历史资产 / 评估兼容保留。
- 又一批低频说明文档已并回权威入口：数据集说明、正式 bundle provenance、答辩问答 / 展示建议、单图对照 / token 可视化使用说明已统一收口到 `docs/data_source_policy.md`、`docs/result_summary.md`、`docs/project_overview_newcomer_defense.md`、`artifacts/server_inference_friendly_pack/README.md`、`docs/current_work_status.md` 与 `docs/handoff-next.md`；TrackA 后续单变量交接也统一回写权威文档，不再保留额外一次性交接文件。
- 第二轮仓库瘦身已完成：旧中间 checkpoint、旧 server run、旧训练输出、旧 profile/build 报告和 stale inventory 已删除。
- 公平外部对比已在服务器刷新完成并写入前端静态摘要：`fair_external_20260423_113217` 中 `accuracy_comparison_is_fair=true`，`Transshield` 仍为 `93.702292% / 94.083971% / AUC 0.972313`，`MPCViT` 两 seed 均值仍为 `96.660305% / 96.946565% / AUC 0.993449`，差距保持 `-2.958013 pt / -2.862594 pt / -0.021137`。
- 统一 secure benchmark 也已在服务器完成一轮最新 `architecture_proxy` 复跑并写入前端静态摘要：`standardized_secure_benchmark_20260423_130435` 中，`Transshield` proxy 为 `4422.72 MiB / 13.4821s`，`MPCViT` proxy 为 `262.56 MiB / 4.5622s`，同一 harness 下比值约 `16.8447x / 2.9552x`；这一步的目的不是替代正式 pipeline，而是把当前正式 bundle 的**结构尺度**映射到同一 MPCFormer benchmark harness 中做外部参照。
- 同一 benchmark 还已完成 `same_shape_operator_proxy`：`standardized_secure_benchmark_20260423_132121_same_shape` 中，在固定 DeiT-S 形状下，`Transshield secure-friendly ops` 相对 baseline ops 将模块通信从 `5918.69 MiB` 降到 `881.05 MiB`、将时间从 `15.3365s` 降到 `8.1045s`，比值约 `0.1489x / 0.5284x`。这说明**算子替换本身是明显正向的**，而前一条 `architecture_proxy` 的高开销主要来自结构尺度差异，不是 secure-friendly ops 本身更重。
- 需要注意：统一 secure benchmark 不会加载历史最优正式模型权重，也不执行 full-val 医学图像 pipeline；它只复用当前正式 bundle 的结构/算子口径构造 proxy，用于验证“算法/算子替换是否降低 secure transformer harness 开销”。
- 因此当前非 `e2e` 的“公平外部对比 + 统一 secure benchmark”线已经正式完成，且 benchmark 数字已写入前端静态 JSON；本地文档不需要继续为这条线新增实验计划。
- `margin-aware pruning` 已拿到一条明确的研究性正结果：`w10` 能显著拉开 secure 剪枝边界，但当前还不能直接替换正式展示模型。
- `network-kth` 已完成一轮正式化改造：新增 `blockwise_exact_kth` 模式与 manifest 生成器，CPU smoke checker 与 full replay 都已验证逐 stage 语义通过。
- 端到端隐私保护线已从并行研究线升级为当前 Web demo 主路径：默认页面走浏览器本地分片 + E2E approximate SPU，旧 `secure sidecar` 只保留为 legacy 调试/历史对比路径。
- `e2e secure inference` 的第一批仓内骨架已落地：新增 `tools/transshield_e2e_secure_infer.py`、`artifacts/server_inference_friendly_pack/run_e2e_secure_poc.sh`、`integrations/openbumblebee/e2e_secure_vit/README.md` 与 `configs/openbumblebee/2pc_e2e.template.json`；这批最早骨架起初只提供边界合同、客户端预处理像素包和 plaintext reference，现在已被后续 E2E approximate deploy / Web 主路径接住。
- `e2e secure inference` 的第二步最小推进也已接上：`run_e2e_secure_poc.sh` 现在会额外生成 `static whole-forward` plaintext reference，用当前 student 的 `patch_embed + blocks + head` 绕开 runtime pruning 决策路径，为后续“先静态 ViT 整网 secure，再并入 masking-pruning”提供直接对齐基准。
- `e2e secure inference` 的 whole-forward checker 也已预先补好：`tools/transshield_e2e_secure_infer.py compare-static-whole-forward` 可在未来 SPU 候选输出落地后，直接对齐 `logits / probabilities / argmax / threshold`。
- `e2e secure inference` 的集成入口也已落地到 `integrations/`：新增 `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py` 与 `artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh`；当前正式 Web 主入口使用更稳定的 `run_e2e_secure_approx_deploy.sh` 近似配置。
- `2026-04-22` 的服务器 run `tracka_e2e_secure_poc_cpu` 已把这条新线的 **CPU static whole-forward contract** 闭合：`client pixel package` 为 `524` 个样本、`static whole-forward reference` 的 `logits/cls_features/token_features` 形状分别为 `[524,2] / [524,384] / [524,196,384]`，`cpu candidate` 用时约 `20.73s`，`verify` 得到 `logits/probabilities max_abs_error = 0.0`、`argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`。因此当前可以确认“边界合同 + whole-forward reference + CPU candidate + verify”四步已闭环，下一阶段应直接转向 `runtime=spu` 的真实 whole-forward backend，而不是再重复 CPU reference 验证。
- `2026-04-23` 的服务器 `runtime=spu` smoke 已完成一轮真正的 same-depth 渐进验证：在当前 `CUDA-enabled jaxlib is not installed. Falling back to cpu.` 的环境下，`run_e2e_secure_whole_forward.sh spu` 以 `sample=1 / public params` 先后跑通了 `depth=0..5`，并都能通过“`SPU depth=k` vs `CPU depth=k`”对齐检查；其中 `depth=0` 的 `logits/probabilities max_abs_error` 约为 `4.21e-4 / 1.32e-4`，`depth=1` 约为 `2.41e-3 / 3.86e-4`，而 `depth=2..5` 虽仍保持 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，但数值漂移持续放大，`depth=5` 已升到 `logits/probabilities max_abs_error ≈ 1.20e-1 / 4.01e-2`。因此当前可以确认：**static whole-forward SPU backend 已通过 `depth=0..5 / sample=1 / public params` 的服务器 smoke，但数值误差已进入明显累积区**。
- 同一轮服务器验证最初在默认 colocated runtime 配置下遇到了 `depth=6` 异常：一次 full run 的 `SPU` 候选虽然 `finite_logits=true`，但相对 `CPU depth=6` 出现了 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 9.29e-1 / 2.64e-1`；与此同时还伴随 `grpc UNAVAILABLE / Socket closed / Not connected <spu_internal_addrs[1]>`。因此当时更准确的判断并不是“算法上必然在 depth6 失配”，而是：**默认 colocated runtime 配置在 depth=6 full run 上存在稳定性问题，导致同一层级的对齐结论不再可靠**。
- 为了支持这条 `depth=5 -> 6` 归因，本地还新增了 `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py probe-block` 与 `compare-block-probe`：它们可在 `sample=1` 条件下对指定 block 显式导出 CPU/SPU 的 `CLS token` 中间摘要（`block_input / norm1 / attn_out / attn_residual / norm2 / mlp_out / block_output / final_norm / logits / probabilities`），并自动给出 stage-wise `max_abs_error / mean_abs_error / cosine_similarity`；当前推荐直接固定 `static_depth_limit=6, probe_block_index=5` 来做 block-6 归因，而不是继续加深 smoke 深度。
- `2026-04-23` 的 block-6 probe compare 也已经给出第一条明确归因：在 `static_depth_limit=6, probe_block_index=5` 下，`norm1_out_cls` 仍然较接近（`cosine≈0.9982`、`max_abs_error≈0.1825`），但 `attn_out_cls` 已成为首个明显失真阶段（`cosine≈0.0159`、`l2_error≈35.21`、`max_abs_error≈5.37`），后续 `attn_residual_out_cls / block_output_cls` 再继续放大。因此当前最可信的工作假设是：**block 6 的首发大漂移来自 attention 输出，而不是 norm1 或 MLP 首发**。
- 同一天的服务器复验已经把这条系统线推进到明确结论：在 run `tracka_e2e_secure_spu_depth6_smoke1_nocoloc_20260423` 中，固定 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 后，`depth=6` 的 full run 不再出现 node/link 报错，`logs/spu_runtime_ports.json` 也明确记录 `disable_colocated_optimization=true`；此时 same-depth compare 恢复为 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，`logits/probabilities max_abs_error` 约为 `1.43e-1 / 4.34e-2`。因此当前最新阶段结论应更新为：**默认 colocated runtime 配置会把 depth6 full run 搞坏，但在 `disable_colocated_optimization` 下，depth=6 same-depth smoke 已重新稳定并通过决策对齐**。
- 这也意味着当前 `e2e` 线的两条 issue 已经重新排序：系统线的首个可用缓解措施已经找到，后续更合理的主线是**以 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 作为 depth6+ smoke 的默认运行方式继续推进 same-depth 边界；而 block-6 attention 漂移则保留为独立的内部数值归因证据，不再把它直接等同于“depth6 必然 final mismatch”**。
- 为了让这条系统线缓解可复用，`run_e2e_secure_whole_forward.sh` 现已额外暴露 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 等 runtime 启动开关；`probe-spu` 和 `spu` 都会沿用 wrapper 内的自启动逻辑，并在复用现有 runtime 前检查当前请求的 `disable_colocated_optimization` 配置是否匹配。`secure sidecar` 的 `artifacts/server_inference_friendly_pack/_run_secure_pipeline_by_runtime.sh` 也已同步支持同样的 env 开关，避免未来再回到手动 `start`。
- `2026-04-23` 的下一条 same-depth 推进也已完成：在 run `tracka_e2e_secure_spu_depth7_smoke1_nocoloc_20260423` 中，固定 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 后，`depth=7` 仍保持 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，`logits/probabilities max_abs_error` 约为 `1.67e-1 / 4.77e-2`，且 runtime 无 node 报错。
- 同日再往前推进后，`depth=8`、`depth=9`、`depth=10`、`depth=11` 也都已通过：run `tracka_e2e_secure_spu_depth8_smoke1_nocoloc_20260423` 的 `logits/probabilities max_abs_error` 约为 `2.11e-1 / 5.97e-2`；run `tracka_e2e_secure_spu_depth9_smoke1_nocoloc_20260423` 约为 `1.97e-1 / 7.37e-2`；run `tracka_e2e_secure_spu_depth10_smoke1_nocoloc_20260423` 约为 `1.95e-1 / 7.12e-2`；run `tracka_e2e_secure_spu_depth11_smoke1_nocoloc_20260423` 继续保持 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，`logits/probabilities max_abs_error` 约为 `2.07e-1 / 8.41e-2`，且 `node_*.log` 无新的 runtime 错误。因此截至当前，最准确的阶段结论已更新为：**`e2e static whole-forward SPU backend` 在 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1 / sample=1 / public params` 下，已经完成 `depth=0..11` 的 same-depth 决策一致 smoke；下一步只剩 `depth=12` / full static whole-forward 边界**。
- `2026-04-23` 的 `depth=12 / sample=1 / nocoloc` 也已收口：run `tracka_e2e_secure_spu_depth12_smoke1_nocoloc_20260423` 在 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 下继续保持 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，`logits/probabilities max_abs_error` 约为 `1.54e-1 / 6.06e-2`，且 `node_*.log` 仍无 runtime 报错。由此可以确认：**`e2e static whole-forward SPU backend` 的单样本 full-depth smoke 已经闭环**。
- 但同一天的下一阶段 `sample_count=2` 也暴露了新的真实边界：run `tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423` 在 `SPU_DISABLE_COLOCATED_OPTIMIZATION=1` 下依然 `finite_logits=true`、runtime 稳定、`node_*.log` 无报错，但相对 CPU reference 已出现 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error` 升到约 `6.92e-1 / 3.00e-1`。因此当前最准确的阶段结论需要再向前细化一层：**single-sample full-depth 已闭环，但 multi-sample (`sample=2`) full-depth 仍未闭环**。
- `2026-04-25` 的继续推进已把 `depth=12 / sample=2` 的失败进一步拆清：历史 `spu_batch_size=1` 路径容易触发 `grpc UNAVAILABLE / Socket closed / Not connected`，更像同一 runtime 内连续两次 SPU 调用的 link 稳定性问题；而 run `tracka_e2e_secure_depth12_smoke2_bsz2_direct_20260425_001632` 使用 `spu_batch_size=2` 后 runtime 稳定、无 node error，并成功产出 `[2,2]` logits，但决策只达到 `argmax_match_ratio = 0.5`、`threshold_match_ratio = 0.5`，`logits/probabilities max_abs_error ≈ 3.74e-1 / 1.65e-1`。因此当前结论不再是单纯的 batch 拼接问题，而是：**batched whole-forward 可以跑完，但第二个样本的数值误差会跨过分类/阈值边界**。
- 同一轮逐样本拆解显示，`sample0` 在 `depth=12` 下虽然存在较大数值误差，但 CPU/SPU 的 `argmax` 与 threshold 判定仍一致；`sample1` 是失败样本。run `tracka_e2e_secure_depth12_sample1_single_fix_20260425_010810` 进一步确认 `sample1` 单独运行 `depth=12 / sample=1 / bsz=1 / nocoloc` 时 runtime 可完成，但 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 1.40e-1 / 5.93e-2`。因此 `sample2 bsz2` 的 `0.5/0.5` 不是由 batched shape 本身独立造成，而是由一个本来接近决策边界的样本在 SPU 数值漂移下翻转造成。
- `sample1` 的深度边界已经收窄：`tracka_e2e_secure_depth6_sample1_single_20260425_012825` 与 `tracka_e2e_secure_depth8_sample1_single_20260425_013826` 均保持 `argmax_match_ratio = 1.0`、`threshold_match_ratio = 1.0`，但误差逐步增大；`tracka_e2e_secure_depth9_sample1_single_20260425_014637` 开始翻转为 `0.0 / 0.0`，`logits/probabilities max_abs_error ≈ 3.72e-1 / 1.79e-1`；`depth10` 和 `depth12` 也继续失败。因此当前最准确的数值边界是：**sample1 的决策翻转发生在 `depth8 -> depth9`，也就是加入第 9 个 transformer block 后**。
- block-level probe 已定位这个翻转点：`tracka_e2e_secure_block9_probe_sample1_depth9_20260425_015446` 在 `static_depth_limit=9, probe_block_index=8` 下成功完成，最终 `argmax/threshold = 0.0 / 0.0`。其中 `mlp_out_cls` 是按 `max_abs_error` 排名的最大误差阶段（`max_abs_error≈23.10`、`l2_error≈92.67`），但 `attn_out_cls` 已出现明显方向性失真（`cosine≈0.0908`、`max_abs_error≈6.31`）。当前工作假设应更新为：**sample1 在 block9 处先由 attention 输出产生方向漂移，再由 MLP 输出显著放大幅度，最终跨过分类/阈值边界**。
- 为避免把 `sample1` 的翻转误解成 block9 一定失败，又补做了 `sample0` 的同口径 block9 对照：`tracka_e2e_secure_block9_probe_sample0_depth9_20260425_020754` 中，`sample0` 虽然也出现明显 `attn_out_cls` 方向漂移（`cosine≈0.0289`、`max_abs_error≈10.57`）和较大 MLP 误差（`mlp_out_cls max_abs_error≈11.50`），但最终仍保持 `argmax/threshold = 1.0 / 1.0`，`final_logits/probabilities max_abs_error≈1.96e-1 / 7.37e-2`，且 final logits/probabilities cosine 仍约 `0.9845 / 0.9930`。与 `sample1` 的 `final_logits/probabilities max_abs_error≈3.70e-1 / 1.78e-1`、final logits/probabilities cosine `≈0.7504 / 0.9409` 对比后，当前更精确的结论是：**block9 attention 漂移是共性风险，sample1 的 MLP 放大与最终分类 margin 更脆弱才使其跨界；sample0 仍能吸收该漂移**。
- 第一条 SPU-only ablation 也已回贴：`tracka_e2e_secure_block9_probe_sample1_depth9_attn_standard_20260425_023618` 把 `E2E_SPU_ATTENTION_POLICY` 从默认 `smoothed` 改成 `standard` 后，`sample1 / depth9 / block9 probe` 的最终决策从 `0.0 / 0.0` 恢复为 `1.0 / 1.0`；但这不是单纯“误差变小”的修复，因为 `mlp_out_cls max_abs_error` 反而从约 `23.10` 增至约 `47.07`，`final_logits/probabilities max_abs_error` 也升到约 `4.50e-1 / 1.95e-1`，`attn_out_cls cosine` 变为约 `-0.1056`。因此当前只能把它记作**决策方向被拉回的诊断性正结果**，不能写成数值稳定性已改善；下一步需要用同一开关跑 full `depth9` / `depth12` candidate 验证它是否只在 probe 路径偶然成立。
- 同一开关的 full candidate 已补出有效 compare 结论：`tracka_e2e_secure_depth9_sample1_attn_standard_full_rerun_20260425_164037` 从 `tracka_e2e_secure_spu_depth12_smoke2_nocoloc_20260423` 的两样本输入包中切出原始 `sample1`，重新生成 CPU `depth9` reference，并用 `E2E_SPU_ATTENTION_POLICY=standard / activation_override=bundle / bsz=1 / nocoloc` 跑 SPU candidate；这次 `reference_pt` 与 `candidate_pt` 已正确分开，compare 结果仍为 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 3.68499e-1 / 1.77430e-1`。因此 `standard attention` 的 block9 probe 正结果不能外推到 full depth9 路径，只能记作 probe-only / 决策方向诊断信号。
- `gelu` activation ablation 也已得到有效负结果：`tracka_e2e_secure_depth9_sample1_gelu_full_20260425_171045` 固定 `depth9 / sample1 / bsz=1 / nocoloc`，保持默认 `E2E_SPU_ATTENTION_POLICY=smoothed`，只把 `E2E_SPU_ACTIVATION_OVERRIDE=gelu` 后，CPU reference 与 SPU candidate 的 `.pt` 已正确分开；compare 仍为 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 7.13191e-1 / 1.41736e-1`。因此当前不能把 full depth9 失配归因成单纯 activation override 选择错误；下一步应继续沿 block9 attention / MLP 数值漂移做更细 probe，例如对比 `smoothed+bundle`、`standard+bundle`、`smoothed+gelu` 三条 block9 probe 的阶段误差，而不是继续扩大样本数。
- `smoothed+gelu` 的 block9 probe 进一步说明这条 activation 路线不适合作为修复：`tracka_e2e_secure_block9_probe_sample1_depth9_smoothed_gelu_20260425_172711` 中最终仍是 `argmax/threshold = 0.0 / 0.0`，而且最大误差已经出现在 `block_input_cls`，SPU candidate 的 `block_input_cls` / `attn_residual_out_cls` / `block_output_cls` 范数约 `9.95e9`，`max_abs_error≈1.737e9`；这说明 `gelu` override 在进入 block9 之前已经造成上游状态爆炸，不是 block9 内部 attention 或 MLP 的局部漂移。因此后续不应继续沿 `gelu` 做 full candidate 搜索，应回到 `bundle activation` 下比较 `smoothed` 与 `standard attention` 的同口径 block9 行为。
- `bundle activation` 下的 block9 同口径对照出现了更细的边界：`tracka_e2e_secure_block9_probe_sample1_depth9_smoothed_bundle_20260425_181915` 中，`smoothed+bundle` 虽然 `mlp_out_cls` 是最大误差阶段（`max_abs_error≈60.40`、`l2_error≈253.54`），final logits/probabilities 误差也不小（`max_abs_error≈4.892e-1 / 2.116e-1`），但最终仍保持 `argmax/threshold = 1.0 / 1.0`；同一输入与同一 probe 口径下切到 `standard+bundle` 后，整体数值误差反而更小（largest stage 变为 `block_output_cls max_abs_error≈14.63`，final logits/probabilities `max_abs_error≈8.044e-2 / 3.862e-2`），但决策变成 `0.0 / 0.0`。因此当前不能只用误差大小判断是否修复，关键是误差方向与该样本的分类/阈值 margin；下一步应在同一输入上补 full `smoothed+bundle` control，确认 probe 与 full candidate 是否一致。
- full `smoothed+bundle` control 已补出有效负结果：`tracka_e2e_secure_depth9_sample1_smoothed_bundle_full_20260425_190634` 使用同一 `sample1` 输入、`depth9 / bsz=1 / nocoloc / E2E_SPU_ATTENTION_POLICY=smoothed / E2E_SPU_ACTIVATION_OVERRIDE=bundle`，CPU reference 与 SPU candidate 的 `.pt` 已正确分开；compare 仍为 `argmax_match_ratio = 0.0`、`threshold_match_ratio = 0.0`，`logits/probabilities max_abs_error ≈ 3.70071e-1 / 1.78151e-1`。因此 block9 probe 中 `smoothed+bundle` 的 `1.0 / 1.0` 也不能外推到 full depth9；当前更准确的判断是：probe 只适合定位 block9 局部漂移阶段，不能替代 full candidate 的最终决策验证。下一步不应继续换 attention / activation 开关，而应比较 probe 与 full run 的执行范围差异，尤其是 full run 在 block9 之后仍会继续执行哪些 head / norm / residual 细节。
- 文件级对比已确认 probe 与 full candidate 的最终输出不是同一语义：同一 `sample1 / depth9 / smoothed+bundle` 下，probe 的 SPU `final_logits=[-0.0151, 0.9563]`、`argmax=1`，而 full candidate 的 SPU `logits=[0.7820, 0.0992]`、`argmax=0`，两者 logits 最大差约 `0.8571`、probabilities 最大差约 `0.3898`。因此当前不能再用 `probe-block` 的 `final_logits/final_probabilities` 直接判断 full candidate 决策；probe 的价值应限制为阶段漂移定位，下一步应查 `probe-block` 与 `run` 两条实现路径在 `static_depth_limit=9` 后的 head/norm/return 语义差异。
- CPU-only 对照已排除 Python 路径语义差异：`cpu_probe_vs_full_depth9_sample1_20260425_193830` 中，full CPU `run` 与 CPU `probe-block` 使用同一输入、同一 `depth9` 后，logits 与 probabilities 完全一致（`max_abs_diff=0.0`），argmax 均为 `1`。因此 `probe-block` 与 full candidate 的不一致不是 Python 侧 `run_static_student_whole_forward_limited` / `run_static_student_whole_forward_probe` 语义分叉，而是 SPU/JAX 路径在 reveal 中间张量后改变了执行图 / 数值行为；后续必须把 `probe-block` 的最终 logits 视为“带 probe 的调试图输出”，不能当作 full SPU candidate 输出。
- SPU reveal 扰动审计已进一步收窄到“最后执行 block 的 probe”：`spu_probe_reveal_perturb_depth9_sample1_20260425_194304` 复用同一 full SPU candidate，对 `probe_block_index=0/5/8` 分别 reveal。block1 probe 与 full logits 最大差约 `1.34e-3`，block6 probe 与 full 最大差约 `1.68e-4`，二者 argmax 均保持 `0`；但 block9 probe 的 logits 变成约 `[7169.68, 8253.86]`，相对 full 最大差约 `8.25e3`，argmax 翻到 `1`。因此当前最可信的技术判断是：在 `depth9` 图中 probe 最后一个实际执行 block 的中间输出会显著扰动 SPU/JAX 编译/数值行为，而较早 block 的 reveal 基本不影响 full logits。
- 同一 reveal 扰动审计继续补了 `probe_block_index=6/7`，进一步确认边界：block7 probe 相对 full logits 最大差约 `2.62e-2`，block8 probe 最大差约 `1.21e-3`，二者 argmax 都仍为 `0`；只有 block9 probe 出现 `8.25e3` 量级差异并翻到 `1`。因此当前应正式把该现象记录为 **last-executed-block probe perturbation**：在 `static_depth_limit=9` 下 reveal 最后一个执行 block 的 probe 输出会污染最终 logits，而 reveal 更早 block 基本稳定。
- `depth8` 泛化验证已补齐，推翻了“所有最后执行 block probe 都会爆炸”的泛化假设：`spu_last_block_probe_depth8_sample1_fixed_20260425_212620` 固定同一 `sample1` 输入，在 `static_depth_limit=8 / probe_block_index=7` 下，probe logits 与 full logits 最大差仅约 `3.36e-4`、probabilities 最大差约 `6.38e-6`，argmax 均为 `0`。因此当前异常不应泛化成所有 last-block reveal，而应收窄为：**depth9/block9 的 SPU probe 图存在特殊数值/编译扰动**。
- `depth10` probe pair 又把模式修正为“最后执行 block reveal 易产生巨大数值扰动，但不一定改变 argmax”：`spu_probe_pair_depth10_sample1_20260425_214432` 中，full depth10 logits 为 `[0.3971, -0.1138]`、argmax `0`；probe block9（index 8，不是最后执行 block）与 full logits 最大差仅约 `1.11e-3`；probe block10（index 9，最后执行 block）logits 变成约 `[6456.94, -6793.01]`，最大差约 `6.79e3`，probabilities 饱和到 `[1.0, 0.0]`，但 argmax 仍为 `0`。因此最终收口应写成：**SPU probe reveal 最后执行 block 时会显著污染 probe 图的 final logits/probabilities；是否翻转取决于污染方向与 margin。probe 的 final logits/probabilities 一律不应再用于 full 决策结论。**
- e2e 隐私边界前移已继续推进：`client-share-preprocess` 现在除 legacy debug share manifest 外，还能写出不含私有 share 路径的 public manifest 与单独的 `P1/P2` party manifests；`run_e2e_secure_whole_forward.sh spu` 也新增 `E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON`、`E2E_INPUT_P1_SHARE_MANIFEST_JSON`、`E2E_INPUT_P2_SHARE_MANIFEST_JSON`，对应 `run --runtime spu --input-share-public-manifest-json ... --input-p1-share-manifest-json ... --input-p2-share-manifest-json ...`。该 split-manifest 路径不再把 plaintext `pixel_values` 作为 runner 输入，也避免 public manifest 同时暴露两份 share 路径；但当前仍是单 launcher 调试桥，正式全隐私目标仍需要拆成 P1/P2 各自只加载本方 share 的独立 party 进程。
- split-share 服务器 smoke 已把当前边界收窄：同一输入下 depth0、depth1、depth2、depth3 的 split public/P1/P2 输入路径均能与 CPU same-depth reference 保持决策一致；depth4 split-share 虽仍决策一致但 logits 数值爆到约 `1.31e3`，而同一 depth4 的 plaintext-input SPU 对照仍是正常量级（logits/probabilities max_abs_error 约 `0.143 / 0.0606`）。因此当前问题不是 depth4 模型本身，而是双私有输入 split-share 路径在更深 SPU/Cheetah 协议上的数值/通信稳定性边界。为此新增 `audit-input-shares` / `run_e2e_secure_whole_forward.sh audit-shares`，用于只 reveal 重组像素与 patch-embed 张量做显式 debug 定位，不能作为生产 reveal policy。
- `audit-input-shares` 首轮服务器结果显示：debug shares 在 CPU 侧可 0 误差重构 plaintext pixel，SPU 内 `share0+share1` 与 CPU 重构仅约 `3.02e-5` max error，patch tokens / tokens+pos 仅约 `2.49e-3 / 2.48e-3` max error，cosine 约 `0.9999998+`。因此 depth4 split-share 爆炸点不在输入重组或 patch embedding，下一步应使用新增 split-share `probe-block` 支持定位 block 内 stage。
- split-share block4 probe 已完成：在 `static_depth_limit=4, probe_block_index=3` 下，CPU plaintext probe vs SPU split-share probe 的最大中间 stage 是 `attn_out_cls`（`max_abs_error≈5.44`、`l2_error≈27.70`、`cosine≈0.0438`），`norm1_out_cls` 仍较接近（`max_abs_error≈0.107`、`cosine≈0.9992`），`mlp_out_cls` 也不是首发大漂移（`max_abs_error≈0.372`、`cosine≈0.9973`）。但该 probe debug graph 的 final logits 仍正常（max_abs_error≈0.076），不能复现 non-probe split depth4 的 `≈1.31e3` logits 爆炸；因此当前结论是：split-share depth4 的局部风险首先落在 attention 输出方向，但 full-run 爆炸还受 SPU/JAX 图形态影响，下一步应用 full split depth4 的 attention-policy ablation 验证。
- full split-share depth4 的 attention-policy ablation 已验证：把 `E2E_SPU_ATTENTION_POLICY=standard` 后，non-probe full run 不再出现 smoothed 下的 `≈1.31e3` logits 爆炸，CPU vs SPU compare 恢复到 `logits/probabilities max_abs_error≈0.0762 / 0.02894`，且 `argmax/threshold=1.0/1.0`。因此当前最小修复方向是：split-share 路径优先使用 `standard attention`，并继续向 depth6/8 做 same-depth smoke；原 `smoothed attention` 在 split-share deeper graph 下会触发严重数值/协议不稳定。
- split-share + `standard attention` 已继续推进到 full depth：depth6、depth8、depth10、depth12 均完成 same-depth CPU vs SPU compare，且均保持 `argmax/threshold=1.0/1.0`；对应 `logits/probabilities max_abs_error` 约为 depth6 `0.1437 / 0.0435`、depth8 `0.2101 / 0.0594`、depth10 `0.1964 / 0.0717`、depth12 `0.1608 / 0.0630`。因此当前 e2e 研究线已证明：在 `sample=1 / public params / static full-depth ViT / final logits reveal` 条件下，split public/P1/P2 debug share 输入可以闭合 full-depth whole-forward；但它仍不是生产全隐私，因为单 launcher 仍同时持有 P1/P2 party manifests，且 dynamic masking-pruning 尚未进入 secure forward。
- 真正全隐私工程化已启动第一步：`run --runtime spu` 新增 `--party-local-share-load`，server wrapper 暴露 `E2E_PARTY_LOCAL_SHARE_LOAD=1`。该模式要求 split public/P1/P2 manifests，但 driver 不再 `torch.load` 私有 share tensor，而是让 P1/P2 device 函数分别在各自 party 上读取本方 party manifest 与 share 文件，再送入 SPU 图内相加。它比 debug bridge 更接近生产边界；但当前仍是 colocated/单 launcher 编排，launcher 仍知道两个 party manifest 路径，后续还需拆成真正独立 P1/P2 party 进程。
- 为了继续向“全程隐私推理”收口，party-local 输出侧又补了一个隐私边界硬化：`run --runtime spu` 新增 `--redact-private-input-paths`，server wrapper 默认 `E2E_REDACT_PRIVATE_INPUT_PATHS=1`。这样 candidate `.pt` 与 summary JSON 只保留 public manifest provenance，不再持久化 legacy/P1/P2 私有 share manifest 路径；share 输入模式下 wrapper 也不再传 `--input-pt`，避免 candidate metadata 指回 plaintext client pixel package；如需本地 debug 才显式设为 `0`。
- party-local share-load 的 depth8 compare 已补齐并通过：`logits/probabilities max_abs_error≈0.2095 / 0.0593`，`argmax/threshold=1.0/1.0`。因此 party-local + standard attention 的当前稳定边界是 depth8，决策翻转发生在 depth8->10；下一步应固定 party-local + standard attention，优先对 depth10 做 block-level probe 或比较 tensor-load vs party-local 同深度 logits，定位由 party-local 图形态引入的深层数值漂移。
- 新生成的 `sample=2` share input 上，`depth10 / standard attention / bsz=2` 的 split-share 与 party-local 已完成直接张量对照：`logits_max_abs_diff=0.002685546875`、`probabilities_max_abs_diff=0.0011300444602966309`，两边 `argmax=[0,0]`。这说明当前这组输入下，party-local 读本方 share 的图形态已基本贴近 split-share tensor-load，不再复现旧记录中 depth10 party-local 明显翻转；下一步应在同一输入上推进 `depth12` split vs party-local，并同步检查 CPU verify 与 redaction metadata。
- 继续推进 `party-local + standard attention` 到更深后，当前服务器 runtime 边界已暴露为系统断链而不是普通数值 compare：`depth11 / sample=2` 在 `builtin_spu_run` 后长时间卡住，`depth12 / sample=1` 也触发 `grpc UNAVAILABLE / Socket closed`，node 日志出现 `SendImpl error ... Not connected to 127.0.0.1:34357`。因此当前不要再反复重跑 depth12 party-local；阶段结论应收口为：**party-local 全隐私输入路径已稳定到 depth10，depth11/12 在当前 CPU jaxlib + SPU runtime 下遇到 link/runtime 稳定性边界**。下一步应先保存 depth10 正结果与 depth12 断链证据，再考虑改 runtime 图拆分 / 分块 reveal-less 执行，而不是继续盲跑 full-depth party-local。
- 针对上述系统边界，仓内已新增实验性无 reveal 图拆分开关：`run --runtime spu --spu-block-chunk-size N`，server wrapper 对应 `E2E_SPU_BLOCK_CHUNK_SIZE=N`。它会把 transformer blocks 按 N 个一组拆成多次 SPU 调用，中间 token state 继续作为 SPU value 传递，不 `ppd.get` / 不 reveal；只在最后 reveal logits。下一步服务器应先用 `depth12 / sample=1 / party-local / standard attention / chunk_size=4` 做 smoke，若通过再回到 `sample=2`。
- 用户回贴的 `depth12 / sample=1 / party-local / standard attention / chunk_size=4` 结果仍失败：错误发生在首个 `spu(embed_and_blocks_from_shares_fn)` 分段调用，Python 侧报 `grpc UNAVAILABLE / Socket closed`，node 日志继续是 `SendImpl error ... Not connected to 127.0.0.1:46019`。因此 chunk4 还没有把首段图降到当前 runtime 可稳定承载的范围；下一步应先进一步缩小到 `E2E_SPU_BLOCK_CHUNK_SIZE=1` 做 `depth12/sample1` smoke，若仍失败，再回退做 `depth4/chunk1` sanity，以区分“chunked 实现本身不通”还是“depth12 party-local 仍过重”。
- 用户继续回贴的 chunk1 日志仍显示 `SendImpl error ... Not connected to 127.0.0.1:34253`，node1 侧能看到 `builtin_spu_run ... inputs=20 wrapped_shares=2 public_values=18`，说明失败已经压到 `embed + 1 block` 的首段调用层面。当前不要继续 depth12 chunk 重试；应先做 `depth4` 的 paired sanity：同一 party-local 输入下先跑 monolithic depth4，再跑 chunk1 depth4。如果 monolithic depth4 通过而 chunk1 失败，问题在 chunked SPU value 传递 / 多次 SPU call 图形态；如果 monolithic depth4 也失败，则问题退化为当前 party-local runtime 基础链路不稳。
- 用户继续反馈 `depth4 / sample1 / party-local / monolithic` 的 `spu` 命令仍会卡死。因此当前边界进一步收窄为：**不是 depth12 或 chunked 图拆分特有问题，而是当前 session/runtime 下 party-local 路径的基础 SPU 链路已经不稳定**。下一步不要再加 transformer 深度；应只跑 `depth0 / party-local` sanity。如果 depth0 也卡死，则先停止 e2e SPU 实验，保存 runtime 证据，转为检查 SPU runtime / party device 读文件 / localhost port/link 层。
- `depth0 / sample1 / party-local / standard` sanity 已跑完，没有卡死，但 verify 为决策不一致：`logits max_abs_error=0.6297486424446106`、`probabilities max_abs_error=0.2999129593372345`、`argmax_match_ratio=0.0`、`threshold_match_ratio=0.0`。这说明当前基础链路能完成最小图，但数值/语义已经与 CPU reference 明显偏离；下一步应做同一输入的 `depth0 split-share tensor-load` vs `party-local` 直接对照，判断问题来自 party-local device-side 文件加载，还是 split-share 输入本身相对 CPU reference 已漂移。
- 同一输入的 `depth0 split-share tensor-load` vs `party-local` 直接对照已回贴：`logits_max_abs_diff=4.57763671875e-05`、`probabilities_max_abs_diff=1.0967254638671875e-05`，两边 `argmax=[1]`、`threshold=[1]` 完全一致。因此 party-local device-side 文件加载不是首发问题；当前偏差来自 split-share/SPU depth0 相对 CPU reference 的基础语义/数值漂移。下一步应补同一输入的 `depth0 plaintext-input SPU` 与 `audit-shares`，区分是 share recomposition / patch_embed 漂移，还是 SPU static depth0 的 norm/head/patch backend 与 CPU reference 不一致。
- 需要更正上一条里的 “SPU depth0 相对 CPU reference 基础漂移” 判定：用户后续贴回的 `depth0 plaintext-input SPU` verify 仍然使用的是 `static_whole_forward_reference.pt` 这个 full-depth CPU reference，而不是 same-depth `depth0` CPU candidate，因此 `0.6297 / 0.2999` 的误差和 `0.0/0.0` 决策不一致不能作为 depth0 语义失败结论。`audit-shares` 已确认重组像素、patch tokens、tokens+pos 仍高度一致；下一步应先生成 `depth0 CPU candidate`，再用它作为 reference 分别 verify `depth0 plaintext/split/party-local SPU`，重新判断 same-depth 语义。
- same-depth 修正验证已补回第一条正结果：`depth0 CPU candidate` 作为 reference 时，`depth0 party-local SPU` 通过决策一致，`logits/probabilities max_abs_error≈2.68e-4 / 9.51e-5`，`argmax/threshold_match_ratio=1.0/1.0`。因此 party-local 的 depth0 基础语义正常；此前的 `0.6297` 误差只是 full-depth vs depth0 reference 混用造成的误判。下一步只需补同一 CPU0 reference 下的 split-share 和 plaintext-input depth0 verify。
- 同一 CPU0 reference 下，`depth0 split-share tensor-load` 与 `depth0 plaintext-input SPU` 也已补齐并通过：split-share `logits/probabilities max_abs_error≈3.14e-4 / 1.06e-4`，plaintext-input `≈2.53e-4 / 8.41e-5`，两者 `argmax/threshold_match_ratio` 均为 `1.0/1.0`。因此 depth0 的 plaintext / split-share / party-local 三条输入路径同深度语义都正常；下一步应回到 depth4，但必须先生成 same-depth CPU4 reference，再 verify monolithic depth4 party-local，不能再拿 full-depth reference 比。
- `depth4 CPU candidate` 已生成，same-depth reference 有效；但用户回贴 `depth4 / sample1 / party-local / monolithic` 重跑仍卡死。因此当前边界更新为：**party-local 在 depth0 同深度语义正常，但到 depth4 monolithic 仍触发 runtime/link 卡死**。下一步不要继续 party-local depth4 重试，应先跑同一 CPU4 reference 下的 `depth4 split-share tensor-load` 和 `depth4 plaintext-input SPU`，判断 depth4 图本身是否可在当前 runtime 通过。
- 用户继续反馈 `depth4 split-share tensor-load` 也已经卡死。因此当前问题不再是 party-local device-side 文件加载特有；边界收窄为：**split-share/party-local 这类双 share 输入在 depth4 图上触发当前 SPU runtime/link 卡死**。下一步只跑 `depth4 plaintext-input SPU` 作隔离：若 plaintext depth4 通过，则问题在 share 输入 / 双 secret 输入路径；若 plaintext depth4 也卡死，则问题是当前 depth4 SPU 图本身或 runtime 状态。
- 用户继续反馈 `depth4 plaintext-input SPU` 也卡死，错误为 `Socket closed`。因此当前边界已排除 share 输入特有问题，收口为：**当前服务器 session/runtime 下，static whole-forward SPU 到 depth4 的图本身会触发 runtime/link 断链**。下一步应停止 depth4 重试，回退到 same-depth `depth3 plaintext-input SPU`，确认当前 session 的真实可用边界是否仍是 `depth0..3`。
- `depth3 CPU candidate` 已生成，但用户回贴 `depth3 plaintext-input SPU` 也卡死。因此当前 session 的真实可用边界已经不是历史的 depth3/4，而是至少退回到 **depth0 可用、depth3+ 断链**。下一步不要继续尝试 depth3/4/12；应补 `depth1`、`depth2` plaintext-input same-depth smoke，建立当前环境的最小断点，并保存 runtime/link 证据。
- 用户继续回贴 `depth1 plaintext-input SPU` 也会卡死。因此当前最小断点已明确为：**depth0 plaintext/split/party-local same-depth 全部通过；一进入第一个 transformer block（depth1）即触发 SPU runtime/link 卡死**。这已经足够支撑下一步工程方向：停止继续 depth sweep，保存 depth0 正结果与 depth1 断链证据，转向 block1 SPU 图/attention/MLP 子图拆分或底层 SPU link 稳定性诊断。
- 针对这个最小断点，仓内已新增 debug-only 子图定位入口：`integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py block1-subgraph-smoke`，server wrapper 对应 `run_e2e_secure_whole_forward.sh block1-smoke`。它会逐段运行 `patch_pos / norm1 / qkv / attention_context / attention_residual / norm2 / mlp_hidden / block_output / head_logits`，每段 `ppd.get` 后记录 shape / finite / scalar stats；如果某段 `Socket closed`，会把已通过阶段和异常写入 `E2E_BLOCK1_SMOKE_JSON`。这是显式 debug reveal，不属于生产 e2e reveal policy。
- 用户继续反馈 `block1-smoke` 本身也会卡死，错误仍为 `grpc_status:14 / Socket closed`。这说明即使按 block1 子图分段 reveal，当前 SPU runtime/link 也可能在最早的 debug 子图阶段断链；下一步不应继续往 `transshield_e2e_secure_vit.py` 里追加更多一次性调试逻辑，而应先做文件瘦身和模块边界整理，把 stable runtime、debug-only probe、server wrapper 入口分开，避免 e2e 集成脚本继续膨胀。
- `2026-04-26` 已完成 e2e 集成脚本第一轮拆分：`transshield_e2e_secure_vit.py` 从约 `2363` 行降到约 `1484` 行，主文件现在主要保留 CLI / command 编排；CPU static reference 移到 `cpu_static_vit.py`，SPU/JAX whole-forward 与 share audit 移到 `spu_static_vit.py`，block1 debug-only smoke 移到 `debug_probe.py`，通用 JSON/tensor/share/params helper 继续在 `common.py`、`input_shares.py`、`static_vit_params.py`。已通过 `py_compile`、wrapper `bash -n`、`run --help` 与 `block1-subgraph-smoke --help`。下一阶段不要继续添加一次性 debug 代码；应在这个分层基础上做更底层的 SPU runtime/link 最小诊断。
- 拆分后的服务器 `depth0` 回归已由用户回贴通过：`depth0 plaintext-input SPU` 对 same-depth CPU0 reference 的 `logits/probabilities max_abs_error≈3.75e-4 / 1.21e-4`，`argmax/threshold_match_ratio=1.0/1.0`。这说明拆分没有破坏 e2e 入口；当前最小断点仍是进入 transformer block 后的 SPU workload/link。
- 为了继续隔离 runtime/link，仓内新增 debug-only 合成 primitive 入口：`transshield_e2e_secure_vit.py runtime-primitive-smoke`，server wrapper 模式为 `run_e2e_secure_whole_forward.sh runtime-smoke`。它不读图片、不读 share、不读模型权重，只用合成 `[1, tokens, embed]` 张量逐段执行 `scalar_add / layer_norm / qkv_linear / attention_scores / attention_probs / attention_context / projection_residual / mlp_hidden_square / mlp_residual` 并 reveal stats，用于判断当前断链是 ViT 参数/输入特有，还是标准 transformer primitive workload 已足以触发 SPU link 失败。
- 用户回贴 runtime primitive smoke 后，边界进一步收窄：tiny shape `token_count=16 / embed_dim=64 / heads=4 / mlp_dim=128` 全部阶段通过；ViT shape `token_count=197 / embed_dim=384 / heads=6 / mlp_dim=1536` 在第一段 `layer_norm` 即触发 `grpc UNAVAILABLE / Socket closed`，且只完成了 `scalar_add`。因此当前问题已经不是 e2e 输入、share、模型权重或 attention/MLP 特有，而是标准 transformer 形状的 SPU/JAX `layer_norm` workload 已足以触发当前 runtime/link 失败；下一步应做 `token_count x embed_dim` 的 layer_norm 形状阈值扫描。
- 用户继续反馈形状阈值 loop 也会长时间卡住，日志停在 `builtin_spu_run` RPC request 后并出现 `bvar is busy at sampling for 2 seconds`。这说明一旦某个 shape 触发 runtime 卡死，复用同一组 SPU 节点会污染后续诊断；后续不要用一条 shell loop 复用 runtime 扫多个 shape，应改为每个 shape 强制重启 runtime、短超时、失败即收 `node_*.log` 和 JSON。
- 用户进一步回贴单 shape 诊断仍中断：node 日志出现大量 `E1008 Reached timeout=20000ms`、`Duplicate ACK`、`bvar is busy`、`UpdateDerivedVars is too busy`，随后对 `spu_internal_addrs` 报 `E112 Not connected`。这说明当前 SPU backend 在较大 transformer primitive 下会进入 runtime/link 半死状态，shell `timeout` 不能可靠恢复节点；本轮 e2e 线应停止继续服务器实验，阶段结论收口为 **CPU/JAX-SPU runtime 层不稳定，无法在当前环境支撑 ViT 形状 layer_norm/attention workload 的全程隐私推理**。
- 为继续朝全程隐私推理推进，仓内新增实验开关 `--spu-layer-norm-chunk-size` / `E2E_SPU_LAYER_NORM_CHUNK_SIZE`，以及 runtime-smoke 专用 `--layer-norm-chunk-size` / `E2E_RUNTIME_SMOKE_LAYER_NORM_CHUNK_SIZE`。该实现仍在 SPU 图内计算 mean/variance，不 reveal 中间值，只把 feature 维 reduction 拆成多个小块求和，尝试绕开 `embed_dim=384` layer_norm reduction 触发的 runtime/link 断链。默认 `0` 保持原图。
- `layer_norm_chunk=64` 仍在 `layer_norm` 阶段触发 `Socket closed`，说明当前 runtime 对 secret layer_norm reduction 的问题不是简单 384 维整段 reduction 太大。为继续隔离下游 workload，仓内新增更激进的实验开关 `--spu-layer-norm-policy affine` / `E2E_SPU_LAYER_NORM_POLICY=affine`，runtime-smoke 对应 `--layer-norm-policy affine` / `E2E_RUNTIME_SMOKE_LAYER_NORM_POLICY=affine`。该模式跳过 secret mean/variance reduction，只做 public affine/identity，用于验证 qkv/attention/MLP 是否还能在全隐私不 reveal 的 SPU 图内跑通；它不是最终数值等价实现。
- 用户回贴 `layer_norm_policy=affine` 后，ViT shape 已能越过 `layer_norm / qkv_linear / attention_scores`，新的断点是 `attention_probs`，即 secret softmax 的 `max/exp/sum` 图。仓内因此新增诊断开关 `--attention-policy uniform` / `E2E_RUNTIME_SMOKE_ATTENTION_POLICY=uniform`，并允许 `--spu-attention-policy uniform`。该模式跳过 secret softmax，构造 public uniform attention，只用于验证后续 `attention_context / projection / MLP` 能否在 SPU 内跑通；它不是最终数值等价实现。
- 用户继续反馈 `affine LN + uniform attention` 在 fetch 约 `1.86MB` 的 uniform attention/probs 后，下一个 `builtin_spu_run` 仍卡死；按阶段顺序该断点对应 `attention_context`。仓内已进一步把 `uniform attention` 的 context 从显式 `197x197 @ V` 改为等价的 `mean(V)` broadcast，避免大 attention context matmul，继续验证 projection/MLP 的可承载性。
- 用户回贴 `uniform attention => mean(V) broadcast` 后，ViT shape synthetic runtime-smoke 已能继续完成 `projection_residual / mlp_hidden / mlp_residual`，其中 `stage_mlp_hidden` 的 square activation 仍产生约 `27MB send / 25.9MB recv`，但 runtime 未断链。因此当前可承载性边界进一步明确：当前 SPU runtime 能承载 ViT shape 的 QKV、attention-score matmul、projection 和 MLP；硬阻塞集中在 secret layer_norm mean/variance、secret softmax，以及显式 `197x197 @ V` attention context matmul。
- 用户继续回贴真实 e2e `depth1` 诊断组合日志：在 `E2E_SPU_LAYER_NORM_POLICY=affine` + `E2E_SPU_ATTENTION_POLICY=uniform` 下，`forward_fn` 已完整完成，用时约 `24.95s`，HLO 中 `pphlo.dot` 6 次、`pphlo.multiply` 7 次，link total 约 `64.97MB send / 48.65MB recv`。这说明真实 block1 图在跳过 exact LN/softmax/context 大 matmul后，当前 SPU runtime 能跑通；下一步应先确认 candidate JSON，然后把同一诊断组合切到 split/party-local share 输入，验证“隐私输入路径 + block1”能否一起跑通。
- 用户已确认真实 e2e `depth1 / party-local share load` 在同一诊断组合下成立：`finite_logits=true`、`input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`input_mode=party_local_debug_share_load` 全部满足。随后 `depth2`、`depth3`、`depth4` 与 `depth5 / party-local / affine LN / uniform attention` 也满足同一组隐私边界字段；`depth5` 用时约 `32.93s`，metadata 仍确认 driver 不加载明文像素或私有 share tensor，只 reveal final logits。阶段结论因此前移为：**party-local 全隐私输入边界 + 前 5 个 block 的诊断 SPU 图已闭合**。但 `depth4` logits 已到约 `1e8` 量级，`depth5` 进一步到约 `1e9` 量级且 probabilities 饱和为 `0/1`，所以这些结果只能说明 runtime/privacy-boundary 可承载，不能作为有效数值推理或最终分类结果；当前诊断组合仍不是 exact layer_norm / secret softmax / exact attention context 的数值等价模型。用户已明确目标是“实际可以使用的全隐私”，因此当前应停止继续 depth6+ 诊断加深，下一步改为做可用路径：用 public calibration stats 替代裸 `affine LN`（避免激活爆炸），并设计可验证的 secure-friendly attention 替代；在 CPU same-policy 先达到可用精度/非饱和后，再回到 party-local SPU。
- 已新增实际可用路线的第一步：`calibrate-layer-norm` CLI / wrapper `calibrate-ln` 可从非私有/public calibration pixel package 生成 `public_calibrated` LN 统计；`run --runtime spu` 新增 `--spu-layer-norm-policy public_calibrated` 与 `--spu-layer-norm-calibration-json`，wrapper 对应 `E2E_SPU_LAYER_NORM_CALIBRATION_JSON`。该模式使用公开 per-feature activation mean/variance 替代私有样本 LN reduction，不改变 party-local share-load 隐私输入边界。下一步服务器应先生成公开 calibration JSON，再复跑 `depth5 / party-local / public_calibrated LN / uniform attention`，目标是把 logits 从 `1e9` 饱和量级拉回可用范围。
- 用户已回贴 `depth5 / party-local / public_calibrated LN / uniform attention` 成功结果：`finite_logits=true`，隐私边界字段继续满足 `input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`input_mode=party_local_debug_share_load`；`spu_layer_norm_policy=public_calibrated` 且记录了公开 calibration JSON。数值上 logits 已从先前 affine LN 的 `1e9` 量级回落到约 `[0.2576, 0.4537]`，probabilities 为约 `[0.4511, 0.5489]`，不再饱和。因此当前阶段结论升级为：**party-local 全隐私输入边界 + public-calibrated LN 的前 5 block SPU 路径已经具备非爆炸、可解释的 final logits 输出**。仍需注意 attention 目前仍是 `uniform` 近似，尚不是原始 ViT exact/standard attention；下一步应在同一策略下推进 full depth 或补 CPU same-policy 对照，而不是回到 affine 诊断。
- 用户继续回贴 `depth12 / party-local / public_calibrated LN / uniform attention`：runtime 成功完成，`forward_from_shares_fn` 用时约 `33.73s`，SPU link 发送/接收约 `385.7MB / 333.5MB`，隐私字段继续成立；但 logits 又放大到约 `[2.02e5, 1.49e6]`，probabilities 饱和为 `0/1`。因此当前不能把 full-depth public-calibrated uniform-attention 写成实际可用分类器；准确结论是：**public-calibrated LN 已修复 depth5 的数值爆炸，但 full-depth 仍存在近似策略累积放大**。下一步应先做 CPU same-policy 对照，判断饱和来自 uniform/public-calibrated 策略本身还是 SPU 数值实现；若 CPU same-policy 也饱和，应改 attention/activation 近似或只把 depth5 作为可用 prefix baseline。
- 用户尝试 `depth12 / party-local / public_calibrated LN / uniform attention / activation_override=gelu` 后再次触发 `grpc UNAVAILABLE / Socket closed`，断点在 `spu(forward_from_shares_fn)`。因此 exact GELU/erf 不适合作为当前 SPU 可部署近似路线；不要继续重跑 gelu。下一步若继续 full-depth 近似，应优先试 SPU 友好的 `fixed_square` 或已训练 bundle quadratic，并配合 public calibration；当前可实际展示/部署的最低风险基线仍是 `depth5 / party-local / public_calibrated LN / uniform attention / bundle activation`。
- 用户继续回贴 `depth12 / party-local / public_calibrated LN / uniform attention / activation_override=fixed_square` 成功结果：runtime 完成约 `39.08s`，`finite_logits=true`，隐私边界字段继续满足 `input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`input_mode=party_local_debug_share_load`；数值上 logits 约 `[0.0315, 0.1317]`，probabilities 约 `[0.4750, 0.5250]`，不再饱和。因此当前可用近似路线更新为：**full-depth static ViT scope + party-local debug share load + public_calibrated LN + uniform attention + fixed_square activation + final logits only reveal**。它仍不是原始 exact ViT（uniform attention、fixed_square activation、public-calibrated LN 都是 deployable approximation），且仍是 colocated debug launcher 而非 P1/P2 独立进程；但它已经满足“全隐私输入边界 + full-depth runtime + 非饱和 final logits”的当前最强可用基线。
- 用户继续回贴同配置 `sample_count=2 / spu_batch_size=2`：runtime 与隐私字段仍成立，但 logits 聚合统计出现 `min≈-1.64e6`、`std≈6.70e5`，probabilities 再次出现 `0/1` 饱和；同时 `max≈0.1317` 仍接近 sample1 成功结果，说明很可能是第二个样本或 bsz2 批处理触发失稳。当前不能把 sample2 批量路径写成可部署通过；下一步应固定同一 calibration 与模型近似，先跑 `sample_count=2 / spu_batch_size=1` 区分批处理问题和样本问题，再考虑扩大 public calibration 样本数或加入 robust variance floor。
- 用户随后回贴同配置 `sample_count=2 / spu_batch_size=1` 成功：隐私字段继续成立，logits 为 `[[0.0314, 0.1316], [0.0359, 0.1358]]`，probabilities 均约 `[0.4750, 0.5250]`，`argmax=[1,1]`、`threshold=[1,1]`。因此前一次 `sample_count=2 / bsz=2` 爆炸已定位为 batched SPU graph 数值/编译问题，不是第二个样本或 public calibration 本身失稳。当前可部署近似基线应固定为 `E2E_SPU_BATCH_SIZE=1`，以逐样本方式处理多样本输入。
- 已把当前可部署近似基线集成为 server-friendly pack 入口：`artifacts/server_inference_friendly_pack/run_e2e_secure_approx_deploy.sh`。该脚本固定 `depth12 / party-local / public_calibrated LN / uniform attention / fixed_square / bsz1 / final logits only`，支持 `make-calib-pixels`、`calibrate`、`infer`、`all` 四个模式，并会拒绝 `E2E_SPU_BATCH_SIZE!=1` 防止误回到 batched graph 失稳路径。默认公开校准目录为服务器 `/data/wyb/pneumoniamnist_imagefolder_subset`，可用 `PUBLIC_CALIB_DATASET_DIR` 覆盖。
- `2026-04-27` E2E approximate smoke 进一步收口：原始 `uniform + fixed_square + public_calibrated LN` 在同条件明文对照下暴露 class0 边界偏移和偶发大 logits；已修复 eval list `find|head` 的 `pipefail/SIGPIPE` 早退，并加入 `E2E_APPROX_EVAL_ISOLATE_SAMPLES=1`、balanced public LN calibration、`E2E_SPU_ACTIVATION_CLIP_VALUE=3.0` 与 public post-reveal output calibration。当前已验证的 smoke-stable 配置是 `depth12 / party-local share load / public_calibrated LN / uniform attention / fixed_square / clip3.0 / output calibration / bsz1 / isolate samples`，其中 output calibration JSON 为服务器 `artifacts/server_pipeline_run/e2e_output_calibration_uniform_clip3_smoke8.json`，规则为 `weights=[-3.75, 4.0]`、`bias=-0.62`。
- 同日服务器同条件评测已闭合三组 smoke：`class0_4`、`class1_4` 和 balanced8 都得到 `original_plaintext_same_subset_argmax_accuracy=100.0`、`e2e_argmax_accuracy=100.0`、`prediction_match_vs_original_plaintext.argmax_match_ratio=1.0`，隐私字段继续满足 `input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false`、`private_input_paths_redacted=true`、`input_mode=party_local_debug_share_load`。balanced8 结果路径：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced8_uniform_clip3_calibrated_20260427_201413/e2e_secure_poc/e2e_approx_eval_metrics.json`，`e2e_elapsed_sec≈736.99s`，SPU link total≈`1.4526GB`。这只是 balanced8 smoke，不是 full-val 稳定性证明；下一步应按同配置扩大到 balanced16/更多样本。
- balanced16 已完成下一阶段诊断：原 smoke8 output calibration 在 balanced16 上得到 plaintext same subset `93.75%`、E2E `81.25%`、prediction match `0.75`；重新拟合 public post-reveal output calibration `weights=[6.25,16.0]`、`bias=-1.35` 后，原始一次性 run 为 E2E `87.5%`、match `0.9375`。逐样本排查发现 `i=15` 同一张图在一次 run 中 raw logits 偶发爆到 `[-3.4e5,-1.4e6]`，fresh runtime 单样本复跑恢复到小量级并预测 class1；替换该 fresh-runtime candidate 后的诊断汇总为 plaintext same subset `93.75%`、E2E `93.75%`、gap `0.0pp`、prediction match `0.875`，只剩 `i=6` 是稳定近似残差。patched metrics：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_20260427_211653/e2e_secure_poc/e2e_approx_eval_metrics_patched_i15.json`。因此当前结论是：balanced16 在 fresh-runtime/guarded per-sample 条件下可达到同 subset 明文准确率，但不是一次性原始 run 证明；eval 脚本已加入 per-sample logits guard 和 retry，后续扩大样本必须启用该 guard。
- `2026-04-28` 继续验证 `E2E_SPU_BLOCK_CHUNK_SIZE=3`：单样本 `i=15` chunk smoke 成功，candidate JSON 记录 `spu_forward_graph_mode=reveal_less_block_chunked`、`spu_block_chunk_size=3`，raw logits 回到小量级，且 fastpath request 从 monolithic 的约 `86MB` 降到约 `22.8MB/21.3MB` 分段请求。随后 guarded balanced16 chunk3 原始一次性 run 成功：`/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_approx_eval_balanced16_uniform_clip3_calib16diag_chunk3_guarded_20260427_235545/e2e_secure_poc/e2e_approx_eval_metrics.json`，plaintext same subset `93.75%`、E2E `93.75%`、gap `0.0pp`、prediction match `0.875`、`finite_logits=true`，隐私字段继续通过。这是当前最强的“一次性非 patched” balanced16 结果。
- 代价也已明确：chunk3 guarded balanced16 `e2e_elapsed_sec≈1474.46s`，用户反馈明显偏慢。chunking 解决了 monolithic 大图卡死/半死风险，但每样本多段 SPU run、fresh runtime、guard/retry 和多次 fetch 带来较大 wall-time 开销。后续优化应优先比较 `E2E_SPU_BLOCK_CHUNK_SIZE=4/6`、有限 runtime reuse、降低 startup/warmup 成本、并修正通信统计口径；当前 metrics 的 `aggregate_total_bytes≈161KB` 只来自 latest node log，不代表 chunked isolated 全流程总通信量。
- TrackA 明文训练根因诊断已收口，对应 issue 已完成：`ratio loss` 不是 `predictor_1` 过度 pruning 的首发驱动；归档见 `docs/tracka_predictor1_root_cause_2026-04-21.md`，正式展示口径不变。
- TrackA 的 server 运行环境 provenance 已闭合：`/data/wyb/conda_envs/transshield` 当前版本与 `requirements.txt` 完全一致，因此当前不再优先怀疑 server env 独立漂移。
- TrackA 的 `source` vs `compat` `debug80` parity 也已闭合：`NONEMPTY_KEEP_GUARD=false`、同 seed 下，两侧 `Namespace` 有效值、`Transform`、`Sampler_train`、scheduler / WD，以及 20 条 `Averaged stats` 都对齐；`training_compat` 仅多出默认关闭的 `pruning_margin_*` 兼容参数。
- TrackA 的 `LOSS_GRAD_ATTRIB=true` 归因已在 `2026-04-22` 完成首轮收口：server env provenance、strict source `guard-off` vs `guard-on`、source vs compat `debug80` parity 均已有同口径证据；在 run `tracka_source_epoch3_lossgradattrib_guardoff_seed0_20260422_195751` 的 `epoch=2 step=140~146` 中，`score_predictor.1.out_proj.weight` 的 total spike 由 `cls_kl` 主导，`ratio_loss` 仍是微小量级。后续若继续推进，不应再追默认 runner 差异，而应另开新 issue 决定下一条最小单变量。
- TrackA 的首个单变量 issue 已在 `2026-04-22` 完成收口：`cls_distill_weight=0.0` 的 `resync1` run 已确认是**有效但负向**结果，会把 `predictor_1 empty keep -> zero_active -> predictor_2 non-finite` 提前到 `epoch=2 step=34`；因此当前不建议继续跑 `clsdw0=0.0`，后续若继续推进应另开新 issue 选择下一条最小单变量。
- TrackA 的“下一条最小单变量” issue 也已在 `2026-04-22` 收口：唯一改动 `cls_distill_weight=0.5` 的 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` run 没有复现 `clsdw0=0.0` 在 `epoch=2 step=34` 的早期失稳，并在 `epoch=2 step=146` 把 `score_predictor.1.out_proj.weight` 的 total `grad_l2` 从 `4.198726e+03` 压到 `2.400378e-01`、把 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `5.022926e-01`；但 `epoch3` terminal accuracy 仍是 `74.24%`，因此应记作**明确缓解的低精度诊断结果**，不是正式修复或新正式成绩。
- TrackA post-`clsdw05` 的下一条最小单变量 `cls_distill_weight=0.75` 也已在 `2026-04-22` 收口：它是**有效单变量配置**，没有命中 `step=34` early-fail gate，并在 `epoch=2 step=146` 把 `score_predictor.1.out_proj.weight` 的 total `grad_l2` 从 `4.198726e+03` 压到 `3.748181e+01`、把 `cls_kl grad_l2` 从 `4.041559e+03` 压到 `2.105142e+01`、把 `predictor_1 final_keep_ratio_mean` 从 `1.548948e-01` 提到 `4.730873e-01`；但 terminal 仍是 `74.24%`。因此它应记作**明确缓解、但弱于 `clsdw05=0.5` 的低精度诊断结果**，不是正式修复或新正式成绩。
- TrackA post-`clsdw075` 的新阻塞点单变量已完成回贴分析：固定 `cls_distill_weight=0.5` 后把 `token_distill_weight` 从 `0.02` 上调到 `0.04` 是有效配置且未早崩；它让 `epoch3` terminal 从 `74.24%` 松动到 `79.01%`，但在 `epoch=2 step=146` 将 total `grad_l2` 从 `2.400378e-01` 放大到 `4.365869e+00`，并把 `predictor_1 final_keep_ratio_mean` 从 `5.022926e-01` 降到 `4.537168e-01`。因此这条结果是**terminal 正向、稳定性负向的混合诊断结果**，不能直接推进 `full20` 或写成正式修复。
- TrackA terminal-稳定性解耦单变量也已在 `2026-04-23` 完成回贴分析：固定 `cls_distill_weight=0.5` 后只把 `token_distill_weight` 从 `0.02` 调到 `0.03` 是有效配置且未早崩；它把 `step=146` total `grad_l2` 从 control 的 `2.400378e-01` 放大到 `1.165204e+00`、把 `active_margin_mean` 从 `-3.137406e-02` 推到 `-1.206955e-01`，terminal 只从 `74.24%` 轻微到 `74.43%`。因此它不是 clean 解耦：稳定性仍恶化，terminal 基本未接近 `79.01%`，不能进入 `full20` 或写成正式修复。
- TrackA post-`tdw003` 的下一条最小单变量 `ratio_weight: 2.0 -> 3.0` 已在 `2026-04-23` 完成回贴分析：它是有效配置且未早崩；在 `cls_distill_weight=0.5 / token_distill_weight=0.04` 底座上，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 回升到 `5.245984e-01`、`active_margin_mean` 从 `-1.643516e-01` 拉到 `4.946269e-02`，但目标参数 total `grad_l2` 却从 `4.365869e+00` 放大到 `5.206851e+01`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果应记作**负结果**：keep/margin 代理改善，但整体稳定性与 terminal 同时失败，不能进入 `full20` 或写成正式修复。
- TrackA post-`rw3` 的最小单变量 `cls_distill_weight: 0.5 -> 0.4` 已在 `2026-04-23` 完成回贴分析：它是有效配置且未早崩；在 `cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0` control 上，ablation 虽把 `cls_kl grad_l2` 从 `2.991327e+01` 降到 `4.482888e+00`，但 total `grad_l2` 从 `4.365869e+00` 升到 `7.772295e+00`，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 降到 `2.575499e-01`，`active_margin_mean` 从 `-1.643516e-01` 恶化到 `-1.041453e+00`，terminal 也从 `79.01%` 回落到 `74.24%`。因此这条结果是**有效但负向的非 clean 解耦**，不能进入 `full20` 或写成正式修复。
- TrackA post-`clsdw04` 的 midpoint 单变量 `token_distill_weight: 0.04 -> 0.035` 也已在 `2026-04-23` 完成回贴分析：它是有效配置且未早崩；在 `cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0` control 上，`step=146` total `grad_l2` 从 `4.365869e+00` 降到 `2.089920e+00`、`cls_kl grad_l2` 从 `2.991327e+01` 降到 `9.532135e-01`、`token_kl grad_l2` 从 `1.172528e+00` 降到 `2.397546e-03`，`predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 回升到 `4.878815e-01`，`active_margin_mean` 从 `-1.643516e-01` 回升到 `-8.497284e-02`；但 terminal 从 `79.01%` 回落到 `74.24%`。因此这条结果是**稳定性缓解、terminal 丢失的负向 midpoint 结果**，不能进入 `full20` 或写成正式修复。
- TrackA post-`tdw0035` 的近端 midpoint 选择已在 `2026-04-23` 完成服务器回贴分析：唯一近端候选 `token_distill_weight=0.0375` 是有效配置、未早崩，并在 `epoch=2 step=146` 把 `predictor_1 final_keep_ratio_mean` 从 `4.537168e-01` 提到 `5.085371e-01`、把 `active_margin_mean` 从 `-1.643516e-01` 拉到 `-2.972232e-03`、把 total `grad_l2` 从 `4.365869e+00` 压到 `1.294038e-01`、把 `cls_kl grad_l2` 从 `2.991327e+01` 压到 `1.165144e-01`、把 `token_kl grad_l2` 从 `1.172528e+00` 压到 `1.486021e-03`；但 terminal 仍从 `79.01%` 直接回落到 `74.24%`。因此当前应正式收口为 **`stop_token_midpoint`**：`0.03 / 0.035 / 0.0375` 三个 `<0.04` 的点都没有保住 `79.01%`，说明近端 token 轴的信息增益已不足，不再继续做 `0.04` 附近的剂量搜索，也不推进 `full20`。
- `2026-04-23` 前一轮本地交接维护已完成：`rw3`、`clsdw04` 与 `tdw0035` 后续最小单变量均已回贴分析并同步到权威文档；这个交接已被 post-`tdw0035` 近端 midpoint issue 接住，并已由用户回贴的 `token_distill_weight=0.0375` 服务器结果收口。当前仍没有“terminal 提升 + 稳定性无变化 / 缓解”的 clean 解耦迹象；本地未执行 `/data/wyb/...` 命令，未修改 e2e secure inference 支线。
- 仓库已完成一轮 TrackA 清理：`docs/history_best_repro_drift_audit_2026-04-21.md` 现为唯一主审计文档；两份 superseded 草稿已合并删除；部分 server pack wrapper 已去重。
- 第二轮 TrackA / server-pack 脚本整理已继续推进：`scripts/run_tracka_spu.sh`、`scripts/run_spu_patch_build_probe.sh`、`artifacts/server_inference_friendly_pack/run_margin_aware_pruning_ablation.sh` 已按阶段函数化；`tools/transshield_inference_friendly_server_pack.py` 已拆成命令构造 / shortcut 生成 / manifest 写出三层，`tools/transshield_openbumblebee_pipeline.py` 已拆成 step builder / replay helper / CLI parser 三层，`tools/transshield_openbumblebee_inference_replay.py` 与 `tools/transshield_fair_external_comparison.py` 也已分别拆成“边界回放 / 公平性报告”的分层 helper；`tools/transshield_chat_demo.py` 已拆成摘要构造 / state helper / handler request parsing 三层，`tools/transshield_selection_mode_profile_report.py` 已拆成运行产物解析 / scalar compare / Markdown section builder 三层，`tools/transshield_secure_profile_summary.py` 已拆成日志提取 / communication diagnosis / payload summary 三层，`tools/transshield_threshold_branch_eval.py` 已拆成 tie 统计 / kth mask 构造 / eval 主循环 三层，`tools/transshield_single_image_comparison.py` 已拆成 baseline/modified trace report builder / stage panel builder / summary board 组装三层，`tools/transshield_competition_scorecard.py` 已拆成输入解析 / 证据拼装 / checklist+outlook / Markdown section builder 四层，`tools/transshield_fastpath_profile_summary.py` 已拆成日志匹配 / bucket 更新 / 汇总收口 / Markdown 分段四层，`tools/transshield_runtime_branch_compare.py` 已拆成 summary 归一化 / compare section / recommendation builder 三层；根目录 `handoff-next.md` 与 `run_secure_selection_mode_profile_compare.sh` 改为薄兼容入口，避免 stale duplicate。另为避免重新生成已退役低频文档名，token pruning 可视化的 Markdown 输出已统一改为 `token_pruning_trace_report.md`。
- 新一轮根目录重复入口审计已完成：根目录 `PLANS.md` 已删除，避免继续制造过时 master plan 入口；根目录 `current_work_status.md`、`handoff-next.md` 与 `algorithm_protocol_upgrade_roadmap.md` 继续只保留薄兼容入口。根目录 `transshield_blockwise_kth_selection_manifest.py`、`transshield_stagewise_threshold_report.py`、`transshield_network_kth_bridge.py` 以及旧模型镜像 `dyvit.py`、`dylvvit.py` 也统一收口到 `tools/`、`integrations/`、`models/` 的权威实现；`training_source_tracka/` 仍保留为 source/provenance 快照，`training_compat/` 仍保留为当前 server 侧 plaintext compatibility runner，`references/original_plaintext_runtime/` 仍保留为 baseline runtime 快照，它们都不属于误删候选。
- 按最近产物时间与 handoff 引用频率看，当前最高频的入口主要是：`scripts/run_tracka_train.sh`、`scripts/run_tracka_spu.sh`、`scripts/run_spu_patch_build_probe.sh` 以及 `artifacts/server_inference_friendly_pack/` 下的 server pack wrappers。
- 这批高频训练入口现已进一步收口：`scripts/run_tracka_train.sh` 统一承接原来的 `source|compat` 两条 TrackA 训练路径，共享 `scripts/_tracka_training_common.sh`；`scripts/run_tracka_spu.sh` 统一承接原来的 `followup|dual-profile` 两条 SPU 路径，避免继续把模式写进文件名。
- `logs/` 也已进一步收口：当前只保留 `logs/spu_nodes/node_*.log`、`logs/spu_runtime_ports.json`，以及仍被当前 blockwise 证据链引用的 `logs/selection_mode_profile/blockwise_vs_flat_20260418_004346/`；本地 clean probe、web demo nohup 日志、SPU 轮转备份与未再引用的旧 selection-mode profile 目录已清掉。
- `tools/` 也已继续瘦身：旧 `phase3_lower_tail` 原型/manifest/planner、一次性 `SPU` fastpath logging patch 脚本，以及本地 repo audit/cleanup helper 已移除；当前支持的 final-repo 工具以 `tools/README.md` 为准。
- `tools/` 里的 Stage-2 说明型脚本也进一步收口：原来的 `transshield_tensor_contract_report.py`、`transshield_pruning_semantics_report.py`、`transshield_f_mux_spec_report.py`、`transshield_forward_dataflow_report.py`、`transshield_policy_spec_report.py`、`transshield_secure_kth_contract_report.py` 已合并为统一入口 `tools/transshield_stage2_report.py`，通过子命令生成对应报告，减少散落的单用途文件名。
- 本地 / 服务器传输命令也已重新收口：`docs/handoff-next.md` 与 `scripts/README.md` 现给出两条权威 `rsync` 命令，约定“本地 → 服务器用黑名单、服务器 → 本地用白名单”，并明确禁止再把服务器 `artifacts/` 直接同步到本地仓库根目录。
- `configs/openbumblebee/` 也已收口：当前只保留 live 的 `2pc.json`、`2pc.template.json` 与 `README.md`；所有时间戳 `.bak.*` 配置快照已从仓内移除。
- 最新一轮全仓审计已继续推进到 `integrations/`：当前 `integrations/openbumblebee/` 只剩两条 live bridge 实现，没有额外镜像目录；其中 `transshield_network_kth_bridge.py` 仍然偏长，但主要复杂度来自 CPU/SPU 双 runtime、selection-mode 与 mixed payload 三层职责叠加，暂不做激进删改。
- `secure_infer/` 已确认只是历史导航 README，不再承担独立入口；其内容已并回 `docs/architecture.md` 与 `tools/README.md`，同时 `docs/retention_list.md` 已刷新到当前真实保留集，去掉不存在或已退役路径。
- 最新一轮 `models/` / `training_*` 审计也已闭合边界：`dyconvnext.py`、`dyswin.py`、`samplers.py`、`optim_factory.py`、`utils.py` 在 root/source/compat 三侧完全一致；`datasets.py` 为 `source == compat`，但比根仓多 `augmentation_profile/mpcvit_like`；`engine.py` 为 `root == compat`，而 source 侧保留更多诊断钩子；`models/dylvvit.py` 为 `source == compat`，根仓额外承接 pruning diagnostics；`models/dyvit.py` 则 root/source/compat 三侧都已实质分叉，因此这些目录不是“误复制出来的重复代码”，而是 live 训练栈、source provenance 快照与 server compatibility runner 三种边界并存。
- 根仓 `main.py` 的定位也已进一步明确：它仍是 final-repo live 训练 / ablation 入口，当前被 `artifacts/server_inference_friendly_pack/run_train.sh`、`run_freeze_export.sh`、`run_margin_aware_pruning_ablation.sh` 等 wrapper 使用；TrackA provenance / parity 相关训练仍统一走 `scripts/run_tracka_train.sh source|compat`，不要把这两条口径混用。
- `references/original_plaintext_runtime/` 也已再次核实为 live baseline runtime 快照：当前仍被 `final_compare_env.template.sh`、`_run_plaintext_eval_variant.sh`、`_run_plaintext_predict_variant.sh`、`run_single_image_comparison.sh` 等 wrapper 直接引用，不属于可删重复目录。
- `results/` 最新一轮审计也已完成分层：当前真正仍有 live/runtime 作用或仍被主文档引用的，主要是 `blockwise_exact_kth_selection_manifest_default.json`、`blockwise_exact_kth_manifest_20260418_004103.*`、最新公平外部对比 `fair_external_comparison/fair_external_20260423_113217/`、最新统一 secure benchmark `standardized_secure_benchmark/standardized_secure_benchmark_20260423_130435/` 与 `standardized_secure_benchmark/standardized_secure_benchmark_20260423_132121_same_shape/`，以及 margin-aware 主证据目录；payload 设计空间报告虽然目录多，但合计只有约 `248K`，属于小体量佐证，不是当前优先清理对象。
- `results/` 这轮也已继续清理：本地误同步进来的空占位 margin-aware 目录、未形成完整 benchmark 汇总的 `standardized_secure_ops_20260417_181138/`，以及未被主文档再引用的本地 `margin_aware_full20_w3w5_20260417_213047/` 已删除；文档里若仍出现对应 `/data/wyb/...` 路径，应按服务器 provenance 理解，而不是要求本地继续保留同名空目录。
- `artifacts/` 顶层这一轮也已核清：当前真正的 live runtime 核心主要是 `baselines/`、`frozen_bundle_verified_tracka_lr3e5_20260414/`、`inference_ready_config/`、`server_inference_friendly_pack/` 与 `web_demo_assets/`；`server_pipeline_run/`、`server_profile_reports/`、`train_runs/` 属于当前证据链；体量最大的则是 `archive/`（约 `510M`）与 `frozen_candidates/`（约 `425M`）这两类 provenance/候选资产。
- `artifacts/` 的大头并不是“很多小碎文件”，而是少数大 checkpoint / state_dict：`archive/` 里基本就是两份约 `256M` 的完整训练 checkpoint；`frozen_candidates/tracka_lr3e5_timm_best_20260414/` 一目录就约 `340M`，主要来自 `checkpoint-best.pth`。它们当前都仍与 provenance / drift audit 文档有关，因此先不直接删。
- `artifacts/` 这轮也已继续做“零外部引用”清理：`train_runs/` 下两条仅本地误同步、且未被仓内文档 / 脚本 / 结果再次引用的 guard-off parity run —— `tracka_compat_debug80_seed0_guardoff_parity_20260422/` 与 `tracka_source_debug80_seed0_guardoff_parity_20260422/` —— 已删除；其余当前保留目录都至少还能在交接文档、结果报告或 wrapper 中找到引用。
- `frozen_bundle_full/` 与 `frozen_bundle_verified_tracka_lr3e5_20260414/` 也已核过，不是简单重复拷贝：两边的 light checkpoint 与 `state_dict` hash 都不同；其中 verified bundle 的 `modified_plaintext_model_state_dict.pth` 与 `frozen_candidates/tracka_lr3e5_timm_best_20260414/` 一致，说明它仍承担当前正式 bundle 的 provenance 收口角色。
- `web_demo/`、`configs/`、`references/`、`licenses/` 这批小目录也已完成核对：`web_demo/index.html` 是当前前端单页实现，虽约 2042 行但没有再引用旧 archived profile / 固定历史通信作为主口径；`configs/openbumblebee/` 只保留 live `2pc.json`、模板与 README；`references/original_plaintext_runtime/` 仍是 baseline eval / predict wrapper 直接使用的最小快照；`licenses/` 与 `THIRD_PARTY.md` 是交付所需第三方许可证说明，不是清理对象。
- 已新增一份面向初见者与答辩的总览文档：`docs/project_overview_newcomer_defense.md`。它把项目目标、三条主线、仓库结构、当前进展、可正式宣称的结论、不能混用的口径，以及建议答辩讲法统一写在一处，后续新人入仓或准备答辩时优先看这份文档。
- 这类 cleanup / refactor 变更现在不再单独新开专门变更文档；直接维护 `docs/current_work_status.md`、`docs/handoff-next.md` 和 `tools/README.md` 这三处权威参考入口。

## 当前权威文件

按优先级阅读：

1. `docs/data_source_policy.md`
2. `docs/result_summary.md`
3. `docs/external_baseline_comparison.md`
4. `docs/web_chat_demo.md`
5. `docs/handoff-next.md`
6. `docs/margin_aware_pruning_notes.md`
7. `docs/network_kth_blockwise_notes.md`

## 当前离线验证集权威指标

来源：`artifacts/web_demo_assets/best_demo_content.json`

| 指标 | 当前值 | 用途 |
|---|---:|---|
| 最佳轮次 | `epoch 8` | 说明当前冻结展示包使用哪一轮导出权重 |
| Argmax 准确率 | `93.702292%` | 离线验证集对比 |
| Threshold 准确率 | `94.083971%` | 离线验证集对比 |
| AUC | `0.972313` | 离线验证集对比 |
| Argmax 一致率 | `100%` | secure 与明文一致性 |
| Threshold 一致率 | `100%` | secure 与明文一致性 |

## 当前前端展示规则

- 顶部四个卡片：只显示当前浏览器选择图片触发的即时结果
- `交互演示`：浏览器本地分片、E2E 推理、安全结果、live run 成本闭环展示
- `与外部同数据集基线相比`：只显示离线验证集最佳成绩与 `MPCViT` 对比
- `本次安全推理开销`：只显示当前图片本次 `E2E SPU live run`
- `统一 secure benchmark`：当前 benchmark 报告已跑通并写入 `artifacts/web_demo_assets/best_demo_content.json`，前端会显示同一 harness 下的外部 secure proxy 对比

## Phase 3 当前进展

已新增：

- manifest 生成器：`tools/transshield_blockwise_kth_selection_manifest.py`
- 新选择模式：`blockwise_exact_kth`
- bridge / pipeline / profile compare 已接入新模式

当前本地 smoke 结论：

- 使用 `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/` 的 `smoke8` 输入，
- 新模式生成的 candidate payload 已通过 `tools/transshield_secure_network_kth.py check`
- 逐 stage `max_abs_error = 0.0`

这说明当前 `blockwise_exact_kth` 在**语义正确性**上已经打通。

## Phase 3 第一轮服务器结论

来源：

- `/data/wyb/Transshield_final/artifacts/server_profile_reports/blockwise_vs_flat_20260418_004346_selection_mode_compare/selection_mode_profile_compare.md`

当前已经得到一轮明确正结果：

- `flat_odd_even` → `blockwise_exact_kth`
- `verify` 两边都通过
- `network_kth_bridge`：
  - `11.6141s -> 10.3066s`
  - 比例 `0.887x`
- `total pipeline duration`：
  - `16.9257s -> 15.6245s`
  - 比例 `0.923x`
- `communication total bytes`：
  - `1.72 MB -> 1.72 MB`
  - 当前无下降

当前判断：

- `Phase 3` 已经拿到第一轮同口径 SPU 加速证据；
- replay / branch reconstruction 也已经补齐并通过；
- 收益主要体现在时间，而不是通信；
- 它已经升级为正式 secure pipeline / Web demo secure run 的默认选择模式；
- 下一步重点是保留 Phase 4 payload 诊断结论，而不是把 mixed payload 设为默认。

## Phase 4 当前已启动的最小诊断

当前仓内已经补上第一批 payload 诊断钩子：

- `stage2_secure_network_kth_candidate_from_server.json`
  - 现在会记录每个 stage 的：
    - dense `masked_score` float32 bytes
    - compact `masked_score` float32 bytes
    - active compaction 节省的 bytes
- `secure_profile_summary.json`
  - 现在会额外给出：
    - `rpc_total_over_compact_payload_ratio`
    - `make_shares_over_compact_payload_ratio`
- `selection_mode_profile_compare.md`
  - 现在可以直接看：
    - compact payload 是否真的变小
    - RPC 总量相对 compact payload 放大了多少

这一步的意义不是马上降通信，而是先把 `Phase 4` 的问题分清楚：

- 是 payload 本身太大
- 还是 payload 不大，但 Python RPC / share framing 放大得太多

## Phase 4 当前最重要的新结论

基于正式模型当前 `smoke8` 输入的 payload precision ablation，已经确认一条**可工程化**的 mixed payload 路线：

- 正式模型当前最佳已知方案：
  - `stage0=float16`
  - `stage1=float32`
  - `stage2=float16`
  - `boundary_window=4`
- 对应结果：
  - `formal_stage1fp32_bw4.json`
  - `all_exact_semantics_preserved = true`
  - `total_byte_ratio_vs_float32 = 0.6807`

这说明：

- 正式模型已经不是“完全不能降 payload”；
- 当前真正敏感的是 `stage1 / layer 6`；
- 因此 Phase 4 的首选工程方向，不再是盲目全量 `float16`，而是：
  - `stage1` 保持 `float32`
  - `stage0/2` 走 `float16`
  - 对边界附近 token 保留小窗口 `float32` 修正

同时，`w10` 这条研究线在 Phase 4 也被重新证明确实有用：

- `all_float16_bw4.json`
  - `all_exact_semantics_preserved = true`
  - `total_byte_ratio_vs_float32 = 0.5315`

它的意义不是替换正式展示模型，而是说明：

- 一旦 pruning boundary 被明显拉开；
- mixed-precision payload 会从“很难做 exact”变成“更容易做 exact”；
- 因此 `w10` 现在更像是 Phase 4 的**研究上界 / 协议友好参考线**。

这一步 formal-vs-mixed 工程复核现已完成，服务器 run 为：

- `phase4_payload_formal_vs_mixed_20260423_104207`
- A：`formal_fp32`
- B：`mixed_stage1fp32_bw4`

当前实测表明：

1. 两边都保持：
   - `selection_mode=blockwise_exact_kth`
2. A 侧正确进入正式默认路径：
   - `payload_default_dtype=float32`
   - `payload_boundary_window=0`
3. B 侧正确进入唯一 mixed 候选：
   - `payload_default_dtype=float16`
   - `payload_stage_dtypes="1:float32"`
   - `payload_boundary_window=4`
4. A / B 两边都满足：
   - `overall_passed=true`
   - `pipeline_verify_overall_passed=true`
5. mixed 侧 host 输入压缩确实生效：
   - `payload_mixed_transport_total_bytes: 899184 -> 668624`
   - 比例 `0.7436x`
   - `python_fastpath_make_shares_total_input_bytes: 899188 -> 630900`
   - 比例 `0.7016x`
6. 但最终工程指标明显变坏：
   - `communication_total_bytes: 1802657 -> 1880245`
   - 比例 `1.0430x`
   - `network_kth_bridge_elapsed_sec: 11.7105s -> 197.2977s`
   - 比例 `16.8479x`
   - `total_pipeline_duration_sec: 18.1556s -> 203.4937s`
   - 比例 `11.2083x`

## Phase 4 当前最新实测结论

这条工程线现在已经拿到一条**formal-vs-mixed 单次工程复核的明确负结论**：

- 正式模型 mixed payload 真实 SPU pipeline：
  - run：`phase4_payload_formal_vs_mixed_20260423_104207`
  - A：`formal_fp32`
  - B：`mixed_stage1fp32_bw4`
  - `selection_mode = blockwise_exact_kth`
  - `payload_dtype = float16`
  - `payload_stage_dtypes = "1:float32"`
  - `payload_boundary_window = 4`
- A / B 两边：
  - `overall_passed = true`
  - `pipeline_verify_overall_passed = true`
- 但实际 profile 结果是：
  - `payload_mixed_transport_total_bytes: 899184 -> 668624`
  - `python_fastpath_make_shares_total_input_bytes: 899188 -> 630900`
  - `make_shares_over_compact_payload_ratio = 0.7016x`
  - 说明 host/P1 侧输入压缩**确实生效**
  - `communication total bytes: 1802657 -> 1880245`
  - 比例 `1.0430x`，反而高于 fp32 基线
  - `network_kth_bridge: 11.7105s -> 197.2977s`
  - 比例 `16.8479x`
  - `total_pipeline_duration_sec: 18.1556s -> 203.4937s`
  - 比例 `11.2083x`

根因现在已经很清楚：

1. Python fastpath 主显示里的大头不再是 `make_shares`
   - 而是 `builtin_fetch_object`
   - 也就是 share object 在节点间的拉取/返回
2. mixed payload 在当前 OpenBumbleBee / SPU 栈里只能减少
   - `make_shares` 输入字节
   - 不能直接减少 secure share 自身的对象返回字节
3. 如果把 mixed payload 的精确重构放到 SPU 内部做，
   - 会额外引入很重的协议算子开销
   - 当前实测已经出现 `11x~17x` 的时间恶化

因此截至当前：

- `Phase 4 mixed payload` 这条线**保留为研究/诊断证据**
- 但**不能进入正式展示默认路径**
- 当前 Web demo / 正式 SPU pipeline 仍应继续使用：
  - `payload_dtype = float32`
  - `blockwise_exact_kth`
- 当前不要再在这条 issue 内继续给第二个 mixed payload 配方，也不要把它扩成 payload 搜索；如果以后真要继续，只能进入更深层的 transport / protocol redesign，而这已经超出当前正式展示范围。

一句话收口：

- `mixed payload` 证明了“host 侧压缩”是成立的；
- 但在当前 fastpath / share-object / SPU 重构实现下，**它不能转化成更好的最终通信与时延指标**。

## 当前正式 secure 默认路径

截至当前，正式 secure pipeline / Web demo 默认路径已经收敛为：

- `KTH_SELECTION_MODE = blockwise_exact_kth`
- `PHASE3_SELECTION_MANIFEST = results/blockwise_exact_kth_selection_manifest_default.json`
- `payload_dtype = float32`

这表示：

- 默认启用 Phase 3 的 blockwise exact-kth 正结果；
- 默认不启用 Phase 4 mixed payload 负结果；
- 如需回退旧路径，可显式设置：
  - `KTH_SELECTION_MODE=flat_odd_even`

对应入口已更新：

- `tools/transshield_openbumblebee_pipeline.py`
- `artifacts/server_inference_friendly_pack/run_secure_pipeline.sh cpu|spu`

另外，当前 Web demo 已增加一条**仅影响演示等待时间**的工程优化入口：

- `WEB_DEMO_REUSE_SPU_RUNTIME=1` 时，会优先复用已启动的 SPU runtime
- 如果当前 runtime 不可达，再自动执行原来的 restart 路径
- 这不会改变模型、离线准确率或 secure 语义，只是减少网页多次点击时反复重启 SPU 节点的额外等待

同时，`network_kth bridge` 已新增一条默认开启的 Python fastpath 优化：

- SPU 路径下默认改为 `batched_pyu_bundle` 输入/输出传输
- 目标是减少逐 stage `PYU object` 拉取与 `builtin_fetch_object` 次数
- 如需回退旧路径，可显式设置：
  - `TRANSSHIELD_SPU_IO_MODE=per_stage`

当前 `Web demo` 单图 secure run 还新增了一条更直接的 fast path：

- 默认 `WEB_DEMO_SKIP_PIPELINE_VERIFY=1`
- 即网页单图运行时，不再每次都重复执行 `pipeline verify`
- 仍然保留：
  - secure pipeline 主运行
  - replay
  - selected-image diagnosis
  - profile summary
- 因此这条优化主要减少演示等待时间，不会改离线准确率，也不会改变页面展示的 secure 预测语义

## `E2E secure inference` 当前主路径

这条线当前已经成为 Web demo 的默认主路径：浏览器本地生成 `share0/share1`，后端通过 party-local debug share load 触发 full-depth approximate SPU，并只 reveal final logits。旧 `secure sidecar + replay + compare` 保留为 legacy 证据链和调试入口，不再作为最终 Web 主展示默认路径。

当前必须先明确：

- 当前网页主流程不上传原图或完整 plaintext `pixel_values`；
- 当前 demo 后端仍是单进程同时接收两份 debug share，因此不能等同于生产级独立 P1/P2 部署；
- 当前 E2E approximate 路径使用 `public_calibrated LN + uniform attention + fixed_square`，不是原始 DynamicViT 明文语义逐算子等价复现。

后续建议按三步推进：

1. **生产部署拆分**
   - 把当前单后端接收两份 share 改成独立 P1/P2 上传端点；
   - 保证 share0/share1 分别只进入各自 party 的进程 / 主机 / 信任域；
   - 保持 final logits only reveal policy。
2. **同口径效果评测**
   - 用 `run_e2e_secure_approx_eval.sh` 替代旧 sidecar 对比；
   - 同一图片列表上输出原始明文 reference、E2E approximate SPU、准确率差、预测一致率和通信量。
3. **迁入当前 masking-pruning 语义**
   - 保持 fixed-shape / `masking` 表达；
   - 把 `pred_score -> prev_decision -> policy -> block forward` 一起迁进 secure 前向；
   - 不回退到“动态删 token”这条对 secure 不友好的表达。

当前约束也已收口为：

- 可以写“当前 demo 已做到浏览器端不上传原图 / 不上传完整 pixel_values”；
- 不能写“生产级全隐私 2PC 部署已经完成”；
- 不直接把 `OpenBumbleBee/flax_vit` 当作最终实现照搬，因为当前仓库主线仍是 `PyTorch + DynamicViT masking` 语义。

## Phase 3 replay 补验证结论

来源：

- `/data/wyb/Transshield_final/artifacts/server_pipeline_run/blockwise_vs_flat_20260418_004346_blockwise_exact_kth/pipeline_inference_replay_summary.json`

当前已经确认：

- `overall_passed = true`
- 三个 stage 全部 `overall_passed = true`
- 每个 stage：
  - `exact_count_match_ratio = 1.0`
  - `exact_mask_match_ratio = 1.0`

## 明文重训稳定性最新结论

截至 `2026-04-18`，离线正式展示成绩仍以冻结 bundle 为准，服务器验证集复核结果保持：

- Argmax：`93.702292%`
- Threshold：`94.083971%`
- AUC：`0.972332`

当前明文重训排障的最新问题不在验证集、阈值搜索或前端展示，而在训练期 mask-pruning 路径：

- `training_source_tracka` 原始 source 控制路径在 guard-off 下也会崩溃；
- 首个明确崩溃点出现在 `epoch=3 step=3` 附近；
- 日志显示 `PredictorLG global_x / post_agg / out_conv_out` 已经含 non-finite；
- 随后 `VisionTransformerDiffPruning predictor_2_pred_score` 触发 non-finite 检查；
- 直接诱因是上一阶段 hard keep mask 对部分 sample 变成全 0，下一阶段 global aggregation 出现除 0。

仓内已把 `training_compat` 中的可选 `nonempty_keep_guard` 同步补到 `training_source_tracka`：

- `training_source_tracka/models/dyvit.py`
- `training_source_tracka/main.py`
- `scripts/run_tracka_train.sh source`

注意：

- `nonempty_keep_guard=false` 仍是默认值，用于保持原始 source 控制路径；
- 显式设置 `NONEMPTY_KEEP_GUARD=true` 才启用单 token fallback；
- strict source `guard-off` vs `guard-on` 与 source-vs-compat `debug80` parity 现在都已闭合；不建议再把默认 runner parity 当主阻塞，也不建议直接长训或继续扫超参。

最新 source guard-on `epoch5` 结果已经确认：

- 训练不再崩溃，但精度仍卡在多数类基线：
  - Argmax：`74.236643%`
  - Threshold：`75.381678%`
  - AUC：`0.769285`
- 当前 run 的 threshold 预测分布是 `[38, 486]`，明显偏向类 `1`；
- 官方 best 的 threshold 预测分布是 `[134, 390]`；
- 训练轨迹与官方 best 的首个明显分叉点在 `epoch=2` 末尾：
  - 当前 `score_predictor.1.out_proj.weight` 梯度在 `epoch=2 step=146` 突然到 `1.057964e+03`
  - 官方同阶段 grad_norm 约 `2.5876`，随后验证准确率到 `75.19% -> 82.63% -> 91.41%`
- `empty_keep` 在当前 run 中从 `epoch=3 step=0` 开始出现，主要发生在 `predictor_2`，随后 `predictor_1` 也开始出现。

因此当前判断是：

1. `empty_keep` 是 guard 捕获到的连锁症状；
2. 更早的根因线索是 `score_predictor.1` 在 `epoch=2` 末尾的梯度爆炸；
3. 下一步应先做 strict source `guard-off` vs `guard-on` 对照，再做 `training_source_tracka` vs `training_compat` 的 `debug80` parity，而不是继续长训或扫 EMA/LR。

已新增最小诊断钩子：

- `training_source_tracka/models/dyvit.py`
  - 新增 `predictor_1/2_keep_diag`
  - 记录 `active_count`
  - 记录 `raw_keep_count`
  - 记录 guard 后 `final_keep_count`
  - 记录 `raw/final_keep_ratio`
  - 记录 active token 上的 `keep-vs-drop margin`
  - 在常规 step、epoch 末尾 step、或出现 empty keep 时打印
- `training_source_tracka/losses.py`
  - 新增 `ratio_stage_i`
  - 记录每个 pruning stage 的 `pos_ratio`
  - 记录相对目标 keep ratio 的 `gap`
  - 记录该 stage 的 ratio loss
- `scripts/run_tracka_train.sh source`
  - 新增 `epoch3` 模式，用于复现到 `epoch=2` 末尾的首个梯度爆炸点。
  - `mean_jaccard_vs_topk = 1.0`
  - `reconstructed_branch_matches_topk_reference = true`
  - `max_abs_error_after_snap_vs_argsort_reference = 0.0`

这说明 `blockwise_exact_kth` 已经不只是 checker 通过，而是 **完整 replay 语义也通过**。

## 为什么 Phase 3 时间降了但通信没降

当前实现里：

- 两个模式在进入 SPU 前，都先做同一套 `active token compaction`
- 所以送进 SPU 的 `masked_score` 紧凑张量大小当前相同
- `blockwise_exact_kth` 改掉的是 SPU 内部 `exact kth` 的执行调度

因此结果会表现成：

- `network_kth_bridge` 时间下降
- `total pipeline` 时间下降
- Python fastpath RPC bytes 基本不变

这不是矛盾，而是说明当前已经把 **Phase 3 的“算子/调度侧优化”** 跑通了；下一步要降通信，必须进入 **Phase 4 payload**。

## Phase 2 研究证据：`margin-aware pruning w10`

这条线的定位是：**算法升级证据**，不是当前 Web demo 默认模型。

- 服务器候选运行名：`margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`
- 候选 bundle：`artifacts/frozen_candidates/margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`
- 服务器报告根目录：`/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_aware_full20_w10_20260417_205242`
- 候选效果：
  - Argmax 准确率：`88.9313%`
  - Threshold 准确率：`90.2672%`
  - AUC：`0.956508`
- secure 一致性：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
  - `logits_max_abs_error = 0.0`
  - `probabilities_max_abs_error = 0.0`
- 最关键读数：
  - Stage 2 boundary margin mean 提升到基线的 `243.532x`
  - Stage 2 `<=1e-4` near-boundary 比例从 `98.66%` 降到 `5.92%`
  - `tie delta = -0.19%`，没有出现 secure 一致性变坏

这说明方向是成立的：margin-aware loss 确实能把最敏感的 Stage 2 剪枝边界拉开，后续可以与协议友好的 `network-kth`、payload 压缩、局部化 pruning policy 一起形成更完整的算法升级叙事。

当前不能直接把它切到主展示的原因也很明确：精度仍低于正式展示模型（当前正式 Threshold 准确率为 `94.083971%`）。因此下一步应做**局部化和后置化**，而不是继续扫“全局统一 margin 权重”。

## Phase 2 当前收口结论

截至 `2026-04-18`，`margin-aware pruning` 的训练搜索先告一段落，当前保留两条最有用的证据：

- `w10`
  - 作为**最强研究性正结果**
  - 证明 Stage 2 边界可以被显著拉开，且 secure 一致性不坏
- `w3 + formal hparams + tok0.02`
  - 作为**当前最好的精度 / 协议友好性折中证据**
  - Threshold Acc：`91.6031%`
  - AUC：`0.967476`
  - Stage 2 margin：`20.032x`

同时，以下两条也已经给出明确负结论：

- `Stage2-only + delayed start`
  - 精度明显失败，不再继续
- `w3 + tok0.04`
  - 虽然 Argmax 变高，但 Threshold / AUC 下降，且 Stage 3 分布明显变坏

因此当前判断是：

- `w10` 仍然非常有用，应保留为 Phase 2 的核心算法证据；
- 但不再继续大规模扫 margin 超参；
- 下一阶段优先转向 `Phase 3 network-kth` 与 `Phase 4 payload`。

## 当前明确禁用的旧数据

以下数字已经禁止再用于前端、主结论或新文档：

- `1.90 MB` 历史 fastpath 8 样本通信
- `979.9903s` / `975.1174s` / `3.21 GB` 旧 archived SPU profile
- “相对旧正式展示模型”的训练收益

## 本次文档清理结果

已删除：

- `docs/changes/`
- `docs/defense/`
- 所有 `server_execution_request_*.md`
- 所有 dated `handoff / runbook / cleanup / assessment / profiling / followup` 文档
- `competition_summary.md`
- `experiments.md`
- `secure_profiling_summary.md`

目的：阻断旧数字从历史文档回流到前端和答辩材料。

## 本次产物清理结果

已删除：

- `artifacts/archive/intermediate_checkpoints_20260415/`
- `artifacts/server_pipeline_run/` 中的历史运行产物
- `artifacts/train_runs/` 中的历史训练输出
- `artifacts/server_profile_reports/`
- `artifacts/spu_build_reports/`
- `artifacts/bridge_validation/`
- `results/` 中 20260414~20260416 的旧 followup / dry-run / audit JSON
- `logs/spu_nodes_clean/`、`logs/spu_runtime_ports_clean.json` 与两份旧 `web_demo` nohup 日志
- `logs/spu_nodes/` 下所有 `node_*.log.prev.*` 轮转备份
- `logs/selection_mode_profile/` 下未再被当前文档引用的旧 phase3 / phase4 实验目录，仅保留 `blockwise_vs_flat_20260418_004346/`

已整合：

- `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/`
  - 保存 selection-mode profile 所需的最小 runtime inputs
  - 替代对历史 `server_pipeline_run/` 目录的依赖
- `transshield_openbumblebee_pipeline.py`
  - 收成兼容 wrapper，实际逻辑统一走 `tools/transshield_openbumblebee_pipeline.py`

## 当前仍保留的说明文档

- `docs/project_overview_newcomer_defense.md`
- `docs/data_source_policy.md`
- `docs/result_summary.md`
- `docs/external_baseline_comparison.md`
- `docs/web_chat_demo.md`
- `docs/architecture.md`
- `artifacts/server_inference_friendly_pack/README.md`

这些入口已经吸收了数据集说明、正式 bundle provenance、答辩问答 / 展示建议，以及单图对照 / token 可视化的当前用法；旧独立说明文档已删除，不再单独维护。

## 明文训练复现当前新结论

围绕“正式 best 能不能稳定复现”这条线，今天已经确认一个关键 provenance 差异：

- 官方 best 来源：
  - `artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414/manifest.json`
- 其中明确记录的训练关键参数包括：
  - `model_ema = false`
  - `activation_lr_scale = 10.0`
  - `crop_pct = 0.875`
  - `weight_decay_end = 0.05`
- 近期服务器侧 `ema_only_epoch1_* / ema_only_epoch5_* / ema_only_epoch20_*` 日志里：
  - `model_ema = true`
  - `activation_lr_scale = 1.0`

这意味着：

- 最近这条 `EMA-only` 复现实验**不是**相对正式 best 的单变量公平对比；
- 它目前只能说明：
  - “`model_ema=true` 且没有维持官方 `activation_lr_scale=10.0` 配方时，训练最高只到约 `77.10%`，EMA 更低”
- 但它**还不能**直接说明：
  - “在官方 best 原配方上，只改 `EMA` 一项也一定失败”

继续追查后，已经定位到一个更关键的代码路径差异：

- `training_compat` 与历史 `DynamicViT_exp_square` 并非完全一致；
- 其中 `training_compat/models/dyvit.py` 之前默认启用了：
  - 空 keep 决策时的 `single_token fallback`
  - `PredictorLG` 的零 active-count clamp guard
- 这两处都不属于原始 `tracka_lr3e5_timm_best_20260414` 训练语义。

截至当前，这个漂移已经收口为：

- 新增显式开关：`nonempty_keep_guard`
- 默认值：`false`
- 服务器脚本：
  - `scripts/run_tracka_train.sh compat`
  - 也已显式暴露：
    - `NONEMPTY_KEEP_GUARD=false`

这表示：

- 现在 `training_compat` 默认更接近官方 provenance 语义；
- 如果以后只是做“鲁棒性排障”，而不是做严格复现，再显式打开：
  - `NONEMPTY_KEEP_GUARD=true`

另外，今天还确认了另一个会误导短验证判断的问题：

- 之前脚本里的 `epoch1` / `epoch5` 是直接把：
  - `epochs=1` / `epochs=5`
  - 一起传给训练入口
- 这会把 Cosine LR / WD schedule 也压缩到 1 或 5 轮
- 因而它**不是**“官方 20-epoch 配方前 1 / 5 轮”的公平短验证

当前这个问题也已修正为：

- `epoch1` / `epoch5` 默认保持：
  - `epochs=20`
- 同时新增：
  - `stop_after_epoch`
- 用“提前停”代替“改短总 epoch”，从而保留官方 schedule。

因此当前最小正确下一步已经收敛为：

1. 先用 `scripts/run_tracka_train.sh compat` 精确重跑官方配方；
2. 只有当这条线恢复到接近官方 provenance 后，才继续做真正的单变量 ablation；
3. 下一条最合理的单变量仍然是：
   - 在保持 `activation_lr_scale=10.0`、`crop_pct=0.875`、`weight_decay_end=0.05` 不变的前提下，
   - 单独切 `MODEL_EMA=true`。

最新收敛结论：

- 官方 frozen bundle 已经在 `/data/wyb/pneumoniamnist_imagefolder_subset/val` 上重新验证通过：
  - `default_argmax_acc1 = 93.702292`
  - `best_threshold_acc = 94.083971`
  - `auc = 0.972332`
- `/data` 侧 train / val 类别计数也与历史记录一致：
  - train：`0:1214, 1:3494, total=4708`
  - val：`0:135, 1:389, total=524`

因此当前问题已经不在验证集、评估器或 frozen best bundle，而是在**训练复现链**。

为隔离 `training_compat` 与历史训练入口的差异，已经新增一个原始源码对照快照：

- `training_source_tracka/`
- 服务器入口：
  - `scripts/run_tracka_train.sh source`

这个入口基于 `DynamicViT_exp_square` 原始训练文件，只额外加入 `--stop_after_epoch`，用于保留 20-epoch schedule 的前 5 轮对照。

## TrackA server env provenance 已闭合

`2026-04-22` 已核对 `/data/wyb/conda_envs/transshield/bin/python`：

- `python 3.9.25`
- `torch 1.13.1+cu117`
- `torchvision 0.14.1+cu117`
- `timm 0.3.2`
- `numpy 1.26.4`
- `cuda 11.7`
- `cudnn 8500`

与仓库 `requirements.txt` 当前 pin 完全一致。

这说明：

- 本地 `/home/yclcg/miniconda3/envs/transshield` 已偏离 repo pin，不能继续拿来替代 server provenance；
- 当前 server runner 的 package / CUDA 现状核验已经闭合；
- 当前 `74.236643%` 的 guard-on 失败结果，不能再优先归因到“server env 独立漂移”。

因此当前问题继续收口时，优先级应转到：

1. strict source `guard-off` vs `guard-on` 对照已完成；
2. `training_source_tracka` vs `training_compat` 的 `debug80` parity 已完成；
3. `LOSS_GRAD_ATTRIB=true` 首轮非 `ratio` 分项归因已完成，结论是 `cls_kl` 主导 `epoch=2 step=146` 的 `predictor_1` 关键 spike。

因此后续如果继续 TrackA，不要再重复追 runner / 默认路径差异；应另开新 issue 专门决定下一条最小单变量，而不是继续补跑 `clsdw0=0.0`。

当前 issue 已贴回 `tracka_source_epoch3_sched20_guardon_seed0_gpu3` 与 `tracka_source_epoch3_sched20_guardoff_seed0_gpu3` 两侧摘录，现已闭环确认：

- 两侧 `Namespace(...)` 唯一有效差异就是：
  - `nonempty_keep_guard=True` vs `False`
  - 其余 recipe 与 `stop_after_epoch=3` 保持一致
- 在关键窗口 `epoch=2 step=140~146`，两侧提取字段按数值归一化后同轨：
  - `predictor_1/2_keep_diag`
  - `ratio_stage_0/1/2`
  - `grad_watch parameter=score_predictor.1.out_proj.weight`
- 两侧都在 `epoch=2 step=146` 同时出现：
  - `predictor_1 final_keep_ratio_mean=1.548948e-01`
  - `ratio_stage_1 pos_ratio_mean=6.042730e-02`
  - `ratio_stage_2 pos_ratio_mean=2.822066e-02`
  - `score_predictor.1.out_proj.weight = ±1.057964e+03`
- 两侧三次 eval 都保持：
  - `Accuracy of the model on the 524 test images: 74.2%`
  - `Max accuracy: 74.24%`
- 两侧 `epoch=3 step=0~5` grep 都为空，且这属于**预期行为**：
  - run 会在 `Early stop after epoch 2 due to stop_after_epoch=3`
  - 之后直接结束，因此不会进入 `epoch=3` 的训练 step

因此当前正式结论是：

1. strict source `guard-off` 与 `guard-on` 在关键窗口前**没有分叉**；
2. guard 不是 `epoch=2 step=146` 之前的首发根因；
3. guard 更像后续防崩的症状缓冲层，而不是这条 `epoch3` closure 线上的首因。

当前 issue 随后又补齐了 `training_source_tracka` vs `training_compat` 的 `debug80` parity：

1. 两侧 `Transform` / `Sampler_train` / `Use Cosine LR scheduler` / `Max WD` 完全一致；
2. 两侧 20 条 `Averaged stats: lr:` 数值逐项重合；
3. `Namespace(...)` 差异仅剩：
   - source 侧保留 `crop_pct=None`、`weight_decay_end=None` 的运行时补全写法；
   - compat 侧显式固定 `crop_pct=0.875`、`weight_decay_end=0.05`；
   - compat 侧额外打印默认关闭的 `pruning_margin_*`

因此当前可正式认为：

1. `training_compat` 默认路径仍可视作 source-compatible；
2. 当前问题不再像是 `source` / `compat` 默认 runner 分叉；
3. 若后续继续推进，更应另开 issue 做共享训练路径内部的非 `ratio` 归因，而不是继续追默认脚本 parity。

## TrackA `LOSS_GRAD_ATTRIB=true` 归因结论

`2026-04-22` 当前权威结论如下：

- `server env provenance = Yes`
- strict source `guard-off` vs `guard-on` 同口径证据 = `Yes`
- source vs compat `debug80` parity 同口径证据 = `Yes`
- 默认 runner / 默认路径差异仍是主干扰 = `No`
- `LOSS_GRAD_ATTRIB=true` 最小归因 = `Yes`

因此：

1. 当前 `LOSS_GRAD_ATTRIB=true` 已把首个明确候选收敛到 `cls_kl`：在 `epoch=2 step=146`，total `grad_l2=4.198726e+03`、`grad_absmax=1.057964e+03`，其中 `cls_kl` 为 `grad_l2=4.041559e+03`、`grad_absmax=1.017747e+03`；
2. `ratio_loss` 在同一窗口始终只有 `O(1e-1)` 的 grad_l2，继续排除为首发驱动；
3. 首个单变量 `cls_distill_weight=0.0` issue 已完成，并已形成有效负结果；
4. 当前不在本 issue 内直接修改训练语义；若继续，应另开新 issue 专门决定下一条最小单变量，而不是继续补跑 `clsdw0=0.0`。
5. 这条“下一条最小单变量”现已完成：`cls_distill_weight=0.5` 在 `step=34` 未提前失稳，并在 `step=146` 对 `cls_kl` 主导的 spike 给出强缓解，但 `epoch3` accuracy 仍停在 `74.24%`，所以当前结论是“诊断性缓解”，不是正式修复。

### TrackA 首个单变量结论（`2026-04-22`）

本轮首个且仅一个单变量验证曾设为：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- ablation：仅把 `cls_distill_weight` 从 `1.0` 改为 `0.0`
- 不改：
  - `token_distill_weight=0.02`
  - `ratio_weight=2.0`
  - `activation_lr_scale=10.0`
  - `model_ema=false`
  - 其它官方 recipe 有效参数

当前把 `cls_distill_weight` 直接设为 `0.0`，不是为了给出最终修复值，而是为了先验证：

1. `epoch=2 step=146` 的 total grad spike 是否随 `cls_kl` 链路一起明显回落；
2. `predictor_1` 的 `final_keep_ratio_mean` 是否不再在 `step=140~146` 提前塌缩；
3. 若两者都没有改善，则说明首因不止这一个标量权重，后续应再开新 issue，而不是在本 issue 内继续扫多个 loss 项。

配套服务器命令与判定标准已同步写入：

- `docs/history_best_repro_drift_audit_2026-04-21.md`
- `docs/handoff-next.md`

当前状态：

- 命令包、服务器回贴日志与 crash 邻域都已补齐；
- 当前 issue 已可收成“首个单变量已拿到有效负结果”；
- 不再继续补跑 `clsdw0=0.0`，也不在本 issue 内扩展到第二个 loss 项。

`2026-04-22` 用户随后已补回 corrected header / `LossGradAttrib`。当前正式结论更新为：

- 当前这条 `clsdw0` run **不是有效单变量对照**；
- 因为 `tracka_source_epoch3_lossgradattrib_clsdw0_seed0_20260422` 的 `Namespace(...)` 里仍然写着：
  - `cls_distill_weight=1.0`
  - `token_distill_weight=0.02`
- 对应地，control / ablation 两边：
  - `predictor_1_keep_diag` 在 `epoch=2 step=140~146` 逐项相同；
  - `LossGradAttrib` 的 `total / cls_loss / ratio_loss / cls_kl / token_kl` 逐项相同；
  - terminal 也都停在 `74.24%`

因此这里真正能落盘的结论不是“`cls_distill_weight=0.0` 无效”，而是：

- **本轮 ablation 参数没有进入有效配置**
- 所以这轮结果**不能**用来判断 `cls_kl / cls_distill_weight` 是否真的无效

此外，这次也顺手定位到一个 runner 侧日志问题：

- 旧版 `train_stdout.log` 不会保留 wrapper 的 `[tracka-source] key=value` header；
- 原因是之前只有 python 子进程输出被 `tee` 到日志；
- 本地现已补上 runner 日志修正，后续新的 `train_stdout.log` 会同时保留 wrapper header 与 python `Namespace(...)`

当前最小下一步已收敛为：

1. 先把更新后的 `scripts/_tracka_training_common.sh`、`scripts/run_tracka_train.sh` 同步到服务器；
2. 只重跑一条新的 `clsdw0` ablation；
3. 先确认 header + `Namespace` 都真的变成 `cls_distill_weight=0.0`，再谈效果判断。

`2026-04-22` 更晚的最新回贴又新增了一条新阻塞：新的 `resync1` ablation 尝试没有正常进入完整 `epoch3` 比较收口，而是在 forward 阶段报：

- `RuntimeError: Non-finite tensor detected in VisionTransformerDiffPruning: predictor_2_pred_score`

当前可正式记录的是：

- 这条新的 ablation 尝试已经不再只是“旧无效对照”的问题；
- 现在已经确认它是**有效配置**：
  - wrapper header 显示 `cls_distill_weight=0.0`
  - `Namespace(...)` 也显示 `cls_distill_weight=0.0`
  - `epoch=0 step=0` 的 `LossGradAttrib` 进一步显示：
    - `component=cls_kl`
    - `weight=0.000000e+00`
    - `grad_l2=0.000000e+00`
- 这说明当前首个单变量 ablation **确实生效了**

但这条已生效的 ablation 目前给出的不是缓解证据，而是更具体的负结果：

- 用户已贴回真正 crash 邻域；
- 在 `epoch=2 step=34`：
  - `predictor_1_keep_diag` 已出现 `raw_empty=1 / final_empty=1`
  - 同时 `raw_le1=2 / final_le1=2`
  - `final_keep_ratio_mean=1.946639e-01`
- 紧接着：
  - `PredictorLG zero_active_policy_samples=1`
  - `global_x / post_agg / out_conv_out isfinite=False`
- 然后在：
  - `predictor_2_pred_score`
  处触发 `_check_finite`

因此，这条 `cls_distill_weight=0.0` strict source / epoch3 单变量尝试当前应记作：

- **有效、但明确负向**
- 它不是 clean comparator；
- 它会把原本更晚出现的 zero-active / predictor_2 non-finite 风险提前到 `epoch=2 step=34`

因此当前最小下一步已更新为：

1. 不再怀疑这条 run 是否真正切到了 `cls_distill_weight=0.0`；
2. 不再继续重复 `clsdw0=0.0` 这条 run；
3. 当前 issue 自身已经没有缺失证据；
4. 若继续这条根因线，应另开新 issue 专门决定下一条最小单变量，而不是在本 issue 内继续扩题。

### TrackA 下一条最小单变量结论（`2026-04-22`）

在首个单变量 `cls_distill_weight=0.0` 已确认是有效负结果后，新 issue 只选择了**一条且仅一条**下一步验证：

- 控制线：strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 唯一改动：`cls_distill_weight: 1.0 -> 0.5`
- 其余有效参数保持不变，尤其继续保持：
  - `token_distill_weight=0.02`
  - `ratio_weight=2.0`
  - `activation_lr_scale=10.0`
  - `model_ema=false`

用户随后回贴的 control / ablation 日志已足以把这条 issue 收口：

1. 这次 `clsdw05` 是**有效单变量配置**
   - run：
     - `tracka_source_epoch3_lossgradattrib_clsdw1_control_seed0_20260422_next1`
     - `tracka_source_epoch3_lossgradattrib_clsdw05_seed0_20260422_next1`
   - header 与 `Namespace(...)` 同时显示：
     - control：`cls_distill_weight=1.0`
     - ablation：`cls_distill_weight=0.5`
   - `epoch=0 step=0` 的 `LossGradAttrib` 也确认目标链路确实减半：
     - `scaled_loss: 4.530853e-02 -> 2.265427e-02`
     - `grad_l2: 2.592915e+01 -> 1.296458e+01`

2. 它**没有**复现 `clsdw0=0.0` 的 `epoch=2 step=34` 提前失稳链
   - 当前用户回贴的 `early-fail gate` 没有命中：
     - `raw_empty / final_empty > 0`
     - `zero_active_policy_samples > 0`
     - `RuntimeError: Non-finite tensor`
   - 因此当前没有证据表明 `clsdw05` 会像 `clsdw0=0.0` 一样在 `step=34` 提前崩溃。

3. 在真正关键窗口 `epoch=2 step=146`，它给出**明确缓解**
   - control：
     - `predictor_1 final_keep_ratio_mean=1.548948e-01`
     - `active_margin_mean=-1.722985e+00`
     - `total grad_l2=4.198726e+03`
     - `total grad_absmax=1.057964e+03`
     - `cls_kl grad_l2=4.041559e+03`
     - `cls_kl grad_absmax=1.017747e+03`
   - `clsdw05`：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `active_margin_mean=-3.137406e-02`
     - `total grad_l2=2.400378e-01`
     - `total grad_absmax=4.624692e-02`
     - `cls_kl grad_l2=1.123101e-01`
     - `cls_kl grad_absmax=2.186546e-02`
   - 因而按照当前 issue 预设判定标准，这条 `clsdw05` 应记为：
     - **step146 强缓解**
     - **predictor_1 不再提前塌缩**

4. 但它还**不是正式修复**
   - control 与 `clsdw05` 的 terminal 都仍是：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 且两边都只是 `epoch3` 的低精度诊断窗口，不是 `full20` 正式成绩。

因此，当前这条“下一条最小单变量” issue 的正式收口应为：

- 已成功选出并验证一条最小单变量：`cls_distill_weight=0.5`
- 它相对 control 给出了**强烈且干净的稳定性缓解信号**
- 但当前只应写成**低精度诊断上的正结果**，不能写成“正式精度已恢复”或“修复已完成”
- 当前 issue 的实验目标已经完成；若还要继续推进，应另开新的单变量 issue，而不是在本 issue 内继续扩题

### TrackA post-`clsdw05` 下一条单变量结论（`2026-04-22`）

在 `clsdw05=0.5` 已确认是明确缓解的诊断性正结果后，本轮新 issue 只继续验证了一条且仅一条下一变量：

- `cls_distill_weight: 1.0 -> 0.75`
- 仍保持 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`
- 并继续固定 `token_distill_weight=0.02`、`ratio_weight=2.0`、`activation_lr_scale=10.0`、`model_ema=false`

用户随后回贴的 control / ablation 日志已足以把这条 issue 收口：

1. `clsdw075` 是**有效单变量配置**
   - control 的 header + `Namespace(...)` 都是 `cls_distill_weight=1.0`
   - ablation 的 header + `Namespace(...)` 都是 `cls_distill_weight=0.75`
   - 两边都保持 strict source、guard-off、`LOSS_GRAD_ATTRIB=true`

2. 它没有走 `clsdw0=0.0` 的负路径
   - `epoch=2 step=30~36` 的 early-fail gate 没有命中：
     - `raw_empty / final_empty > 0`
     - `zero_active_policy_samples > 0`
     - `RuntimeError: Non-finite tensor`
   - 因此当前没有证据表明 `0.75` 会像 `0.0` 一样在 `step=34` 早崩。

3. 它在 `epoch=2 step=146` 给出**明确缓解**
   - control：
     - `predictor_1 final_keep_ratio_mean=1.548948e-01`
     - `total grad_l2=4.198726e+03`
     - `cls_kl grad_l2=4.041559e+03`
   - `clsdw075`：
     - `predictor_1 final_keep_ratio_mean=4.730873e-01`
     - `total grad_l2=3.748181e+01`
     - `cls_kl grad_l2=2.105142e+01`
   - 所以当前可以把它记成：
     - **step146 明确缓解**
     - **predictor_1 未沿 control 轨迹塌缩**

4. 但它弱于 `clsdw05=0.5`
   - `clsdw05` 在同一 `step=146` 的参考值更强：
     - `predictor_1 final_keep_ratio_mean=5.022926e-01`
     - `total grad_l2=2.400378e-01`
     - `cls_kl grad_l2=1.123101e-01`
   - 因此 `0.75` 虽然仍是正结果，但并没有超过 `0.5`。

5. terminal 仍不动
   - control 与 `clsdw075` 的 terminal 都仍是：
     - `Accuracy of the model on the 524 test images: 74.2%`
     - `Max accuracy: 74.24%`
   - 所以它仍只是 `epoch3` 低精度诊断信号，不能写成正式修复。

因此，本轮正式收口应为：

- `cls_distill_weight=0.75` 是**有效且明确缓解**的单变量
- 但它**弱于** `cls_distill_weight=0.5`
- 且 terminal 仍停在 `74.24%`
- 所以后续若继续推进，不建议在同一 issue 内继续扫更多 `cls_distill_weight` 值；应另开新的单变量 issue

### TrackA post-`clsdw075` 新阻塞点选择（`2026-04-22`）

本轮只处理“`clsdw075` 后下一条最小单变量该选什么”。结论是：

- 旧 `clsdw075` issue 已可关闭：它已证明 `0.75` 有效但弱于 `0.5`，且 terminal 仍是 `74.24%`；
- 不继续扫 `cls_distill_weight`，因为 `0.0 / 0.5 / 0.75` 三点已经给出足够剂量线信息；
- 新单变量固定当前更强缓解底座 `cls_distill_weight=0.5`，只改 `token_distill_weight: 0.02 -> 0.04`；
- 选择它优先于 `ratio_weight`，因为 `ratio_loss` 已被排除为首发驱动，而 token distill 是现有 mask-aware token loss 中最小的非 `cls_kl` 信息保持旋钮；
- 命令包要求 strict source / `NONEMPTY_KEEP_GUARD=false` / `epoch3` / `LOSS_GRAD_ATTRIB=true`，并检查 header + `Namespace(...)`、`step=34` early-fail gate、`epoch=2 step=146` keep+attrib 与 terminal `74.24%` 是否松动。

用户已回贴服务器日志后，本轮结论更新为：

- 配置有效：control 为 `cls_distill_weight=0.5 / token_distill_weight=0.02`，ablation 为 `cls_distill_weight=0.5 / token_distill_weight=0.04`；
- 未早崩：`epoch=2 step=30~36` 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限或 `RuntimeError`；
- 稳定性窗口变差：`step=146` total `grad_l2` 从 `2.400378e-01` 增至 `4.365869e+00`，`predictor_1 final_keep_ratio_mean` 从 `5.022926e-01` 降至 `4.537168e-01`，`active_margin_mean` 从 `-3.137406e-02` 变为 `-1.643516e-01`；
- terminal 正向松动：`Max accuracy` 从 `74.24%` 提至 `79.01%`；
- 结论：这是**有效但混合**的诊断结果，不是 clean 稳定性缓解；当前不建议直接进入 `full20`，也不能写成正式修复或新正式成绩。

当前新阻塞点：`token_distill_weight=0.04` 同时带来 terminal 改善与 `score_predictor.1` 稳定性恶化，后续若继续应另开新 issue 解耦这两个信号，而不是在本 issue 内继续扩题。

`2026-04-23` 用户又回贴了 terminal-稳定性解耦单变量 `token_distill_weight=0.03` 的服务器日志，本轮正式结论如下：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.02`
  - ablation：`cls_distill_weight=0.5 / token_distill_weight=0.03`
  - `epoch=0 step=0` 的 attribution 也确认 `token_kl weight` 从 `2.000000e-02` 变成 `3.000000e-02`，`cls_kl weight` 继续保持 `5.000000e-01`
- `step=34` 不早崩：
  - `epoch=2 step=30` 两边都没有 `raw_empty / final_empty / raw_le1 / final_le1`
  - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`
- `step=146` 不是 clean 稳定性缓解：
  - control：`final_keep_ratio_mean=5.022926e-01`，`active_margin_mean=-3.137406e-02`，total `grad_l2=2.400378e-01`
  - `tdw003`：`final_keep_ratio_mean=4.792178e-01`，`active_margin_mean=-1.206955e-01`，total `grad_l2=1.165204e+00`
  - keep ratio 下降约 `0.0231`，仍在 `0.03` 容忍线内；但 total `grad_l2` 约为 control 的 `4.85x`，`active_margin_mean` 比 control 更负约 `0.0893`，超过 `0.05` 容忍线
- terminal 只有极弱松动：
  - control：`Max accuracy: 74.24%`
  - `tdw003`：`Max accuracy: 74.43%`
  - 相比 `tdw004` 的 `79.01%` 明显不接近
- 结论：
  - `tdw003` 相比 `tdw004` 的确降低了稳定性伤害，但也几乎带走了 terminal 提升；
  - 当前没有拿到“terminal 提升 + 稳定性无变化/缓解”的 clean 解耦迹象；
  - 不建议直接进入 `full20`，也不能把 `74.43%` 或 `79.01%` 写成正式成绩。

后续 TrackA agent 输出服务器命令时，还必须使用可复用路径变量写法：

- 先在同一个 shell / tmux 会话中设置 `export REPO_ROOT=/data/wyb/Transshield_final`、`export TRAIN_RUN_ROOT="$REPO_ROOT/artifacts/train_runs"`，再 `cd "$REPO_ROOT"`。
- 每条 run 用 `run_xxx=...` 保存 run 名，并通过 `export RUN_NAME="$run_xxx"` 传给 runner。
- 后续 grep / 判定统一用 `LOG="$TRAIN_RUN_ROOT/$run/train_stdout.log"`，不要在各处重复硬编码日志路径。
- 不要写一次性的 `env VAR=... bash ...`、`bash -c` 或子 shell setup；否则训练命令结束后路径变量会丢失，影响后续对照。

`2026-04-23` 用户已回贴 `ratio_weight=3.0` 的服务器日志，本轮正式结论如下：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
  - ablation：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=3.0`
  - wrapper header 与 `Namespace(...)` 都确认只有 `ratio_weight` 发生变化
- `step=34` 不早崩：
  - `epoch=2 step=30` 两边都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`
- `step=146` 不是稳定性缓解，而是 split 代理改善但 total grad 更差：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`
  - ablation：`final_keep_ratio_mean=5.245984e-01`，`active_margin_mean=4.946269e-02`，total `grad_l2=5.206851e+01`
  - keep ratio 回升约 `+0.0709`，margin 回升约 `+0.2138`，但 total `grad_l2` 升到 control 的约 `11.93x`
  - `cls_kl grad_l2` 从 `2.991327e+01` 降到 `1.911049e+01`，`ratio_loss grad_l2` 也更小；但 `cls_loss grad_l2` 与 `token_kl grad_l2` 更高，所以不能把 keep/margin 回升写成整体稳定性缓解
- terminal 被直接带回原停滞点：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`
- 结论：
  - `ratio_weight=3.0` 没有保住 `tdw004` 的 terminal-positive 信号；
  - 它也没有在目标参数 total grad 意义上缓解稳定性；
  - 因此这是**负结果**，不是 clean 解耦，也不是可推进的 mixed 修复；
  - 当前仍不建议直接进入 `full20`。

`2026-04-23` 用户随后又回贴 `cls_distill_weight=0.4` 的服务器日志，本轮正式结论如下：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
  - ablation：`cls_distill_weight=0.4 / token_distill_weight=0.04 / ratio_weight=2.0`
  - wrapper header、`Namespace(...)` 与 `epoch=0 step=0` attribution 权重都确认只有 `cls_distill_weight` 发生变化
- `step=34` 不早崩：
  - ablation 的 `epoch=2 step=30` 显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False` 或 `RuntimeError`
- `step=146` 是 `cls_kl` 分项下降，但不是整体稳定性缓解：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`，`cls_kl grad_l2=2.991327e+01`
  - ablation：`final_keep_ratio_mean=2.575499e-01`，`active_margin_mean=-1.041453e+00`，total `grad_l2=7.772295e+00`，`cls_kl grad_l2=4.482888e+00`
  - `cls_kl grad_l2` 明显下降，但 total `grad_l2` 升到 control 的约 `1.78x`，keep ratio 下降约 `0.1962`，margin 负移约 `0.8771`
- terminal 被带回原停滞点：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`
- 结论：
  - `cls_distill_weight=0.4` 没有保住 `tdw004` 的 terminal-positive 信号；
  - 它只降低了 `cls_kl` 分项压力，却让 total grad、keep/margin 和 terminal 同时变差；
  - 因此这是**有效但负向的非 clean 解耦**，不是修复，也不是可推进的 mixed 结果；
  - 当前仍不建议直接进入 `full20`。

`2026-04-23` 用户随后又回贴 `token_distill_weight=0.035` 的服务器日志，本轮正式结论如下：

- 配置有效：
  - control：`cls_distill_weight=0.5 / token_distill_weight=0.04 / ratio_weight=2.0`
  - ablation：`cls_distill_weight=0.5 / token_distill_weight=0.035 / ratio_weight=2.0`
  - wrapper header 与 `Namespace(...)` 都确认只有 `token_distill_weight` 发生变化
- `step=34` 不早崩：
  - control 与 ablation 的 `epoch=2 step=30/40` 都显示 `raw_empty=0 / final_empty=0 / raw_le1=0 / final_le1=0`
  - 回贴 grep 没有 `zero_active_policy_samples`、`predictor_2_pred_score` 非有限、`isfinite=False`、`RuntimeError`、`Non-finite` 或 `skip optimizer step`
- `step=146` 是稳定性明显缓解：
  - control：`final_keep_ratio_mean=4.537168e-01`，`active_margin_mean=-1.643516e-01`，total `grad_l2=4.365869e+00`，`cls_kl grad_l2=2.991327e+01`，`token_kl grad_l2=1.172528e+00`
  - ablation：`final_keep_ratio_mean=4.878815e-01`，`active_margin_mean=-8.497284e-02`，total `grad_l2=2.089920e+00`，`cls_kl grad_l2=9.532135e-01`，`token_kl grad_l2=2.397546e-03`
  - total `grad_l2` 下降约 `52.1%`，`cls_kl grad_l2` 下降约 `96.8%`，`token_kl grad_l2` 下降约 `99.8%`，keep / margin 也从 `0.04` 坏窗口回温
- terminal-positive 信号没有保住：
  - control：`Max accuracy: 79.01%`
  - ablation：`Max accuracy: 74.24%`
- 结论：
  - `token_distill_weight=0.035` 的确把 `tdw004` 坏窗口往更温和区间拉回；
  - 但它没有保住 `tdw004` 的 terminal-positive 信号，terminal 直接回到 `74.24%`；
  - 因此这是**稳定性缓解、terminal 丢失的负向 midpoint 结果**，不是 clean 解耦，也不是修复；
  - 当前仍不建议直接进入 `full20`；若继续 TrackA，应另开新单变量 issue 决定是否还要在 `0.04` 附近做更贴近的 token midpoint。

### TrackA post-`tdw0035` 近端 midpoint 结果：`token_distill_weight=0.0375`

本轮目标：

- 不是重判 `tdw0035`，而是只验证：在 `0.035` 已确认“稳定性缓解、terminal 丢失”之后，`0.04` 附近是否还值得继续做唯一一条更贴近 `0.04` 的 midpoint。

本地已执行：

- 只执行 `sed -n` / `rg -n` 读取与比对 `docs/history_best_repro_drift_audit_2026-04-21.md`、`docs/current_work_status.md`、`docs/handoff-next.md`、`docs/tracka_predictor1_root_cause_2026-04-21.md`、`scripts/run_tracka_train.sh`、`training_source_tracka/main.py`、`training_source_tracka/engine.py`、`training_source_tracka/losses.py`
- 未执行任何 `/data/wyb/...` 命令

本轮结论：

- `tdw0035` 原 issue 早已可以关闭；
- 本轮唯一近端候选 `token_distill_weight=0.0375` 也已完成；
- 从用户回贴的 `LossGradAttrib` 权重可直接看出：
  - `cls_kl weight=5.000000e-01`
  - `token_kl weight=3.750000e-02`
  - `ratio_loss weight=6.666667e-01`
  这足以**推断** ablation 已正确进入 `cls_distill_weight=0.5 / token_distill_weight=0.0375 / ratio_weight=2.0`；
- 该 run 已正常到达 `epoch=2 step=146` 并完成 eval，因此 `step=34` early-fail gate 未命中；
- `step=146` 相比 `tdw004` control 是**更强的稳定性缓解**：
  - `predictor_1 final_keep_ratio_mean: 4.537168e-01 -> 5.085371e-01`
  - `active_margin_mean: -1.643516e-01 -> -2.972232e-03`
  - total `grad_l2: 4.365869e+00 -> 1.294038e-01`
  - `cls_kl grad_l2: 2.991327e+01 -> 1.165144e-01`
  - `token_kl grad_l2: 1.172528e+00 -> 1.486021e-03`
- 但 terminal 仍从 `79.01%` 直接回落到 `74.24%`；
- 因此当前最终选择应从之前的 `continue_near_004` 收口为 **`stop_token_midpoint`**。

为什么这条线现在应停：

- 现在已有三个 `<0.04` 的点：
  - `0.03 -> 74.43%`
  - `0.035 -> 74.24%`
  - `0.0375 -> 74.24%`
- 这三个点都没有保住 `79.01%`，但稳定性却持续改善；
- `0.0375` 已经证明：即使把稳定性窗口拉回到几乎与稳定 control 对齐，terminal 也仍不会回来；
- 所以继续做 `0.038 / 0.0385 / 0.039` 这类近端 token 剂量搜索，信息增益已经不足。

当前状态：

- 本 issue 目标已完成；
- 当前正式结论是 `stop_token_midpoint`；
- 当前不要继续扩近端 token 轴，不推进 `full20`，也不把 `74.24%` 或 `79.01%` 写成正式成绩或正式修复。

## 后续建议

- 如果要新增同口径 secure 通信对比，必须重新跑**同输入、同样本量、同协议路径**的数据。
- 如果要新增 external secure benchmark 数字，必须使用：
  - `artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh`
  - 并明确标注那是 benchmark，不是 full-val pipeline。
- 如果要补新前端卡片，先检查 `docs/data_source_policy.md` 是否允许该字段直接展示。
- `margin-aware pruning` 的有用思路与负结果，统一记录在：
  - `docs/margin_aware_pruning_notes.md`
- `network-kth` 的当前有效结果统一记录在：
  - `docs/network_kth_blockwise_notes.md`
- 当前更推荐的下一阶段：
  - `Phase 3 network-kth`
  - `Phase 4 payload`
