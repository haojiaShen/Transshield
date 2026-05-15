# Tools

This directory now keeps only the scripts needed for the final Transshield run path.

Canonical implementations live here unless a section below says otherwise.

- Root `transshield_openbumblebee_pipeline.py`
- Root `transshield_blockwise_kth_selection_manifest.py`
- Root `transshield_stagewise_threshold_report.py`

are compatibility wrappers only.

The canonical `network-kth` bridge implementation lives in:

- `integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py`
- `integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py`
  - whole-forward e2e integration entry for the new parallel track
  - now supports current whole-forward SPU runs, same-policy verify, calibration, and guarded secret-runtime helpers alongside the legacy sidecar path

Root `transshield_network_kth_bridge.py` is also a compatibility wrapper only.

Secure-code navigation is maintained here plus `docs/architecture.md`; the old
`secure_infer/` navigation directory is retired.

Legacy `phase3_lower_tail` prototypes/planners, one-off `SPU` fastpath logging patchers,
and local repo audit/cleanup helpers have been removed from `tools/`; if a script is not
listed below, treat it as retired rather than a supported final-repo entrypoint.

Before using this toolchain, keep the current project boundary fixed:

- current mainline model = `ViT / DynamicViT`, because token-level pruning boundary is the carrier of the current `F_less / F_mux` innovation
- current secure-facing pruning semantics = masking-friendly `keep/zero`, not direct token deletion
- current dynamic pruning boundary = sample/stage-specific `kth` threshold, not the final binary classification threshold
- current secure backend = `OpenBumbleBee / SPU`, shared by:
  - the formal delivery line `secure sidecar + replay`
  - the later whole-forward secure ViT expansion line
- `CNN + ViT` hybrid is not part of the current mainline toolchain
- `embedding / position encoding` secure optimization is only a later `P2` candidate

## Main entrypoints

- `transshield_inference_friendly_server_pack.py`
  - generate the server-side runnable script pack
  - internal structure is now split into command builders, shortcut emitters, and manifest/script writers
- `transshield_build_clean_deploy_repo.py`
  - build a clean self-contained server runtime repo from the full local repository
  - materializes the current delivery bundle into regular files so server deployment no longer depends on broken bundle symlinks
- `transshield_openbumblebee_pipeline.py`
  - run / verify / replay the secure bridge pipeline
  - internal structure is now split into step-command builders, replay helpers, and CLI parser construction
- `transshield_openbumblebee_inference_replay.py`
  - replay the final secure pruning boundary
  - internal structure is now split into boundary-report helpers, replay-stage helpers, and CLI parser construction
- `transshield_fair_external_comparison.py`
  - build the fair same-dataset comparison payload and Markdown report
  - internal structure is now split into parsing, fairness checks, table building, and Markdown section builders
- `transshield_chat_demo.py`
  - run the upload / plaintext / secure web demo backend
  - internal structure is now split into summary builders, state helpers, request validation, and CLI parser construction
- `transshield_e2e_secure_infer.py`
  - early e2e secure-inference scaffolding for the new parallel track
  - currently provides privacy-boundary contract export, client-side pixel preprocessing, plaintext reference inference, static whole-forward plaintext reference, future-candidate comparison, and SPU implementation-plan output
- `update_web_demo_summary.py`
  - regenerate `artifacts/web_demo_assets/best_demo_content.json` for the current verified bundle / report set
- `transshield_plaintext_secure_score_compare.py`
  - compare plaintext bundle inference against secure replay scores sample-by-sample
- `transshield_plaintext_checkpoint_eval.py`
  - evaluate a plaintext checkpoint from either the baseline repo or the modified Transshield repo
- `transshield_plaintext_checkpoint_infer.py`
  - run checkpoint-based plaintext inference for selected images / report wrappers
- `transshield_plaintext_eval_compare.py`
  - compare two plaintext evaluation summaries with Accuracy / AUC / F1 deltas
- `transshield_external_threshold_search.py`
  - generate a compatible `threshold_best.json` for an external baseline DynamicViT checkpoint
- `transshield_secure_profile_summary.py`
  - summarize secure run time / communication / SPU log profiling into one JSON
  - internal structure is now split into log extraction, communication diagnosis, and payload-summary builders
- `transshield_e2e_whole_forward_summary.py`
  - summarize E2E whole-forward SPU candidate JSON plus `logs/spu_nodes` communication diagnostics for the current dual-private ViT path
- `transshield_e2e_keepmask_result_summary.py`
  - summarize a completed keep-mask whole-forward wrapper run directory from its `candidate JSON + compare JSON`
  - useful after remote sync, when you want the key privacy/runtime/error metrics without manually chaining `jq`
- `transshield_e2e_keepmask_doc_snippet.py`
  - renders `keepmask_result_summary.json` into a doc-ready Markdown snippet
  - useful for quickly pasting a finished smoke run into `current_work_status.md` or `handoff-next.md`
- `transshield_e2e_keepmask_doc_block.py`
  - merges multiple keep-mask summary JSONs into one doc-ready Markdown block with shared privacy fields and per-run metrics
  - useful when updating the combined `smoke1/smoke8/smoke16/smoke32` section in handoff/status docs
- `transshield_e2e_keepmask_scaling_report.py`
  - aggregates multiple keep-mask summary JSONs into one scaling report JSON/Markdown with privacy-boundary consistency, sec/sample trend, and decision-stability checks
  - useful once `smoke1/smoke8/smoke16/smoke32` have been synced locally and you want one report instead of scattered run notes
- `transshield_e2e_runtime_efficiency_report.py`
  - compare E2E eval metrics across isolated/non-isolated runs and report elapsed time, sec/sample, speedup, accuracy, finite-logit status, and privacy-field checks
- `transshield_e2e_calibration_drift_report.py`
  - analyze existing E2E raw logits under multiple public output calibrations and rank wrong samples for follow-up block probes
- `transshield_e2e_calibration_decision_report.py`
  - combine multiple E2E calibration transfer reports into an accuracy-first vs loss-first deployment decision summary
- `transshield_e2e_calibration_sample_report.py`
  - summarize per-sample E2E calibration transfer CSVs, including static-vs-SPU-aware recovered samples and block-probe recommendations
- `transshield_e2e_image_list_overlap_report.py`
  - summarize class balance and pairwise overlap across E2E image lists, useful for calibration fit/held-out split audits
- `transshield_e2e_block_sweep_summary.py`
  - summarize CPU-vs-SPU block-probe compare JSONs across multiple blocks, including max-abs drift, relative L2 drift, high-cosine checks, and debug-graph safety notes
- `transshield_e2e_block_probe_batch_report.py`
  - aggregate multiple single-sample block-probe summaries into one batch judgement
  - used to test whether heldout residual wrong samples share a consistent late-block cumulative drift pattern
- `transshield_e2e_public_threshold_recovery.py`
  - sweep public score thresholds on existing E2E logits and test cross-subset transfer for lightweight SPU-aware output calibration
- `transshield_public_logit_affine_calibration.py`
  - fit an accuracy-constrained public affine calibration for binary E2E logits and export `weights=[-scale,scale]` plus public bias for the OpenBumbleBee E2E runner
- `transshield_public_logit_temperature_calibration.py`
  - fit a boundary-preserving public temperature calibration that scales the bias-only score without changing its final decision boundary
- `transshield_slice_debug_shares.py`
  - slice an existing debug additive-share manifest into a smaller public/P1/P2 manifest set for isolated E2E repro runs
- `transshield_delivery_acceptance_report.py`
  - aggregate the current delivery-line evidence into one JSON / Markdown acceptance report
  - inputs can include plaintext full-val, fairness, boundary checks, legacy replay compare, E2E same-policy verify, and guarded secret-runtime summary
- `transshield_secure_profile_compare.py`
  - compare two secure profiling summaries
- `transshield_selection_mode_profile_report.py`
  - compare two secure runs from different `network-kth` selection modes
  - internal structure is now split into artifact parsing, scalar/step compare helpers, and Markdown section builders
- `transshield_followup_tracker.py`
  - summarize an experiment-repo follow-up log into final-repo JSON / Markdown for controlled result handoff
- `transshield_comparison_report.py`
  - assemble the final baseline / modified / secure comparison report from wrapper outputs
- `transshield_cpu_spu_profile_report.py`
  - summarize CPU-vs-SPU profile wrapper outputs into one comparison report
- `transshield_margin_ablation_report.py`
  - aggregate margin-aware pruning server runs into one report bundle
- `transshield_competition_scorecard.py`
  - generate a promotion / readiness / external-gap scorecard for the current verified candidate
  - internal structure is now split into input parsing, secure-evidence/checklist builders, and Markdown section builders
- `transshield_secure_diagnosis_report.py`
  - build the secure-side selected-image diagnosis summary
- `transshield_single_image_comparison.py`
  - build a baseline-vs-modified single-image comparison board
  - internal structure is now split into trace-report builders, stage-panel builders, and summary-board helpers
- `transshield_selected_image_report.py`
  - render selected-image plaintext diagnostics into a compact report
- `transshield_extract_spu_followup_summary.py`
  - extract a compact SPU follow-up summary from one secure run directory
- `transshield_runtime_branch_compare.py`
  - compare fast-vs-communication-visible SPU runtime branches for the verified candidate
  - internal structure is now split into summary normalization, compare sections, and recommendation builders
- `transshield_merge_aux_comm_profile.py`
  - merge fast-runtime timings with auxiliary communication-visible counters into one display-ready profile
- `transshield_patch_spu_multirank_profile.py`
  - hot-patch installed `spu/utils/distributed_impl.py` so nonzero ranks can keep profiling behind `SPU_ENABLE_MULTI_RANK_PROFILE=1`
- `transshield_spu_runtime_setup.py`
  - start / stop / restart colocated `SPU` runtime nodes and record `logs/spu_runtime_ports.json`
- `transshield_extract_spu_build_errors.py`
  - extract actionable compiler-error excerpts for the current `SPU` rebuild blockers from a Bazel build log
- `transshield_fastpath_profile_summary.py`
  - summarize Python fastpath RPC/profile logs into JSON / Markdown
  - internal structure is now split into log matching, bucket updates, summary finalization, and Markdown section builders
- `transshield_standardized_secure_benchmark_report.py`
  - summarize standardized secure external benchmark runs into JSON / Markdown
- `transshield_token_pruning_visualization.py`
  - export token-pruning visualization assets plus `token_pruning_trace_report.md` for the demo / docs

These are wrapped by:

- `artifacts/server_inference_friendly_pack/run_plaintext_eval.sh`
- `artifacts/server_inference_friendly_pack/run_plaintext_model_compare.sh`
- `artifacts/server_inference_friendly_pack/run_secure_score_compare.sh`
- `artifacts/server_inference_friendly_pack/run_full_final_comparison_suite.sh`
- `artifacts/server_inference_friendly_pack/run_secure_profile_summary.sh`
- `artifacts/server_inference_friendly_pack/run_secure_profile_compare.sh`
- `artifacts/server_inference_friendly_pack/run_secure_selection_mode_profile_compare.sh`

## Required comparison scenarios retained in this repo

- `original plaintext` vs `modified plaintext`
  - baseline eval: `run_plaintext_eval.sh baseline`
  - modified eval: `run_plaintext_eval.sh modified`
  - summary compare: `run_plaintext_model_compare.sh`
- `modified plaintext` vs `secure 2PC replay`
  - secure replay compare: `run_secure_score_compare.sh`
  - one-shot wrapper: `run_full_final_comparison_suite.sh`

The original plaintext path is kept self-contained via:

- `references/original_plaintext_runtime/datasets.py`
- `references/original_plaintext_runtime/utils.py`
- `references/original_plaintext_runtime/models/dyvit.py`

Bundled evaluation assets kept in-repo:

- `artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth`
- `artifacts/baselines/original_plaintext_threshold_best_fix3.json`
- `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/modified_plaintext_model_state_dict.pth`
- current bundle threshold payload is resolved from the bundle metadata / generated threshold JSON during runtime

## ViT training and bundle export

- `transshield_binary_threshold_search.py`
- `freeze_export_candidate.py`
- `transshield_prepare_verified_bundle.py`
- `verify_frozen_candidate.py`

## Secure sidecar export and checking

- `transshield_secure_sidecar_export_suite.py`
- `transshield_secure_network_kth.py`
  - 统一收口原来的 `input_export` / `manifest` / `export` / `checker` / `branch_eval`
  - 子命令：`input-export` / `manifest` / `export` / `check` / `branch-eval`
- `transshield_secure_tie_payload.py`
  - 统一收口原来的 `export` / `checker` / `branch_eval`
  - 子命令：`export` / `check` / `branch-eval`

## Bridge helpers kept for pipeline support

- `transshield_openbumblebee_bridge.py`
- `transshield_openbumblebee_tie_bridge.py`

## Internal support modules

- `transshield_input_selection.py`
- `transshield_stage2_bundle.py`
- `transshield_stagewise_threshold_report.py`
- `transshield_threshold_branch_eval.py`
  - internal structure is now split into tie-stat helpers, kth-mask builders, and eval-loop helpers
- `transshield_slice_debug_shares.py`
  - slices E2E debug additive-share manifests by contiguous ranges or explicit non-contiguous indices
  - supports `--source-paths-file` so selected probe reports can keep original image paths
- `transshield_e2e_policy_probe_report.py`
  - compares selected-sample E2E candidate `.pt` files across secure graph policy variants
  - used by selected/mixed heldout238 policy probes to distinguish output calibration recovery from secure-graph/window sensitivity
- `transshield_e2e_gap_attribution_report.py`
  - compares `reference static -> cpu static -> spu candidate` PT payloads under either raw or calibrated logits
  - used to separate raw secure-graph drift from SPU-side public output calibration gains on heldout subsets
- `transshield_e2e_plaintext_static_gap_report.py`
  - compares full-model plaintext per-sample logits against static whole-forward per-sample logits
  - reports score correlation, affine boundary shift, and whether the gap is mainly a public-threshold misalignment rather than ranking failure
- `transshield_e2e_plaintext_bridge_calibration.py`
  - translates plaintext best-threshold into static/E2E raw-score space through an affine bridge and evaluates the resulting public calibration against held-out raw E2E logits
  - used to test whether the plaintext-static boundary insight is deployable, or only explanatory
- `transshield_e2e_output_profile_compare.py`
  - compares completed real E2E run directories across output-calibration profiles such as `accuracy_first`, `loss_first_affine`, and `loss_first_temperature`
  - summarizes threshold/argmax accuracy, calibrated BCE, raw-graph-vs-static drift, latency, and communication into one JSON/Markdown report
- `transshield_e2e_approx_eval_metrics.py`
  - rebuilds `e2e_approx_eval_metrics.json` from existing share/reference/candidate artifacts
  - reports both full-model plaintext context and static whole-forward reference metrics, and isolates raw secure-graph drift when calibration is present
- `transshield_remote_pty.py`
  - local maintenance helper for password-based SSH/rsync in this development environment; reads the password from an environment variable and does not store it in repo files
- `transshield_remote_run_watch.py`
  - local helper for long-running remote E2E/keep-mask jobs
  - can query remote run-dir status, inspect whether `candidate/compare JSON` already landed, and sync the finished run directory back to the local repo

## Supplementary stage-2 analysis scripts retained for explanation and trace reproduction

- `transshield_forward_trace.py`
- `transshield_pruning_trace.py`
- `transshield_reference_checker.py`
- `transshield_stage2_report.py`
  - 统一收口原来的 Stage-2 说明 / contract 报告入口：
  - `tensor-contract`
  - `pruning-semantics`
  - `f-mux-spec`
  - `forward-dataflow`
  - `policy-spec`
  - `secure-kth-contract`
- `transshield_kth_threshold_report.py`
- `transshield_secure_kth_checker.py`

These scripts are not the default competition entrypoints, but they now stay in-repo to support:

- historical result backtracking
- stage-2 design explanation
- trace / contract / checker regeneration from the frozen bundle
