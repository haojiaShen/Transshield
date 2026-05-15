# Keep-mask Whole-forward Wrapper Scaling Report

## Judgement
- `status = scaling_observed_but_needs_more_points`: current keep-mask wrapper runs are valid and privacy-consistent, but more sample-count points are still needed to judge scaling stability with confidence.

## Privacy Boundary
- `host_plaintext_pixel_values_materialized = false`
- `host_private_share_tensors_loaded = false`
- `input_mode = party_local_debug_share_load`
- `input_pt = null`
- `private_input_paths_redacted = true`
- `reveal_policy = final_logits_only`
- `runtime_pruning_keep_mask_stage_count = 3`
- `spu_forward_graph_mode = monolithic`
- `spu_params_mode = secret`
- `privacy_consistent = true`

## Aggregate
- `run_count = 4`
- `sample_count_min/max/total = 1 / 32 / 57`
- `elapsed_sec_total = 11277.8594`
- `sec_per_sample mean/min/max = 207.5605 / 194.6303 / 233.8283`
- `sec_per_sample_spread_ratio = 1.2014`
- `logits_max_abs_error_max = 0.003555`
- `probabilities_max_abs_error_max = 0.001773`
- `all_finite_logits = true`
- `all_argmax_match_ratio_one = true`
- `all_threshold_match_ratio_one = true`

## Runs
- `smoke1_partylocal_secret_20260509_2`: `/home/yclcg/Transshield_final/results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke1_partylocal_secret_20260509_2/`
  - `sample_count = 1`
  - `elapsed_sec = 233.8283`
  - `sec_per_sample = 233.8283`
  - `logits/probabilities max_abs_error = 0.002585 / 0.001197`
  - `argmax/threshold match = 1.0 / 1.0`
- `smoke8_partylocal_secret_20260509_1`: `/home/yclcg/Transshield_final/results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke8_partylocal_secret_20260509_1/`
  - `sample_count = 8`
  - `elapsed_sec = 1612.6744`
  - `sec_per_sample = 201.5843`
  - `logits/probabilities max_abs_error = 0.002789 / 0.001353`
  - `argmax/threshold match = 1.0 / 1.0`
- `smoke16_partylocal_secret_20260509_1`: `/home/yclcg/Transshield_final/results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke16_partylocal_secret_20260509_1/`
  - `sample_count = 16`
  - `elapsed_sec = 3203.1877`
  - `sec_per_sample = 200.1992`
  - `logits/probabilities max_abs_error = 0.002633 / 0.001287`
  - `argmax/threshold match = 1.0 / 1.0`
- `smoke32_partylocal_secret_20260509_1`: `/home/yclcg/Transshield_final/results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_smoke32_partylocal_secret_20260509_1/`
  - `sample_count = 32`
  - `elapsed_sec = 6228.1691`
  - `sec_per_sample = 194.6303`
  - `logits/probabilities max_abs_error = 0.003555 / 0.001773`
  - `argmax/threshold match = 1.0 / 1.0`

## Pairwise Scaling
- `smoke1_partylocal_secret_20260509_2 -> smoke8_partylocal_secret_20260509_1`: `sample_ratio = 8.0000`, `elapsed_ratio = 6.8968`, `sec_per_sample_ratio = 0.8621`, `incremental_sec_per_new_sample = 196.9780`
- `smoke8_partylocal_secret_20260509_1 -> smoke16_partylocal_secret_20260509_1`: `sample_ratio = 2.0000`, `elapsed_ratio = 1.9863`, `sec_per_sample_ratio = 0.9931`, `incremental_sec_per_new_sample = 198.8142`
- `smoke16_partylocal_secret_20260509_1 -> smoke32_partylocal_secret_20260509_1`: `sample_ratio = 2.0000`, `elapsed_ratio = 1.9444`, `sec_per_sample_ratio = 0.9722`, `incremental_sec_per_new_sample = 189.0613`
