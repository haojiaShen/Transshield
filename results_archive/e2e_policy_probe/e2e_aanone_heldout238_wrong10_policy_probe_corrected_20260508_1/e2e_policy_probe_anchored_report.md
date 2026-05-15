# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_wrong10_policy_probe_corrected_20260508_1_anchored`
- sample_count: `10`
- baseline_variant: `source_heldout238_exact_uniform_clip0`
- status: `policy_variant_recovers_baseline_wrong_samples`
- recovered_by_any_nonbaseline_variant: `4`
- regressed_by_any_nonbaseline_variant: `0`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| source_heldout238_exact_uniform_clip0 | 0.0000 | 0 | 10 | 0.343417 | True |
| rerun_exact_uniform_clip0 | 0.0000 | 0 | 10 | 0.342816 | True |
| rerun_exact_uniform_clip3 | 40.0000 | 4 | 6 | 0.636949 | True |
| rerun_exact_uniform_clip0_lncmp64 | 0.0000 | 0 | 10 | 0.342851 | True |

## Per-Sample Scores

| source_index | target | image | source_heldout238_exact_uniform_clip0 pred/score | rerun_exact_uniform_clip0 pred/score | rerun_exact_uniform_clip3 pred/score | rerun_exact_uniform_clip0_lncmp64 pred/score |
|---:|---:|---|---:|---:|---:|---:|
| 121 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png | 0/-0.692276 wrong | 0/-0.692398 wrong | 0/-0.709396 wrong | 0/-0.690796 wrong |
| 220 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png | 0/-0.64122 wrong | 0/-0.642838 wrong | 0/-0.695862 wrong | 0/-0.640823 wrong |
| 167 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png | 0/-0.553375 wrong | 0/-0.551956 wrong | 0/-0.724808 wrong | 0/-0.552643 wrong |
| 227 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00474.png | 0/-0.48085 wrong | 0/-0.481613 wrong | 0/-0.66684 wrong | 0/-0.480759 wrong |
| 21 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png | 1/0.32608 wrong | 1/0.325516 wrong | 0/-0.622375 ok | 1/0.32576 wrong |
| 49 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00210.png | 1/0.26976 wrong | 1/0.268997 wrong | 0/-0.515945 ok | 1/0.268295 wrong |
| 217 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00434.png | 0/-0.0510559 wrong | 0/-0.0491333 wrong | 0/-0.617661 wrong | 0/-0.0481568 wrong |
| 196 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00342.png | 0/-0.0606842 wrong | 0/-0.0582581 wrong | 0/-0.665878 wrong | 0/-0.0594788 wrong |
| 23 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00114.png | 1/0.143997 wrong | 1/0.143936 wrong | 0/-0.64212 ok | 1/0.146866 wrong |
| 54 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00223.png | 1/0.214874 wrong | 1/0.213516 wrong | 0/-0.508606 ok | 1/0.214935 wrong |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
