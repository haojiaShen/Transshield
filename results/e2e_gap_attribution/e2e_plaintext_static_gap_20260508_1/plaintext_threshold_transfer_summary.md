# Plaintext Score Threshold Transfer

- Scope: `plaintext_same_images_reference.json` on heldout64 / heldout128 / heldout238
- Score rule: `class1_score = logit1 - logit0`
- Goal: test whether the large `original_plaintext_same_subset≈50%` drop is mainly a zero-boundary issue, and whether a public threshold learned on one split transfers to the others.

## Best Threshold Per Split

- heldout64: best threshold `0.6570841521`, best accuracy `93.75%`
- heldout128: best threshold `0.6230766103`, best accuracy `89.84375%`
- heldout238: best threshold `0.5688221827`, best accuracy `91.17647%`

## Transfer Matrix

| Threshold source | heldout64 | heldout128 | heldout238 |
|---|---:|---:|---:|
| heldout64 (`0.657084`) | `93.75%` | `89.0625%` | `89.91597%` |
| heldout128 (`0.623077`) | `93.75%` | `89.84375%` | `90.75630%` |
| heldout238 (`0.568822`) | `92.1875%` | `88.28125%` | `91.17647%` |

## Interpretation

- `plaintext full-model` 与 `static whole-forward` 的 score ranking 在三个 heldout 上都保持高相关，主要问题不是 ranking failure，而是 zero boundary 明显右移。
- 与之前 `SPU-aware threshold-only` 在 heldout128 上容易明显回退不同，这组 `plaintext score` threshold 的 cross-split transfer 仍保持在 `88.28% ~ 93.75%`。
- 这说明若后续要继续找“不重训”的轻量恢复路线，`full-model / static score boundary calibration` 是值得继续保留的候选，而不应再把 `original_plaintext_same_subset≈50%` 直接视为 full-model 本体失效。
