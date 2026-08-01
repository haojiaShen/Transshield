# Transshield runtime e2e secure ViT track

This directory hosts the retained **whole-forward e2e secure inference** path.

It is the only integration subtree in `integrations/transshield_runtime/` that still
belongs to the current final-delivery runtime chain.

## Deployment boundary

The current runnable showcase is still a **single-machine colocated 2PC/SPU prototype**. This subtree already contains the privacy-forward bridge needed for migration, especially split public/P1/P2 manifests, `--party-local-share-load`, and `--redact-private-input-paths`. The repo also provides a deployment-bundle generator, basic SPU party-node launcher, and minimal hospital/AI/coordinator split gateway through `tools/transshield_spu_runtime_setup.py render-deployment`, `start-party`, and `showcase_api.split_gateway`; however, it does not yet provide production-grade hospital/AI authentication, scheduling, auditing, or private model-parameter loading.

For the Chinese migration checklist, see `docs/party_split_2pc.md`.

## Current status

The repository now contains the first e2e scaffolding in:

- `tools/transshield_e2e_secure_infer.py`
- `integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py`
- `integrations/transshield_runtime/e2e_secure_vit/cpu_static_vit.py`
- `integrations/transshield_runtime/e2e_secure_vit/spu_static_vit.py`
- `integrations/transshield_runtime/e2e_secure_vit/debug_probe.py`

That scaffolding currently does:

1. define the e2e privacy boundary contract
2. simulate **client-side** image preprocessing into `pixel_values`
3. optionally simulate client-side preprocessing directly into debug additive
   share files plus public/P1/P2 manifests, without writing plaintext
   `client_pixel_values.pt`
4. run a **plaintext reference** against the frozen verified bundle
5. run a **static whole-forward plaintext reference** that keeps the student blocks/head but bypasses runtime pruning decisions
6. run a **CPU candidate** for the same whole-forward contract
7. run an experimental **static JAX/SPU candidate** for the same whole-forward contract
8. compare a candidate against that whole-forward reference

The `SPU` backend is intentionally narrow: it covers the static DeiT-S/ViT
whole-forward contract only, secret-shares `pixel_values`, uses public model
parameters by default for the first smoke, reveals final logits only, and does
not yet include runtime `masking-pruning` decisions.

For the privacy-boundary migration, the retained whole-forward wrapper exposes
two relevant modes:

- `E2E_SHARE_ONLY=1`: writes a debug additive share bundle manifest and exits
  before writing a plaintext pixel package. It also writes a public manifest
  without private share paths and separate `P1` / `P2` party manifests.
- `E2E_RECONSTRUCT_FROM_SHARES=1`: reconstructs plaintext pixels from that
  debug share bundle only as a temporary compatibility bridge for the current
  CPU/SPU backend.

These debug additive shares are **not** production MPC shares. The intended
next milestone is to replace host reconstruction with true per-party share
ingestion by the SPU runtime.

The whole-forward runner also has an experimental bridge for that milestone:
`run --runtime spu --input-share-manifest-json <manifest>` loads the debug
additive share files and sends the two shares through `P1` and `P2` separately
before adding them inside the SPU graph. This avoids feeding a plaintext
`pixel_values` tensor as the runner input, but it is still a local debug bridge
because the current launcher can see both share files. The production target is
to have each party load only its own share.

The preferred debug bridge is now the split-manifest form:
`--input-share-public-manifest-json`, `--input-p1-share-manifest-json`, and
`--input-p2-share-manifest-json`. The public manifest contains no private share
paths, while each party manifest contains only its own share path. This is one
step closer to the production party boundary, but still not final MPC serving
until the web/API gateway and runner no longer see both private party
manifests.

For split-share failures, use `audit-input-shares` through
`run_e2e_secure_whole_forward.sh audit-shares` before running deeper blocks. It
reveals reconstructed pixels and patch-embed tensors only for explicit debug
localization, so it must not be treated as the production reveal policy.

For the next privacy-boundary step, `run --runtime spu --party-local-share-load`
lets P1/P2 load their own share files inside party devices from the split party
manifests. The driver no longer materializes private share tensors in memory,
although the current demo runner can still be configured with both party
manifest paths.
The server wrapper now also defaults `E2E_REDACT_PRIVATE_INPUT_PATHS=1`, which
passes `--redact-private-input-paths` so candidate `.pt` files and summary JSON
do not persist legacy/P1/P2 private share manifest paths. Keep it enabled for
privacy-forward runs; set it to `0` only for explicit local debugging. In share
input modes the wrapper also omits `--input-pt`, so the candidate metadata does
not point back to a plaintext client pixel package.

After the `depth11/12` party-local runtime/link boundary, the runner now also
exposes `--spu-block-chunk-size` via `E2E_SPU_BLOCK_CHUNK_SIZE`. This splits the
SPU transformer blocks into multiple SPU calls while keeping intermediate token
states as SPU values and revealing only final logits. It is intended for
runtime-stability diagnosis and should first be tested with public params and
`standard` attention.

After the later same-depth correction, the current minimal failure boundary is
`depth0` passing and `depth1` failing. Use
`block1-subgraph-smoke` through `run_e2e_secure_whole_forward.sh block1-smoke`
to run debug-only SPU subgraphs for the first transformer block. This command
reveals intermediate tensors intentionally and must not be used as the
production e2e reveal policy.

After `block1-smoke` also hit `grpc_status:14 / Socket closed`, the integration
entry was split so new diagnosis does not keep growing one monolithic file:
`transshield_e2e_secure_vit.py` is the CLI/command layer, `cpu_static_vit.py`
holds CPU static reference helpers, `spu_static_vit.py` holds SPU/JAX
whole-forward and share-audit routines, and `debug_probe.py` holds debug-only
subgraph probes.

The next lower-level diagnostic is `runtime-primitive-smoke`, exposed through
`run_e2e_secure_whole_forward.sh runtime-smoke`. It uses synthetic transformer
shaped tensors and does not load private images, share manifests, or model
weights. Its stages are `scalar_add`, `layer_norm`, `qkv_linear`,
`attention_scores`, `attention_probs`, `attention_context`,
`projection_residual`, `mlp_hidden_square`, and `mlp_residual`.

The latest server milestone is `depth=5` with party-local debug share loading
under the diagnostic runtime-carriability combo:
`E2E_SPU_LAYER_NORM_POLICY=affine` and `E2E_SPU_ATTENTION_POLICY=uniform`.
The summary metadata reports finite logits, no `input_pt`, no host plaintext
pixel materialization, no host private share tensor loading, redacted private
input paths, and `input_mode=party_local_debug_share_load`. This closes the
current party-local privacy input boundary for the first five block diagnostic
graph, with final-logits-only reveal. It is not the final numerically
equivalent secure ViT path because exact secret layer norm, secret softmax, and
full attention-context matmul remain unresolved runtime blockers. At `depth=4`
and `depth=5` the logits are already very large and probabilities saturate, so
this milestone must be treated as runtime/privacy-boundary evidence rather than
a valid classification result.

The next deployable direction is `E2E_SPU_LAYER_NORM_POLICY=public_calibrated`.
It uses public calibration activation statistics instead of private per-sample
layer-normalization reductions. The calibration JSON is generated offline with
`run_e2e_secure_whole_forward.sh calibrate-ln` from a non-private/public
calibration pixel package, then passed to the SPU run as
`E2E_SPU_LAYER_NORM_CALIBRATION_JSON`. This keeps the party-local private input
boundary unchanged while avoiding the activation explosion caused by bare
`affine` layer norm.

The first server result for this path is positive at `depth=5` with
party-local share loading and `uniform` attention: final logits return to a
small finite range and probabilities no longer saturate. This is a usable
privacy/runtime milestone for the approximate path, but it is still not the
original exact ViT because attention is still public-uniform rather than secret
softmax attention.

The strongest current approximate baseline is now full-depth `depth=12` with
party-local share loading, `public_calibrated` layer norm, `uniform` attention,
and `fixed_square` activation. It completes on SPU with final-logits-only
reveal and non-saturated probabilities. This is the deployable approximate
baseline to compare and harden next; exact secret layer norm, secret softmax
attention, dynamic pruning, and independent P1/P2 launch remain future work.
Use `E2E_SPU_BATCH_SIZE=1` for this baseline. A two-sample run with
`spu_batch_size=2` exposed a batched-graph numeric explosion, while the same
two samples with `spu_batch_size=1` produced small non-saturated logits for
both samples.

The historical server-side wrapper has been archived under
`archive/deprecated/artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh`.
Older wrapper names may appear in historical notes, but they are not part of
the current cleaned final repository.

As of `2026-04-22`, the server run `tracka_e2e_secure_poc_cpu` has already
closed the current CPU contract on `524` samples:

- `static_whole_forward_reference` is finite
- `cpu candidate` is finite
- `logits/probabilities max_abs_error = 0.0`
- `argmax_match_ratio = 1.0`
- `threshold_match_ratio = 1.0`

So the next real milestone is no longer another CPU check, but the first
minimal `run --runtime spu` backend for the same whole-forward contract.

As of `2026-04-23`, that minimal backend is no longer only local scaffolding:
the server has already completed same-depth smoke validation for
`depth=0..5 / sample=1 / public params`. A historical default-colocated
`depth=6` full run did fail, but the later rerun with
`SPU_DISABLE_COLOCATED_OPTIMIZATION=1` restored `depth=6` to a stable
same-depth pass against the CPU reference, and the same `nocoloc` setting has
already kept `depth=7`, `depth=8`, `depth=9`, `depth=10`, `depth=11`, and
`depth=12` stable for the single-sample case as well. The later `2026-04-25`
follow-up refined the previous `depth=12 / sample_count=2` mismatch: using
`spu_batch_size=2` makes the batched run complete without node errors, but the
two-sample compare still only reaches `argmax/threshold match = 0.5 / 0.5`.
Per-sample inspection shows that sample 0 remains decision-consistent while
sample 1 flips. Running sample 1 alone at full depth also flips, so the current
working conclusion is that this is primarily a near-boundary sample plus
accumulated SPU numeric drift, not merely a batched-shape runtime failure.

The current block-level evidence is also stronger now. The server-side
`block6_probe_compare_cpu_vs_spu_depth6.json` shows that `norm1_out_cls` is
still close, while `attn_out_cls` is the first stage with a large distortion.
For the later failing sample-1 path, `depth8` is still decision-consistent and
`depth9` flips; `block9_probe_compare_cpu_vs_spu_depth9_sample1.json` ranks
`mlp_out_cls` as the largest-amplitude error, while `attn_out_cls` already has
a very low cosine similarity. A matched sample-0 block-9 probe remains
decision-consistent even though it also has a strong attention-direction drift,
so the current hypothesis is more specific: block-9 attention drift is a common
risk, and sample 1 crosses the final boundary because the later MLP/head path
amplifies that drift on a more fragile near-boundary input.

There is also a separate runtime-stability issue on the server for some
`depth=6` full runs: the SPU job can fail mid-run with `grpc UNAVAILABLE`,
`Socket closed`, or `Not connected` errors on the internal SPU link. The
server-friendly wrapper therefore now exposes
`SPU_DISABLE_COLOCATED_OPTIMIZATION=1` so that the same `spu` / `probe-spu`
entrypoint can rerun with `--disable-colocated-optimization` without manually
starting the runtime first. On `2026-04-23`, that switch already proved useful:
the `tracka_e2e_secure_spu_depth6_smoke1_nocoloc_20260423` rerun completed
without node errors and recovered `argmax/threshold match = 1.0 / 1.0`.

## Current integration entry

The dedicated integration entry is now:

- `integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py`

and the archived historical wrapper is:

- `archive/deprecated/artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh`

Current supported modes:

1. `prepare`
2. `run --runtime cpu`
3. `run --runtime spu`
4. `verify`
5. `audit-input-shares`
6. `probe-block`
7. `compare-block-probe`
8. `block1-subgraph-smoke`
9. `runtime-primitive-smoke`

### Secure pruning performance switches

The runner keeps two compatible secure-pruning graph modes:

- `--spu-secure-pruning-mode mask` keeps the historical 196-spatial-token mask graph for rollback and A/B checks.
- `--spu-secure-pruning-mode compact` uses a fixed bitonic schedule to move token payloads with their secret scores and then runs later blocks at 137/96/67 spatial tokens. Uniform-attention blocks account for the omitted logical zero-token contribution before averaging.
- `--spu-final-block-cls-only` is an optional uniform-attention optimization: the final block computes the selected-token value aggregate and the CLS output only. When the last block is also a pruning point, it builds the exact keep mask but skips the final oblivious sort of the 384-dimensional token payload. Omit the flag to restore the previous graph for A/B checks.

Uniform-attention blocks also project their shared mean-V output once and then
broadcast the projected value. This removes repeated identical secure matrix
multiplications without changing the graph's mathematical result or any
privacy boundary.

`--spu-uniform-attention-value-fusion` additionally uses
`mean(linear(x)) = linear(mean(x))` to aggregate normalized tokens before the
V projection. The identity is exact over real arithmetic, but it changes the
fixed-point truncation order; keep the switch tied to the validated FM64
profile and re-check logit/probability drift before enabling it for another
field or fraction-bit setting.

The CLS-only switch does not change weights, depth, keep counts, parameter
visibility, input-share loading, or the final-logits-only reveal policy. It is
restricted to `uniform` attention because only that policy has a token-shared
value aggregate that makes the unused spatial outputs removable without an
attention-matrix approximation.

`--spu-compile-cache-dir PATH` enables a content-addressed cache across fresh
runner processes. Cache entries contain the compiled SPU executable and public
output shape tree only; input values, model values, and secret shares are not
serialized. Candidate JSON now records model-load, SPU-runtime, compile-cache,
per-chunk execute, and final-reveal timing separately.

The online `2pc.template.json` disables HAL/PPHLO per-op profiling. Use the
profile-enabled `2pc.json` or `2pc_e2e.template.json` when collecting operator
or communication evidence, and compare latency only between runs with the same
profiling setting.

`run --runtime spu` is experimental. Start with `--max-samples 1
--spu-batch-size 1 --spu-params-mode public`, then verify with
`--allow-prefix-candidate` before trying larger sample counts. Use
`--spu-params-mode secret` only after the public-parameter smoke is stable.

For block-level numeric-drift ablation, the SPU path also exposes two
experimental knobs. `--spu-attention-policy smoothed|standard` preserves the
existing smoothed policy by default and can switch to a standard softmax for
attention-only diagnosis. `--spu-activation-override bundle|...` preserves the
bundle activation by default; non-`bundle` values are SPU-only diagnostics and
must not be reported as same-semantics CPU/SPU comparisons.

On the current server, JAX may warn that CUDA-enabled `jaxlib` is missing and
fall back to CPU. Do not start with all 12 transformer blocks in that state.
Use `--static-depth-limit 0` first, then try `1`, and only scale up after each
prefix passes.

For the current `depth=5 -> 6` drift issue, prefer the explicit block-level
probe path instead of continuing to push deeper immediately. The intended
workflow is:

1. run `probe-block --runtime cpu --static-depth-limit 6 --probe-block-index 5`
2. run `probe-block --runtime spu --static-depth-limit 6 --probe-block-index 5`
3. run `compare-block-probe` on those two JSON outputs

This debug path intentionally reveals intermediate CLS-token summaries for the
selected block, so it must be treated as explicit offline debugging only, not
as the default e2e reveal policy.

After the `2026-04-25` depth-9 sample-1 analysis, there is an additional
important rule: `probe-block` final logits/probabilities are **debug-graph
outputs**, not full-candidate decision outputs. Revealing intermediate tensors
can change the SPU/JAX graph and its numeric behavior, especially when probing
the last executed block. Use probe JSONs only for intermediate stage drift
attribution (`block_input_cls`, `attn_out_cls`, `mlp_out_cls`,
`block_output_cls`, etc.). Full decisions must come from non-probe
`run --runtime spu` followed by `verify`.

## Intended implementation order

1. client-side preprocess and secret-share `pixel_values`
2. static `deit-s / ViT` whole-forward secure inference without pruning
3. plaintext-vs-secure checker on final logits
4. migrate current `dyvit` masking-pruning semantics into the secure forward

## Why this is separate

The current bridges only protect the pruning boundary payload.

The e2e track aims to move the privacy boundary earlier, so it should not be mixed into the existing live bridge entrypoints until the whole-forward secure path is stable enough to validate independently.
