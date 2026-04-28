# OpenBumbleBee Transshield network-kth bridge

This example is the smallest bridge from the frozen Transshield stage-2 bundle into the OpenBumbleBee side.

It does **not** run the full ViT model. Instead it only takes over the pruning-side compare-network step:

1. load the exported `masked_score` input sidecar from the standalone `Transshield` repo
2. load the machine-readable compare-network manifest
3. run the fixed odd-even compare-swap network
4. emit a checker-compatible `kth_threshold` candidate sidecar

## Why this exists

This is the safest first BumbleBee integration step because:

- the frozen Transshield baseline stays untouched
- the interface is already fixed by checker/eval artifacts
- the output can be validated immediately by the existing DynamicViT checker

## Inputs

- `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_network_kth_manifest.json`
- `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_network_kth_input_*.pt`

Both are staged as a minimal runtime-input pack so this bridge no longer depends on historical `server_pipeline_run/` outputs.

## CPU smoke run

```sh
mkdir -p artifacts/server_pipeline_run/network_kth_smoke
python integrations/openbumblebee/transshield_network_kth_bridge/transshield_network_kth_bridge.py \
  --manifest-json artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_network_kth_manifest.json \
  --input-pt artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_network_kth_input_smoke8.pt \
  --output-pt artifacts/server_pipeline_run/network_kth_smoke/stage2_secure_network_kth_candidate_from_server.pt \
  --output-json artifacts/server_pipeline_run/network_kth_smoke/stage2_secure_network_kth_candidate_from_server.json \
  --runtime cpu
```

Then verify it inside `Transshield` with:

```sh
python tools/transshield_secure_network_kth.py check \
  --reference-pt artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_network_kth_reference_smoke8.pt \
  --candidate-pt artifacts/server_pipeline_run/network_kth_smoke/stage2_secure_network_kth_candidate_from_server.pt
```

## SPU follow-up

The script also exposes `--runtime spu` plus `--config`, so the same bridge boundary can later be moved from CPU/JAX to SPU without changing the Transshield-side checker contract.

## Server-friendly workflow

If local hardware is limited, prepare a portable pack first:

```sh
python tools/transshield_openbumblebee_bridge.py prepare \
  --output-dir artifacts/server_bridge_pack_smoke8
```

That step only copies inputs and emits command templates; it does **not** run BumbleBee locally.
