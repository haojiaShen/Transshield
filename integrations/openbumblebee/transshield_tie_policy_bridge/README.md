# OpenBumbleBee Transshield tie-policy bridge

This bridge is the follow-up step after `network-kth`.

It consumes:

- `masked_score` input sidecar
- `kth_threshold` candidate sidecar from the previous bridge step

and emits:

- checker-compatible `selected_equal_mask` / `tie_keep_quota` sidecar

## Local non-runtime preparation

Prepare a server pack without running BumbleBee:

```sh
python tools/transshield_openbumblebee_tie_bridge.py prepare \
  --output-dir artifacts/server_tie_bridge_pack_smoke8
```

## Expected server order

1. Run the `network-kth` bridge first
2. Place `stage2_secure_network_kth_candidate_from_server.pt` into the tie pack directory
3. Run the tie bridge command from `commands.json`
4. Run the tie checker command from `commands.json`

## Validation contract

The tie bridge output is checked by:

- `tools/transshield_secure_tie_payload.py check`

The current smoke reference is:

- `artifacts/inference_ready_config/selection_mode_runtime_inputs_verified/stage2_secure_tie_policy_lowest_smoke8.pt`
