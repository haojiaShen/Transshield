# OpenBumbleBee runtime configs

These files are the in-repo runtime configs for the Transshield secure bridge.

## Files

- `2pc.json`: default colocated 2-party runtime config used by Transshield server scripts
- `2pc.template.json`: copy/edit template if your server ports or temp folders differ
- `2pc_e2e.template.json`: reserved template for the future e2e whole-forward secure inference track; keep it separate from the current live sidecar config

## When to edit

Edit these values on the server if needed:

- `nodes`
- `spu_internal_addrs`
- `experimental_data_folder`

The default paths are colocated localhost settings so the full `Transshield` repo can be copied to a server and run without depending on an external OpenBumbleBee checkout just for config files.

For server runs, prefer the generated SPU entrypoints or:

```bash
python tools/transshield_spu_runtime_setup.py start \
  --config configs/openbumblebee/2pc.json \
  --template configs/openbumblebee/2pc.template.json \
  --backup \
  --restart \
  --remove-unsupported-cheetah-fields \
  --log-dir logs/spu_nodes \
  --state-json logs/spu_runtime_ports.json
```

This rewrites the config with currently free localhost ports, removes the
`approx_less_precision` Cheetah field for older installed `spu` wheels that do
not expose it, starts each colocated node service separately, warms up the SPU
runtime, and records the selected ports and node PIDs in
`logs/spu_runtime_ports.json`.

When restarted through this helper, existing `logs/spu_nodes/node_*.log` files
are rotated to `*.prev.<timestamp>` before new node processes start, so each
fresh run gets clean node logs for communication/profile parsing.

If a specific `spu` build keeps emitting all-zero `Link details` counters, you
can also temporarily disable colocated optimization for diagnosis:

```bash
python tools/transshield_spu_runtime_setup.py start \
  --config configs/openbumblebee/2pc.json \
  --template configs/openbumblebee/2pc.template.json \
  --backup \
  --restart \
  --remove-unsupported-cheetah-fields \
  --disable-colocated-optimization \
  --log-dir logs/spu_nodes \
  --state-json logs/spu_runtime_ports.json
```
