# SPU Python Fast Path Profile Summary

- Matched fastpath lines: `100`
- RPC request bytes: `5830636`
- RPC response bytes: `1855657`
- RPC total bytes: `7686293`
- make_shares input bytes: `899188`

Diagnosis: Python fastpath RPC/cloudpickle traffic is nonzero while SPU Link details remain zero; default fast runtime communication is visible at the Python distributed layer, not in the inspected yacl link counters.

## Diagnostic Caveat
- C++ LinkDetails are diagnostic-only for this fastpath: count=`2`, all_zero=`True`
- Primary communication display should use Python fastpath RPC bytes, not C++ LinkDetails zero counters.

## Top RPC Requests
- `builtin_spu_run` -> `node:0@127.0.0.1:32989`: bytes=`1997856`, count=`2`, max=`998928`
- `builtin_spu_run` -> `node:1@127.0.0.1:41879`: bytes=`1997856`, count=`2`, max=`998928`
- `<lambda>` -> `node:0@127.0.0.1:32989`: bytes=`1805754`, count=`6`, max=`412047`
- `make_shares` -> `node:0@127.0.0.1:32989`: bytes=`13836`, count=`6`, max=`2306`
- `builtin_fetch_object` -> `node:0@127.0.0.1:32989`: bytes=`8250`, count=`10`, max=`825`
- `builtin_fetch_object` -> `node:1@127.0.0.1:41879`: bytes=`4950`, count=`6`, max=`825`
- `builtin_fetch_meta` -> `node:0@127.0.0.1:32989`: bytes=`1566`, count=`8`, max=`279`
- `builtin_spu_init` -> `node:0@127.0.0.1:32989`: bytes=`284`, count=`2`, max=`142`
- `builtin_spu_init` -> `node:1@127.0.0.1:41879`: bytes=`284`, count=`2`, max=`142`

## Top RPC Responses
- `builtin_fetch_object` <- `node:0@127.0.0.1:32989`: bytes=`1825721`, count=`10`, max=`821883`
- `builtin_fetch_object` <- `node:1@127.0.0.1:41879`: bytes=`26364`, count=`6`, max=`4394`
- `make_shares` <- `node:0@127.0.0.1:32989`: bytes=`1146`, count=`6`, max=`191`
- `<lambda>` <- `node:0@127.0.0.1:32989`: bytes=`804`, count=`6`, max=`134`
- `builtin_fetch_meta` <- `node:0@127.0.0.1:32989`: bytes=`614`, count=`8`, max=`80`
- `builtin_spu_run` <- `node:1@127.0.0.1:41879`: bytes=`496`, count=`2`, max=`248`
- `builtin_spu_run` <- `node:0@127.0.0.1:32989`: bytes=`496`, count=`2`, max=`248`
- `builtin_spu_init` <- `node:0@127.0.0.1:32989`: bytes=`8`, count=`2`, max=`4`
- `builtin_spu_init` <- `node:1@127.0.0.1:41879`: bytes=`8`, count=`2`, max=`4`

## Fetch Objects
- `node:0` -> `node:1`: count=`4`

## Parsed Files
- `logs/spu_nodes/node_0.log`
- `logs/spu_nodes/node_1.log`
- `/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_w10_secure_check_20260417_212831/secure_spu/step_logs/00_network_kth_bridge.log`
- `/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_w10_secure_check_20260417_212831/secure_spu/step_logs/00_network_kth_bridge.log`
