# Vendored SPU modification notes

This directory redistributes a vendored SPU subtree under the upstream Apache License 2.0 terms.

## License retention

- Upstream license text is retained in `spu_vendored/LICENSE`.
- The upstream project already carries its own per-file Apache 2.0 headers.
- At the TransShield repository level, modified or locally maintained vendored files may include an additional
  `TransShield local modification notice` or `TransShield local vendor notice` header to make provenance explicit
  in the competition deliverable.

## Maintained provenance references

- `docs/evidence/spu_bumblebee_local_modifications.md`
- `docs/transshield_modifications_improvements_log.md`

## Files explicitly called out by the current deliverable notes

The current report and provenance documents reference the following vendored files as part of the locally adapted tree:

- `spu_vendored/libspu/spu.proto`
- `spu_vendored/libspu/mpc/cheetah/arithmetic.h`
- `spu_vendored/libspu/mpc/cheetah/arithmetic.cc`
- `spu_vendored/libspu/mpc/cheetah/protocol.cc`

These notes do not replace the upstream copyright or Apache 2.0 license headers.

## Runtime performance layer

The final repository keeps the pip-installed SPU package unchanged and applies
project-scoped runtime optimizations from:

- `integrations/transshield_runtime/e2e_secure_vit/spu_compile_cache.py`: content-addressed JAX-to-SPU executable cache; cached entries contain graph code and public output shape metadata, never request values or secret shares.
- `integrations/transshield_runtime/e2e_secure_vit/secure_pruning_ops.py`: fixed-schedule exact top-k and optional oblivious token compaction.

Both optimizations are selectable at the runner layer, so the historical mask
graph remains available for A/B comparison and rollback.

An additional opt-in SPU 0.9.3b0 source patch is preserved at
`spu_vendored/patches/cheetah_parallel_rlwe_packing.patch`. It retains the
upstream internally parallel path for a single response group and parallelizes
only multiple independent RLWE packing groups with non-overlapping output
storage. The patch is validated in an isolated runtime and is not applied by
the repository's default build.

`spu_vendored/patches/cheetah_packing_sumdiff.patch` is a second optional
source-level patch for the same isolated runtime. It computes the two RLWE
packing butterfly branches as an in-place sum plus a separate difference,
removing one full ciphertext copy from the common non-empty path without
changing the ring expression. Its report-scale same-VPS A/B result is a
marginal 0.61% latency reduction, so it remains an explicit research patch and
is not applied by the default build.
