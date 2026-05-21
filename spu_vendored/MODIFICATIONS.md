# Vendored SPU modification notes

This directory redistributes a vendored SPU subtree under the upstream Apache License 2.0 terms.

## License retention

- Upstream license text is retained in `spu_vendored/LICENSE`.
- The upstream project already carries its own per-file Apache 2.0 headers.
- At the TransShield repository level, modified or locally maintained vendored files may include an additional
  `TransShield local modification notice` or `TransShield local vendor notice` header to make provenance explicit
  in the competition deliverable.

## Maintained provenance references

- `docs/transshield_bumblebee_spu_modifications.md`
- `docs/transshield_modifications_improvements_log.md`

## Files explicitly called out by the current deliverable notes

The current report and provenance documents reference the following vendored files as part of the locally adapted tree:

- `spu_vendored/libspu/spu.proto`
- `spu_vendored/libspu/mpc/cheetah/arithmetic.h`
- `spu_vendored/libspu/mpc/cheetah/arithmetic.cc`
- `spu_vendored/libspu/mpc/cheetah/protocol.cc`

These notes do not replace the upstream copyright or Apache 2.0 license headers.
