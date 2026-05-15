# E2E Chunked Runtime Axis Report

- status: `chunked_runtime_does_not_recover_selected_wrong_sample`
- reason: E2E_SPU_BLOCK_CHUNK_SIZE=3 changes the SPU graph mode to reveal_less_block_chunked, but the selected high-margin wrong sample remains class 0 and its calibrated score stays negative.
- sample: heldout238 index `121`, target `1`, image `/data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png`

| mode | graph mode | score | prediction | correct |
|---|---|---:|---:|---|
| monolithic heldout238 | default | -0.692276 | 0 | False |
| chunk3 single-sample | reveal_less_block_chunked | -0.69191 | 0 | False |

## Interpretation

- Treat block chunking primarily as runtime/communication graph-size optimization; continue accuracy work through numeric approximation reduction or boundary robustness rather than chunk size alone.
