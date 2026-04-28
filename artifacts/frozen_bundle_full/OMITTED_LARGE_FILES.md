Large training checkpoints were intentionally not copied into this GitHub-ready bundle.

Omitted items:

- `checkpoint-best.pth`
- `checkpoint-best-repro1.pth`

Reason:

- each file is larger than the practical GitHub single-repo submission target
- `modified_plaintext_model_state_dict.pth` is already included here and is the smaller portable weight artifact to keep with this competition repo

Other originally linked small files were materialized locally:

- `threshold_best.json`
- `train_stdout.log`
- `eval_threshold_stdout.log`
