# Original plaintext baseline runtime

This directory keeps the minimal code needed to evaluate the original plaintext DynamicViT checkpoint inside the single final `Transshield` repository.

Included files:

- `datasets.py`
- `utils.py`
- `models/dyvit.py`
- `models/__init__.py`

Purpose:

- support `original plaintext` vs `modified plaintext` comparison
- avoid keeping a second standalone baseline repository as a final deliverable

This runtime is intentionally minimal and is only meant for checkpoint evaluation and comparison scripts.
