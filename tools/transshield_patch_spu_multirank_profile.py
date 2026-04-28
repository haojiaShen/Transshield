#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path


TARGET_SNIPPET = """    if my_rank != 0:
        spu_config.enable_action_trace = False
        spu_config.enable_hal_profile = False
        spu_config.enable_pphlo_profile = False
"""

REPLACEMENT_SNIPPET = """    preserve_multi_rank_profile = os.environ.get("SPU_ENABLE_MULTI_RANK_PROFILE", "0") == "1"
    if my_rank != 0:
        spu_config.enable_action_trace = False
        if not preserve_multi_rank_profile:
            spu_config.enable_hal_profile = False
            spu_config.enable_pphlo_profile = False
"""


def backup_file(path: Path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = path.with_name(path.name + f'.bak.{timestamp}')
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def ensure_os_import(text: str):
    if '\nimport os\n' in text or text.startswith('import os\n') or '\nimport os\r\n' in text:
        return text
    lines = text.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (lines[insert_at].startswith('import ') or lines[insert_at].startswith('from ')):
        insert_at += 1
    lines.insert(insert_at, 'import os')
    return '\n'.join(lines) + ('\n' if text.endswith('\n') else '')


def patch_file(path: Path):
    original = path.read_text(encoding='utf-8', errors='replace')
    if 'SPU_ENABLE_MULTI_RANK_PROFILE' in original:
        return 'already_patched', None
    if TARGET_SNIPPET not in original:
        raise RuntimeError('Target snippet not found; installed SPU layout may differ from expected 0.9.3b0 wheel')

    patched = ensure_os_import(original)
    patched = patched.replace(TARGET_SNIPPET, REPLACEMENT_SNIPPET)
    backup_path = backup_file(path)
    path.write_text(patched, encoding='utf-8')
    return 'patched', backup_path


def main():
    parser = argparse.ArgumentParser(description='Hot-patch installed SPU distributed_impl.py to preserve profile on nonzero ranks behind an env flag.')
    parser.add_argument(
        '--target',
        default=os.environ.get(
            'SPU_DISTRIBUTED_IMPL',
            str(
                Path(os.environ.get('SPU_CONDA_PREFIX', '/data/wyb/conda_envs/transshield'))
                / 'lib/python3.9/site-packages/spu/utils/distributed_impl.py'
            ),
        ),
        help='Path to installed spu/utils/distributed_impl.py',
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        raise SystemExit(f'Missing target file: {target}')

    status, backup_path = patch_file(target)
    print(f'status = {status}')
    print(f'target = {target}')
    if backup_path is not None:
        print(f'backup = {backup_path}')
    print('usage = SPU_ENABLE_MULTI_RANK_PROFILE=1 ...')


if __name__ == '__main__':
    main()
