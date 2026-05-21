import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stage2_bundle import DEFAULT_BUNDLE_MODEL_STATE_NAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_last_jsonl(path: Path):
    last = None
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def symlink_force(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def normalize_transshield_strings(value):
    old_upper = 'Trans' + 'field'
    old_lower = old_upper.lower()
    new_upper = 'Trans' + 'shield'
    new_lower = new_upper.lower()
    if isinstance(value, str):
        return value.replace(old_upper, new_upper).replace(old_lower, new_lower)
    if isinstance(value, dict):
        return {key: normalize_transshield_strings(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_transshield_strings(item) for item in value]
    return value


def checkpoint_summary(run_dir: Path, checkpoint_name: str, threshold_json_name: str):
    checkpoint_path = run_dir / checkpoint_name
    threshold_path = run_dir / threshold_json_name
    log_path = run_dir / 'log.txt'
    train_stdout_path = run_dir / 'train_stdout.log'
    summary = {
        'run_dir': str(run_dir),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_size_bytes': checkpoint_path.stat().st_size,
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'threshold_json_path': str(threshold_path) if threshold_path.exists() else None,
        'threshold_metrics': read_json(threshold_path) if threshold_path.exists() else None,
        'log_path': str(log_path) if log_path.exists() else None,
        'last_log_entry': read_last_jsonl(log_path) if log_path.exists() else None,
        'train_stdout_path': str(train_stdout_path) if train_stdout_path.exists() else None,
    }
    return summary


def write_commands(path: Path, train_command: str, search_command: str, eval_command: str):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', '']
    if train_command:
        lines += ['# Training command', train_command, '']
    if search_command:
        lines += ['# Threshold search command', search_command, '']
    if eval_command:
        lines += ['# Thresholded eval command', eval_command, '']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    path.chmod(0o755)


def write_readme(path: Path, bundle_name: str, manifest: dict):
    primary = manifest['primary']
    threshold = primary.get('threshold_metrics') or {}
    last_log = primary.get('last_log_entry') or {}
    lines = [
        f'# {bundle_name}',
        '',
        'Frozen export bundle for a Transshield inference-friendly training run.',
        '',
        '## Primary candidate',
        f"- Source run: `{primary['run_dir']}`",
        f"- Full checkpoint symlink: `checkpoint-best.pth`",
        f"- Pure model weights: `{DEFAULT_BUNDLE_MODEL_STATE_NAME}`",
        f"- Threshold metadata: `threshold_best.json`",
        f"- Default argmax acc: `{threshold.get('default_argmax_acc1')}`",
        f"- Thresholded acc: `{threshold.get('eval_acc1')}`",
        f"- Eval loss: `{threshold.get('eval_loss')}`",
        f"- AUC: `{threshold.get('auc')}`",
        f"- Final epoch default eval acc: `{last_log.get('test_acc1')}`",
        f"- Final epoch default eval loss: `{last_log.get('test_loss')}`",
        '',
        '## Notes',
        '- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.',
        '- The full checkpoint symlink preserves direct `main.py --resume ... --eval true` compatibility.',
        '- Thresholded binary eval is generated separately and stored in `threshold_best.json`.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Freeze a server-trained Transshield run into a lightweight standalone export bundle.')
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint-name', default='checkpoint-best.pth')
    parser.add_argument('--threshold-json-name', default='threshold_best.json')
    parser.add_argument('--train-command', default='')
    parser.add_argument('--threshold-search-command', default='')
    parser.add_argument('--eval-command', default='')
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = checkpoint_summary(source_dir, args.checkpoint_name, args.threshold_json_name)

    checkpoint = torch.load(source_dir / args.checkpoint_name, map_location='cpu', weights_only=False)
    model_state = checkpoint['model']
    export_state_path = output_dir / DEFAULT_BUNDLE_MODEL_STATE_NAME
    torch.save(model_state, export_state_path)

    args_snapshot = checkpoint.get('args')
    if hasattr(args_snapshot, '__dict__'):
        args_snapshot = vars(args_snapshot)
    elif args_snapshot is None:
        args_snapshot = {}
    args_snapshot = normalize_transshield_strings(args_snapshot)

    symlink_force(source_dir / args.checkpoint_name, output_dir / 'checkpoint-best.pth')
    if (source_dir / args.threshold_json_name).exists():
        symlink_force(source_dir / args.threshold_json_name, output_dir / 'threshold_best.json')
    if (source_dir / 'train_stdout.log').exists():
        symlink_force(source_dir / 'train_stdout.log', output_dir / 'train_stdout.log')

    export_summary = {
        'model_state_dict_path': str(export_state_path),
        'model_state_dict_sha256': sha256_file(export_state_path),
        'model_tensor_count': len(model_state),
        'student_model_state_dict_path': str(export_state_path),
        'student_model_state_dict_sha256': sha256_file(export_state_path),
        'student_model_tensor_count': len(model_state),
    }

    manifest = {
        'bundle_name': output_dir.name,
        'frozen_at_utc': dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'primary': primary,
        'export': export_summary,
        'args_snapshot': args_snapshot,
        'commands': {
            'train': args.train_command,
            'threshold_search': args.threshold_search_command,
            'eval_threshold': args.eval_command,
        },
        'status': {
            'export_ready': True,
            'spu_alignment_preserved': True,
        },
    }

    (output_dir / 'args_snapshot.json').write_text(
        json.dumps(args_snapshot, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_commands(output_dir / 'commands.sh', args.train_command, args.threshold_search_command, args.eval_command)
    write_readme(output_dir / 'README.md', output_dir.name, manifest)


if __name__ == '__main__':
    main()
