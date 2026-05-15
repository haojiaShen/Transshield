#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def resolve_threshold_payload(bundle_dir: Path):
    threshold_path = bundle_dir / 'threshold_best.json'
    if threshold_path.exists() and not threshold_path.is_symlink():
        return load_json(threshold_path)
    if threshold_path.exists():
        try:
            return load_json(threshold_path.resolve())
        except FileNotFoundError:
            pass
    manifest = load_json(bundle_dir / 'manifest.json')
    threshold_metrics = ((manifest.get('primary') or {}).get('threshold_metrics') or {})
    if not threshold_metrics:
        raise FileNotFoundError(f'Cannot resolve threshold_best.json from {bundle_dir}')
    return threshold_metrics


def build_eval_checkpoint_light(state_dict_path: Path, args_snapshot_path: Path, output_path: Path, source_bundle_dir: Path):
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
    args_snapshot = load_json(args_snapshot_path)
    payload = {
        'args': args_snapshot,
        'model': state_dict,
        'source_bundle_dir': str(source_bundle_dir),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def write_readme(path: Path, source_bundle_dir: Path, promotion_manifest: dict):
    metrics = promotion_manifest.get('verified_metrics', {})
    lines = [
        f'# {promotion_manifest["bundle_name"]}',
        '',
        'Safe promotion-ready bundle generated from a verified candidate without overwriting the current official bundle.',
        '',
        '## Source',
        f'- source candidate bundle: `{source_bundle_dir}`',
        f'- source manifest: `source_manifest.json`',
        '',
        '## Bundled assets',
        '- `modified_plaintext_model_state_dict.pth`',
        '- `modified_plaintext_eval_checkpoint_light.pth`',
        '- `threshold_best.json`',
        '- `args_snapshot.json`',
        '- `promotion_manifest.json`',
        '',
        '## Verified metrics',
        f'- plaintext / secure argmax accuracy: `{metrics.get("argmax_accuracy")}`',
        f'- plaintext / secure threshold accuracy: `{metrics.get("threshold_accuracy")}`',
        f'- argmax match ratio: `{metrics.get("argmax_match_ratio")}`',
        f'- threshold match ratio: `{metrics.get("threshold_match_ratio")}`',
        f'- communication status: `{metrics.get("communication_status")}`',
        '',
        '## Note',
        '- This bundle is promotion-ready and self-contained for repo-side evaluation.',
        '- It does not overwrite `artifacts/frozen_bundle_full/`; promotion remains an explicit human decision.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Prepare a safe, non-symlink verified bundle from a candidate bundle.')
    parser.add_argument('--source-bundle-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--spu-summary-json', default='')
    parser.add_argument('--plaintext-summary-json', default='')
    args = parser.parse_args()

    source_bundle_dir = Path(args.source_bundle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_bundle_dir / 'manifest.json'
    state_dict_path = source_bundle_dir / 'modified_plaintext_model_state_dict.pth'
    args_snapshot_path = source_bundle_dir / 'args_snapshot.json'
    commands_path = source_bundle_dir / 'commands.sh'

    threshold_payload = resolve_threshold_payload(source_bundle_dir)
    source_manifest = load_json(source_manifest_path)
    plaintext_summary = load_json(Path(args.plaintext_summary_json).resolve()) if args.plaintext_summary_json else {}
    spu_summary = load_json(Path(args.spu_summary_json).resolve()) if args.spu_summary_json else {}

    copy_file(state_dict_path, output_dir / 'modified_plaintext_model_state_dict.pth')
    copy_file(args_snapshot_path, output_dir / 'args_snapshot.json')
    if commands_path.exists():
        copy_file(commands_path, output_dir / 'source_commands.sh')
    copy_file(source_manifest_path, output_dir / 'source_manifest.json')
    write_json(output_dir / 'threshold_best.json', threshold_payload)
    build_eval_checkpoint_light(
        state_dict_path=state_dict_path,
        args_snapshot_path=args_snapshot_path,
        output_path=output_dir / 'modified_plaintext_eval_checkpoint_light.pth',
        source_bundle_dir=source_bundle_dir,
    )

    verified_metrics = {
        'argmax_accuracy': spu_summary.get('plaintext_argmax_accuracy'),
        'threshold_accuracy': spu_summary.get('plaintext_threshold_accuracy'),
        'argmax_match_ratio': spu_summary.get('argmax_match_ratio'),
        'threshold_match_ratio': spu_summary.get('threshold_match_ratio'),
        'communication_status': spu_summary.get('communication_status'),
        'spu_pipeline_overall_passed': spu_summary.get('spu_pipeline_overall_passed'),
        'spu_replay_overall_passed': spu_summary.get('spu_replay_overall_passed'),
        'best_epoch': (plaintext_summary.get('best_by_acc1') or {}).get('epoch'),
    }
    promotion_manifest = {
        'bundle_name': output_dir.name,
        'source_bundle_dir': str(source_bundle_dir),
        'source_manifest_path': str(source_manifest_path),
        'status': 'promotion_ready_verified_bundle',
        'verified_metrics': verified_metrics,
        'notes': [
            'Generated without modifying artifacts/frozen_bundle_full.',
            'threshold_best.json is materialized as a regular file to avoid broken symlink issues on remote hosts.',
            'modified_plaintext_eval_checkpoint_light.pth is rebuilt from args_snapshot + model_state_dict for repo-friendly evaluation.',
        ],
    }
    write_json(output_dir / 'promotion_manifest.json', promotion_manifest)
    write_readme(output_dir / 'README.md', source_bundle_dir, promotion_manifest)

    print(json.dumps({
        'output_dir': str(output_dir),
        'status': promotion_manifest['status'],
        'argmax_accuracy': verified_metrics['argmax_accuracy'],
        'threshold_accuracy': verified_metrics['threshold_accuracy'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
