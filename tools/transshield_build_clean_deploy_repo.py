#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_prepare_verified_bundle import build_eval_checkpoint_light
from tools.transshield_stage2_bundle import resolve_threshold_payload


IGNORE_NAMES = {
    '.git',
    '__pycache__',
    '.DS_Store',
    'tmp_acceptance_check',
}

FULL_COPY_DIRS = [
    'configs',
    'docs',
    'integrations',
    'licenses',
    'models',
    'references',
    'scripts',
    'tools',
    'training_compat',
    'training_source_tracka',
    'web_demo',
]

ARTIFACT_COPY_DIRS = [
    'artifacts/baselines',
    'artifacts/inference_ready_config',
    'artifacts/server_inference_friendly_pack',
    'artifacts/web_demo_assets',
]

BUNDLE_DIRS = [
    'artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430',
]

RESULT_FILES = [
    'results/README.md',
    'results/blockwise_exact_kth_selection_manifest_default.json',
    'results/blockwise_exact_kth_manifest_20260418_004103.json',
    'results/blockwise_exact_kth_manifest_20260418_004103.md',
]

RESULT_DIRS = [
    'results/delivery_acceptance/delivery_acceptance_20260505_clean',
    'results/fair_external_comparison/fair_external_secure_static_20260505_clean',
    'results/stage_cost_risk_model/stage_cost_risk_20260505_clean',
    'results/secure_static_train_depth_evidence/secure_static_train_depth_20260505_clean',
    'results/protocol_aware_pruning_objective/protocol_aware_recipe_20260505_clean',
]

RUNTIME_STATIC_FILES = [
    'artifacts/server_pipeline_run/e2e_output_calibration_secret_depth6_clip0_balanced8_20260502.json',
    'artifacts/server_pipeline_run/secret_depth6_clip0_balanced8_offline_eval_20260502.json',
    'artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean/secret_isolated_eval_summary.json',
    'artifacts/server_pipeline_run/secure_static_depth12_epoch8_secret_depth_boundary_calib_clip0_20260430/e2e_secure_poc/e2e_public_layer_norm_calibration_depth6_uniform_fixed_square_clip0.json',
    'artifacts/server_pipeline_run/e2e_output_calibration_secure_static_depth12_epoch8_clip3_balanced8_20260430.json',
    'artifacts/server_pipeline_run/secure_static_depth12_epoch8_publicraw_balanced8_clip3_20260430/e2e_secure_poc/e2e_public_layer_norm_calibration_depth12_uniform_fixed_square_clip3p0.json',
]

RUNTIME_STATIC_DIRS = [
    'artifacts/server_pipeline_run/delivery_line_suite_20260505_clean',
]

EMPTY_DIRS = [
    'artifacts/server_pipeline_run',
    'results/delivery_acceptance',
    'results/fair_external_comparison',
    'logs/spu_nodes',
]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        symlinks=False,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.DS_Store'),
    )


def copy_root_files(source_root: Path, output_root: Path):
    copied = []
    for item in sorted(source_root.iterdir(), key=lambda path: path.name):
        if item.name in IGNORE_NAMES:
            continue
        if item.is_dir():
            continue
        if item.name.endswith(('.pyc', '.pyo')):
            continue
        copy_file(item, output_root / item.name)
        copied.append(item.name)
    return copied


def copy_optional_files(source_root: Path, output_root: Path, relative_paths):
    copied = []
    missing = []
    for relative in relative_paths:
        src = source_root / relative
        if src.is_file():
            copy_file(src, output_root / relative)
            copied.append(relative)
        else:
            missing.append(relative)
    return copied, missing


def copy_optional_dirs(source_root: Path, output_root: Path, relative_paths):
    copied = []
    missing = []
    for relative in relative_paths:
        src = source_root / relative
        if src.is_dir():
            copy_tree(src, output_root / relative)
            copied.append(relative)
        else:
            missing.append(relative)
    return copied, missing


def materialize_bundle(source_bundle_dir: Path, output_bundle_dir: Path):
    output_bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_regular_files = []
    skipped_items = []
    for item in sorted(source_bundle_dir.iterdir(), key=lambda path: path.name):
        if item.is_dir():
            continue
        if item.name in {
            'threshold_best.json',
            'modified_plaintext_eval_checkpoint_light.pth',
            'checkpoint-best.pth',
            'train_stdout.log',
        }:
            skipped_items.append(item.name)
            continue
        if item.is_symlink() and not item.exists():
            skipped_items.append(item.name)
            continue
        copy_file(item.resolve() if item.is_symlink() else item, output_bundle_dir / item.name)
        copied_regular_files.append(item.name)

    threshold_payload = resolve_threshold_payload(source_bundle_dir)
    write_json(output_bundle_dir / 'threshold_best.json', threshold_payload)

    state_dict_path = output_bundle_dir / 'modified_plaintext_model_state_dict.pth'
    args_snapshot_path = output_bundle_dir / 'args_snapshot.json'
    if state_dict_path.exists() and args_snapshot_path.exists():
        build_eval_checkpoint_light(
            state_dict_path=state_dict_path,
            args_snapshot_path=args_snapshot_path,
            output_path=output_bundle_dir / 'modified_plaintext_eval_checkpoint_light.pth',
            source_bundle_dir=source_bundle_dir,
        )

    deploy_manifest = {
        'bundle_dir': str(output_bundle_dir),
        'source_bundle_dir': str(source_bundle_dir),
        'self_contained_threshold_json': True,
        'self_contained_light_checkpoint': (output_bundle_dir / 'modified_plaintext_eval_checkpoint_light.pth').exists(),
        'full_checkpoint_included': False,
        'copied_regular_files': copied_regular_files,
        'skipped_items': skipped_items,
    }
    write_json(output_bundle_dir / 'clean_deploy_manifest.json', deploy_manifest)
    return deploy_manifest


def build_clean_repo(source_root: Path, output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    root_files = copy_root_files(source_root, output_root)

    copied_dirs = []
    for relative in FULL_COPY_DIRS:
        copy_tree(source_root / relative, output_root / relative)
        copied_dirs.append(relative)

    for relative in ARTIFACT_COPY_DIRS:
        copy_tree(source_root / relative, output_root / relative)
        copied_dirs.append(relative)

    bundle_manifests = []
    for relative in BUNDLE_DIRS:
        bundle_manifests.append(
            materialize_bundle(
                source_bundle_dir=source_root / relative,
                output_bundle_dir=output_root / relative,
            )
        )

    copied_results = []
    for relative in RESULT_FILES:
        copy_file(source_root / relative, output_root / relative)
        copied_results.append(relative)

    copied_result_dirs = []
    for relative in RESULT_DIRS:
        copy_tree(source_root / relative, output_root / relative)
        copied_result_dirs.append(relative)

    for relative in EMPTY_DIRS:
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    copied_runtime_static_files, missing_runtime_static_files = copy_optional_files(
        source_root,
        output_root,
        RUNTIME_STATIC_FILES,
    )
    copied_runtime_static_dirs, missing_runtime_static_dirs = copy_optional_dirs(
        source_root,
        output_root,
        RUNTIME_STATIC_DIRS,
    )

    manifest = {
        'profile': 'server_runtime_clean_v1',
        'source_root': str(source_root),
        'output_root': str(output_root),
        'copied_root_files': root_files,
        'copied_directories': copied_dirs,
        'materialized_bundles': bundle_manifests,
        'copied_results': copied_results,
        'copied_result_directories': copied_result_dirs,
        'created_empty_directories': EMPTY_DIRS,
        'copied_runtime_static_files': copied_runtime_static_files,
        'missing_runtime_static_files': missing_runtime_static_files,
        'copied_runtime_static_directories': copied_runtime_static_dirs,
        'missing_runtime_static_directories': missing_runtime_static_dirs,
        'omitted_top_level_paths': [
            'artifacts/archive',
            'artifacts/frozen_bundle_full',
            'artifacts/frozen_candidates',
            'artifacts/frozen_bundle_verified_tracka_lr3e5_20260414',
            'artifacts/server_pipeline_run (historical contents except selected runtime static assets)',
            'artifacts/train_runs',
            'results/fair_external_comparison (historical contents except current official clean evidence)',
            'results/margin_aware_pruning_ablation',
            'results/standardized_secure_benchmark',
            'tmp_acceptance_check',
        ],
    }
    write_json(output_root / 'clean_deploy_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Build a clean self-contained Transshield server runtime repo for rsync --delete deployment.'
    )
    parser.add_argument('--source-root', default=str(REPO_ROOT))
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    manifest = build_clean_repo(source_root, output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
