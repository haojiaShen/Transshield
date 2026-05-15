import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_SPECS = [
    ('run_train.sh', 'train', True, True),
    ('run_freeze_export.sh', 'freeze_export', False, False),
    ('run_verify_bundle.sh', 'verify_bundle', False, False),
    ('run_secure_export_inputs.sh', 'secure_export_inputs', False, True),
    ('run_secure_replay.sh', 'secure_replay', False, False),
    ('run_secure_score_compare.sh', 'secure_compare', False, False),
    ('run_plaintext_model_compare.sh', 'plaintext_model_compare', False, False),
    ('run_secure_profile_summary.sh', 'secure_profile_summary', False, False),
    ('run_secure_profile_compare.sh', 'secure_profile_compare', False, False),
]


def write_text(path: Path, text: str, executable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if executable:
        path.chmod(0o755)


def write_json(path: Path, payload):
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def shell_quote(value: str):
    return '"' + value.replace('"', '\\"') + '"'


def shell_bool(value) -> str:
    return str(bool(value)).lower()


def join_shell_parts(*parts: str) -> str:
    return ' '.join(str(part) for part in parts if part)


def join_shell_lines(*lines: str) -> str:
    return '\n'.join(line for line in lines if line)


def build_train_command(args):
    return join_shell_parts(
        '"$PYTHON_BIN" main.py',
        f'--model {args.model}',
        '--data_set image_folder',
        '--data_path "$TRAIN_DATA_PATH"',
        '--eval_data_path "$VAL_DATA_PATH"',
        f'--nb_classes {args.nb_classes}',
        '--output_dir "$RUN_DIR"',
        '--log_dir "$RUN_DIR/tb"',
        f'--input_size {args.input_size}',
        f'--batch_size {args.batch_size}',
        f'--epochs {args.epochs}',
        f'--num_workers {args.num_workers}',
        f'--base_rate {args.base_rate}',
        f'--ratio_weight {args.ratio_weight}',
        f'--lr {args.lr}',
        f'--warmup_epochs {args.warmup_epochs}',
        f'--warmup_steps {args.warmup_steps}',
        f'--clip_grad {args.clip_grad}',
        f'--device {args.device}',
        '--model_ema false',
        '--save_ckpt true',
        f'--save_ckpt_freq {args.save_ckpt_freq}',
        f'--save_ckpt_num {args.save_ckpt_num}',
        '--auto_resume false',
        f'--use_amp {shell_bool(args.use_amp)}',
        '--mixup 0',
        '--cutmix 0',
        f'--seed {args.seed}',
        f'--lr_scale {args.lr_scale}',
        f'--groupa_lr_scale {args.groupa_lr_scale}',
        f'--activation_lr_scale {args.activation_lr_scale}',
        f'--cls_distill_weight {args.cls_distill_weight}',
        f'--token_distill_weight {args.token_distill_weight}',
        f'--square_activation_mode {args.square_activation_mode}',
        f'--approx_attn_mode {args.approx_attn_mode}',
        f'--eval_tie_policy {args.eval_tie_policy}',
        f'--patch_embed_bias_init_mode {args.patch_embed_bias_init_mode}',
        f'--freeze_patch_embed_proj {shell_bool(args.freeze_patch_embed_proj)}',
        f'--pretrained_fix_step {args.pretrained_fix_step}',
        '--inference_friendly_ops true',
    )


def build_threshold_command(args, mode: str):
    base_parts = [
        f'"$PYTHON_BIN" tools/transshield_binary_threshold_search.py {mode}',
        '--checkpoint "$RUN_DIR/checkpoint-best.pth"',
    ]
    if mode == 'eval':
        base_parts.append('--threshold-json "$RUN_DIR/threshold_best.json"')
    base_parts.extend(
        [
            '--data-path "$VAL_DATA_PATH"',
            f'--device {args.device}',
            f'--batch-size {args.batch_size}',
            f'--num-workers {args.num_workers}',
            f'--output-json "$RUN_DIR/threshold_{"best" if mode == "search" else "eval"}.json"',
        ]
    )
    return join_shell_parts(*base_parts)


def build_freeze_export_command(train_command: str, threshold_search_command: str, threshold_eval_command: str):
    return join_shell_parts(
        '"$PYTHON_BIN" tools/freeze_export_candidate.py',
        '--source-dir "$RUN_DIR"',
        '--output-dir "$BUNDLE_DIR"',
        f'--train-command {shell_quote(train_command)}',
        f'--threshold-search-command {shell_quote(threshold_search_command)}',
        f'--eval-command {shell_quote(threshold_eval_command)}',
    )


def build_verify_bundle_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/verify_frozen_candidate.py',
        '--bundle-dir "$BUNDLE_DIR"',
        '--device cpu',
    )


def build_secure_export_inputs_command(args):
    secure_export_lines = [
        'mkdir -p "$SECURE_RUN_DIR"',
        join_shell_parts(
            '"$PYTHON_BIN" tools/transshield_secure_sidecar_export_suite.py',
            '--bundle-dir "$BUNDLE_DIR"',
            '--data-path "$VAL_DATA_PATH"',
            f'--device {args.device}',
            f'--batch-size {args.batch_size}',
            f'--num-workers {args.secure_num_workers}',
            f'--max-samples {args.secure_max_samples}',
            '--input-output-pt "$SECURE_RUN_DIR/stage2_secure_network_kth_input_smoke8.pt"',
            '--input-output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_input_smoke8.json"',
            '--kth-output-pt "$SECURE_RUN_DIR/stage2_secure_network_kth_reference_smoke8.pt"',
            '--kth-output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_reference_smoke8.json"',
            '--tie-output-pt "$SECURE_RUN_DIR/stage2_secure_tie_policy_lowest_smoke8.pt"',
            '--tie-output-json "$SECURE_RUN_DIR/stage2_secure_tie_policy_lowest_smoke8.json"',
        ),
        join_shell_parts(
            '"$PYTHON_BIN" tools/transshield_secure_network_kth.py manifest',
            '--bundle-dir "$BUNDLE_DIR"',
            '--output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_manifest.json"',
        ),
    ]
    return join_shell_lines(*secure_export_lines)


def build_pipeline_run_command(runtime: str, include_config: bool = False):
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py run',
        f'--runtime {runtime}',
        '--bundle-dir "$BUNDLE_DIR"',
        '--config "$CONFIG_PATH"' if include_config else '',
        '--output-dir "$SECURE_RUN_DIR"',
    )


def build_pipeline_verify_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py verify',
        '--output-dir "$SECURE_RUN_DIR"',
    )


def build_secure_pipeline_command(runtime: str):
    lines = []
    if runtime == 'spu':
        lines.append(
            join_shell_parts(
                '"$PYTHON_BIN" tools/transshield_spu_runtime_setup.py start',
                '--config "$CONFIG_PATH"',
                '--template configs/openbumblebee/2pc.template.json',
                '--backup',
                '--restart',
                '--remove-unsupported-cheetah-fields',
                '--log-dir logs/spu_nodes',
                '--state-json logs/spu_runtime_ports.json',
            )
        )
    lines.append(build_pipeline_run_command(runtime, include_config=(runtime == 'spu')))
    lines.append(build_pipeline_verify_command())
    return join_shell_lines(*lines)


def build_secure_replay_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py replay',
        '--output-dir "$SECURE_RUN_DIR"',
        '--bundle-dir "$BUNDLE_DIR"',
        '--device cpu',
        '--enable-model-replay',
    )


def build_secure_compare_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_plaintext_secure_score_compare.py',
        '--bundle-dir "$BUNDLE_DIR"',
        '--secure-replay-json "$SECURE_RUN_DIR/pipeline_inference_replay_summary.json"',
        '--device cpu',
        '--batch-size 16',
        '--num-workers 0',
        '--output-json "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.json"',
        '--output-csv "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.csv"',
    )


def build_plaintext_eval_modified_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py',
        '--bundle-dir "$BUNDLE_DIR"',
        '--data-path "$VAL_DATA_PATH"',
        '--device "$PLAINTEXT_EVAL_DEVICE"',
        '--batch-size "$PLAINTEXT_EVAL_BATCH_SIZE"',
        '--num-workers "$PLAINTEXT_EVAL_NUM_WORKERS"',
        '--threshold-json "$BUNDLE_DIR/threshold_best.json"',
        '--label modified_plaintext',
        '--output-json "$SECURE_RUN_DIR/plaintext_modified_eval.json"',
        '--output-csv "$SECURE_RUN_DIR/plaintext_modified_eval.csv"',
    )


def build_plaintext_eval_baseline_command():
    return join_shell_lines(
        'if [[ -z "$BASELINE_REPO_ROOT" || -z "$BASELINE_CHECKPOINT" ]]; then',
        '  echo "Please set BASELINE_REPO_ROOT and BASELINE_CHECKPOINT before running." >&2',
        '  exit 1',
        'fi',
        'THRESHOLD_ARGS=()',
        'if [[ -n "$BASELINE_THRESHOLD_JSON" ]]; then',
        '  THRESHOLD_ARGS+=(--threshold-json "$BASELINE_THRESHOLD_JSON")',
        'fi',
        join_shell_parts(
            '"$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py',
            '--repo-root "$BASELINE_REPO_ROOT"',
            '--checkpoint "$BASELINE_CHECKPOINT"',
            '--data-path "$VAL_DATA_PATH"',
            '--device "$PLAINTEXT_EVAL_DEVICE"',
            '--batch-size "$PLAINTEXT_EVAL_BATCH_SIZE"',
            '--num-workers "$PLAINTEXT_EVAL_NUM_WORKERS"',
            '--label "$BASELINE_LABEL"',
            '--output-json "$SECURE_RUN_DIR/plaintext_baseline_eval.json"',
            '--output-csv "$SECURE_RUN_DIR/plaintext_baseline_eval.csv"',
            '${THRESHOLD_ARGS[@]}',
        ),
    )


def build_plaintext_model_compare_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_plaintext_eval_compare.py',
        '--eval-a "$SECURE_RUN_DIR/plaintext_baseline_eval.json"',
        '--eval-b "$SECURE_RUN_DIR/plaintext_modified_eval.json"',
        '--label-a "$BASELINE_LABEL"',
        '--label-b modified_plaintext',
        '--output-json "$SECURE_RUN_DIR/plaintext_model_compare.json"',
    )


def build_secure_profile_summary_command():
    return join_shell_parts(
        '"$PYTHON_BIN" tools/transshield_secure_profile_summary.py',
        '--secure-run-dir "$SECURE_RUN_DIR"',
        '--spu-state-json logs/spu_runtime_ports.json',
        '--spu-log-dir logs/spu_nodes',
        '--output-json "$SECURE_RUN_DIR/secure_profile_summary.json"',
    )


def build_secure_profile_compare_command():
    return join_shell_lines(
        'if [[ -z "$SECURE_BASELINE_PROFILE_JSON" ]]; then',
        '  echo "Please set SECURE_BASELINE_PROFILE_JSON before running." >&2',
        '  exit 1',
        'fi',
        join_shell_parts(
            '"$PYTHON_BIN" tools/transshield_secure_profile_compare.py',
            '--summary-a "$SECURE_BASELINE_PROFILE_JSON"',
            '--summary-b "$SECURE_RUN_DIR/secure_profile_summary.json"',
            '--label-a "$SECURE_BASELINE_LABEL"',
            '--label-b transshield_secure',
            '--output-json "$SECURE_RUN_DIR/secure_profile_compare.json"',
        ),
    )


def build_commands(args):
    train_command = build_train_command(args)
    threshold_search_command = build_threshold_command(args, 'search')
    threshold_eval_command = build_threshold_command(args, 'eval')
    run_dir = Path(args.run_root) / args.run_name
    bundle_dir = Path(args.bundle_root) / f'{args.run_name}_bundle'

    return {
        'run_dir': str(run_dir),
        'bundle_dir': str(bundle_dir),
        'train': train_command,
        'threshold_search': threshold_search_command,
        'threshold_eval': threshold_eval_command,
        'freeze_export': build_freeze_export_command(
            train_command,
            threshold_search_command,
            threshold_eval_command,
        ),
        'verify_bundle': build_verify_bundle_command(),
        'secure_export_inputs': build_secure_export_inputs_command(args),
        'secure_pipeline_cpu': build_secure_pipeline_command('cpu'),
        'secure_pipeline_spu': build_secure_pipeline_command('spu'),
        'secure_replay': build_secure_replay_command(),
        'secure_compare': build_secure_compare_command(),
        'plaintext_eval_modified': build_plaintext_eval_modified_command(),
        'plaintext_eval_baseline': build_plaintext_eval_baseline_command(),
        'plaintext_model_compare': build_plaintext_model_compare_command(),
        'secure_profile_summary': build_secure_profile_summary_command(),
        'secure_profile_compare': build_secure_profile_compare_command(),
    }


def build_script_preamble(run_name: str, require_train_data: bool = False, require_val_data: bool = False):
    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"',
        'cd "$REPO_ROOT"',
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        f'RUN_NAME="${{RUN_NAME:-{run_name}}}"',
        'TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"',
        'VAL_DATA_PATH="${VAL_DATA_PATH:-}"',
        'RUN_DIR="${RUN_DIR:-artifacts/server_runs/${RUN_NAME}}"',
        'BUNDLE_DIR="${BUNDLE_DIR:-artifacts/server_bundles/${RUN_NAME}_bundle}"',
        'SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"',
        'CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"',
        'BASELINE_REPO_ROOT="${BASELINE_REPO_ROOT:-}"',
        'BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"',
        'BASELINE_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-}"',
        'BASELINE_LABEL="${BASELINE_LABEL:-baseline_plaintext}"',
        'PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"',
        'PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"',
        'PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"',
        'SECURE_BASELINE_PROFILE_JSON="${SECURE_BASELINE_PROFILE_JSON:-}"',
        'SECURE_BASELINE_LABEL="${SECURE_BASELINE_LABEL:-baseline_secure}"',
    ]
    if require_train_data:
        lines.extend(
            [
                'if [[ -z "$TRAIN_DATA_PATH" ]]; then',
                '  echo "Please set TRAIN_DATA_PATH before running." >&2',
                '  exit 1',
                'fi',
            ]
        )
    if require_val_data:
        lines.extend(
            [
                'if [[ -z "$VAL_DATA_PATH" ]]; then',
                '  echo "Please set VAL_DATA_PATH before running." >&2',
                '  exit 1',
                'fi',
            ]
        )
    return lines


def build_script_body(command: str, run_name: str, require_train_data: bool = False, require_val_data: bool = False):
    lines = build_script_preamble(
        run_name,
        require_train_data=require_train_data,
        require_val_data=require_val_data,
    )
    lines.append(command)
    return join_shell_lines(*lines) + '\n'


def build_dispatch_script(
    run_name: str,
    dispatch_name: str,
    usage: str,
    cases,
    require_train_data: bool = False,
    require_val_data: bool = False,
):
    lines = build_script_preamble(
        run_name,
        require_train_data=require_train_data,
        require_val_data=require_val_data,
    )
    lines.append(f'{dispatch_name}="${{1:-}}"')
    lines.append(f'case "${dispatch_name}" in')
    for label, command in cases:
      lines.append(f'  {label})')
      for command_line in command.splitlines():
          lines.append(f'    {command_line}')
      lines.append('    ;;')
    lines.extend(
        [
            '  *)',
            f'    echo "Usage: $0 {usage}" >&2',
            '    exit 1',
            '    ;;',
            'esac',
        ]
    )
    return join_shell_lines(*lines) + '\n'


def build_shortcut_script(*script_names: str):
    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
    ]
    lines.extend(f'"$SCRIPT_DIR/{script_name}"' for script_name in script_names)
    return join_shell_lines(*lines) + '\n'


def build_full_final_comparison_suite_script():
    return join_shell_lines(
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'SECURE_RUNTIME="${SECURE_RUNTIME:-spu}"',
        '"$SCRIPT_DIR/run_full_transshield_pipeline.sh" "$SECURE_RUNTIME"',
        'if [[ -n "${BASELINE_REPO_ROOT:-}" && -n "${BASELINE_CHECKPOINT:-}" ]]; then',
        '  "$SCRIPT_DIR/run_plaintext_eval.sh" baseline',
        '  "$SCRIPT_DIR/run_plaintext_model_compare.sh"',
        'else',
        '  echo "[skip] plaintext baseline compare: set BASELINE_REPO_ROOT and BASELINE_CHECKPOINT"',
        'fi',
        'if [[ -n "${SECURE_BASELINE_PROFILE_JSON:-}" ]]; then',
        '  "$SCRIPT_DIR/run_secure_profile_compare.sh"',
        'else',
        '  echo "[skip] secure profile compare: set SECURE_BASELINE_PROFILE_JSON"',
        'fi',
    ) + '\n'


def build_final_env_template():
    return join_shell_lines(
        '# Server environment template for the final Transshield comparison suite.',
        '# Fill in the placeholder paths before running.',
        '',
        'source /home/wyb/miniconda3/etc/profile.d/conda.sh',
        'conda activate /data/wyb/conda_envs/transshield',
        'unset PYTHONPATH',
        'export PYTHONNOUSERSITE=1',
        'export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python',
        '',
        'export TRANSHIELD_TMP_ROOT="${TRANSHIELD_TMP_ROOT:-/data/wyb/bazel_clean/tmp}"',
        'mkdir -p "$TRANSHIELD_TMP_ROOT"',
        'export TMPDIR="$TRANSHIELD_TMP_ROOT"',
        'export TEMP="$TRANSHIELD_TMP_ROOT"',
        'export TMP="$TRANSHIELD_TMP_ROOT"',
        'export TEST_TMPDIR="$TRANSHIELD_TMP_ROOT"',
        '',
        'cd /home/yclcg/Transshield_final',
        '',
        'export RUN_NAME=transshield_comp_full_compare_YYYYMMDD',
        'export TRAIN_DATA_PATH=$DATA_ROOT/train',
        'export VAL_DATA_PATH=$DATA_ROOT/val',
        'export RUN_DIR=/home/yclcg/Transshield_final/artifacts/server_runs/${RUN_NAME}',
        'export BUNDLE_DIR=/home/yclcg/Transshield_final/artifacts/server_bundles/${RUN_NAME}_bundle',
        'export SECURE_RUN_DIR=/home/yclcg/Transshield_final/artifacts/server_pipeline_run/${RUN_NAME}',
        'export CONFIG_PATH=/home/yclcg/Transshield_final/configs/openbumblebee/2pc.json',
        'export SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-0}"',
        'export PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"',
        'export PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"',
        'export PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"',
        '',
        '# Group 1 baseline plaintext',
        'export BASELINE_REPO_ROOT=/data/wyb/DynamicViT_baseline',
        'export BASELINE_CHECKPOINT=/data/wyb/DynamicViT_baseline/path/to/checkpoint-best.pth',
        'export BASELINE_THRESHOLD_JSON=',
        'export BASELINE_LABEL=original_plaintext',
        '',
        '# Group 3 original secure baseline profile summary',
        'export SECURE_BASELINE_PROFILE_JSON=',
        'export SECURE_BASELINE_LABEL=original_secure',
        '',
    )


def build_pack_readme():
    return join_shell_lines(
        '# Transshield inference-friendly server pack',
        '',
        'ViT-only run order:',
        '1. `run_train.sh`',
        '2. `run_threshold.sh search`',
        '3. `run_threshold.sh eval`',
        '4. `run_freeze_export.sh`',
        '5. `run_verify_bundle.sh`',
        '',
        'Full secure run order:',
        '6. `run_secure_export_inputs.sh`',
        '7. `run_secure_pipeline.sh cpu|spu`',
        '8. `run_secure_replay.sh`',
        '9. `run_secure_score_compare.sh`',
        '10. `run_secure_profile_summary.sh`',
        '',
        'Plaintext comparison helpers:',
        '- `run_plaintext_eval.sh baseline|modified`',
        '- `run_plaintext_model_compare.sh`',
        '',
        'Secure profile helpers:',
        '- `run_secure_profile_summary.sh`',
        '- `run_secure_profile_compare.sh`',
        '',
        '`run_secure_pipeline.sh spu` automatically rewrites `configs/openbumblebee/2pc.json`',
        'with free localhost ports, removes SPU-version-incompatible Cheetah fields,',
        'restarts each colocated SPU node separately, warms up the SPU runtime, and',
        'records the selected ports and node PIDs in `logs/spu_runtime_ports.json`.',
        '',
        'Shortcuts:',
        '- `run_full_vit_pipeline.sh`',
        '- `run_full_transshield_pipeline.sh cpu|spu`',
        '- `run_full_final_comparison_suite.sh`',
        '',
        'Environment template:',
        '- `final_compare_env.template.sh`',
        '',
        'Default SPU config path: `configs/openbumblebee/2pc.json`',
        '',
    )


def build_manifest(output_dir: Path, commands):
    return {
        'repo_root': str(REPO_ROOT),
        'pack_dir': str(output_dir),
        'purpose': 'server-ready pack for inference-friendly ViT training, threshold calibration, and frozen export',
        'recommended_recipe': str(
            (REPO_ROOT / 'artifacts' / 'inference_ready_config' / 'transshield_inference_friendly_ops_recipe.json').resolve()
        ),
        'commands': commands,
        'constraints': {
            'does_not_run_bumblebee': True,
            'does_not_run_spu': True,
            'focuses_on_vit_side_completion_before_secure_integration': True,
        },
    }


def write_primary_scripts(output_dir: Path, run_name: str, commands):
    for script_name, command_key, require_train_data, require_val_data in SCRIPT_SPECS:
        write_text(
            output_dir / script_name,
            build_script_body(
                commands[command_key],
                run_name,
                require_train_data=require_train_data,
                require_val_data=require_val_data,
            ),
            executable=True,
        )

    dispatch_scripts = {
        'run_threshold.sh': build_dispatch_script(
            run_name,
            'THRESHOLD_MODE',
            '[search|eval]',
            [
                ('search', commands['threshold_search']),
                ('eval', commands['threshold_eval']),
            ],
            require_val_data=True,
        ),
        'run_plaintext_eval.sh': build_dispatch_script(
            run_name,
            'PLAINTEXT_VARIANT',
            '[baseline|modified]',
            [
                ('baseline', commands['plaintext_eval_baseline']),
                ('modified', commands['plaintext_eval_modified']),
            ],
            require_val_data=True,
        ),
        'run_secure_pipeline.sh': build_dispatch_script(
            run_name,
            'SECURE_RUNTIME',
            '[cpu|spu]',
            [
                ('cpu', commands['secure_pipeline_cpu']),
                ('spu', commands['secure_pipeline_spu']),
            ],
        ),
    }
    for script_name, content in dispatch_scripts.items():
        write_text(output_dir / script_name, content, executable=True)


def write_shortcut_scripts(output_dir: Path):
    shortcut_scripts = {
        'run_full_vit_pipeline.sh': join_shell_lines(
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
            '"$SCRIPT_DIR/run_train.sh"',
            '"$SCRIPT_DIR/run_threshold.sh" search',
            '"$SCRIPT_DIR/run_threshold.sh" eval',
            '"$SCRIPT_DIR/run_freeze_export.sh"',
            '"$SCRIPT_DIR/run_verify_bundle.sh"',
        ),
        'run_full_transshield_pipeline.sh': join_shell_lines(
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
            'SECURE_RUNTIME="${1:-cpu}"',
            'case "$SECURE_RUNTIME" in',
            '  cpu|spu) ;;',
            '  *)',
            '    echo "Usage: $0 [cpu|spu]" >&2',
            '    exit 1',
            '    ;;',
            'esac',
            '"$SCRIPT_DIR/run_full_vit_pipeline.sh"',
            '"$SCRIPT_DIR/run_plaintext_eval.sh" modified',
            '"$SCRIPT_DIR/run_secure_export_inputs.sh"',
            '"$SCRIPT_DIR/run_secure_pipeline.sh" "$SECURE_RUNTIME"',
            '"$SCRIPT_DIR/run_secure_replay.sh"',
            '"$SCRIPT_DIR/run_secure_score_compare.sh"',
            '"$SCRIPT_DIR/run_secure_profile_summary.sh"',
        ),
        'run_full_final_comparison_suite.sh': build_full_final_comparison_suite_script(),
    }
    for script_name, content in shortcut_scripts.items():
        write_text(output_dir / script_name, content, executable=True)


def write_pack(args, commands):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_primary_scripts(output_dir, args.run_name, commands)
    write_shortcut_scripts(output_dir)
    write_text(output_dir / 'final_compare_env.template.sh', build_final_env_template())

    manifest = build_manifest(output_dir, commands)
    write_json(output_dir / 'commands.json', manifest)
    write_text(output_dir / 'README.md', build_pack_readme())
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Prepare a server-ready Transshield pack for inference-friendly ViT training and export.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-data-path', required=True)
    parser.add_argument('--val-data-path', required=True)
    parser.add_argument('--run-name', default='transshield_inference_friendly_deits')
    parser.add_argument('--run-root', default='artifacts/server_runs')
    parser.add_argument('--bundle-root', default='artifacts/server_bundles')
    parser.add_argument('--model', default='deit-s')
    parser.add_argument('--nb-classes', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--secure-num-workers', type=int, default=0)
    parser.add_argument('--secure-max-samples', type=int, default=8)
    parser.add_argument('--base-rate', type=float, default=0.7)
    parser.add_argument('--ratio-weight', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-ckpt-freq', type=int, default=1)
    parser.add_argument('--save-ckpt-num', type=int, default=2)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr-scale', type=float, default=1.0)
    parser.add_argument('--groupa-lr-scale', type=float, default=0.1)
    parser.add_argument('--activation-lr-scale', type=float, default=10.0)
    parser.add_argument('--cls-distill-weight', type=float, default=1.0)
    parser.add_argument('--token-distill-weight', type=float, default=0.02)
    parser.add_argument('--square-activation-mode', default='learnable_quadratic_gelu_init')
    parser.add_argument('--approx-attn-mode', default='relu')
    parser.add_argument('--eval-tie-policy', default='lowest_index')
    parser.add_argument('--patch-embed-bias-init-mode', default='zero')
    parser.add_argument('--freeze-patch-embed-proj', action='store_true')
    parser.add_argument('--pretrained-fix-step', type=int, default=0)
    args = parser.parse_args()
    if not args.freeze_patch_embed_proj:
        args.freeze_patch_embed_proj = True

    commands = build_commands(args)
    manifest = write_pack(args, commands)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
