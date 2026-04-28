import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_openbumblebee_bridge import prepare_bridge_pack as prepare_kth_pack
from tools.transshield_openbumblebee_bridge import DEFAULT_BUNDLE_DIR as DEFAULT_KTH_BUNDLE_DIR
from tools.transshield_openbumblebee_bridge import DEFAULT_BRIDGE_SCRIPT as DEFAULT_KTH_BRIDGE_SCRIPT
from tools.transshield_openbumblebee_bridge import DEFAULT_SPU_CONFIG as DEFAULT_SPU_CONFIG
from tools.transshield_openbumblebee_tie_bridge import prepare_bridge_pack as prepare_tie_pack
from tools.transshield_openbumblebee_tie_bridge import DEFAULT_BRIDGE_SCRIPT as DEFAULT_TIE_BRIDGE_SCRIPT
from tools.transshield_stage2_bundle import DEFAULT_BUNDLE_MODEL_STATE_NAME, resolve_model_state_dict_path

DEFAULT_THRESHOLD_TOLERANCE = 5e-5
DEFAULT_PIPELINE_SELECTION_MODE = 'blockwise_exact_kth'
DEFAULT_PHASE3_SELECTION_MANIFEST = REPO_ROOT / 'results' / 'blockwise_exact_kth_selection_manifest_default.json'
DEFAULT_PRESENTATION_BUNDLE_DIR = REPO_ROOT / 'artifacts' / 'frozen_bundle_verified_tracka_lr3e5_20260414'
RUNTIME_INPUT_FILES = [
    'stage2_secure_network_kth_manifest.json',
    'stage2_secure_network_kth_input_smoke8.pt',
    'stage2_secure_network_kth_input_smoke8.json',
    'stage2_secure_network_kth_reference_smoke8.pt',
    'stage2_secure_network_kth_reference_smoke8.json',
    'stage2_secure_tie_policy_lowest_smoke8.pt',
    'stage2_secure_tie_policy_lowest_smoke8.json',
]


def write_json(path: Path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def tail_text(path: Path, max_lines: int = 80):
    if not path.exists():
        return ''
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-max_lines:])


def bundle_supports_eval_replay(bundle_dir: Path):
    try:
        resolve_model_state_dict_path(bundle_dir)
        return True
    except FileNotFoundError:
        return False


def ensure_runtime_inputs(output_dir: Path, bundle_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_INPUT_FILES:
        dst = output_dir / name
        if dst.exists():
            continue
        src = bundle_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f'missing runtime input artifact: {src}; '
                f'either export it into {output_dir} first or place it under the bundle directory'
            )
        shutil.copy2(src, dst)


def build_step(name: str, command, outputs):
    return {'name': name, 'command': command, 'outputs': outputs}


def build_selection_mode_args(selection_mode: str, phase3_selection_manifest: str):
    extra_args = ['--selection-mode', selection_mode]
    if selection_mode != 'flat_odd_even':
        if not phase3_selection_manifest:
            raise ValueError('--phase3-selection-manifest is required when selection_mode is not flat_odd_even')
        extra_args.extend(['--phase3-selection-manifest', phase3_selection_manifest])
    return extra_args


def build_payload_args(payload_dtype: str, payload_stage_dtypes: str, payload_boundary_window: int):
    payload_args = ['--payload-dtype', payload_dtype]
    if payload_stage_dtypes:
        payload_args.extend(['--payload-stage-dtypes', payload_stage_dtypes])
    if int(payload_boundary_window) > 0:
        payload_args.extend(['--payload-boundary-window', str(int(payload_boundary_window))])
    return payload_args


def build_kth_bridge_command(
    output_dir: Path,
    runtime: str,
    config: str,
    selection_mode: str,
    phase3_selection_manifest: str,
    payload_dtype: str,
    payload_stage_dtypes: str,
    payload_boundary_window: int,
):
    command = [
        sys.executable,
        str(REPO_ROOT / 'integrations' / 'openbumblebee' / 'transshield_network_kth_bridge' / 'transshield_network_kth_bridge.py'),
        '--manifest-json', str(output_dir / 'stage2_secure_network_kth_manifest.json'),
        '--input-pt', str(output_dir / 'stage2_secure_network_kth_input_smoke8.pt'),
        '--output-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
        '--output-json', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.json'),
        '--runtime', runtime,
    ]
    command.extend(build_selection_mode_args(selection_mode, phase3_selection_manifest))
    command.extend(build_payload_args(payload_dtype, payload_stage_dtypes, payload_boundary_window))
    if runtime == 'spu':
        if not config:
            raise ValueError('--config is required when runtime=spu')
        command.extend(['--config', config])
    return command


def build_kth_checker_command(output_dir: Path):
    return [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'transshield_secure_network_kth.py'),
        'check',
        '--reference-pt', str(output_dir / 'stage2_secure_network_kth_reference_smoke8.pt'),
        '--candidate-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
        '--tolerance', '5e-5',
        '--output-json', str(output_dir / 'stage2_secure_network_kth_candidate_check.json'),
    ]


def build_tie_bridge_command(output_dir: Path):
    return [
        sys.executable,
        str(REPO_ROOT / 'integrations' / 'openbumblebee' / 'transshield_tie_policy_bridge' / 'transshield_tie_policy_bridge.py'),
        '--input-pt', str(output_dir / 'stage2_secure_network_kth_input_smoke8.pt'),
        '--kth-payload-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
        '--output-pt', str(output_dir / 'stage2_secure_tie_candidate_from_server.pt'),
        '--output-json', str(output_dir / 'stage2_secure_tie_candidate_from_server.json'),
        '--tie-policy', 'lowest_index',
        '--threshold-tolerance', '5e-5',
    ]


def build_tie_checker_command(output_dir: Path):
    return [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'transshield_secure_tie_payload.py'),
        'check',
        '--reference-pt', str(output_dir / 'stage2_secure_tie_policy_lowest_smoke8.pt'),
        '--candidate-pt', str(output_dir / 'stage2_secure_tie_candidate_from_server.pt'),
        '--input-pt', str(output_dir / 'stage2_secure_network_kth_input_smoke8.pt'),
        '--kth-payload-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
        '--threshold-tolerance', '5e-5',
        '--output-json', str(output_dir / 'stage2_secure_tie_candidate_check.json'),
    ]


def build_eval_replay_steps(output_dir: Path, bundle_dir: Path, eval_data_path: str, eval_max_samples: int):
    if not eval_data_path:
        raise ValueError('--eval-data-path is required when --eval-replay is enabled')
    return [
        build_step(
            'network_kth_eval_replay',
            [
                sys.executable,
                str(REPO_ROOT / 'tools' / 'transshield_secure_network_kth.py'),
                'branch-eval',
                '--bundle-dir', str(bundle_dir),
                '--kth-payload-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
                '--data-path', eval_data_path,
                '--device', 'cpu',
                '--batch-size', '16',
                '--num-workers', '0',
                '--max-samples', str(eval_max_samples),
                '--output-json', str(output_dir / 'stage2_secure_network_kth_eval_replay.json'),
            ],
            [str(output_dir / 'stage2_secure_network_kth_eval_replay.json')],
        ),
        build_step(
            'tie_policy_eval_replay',
            [
                sys.executable,
                str(REPO_ROOT / 'tools' / 'transshield_secure_tie_payload.py'),
                'branch-eval',
                '--bundle-dir', str(bundle_dir),
                '--tie-payload-pt', str(output_dir / 'stage2_secure_tie_candidate_from_server.pt'),
                '--data-path', eval_data_path,
                '--device', 'cpu',
                '--batch-size', '16',
                '--num-workers', '0',
                '--max-samples', str(eval_max_samples),
                '--output-json', str(output_dir / 'stage2_secure_tie_eval_replay.json'),
            ],
            [str(output_dir / 'stage2_secure_tie_eval_replay.json')],
        ),
    ]


def build_step_commands(
    output_dir: Path,
    bundle_dir: Path,
    runtime: str,
    config: str,
    eval_replay: bool,
    eval_data_path: str,
    eval_max_samples: int,
    selection_mode: str,
    phase3_selection_manifest: str,
    payload_dtype: str,
    payload_stage_dtypes: str,
    payload_boundary_window: int,
):
    steps = [
        build_step(
            'network_kth_bridge',
            build_kth_bridge_command(
                output_dir,
                runtime,
                config,
                selection_mode,
                phase3_selection_manifest,
                payload_dtype,
                payload_stage_dtypes,
                payload_boundary_window,
            ),
            [
                str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
                str(output_dir / 'stage2_secure_network_kth_candidate_from_server.json'),
            ],
        ),
        build_step(
            'network_kth_checker',
            build_kth_checker_command(output_dir),
            [str(output_dir / 'stage2_secure_network_kth_candidate_check.json')],
        ),
        build_step(
            'tie_policy_bridge',
            build_tie_bridge_command(output_dir),
            [
                str(output_dir / 'stage2_secure_tie_candidate_from_server.pt'),
                str(output_dir / 'stage2_secure_tie_candidate_from_server.json'),
            ],
        ),
        build_step(
            'tie_policy_checker',
            build_tie_checker_command(output_dir),
            [str(output_dir / 'stage2_secure_tie_candidate_check.json')],
        ),
    ]

    if eval_replay:
        steps.extend(build_eval_replay_steps(output_dir, bundle_dir, eval_data_path, eval_max_samples))
    return steps


def render_shell_script(path: Path, runtime: str, bundle_dir: Path):
    config_line = '--config "$CONFIG_PATH" ' if runtime == 'spu' else ''
    spu_runtime_setup = ''
    if runtime == 'spu':
        spu_runtime_setup = (
            '"$PYTHON_BIN" tools/transshield_spu_runtime_setup.py start '
            '--config "$CONFIG_PATH" '
            '--template configs/openbumblebee/2pc.template.json '
            '--backup '
            '--restart '
            '--remove-unsupported-cheetah-fields '
            '--log-dir logs/spu_nodes '
            '--state-json logs/spu_runtime_ports.json\n'
        )
    content = f"""#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=\"$(cd \"$(dirname \"$0\")/../..\" && pwd)\"
cd \"$REPO_ROOT\"
PYTHON_BIN=\"${{PYTHON_BIN:-python}}\"
BUNDLE_DIR=\"${{BUNDLE_DIR:-{bundle_dir}}}\"
CONFIG_PATH=\"${{CONFIG_PATH:-configs/openbumblebee/2pc.json}}\"
KTH_SELECTION_MODE=\"${{KTH_SELECTION_MODE:-{DEFAULT_PIPELINE_SELECTION_MODE}}}\"
PHASE3_SELECTION_MANIFEST=\"${{PHASE3_SELECTION_MANIFEST:-results/blockwise_exact_kth_selection_manifest_default.json}}\"
{spu_runtime_setup}\
EXTRA_ARGS=()
if [[ \"$KTH_SELECTION_MODE\" != \"flat_odd_even\" ]]; then
  EXTRA_ARGS+=(--phase3-selection-manifest \"$PHASE3_SELECTION_MANIFEST\")
fi
\"$PYTHON_BIN\" tools/transshield_openbumblebee_pipeline.py run --runtime {runtime} --bundle-dir \"$BUNDLE_DIR\" {config_line}--selection-mode \"$KTH_SELECTION_MODE\" \"${{EXTRA_ARGS[@]}}\" --output-dir artifacts/server_pipeline_run
"""
    path.write_text(content, encoding='utf-8')
    path.chmod(0o755)


def build_replay_command(
    output_dir: Path,
    bundle_dir: Path,
    enable_model_replay: bool,
    device: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    sample_root_from: str,
    sample_root_to: str,
    threshold_tolerance: float,
):
    command = [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'transshield_openbumblebee_inference_replay.py'),
        '--bundle-dir', str(bundle_dir),
        '--input-pt', str(output_dir / 'stage2_secure_network_kth_input_smoke8.pt'),
        '--kth-payload-pt', str(output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'),
        '--tie-payload-pt', str(output_dir / 'stage2_secure_tie_candidate_from_server.pt'),
        '--device', device,
        '--max-samples', str(max_samples),
        '--batch-size', str(batch_size),
        '--num-workers', str(num_workers),
        '--threshold-tolerance', str(threshold_tolerance),
        '--output-json', str(output_dir / 'pipeline_inference_replay_summary.json'),
    ]
    if enable_model_replay:
        command.append('--enable-model-replay')
    if sample_root_from:
        command.extend(['--sample-root-from', sample_root_from])
    if sample_root_to:
        command.extend(['--sample-root-to', sample_root_to])
    return command


def prepare(output_dir: Path, bundle_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    kth_dir = output_dir / 'network_kth_pack'
    tie_dir = output_dir / 'tie_policy_pack'
    kth_manifest = prepare_kth_pack(bundle_dir, kth_dir, DEFAULT_KTH_BRIDGE_SCRIPT, DEFAULT_SPU_CONFIG)
    tie_manifest = prepare_tie_pack(bundle_dir, tie_dir, DEFAULT_TIE_BRIDGE_SCRIPT)

    summary = {
        'repo_root': str(REPO_ROOT),
        'output_dir': str(output_dir),
        'bundle_dir': str(bundle_dir),
        'default_spu_config': str(DEFAULT_SPU_CONFIG),
        'packs': {
            'network_kth': kth_manifest,
            'tie_policy': tie_manifest,
        },
        'server_entrypoints': {
            'cpu_script': str(output_dir / 'run_server_cpu.sh'),
            'spu_script': str(output_dir / 'run_server_spu.sh'),
            'python_run_command': f'python tools/transshield_openbumblebee_pipeline.py run --runtime spu --bundle-dir {bundle_dir} --config configs/openbumblebee/2pc.json --output-dir artifacts/server_pipeline_run',
            'python_replay_command': f'python tools/transshield_openbumblebee_pipeline.py replay --output-dir artifacts/server_pipeline_run --bundle-dir {bundle_dir}',
        },
        'constraints': {
            'prepare_does_not_run_openbumblebee': True,
            'prepare_does_not_run_spu': True,
        },
    }
    write_json(output_dir / 'pipeline_manifest.json', summary)
    render_shell_script(output_dir / 'run_server_cpu.sh', 'cpu', bundle_dir)
    render_shell_script(output_dir / 'run_server_spu.sh', 'spu', bundle_dir)
    (output_dir / 'README.md').write_text(
        "# Transshield OpenBumbleBee server pipeline pack\n\n"
        "This pack contains both bridge stages and ready-to-run server entrypoints.\n"
        "Use `run_server_cpu.sh` or `run_server_spu.sh` after copying the repo to the server.\n"
        "By default the SPU entrypoint uses `configs/openbumblebee/2pc.json` inside this repo.\n"
        "`run_server_spu.sh` calls `tools/transshield_spu_runtime_setup.py` first to "
        "rewrite free localhost ports, restart each node separately, and warm up the SPU runtime.\n",
        encoding='utf-8',
    )
    return summary


def execute_steps(steps, dry_run: bool, step_log_dir: Path):
    results = []
    step_log_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        command = [str(item) for item in step['command']]
        result = {'name': step['name'], 'command': command, 'outputs': step['outputs']}
        log_path = step_log_dir / f'{len(results):02d}_{step["name"]}.log'
        result['log_path'] = str(log_path)
        if dry_run:
            result['status'] = 'dry_run'
            results.append(result)
            continue

        print(f'[pipeline] starting step: {step["name"]}', flush=True)
        started_at = time.time()
        with log_path.open('w', encoding='utf-8') as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finished_at = time.time()

        result['returncode'] = completed.returncode
        result['status'] = 'ok' if completed.returncode == 0 else 'failed'
        result['duration_sec'] = float(finished_at - started_at)
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f'step failed: {step["name"]}\n'
                f'log: {log_path}\n'
                f'last lines:\n{tail_text(log_path)}'
            )
        print(f'[pipeline] finished step: {step["name"]} -> {log_path}', flush=True)
    return results


def run_pipeline(
    output_dir: Path,
    bundle_dir: Path,
    runtime: str,
    config: str,
    eval_replay: bool,
    eval_data_path: str,
    eval_max_samples: int,
    selection_mode: str,
    phase3_selection_manifest: str,
    payload_dtype: str,
    payload_stage_dtypes: str,
    payload_boundary_window: int,
    dry_run: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_inputs(output_dir, bundle_dir)
    if eval_replay and not bundle_supports_eval_replay(bundle_dir):
        raise FileNotFoundError(
            f'eval replay requested but {bundle_dir / DEFAULT_BUNDLE_MODEL_STATE_NAME} is missing; '
            'copy the frozen model into the standalone repo first.'
        )
    steps = build_step_commands(
        output_dir,
        bundle_dir,
        runtime,
        config,
        eval_replay,
        eval_data_path,
        eval_max_samples,
        selection_mode,
        phase3_selection_manifest,
        payload_dtype,
        payload_stage_dtypes,
        payload_boundary_window,
    )
    results = execute_steps(steps, dry_run, output_dir / 'step_logs')
    summary = {
        'repo_root': str(REPO_ROOT),
        'output_dir': str(output_dir),
        'bundle_dir': str(bundle_dir),
        'runtime': runtime,
        'selection_mode': selection_mode,
        'phase3_selection_manifest': phase3_selection_manifest,
        'payload_dtype': payload_dtype,
        'payload_stage_dtypes': payload_stage_dtypes,
        'payload_boundary_window': int(payload_boundary_window),
        'dry_run': dry_run,
        'eval_replay': eval_replay,
        'eval_max_samples': eval_max_samples,
        'steps': results,
    }
    write_json(output_dir / 'pipeline_run_summary.json', summary)
    return summary


def verify_pipeline(output_dir: Path):
    output_dir = output_dir.resolve()
    required = {
        'network_kth_check': output_dir / 'stage2_secure_network_kth_candidate_check.json',
        'tie_policy_check': output_dir / 'stage2_secure_tie_candidate_check.json',
    }
    report = {'output_dir': str(output_dir), 'required_checks': {}, 'overall_passed': True}
    for key, path in required.items():
        exists = path.exists()
        payload = json.loads(path.read_text(encoding='utf-8')) if exists else None
        passed = bool(payload.get('overall_passed')) if payload else False
        report['required_checks'][key] = {'path': str(path), 'exists': exists, 'overall_passed': passed}
        report['overall_passed'] = report['overall_passed'] and exists and passed
    write_json(output_dir / 'pipeline_verify_summary.json', report)
    return report


def replay_pipeline(
    output_dir: Path,
    bundle_dir: Path,
    enable_model_replay: bool,
    device: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    sample_root_from: str,
    sample_root_to: str,
    threshold_tolerance: float,
):
    output_dir = output_dir.resolve()
    required = [
        output_dir / 'stage2_secure_network_kth_input_smoke8.pt',
        output_dir / 'stage2_secure_network_kth_candidate_from_server.pt',
        output_dir / 'stage2_secure_tie_candidate_from_server.pt',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'replay requires pipeline outputs that are missing: {missing}')

    command = build_replay_command(
        output_dir=output_dir,
        bundle_dir=bundle_dir.resolve(),
        enable_model_replay=enable_model_replay,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_root_from=sample_root_from,
        sample_root_to=sample_root_to,
        threshold_tolerance=threshold_tolerance,
    )
    started_at = time.time()
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    finished_at = time.time()
    if completed.returncode != 0:
        raise RuntimeError(f'replay step failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}')
    summary = json.loads((output_dir / 'pipeline_inference_replay_summary.json').read_text(encoding='utf-8'))
    summary['command'] = command
    summary['command_duration_sec'] = float(finished_at - started_at)
    write_json(output_dir / 'pipeline_inference_replay_summary.json', summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description='Unified Transshield OpenBumbleBee server pipeline.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_prepare = subparsers.add_parser('prepare', help='emit a server-ready combined pipeline pack')
    parser_prepare.add_argument('--output-dir', required=True)
    parser_prepare.add_argument('--bundle-dir', default=str(DEFAULT_PRESENTATION_BUNDLE_DIR))

    parser_run = subparsers.add_parser('run', help='run the combined server pipeline')
    parser_run.add_argument('--output-dir', required=True)
    parser_run.add_argument('--bundle-dir', default=str(DEFAULT_PRESENTATION_BUNDLE_DIR))
    parser_run.add_argument('--runtime', choices=['cpu', 'spu'], default='cpu')
    parser_run.add_argument('--config', default=str(DEFAULT_SPU_CONFIG))
    parser_run.add_argument('--eval-replay', action='store_true')
    parser_run.add_argument('--eval-data-path', default='')
    parser_run.add_argument('--eval-max-samples', type=int, default=0)
    parser_run.add_argument('--selection-mode', choices=['flat_odd_even', 'phase3_lower_tail', 'blockwise_exact_kth'], default=DEFAULT_PIPELINE_SELECTION_MODE)
    parser_run.add_argument('--phase3-selection-manifest', default=str(DEFAULT_PHASE3_SELECTION_MANIFEST))
    parser_run.add_argument('--payload-dtype', choices=['float32', 'float16'], default='float32')
    parser_run.add_argument('--payload-stage-dtypes', default='')
    parser_run.add_argument('--payload-boundary-window', type=int, default=0)
    parser_run.add_argument('--dry-run', action='store_true')

    parser_verify = subparsers.add_parser('verify', help='verify combined pipeline outputs')
    parser_verify.add_argument('--output-dir', required=True)

    parser_replay = subparsers.add_parser('replay', help='replay the final inference boundary from pipeline outputs')
    parser_replay.add_argument('--output-dir', required=True)
    parser_replay.add_argument('--bundle-dir', default=str(DEFAULT_PRESENTATION_BUNDLE_DIR))
    parser_replay.add_argument('--enable-model-replay', action='store_true')
    parser_replay.add_argument('--device', default='cpu')
    parser_replay.add_argument('--max-samples', type=int, default=0)
    parser_replay.add_argument('--batch-size', type=int, default=32)
    parser_replay.add_argument('--num-workers', type=int, default=0)
    parser_replay.add_argument('--sample-root-from', default='')
    parser_replay.add_argument('--sample-root-to', default='')
    parser_replay.add_argument('--threshold-tolerance', type=float, default=DEFAULT_THRESHOLD_TOLERANCE)
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == 'prepare':
        summary = prepare(Path(args.output_dir).resolve(), Path(args.bundle_dir).resolve())
    elif args.command == 'run':
        summary = run_pipeline(
            Path(args.output_dir).resolve(),
            Path(args.bundle_dir).resolve(),
            args.runtime,
            args.config,
            args.eval_replay,
            args.eval_data_path,
            args.eval_max_samples,
            args.selection_mode,
            args.phase3_selection_manifest,
            args.payload_dtype,
            args.payload_stage_dtypes,
            args.payload_boundary_window,
            args.dry_run,
        )
    elif args.command == 'replay':
        summary = replay_pipeline(
            Path(args.output_dir).resolve(),
            Path(args.bundle_dir).resolve(),
            args.enable_model_replay,
            args.device,
            args.max_samples,
            args.batch_size,
            args.num_workers,
            args.sample_root_from,
            args.sample_root_to,
            args.threshold_tolerance,
        )
    else:
        summary = verify_pipeline(Path(args.output_dir).resolve())

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
