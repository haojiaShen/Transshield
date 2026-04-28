import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_secure_network_kth import build_report, load_payload


DEFAULT_BUNDLE_DIR = REPO_ROOT / 'artifacts' / 'frozen_bundle'
DEFAULT_BRIDGE_SCRIPT = REPO_ROOT / 'integrations' / 'openbumblebee' / 'transshield_network_kth_bridge' / 'transshield_network_kth_bridge.py'
DEFAULT_SPU_CONFIG = REPO_ROOT / 'configs' / 'openbumblebee' / '2pc.json'
DEFAULT_REFERENCE_PT = DEFAULT_BUNDLE_DIR / 'stage2_secure_network_kth_reference_smoke8.pt'
DEFAULT_INPUT_PT = DEFAULT_BUNDLE_DIR / 'stage2_secure_network_kth_input_smoke8.pt'
DEFAULT_MANIFEST_JSON = DEFAULT_BUNDLE_DIR / 'stage2_secure_network_kth_manifest.json'


def write_json(path: Path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def copy_required_files(bundle_dir: Path, output_dir: Path):
    required = [
        'stage2_secure_network_kth_manifest.json',
        'stage2_secure_network_kth_input_smoke8.pt',
        'stage2_secure_network_kth_input_smoke8.json',
        'stage2_secure_network_kth_reference_smoke8.pt',
        'stage2_secure_network_kth_reference_smoke8.json',
        'stage2_secure_network_kth_contract.json',
        'stage2_secure_kth_selection_manifest.json',
    ]
    copied = []
    for name in required:
        src = bundle_dir / name
        dst = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f'missing required bridge artifact: {src}')
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def build_prepare_manifest(bundle_dir: Path, output_dir: Path, bridge_script: Path, spu_config_path: Path):
    copied = copy_required_files(bundle_dir, output_dir)
    candidate_pt = output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'
    candidate_json = output_dir / 'stage2_secure_network_kth_candidate_from_server.json'
    check_json = output_dir / 'stage2_secure_network_kth_candidate_check.json'
    eval_json = output_dir / 'threshold_branch_eval_smoke8.json'

    cpu_command = (
        f'python {bridge_script} '
        f'--manifest-json {output_dir / "stage2_secure_network_kth_manifest.json"} '
        f'--input-pt {output_dir / "stage2_secure_network_kth_input_smoke8.pt"} '
        f'--output-pt {candidate_pt} '
        f'--output-json {candidate_json} '
        f'--runtime cpu'
    )
    spu_command = (
        f'python {bridge_script} '
        f'--manifest-json {output_dir / "stage2_secure_network_kth_manifest.json"} '
        f'--input-pt {output_dir / "stage2_secure_network_kth_input_smoke8.pt"} '
        f'--output-pt {candidate_pt} '
        f'--output-json {candidate_json} '
        f'--runtime spu '
        f'--config {spu_config_path}'
    )
    checker_command = (
        f'python {REPO_ROOT / "tools" / "transshield_secure_network_kth.py"} check '
        f'--reference-pt {output_dir / "stage2_secure_network_kth_reference_smoke8.pt"} '
        f'--candidate-pt {candidate_pt} '
        f'--output-json {check_json}'
    )
    eval_command = (
        f'python {REPO_ROOT / "tools" / "transshield_secure_network_kth.py"} branch-eval '
        f'--bundle-dir {bundle_dir} '
        f'--kth-payload-pt {candidate_pt} '
        f'--device cpu --batch-size 4 --num-workers 0 --max-samples 8 '
        f'--output-json {eval_json}'
    )

    return {
        'repo_root': str(REPO_ROOT),
        'bundle_dir': str(bundle_dir),
        'bridge_script': str(bridge_script),
        'spu_config_path': str(spu_config_path),
        'output_dir': str(output_dir),
        'copied_files': copied,
        'expected_outputs': {
            'candidate_pt': str(candidate_pt),
            'candidate_json': str(candidate_json),
            'check_json': str(check_json),
            'eval_json': str(eval_json),
        },
        'commands': {
            'cpu_bridge': cpu_command,
            'spu_bridge_template': spu_command,
            'checker': checker_command,
            'eval_replay': eval_command,
        },
        'constraints': {
            'prepare_step_does_not_run_openbumblebee': True,
            'prepare_step_does_not_run_spu': True,
            'server_execution_expected_later': True,
        },
    }


def write_prepare_readme(output_dir: Path):
    text = f"""# Transshield OpenBumbleBee bridge pack

This directory is a portable bridge pack prepared from the standalone `Transshield` repo.

## Included

- compare-network manifest
- `masked_score` smoke input sidecar
- reference `kth_threshold` smoke sidecar
- command templates for CPU / SPU bridge execution

## Expected workflow

1. Copy this pack plus the `Transshield` repo to the server
2. Run the OpenBumbleBee bridge there to produce `stage2_secure_network_kth_candidate_from_server.pt`
3. Run the checker command in `commands.json`
4. Optionally run the eval replay command in `commands.json`

## Important note

This pack is a **prepare-only** artifact. It does not run BumbleBee locally.
"""
    (output_dir / 'README.md').write_text(text, encoding='utf-8')


def prepare_bridge_pack(bundle_dir: Path, output_dir: Path, bridge_script: Path, spu_config_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_prepare_manifest(bundle_dir, output_dir, bridge_script, spu_config_path)
    write_json(output_dir / 'commands.json', manifest)
    write_prepare_readme(output_dir)
    return manifest


def verify_candidate(reference_pt: Path, candidate_pt: Path, output_json: Path, tolerance: float):
    report = build_report(
        load_payload(reference_pt.resolve()),
        load_payload(candidate_pt.resolve()),
        reference_pt.resolve(),
        candidate_pt.resolve(),
        tolerance,
    )
    write_json(output_json, report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Prepare and verify the standalone Transshield OpenBumbleBee bridge without running BumbleBee locally.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_prepare = subparsers.add_parser('prepare', help='copy a server-friendly bridge pack and emit command templates')
    parser_prepare.add_argument('--bundle-dir', default=str(DEFAULT_BUNDLE_DIR))
    parser_prepare.add_argument('--output-dir', required=True)
    parser_prepare.add_argument('--bridge-script', default=str(DEFAULT_BRIDGE_SCRIPT))
    parser_prepare.add_argument('--spu-config', default=str(DEFAULT_SPU_CONFIG))

    parser_verify = subparsers.add_parser('verify', help='verify a candidate kth-threshold sidecar against the smoke reference')
    parser_verify.add_argument('--reference-pt', default=str(DEFAULT_REFERENCE_PT))
    parser_verify.add_argument('--candidate-pt', required=True)
    parser_verify.add_argument('--output-json', required=True)
    parser_verify.add_argument('--tolerance', type=float, default=1e-6)

    args = parser.parse_args()

    if args.command == 'prepare':
        manifest = prepare_bridge_pack(
            Path(args.bundle_dir).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.bridge_script).resolve(),
            Path(args.spu_config).resolve(),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    report = verify_candidate(
        Path(args.reference_pt).resolve(),
        Path(args.candidate_pt).resolve(),
        Path(args.output_json).resolve(),
        args.tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
