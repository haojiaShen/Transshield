import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_secure_tie_payload import build_report, load_payload


DEFAULT_BUNDLE_DIR = REPO_ROOT / 'artifacts' / 'frozen_bundle'
DEFAULT_BRIDGE_SCRIPT = REPO_ROOT / 'integrations' / 'openbumblebee' / 'transshield_tie_policy_bridge' / 'transshield_tie_policy_bridge.py'
DEFAULT_REFERENCE_PT = DEFAULT_BUNDLE_DIR / 'stage2_secure_tie_policy_lowest_smoke8.pt'
DEFAULT_INPUT_PT = DEFAULT_BUNDLE_DIR / 'stage2_secure_network_kth_input_smoke8.pt'


def write_json(path: Path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def copy_required_files(bundle_dir: Path, output_dir: Path):
    required = [
        'stage2_secure_network_kth_input_smoke8.pt',
        'stage2_secure_network_kth_input_smoke8.json',
        'stage2_secure_tie_payload_contract.json',
        'stage2_secure_tie_policy_lowest_smoke8.pt',
        'stage2_secure_tie_policy_lowest_smoke8.json',
    ]
    copied = []
    for name in required:
        src = bundle_dir / name
        dst = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f'missing required tie bridge artifact: {src}')
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def build_prepare_manifest(bundle_dir: Path, output_dir: Path, bridge_script: Path):
    copied = copy_required_files(bundle_dir, output_dir)
    candidate_pt = output_dir / 'stage2_secure_tie_candidate_from_server.pt'
    candidate_json = output_dir / 'stage2_secure_tie_candidate_from_server.json'
    check_json = output_dir / 'stage2_secure_tie_candidate_check.json'

    kth_payload_placeholder = output_dir / 'stage2_secure_network_kth_candidate_from_server.pt'
    cpu_command = (
        f'python {bridge_script} '
        f'--input-pt {output_dir / "stage2_secure_network_kth_input_smoke8.pt"} '
        f'--kth-payload-pt {kth_payload_placeholder} '
        f'--output-pt {candidate_pt} '
        f'--output-json {candidate_json} '
        f'--tie-policy lowest_index'
    )
    checker_command = (
        f'python {REPO_ROOT / "tools" / "transshield_secure_tie_payload.py"} check '
        f'--reference-pt {output_dir / "stage2_secure_tie_policy_lowest_smoke8.pt"} '
        f'--candidate-pt {candidate_pt} '
        f'--output-json {check_json}'
    )

    return {
        'repo_root': str(REPO_ROOT),
        'bundle_dir': str(bundle_dir),
        'bridge_script': str(bridge_script),
        'output_dir': str(output_dir),
        'copied_files': copied,
        'depends_on': {
            'required_kth_payload_pt': str(kth_payload_placeholder),
            'note': 'This file is expected to be produced by the network-kth bridge step first.',
        },
        'expected_outputs': {
            'candidate_pt': str(candidate_pt),
            'candidate_json': str(candidate_json),
            'check_json': str(check_json),
        },
        'commands': {
            'tie_bridge_lowest_index': cpu_command,
            'checker': checker_command,
        },
        'constraints': {
            'prepare_step_does_not_run_openbumblebee': True,
            'prepare_step_does_not_run_spu': True,
            'server_execution_expected_later': True,
        },
    }


def write_prepare_readme(output_dir: Path):
    text = """# Transshield OpenBumbleBee tie bridge pack

This pack prepares the deterministic `lowest_index` tie-policy bridge.

## Expected order

1. Run the network-kth bridge first and place `stage2_secure_network_kth_candidate_from_server.pt` in this directory
2. Run the tie-policy bridge command from `commands.json`
3. Run the tie checker command from `commands.json`

This pack does **not** run BumbleBee locally.
"""
    (output_dir / 'README.md').write_text(text, encoding='utf-8')


def prepare_bridge_pack(bundle_dir: Path, output_dir: Path, bridge_script: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_prepare_manifest(bundle_dir, output_dir, bridge_script)
    write_json(output_dir / 'commands.json', manifest)
    write_prepare_readme(output_dir)
    return manifest


def verify_candidate(reference_pt: Path, candidate_pt: Path, output_json: Path):
    report = build_report(
        load_payload(reference_pt.resolve()),
        load_payload(candidate_pt.resolve()),
        reference_pt.resolve(),
        candidate_pt.resolve(),
    )
    write_json(output_json, report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Prepare and verify the standalone Transshield tie-policy bridge without running BumbleBee locally.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_prepare = subparsers.add_parser('prepare', help='copy a server-friendly tie bridge pack and emit command templates')
    parser_prepare.add_argument('--bundle-dir', default=str(DEFAULT_BUNDLE_DIR))
    parser_prepare.add_argument('--output-dir', required=True)
    parser_prepare.add_argument('--bridge-script', default=str(DEFAULT_BRIDGE_SCRIPT))

    parser_verify = subparsers.add_parser('verify', help='verify a candidate tie sidecar against the deterministic lowest-index smoke reference')
    parser_verify.add_argument('--reference-pt', default=str(DEFAULT_REFERENCE_PT))
    parser_verify.add_argument('--candidate-pt', required=True)
    parser_verify.add_argument('--output-json', required=True)

    args = parser.parse_args()

    if args.command == 'prepare':
        manifest = prepare_bridge_pack(
            Path(args.bundle_dir).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.bridge_script).resolve(),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    report = verify_candidate(
        Path(args.reference_pt).resolve(),
        Path(args.candidate_pt).resolve(),
        Path(args.output_json).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
