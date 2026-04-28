#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_merged(primary_profile, aux_comm, primary_label, aux_label):
    merged = json.loads(json.dumps(primary_profile))
    communication_profile = merged.setdefault('communication_profile', {})
    communication_profile['status'] = 'available_via_aux_probe'
    communication_profile['supported'] = True
    communication_profile['aggregate_link_metrics'] = aux_comm.get('aggregate_link_metrics')
    communication_profile['diagnosis'] = (
        'Primary fast runtime keeps its original timing path; communication metrics are injected from an auxiliary '
        'communication-visible rerun where colocated optimization was disabled.'
    )
    communication_profile['note'] = (
        f'Runtime timings come from `{primary_label}`; communication bytes come from `{aux_label}`.'
    )
    communication_profile['auxiliary_probe'] = {
        'source': aux_label,
        'communication_status': aux_comm.get('communication_status'),
        'link_detail_count': aux_comm.get('link_detail_count'),
        'nonzero_link_detail_count': aux_comm.get('nonzero_link_detail_count'),
        'runtime_config_note': aux_comm.get('runtime_config_note'),
        'total_pipeline_duration_sec': aux_comm.get('total_pipeline_duration_sec'),
        'replay_duration_sec': aux_comm.get('replay_duration_sec'),
    }
    merged['profile_merge_note'] = (
        'This merged summary preserves the fast runtime timing path while attaching communication metrics from '
        'an auxiliary communication-visible diagnostic rerun.'
    )
    merged['profile_sources'] = {
        'primary_runtime_profile': primary_label,
        'auxiliary_communication_profile': aux_label,
    }
    return merged


def get_total_duration(payload):
    step_profile = payload.get('step_profile') or {}
    if step_profile.get('total_pipeline_duration_sec') is not None:
        return step_profile.get('total_pipeline_duration_sec')
    return payload.get('total_pipeline_duration_sec')


def get_replay_duration(payload):
    step_profile = payload.get('step_profile') or {}
    if step_profile.get('replay_duration_sec') is not None:
        return step_profile.get('replay_duration_sec')
    return payload.get('replay_duration_sec')


def render_markdown(merged):
    comm = merged.get('communication_profile', {}) or {}
    agg = comm.get('aggregate_link_metrics') or {}
    aux = comm.get('auxiliary_probe') or {}
    lines = [
        '# Merged Secure Profile Summary — Fast Runtime + Auxiliary Communication Probe',
        '',
        f"- merged communication status: `{comm.get('status')}`",
        f"- total pipeline duration (fast runtime): `{get_total_duration(merged)}`",
        f"- replay duration (fast runtime): `{get_replay_duration(merged)}`",
        f"- max total bytes (aux probe): `{agg.get('max_total_bytes')}`",
        f"- max send bytes (aux probe): `{agg.get('max_send_bytes')}`",
        f"- max recv bytes (aux probe): `{agg.get('max_recv_bytes')}`",
        '',
        '## Sources',
        '',
        f"- primary runtime profile: `{(merged.get('profile_sources') or {}).get('primary_runtime_profile')}`",
        f"- auxiliary communication profile: `{(merged.get('profile_sources') or {}).get('auxiliary_communication_profile')}`",
        '',
        '## Interpretation',
        '',
        '- 该结果保留快分支的默认时间口径。',
        '- 通信字节来自单独的 communication-visible 辅助 rerun。',
        '- 因此它适合作为展示/答辩时的 merged profile 口径，但应明确注明来源是“双分支合并”，不是单次原生快分支直接吐出的通信计数。',
    ]
    if aux.get('runtime_config_note'):
        lines.append(f"- auxiliary runtime note: {aux.get('runtime_config_note')}")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Merge a fast runtime secure profile with an auxiliary communication-visible rerun.')
    parser.add_argument('--primary-profile-json', required=True)
    parser.add_argument('--aux-communication-json', required=True)
    parser.add_argument('--primary-label', default='default_fast_runtime')
    parser.add_argument('--aux-label', default='diagnostic_comm_runtime')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    primary = load_json(Path(args.primary_profile_json).resolve())
    aux = load_json(Path(args.aux_communication_json).resolve())
    merged = build_merged(primary, aux, args.primary_label, args.aux_label)
    write_text(Path(args.output_json).resolve(), json.dumps(merged, indent=2, sort_keys=True) + '\n')
    write_text(Path(args.output_md).resolve(), render_markdown(merged))
    print(json.dumps({
        'merged_status': merged.get('communication_profile', {}).get('status'),
        'output_json': str(Path(args.output_json).resolve()),
        'output_md': str(Path(args.output_md).resolve()),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
