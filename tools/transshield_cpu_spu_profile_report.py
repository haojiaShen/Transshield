import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def scalar_delta(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return float(rhs - lhs)


def scalar_ratio(lhs, rhs):
    if lhs in (None, 0) or rhs is None:
        return None
    return float(rhs / lhs)


def human_bytes(value):
    if value is None:
        return 'N/A'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(value)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f'{size:.2f} {units[unit_index]}'


def human_seconds(value):
    if value is None:
        return 'N/A'
    return f'{float(value):.4f}s'


def collect_metric_compare(cpu_summary, spu_summary):
    metric_names = [
        'total_pipeline_duration_sec',
        'replay_duration_sec',
        'network_kth_bridge_elapsed_sec',
        'tie_bridge_elapsed_sec',
    ]
    compare = {}
    cpu_step_profile = cpu_summary.get('step_profile', {}) or {}
    spu_step_profile = spu_summary.get('step_profile', {}) or {}
    for name in metric_names:
        cpu_value = cpu_step_profile.get(name)
        spu_value = spu_step_profile.get(name)
        compare[name] = {
            'cpu': cpu_value,
            'spu': spu_value,
            'delta_sec': scalar_delta(cpu_value, spu_value),
            'ratio_spu_over_cpu': scalar_ratio(cpu_value, spu_value),
        }

    step_names = sorted(
        set((cpu_step_profile.get('durations_sec') or {}).keys())
        | set((spu_step_profile.get('durations_sec') or {}).keys())
    )
    step_compare = {}
    for step_name in step_names:
        cpu_value = (cpu_step_profile.get('durations_sec') or {}).get(step_name)
        spu_value = (spu_step_profile.get('durations_sec') or {}).get(step_name)
        step_compare[step_name] = {
            'cpu': cpu_value,
            'spu': spu_value,
            'delta_sec': scalar_delta(cpu_value, spu_value),
            'ratio_spu_over_cpu': scalar_ratio(cpu_value, spu_value),
        }
    compare['pipeline_steps'] = step_compare
    return compare


def build_output(cpu_summary_path: Path, spu_summary_path: Path, cpu_summary, spu_summary):
    cpu_profile = cpu_summary.get('communication_profile') or {}
    spu_profile = spu_summary.get('communication_profile') or {}
    cpu_comm = cpu_profile.get('aggregate_link_metrics') or {}
    spu_link_comm = spu_profile.get('aggregate_link_metrics') or {}
    spu_fastpath_comm = spu_profile.get('aggregate_python_fastpath_metrics') or {}
    spu_comm_source = (
        'python_distributed_rpc_cloudpickle'
        if spu_fastpath_comm.get('rpc_total_bytes') is not None
        else 'cpp_yacl_linkdetails'
    )
    spu_comm_total = (
        spu_fastpath_comm.get('rpc_total_bytes')
        if spu_comm_source == 'python_distributed_rpc_cloudpickle'
        else spu_link_comm.get('sum_total_bytes')
    )
    output = {
        'cpu_summary': str(cpu_summary_path),
        'spu_summary': str(spu_summary_path),
        'cpu_runtime': cpu_summary.get('runtime'),
        'spu_runtime': spu_summary.get('runtime'),
        'time_compare': collect_metric_compare(cpu_summary, spu_summary),
        'communication_compare': {
            'cpu_supported': bool(cpu_profile.get('supported')),
            'spu_supported': bool(spu_profile.get('supported')),
            'spu_communication_source': spu_comm_source,
            'spu_sum_total_bytes': spu_comm_total,
            'spu_python_fastpath_rpc_request_bytes': spu_fastpath_comm.get('rpc_request_total_bytes'),
            'spu_python_fastpath_rpc_response_bytes': spu_fastpath_comm.get('rpc_response_total_bytes'),
            'spu_max_total_bytes_per_node': spu_link_comm.get('max_total_bytes'),
            'spu_sum_send_bytes': spu_link_comm.get('sum_send_bytes'),
            'spu_sum_recv_bytes': spu_link_comm.get('sum_recv_bytes'),
            'cpu_sum_total_bytes': cpu_comm.get('sum_total_bytes'),
        },
        'consistency': {
            'cpu_overall_passed': cpu_summary.get('overall_passed'),
            'spu_overall_passed': spu_summary.get('overall_passed'),
            'cpu_replay_overall_passed': cpu_summary.get('replay_overall_passed'),
            'spu_replay_overall_passed': spu_summary.get('replay_overall_passed'),
        },
        'notes': [
            'CPU runtime is a plaintext reference path, so communication bytes are expected to be unavailable.',
            'SPU default fast runtime communication is displayed from Python distributed RPC/cloudpickle metrics when available; C++ LinkDetails counters are treated as diagnostic-only for that path.',
        ],
    }
    return output


def render_markdown(report):
    total_compare = report['time_compare']['total_pipeline_duration_sec']
    replay_compare = report['time_compare']['replay_duration_sec']
    comm = report['communication_compare']
    total_ratio_text = 'N/A' if total_compare['ratio_spu_over_cpu'] is None else f"{total_compare['ratio_spu_over_cpu']:.2f}x"
    replay_ratio_text = 'N/A' if replay_compare['ratio_spu_over_cpu'] is None else f"{replay_compare['ratio_spu_over_cpu']:.2f}x"
    lines = [
        '# CPU vs SPU Secure Profiling',
        '',
        '## 1. Overview',
        '',
        f"- CPU summary: `{report['cpu_summary']}`",
        f"- SPU summary: `{report['spu_summary']}`",
        f"- CPU verify passed: `{report['consistency']['cpu_overall_passed']}`",
        f"- SPU verify passed: `{report['consistency']['spu_overall_passed']}`",
        '',
        '## 2. Key timings',
        '',
        '| Metric | CPU | SPU | Delta (SPU-CPU) | Ratio (SPU/CPU) |',
        '|---|---:|---:|---:|---:|',
        f"| total_pipeline_duration_sec | {human_seconds(total_compare['cpu'])} | {human_seconds(total_compare['spu'])} | {human_seconds(total_compare['delta_sec'])} | {total_ratio_text} |",
        f"| replay_duration_sec | {human_seconds(replay_compare['cpu'])} | {human_seconds(replay_compare['spu'])} | {human_seconds(replay_compare['delta_sec'])} | {replay_ratio_text} |",
        '',
        '## 3. Pipeline step timings',
        '',
        '| Step | CPU | SPU | Delta (SPU-CPU) | Ratio (SPU/CPU) |',
        '|---|---:|---:|---:|---:|',
    ]
    for step_name, payload in report['time_compare']['pipeline_steps'].items():
        ratio_text = 'N/A' if payload['ratio_spu_over_cpu'] is None else f"{payload['ratio_spu_over_cpu']:.2f}x"
        lines.append(
            f"| {step_name} | {human_seconds(payload['cpu'])} | {human_seconds(payload['spu'])} | "
            f"{human_seconds(payload['delta_sec'])} | {ratio_text} |"
        )

    lines.extend(
        [
            '',
            '## 4. Communication',
            '',
            f"- CPU communication bytes: `{human_bytes(comm['cpu_sum_total_bytes'])}`",
            f"- SPU communication source: `{comm['spu_communication_source']}`",
            f"- SPU total communication bytes: `{human_bytes(comm['spu_sum_total_bytes'])}`",
            f"- SPU Python fastpath request bytes: `{human_bytes(comm.get('spu_python_fastpath_rpc_request_bytes'))}`",
            f"- SPU Python fastpath response bytes: `{human_bytes(comm.get('spu_python_fastpath_rpc_response_bytes'))}`",
            '',
            '## 5. Notes',
            '',
            '- CPU is the plaintext reference runtime, so missing communication metrics are expected.',
            '- SPU communication uses Python fastpath RPC metrics when present; C++ LinkDetails zero counters are not shown in the primary table.',
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate a CPU-vs-SPU secure profiling report.')
    parser.add_argument('--cpu-summary', required=True)
    parser.add_argument('--spu-summary', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    cpu_summary_path = Path(args.cpu_summary).resolve()
    spu_summary_path = Path(args.spu_summary).resolve()
    cpu_summary = load_json(cpu_summary_path)
    spu_summary = load_json(spu_summary_path)
    output = build_output(cpu_summary_path, spu_summary_path, cpu_summary, spu_summary)
    write_json(Path(args.output_json).resolve(), output)
    write_text(Path(args.output_md).resolve(), render_markdown(output) + '\n')
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
