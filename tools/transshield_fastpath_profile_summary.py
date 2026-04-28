#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


RPC_REQUEST_RE = re.compile(
    r'\[fastpath-profile\] rpc request fn=(?P<fn>\S+) peer=(?P<peer>\S+) '
    r'req_bytes=(?P<bytes>\d+) req_chunks=(?P<chunks>\d+)'
)
RPC_RESPONSE_RE = re.compile(
    r'\[fastpath-profile\] rpc response fn=(?P<fn>\S+) peer=(?P<peer>\S+) '
    r'rsp_bytes=(?P<bytes>\d+) rsp_chunks=(?P<chunks>\d+)'
)
FETCH_OBJECT_RE = re.compile(
    r'\[fastpath-profile\] fetch object dst_node=(?P<dst>\S+) '
    r'src_node=(?P<src>\S+) ref=(?P<ref>\S+)'
)
MAKE_SHARES_RE = re.compile(
    r'\[fastpath-profile\] pyu_make_shares owner_rank=(?P<owner_rank>-?\d+) '
    r'world_size=(?P<world_size>\d+) shape=(?P<shape>.*?) dtype=(?P<dtype>\S+) '
    r'x_bytes=(?P<x_bytes>\d+) share_chunks=(?P<share_chunks>\d+) vtype=(?P<vtype>\d+)'
)
BUILTIN_RUN_RE = re.compile(
    r'\[fastpath-profile\] builtin_spu_run node=(?P<node>\S+) device=(?P<device>\S+) '
    r'inputs=(?P<inputs>\d+) wrapped_shares=(?P<wrapped>\d+) public_values=(?P<public>\d+)'
)
LINK_DETAILS_RE = re.compile(
    r'Link details: total send bytes (?P<send_bytes>\d+), recv bytes (?P<recv_bytes>\d+), '
    r'send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)'
)


def make_rpc_peer_entry():
    return {'fn': '', 'peer': '', 'count': 0, 'bytes': 0, 'chunks': 0, 'max_bytes': 0}


def make_rpc_fn_entry():
    return {'fn': '', 'count': 0, 'bytes': 0, 'chunks': 0, 'max_bytes': 0}


def make_rpc_combined_entry():
    return {
        'fn': '',
        'request_count': 0,
        'request_bytes': 0,
        'request_chunks': 0,
        'request_max_bytes': 0,
        'response_count': 0,
        'response_bytes': 0,
        'response_chunks': 0,
        'response_max_bytes': 0,
        'total_count': 0,
        'total_bytes': 0,
        'total_chunks': 0,
    }


def make_fetch_entry():
    return {'src': '', 'dst': '', 'count': 0}


def make_builtin_run_entry():
    return {'node': '', 'device': '', 'count': 0, 'inputs': 0, 'wrapped_shares': 0, 'public_values': 0}


def iter_paths(paths):
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            yield path
        else:
            yield from sorted(path.rglob('*.log'))


def build_parse_state():
    return {
        'rpc_requests': defaultdict(make_rpc_peer_entry),
        'rpc_responses': defaultdict(make_rpc_peer_entry),
        'rpc_requests_by_fn': defaultdict(make_rpc_fn_entry),
        'rpc_responses_by_fn': defaultdict(make_rpc_fn_entry),
        'rpc_combined_by_fn': defaultdict(make_rpc_combined_entry),
        'fetch_objects': defaultdict(make_fetch_entry),
        'make_shares': [],
        'builtin_runs': defaultdict(make_builtin_run_entry),
        'link_details': [],
        'matched_line_count': 0,
        'parsed_files': [],
    }


def record_rpc(bucket, fn, peer, byte_count, chunk_count):
    key = f'{fn}|{peer}'
    item = bucket[key]
    item['fn'] = fn
    item['peer'] = peer
    item['count'] += 1
    item['bytes'] += byte_count
    item['chunks'] += chunk_count
    item['max_bytes'] = max(item['max_bytes'], byte_count)


def record_rpc_by_fn(bucket, fn, byte_count, chunk_count):
    item = bucket[fn]
    item['fn'] = fn
    item['count'] += 1
    item['bytes'] += byte_count
    item['chunks'] += chunk_count
    item['max_bytes'] = max(item['max_bytes'], byte_count)


def record_rpc_combined(bucket, fn, direction, byte_count, chunk_count):
    item = bucket[fn]
    item['fn'] = fn
    prefix = 'request' if direction == 'request' else 'response'
    item[f'{prefix}_count'] += 1
    item[f'{prefix}_bytes'] += byte_count
    item[f'{prefix}_chunks'] += chunk_count
    item[f'{prefix}_max_bytes'] = max(item[f'{prefix}_max_bytes'], byte_count)
    item['total_count'] = item['request_count'] + item['response_count']
    item['total_bytes'] = item['request_bytes'] + item['response_bytes']
    item['total_chunks'] = item['request_chunks'] + item['response_chunks']


def record_fetch_object(bucket, src, dst):
    key = f'{src}->{dst}'
    item = bucket[key]
    item['src'] = src
    item['dst'] = dst
    item['count'] += 1


def record_make_shares(items, match, path):
    items.append(
        {
            'owner_rank': int(match.group('owner_rank')),
            'world_size': int(match.group('world_size')),
            'shape': match.group('shape'),
            'dtype': match.group('dtype'),
            'x_bytes': int(match.group('x_bytes')),
            'share_chunks': int(match.group('share_chunks')),
            'vtype': int(match.group('vtype')),
            'path': str(path),
        }
    )


def record_builtin_run(bucket, match):
    key = f'{match.group("node")}|{match.group("device")}'
    item = bucket[key]
    item['node'] = match.group('node')
    item['device'] = match.group('device')
    item['count'] += 1
    item['inputs'] += int(match.group('inputs'))
    item['wrapped_shares'] += int(match.group('wrapped'))
    item['public_values'] += int(match.group('public'))


def match_rpc_line(line, state):
    request_match = RPC_REQUEST_RE.search(line)
    if request_match:
        byte_count = int(request_match.group('bytes'))
        chunk_count = int(request_match.group('chunks'))
        fn = request_match.group('fn')
        peer = request_match.group('peer')
        record_rpc(state['rpc_requests'], fn, peer, byte_count, chunk_count)
        record_rpc_by_fn(state['rpc_requests_by_fn'], fn, byte_count, chunk_count)
        record_rpc_combined(state['rpc_combined_by_fn'], fn, 'request', byte_count, chunk_count)
        return True

    response_match = RPC_RESPONSE_RE.search(line)
    if response_match:
        byte_count = int(response_match.group('bytes'))
        chunk_count = int(response_match.group('chunks'))
        fn = response_match.group('fn')
        peer = response_match.group('peer')
        record_rpc(state['rpc_responses'], fn, peer, byte_count, chunk_count)
        record_rpc_by_fn(state['rpc_responses_by_fn'], fn, byte_count, chunk_count)
        record_rpc_combined(state['rpc_combined_by_fn'], fn, 'response', byte_count, chunk_count)
        return True
    return False


def match_non_rpc_line(line, state, path):
    fetch_match = FETCH_OBJECT_RE.search(line)
    if fetch_match:
        record_fetch_object(state['fetch_objects'], fetch_match.group('src'), fetch_match.group('dst'))
        return True

    shares_match = MAKE_SHARES_RE.search(line)
    if shares_match:
        record_make_shares(state['make_shares'], shares_match, path)
        return True

    run_match = BUILTIN_RUN_RE.search(line)
    if run_match:
        record_builtin_run(state['builtin_runs'], run_match)
        return True

    link_match = LINK_DETAILS_RE.search(line)
    if link_match:
        state['link_details'].append({name: int(value) for name, value in link_match.groupdict().items()})
        return True
    return False


def parse_log_file(path, state):
    saw_match = False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        matched = match_rpc_line(line, state) or match_non_rpc_line(line, state, path)
        if matched:
            saw_match = True
            state['matched_line_count'] += 1
    if saw_match:
        state['parsed_files'].append(str(path))


def sort_summary_items(state):
    return {
        'rpc_requests_by_fn_peer': sorted(state['rpc_requests'].values(), key=lambda item: item['bytes'], reverse=True),
        'rpc_responses_by_fn_peer': sorted(state['rpc_responses'].values(), key=lambda item: item['bytes'], reverse=True),
        'rpc_requests_by_fn': sorted(state['rpc_requests_by_fn'].values(), key=lambda item: item['bytes'], reverse=True),
        'rpc_responses_by_fn': sorted(state['rpc_responses_by_fn'].values(), key=lambda item: item['bytes'], reverse=True),
        'rpc_combined_by_fn': sorted(state['rpc_combined_by_fn'].values(), key=lambda item: item['total_bytes'], reverse=True),
        'fetch_objects': sorted(state['fetch_objects'].values(), key=lambda item: item['count'], reverse=True),
        'builtin_runs': sorted(state['builtin_runs'].values(), key=lambda item: item['count'], reverse=True),
    }


def build_diagnosis(request_total, response_total, link_details, link_send_total, link_recv_total):
    if (request_total + response_total) > 0 and link_details and link_send_total == 0 and link_recv_total == 0:
        return (
            'Python fastpath RPC/cloudpickle traffic is nonzero while SPU Link details remain zero; '
            'default fast runtime communication is visible at the Python distributed layer, not in the inspected yacl link counters.'
        )
    return ''


def finalize_summary(state):
    request_total = sum(item['bytes'] for item in state['rpc_requests'].values())
    response_total = sum(item['bytes'] for item in state['rpc_responses'].values())
    link_send_total = sum(item['send_bytes'] for item in state['link_details'])
    link_recv_total = sum(item['recv_bytes'] for item in state['link_details'])
    sorted_items = sort_summary_items(state)
    return {
        'matched_line_count': state['matched_line_count'],
        'parsed_files': state['parsed_files'],
        'rpc_request_total_bytes': request_total,
        'rpc_response_total_bytes': response_total,
        'rpc_total_bytes': request_total + response_total,
        'rpc_requests_by_fn_peer': sorted_items['rpc_requests_by_fn_peer'],
        'rpc_responses_by_fn_peer': sorted_items['rpc_responses_by_fn_peer'],
        'rpc_requests_by_fn': sorted_items['rpc_requests_by_fn'],
        'rpc_responses_by_fn': sorted_items['rpc_responses_by_fn'],
        'rpc_combined_by_fn': sorted_items['rpc_combined_by_fn'],
        'fetch_objects': sorted_items['fetch_objects'],
        'make_shares': state['make_shares'],
        'make_shares_total_input_bytes': sum(item['x_bytes'] for item in state['make_shares']),
        'builtin_runs': sorted_items['builtin_runs'],
        'link_detail_count': len(state['link_details']),
        'link_send_total_bytes': link_send_total,
        'link_recv_total_bytes': link_recv_total,
        'link_details_all_zero': bool(state['link_details']) and link_send_total == 0 and link_recv_total == 0,
        'diagnosis': build_diagnosis(
            request_total,
            response_total,
            state['link_details'],
            link_send_total,
            link_recv_total,
        ),
    }


def parse_logs(paths):
    state = build_parse_state()
    for path in iter_paths(paths):
        parse_log_file(path, state)
    return finalize_summary(state)


def render_rpc_combined_item(item):
    return (
        f"- `{item['fn']}`: total=`{item['total_bytes']}`, "
        f"request=`{item['request_bytes']}`, response=`{item['response_bytes']}`, count=`{item['total_count']}`"
    )


def render_rpc_peer_item(item, direction):
    arrow = '->' if direction == 'request' else '<-'
    return f"- `{item['fn']}` {arrow} `{item['peer']}`: bytes=`{item['bytes']}`, count=`{item['count']}`, max=`{item['max_bytes']}`"


def render_fetch_object_item(item):
    return f"- `{item['src']}` -> `{item['dst']}`: count=`{item['count']}`"


def render_path_item(path):
    return f"- `{path}`"


def render_section(title, items, formatter):
    lines = ['', title]
    for item in items:
        lines.append(formatter(item))
    return lines


def render_markdown(summary):
    lines = [
        '# SPU Python Fast Path Profile Summary',
        '',
        f"- Matched fastpath lines: `{summary['matched_line_count']}`",
        f"- RPC request bytes: `{summary['rpc_request_total_bytes']}`",
        f"- RPC response bytes: `{summary['rpc_response_total_bytes']}`",
        f"- RPC total bytes: `{summary['rpc_total_bytes']}`",
        f"- make_shares input bytes: `{summary['make_shares_total_input_bytes']}`",
    ]
    if summary['diagnosis']:
        lines.extend(['', f"Diagnosis: {summary['diagnosis']}"])
    lines.extend(
        [
            '',
            '## Diagnostic Caveat',
            f"- C++ LinkDetails are diagnostic-only for this fastpath: count=`{summary['link_detail_count']}`, all_zero=`{summary['link_details_all_zero']}`",
            '- Primary communication display should use Python fastpath RPC bytes, not C++ LinkDetails zero counters.',
        ]
    )
    lines.extend(render_section('## Top RPC Functions', summary['rpc_combined_by_fn'][:12], render_rpc_combined_item))
    lines.extend(render_section('## Top RPC Requests', summary['rpc_requests_by_fn_peer'][:20], lambda item: render_rpc_peer_item(item, 'request')))
    lines.extend(render_section('## Top RPC Responses', summary['rpc_responses_by_fn_peer'][:20], lambda item: render_rpc_peer_item(item, 'response')))
    lines.extend(render_section('## Fetch Objects', summary['fetch_objects'][:20], render_fetch_object_item))
    lines.extend(render_section('## Parsed Files', summary['parsed_files'], render_path_item))
    return '\n'.join(lines) + '\n'


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description='Summarize SPU Python fastpath profile logs.')
    parser.add_argument('paths', nargs='+', help='Log files or directories to scan')
    parser.add_argument('--output-json', default='', help='Optional JSON output path')
    parser.add_argument('--output-md', default='', help='Optional Markdown output path')
    return parser


def main():
    args = build_parser().parse_args()
    summary = parse_logs(args.paths)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        write_text(Path(args.output_json), json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if args.output_md:
        write_text(Path(args.output_md), render_markdown(summary))


if __name__ == '__main__':
    main()
