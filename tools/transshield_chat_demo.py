import argparse
import email.policy
import hashlib
import heapq
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEB_DEMO_ROOT = REPO_ROOT / 'web_demo'
HTML_PATH = WEB_DEMO_ROOT / 'index.html'
DEMO_SUMMARY_PATH = REPO_ROOT / 'artifacts' / 'web_demo_assets' / 'demo_content_summary.json'
ALLOWED_UPLOAD_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
DEFAULT_MAX_UPLOAD_MB = 10
DEFAULT_SECURE_TIMEOUT_SEC = 600
DEFAULT_E2E_PROFILE = 'secret_depth6_clip0_showcase'
DEFAULT_FINANCE_BUNDLE_DIR = REPO_ROOT / 'artifacts' / 'frozen_bundle_finance_lrd_rank192_20260515'
DEFAULT_FINANCE_DATA_ROOT = REPO_ROOT / 'data' / 'finance_fraud_v3' / 'val'
DEFAULT_FINANCE_TIMEOUT_SEC = 1800
DEFAULT_FINANCE_SAMPLES_PER_CLASS = 2
E2E_SHARE_SHAPE = [1, 3, 224, 224]
E2E_SHARE_FLOAT_COUNT = 1 * 3 * 224 * 224
E2E_SHARE_BYTE_COUNT = E2E_SHARE_FLOAT_COUNT * 4
MEDICAL_CONTROL_PLANE_CONTRACT_VERSION = 'medical_control_plane_v7'
MEDICAL_MULTIPART_MAX_BYTES = 5 * 1024 * 1024
MEDICAL_MULTIPART_MAX_BOUNDARIES = 12
MEDICAL_MULTIPART_MAX_CONTENT_DISPOSITION = 10
MEDICAL_MULTIPART_MAX_HEADER_BYTES = 8 * 1024
MEDICAL_MULTIPART_MAX_TOP_LEVEL_PARTS = 7
JSON_PART_MAX_BYTES = 4096
QUALITY_SUMMARY_MAX_BYTES = 1024
CONTROL_PLANE_METRICS_MAX_BYTES = 1024
AUDIT_MANIFEST_MAX_BYTES = 2048
MEDICAL_REQUEST_FIELDS = {
    'domain',
    'client_contract_version',
    'share0',
    'share1',
    'client_quality_summary',
    'client_audit_manifest',
    'client_control_plane_metrics',
}
FINANCE_REQUEST_FIELDS = {'domain', 'sample_id'}
JSON_INT_MAX_DIGITS = 32
JSON_FLOAT_MAX_CHARS = 64
JSON_FLOAT_MAX_EXPONENT = 6
REPLAY_NONCE_TTL_SEC = 600.0
REPLAY_PAYLOAD_TTL_SEC = 120.0
REPLAY_GUARD_MAX_ITEMS = 20_000
IP_WINDOW_LIMIT = 6
IP_WINDOW_SEC = 60.0
IP_INFLIGHT_LIMIT = 2
IP_SHARD_COUNT = 64
IP_SHARD_CAPACITY = 1536
IP_STALE_TTL_SEC = 600.0
IP_GUARD_EVICT_BATCH = 500
REPLAY_GUARD_EVICT_BATCH = 500
REQUEST_JSON_PARSE_RE = re.compile(rb'[\r\n]')
SPU_LINK_DETAILS_RE = re.compile(
    r'Link details: total send bytes (?P<send>\d+), recv bytes (?P<recv>\d+), '
    r'send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)'
)
HEX_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
WEB_DEMO_ALLOWED_EXTENSIONS = {
    '.html',
    '.css',
    '.js',
    '.png',
    '.jpg',
    '.jpeg',
    '.webp',
    '.svg',
    '.json',
}


@dataclass
class IpState:
    window: deque = field(default_factory=deque)
    inflight: int = 0
    last_seen_monotonic: float = 0.0


@dataclass
class RawPart:
    name: str
    headers: dict
    body_start: int
    body_end: int
    filename: Optional[str] = None
    content_type: Optional[str] = None


def bool_from_env(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value in (None, ''):
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


def positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value in (None, ''):
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def float_from_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value in (None, ''):
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def clip_tag(value: float) -> str:
    return 'clip' + f'{value:g}'.replace('.', 'p')


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def empty_demo_summary():
    return {
        'updated_at': None,
        'default_bundle': None,
        'plaintext_stability_closure': None,
    }


def build_manifest_demo_summary(manifest):
    verified = manifest.get('verified_metrics', {})
    return {
        'updated_at': None,
        'default_bundle': {
            'title': '当前正式展示口径',
            'bundle_name': manifest.get('bundle_name'),
            'bundle_dir': 'artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430',
            'status': manifest.get('status'),
            'argmax_accuracy': verified.get('argmax_accuracy'),
            'threshold_accuracy': verified.get('threshold_accuracy'),
            'best_epoch': verified.get('best_epoch'),
            'argmax_match_ratio': verified.get('argmax_match_ratio'),
            'threshold_match_ratio': verified.get('threshold_match_ratio'),
            'spu_pipeline_overall_passed': verified.get('spu_pipeline_overall_passed'),
            'spu_replay_overall_passed': verified.get('spu_replay_overall_passed'),
            'communication_source': '本页面仅展示当前保留的正式运行通信量',
            'summary': '前端默认加载当前正式展示口径：医疗采用动态安全剪枝与完整隐私推理；金融只保留低秩压缩压力样本与完整隐私压力验证。',
        },
        'plaintext_stability_closure': None,
    }


def load_demo_summary():
    if DEMO_SUMMARY_PATH.exists():
        return load_json(DEMO_SUMMARY_PATH)
    return empty_demo_summary()


def parse_class_names(raw_value: str):
    values = [item.strip() for item in raw_value.split(',') if item.strip()]
    if len(values) != 2:
        raise ValueError('--class-names must contain exactly two comma-separated names')
    return values


def class_label(class_names, index):
    if index is None:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(class_names):
        return str(index)
    return class_names[index]


def public_sample_metadata(sample: dict):
    return {
        'id': sample['id'],
        'label': sample['label'],
        'category': sample['category'],
        'category_label': sample['category_label'],
        'ground_truth_index': sample['ground_truth_index'],
        'ground_truth_label': sample['ground_truth_label'],
        'preview_url': sample['preview_url'],
    }


def parse_content_type_boundary(content_type: str) -> Optional[bytes]:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    if not value:
        return None
    return value.encode('utf-8', 'strict')


def parse_part_headers_bytes(header_bytes: bytes) -> dict:
    headers = {}
    for line in header_bytes.split(b'\r\n'):
        if not line:
            continue
        if b':' not in line:
            raise ValueError('invalid multipart header line')
        key, value = line.split(b':', 1)
        headers[key.decode('ascii', 'strict').strip().lower()] = value.decode('latin-1').strip()
    return headers


def parse_content_disposition(value: str) -> dict:
    pieces = [piece.strip() for piece in value.split(';') if piece.strip()]
    if not pieces:
        return {}
    result = {'_kind': pieces[0].lower()}
    for piece in pieces[1:]:
        if '=' not in piece:
            continue
        key, raw = piece.split('=', 1)
        result[key.strip().lower()] = raw.strip().strip('"')
    return result


def build_mime_message(content_type: str, body: bytes) -> bytes:
    return (
        f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode('utf-8')
        + body
    )


def strict_json_int(value: str):
    if len(value.lstrip('+-')) > JSON_INT_MAX_DIGITS:
        raise ValueError('json integer token too large')
    return int(value)


def strict_json_float(value: str):
    if len(value) > JSON_FLOAT_MAX_CHARS:
        raise ValueError('json float token too large')
    lower = value.lower()
    if 'e' in lower:
        _, exponent = lower.split('e', 1)
        exponent_value = int(exponent)
        if abs(exponent_value) > JSON_FLOAT_MAX_EXPONENT:
            raise ValueError('json float exponent too large')
    return float(value)


def strict_json_constant(value: str):
    raise ValueError(f'json constant not allowed: {value}')


def ensure_sha256_hex(name: str, value) -> str:
    text = str(value or '').strip().lower()
    if not HEX_SHA256_RE.fullmatch(text):
        raise ValueError(f'{name} must be a lowercase 64-char sha256 hex string')
    return text


def server_sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_share_digest(label: str, payload: bytes) -> str:
    length_prefix = len(payload).to_bytes(8, byteorder='big', signed=False)
    digest = hashlib.blake2s()
    digest.update(label.encode('ascii'))
    digest.update(b'|')
    digest.update(length_prefix)
    digest.update(payload)
    return digest.hexdigest()


def make_payload_fingerprint(share0_bytes: bytes, share1_bytes: bytes) -> str:
    share0_digest = make_share_digest('share0', share0_bytes)
    share1_digest = make_share_digest('share1', share1_bytes)
    payload = f'v7|l0:{len(share0_bytes)}|l1:{len(share1_bytes)}|s0:{share0_digest}|s1:{share1_digest}'
    return hashlib.blake2s(payload.encode('utf-8')).hexdigest()


def bytes_to_float32_aligned(payload: bytes) -> np.ndarray:
    if len(payload) % 4 != 0:
        raise ValueError('payload length is not aligned to 4 bytes')
    return np.frombuffer(payload, dtype='<u4').copy()


def flush_tiny_values_inplace(values: np.ndarray, threshold: float = 1e-30):
    mask = np.abs(values) < threshold
    if np.any(mask):
        values[mask] = np.float32(0.0)


def contains_subnormal_values(share_u32: np.ndarray) -> bool:
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponent_zero = (share_u32 & exponent_mask) == 0
    mantissa_nonzero = (share_u32 & mantissa_mask) != 0
    return bool(np.any(exponent_zero & mantissa_nonzero))


def compute_luma_metrics(rgb_tensor: np.ndarray) -> dict:
    rgb = np.clip(rgb_tensor.astype(np.float32, copy=False), 0.0, 1.0)
    r = rgb[0, 0]
    g = rgb[0, 1]
    b = rgb[0, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    p05, p95 = np.percentile(luma, [5, 95])
    overexposed_ratio = float(np.mean(luma >= 0.95))
    underexposed_ratio = float(np.mean(luma <= 0.05))
    effective_luma_ratio = float(np.mean((luma >= 0.02) & (luma <= 0.98)))
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return {
        'mean_luma': float(np.mean(luma)),
        'std_luma': float(np.std(luma)),
        'overexposed_ratio': overexposed_ratio,
        'underexposed_ratio': underexposed_ratio,
        'effective_luma_ratio': effective_luma_ratio,
        'dynamic_range_p95_p05': float(p95 - p05),
        'laplacian_variance': float(np.var(lap)),
    }


def validate_quality_summary_object(payload: dict):
    required_keys = (
        'mean_luma',
        'std_luma',
        'overexposed_ratio',
        'underexposed_ratio',
        'effective_luma_ratio',
        'dynamic_range_p95_p05',
        'laplacian_variance',
    )
    for key in required_keys:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ValueError(f'client_quality_summary.{key} must be a finite number')


def validate_control_plane_metrics_object(payload: Optional[dict]):
    if payload is None:
        return
    allowed_keys = {
        'decode_ms',
        'preprocess_ms',
        'dqa_ms',
        'hash_ms',
        'share_build_ms',
        'total_ms',
    }
    for key, value in payload.items():
        if key not in allowed_keys:
            continue
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f'client_control_plane_metrics.{key} must be a finite non-negative number')


def build_quality_assurance(server_summary: dict, client_summary: Optional[dict]) -> dict:
    severe_reasons = []
    warning_reasons = []
    if server_summary['overexposed_ratio'] > 0.40:
        severe_reasons.append('overexposed_ratio>0.40')
    elif server_summary['overexposed_ratio'] > 0.15:
        warning_reasons.append('overexposed_ratio>0.15')
    if server_summary['underexposed_ratio'] > 0.40:
        severe_reasons.append('underexposed_ratio>0.40')
    elif server_summary['underexposed_ratio'] > 0.15:
        warning_reasons.append('underexposed_ratio>0.15')
    if server_summary['laplacian_variance'] < 1e-4 and (
        server_summary['effective_luma_ratio'] < 0.10
        or server_summary['dynamic_range_p95_p05'] < 0.02
    ):
        severe_reasons.append('degenerate_structure')
    elif server_summary['laplacian_variance'] < 5e-4:
        warning_reasons.append('laplacian_variance<5e-4')
    if server_summary['effective_luma_ratio'] < 0.20:
        warning_reasons.append('effective_luma_ratio<0.20')
    if server_summary['dynamic_range_p95_p05'] < 0.05:
        warning_reasons.append('dynamic_range_p95_p05<0.05')

    drift = {}
    significant_drift_keys = []
    if isinstance(client_summary, dict):
        for key in (
            'mean_luma',
            'std_luma',
            'overexposed_ratio',
            'underexposed_ratio',
            'effective_luma_ratio',
            'dynamic_range_p95_p05',
            'laplacian_variance',
        ):
            client_value = client_summary.get(key)
            if isinstance(client_value, (int, float)):
                abs_diff = float(abs(float(client_value) - float(server_summary[key])))
                drift[key] = {
                    'client': float(client_value),
                    'server': float(server_summary[key]),
                    'abs_diff': abs_diff,
                    'within_tolerance_1e-4': abs_diff <= 1e-4,
                }
                if abs_diff > 1e-4:
                    significant_drift_keys.append(key)

    if severe_reasons:
        status = 'block'
    elif warning_reasons:
        status = 'warn'
    else:
        status = 'pass'

    return {
        'status': status,
        'integrity_status': 'client_summary_drifted' if significant_drift_keys else 'consistent',
        'integrity_reasons': significant_drift_keys,
        'server_quality_summary': server_summary,
        'client_vs_server_drift': drift,
        'blocking_reasons': severe_reasons,
        'warning_reasons': warning_reasons,
    }


class DemoState:
    def __init__(self, bundle_dir: Path, device: str, class_names, upload_dir: Path, repo_root: Path):
        self.bundle_dir = bundle_dir.resolve()
        self.device = device
        self.class_names = list(class_names)
        self.finance_class_names = parse_class_names(
            os.environ.get('WEB_DEMO_FINANCE_CLASS_NAMES', '可疑欺诈,正常交易')
        )
        self.upload_dir = upload_dir.resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = repo_root.resolve()
        self.run_root = self.repo_root / 'artifacts' / 'web_demo_runs'
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.finance_bundle_dir = Path(
            os.environ.get('WEB_DEMO_FINANCE_BUNDLE_DIR', str(DEFAULT_FINANCE_BUNDLE_DIR))
        ).expanduser().resolve()
        self.finance_data_root = Path(
            os.environ.get('WEB_DEMO_FINANCE_DATA_ROOT', str(DEFAULT_FINANCE_DATA_ROOT))
        ).expanduser().resolve()
        self.bundle = None
        self.threshold = None
        self.sessions = {}
        self.e2e_profile = self.build_e2e_profile()
        self.finance_profile = self.build_finance_profile()
        self.max_upload_bytes = positive_int_from_env('WEB_DEMO_MAX_UPLOAD_MB', DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024
        self.command_timeout_sec = positive_int_from_env('WEB_DEMO_SECURE_TIMEOUT_SEC', DEFAULT_SECURE_TIMEOUT_SEC)
        self.finance_timeout_sec = positive_int_from_env('WEB_DEMO_FINANCE_TIMEOUT_SEC', DEFAULT_FINANCE_TIMEOUT_SEC)
        self.e2e_execution_mode = os.environ.get('WEB_DEMO_E2E_EXECUTION_MODE', 'local').strip().lower() or 'local'
        if self.e2e_execution_mode not in {'local', 'ssh'}:
            raise ValueError('WEB_DEMO_E2E_EXECUTION_MODE must be local or ssh')
        self.remote_ssh_target = os.environ.get('WEB_DEMO_REMOTE_SSH_TARGET', '').strip()
        self.remote_ssh_port = os.environ.get('WEB_DEMO_REMOTE_SSH_PORT', '22').strip() or '22'
        self.remote_ssh_user = os.environ.get('WEB_DEMO_REMOTE_SSH_USER', '').strip()
        self.remote_ssh_password = os.environ.get('WEB_DEMO_REMOTE_SSH_PASSWORD', '').strip()
        self.remote_repo_root = Path(
            os.environ.get('WEB_DEMO_REMOTE_REPO_ROOT', '/home/yclcg/Transshield_final')
        )
        self.remote_python_bin = os.environ.get(
            'WEB_DEMO_REMOTE_PYTHON_BIN',
            '/data/wyb/conda_envs/transshield/bin/python',
        )
        if self.e2e_execution_mode == 'ssh' and not self.remote_ssh_target:
            raise ValueError('WEB_DEMO_REMOTE_SSH_TARGET is required when WEB_DEMO_E2E_EXECUTION_MODE=ssh')
        self.finance_samples = self.build_finance_sample_library()
        self.finance_samples_by_id = {sample['id']: sample for sample in self.finance_samples}
        self.demo_summary = self.build_demo_summary()
        self.audit_dir = self.repo_root / 'artifacts' / 'web_demo_audit'
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_events_path = self.audit_dir / 'audit_events.jsonl'
        self.audit_rejections_path = self.audit_dir / 'audit_rejections.jsonl'
        self.audit_lock = threading.Lock()
        self.replay_lock = threading.Lock()
        self.recent_nonces = {}
        self.recent_payloads = {}
        self.nonce_expiry_heap = []
        self.payload_expiry_heap = []
        self.ip_shards = [
            {
                'lock': threading.Lock(),
                'states': OrderedDict(),
            }
            for _ in range(IP_SHARD_COUNT)
        ]
        self.global_inflight = threading.BoundedSemaphore(4)
        self.cleaner_stop = threading.Event()
        self.cleaner_threads = [
            threading.Thread(target=self._replay_guard_cleaner, name='web-demo-replay-cleaner', daemon=True),
            threading.Thread(target=self._ip_guard_cleaner, name='web-demo-ip-cleaner', daemon=True),
        ]
        for thread in self.cleaner_threads:
            thread.start()

    def build_finance_profile(self):
        profile = {
            'profile_name': 'finance_secret_uniform_live',
            'description': 'finance / party-local secret / secure internal pruning',
            'run_suffix': 'finance_secret_uniform_live',
            'candidate_basename': 'e2e_static_whole_forward_candidate_from_server',
            'static_depth_limit': -1,
            'spu_params_mode': 'secret',
            'spu_layer_norm_policy': 'exact',
            'spu_attention_policy': 'uniform',
            'spu_activation_override': 'bundle',
            'spu_activation_clip_value': 0.0,
            'spu_block_chunk_size': 0,
            'spu_layer_norm_chunk_size': 0,
            'spu_batch_size': positive_int_from_env('WEB_DEMO_FINANCE_E2E_SPU_BATCH_SIZE', 1),
            'party_local_share_load': True,
            'spu_runtime_reuse': bool_from_env('WEB_DEMO_REUSE_SPU_RUNTIME', False),
            'spu_disable_colocated_optimization': bool_from_env('SPU_DISABLE_COLOCATED_OPTIMIZATION', True),
            'ln_calibration_json': None,
            'output_calibration_json': None,
        }
        profile['candidate_pt_name'] = f"{profile['candidate_basename']}.pt"
        profile['candidate_json_name'] = f"{profile['candidate_basename']}.json"
        return profile

    def local_e2e_stack_available(self):
        required = [
            self.repo_root / 'tools' / 'transshield_e2e_secure_infer.py',
            self.repo_root / 'tools' / 'transshield_stage2_bundle.py',
        ]
        return all(path.exists() for path in required)

    def finance_live_supported(self):
        if not self.finance_samples:
            return False
        if self.e2e_execution_mode == 'ssh':
            return bool(self.remote_ssh_target)
        return self.local_e2e_stack_available()

    def build_finance_sample_library(self):
        categories = [('fraud', '欺诈交易'), ('normal', '正常交易')]
        per_class = positive_int_from_env(
            'WEB_DEMO_FINANCE_SAMPLES_PER_CLASS',
            DEFAULT_FINANCE_SAMPLES_PER_CLASS,
        )
        samples = []
        for category, category_label in categories:
            category_dir = self.finance_data_root / category
            if not category_dir.exists():
                continue
            for index, path in enumerate(sorted(category_dir.glob('*.png'))[:per_class], start=1):
                ground_truth_index = 0 if category == 'fraud' else 1
                sample_id = f'{category}-{path.stem}'
                samples.append(
                    {
                        'id': sample_id,
                        'label': f'{category_label}样本 {index}',
                        'category': category,
                        'category_label': category_label,
                        'ground_truth_index': ground_truth_index,
                        'ground_truth_label': self.finance_class_names[ground_truth_index],
                        'path': path.resolve(),
                        'relative_path': str(path.resolve().relative_to(self.repo_root)),
                        'preview_url': f'/samples/finance/{sample_id}',
                    }
                )
        return samples

    def build_demo_summary(self):
        summary = load_demo_summary()
        items = ((summary.get('showcase_domains') or {}).get('items') or [])
        for item in items:
            if item.get('id') == 'medical':
                item['live_demo_supported'] = True
                item['live_demo_mode'] = 'browser_private_shares'
                item['demo_note'] = '浏览器本地完成控制面预检与分片，再触发一次完整隐私推理；正式落地时由医院侧服务器和 AI 公司侧服务器两方协同执行。'
            elif item.get('id') == 'finance':
                item['eyebrow'] = 'Boundary Stress Test'
                item['live_demo_supported'] = self.finance_live_supported()
                item['live_demo_mode'] = 'builtin_sample_secure_run'
                item['summary'] = (
                    '金融特征先编码为图像，再沿用同一套动态安全剪枝与双向隐私方法。'
                    '这一域只作为边界压力测试：重点观察极端分布输入、稀疏样本和动态路由稳定性。'
                    '页面支持从内置压力样本中触发一次完整隐私推理，同时保留已经完成的批量实测结果作为压力测试对照。'
                )
                item['sample_library'] = {
                    'title': '金融边界压力样本',
                    'summary': (
                        '选择一条内置压力样本，直接触发一次真实完整隐私推理。'
                        if self.finance_live_supported()
                        else '当前环境未配置远端完整隐私运行环境，因此金融域先展示已验证的边界压力结果。'
                    ),
                    'items': [public_sample_metadata(sample) for sample in self.finance_samples],
                }
                item['display_notes'] = [
                    '金融域支持现场触发一次真实完整隐私推理，但只作为边界压力验证。',
                    '正式部署参与方固定为银行侧服务器与 AI 公司侧服务器两方。',
                    '页面同时保留批量动态实测结果，便于和单次压力样本结果对照理解。',
                ]
        if summary.get('showcase_domains'):
            summary['showcase_domains']['summary'] = (
                '这页展示一条正式主线和一条压力测试线：医疗支持浏览器本地控制面演示；金融只保留边界压力验证。'
            )
        summary['runtime_capability'] = {
            'medical_live_demo': True,
            'finance_live_demo': self.finance_live_supported(),
            'execution_mode': self.e2e_execution_mode,
        }
        return summary

    def record_audit_event(self, accepted: bool, payload: dict):
        path = self.audit_events_path if accepted else self.audit_rejections_path
        with self.audit_lock:
            with path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')

    def shard_index_for_ip(self, ip: str) -> int:
        return hashlib.blake2s(ip.encode('utf-8'), digest_size=1).digest()[0] % IP_SHARD_COUNT

    def check_ip_rate_limit(self, ip: str):
        now = time.monotonic()
        shard_index = self.shard_index_for_ip(ip)
        shard = self.ip_shards[shard_index]
        with shard['lock']:
            states = shard['states']
            state = states.get(ip)
            if state is None:
                if len(states) >= IP_SHARD_CAPACITY and not self._evict_idle_ip_locked(states, now, attempts=8):
                    return False, 'ip_state_saturated'
                state = IpState()
                states[ip] = state
            while state.window and state.window[0] <= now - IP_WINDOW_SEC:
                state.window.popleft()
            if len(state.window) >= IP_WINDOW_LIMIT:
                state.last_seen_monotonic = now
                states.move_to_end(ip)
                return False, 'rate_limited_ip'
            state.window.append(now)
            state.last_seen_monotonic = now
            states.move_to_end(ip)
            return True, None

    def reserve_ip_inflight(self, ip: str):
        now = time.monotonic()
        shard_index = self.shard_index_for_ip(ip)
        shard = self.ip_shards[shard_index]
        with shard['lock']:
            states = shard['states']
            state = states.get(ip)
            if state is None:
                if len(states) >= IP_SHARD_CAPACITY and not self._evict_idle_ip_locked(states, now, attempts=8):
                    return None, 'ip_state_saturated'
                state = IpState()
                states[ip] = state
            if state.inflight >= IP_INFLIGHT_LIMIT:
                state.last_seen_monotonic = now
                states.move_to_end(ip)
                return None, 'busy_retry_later'
            state.inflight += 1
            state.last_seen_monotonic = now
            states.move_to_end(ip)
        if not self.global_inflight.acquire(blocking=False):
            with shard['lock']:
                state = shard['states'].get(ip)
                if state is not None and state.inflight > 0:
                    state.inflight -= 1
                    state.last_seen_monotonic = time.monotonic()
                    shard['states'].move_to_end(ip)
            return None, 'busy_retry_later'
        return shard_index, None

    def release_ip_inflight(self, ip: str):
        shard = self.ip_shards[self.shard_index_for_ip(ip)]
        with shard['lock']:
            state = shard['states'].get(ip)
            if state is not None and state.inflight > 0:
                state.inflight -= 1
                state.last_seen_monotonic = time.monotonic()
                shard['states'].move_to_end(ip)
        try:
            self.global_inflight.release()
        except ValueError:
            return

    def _evict_idle_ip_locked(self, states: OrderedDict, now: float, attempts: int) -> bool:
        for _ in range(attempts):
            if not states:
                return True
            oldest_ip, oldest_state = next(iter(states.items()))
            while oldest_state.window and oldest_state.window[0] <= now - IP_WINDOW_SEC:
                oldest_state.window.popleft()
            if (
                oldest_state.inflight == 0
                and not oldest_state.window
                and oldest_state.last_seen_monotonic <= now - IP_STALE_TTL_SEC
            ):
                states.popitem(last=False)
                return True
            states.move_to_end(oldest_ip)
        return False

    def check_and_remember_replay(self, audit_nonce: str, payload_fingerprint: str):
        now = time.monotonic()
        with self.replay_lock:
            nonce_expiry = self.recent_nonces.get(audit_nonce)
            if nonce_expiry is not None:
                if nonce_expiry > now:
                    return False, 'duplicate_nonce'
                self.recent_nonces.pop(audit_nonce, None)
            payload_expiry = self.recent_payloads.get(payload_fingerprint)
            if payload_expiry is not None:
                if payload_expiry > now:
                    return False, 'duplicate_payload'
                self.recent_payloads.pop(payload_fingerprint, None)
            if len(self.recent_nonces) >= REPLAY_GUARD_MAX_ITEMS:
                self._opportunistic_replay_cleanup_locked(now)
            if len(self.recent_payloads) >= REPLAY_GUARD_MAX_ITEMS:
                self._opportunistic_replay_cleanup_locked(now)
            if len(self.recent_nonces) >= REPLAY_GUARD_MAX_ITEMS or len(self.recent_payloads) >= REPLAY_GUARD_MAX_ITEMS:
                return False, 'guard_cache_saturated'
            nonce_deadline = now + REPLAY_NONCE_TTL_SEC
            payload_deadline = now + REPLAY_PAYLOAD_TTL_SEC
            self.recent_nonces[audit_nonce] = nonce_deadline
            self.recent_payloads[payload_fingerprint] = payload_deadline
            heapq.heappush(self.nonce_expiry_heap, (nonce_deadline, audit_nonce))
            heapq.heappush(self.payload_expiry_heap, (payload_deadline, payload_fingerprint))
            return True, None

    def _opportunistic_replay_cleanup_locked(self, now: float):
        for _ in range(8):
            if self.nonce_expiry_heap and self.nonce_expiry_heap[0][0] <= now:
                expiry, nonce = heapq.heappop(self.nonce_expiry_heap)
                if self.recent_nonces.get(nonce) == expiry:
                    self.recent_nonces.pop(nonce, None)
                continue
            if self.payload_expiry_heap and self.payload_expiry_heap[0][0] <= now:
                expiry, key = heapq.heappop(self.payload_expiry_heap)
                if self.recent_payloads.get(key) == expiry:
                    self.recent_payloads.pop(key, None)
                continue
            break

    def _replay_guard_cleaner(self):
        while not self.cleaner_stop.wait(5.0):
            now = time.monotonic()
            while True:
                deleted = 0
                self.replay_lock.acquire()
                try:
                    for _ in range(REPLAY_GUARD_EVICT_BATCH):
                        if self.nonce_expiry_heap and self.nonce_expiry_heap[0][0] <= now:
                            expiry, nonce = heapq.heappop(self.nonce_expiry_heap)
                            if self.recent_nonces.get(nonce) == expiry:
                                self.recent_nonces.pop(nonce, None)
                                deleted += 1
                            continue
                        if self.payload_expiry_heap and self.payload_expiry_heap[0][0] <= now:
                            expiry, key = heapq.heappop(self.payload_expiry_heap)
                            if self.recent_payloads.get(key) == expiry:
                                self.recent_payloads.pop(key, None)
                                deleted += 1
                            continue
                        break
                finally:
                    self.replay_lock.release()
                if deleted == 0:
                    break
                time.sleep(0.01)

    def _ip_guard_cleaner(self):
        shard_cursor = 0
        while not self.cleaner_stop.wait(5.0):
            for _ in range(IP_SHARD_COUNT):
                if self.cleaner_stop.is_set():
                    return
                shard = self.ip_shards[shard_cursor]
                shard_cursor = (shard_cursor + 1) % IP_SHARD_COUNT
                deleted = 0
                now = time.monotonic()
                shard['lock'].acquire()
                try:
                    states = shard['states']
                    for _ in range(IP_GUARD_EVICT_BATCH):
                        if not states:
                            break
                        ip, state = next(iter(states.items()))
                        while state.window and state.window[0] <= now - IP_WINDOW_SEC:
                            state.window.popleft()
                        if state.inflight == 0 and not state.window and state.last_seen_monotonic <= now - IP_STALE_TTL_SEC:
                            states.popitem(last=False)
                            deleted += 1
                            continue
                        break
                finally:
                    shard['lock'].release()
                if deleted:
                    time.sleep(0.01)

    def ensure_bundle(self):
        if self.bundle is None:
            from tools.transshield_stage2_bundle import load_frozen_bundle, resolve_threshold

            self.bundle = load_frozen_bundle(self.bundle_dir, self.device)
            self.threshold = resolve_threshold(self.bundle_dir, None)
        return self.bundle

    def analyze_image(self, image_path: Path):
        from tools.transshield_pruning_trace import collect_predictor_outputs, reconstruct_eval_masks
        from tools.transshield_stage2_bundle import postprocess_binary_output, preprocess_image

        bundle = self.ensure_bundle()
        image_path, input_tensor = preprocess_image(image_path, bundle['transform'], self.device)
        logits, probs, predictor_outputs = collect_predictor_outputs(bundle['model'], input_tensor)
        base_rate = float(bundle['args_snapshot']['base_rate'])
        token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
        trace = reconstruct_eval_masks(predictor_outputs, token_ratio)
        prediction = postprocess_binary_output(probs, threshold=self.threshold)
        return {
            'image_path': str(image_path),
            'probabilities': [float(value) for value in probs.squeeze(0).detach().cpu().tolist()],
            'logits': [float(value) for value in logits.squeeze(0).detach().cpu().tolist()],
            'prediction': prediction,
            'pruning_trace': trace,
        }

    def build_e2e_profile(self):
        profile_name = os.environ.get('WEB_DEMO_E2E_PROFILE', DEFAULT_E2E_PROFILE).strip() or DEFAULT_E2E_PROFILE
        profile_defs = {
            'secret_depth6_clip0_showcase': {
                'description': 'secret_blockwise_stage / depth6 / clip0 showcase',
                'run_suffix': 'secret_depth6_clip0',
                'static_depth_limit': 6,
                'spu_params_mode': 'secret_blockwise_stage',
                'spu_layer_norm_policy': 'public_calibrated',
                'spu_attention_policy': 'uniform',
                'spu_activation_override': 'fixed_square',
                'spu_activation_clip_value': 0.0,
                'spu_block_chunk_size': 0,
                'spu_layer_norm_chunk_size': 0,
                'spu_batch_size': 1,
                'party_local_share_load': True,
                'spu_runtime_reuse': False,
                'spu_disable_colocated_optimization': True,
                'candidate_basename': 'e2e_static_whole_forward_candidate_spu_depth6_partylocal_publiccalibln_uniform_fixed_square_clip0_eval',
                'ln_calibration_json': self.repo_root / 'artifacts' / 'server_pipeline_run' / 'secure_static_depth12_epoch8_secret_depth_boundary_calib_clip0_20260430' / 'e2e_secure_poc' / 'e2e_public_layer_norm_calibration_depth6_uniform_fixed_square_clip0.json',
                'output_calibration_json': self.repo_root / 'artifacts' / 'server_pipeline_run' / 'e2e_output_calibration_secret_depth6_clip0_balanced8_20260502.json',
            },
            'public_depth12_clip3_showcase': {
                'description': 'public params / depth12 / clip3 showcase',
                'run_suffix': 'public_depth12_clip3',
                'static_depth_limit': 12,
                'spu_params_mode': 'public',
                'spu_layer_norm_policy': 'public_calibrated',
                'spu_attention_policy': 'uniform',
                'spu_activation_override': 'fixed_square',
                'spu_activation_clip_value': 3.0,
                'spu_block_chunk_size': 0,
                'spu_layer_norm_chunk_size': 0,
                'spu_batch_size': 1,
                'party_local_share_load': True,
                'spu_runtime_reuse': False,
                'spu_disable_colocated_optimization': True,
                'candidate_basename': 'e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_uniform_fixed_square_clip3_eval',
                'ln_calibration_json': self.repo_root / 'artifacts' / 'server_pipeline_run' / 'secure_static_depth12_epoch8_publicraw_balanced8_clip3_20260430' / 'e2e_secure_poc' / 'e2e_public_layer_norm_calibration_depth12_uniform_fixed_square_clip3p0.json',
                'output_calibration_json': self.repo_root / 'artifacts' / 'server_pipeline_run' / 'e2e_output_calibration_secure_static_depth12_epoch8_clip3_balanced8_20260430.json',
            },
        }
        if profile_name not in profile_defs:
            raise ValueError(f'unsupported WEB_DEMO_E2E_PROFILE: {profile_name}')
        profile = dict(profile_defs[profile_name])
        profile['profile_name'] = profile_name
        profile['static_depth_limit'] = positive_int_from_env(
            'WEB_DEMO_E2E_STATIC_DEPTH_LIMIT',
            int(profile['static_depth_limit']),
        )
        profile['spu_batch_size'] = positive_int_from_env(
            'WEB_DEMO_E2E_SPU_BATCH_SIZE',
            int(profile['spu_batch_size']),
        )
        profile['spu_block_chunk_size'] = positive_int_from_env(
            'WEB_DEMO_E2E_SPU_BLOCK_CHUNK_SIZE',
            int(profile['spu_block_chunk_size']),
        )
        profile['spu_layer_norm_chunk_size'] = positive_int_from_env(
            'WEB_DEMO_E2E_SPU_LAYER_NORM_CHUNK_SIZE',
            int(profile['spu_layer_norm_chunk_size']),
        )
        profile['spu_params_mode'] = os.environ.get('WEB_DEMO_E2E_PARAMS_MODE', profile['spu_params_mode'])
        profile['spu_layer_norm_policy'] = os.environ.get(
            'WEB_DEMO_E2E_LAYER_NORM_POLICY',
            profile['spu_layer_norm_policy'],
        )
        profile['spu_attention_policy'] = os.environ.get(
            'WEB_DEMO_E2E_ATTENTION_POLICY',
            profile['spu_attention_policy'],
        )
        profile['spu_activation_override'] = os.environ.get(
            'WEB_DEMO_E2E_ACTIVATION_OVERRIDE',
            profile['spu_activation_override'],
        )
        profile['spu_activation_clip_value'] = float_from_env(
            'WEB_DEMO_E2E_ACTIVATION_CLIP_VALUE',
            float(profile['spu_activation_clip_value']),
        )
        profile['party_local_share_load'] = bool_from_env(
            'WEB_DEMO_E2E_PARTY_LOCAL_SHARE_LOAD',
            bool(profile['party_local_share_load']),
        )
        profile['spu_runtime_reuse'] = bool_from_env(
            'WEB_DEMO_REUSE_SPU_RUNTIME',
            bool(profile['spu_runtime_reuse']),
        )
        profile['spu_disable_colocated_optimization'] = bool_from_env(
            'SPU_DISABLE_COLOCATED_OPTIMIZATION',
            bool(profile['spu_disable_colocated_optimization']),
        )
        profile['candidate_basename'] = os.environ.get(
            'WEB_DEMO_E2E_CANDIDATE_BASENAME',
            profile['candidate_basename'],
        )
        profile['candidate_pt_name'] = f"{profile['candidate_basename']}.pt"
        profile['candidate_json_name'] = f"{profile['candidate_basename']}.json"
        profile['ln_calibration_json'] = Path(
            os.environ.get('WEB_DEMO_E2E_LN_CALIBRATION_JSON', str(profile['ln_calibration_json']))
        ).expanduser().resolve()
        profile['output_calibration_json'] = Path(
            os.environ.get('WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON', str(profile['output_calibration_json']))
        ).expanduser().resolve()
        profile['activation_clip_tag'] = clip_tag(float(profile['spu_activation_clip_value']))
        return profile

    def remote_target_parts(self):
        target = self.remote_ssh_target.strip()
        user = self.remote_ssh_user
        host = target
        if '@' in target:
            user, host = target.rsplit('@', 1)
        if not user:
            user = os.environ.get('USER', '').strip()
        if not host:
            raise ValueError('missing WEB_DEMO_REMOTE_SSH_TARGET host')
        return user, host

    def open_remote_client(self):
        import paramiko

        user, host = self.remote_target_parts()
        connect_kwargs = {
            'hostname': host,
            'port': int(self.remote_ssh_port),
            'username': user,
            'timeout': 20,
            'banner_timeout': 20,
            'auth_timeout': 20,
        }
        if self.remote_ssh_password:
            connect_kwargs['password'] = self.remote_ssh_password
            connect_kwargs['look_for_keys'] = False
            connect_kwargs['allow_agent'] = False
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(**connect_kwargs)
        return client

    def ensure_remote_dir(self, sftp, remote_dir: Path):
        remote_dir = Path(remote_dir)
        parts = remote_dir.as_posix().strip('/').split('/')
        current = '/'
        for part in parts:
            current = f'{current.rstrip("/")}/{part}'
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)

    def sftp_put_tree(self, sftp, local_dir: Path, remote_dir: Path):
        local_dir = Path(local_dir).resolve()
        remote_dir = Path(remote_dir)
        self.ensure_remote_dir(sftp, remote_dir)
        for root, dirnames, filenames in os.walk(local_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(local_dir)
            remote_root = remote_dir / rel_root
            self.ensure_remote_dir(sftp, remote_root)
            for dirname in dirnames:
                self.ensure_remote_dir(sftp, remote_root / dirname)
            for filename in filenames:
                local_path = root_path / filename
                remote_path = remote_root / filename
                sftp.put(str(local_path), remote_path.as_posix())

    def sftp_get_tree(self, sftp, remote_dir: Path, local_dir: Path):
        remote_dir = Path(remote_dir)
        local_dir = Path(local_dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir.as_posix()):
            remote_path = remote_dir / entry.filename
            local_path = local_dir / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                self.sftp_get_tree(sftp, remote_path, local_path)
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path.as_posix(), str(local_path))

    def run_remote_command(self, client, command: str, log_path: Path, step_name: str, timeout_sec: int):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        channel = stdout.channel
        collected = []
        with log_path.open('w', encoding='utf-8') as handle:
            while True:
                while channel.recv_ready():
                    chunk = channel.recv(4096).decode('utf-8', errors='replace')
                    collected.append(chunk)
                    handle.write(chunk)
                    handle.flush()
                while channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                    collected.append(chunk)
                    handle.write(chunk)
                    handle.flush()
                if channel.exit_status_ready():
                    break
                if time.time() - started_at > timeout_sec:
                    channel.close()
                    raise TimeoutError(
                        f'{step_name} timed out after {timeout_sec}s\n'
                        f'log_path={log_path}\n'
                        f'log_tail:\n{"".join(collected)[-4000:]}'
                    )
                time.sleep(0.2)
            while channel.recv_ready():
                chunk = channel.recv(4096).decode('utf-8', errors='replace')
                collected.append(chunk)
                handle.write(chunk)
            while channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                collected.append(chunk)
                handle.write(chunk)
        returncode = channel.recv_exit_status()
        if returncode != 0:
            raise RuntimeError(
                f'{step_name} failed with returncode={returncode}\n'
                f'log_path={log_path}\n'
                f'log_tail:\n{"".join(collected)[-4000:]}'
            )

    def create_pruning_visual_assets(self, session_id: str, analysis: dict):
        from tools.transshield_token_pruning_visualization import (
            draw_mask_overlay,
            draw_original_panel,
            fit_display_image,
            make_contact_sheet,
        )

        image_path = Path(analysis['image_path'])
        base_image = fit_display_image(image_path, 224)
        original_panel = draw_original_panel(
            base_image,
            title='Original Image',
            subtitle='Resized input for patch visualization',
        )
        original_name = f'{session_id}_original_panel.png'
        original_path = self.upload_dir / original_name
        original_panel.save(original_path)

        panels = [original_panel]
        overlay_urls = []
        for stage in analysis['pruning_trace']['stages']:
            keep_count = int(stage['active_after_per_sample'][0])
            total_count = int(analysis['pruning_trace']['init_spatial_tokens'])
            density = float(stage['active_after_density_per_sample'][0])
            panel = draw_mask_overlay(
                base_image=base_image,
                mask_grid=stage['first_sample_mask_grid'],
                title=f"Stage {stage['stage_index'] + 1} | layer {stage['pruning_layer']}",
                subtitle=f"keep={keep_count}/{total_count} ({density:.2%})",
            )
            panel_name = f"{session_id}_stage_{stage['stage_index'] + 1}_overlay.png"
            panel_path = self.upload_dir / panel_name
            panel.save(panel_path)
            panels.append(panel)
            overlay_urls.append(f"/uploads/{panel_name}")

        summary_board = make_contact_sheet(panels, columns=2)
        summary_name = f'{session_id}_token_pruning_summary.png'
        summary_path = self.upload_dir / summary_name
        summary_board.save(summary_path)

        return {
            'summary_board_url': f'/uploads/{summary_name}',
            'original_panel_url': f'/uploads/{original_name}',
            'overlay_urls': overlay_urls,
        }

    def store_session(self, session_id: str, upload_name: str, analysis: dict, pruning_assets: dict):
        self.sessions[session_id] = {
            'image_name': upload_name,
            'analysis': analysis,
            'pruning_assets': pruning_assets,
        }
        return self.sessions[session_id]

    def build_secure_run_context(self, session_id: str, runtime: str):
        session = self.sessions[session_id]
        image_path = Path(session['analysis']['image_path']).resolve()
        run_name = f'web_demo_{session_id}_{runtime}'
        secure_run_dir = self.run_root / run_name
        secure_run_dir.mkdir(parents=True, exist_ok=True)
        suite_log_path = secure_run_dir / f'web_demo_{runtime}_suite.log'
        profile_log_path = secure_run_dir / f'web_demo_{runtime}_profile.log'
        return {
            'session': session,
            'image_path': image_path,
            'run_name': run_name,
            'secure_run_dir': secure_run_dir,
            'suite_log_path': suite_log_path,
            'profile_log_path': profile_log_path,
        }

    def build_secure_env(self, image_path: Path, run_name: str, secure_run_dir: Path, runtime: str):
        env = os.environ.copy()
        env.update(
            {
                'REPO_ROOT': str(self.repo_root),
                'PYTHON_BIN': sys.executable,
                'BUNDLE_DIR': str(self.bundle_dir),
                'PLAINTEXT_EVAL_DEVICE': self.device,
                'SECURE_EXPORT_DEVICE': self.device,
                'INPUT_IMAGE': str(image_path),
                'RUN_NAME': run_name,
                'SECURE_RUN_DIR': str(secure_run_dir),
                'SECURE_RUNTIME': runtime,
                'CLASS_NAMES': ','.join(self.class_names),
            }
        )
        if runtime == 'spu':
            env['SPU_RUNTIME_REUSE'] = os.environ.get('WEB_DEMO_REUSE_SPU_RUNTIME', '1')
        env['SKIP_PIPELINE_VERIFY'] = os.environ.get('WEB_DEMO_SKIP_PIPELINE_VERIFY', '1')
        return env

    def build_profiled_e2e_run_context(self, run_name: str, profile: dict, image_path: Optional[Path] = None, session: Optional[dict] = None):
        secure_run_dir = self.run_root / run_name
        e2e_run_dir = secure_run_dir / 'e2e_secure_poc'
        e2e_run_dir.mkdir(parents=True, exist_ok=True)
        run_relative_dir = secure_run_dir.relative_to(self.repo_root)
        remote_secure_run_dir = self.remote_repo_root / run_relative_dir
        remote_e2e_run_dir = remote_secure_run_dir / 'e2e_secure_poc'
        return {
            'session': session,
            'image_path': None if image_path is None else Path(image_path).resolve(),
            'run_name': run_name,
            'secure_run_dir': secure_run_dir,
            'e2e_run_dir': e2e_run_dir,
            'share_log_path': secure_run_dir / 'web_demo_e2e_share_preprocess.log',
            'remote_secure_run_dir': remote_secure_run_dir,
            'remote_e2e_run_dir': remote_e2e_run_dir,
            'calib_log_path': secure_run_dir / 'web_demo_e2e_calibration.log',
            'infer_log_path': secure_run_dir / 'web_demo_browser_e2e_infer.log',
            'share_prefix': e2e_run_dir / 'client_pixel_values_debug_share',
            'share_manifest_json': e2e_run_dir / 'client_pixel_values_debug_share_manifest.json',
            'share_public_json': e2e_run_dir / 'client_pixel_values_debug_share_public_manifest.json',
            'share_party_dir': e2e_run_dir / 'client_pixel_values_debug_share_party_manifests',
            'candidate_pt': e2e_run_dir / profile['candidate_pt_name'],
            'candidate_json': e2e_run_dir / profile['candidate_json_name'],
        }

    def build_e2e_run_context(self, session_id: str):
        session = self.sessions[session_id]
        image_path = Path(session['analysis']['image_path']).resolve()
        run_name = f"web_demo_{session_id}_e2e_{self.e2e_profile['run_suffix']}"
        return self.build_profiled_e2e_run_context(
            run_name=run_name,
            profile=self.e2e_profile,
            image_path=image_path,
            session=session,
        )

    def build_browser_e2e_run_context(self, job_id: str):
        run_name = f"web_demo_browser_e2e_{job_id}_{self.e2e_profile['run_suffix']}"
        return self.build_profiled_e2e_run_context(run_name=run_name, profile=self.e2e_profile)

    def build_finance_sample_run_context(self, sample: dict):
        run_name = f"web_demo_finance_{sample['id']}_{uuid.uuid4().hex[:8]}_{self.finance_profile['run_suffix']}"
        return self.build_profiled_e2e_run_context(
            run_name=run_name,
            profile=self.finance_profile,
            image_path=sample['path'],
        )

    def remote_path_for(self, path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        try:
            relative = path.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f'cannot map non-repo path to remote repo root: {path}') from exc
        return self.remote_repo_root / relative

    def execution_path(self, path: Path) -> Path:
        if self.e2e_execution_mode == 'ssh':
            return self.remote_path_for(path)
        return Path(path).expanduser().resolve()

    def context_execution_path(self, context: dict, key: str) -> Path:
        if self.e2e_execution_mode == 'ssh':
            mapping = {
                'secure_run_dir': 'remote_secure_run_dir',
                'e2e_run_dir': 'remote_e2e_run_dir',
            }
            if key in mapping:
                return context[mapping[key]]
        return context[key]

    def write_browser_e2e_share_payloads(self, context: dict, share0_bytes: bytes, share1_bytes: bytes):
        if len(share0_bytes) != E2E_SHARE_BYTE_COUNT or len(share1_bytes) != E2E_SHARE_BYTE_COUNT:
            raise ValueError(
                f'invalid share byte size: got {len(share0_bytes)} and {len(share1_bytes)}, '
                f'expected {E2E_SHARE_BYTE_COUNT} bytes each'
            )
        share_paths = [
            context['e2e_run_dir'] / 'browser_client_pixel_values_debug_share_p1_share.float32le',
            context['e2e_run_dir'] / 'browser_client_pixel_values_debug_share_p2_share.float32le',
        ]
        manifest_share_paths = [
            self.execution_path(share_paths[0]),
            self.execution_path(share_paths[1]),
        ]
        sample_ids = ['browser_sample_000000']
        for share_path, share_bytes in zip(share_paths, [share0_bytes, share1_bytes]):
            share_path.write_bytes(share_bytes)

        public_manifest = {
            'manifest_type': 'transshield_e2e_debug_float_additive_share_public_manifest_v0',
            'share_count': 2,
            'party_ids': ['P1', 'P2'],
            'share_semantics': 'debug_float_additive_share_not_production_mpc_share',
            'share_dtype': 'torch.float32',
            'share_shape': E2E_SHARE_SHAPE,
            'sample_count': 1,
            'sample_ids': sample_ids,
            'targets_included': False,
            'source_paths_included': False,
            'private_share_paths_included': False,
            'privacy_status': (
                'browser_generated_split_shares; original image and plaintext pixel_values are not uploaded '
                'to the web backend'
            ),
        }
        context['share_public_json'].write_text(json.dumps(public_manifest, indent=2, sort_keys=True), encoding='utf-8')

        context['share_party_dir'].mkdir(parents=True, exist_ok=True)
        for rank, party_id in enumerate(['P1', 'P2']):
            party_manifest = {
                'manifest_type': 'transshield_e2e_debug_float_additive_share_party_manifest_v0',
                'party_id': party_id,
                'share_rank': rank,
                'share_count': 2,
                'share_path': str(manifest_share_paths[rank]),
                'share_storage_format': 'raw_float32_le',
                'public_manifest_json': str(self.execution_path(context['share_public_json'])),
                'share_semantics': public_manifest['share_semantics'],
                'share_dtype': public_manifest['share_dtype'],
                'share_shape': public_manifest['share_shape'],
                'sample_count': 1,
                'sample_ids': sample_ids,
                'privacy_status': 'browser_share_upload_for_demo; production must store this only on its owning party',
            }
            (context['share_party_dir'] / f'{party_id.lower()}_share_manifest.json').write_text(
                json.dumps(party_manifest, indent=2, sort_keys=True),
                encoding='utf-8',
            )

    def run_e2e_approx_for_browser_shares(self, share0_bytes: bytes, share1_bytes: bytes):
        job_id = uuid.uuid4().hex
        context = self.build_browser_e2e_run_context(job_id)
        self.write_browser_e2e_share_payloads(context, share0_bytes, share1_bytes)
        calibration_json, calibration_exists = self.resolve_e2e_calibration_path(context['e2e_run_dir'])
        output_calibration_json = self.resolve_e2e_output_calibration_path()
        if output_calibration_json is None:
            raise RuntimeError(
                'E2E output calibration JSON is missing for the selected web demo profile.'
            )
        env = self.build_e2e_env(
            context,
            profile=self.e2e_profile,
            bundle_dir=self.bundle_dir,
            calibration_json=calibration_json,
            output_calibration_json=output_calibration_json,
        )
        scripts = self.secure_script_paths()
        if not calibration_exists:
            if not bool_from_env('WEB_DEMO_AUTO_CALIBRATE_E2E', False):
                raise RuntimeError(
                    'E2E public layer norm calibration JSON is missing. '
                    'Pre-generate it on the server, or set WEB_DEMO_AUTO_CALIBRATE_E2E=1 for an explicit debug run.'
                )
            raise RuntimeError(
                'Web demo currently expects a pre-generated layer norm calibration JSON for the selected E2E profile.'
            )
        if self.e2e_execution_mode == 'ssh':
            self.run_remote_e2e_whole_forward(context, env, scripts['whole_forward'])
        else:
            self._run_command_with_log(
                ['bash', str(scripts['whole_forward']), 'spu'],
                cwd=self.repo_root,
                env=env,
                log_path=context['infer_log_path'],
                step_name='browser_e2e_secure_whole_forward_spu',
            )
        return self.load_profiled_e2e_result(
            context,
            profile=self.e2e_profile,
            class_names=self.class_names,
            calibration_json=calibration_json,
            output_calibration_json=output_calibration_json,
        )

    def validate_medical_control_plane_payload(
        self,
        share0_bytes: bytes,
        share1_bytes: bytes,
        client_quality_summary: Optional[dict],
        client_audit_manifest: dict,
        client_control_plane_metrics: Optional[dict],
    ):
        if len(share0_bytes) != E2E_SHARE_BYTE_COUNT or len(share1_bytes) != E2E_SHARE_BYTE_COUNT:
            raise ValueError('share payload size is invalid')

        if not isinstance(client_quality_summary, dict):
            raise ValueError('client_quality_summary must be a JSON object')
        if not isinstance(client_audit_manifest, dict):
            raise ValueError('client_audit_manifest must be a JSON object')
        if client_control_plane_metrics is not None and not isinstance(client_control_plane_metrics, dict):
            raise ValueError('client_control_plane_metrics must be a JSON object')
        validate_quality_summary_object(client_quality_summary)
        validate_control_plane_metrics_object(client_control_plane_metrics)

        share0_sha256 = server_sha256_hex(share0_bytes)
        share1_sha256 = server_sha256_hex(share1_bytes)
        client_share0_sha256 = ensure_sha256_hex('share0_sha256', client_audit_manifest.get('share0_sha256'))
        client_share1_sha256 = ensure_sha256_hex('share1_sha256', client_audit_manifest.get('share1_sha256'))
        client_source_image_sha256 = ensure_sha256_hex(
            'source_image_sha256',
            client_audit_manifest.get('source_image_sha256'),
        )
        client_normalized_tensor_sha256 = ensure_sha256_hex(
            'normalized_tensor_sha256',
            client_audit_manifest.get('normalized_tensor_sha256'),
        )
        client_audit_chain_sha256 = ensure_sha256_hex(
            'audit_chain_sha256',
            client_audit_manifest.get('audit_chain_sha256'),
        )
        audit_nonce = str(client_audit_manifest.get('audit_nonce') or '').strip()
        if not audit_nonce or len(audit_nonce) > 128:
            raise ValueError('audit_nonce is missing or too long')
        if client_share0_sha256 != share0_sha256:
            raise RuntimeError('share0 sha256 mismatch')
        if client_share1_sha256 != share1_sha256:
            raise RuntimeError('share1 sha256 mismatch')
        expected_audit_chain_sha256 = hashlib.sha256(
            (
                f'v7|{audit_nonce}|{client_source_image_sha256}|'
                f'{client_normalized_tensor_sha256}|{share0_sha256}|{share1_sha256}'
            ).encode('utf-8')
        ).hexdigest()
        if client_audit_chain_sha256 != expected_audit_chain_sha256:
            raise RuntimeError('audit chain sha256 mismatch')

        share0_u32 = bytes_to_float32_aligned(share0_bytes)
        share1_u32 = bytes_to_float32_aligned(share1_bytes)
        if contains_subnormal_values(share0_u32) or contains_subnormal_values(share1_u32):
            raise RuntimeError('subnormal share values detected')

        share0_f32 = share0_u32.view('<f4')
        share1_f32 = share1_u32.view('<f4')
        flush_tiny_values_inplace(share0_f32)
        flush_tiny_values_inplace(share1_f32)
        if not np.isfinite(share0_f32).all() or not np.isfinite(share1_f32).all():
            raise RuntimeError('share contains non-finite values')
        if float(np.max(np.abs(share0_f32))) > 1e3 or float(np.max(np.abs(share1_f32))) > 1e3:
            raise RuntimeError('share magnitude exceeds allowed bound')

        reconstructed = np.add(share0_f32, share1_f32, dtype=np.float32).reshape(E2E_SHARE_SHAPE)
        if not np.isfinite(reconstructed).all():
            raise RuntimeError('reconstructed tensor contains non-finite values')

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        rgb = reconstructed * std + mean
        rgb_min = float(np.min(rgb))
        rgb_max = float(np.max(rgb))
        if rgb_min < -0.05 or rgb_max > 1.05:
            raise RuntimeError('reconstructed rgb range is invalid')

        server_quality_summary = compute_luma_metrics(rgb)
        quality_assurance = build_quality_assurance(server_quality_summary, client_quality_summary)
        if quality_assurance['status'] == 'block':
            raise RuntimeError('quality assurance blocked request')

        client_metrics = client_control_plane_metrics if isinstance(client_control_plane_metrics, dict) else {}
        audit = {
            'audit_nonce': audit_nonce,
            'client_source_image_sha256': client_source_image_sha256,
            'client_normalized_tensor_sha256': client_normalized_tensor_sha256,
            'client_audit_chain_sha256': client_audit_chain_sha256,
            'server_audit_chain_sha256': expected_audit_chain_sha256,
            'audit_chain_consistent': True,
            'server_share0_sha256': share0_sha256,
            'server_share1_sha256': share1_sha256,
            'server_payload_fingerprint': make_payload_fingerprint(share0_bytes, share1_bytes),
        }
        control_plane_metrics = {
            'client': client_metrics,
            'server_pre_spu_checks_ms': None,
        }
        return {
            'share0_bytes': share0_bytes,
            'share1_bytes': share1_bytes,
            'audit': audit,
            'quality_assurance': quality_assurance,
            'control_plane_metrics': control_plane_metrics,
        }

    def build_mock_web_demo_result(
        self,
        class_names,
        sleep_sec: float,
        sample: Optional[dict] = None,
        profile: Optional[dict] = None,
    ):
        active_profile = profile or self.e2e_profile
        threshold_label = class_names[0] if class_names else 'class_0'
        argmax_label = class_names[1] if len(class_names) > 1 else threshold_label
        result = {
            'runtime': 'e2e',
            'prediction': {
                'argmax_label': argmax_label,
                'threshold_label': threshold_label,
                'prob_class_0': 0.41,
                'prob_class_1': 0.59,
            },
            'profile': {
                'sample_count': 1,
                'total_pipeline_duration_sec': float(sleep_sec),
                'communication': {
                    'total_bytes': 0,
                    'status': 'mock',
                    'supported': True,
                },
            },
            'summary': {
                'finite_logits': True,
                'effective_static_depth': self.e2e_profile['static_depth_limit'],
                'privacy_note': 'mock secure result for guard stress testing',
                'spu': {
                    'input_mode': 'party_local_debug_share_load',
                    'host_plaintext_pixel_values_materialized': False,
                    'host_private_share_tensors_loaded': False,
                    'private_input_paths_redacted': True,
                    'driver_private_share_manifest_paths_recorded': False,
                    'share_semantics': 'debug_float_additive_share_not_production_mpc_share',
                    'spu_layer_norm_policy': active_profile['spu_layer_norm_policy'],
                    'static_forward_metadata': {
                        'attention_policy': active_profile['spu_attention_policy'],
                        'activation_kind': active_profile['spu_activation_override'],
                    },
                },
            },
            'candidate_json': 'mock_candidate.json',
            'candidate_pt': 'mock_candidate.pt',
            'calibration_json': None,
        }
        if sample is not None:
            result['sample'] = sample
        return result

    def resolve_e2e_calibration_path(self, e2e_run_dir: Path):
        raw_path = (
            os.environ.get('WEB_DEMO_E2E_LN_CALIBRATION_JSON')
            or os.environ.get('E2E_SPU_LAYER_NORM_CALIBRATION_JSON')
        )
        if raw_path:
            path = Path(raw_path).expanduser().resolve()
            return path, path.exists()
        path = Path(self.e2e_profile['ln_calibration_json']).expanduser().resolve()
        return path, path.exists()

    def resolve_e2e_output_calibration_path(self):
        raw_path = (
            os.environ.get('WEB_DEMO_E2E_OUTPUT_CALIBRATION_JSON')
            or os.environ.get('E2E_OUTPUT_CALIBRATION_JSON')
        )
        if raw_path:
            path = Path(raw_path).expanduser().resolve()
            return path if path.exists() else None
        path = Path(self.e2e_profile['output_calibration_json']).expanduser().resolve()
        return path if path.exists() else None

    def build_e2e_env(
        self,
        context: dict,
        profile: dict,
        bundle_dir: Path,
        calibration_json: Optional[Path] = None,
        output_calibration_json: Optional[Path] = None,
    ):
        env = os.environ.copy()
        repo_root = self.remote_repo_root if self.e2e_execution_mode == 'ssh' else self.repo_root
        python_bin = self.remote_python_bin if self.e2e_execution_mode == 'ssh' else sys.executable
        e2e_run_dir = self.context_execution_path(context, 'e2e_run_dir')
        candidate_pt = self.execution_path(context['candidate_pt'])
        candidate_json = self.execution_path(context['candidate_json'])
        share_public_json = self.execution_path(context['share_public_json'])
        share_party_dir = self.execution_path(context['share_party_dir'])
        env.update(
            {
                'REPO_ROOT': str(repo_root),
                'PYTHON_BIN': python_bin,
                'BUNDLE_DIR': str(self.execution_path(bundle_dir)),
                'CONFIG_PATH': str(repo_root / 'configs' / 'openbumblebee' / '2pc.json'),
                'RUN_NAME': context['run_name'],
                'E2E_RUN_DIR': str(e2e_run_dir),
                'E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON': str(share_public_json),
                'E2E_INPUT_P1_SHARE_MANIFEST_JSON': str(share_party_dir / 'p1_share_manifest.json'),
                'E2E_INPUT_P2_SHARE_MANIFEST_JSON': str(share_party_dir / 'p2_share_manifest.json'),
                'E2E_CANDIDATE_PT': str(candidate_pt),
                'E2E_CANDIDATE_JSON': str(candidate_json),
                'E2E_STATIC_DEPTH_LIMIT': str(profile['static_depth_limit']),
                'E2E_RUN_MAX_SAMPLES': '1',
                'E2E_SPU_BATCH_SIZE': str(profile['spu_batch_size']),
                'E2E_SPU_BLOCK_CHUNK_SIZE': str(profile['spu_block_chunk_size']),
                'E2E_SPU_LAYER_NORM_CHUNK_SIZE': str(profile['spu_layer_norm_chunk_size']),
                'E2E_PARTY_LOCAL_SHARE_LOAD': '1' if profile['party_local_share_load'] else '0',
                'E2E_REDACT_PRIVATE_INPUT_PATHS': '1',
                'E2E_SPU_LAYER_NORM_POLICY': profile['spu_layer_norm_policy'],
                'E2E_SPU_PARAMS_MODE': profile['spu_params_mode'],
                'E2E_SPU_ATTENTION_POLICY': profile['spu_attention_policy'],
                'E2E_SPU_ACTIVATION_OVERRIDE': profile['spu_activation_override'],
                'E2E_SPU_ACTIVATION_CLIP_VALUE': str(profile['spu_activation_clip_value']),
                'SPU_RUNTIME_REUSE': '1' if profile['spu_runtime_reuse'] else '0',
                'SPU_DISABLE_COLOCATED_OPTIMIZATION': '1' if profile['spu_disable_colocated_optimization'] else '0',
                'PUBLIC_CALIB_DATASET_DIR': os.environ.get(
                    'WEB_DEMO_PUBLIC_CALIB_DATASET_DIR',
                    os.environ.get('DATA_ROOT', str(REPO_ROOT / 'data')),
                ),
            }
        )
        if calibration_json is not None:
            calibration_path = self.execution_path(calibration_json)
            env['E2E_SPU_LAYER_NORM_CALIBRATION_JSON'] = str(calibration_path)
        if output_calibration_json is not None:
            env['E2E_OUTPUT_CALIBRATION_JSON'] = str(self.execution_path(output_calibration_json))
        return env

    def remote_ssh_base(self):
        return [
            'ssh',
            '-p',
            self.remote_ssh_port,
            '-o',
            'StrictHostKeyChecking=no',
            '-o',
            f'UserKnownHostsFile={Path.home() / ".ssh" / "known_hosts"}',
        ]

    def remote_command_env(self, context: dict):
        env = os.environ.copy()
        password = os.environ.get('WEB_DEMO_REMOTE_SSH_PASSWORD', '')
        if password:
            askpass_path = context['secure_run_dir'] / '.web_demo_ssh_askpass.sh'
            askpass_path.write_text(
                '#!/usr/bin/env bash\n'
                f"printf '%s\\n' {shlex.quote(password)}\n",
                encoding='utf-8',
            )
            askpass_path.chmod(0o700)
            env['SSH_ASKPASS'] = str(askpass_path)
            env['SSH_ASKPASS_REQUIRE'] = 'force'
            env.setdefault('DISPLAY', ':0')
        return env

    def build_remote_forward_command(self, env: dict, whole_forward_script: Path):
        forward_env_keys = [
            'REPO_ROOT',
            'PYTHON_BIN',
            'BUNDLE_DIR',
            'CONFIG_PATH',
            'RUN_NAME',
            'E2E_RUN_DIR',
            'E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON',
            'E2E_INPUT_P1_SHARE_MANIFEST_JSON',
            'E2E_INPUT_P2_SHARE_MANIFEST_JSON',
            'E2E_CANDIDATE_PT',
            'E2E_CANDIDATE_JSON',
            'E2E_SPU_LAYER_NORM_CALIBRATION_JSON',
            'E2E_STATIC_DEPTH_LIMIT',
            'E2E_RUN_MAX_SAMPLES',
            'E2E_SPU_BATCH_SIZE',
            'E2E_SPU_BLOCK_CHUNK_SIZE',
            'E2E_SPU_LAYER_NORM_CHUNK_SIZE',
            'E2E_PARTY_LOCAL_SHARE_LOAD',
            'E2E_REDACT_PRIVATE_INPUT_PATHS',
            'E2E_SPU_LAYER_NORM_POLICY',
            'E2E_SPU_PARAMS_MODE',
            'E2E_SPU_ATTENTION_POLICY',
            'E2E_SPU_ACTIVATION_OVERRIDE',
            'E2E_SPU_ACTIVATION_CLIP_VALUE',
            'E2E_OUTPUT_CALIBRATION_JSON',
            'SPU_RUNTIME_REUSE',
            'SPU_DISABLE_COLOCATED_OPTIMIZATION',
            'PUBLIC_CALIB_DATASET_DIR',
        ]
        exported = ' '.join(
            f'{key}={shlex.quote(value)}'
            for key in forward_env_keys
            if (value := env.get(key)) is not None
        )
        return (
            f'cd {shlex.quote(str(self.remote_repo_root))} && '
            f'{exported} bash {shlex.quote(str(self.remote_path_for(whole_forward_script)))} spu'
        )

    def run_remote_e2e_whole_forward(self, context: dict, env: dict, whole_forward_script: Path):
        remote_command = self.build_remote_forward_command(env, whole_forward_script)
        with self.open_remote_client() as client:
            sftp = client.open_sftp()
            self.sftp_put_tree(sftp, context['secure_run_dir'], context['remote_secure_run_dir'])
            self.run_remote_command(
                client,
                remote_command,
                context['infer_log_path'],
                'browser_e2e_remote_secure_whole_forward_spu',
                self.command_timeout_sec,
            )
            self.sftp_get_tree(sftp, context['remote_e2e_run_dir'], context['e2e_run_dir'])
            self.sftp_get_tree(
                sftp,
                self.remote_repo_root / 'logs' / 'spu_nodes',
                context['secure_run_dir'] / 'remote_spu_nodes',
            )

    def run_finance_secure_sample(self, sample_id: str):
        sample = self.finance_samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f'unknown finance sample_id: {sample_id}')
        context = self.build_finance_sample_run_context(sample)
        whole_forward_script = self.secure_script_paths()['whole_forward']
        if self.e2e_execution_mode != 'ssh' and not self.local_e2e_stack_available():
            raise RuntimeError(
                'local E2E stack is incomplete for finance live demo; use WEB_DEMO_E2E_EXECUTION_MODE=ssh.'
            )
        env = self.build_e2e_env(
            context,
            profile=self.finance_profile,
            bundle_dir=self.finance_bundle_dir,
        )
        if self.e2e_execution_mode == 'ssh':
            remote_bundle_dir = self.execution_path(self.finance_bundle_dir)
            remote_sample_path = self.execution_path(sample['path'])
            remote_share_prefix = self.execution_path(context['share_prefix'])
            remote_share_manifest_json = self.execution_path(context['share_manifest_json'])
            remote_share_public_json = self.execution_path(context['share_public_json'])
            remote_share_party_dir = self.execution_path(context['share_party_dir'])
            share_seed = os.environ.get('WEB_DEMO_E2E_SHARE_SEED', '0')
            share_command = (
                f'cd {shlex.quote(str(self.remote_repo_root))} && '
                f'mkdir -p {shlex.quote(str(context["remote_e2e_run_dir"]))} '
                f'{shlex.quote(str(remote_share_party_dir))} && '
                f'{shlex.quote(self.remote_python_bin)} tools/transshield_e2e_secure_infer.py client-share-preprocess '
                f'--bundle-dir {shlex.quote(str(remote_bundle_dir))} '
                f'--image {shlex.quote(str(remote_sample_path))} '
                f'--max-samples 1 '
                f'--output-prefix {shlex.quote(str(remote_share_prefix))} '
                f'--output-json {shlex.quote(str(remote_share_manifest_json))} '
                f'--output-public-json {shlex.quote(str(remote_share_public_json))} '
                f'--output-party-manifest-dir {shlex.quote(str(remote_share_party_dir))} '
                f'--include-source-paths --include-targets --share-seed {shlex.quote(str(share_seed))}'
            )
            remote_forward = self.build_remote_forward_command(env, whole_forward_script)
            with self.open_remote_client() as client:
                sftp = client.open_sftp()
                self.ensure_remote_dir(sftp, context['remote_e2e_run_dir'])
                self.run_remote_command(
                    client,
                    share_command,
                    context['share_log_path'],
                    'finance_remote_share_preprocess',
                    self.finance_timeout_sec,
                )
                self.run_remote_command(
                    client,
                    remote_forward,
                    context['infer_log_path'],
                    'finance_remote_secure_whole_forward_spu',
                    self.finance_timeout_sec,
                )
                self.sftp_get_tree(sftp, context['remote_e2e_run_dir'], context['e2e_run_dir'])
                self.sftp_get_tree(
                    sftp,
                    self.remote_repo_root / 'logs' / 'spu_nodes',
                    context['secure_run_dir'] / 'remote_spu_nodes',
                )
        else:
            share_seed = os.environ.get('WEB_DEMO_E2E_SHARE_SEED', '0')
            self._run_command_with_log(
                [
                    sys.executable,
                    'tools/transshield_e2e_secure_infer.py',
                    'client-share-preprocess',
                    '--bundle-dir',
                    str(self.finance_bundle_dir),
                    '--image',
                    str(sample['path']),
                    '--max-samples',
                    '1',
                    '--output-prefix',
                    str(context['share_prefix']),
                    '--output-json',
                    str(context['share_manifest_json']),
                    '--output-public-json',
                    str(context['share_public_json']),
                    '--output-party-manifest-dir',
                    str(context['share_party_dir']),
                    '--include-source-paths',
                    '--include-targets',
                    '--share-seed',
                    share_seed,
                ],
                cwd=self.repo_root,
                env=os.environ.copy(),
                log_path=context['share_log_path'],
                step_name='finance_local_share_preprocess',
            )
            self._run_command_with_log(
                ['bash', str(whole_forward_script), 'spu'],
                cwd=self.repo_root,
                env=env,
                log_path=context['infer_log_path'],
                step_name='finance_local_secure_whole_forward_spu',
            )
        result = self.load_profiled_e2e_result(
            context,
            profile=self.finance_profile,
            class_names=self.finance_class_names,
        )
        result['sample'] = public_sample_metadata(sample)
        return result

    def secure_script_paths(self):
        script_dir = self.repo_root / 'artifacts' / 'server_inference_friendly_pack'
        return {
            'suite': script_dir / 'run_selected_image_secure_suite.sh',
            'profile': script_dir / 'run_secure_profile_summary.sh',
            'whole_forward': script_dir / 'run_e2e_secure_whole_forward.sh',
        }

    def load_secure_result(self, runtime: str, run_name: str, secure_run_dir: Path, suite_log_path: Path, profile_log_path: Path):
        diagnosis_path = secure_run_dir / 'selected_image_secure_diagnosis.json'
        profile_path = secure_run_dir / 'secure_profile_summary.json'
        diagnosis = load_json(diagnosis_path)
        profile = load_json(profile_path)
        return {
            'runtime': runtime,
            'run_name': run_name,
            'secure_run_dir': str(secure_run_dir),
            'suite_log_path': str(suite_log_path),
            'profile_log_path': str(profile_log_path),
            'diagnosis': diagnosis,
            'profile': profile,
        }

    def validate_e2e_summary(self, summary: dict):
        if not bool(summary.get('finite_logits', True)):
            raise RuntimeError('E2E candidate JSON reports non-finite logits.')
        guard = float_from_env('WEB_DEMO_E2E_LOGIT_ABS_GUARD', 10.0)
        stats = summary.get('raw_logits_before_output_calibration') or summary.get('logits') or {}
        if not isinstance(stats, dict):
            return
        min_value = stats.get('min')
        max_value = stats.get('max')
        if min_value is None or max_value is None:
            return
        max_abs = max(abs(float(min_value)), abs(float(max_value)))
        if max_abs > guard:
            raise RuntimeError(
                f'E2E candidate JSON exceeded logit guard: max_abs={max_abs:.6f}, guard={guard:.6f}.'
            )

    def build_prediction_payload(self, class_names, argmax_index, threshold_index, prob0, prob1):
        return {
            'argmax_label': int(argmax_index),
            'argmax_label_name': class_label(class_names, argmax_index),
            'threshold_label': None if threshold_index is None else int(threshold_index),
            'threshold_label_name': class_label(class_names, threshold_index),
            'prob_class_0': prob0,
            'prob_class_1': prob1,
            'confidence_margin': None if prob0 is None or prob1 is None else abs(prob1 - prob0),
        }

    def load_profiled_e2e_result(
        self,
        context: dict,
        profile: dict,
        class_names,
        calibration_json: Optional[Path] = None,
        output_calibration_json: Optional[Path] = None,
    ):
        summary = load_json(context['candidate_json'])
        self.validate_e2e_summary(summary)
        preview = summary.get('prediction_preview') or {}
        probabilities = preview.get('probabilities')
        logits = preview.get('logits')
        argmax_predictions = preview.get('argmax_predictions')
        threshold_predictions = preview.get('threshold_predictions')
        if probabilities is None or logits is None or argmax_predictions is None:
            raise RuntimeError(
                'candidate JSON missing prediction_preview; rerun with the updated '
                'transshield_e2e_secure_vit.py so the web backend does not need to import torch.'
            )
        row_index = 0
        row_probabilities = probabilities[row_index] if len(probabilities) > row_index else []
        prob0 = float(row_probabilities[0]) if len(row_probabilities) > 0 else None
        prob1 = float(row_probabilities[1]) if len(row_probabilities) > 1 else None
        threshold_index = (
            None
            if threshold_predictions is None or len(threshold_predictions) <= row_index
            else int(threshold_predictions[row_index])
        )
        remote_log_dir = context['secure_run_dir'] / 'remote_spu_nodes'
        log_dir = remote_log_dir if remote_log_dir.exists() else self.repo_root / 'logs' / 'spu_nodes'
        link_details = latest_nonzero_spu_link_details(log_dir)
        link_total_bytes = None if link_details is None else link_details['total_bytes']
        return {
            'runtime': 'e2e',
            'run_name': context['run_name'],
            'secure_run_dir': str(context['secure_run_dir']),
            'e2e_run_dir': str(context['e2e_run_dir']),
            'share_log_path': str(context['share_log_path']),
            'infer_log_path': str(context['infer_log_path']),
            'candidate_json': str(context['candidate_json']),
            'candidate_pt': str(context['candidate_pt']),
            'web_demo_profile': profile['profile_name'],
            'web_demo_profile_description': profile['description'],
            'calibration_json': None if calibration_json is None else str(calibration_json),
            'output_calibration_json': None if output_calibration_json is None else str(output_calibration_json),
            'summary': summary,
            'prediction': self.build_prediction_payload(
                class_names,
                argmax_predictions[row_index],
                threshold_index,
                prob0,
                prob1,
            ),
            'profile': {
                'sample_count': int(summary.get('sample_count') or len(probabilities)),
                'total_pipeline_duration_sec': summary.get('elapsed_sec'),
                'communication': {
                    'source': 'SPU/JAX e2e node logs',
                    'source_detail': f'Parsed from latest nonzero Link details in {log_dir}/node_*.log.',
                    'status': 'available' if link_details else 'missing',
                    'total_bytes': link_total_bytes,
                    'request_bytes': None if link_details is None else link_details['send_bytes'],
                    'response_bytes': None if link_details is None else link_details['recv_bytes'],
                    'make_shares_input_bytes': None,
                    'link_total_bytes': link_total_bytes,
                    'node_latest_nonzero_link_details': [] if link_details is None else link_details['node_latest_nonzero_link_details'],
                    'note': 'e2e path reveals final logits only; communication is parsed from current SPU node logs.',
                    'warning': None,
                },
                'supported': True,
                'web_demo_profile': profile['profile_name'],
                'web_demo_profile_description': profile['description'],
                'web_demo_profile_config': {
                    'static_depth_limit': int(profile['static_depth_limit']),
                    'spu_params_mode': profile['spu_params_mode'],
                    'spu_layer_norm_policy': profile['spu_layer_norm_policy'],
                    'spu_attention_policy': profile['spu_attention_policy'],
                    'spu_activation_override': profile['spu_activation_override'],
                    'spu_activation_clip_value': float(profile['spu_activation_clip_value']),
                },
            },
        }

    def get_cached_secure_result(self, session_id: str, runtime: str):
        return self.sessions[session_id].get('secure_results', {}).get(runtime)

    def run_secure_pipeline_for_session(self, session_id: str, runtime: str):
        if runtime not in {'cpu', 'spu'}:
            raise ValueError(f'unsupported runtime: {runtime}')
        context = self.build_secure_run_context(session_id, runtime)
        env = self.build_secure_env(
            context['image_path'],
            context['run_name'],
            context['secure_run_dir'],
            runtime,
        )
        scripts = self.secure_script_paths()

        self._run_command_with_log(
            ['bash', str(scripts['suite'])],
            cwd=self.repo_root,
            env=env,
            log_path=context['suite_log_path'],
            step_name=f'{runtime}_selected_image_secure_suite',
        )
        self._run_command_with_log(
            ['bash', str(scripts['profile'])],
            cwd=self.repo_root,
            env=env,
            log_path=context['profile_log_path'],
            step_name=f'{runtime}_secure_profile_summary',
        )

        result = self.load_secure_result(
            runtime,
            context['run_name'],
            context['secure_run_dir'],
            context['suite_log_path'],
            context['profile_log_path'],
        )
        context['session'].setdefault('secure_results', {})[runtime] = result
        return result

    def run_e2e_approx_for_session(self, session_id: str):
        context = self.build_e2e_run_context(session_id)
        calibration_json, calibration_exists = self.resolve_e2e_calibration_path(context['e2e_run_dir'])
        output_calibration_json = self.resolve_e2e_output_calibration_path()
        if output_calibration_json is None:
            raise RuntimeError(
                'E2E output calibration JSON is missing for the selected web demo profile.'
            )
        env = self.build_e2e_env(
            context,
            profile=self.e2e_profile,
            bundle_dir=self.bundle_dir,
            calibration_json=calibration_json,
            output_calibration_json=output_calibration_json,
        )
        scripts = self.secure_script_paths()

        self._run_command_with_log(
            [
                sys.executable,
                'tools/transshield_e2e_secure_infer.py',
                'client-share-preprocess',
                '--bundle-dir',
                str(self.bundle_dir),
                '--image',
                str(context['image_path']),
                '--max-samples',
                '1',
                '--output-prefix',
                str(context['share_prefix']),
                '--output-json',
                str(context['share_manifest_json']),
                '--output-public-json',
                str(context['share_public_json']),
                '--output-party-manifest-dir',
                str(context['share_party_dir']),
                '--share-seed',
                os.environ.get('WEB_DEMO_E2E_SHARE_SEED', '0'),
            ],
            cwd=self.repo_root,
            env=env,
            log_path=context['share_log_path'],
            step_name='e2e_client_share_preprocess',
        )
        if not calibration_exists:
            raise RuntimeError(
                'E2E public layer norm calibration JSON is missing for the selected web demo profile.'
            )
        self._run_command_with_log(
            ['bash', str(scripts['whole_forward']), 'spu'],
            cwd=self.repo_root,
            env=env,
            log_path=context['infer_log_path'],
            step_name='e2e_secure_whole_forward_spu',
        )

        result = self.load_profiled_e2e_result(
            context,
            profile=self.e2e_profile,
            class_names=self.class_names,
            calibration_json=calibration_json,
            output_calibration_json=output_calibration_json,
        )
        context['session'].setdefault('secure_results', {})['e2e'] = result
        return result

    @staticmethod
    def _tail_text(path: Path, max_lines: int = 40):
        if not path.exists():
            return ''
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(lines[-max_lines:])

    def _run_command_with_log(self, command, cwd: Path, env: dict, log_path: Path, step_name: str):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=self.command_timeout_sec)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                tail = self._tail_text(log_path)
                raise TimeoutError(
                    f'{step_name} timed out after {self.command_timeout_sec}s\n'
                    f'log_path={log_path}\n'
                    f'log_tail:\n{tail}'
                ) from exc
        if returncode != 0:
            tail = self._tail_text(log_path)
            raise RuntimeError(
                f'{step_name} failed with returncode={returncode}\n'
                f'log_path={log_path}\n'
                f'log_tail:\n{tail}'
            )


def summarize_stage_counts(trace):
    stage_lines = []
    for stage in trace['stages']:
        keep_count = int(stage['active_after_per_sample'][0])
        density = float(stage['active_after_density_per_sample'][0])
        stage_lines.append(f"stage {stage['stage_index'] + 1} 保留 {keep_count} 个 token（密度 {density:.2%}）")
    return '；'.join(stage_lines)


def build_stage_card(stage, init_spatial_tokens: int):
    keep_count = int(stage['active_after_per_sample'][0])
    active_before = int(stage['active_before_per_sample'][0])
    pruned_count = active_before - keep_count
    density = float(stage['active_after_density_per_sample'][0])
    return {
        'stage_index': int(stage['stage_index']) + 1,
        'pruning_layer': int(stage['pruning_layer']),
        'active_before': active_before,
        'keep_count': keep_count,
        'pruned_count': pruned_count,
        'density': density,
        'density_percent': f'{density:.2%}',
        'total_tokens': int(init_spatial_tokens),
    }


def pruning_stage_cards(trace):
    init_spatial_tokens = int(trace['init_spatial_tokens'])
    return [build_stage_card(stage, init_spatial_tokens) for stage in trace['stages']]


def stage_reason(index: int, total_stages: int):
    if index == 0:
        return '先做一轮粗筛，优先去掉贡献较低的 patch，保留主要上下文。'
    if index == total_stages - 1:
        return '最后一轮按更深层语义继续压缩，只留下更关键的 token。'
    return '在上一阶段保留集合上继续筛选，进一步去掉贡献较弱的 token。'


def stage_explanation_lines(stage_cards):
    if not stage_cards:
        return []
    lines = []
    total_stages = len(stage_cards)
    for index, stage in enumerate(stage_cards):
        lines.append(
            {
                'stage_index': stage['stage_index'],
                'title': f"Stage {stage['stage_index']}（layer {stage['pruning_layer']}）",
                'summary': (
                    f"保留 {stage['keep_count']} / {stage['active_before']}，"
                    f"本阶段新屏蔽 {stage['pruned_count']} 个，保留密度 {stage['density_percent']}。"
                ),
                'reason': stage_reason(index, total_stages),
            }
        )
    return lines


def human_bytes(value):
    if value in (None, ''):
        return 'N/A'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(value)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f'{size:.2f} {units[unit_index]}'


def build_communication_summary(source, source_detail, status, total_bytes, request_bytes, response_bytes, make_shares_input_bytes, link_total_bytes, note, warning):
    return {
        'source': source,
        'source_detail': source_detail,
        'status': status,
        'total_bytes': total_bytes,
        'request_bytes': request_bytes,
        'response_bytes': response_bytes,
        'make_shares_input_bytes': make_shares_input_bytes,
        'link_total_bytes': link_total_bytes,
        'note': note,
        'warning': warning,
    }


def latest_nonzero_spu_link_details(log_dir: Path):
    node_details = []
    for log_path in sorted(log_dir.glob('node_*.log')):
        latest = None
        for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = SPU_LINK_DETAILS_RE.search(line)
            if not match:
                continue
            item = {
                'log_path': str(log_path),
                'send_bytes': int(match.group('send')),
                'recv_bytes': int(match.group('recv')),
                'send_actions': int(match.group('send_actions')),
                'recv_actions': int(match.group('recv_actions')),
            }
            if item['send_bytes'] or item['recv_bytes'] or item['send_actions'] or item['recv_actions']:
                latest = item
        if latest is not None:
            node_details.append(latest)
    if not node_details:
        return None
    send_bytes = sum(item['send_bytes'] for item in node_details)
    recv_bytes = sum(item['recv_bytes'] for item in node_details)
    return {
        'node_latest_nonzero_link_details': node_details,
        'send_bytes': send_bytes,
        'recv_bytes': recv_bytes,
        'total_bytes': send_bytes + recv_bytes,
    }


def summarize_communication(profile):
    communication = profile.get('communication_profile') or {}
    if profile.get('runtime') != 'spu':
        return build_communication_summary(
            'CPU reference',
            'local plaintext reference',
            communication.get('status'),
            None,
            None,
            None,
            None,
            None,
            'CPU reference 是本地参考执行，不产生 SPU/2PC 通信量。',
            None,
        )
    fastpath = communication.get('aggregate_python_fastpath_metrics') or {}
    link_metrics = communication.get('aggregate_link_metrics') or {}
    if fastpath.get('rpc_total_bytes') is not None:
        return build_communication_summary(
            'Python RPC fastpath',
            fastpath.get('source') or 'python_distributed_rpc_cloudpickle',
            communication.get('status'),
            fastpath.get('rpc_total_bytes'),
            fastpath.get('rpc_request_total_bytes'),
            fastpath.get('rpc_response_total_bytes'),
            fastpath.get('make_shares_total_input_bytes'),
            link_metrics.get('sum_total_bytes'),
            communication.get('note'),
            communication.get('warning'),
        )
    if link_metrics.get('sum_total_bytes') is not None:
        return build_communication_summary(
            'SPU LinkDetails',
            'C++ yacl LinkDetails',
            communication.get('status'),
            link_metrics.get('sum_total_bytes'),
            None,
            None,
            None,
            link_metrics.get('sum_total_bytes'),
            communication.get('note'),
            communication.get('warning'),
        )
    return build_communication_summary(
        'N/A',
        '',
        communication.get('status'),
        None,
        None,
        None,
        None,
        None,
        communication.get('note'),
        communication.get('warning'),
    )


def build_secure_prediction(row):
    return {
        'argmax_label': row.get('secure_argmax_label'),
        'threshold_label': row.get('secure_threshold_label'),
        'prob_class_0': row.get('prob_class_0'),
        'prob_class_1': row.get('prob_class_1'),
        'confidence_margin': row.get('confidence_margin'),
    }


def summarize_secure_result(result):
    if result.get('runtime') == 'e2e':
        summary = result.get('summary') or {}
        spu = summary.get('spu') or {}
        return {
            'runtime': 'e2e',
            'prediction': result.get('prediction') or {},
            'profile': result.get('profile') or {},
            'e2e': {
                'candidate_json': result.get('candidate_json'),
                'candidate_pt': result.get('candidate_pt'),
                'calibration_json': result.get('calibration_json'),
                'input_mode': spu.get('input_mode'),
                'host_plaintext_pixel_values_materialized': spu.get('host_plaintext_pixel_values_materialized'),
                'host_private_share_tensors_loaded': spu.get('host_private_share_tensors_loaded'),
                'private_input_paths_redacted': spu.get('private_input_paths_redacted'),
                'driver_private_share_manifest_paths_recorded': spu.get('driver_private_share_manifest_paths_recorded'),
                'share_semantics': spu.get('share_semantics'),
                'layer_norm_policy': spu.get('spu_layer_norm_policy'),
                'attention_policy': (spu.get('static_forward_metadata') or {}).get('attention_policy'),
                'activation_kind': (spu.get('static_forward_metadata') or {}).get('activation_kind'),
                'effective_static_depth': summary.get('effective_static_depth'),
                'finite_logits': summary.get('finite_logits'),
                'privacy_note': summary.get('privacy_note'),
                'web_boundary_note': (
                    'The default web path generates shares in the browser and does not upload the raw image or '
                    'plaintext pixel_values. The current demo server still receives both debug shares in one process; '
                    'production deployment should route share0/share1 to independent P1/P2 services.'
                ),
            },
        }
    diagnosis_rows = result['diagnosis'].get('results', [])
    row = diagnosis_rows[0] if diagnosis_rows else {}
    profile = result['profile']
    communication = summarize_communication(profile)
    total_duration = (profile.get('step_profile') or {}).get('total_pipeline_duration_sec')
    return {
        'runtime': result['runtime'],
        'prediction': build_secure_prediction(row),
        'profile': {
            'sample_count': len(diagnosis_rows) or None,
            'total_pipeline_duration_sec': total_duration,
            'communication': communication,
            'sum_total_bytes': communication.get('total_bytes'),
            'supported': bool((profile.get('communication_profile') or {}).get('supported')),
        },
    }


def compare_secure_results(cpu_result, spu_result):
    cpu_summary = summarize_secure_result(cpu_result)
    spu_summary = summarize_secure_result(spu_result)
    cpu_time = cpu_summary['profile']['total_pipeline_duration_sec']
    spu_time = spu_summary['profile']['total_pipeline_duration_sec']
    ratio = None if cpu_time in (None, 0) or spu_time is None else float(spu_time / cpu_time)
    same_input_sample_count = (
        cpu_summary['profile'].get('sample_count')
        if cpu_summary['profile'].get('sample_count') == spu_summary['profile'].get('sample_count')
        else None
    )
    return {
        'cpu': cpu_summary,
        'spu': spu_summary,
        'same_input_sample_count': same_input_sample_count,
        'time_ratio_spu_over_cpu': ratio,
        'communication_note': 'Legacy CPU reference 和 SPU secure 使用同一批输入；CPU 不产生 2PC/SPU 通信，通信量只展示这次 legacy SPU live run。',
    }


class DemoHandler(BaseHTTPRequestHandler):
    server_version = 'TransshieldFlowDemo/0.1'

    @property
    def state(self) -> DemoState:
        return self.server.state

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_error_json(self, error, status, details=None, error_code=None, retryable=False):
        payload = {'error': error, 'retryable': bool(retryable)}
        if error_code is not None:
            payload['error_code'] = error_code
        if details is not None:
            payload['details'] = details
        self._send_json(payload, status=status)

    def _send_html(self, text: str, status=HTTPStatus.OK):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, 'File not found')
            return
        body = path.read_bytes()
        content_type = 'application/octet-stream'
        suffix = path.suffix.lower()
        if suffix in ['.html', '.htm']:
            content_type = 'text/html; charset=utf-8'
        elif suffix == '.css':
            content_type = 'text/css; charset=utf-8'
        elif suffix == '.js':
            content_type = 'application/javascript; charset=utf-8'
        elif suffix == '.json':
            content_type = 'application/json; charset=utf-8'
        elif suffix == '.png':
            content_type = 'image/png'
        elif suffix in ['.jpg', '.jpeg']:
            content_type = 'image/jpeg'
        elif suffix == '.webp':
            content_type = 'image/webp'
        elif suffix == '.svg':
            content_type = 'image/svg+xml'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _resolve_web_demo_path(self, request_path: str) -> Optional[Path]:
        relative = request_path.lstrip('/') or 'index.html'
        candidate = (WEB_DEMO_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_DEMO_ROOT.resolve())
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        if candidate.suffix.lower() not in WEB_DEMO_ALLOWED_EXTENSIONS:
            return None
        return candidate

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send_html(HTML_PATH.read_text(encoding='utf-8'))
            return
        if parsed.path == '/artifacts/web_demo_assets/demo_content_summary.json':
            self._send_file(DEMO_SUMMARY_PATH)
            return
        if parsed.path == '/control_plane_worker.js':
            worker_path = REPO_ROOT / 'web_demo' / 'control_plane_worker.js'
            if not worker_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, 'Worker not found')
                return
            body = worker_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/javascript; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == '/worker_selftest.html':
            page_path = REPO_ROOT / 'web_demo' / 'worker_selftest.html'
            if not page_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, 'File not found')
                return
            self._send_html(page_path.read_text(encoding='utf-8'))
            return
        if parsed.path.startswith('/uploads/'):
            name = parsed.path.split('/uploads/', 1)[1]
            safe_name = Path(name).name
            if safe_name != name:
                self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid upload path')
                return
            self._send_file(self.state.upload_dir / safe_name)
            return
        if parsed.path.startswith('/samples/finance/'):
            sample_id = parsed.path.split('/samples/finance/', 1)[1]
            safe_sample_id = Path(sample_id).name
            if safe_sample_id != sample_id:
                self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid sample path')
                return
            sample = self.state.finance_samples_by_id.get(safe_sample_id)
            if sample is None:
                self.send_error(HTTPStatus.NOT_FOUND, 'Sample not found')
                return
            self._send_file(sample['path'])
            return
        if parsed.path == '/api/demo_summary':
            self._send_json(self.state.demo_summary)
            return
        if parsed.path == '/api/health':
            self._send_json({'ok': True})
            return
        static_path = self._resolve_web_demo_path(parsed.path)
        if static_path is not None:
            self._send_file(static_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, 'Not found')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/upload':
            if not bool_from_env('WEB_DEMO_ENABLE_LEGACY_SIDECAR', False):
                self._send_error_json(
                    (
                        'Legacy plaintext upload is disabled. The default demo path is '
                        '/api/e2e/analyze_private_shares with browser-side share generation.'
                    ),
                    HTTPStatus.NOT_FOUND,
                )
                return
            self.handle_upload()
            return
        if parsed.path == '/api/run_secure':
            if not bool_from_env('WEB_DEMO_ENABLE_LEGACY_SIDECAR', False):
                self._send_error_json(
                    (
                        'Legacy CPU/SPU sidecar endpoint is disabled. Set '
                        'WEB_DEMO_ENABLE_LEGACY_SIDECAR=1 only for debugging old flows.'
                    ),
                    HTTPStatus.NOT_FOUND,
                )
                return
            self.handle_run_secure()
            return
        if parsed.path == '/api/e2e/analyze_private_shares':
            self.handle_e2e_private_share_analysis()
            return
        self.send_error(HTTPStatus.NOT_FOUND, 'Not found')

    def _parse_content_length(self):
        raw_value = self.headers.get('Content-Length', '0')
        try:
            return int(raw_value)
        except ValueError:
            return 0

    def _send_best_effort_413(self, error: str, details: Optional[str] = None):
        payload = {'error': error, 'error_code': 'payload_too_large', 'retryable': False}
        if details is not None:
            payload['details'] = details
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass

    def _read_request_body(self, max_bytes: int):
        transfer_encoding = self.headers.get('Transfer-Encoding')
        if transfer_encoding:
            self._send_error_json(
                '当前演示接口不接受 Transfer-Encoding 请求体。',
                HTTPStatus.BAD_REQUEST,
                error_code='transfer_encoding_not_supported',
            )
            return None
        content_length = self._parse_content_length()
        if content_length <= 0:
            self._send_error_json('请求体为空或 Content-Length 非法。', HTTPStatus.BAD_REQUEST, error_code='invalid_content_length')
            return None
        if content_length > max_bytes:
            self._send_best_effort_413(
                '请求体超过当前演示接口允许的大小。',
                details=f'current={human_bytes(content_length)} limit={human_bytes(max_bytes)}',
            )
            return None
        remaining = content_length
        chunks = []
        total = 0
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                self._send_error_json('请求体在传输过程中提前结束。', HTTPStatus.BAD_REQUEST, error_code='truncated_body')
                return None
            chunks.append(chunk)
            total += len(chunk)
            remaining -= len(chunk)
            if total > max_bytes:
                self._send_best_effort_413('请求体超过当前演示接口允许的大小。')
                return None
        return b''.join(chunks)

    def _precheck_raw_multipart(self, content_type: str, raw_body: bytes):
        if 'multipart/form-data' not in content_type:
            raise ValueError('content type is not multipart/form-data')
        boundary = parse_content_type_boundary(content_type)
        if not boundary:
            raise ValueError('multipart boundary missing')
        marker = b'--' + boundary
        if not raw_body.startswith(marker + b'\r\n'):
            raise ValueError('multipart body does not start with boundary')

        parts = {}
        boundary_count = 1
        position = len(marker) + 2
        while True:
            header_end = raw_body.find(b'\r\n\r\n', position)
            if header_end == -1:
                raise ValueError('multipart part headers not terminated')
            header_bytes = raw_body[position:header_end]
            if len(header_bytes) > MEDICAL_MULTIPART_MAX_HEADER_BYTES:
                raise ValueError('multipart part headers too large')
            headers = parse_part_headers_bytes(header_bytes)
            if len(parts) + 1 > MEDICAL_MULTIPART_MAX_CONTENT_DISPOSITION:
                raise ValueError('too many multipart parts')
            disposition = parse_content_disposition(headers.get('content-disposition', ''))
            if disposition.get('_kind') != 'form-data':
                raise ValueError('invalid content-disposition')
            name = disposition.get('name')
            if not name:
                raise ValueError('multipart field name missing')
            if name in parts:
                raise ValueError(f'duplicate multipart field: {name}')
            if headers.get('content-type', '').lower().startswith('multipart/'):
                raise ValueError('nested multipart not allowed')

            body_start = header_end + 4
            next_boundary = raw_body.find(b'\r\n' + marker, body_start)
            if next_boundary == -1:
                raise ValueError('multipart closing boundary missing')
            body_end = next_boundary
            parts[name] = RawPart(
                name=name,
                headers=headers,
                body_start=body_start,
                body_end=body_end,
                filename=disposition.get('filename'),
                content_type=headers.get('content-type'),
            )

            boundary_line_start = next_boundary + 2
            boundary_line_end = boundary_line_start + len(marker)
            boundary_count += 1
            if boundary_count > MEDICAL_MULTIPART_MAX_BOUNDARIES:
                raise ValueError('too many multipart boundaries')
            if raw_body.startswith(b'--', boundary_line_end):
                if raw_body[boundary_line_end + 2:] not in (b'', b'\r\n'):
                    raise ValueError('multipart epilogue is not allowed')
                break
            if raw_body[boundary_line_end:boundary_line_end + 2] != b'\r\n':
                raise ValueError('invalid multipart boundary separator')
            position = boundary_line_end + 2
        return parts

    def _validate_email_multipart_structure(self, content_type: str, raw_body: bytes, expected_fields: set):
        try:
            message = BytesParser(policy=email.policy.default).parsebytes(build_mime_message(content_type, raw_body))
        except Exception as exc:
            raise ValueError('email multipart parser rejected payload') from exc
        if not message.is_multipart():
            raise ValueError('mime root is not multipart')
        payload = message.get_payload()
        if not isinstance(payload, list):
            raise ValueError('multipart payload is not a list')
        if len(payload) != len(expected_fields):
            raise ValueError('unexpected multipart part count')
        for part in payload:
            if part.is_multipart():
                raise ValueError('nested multipart is not allowed')

    def _parse_multipart_request(self, max_bytes: int):
        raw_body = self._read_request_body(max_bytes)
        if raw_body is None:
            return None, None
        content_type = self.headers.get('Content-Type', '')
        try:
            parts = self._precheck_raw_multipart(content_type, raw_body)
            if 'domain' not in parts:
                raise ValueError('domain field missing')
            domain = self._read_small_text_part(raw_body, parts['domain'], 64)
            expected_fields = MEDICAL_REQUEST_FIELDS if domain == 'medical' else FINANCE_REQUEST_FIELDS
            if set(parts.keys()) != expected_fields:
                raise ValueError('unexpected multipart field set')
            self._validate_email_multipart_structure(content_type, raw_body, expected_fields)
        except ValueError as exc:
            self._send_error_json(
                '请求体不是受支持的 multipart/form-data 结构。',
                HTTPStatus.BAD_REQUEST,
                details=str(exc),
                error_code='malformed_multipart_precheck_failed',
            )
            return None, None
        return raw_body, parts

    def _read_small_text_part(self, raw_body: bytes, part: RawPart, max_bytes: int):
        part_bytes = memoryview(raw_body)[part.body_start:part.body_end]
        if len(part_bytes) > max_bytes:
            raise ValueError(f'{part.name} is too large')
        return bytes(part_bytes).decode('utf-8', 'strict').strip()

    def _load_json_part(self, raw_body: bytes, part: RawPart):
        part_view = memoryview(raw_body)[part.body_start:part.body_end]
        body_len = len(part_view)
        if body_len > JSON_PART_MAX_BYTES:
            raise ValueError('json part exceeds global size limit')
        field_limit = {
            'client_quality_summary': QUALITY_SUMMARY_MAX_BYTES,
            'client_control_plane_metrics': CONTROL_PLANE_METRICS_MAX_BYTES,
            'client_audit_manifest': AUDIT_MANIFEST_MAX_BYTES,
        }.get(part.name, JSON_PART_MAX_BYTES)
        if body_len > field_limit:
            raise ValueError(f'{part.name} exceeds field size limit')
        decoded = bytes(part_view).decode('utf-8', 'strict')
        return json.loads(
            decoded,
            parse_int=strict_json_int,
            parse_float=strict_json_float,
            parse_constant=strict_json_constant,
        )

    def _extract_part_bytes(self, raw_body: bytes, part: RawPart) -> bytes:
        return raw_body[part.body_start:part.body_end]

    def _extract_image_part(self, raw_body: bytes, parts: dict):
        image_part = parts.get('image')
        if image_part is None or not image_part.filename:
            self._send_error_json('没有收到图片文件，请重新选择一张图片。', HTTPStatus.BAD_REQUEST, error_code='missing_image')
            return None
        suffix = Path(image_part.filename).suffix.lower() or '.png'
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            allowed = ', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))
            self._send_error_json(
                f'不支持的图片格式：{suffix}。当前支持：{allowed}。',
                HTTPStatus.BAD_REQUEST,
                error_code='unsupported_image_suffix',
            )
            return None
        if image_part.content_type and not image_part.content_type.startswith('image/'):
            self._send_error_json('上传文件不是图片类型，请重新选择图片。', HTTPStatus.BAD_REQUEST, error_code='invalid_image_content_type')
            return None
        payload = self._extract_part_bytes(raw_body, image_part)
        if len(payload) > self.state.max_upload_bytes:
            self._send_error_json(
                f'上传图片超过 {human_bytes(self.state.max_upload_bytes)} 限制，请压缩后再试。',
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                error_code='image_too_large',
            )
            return None
        return image_part, payload, suffix

    def _build_upload_response(self, session_id: str, upload_name: str, analysis: dict, pruning_assets: dict):
        pruning_stages = pruning_stage_cards(analysis['pruning_trace'])
        return {
            'session_id': session_id,
            'image_url': f'/uploads/{upload_name}',
            'prediction': analysis['prediction'],
            'probabilities': analysis['probabilities'],
            'pruning_trace_summary': summarize_stage_counts(analysis['pruning_trace']),
            'pruning_stages': pruning_stages,
            'pruning_explanations': stage_explanation_lines(pruning_stages),
            'pruning_assets': pruning_assets,
            'pruning_asset_count': len(pruning_assets.get('overlay_urls', [])) + (1 if pruning_assets.get('original_panel_url') else 0),
        }

    def _parse_json_request_body(self):
        content_length = self._parse_content_length()
        if content_length <= 0:
            self._send_error_json('运行请求为空，请先上传图片后再点击运行。', HTTPStatus.BAD_REQUEST)
            return None
        try:
            return json.loads(self.rfile.read(content_length).decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json('运行请求格式不正确，请刷新页面后重试。', HTTPStatus.BAD_REQUEST)
            return None

    def _validate_runtime_mode(self, mode):
        if mode not in {'cpu', 'spu', 'both', 'e2e'}:
            self._send_error_json(
                f'不支持的运行模式：{mode}。请选择 CPU reference、SPU 安全推理或 e2e 近似全隐私路径。',
                HTTPStatus.BAD_REQUEST,
            )
            return False
        return True

    def _resolve_session(self, session_id):
        if not session_id or session_id not in self.state.sessions:
            self._send_error_json('当前会话无效，请先重新上传图片。', HTTPStatus.BAD_REQUEST)
            return None
        return self.state.sessions[session_id]

    def _run_both_secure_modes(self, session_id: str):
        cpu_result = self.state.get_cached_secure_result(session_id, 'cpu') or self.state.run_secure_pipeline_for_session(session_id, 'cpu')
        spu_result = self.state.get_cached_secure_result(session_id, 'spu') or self.state.run_secure_pipeline_for_session(session_id, 'spu')
        self._send_json({'session_id': session_id, 'mode': 'both', 'compare': compare_secure_results(cpu_result, spu_result)})

    def _run_single_secure_mode(self, session_id: str, mode: str):
        if mode == 'e2e':
            result = self.state.get_cached_secure_result(session_id, mode) or self.state.run_e2e_approx_for_session(session_id)
        else:
            result = self.state.get_cached_secure_result(session_id, mode) or self.state.run_secure_pipeline_for_session(session_id, mode)
        self._send_json({
            'session_id': session_id,
            'mode': mode,
            'result': summarize_secure_result(result),
        })

    def handle_upload(self):
        raw_body = self._read_request_body(self.state.max_upload_bytes)
        if raw_body is None:
            return
        content_type = self.headers.get('Content-Type', '')
        try:
            parts = self._precheck_raw_multipart(content_type, raw_body)
            if set(parts.keys()) != {'image'}:
                raise ValueError('legacy upload expects exactly one image field')
            self._validate_email_multipart_structure(content_type, raw_body, {'image'})
        except ValueError as exc:
            self._send_error_json(
                '上传请求格式不正确，请使用页面上的图片上传控件。',
                HTTPStatus.BAD_REQUEST,
                details=str(exc),
                error_code='malformed_upload_multipart',
            )
            return
        extracted = self._extract_image_part(raw_body, parts)
        if extracted is None:
            return
        _, image_bytes, suffix = extracted
        session_id = uuid.uuid4().hex
        upload_name = f'{session_id}{suffix}'
        upload_path = self.state.upload_dir / upload_name
        with upload_path.open('wb') as handle:
            handle.write(image_bytes)
        upload_path = self.state.upload_dir / upload_name

        analysis = self.state.analyze_image(upload_path)
        pruning_assets = self.state.create_pruning_visual_assets(session_id, analysis)
        self.state.store_session(session_id, upload_name, analysis, pruning_assets)
        self._send_json(self._build_upload_response(session_id, upload_name, analysis, pruning_assets))

    def handle_run_secure(self):
        payload = self._parse_json_request_body()
        if payload is None:
            return
        session_id = payload.get('session_id')
        mode = payload.get('mode', 'cpu')
        if not self._validate_runtime_mode(mode):
            return
        if self._resolve_session(session_id) is None:
            return

        try:
            if mode == 'both':
                self._run_both_secure_modes(session_id)
                return

            self._run_single_secure_mode(session_id, mode)
        except Exception as exc:
            self._send_error_json(
                f'secure pipeline 运行失败（runtime={mode}）',
                HTTPStatus.INTERNAL_SERVER_ERROR,
                details=str(exc),
            )

    def handle_e2e_private_share_analysis(self):
        request_started_at = time.perf_counter()
        client_ip = self.client_address[0] if self.client_address else 'unknown'
        allowed, rate_error = self.state.check_ip_rate_limit(client_ip)
        if not allowed:
            self._send_error_json(
                '当前来源请求过于频繁，请稍后再试。',
                HTTPStatus.TOO_MANY_REQUESTS,
                error_code=rate_error,
                retryable=True,
            )
            return

        raw_body, parts = self._parse_multipart_request(MEDICAL_MULTIPART_MAX_BYTES)
        if raw_body is None or parts is None:
            return

        try:
            domain = self._read_small_text_part(raw_body, parts['domain'], 64) or 'medical'
        except ValueError as exc:
            self._send_error_json('domain 字段不合法。', HTTPStatus.BAD_REQUEST, details=str(exc), error_code='invalid_domain')
            return

        if domain == 'finance':
            try:
                sample_id = self._read_small_text_part(raw_body, parts['sample_id'], 256)
            except ValueError as exc:
                self._send_error_json('金融压力验证缺少 sample_id。', HTTPStatus.BAD_REQUEST, details=str(exc), error_code='missing_sample_id')
                return
            if sample_id not in self.state.finance_samples_by_id:
                self._send_error_json(
                    '未知的金融压力样本标识。',
                    HTTPStatus.BAD_REQUEST,
                    error_code='unknown_finance_sample_id',
                )
                return
            reservation, reserve_error = self.state.reserve_ip_inflight(client_ip)
            if reservation is None:
                status = HTTPStatus.SERVICE_UNAVAILABLE if reserve_error == 'ip_state_saturated' else HTTPStatus.TOO_MANY_REQUESTS
                self._send_error_json(
                    '当前演示环境繁忙，请稍后重试。',
                    status,
                    error_code=reserve_error,
                    retryable=reserve_error != 'ip_state_saturated',
                )
                return
            try:
                mock_sleep = float_from_env('WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC', 0.0)
                if mock_sleep > 0:
                    time.sleep(mock_sleep)
                    result = self.state.build_mock_web_demo_result(
                        self.state.finance_class_names,
                        mock_sleep,
                        sample=public_sample_metadata(self.state.finance_samples_by_id[sample_id]),
                        profile=self.state.finance_profile,
                    )
                else:
                    result = self.state.run_finance_secure_sample(sample_id)
            except Exception as exc:
                self._send_error_json(
                    '金融边界压力验证失败',
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    details=str(exc),
                    error_code='finance_live_demo_failed',
                )
                return
            finally:
                self.state.release_ip_inflight(client_ip)
            self._send_json(
                {
                    'domain': 'finance',
                    'mode': 'finance_builtin_sample_secure_run',
                    'sample': result.get('sample'),
                    'result': summarize_secure_result(result),
                    'privacy': {
                        'builtin_demo_sample': True,
                        'host_plaintext_pixel_values_materialized': False,
                        'host_model_params_materialized': False,
                        'reveal_policy': 'final_logits_only',
                        'production_note': (
                            '当前按钮触发的是内置压力样本的完整隐私推理。正式落地时由银行侧服务器与 AI 公司侧服务器两方协同执行。'
                        ),
                    },
                }
            )
            return

        if domain != 'medical':
            self._send_error_json('domain 字段只支持 medical 或 finance。', HTTPStatus.BAD_REQUEST, error_code='unsupported_domain')
            return

        try:
            contract_version = self._read_small_text_part(raw_body, parts['client_contract_version'], 128)
            if contract_version != MEDICAL_CONTROL_PLANE_CONTRACT_VERSION:
                raise ValueError(f'expected {MEDICAL_CONTROL_PLANE_CONTRACT_VERSION}, got {contract_version}')
            client_quality_summary = self._load_json_part(raw_body, parts['client_quality_summary'])
            client_audit_manifest = self._load_json_part(raw_body, parts['client_audit_manifest'])
            client_control_plane_metrics = self._load_json_part(raw_body, parts['client_control_plane_metrics'])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(
                '医疗控制面文本字段解析失败。',
                HTTPStatus.BAD_REQUEST,
                details=str(exc),
                error_code='invalid_control_plane_payload',
            )
            return

        share0 = self._extract_part_bytes(raw_body, parts['share0'])
        share1 = self._extract_part_bytes(raw_body, parts['share1'])
        audit_nonce = str(client_audit_manifest.get('audit_nonce') or '').strip()
        if not audit_nonce or len(audit_nonce) > 128:
            self._send_error_json(
                'audit_nonce 缺失或长度非法。',
                HTTPStatus.BAD_REQUEST,
                error_code='invalid_audit_nonce',
            )
            return

        validation_started_at = time.perf_counter()
        try:
            validated = self.state.validate_medical_control_plane_payload(
                share0,
                share1,
                client_quality_summary,
                client_audit_manifest,
                client_control_plane_metrics,
            )
        except RuntimeError as exc:
            details = str(exc)
            error_code = {
                'share0 sha256 mismatch': 'audit_share_hash_mismatch',
                'share1 sha256 mismatch': 'audit_share_hash_mismatch',
                'audit chain sha256 mismatch': 'audit_chain_mismatch',
                'subnormal share values detected': 'invalid_subnormal_share',
                'share contains non-finite values': 'non_finite_share',
                'share magnitude exceeds allowed bound': 'share_magnitude_out_of_range',
                'reconstructed tensor contains non-finite values': 'non_finite_tensor',
                'reconstructed rgb range is invalid': 'invalid_tensor_rgb_range',
                'quality assurance blocked request': 'quality_assurance_blocked',
            }.get(details, 'invalid_medical_share_payload')
            self.state.record_audit_event(
                False,
                {
                    'ts': time.time(),
                    'ip': client_ip,
                    'audit_nonce': audit_nonce,
                    'error_code': error_code,
                    'details': details,
                },
            )
            self._send_error_json(
                '医疗控制面快检未通过。',
                HTTPStatus.UNPROCESSABLE_ENTITY,
                details=details,
                error_code=error_code,
            )
            return
        except ValueError as exc:
            self._send_error_json(
                'share 大小不正确。',
                HTTPStatus.BAD_REQUEST,
                details=str(exc),
                error_code='invalid_share_length',
            )
            return

        validated['control_plane_metrics']['server_pre_spu_checks_ms'] = round(
            (time.perf_counter() - validation_started_at) * 1000.0,
            3,
        )
        replay_ok, replay_error = self.state.check_and_remember_replay(
            audit_nonce,
            validated['audit']['server_payload_fingerprint'],
        )
        if not replay_ok:
            status = HTTPStatus.SERVICE_UNAVAILABLE if replay_error == 'guard_cache_saturated' else HTTPStatus.CONFLICT
            self.state.record_audit_event(
                False,
                {
                    'ts': time.time(),
                    'ip': client_ip,
                    'audit_nonce': audit_nonce,
                    'payload_fingerprint': validated['audit']['server_payload_fingerprint'],
                    'error_code': replay_error,
                },
            )
            self._send_error_json(
                '检测到重复或饱和的控制面请求。',
                status,
                error_code=replay_error,
                retryable=replay_error == 'guard_cache_saturated',
            )
            return

        reservation, reserve_error = self.state.reserve_ip_inflight(client_ip)
        if reservation is None:
            status = HTTPStatus.SERVICE_UNAVAILABLE if reserve_error == 'ip_state_saturated' else HTTPStatus.TOO_MANY_REQUESTS
            self.state.record_audit_event(
                False,
                {
                    'ts': time.time(),
                    'ip': client_ip,
                    'audit_nonce': audit_nonce,
                    'payload_fingerprint': validated['audit']['server_payload_fingerprint'],
                    'error_code': reserve_error,
                },
            )
            self._send_error_json(
                '当前演示环境繁忙，请稍后重试。',
                status,
                error_code=reserve_error,
                retryable=reserve_error != 'ip_state_saturated',
            )
            return

        try:
            mock_sleep = float_from_env('WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC', 0.0)
            if mock_sleep > 0:
                time.sleep(mock_sleep)
                result = self.state.build_mock_web_demo_result(
                    self.state.class_names,
                    mock_sleep,
                    profile=self.state.e2e_profile,
                )
            else:
                result = self.state.run_e2e_approx_for_browser_shares(validated['share0_bytes'], validated['share1_bytes'])
        except Exception as exc:
            self.state.record_audit_event(
                False,
                {
                    'ts': time.time(),
                    'ip': client_ip,
                    'audit_nonce': audit_nonce,
                    'payload_fingerprint': validated['audit']['server_payload_fingerprint'],
                    'error_code': 'medical_secure_run_failed',
                    'details': str(exc),
                },
            )
            self._send_error_json(
                '医疗正式主线推理失败',
                HTTPStatus.INTERNAL_SERVER_ERROR,
                details=str(exc),
                error_code='medical_secure_run_failed',
                retryable=True,
            )
            return
        finally:
            self.state.release_ip_inflight(client_ip)

        response_payload = {
            'domain': 'medical',
            'mode': 'e2e_browser_private_shares',
            'result': summarize_secure_result(result),
            'quality_assurance': validated['quality_assurance'],
            'audit': validated['audit'],
            'control_plane_metrics': validated['control_plane_metrics'],
            'privacy': {
                'browser_generated_shares': True,
                'server_received_plain_image': False,
                'server_received_plain_pixel_values': False,
                'server_received_share0_and_share1_in_demo_process': True,
                'production_note': (
                    '正式部署应将 share0/share1 分别上传到独立 P1/P2 服务；当前 web demo 为单进程演示接口。'
                    '另外，当前同步阻塞原型在长耗时 SPU 计算期间不能主动感知客户端断连并终止任务。'
                ),
            },
        }
        self.state.record_audit_event(
            True,
            {
                'ts': time.time(),
                'ip': client_ip,
                'audit_nonce': audit_nonce,
                'payload_fingerprint': validated['audit']['server_payload_fingerprint'],
                'quality_status': validated['quality_assurance']['status'],
                'request_total_ms': round((time.perf_counter() - request_started_at) * 1000.0, 3),
            },
        )
        self._send_json(response_payload)


def build_server(host: str, port: int, state: DemoState):
    class DemoHTTPServer(ThreadingHTTPServer):
        def __init__(self, server_address, handler_cls):
            super().__init__(server_address, handler_cls)
            self.state = state

    return DemoHTTPServer((host, port), DemoHandler)


def build_parser():
    parser = argparse.ArgumentParser(description='Run a lightweight upload-and-flow demo for Transshield image inference.')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--class-names', default='class_0,class_1')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--upload-dir', default='artifacts/web_demo_uploads')
    return parser


def main():
    args = build_parser().parse_args()

    class_names = parse_class_names(args.class_names)
    state = DemoState(
        bundle_dir=Path(args.bundle_dir),
        device=args.device,
        class_names=class_names,
        upload_dir=Path(args.upload_dir),
        repo_root=REPO_ROOT,
    )
    server = build_server(args.host, args.port, state)
    print(f'[*] Transshield flow demo listening on http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
