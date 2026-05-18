import argparse
import cgi
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HTML_PATH = REPO_ROOT / 'web_demo' / 'index.html'
DEMO_SUMMARY_PATH = REPO_ROOT / 'artifacts' / 'web_demo_assets' / 'best_demo_content.json'
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
SPU_LINK_DETAILS_RE = re.compile(
    r'Link details: total send bytes (?P<send>\d+), recv bytes (?P<recv>\d+), '
    r'send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)'
)


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
            'title': '当前默认展示 bundle',
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
            'communication_source': '本页面仅展示当前 E2E SPU live run 通信量',
            'summary': '前端默认加载当前 secure-static 正式交付 bundle，里面放的是当前主展示模型的权重、阈值和运行配置。',
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
                item['demo_note'] = '浏览器本地生成两份 share，再触发一次完整隐私推理。'
            elif item.get('id') == 'finance':
                item['eyebrow'] = 'Live Secure Demo'
                item['live_demo_supported'] = self.finance_live_supported()
                item['live_demo_mode'] = 'builtin_sample_secure_run'
                item['summary'] = (
                    '金融特征先编码成图像，再沿用同一套 Transshield 方法。页面支持从内置验证样本中触发一次完整隐私推理，'
                    '同时保留已经跑完的批量实测结果作为对照。'
                )
                item['sample_library'] = {
                    'title': '金融内置验证样本',
                    'summary': (
                        '选择一条内置验证样本，直接触发一次完整隐私推理。'
                        if self.finance_live_supported()
                        else '当前环境未配置远端 secure runtime，因此金融域先显示已验证结果。'
                    ),
                    'items': [public_sample_metadata(sample) for sample in self.finance_samples],
                }
                item['display_notes'] = [
                    '金融域也支持现场触发一次真实 secure run。',
                    '页面同时保留批量实测结果，便于和单次现场结果对照理解。',
                ]
        if summary.get('showcase_domains'):
            summary['showcase_domains']['summary'] = (
                '这页同时展示两条线：医疗支持浏览器本地分片演示；金融支持从内置验证样本触发完整隐私推理。'
            )
        summary['runtime_capability'] = {
            'medical_live_demo': True,
            'finance_live_demo': self.finance_live_supported(),
            'execution_mode': self.e2e_execution_mode,
        }
        return summary

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
        self.wfile.write(body)

    def _send_error_json(self, error, status, details=None):
        payload = {'error': error}
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
        content_type = 'image/png'
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            content_type = 'image/jpeg'
        elif path.suffix.lower() == '.webp':
            content_type = 'image/webp'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send_html(HTML_PATH.read_text(encoding='utf-8'))
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

    def _parse_upload_form(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._send_error_json('上传请求格式不正确，请使用页面上的图片上传控件。', HTTPStatus.BAD_REQUEST)
            return None
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers.get('Content-Type'),
            },
        )

    def _resolve_image_field(self, form):
        image_field = form['image'] if 'image' in form else None
        if isinstance(image_field, list):
            image_field = image_field[0] if image_field else None
        return image_field

    def _resolve_form_file_field(self, form, name: str):
        field = form[name] if name in form else None
        if isinstance(field, list):
            field = field[0] if field else None
        return field

    def _read_share_field(self, form, name: str):
        field = self._resolve_form_file_field(form, name)
        if field is None or not getattr(field, 'file', None):
            self._send_error_json(f'缺少 {name}，浏览器本地 share 生成可能失败。', HTTPStatus.BAD_REQUEST)
            return None
        data = field.file.read(E2E_SHARE_BYTE_COUNT + 1)
        if len(data) != E2E_SHARE_BYTE_COUNT:
            self._send_error_json(
                f'{name} 大小不正确：收到 {len(data)} bytes，期望 {E2E_SHARE_BYTE_COUNT} bytes。',
                HTTPStatus.BAD_REQUEST,
            )
            return None
        return data

    def _read_text_field(self, form, name: str, default: Optional[str] = None):
        if name not in form:
            return default
        field = form[name]
        if isinstance(field, list):
            field = field[0] if field else None
        if field is None:
            return default
        value = getattr(field, 'value', default)
        if value in (None, ''):
            return default
        return str(value).strip()

    def _validate_image_field(self, image_field):
        if image_field is None or not getattr(image_field, 'filename', ''):
            self._send_error_json('没有收到图片文件，请重新选择一张图片。', HTTPStatus.BAD_REQUEST)
            return None
        suffix = Path(image_field.filename).suffix.lower() or '.png'
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            allowed = ', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))
            self._send_error_json(f'不支持的图片格式：{suffix}。当前支持：{allowed}。', HTTPStatus.BAD_REQUEST)
            return None
        if getattr(image_field, 'type', '') and not image_field.type.startswith('image/'):
            self._send_error_json('上传文件不是图片类型，请重新选择图片。', HTTPStatus.BAD_REQUEST)
            return None
        return suffix

    def _save_uploaded_image(self, image_field, suffix: str):
        image_bytes = image_field.file.read(self.state.max_upload_bytes + 1)
        if len(image_bytes) > self.state.max_upload_bytes:
            self._send_error_json(
                f'上传图片超过 {human_bytes(self.state.max_upload_bytes)} 限制，请压缩后再试。',
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return None, None
        session_id = uuid.uuid4().hex
        upload_name = f'{session_id}{suffix}'
        upload_path = self.state.upload_dir / upload_name
        with upload_path.open('wb') as handle:
            handle.write(image_bytes)
        return session_id, upload_name

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
        content_length = self._parse_content_length()
        if content_length > self.state.max_upload_bytes:
            self._send_error_json(
                (
                    f'上传文件过大：当前请求约 {human_bytes(content_length)}，'
                    f'演示页限制 {human_bytes(self.state.max_upload_bytes)}。请换一张压缩后的图片。'
                ),
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        form = self._parse_upload_form()
        if form is None:
            return
        image_field = self._resolve_image_field(form)
        suffix = self._validate_image_field(image_field)
        if suffix is None:
            return
        session_id, upload_name = self._save_uploaded_image(image_field, suffix)
        if session_id is None:
            return
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
        form = self._parse_upload_form()
        if form is None:
            return
        domain = self._read_text_field(form, 'domain', 'medical') or 'medical'
        try:
            if domain == 'finance':
                sample_id = self._read_text_field(form, 'sample_id')
                if not sample_id:
                    self._send_error_json('金融现场验证缺少 sample_id。', HTTPStatus.BAD_REQUEST)
                    return
                result = self.state.run_finance_secure_sample(sample_id)
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
                                '当前按钮触发的是内置验证样本的完整隐私推理。真实部署时，数据使用方应先在本地完成分片，'
                                '再把分片分别交给独立计算节点。'
                            ),
                        },
                    }
                )
                return

            content_length = self._parse_content_length()
            max_share_request_bytes = E2E_SHARE_BYTE_COUNT * 2 + 1024 * 1024
            if content_length > max_share_request_bytes:
                self._send_error_json(
                    f'密态 share 请求过大：当前约 {human_bytes(content_length)}，限制 {human_bytes(max_share_request_bytes)}。',
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            share0 = self._read_share_field(form, 'share0')
            if share0 is None:
                return
            share1 = self._read_share_field(form, 'share1')
            if share1 is None:
                return
            result = self.state.run_e2e_approx_for_browser_shares(share0, share1)
            self._send_json(
                {
                    'domain': 'medical',
                    'mode': 'e2e_browser_private_shares',
                    'result': summarize_secure_result(result),
                    'privacy': {
                        'browser_generated_shares': True,
                        'server_received_plain_image': False,
                        'server_received_plain_pixel_values': False,
                        'server_received_share0_and_share1_in_demo_process': True,
                        'production_note': (
                            '生产部署应将 share0/share1 分别上传到独立 P1/P2 服务；当前 web demo '
                            '为单进程演示接口，但不上传原图或完整 pixel_values。'
                        ),
                    },
                }
            )
        except Exception as exc:
            self._send_error_json(
                'E2E 浏览器密态 share 推理失败',
                HTTPStatus.INTERNAL_SERVER_ERROR,
                details=str(exc),
            )


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
