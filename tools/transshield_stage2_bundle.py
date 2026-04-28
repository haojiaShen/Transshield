import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_transform
from models.dyvit import VisionTransformerDiffPruning

DEFAULT_BUNDLE_MODEL_STATE_NAME = 'modified_plaintext_model_state_dict.pth'
LEGACY_BUNDLE_MODEL_STATE_NAME = 'student_model_state_dict.pth'


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_model_from_args_snapshot(args_snapshot: dict):
    base_rate = float(args_snapshot['base_rate'])
    keep_rate = [base_rate, base_rate ** 2, base_rate ** 3]
    return VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=int(args_snapshot['nb_classes']),
        pruning_loc=[3, 6, 9],
        token_ratio=keep_rate,
        distill=True,
        act_layer=args_snapshot['square_activation_mode'] if args_snapshot['use_square_gelu'] else 'gelu',
        use_mask_pruning=bool(args_snapshot['use_mask_pruning']),
        use_approx_attn=bool(args_snapshot['use_approx_attn']),
        approx_attn_mode=args_snapshot['approx_attn_mode'],
        fp32_attention=True,
        eval_pruning_mode=args_snapshot.get('eval_pruning_mode', 'topk_argsort'),
        eval_tie_policy=args_snapshot.get('eval_tie_policy', 'lowest_index'),
    )


def build_eval_transform_from_args_snapshot(args_snapshot: dict):
    transform_args = SimpleNamespace(
        input_size=int(args_snapshot['input_size']),
        imagenet_default_mean_and_std=bool(args_snapshot['imagenet_default_mean_and_std']),
        crop_pct=args_snapshot.get('crop_pct'),
        color_jitter=args_snapshot.get('color_jitter', 0.4),
        aa=args_snapshot.get('aa', 'rand-m9-mstd0.5-inc1'),
        train_interpolation=args_snapshot.get('train_interpolation', 'bicubic'),
        reprob=float(args_snapshot.get('reprob', 0.25)),
        remode=args_snapshot.get('remode', 'pixel'),
        recount=int(args_snapshot.get('recount', 1)),
    )
    return build_transform(is_train=False, args=transform_args)


def resolve_threshold(bundle_dir: Path, threshold_override=None):
    threshold_json = load_json(bundle_dir / 'threshold_best.json')
    if threshold_override is None:
        return threshold_json.get('eval_binary_threshold')
    return threshold_override


def resolve_model_state_dict_path(bundle_dir: Path) -> Path:
    bundle_dir = Path(bundle_dir).resolve()
    candidates = [
        bundle_dir / DEFAULT_BUNDLE_MODEL_STATE_NAME,
        bundle_dir / LEGACY_BUNDLE_MODEL_STATE_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        'missing frozen model state_dict; checked: '
        + ', '.join(str(path) for path in candidates)
    )


def stage2_semantics_map():
    return {
        'single_source_of_truth': {
            'model_state_dict': f'outputs/pneumonia_transshield_candidate_frozen_export_v1/{DEFAULT_BUNDLE_MODEL_STATE_NAME}',
            'threshold_sidecar': 'outputs/pneumonia_transshield_candidate_frozen_export_v1/threshold_best.json',
        },
        'f_less_candidates': [
            {
                'kind': 'final_binary_eval_threshold',
                'location': 'engine.py::_binary_threshold_accuracy',
                'operation': 'class1_prob >= threshold',
            },
            {
                'kind': 'final_binary_infer_threshold',
                'location': 'infer.py::validate',
                'operation': 'class1_prob >= threshold',
            },
            {
                'kind': 'pruning_kth_threshold_design_reference',
                'location': 'tools/transshield_kth_threshold_report.py',
                'operation': 'keep can be factored into per-sample kth threshold extraction plus score comparison, with separate tie handling if boundary ties exist',
            },
            {
                'kind': 'eval_pruning_f_less_tie_path',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning._build_eval_keep_decision',
                'operation': 'compare-network kth threshold -> score > threshold plus deterministic equal-score tie selection',
            },
        ],
        'f_mux_candidates': [
            {
                'kind': 'spatial_mask_apply_helper',
                'location': 'models/dyvit.py::_apply_spatial_mask',
                'operation': 'x[:, 1:] * decision',
            },
            {
                'kind': 'eval_mask_apply_inline',
                'location': 'models/dyvit.py::VisionTransformerDiffPruning.forward',
                'operation': 'torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)',
            },
        ],
        'notes': [
            'Current dynamic pruning eval path uses top-k selection via argsort/scatter for token keep mask generation.',
            'The inference-friendly eval pruning path can now replace top-k keep generation with compare-network kth extraction plus F_less-style comparison and deterministic tie selection.',
            'The final classification threshold is an explicit compare.',
            'The frozen baseline remains top-k unless eval_pruning_mode is set to compare_network_tie/f_less_tie.',
        ],
    }


def load_frozen_bundle(bundle_dir, device='cpu'):
    bundle_dir = Path(bundle_dir).resolve()
    args_snapshot = load_json(bundle_dir / 'args_snapshot.json')
    model = build_model_from_args_snapshot(args_snapshot).to(device)
    state_dict_path = resolve_model_state_dict_path(bundle_dir)
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            f'non-strict state_dict load: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}'
        )
    model.eval()
    transform = build_eval_transform_from_args_snapshot(args_snapshot)
    return {
        'bundle_dir': bundle_dir,
        'args_snapshot': args_snapshot,
        'model': model,
        'transform': transform,
    }


def preprocess_image(image_path, transform, device='cpu'):
    image_path = Path(image_path).resolve()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return image_path, tensor


def infer_logits(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
    return logits, probs


def postprocess_binary_output(probs, threshold=None):
    probs_cpu = probs.squeeze(0).detach().cpu()
    argmax_class = int(probs_cpu.argmax().item())
    threshold_class = None
    if threshold is not None and probs_cpu.numel() == 2:
        threshold_class = int((probs_cpu[1].item() >= threshold))
    return {
        'probabilities': [float(value) for value in probs_cpu.tolist()],
        'argmax_class': argmax_class,
        'threshold': threshold,
        'threshold_class': threshold_class,
    }


def input_tensor_stats(input_tensor):
    return {
        'min': float(input_tensor.min().item()),
        'max': float(input_tensor.max().item()),
        'mean': float(input_tensor.mean().item()),
        'std': float(input_tensor.std(unbiased=False).item()),
    }
