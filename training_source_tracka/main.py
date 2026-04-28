# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset, build_transform
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from losses import DistillDiffPruningLoss_dynamic
from samplers import RASampler
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from calc_flops import calc_flops, throughput

import warnings
warnings.filterwarnings('ignore')

def _format_tensor_stats(tensor):
    if tensor is None:
        return "value=None"
    detached = tensor.detach().float()
    if detached.numel() == 0:
        return f"shape={tuple(detached.shape)} empty=True"
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return f"shape={tuple(detached.shape)} finite_count=0"
    std = finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0
    norm = finite.norm().item()
    return (
        f"shape={tuple(detached.shape)} "
        f"min={finite.min().item():.6e} "
        f"max={finite.max().item():.6e} "
        f"mean={finite.mean().item():.6e} "
        f"std={std:.6e} "
        f"norm={norm:.6e}"
    )

def _log_patch_embed_diagnostics(student_model, teacher_model):
    for name in ['patch_embed.proj.weight', 'patch_embed.proj.bias']:
        student_param = dict(student_model.named_parameters()).get(name)
        teacher_param = dict(teacher_model.named_parameters()).get(name)
        print(f"[InitDiag] student {name} {_format_tensor_stats(student_param)}")
        print(f"[InitDiag] teacher {name} {_format_tensor_stats(teacher_param)}")
        if student_param is None or teacher_param is None:
            continue
        if student_param.shape != teacher_param.shape:
            print(
                f"[InitDiag] student/teacher {name} shape_mismatch "
                f"student_shape={tuple(student_param.shape)} teacher_shape={tuple(teacher_param.shape)}"
            )
            continue
        diff = (student_param.detach().float() - teacher_param.detach().float()).abs()
        print(
            f"[InitDiag] student/teacher {name} abs_diff "
            f"max={diff.max().item():.6e} mean={diff.mean().item():.6e}"
        )

def _resolve_patch_embed_bias_init_mode(args):
    if args.patch_embed_bias_init_mode != 'pretrained':
        return args.patch_embed_bias_init_mode
    if args.skip_patch_embed_bias_pretrained:
        return 'default'
    return 'pretrained'

def _drop_student_pretrained_keys(state_dict, args):
    if _resolve_patch_embed_bias_init_mode(args) == 'pretrained':
        return state_dict
    filtered = state_dict.copy()
    if 'patch_embed.proj.bias' in filtered:
        print("Removing key patch_embed.proj.bias from student pretrained checkpoint")
        del filtered['patch_embed.proj.bias']
    return filtered

def _apply_patch_embed_bias_init(model, args):
    mode = _resolve_patch_embed_bias_init_mode(args)
    print(f"[InitDiag] patch_embed_bias_init_mode={mode}")
    if mode != 'zero':
        return
    patch_embed = getattr(model, 'patch_embed', None)
    proj = getattr(patch_embed, 'proj', None)
    bias = getattr(proj, 'bias', None)
    if bias is None:
        print("[InitDiag] patch_embed.proj.bias unavailable for zero init")
        return
    with torch.no_grad():
        bias.zero_()
    print("[InitDiag] student patch_embed.proj.bias zero-initialized after load")

def _freeze_patch_embed_bias_if_needed(model, args):
    if not args.freeze_patch_embed_bias:
        return
    patch_embed = getattr(model, 'patch_embed', None)
    proj = getattr(patch_embed, 'proj', None)
    bias = getattr(proj, 'bias', None)
    if bias is None:
        print("[InitDiag] patch_embed.proj.bias unavailable for freeze")
        return
    bias.requires_grad_(False)
    print("[InitDiag] student patch_embed.proj.bias frozen from training")

def _freeze_patch_embed_proj_if_needed(model, args):
    if not args.freeze_patch_embed_proj:
        return
    patch_embed = getattr(model, 'patch_embed', None)
    proj = getattr(patch_embed, 'proj', None)
    if proj is None:
        print("[InitDiag] patch_embed.proj unavailable for freeze")
        return
    frozen_names = []
    if getattr(proj, 'weight', None) is not None:
        proj.weight.requires_grad_(False)
        frozen_names.append('weight')
    if getattr(proj, 'bias', None) is not None:
        proj.bias.requires_grad_(False)
        frozen_names.append('bias')
    print(f"[InitDiag] student patch_embed.proj frozen from training parts={frozen_names}")

def _load_checkpoint_state_dict(path, model_key='model|module'):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    checkpoint_model = None
    for key in model_key.split('|'):
        if key in checkpoint:
            checkpoint_model = checkpoint[key]
            print("Load state_dict by model_key = %s" % key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    return checkpoint_model

def _load_teacher_checkpoint_if_needed(teacher_model, args):
    if not args.teacher_checkpoint_path:
        return
    print("Load teacher ckpt from %s" % args.teacher_checkpoint_path)
    teacher_checkpoint_model = _load_checkpoint_state_dict(
        args.teacher_checkpoint_path, model_key=args.model_key)
    utils.load_state_dict(teacher_model, teacher_checkpoint_model)

def _freeze_patch_embed_weight_if_needed(model, args):
    if not args.freeze_patch_embed_weight:
        return
    patch_embed = getattr(model, 'patch_embed', None)
    proj = getattr(patch_embed, 'proj', None)
    weight = getattr(proj, 'weight', None)
    if weight is None:
        print("[InitDiag] patch_embed.proj.weight unavailable for freeze")
        return
    weight.requires_grad_(False)
    print("[InitDiag] student patch_embed.proj.weight frozen from training")


def _build_deit_s_model_bundle(args, sparse_ratio, student_act_layer):
    pruning_loc = [3, 6, 9]
    keep_rate = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
    print('token_ratio =', keep_rate, 'at layer', pruning_loc)
    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=args.nb_classes,
        pruning_loc=pruning_loc,
        token_ratio=keep_rate,
        distill=True,
        act_layer=student_act_layer,
        use_mask_pruning=args.use_mask_pruning,
        use_approx_attn=args.use_approx_attn,
        approx_attn_mode=args.approx_attn_mode,
        fp32_attention=True,
        nonempty_keep_guard=args.nonempty_keep_guard,
    )
    pretrained = torch.load(
        './pretrained/deit_small_patch16_224-cd65a155.pth',
        map_location='cpu',
    )['model']
    teacher_model = VisionTransformerTeacher(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=args.nb_classes,
    )
    return model, teacher_model, pretrained, keep_rate


def _build_class_weights(dataset, nb_classes, mode, power=1.0):
    if mode == 'none':
        return None
    if mode not in {'inverse_freq', 'sqrt_inverse_freq', 'power_inverse_freq'}:
        raise ValueError(f"Unsupported class_weight_mode: {mode}")

    targets = getattr(dataset, 'targets', None)
    if targets is None:
        raise ValueError('class_weight_mode requires dataset.targets')

    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(target_tensor, minlength=nb_classes).float()
    if (class_counts <= 0).any():
        raise ValueError(f'class_weight_mode requires all classes present, got counts={class_counts.tolist()}')

    if mode == 'power_inverse_freq':
        class_weights = class_counts.pow(-float(power))
        class_weights = class_weights * (float(nb_classes) / class_weights.sum())
    else:
        class_weights = class_counts.sum() / (class_counts * float(nb_classes))
        if mode == 'sqrt_inverse_freq':
            class_weights = class_weights.sqrt()
    print(
        f"[ClassWeight] mode={mode} "
        f"power={power} "
        f"counts={class_counts.tolist()} "
        f"weights={class_weights.tolist()}"
    )
    return class_weights


def _build_sample_weights(dataset, nb_classes, mode):
    if mode == 'distributed':
        return None
    if mode not in {'weighted_inverse_freq', 'weighted_sqrt_inverse_freq'}:
        raise ValueError(f"Unsupported train_sampler_mode: {mode}")

    targets = getattr(dataset, 'targets', None)
    if targets is None:
        raise ValueError('train_sampler_mode requires dataset.targets')

    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(target_tensor, minlength=nb_classes).float()
    if (class_counts <= 0).any():
        raise ValueError(f'train_sampler_mode requires all classes present, got counts={class_counts.tolist()}')

    class_weights = class_counts.sum() / (class_counts * float(nb_classes))
    if mode == 'weighted_sqrt_inverse_freq':
        class_weights = class_weights.sqrt()
    sample_weights = class_weights[target_tensor].double()
    print(
        f"[SamplerDiag] mode={mode} "
        f"counts={class_counts.tolist()} "
        f"class_weights={class_weights.tolist()} "
        f"sample_weight_min={sample_weights.min().item():.6e} "
        f"sample_weight_max={sample_weights.max().item():.6e}"
    )
    return sample_weights

def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic training script', add_help=False)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='deit-s', choices=['deit-s'], type=str, metavar='MODEL',
                        help='Name of model to train; current TrackA source training only supports deit-s.')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=True, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--augmentation_profile', type=str, default='timm',
                        choices=['timm', 'mpcvit_like', 'mpcvit_like_hflip'],
                        help='Training augmentation profile. "timm" keeps the existing pipeline; '
                             '"mpcvit_like" uses fixed resize with no color jitter/auto-augment/random erasing, '
                             'inspired by the external MPCViT PneumoniaMNIST reproduction.')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=utils.str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=utils.str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=utils.str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--eval_binary_threshold', type=float, default=None,
                        help='Optional probability threshold for class-1 during binary eval; default None keeps argmax.')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=utils.str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=utils.str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--groupa_lr_scale', type=float, default=0.1,
                        help='LR scale for group A: patch_embed.proj.{weight,bias} and cls_token.')
    parser.add_argument('--activation_lr_scale', type=float, default=1.0,
                        help='LR scale for learnable square/quadratic activation coefficients (.raw_alpha/.alpha/.beta).')
    parser.add_argument('--cls_token_full_lr', type=utils.str2bool, default=False,
                        help='Keep cls_token out of the low-lr group so it uses the standard pretrained lr scale.')
    parser.add_argument('--train_pos_embed', type=utils.str2bool, default=False,
                        help='Include pos_embed in optimizer parameter groups.')
    parser.add_argument('--pretrained_fix_step', type=int, default=5,
                        help='Number of initial epochs with lr forced to 0 for pretrained decay/no_decay parameter groups.')
    parser.add_argument('--skip_patch_embed_bias_pretrained', type=utils.str2bool, default=False,
                        help='Skip loading pretrained patch_embed.proj.bias for the student model only.')
    parser.add_argument('--teacher_checkpoint_path', default='',
                        help='Optional checkpoint path for overriding teacher weights after base pretrained load.')
    parser.add_argument('--patch_embed_bias_init_mode', type=str, default='pretrained',
                        choices=['pretrained', 'default', 'zero'],
                        help='Student patch_embed.proj.bias init mode: pretrained, default(skip pretrained), or zero(skip pretrained + zero init).')
    parser.add_argument('--freeze_patch_embed_proj', type=utils.str2bool, default=False,
                        help='Freeze student patch_embed.proj.{weight,bias} from training.')
    parser.add_argument('--class_weight_mode', type=str, default='none',
                        choices=['none', 'inverse_freq', 'sqrt_inverse_freq', 'power_inverse_freq'],
                        help='Apply class weights to the base CrossEntropyLoss.')
    parser.add_argument('--class_weight_power', type=float, default=1.0,
                        help='Exponent for class_weight_mode=power_inverse_freq; weight is proportional to class_count^-power.')
    parser.add_argument('--train_sampler_mode', type=str, default='distributed',
                        choices=['distributed', 'weighted_inverse_freq', 'weighted_sqrt_inverse_freq'],
                        help='Training sampler mode; default preserves the original DistributedSampler behavior.')
    parser.add_argument('--freeze_patch_embed_weight', type=utils.str2bool, default=False,
                        help='Freeze student patch_embed.proj.weight from training.')
    parser.add_argument('--freeze_patch_embed_bias', type=utils.str2bool, default=False,
                        help='Freeze student patch_embed.proj.bias from training.')
    parser.add_argument('--base_rate', type=float, default='0.9')
    parser.add_argument('--ratio_weight', type=float, default='2.0')
    parser.add_argument('--cls_distill_weight', type=float, default=None,
                        help='Override classification distillation weight; default None keeps current distill_weight behavior.')
    parser.add_argument('--token_distill_weight', type=float, default=None,
                        help='Override token distillation weight; default None keeps current distill_weight behavior.')
    parser.add_argument('--use_square_gelu', type=utils.str2bool, default=True,
                        help='Use square activation in student MLP/predictor modules.')
    parser.add_argument('--square_activation_mode', type=str, default='learnable_quadratic',
                        choices=['fixed_square', 'learnable_square', 'learnable_quadratic', 'learnable_quadratic_gelu_init'],
                        help='Student square activation mode when use_square_gelu=true.')
    parser.add_argument('--use_approx_attn', type=utils.str2bool, default=False,
                        help='Use security-friendly approximate attention in student attention blocks.')
    parser.add_argument('--approx_attn_mode', type=str, default='relu',
                        help='Approximate attention mode for student model.')
    parser.add_argument('--use_mask_pruning', type=utils.str2bool, default=False,
                        help='Use mask-based pruning in training instead of keeping pruned token features active.')
    parser.add_argument('--nonempty_keep_guard', type=utils.str2bool, default=False,
                        help='Ensure each sample keeps at least one active token after hard pruning.')
    parser.add_argument('--debug_nan', type=utils.str2bool, default=False,
                        help='Enable minimal non-finite checks in student forward and loss computation.')
    parser.add_argument('--loss_grad_attrib', type=utils.str2bool, default=False,
                        help='Log per-loss gradient attribution for one target parameter without changing training semantics.')
    parser.add_argument('--loss_grad_attrib_param', type=str, default='score_predictor.1.out_proj.weight',
                        help='Parameter name used by --loss_grad_attrib diagnostics.')
    parser.add_argument('--debug_max_steps', default=0, type=int,
                        help='If > 0, stop training epoch early after this many steps for debug runs.')
    parser.add_argument('--stop_after_epoch', default=0, type=int,
                        help='Optional early stop after N completed epochs while preserving the full epochs LR/WD schedule.')

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sample_weights = _build_sample_weights(dataset_train, args.nb_classes, args.train_sampler_mode)
    if sample_weights is not None:
        if num_tasks != 1:
            raise ValueError('train_sampler_mode=weighted_inverse_freq currently supports only single-process runs')
        sampler_train = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    class_weights = _build_class_weights(
        dataset_train, args.nb_classes, args.class_weight_mode,
        power=args.class_weight_power)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        if class_weights is not None:
            raise ValueError('class_weight_mode is not supported together with mixup/cutmix')
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        if class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(args.model)

    SPARSE_RATIO = [args.base_rate, args.base_rate - 0.2, args.base_rate - 0.4]
    student_act_layer = args.square_activation_mode if args.use_square_gelu else 'gelu'
    model, teacher_model, pretrained, keep_rate = _build_deit_s_model_bundle(
        args, SPARSE_RATIO, student_act_layer
    )
    student_pretrained = _drop_student_pretrained_keys(pretrained, args)
    utils.load_state_dict(model, student_pretrained)
    _apply_patch_embed_bias_init(model, args)
    _freeze_patch_embed_proj_if_needed(model, args)
    _freeze_patch_embed_weight_if_needed(model, args)
    _freeze_patch_embed_bias_if_needed(model, args)
    utils.load_state_dict(teacher_model, pretrained)
    _load_teacher_checkpoint_if_needed(teacher_model, args)
    if args.debug_nan:
        _log_patch_embed_diagnostics(model, teacher_model)
    teacher_model.eval()
    teacher_model = teacher_model.to(device)
    print('success load teacher model weight')
    criterion = DistillDiffPruningLoss_dynamic(
        teacher_model,
        criterion,
        clf_weight=1.0,
        keep_ratio=keep_rate,
        mse_token=True,
        ratio_weight=args.ratio_weight,
        distill_weight=0.5,
        cls_distill_weight=args.cls_distill_weight,
        token_distill_weight=args.token_distill_weight
    )

    model.eval()
    if utils.is_main_process():
        flops = calc_flops(model, args.input_size)
        print('FLOPs: {}'.format(flops))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print('number of params:', n_parameters)

    if args.finetune:
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = _load_checkpoint_state_dict(
            args.finetune, model_key=args.model_key)
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        checkpoint_model = _drop_student_pretrained_keys(checkpoint_model, args)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        _apply_patch_embed_bias_init(model, args)
        _freeze_patch_embed_proj_if_needed(model, args)
        _freeze_patch_embed_weight_if_needed(model, args)
        _freeze_patch_embed_bias_if_needed(model, args)
    
    model.to(device)

    if utils.is_main_process() and args.throughput:
        print('# throughput test')
        image = torch.randn(32, 3, args.input_size, args.input_size)
        throughput(image, model)
        del image
        import sys
        sys.exit(1)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay != 1.0:
        raise ValueError('TrackA source training currently only supports deit-s with layer_decay=1.0.')
    assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if hasattr(model_without_ddp, 'set_debug_nan'):
        model_without_ddp.set_debug_nan(args.debug_nan)
    if hasattr(criterion, 'set_debug_nan'):
        criterion.set_debug_nan(args.debug_nan)

    low_lr_names = {'patch_embed.proj.weight', 'patch_embed.proj.bias'}
    if not args.cls_token_full_lr:
        low_lr_names.add('cls_token')

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        bone_lr_scale=args.lr_scale,
        low_lr_names=low_lr_names,
        low_lr_scale=args.groupa_lr_scale)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used
    if hasattr(loss_scaler, 'set_debug_nan'):
        loss_scaler.set_debug_nan(args.debug_nan)
    if args.debug_nan:
        for name, param in model_without_ddp.named_parameters():
            setattr(param, '_debug_name', name)

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))


    print("criterion = %s" % str(criterion))

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    max_accuracy, max_accuracy_ema = utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(
            data_loader_val, model, device, use_amp=args.use_amp,
            binary_threshold=args.eval_binary_threshold)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(model_without_ddp, 'get_square_alpha_summary'):
            print(f"Square alpha before epoch {epoch}: {model_without_ddp.get_square_alpha_summary()}")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp, debug_nan=args.debug_nan, debug_max_steps=args.debug_max_steps,
            loss_grad_attrib=args.loss_grad_attrib, loss_grad_attrib_param=args.loss_grad_attrib_param
        )
        if hasattr(model_without_ddp, 'get_square_alpha_summary'):
            print(f"Square alpha after epoch {epoch}: {model_without_ddp.get_square_alpha_summary()}")

        if data_loader_val is not None:
            test_stats = evaluate(
                data_loader_val, model, device, use_amp=args.use_amp,
                binary_threshold=args.eval_binary_threshold)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(
                    data_loader_val, model_ema.ema, device, use_amp=args.use_amp,
                    binary_threshold=args.eval_binary_threshold)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.stop_after_epoch > 0 and (epoch + 1) >= args.stop_after_epoch:
            print(f"Early stop after epoch {epoch} due to stop_after_epoch={args.stop_after_epoch}")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





















