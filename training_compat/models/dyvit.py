""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_index_select

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class SquareAct(nn.Module):
    def __init__(self, init_alpha=0.25, learnable=False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            raw_init = math.log(math.expm1(init_alpha))
            self.raw_alpha = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_buffer('fixed_alpha', torch.tensor(init_alpha, dtype=torch.float32))

    def current_alpha(self):
        if self.learnable:
            return F.softplus(self.raw_alpha)
        return self.fixed_alpha

    def forward(self, x):
        output_dtype = x.dtype
        if x.dtype != torch.float32 or torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                y = self.current_alpha() * (x_fp32 * x_fp32)
            return y.to(output_dtype)
        return self.current_alpha() * (x * x)

    def extra_repr(self):
        mode = 'learnable' if self.learnable else 'fixed'
        return f'mode={mode}, alpha={self.current_alpha().detach().item():.6f}'


class QuadraticAct(nn.Module):
    def __init__(self, init_alpha=0.0, init_beta=1.0, learnable=True):
        super().__init__()
        if not learnable:
            raise ValueError('QuadraticAct is intended for learnable coefficients.')
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

    def current_alpha(self):
        return self.alpha

    def current_beta(self):
        return self.beta

    def forward(self, x):
        output_dtype = x.dtype
        if x.dtype != torch.float32 or torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                y = self.alpha * (x_fp32 * x_fp32) + self.beta * x_fp32
            return y.to(output_dtype)
        return self.alpha * (x * x) + self.beta * x

    def extra_repr(self):
        return (
            f"alpha={self.current_alpha().detach().item():.6f}, "
            f"beta={self.current_beta().detach().item():.6f}"
        )


def get_act_layer(act_layer):
    if isinstance(act_layer, str):
        act_layer = act_layer.lower()
        if act_layer == 'gelu':
            return nn.GELU
        if act_layer == 'square' or act_layer == 'fixed_square':
            return partial(SquareAct, learnable=False, init_alpha=0.25)
        if act_layer == 'learnable_square':
            return partial(SquareAct, learnable=True, init_alpha=1.0)
        if act_layer == 'learnable_quadratic':
            return partial(QuadraticAct, init_alpha=0.0, init_beta=1.0, learnable=True)
        if act_layer == 'learnable_quadratic_gelu_init':
            return partial(QuadraticAct, init_alpha=0.3, init_beta=0.5, learnable=True)
        raise ValueError(f'Unsupported act_layer: {act_layer}')
    if isinstance(act_layer, nn.Module):
        return act_layer.__class__
    return act_layer


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        act_layer = get_act_layer(act_layer)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.use_fp32_square_family = isinstance(self.act, (SquareAct, QuadraticAct))
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_fp32_square_family and torch.is_autocast_enabled():
            output_dtype = x.dtype
            with torch.cuda.amp.autocast(enabled=False):
                x = self.fc1(x.float())
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.drop(x)
            return x.to(output_dtype)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_approx_attn=False, approx_attn_mode='relu', fp32_attention=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_approx_attn = use_approx_attn
        self.approx_attn_mode = approx_attn_mode
        self.fp32_attention = fp32_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_attn_policy(self, policy):
        B, N, _ = policy.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        return attn_policy + (1.0 - attn_policy) * eye

    def _relu_attention(self, attn, attn_policy=None, eps=1e-6):
        attn = F.relu(attn)
        if attn_policy is not None:
            attn = attn * attn_policy.to(attn.dtype)
        N = attn.shape[-1]
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, _, _ = policy.size()
        _, _, N, _ = attn.size()
        attn_policy = self._build_attn_policy(policy)
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn

    def forward(self, x, policy):
        if self.fp32_attention and torch.is_autocast_enabled():
            output_dtype = x.dtype
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                policy_fp32 = None if policy is None else policy.float()
                out = self._forward_impl(x_fp32, policy_fp32)
            return out.to(output_dtype)
        return self._forward_impl(x, policy)

    def _forward_impl(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_approx_attn:
            attn_policy = None if policy is None else self._build_attn_policy(policy)
            attn = self._relu_attention(attn, attn_policy=attn_policy)
        elif policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer='gelu', norm_layer=nn.LayerNorm,
                 use_approx_attn=False, approx_attn_mode='relu', fp32_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_approx_attn=use_approx_attn, approx_attn_mode=approx_attn_mode, fp32_attention=fp32_attention)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None, return_debug=False):
        norm1_out = self.norm1(x)
        attn_out = self.attn(norm1_out, policy=policy)
        x = x + self.drop_path(attn_out)
        attn_residual_out = x

        if self.mlp.use_fp32_square_family and torch.is_autocast_enabled():
            output_dtype = x.dtype
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                norm2_out = self.norm2(x_fp32)
                mlp_out = self.drop_path(self.mlp(norm2_out))
                x = x_fp32 + mlp_out
            x = x.to(output_dtype)
        else:
            norm2_out = self.norm2(x)
            mlp_out = self.drop_path(self.mlp(norm2_out))
            x = x + mlp_out

        if return_debug:
            return x, {
                'norm1_out': norm1_out,
                'attn_out': attn_out,
                'attn_residual_out': attn_residual_out,
                'norm2_out': norm2_out,
                'mlp_out': mlp_out,
            }
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, act_layer='gelu', nonempty_keep_guard=False):
        super().__init__()
        predictor_act_layer = get_act_layer(act_layer)
        self.nonempty_keep_guard = bool(nonempty_keep_guard)
        self.debug_nan = False
        self.debug_context = ""
        self._debug_predictor_logged = set()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            predictor_act_layer()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            predictor_act_layer(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            predictor_act_layer(),
        )
        self.out_proj = nn.Linear(embed_dim // 4, 2)

    def set_debug_nan(self, enabled):
        self.debug_nan = enabled
        if not enabled:
            self._debug_predictor_logged.clear()

    def set_debug_context(self, context):
        self.debug_context = context

    def _tensor_stats(self, tensor):
        detached = tensor.detach().float()
        finite_mask = torch.isfinite(detached)
        finite = detached[finite_mask]
        if finite.numel() == 0:
            return f"isfinite=False shape={tuple(detached.shape)} finite_count=0"
        return (
            f"isfinite={bool(finite_mask.all())} shape={tuple(detached.shape)} "
            f"min={finite.min().item():.6e} max={finite.max().item():.6e} mean={finite.mean().item():.6e}"
        )

    def _debug_log(self, name, tensor):
        if not self.debug_nan:
            return
        key = name
        is_all_finite = torch.isfinite(tensor).all().item()
        if is_all_finite and key in self._debug_predictor_logged:
            return
        print(f"[NaNDebug][{self.debug_context}] module=PredictorLG tensor={name} {self._tensor_stats(tensor)}")
        if is_all_finite:
            self._debug_predictor_logged.add(key)

    def forward(self, x, policy):
        output_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = self.in_conv(x.float())
            self._debug_log('in_conv_out', x)
            B, N, C = x.size()
            policy_float = policy.float()
            global_input = x[:, :, C//2:] * policy_float
            self._debug_log('global_input', global_input)
            active_count = torch.sum(policy_float, dim=1, keepdim=True)
            if self.debug_nan and torch.any(active_count == 0):
                zero_count = int((active_count == 0).sum().item())
                print(
                    f"[NaNDebug][{self.debug_context}] "
                    f"module=PredictorLG zero_active_policy_samples={zero_count}"
                )
            if self.nonempty_keep_guard:
                global_x = global_input.sum(dim=1, keepdim=True) / active_count.clamp_min(1.0)
            else:
                global_x = global_input.sum(dim=1, keepdim=True) / active_count
            self._debug_log('global_x', global_x)
            x = torch.cat([x[:, :, :C//2], global_x.expand(B, N, C//2)], dim=-1)
            self._debug_log('post_agg', x)
            x = self.out_conv(x)
            logits = self.out_proj(x)
            logits = logits.clamp(-10.0, 10.0)
            self._debug_log('out_conv_out', logits)
            x = F.log_softmax(logits, dim=-1)
        return x.to(output_dtype)


class VisionTransformerDiffPruning(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 pruning_loc=None, token_ratio=None, distill=False, act_layer='square',
                 use_mask_pruning=False, use_approx_attn=False, approx_attn_mode='relu',
                 fp32_attention=False, nonempty_keep_guard=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print('## diff vit pruning method')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,
                use_approx_attn=use_approx_attn, approx_attn_mode=approx_attn_mode, fp32_attention=fp32_attention)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        predictor_list = [
            PredictorLG(embed_dim, act_layer=act_layer, nonempty_keep_guard=nonempty_keep_guard)
            for _ in range(len(pruning_loc))
        ]

        self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.use_mask_pruning = use_mask_pruning
        self.use_approx_attn = use_approx_attn
        self.approx_attn_mode = approx_attn_mode
        self.nonempty_keep_guard = bool(nonempty_keep_guard)
        self.debug_nan = False
        self.debug_context = ""
        self._debug_tensor_logged = set()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_debug_nan(self, enabled):
        self.debug_nan = enabled
        for predictor in self.score_predictor:
            predictor.set_debug_nan(enabled)
        if not enabled:
            self._debug_tensor_logged.clear()

    def set_debug_context(self, context):
        self.debug_context = context
        for predictor in self.score_predictor:
            predictor.set_debug_context(context)

    def get_square_alpha_summary(self, max_items=4):
        alpha_values = []
        beta_values = []
        for module in self.modules():
            if isinstance(module, SquareAct):
                alpha_values.append(module.current_alpha().detach().item())
            elif isinstance(module, QuadraticAct):
                alpha_values.append(module.current_alpha().detach().item())
                beta_values.append(module.current_beta().detach().item())
        if not alpha_values:
            return "count=0"
        alpha_preview = ", ".join(f"{value:.6f}" for value in alpha_values[:max_items])
        mean_alpha = sum(alpha_values) / len(alpha_values)
        summary = (
            f"count={len(alpha_values)} alpha_first=[{alpha_preview}] "
            f"alpha_min={min(alpha_values):.6f} alpha_max={max(alpha_values):.6f} alpha_mean={mean_alpha:.6f}"
        )
        if beta_values:
            beta_preview = ", ".join(f"{value:.6f}" for value in beta_values[:max_items])
            mean_beta = sum(beta_values) / len(beta_values)
            summary += (
                f" beta_first=[{beta_preview}] "
                f"beta_min={min(beta_values):.6f} beta_max={max(beta_values):.6f} beta_mean={mean_beta:.6f}"
            )
        return summary

    def _tensor_stats(self, tensor):
        if tensor is None:
            return "value=None"
        detached = tensor.detach()
        shape = tuple(detached.shape)
        if detached.numel() == 0:
            return f"shape={shape} empty=True"
        detached = detached.float()
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            return f"shape={shape} finite_count=0"
        return (
            f"shape={shape} "
            f"min={finite.min().item():.6e} "
            f"max={finite.max().item():.6e} "
            f"mean={finite.mean().item():.6e}"
        )

    def _input_stats(self, tensor):
        if tensor is None:
            return "value=None"
        detached = tensor.detach()
        shape = tuple(detached.shape)
        if detached.numel() == 0:
            return f"shape={shape} empty=True"
        detached = detached.float()
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            return f"shape={shape} finite_count=0"
        std = finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0
        return (
            f"shape={shape} "
            f"min={finite.min().item():.6e} "
            f"max={finite.max().item():.6e} "
            f"mean={finite.mean().item():.6e} "
            f"std={std:.6e}"
        )

    def _check_finite(self, name, tensor):
        if not self.debug_nan or tensor is None:
            return
        if torch.is_tensor(tensor) and torch.isfinite(tensor).all():
            return
        print(
            f"[NaNDebug][{self.debug_context}] "
            f"module=VisionTransformerDiffPruning tensor={name} {self._tensor_stats(tensor)}"
        )
        raise RuntimeError(f"Non-finite tensor detected in VisionTransformerDiffPruning: {name}")

    def _debug_step(self):
        if 'step=' not in self.debug_context:
            return None
        try:
            return int(self.debug_context.split('step=')[-1])
        except ValueError:
            return None

    def _debug_log_tensor(self, name, tensor, threshold=1e3):
        if not self.debug_nan or tensor is None:
            return
        step = self._debug_step()
        should_log = step in {0, 10, 20, 30, 40}
        tensor_float = tensor.detach().float()
        finite_mask = torch.isfinite(tensor_float)
        finite = tensor_float[finite_mask]
        max_abs = finite.abs().max().item() if finite.numel() > 0 else float('inf')
        if not finite_mask.all().item() or max_abs > threshold:
            should_log = True
        key = f"{step}:{name}"
        if not should_log or key in self._debug_tensor_logged:
            return
        self._debug_tensor_logged.add(key)
        print(f"[NaNDebug][{self.debug_context}] module=VisionTransformerDiffPruning tensor={name} {self._tensor_stats(tensor)}")

    def _debug_log_input(self, name, tensor):
        if not self.debug_nan or tensor is None:
            return
        step = self._debug_step()
        if step not in {0, 10, 20, 30, 40}:
            return
        key = f"{step}:{name}"
        if key in self._debug_tensor_logged:
            return
        self._debug_tensor_logged.add(key)
        print(f"[NaNDebug][{self.debug_context}] module=VisionTransformerDiffPruning tensor={name} {self._input_stats(tensor)}")

    def _apply_spatial_mask(self, x, decision):
        if decision is None:
            return x
        return torch.cat([x[:, :1], x[:, 1:] * decision], dim=1)

    def _ensure_non_empty_keep_decision(self, hard_keep_decision, pred_score, prev_decision, predictor_index):
        if not self.nonempty_keep_guard:
            return hard_keep_decision
        keep_counts = hard_keep_decision.sum(dim=1)
        empty_mask = keep_counts.squeeze(-1) <= 0
        if not empty_mask.any():
            return hard_keep_decision

        active_before = prev_decision.squeeze(-1) > 0
        keep_score = pred_score[:, :, 0].masked_fill(~active_before, float('-inf'))
        fallback_index = keep_score.argmax(dim=1, keepdim=True)
        fallback = torch.zeros_like(hard_keep_decision.squeeze(-1))
        fallback.scatter_(1, fallback_index, 1.0)
        recovered = torch.where(
            empty_mask.unsqueeze(-1),
            fallback,
            hard_keep_decision.squeeze(-1),
        ).unsqueeze(-1)
        recovered = recovered * prev_decision
        if self.debug_nan:
            empty_count = int(empty_mask.sum().item())
            print(
                f"[NaNDebug][{self.debug_context}] "
                f"module=VisionTransformerDiffPruning "
                f"predictor_{predictor_index}_empty_keep_samples={empty_count} "
                f"fallback=single_token"
            )
        return recovered

    def forward(self, x):
        B = x.shape[0]
        self._debug_log_input('patch_embed_input', x)
        self._check_finite('patch_embed_input', x)
        x = self.patch_embed(x)
        self._debug_log_tensor('patch_embed', x)
        self._check_finite('patch_embed', x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        self._debug_log_tensor('seq_with_cls_pos', x)
        self._check_finite('pos_drop', x)

        p_count = 0
        out_pred_prob = []
        init_n = 14 * 14
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + 1, 1, dtype=x.dtype, device=x.device)
        self._check_finite('prev_decision_init', prev_decision)
        self._check_finite('policy_init', policy)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                if self.use_mask_pruning:
                    x = self._apply_spatial_mask(x, prev_decision)
                    self._check_finite(f'block_{i}_masked_input', x)
                spatial_x = x[:, 1:]
                self._check_finite(f'block_{i}_spatial_x', spatial_x)
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                self._debug_log_tensor(f'predictor_{p_count}_pred_score', pred_score)
                self._check_finite(f'predictor_{p_count}_pred_score', pred_score)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    hard_keep_decision = self._ensure_non_empty_keep_decision(
                        hard_keep_decision, pred_score, prev_decision, p_count
                    )
                    self._debug_log_tensor(f'predictor_{p_count}_hard_keep_decision', hard_keep_decision)
                    self._check_finite(f'predictor_{p_count}_hard_keep_decision', hard_keep_decision)
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                    self._debug_log_tensor(f'predictor_{p_count}_policy', policy)
                    self._check_finite(f'predictor_{p_count}_policy', policy)
                    x = blk(x, policy=policy)
                    self._check_finite(f'block_{i}_output', x)
                    if self.use_mask_pruning:
                        x = self._apply_spatial_mask(x, hard_keep_decision)
                        self._check_finite(f'block_{i}_masked_output', x)
                    prev_decision = hard_keep_decision
                    self._check_finite(f'predictor_{p_count}_prev_decision', prev_decision)
                else:
                    score = pred_score[:, :, 0]
                    self._debug_log_tensor(f'predictor_{p_count}_score_eval', score)
                    score = score.masked_fill(prev_decision.squeeze(-1) == 0, float('-inf'))
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    new_mask = torch.zeros_like(prev_decision)
                    new_mask.scatter_(1, keep_policy.unsqueeze(-1), 1.0)
                    prev_decision = new_mask * prev_decision
                    self._debug_log_tensor(f'predictor_{p_count}_prev_decision_eval', prev_decision)
                    x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
                    cls_policy = torch.ones(B, 1, 1, dtype=prev_decision.dtype, device=prev_decision.device)
                    policy = torch.cat([cls_policy, prev_decision], dim=1)
                    self._debug_log_tensor(f'predictor_{p_count}_policy_eval', policy)
                    x = blk(x, policy=policy)
                    x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)
                p_count += 1
            else:
                if self.training:
                    if i == 0 and self.debug_nan:
                        x, block_debug = blk(x, policy, return_debug=True)
                        self._debug_log_tensor('block0_norm1_out', block_debug['norm1_out'])
                        self._debug_log_tensor('block0_attn_out', block_debug['attn_out'])
                        self._debug_log_tensor('block0_attn_residual_out', block_debug['attn_residual_out'])
                        self._debug_log_tensor('block0_norm2_out', block_debug['norm2_out'])
                        self._debug_log_tensor('block0_mlp_out', block_debug['mlp_out'])
                    else:
                        x = blk(x, policy)
                    self._check_finite(f'block_{i}_output', x)
                    if self.use_mask_pruning:
                        x = self._apply_spatial_mask(x, prev_decision)
                        self._check_finite(f'block_{i}_masked_output', x)
                else:
                    x = blk(x, policy=policy)
                    x = torch.cat([x[:, :1], x[:, 1:] * prev_decision], dim=1)

        x = self.norm(x)
        self._check_finite('norm_output', x)
        features = x[:, 1:]
        self._check_finite('features', features)
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        self._check_finite('head_output', x)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob
            else:
                return x, out_pred_prob
        else:
            return x

class VisionTransformerTeacher(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print('## diff vit pruning method')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        feature = self.norm(x)
        cls = feature[:, 0]
        tokens = feature[:, 1:]
        cls = self.pre_logits(cls)
        cls = self.head(cls)
        return cls, tokens

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
