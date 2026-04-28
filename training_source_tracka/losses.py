"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class ConvNextDistillDiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=10.0, distill_weight=0.5, keep_ratio=[0.9, 0.7, 0.5], clf_weight=0, mse_token=False, print_mode=True, swin_token=False):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.swin_token = swin_token

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, out_pred_score = outputs

        pred_loss = 0.0

        ratio = self.keep_ratio
            
        for i, score in enumerate(out_pred_score):
            if not self.swin_token:
                pos_ratio = score.mean(dim=(2,3))
            else:
                pos_ratio = score.mean(dim=1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t, token_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        token_kl_loss = torch.pow(token_pred - token_t, 2).mean()

        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(out_pred_score) + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss 
        loss_part = []

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            self.count += 1
            loss_part.append(cls_loss)
            loss_part.append(pred_loss)
            loss_part.append(cls_kl_loss)
            loss_part.append(token_kl_loss)
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f, layer_mse=%.4f, feat_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100, self.layer_mse_loss / 100, self.feat_distill_loss / 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss, loss_part



class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5,
                 cls_distill_weight=None, token_distill_weight=None, dynamic=False, pruning_loc=[3,6,9],
                 keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.cls_distill_weight = distill_weight if cls_distill_weight is None else cls_distill_weight
        self.token_distill_weight = distill_weight if token_distill_weight is None else token_distill_weight
        self.debug_nan = False
        self.debug_context = ""

        print(
            'ratio_weight=', ratio_weight,
            'cls_distill_weight', self.cls_distill_weight,
            'token_distill_weight', self.token_distill_weight
        )


        if dynamic:
            print('using dynamic loss')

    def set_debug_nan(self, enabled):
        self.debug_nan = enabled

    def set_debug_context(self, context):
        self.debug_context = context

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

    def _check_finite(self, name, tensor):
        if not self.debug_nan or tensor is None:
            return
        if torch.is_tensor(tensor) and torch.isfinite(tensor).all():
            return
        print(
            f"[NaNDebug][{self.debug_context}] "
            f"module=DistillDiffPruningLoss_dynamic tensor={name} {self._tensor_stats(tensor)}"
        )
        raise RuntimeError(f"Non-finite tensor detected in DistillDiffPruningLoss_dynamic: {name}")

    def _debug_step(self):
        if 'step=' not in self.debug_context:
            return None
        try:
            return int(self.debug_context.split('step=')[-1])
        except ValueError:
            return None

    def _debug_vector_stats(self, name, values):
        values = values.detach().float().reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return f"{name}_finite_count=0"
        return (
            f"{name}_min={finite.min().item():.6e} "
            f"{name}_max={finite.max().item():.6e} "
            f"{name}_mean={finite.mean().item():.6e}"
        )

    def _debug_log_ratio_stage(self, stage_index, pos_ratio, target_ratio, stage_loss):
        if not self.debug_nan:
            return
        step = self._debug_step()
        gap = pos_ratio.detach().float() - float(target_ratio)
        should_log = (
            step in {0, 10, 20, 30, 40}
            or (step is not None and step >= 140)
            or gap.abs().max().item() > 0.2
        )
        if not should_log:
            return
        print(
            f"[NaNDebug][{self.debug_context}] "
            f"module=DistillDiffPruningLoss_dynamic "
            f"ratio_stage_{stage_index} "
            f"target={float(target_ratio):.6f} "
            f"{self._debug_vector_stats('pos_ratio', pos_ratio)} "
            f"{self._debug_vector_stats('gap', gap)} "
            f"stage_loss={stage_loss.detach().float().item():.6e}"
        )

    def _resolve_token_mask(self, mask, token_pred):
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            return None

        B, N = token_pred.shape[:2]
        token_mask = mask.detach()

        if token_mask.dim() == 3 and token_mask.shape[-1] == 1:
            token_mask = token_mask.squeeze(-1)
        elif token_mask.dim() > 2:
            token_mask = token_mask.reshape(B, -1)

        if token_mask.dim() != 2 or token_mask.shape[0] != B:
            return None

        if token_mask.shape[1] == N + 1:
            token_mask = token_mask[:, 1:]
        elif token_mask.shape[1] != N:
            return None

        return token_mask > 0.5

    def _compute_token_distill_loss(self, token_pred, token_t, token_mask):
        token_pred = token_pred.float()
        token_t = token_t.float()

        if token_mask is None:
            token_pred_flat = token_pred.reshape(-1, token_pred.shape[-1])
            token_t_flat = token_t.reshape(-1, token_t.shape[-1])
            if self.mse_token:
                return torch.pow(token_pred_flat - token_t_flat, 2).mean()
            return F.kl_div(
                F.log_softmax(token_pred_flat, dim=-1),
                F.log_softmax(token_t_flat, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        valid_counts = token_mask.sum(dim=1)
        valid_samples = valid_counts > 0
        if not valid_samples.any():
            return token_pred.new_zeros(())

        if self.mse_token:
            per_token_loss = torch.pow(token_pred - token_t, 2).mean(dim=-1)
        else:
            per_token_loss = F.kl_div(
                F.log_softmax(token_pred, dim=-1),
                F.log_softmax(token_t, dim=-1),
                reduction='none',
                log_target=True
            ).sum(dim=-1)

        mask_float = token_mask.float()
        per_sample_loss = (per_token_loss * mask_float).sum(dim=1) / valid_counts.clamp_min(1).float()
        return per_sample_loss[valid_samples].mean()

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        if not isinstance(outputs, (list, tuple)):
            raise TypeError("DistillDiffPruningLoss_dynamic expects tuple/list outputs")
        if len(outputs) == 4:
            pred, token_pred, mask, out_pred_score = outputs
        elif len(outputs) == 3:
            pred, token_pred, out_pred_score = outputs
            mask = None
        else:
            raise ValueError(
                "DistillDiffPruningLoss_dynamic expects outputs of length 3 or 4, "
                f"got {len(outputs)}"
            )
        self._check_finite('pred', pred)
        self._check_finite('token_pred', token_pred)
        self._check_finite('mask', mask)
        for i, score in enumerate(out_pred_score):
            self._check_finite(f'out_pred_score_{i}', score)

        pred_loss = 0.0

        ratio = self.keep_ratio
        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            stage_loss = ((pos_ratio - ratio[i]) ** 2).mean()
            self._debug_log_ratio_stage(i, pos_ratio, ratio[i], stage_loss)
            pred_loss = pred_loss + stage_loss
        self._check_finite('ratio_loss', pred_loss)

        cls_loss = self.base_criterion(pred, labels)
        self._check_finite('cls_loss', cls_loss)

        with torch.no_grad():

            cls_t, token_t = self.teacher_model(inputs)
        self._check_finite('teacher_cls', cls_t)
        self._check_finite('teacher_token', token_t)

        pred_fp32 = pred.float()
        cls_t_fp32 = cls_t.float()
        cls_kl_loss = F.kl_div(
                F.log_softmax(pred_fp32, dim=-1),
                F.log_softmax(cls_t_fp32, dim=-1),
                reduction='batchmean',
                log_target=True
            )
        self._check_finite('cls_kl', cls_kl_loss)

        token_mask = self._resolve_token_mask(mask, token_pred)
        if token_mask is not None:
            self._check_finite('token_mask', token_mask.float())
        token_kl_loss = self._compute_token_distill_loss(token_pred, token_t, token_mask)
        self._check_finite('token_kl', token_kl_loss)
        
        # print(cls_loss, pred_loss)
        loss = (
            self.clf_weight * cls_loss
            + self.ratio_weight * pred_loss / len(self.pruning_loc)
            + self.cls_distill_weight * cls_kl_loss
            + self.token_distill_weight * token_kl_loss
        )
        self._check_finite('total_loss', loss)
        zero = loss.new_zeros(())
        loss_part = [cls_loss, pred_loss, cls_kl_loss, token_kl_loss, zero, zero]

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss, loss_part
