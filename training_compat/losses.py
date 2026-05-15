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

        pred_loss = pred.new_zeros(())

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
                 keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True,
                 pruning_margin_weight=0.0, pruning_margin_target=1e-4, pruning_margin_mode='hinge',
                 pruning_margin_stage_weights=None, pruning_margin_start_epoch=0):
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
        self.pruning_margin_loss = 0
        self.last_pruning_margin_stats = []
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.cls_distill_weight = distill_weight if cls_distill_weight is None else cls_distill_weight
        self.token_distill_weight = distill_weight if token_distill_weight is None else token_distill_weight
        self.pruning_margin_weight = pruning_margin_weight
        self.pruning_margin_target = pruning_margin_target
        self.pruning_margin_mode = pruning_margin_mode
        self.pruning_margin_stage_weights = self._normalize_pruning_margin_stage_weights(pruning_margin_stage_weights)
        self.pruning_margin_start_epoch = max(int(pruning_margin_start_epoch), 0)
        self.current_epoch = 0
        self.debug_nan = False
        self.debug_context = ""

        print(
            'ratio_weight=', ratio_weight,
            'cls_distill_weight', self.cls_distill_weight,
            'token_distill_weight', self.token_distill_weight,
            'pruning_margin_weight', self.pruning_margin_weight,
            'pruning_margin_target', self.pruning_margin_target,
            'pruning_margin_mode', self.pruning_margin_mode,
            'pruning_margin_stage_weights', self.pruning_margin_stage_weights,
            'pruning_margin_start_epoch', self.pruning_margin_start_epoch
        )


        if dynamic:
            print('using dynamic loss')

    def set_debug_nan(self, enabled):
        self.debug_nan = enabled

    def set_debug_context(self, context):
        self.debug_context = context

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _normalize_pruning_margin_stage_weights(self, stage_weights):
        if stage_weights is None:
            return None
        if isinstance(stage_weights, str):
            raw = stage_weights.strip()
            if not raw:
                return None
            parts = raw.replace(',', ' ').replace(';', ' ').split()
            stage_weights = parts
        else:
            stage_weights = list(stage_weights)
        if not stage_weights:
            return None
        normalized = [float(item) for item in stage_weights]
        if len(normalized) != len(self.pruning_loc):
            raise ValueError(
                f'pruning_margin_stage_weights length mismatch: '
                f'expected {len(self.pruning_loc)}, got {len(normalized)}'
            )
        return normalized

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

    def _compute_pruning_margin_loss(self, pruning_diagnostics, reference_tensor):
        self.last_pruning_margin_stats = []
        if self.pruning_margin_weight <= 0 or pruning_diagnostics is None:
            return reference_tensor.new_zeros(())
        if self.current_epoch < self.pruning_margin_start_epoch:
            self.last_pruning_margin_stats.append(
                {
                    'status': 'before_start_epoch',
                    'current_epoch': self.current_epoch,
                    'start_epoch': self.pruning_margin_start_epoch,
                }
            )
            return reference_tensor.new_zeros(())

        stage_reports = pruning_diagnostics.get('stage_reports')
        if not stage_reports:
            return reference_tensor.new_zeros(())

        weighted_stage_losses = []
        active_stage_weights = []
        for stage_report in stage_reports:
            keep_log_score = stage_report.get('keep_log_score')
            active_before = stage_report.get('active_before')
            stage_index = int(stage_report.get('stage_index', len(weighted_stage_losses)))
            stage_weight = 1.0
            if self.pruning_margin_stage_weights is not None:
                if stage_index < 0 or stage_index >= len(self.pruning_margin_stage_weights):
                    self.last_pruning_margin_stats.append(
                        {'stage_index': stage_index, 'status': 'stage_weight_index_mismatch'}
                    )
                    continue
                stage_weight = float(self.pruning_margin_stage_weights[stage_index])
                if stage_weight <= 0:
                    self.last_pruning_margin_stats.append(
                        {
                            'stage_index': stage_index,
                            'status': 'disabled_by_stage_weight',
                            'stage_weight': stage_weight,
                        }
                    )
                    continue
            if keep_log_score is None or active_before is None:
                self.last_pruning_margin_stats.append(
                    {'stage_index': stage_index, 'status': 'missing_inputs', 'stage_weight': stage_weight}
                )
                continue

            keep_log_score = keep_log_score.float()
            active_before = active_before.bool()
            if keep_log_score.ndim != 2 or active_before.shape != keep_log_score.shape:
                self.last_pruning_margin_stats.append(
                    {'stage_index': stage_index, 'status': 'shape_mismatch', 'stage_weight': stage_weight}
                )
                continue

            active_counts = active_before.sum(dim=1)
            masked_score = keep_log_score.masked_fill(~active_before, float('-inf'))

            if stage_index <= 0:
                conditional_keep_ratio = float(self.keep_ratio[0])
            else:
                previous_keep_ratio = max(float(self.keep_ratio[stage_index - 1]), 1e-12)
                conditional_keep_ratio = float(self.keep_ratio[stage_index]) / previous_keep_ratio
            conditional_keep_ratio = max(0.0, min(1.0, conditional_keep_ratio))

            sample_margins = []
            sample_keep_counts = []
            for sample_index in range(masked_score.shape[0]):
                active_count = int(active_counts[sample_index].item())
                if active_count <= 1:
                    continue
                keep_count = int(round(active_count * conditional_keep_ratio))
                keep_count = max(1, min(keep_count, active_count - 1))
                topk_values = torch.topk(masked_score[sample_index], k=keep_count + 1, dim=0).values
                sample_margins.append(topk_values[keep_count - 1] - topk_values[keep_count])
                sample_keep_counts.append(keep_count)

            if not sample_margins:
                self.last_pruning_margin_stats.append(
                    {
                        'stage_index': stage_index,
                        'status': 'no_valid_samples',
                        'conditional_keep_ratio': conditional_keep_ratio,
                        'stage_weight': stage_weight,
                    }
                )
                continue

            margin = torch.stack(sample_margins)
            self._check_finite('pruning_margin_stage_margin', margin)

            margin_gap = self.pruning_margin_target - margin
            if self.pruning_margin_mode == 'hinge':
                stage_loss = F.relu(margin_gap)
            elif self.pruning_margin_mode == 'softplus':
                stage_loss = F.softplus(margin_gap)
            else:
                raise ValueError(f'unsupported pruning_margin_mode: {self.pruning_margin_mode}')
            self.last_pruning_margin_stats.append(
                {
                    'stage_index': stage_index,
                    'status': 'ok',
                    'stage_weight': stage_weight,
                    'conditional_keep_ratio': conditional_keep_ratio,
                    'sample_count': int(margin.numel()),
                    'active_mean': float(active_counts.float().mean().item()),
                    'keep_mean': float(torch.tensor(sample_keep_counts, dtype=torch.float32).mean().item()),
                    'margin_mean': float(margin.mean().item()),
                    'margin_min': float(margin.min().item()),
                    'margin_max': float(margin.max().item()),
                    'violation_ratio': float((margin_gap > 0).float().mean().item()),
                    'stage_loss_mean': float(stage_loss.mean().item()),
                }
            )
            weighted_stage_losses.append(stage_loss.mean() * stage_weight)
            active_stage_weights.append(stage_weight)

        if not weighted_stage_losses:
            return reference_tensor.new_zeros(())
        total_weight = reference_tensor.new_tensor(active_stage_weights).sum().clamp_min(1e-12)
        return torch.stack(weighted_stage_losses).sum() / total_weight

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
        pruning_diagnostics = None
        if len(outputs) == 5:
            pred, token_pred, mask, out_pred_score, pruning_diagnostics = outputs
        elif len(outputs) == 4:
            pred, token_pred, mask, out_pred_score = outputs
        elif len(outputs) == 3:
            pred, token_pred, out_pred_score = outputs
            mask = None
        else:
            raise ValueError(
                "DistillDiffPruningLoss_dynamic expects outputs of length 3, 4, or 5, "
                f"got {len(outputs)}"
            )
        self._check_finite('pred', pred)
        self._check_finite('token_pred', token_pred)
        self._check_finite('mask', mask)
        for i, score in enumerate(out_pred_score):
            self._check_finite(f'out_pred_score_{i}', score)

        pred_loss = pred.new_zeros(())

        ratio = self.keep_ratio
        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()
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
        pruning_margin_loss = self._compute_pruning_margin_loss(pruning_diagnostics, pred)
        self._check_finite('pruning_margin_loss', pruning_margin_loss)
        
        # print(cls_loss, pred_loss)
        loss = (
            self.clf_weight * cls_loss
            + self.ratio_weight * pred_loss / max(len(out_pred_score), 1)
            + self.cls_distill_weight * cls_kl_loss
            + self.token_distill_weight * token_kl_loss
            + self.pruning_margin_weight * pruning_margin_loss
        )
        self._check_finite('total_loss', loss)
        zero = loss.new_zeros(())
        loss_part = [cls_loss, pred_loss, cls_kl_loss, token_kl_loss, zero, zero, pruning_margin_loss]

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            self.pruning_margin_loss += pruning_margin_loss.item()
            self.count += 1
            if self.count == 100:
                margin_stats_text = ' margin_stats=none'
                if self.last_pruning_margin_stats:
                    parts = []
                    for item in self.last_pruning_margin_stats:
                        if item.get('status') != 'ok':
                            if 'stage_index' in item:
                                parts.append(f"s{item['stage_index']}:{item['status']}")
                            elif item.get('status') == 'before_start_epoch':
                                parts.append(
                                    "global:{status}(epoch={epoch},start={start})".format(
                                        status=item['status'],
                                        epoch=item.get('current_epoch', 'na'),
                                        start=item.get('start_epoch', 'na'),
                                    )
                                )
                            else:
                                parts.append(f"global:{item['status']}")
                            continue
                        parts.append(
                            "s{stage}:w={weight:.2f},mean={mean:.3e},viol={viol:.2f},loss={loss:.3e}".format(
                                stage=item['stage_index'],
                                weight=item.get('stage_weight', 1.0),
                                mean=item['margin_mean'],
                                viol=item['violation_ratio'],
                                loss=item['stage_loss_mean'],
                            )
                        )
                    margin_stats_text = " margin_stats=[" + "; ".join(parts) + "]"
                print(
                    'loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f, pruning_margin=%.6e%s'
                    % (
                        self.cls_loss / 100,
                        self.ratio_loss / 100,
                        self.cls_distill_loss / 100,
                        self.token_distill_loss / 100,
                        self.pruning_margin_loss / 100,
                        margin_stats_text,
                    )
                )
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
                self.pruning_margin_loss = 0
        return loss, loss_part
