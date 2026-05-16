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
                self.count = 0