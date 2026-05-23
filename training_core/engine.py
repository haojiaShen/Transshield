# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from . import utils

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                    debug_nan=False, debug_max_steps=0):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    debug_context = None

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if debug_nan:
            debug_context = f"epoch={epoch} step={data_iter_step}"
            debug_model = model.module if hasattr(model, 'module') else model
            if hasattr(debug_model, 'set_debug_context'):
                debug_model.set_debug_context(debug_context)
            if hasattr(criterion, 'set_debug_context'):
                criterion.set_debug_context(debug_context)
            if hasattr(loss_scaler, 'set_debug_context'):
                loss_scaler.set_debug_context(debug_context)
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss, loss_part = criterion(samples, output, targets)
        else: # full precision
            output = model(samples)
            loss, loss_part = criterion(samples, output, targets)

        if not isinstance(loss_part, (list, tuple)):
            loss_part = [loss_part]
        loss_part = list(loss_part)
        if len(loss_part) < 7:
            zero = loss.new_zeros(())
            loss_part.extend([zero] * (7 - len(loss_part)))
        elif len(loss_part) > 7:
            loss_part = loss_part[:7]

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        grad_norm = None

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                parameters = [p for p in model.parameters() if p.requires_grad]
                nonfinite_grad = utils.find_first_nonfinite_grad(
                    parameters, debug_nan=debug_nan, debug_context=debug_context)
                if nonfinite_grad is not None:
                    grad_norm = torch.tensor(
                        float('nan'),
                        device=parameters[0].grad.device if parameters and parameters[0].grad is not None else samples.device
                    )
                elif max_norm is not None:
                    utils.maybe_log_grad_watch(parameters, debug_nan=debug_nan, debug_context=debug_context)
                    grad_norm = utils.get_grad_norm_(parameters, debug_nan=False, debug_context=debug_context)
                    if torch.isfinite(grad_norm):
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
                else:
                    grad_norm = utils.get_grad_norm_(parameters, debug_nan=debug_nan, debug_context=debug_context)

                if grad_norm is None or torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print("[NaNDebug][grad] skip optimizer step due to non-finite gradient")
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if mixup_fn is None:
            output_for_acc = output[0] if isinstance(output, tuple) else output
            class_acc = (output_for_acc.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(cls_loss=loss_part[0], head="loss")
            log_writer.update(ratio_loss=loss_part[1], head="loss")
            log_writer.update(cls_distill_loss=loss_part[2], head="loss")
            log_writer.update(token_distill_loss=loss_part[3], head="loss")
            log_writer.update(layer_mse_loss=loss_part[4], head="loss")
            log_writer.update(feat_distill_loss=loss_part[5], head="loss")
            log_writer.update(pruning_margin_loss=loss_part[6], head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if grad_norm is not None:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if debug_max_steps > 0 and (data_iter_step + 1) >= debug_max_steps:
            print(f"Debug max steps reached at epoch={epoch} step={data_iter_step}")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def _binary_threshold_accuracy(output, target, threshold):
    if output.shape[-1] != 2:
        raise ValueError('binary threshold evaluation requires exactly 2 logits')
    class1_prob = torch.softmax(output, dim=-1)[:, 1]
    pred = (class1_prob >= threshold).long()
    acc1 = (pred == target).float().mean() * 100.0
    return acc1


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, binary_threshold=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    i = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        i += 1
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        max_k = min(5, output.shape[-1])
        acc_values = accuracy(output, target, topk=(1, max_k))
        if binary_threshold is not None:
            acc1 = _binary_threshold_accuracy(output, target, binary_threshold)
        else:
            acc1 = acc_values[0]
        acc5 = acc_values[1] if max_k > 1 else acc_values[0]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
