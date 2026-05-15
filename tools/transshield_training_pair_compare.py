#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def maybe_load_json(path_str: str) -> Optional[Dict[str, Any]]:
    if not path_str:
        return None
    path = Path(path_str).resolve()
    if not path.exists():
        return None
    return load_json(path)


def maybe_load_train_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return None
    raw = log_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # training_compat/main.py appends one JSON object per epoch line.
        # For multi-epoch runs we take the last valid JSON line as the final summary.
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"cannot parse appended JSON training log: {log_path}")


def maybe_parse_command(run_dir: Path) -> Optional[Dict[str, Any]]:
    command_path = run_dir / "command.sh"
    if not command_path.exists():
        return None
    raw = command_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    tokens = shlex.split(raw)
    parsed: Dict[str, Any] = {"raw": raw}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            parsed[key] = tokens[i + 1]
            i += 2
        else:
            parsed[key] = True
            i += 1
    return parsed


def metric_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return float(rhs - lhs)


def safe_get(container: Optional[Dict[str, Any]], *keys: str) -> Any:
    current: Any = container
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def list_accuracy(predictions: List[int], targets: List[int]) -> float:
    return float(sum(1 for pred, target in zip(predictions, targets) if pred == target) / len(targets) * 100.0)


def list_binary_f1(predictions: List[int], targets: List[int]) -> float:
    true_positive = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    false_positive = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    false_negative = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def list_binary_auc(scores: List[float], targets: List[int]) -> Optional[float]:
    positives = [score for score, target in zip(scores, targets) if target == 1]
    negatives = [score for score, target in zip(scores, targets) if target == 0]
    if not positives or not negatives:
        return None
    greater = 0
    equal = 0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                greater += 1
            elif positive == negative:
                equal += 1
    return float((greater + 0.5 * equal) / (len(positives) * len(negatives)))


def list_cross_entropy(logit_pairs: List[List[float]], targets: List[int]) -> float:
    total = 0.0
    for logits, target in zip(logit_pairs, targets):
        z0, z1 = logits
        max_logit = max(z0, z1)
        logsumexp = max_logit + math.log(math.exp(z0 - max_logit) + math.exp(z1 - max_logit))
        total += -(z1 if target == 1 else z0) + logsumexp
    return float(total / len(targets))


def summarize_public_logit_bias_calibration(
    eval_payload: Optional[Dict[str, Any]],
    threshold_payload: Optional[Dict[str, Any]],
    csv_path_str: str,
    epsilon: float,
) -> Optional[Dict[str, Any]]:
    if eval_payload is None or threshold_payload is None or not csv_path_str:
        return None
    csv_path = Path(csv_path_str).resolve()
    if not csv_path.exists():
        return None
    rows = load_csv_rows(csv_path)
    if not rows:
        return None

    threshold = threshold_payload.get("eval_binary_threshold")
    if threshold is None:
        threshold = safe_get(eval_payload, "metrics", "threshold")
    if threshold is None:
        return None
    threshold = float(threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        return {
            "status": "public_bias_not_applicable",
            "reason": "threshold is outside the open interval (0, 1), so it cannot be converted to a finite logit bias.",
            "threshold": threshold,
            "plaintext_eval_csv": str(csv_path),
        }

    class1_bias = math.log((1.0 - threshold) / threshold)
    effective_bias = class1_bias + float(epsilon)

    targets = [int(row["target"]) for row in rows]
    logits = [[float(row["logit_0"]), float(row["logit_1"])] for row in rows]
    class1_probs = [float(row["prob_1"]) for row in rows]
    original_scores = [z1 - z0 for z0, z1 in logits]
    calibrated_logits = [[z0, z1 + effective_bias] for z0, z1 in logits]
    calibrated_scores = [z1 - z0 for z0, z1 in calibrated_logits]

    original_argmax = [1 if z1 >= z0 else 0 for z0, z1 in logits]
    threshold_predictions = [1 if prob >= threshold else 0 for prob in class1_probs]
    calibrated_argmax = [1 if z1 >= z0 else 0 for z0, z1 in calibrated_logits]

    original_ce = list_cross_entropy(logits, targets)
    calibrated_ce = list_cross_entropy(calibrated_logits, targets)
    threshold_accuracy = list_accuracy(threshold_predictions, targets)
    calibrated_accuracy = list_accuracy(calibrated_argmax, targets)
    status = (
        "public_bias_recovers_threshold_argmax"
        if calibrated_accuracy >= threshold_accuracy - 1e-9
        else "public_bias_nearly_recovers_threshold_argmax"
    )

    return {
        "status": status,
        "reason": "公开 class-1 logit bias 将 best threshold 等价搬到 argmax 边界；该操作不改变 ViT/SPU 主体算子。",
        "plaintext_eval_csv": str(csv_path),
        "threshold": threshold,
        "effective_class1_logit_bias": effective_bias,
        "sample_count": len(rows),
        "sample_paths_sha256": eval_payload.get("sample_paths_sha256"),
        "metrics": {
            "original_argmax_accuracy": list_accuracy(original_argmax, targets),
            "threshold_accuracy": threshold_accuracy,
            "calibrated_argmax_accuracy": calibrated_accuracy,
            "original_argmax_f1": list_binary_f1(original_argmax, targets),
            "threshold_f1": list_binary_f1(threshold_predictions, targets),
            "calibrated_argmax_f1": list_binary_f1(calibrated_argmax, targets),
            "original_auc": list_binary_auc(original_scores, targets),
            "calibrated_auc": list_binary_auc(calibrated_scores, targets),
            "original_ce_loss": original_ce,
            "calibrated_ce_loss": calibrated_ce,
            "calibrated_minus_original_argmax_accuracy": calibrated_accuracy - list_accuracy(original_argmax, targets),
            "calibrated_minus_original_ce_loss": calibrated_ce - original_ce,
        },
        "deployment_note": {
            "operation": "add public scalar to class-1 logit before final argmax decision",
            "secure_friendly": True,
            "requires_retraining": False,
            "changes_auc_ranking": False,
        },
    }


def summarize_eval(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    metrics = payload.get("metrics", {})
    return {
        "eval_loss": metrics.get("eval_loss"),
        "auc": metrics.get("auc"),
        "argmax_accuracy": metrics.get("argmax_accuracy"),
        "argmax_f1": metrics.get("argmax_f1"),
        "threshold": metrics.get("threshold"),
        "threshold_accuracy": metrics.get("threshold_accuracy"),
        "threshold_f1": metrics.get("threshold_f1"),
        "finite_logits": payload.get("finite_logits"),
        "sample_count": payload.get("sample_count"),
        "sample_paths_sha256": payload.get("sample_paths_sha256"),
        "args_snapshot_summary": payload.get("args_snapshot_summary", {}),
    }


def summarize_threshold_search(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    return {
        "eval_binary_threshold": payload.get("eval_binary_threshold"),
        "eval_acc1": payload.get("eval_acc1"),
        "eval_loss": payload.get("eval_loss"),
        "auc": payload.get("auc"),
        "default_argmax_acc1": payload.get("default_argmax_acc1"),
        "finite_logits": payload.get("finite_logits"),
        "sample_count": payload.get("sample_count"),
    }


def summarize_margin_report(payload: Optional[Dict[str, Any]], focus_stage_index: int) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    stage_payload = safe_get(payload, "stage_summaries", str(focus_stage_index)) or {}
    return {
        "interpretation_status": safe_get(payload, "interpretation", "status"),
        "interpretation_reason": safe_get(payload, "interpretation", "reason"),
        "selected_profile": safe_get(payload, "recipe_comparison", "profile_name"),
        "recipe_matches_weight": safe_get(payload, "recipe_comparison", "matches_weight"),
        "recipe_matches_target": safe_get(payload, "recipe_comparison", "matches_target"),
        "recipe_matches_mode": safe_get(payload, "recipe_comparison", "matches_mode"),
        "recipe_matches_stage_weights_csv": safe_get(payload, "recipe_comparison", "matches_stage_weights_csv"),
        "max_pruning_margin": safe_get(payload, "log_summary", "max_pruning_margin"),
        "nonzero_pruning_margin_line_count": safe_get(payload, "log_summary", "nonzero_pruning_margin_line_count"),
        "focus_stage_index": focus_stage_index,
        "focus_stage_margin_mean": stage_payload.get("mean_margin_mean"),
        "focus_stage_violation_ratio": stage_payload.get("mean_violation_ratio"),
        "focus_stage_loss_mean": stage_payload.get("mean_stage_loss_mean"),
        "focus_stage_weight": stage_payload.get("mean_stage_weight"),
        "all_stage_summaries": payload.get("stage_summaries", {}),
    }


def summarize_distill_report(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    configured = payload.get("configured_distill", {})
    summary = payload.get("log_summary", {})
    return {
        "interpretation_status": safe_get(payload, "interpretation", "status"),
        "interpretation_reason": safe_get(payload, "interpretation", "reason"),
        "cls_distill_weight": configured.get("cls_distill_weight"),
        "token_distill_weight": configured.get("token_distill_weight"),
        "mean_cls_kl": summary.get("mean_cls_kl"),
        "mean_token_kl": summary.get("mean_token_kl"),
        "mean_effective_cls_term": summary.get("mean_effective_cls_term"),
        "mean_effective_token_term": summary.get("mean_effective_token_term"),
        "nonzero_effective_distill_line_count": summary.get("nonzero_effective_distill_line_count"),
        "loss_info_line_count": summary.get("loss_info_line_count"),
    }


def compare_configs(
    baseline_command: Optional[Dict[str, Any]],
    candidate_command: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    default_values = {
        "activation_lr_scale": "1.0",
        "aa": "rand-m9-mstd0.5-inc1",
        "augmentation_profile": "timm",
        "clip_grad": "1.0",
        "color_jitter": "0.4",
        "cls_token_full_lr": "false",
        "freeze_patch_embed_proj": "false",
        "freeze_patch_embed_weight": "false",
        "freeze_patch_embed_bias": "false",
        "patch_embed_bias_init_mode": "pretrained",
        "skip_patch_embed_bias_pretrained": "false",
        "train_pos_embed": "false",
        "reprob": "0.25",
        "weight_decay": "0.05",
    }
    interesting_keys = [
        "epochs",
        "seed",
        "batch_size",
        "augmentation_profile",
        "color_jitter",
        "aa",
        "reprob",
        "clip_grad",
        "lr",
        "min_lr",
        "warmup_steps",
        "lr_scale",
        "groupa_lr_scale",
        "activation_lr_scale",
        "cls_token_full_lr",
        "train_pos_embed",
        "freeze_patch_embed_proj",
        "freeze_patch_embed_weight",
        "freeze_patch_embed_bias",
        "patch_embed_bias_init_mode",
        "skip_patch_embed_bias_pretrained",
        "pretrained_fix_step",
        "model_ema",
        "smoothing",
        "weight_decay",
        "class_weight_mode",
        "class_weight_power",
        "train_sampler_mode",
        "secure_static_train_depth",
        "secure_static_skip_pruning",
        "use_mask_pruning",
        "cls_distill_weight",
        "token_distill_weight",
        "teacher_checkpoint_path",
        "pruning_margin_weight",
        "pruning_margin_target",
        "pruning_margin_mode",
        "pruning_margin_stage_weights",
        "pruning_margin_start_epoch",
        "finetune",
    ]
    baseline = {}
    candidate = {}
    changed = {}
    for key in interesting_keys:
        lhs = None if baseline_command is None else baseline_command.get(key)
        rhs = None if candidate_command is None else candidate_command.get(key)
        if key in default_values:
            if lhs is None:
                lhs = default_values[key]
            if rhs is None:
                rhs = default_values[key]
        baseline[key] = lhs
        candidate[key] = rhs
        if lhs != rhs:
            changed[key] = {"baseline": lhs, "candidate": rhs}
    return {"baseline": baseline, "candidate": candidate, "changed": changed}


def build_judgement(
    study_kind: str,
    baseline_eval: Optional[Dict[str, Any]],
    candidate_eval: Optional[Dict[str, Any]],
    baseline_margin: Optional[Dict[str, Any]],
    candidate_margin: Optional[Dict[str, Any]],
    baseline_distill: Optional[Dict[str, Any]],
    candidate_distill: Optional[Dict[str, Any]],
    focus_stage_index: int,
    config_compare: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    baseline_threshold_acc = safe_get(baseline_eval, "threshold_accuracy")
    candidate_threshold_acc = safe_get(candidate_eval, "threshold_accuracy")
    baseline_auc = safe_get(baseline_eval, "auc")
    candidate_auc = safe_get(candidate_eval, "auc")
    candidate_margin_status = safe_get(candidate_margin, "interpretation_status")
    baseline_stage_violation = safe_get(baseline_margin, "focus_stage_violation_ratio")
    candidate_stage_violation = safe_get(candidate_margin, "focus_stage_violation_ratio")
    candidate_stage_margin = safe_get(candidate_margin, "focus_stage_margin_mean")
    candidate_recipe_matches_weight = safe_get(candidate_margin, "recipe_matches_weight")
    candidate_recipe_matches_target = safe_get(candidate_margin, "recipe_matches_target")
    candidate_recipe_matches_mode = safe_get(candidate_margin, "recipe_matches_mode")
    candidate_recipe_matches_stage_weights = safe_get(candidate_margin, "recipe_matches_stage_weights_csv")

    if study_kind == "protocol_aware_pruning":
        recipe_mismatch = any(
            value is False
            for value in [
                candidate_recipe_matches_weight,
                candidate_recipe_matches_target,
                candidate_recipe_matches_mode,
                candidate_recipe_matches_stage_weights,
            ]
        )
        if recipe_mismatch:
            return {
                "status": "candidate_profile_not_applied",
                "reason": "候选 run 的 recipe 对照出现不匹配，说明 candidate profile 没有按预期真正注入训练命令。",
                "focus_stage_index": focus_stage_index,
                "recipe_matches_weight": candidate_recipe_matches_weight,
                "recipe_matches_target": candidate_recipe_matches_target,
                "recipe_matches_mode": candidate_recipe_matches_mode,
                "recipe_matches_stage_weights_csv": candidate_recipe_matches_stage_weights,
            }
        if candidate_margin_status != "protocol_margin_stats_available":
            return {
                "status": "candidate_margin_not_observed",
                "reason": "候选 protocol-aware run 还没有产出可解析的 pruning margin stats。",
                "focus_stage_index": focus_stage_index,
            }
        if candidate_stage_violation is None:
            return {
                "status": "candidate_focus_stage_missing",
                "reason": "候选 margin report 缺少 focus stage 指标，无法判断边界是否缓解。",
                "focus_stage_index": focus_stage_index,
            }
        if baseline_stage_violation is not None and candidate_stage_violation < baseline_stage_violation:
            return {
                "status": "boundary_relief_observed",
                "reason": "候选 run 的 focus stage violation ratio 低于 baseline，可视为出现了边界缓解信号。",
                "focus_stage_index": focus_stage_index,
            }
        if candidate_stage_violation >= 0.999:
            threshold_delta = metric_delta(baseline_threshold_acc, candidate_threshold_acc)
            auc_delta = metric_delta(baseline_auc, candidate_auc)
            return {
                "status": "no_boundary_relief_yet",
                "reason": "候选 run 的 focus stage violation ratio 仍接近 1.0，当前更像是 objective 已接线但尚未缓解边界歧义。",
                "focus_stage_index": focus_stage_index,
                "candidate_focus_stage_margin_mean": candidate_stage_margin,
                "threshold_accuracy_delta": threshold_delta,
                "auc_delta": auc_delta,
            }
        return {
            "status": "partial_boundary_change_observed",
            "reason": "候选 run 已出现非满额 violation，但还需要更长训练与 full-val compare 才能判断是否值得升级为正式收益证据。",
            "focus_stage_index": focus_stage_index,
            "candidate_focus_stage_margin_mean": candidate_stage_margin,
        }

    if study_kind == "distill_compensation":
        candidate_distill_status = safe_get(candidate_distill, "interpretation_status")
        baseline_distill_status = safe_get(baseline_distill, "interpretation_status")
        candidate_effective_cls = safe_get(candidate_distill, "mean_effective_cls_term")
        candidate_effective_token = safe_get(candidate_distill, "mean_effective_token_term")
        candidate_nonzero_lines = safe_get(candidate_distill, "nonzero_effective_distill_line_count")
        threshold_delta = metric_delta(baseline_threshold_acc, candidate_threshold_acc)
        auc_delta = metric_delta(baseline_auc, candidate_auc)
        argmax_delta = metric_delta(safe_get(baseline_eval, "argmax_accuracy"), safe_get(candidate_eval, "argmax_accuracy"))

        if baseline_distill_status not in (None, "distill_disabled_reference"):
            return {
                "status": "baseline_distill_reference_invalid",
                "reason": "baseline 侧没有保持 no-distill 参考语义，当前配对不能当蒸馏补偿证据使用。",
                "baseline_distill_status": baseline_distill_status,
            }
        if candidate_distill_status != "distill_terms_observed":
            return {
                "status": "candidate_distill_not_observed",
                "reason": "候选 run 没有给出可确认的有效 distill term 读数，当前不能当蒸馏补偿证据使用。",
                "candidate_distill_status": candidate_distill_status,
            }
        if threshold_delta is not None and auc_delta is not None and threshold_delta >= 0 and auc_delta >= 0:
            return {
                "status": "distill_benefit_observed",
                "reason": "候选 run 在 threshold/AUC 上都没有低于 no-distill baseline，可作为蒸馏补偿的正向证据。",
                "threshold_accuracy_delta": threshold_delta,
                "auc_delta": auc_delta,
                "argmax_accuracy_delta": argmax_delta,
                "candidate_effective_cls_term": candidate_effective_cls,
                "candidate_effective_token_term": candidate_effective_token,
                "candidate_nonzero_effective_distill_line_count": candidate_nonzero_lines,
            }
        if threshold_delta is not None and threshold_delta >= 0 and (auc_delta is None or auc_delta > -0.001):
            return {
                "status": "distill_neutral_or_mixed",
                "reason": "候选 run 的 threshold 指标没有下降，但 AUC 改善也不明确；当前更像蒸馏已接线但收益仍偏中性。",
                "threshold_accuracy_delta": threshold_delta,
                "auc_delta": auc_delta,
                "argmax_accuracy_delta": argmax_delta,
                "candidate_effective_cls_term": candidate_effective_cls,
                "candidate_effective_token_term": candidate_effective_token,
                "candidate_nonzero_effective_distill_line_count": candidate_nonzero_lines,
            }
        return {
            "status": "no_clear_distill_benefit_yet",
            "reason": "候选 run 虽已观测到有效 distill term，但当前 full-val compare 还没有形成明确收益。",
            "threshold_accuracy_delta": threshold_delta,
            "auc_delta": auc_delta,
            "argmax_accuracy_delta": argmax_delta,
            "candidate_effective_cls_term": candidate_effective_cls,
            "candidate_effective_token_term": candidate_effective_token,
            "candidate_nonzero_effective_distill_line_count": candidate_nonzero_lines,
        }

    if study_kind == "secure_static_train_depth":
        changed = (config_compare or {}).get("changed") or {}
        changed_keys = sorted(changed.keys())
        baseline_depth = safe_get(config_compare, "baseline", "secure_static_train_depth")
        candidate_depth = safe_get(config_compare, "candidate", "secure_static_train_depth")
        threshold_delta = metric_delta(baseline_threshold_acc, candidate_threshold_acc)
        auc_delta = metric_delta(baseline_auc, candidate_auc)
        argmax_delta = metric_delta(safe_get(baseline_eval, "argmax_accuracy"), safe_get(candidate_eval, "argmax_accuracy"))
        allowed_changed_keys = {"secure_static_train_depth"}

        if baseline_depth is None or candidate_depth is None:
            return {
                "status": "depth_flag_missing",
                "reason": "配对训练命令里缺少 secure_static_train_depth，当前不能当作 train-depth 单因子证据使用。",
                "changed_keys": changed_keys,
            }
        if baseline_depth == candidate_depth:
            return {
                "status": "depth_control_invalid",
                "reason": "baseline 与 candidate 的 secure_static_train_depth 相同，当前不是有效的 depth pair-study。",
                "baseline_secure_static_train_depth": baseline_depth,
                "candidate_secure_static_train_depth": candidate_depth,
                "changed_keys": changed_keys,
            }
        unexpected_changed_keys = [key for key in changed_keys if key not in allowed_changed_keys]
        if unexpected_changed_keys:
            return {
                "status": "depth_control_not_isolated",
                "reason": "当前配对除了 secure_static_train_depth 外还改变了其他关键训练配置，不能当作单因子 depth 证据使用。",
                "baseline_secure_static_train_depth": baseline_depth,
                "candidate_secure_static_train_depth": candidate_depth,
                "changed_keys": changed_keys,
                "unexpected_changed_keys": unexpected_changed_keys,
            }
        if threshold_delta is not None and auc_delta is not None and threshold_delta >= 0 and auc_delta >= 0:
            return {
                "status": "depth_alignment_benefit_observed",
                "reason": "在其余关键训练配置保持一致时，更深的 secure_static_train_depth 没有低于 depth control，可视为 train-depth 对齐的正向证据。",
                "baseline_secure_static_train_depth": baseline_depth,
                "candidate_secure_static_train_depth": candidate_depth,
                "threshold_accuracy_delta": threshold_delta,
                "auc_delta": auc_delta,
                "argmax_accuracy_delta": argmax_delta,
                "changed_keys": changed_keys,
            }
        if threshold_delta is not None and threshold_delta >= 0 and (auc_delta is None or auc_delta > -0.001):
            return {
                "status": "depth_alignment_neutral_or_mixed",
                "reason": "更深的 secure_static_train_depth 没有带来明确 full-val 提升，但 threshold 指标未变差；当前更像 deployment-aligned 训练语义已对齐，收益仍偏中性。",
                "baseline_secure_static_train_depth": baseline_depth,
                "candidate_secure_static_train_depth": candidate_depth,
                "threshold_accuracy_delta": threshold_delta,
                "auc_delta": auc_delta,
                "argmax_accuracy_delta": argmax_delta,
                "changed_keys": changed_keys,
            }
        return {
            "status": "no_clear_depth_benefit_yet",
            "reason": "当前 depth pair-study 已实现单因子控制，但更深的 secure_static_train_depth 还没有形成明确收益。",
            "baseline_secure_static_train_depth": baseline_depth,
            "candidate_secure_static_train_depth": candidate_depth,
            "threshold_accuracy_delta": threshold_delta,
            "auc_delta": auc_delta,
            "argmax_accuracy_delta": argmax_delta,
            "changed_keys": changed_keys,
        }

    threshold_delta = metric_delta(baseline_threshold_acc, candidate_threshold_acc)
    auc_delta = metric_delta(baseline_auc, candidate_auc)
    if threshold_delta is not None and auc_delta is not None and threshold_delta >= 0 and auc_delta >= 0:
        return {
            "status": "candidate_eval_not_worse",
            "reason": "候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。",
            "threshold_accuracy_delta": threshold_delta,
            "auc_delta": auc_delta,
        }
    return {
        "status": "candidate_eval_not_improved",
        "reason": "候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。",
        "threshold_accuracy_delta": threshold_delta,
        "auc_delta": auc_delta,
    }


def build_markdown(report: Dict[str, Any]) -> str:
    baseline = report["labels"]["baseline"]
    candidate = report["labels"]["candidate"]
    train = report["train_metrics"]
    eval_compare = report["plaintext_eval_compare"]
    public_bias = report.get("public_logit_bias_calibration_compare") or {}
    margin = report["margin_compare"]
    distill = report["distill_compare"]
    judgement = report["judgement"]

    lines = [
        f"# {report['study_kind']} Pair Compare",
        "",
        "## 1. 结论",
        "",
        f"- status: `{judgement.get('status')}`",
        f"- reason: {judgement.get('reason')}",
        "",
        "## 2. 训练指标",
        "",
        f"- baseline `{baseline}` test_acc1: `{train['baseline'].get('test_acc1')}`",
        f"- candidate `{candidate}` test_acc1: `{train['candidate'].get('test_acc1')}`",
        f"- delta candidate-baseline test_acc1: `{train['delta_candidate_minus_baseline'].get('test_acc1')}`",
        f"- delta candidate-baseline train_loss: `{train['delta_candidate_minus_baseline'].get('train_loss')}`",
        "",
        "## 3. 明文评估对照",
        "",
        f"- threshold_accuracy delta: `{safe_get(eval_compare, 'delta_candidate_minus_baseline', 'threshold_accuracy')}`",
        f"- auc delta: `{safe_get(eval_compare, 'delta_candidate_minus_baseline', 'auc')}`",
        f"- argmax_accuracy delta: `{safe_get(eval_compare, 'delta_candidate_minus_baseline', 'argmax_accuracy')}`",
        "",
    ]

    has_margin = safe_get(margin, 'baseline') is not None or safe_get(margin, 'candidate') is not None
    has_distill = safe_get(distill, 'baseline') is not None or safe_get(distill, 'candidate') is not None
    has_public_bias = safe_get(public_bias, 'baseline') is not None or safe_get(public_bias, 'candidate') is not None

    next_section_index = 4
    if has_public_bias:
        lines.extend(
            [
                f"## {next_section_index}. Public Logit-Bias Calibration",
                "",
                f"- baseline calibration status: `{safe_get(public_bias, 'baseline', 'status')}`",
                f"- candidate calibration status: `{safe_get(public_bias, 'candidate', 'status')}`",
                f"- candidate class1_logit_bias: `{safe_get(public_bias, 'candidate', 'effective_class1_logit_bias')}`",
                f"- calibrated_argmax_accuracy delta: `{safe_get(public_bias, 'delta_candidate_minus_baseline', 'calibrated_argmax_accuracy')}`",
                f"- calibrated_ce_loss delta: `{safe_get(public_bias, 'delta_candidate_minus_baseline', 'calibrated_ce_loss')}`",
                f"- calibrated_auc delta: `{safe_get(public_bias, 'delta_candidate_minus_baseline', 'calibrated_auc')}`",
                "",
            ]
        )
        next_section_index += 1

    if has_margin:
        lines.extend(
            [
                f"## {next_section_index}. Margin / Boundary",
                "",
                f"- candidate margin status: `{safe_get(margin, 'candidate', 'interpretation_status')}`",
                f"- focus_stage_index: `{safe_get(margin, 'candidate', 'focus_stage_index')}`",
                f"- candidate focus_stage_violation_ratio: `{safe_get(margin, 'candidate', 'focus_stage_violation_ratio')}`",
                f"- candidate focus_stage_margin_mean: `{safe_get(margin, 'candidate', 'focus_stage_margin_mean')}`",
                f"- baseline focus_stage_violation_ratio: `{safe_get(margin, 'baseline', 'focus_stage_violation_ratio')}`",
                "",
            ]
        )
        next_section_index += 1

    if has_distill:
        lines.extend(
            [
                f"## {next_section_index}. Distill",
                "",
                f"- baseline distill status: `{safe_get(distill, 'baseline', 'interpretation_status')}`",
                f"- candidate distill status: `{safe_get(distill, 'candidate', 'interpretation_status')}`",
                f"- candidate mean_effective_cls_term: `{safe_get(distill, 'candidate', 'mean_effective_cls_term')}`",
                f"- candidate mean_effective_token_term: `{safe_get(distill, 'candidate', 'mean_effective_token_term')}`",
                f"- candidate nonzero_effective_distill_line_count: `{safe_get(distill, 'candidate', 'nonzero_effective_distill_line_count')}`",
                "",
            ]
        )
        next_section_index += 1

    lines.extend(
        [
            f"## {next_section_index}. 配置差异",
            "",
        ]
    )
    changed = report["config_compare"]["changed"]
    if not changed:
        lines.append("- none")
    else:
        for key, value in changed.items():
            lines.append(
                f"- `{key}`: baseline=`{value.get('baseline')}` candidate=`{value.get('candidate')}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two secure-static training runs under a paired study.")
    parser.add_argument("--study-kind", default="generic_pair")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-threshold-search-json", default="")
    parser.add_argument("--candidate-threshold-search-json", default="")
    parser.add_argument("--baseline-plaintext-eval-json", default="")
    parser.add_argument("--candidate-plaintext-eval-json", default="")
    parser.add_argument("--baseline-plaintext-eval-csv", default="")
    parser.add_argument("--candidate-plaintext-eval-csv", default="")
    parser.add_argument("--baseline-margin-report-json", default="")
    parser.add_argument("--candidate-margin-report-json", default="")
    parser.add_argument("--baseline-distill-report-json", default="")
    parser.add_argument("--candidate-distill-report-json", default="")
    parser.add_argument("--enable-public-logit-bias-calibration", action="store_true")
    parser.add_argument("--public-logit-bias-epsilon", type=float, default=1e-6)
    parser.add_argument("--focus-stage-index", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    baseline_run_dir = Path(args.baseline_run_dir).resolve()
    candidate_run_dir = Path(args.candidate_run_dir).resolve()

    baseline_command = maybe_parse_command(baseline_run_dir)
    candidate_command = maybe_parse_command(candidate_run_dir)
    baseline_train = maybe_load_train_metrics(baseline_run_dir)
    candidate_train = maybe_load_train_metrics(candidate_run_dir)
    baseline_eval_payload = maybe_load_json(args.baseline_plaintext_eval_json)
    candidate_eval_payload = maybe_load_json(args.candidate_plaintext_eval_json)
    baseline_threshold_payload = maybe_load_json(args.baseline_threshold_search_json)
    candidate_threshold_payload = maybe_load_json(args.candidate_threshold_search_json)
    baseline_margin_payload = maybe_load_json(args.baseline_margin_report_json)
    candidate_margin_payload = maybe_load_json(args.candidate_margin_report_json)
    baseline_distill_payload = maybe_load_json(args.baseline_distill_report_json)
    candidate_distill_payload = maybe_load_json(args.candidate_distill_report_json)

    baseline_eval = summarize_eval(baseline_eval_payload)
    candidate_eval = summarize_eval(candidate_eval_payload)
    baseline_threshold = summarize_threshold_search(baseline_threshold_payload)
    candidate_threshold = summarize_threshold_search(candidate_threshold_payload)
    baseline_margin = summarize_margin_report(baseline_margin_payload, args.focus_stage_index)
    candidate_margin = summarize_margin_report(candidate_margin_payload, args.focus_stage_index)
    baseline_distill = summarize_distill_report(baseline_distill_payload)
    candidate_distill = summarize_distill_report(candidate_distill_payload)
    baseline_public_bias = None
    candidate_public_bias = None
    if args.enable_public_logit_bias_calibration:
        baseline_public_bias = summarize_public_logit_bias_calibration(
            baseline_eval_payload,
            baseline_threshold_payload,
            args.baseline_plaintext_eval_csv,
            args.public_logit_bias_epsilon,
        )
        candidate_public_bias = summarize_public_logit_bias_calibration(
            candidate_eval_payload,
            candidate_threshold_payload,
            args.candidate_plaintext_eval_csv,
            args.public_logit_bias_epsilon,
        )

    report = {
        "study_kind": args.study_kind,
        "labels": {
            "baseline": args.baseline_label,
            "candidate": args.candidate_label,
        },
        "inputs": {
            "baseline_run_dir": str(baseline_run_dir),
            "candidate_run_dir": str(candidate_run_dir),
            "baseline_threshold_search_json": str(Path(args.baseline_threshold_search_json).resolve()) if args.baseline_threshold_search_json else None,
            "candidate_threshold_search_json": str(Path(args.candidate_threshold_search_json).resolve()) if args.candidate_threshold_search_json else None,
            "baseline_plaintext_eval_json": str(Path(args.baseline_plaintext_eval_json).resolve()) if args.baseline_plaintext_eval_json else None,
            "candidate_plaintext_eval_json": str(Path(args.candidate_plaintext_eval_json).resolve()) if args.candidate_plaintext_eval_json else None,
            "baseline_plaintext_eval_csv": str(Path(args.baseline_plaintext_eval_csv).resolve()) if args.baseline_plaintext_eval_csv else None,
            "candidate_plaintext_eval_csv": str(Path(args.candidate_plaintext_eval_csv).resolve()) if args.candidate_plaintext_eval_csv else None,
            "baseline_margin_report_json": str(Path(args.baseline_margin_report_json).resolve()) if args.baseline_margin_report_json else None,
            "candidate_margin_report_json": str(Path(args.candidate_margin_report_json).resolve()) if args.candidate_margin_report_json else None,
            "baseline_distill_report_json": str(Path(args.baseline_distill_report_json).resolve()) if args.baseline_distill_report_json else None,
            "candidate_distill_report_json": str(Path(args.candidate_distill_report_json).resolve()) if args.candidate_distill_report_json else None,
        },
        "config_compare": compare_configs(baseline_command, candidate_command),
        "train_metrics": {
            "baseline": baseline_train or {},
            "candidate": candidate_train or {},
            "delta_candidate_minus_baseline": {
                "train_loss": metric_delta(safe_get(baseline_train, "train_loss"), safe_get(candidate_train, "train_loss")),
                "train_class_acc": metric_delta(safe_get(baseline_train, "train_class_acc"), safe_get(candidate_train, "train_class_acc")),
                "test_loss": metric_delta(safe_get(baseline_train, "test_loss"), safe_get(candidate_train, "test_loss")),
                "test_acc1": metric_delta(safe_get(baseline_train, "test_acc1"), safe_get(candidate_train, "test_acc1")),
            },
        },
        "threshold_search_compare": {
            "baseline": baseline_threshold,
            "candidate": candidate_threshold,
            "delta_candidate_minus_baseline": {
                "eval_binary_threshold": metric_delta(safe_get(baseline_threshold, "eval_binary_threshold"), safe_get(candidate_threshold, "eval_binary_threshold")),
                "eval_acc1": metric_delta(safe_get(baseline_threshold, "eval_acc1"), safe_get(candidate_threshold, "eval_acc1")),
                "auc": metric_delta(safe_get(baseline_threshold, "auc"), safe_get(candidate_threshold, "auc")),
            },
        },
        "plaintext_eval_compare": {
            "sample_count_match": safe_get(baseline_eval, "sample_count") == safe_get(candidate_eval, "sample_count"),
            "sample_paths_match": safe_get(baseline_eval, "sample_paths_sha256") == safe_get(candidate_eval, "sample_paths_sha256"),
            "baseline": baseline_eval,
            "candidate": candidate_eval,
            "delta_candidate_minus_baseline": {
                "eval_loss": metric_delta(safe_get(baseline_eval, "eval_loss"), safe_get(candidate_eval, "eval_loss")),
                "auc": metric_delta(safe_get(baseline_eval, "auc"), safe_get(candidate_eval, "auc")),
                "argmax_accuracy": metric_delta(safe_get(baseline_eval, "argmax_accuracy"), safe_get(candidate_eval, "argmax_accuracy")),
                "argmax_f1": metric_delta(safe_get(baseline_eval, "argmax_f1"), safe_get(candidate_eval, "argmax_f1")),
                "threshold_accuracy": metric_delta(safe_get(baseline_eval, "threshold_accuracy"), safe_get(candidate_eval, "threshold_accuracy")),
                "threshold_f1": metric_delta(safe_get(baseline_eval, "threshold_f1"), safe_get(candidate_eval, "threshold_f1")),
            },
        },
        "margin_compare": {
            "baseline": baseline_margin,
            "candidate": candidate_margin,
            "delta_candidate_minus_baseline": {
                "max_pruning_margin": metric_delta(safe_get(baseline_margin, "max_pruning_margin"), safe_get(candidate_margin, "max_pruning_margin")),
                "focus_stage_margin_mean": metric_delta(safe_get(baseline_margin, "focus_stage_margin_mean"), safe_get(candidate_margin, "focus_stage_margin_mean")),
                "focus_stage_violation_ratio": metric_delta(safe_get(baseline_margin, "focus_stage_violation_ratio"), safe_get(candidate_margin, "focus_stage_violation_ratio")),
            },
        },
        "distill_compare": {
            "baseline": baseline_distill,
            "candidate": candidate_distill,
            "delta_candidate_minus_baseline": {
                "mean_cls_kl": metric_delta(safe_get(baseline_distill, "mean_cls_kl"), safe_get(candidate_distill, "mean_cls_kl")),
                "mean_token_kl": metric_delta(safe_get(baseline_distill, "mean_token_kl"), safe_get(candidate_distill, "mean_token_kl")),
                "mean_effective_cls_term": metric_delta(safe_get(baseline_distill, "mean_effective_cls_term"), safe_get(candidate_distill, "mean_effective_cls_term")),
                "mean_effective_token_term": metric_delta(safe_get(baseline_distill, "mean_effective_token_term"), safe_get(candidate_distill, "mean_effective_token_term")),
            },
        },
        "public_logit_bias_calibration_compare": {
            "enabled": bool(args.enable_public_logit_bias_calibration),
            "baseline": baseline_public_bias,
            "candidate": candidate_public_bias,
            "delta_candidate_minus_baseline": {
                "effective_class1_logit_bias": metric_delta(
                    safe_get(baseline_public_bias, "effective_class1_logit_bias"),
                    safe_get(candidate_public_bias, "effective_class1_logit_bias"),
                ),
                "calibrated_argmax_accuracy": metric_delta(
                    safe_get(baseline_public_bias, "metrics", "calibrated_argmax_accuracy"),
                    safe_get(candidate_public_bias, "metrics", "calibrated_argmax_accuracy"),
                ),
                "calibrated_argmax_f1": metric_delta(
                    safe_get(baseline_public_bias, "metrics", "calibrated_argmax_f1"),
                    safe_get(candidate_public_bias, "metrics", "calibrated_argmax_f1"),
                ),
                "calibrated_auc": metric_delta(
                    safe_get(baseline_public_bias, "metrics", "calibrated_auc"),
                    safe_get(candidate_public_bias, "metrics", "calibrated_auc"),
                ),
                "calibrated_ce_loss": metric_delta(
                    safe_get(baseline_public_bias, "metrics", "calibrated_ce_loss"),
                    safe_get(candidate_public_bias, "metrics", "calibrated_ce_loss"),
                ),
            },
        },
    }
    report["judgement"] = build_judgement(
        study_kind=args.study_kind,
        baseline_eval=baseline_eval,
        candidate_eval=candidate_eval,
        baseline_margin=baseline_margin,
        candidate_margin=candidate_margin,
        baseline_distill=baseline_distill,
        candidate_distill=candidate_distill,
        focus_stage_index=args.focus_stage_index,
        config_compare=report["config_compare"],
    )

    write_json(Path(args.output_json).resolve(), report)
    write_text(Path(args.output_md).resolve(), build_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
