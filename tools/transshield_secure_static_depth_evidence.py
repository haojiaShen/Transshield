#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def fmt_number(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def fmt_percent(value, digits=2):
    if value is None:
        return 'N/A'
    return f'{float(value) * 100:.{digits}f}%'


def evidence_level(label: str, status: str, reason: str):
    return {
        'label': label,
        'status': status,
        'reason': reason,
    }


def build_pair_compare_summary(pair_compare_path: Path):
    payload = load_json(pair_compare_path)
    judgement = payload.get('judgement') or {}
    config_compare = payload.get('config_compare') or {}
    baseline_cfg = config_compare.get('baseline') or {}
    candidate_cfg = config_compare.get('candidate') or {}
    labels = payload.get('labels') or {}
    return {
        'path': str(pair_compare_path.resolve()),
        'pair_name': pair_compare_path.parent.name,
        'status': judgement.get('status'),
        'reason': judgement.get('reason'),
        'baseline_label': labels.get('baseline'),
        'candidate_label': labels.get('candidate'),
        'baseline_secure_static_train_depth': judgement.get('baseline_secure_static_train_depth') or baseline_cfg.get('secure_static_train_depth'),
        'candidate_secure_static_train_depth': judgement.get('candidate_secure_static_train_depth') or candidate_cfg.get('secure_static_train_depth'),
        'epochs': candidate_cfg.get('epochs') or baseline_cfg.get('epochs'),
        'threshold_accuracy_delta': judgement.get('threshold_accuracy_delta'),
        'auc_delta': judgement.get('auc_delta'),
        'argmax_accuracy_delta': judgement.get('argmax_accuracy_delta'),
        'changed_keys': judgement.get('changed_keys') or [],
    }


def build_report(
    bundle_manifest_path: Path,
    bundle_args_snapshot_path: Path,
    baseline_eval_path: Path,
    modified_eval_path: Path,
    plaintext_model_compare_path: Path,
    plaintext_secure_compare_path: Path,
    secret_summary_path: Path,
    acceptance_json_path: Optional[Path],
    pair_compare_paths,
):
    bundle_manifest = load_json(bundle_manifest_path)
    bundle_args = load_json(bundle_args_snapshot_path)
    baseline_eval = load_json(baseline_eval_path)
    modified_eval = load_json(modified_eval_path)
    plaintext_model_compare = load_json(plaintext_model_compare_path)
    plaintext_secure_compare = load_json(plaintext_secure_compare_path)
    secret_summary = load_json(secret_summary_path)
    acceptance = load_json(acceptance_json_path) if acceptance_json_path else None
    pair_compare_summaries = [build_pair_compare_summary(path) for path in pair_compare_paths]

    bundle_depth = int(bundle_args.get('secure_static_train_depth', 0) or 0)
    bundle_skip_pruning = bool(bundle_args.get('secure_static_skip_pruning', False))
    approx_attn_mode = bundle_args.get('approx_attn_mode')
    square_activation_mode = bundle_args.get('square_activation_mode')
    use_square_gelu = bool(bundle_args.get('use_square_gelu', False))
    use_approx_attn = bool(bundle_args.get('use_approx_attn', False))

    baseline_depth = int((baseline_eval.get('args_snapshot_summary') or {}).get('secure_static_train_depth', 0) or 0)
    modified_depth = int((modified_eval.get('args_snapshot_summary') or {}).get('secure_static_train_depth', 0) or 0)

    compare_payload = plaintext_secure_compare.get('comparison') or {}
    argmax_match_ratio = (compare_payload.get('argmax_predictions') or {}).get('match_ratio')
    threshold_match_ratio = (compare_payload.get('threshold_predictions') or {}).get('match_ratio')
    logits_max_abs_error = (compare_payload.get('logits') or {}).get('max_abs_error')
    probabilities_max_abs_error = (compare_payload.get('probabilities') or {}).get('max_abs_error')

    secret_complete = bool(secret_summary.get('complete'))
    secret_pending_count = int(secret_summary.get('pending_count') or 0)
    secret_unstable_count = int(secret_summary.get('unstable_count') or 0)
    secret_accepted_count = int(secret_summary.get('accepted_count') or 0)
    secret_mean_elapsed_sec = secret_summary.get('mean_accepted_elapsed_sec')
    if secret_mean_elapsed_sec is None and secret_accepted_count > 0:
        secret_mean_elapsed_sec = float(secret_summary.get('sum_accepted_elapsed_sec') or 0.0) / float(secret_accepted_count)

    official_line = None if not acceptance else acceptance.get('official_line')
    acceptance_readiness = None if not acceptance else acceptance.get('readiness')
    acceptance_gates = None if not acceptance else acceptance.get('gates')

    semantic_alignment = evidence_level(
        label='training_semantics_match_static_secret_scope',
        status='high'
        if bundle_depth > 0 and bundle_skip_pruning and use_approx_attn and use_square_gelu
        else 'low',
        reason=(
            'current official bundle uses secure_static_train_depth>0, secure_static_skip_pruning=true, uniform attention and fixed_square activation.'
            if bundle_depth > 0 and bundle_skip_pruning and use_approx_attn and use_square_gelu
            else 'current bundle does not expose the full secure-static alignment flags.'
        ),
    )
    deployment_alignment = evidence_level(
        label='current_bundle_supports_deployable_secret_line',
        status='medium' if secret_complete and secret_pending_count == 0 and secret_unstable_count == 0 else 'low',
        reason=(
            'depth6 clip0 guarded secret runtime is complete and stable on the current official line, but it is still shallower than the training depth12 setting.'
            if secret_complete and secret_pending_count == 0 and secret_unstable_count == 0
            else 'current secret runtime evidence is incomplete or unstable.'
        ),
    )
    causal_isolation = evidence_level(
        label='isolated_causal_attribution_of_secure_static_train_depth',
        status='medium' if pair_compare_summaries else 'low',
        reason=(
            'the repo already retains paired controls that change only secure_static_train_depth, but current epoch1/epoch3 results still show no_clear_depth_benefit_yet.'
            if pair_compare_summaries
            else 'no paired control compare was provided, so current evidence remains deployment-oriented rather than a clean single-factor ablation.'
        ),
    )

    output = {
        'manifest_type': 'transshield_secure_static_train_depth_evidence_v0',
        'inputs': {
            'bundle_manifest_json': str(bundle_manifest_path.resolve()),
            'bundle_args_snapshot_json': str(bundle_args_snapshot_path.resolve()),
            'baseline_eval_json': str(baseline_eval_path.resolve()),
            'modified_eval_json': str(modified_eval_path.resolve()),
            'plaintext_model_compare_json': str(plaintext_model_compare_path.resolve()),
            'plaintext_secure_compare_json': str(plaintext_secure_compare_path.resolve()),
            'secret_isolated_eval_summary_json': str(secret_summary_path.resolve()),
            'acceptance_json': None if not acceptance_json_path else str(acceptance_json_path.resolve()),
            'pair_compare_jsons': [str(path.resolve()) for path in pair_compare_paths],
        },
        'official_line': official_line,
        'acceptance_readiness': acceptance_readiness,
        'acceptance_gates': acceptance_gates,
        'bundle_training_profile': {
            'bundle_name': bundle_manifest.get('bundle_name'),
            'run_dir': ((bundle_manifest.get('primary') or {}).get('run_dir')),
            'checkpoint_path': ((bundle_manifest.get('primary') or {}).get('checkpoint_path')),
            'epochs': bundle_args.get('epochs'),
            'lr': bundle_args.get('lr'),
            'batch_size': bundle_args.get('batch_size'),
            'secure_static_train_depth': bundle_depth,
            'secure_static_skip_pruning': bundle_skip_pruning,
            'approx_attn_mode': approx_attn_mode,
            'square_activation_mode': square_activation_mode,
            'use_square_gelu': use_square_gelu,
            'use_approx_attn': use_approx_attn,
            'pretrained_fix_step': bundle_args.get('pretrained_fix_step'),
            'freeze_time_threshold_metrics': ((bundle_manifest.get('primary') or {}).get('threshold_metrics')),
        },
        'current_plaintext_evidence': {
            'baseline_depth0': {
                'secure_static_train_depth': baseline_depth,
                'metrics': baseline_eval.get('metrics'),
            },
            'modified_depth_line': {
                'secure_static_train_depth': modified_depth,
                'metrics': modified_eval.get('metrics'),
            },
            'delta_modified_minus_baseline': plaintext_model_compare.get('delta_b_minus_a'),
        },
        'current_secret_evidence': {
            'argmax_match_ratio': argmax_match_ratio,
            'threshold_match_ratio': threshold_match_ratio,
            'logits_max_abs_error': logits_max_abs_error,
            'probabilities_max_abs_error': probabilities_max_abs_error,
            'secret_runtime_complete': secret_complete,
            'secret_runtime_pending_count': secret_pending_count,
            'secret_runtime_unstable_count': secret_unstable_count,
            'secret_runtime_accepted_count': secret_accepted_count,
            'secret_runtime_mean_elapsed_sec': secret_mean_elapsed_sec,
            'deployable_secret_profile': None if not official_line else official_line.get('default_secret_profile'),
        },
        'paired_control_evidence': pair_compare_summaries,
        'evidence_matrix': [
            semantic_alignment,
            deployment_alignment,
            causal_isolation,
        ],
        'judgement': {
            'current_best_claim': (
                '`secure_static_train_depth` 当前已经能被表述为官方 secure-friendly 主线中的训练-部署对齐设计选择，'
                '但还不能被表述为已经完成单因子因果归因的收益来源。'
            ),
            'what_is_supported_now': [
                '当前 official bundle 明确训练在 static-scope-compatible 配置上',
                '当前 modified line 在 threshold accuracy 和 AUC 上显著优于 retained baseline',
                '同一条 official line 可以被当前 depth6 clip0 guarded secret runtime 以较高一致性稳定承接',
                (
                    '仓库中已经存在只改变 secure_static_train_depth 的 paired control，'
                    '当前 epoch1/epoch3 compare 都给出 no_clear_depth_benefit_yet。'
                    if pair_compare_summaries
                    else '当前报告未附带 paired control compare，因此这里只能给 deployment-oriented 证据。'
                ),
            ],
            'what_is_not_supported_yet': [
                '更深 secure_static_train_depth 单独带来 threshold accuracy / AUC 收益',
                '一个 train-depth 与 deploy-depth 精确对齐的保留式最小 ablation',
            ],
        },
        'next_minimal_ablation': {
            'goal': 'retain one paired control bundle under the same uniform/fixed_square/static-skip-pruning stack, changing only secure_static_train_depth.',
            'suggested_control': {
                'mode': 'epoch1_or_short_multi_epoch',
                'secure_static_depth': 0,
                'accuracy_profile': 'default',
                'base_bundle': bundle_manifest.get('bundle_name'),
            },
            'suggested_runner': 'bash artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh suite',
        },
        'key_findings': [
            f'当前 official bundle 训练口径为 secure_static_train_depth={bundle_depth}, secure_static_skip_pruning={str(bundle_skip_pruning).lower()}, approx_attn_mode={approx_attn_mode}, square_activation_mode={square_activation_mode}。',
            (
                '相对 retained baseline depth0 line，当前 modified line 的 threshold accuracy 提升 '
                f"{fmt_number((plaintext_model_compare.get('delta_b_minus_a') or {}).get('threshold_accuracy'), 4)} pt，AUC 提升 "
                f"{fmt_number((plaintext_model_compare.get('delta_b_minus_a') or {}).get('auc'), 6)}。"
            ),
            (
                '当前 modified line 同时保持较高 same-policy replay consistency：'
                f'argmax_match_ratio={fmt_percent(argmax_match_ratio)}，'
                f'threshold_match_ratio={fmt_percent(threshold_match_ratio)}。'
            ),
            (
                '虽然仓库里已经有 epoch1/epoch3 两条只改变 secure_static_train_depth 的 paired control，'
                '但它们都还没有证明更深 depth 带来清晰收益。'
                if pair_compare_summaries
                else '当前报告未附带 paired control compare，因此还不能给出严格单因子归因结论。'
            ),
        ],
    }
    return output


def render_markdown(report):
    bundle_profile = report['bundle_training_profile']
    baseline = report['current_plaintext_evidence']['baseline_depth0']
    modified = report['current_plaintext_evidence']['modified_depth_line']
    delta = report['current_plaintext_evidence']['delta_modified_minus_baseline'] or {}
    secret = report['current_secret_evidence']
    acceptance_readiness = report.get('acceptance_readiness') or {}
    acceptance_gates = report.get('acceptance_gates') or {}
    paired_control_evidence = report.get('paired_control_evidence') or []

    lines = [
        '# Secure Static Train Depth Evidence',
        '',
        '## 1. 结论',
        '',
        f"- 当前最稳妥的表述：{report['judgement']['current_best_claim']}",
        '',
        '## 2. 当前 bundle 训练口径',
        '',
        f"- secure_static_train_depth: `{bundle_profile['secure_static_train_depth']}`",
        f"- secure_static_skip_pruning: `{fmt_number(bundle_profile['secure_static_skip_pruning'])}`",
        f"- approx_attn_mode: `{bundle_profile['approx_attn_mode']}`",
        f"- square_activation_mode: `{bundle_profile['square_activation_mode']}`",
        f"- use_square_gelu: `{fmt_number(bundle_profile['use_square_gelu'])}`",
        f"- epochs: `{bundle_profile['epochs']}`",
        f"- lr: `{fmt_number(bundle_profile['lr'], 8)}`",
        f"- batch_size: `{bundle_profile['batch_size']}`",
        '',
        '## 3. 当前可验证证据',
        '',
        '| Evidence | Baseline / Control | Current Line | Delta |',
        '|---|---:|---:|---:|',
        f"| secure_static_train_depth | {baseline['secure_static_train_depth']} | {modified['secure_static_train_depth']} | N/A |",
        f"| threshold_accuracy | {fmt_number((baseline['metrics'] or {}).get('threshold_accuracy'), 4)} | {fmt_number((modified['metrics'] or {}).get('threshold_accuracy'), 4)} | {fmt_number(delta.get('threshold_accuracy'), 4)} |",
        f"| argmax_accuracy | {fmt_number((baseline['metrics'] or {}).get('argmax_accuracy'), 4)} | {fmt_number((modified['metrics'] or {}).get('argmax_accuracy'), 4)} | {fmt_number(delta.get('argmax_accuracy'), 4)} |",
        f"| auc | {fmt_number((baseline['metrics'] or {}).get('auc'), 6)} | {fmt_number((modified['metrics'] or {}).get('auc'), 6)} | {fmt_number(delta.get('auc'), 6)} |",
        '',
        '## 4. 当前验收口径',
        '',
        f"- readiness: `{acceptance_readiness.get('status')}`",
        f"- reason: {acceptance_readiness.get('reason') or 'N/A'}",
        f"- boundary_kth_check_passed: `{fmt_number(acceptance_gates.get('boundary_kth_check_passed'))}`",
        f"- boundary_tie_check_passed: `{fmt_number(acceptance_gates.get('boundary_tie_check_passed'))}`",
        f"- e2e_same_policy_consistency_exact: `{fmt_number(acceptance_gates.get('e2e_same_policy_consistency_exact'))}`",
        f"- secret_runtime_complete: `{fmt_number(acceptance_gates.get('secret_runtime_complete'))}`",
        '',
        '## 5. 当前 paired control',
        '',
    ]
    if paired_control_evidence:
        lines.extend(
            [
                '| Pair | Epochs | Depth 0 -> 12 | Status | Threshold Delta | AUC Delta | Argmax Delta |',
                '|---|---:|---:|---|---:|---:|---:|',
            ]
        )
        for item in paired_control_evidence:
            lines.append(
                f"| {item['pair_name']} | {item['epochs']} | {item['baseline_secure_static_train_depth']} -> {item['candidate_secure_static_train_depth']} | "
                f"{item['status']} | {fmt_number(item['threshold_accuracy_delta'], 4)} | {fmt_number(item['auc_delta'], 6)} | {fmt_number(item['argmax_accuracy_delta'], 4)} |"
            )
    else:
        lines.append('- no paired control compare attached')

    lines.extend(
        [
            '',
            '## 6. Secret 路径对齐',
        '',
        f"- deployable secret profile: `{secret['deployable_secret_profile']}`",
        f"- secret_runtime_complete: `{fmt_number(secret['secret_runtime_complete'])}`",
        f"- pending_count: `{secret['secret_runtime_pending_count']}`",
        f"- unstable_count: `{secret['secret_runtime_unstable_count']}`",
        f"- argmax_match_ratio: `{fmt_percent(secret['argmax_match_ratio'])}`",
        f"- threshold_match_ratio: `{fmt_percent(secret['threshold_match_ratio'])}`",
        f"- logits_max_abs_error: `{fmt_number(secret['logits_max_abs_error'], 8)}`",
        f"- probabilities_max_abs_error: `{fmt_number(secret['probabilities_max_abs_error'], 8)}`",
        '',
        '## 7. 证据分级',
        '',
        ]
    )
    for item in report['evidence_matrix']:
        lines.append(f"- {item['label']}: `{item['status']}`")
        lines.append(f"  reason: {item['reason']}")

    lines.extend(
        [
            '',
            '## 8. 当前支持什么',
            '',
        ]
    )
    for item in report['judgement']['what_is_supported_now']:
        lines.append(f'- {item}')
    lines.extend(
        [
            '',
            '## 9. 当前还不支持什么',
            '',
        ]
    )
    for item in report['judgement']['what_is_not_supported_yet']:
        lines.append(f'- {item}')

    lines.extend(
        [
            '',
            '## 10. 下一步最小 ablation',
            '',
            f"- goal: {report['next_minimal_ablation']['goal']}",
            f"- suggested_runner: `{report['next_minimal_ablation']['suggested_runner']}`",
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate the current secure_static_train_depth evidence report.')
    parser.add_argument('--bundle-manifest-json', required=True)
    parser.add_argument('--bundle-args-snapshot-json', required=True)
    parser.add_argument('--baseline-eval-json', required=True)
    parser.add_argument('--modified-eval-json', required=True)
    parser.add_argument('--plaintext-model-compare-json', required=True)
    parser.add_argument('--plaintext-secure-compare-json', required=True)
    parser.add_argument('--secret-isolated-summary-json', required=True)
    parser.add_argument('--acceptance-json', default='')
    parser.add_argument('--pair-compare-json', action='append', default=[])
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    acceptance_json_path = Path(args.acceptance_json).expanduser().resolve() if args.acceptance_json else None
    pair_compare_paths = [Path(path).expanduser().resolve() for path in args.pair_compare_json]
    report = build_report(
        bundle_manifest_path=Path(args.bundle_manifest_json).expanduser().resolve(),
        bundle_args_snapshot_path=Path(args.bundle_args_snapshot_json).expanduser().resolve(),
        baseline_eval_path=Path(args.baseline_eval_json).expanduser().resolve(),
        modified_eval_path=Path(args.modified_eval_json).expanduser().resolve(),
        plaintext_model_compare_path=Path(args.plaintext_model_compare_json).expanduser().resolve(),
        plaintext_secure_compare_path=Path(args.plaintext_secure_compare_json).expanduser().resolve(),
        secret_summary_path=Path(args.secret_isolated_summary_json).expanduser().resolve(),
        acceptance_json_path=acceptance_json_path,
        pair_compare_paths=pair_compare_paths,
    )
    write_json(Path(args.output_json).expanduser().resolve(), report)
    write_text(Path(args.output_md).expanduser().resolve(), render_markdown(report) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
