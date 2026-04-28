#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional


CHECKLIST_LABELS = {
    'verified_bundle_exists': 'verified bundle exists',
    'official_bundle_exists': 'official bundle exists',
    'promotion_manifest_status_ok': 'promotion manifest status is ready',
    'candidate_best_epoch_recorded': 'candidate best epoch is recorded',
    'candidate_improves_official_argmax': 'candidate improves official argmax',
    'candidate_improves_official_threshold': 'candidate improves official threshold',
    'candidate_improves_official_auc': 'candidate improves official AUC',
    'fullval_spu_pipeline_passed': 'full-val SPU pipeline passed',
    'fullval_spu_replay_passed': 'full-val SPU replay passed',
    'prediction_semantics_match': 'prediction semantics match exactly',
    'communication_evidence_hardened': 'communication evidence is hardened',
    'already_promoted_as_default_bundle': 'already promoted as default bundle',
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def fmt(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def find_method(rows, method_name):
    for row in rows:
        if row.get('method') == method_name:
            return row
    raise KeyError(f'Method not found: {method_name}')


def load_optional_json(path: Optional[Path]):
    if path is None:
        return None
    return load_json(path)


def normalize_fastpath_metrics(communication_profile, communication_summary):
    metrics = communication_profile.get('aggregate_python_fastpath_metrics') or {}
    if metrics:
        return metrics
    if not isinstance(communication_summary, dict):
        return {}
    if communication_summary.get('python_fastpath_rpc_total_bytes') is None:
        return {}
    return {
        'rpc_request_total_bytes': communication_summary.get('python_fastpath_rpc_request_bytes'),
        'rpc_response_total_bytes': communication_summary.get('python_fastpath_rpc_response_bytes'),
        'rpc_total_bytes': communication_summary.get('python_fastpath_rpc_total_bytes'),
        'make_shares_total_input_bytes': communication_summary.get('make_shares_input_bytes'),
        'source': communication_summary.get('communication_source', 'Python distributed RPC/cloudpickle fastpath'),
    }


def load_scorecard_inputs(
    matrix_json,
    plaintext_json,
    spu_json,
    promotion_manifest_json,
    communication_json=None,
):
    matrix = load_json(matrix_json)
    plaintext = load_json(plaintext_json)
    spu = load_json(spu_json)
    promotion_manifest = load_json(promotion_manifest_json)
    communication_summary = load_optional_json(communication_json)
    communication_profile = (
        communication_summary.get('communication_profile')
        if isinstance(communication_summary, dict)
        else None
    ) or (communication_summary if isinstance(communication_summary, dict) else {})
    return {
        'matrix': matrix,
        'plaintext': plaintext,
        'spu': spu,
        'promotion_manifest': promotion_manifest,
        'communication_summary': communication_summary,
        'communication_profile': communication_profile,
    }


def extract_accuracy_rows(matrix):
    accuracy_rows = matrix.get('model_accuracy_axis', [])
    return {
        'official': find_method(accuracy_rows, 'Transshield official plaintext baseline'),
        'verified': find_method(accuracy_rows, 'Transshield lr3e-5 verified candidate plaintext'),
        'mpcvit': find_method(accuracy_rows, 'MPCViT same-dataset'),
        'mpcvit_3seed': find_method(accuracy_rows, 'MPCViT 3-seed mean'),
        'gap_section': matrix.get('current_gap_to_mpcvit_same_dataset', {}),
    }


def build_delta_vs_official(official, verified):
    return {
        'argmax_accuracy': verified.get('argmax_accuracy') - official.get('argmax_accuracy'),
        'threshold_accuracy': verified.get('threshold_accuracy') - official.get('threshold_accuracy'),
        'auc': verified.get('auc') - official.get('auc'),
    }


def build_secure_evidence(spu, communication_summary, communication_profile):
    communication_status = communication_profile.get(
        'status',
        communication_profile.get('communication_status', spu.get('communication_status')),
    )
    secure_evidence = {
        'sample_count': spu.get('sample_count'),
        'spu_pipeline_overall_passed': spu.get('spu_pipeline_overall_passed'),
        'spu_replay_overall_passed': spu.get('spu_replay_overall_passed'),
        'secure_model_replay_status': spu.get('secure_model_replay_status'),
        'argmax_match_ratio': spu.get('argmax_match_ratio'),
        'threshold_match_ratio': spu.get('threshold_match_ratio'),
        'logits_max_abs_error': spu.get('logits_max_abs_error'),
        'probabilities_max_abs_error': spu.get('probabilities_max_abs_error'),
        'total_pipeline_duration_sec': spu.get('total_pipeline_duration_sec'),
        'replay_duration_sec': spu.get('replay_duration_sec'),
        'communication_status': communication_status,
    }
    if communication_summary is None:
        return secure_evidence, communication_status

    python_fastpath_metrics = normalize_fastpath_metrics(communication_profile, communication_summary)
    secure_evidence.update(
        {
            'communication_link_detail_count': communication_profile.get(
                'link_detail_count',
                communication_summary.get('link_detail_count'),
            ),
            'communication_nonzero_link_detail_count': communication_profile.get(
                'nonzero_link_detail_count',
                communication_summary.get('nonzero_link_detail_count'),
            ),
            'communication_aggregate_link_metrics': communication_profile.get(
                'aggregate_link_metrics',
                communication_summary.get('aggregate_link_metrics'),
            ),
            'communication_python_fastpath_metrics': python_fastpath_metrics,
            'communication_python_fastpath_rpc_total_bytes': python_fastpath_metrics.get('rpc_total_bytes'),
            'communication_probe_total_pipeline_duration_sec': (
                communication_summary.get('step_profile', {}).get('total_pipeline_duration_sec')
                if communication_summary.get('step_profile') is not None
                else communication_summary.get('total_pipeline_duration_sec')
            ),
            'communication_probe_replay_duration_sec': (
                communication_summary.get('step_profile', {}).get('replay_duration_sec')
                if communication_summary.get('step_profile') is not None
                else communication_summary.get('replay_duration_sec')
            ),
            'communication_runtime_config_note': communication_profile.get(
                'note',
                communication_summary.get(
                    'runtime_config_note',
                    'Primary communication display uses Python distributed RPC/cloudpickle fastpath metrics when available.',
                ),
            ),
        }
    )
    return secure_evidence, communication_status


def build_promotion_checklist(
    plaintext,
    promotion_manifest,
    official_bundle_dir,
    verified_bundle_dir,
    delta_vs_official,
    secure_evidence,
    communication_status,
):
    return {
        'verified_bundle_exists': verified_bundle_dir.exists(),
        'official_bundle_exists': official_bundle_dir.exists(),
        'promotion_manifest_status_ok': promotion_manifest.get('status') == 'promotion_ready_verified_bundle',
        'candidate_best_epoch_recorded': (plaintext.get('best_by_acc1') or {}).get('epoch') is not None,
        'candidate_improves_official_argmax': delta_vs_official['argmax_accuracy'] > 0,
        'candidate_improves_official_threshold': delta_vs_official['threshold_accuracy'] > 0,
        'candidate_improves_official_auc': delta_vs_official['auc'] > 0,
        'fullval_spu_pipeline_passed': bool(secure_evidence.get('spu_pipeline_overall_passed')),
        'fullval_spu_replay_passed': bool(secure_evidence.get('spu_replay_overall_passed')),
        'prediction_semantics_match': (
            secure_evidence.get('argmax_match_ratio') == 1.0
            and secure_evidence.get('threshold_match_ratio') == 1.0
        ),
        'communication_evidence_hardened': communication_status not in (
            None,
            '',
            'unreliable_zero_counters',
            'colocated_private_path_no_link_counters',
        ),
        'already_promoted_as_default_bundle': False,
    }


def build_remaining_blockers(checklist, communication_status):
    blockers = []
    if not checklist['communication_evidence_hardened']:
        blockers.append('communication counters are still unreliable for the verified candidate')
    elif communication_status == 'available_via_aux_probe':
        blockers.append(
            'communication counters are available only under the diagnostic runtime branch with colocated optimization disabled'
        )
    if not checklist['already_promoted_as_default_bundle']:
        blockers.append('verified candidate has not yet been formally promoted as the default display bundle')
    return blockers


def build_next_actions():
    return [
        'Decide whether to promote artifacts/frozen_bundle_verified_tracka_lr3e5_20260414 as the default presentation bundle',
        'Keep Python fastpath RPC/cloudpickle metrics as the default fast-runtime communication display',
        'Run 1-2 additional lr=3e-5 confirmation seeds or a minimal stability ablation to show the candidate is not an isolated peak',
        'Convert the current matrices into presentation visuals: architecture, token pruning, CPU/SPU runtime, and external comparison',
    ]


def build_competition_outlook(checklist, argmax_gap):
    if (
        checklist['fullval_spu_pipeline_passed']
        and checklist['prediction_semantics_match']
        and argmax_gap is not None
        and argmax_gap <= 3.0
    ):
        if checklist['communication_evidence_hardened']:
            return {
                'status': 'competitive_for_award_with_hardened_comm_but_not_promoted',
                'reason': (
                    'The verified candidate is close enough in plaintext accuracy to remain credible, '
                    'the secure sidecar path is fully revalidated, and communication evidence is now available; '
                    'the main remaining decision is formal promotion.'
                ),
            }
        return {
            'status': 'competitive_for_award_but_not_fully_hardened',
            'reason': (
                'The verified candidate is close enough in plaintext accuracy to remain credible, '
                'and the secure sidecar path is fully revalidated, but promotion and communication evidence are not fully hardened.'
            ),
        }
    return {
        'status': 'promising_but_still_midproof',
        'reason': (
            'The project has real secure-system evidence, but either the model gap or the evidence gap '
            'is still too large for a stronger claim.'
        ),
    }


def build_artifact_paths(
    official_bundle_dir,
    verified_bundle_dir,
    matrix_json,
    plaintext_json,
    spu_json,
    promotion_manifest_json,
    communication_json,
):
    return {
        'official_bundle_dir': str(official_bundle_dir),
        'verified_bundle_dir': str(verified_bundle_dir),
        'promotion_manifest_json': str(promotion_manifest_json),
        'plaintext_followup_json': str(plaintext_json),
        'spu_summary_json': str(spu_json),
        'communication_summary_json': str(communication_json) if communication_json else '',
        'comparison_matrix_json': str(matrix_json),
    }


def build_scorecard(
    matrix_json,
    plaintext_json,
    spu_json,
    promotion_manifest_json,
    official_bundle_dir: Path,
    verified_bundle_dir: Path,
    communication_json=None,
):
    inputs = load_scorecard_inputs(
        matrix_json,
        plaintext_json,
        spu_json,
        promotion_manifest_json,
        communication_json,
    )
    rows = extract_accuracy_rows(inputs['matrix'])
    delta_vs_official = build_delta_vs_official(rows['official'], rows['verified'])
    secure_evidence, communication_status = build_secure_evidence(
        inputs['spu'],
        inputs['communication_summary'],
        inputs['communication_profile'],
    )
    checklist = build_promotion_checklist(
        inputs['plaintext'],
        inputs['promotion_manifest'],
        official_bundle_dir,
        verified_bundle_dir,
        delta_vs_official,
        secure_evidence,
        communication_status,
    )
    gap_best = rows['gap_section'].get('transshield_verified_candidate_vs_mpcvit_best_by_argmax', {})
    return {
        'official_baseline': rows['official'],
        'verified_candidate': {
            'plaintext': rows['verified'],
            'best_epoch': (inputs['plaintext'].get('best_by_acc1') or {}).get('epoch'),
            'final_epoch': inputs['plaintext'].get('final_epoch'),
            'secure_summary': secure_evidence,
        },
        'external_references': {
            'mpcvit_same_dataset': rows['mpcvit'],
            'mpcvit_3seed_mean': rows['mpcvit_3seed'],
            'gap_to_mpcvit_same_dataset': rows['gap_section'],
        },
        'delta_vs_official': delta_vs_official,
        'promotion_checklist': checklist,
        'remaining_blockers': build_remaining_blockers(checklist, communication_status),
        'next_actions': build_next_actions(),
        'competition_outlook': build_competition_outlook(
            checklist,
            gap_best.get('argmax_accuracy_gap'),
        ),
        'artifact_paths': build_artifact_paths(
            official_bundle_dir,
            verified_bundle_dir,
            matrix_json,
            plaintext_json,
            spu_json,
            promotion_manifest_json,
            communication_json,
        ),
    }


def render_core_metrics(scorecard):
    official = scorecard['official_baseline']
    verified = scorecard['verified_candidate']['plaintext']
    delta = scorecard['delta_vs_official']
    return [
        '## Core Metrics',
        '',
        '| Item | Official Baseline | Verified Candidate | Delta |',
        '| --- | ---: | ---: | ---: |',
        f'| argmax accuracy | `{fmt(official.get("argmax_accuracy"))}` | `{fmt(verified.get("argmax_accuracy"))}` | `+{fmt(delta["argmax_accuracy"])}` |',
        f'| threshold accuracy | `{fmt(official.get("threshold_accuracy"))}` | `{fmt(verified.get("threshold_accuracy"))}` | `+{fmt(delta["threshold_accuracy"])}` |',
        f'| AUC | `{fmt(official.get("auc"))}` | `{fmt(verified.get("auc"))}` | `+{fmt(delta["auc"])}` |',
    ]


def render_secure_evidence(scorecard):
    secure = scorecard['verified_candidate']['secure_summary']
    lines = [
        '## Secure Evidence',
        '',
        f'- full-val SPU pipeline passed: `{fmt(secure.get("spu_pipeline_overall_passed"))}`',
        f'- full-val SPU replay passed: `{fmt(secure.get("spu_replay_overall_passed"))}`',
        f'- secure replay status: `{secure.get("secure_model_replay_status")}`',
        f'- argmax match ratio: `{fmt(secure.get("argmax_match_ratio"))}`',
        f'- threshold match ratio: `{fmt(secure.get("threshold_match_ratio"))}`',
        f'- full-val SPU total pipeline duration: `{fmt(secure.get("total_pipeline_duration_sec"))}s`',
        f'- full-val SPU replay duration: `{fmt(secure.get("replay_duration_sec"))}s`',
        f'- communication status: `{secure.get("communication_status")}`',
    ]
    if secure.get('communication_aggregate_link_metrics') is not None:
        aggregate = secure['communication_aggregate_link_metrics']
        lines.extend(
            [
                f'- max total bytes: `{fmt(aggregate.get("max_total_bytes"), 0)}`',
                f'- max send bytes: `{fmt(aggregate.get("max_send_bytes"), 0)}`',
                f'- max recv bytes: `{fmt(aggregate.get("max_recv_bytes"), 0)}`',
            ]
        )
    if secure.get('communication_python_fastpath_metrics'):
        fastpath = secure['communication_python_fastpath_metrics']
        lines.extend(
            [
                f'- Python fastpath RPC total bytes: `{fmt(fastpath.get("rpc_total_bytes"), 0)}`',
                f'- Python fastpath RPC request bytes: `{fmt(fastpath.get("rpc_request_total_bytes"), 0)}`',
                f'- Python fastpath RPC response bytes: `{fmt(fastpath.get("rpc_response_total_bytes"), 0)}`',
                f'- Python fastpath make_shares input bytes: `{fmt(fastpath.get("make_shares_total_input_bytes"), 0)}`',
            ]
        )
    if secure.get('communication_probe_total_pipeline_duration_sec') is not None:
        lines.append(
            f'- communication probe total pipeline duration: `{fmt(secure.get("communication_probe_total_pipeline_duration_sec"))}s`'
        )
    if secure.get('communication_runtime_config_note'):
        lines.append(f'- communication note: {secure.get("communication_runtime_config_note")}')
    return lines


def render_external_gap(scorecard):
    gap_best = scorecard['external_references']['gap_to_mpcvit_same_dataset'][
        'transshield_verified_candidate_vs_mpcvit_best_by_argmax'
    ]
    return [
        '## External Gap',
        '',
        (
            f'- gap vs `MPCViT` best-by-argmax: argmax `{fmt(gap_best.get("argmax_accuracy_gap"))}`, '
            f'threshold `{fmt(gap_best.get("threshold_accuracy_gap"))}`, '
            f'AUC `{fmt(gap_best.get("auc_gap"))}`'
        ),
    ]


def render_checklist(scorecard):
    checklist = scorecard['promotion_checklist']
    lines = ['## Promotion Checklist', '']
    for key, label in CHECKLIST_LABELS.items():
        lines.append(f'- {label}: `{fmt(checklist.get(key))}`')
    return lines


def render_simple_list(title, items):
    lines = [title, '']
    for item in items:
        lines.append(f'- {item}')
    return lines


def render_markdown(scorecard):
    lines = [
        '# Competition Scorecard — `tracka_lr3e5_timm` Verified Candidate',
        '',
        f'> outlook: `{scorecard["competition_outlook"]["status"]}`  ',
        f'> reason: {scorecard["competition_outlook"]["reason"]}',
        '',
    ]
    for section in (
        render_core_metrics(scorecard),
        render_secure_evidence(scorecard),
        render_external_gap(scorecard),
        render_checklist(scorecard),
        render_simple_list('## Remaining Blockers', scorecard['remaining_blockers']),
        render_simple_list('## Recommended Next Actions', scorecard['next_actions']),
    ):
        lines.extend(section)
        lines.append('')
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate a promotion and competition scorecard for the current verified candidate.'
    )
    parser.add_argument('--comparison-matrix-json', required=True)
    parser.add_argument('--plaintext-followup-json', required=True)
    parser.add_argument('--spu-summary-json', required=True)
    parser.add_argument('--promotion-manifest-json', required=True)
    parser.add_argument('--official-bundle-dir', required=True)
    parser.add_argument('--verified-bundle-dir', required=True)
    parser.add_argument('--communication-summary-json', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    scorecard = build_scorecard(
        matrix_json=Path(args.comparison_matrix_json).resolve(),
        plaintext_json=Path(args.plaintext_followup_json).resolve(),
        spu_json=Path(args.spu_summary_json).resolve(),
        promotion_manifest_json=Path(args.promotion_manifest_json).resolve(),
        official_bundle_dir=Path(args.official_bundle_dir).resolve(),
        verified_bundle_dir=Path(args.verified_bundle_dir).resolve(),
        communication_json=Path(args.communication_summary_json).resolve() if args.communication_summary_json else None,
    )

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    write_text(output_json, json.dumps(scorecard, indent=2, sort_keys=True) + '\n')
    write_text(output_md, render_markdown(scorecard) + '\n')
    print(
        json.dumps(
            {
                'competition_outlook': scorecard['competition_outlook']['status'],
                'argmax_delta_vs_official': scorecard['delta_vs_official']['argmax_accuracy'],
                'gap_to_mpcvit_argmax': scorecard['external_references']['gap_to_mpcvit_same_dataset'][
                    'transshield_verified_candidate_vs_mpcvit_best_by_argmax'
                ]['argmax_accuracy_gap'],
                'output_json': str(output_json),
                'output_md': str(output_md),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
