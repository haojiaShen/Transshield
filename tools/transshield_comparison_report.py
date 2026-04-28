import argparse
import json
from pathlib import Path


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def format_float(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def collect_summary(run_dir: Path):
    baseline_eval = load_json_if_exists(run_dir / 'plaintext_baseline_eval.json')
    modified_eval = load_json_if_exists(run_dir / 'plaintext_modified_eval.json')
    plaintext_compare = load_json_if_exists(run_dir / 'plaintext_model_compare.json')
    secure_compare = load_json_if_exists(run_dir / 'plaintext_vs_secure_score_compare.json')
    secure_profile_compare = load_json_if_exists(run_dir / 'secure_profile_compare.json')

    summary = {
        'run_dir': str(run_dir.resolve()),
        'available': {
            'baseline_eval': baseline_eval is not None,
            'modified_eval': modified_eval is not None,
            'plaintext_compare': plaintext_compare is not None,
            'secure_compare': secure_compare is not None,
            'secure_profile_compare': secure_profile_compare is not None,
        },
        'baseline_eval': None,
        'modified_eval': None,
        'plaintext_compare': None,
        'secure_compare': None,
        'secure_profile_compare': None,
    }

    if baseline_eval is not None:
        metrics = baseline_eval.get('metrics', {})
        summary['baseline_eval'] = {
            'sample_count': baseline_eval.get('sample_count'),
            'argmax_accuracy': metrics.get('argmax_accuracy'),
            'argmax_f1': metrics.get('argmax_f1'),
            'auc': metrics.get('auc'),
            'threshold': metrics.get('threshold'),
            'threshold_accuracy': metrics.get('threshold_accuracy'),
            'threshold_f1': metrics.get('threshold_f1'),
        }

    if modified_eval is not None:
        metrics = modified_eval.get('metrics', {})
        summary['modified_eval'] = {
            'sample_count': modified_eval.get('sample_count'),
            'argmax_accuracy': metrics.get('argmax_accuracy'),
            'argmax_f1': metrics.get('argmax_f1'),
            'auc': metrics.get('auc'),
            'threshold': metrics.get('threshold'),
            'threshold_accuracy': metrics.get('threshold_accuracy'),
            'threshold_f1': metrics.get('threshold_f1'),
        }

    if plaintext_compare is not None:
        summary['plaintext_compare'] = {
            'sample_count_match': plaintext_compare.get('sample_count_match'),
            'sample_paths_match': plaintext_compare.get('sample_paths_match'),
            'delta_auc_b_minus_a': plaintext_compare.get('delta_b_minus_a', {}).get('auc'),
            'delta_argmax_accuracy_b_minus_a': plaintext_compare.get('delta_b_minus_a', {}).get('argmax_accuracy'),
            'delta_argmax_f1_b_minus_a': plaintext_compare.get('delta_b_minus_a', {}).get('argmax_f1'),
            'delta_threshold_accuracy_b_minus_a': plaintext_compare.get('delta_b_minus_a', {}).get('threshold_accuracy'),
            'delta_threshold_f1_b_minus_a': plaintext_compare.get('delta_b_minus_a', {}).get('threshold_f1'),
        }

    if secure_compare is not None:
        comparison = secure_compare.get('comparison', {})
        summary['secure_compare'] = {
            'sample_count': secure_compare.get('sample_count'),
            'secure_overall_passed': secure_compare.get('source_status', {}).get('secure_overall_passed'),
            'secure_model_replay_status': secure_compare.get('source_status', {}).get('secure_model_replay_status'),
            'argmax_match_ratio': comparison.get('argmax_predictions', {}).get('match_ratio'),
            'threshold_match_ratio': comparison.get('threshold_predictions', {}).get('match_ratio'),
            'logits_max_abs_error': comparison.get('logits', {}).get('max_abs_error'),
            'probabilities_max_abs_error': comparison.get('probabilities', {}).get('max_abs_error'),
            'plaintext_argmax_accuracy': comparison.get('plaintext_argmax_accuracy'),
            'secure_argmax_accuracy': comparison.get('secure_argmax_accuracy'),
            'plaintext_threshold_accuracy': comparison.get('plaintext_threshold_accuracy'),
            'secure_threshold_accuracy': comparison.get('secure_threshold_accuracy'),
        }

    if secure_profile_compare is not None:
        summary['secure_profile_compare'] = secure_profile_compare

    return summary


def render_text(summary):
    lines = []
    lines.append(f"Run Dir: {summary['run_dir']}")

    baseline_eval = summary.get('baseline_eval')
    if baseline_eval is not None:
        lines.append('')
        lines.append('[Baseline Plaintext]')
        lines.append(f"sample_count={format_float(baseline_eval.get('sample_count'), 0)}")
        lines.append(f"argmax_accuracy={format_float(baseline_eval.get('argmax_accuracy'))}")
        lines.append(f"argmax_f1={format_float(baseline_eval.get('argmax_f1'))}")
        lines.append(f"auc={format_float(baseline_eval.get('auc'))}")
        lines.append(f"threshold={format_float(baseline_eval.get('threshold'))}")
        lines.append(f"threshold_accuracy={format_float(baseline_eval.get('threshold_accuracy'))}")
        lines.append(f"threshold_f1={format_float(baseline_eval.get('threshold_f1'))}")

    modified_eval = summary.get('modified_eval')
    if modified_eval is not None:
        lines.append('')
        lines.append('[Modified Plaintext]')
        lines.append(f"sample_count={format_float(modified_eval.get('sample_count'), 0)}")
        lines.append(f"argmax_accuracy={format_float(modified_eval.get('argmax_accuracy'))}")
        lines.append(f"argmax_f1={format_float(modified_eval.get('argmax_f1'))}")
        lines.append(f"auc={format_float(modified_eval.get('auc'))}")
        lines.append(f"threshold={format_float(modified_eval.get('threshold'))}")
        lines.append(f"threshold_accuracy={format_float(modified_eval.get('threshold_accuracy'))}")
        lines.append(f"threshold_f1={format_float(modified_eval.get('threshold_f1'))}")

    plaintext_compare = summary.get('plaintext_compare')
    if plaintext_compare is not None:
        lines.append('')
        lines.append('[Baseline vs Modified]')
        lines.append(f"sample_count_match={format_float(plaintext_compare.get('sample_count_match'))}")
        lines.append(f"sample_paths_match={format_float(plaintext_compare.get('sample_paths_match'))}")
        lines.append(f"delta_auc_b_minus_a={format_float(plaintext_compare.get('delta_auc_b_minus_a'))}")
        lines.append(f"delta_argmax_accuracy_b_minus_a={format_float(plaintext_compare.get('delta_argmax_accuracy_b_minus_a'))}")
        lines.append(f"delta_argmax_f1_b_minus_a={format_float(plaintext_compare.get('delta_argmax_f1_b_minus_a'))}")
        lines.append(f"delta_threshold_accuracy_b_minus_a={format_float(plaintext_compare.get('delta_threshold_accuracy_b_minus_a'))}")
        lines.append(f"delta_threshold_f1_b_minus_a={format_float(plaintext_compare.get('delta_threshold_f1_b_minus_a'))}")

    secure_compare = summary.get('secure_compare')
    if secure_compare is not None:
        lines.append('')
        lines.append('[Modified Plaintext vs Secure]')
        lines.append(f"sample_count={format_float(secure_compare.get('sample_count'), 0)}")
        lines.append(f"secure_overall_passed={format_float(secure_compare.get('secure_overall_passed'))}")
        lines.append(f"secure_model_replay_status={secure_compare.get('secure_model_replay_status', 'N/A')}")
        lines.append(f"argmax_match_ratio={format_float(secure_compare.get('argmax_match_ratio'))}")
        lines.append(f"threshold_match_ratio={format_float(secure_compare.get('threshold_match_ratio'))}")
        lines.append(f"logits_max_abs_error={format_float(secure_compare.get('logits_max_abs_error'))}")
        lines.append(f"probabilities_max_abs_error={format_float(secure_compare.get('probabilities_max_abs_error'))}")
        lines.append(f"plaintext_argmax_accuracy={format_float(secure_compare.get('plaintext_argmax_accuracy'))}")
        lines.append(f"secure_argmax_accuracy={format_float(secure_compare.get('secure_argmax_accuracy'))}")
        lines.append(f"plaintext_threshold_accuracy={format_float(secure_compare.get('plaintext_threshold_accuracy'))}")
        lines.append(f"secure_threshold_accuracy={format_float(secure_compare.get('secure_threshold_accuracy'))}")

    if summary.get('secure_profile_compare') is not None:
        lines.append('')
        lines.append('[Secure Profile Compare]')
        lines.append('see comparison_report_summary.json for full profile diff')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Summarize Transshield comparison outputs into direct human-readable metrics.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-json', default='')
    parser.add_argument('--output-txt', default='')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary = collect_summary(run_dir)
    text = render_text(summary)

    output_json = Path(args.output_json).resolve() if args.output_json else run_dir / 'comparison_report_summary.json'
    output_txt = Path(args.output_txt).resolve() if args.output_txt else run_dir / 'comparison_report_summary.txt'

    write_text(output_json, json.dumps(summary, indent=2, sort_keys=True) + '\n')
    write_text(output_txt, text)
    print(text, end='')


if __name__ == '__main__':
    main()
