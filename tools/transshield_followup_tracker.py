#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json_lines(path: Path):
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def summarize_records(records, reference_acc=None, reference_loss=None):
    if not records:
        raise ValueError('No records found in log file')

    best_acc_record = max(records, key=lambda item: item.get('test_acc1', float('-inf')))
    best_loss_record = min(records, key=lambda item: item.get('test_loss', float('inf')))
    final_record = records[-1]

    summary = {
        'epoch_count': len(records),
        'final_epoch': final_record.get('epoch'),
        'final': final_record,
        'best_by_acc1': best_acc_record,
        'best_by_loss': best_loss_record,
        'all_finite_epochs_logged': True,
    }

    if reference_acc is not None:
        summary['delta_vs_reference_acc1'] = best_acc_record.get('test_acc1') - reference_acc
        summary['final_delta_vs_reference_acc1'] = final_record.get('test_acc1') - reference_acc
    if reference_loss is not None:
        summary['delta_vs_reference_loss'] = best_loss_record.get('test_loss') - reference_loss
        summary['final_delta_vs_reference_loss'] = final_record.get('test_loss') - reference_loss

    return summary


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def fmt(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.{digits}f}'


def render_markdown(summary, run_name, source_log):
    final_record = summary['final']
    best_acc = summary['best_by_acc1']
    best_loss = summary['best_by_loss']

    lines = [
        f'# Follow-Up Summary: {run_name}',
        '',
        f'- source log: `{source_log}`',
        f'- logged epoch count: `{summary["epoch_count"]}`',
        f'- final epoch index: `{summary["final_epoch"]}`',
        '',
        '## Best Checkpoints by Logged Eval',
        '',
        '| Selection | Epoch | Test Acc@1 | Test Loss | Train Loss | Train Class Acc | Grad Norm |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        f'| best-by-acc1 | {best_acc.get("epoch")} | {fmt(best_acc.get("test_acc1"))} | {fmt(best_acc.get("test_loss"))} | {fmt(best_acc.get("train_loss"))} | {fmt(best_acc.get("train_class_acc"))} | {fmt(best_acc.get("train_grad_norm"))} |',
        f'| best-by-loss | {best_loss.get("epoch")} | {fmt(best_loss.get("test_acc1"))} | {fmt(best_loss.get("test_loss"))} | {fmt(best_loss.get("train_loss"))} | {fmt(best_loss.get("train_class_acc"))} | {fmt(best_loss.get("train_grad_norm"))} |',
        f'| final | {final_record.get("epoch")} | {fmt(final_record.get("test_acc1"))} | {fmt(final_record.get("test_loss"))} | {fmt(final_record.get("train_loss"))} | {fmt(final_record.get("train_class_acc"))} | {fmt(final_record.get("train_grad_norm"))} |',
    ]

    if 'delta_vs_reference_acc1' in summary:
        lines.extend([
            '',
            '## Comparison vs Current Official Baseline',
            '',
            f'- best acc@1 delta vs official baseline: `{fmt(summary.get("delta_vs_reference_acc1"))}`',
            f'- final acc@1 delta vs official baseline: `{fmt(summary.get("final_delta_vs_reference_acc1"))}`',
        ])
    if 'delta_vs_reference_loss' in summary:
        lines.append(f'- best loss delta vs official baseline: `{fmt(summary.get("delta_vs_reference_loss"))}`')
        lines.append(f'- final loss delta vs official baseline: `{fmt(summary.get("final_delta_vs_reference_loss"))}`')

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- 本摘要只反映实验仓 follow-up 的已记录结果，不自动替代最终作品正式基线。',
        '- 若 `acc@1` 未超过当前正式基线，则该分支应视为负结果或待继续分析分支。',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Summarize a follow-up training log into JSON/Markdown.')
    parser.add_argument('--log-path', required=True)
    parser.add_argument('--run-name', default='')
    parser.add_argument('--reference-acc1', type=float, default=None)
    parser.add_argument('--reference-loss', type=float, default=None)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve()
    run_name = args.run_name or log_path.parent.name
    records = load_json_lines(log_path)
    summary = summarize_records(records, reference_acc=args.reference_acc1, reference_loss=args.reference_loss)
    summary['run_name'] = run_name
    summary['source_log'] = str(log_path)

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    write_text(output_json, json.dumps(summary, indent=2, sort_keys=True) + '\n')
    write_text(output_md, render_markdown(summary, run_name=run_name, source_log=log_path))
    print(json.dumps({
        'run_name': run_name,
        'best_epoch': summary['best_by_acc1'].get('epoch'),
        'best_acc1': summary['best_by_acc1'].get('test_acc1'),
        'final_epoch': summary['final_epoch'],
        'final_acc1': summary['final'].get('test_acc1'),
        'output_json': str(output_json),
        'output_md': str(output_md),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
