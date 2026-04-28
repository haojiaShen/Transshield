import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_pruning_trace import build_trace_report


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def load_font(size: int):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


def fit_display_image(image_path: Path, size: int):
    image = Image.open(image_path).convert('RGB')
    resample = getattr(Image, 'Resampling', Image).BICUBIC
    return ImageOps.fit(image, (size, size), method=resample)


def draw_mask_overlay(base_image: Image.Image, mask_grid, title: str, subtitle: str):
    panel_width = base_image.width
    panel_height = base_image.height + 56
    panel = Image.new('RGBA', (panel_width, panel_height), 'white')
    panel.paste(base_image.convert('RGBA'), (0, 0))

    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay, 'RGBA')
    draw_panel = ImageDraw.Draw(panel)
    grid_size = len(mask_grid)
    patch_size = base_image.width / grid_size

    keep_fill = ImageColor.getrgb('#4CAF50') + (36,)
    prune_fill = ImageColor.getrgb('#111827') + (138,)
    keep_outline = ImageColor.getrgb('#2E7D32') + (170,)
    prune_outline = ImageColor.getrgb('#CBD5E1') + (165,)
    grid_outline = ImageColor.getrgb('#FAFAFA') + (72,)

    for row_index, row in enumerate(mask_grid):
        for col_index, keep_flag in enumerate(row):
            x0 = int(round(col_index * patch_size))
            y0 = int(round(row_index * patch_size))
            x1 = int(round((col_index + 1) * patch_size))
            y1 = int(round((row_index + 1) * patch_size))
            fill = keep_fill if keep_flag else prune_fill
            outline = keep_outline if keep_flag else prune_outline
            draw_overlay.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=1)
            draw_overlay.rectangle([x0, y0, x1, y1], outline=grid_outline, width=1)

    panel.alpha_composite(overlay, (0, 0))
    title_font = load_font(18)
    body_font = load_font(11)
    draw_panel.text((12, base_image.height + 8), title, fill='black', font=title_font)
    draw_panel.text((12, base_image.height + 38), subtitle, fill='#333333', font=body_font)
    return panel.convert('RGB')


def draw_original_panel(base_image: Image.Image, title: str, subtitle: str):
    panel = Image.new('RGB', (base_image.width, base_image.height + 56), 'white')
    panel.paste(base_image, (0, 0))
    draw = ImageDraw.Draw(panel)
    title_font = load_font(18)
    body_font = load_font(11)
    draw.text((12, base_image.height + 8), title, fill='black', font=title_font)
    draw.text((12, base_image.height + 38), subtitle, fill='#333333', font=body_font)
    return panel


def make_contact_sheet(panels, columns=2, background='white', padding=16):
    if not panels:
        raise ValueError('no panels provided')
    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    rows = math.ceil(len(panels) / columns)
    canvas_width = columns * panel_width + padding * (columns + 1)
    canvas_height = rows * panel_height + padding * (rows + 1)
    canvas = Image.new('RGB', (canvas_width, canvas_height), background)
    for index, panel in enumerate(panels):
        row_index = index // columns
        col_index = index % columns
        x = padding + col_index * (panel_width + padding)
        y = padding + row_index * (panel_height + padding)
        canvas.paste(panel, (x, y))
    return canvas


def build_visualization_summary(trace_report, output_dir: Path):
    stage_rows = []
    for stage in trace_report['pruning_trace']['stages']:
        active_before = stage['active_before_per_sample'][0]
        active_after = stage['active_after_per_sample'][0]
        density = stage['active_after_density_per_sample'][0]
        stage_rows.append({
            'stage_index': stage['stage_index'],
            'pruning_layer': stage['pruning_layer'],
            'keep_count': int(active_after),
            'pruned_count': int(active_before - active_after),
            'density': float(density),
            'overlay_png': str((output_dir / f"stage_{stage['stage_index'] + 1}_overlay.png").resolve()),
            'mask_json_keys': {
                'keep_indices': 'first_sample_keep_indices',
                'pruned_indices': 'first_sample_pruned_indices',
                'mask_grid': 'first_sample_mask_grid',
            },
        })
    return {
        'image_path': trace_report['image_path'],
        'argmax_class': trace_report['argmax_class'],
        'threshold_class': trace_report.get('threshold_class'),
        'probabilities': trace_report['probabilities'],
        'summary_board_png': str((output_dir / 'token_pruning_summary.png').resolve()),
        'trace_json': str((output_dir / 'token_pruning_trace.json').resolve()),
        'stages': stage_rows,
    }


def render_markdown(trace_report, summary):
    probability_values = ', '.join(f'{value:.6f}' for value in trace_report['probabilities'])
    lines = [
        '# Token Pruning Visualization',
        '',
        f"- Image: `{trace_report['image_path']}`",
        f"- Summary board: `{summary['summary_board_png']}`",
        f"- Trace JSON: `{summary['trace_json']}`",
        f"- Probabilities: `{probability_values}`",
        f"- Argmax class: `{trace_report['argmax_class']}`",
        f"- Threshold class: `{trace_report.get('threshold_class')}`",
        '',
        '## Stage summary',
        '',
        '| Stage | Layer | Keep Count | Pruned Count | Keep Density |',
        '|---|---:|---:|---:|---:|',
    ]
    for stage in summary['stages']:
        lines.append(
            f"| stage_{stage['stage_index'] + 1} | {stage['pruning_layer']} | {stage['keep_count']} | "
            f"{stage['pruned_count']} | {stage['density']:.4f} |"
        )
    lines.extend(
        [
            '',
            '## How to explain',
            '',
            '- 浅绿色区域表示当前阶段仍被保留的 token。',
            '- 灰黑遮罩表示当前阶段已经被 masking 停用的 token。',
            '- 这不是病灶分割图，颜色只表示 token 是否继续参与后续计算。',
            '- 可以直接用这组图说明：当前项目不是直接删除 token，而是将 pruning 改写成 masking 表达，以便与 `F_mux` 语义对齐。',
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate token pruning visualization panels for a single input image.')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_report = build_trace_report(
        bundle_dir=args.bundle_dir,
        image_path=args.image_path,
        device=args.device,
    )
    write_json(output_dir / 'token_pruning_trace.json', trace_report)

    base_image = fit_display_image(Path(trace_report['image_path']), args.image_size)
    original_panel = draw_original_panel(
        base_image,
        title='Original Image',
        subtitle='Display-only resized input for patch visualization',
    )
    panels = [original_panel]

    for stage in trace_report['pruning_trace']['stages']:
        keep_count = int(stage['active_after_per_sample'][0])
        total_count = trace_report['pruning_trace']['init_spatial_tokens']
        density = stage['active_after_density_per_sample'][0]
        panel = draw_mask_overlay(
            base_image=base_image,
            mask_grid=stage['first_sample_mask_grid'],
            title=f"Stage {stage['stage_index'] + 1} | layer {stage['pruning_layer']}",
            subtitle=f"keep={keep_count}/{total_count} ({density:.2%})",
        )
        panel_path = output_dir / f"stage_{stage['stage_index'] + 1}_overlay.png"
        panel.save(panel_path)
        panels.append(panel)

    summary_board = make_contact_sheet(panels, columns=2)
    summary_board.save(output_dir / 'token_pruning_summary.png')

    summary = build_visualization_summary(trace_report, output_dir)
    write_json(output_dir / 'token_pruning_visualization_summary.json', summary)
    write_text(output_dir / 'token_pruning_trace_report.md', render_markdown(trace_report, summary) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
