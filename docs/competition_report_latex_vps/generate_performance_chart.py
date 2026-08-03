#!/usr/bin/env python3
"""Generate the report performance chart from structured metric data.

The chart is drawn from a blank canvas.  It never reads, retouches, masks, or
reuses pixels from the submitted report image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2726
HEIGHT = 1452
BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#FCFDFE"
PANEL_BORDER = "#E8EBEF"
AXIS_COLOR = "#666B73"
TEXT_COLOR = "#20242A"
GRID_COLOR = "#D9DEE5"


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = TEXT_COLOR,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def _multiline_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    line_gap: int = 6,
) -> None:
    for line_index, line in enumerate(text.splitlines()):
        _centered_text(draw, center_x, y + line_index * (font.size + line_gap), line, font)


def _rotated_axis_label(
    canvas: Image.Image,
    center_x: int,
    center_y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    box = font.getbbox(text)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    label = Image.new("RGBA", (text_width + 24, text_height + 24), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((12, 12 - box[1]), text, font=font, fill=TEXT_COLOR)
    rotated = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(
        rotated,
        (int(center_x - rotated.width / 2), int(center_y - rotated.height / 2)),
    )


def _format_tick(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _load_chart_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    series = data.get("series")
    metrics = data.get("metrics")
    if not isinstance(series, list) or len(series) != 2:
        raise ValueError("performance chart requires exactly two series")
    if not isinstance(metrics, list) or len(metrics) != 5:
        raise ValueError("performance chart requires exactly five metrics")
    for metric in metrics:
        if len(metric.get("values", [])) != len(series):
            raise ValueError(f"metric {metric.get('key')} has an invalid value count")
        if len(metric.get("display", [])) != len(series):
            raise ValueError(f"metric {metric.get('key')} has an invalid display count")
    return data


def generate_performance_chart(
    data_path: Path,
    output_path: Path,
    *,
    chinese_font_path: Path,
    latin_font_path: Path,
) -> Path:
    data = _load_chart_data(data_path)
    series = data["series"]
    metrics = data["metrics"]

    canvas = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(chinese_font_path, 30)
    axis_font = _font(chinese_font_path, 25)
    label_font = _font(chinese_font_path, 26)
    tick_font = _font(latin_font_path, 21)
    value_font = _font(latin_font_path, 25)

    outer_left = 46
    outer_right = 24
    panel_gap = 34
    panel_width = (WIDTH - outer_left - outer_right - panel_gap * 4) / 5
    panel_top = 18
    panel_bottom = HEIGHT - 20
    plot_top = 142
    plot_bottom = 1162

    for metric_index, metric in enumerate(metrics):
        panel_left = outer_left + metric_index * (panel_width + panel_gap)
        panel_right = panel_left + panel_width
        draw.rounded_rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            radius=18,
            fill=PANEL_BACKGROUND,
            outline=PANEL_BORDER,
            width=2,
        )

        plot_left = panel_left + 106
        plot_right = panel_right - 24
        plot_height = plot_bottom - plot_top
        axis_max = float(metric["axis_max"])
        tick_step = float(metric["tick_step"])

        _centered_text(
            draw,
            (panel_left + panel_right) / 2,
            54,
            metric["title"],
            title_font,
        )

        tick_value = 0.0
        while tick_value <= axis_max + tick_step / 100:
            y = plot_bottom - plot_height * tick_value / axis_max
            if tick_value > 0:
                dash_x = plot_left
                while dash_x < plot_right:
                    draw.line(
                        (dash_x, y, min(dash_x + 12, plot_right), y),
                        fill=GRID_COLOR,
                        width=2,
                    )
                    dash_x += 22
            tick_text = _format_tick(tick_value)
            tick_box = draw.textbbox((0, 0), tick_text, font=tick_font)
            draw.text(
                (plot_left - 16 - (tick_box[2] - tick_box[0]), y - (tick_box[3] - tick_box[1]) / 2),
                tick_text,
                font=tick_font,
                fill=AXIS_COLOR,
            )
            tick_value += tick_step

        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS_COLOR, width=3)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS_COLOR, width=3)

        axis_center_x = int(panel_left + 28)
        _rotated_axis_label(
            canvas,
            axis_center_x,
            int((plot_top + plot_bottom) / 2),
            metric["axis_label"],
            axis_font,
        )

        bar_centers = (
            plot_left + (plot_right - plot_left) * 0.30,
            plot_left + (plot_right - plot_left) * 0.76,
        )
        bar_width = min(92, (plot_right - plot_left) * 0.28)

        for series_index, item in enumerate(series):
            value = float(metric["values"][series_index])
            bar_top = plot_bottom - plot_height * value / axis_max
            center_x = bar_centers[series_index]
            draw.rounded_rectangle(
                (
                    center_x - bar_width / 2,
                    bar_top,
                    center_x + bar_width / 2,
                    plot_bottom,
                ),
                radius=5,
                fill=item["color"],
            )
            _centered_text(
                draw,
                center_x,
                max(plot_top + 8, bar_top - 40),
                metric["display"][series_index],
                value_font,
            )
            _multiline_centered_text(
                draw,
                center_x,
                plot_bottom + 28,
                item["label"],
                label_font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG", optimize=True)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    generated = generate_performance_chart(
        root / "performance_chart_data.json",
        root / "output" / "intermediate" / "strict_format" / "performance_chart_generated.png",
        chinese_font_path=Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        latin_font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    print(generated)
