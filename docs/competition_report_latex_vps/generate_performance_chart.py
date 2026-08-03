#!/usr/bin/env python3
"""Generate the report performance dashboard from structured metric data.

The complete figure is drawn from a blank canvas.  It never opens, masks,
retouches, or reuses pixels from the submitted report image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2726
HEIGHT = 1452
WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
GRID = "#DCE4ED"
PANEL = "#F8FAFC"
BORDER = "#D7E0EA"
NAVY = "#2D5B9B"
NAVY_DARK = "#183A6B"
AMBER = "#E99A2E"


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[float, float]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _center_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = INK,
) -> None:
    width, _ = _text_size(draw, text, font)
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


def _right_text(
    draw: ImageDraw.ImageDraw,
    right_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = INK,
) -> None:
    width, _ = _text_size(draw, text, font)
    draw.text((right_x - width, y), text, font=font, fill=fill)


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


def _single_line_label(label: str) -> str:
    return label.replace("\n", "")


def _metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    metric: dict[str, Any],
    series: list[dict[str, Any]],
    *,
    chinese_font_path: Path,
    latin_font_path: Path,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((left + 28, top + 28, left + 42, top + 98), radius=7, fill=NAVY)
    draw.text((left + 70, top + 18), metric["title"], font=_font(chinese_font_path, 52), fill=INK)
    draw.text((left + 72, top + 88), metric["axis_label"], font=_font(chinese_font_path, 34), fill=MUTED)

    plot_left = left + 390
    plot_right = right - 78
    bar_height = 64
    row_centers = (top + 218, top + 378)
    axis_max = float(metric["axis_max"])
    tick_count = 5
    tick_font = _font(latin_font_path, 30)

    for tick_index in range(tick_count + 1):
        value = axis_max * tick_index / tick_count
        x = plot_left + (plot_right - plot_left) * tick_index / tick_count
        draw.line((x, top + 145, x, bottom - 72), fill=GRID, width=2)
        label = f"{value:.1f}" if axis_max <= 5 and not value.is_integer() else f"{value:.0f}"
        _center_text(draw, x, bottom - 52, label, tick_font, fill=MUTED)

    for index, item in enumerate(series):
        y = row_centers[index]
        label = item.get("short_label", _single_line_label(item["label"]))
        draw.text((left + 70, y - 34), label, font=_font(chinese_font_path, 40), fill=INK)
        value = float(metric["values"][index])
        end_x = plot_left + min(value / axis_max, 1.0) * (plot_right - plot_left)
        draw.rounded_rectangle((plot_left, y - bar_height / 2, plot_right, y + bar_height / 2), radius=bar_height // 2, fill="#E9EEF4")
        draw.rounded_rectangle((plot_left, y - bar_height / 2, max(plot_left + bar_height, end_x), y + bar_height / 2), radius=bar_height // 2, fill=item["color"])
        display = metric["display"][index]
        value_width = _text_size(draw, display, _font(latin_font_path, 44))[0]
        if end_x - plot_left > value_width + 62:
            _right_text(draw, end_x - 24, y - 28, display, _font(latin_font_path, 44), fill=WHITE)
        else:
            draw.text((end_x + 16, y - 28), display, font=_font(latin_font_path, 44), fill=item["color"])


def generate_performance_chart(
    data_path: Path,
    output_path: Path,
    *,
    chinese_font_path: Path,
    latin_font_path: Path,
) -> Path:
    data = _load_chart_data(data_path)
    series = data["series"]
    metrics = {metric["key"]: metric for metric in data["metrics"]}

    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = _font(chinese_font_path, 58)
    subtitle_font = _font(chinese_font_path, 32)
    label_font = _font(chinese_font_path, 34)
    number_font = _font(latin_font_path, 44)

    draw.text((52, 31), "端到端性能与通信量概览", font=title_font, fill=INK)
    draw.text((54, 107), "总量指标与单样本指标分区展示，避免样本规模差异造成误读", font=subtitle_font, fill=MUTED)

    sample_metric = metrics["sample_count"]
    chip_left = 1510
    chip_width = 555
    for index, item in enumerate(series):
        left = chip_left + index * (chip_width + 34)
        top = 27
        draw.rounded_rectangle((left, top, left + chip_width, top + 122), radius=24, fill="#F4F7FA", outline=BORDER, width=2)
        draw.rounded_rectangle((left + 22, top + 21, left + 36, top + 99), radius=7, fill=item["color"])
        draw.text((left + 62, top + 22), _single_line_label(item["label"]), font=label_font, fill=INK)
        _right_text(draw, left + chip_width - 28, top + 18, sample_metric["display"][index], number_font, fill=item["color"])
        _right_text(draw, left + chip_width - 28, top + 70, "样本", subtitle_font, fill=MUTED)

    cards = [
        (metrics["total_seconds"], (52, 190, 1346, 774)),
        (metrics["seconds_per_sample"], (1380, 190, 2674, 774)),
        (metrics["communication_gib"], (52, 806, 1346, 1390)),
        (metrics["communication_per_sample_gib"], (1380, 806, 2674, 1390)),
    ]
    for metric, box in cards:
        _metric_card(
            draw,
            box,
            metric,
            series,
            chinese_font_path=chinese_font_path,
            latin_font_path=latin_font_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    generated = generate_performance_chart(
        root / "performance_chart_data.json",
        root / "output" / "intermediate" / "strict_format" / "performance_chart_generated.png",
        chinese_font_path=Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        latin_font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    print(generated)
