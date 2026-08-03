#!/usr/bin/env python3
"""Generate the report's statistical and assessment figures from data.

Every figure is drawn on a blank canvas.  The generator does not open,
retouch, trace, or reuse pixels from the submitted report figures.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
GRID = "#DCE4ED"
PANEL = "#F7F9FC"
PANEL_ALT = "#F0F4F9"
BORDER = "#D7E0EA"
NAVY = "#2D5B9B"
NAVY_DARK = "#183A6B"
TEAL = "#1E9B91"
AMBER = "#E99A2E"
CORAL = "#DF6B67"
GREEN = "#2F8B57"
VIOLET = "#7867B7"
SLATE = "#6B7280"


class FigureStyle:
    def __init__(self, regular_font: Path, bold_font: Path, latin_font: Path) -> None:
        self.regular_font = regular_font
        self.bold_font = bold_font
        self.latin_font = latin_font
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, *, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
        kind = "latin" if latin else ("bold" if bold else "regular")
        key = (kind, size)
        if key not in self._cache:
            path = self.latin_font if latin else (self.bold_font if bold else self.regular_font)
            self._cache[key] = ImageFont.truetype(str(path), size)
        return self._cache[key]


def _load_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "base_rate_scan",
        "baseline_comparison",
        "probability_distribution",
        "secure_function_benchmark",
        "primitive_benchmark",
        "guard_matrix",
        "ablation",
        "capability_matrix",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing report figure data: {sorted(missing)}")
    return data


def _canvas(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, WHITE)
    return image, ImageDraw.Draw(image)


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


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    *,
    max_lines: int | None = None,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and _text_size(draw, candidate, font)[0] > max_width:
            lines.append(current)
            current = char
            if max_lines is not None and len(lines) >= max_lines:
                return lines
        else:
            current = candidate
    if current and (max_lines is None or len(lines) < max_lines):
        lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    *,
    fill: str = INK,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    lines = _wrap_text(draw, text, font, max_width, max_lines=max_lines)
    for index, line in enumerate(lines):
        draw.text((x, y + index * (font.size + line_gap)), line, font=font, fill=fill)
    return len(lines)


def _rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: str = PANEL,
    outline: str = BORDER,
    radius: int = 22,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _legend_item(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    color: str,
    font: ImageFont.FreeTypeFont,
    *,
    line: bool = False,
) -> float:
    if line:
        draw.line((x, y + 15, x + 52, y + 15), fill=color, width=7)
        draw.ellipse((x + 20, y + 5, x + 40, y + 25), fill=color, outline=WHITE, width=3)
    else:
        draw.rounded_rectangle((x, y + 2, x + 36, y + 30), radius=6, fill=color)
    draw.text((x + 50, y), label, font=font, fill=INK)
    return x + 50 + _text_size(draw, label, font)[0] + 46


def _rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = MUTED,
) -> None:
    box = font.getbbox(text)
    width = box[2] - box[0]
    height = box[3] - box[1]
    layer = Image.new("RGBA", (width + 30, height + 30), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((15, 15 - box[1]), text, font=font, fill=fill)
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, (center[0] - layer.width // 2, center[1] - layer.height // 2), layer)


def _nice_max(value: float, target_ticks: int = 5) -> float:
    if value <= 0:
        return 1.0
    raw = value / target_ticks
    exponent = 10 ** math.floor(math.log10(raw))
    fraction = raw / exponent
    nice_fraction = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice_fraction * exponent * target_ticks


def _save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def _generate_base_rate_scan(data: dict[str, Any], output: Path, style: FigureStyle) -> Path:
    image, draw = _canvas((2440, 1288))
    plot = (190, 155, 2290, 925)
    rates = data["base_rate"]
    threshold = data["threshold_accuracy"]
    argmax = data["argmax_accuracy"]
    auc_pct = [value * 100 for value in data["auc"]]
    selected = data["selected_base_rate"]

    title_font = style.font(50, bold=True)
    label_font = style.font(38)
    tick_font = style.font(32, latin=True)
    value_font = style.font(30, bold=True, latin=True)

    draw.text((72, 38), "部署口径扫描：精度、AUC 与三阶段保留率", font=title_font, fill=INK)
    legend_x = 1190
    legend_x = _legend_item(draw, legend_x, 44, "阈值精度", NAVY, label_font, line=True)
    legend_x = _legend_item(draw, legend_x, 44, "AUC × 100", AMBER, label_font, line=True)
    _legend_item(draw, legend_x, 44, "argmax 精度", SLATE, label_font, line=True)

    x0, y0, x1, y1 = plot
    selected_index = rates.index(selected)
    x_positions = [x0 + i * (x1 - x0) / (len(rates) - 1) for i in range(len(rates))]
    selected_x = x_positions[selected_index]
    draw.rounded_rectangle((selected_x - 105, y0 - 25, selected_x + 105, y1 + 26), radius=20, fill="#EDF4FF")

    for value in range(60, 101, 5):
        y = y1 - (value - 60) / 40 * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        _right_text(draw, x0 - 24, y - 14, str(value), tick_font, fill=MUTED)
    draw.line((x0, y0, x0, y1), fill=SLATE, width=3)
    draw.line((x0, y1, x1, y1), fill=SLATE, width=3)
    _rotated_label(image, (62, (y0 + y1) // 2), "指标值（%）", label_font)

    def y_of(value: float) -> float:
        return y1 - (value - 60) / 40 * (y1 - y0)

    series = [
        (threshold, NAVY, 9),
        (auc_pct, AMBER, 8),
        (argmax, SLATE, 6),
    ]
    for values, color, width in series:
        points = [(x, y_of(v)) for x, v in zip(x_positions, values)]
        draw.line(points, fill=color, width=width, joint="curve")
        for px, py in points:
            draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=color, outline=WHITE, width=4)

    for index, x in enumerate(x_positions):
        th_y = y_of(threshold[index])
        auc_y = y_of(auc_pct[index])
        arg_y = y_of(argmax[index])
        if index == 0:
            draw.text((x + 20, th_y + 20), f"{threshold[index]:.4f}", font=value_font, fill=NAVY_DARK)
            draw.text((x + 20, arg_y + 24), f"{argmax[index]:.4f}", font=value_font, fill=SLATE)
            draw.text((x + 20, auc_y - 48), f"{data['auc'][index]:.4f}", font=value_font, fill="#A96510")
        elif index == len(x_positions) - 1:
            _center_text(draw, x, th_y - 48, f"{threshold[index]:.4f}", value_font, fill=NAVY_DARK)
            _center_text(draw, x, arg_y + 30, f"{argmax[index]:.4f}", value_font, fill=SLATE)
            _center_text(draw, x, auc_y - 48, f"{data['auc'][index]:.4f}", value_font, fill="#A96510")
        else:
            _center_text(draw, x, th_y + 20, f"{threshold[index]:.4f}", value_font, fill=NAVY_DARK)
            _center_text(draw, x, arg_y + 24, f"{argmax[index]:.4f}", value_font, fill=SLATE)
            _center_text(draw, x, auc_y - 48, f"{data['auc'][index]:.4f}", value_font, fill="#A96510")
        _center_text(draw, x, 960, f"base_rate {rates[index]:.1f}", style.font(32, bold=True), fill=INK)
        keep_label = f"保留率 {data['three_stage_keep_rate'][index]}"
        if index == len(x_positions) - 1:
            _right_text(draw, 2370, 1013, keep_label, style.font(26), fill=MUTED)
        else:
            _center_text(draw, x, 1013, keep_label, style.font(26), fill=MUTED)
    draw.line((selected_x, y0 - 10, selected_x, y1 + 25), fill=CORAL, width=4)
    badge = (selected_x - 210, 93, selected_x + 210, 150)
    draw.rounded_rectangle(badge, radius=24, fill=CORAL)
    _center_text(draw, selected_x, 100, "正式主线 · base_rate = 0.70", style.font(27, bold=True), fill=WHITE)

    note = "统一把 AUC 换算为百分制，避免双纵轴造成视觉误判；浅蓝色带标记最终部署口径。"
    draw.rounded_rectangle((72, 1110, 2368, 1225), radius=20, fill=PANEL_ALT)
    draw.text((108, 1139), note, font=style.font(32), fill=MUTED)
    return _save(image, output)


def _generate_baseline_comparison(rows: list[dict[str, Any]], output: Path, style: FigureStyle) -> Path:
    image, draw = _canvas((2056, 1104))
    plot_left, plot_right = 700, 1930
    plot_top, plot_bottom = 205, 910
    x_min, x_max = 60.0, 100.0
    title_font = style.font(48, bold=True)
    method_font = style.font(36, bold=True)
    detail_font = style.font(28)
    tick_font = style.font(30, latin=True)
    value_font = style.font(28, bold=True, latin=True)

    draw.text((54, 35), "医疗任务性能基线：阈值精度与 AUC 对照", font=title_font, fill=INK)
    legend_x = 1030
    legend_x = _legend_item(draw, legend_x, 42, "阈值精度", NAVY, style.font(32), line=True)
    _legend_item(draw, legend_x, 42, "AUC × 100", AMBER, style.font(32), line=True)

    for tick in range(60, 101, 5):
        x = plot_left + (tick - x_min) / (x_max - x_min) * (plot_right - plot_left)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=2)
        _center_text(draw, x, plot_bottom + 22, str(tick), tick_font, fill=MUTED)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=SLATE, width=3)
    _center_text(draw, (plot_left + plot_right) / 2, 975, "指标值（%）", style.font(34), fill=MUTED)

    row_gap = (plot_bottom - plot_top) / len(rows)
    for index, row in enumerate(rows):
        row_y0 = plot_top + index * row_gap
        center_y = row_y0 + row_gap / 2
        if row.get("highlight"):
            draw.rounded_rectangle((35, row_y0 + 7, 1998, row_y0 + row_gap - 7), radius=18, fill="#EDF4FF")
            draw.rounded_rectangle((43, row_y0 + 24, 55, row_y0 + row_gap - 24), radius=6, fill=NAVY)
        draw.text((78, center_y - 38), row["method"], font=method_font, fill=INK)
        draw.text((78, center_y + 5), row["detail"], font=detail_font, fill=MUTED)

        threshold = float(row["threshold_accuracy"])
        auc = float(row["auc"]) * 100
        threshold_x = plot_left + (threshold - x_min) / (x_max - x_min) * (plot_right - plot_left)
        auc_x = plot_left + (auc - x_min) / (x_max - x_min) * (plot_right - plot_left)
        draw.line((min(threshold_x, auc_x), center_y, max(threshold_x, auc_x), center_y), fill="#AAB5C3", width=7)
        draw.ellipse((threshold_x - 13, center_y - 13, threshold_x + 13, center_y + 13), fill=NAVY, outline=WHITE, width=4)
        draw.ellipse((auc_x - 13, center_y - 13, auc_x + 13, center_y + 13), fill=AMBER, outline=WHITE, width=4)
        _center_text(draw, threshold_x, center_y - 50, f"{threshold:.2f}", value_font, fill=NAVY_DARK)
        _center_text(draw, auc_x, center_y + 24, f"{auc:.2f}", value_font, fill="#A96510")

    draw.rounded_rectangle((54, 1025, 2002, 1082), radius=16, fill=PANEL_ALT)
    _center_text(draw, 1028, 1032, "AUC 统一按百分制展示；连线仅表示同一方案的两项指标差距。", style.font(28), fill=MUTED)
    return _save(image, output)


def _histogram(values: Sequence[float], labels: Sequence[int], bins: int) -> tuple[list[int], list[int]]:
    negative = [0] * bins
    positive = [0] * bins
    for value, label in zip(values, labels):
        index = min(bins - 1, max(0, int(value * bins)))
        (positive if label else negative)[index] += 1
    return negative, positive


def _accuracy(values: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    correct = sum(int((value >= threshold) == bool(label)) for value, label in zip(values, labels))
    return correct / len(values)


def _read_probability_sources(
    config: dict[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    dynamic_json = json.loads((repo_root / config["dynamic_prediction_json"]).read_text(encoding="utf-8"))
    dynamic_probabilities = [float(item[1]) for item in dynamic_json["prediction_preview"]["probabilities"]]
    image_paths = (repo_root / config["dynamic_image_list"]).read_text(encoding="utf-8").splitlines()
    dynamic_labels = [int(Path(item).parent.name) for item in image_paths if item.strip()]
    if len(dynamic_probabilities) != len(dynamic_labels):
        raise ValueError("dynamic probability count does not match label count")

    dense_probabilities: list[float] = []
    dense_labels: list[int] = []
    with (repo_root / config["densenet_prediction_csv"]).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            dense_probabilities.append(float(row["prob_positive"]))
            dense_labels.append(int(row["label"]))
    dense_summary = json.loads((repo_root / config["densenet_summary_json"]).read_text(encoding="utf-8"))

    dynamic_threshold = float(config["dynamic_calibrated_threshold"])
    dense_threshold = float(dense_summary["best_threshold"])
    return [
        {
            "title": "医疗动态安全剪枝路径",
            "values": dynamic_probabilities,
            "labels": dynamic_labels,
            "default_threshold": float(config["dynamic_default_threshold"]),
            "threshold": dynamic_threshold,
            "argmax_accuracy": _accuracy(dynamic_probabilities, dynamic_labels, 0.5),
            "threshold_accuracy": _accuracy(dynamic_probabilities, dynamic_labels, dynamic_threshold),
        },
        {
            "title": "DenseNet121 明文基线",
            "values": dense_probabilities,
            "labels": dense_labels,
            "default_threshold": 0.5,
            "threshold": dense_threshold,
            "argmax_accuracy": _accuracy(dense_probabilities, dense_labels, 0.5),
            "threshold_accuracy": _accuracy(dense_probabilities, dense_labels, dense_threshold),
        },
    ]


def _draw_histogram_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    dataset: dict[str, Any],
    bins: int,
    style: FigureStyle,
) -> None:
    left, top, right, bottom = box
    _rounded_panel(draw, box, fill="#FBFCFE", radius=26)
    draw.text((left + 40, top + 28), dataset["title"], font=style.font(48, bold=True), fill=INK)
    draw.text((left + 42, top + 91), f"n = {len(dataset['values'])} · 横轴：阳性类别概率", font=style.font(30), fill=MUTED)

    badge = (right - 430, top + 24, right - 34, top + 142)
    draw.rounded_rectangle(badge, radius=18, fill="#EEF3F8")
    draw.text((badge[0] + 22, badge[1] + 10), f"argmax  {dataset['argmax_accuracy'] * 100:.2f}%", font=style.font(30, bold=True), fill=INK)
    draw.text((badge[0] + 22, badge[1] + 61), f"校准后   {dataset['threshold_accuracy'] * 100:.2f}%", font=style.font(30, bold=True), fill=GREEN)

    legend_y = top + 153
    legend_x = left + 42
    legend_x = _legend_item(draw, legend_x, legend_y, "阴性样本", NAVY, style.font(30))
    _legend_item(draw, legend_x, legend_y, "阳性样本", CORAL, style.font(30))

    plot_left, plot_top = left + 112, top + 230
    plot_right, plot_bottom = right - 42, bottom - 94
    negative, positive = _histogram(dataset["values"], dataset["labels"], bins)
    max_count = _nice_max(max(max(negative), max(positive)), 4)
    for index in range(5):
        value = max_count * index / 4
        y = plot_bottom - (plot_bottom - plot_top) * index / 4
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        _right_text(draw, plot_left - 18, y - 17, f"{value:.0f}", style.font(27, latin=True), fill=MUTED)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=SLATE, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=SLATE, width=3)

    bin_width = (plot_right - plot_left) / bins
    for index, (neg_count, pos_count) in enumerate(zip(negative, positive)):
        x = plot_left + index * bin_width
        neg_height = (plot_bottom - plot_top) * neg_count / max_count
        pos_height = (plot_bottom - plot_top) * pos_count / max_count
        gap = 3
        draw.rounded_rectangle(
            (x + gap, plot_bottom - neg_height, x + bin_width / 2 - gap, plot_bottom),
            radius=3,
            fill=NAVY,
        )
        draw.rounded_rectangle(
            (x + bin_width / 2 + gap, plot_bottom - pos_height, x + bin_width - gap, plot_bottom),
            radius=3,
            fill=CORAL,
        )

    for tick in range(6):
        value = tick / 5
        x = plot_left + value * (plot_right - plot_left)
        draw.line((x, plot_bottom, x, plot_bottom + 10), fill=SLATE, width=2)
        _center_text(draw, x, plot_bottom + 20, f"{value:.1f}", style.font(27, latin=True), fill=MUTED)

    threshold_lines = [
        (dataset["default_threshold"], SLATE, "默认 0.500"),
        (dataset["threshold"], GREEN, f"校准 {dataset['threshold']:.3f}"),
    ]
    for threshold, color, label in threshold_lines:
        x = plot_left + threshold * (plot_right - plot_left)
        dash_y = plot_top
        while dash_y < plot_bottom:
            draw.line((x, dash_y, x, min(dash_y + 18, plot_bottom)), fill=color, width=4)
            dash_y += 30
        label_width = _text_size(draw, label, style.font(27, bold=True))[0] + 30
        draw.rounded_rectangle((x - label_width / 2, plot_top + 10, x + label_width / 2, plot_top + 55), radius=13, fill=WHITE, outline=color, width=2)
        _center_text(draw, x, plot_top + 14, label, style.font(27, bold=True), fill=color)

    _rotated_label(image, (left + 34, (plot_top + plot_bottom) // 2), "样本数", style.font(30))


def _generate_probability_distribution(
    config: dict[str, Any], repo_root: Path, output: Path, style: FigureStyle
) -> Path:
    image, draw = _canvas((3416, 1656))
    datasets = _read_probability_sources(config, repo_root)
    _draw_histogram_panel(image, draw, (22, 20, 1695, 1634), datasets[0], int(config["bins"]), style)
    _draw_histogram_panel(image, draw, (1721, 20, 3394, 1634), datasets[1], int(config["bins"]), style)
    return _save(image, output)


def _log_position(value: float, minimum: float, maximum: float, left: float, right: float) -> float:
    safe = max(value, minimum)
    return left + (math.log10(safe) - math.log10(minimum)) / (math.log10(maximum) - math.log10(minimum)) * (right - left)


def _generate_benchmark_figure(
    rows: list[dict[str, Any]],
    output: Path,
    style: FigureStyle,
    *,
    size: tuple[int, int],
    latency_keys: tuple[str, str, str],
    latency_unit: str,
    latency_min: float,
    latency_max: float,
    communication_max: float,
) -> Path:
    image, draw = _canvas(size)
    width, height = size
    split = int(width * 0.64)
    left_box = (20, 20, split - 16, height - 20)
    right_box = (split + 16, 20, width - 20, height - 20)
    _rounded_panel(draw, left_box, fill="#FBFCFE", radius=26)
    _rounded_panel(draw, right_box, fill="#FBFCFE", radius=26)

    draw.text((58, 45), f"(a) 通信时延（log10 {latency_unit}）", font=style.font(50, bold=True), fill=INK)
    draw.text((split + 54, 45), "(b) 双端合计通信量（MiB）", font=style.font(50, bold=True), fill=INK)
    legend_x = 1030
    for label, color in (("Local", NAVY), ("LAN", AMBER), ("WAN", GREEN)):
        legend_x = _legend_item(draw, legend_x, 55, label, color, style.font(34), line=True)

    row_top = 215
    row_bottom = height - 120
    row_gap = (row_bottom - row_top) / len(rows)
    name_x = 58
    number_columns = (425, 585, 755)
    plot_left, plot_right = 860, split - 74
    draw.text((name_x, 142), "函数 / 原语", font=style.font(32, bold=True), fill=MUTED)
    for x, label, color in zip(number_columns, ("Local", "LAN", "WAN"), (NAVY, AMBER, GREEN)):
        _center_text(draw, x, 142, label, style.font(31, bold=True), fill=color)

    exponent_start = math.floor(math.log10(latency_min))
    exponent_end = math.ceil(math.log10(latency_max))
    for exponent in range(exponent_start, exponent_end + 1):
        value = 10**exponent
        x = _log_position(value, latency_min, latency_max, plot_left, plot_right)
        draw.line((x, row_top - 14, x, row_bottom), fill=GRID, width=2)
        _center_text(draw, x, 143, f"10^{exponent}", style.font(28, latin=True), fill=MUTED)

    comm_name_x = split + 54
    comm_plot_left, comm_plot_right = split + 500, width - 80
    draw.text((comm_name_x, 142), "函数 / 原语", font=style.font(32, bold=True), fill=MUTED)
    draw.text((comm_plot_left, 142), "0", font=style.font(28, latin=True), fill=MUTED)
    _right_text(draw, comm_plot_right, 142, f"{communication_max:.0f}", style.font(28, latin=True), fill=MUTED)

    for index, row in enumerate(rows):
        y0 = row_top + index * row_gap
        center_y = y0 + row_gap / 2
        if index % 2 == 0:
            draw.rounded_rectangle((42, y0 + 4, split - 38, y0 + row_gap - 4), radius=14, fill="#F4F7FA")
            draw.rounded_rectangle((split + 38, y0 + 4, width - 42, y0 + row_gap - 4), radius=14, fill="#F4F7FA")
        name = row["name"]
        draw.multiline_text((name_x, center_y - (54 if "\n" in name else 27)), name, font=style.font(34, bold=True), fill=INK, spacing=6)
        values = [float(row[key]) for key in latency_keys]
        for x, value, color in zip(number_columns, values, (NAVY, AMBER, GREEN)):
            value_label = f"{value:.4f}" if value < 10 else (f"{value:.3f}" if value < 1000 else f"{value:.2f}")
            _center_text(draw, x, center_y - 22, value_label, style.font(28, latin=True), fill=color)
        positions = [_log_position(value, latency_min, latency_max, plot_left, plot_right) for value in values]
        draw.line((positions[0], center_y, positions[-1], center_y), fill="#B9C4D1", width=7)
        for px, color in zip(positions, (NAVY, AMBER, GREEN)):
            draw.ellipse((px - 12, center_y - 12, px + 12, center_y + 12), fill=color, outline=WHITE, width=4)

        draw.text((comm_name_x, center_y - 22), name.replace("\n", " "), font=style.font(32, bold=True), fill=INK)
        communication = float(row["communication_mib"])
        x = comm_plot_left + communication / communication_max * (comm_plot_right - comm_plot_left)
        draw.line((comm_plot_left, center_y, x, center_y), fill="#D0C9EA", width=12)
        draw.ellipse((x - 13, center_y - 13, x + 13, center_y + 13), fill=VIOLET, outline=WHITE, width=4)
        communication_label = f"{communication:.2f}"
        if x > comm_plot_right - 160:
            _right_text(draw, comm_plot_right - 8, center_y - 22, communication_label, style.font(30, bold=True, latin=True), fill="#54428C")
        else:
            draw.text((x + 20, center_y - 22), communication_label, font=style.font(30, bold=True, latin=True), fill="#54428C")

    return _save(image, output)


def _status_symbol(
    draw: ImageDraw.ImageDraw, center: tuple[float, float], status: str, style: FigureStyle, *, size: int = 32
) -> None:
    radius = size
    box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    if status == "yes":
        draw.ellipse(box, fill=GREEN)
        draw.line(
            (
                center[0] - radius * 0.48,
                center[1] + radius * 0.02,
                center[0] - radius * 0.10,
                center[1] + radius * 0.38,
                center[0] + radius * 0.52,
                center[1] - radius * 0.38,
            ),
            fill=WHITE,
            width=max(4, int(radius * 0.22)),
            joint="curve",
        )
    elif status == "no":
        draw.ellipse(box, fill=SLATE)
        inset = radius * 0.38
        line_width = max(4, int(radius * 0.20))
        draw.line((center[0] - inset, center[1] - inset, center[0] + inset, center[1] + inset), fill=WHITE, width=line_width)
        draw.line((center[0] + inset, center[1] - inset, center[0] - inset, center[1] + inset), fill=WHITE, width=line_width)
    elif status == "partial":
        draw.ellipse(box, fill=AMBER)
        draw.pieslice(box, start=-90, end=90, fill=WHITE)
        draw.ellipse(box, outline=AMBER, width=max(3, int(radius * 0.12)))
    else:
        raise ValueError(f"unknown capability status: {status}")


def _generate_guard_matrix(rows: list[dict[str, str]], output: Path, style: FigureStyle) -> Path:
    image, draw = _canvas((3542, 1912))
    margin = 24
    header_top, header_bottom = 24, 176
    footer_top = 1630
    columns = [margin, 880, 1615, 2345, 3518]
    headers = ["异常场景", "首个拦截层", "结果状态", "系统状态摘要"]
    header_icons = ["01", "02", "03", "04"]

    draw.rounded_rectangle((margin, header_top, 3518, footer_top - 22), radius=26, fill=WHITE, outline=BORDER, width=3)
    draw.rounded_rectangle((margin, header_top, 3518, header_bottom), radius=26, fill="#EAF0F8")
    draw.rectangle((margin, header_bottom - 26, 3518, header_bottom), fill="#EAF0F8")
    for index, header in enumerate(headers):
        x = columns[index] + 46
        draw.rounded_rectangle((x, 64, x + 54, 118), radius=14, fill=NAVY)
        _center_text(draw, x + 27, 72, header_icons[index], style.font(21, bold=True, latin=True), fill=WHITE)
        draw.text((x + 78, 59), header, font=style.font(48, bold=True), fill=NAVY_DARK)

    row_top = header_bottom
    row_height = (footer_top - 22 - row_top) / len(rows)
    for index, row in enumerate(rows):
        y0 = row_top + index * row_height
        y1 = y0 + row_height
        if index % 2 == 0:
            draw.rectangle((margin + 2, y0, 3516, y1), fill="#F8FAFC")
        draw.line((margin, y1, 3518, y1), fill=BORDER, width=2)
        draw.ellipse((columns[0] + 46, y0 + 47, columns[0] + 86, y0 + 87), outline=NAVY, width=5)
        draw.text((columns[0] + 112, y0 + 36), row["scenario"], font=style.font(38, bold=True), fill=INK)
        draw.text((columns[1] + 58, y0 + 36), row["layer"], font=style.font(36), fill=INK)

        costly = "资源代价" in row["status"]
        status_fill = "#FFF4DB" if costly else "#E7F5EC"
        status_color = AMBER if costly else GREEN
        pill = (columns[2] + 58, y0 + 28, columns[3] - 58, y0 + 99)
        draw.rounded_rectangle(pill, radius=22, fill=status_fill, outline=status_color, width=2)
        icon_center = (pill[0] + 39.5, pill[1] + 35.5)
        draw.ellipse((icon_center[0] - 17.5, icon_center[1] - 17.5, icon_center[0] + 17.5, icon_center[1] + 17.5), fill=status_color)
        draw.line(
            (
                icon_center[0] - 8,
                icon_center[1],
                icon_center[0] - 2,
                icon_center[1] + 7,
                icon_center[0] + 9,
                icon_center[1] - 8,
            ),
            fill=WHITE,
            width=4,
            joint="curve",
        )
        _center_text(draw, (pill[0] + pill[2]) / 2 + 18, pill[1] + 13, row["status"], style.font(31, bold=True), fill=status_color)
        _draw_wrapped(draw, columns[3] + 45, y0 + 23, row["summary"], style.font(34), columns[4] - columns[3] - 82, fill=INK, line_gap=7, max_lines=2)

    for x in columns[1:-1]:
        draw.line((x, header_top, x, footer_top - 22), fill=BORDER, width=2)

    draw.rounded_rectangle((margin, footer_top, 3518, 1888), radius=25, fill="#EEF3F8", outline=BORDER, width=2)
    draw.ellipse((68, 1685, 132, 1749), fill=NAVY)
    _center_text(draw, 100, 1694, "i", style.font(38, bold=True, latin=True), fill=WHITE)
    note1 = "结果状态为“通过”表示异常场景被有效拦截且系统保持稳定；“通过（有限时资源代价）”表示存在可控的瞬时资源消耗或短时抖动，但系统未失稳且可恢复。"
    note2 = "验证方法：黑盒测试为主，结合白盒辅助分析与日志审计，覆盖协议解析、会话管理、状态机、认证鉴权与流量控制等关键层级。"
    _draw_wrapped(draw, 172, 1658, note1, style.font(34), 3285, fill=INK, line_gap=9, max_lines=2)
    _draw_wrapped(draw, 172, 1770, note2, style.font(34), 3285, fill=MUTED, line_gap=9, max_lines=2)
    return _save(image, output)


def _generate_ablation(rows: list[dict[str, Any]], output: Path, style: FigureStyle) -> Path:
    image, draw = _canvas((3204, 1788))
    draw.text((64, 37), "动态剪枝消融：统一百分制指标对照", font=style.font(60, bold=True), fill=INK)
    legend_x = 1600
    legend_x = _legend_item(draw, legend_x, 57, "阈值精度", NAVY, style.font(37), line=True)
    legend_x = _legend_item(draw, legend_x, 57, "AUC × 100", AMBER, style.font(37), line=True)
    _legend_item(draw, legend_x, 57, "argmax 精度", GREEN, style.font(37), line=True)

    plot_left, plot_right = 900, 3060
    plot_top, plot_bottom = 245, 1430
    x_min, x_max = 60, 100
    for tick in range(60, 101, 5):
        x = plot_left + (tick - x_min) / (x_max - x_min) * (plot_right - plot_left)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=3)
        _center_text(draw, x, plot_bottom + 25, str(tick), style.font(34, latin=True), fill=MUTED)
    _center_text(draw, (plot_left + plot_right) / 2, 1515, "指标值（%）", style.font(40), fill=MUTED)

    row_gap = (plot_bottom - plot_top) / len(rows)
    offsets = (-58, 0, 58)
    metrics = (("threshold_accuracy", NAVY), ("auc", AMBER), ("argmax_accuracy", GREEN))
    for index, row in enumerate(rows):
        y0 = plot_top + index * row_gap
        center_y = y0 + row_gap / 2
        if row.get("highlight"):
            draw.rounded_rectangle((45, y0 + 12, 3159, y0 + row_gap - 12), radius=24, fill="#EDF4FF")
            draw.rounded_rectangle((55, y0 + 54, 69, y0 + row_gap - 54), radius=7, fill=NAVY)
        draw.text((100, center_y - 63), row["method"], font=style.font(46, bold=True), fill=INK)
        draw.text((100, center_y + 10), row["detail"], font=style.font(36), fill=MUTED)
        for offset, (key, color) in zip(offsets, metrics):
            value = float(row[key]) * 100 if key == "auc" else float(row[key])
            x = plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
            y = center_y + offset
            draw.line((plot_left, y, x, y), fill=color, width=8)
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=color, outline=WHITE, width=5)
            label_x = min(x + 24, plot_right - 80)
            draw.text((label_x, y - 25), f"{value:.2f}", font=style.font(34, bold=True, latin=True), fill=color)

    draw.rounded_rectangle((64, 1600, 3140, 1724), radius=22, fill=PANEL_ALT)
    draw.text((104, 1630), "同一横轴直接比较三项指标；AUC 乘以 100 后与准确率统一为百分制。", font=style.font(40), fill=MUTED)
    return _save(image, output)


def _generate_capability_matrix(data: dict[str, Any], output: Path, style: FigureStyle) -> Path:
    image, draw = _canvas((2692, 1192))
    columns = [22, 560, 1075, 1590, 2105, 2670]
    header_top, header_bottom = 20, 160
    footer_top = 1050
    draw.rounded_rectangle((22, 20, 2670, 1030), radius=24, fill=WHITE, outline=BORDER, width=3)
    draw.rounded_rectangle((22, header_top, 2670, header_bottom), radius=24, fill=NAVY_DARK)
    draw.rectangle((22, header_bottom - 24, 2670, header_bottom), fill=NAVY_DARK)
    headers = ["能力项", *data["columns"]]
    for index, header in enumerate(headers):
        _center_text(draw, (columns[index] + columns[index + 1]) / 2, 59, header, style.font(36, bold=True), fill=WHITE)
    draw.rounded_rectangle((columns[-2] + 6, 24, columns[-1] - 6, 1026), radius=22, fill="#EAF1FF", outline="#7EA7E8", width=4)
    draw.rounded_rectangle((columns[-2] + 6, 24, columns[-1] - 6, header_bottom), radius=22, fill="#2C5CC5")
    draw.rectangle((columns[-2] + 6, header_bottom - 24, columns[-1] - 6, header_bottom), fill="#2C5CC5")
    _center_text(draw, (columns[-2] + columns[-1]) / 2, 57, data["columns"][-1], style.font(40, bold=True), fill=WHITE)

    row_height = (1030 - header_bottom) / len(data["rows"])
    for index, row in enumerate(data["rows"]):
        y0 = header_bottom + index * row_height
        y1 = y0 + row_height
        if index % 2 == 0:
            draw.rectangle((24, y0, columns[-2], y1), fill="#F8FAFC")
        draw.line((22, y1, 2670, y1), fill=BORDER, width=2)
        draw.text((68, y0 + 31), row["capability"], font=style.font(34, bold=True), fill=INK)
        for column_index, status in enumerate(row["values"]):
            center_x = (columns[column_index + 1] + columns[column_index + 2]) / 2
            _status_symbol(draw, (center_x, (y0 + y1) / 2), status, style, size=26)
    for x in columns[1:-1]:
        draw.line((x, header_top, x, 1030), fill=BORDER, width=2)

    draw.rounded_rectangle((22, footer_top, 2670, 1170), radius=22, fill=PANEL_ALT, outline=BORDER, width=2)
    legend_x = 180
    for status, label in (("yes", "支持"), ("partial", "部分支持"), ("no", "不支持")):
        _status_symbol(draw, (legend_x, 1110), status, style, size=22)
        draw.text((legend_x + 42, 1085), label, font=style.font(32), fill=INK)
        legend_x += 430
    draw.text((1675, 1085), "蓝色高亮列为本作品能力边界", font=style.font(32, bold=True), fill=NAVY)
    return _save(image, output)


def generate_all_report_figures(
    data_path: Path,
    output_dir: Path,
    *,
    repo_root: Path,
    regular_font_path: Path,
    bold_font_path: Path,
    latin_font_path: Path,
) -> dict[str, Path]:
    data = _load_data(data_path)
    style = FigureStyle(regular_font_path, bold_font_path, latin_font_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig4_1": _generate_base_rate_scan(data["base_rate_scan"], output_dir / "fig4-1-base-rate.png", style),
        "fig4_2": _generate_baseline_comparison(data["baseline_comparison"], output_dir / "fig4-2-baselines.png", style),
        "fig4_3": _generate_probability_distribution(data["probability_distribution"], repo_root, output_dir / "fig4-3-probability.png", style),
        "fig4_5": _generate_benchmark_figure(
            data["secure_function_benchmark"],
            output_dir / "fig4-5-secure-functions.png",
            style,
            size=(3674, 2006),
            latency_keys=("local_s", "lan_s", "wan_s"),
            latency_unit="秒",
            latency_min=0.005,
            latency_max=100,
            communication_max=320,
        ),
        "fig4_6": _generate_benchmark_figure(
            data["primitive_benchmark"],
            output_dir / "fig4-6-primitives.png",
            style,
            size=(3626, 1980),
            latency_keys=("local_ms", "lan_ms", "wan_ms"),
            latency_unit="毫秒",
            latency_min=0.1,
            latency_max=100000,
            communication_max=250,
        ),
        "fig4_7": _generate_guard_matrix(data["guard_matrix"], output_dir / "fig4-7-guard-matrix.png", style),
        "fig4_8": _generate_ablation(data["ablation"], output_dir / "fig4-8-ablation.png", style),
        "fig5_1": _generate_capability_matrix(data["capability_matrix"], output_dir / "fig5-1-capabilities.png", style),
    }
    return outputs


if __name__ == "__main__":
    report_root = Path(__file__).resolve().parent
    generated = generate_all_report_figures(
        report_root / "report_figure_data.json",
        report_root / "output" / "intermediate" / "strict_format" / "generated_figures",
        repo_root=report_root.parents[1],
        regular_font_path=Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        bold_font_path=Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
        latin_font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for key, path in generated.items():
        print(f"{key}: {path}")
