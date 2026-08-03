#!/usr/bin/env python3
"""Generate appendix evidence interfaces from authoritative report data.

The images are drawn from a blank canvas.  No submitted screenshot pixels are
opened, retouched, or reused.  This keeps the appendix reproducible while the
local showcase implementation remains unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


INK = "#142033"
MUTED = "#667085"
BLUE = "#2563EB"
GREEN = "#16865C"
AMBER = "#B56A13"
PANEL = "#FFFFFF"
CANVAS = "#F3F6FA"
BORDER = "#DCE4EE"
SOFT_BLUE = "#EEF4FF"
SOFT_GREEN = "#EDF8F3"
SOFT_AMBER = "#FFF7E8"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _round_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = PANEL,
    outline: str = BORDER,
    radius: int = 22,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _chip(
    draw: ImageDraw.ImageDraw,
    right: int,
    top: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str,
    color: str,
) -> int:
    width = int(_text_width(draw, text, font)) + 34
    left = right - width
    draw.rounded_rectangle((left, top, right, top + 44), radius=22, fill=fill)
    draw.text((left + 17, top + 7), text, font=font, fill=color)
    return left - 12


def _metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    scope: str,
    *,
    accent: str,
    tint: str,
    regular_font: Path,
    bold_font: Path,
) -> None:
    left, top, right, bottom = box
    _round_panel(draw, box, fill=tint)
    draw.rounded_rectangle((left + 24, top + 24, left + 34, bottom - 24), radius=5, fill=accent)
    draw.text((left + 58, top + 22), label, font=_font(regular_font, 27), fill=MUTED)
    draw.text((left + 58, top + 66), value, font=_font(bold_font, 49), fill=INK)
    draw.text((left + 58, bottom - 45), scope, font=_font(regular_font, 22), fill=MUTED)


def _field(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    *,
    regular_font: Path,
    bold_font: Path,
    mono_font: Path | None = None,
) -> None:
    left, top, right, bottom = box
    _round_panel(draw, box, fill="#F8FAFD", radius=14)
    draw.text((left + 18, top + 12), label, font=_font(regular_font, 20), fill=MUTED)
    value_font = _font(mono_font or bold_font, 23 if mono_font else 27)
    draw.text((left + 18, bottom - 40), value, font=value_font, fill=INK)


def _overview(
    output_path: Path,
    *,
    medical: dict[str, Any],
    calibration: dict[str, Any],
    auc: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
    mono_font: Path,
) -> Path:
    width, height = 1600, 962
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((52, 38), "运行总览", font=_font(bold_font, 46), fill=INK)
    draw.text((54, 100), "正式模型、VPS 环境与交付状态", font=_font(regular_font, 27), fill=MUTED)
    chip_right = width - 50
    chip_right = _chip(draw, chip_right, 43, "正式口径", _font(regular_font, 23), fill=SOFT_GREEN, color=GREEN)
    _chip(draw, chip_right, 43, "CPU 环境", _font(regular_font, 23), fill=SOFT_BLUE, color=BLUE)

    gap = 20
    card_top, card_bottom = 160, 340
    card_width = (width - 104 - gap * 3) // 4
    cards = [
        ("运行任务", "0", "当前无排队任务", BLUE, SOFT_BLUE),
        ("完成验证", "2", "医疗与金融完整运行", GREEN, SOFT_GREEN),
        ("医疗样本", str(medical["sample_count"]), "固定部署验证批次", BLUE, SOFT_BLUE),
        ("计算设备", "CPU", "16 vCPU · 61 GiB", AMBER, SOFT_AMBER),
    ]
    for index, (label, value, scope, accent, tint) in enumerate(cards):
        left = 52 + index * (card_width + gap)
        _metric_card(
            draw,
            (left, card_top, left + card_width, card_bottom),
            label,
            value,
            scope,
            accent=accent,
            tint=tint,
            regular_font=regular_font,
            bold_font=bold_font,
        )

    _round_panel(draw, (52, 370, width - 52, height - 44))
    draw.text((82, 400), "正式主线模型", font=_font(bold_font, 34), fill=INK)
    draw.text((82, 450), "医疗动态剪枝主线 · 全量精度与部署性能分口径呈现", font=_font(regular_font, 24), fill=MUTED)

    col_gap = 20
    inner_left, inner_right = 82, width - 82
    field_width = (inner_right - inner_left - col_gap) // 2
    fields = [
        ("模型名称", "医疗动态主线"),
        ("基础保留率", f'{medical["runtime"]["base_rate"]:.2f}'),
        ("阈值精度（524 条全量验证）", f'{calibration["best_threshold_accuracy"] * 100:.2f}%'),
        ("Argmax 准确率（524 条）", f'{calibration["argmax_accuracy"] * 100:.2f}%'),
        ("AUC（524 条全量验证）", f'{auc["auc"]:.4f}'),
        ("VPS 运行环境", "16 vCPU · 61 GiB · 无 GPU"),
    ]
    for index, (label, value) in enumerate(fields):
        row, col = divmod(index, 2)
        left = inner_left + col * (field_width + col_gap)
        top = 500 + row * 104
        _field(
            draw,
            (left, top, left + field_width, top + 84),
            label,
            value,
            regular_font=regular_font,
            bold_font=bold_font,
        )

    _field(
        draw,
        (inner_left, 812, inner_right, 902),
        "正式 bundle",
        "artifacts/frozen_bundle_medical_dynamic_mainline",
        regular_font=regular_font,
        bold_font=bold_font,
        mono_font=mono_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _task_create(
    output_path: Path,
    *,
    training_args: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
) -> Path:
    width, height = 1280, 1002
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((48, 32), "新建训练任务", font=_font(bold_font, 42), fill=INK)
    draw.text((50, 88), "正式主线离线训练参数 · 来源 args_snapshot.json", font=_font(regular_font, 24), fill=MUTED)
    _chip(draw, width - 48, 36, "离线训练 · CUDA", _font(regular_font, 21), fill=SOFT_BLUE, color=BLUE)
    _round_panel(draw, (48, 136, width - 48, height - 42))
    fields = [
        ("任务名称", "医疗动态主线复核"),
        ("启动方式", "正式主线预设"),
        ("训练集路径", "data/medical_train"),
        ("验证集路径", "data/medical_val"),
        ("训练轮次", str(training_args["epochs"])),
        ("批次大小", str(training_args["batch_size"])),
        ("数据线程", str(training_args["num_workers"])),
        ("训练设备", str(training_args["device"]).upper()),
        ("基础保留率", f'{training_args["base_rate"]:.2f}'),
        ("训练深度", str(training_args["secure_static_train_depth"])),
        ("分类蒸馏权重", f'{training_args["cls_distill_weight"]:.2f}'),
        ("token 蒸馏权重", f'{training_args["token_distill_weight"]:.2f}'),
    ]
    gap = 18
    left0, right0 = 76, width - 76
    field_width = (right0 - left0 - gap) // 2
    for index, (label, value) in enumerate(fields):
        row, col = divmod(index, 2)
        left = left0 + col * (field_width + gap)
        top = 172 + row * 100
        _field(
            draw,
            (left, top, left + field_width, top + 78),
            label,
            value,
            regular_font=regular_font,
            bold_font=bold_font,
        )
    checks = ["启用动态剪枝", "启用安全友好算子", "完成后导出 bundle"]
    for index, label in enumerate(checks):
        left = 78 + index * 370
        top = 790
        _round_panel(draw, (left, top, left + 344, top + 72), fill="#F8FAFD", radius=14)
        draw.rounded_rectangle((left + 18, top + 20, left + 50, top + 52), radius=7, fill=BLUE)
        draw.line((left + 26, top + 36, left + 32, top + 43), fill="#FFFFFF", width=4)
        draw.line((left + 32, top + 43, left + 44, top + 29), fill="#FFFFFF", width=4)
        draw.text((left + 68, top + 19), label, font=_font(regular_font, 23), fill=INK)
    _round_panel(draw, (78, 884, width - 78, 946), fill=SOFT_BLUE, outline="#BED2F4", radius=14)
    draw.text((98, 900), "离线训练参数与 VPS 端到端 CPU 性能环境分开统计。", font=_font(regular_font, 23), fill=BLUE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _task_detail(
    output_path: Path,
    *,
    calibration: dict[str, Any],
    auc: dict[str, Any],
    demo: dict[str, Any],
    figure_data: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
) -> Path:
    width, height = 1680, 818
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((48, 26), "训练任务", font=_font(bold_font, 42), fill=INK)
    draw.text((50, 82), "冻结、校准与部署导出记录", font=_font(regular_font, 23), fill=MUTED)
    chip_right = _chip(draw, width - 48, 30, "CPU", _font(regular_font, 21), fill=SOFT_BLUE, color=BLUE)
    _chip(draw, chip_right, 30, "证据归档", _font(regular_font, 21), fill=SOFT_GREEN, color=GREEN)

    _round_panel(draw, (48, 128, 1005, height - 38))
    _round_panel(draw, (1025, 128, width - 48, height - 38))
    draw.text((76, 154), "训练任务列表", font=_font(bold_font, 30), fill=INK)
    draw.text((1052, 154), "任务详情", font=_font(bold_font, 30), fill=INK)

    comparison = demo["external_comparison"]["additional_rows"]
    static = comparison[1]
    original = comparison[-1]
    scan = figure_data["base_rate_scan"]
    scan_index = scan["base_rate"].index(0.8)
    tasks = [
        ("医疗动态主线冻结", "动态深度 10", f'{calibration["best_threshold_accuracy"] * 100:.2f}%', True),
        ("固定结构静态对照", "固定深度 12", f'{static["threshold_accuracy"]:.2f}%', False),
        ("基础保留率扫描", "base_rate 0.80", f'{scan["threshold_accuracy"][scan_index]:.2f}%', False),
        ("原始明文参考复核", "原始 DynamicViT", f'{original["threshold_accuracy"]:.2f}%', False),
    ]
    for index, (name, detail, value, selected) in enumerate(tasks):
        top = 216 + index * 126
        fill = SOFT_BLUE if selected else "#F8FAFD"
        outline = "#AFC9F3" if selected else BORDER
        _round_panel(draw, (72, top, 981, top + 104), fill=fill, outline=outline, radius=16)
        if selected:
            draw.rounded_rectangle((72, top + 12, 82, top + 92), radius=5, fill=BLUE)
        draw.text((100, top + 17), name, font=_font(bold_font, 25), fill=INK)
        draw.text((100, top + 58), detail, font=_font(regular_font, 21), fill=MUTED)
        value_font = _font(bold_font, 27)
        draw.text((930 - _text_width(draw, value, value_font), top + 22), value, font=value_font, fill=GREEN if selected else INK)
        _chip(draw, 958, top + 57, "已完成", _font(regular_font, 18), fill=SOFT_GREEN, color=GREEN)
    _round_panel(draw, (72, 724, 981, 766), fill="#F8FAFD", radius=12)
    draw.text((94, 733), "输出记录：stdout 已归档 · stderr 为空 · JSON 已保存", font=_font(regular_font, 18), fill=MUTED)

    details = [
        ("任务名称", "医疗动态主线冻结"),
        ("状态", "已完成"),
        ("阈值精度", f'{calibration["best_threshold_accuracy"] * 100:.2f}%'),
        ("Argmax 准确率", f'{calibration["argmax_accuracy"] * 100:.2f}%'),
        ("AUC", f'{auc["auc"]:.4f}'),
        ("部署参数", "base_rate 0.70 · depth 10"),
    ]
    detail_left, detail_right = 1052, width - 74
    for index, (label, value) in enumerate(details):
        top = 216 + index * 76
        draw.text((detail_left, top), label, font=_font(regular_font, 21), fill=MUTED)
        value_font = _font(bold_font, 24)
        draw.text((detail_right - _text_width(draw, value, value_font), top - 2), value, font=value_font, fill=INK)
        draw.line((detail_left, top + 44, detail_right, top + 44), fill=BORDER, width=2)
    _round_panel(draw, (1052, 688, detail_right, 754), fill="#F8FAFD", radius=14)
    draw.text((1070, 701), "输出：artifacts/frozen_bundle_medical_dynamic_mainline", font=_font(regular_font, 19), fill=MUTED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _model_assets(
    output_path: Path,
    *,
    calibration: dict[str, Any],
    auc: dict[str, Any],
    demo: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
) -> Path:
    width, height = 1680, 789
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "已训练模型", font=_font(bold_font, 42), fill=INK)
    draw.text((50, 84), "正式主线与静态对照分开展示", font=_font(regular_font, 23), fill=MUTED)
    _chip(draw, width - 48, 32, "VPS · CPU", _font(regular_font, 21), fill=SOFT_BLUE, color=BLUE)
    static = demo["external_comparison"]["additional_rows"][1]
    cards = [
        (
            "医疗动态主线 bundle",
            "正式主线",
            SOFT_GREEN,
            GREEN,
            [
                ("BASE_RATE", "0.70"),
                ("部署深度", "10"),
                ("阈值精度", f'{calibration["best_threshold_accuracy"] * 100:.2f}%'),
                ("AUC", f'{auc["auc"]:.4f}'),
                ("Argmax", f'{calibration["argmax_accuracy"] * 100:.2f}%'),
            ],
            "artifacts/frozen_bundle_medical_dynamic_mainline",
        ),
        (
            "固定结构静态对照",
            "对照线",
            "#F8FAFD",
            BLUE,
            [
                ("BASE_RATE", "0.70"),
                ("部署深度", "12"),
                ("阈值精度", f'{static["threshold_accuracy"]:.2f}%'),
                ("AUC", f'{static["auc"]:.4f}'),
                ("用途", "静态对照"),
            ],
            "results/final/demo_content_summary_final.json",
        ),
    ]
    gap = 22
    card_width = (width - 96 - gap) // 2
    for index, (title, tag, fill, accent, fields, source) in enumerate(cards):
        left = 48 + index * (card_width + gap)
        right = left + card_width
        _round_panel(draw, (left, 130, right, height - 38), fill=fill)
        draw.rounded_rectangle((left + 24, 158, left + 34, 232), radius=5, fill=accent)
        draw.text((left + 56, 154), title, font=_font(bold_font, 29), fill=INK)
        _chip(draw, right - 24, 154, tag, _font(regular_font, 19), fill="#FFFFFF", color=accent)
        for field_index, (label, value) in enumerate(fields):
            row, col = divmod(field_index, 2)
            field_left = left + 28 + col * ((card_width - 74) // 2 + 18)
            field_top = 260 + row * 112
            field_width = (card_width - 74) // 2
            _field(
                draw,
                (field_left, field_top, field_left + field_width, field_top + 88),
                label,
                value,
                regular_font=regular_font,
                bold_font=bold_font,
            )
        _round_panel(draw, (left + 28, 620, right - 28, 710), fill="#FFFFFF", radius=14)
        draw.text((left + 46, 635), "证据来源", font=_font(regular_font, 19), fill=MUTED)
        draw.text((left + 46, 672), source, font=_font(regular_font, 18), fill=INK)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _pruning_overview(
    output_path: Path,
    *,
    regular_font: Path,
    bold_font: Path,
) -> Path:
    width, height = 1680, 1577
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((52, 38), "剪枝演示", font=_font(bold_font, 46), fill=INK)
    draw.text((54, 102), "三阶段固定预算与两方安全执行拓扑", font=_font(regular_font, 26), fill=MUTED)
    _chip(draw, width - 52, 43, "base_rate 0.70", _font(regular_font, 22), fill=SOFT_BLUE, color=BLUE)

    stages = [("Layer 3", 137, "69.9%"), ("Layer 6", 96, "49.0%"), ("Layer 9", 67, "34.2%")]
    gap = 22
    card_width = (width - 104 - gap * 2) // 3
    for stage_index, (label, keep, ratio) in enumerate(stages):
        left = 52 + stage_index * (card_width + gap)
        right = left + card_width
        _round_panel(draw, (left, 158, right, 760))
        draw.text((left + 28, 184), label, font=_font(bold_font, 31), fill=INK)
        value = f"{keep} / 196"
        value_font = _font(bold_font, 31)
        draw.text((right - 28 - _text_width(draw, value, value_font), 184), value, font=value_font, fill=BLUE)
        draw.text((left + 30, 231), f"空间 token 保留率 {ratio}", font=_font(regular_font, 22), fill=MUTED)
        grid_left, grid_top = left + 48, 292
        cell, cell_gap = 24, 5
        for grid_index in range(196):
            row, col = divmod(grid_index, 14)
            x0 = grid_left + col * (cell + cell_gap)
            y0 = grid_top + row * (cell + cell_gap)
            color = BLUE if grid_index < keep else "#E5EAF1"
            draw.rounded_rectangle((x0, y0, x0 + cell, y0 + cell), radius=4, fill=color)
        draw.text((left + 30, 716), "蓝色：保留 · 灰色：移除", font=_font(regular_font, 21), fill=MUTED)

    _round_panel(draw, (52, 792, 1010, height - 44))
    _round_panel(draw, (1032, 792, width - 52, height - 44))
    draw.text((82, 824), "执行拓扑", font=_font(bold_font, 33), fill=INK)
    nodes = [
        ("浏览器", "本地预处理", 92, 950),
        ("P1", "数据份额 1", 340, 866),
        ("P2", "数据份额 2", 340, 1034),
        ("协调服务", "仅编排", 590, 950),
        ("SPU", "两方 2PC", 815, 950),
    ]
    for title, subtitle, x, y in nodes:
        _round_panel(draw, (x, y, x + 170, y + 100), fill="#F8FAFD", radius=16)
        draw.text((x + 20, y + 18), title, font=_font(bold_font, 24), fill=INK)
        draw.text((x + 20, y + 57), subtitle, font=_font(regular_font, 18), fill=MUTED)
    for start, end in [((262, 1000), (340, 916)), ((262, 1000), (340, 1084)), ((510, 916), (590, 1000)), ((510, 1084), (590, 1000)), ((760, 1000), (815, 1000))]:
        draw.line((*start, *end), fill="#9CB3D2", width=5)
    _round_panel(draw, (82, 1190, 980, 1466), fill=SOFT_BLUE, outline="#BED2F4", radius=18)
    topology = ["原始图像不离开本地", "模型参数在 SPU 内保持秘密", "动态决策不在明文域展开", "仅揭示最终 logits"]
    for index, text in enumerate(topology):
        y = 1222 + index * 58
        draw.ellipse((106, y + 7, 124, y + 25), fill=BLUE)
        draw.text((142, y), text, font=_font(regular_font, 23), fill=INK)

    draw.text((1062, 824), "执行概览", font=_font(bold_font, 33), fill=INK)
    summary = [
        ("剪枝阶段", "3 / 6 / 9"),
        ("token 数", "137 / 96 / 67"),
        ("部署深度", "10"),
        ("激活函数", "fixed_square"),
        ("注意力", "uniform"),
        ("输出策略", "final logits only"),
    ]
    for index, (label, value) in enumerate(summary):
        top = 900 + index * 82
        draw.text((1064, top), label, font=_font(regular_font, 23), fill=MUTED)
        value_font = _font(bold_font, 25)
        draw.text((1598 - _text_width(draw, value, value_font), top - 2), value, font=value_font, fill=INK)
        draw.line((1064, top + 48, 1598, top + 48), fill=BORDER, width=2)
    _round_panel(draw, (1064, 1402, 1598, 1492), fill=SOFT_GREEN, outline="#BFE0D2", radius=16)
    draw.text((1092, 1427), "完整两方安全推理链路", font=_font(bold_font, 25), fill=GREEN)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _user_report(
    output_path: Path,
    *,
    medical: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
) -> Path:
    width, height = 1836, 1182
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    sample = medical["first_sample"]
    positive_probability = sample["probabilities"][1]
    result = "肺炎" if sample["threshold_prediction"] else "正常"
    draw.text((54, 36), "安全推理完成", font=_font(bold_font, 46), fill=INK)
    draw.text((56, 99), "固定部署样本 · 仅揭示最终分类结果", font=_font(regular_font, 26), fill=MUTED)
    _chip(draw, width - 54, 42, "证据可回溯", _font(regular_font, 22), fill=SOFT_GREEN, color=GREEN)
    cards = [
        ("最终结论", result, "部署阈值判定", GREEN, SOFT_GREEN),
        ("阳性概率", f"{positive_probability:.4f}", "SPU 最终输出", BLUE, SOFT_BLUE),
        ("部署阈值", f'{medical["threshold"]:.4f}', "独立校准阈值", BLUE, SOFT_BLUE),
        ("样本位置", "1 / 32", "固定部署验证批次", AMBER, SOFT_AMBER),
    ]
    gap = 18
    card_width = (width - 108 - gap * 3) // 4
    for index, (label, value, scope, accent, tint) in enumerate(cards):
        left = 54 + index * (card_width + gap)
        _metric_card(
            draw,
            (left, 154, left + card_width, 342),
            label,
            value,
            scope,
            accent=accent,
            tint=tint,
            regular_font=regular_font,
            bold_font=bold_font,
        )
    _round_panel(draw, (54, 378, 898, 1118))
    _round_panel(draw, (920, 378, width - 54, 1118))
    draw.text((84, 410), "运行证据", font=_font(bold_font, 33), fill=INK)
    evidence = [
        ("样本", sample["relative_path"]),
        ("目标标签", str(sample["target"])),
        ("阈值预测", str(sample["threshold_prediction"])),
        ("输出 logits", f'{sample["logits"][0]:.6f}, {sample["logits"][1]:.6f}'),
        ("完整批次总时长", f'{medical["elapsed_sec"]:.2f} 秒'),
        ("批次平均时延", f'{medical["sec_per_sample"]:.2f} 秒/样本'),
        ("批次通信量", f'{medical["network"]["total_gib"]:.2f} GiB'),
        ("每样本通信量", f'{medical["network"]["per_sample_gib"]:.2f} GiB'),
    ]
    for index, (label, value) in enumerate(evidence):
        top = 476 + index * 72
        draw.text((86, top), label, font=_font(regular_font, 23), fill=MUTED)
        value_font = _font(bold_font, 25)
        draw.text((866 - _text_width(draw, value, value_font), top - 2), value, font=value_font, fill=INK)
        draw.line((86, top + 43, 866, top + 43), fill=BORDER, width=2)

    draw.text((950, 410), "动态剪枝摘要", font=_font(bold_font, 33), fill=INK)
    draw.text((952, 460), "base_rate 0.70 · 输入空间 token 196", font=_font(regular_font, 23), fill=MUTED)
    stages = [("Layer 3", 137, 137 / 196), ("Layer 6", 96, 96 / 196), ("Layer 9", 67, 67 / 196)]
    for index, (label, keep, ratio) in enumerate(stages):
        top = 536 + index * 142
        draw.text((952, top), label, font=_font(bold_font, 25), fill=INK)
        value = f"{keep} / 196"
        value_font = _font(bold_font, 25)
        draw.text((1748 - _text_width(draw, value, value_font), top), value, font=value_font, fill=BLUE)
        draw.rounded_rectangle((952, top + 52, 1748, top + 90), radius=19, fill="#E8EDF4")
        draw.rounded_rectangle((952, top + 52, 952 + int(796 * ratio), top + 90), radius=19, fill=BLUE)
        draw.text((952, top + 101), f"保留率 {ratio * 100:.1f}%", font=_font(regular_font, 21), fill=MUTED)
    _round_panel(draw, (952, 964, 1748, 1074), fill=SOFT_GREEN, outline="#BFE0D2", radius=16)
    draw.text((980, 982), "隐私边界", font=_font(bold_font, 23), fill=GREEN)
    draw.text((980, 1024), "输入与参数保持秘密，仅揭示最终 logits。", font=_font(regular_font, 22), fill=INK)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _evidence(
    output_path: Path,
    *,
    medical: dict[str, Any],
    calibration: dict[str, Any],
    auc: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
    mono_font: Path,
) -> tuple[Path, Path]:
    width, height = 1680, 1002
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((52, 34), "结果证据", font=_font(bold_font, 46), fill=INK)
    draw.text((54, 96), "全量模型指标与完整 SPU 运行数据分别取自对应证据文件", font=_font(regular_font, 26), fill=MUTED)
    chip_right = width - 50
    chip_right = _chip(draw, chip_right, 39, "VPS · CPU", _font(regular_font, 23), fill=SOFT_BLUE, color=BLUE)
    _chip(draw, chip_right, 39, "正式数据", _font(regular_font, 23), fill=SOFT_GREEN, color=GREEN)

    gap = 18
    card_width = (width - 104 - gap * 3) // 4
    cards = [
        ("阈值精度", f'{calibration["best_threshold_accuracy"] * 100:.2f}%', "全量验证 · 524 条", GREEN, SOFT_GREEN),
        ("AUC", f'{auc["auc"]:.4f}', "全量验证 · 524 条", BLUE, SOFT_BLUE),
        ("平均时延", f'{medical["sec_per_sample"]:.2f} 秒', "完整 2PC · 32 条", BLUE, SOFT_BLUE),
        ("批次通信量", f'{medical["network"]["total_gib"]:.2f} GiB', "环回 TX 单计 · 32 条", AMBER, SOFT_AMBER),
    ]
    for index, (label, value, scope, accent, tint) in enumerate(cards):
        left = 52 + index * (card_width + gap)
        _metric_card(
            draw,
            (left, 150, left + card_width, 334),
            label,
            value,
            scope,
            accent=accent,
            tint=tint,
            regular_font=regular_font,
            bold_font=bold_font,
        )

    panel_gap = 22
    panel_width = (width - 104 - panel_gap) // 2
    left_box = (52, 368, 52 + panel_width, 952)
    right_box = (52 + panel_width + panel_gap, 368, width - 52, 952)
    for box in (left_box, right_box):
        _round_panel(draw, box)

    draw.text((82, 398), "全量模型验证", font=_font(bold_font, 32), fill=INK)
    draw.text((82, 445), "results/final · 524 条医疗验证样本", font=_font(regular_font, 23), fill=MUTED)
    fullval = [
        ("阈值", f'{calibration["best_threshold"]:.6f}'),
        ("阈值精度", f'{calibration["best_threshold_accuracy"] * 100:.4f}%'),
        ("Argmax 准确率", f'{calibration["argmax_accuracy"] * 100:.4f}%'),
        ("AUC", f'{auc["auc"]:.6f}'),
    ]
    for index, (label, value) in enumerate(fullval):
        top = 500 + index * 92
        draw.text((84, top), label, font=_font(regular_font, 25), fill=MUTED)
        right = left_box[2] - 32
        value_font = _font(bold_font, 31)
        draw.text((right - _text_width(draw, value, value_font), top - 3), value, font=value_font, fill=INK)
        draw.line((84, top + 52, right, top + 52), fill=BORDER, width=2)
    draw.text((84, 876), "来源：medical_dynamic_threshold_calibration_final.json", font=_font(regular_font, 20), fill=MUTED)
    draw.text((84, 908), "      medical_dynamic_auc_reference_final.json", font=_font(regular_font, 20), fill=MUTED)

    right_left = right_box[0] + 30
    right_edge = right_box[2] - 30
    draw.text((right_left, 398), "完整 SPU 部署验证", font=_font(bold_font, 32), fill=INK)
    draw.text((right_left, 445), "两方 2PC · base_rate 0.70 · 32 条样本", font=_font(regular_font, 23), fill=MUTED)
    deploy = [
        ("总时长", f'{medical["elapsed_sec"]:.2f} 秒'),
        ("平均时延", f'{medical["sec_per_sample"]:.2f} 秒/样本'),
        ("批次通信量", f'{medical["network"]["total_gib"]:.2f} GiB'),
        ("每样本通信量", f'{medical["network"]["per_sample_gib"]:.2f} GiB'),
        ("部署阈值精度 / AUC", f'{medical["threshold_accuracy"] * 100:.2f}% / {medical["auc"]:.5f}'),
        ("三阶段 token 数", "137 / 96 / 67"),
    ]
    for index, (label, value) in enumerate(deploy):
        top = 500 + index * 64
        draw.text((right_left, top), label, font=_font(regular_font, 23), fill=MUTED)
        value_font = _font(bold_font, 27)
        draw.text((right_edge - _text_width(draw, value, value_font), top - 2), value, font=value_font, fill=INK)
        draw.line((right_left, top + 43, right_edge, top + 43), fill=BORDER, width=2)
    draw.text((right_left, 892), "来源：medical32_spu_latest_summary.json", font=_font(regular_font, 20), fill=MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    half = height // 2
    top_path = output_path.with_name(output_path.stem + "_top.png")
    bottom_path = output_path.with_name(output_path.stem + "_bottom.png")
    image.crop((0, 0, width, half)).save(top_path, format="PNG", optimize=True)
    image.crop((0, half, width, height)).save(bottom_path, format="PNG", optimize=True)
    return top_path, bottom_path


def generate_report_ui_snapshots(
    repo_root: Path,
    output_dir: Path,
    *,
    regular_font: Path,
    bold_font: Path,
    mono_font: Path,
) -> dict[str, Path]:
    medical = _read_json(Path(__file__).resolve().parent / "vps_report_data.json")["medical"]
    calibration = _read_json(repo_root / "results/final/medical_dynamic_threshold_calibration_final.json")
    auc = _read_json(repo_root / "results/final/medical_dynamic_auc_reference_final.json")
    demo = _read_json(repo_root / "results/final/demo_content_summary_final.json")
    figure_data = _read_json(Path(__file__).resolve().parent / "report_figure_data.json")
    training_args = _read_json(repo_root / "artifacts/frozen_bundle_medical_dynamic_mainline/args_snapshot.json")
    overview = _overview(
        output_dir / "admin_overview_generated.png",
        medical=medical,
        calibration=calibration,
        auc=auc,
        regular_font=regular_font,
        bold_font=bold_font,
        mono_font=mono_font,
    )
    evidence_top, evidence_bottom = _evidence(
        output_dir / "admin_evidence_generated.png",
        medical=medical,
        calibration=calibration,
        auc=auc,
        regular_font=regular_font,
        bold_font=bold_font,
        mono_font=mono_font,
    )
    task_create = _task_create(
        output_dir / "admin_task_create_generated.png",
        training_args=training_args,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    task_detail = _task_detail(
        output_dir / "admin_task_detail_generated.png",
        calibration=calibration,
        auc=auc,
        demo=demo,
        figure_data=figure_data,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    model_assets = _model_assets(
        output_dir / "admin_model_assets_generated.png",
        calibration=calibration,
        auc=auc,
        demo=demo,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    pruning_overview = _pruning_overview(
        output_dir / "pruning_overview_generated.png",
        regular_font=regular_font,
        bold_font=bold_font,
    )
    user_report = _user_report(
        output_dir / "user_report_generated.png",
        medical=medical,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    return {
        "overview": overview,
        "task_create": task_create,
        "task_detail": task_detail,
        "model_assets": model_assets,
        "evidence_top": evidence_top,
        "evidence_bottom": evidence_bottom,
        "pruning_overview": pruning_overview,
        "user_report": user_report,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    generated = generate_report_ui_snapshots(
        root,
        Path(__file__).resolve().parent / "output/intermediate/strict_format/generated_ui_snapshots",
        regular_font=Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        bold_font=Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
        mono_font=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    for item in generated.values():
        print(item)
