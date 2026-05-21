#!/usr/bin/env python3
"""Generate report figures from current report evidence."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "docs" / "report_evidence" / "assets"
DYNAMIC_PT = REPO_ROOT / "results" / "report_evidence" / "medical_dynamic_threshold_calibration" / "fullval_524_runtime_pruning_depth10_reference.pt"
DYNAMIC_JSON = REPO_ROOT / "results" / "report_evidence" / "medical_dynamic_threshold_calibration" / "fullval_524_runtime_pruning_depth10_reference.json"
DYNAMIC_IMAGE_LIST = REPO_ROOT / "results" / "report_evidence" / "medical_dynamic_threshold_calibration" / "fullval_524_image_list.txt"
CNN_CSV = REPO_ROOT / "artifacts" / "train_runs" / "cnn_plaintext_densenet121_20260519_145509" / "val_predictions.csv"
CNN_THRESHOLD_JSON = REPO_ROOT / "artifacts" / "train_runs" / "cnn_plaintext_densenet121_20260519_145509" / "best_threshold.json"
MEDICAL_THRESHOLD_JSON = REPO_ROOT / "results" / "report_evidence" / "medical_dynamic_threshold_calibration" / "dynamic_fullval_depth10_threshold_summary.json"

THRESHOLD_FIGURE_OUT = ASSET_DIR / "medical_threshold_calibration_shift.png"
TOPOLOGY_FIGURE_OUT = ASSET_DIR / "system_trust_boundary_topology.png"
SEQUENCE_FIGURE_OUT = ASSET_DIR / "software_flow_sequence.png"
ROBUSTNESS_FIGURE_OUT = ASSET_DIR / "robustness_guard_matrix.png"
USER_TOPOLOGY_FIGURE = REPO_ROOT / "2-1.png"
USER_SEQUENCE_FIGURE = REPO_ROOT / "2-2.png"
MANUAL_TOPOLOGY_FIGURE = ASSET_DIR / "curated_system_trust_boundary_topology.png"
MANUAL_SEQUENCE_FIGURE = ASSET_DIR / "curated_software_flow_sequence.png"
MANUAL_ROBUSTNESS_FIGURE = ASSET_DIR / "curated_robustness_guard_matrix.png"

FUZZ_GLOB = "protocol_fuzz*.json"
GUARD_GLOB = "control_plane_guard*.json"

CARD_BG = (255, 255, 255, 255)
CARD_BORDER = (214, 221, 230, 255)
TEXT = (21, 27, 38, 255)
MUTED = (104, 114, 128, 255)
PLAIN = (27, 135, 84, 255)
PLAIN_BG = (235, 248, 240, 255)
SECRET = (198, 48, 65, 255)
SECRET_BG = (254, 241, 243, 255)
BOUNDARY = (120, 129, 146, 255)
INFO = (44, 109, 194, 255)
WARN = (181, 119, 14, 255)


def _copy_manual_figure_if_present(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    shutil.copyfile(source, target)
    return True


def _accuracy(pred, target) -> float:
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    return float((pred_arr == target_arr).mean())


def _load_font(size: int, *, bold: bool = False):
    candidates = []
    try:
        matched = subprocess.run(
            ["fc-match", ":lang=zh", "file"],
            check=False,
            capture_output=True,
            text=True,
        )
        first_line = (matched.stdout or "").strip().splitlines()
        if first_line:
            maybe_path = first_line[0].strip()
            if maybe_path:
                candidates.append(maybe_path)
    except Exception:
        pass
    candidates.extend(
        [
            "/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
            "/mnt/c/Windows/Fonts/simhei.ttf" if bold else "/mnt/c/Windows/Fonts/simsun.ttc",
            "/mnt/c/Windows/Fonts/Dengb.ttf" if bold else "/mnt/c/Windows/Fonts/Deng.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=TEXT, *, anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _draw_rotated_text(image: Image.Image, xy, text: str, font, fill=TEXT, *, angle=90):
    dummy = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_image = Image.new("RGBA", (text_w + 8, text_h + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, dest=(int(xy[0]), int(xy[1])))


def _multiline(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=TEXT, *, spacing=6):
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return "\n".join(lines)


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, max_width: int, fill=TEXT, *, spacing=6):
    wrapped = _wrap_text(draw, text, font, max_width)
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)
    return wrapped


def _rounded(draw: ImageDraw.ImageDraw, box, *, fill, outline=CARD_BORDER, radius=20, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(draw: ImageDraw.ImageDraw, start, end, *, color, width=5, dashed=False):
    x0, y0 = start
    x1, y1 = end
    if dashed:
        steps = max(1, int((((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) // 14))
        for idx in range(0, steps, 2):
            sx = x0 + (x1 - x0) * idx / steps
            sy = y0 + (y1 - y0) * idx / steps
            ex = x0 + (x1 - x0) * min(idx + 1, steps) / steps
            ey = y0 + (y1 - y0) * min(idx + 1, steps) / steps
            draw.line((sx, sy, ex, ey), fill=color, width=width)
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle_x = x1 - x0
    angle_y = y1 - y0
    norm = max((angle_x**2 + angle_y**2) ** 0.5, 1.0)
    ux, uy = angle_x / norm, angle_y / norm
    px, py = -uy, ux
    head_len = 16
    head_half = 8
    p1 = (x1, y1)
    p2 = (x1 - head_len * ux + head_half * px, y1 - head_len * uy + head_half * py)
    p3 = (x1 - head_len * ux - head_half * px, y1 - head_len * uy - head_half * py)
    draw.polygon((p1, p2, p3), fill=color)


def _lock_icon(draw: ImageDraw.ImageDraw, x: int, y: int, *, color=SECRET):
    draw.rounded_rectangle((x, y + 10, x + 22, y + 28), radius=4, fill=color)
    draw.arc((x + 4, y, x + 18, y + 18), 180, 360, fill=color, width=3)


def _domain_header(draw: ImageDraw.ImageDraw, box, title: str, subtitle: str, fonts, *, fill, subtitle_fill):
    x0, y0, x1, y1 = box
    _rounded(draw, box, fill=(255, 255, 255, 0), outline=(0, 0, 0, 0), radius=26, width=0)
    draw.rounded_rectangle((x0, y0, x1, y0 + 58), radius=22, fill=fill)
    _draw_text(draw, (x0 + 26, y0 + 12), title, fonts["title"], (255, 255, 255, 255))
    subtitle_box = (x0 + 20, y0 + 76, x1 - 20, y0 + 116)
    draw.rounded_rectangle(subtitle_box, radius=18, fill=subtitle_fill, outline=(0, 0, 0, 0))
    _draw_text(draw, (subtitle_box[0] + 18, subtitle_box[1] + 7), subtitle, fonts["small"], MUTED)


def _card_with_body(draw: ImageDraw.ImageDraw, box, title: str, body: str, fonts, *, fill=CARD_BG, outline=CARD_BORDER, title_fill=TEXT, body_fill=MUTED):
    _rounded(draw, box, fill=fill, outline=outline, radius=22, width=2)
    x0, y0, x1, y1 = box
    _draw_text(draw, (x0 + 20, y0 + 16), title, fonts["card_title"], title_fill)
    _draw_wrapped(draw, (x0 + 20, y0 + 56), body, fonts["body"], max_width=int(x1 - x0 - 40), fill=body_fill, spacing=8)


def _small_tag(draw: ImageDraw.ImageDraw, box, text: str, fonts, *, fill, outline, text_fill):
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=2)
    _draw_text(draw, (box[0] + 14, box[1] + 7), text, fonts["small"], text_fill)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, fonts):
    draw.line((x, y, x + 54, y), fill=PLAIN, width=5)
    _draw_text(draw, (x + 66, y - 12), "受信任域内明文处理", fonts["small"])
    _arrow(draw, (x, y + 44), (x + 54, y + 44), color=SECRET, width=5, dashed=True)
    _lock_icon(draw, x + 16, y + 28)
    _draw_text(draw, (x + 66, y + 32), "跨边界密态 share / 张量通信", fonts["small"])
    draw.line((x + 8, y + 86, x + 8, y + 128), fill=BOUNDARY, width=3)
    _draw_text(draw, (x + 24, y + 100), "信任边界 / DMZ", fonts["small"])


def _background(size):
    image = Image.new("RGBA", size, (246, 248, 252, 255))
    overlay = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = size
    for idx in range(0, width + height, 36):
        draw.line((idx, 0, 0, idx), fill=(255, 255, 255, 18), width=1)
    image.alpha_composite(overlay)
    return image


def _load_cnn_predictions(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return {
        "prob_positive": np.asarray([float(row["prob_positive"]) for row in rows], dtype=float),
        "label": np.asarray([int(row["label"]) for row in rows], dtype=int),
        "pred_argmax": np.asarray([int(row["pred_argmax"]) for row in rows], dtype=int),
        "pred_threshold": np.asarray([int(row["pred_threshold"]) for row in rows], dtype=int),
    }


def _load_dynamic_predictions(summary_json: Path, image_list: Path):
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    preview = payload["prediction_preview"]
    lines = [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = np.asarray([int(Path(line).parent.name) for line in lines], dtype=int)
    prob_positive = np.asarray([float(row[1]) for row in preview["probabilities"]], dtype=float)
    pred_argmax = np.asarray(preview["argmax_predictions"], dtype=int)
    pred_threshold = np.asarray(preview["threshold_predictions"], dtype=int)
    if not (len(labels) == len(prob_positive) == len(pred_argmax) == len(pred_threshold)):
        raise ValueError("dynamic prediction arrays do not align with image list length")
    return {
        "prob_positive": prob_positive,
        "label": labels,
        "pred_argmax": pred_argmax,
        "pred_threshold": pred_threshold,
    }


def _draw_hist_panel(image, draw, origin, size, probs, labels, threshold, title, argmax_acc, threshold_acc, fonts):
    panel_x, panel_y = origin
    panel_w, panel_h = size
    left = panel_x + 54
    right = panel_x + panel_w - 24
    top = panel_y + 96
    bottom = panel_y + panel_h - 62
    bins = np.linspace(0.0, 1.0, 25)
    hist_neg, _ = np.histogram(probs[labels == 0], bins=bins)
    hist_pos, _ = np.histogram(probs[labels == 1], bins=bins)
    max_count = max(int(hist_neg.max()), int(hist_pos.max()), 1)
    bin_count = len(bins) - 1
    slot_w = (right - left) / bin_count

    _rounded(draw, (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), fill=CARD_BG)
    _draw_text(draw, (panel_x + 18, panel_y + 14), title, fonts["title"])
    threshold_header_y = panel_y + 50
    draw.line((panel_x + 18, threshold_header_y + 7, panel_x + 44, threshold_header_y + 7), fill=(90, 90, 90), width=2)
    _draw_text(draw, (panel_x + 52, threshold_header_y - 4), "默认阈值 0.500", fonts["small"], MUTED)
    green_x = panel_x + 230
    draw.line((green_x, threshold_header_y + 7, green_x + 26, threshold_header_y + 7), fill=PLAIN, width=3)
    _draw_text(draw, (green_x + 34, threshold_header_y - 4), f"校准阈值 {threshold:.3f}", fonts["small"], PLAIN)
    metric_box = (panel_x + panel_w - 208, panel_y + 12, panel_x + panel_w - 18, panel_y + 94)
    draw.rounded_rectangle(metric_box, radius=14, fill=(248, 250, 253, 255), outline=(228, 232, 238, 255), width=1)
    metrics_box = (
        f"argmax：{argmax_acc * 100:.2f}%\n"
        f"阈值：{threshold_acc * 100:.2f}%\n"
        f"最优阈值：{threshold:.3f}"
    )
    _multiline(draw, (metric_box[0] + 12, metric_box[1] + 10), metrics_box, fonts["body"], fill=(50, 50, 50), spacing=4)

    for grid_idx in range(6):
        y = bottom - (bottom - top) * grid_idx / 5
        draw.line((left, y, right, y), fill=(232, 236, 240), width=1)
        _draw_text(draw, (panel_x + 10, y - 8), f"{int(max_count * grid_idx / 5)}", fonts["small"], MUTED)

    draw.line((left, top, left, bottom), fill=(80, 80, 80), width=2)
    draw.line((left, bottom, right, bottom), fill=(80, 80, 80), width=2)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for idx in range(bin_count):
        x0 = left + idx * slot_w + 2
        x_mid = x0 + slot_w / 2
        x1 = left + (idx + 1) * slot_w - 2
        neg_h = 0 if max_count == 0 else (hist_neg[idx] / max_count) * (bottom - top)
        pos_h = 0 if max_count == 0 else (hist_pos[idx] / max_count) * (bottom - top)
        overlay_draw.rectangle((x0, bottom - neg_h, x_mid, bottom), fill=(76, 120, 168, 155))
        overlay_draw.rectangle((x_mid, bottom - pos_h, x1, bottom), fill=(228, 87, 86, 155))
    image.alpha_composite(overlay)

    for tick in np.linspace(0.0, 1.0, 6):
        x = left + (right - left) * tick
        draw.line((x, bottom, x, bottom + 6), fill=(80, 80, 80), width=1)
        label = f"{tick:.1f}"
        text_w = draw.textlength(label, font=fonts["small"])
        _draw_text(draw, (x - text_w / 2, bottom + 12), label, fonts["small"], MUTED)

    threshold_items = [
        {"x_value": 0.5, "color": (90, 90, 90), "dashed": True},
        {"x_value": threshold, "color": PLAIN, "dashed": False},
    ]
    for item in threshold_items:
        x_value = item["x_value"]
        color = item["color"]
        dashed = item["dashed"]
        x = left + (right - left) * x_value
        if dashed:
            for offset in range(int(top), int(bottom), 10):
                draw.line((x, offset, x, min(offset + 5, bottom)), fill=color, width=2)
        else:
            draw.line((x, top, x, bottom), fill=color, width=2)

    x_axis_label = "横轴：阳性类别概率"
    y_axis_label = "纵轴：样本数"
    x_axis_label_w = draw.textlength(x_axis_label, font=fonts["small"])
    _draw_text(draw, (right - x_axis_label_w, bottom + 34), x_axis_label, fonts["small"], MUTED)
    _draw_text(draw, (panel_x + 18, top - 26), y_axis_label, fonts["small"], MUTED)
    legend_y = panel_y + panel_h - 26
    legend_pad_x = 10
    legend_gap = 18
    swatch_w = 16
    label1 = "阴性样本"
    label2 = "阳性样本"
    label1_w = draw.textlength(label1, font=fonts["small"])
    label2_w = draw.textlength(label2, font=fonts["small"])
    blue_width = int(legend_pad_x * 3 + swatch_w + label1_w)
    red_width = int(legend_pad_x * 3 + swatch_w + label2_w)
    blue_card = (panel_x + 16, legend_y - 7, panel_x + 16 + blue_width, legend_y + 20)
    red_card = (blue_card[2] + legend_gap, legend_y - 7, blue_card[2] + legend_gap + red_width, legend_y + 20)
    draw.rounded_rectangle(blue_card, radius=10, fill=(248, 250, 253, 255), outline=(228, 232, 238, 255), width=1)
    draw.rounded_rectangle(red_card, radius=10, fill=(248, 250, 253, 255), outline=(228, 232, 238, 255), width=1)
    draw.rectangle((blue_card[0] + 10, legend_y, blue_card[0] + 26, legend_y + 12), fill=(76, 120, 168))
    _draw_text(draw, (blue_card[0] + 34, legend_y - 3), label1, fonts["small"], (70, 70, 70))
    draw.rectangle((red_card[0] + 10, legend_y, red_card[0] + 26, legend_y + 12), fill=(228, 87, 86))
    _draw_text(draw, (red_card[0] + 34, legend_y - 3), label2, fonts["small"], (70, 70, 70))


def generate_threshold_figure():
    dynamic_payload = _load_dynamic_predictions(DYNAMIC_JSON, DYNAMIC_IMAGE_LIST)
    dynamic_probs = dynamic_payload["prob_positive"]
    dynamic_labels = dynamic_payload["label"]
    dynamic_argmax = dynamic_payload["pred_argmax"]
    dynamic_threshold_pred = dynamic_payload["pred_threshold"]
    dynamic_threshold = float(json.loads(MEDICAL_THRESHOLD_JSON.read_text(encoding="utf-8"))["best_threshold"])

    cnn_payload = _load_cnn_predictions(CNN_CSV)
    cnn_threshold = float(json.loads(CNN_THRESHOLD_JSON.read_text(encoding="utf-8"))["best_threshold"])
    cnn_probs = cnn_payload["prob_positive"]
    cnn_labels = cnn_payload["label"]
    cnn_argmax = cnn_payload["pred_argmax"]
    cnn_threshold_pred = cnn_payload["pred_threshold"]

    image = _background((1600, 860))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": _load_font(30, bold=True),
        "title": _load_font(24, bold=True),
        "body": _load_font(18),
        "small": _load_font(16),
    }
    _draw_text(draw, (120, 40), "医疗任务概率分布与判类阈值偏移示意", fonts["suptitle"])
    _draw_text(
        draw,
        (120, 88),
        "左图为医疗动态安全剪枝路径，右图为 DenseNet121 明文基线；默认阈值与部署校准阈值同时标出。",
        fonts["body"],
        MUTED,
    )
    _draw_hist_panel(
        image,
        draw,
        (90, 150),
        (680, 610),
        dynamic_probs,
        dynamic_labels,
        dynamic_threshold,
        "医疗动态安全剪枝路径",
        _accuracy(dynamic_argmax, dynamic_labels),
        _accuracy(dynamic_threshold_pred, dynamic_labels),
        fonts,
    )
    _draw_hist_panel(
        image,
        draw,
        (830, 150),
        (680, 610),
        cnn_probs,
        cnn_labels,
        cnn_threshold,
        "DenseNet121 明文基线",
        _accuracy(cnn_argmax, cnn_labels),
        _accuracy(cnn_threshold_pred, cnn_labels),
        fonts,
    )
    image.convert("RGB").save(THRESHOLD_FIGURE_OUT, quality=95)


def generate_topology_figure():
    if _copy_manual_figure_if_present(USER_TOPOLOGY_FIGURE, TOPOLOGY_FIGURE_OUT):
        return
    image = _background((1800, 1080))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": _load_font(33, bold=True),
        "title": _load_font(24, bold=True),
        "card_title": _load_font(20, bold=True),
        "body": _load_font(17),
        "small": _load_font(14),
    }
    _draw_text(draw, (88, 40), "图2-1 系统物理与逻辑部署图", fonts["suptitle"])
    _draw_text(draw, (88, 88), "围绕“明文止于本地、跨域只传 share、SPU 前仍有权威快检”三条主线重绘。", fonts["body"], MUTED)

    domains = [
        ((80, 150, 620, 930), "本地明文处理域", "医院终端 + 浏览器主线程 + 单实例 Worker", (19, 124, 62, 255), (239, 248, 242, 255)),
        ((670, 150, 1100, 930), "WAN / DMZ 传输边界", "跨域只承载 multipart、share 与审计摘要", (84, 94, 115, 255), (244, 246, 249, 255)),
        ((1150, 150, 1720, 930), "两方安全执行域", "网关、权威快检、SPU 与审计回显", (179, 49, 49, 255), (255, 244, 245, 255)),
    ]
    for box, title, subtitle, header_fill, subtitle_fill in domains:
        _rounded(draw, box, fill=(255, 255, 255, 190), outline=(223, 228, 234, 255), radius=28)
        _domain_header(draw, (box[0] + 18, box[1] + 14, box[2] - 18, box[1] + 130), title, subtitle, fonts, fill=header_fill, subtitle_fill=subtitle_fill)

    for x in (645, 1125):
        draw.line((x, 152, x, 930), fill=BOUNDARY, width=3)
        for offset in range(152, 930, 18):
            draw.line((x, offset, x, min(offset + 8, 930)), fill=(255, 255, 255, 0), width=0)
        _lock_icon(draw, x - 11, 176, color=BOUNDARY)
        _draw_text(draw, (x - 24, 214), "信任边界", fonts["small"], BOUNDARY)

    _card_with_body(
        draw,
        (110, 320, 285, 560),
        "医院终端",
        "评委或医护人员在本机浏览器中加载本地影像样本；正式演示链路仅向服务端提交分片数据与审计元信息。",
        fonts,
        fill=CARD_BG,
    )
    _card_with_body(
        draw,
        (320, 320, 585, 500),
        "浏览器主线程",
        "负责文件句柄、页面状态、按钮编排与单活 Worker 生命周期管理。",
        fonts,
        fill=CARD_BG,
    )
    _card_with_body(
        draw,
        (320, 530, 585, 790),
        "浏览器 Worker",
        "完成头部嗅探、解码与中心裁剪、DQA 摘要、哈希链构建，以及小端 Float32 share 序列化。",
        fonts,
        fill=PLAIN_BG,
        outline=(159, 209, 179, 255),
        body_fill=(37, 84, 62, 255),
    )
    _small_tag(draw, (112, 820, 595, 875), "明文终止点：原图、像素张量与质量预检都停留在本地 Worker 域内", fonts, fill=(240, 250, 243, 255), outline=(159, 209, 179, 255), text_fill=PLAIN)

    _card_with_body(
        draw,
        (720, 360, 1050, 545),
        "跨域报文",
        "仅提交 multipart/form-data：share0、share1、质量摘要、审计摘要与控制面指标。",
        fonts,
        fill=CARD_BG,
    )
    _small_tag(draw, (742, 575, 1030, 625), "share 生成起点", fonts, fill=(255, 241, 243, 255), outline=(233, 176, 184, 255), text_fill=SECRET)
    _card_with_body(
        draw,
        (720, 670, 1050, 840),
        "边界语义",
        "TLS / HTTP 只承载密态分片与审计摘要；不承载原图，也不承载完整模型参数。",
        fonts,
        fill=(249, 250, 252, 255),
    )

    _card_with_body(
        draw,
        (1200, 300, 1670, 430),
        "业务侧前置网关",
        "完成 body 限长、boundary 解析、header 预检、multipart 结构校验与路由转发。",
        fonts,
        fill=CARD_BG,
    )
    _card_with_body(
        draw,
        (1200, 470, 1670, 640),
        "服务端权威快检",
        "完成 JSON 字节门、share 哈希与审计链校验、内存对齐复制、NaN/Inf/次正规数阻断，以及服务端 DQA 复核。",
        fonts,
        fill=SECRET_BG,
        outline=(233, 176, 184, 255),
        body_fill=(105, 35, 44, 255),
    )
    _small_tag(draw, (1218, 658, 1458, 705), "最后一道预 SPU 权威门", fonts, fill=(255, 248, 236, 255), outline=(244, 201, 123, 255), text_fill=WARN)
    _card_with_body(
        draw,
        (1200, 735, 1415, 860),
        "P1 / P2 / SPU",
        "两方安全执行路径内部完成动态安全剪枝、前向推理与最终分类。",
        fonts,
        fill=CARD_BG,
    )
    _card_with_body(
        draw,
        (1455, 735, 1670, 860),
        "审计与回显",
        "记录载荷指纹、质量裁决与控制面指标，并仅回传分类、审计与质量摘要。",
        fonts,
        fill=CARD_BG,
    )

    _arrow(draw, (285, 440), (320, 410), color=PLAIN, width=5)
    _arrow(draw, (452, 500), (452, 530), color=PLAIN, width=5)
    _arrow(draw, (585, 655), (720, 452), color=SECRET, width=6, dashed=True)
    _lock_icon(draw, 645, 540)
    _arrow(draw, (1050, 452), (1200, 365), color=SECRET, width=6, dashed=True)
    _lock_icon(draw, 1115, 392)
    _arrow(draw, (1435, 430), (1435, 470), color=PLAIN, width=5)
    _arrow(draw, (1435, 640), (1305, 735), color=PLAIN, width=5)
    _arrow(draw, (1415, 795), (1455, 795), color=PLAIN, width=5)

    _rounded(draw, (105, 940, 1695, 1030), fill=(255, 255, 255, 220), outline=CARD_BORDER, radius=24)
    _legend(draw, 135, 985, fonts)
    _small_tag(draw, (1060, 958, 1670, 1012), "对外暴露的唯一业务结果：最终分类标签、质量裁决、审计摘要与控制面开销", fonts, fill=(245, 248, 255, 255), outline=(193, 211, 250, 255), text_fill=INFO)

    image.convert("RGB").save(TOPOLOGY_FIGURE_OUT, quality=95)


def generate_sequence_figure():
    if _copy_manual_figure_if_present(USER_SEQUENCE_FIGURE, SEQUENCE_FIGURE_OUT):
        return
    image = _background((1820, 1240))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": _load_font(33, bold=True),
        "title": _load_font(22, bold=True),
        "card_title": _load_font(18, bold=True),
        "body": _load_font(16),
        "small": _load_font(14),
    }
    _draw_text(draw, (88, 40), "图2-2 端到端软件流转时序图", fonts["suptitle"])
    _draw_text(draw, (88, 88), "流程与第三章 3.2 小节保持一致：八步闭环、两类边界、一个预 SPU 权威门。", fonts["body"], MUTED)

    lane_specs = [
        ("用户 / 主线程", 80, (239, 244, 255, 255), (193, 211, 250, 255)),
        ("浏览器 Worker", 370, (239, 248, 242, 255), (159, 209, 179, 255)),
        ("HTTP / multipart", 660, (255, 249, 236, 255), (244, 201, 123, 255)),
        ("服务端快检", 950, (255, 244, 245, 255), (233, 176, 184, 255)),
        ("SPU / 安全执行", 1240, (245, 242, 255, 255), (199, 179, 230, 255)),
        ("审计与回显", 1530, (239, 244, 255, 255), (193, 211, 250, 255)),
    ]
    top_y = 150
    bottom_y = 1030
    lane_w = 240
    for title, x, fill, outline in lane_specs:
        _rounded(draw, (x, top_y, x + lane_w, bottom_y), fill=(255, 255, 255, 195), outline=(226, 231, 237, 255), radius=26)
        draw.rounded_rectangle((x + 16, top_y + 16, x + lane_w - 16, top_y + 64), radius=18, fill=fill, outline=outline, width=2)
        _draw_text(draw, (x + 30, top_y + 27), title, fonts["title"])
        draw.line((x + lane_w / 2, top_y + 86, x + lane_w / 2, bottom_y - 30), fill=(224, 229, 235, 255), width=2)

    step_y = [245, 345, 445, 555, 665, 775, 885, 980]
    for idx, y in enumerate(step_y, start=1):
        draw.ellipse((35, y - 22, 79, y + 22), fill=(248, 250, 253, 255), outline=(193, 211, 250, 255), width=2)
        _draw_text(draw, (50, y - 11), str(idx), fonts["card_title"], INFO)

    def lane_center(index: int):
        x = lane_specs[index][1]
        return int(x + lane_w / 2)

    def step_card(lane_idx: int, y: int, title: str, body: str, *, fill=CARD_BG, outline=CARD_BORDER, title_fill=TEXT, body_fill=MUTED):
        cx = lane_center(lane_idx)
        box = (cx - 100, y - 42, cx + 100, y + 42)
        _rounded(draw, box, fill=fill, outline=outline, radius=18, width=2)
        _draw_text(draw, (box[0] + 14, box[1] + 10), title, fonts["body"], title_fill)
        _draw_wrapped(draw, (box[0] + 14, box[1] + 34), body, fonts["small"], max_width=176, fill=body_fill, spacing=5)
        return box

    boxes = {}
    boxes["s1a"] = step_card(0, step_y[0], "加载本地影像", "主线程保留文件句柄与页面状态。")
    boxes["s1b"] = step_card(1, step_y[0], "接收句柄", "确认单活 Worker，准备本地计算。", fill=PLAIN_BG, outline=(159, 209, 179, 255), body_fill=(37, 84, 62, 255))

    boxes["s2"] = step_card(1, step_y[1], "头部尺寸嗅探", "解析 PNG/JPEG/WebP 头部，先拦截像素炸弹。", fill=PLAIN_BG, outline=(159, 209, 179, 255), body_fill=(37, 84, 62, 255))
    boxes["s3"] = step_card(1, step_y[2], "解码与预检", "中心裁剪、归一化与 DQA 指标提取。", fill=PLAIN_BG, outline=(159, 209, 179, 255), body_fill=(37, 84, 62, 255))
    boxes["s4a"] = step_card(1, step_y[3], "share 与哈希", "生成两份 share、一次性标识与审计链。", fill=PLAIN_BG, outline=(159, 209, 179, 255), body_fill=(37, 84, 62, 255))
    boxes["s4b"] = step_card(2, step_y[3], "multipart 提交", "跨域仅发送 share 与结构化摘要。", fill=(255, 249, 236, 255), outline=(244, 201, 123, 255), body_fill=(118, 87, 22, 255))

    boxes["s5"] = step_card(3, step_y[4], "协议预检", "长度门、boundary、字段集与 JSON 字节门。", fill=SECRET_BG, outline=(233, 176, 184, 255), body_fill=(105, 35, 44, 255))
    boxes["s6a"] = step_card(3, step_y[5], "张量快检", "share 哈希、NaN/Inf、次正规数与 DQA 复核。", fill=SECRET_BG, outline=(233, 176, 184, 255), body_fill=(105, 35, 44, 255))
    boxes["s6b"] = step_card(4, step_y[5], "进入 SPU", "只有通过权威快检后才触发 2PC 前向。", fill=(245, 242, 255, 255), outline=(199, 179, 230, 255), body_fill=(78, 46, 124, 255))

    boxes["s7a"] = step_card(4, step_y[6], "密态前向", "动态安全剪枝与分类决策在安全图内部完成。", fill=(245, 242, 255, 255), outline=(199, 179, 230, 255), body_fill=(78, 46, 124, 255))
    boxes["s7b"] = step_card(5, step_y[6], "审计落盘", "记录载荷指纹、质量裁决与控制面指标。", fill=(239, 244, 255, 255), outline=(193, 211, 250, 255), body_fill=(44, 109, 194, 255))

    boxes["s8a"] = step_card(5, step_y[7], "结果回显", "仅回传分类结论、质量裁决与审计摘要。", fill=(239, 244, 255, 255), outline=(193, 211, 250, 255), body_fill=(44, 109, 194, 255))
    boxes["s8b"] = step_card(0, step_y[7], "页面展示", "主线程仅展示分类、质量与审计结果。")

    def connect(box_a, box_b, *, color, dashed=False):
        xa = box_a[2]
        xb = box_b[0]
        ya = (box_a[1] + box_a[3]) / 2
        yb = (box_b[1] + box_b[3]) / 2
        _arrow(draw, (xa, ya), (xb, yb), color=color, width=5, dashed=dashed)

    connect(boxes["s1a"], boxes["s1b"], color=PLAIN)
    _arrow(draw, (lane_center(1), step_y[0] + 42), (lane_center(1), step_y[1] - 42), color=PLAIN, width=4)
    _arrow(draw, (lane_center(1), step_y[1] + 42), (lane_center(1), step_y[2] - 42), color=PLAIN, width=4)
    _arrow(draw, (lane_center(1), step_y[2] + 42), (lane_center(1), step_y[3] - 42), color=PLAIN, width=4)
    connect(boxes["s4a"], boxes["s4b"], color=SECRET, dashed=True)
    connect(boxes["s4b"], boxes["s5"], color=SECRET, dashed=True)
    _arrow(draw, (lane_center(3), step_y[4] + 42), (lane_center(3), step_y[5] - 42), color=PLAIN, width=4)
    connect(boxes["s6a"], boxes["s6b"], color=PLAIN)
    _arrow(draw, (lane_center(4), step_y[5] + 42), (lane_center(4), step_y[6] - 42), color=PLAIN, width=4)
    connect(boxes["s7a"], boxes["s7b"], color=PLAIN)
    _arrow(draw, (lane_center(5), step_y[6] + 42), (lane_center(5), step_y[7] - 42), color=PLAIN, width=4)
    _arrow(draw, (boxes["s8a"][0], (boxes["s8a"][1] + boxes["s8a"][3]) / 2), (boxes["s8b"][2], (boxes["s8b"][1] + boxes["s8b"][3]) / 2), color=INFO, width=5, dashed=False)

    _small_tag(draw, (350, 1080, 670, 1130), "明文止于 Worker 域内", fonts, fill=(240, 250, 243, 255), outline=(159, 209, 179, 255), text_fill=PLAIN)
    _small_tag(draw, (690, 1080, 1035, 1130), "跨域仅传 share + 审计摘要", fonts, fill=(255, 241, 243, 255), outline=(233, 176, 184, 255), text_fill=SECRET)
    _small_tag(draw, (1055, 1080, 1410, 1130), "SPU 前仍有权威快检", fonts, fill=(255, 248, 236, 255), outline=(244, 201, 123, 255), text_fill=WARN)
    _small_tag(draw, (1430, 1080, 1740, 1130), "最终只回传分类/质量/审计", fonts, fill=(245, 248, 255, 255), outline=(193, 211, 250, 255), text_fill=INFO)
    _legend(draw, 118, 1180, fonts)

    image.convert("RGB").save(SEQUENCE_FIGURE_OUT, quality=95)


def _latest_json(pattern: str) -> Path | None:
    matches = sorted((REPO_ROOT / "results" / "report_evidence").glob(pattern))
    return matches[-1] if matches else None


def _primary_result_json(kind: str) -> Path | None:
    result_dir = REPO_ROOT / "results" / "report_evidence"
    exact_map = {
        "protocol_fuzz": result_dir / "protocol_fuzz_evidence.json",
        "control_plane_guard": result_dir / "control_plane_guard_evidence.json",
    }
    exact = exact_map.get(kind)
    if exact and exact.exists():
        return exact
    pattern_map = {
        "protocol_fuzz": ["protocol_fuzz_batch_*.json", "web_demo_protocol_fuzz_*.json"],
        "control_plane_guard": ["control_plane_guard_*.json", "web_demo_guard_stress_*.json"],
    }
    for pattern in pattern_map.get(kind, []):
        matches = sorted(path for path in result_dir.glob(pattern) if "_manual" not in path.stem)
        if matches:
            return matches[-1]
    return None


def _load_robustness_rows():
    rows = []
    fuzz_path = _primary_result_json("protocol_fuzz")
    if fuzz_path and fuzz_path.exists():
        payload = json.loads(fuzz_path.read_text(encoding="utf-8"))
        for item in payload.get("results", []):
            rows.append(
                {
                    "name": item.get("name", ""),
                    "status": "PASS" if item.get("passed") else "FAIL",
                    "layer": item.get("interception_layer", ""),
                    "fallback": item.get("fallback_layer", ""),
                    "system_state": (item.get("system_state") or {}).get("summary", "n/a"),
                }
            )
    guard_path = _primary_result_json("control_plane_guard")
    if guard_path and guard_path.exists():
        payload = json.loads(guard_path.read_text(encoding="utf-8"))
        for item in payload.get("checks", []):
            rows.append(
                {
                    "name": item.get("name", ""),
                    "status": "PASS" if item.get("passed") else "FAIL",
                    "layer": item.get("interception_layer", ""),
                    "fallback": item.get("fallback_layer", ""),
                    "system_state": (item.get("system_state") or {}).get("summary", "n/a"),
                }
            )
    return rows


def generate_robustness_figure():
    if _copy_manual_figure_if_present(MANUAL_ROBUSTNESS_FIGURE, ROBUSTNESS_FIGURE_OUT):
        return
    rows = _load_robustness_rows()
    image_height = max(1080, 330 + len(rows) * 48 + 120)
    image = _background((1960, image_height))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": _load_font(34, bold=True),
        "title": _load_font(19, bold=True),
        "body": _load_font(15),
        "small": _load_font(14),
    }
    _draw_text(draw, (90, 42), "图3-3 协议层怪癖与控制面兜底验证矩阵", fonts["suptitle"])
    _draw_text(draw, (90, 92), "矩阵同时记录首个拦截层级与进程资源回落情况；不使用“拦截率 100%”这类失真表述。", fonts["body"], MUTED)
    _rounded(draw, (82, 150, 1878, image_height - 96), fill=CARD_BG, radius=28)

    columns = [
        ("异常场景", 120, 300),
        ("首个拦截层", 450, 310),
        ("兜底层", 790, 240),
        ("结果", 1060, 120),
        ("资源状态摘要", 1210, 620),
    ]
    for title, x, _ in columns:
        _draw_text(draw, (x, 182), title, fonts["title"])
    draw.line((110, 225, 1848, 225), fill=CARD_BORDER, width=3)

    if not rows:
        _draw_text(draw, (118, 270), "尚未找到已落盘的 fuzz / guard 结果 JSON，请先运行验证脚本。", fonts["body"], WARN)
    else:
        row_y = 250
        for index, row in enumerate(rows, start=1):
            if index % 2 == 0:
                draw.rounded_rectangle((100, row_y - 8, 1858, row_y + 36), radius=14, fill=(249, 250, 252, 255))
            status = row["status"]
            status_color = PLAIN if status == "PASS" else SECRET
            _draw_text(draw, (120, row_y), row["name"], fonts["body"])
            _draw_text(draw, (450, row_y), row["layer"][:22], fonts["body"], INFO)
            _draw_text(draw, (790, row_y), row.get("fallback", "")[:16], fonts["body"], WARN)
            draw.rounded_rectangle((1060, row_y - 4, 1140, row_y + 24), radius=14, fill=(236, 248, 240, 255) if status == "PASS" else (255, 239, 241, 255))
            _draw_text(draw, (1079, row_y - 1), status, fonts["body"], status_color)
            _draw_text(draw, (1210, row_y), row["system_state"][:84], fonts["small"], MUTED)
            row_y += 48

    _draw_text(draw, (118, image_height - 62), "系统状态摘要来自本地进程采样：FD 数、socket FD 数、RSS 和线程数；用于证明请求被拒后资源回到稳态。", fonts["small"], MUTED)
    image.convert("RGB").save(ROBUSTNESS_FIGURE_OUT, quality=95)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_threshold_figure()
    generate_topology_figure()
    generate_sequence_figure()
    generate_robustness_figure()
    print(f"generated figures: {THRESHOLD_FIGURE_OUT}, {TOPOLOGY_FIGURE_OUT}, {SEQUENCE_FIGURE_OUT}, {ROBUSTNESS_FIGURE_OUT}")


if __name__ == "__main__":
    main()
