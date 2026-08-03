#!/usr/bin/env python3
"""Build the strict-format VPS report from the submitted PDF.

The submitted PDF is the layout source of truth.  This script removes only
the text that must change, writes the replacements on the original baselines
with the same Windows fonts and point sizes, and leaves every table rule,
paragraph position, page header, page number and unaffected object intact.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "original_report.pdf"
OUTPUT = ROOT / "source" / "report_strict_vps.pdf"
WORK = ROOT / "output" / "intermediate" / "strict_format"

SIMSUN_PATH = Path("/mnt/c/Windows/Fonts/simsun.ttc")
SIMHEI_PATH = Path("/mnt/c/Windows/Fonts/simhei.ttf")
TIMES_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")

SIMSUN_NAME = "SimSunFull"
TIMES_NAME = "TimesNewRomanFull"


def require_inputs() -> None:
    missing = [p for p in (SOURCE, SIMSUN_PATH, SIMHEI_PATH, TIMES_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing strict-format input(s): " + ", ".join(map(str, missing)))
    WORK.mkdir(parents=True, exist_ok=True)


def redact(page: fitz.Page, rects: list[tuple[float, float, float, float]]) -> None:
    for rect in rects:
        page.add_redact_annot(fitz.Rect(*rect), fill=(1, 1, 1))
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )


def text_width(text: str, size: float, all_simsun: bool = False) -> float:
    simsun = fitz.Font(fontfile=str(SIMSUN_PATH))
    if all_simsun:
        return simsun.text_length(text, size)
    times = fitz.Font(fontfile=str(TIMES_PATH))
    width = 0.0
    for part in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", text):
        font = times if part[0].isascii() else simsun
        width += font.text_length(part, size)
    return width


def insert_mixed(
    page: fitz.Page,
    x: float,
    baseline: float,
    text: str,
    size: float = 12,
    *,
    all_simsun: bool = False,
    faux_bold: bool = False,
) -> None:
    def write_run(
        run_x: float,
        run_text: str,
        font_path: Path,
        font_name: str,
    ) -> None:
        if faux_bold:
            # The submitted table uses a normal SimSun fill followed by a
            # 0.3432 pt text stroke to simulate bold.  PDF render mode 2 emits
            # that same fill + stroke while keeping one searchable text run.
            page.insert_text(
                (run_x, baseline),
                run_text,
                fontfile=str(font_path),
                fontname=font_name,
                fontsize=size,
                color=(0, 0, 0),
                fill=(0, 0, 0),
                # PyMuPDF's border_width is expressed relative to the font
                # size.  0.0286 * 12 pt reproduces the source 0.3432 pt line.
                border_width=0.0286,
                render_mode=2,
                overlay=True,
            )
        else:
            page.insert_text(
                (run_x, baseline),
                run_text,
                fontfile=str(font_path),
                fontname=font_name,
                fontsize=size,
                color=(0, 0, 0),
                overlay=True,
            )

    if all_simsun:
        write_run(x, text, SIMSUN_PATH, SIMSUN_NAME)
        return

    cursor = x
    simsun = fitz.Font(fontfile=str(SIMSUN_PATH))
    times = fitz.Font(fontfile=str(TIMES_PATH))
    for part in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", text):
        latin = part[0].isascii()
        path = TIMES_PATH if latin else SIMSUN_PATH
        name = TIMES_NAME if latin else SIMSUN_NAME
        font = times if latin else simsun
        write_run(cursor, part, path, name)
        cursor += font.text_length(part, size)


def insert_centered(
    page: fitz.Page,
    center_x: float,
    baseline: float,
    text: str,
    size: float = 12,
    *,
    all_simsun: bool = False,
    faux_bold: bool = False,
) -> None:
    insert_mixed(
        page,
        center_x - text_width(text, size, all_simsun=all_simsun) / 2,
        baseline,
        text,
        size,
        all_simsun=all_simsun,
        faux_bold=faux_bold,
    )


def patch_page_29(page: fitz.Page) -> None:
    redact(page, [(80, 617, 523, 750)])
    lines = [
        (105, 629.28, "算法、训练过程、离线精度、算子代理基准与鲁棒性证据均保持原稿不变；"),
        (81, 652.68, "本轮仅使用原始代码在当前VPS上复跑32条医疗部署验证样本，环境基准如表"),
        (105, 676.08, "2-4所示。VPS复跑的性能与通信量来自同一次运行，通信量按环回接口TX增量"),
        (81, 699.48, "单计；金融边界压力数据及其他历史实验记录仍保留原报告口径。两组数据硬件、"),
        (81, 722.88, "SPU版本和通信计数方法不完全一致，因此后文按各自口径列示，不作直接比例比"),
        (81, 746.28, "较；当前VPS的详细配置与医疗复跑数据以本轮实测记录为准，后续时延和通信量"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_30(page: fitz.Page) -> None:
    redact(
        page,
        [
            (80, 76, 523, 91),
            (173, 168, 519, 209),
            (173, 214, 519, 233),
            (173, 238, 519, 256),
            (173, 261, 519, 280),
            (173, 308, 519, 328),
            (173, 331, 519, 374),
            (173, 378, 519, 443),
        ],
    )
    # The CPU-cell redaction crosses the original 0.48 pt header rule.  Restore
    # the complete rule before adding cell text so it runs under both columns.
    page.draw_line(
        fitz.Point(81.0, 170.04),
        fitz.Point(522.0, 170.04),
        color=(0, 0, 0),
        width=0.48,
        overlay=True,
    )
    insert_mixed(page, 81, 87.96, "均默认建立在本节明确标注的相应口径之上。")

    insert_mixed(page, 175.8, 182.28, "阿里云KVM实例，Intel Xeon Platinum，16 vCPU")
    insert_mixed(page, 175.8, 205.56, "（1路、8核、每核2线程）")
    insert_mixed(page, 175.8, 228.0, "61 GiB（未配置Swap）")
    insert_mixed(page, 175.8, 251.4, "Ubuntu 24.04.4 LTS")
    insert_mixed(page, 175.8, 274.8, "Linux 6.8.0-136-generic")
    insert_mixed(page, 175.8, 322.32, "SecretFlow SPU 0.9.3b0（Apache License 2.0）")
    insert_mixed(page, 175.8, 345.72, "JAX 0.4.30，NumPy 1.26.4，PyTorch 1.13.1+cpu；")
    insert_mixed(page, 175.8, 369.0, "两方2PC colocated localhost，batch size 8")
    # Keep the original three-line row rhythm.  The previous one-line rewrite
    # left an oversized void above the bottom rule even though its typography
    # was technically unchanged.
    insert_mixed(page, 175.8, 392.4, "未配置、未使用GPU；")
    insert_mixed(page, 175.8, 415.68, "本轮SPU推理使用CPU运行，")
    insert_mixed(page, 175.8, 438.96, "GPU不参与性能统计。")


def patch_page_53(page: fitz.Page) -> None:
    redact(
        page,
        [
            (232, 342, 320, 370),
            (356, 330, 520, 370),
            (95, 400, 186, 416),
            (246, 389, 307, 404),
            (356, 378, 520, 416),
            (95, 423, 187, 464),
            (200, 423, 351, 464),
            (356, 423, 520, 464),
        ],
    )

    insert_mixed(
        page, 237.6, 354.6, "48.69 秒/样本", all_simsun=True, faux_bold=True
    )
    insert_centered(
        page, 438.0, 342.84, "当前VPS原始代码复跑", all_simsun=True, faux_bold=True
    )
    insert_centered(
        page, 438.0, 366.24, "总时长1558.20秒", all_simsun=True, faux_bold=True
    )

    insert_centered(page, 140.16, 412.92, "批次通信量（TX单计）", all_simsun=True)
    insert_centered(page, 276.84, 401.28, "68.38 GiB", all_simsun=True)
    insert_centered(page, 438.0, 389.64, "当前VPS环回接口TX", all_simsun=True)
    insert_centered(page, 438.0, 412.92, "增量单计", all_simsun=True)

    insert_centered(page, 140.16, 436.32, "32条复跑样本", all_simsun=True)
    insert_centered(page, 140.16, 459.6, "阈值精度/AUC", all_simsun=True)
    insert_centered(page, 276.84, 436.32, "93.75%", all_simsun=True)
    insert_centered(page, 276.84, 459.6, "0.96484", all_simsun=True)
    insert_centered(page, 438.0, 436.32, "当前VPS原始代码", all_simsun=True)
    insert_centered(page, 438.0, 459.6, "固定32条医疗样本", all_simsun=True)


def clear_with_tiled_background(
    pixels: np.ndarray,
    target: tuple[int, int, int, int],
    source_x: tuple[int, int],
) -> None:
    x0, y0, x1, y1 = target
    sx0, sx1 = source_x
    strip = pixels[y0:y1, sx0:sx1].copy()
    repeats = (x1 - x0 + strip.shape[1] - 1) // strip.shape[1]
    pixels[y0:y1, x0:x1] = np.tile(strip, (1, repeats, 1))[:, : x1 - x0]


def build_chart(doc: fitz.Document) -> Path:
    page = doc[53]
    chart_xref = 186
    extracted = doc.extract_image(chart_xref)
    original = WORK / f"chart_original.{extracted['ext']}"
    original.write_bytes(extracted["image"])

    upright = ImageOps.flip(Image.open(original).convert("RGB"))
    pixels = np.array(upright)

    # Each edit keeps the original blue bar below the new measured height and
    # restores the original plot background above it from the adjacent blank
    # strip.  Axes, ticks, finance bars and all panel geometry remain unchanged.
    edits = [
        # bar rect, blank-strip x, old-label rect, label, new top
        ((368, 180, 435, 401), (523, 540), (362, 143, 443, 182), "1558.20", 401),
        ((635, 285, 700, 458), (702, 724), (634, 246, 704, 286), "48.69", 458),
        ((910, 157, 977, 260), (980, 998), (908, 119, 980, 158), "68.38", 260),
        ((1185, 269, 1253, 348), (1255, 1273), (1186, 230, 1254, 270), "2.14", 348),
    ]
    # Tick rows in the original 726 px chart.  Repaint only the pixels formerly
    # occupied by the taller medical bars and their labels, then restore the
    # same faint dashed horizontal grid.  This avoids interpolation seams.
    grid_rows = [
        [73, 158, 242, 326, 411, 495, 579, 664],
        [73, 157, 242, 326, 411, 495, 580, 664],
        [73, 191, 309, 428, 546, 664],
        [73, 147, 221, 295, 369, 443, 517, 591, 664],
    ]
    revised = Image.fromarray(pixels)
    cleanup_draw = ImageDraw.Draw(revised)
    for (bar_rect, _, label_rect, _, _), rows in zip(edits, grid_rows):
        cleanup_draw.rectangle(label_rect, fill=(255, 255, 255))
        cleanup_draw.rectangle(bar_rect, fill=(255, 255, 255))
        for region in (label_rect, bar_rect):
            x0, y0, x1, y1 = region
            for row in rows:
                if y0 <= row < y1:
                    start = x0 - (x0 % 12)
                    for dash_x in range(start, x1, 12):
                        cleanup_draw.line(
                            (max(x0, dash_x), row, min(x1, dash_x + 6), row),
                            fill=(224, 224, 224),
                            width=1,
                        )

    draw = ImageDraw.Draw(revised)
    number_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for bar_rect, _, _, label, new_top in edits:
        x0, _, x1, _ = bar_rect
        bbox = draw.textbbox((0, 0), label, font=number_font)
        label_width = bbox[2] - bbox[0]
        draw.text(((x0 + x1 - label_width) / 2, new_top - 29), label, fill=(0, 0, 0), font=number_font)

    # The medical value now uses loopback TX single counting, whereas the
    # finance value retains the historical report definition.
    title_rect = (876, 21, 1104, 61)
    ImageDraw.Draw(revised).rectangle(title_rect, fill=(255, 255, 255))
    cjk_font = ImageFont.truetype(str(SIMHEI_PATH), 21)
    latin_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    title_runs = [("通信量（", cjk_font), ("GiB", latin_font), ("）", cjk_font)]
    title_width = sum(draw.textlength(value, font=font) for value, font in title_runs)
    title_x = (title_rect[0] + title_rect[2] - title_width) / 2
    for value, font in title_runs:
        draw.text((title_x, 27), value, fill=(0, 0, 0), font=font)
        title_x += draw.textlength(value, font=font)

    modified_upright = WORK / "chart_revised_upright.png"
    revised.save(modified_upright)
    modified_raw = WORK / "chart_revised_raw.png"
    ImageOps.flip(revised).save(modified_raw)
    page.replace_image(chart_xref, filename=str(modified_raw))
    return modified_upright


def patch_page_54(page: fitz.Page) -> None:
    redact(page, [(80, 355, 523, 421)])
    lines = [
        (105, 368.76, "医疗交付场景与金融边界压力验证场景的端到端性能与通信量统计结果如图4-4"),
        (81, 392.16, "所示。医疗数据为当前VPS原始代码复跑结果，通信量按环回接口TX增量单计；金"),
        (81, 415.56, "融数据为原报告历史记录。因环境与计数方法不同，两组数据不作直接比例比较。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_61(page: fitz.Page) -> None:
    redact(page, [(378, 194, 515, 212), (401, 224, 492, 241)])
    insert_centered(page, 446.4, 207.36, "原报告历史边界口径", all_simsun=True)
    insert_centered(page, 446.4, 236.88, "原报告历史计数口径", all_simsun=True)


def patch_page_68(page: fitz.Page) -> None:
    redact(page, [(80, 425, 523, 538)])
    lines = [
        (105, 438.96, "实验证据方面，当前VPS原始代码复跑32条样本，平均时延为48.69秒/样本，"),
        (81, 462.36, "环回接口TX增量单计通信量为68.38 GiB；该32条医疗样本阈值精度为93.75%，AUC"),
        (81, 485.76, "为0.96484。服务端不接收明文像素值、模型参数不以明文暴露，对外仅返回最终分"),
        (81, 509.16, "类结果；金融边界压力样本8/8与明文参考逐样本一致，性能数据保留原报告历史口"),
        (81, 532.56, "径。该层内容由第2.1、2.5、3.2、4.4与4.6节共同支撑。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def main() -> None:
    require_inputs()
    doc = fitz.open(SOURCE)
    if doc.page_count != 93:
        raise ValueError(f"Expected 93 pages, found {doc.page_count}")

    patch_page_29(doc[28])
    patch_page_30(doc[29])
    patch_page_53(doc[52])
    build_chart(doc)
    patch_page_54(doc[53])
    patch_page_61(doc[60])
    patch_page_68(doc[67])

    doc.save(OUTPUT, garbage=4, deflate=True, clean=True)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
