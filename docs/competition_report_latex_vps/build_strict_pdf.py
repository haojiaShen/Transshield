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

from generate_performance_chart import generate_performance_chart


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "original_report.pdf"
OUTPUT = ROOT / "source" / "report_strict_vps.pdf"
WORK = ROOT / "output" / "intermediate" / "strict_format"
CHART_DATA = ROOT / "performance_chart_data.json"

SIMSUN_PATH = Path("/mnt/c/Windows/Fonts/simsun.ttc")
SIMHEI_PATH = Path("/mnt/c/Windows/Fonts/simhei.ttf")
TIMES_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")

SIMSUN_NAME = "SimSunFull"
TIMES_NAME = "TimesNewRomanFull"


def require_inputs() -> None:
    missing = [
        p
        for p in (SOURCE, CHART_DATA, SIMSUN_PATH, SIMHEI_PATH, TIMES_PATH)
        if not p.exists()
    ]
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
        (105, 629.28, "本作品医疗交付场景的性能测试在表2-4所示的软硬件环境下执行。安全推理时"),
        (81, 652.68, "延同时受到协议轮次、矩阵规模、底层数值库、并发方式和网络条件影响。为保"),
        (81, 676.08, "证性能结果具有可比性，医疗任务采用固定任务设置和样本集合，并在端到端双向"),
        (81, 699.48, "隐私推理流程中统计时延、通信量与预测结果，重点评估部署精度、AUC和推理效"),
        (81, 722.88, "率。金融任务用于验证边界压力条件下的结果一致性与资源开销，各项实验结果及"),
        (81, 746.28, "对比分析见第4章。"),
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
    insert_mixed(page, 81, 87.96, "表中配置作为医疗交付场景性能与通信量分析的实验基准。")

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
    insert_mixed(page, 175.8, 392.4, "未配置GPU；")
    insert_mixed(page, 175.8, 415.68, "SPU推理统一使用CPU运行，")
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
        page, 438.0, 342.84, "医疗交付场景", all_simsun=True, faux_bold=True
    )
    insert_centered(
        page, 438.0, 366.24, "端到端双向隐私推理", all_simsun=True, faux_bold=True
    )

    insert_centered(page, 140.16, 412.92, "批次通信量", all_simsun=True)
    insert_centered(page, 276.84, 401.28, "68.38 GiB", all_simsun=True)
    insert_centered(page, 438.0, 389.64, "两方2PC安全推理", all_simsun=True)
    insert_centered(page, 438.0, 412.92, "通信量统计", all_simsun=True)

    insert_centered(page, 140.16, 436.32, "32条样本验证", all_simsun=True)
    insert_centered(page, 140.16, 459.6, "阈值精度/AUC", all_simsun=True)
    insert_centered(page, 276.84, 436.32, "93.75%", all_simsun=True)
    insert_centered(page, 276.84, 459.6, "0.96484", all_simsun=True)
    insert_centered(page, 438.0, 436.32, "固定医疗", all_simsun=True)
    insert_centered(page, 438.0, 459.6, "部署验证样本", all_simsun=True)


def build_chart(doc: fitz.Document) -> Path:
    """Generate the complete performance chart from data and embed it.

    The submitted image uses a vertically inverted PDF image matrix.  Remove
    that image object and insert the generated chart at the same page rectangle
    so the new chart remains upright and the old pixels are not retained.
    """

    page = doc[53]
    chart_xref = 186
    chart_rects = page.get_image_rects(chart_xref)
    if len(chart_rects) != 1:
        raise ValueError(f"Expected one performance chart placement, found {len(chart_rects)}")
    generated = generate_performance_chart(
        CHART_DATA,
        WORK / "performance_chart_generated.png",
        chinese_font_path=SIMHEI_PATH,
        latin_font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    page.delete_image(chart_xref)
    page.insert_image(
        chart_rects[0],
        filename=str(generated),
        keep_proportion=False,
        overlay=True,
    )
    return generated


def patch_page_54(page: fitz.Page) -> None:
    redact(page, [(80, 355, 523, 421)])
    lines = [
        (105, 368.76, "医疗交付场景与金融边界压力验证场景的端到端性能与通信量统计结果如图4-4"),
        (81, 392.16, "所示。医疗场景的平均时延为48.69秒/样本，每样本通信量为2.14 GiB；金融边"),
        (81, 415.56, "界压力验证场景的平均时延为106.16秒/样本，每样本通信量为3.16 GiB。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_61(page: fitz.Page) -> None:
    redact(page, [(378, 194, 515, 212), (401, 224, 492, 241)])
    insert_centered(page, 446.4, 207.36, "金融边界压力验证", all_simsun=True)
    insert_centered(page, 446.4, 236.88, "双向通信量统计", all_simsun=True)


def patch_page_68(page: fitz.Page) -> None:
    redact(page, [(80, 425, 523, 538)])
    lines = [
        (105, 438.96, "实验证据方面，医疗交付场景在32条部署验证样本上的平均时延为48.69秒/样本，"),
        (81, 462.36, "批次通信量为68.38 GiB；阈值精度为93.75%，AUC为0.96484。服务端不接收明"),
        (81, 485.76, "文像素值、模型参数不以明文暴露，对外仅返回最终分类结果；金融边界压力样本"),
        (81, 509.16, "8/8与明文参考逐样本一致，平均时延为106.16秒/样本，双向通信量为25.30 GiB。"),
        (81, 532.56, "该层内容由第2.1、2.5、3.2、4.4与4.6节共同支撑。"),
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
