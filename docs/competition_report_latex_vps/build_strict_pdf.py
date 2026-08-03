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
from generate_report_figures import generate_all_report_figures
from generate_report_ui_snapshots import generate_report_ui_snapshots


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "original_report.pdf"
OUTPUT = ROOT / "source" / "report_strict_vps.pdf"
WORK = ROOT / "output" / "intermediate" / "strict_format"
CHART_DATA = ROOT / "performance_chart_data.json"
REPORT_FIGURE_DATA = ROOT / "report_figure_data.json"

SIMSUN_PATH = Path("/mnt/c/Windows/Fonts/simsun.ttc")
SIMHEI_PATH = Path("/mnt/c/Windows/Fonts/simhei.ttf")
MSYH_PATH = Path("/mnt/c/Windows/Fonts/msyh.ttc")
MSYH_BOLD_PATH = Path("/mnt/c/Windows/Fonts/msyhbd.ttc")
TIMES_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")
DEJAVU_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEJAVU_MONO_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
CAMBRIA_PATH = Path("/mnt/c/Windows/Fonts/cambria.ttc")

SIMSUN_NAME = "SimSunFull"
TIMES_NAME = "TimesNewRomanFull"


def require_inputs() -> None:
    missing = [
        p
        for p in (
            SOURCE,
            CHART_DATA,
            REPORT_FIGURE_DATA,
            SIMSUN_PATH,
            SIMHEI_PATH,
            MSYH_PATH,
            MSYH_BOLD_PATH,
            TIMES_PATH,
            DEJAVU_PATH,
            DEJAVU_MONO_PATH,
            CAMBRIA_PATH,
        )
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


def patch_abstracts(doc: fitz.Document) -> None:
    # Same-shape proxy: 1 - 8.10450275739034 / 15.336534976959229.
    chinese = doc[7]
    redact(chinese, [(173.2, 193.8, 210.4, 207.2)])
    insert_mixed(chinese, 173.76, 204.96, "47.16%", all_simsun=True)

    english = doc[9]
    redact(english, [(306.7, 729.8, 344.8, 747.4)])
    insert_mixed(english, 307.2, 743.16, "47.16%")


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
    insert_mixed(
        page,
        175.8,
        369.0,
        "两方2PC colocated localhost；医疗batch size 16，金融batch size 8",
    )
    # Keep the original three-line row rhythm.  The previous one-line rewrite
    # left an oversized void above the bottom rule even though its typography
    # was technically unchanged.
    insert_mixed(page, 175.8, 392.4, "未配置GPU；")
    insert_mixed(page, 175.8, 415.68, "SPU推理统一使用CPU运行，")
    insert_mixed(page, 175.8, 438.96, "GPU不参与性能统计。")


def patch_page_51(page: fitz.Page) -> None:
    # Figure 4-2 compares 92.7481% with the 91.9847% static control,
    # so the improvement here is 0.7634 percentage points.  The 3.0534-point
    # delta belongs to the separate ablation baseline in Figure 4-8.
    redact(page, [(488.5, 636.0, 522.5, 653.5)])
    insert_mixed(page, 489.0, 649.44, "0.7634")


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
        page, 237.6, 354.6, "30.13 秒/样本", all_simsun=True, faux_bold=True
    )
    insert_centered(
        page, 438.0, 342.84, "医疗交付场景", all_simsun=True, faux_bold=True
    )
    insert_centered(
        page, 438.0, 366.24, "端到端双向隐私推理", all_simsun=True, faux_bold=True
    )

    insert_centered(page, 140.16, 412.92, "批次通信量", all_simsun=True)
    insert_centered(page, 276.84, 401.28, "42.57 GiB", all_simsun=True)
    insert_centered(page, 438.0, 389.64, "两方2PC安全推理", all_simsun=True)
    insert_centered(page, 438.0, 412.92, "通信量统计", all_simsun=True)

    insert_centered(page, 140.16, 436.32, "32条样本验证", all_simsun=True)
    insert_centered(page, 140.16, 459.6, "阈值精度/AUC", all_simsun=True)
    insert_centered(page, 276.84, 436.32, "93.75%", all_simsun=True)
    insert_centered(page, 276.84, 459.6, "0.96484", all_simsun=True)
    insert_centered(page, 438.0, 436.32, "固定医疗", all_simsun=True)
    insert_centered(page, 438.0, 459.6, "部署验证样本", all_simsun=True)


def replace_page_image(page: fitz.Page, image_xref: int, generated: Path) -> None:
    """Remove one submitted image object and insert a generated replacement."""

    image_rects = page.get_image_rects(image_xref)
    if len(image_rects) != 1:
        raise ValueError(f"Expected one image placement for xref {image_xref}, found {len(image_rects)}")
    page.delete_image(image_xref)
    page.insert_image(
        image_rects[0],
        filename=str(generated),
        keep_proportion=False,
        overlay=True,
    )


def build_report_figures(doc: fitz.Document) -> dict[str, Path]:
    """Generate and embed all quantitative and assessment figures."""

    generated = generate_all_report_figures(
        REPORT_FIGURE_DATA,
        WORK / "generated_figures",
        repo_root=ROOT.parents[1],
        regular_font_path=MSYH_PATH,
        bold_font_path=MSYH_BOLD_PATH,
        latin_font_path=DEJAVU_PATH,
    )
    placements = {
        "fig4_1": (49, 174),
        "fig4_2": (51, 179),
        "fig4_3": (53, 184),
        "fig4_5": (55, 191),
        "fig4_6": (56, 195),
        "fig4_7": (61, 207),
        "fig4_8": (63, 213),
        "fig5_1": (70, 229),
    }
    for key, (page_index, image_xref) in placements.items():
        replace_page_image(doc[page_index], image_xref, generated[key])
    return generated


def build_chart(doc: fitz.Document) -> Path:
    """Generate the complete performance dashboard from data and embed it.

    The submitted image uses a vertically inverted PDF image matrix.  Remove
    that image object and insert the generated chart at the same page rectangle
    so the new chart remains upright and the old pixels are not retained.
    """

    page = doc[53]
    chart_xref = 186
    generated = generate_performance_chart(
        CHART_DATA,
        WORK / "performance_chart_generated.png",
        chinese_font_path=MSYH_PATH,
        latin_font_path=DEJAVU_PATH,
    )
    replace_page_image(page, chart_xref, generated)
    return generated


def build_appendix_ui(doc: fitz.Document) -> dict[str, Path]:
    """Generate appendix interfaces from current evidence, without editing showcase code."""

    generated = generate_report_ui_snapshots(
        ROOT.parents[1],
        WORK / "generated_ui_snapshots",
        regular_font=MSYH_PATH,
        bold_font=MSYH_BOLD_PATH,
        mono_font=DEJAVU_MONO_PATH,
    )
    replace_page_image(doc[87], 284, generated["overview"])
    replace_page_image(doc[87], 285, generated["task_create"])
    replace_page_image(doc[88], 288, generated["task_detail"])
    replace_page_image(doc[88], 289, generated["model_assets"])
    replace_page_image(doc[89], 292, generated["evidence_top"])
    replace_page_image(doc[89], 293, generated["evidence_bottom"])
    replace_page_image(doc[90], 297, generated["pruning_overview"])
    replace_page_image(doc[91], 303, generated["user_report"])
    return generated


def patch_page_54(page: fitz.Page) -> None:
    redact(page, [(80, 355, 523, 421)])
    lines = [
        (105, 368.76, "医疗交付场景与金融边界压力验证场景的端到端性能与通信量统计结果如图4-4"),
        (81, 392.16, "所示。医疗场景的平均时延为30.13秒/样本，每样本通信量为1.33 GiB；金融边"),
        (81, 415.56, "界压力验证场景的平均时延为47.23秒/样本，每样本通信量为1.93 GiB。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_55(page: fitz.Page) -> None:
    # Reconcile the conceptual single-end send/receive decomposition in
    # Equation (4-2) with the colocated loopback measurement used in this report.
    redact(page, [(80, 493, 540, 723)])
    lines = [
        (105, 508.32, "其中，Ttotal由前端预处理、服务端控制面快检、SPU密态执行与结果回传/审计"),
        (81, 531.72, "收尾四部分组成；式（4-2）的Vtotal采用单端发送/接收分解，通信比值与时间比值"),
        (81, 555.12, "用于相对比较。在colocated环回测试中，接口TX已汇总双方的实际发送字节，"),
        (81, 578.52, "RX为同一流量的镜像，因此报告只计TX增量一次，不重复相加。"),
        (105, 601.92, "性能证据分为两层：其一是端到端双向隐私运行结果，用于说明医疗动态安全剪"),
        (81, 625.32, "枝链路与金融压力样本的总时延和有效通信量；其二是统一安全基准下的算子级代"),
        (81, 648.72, "理对比，用于说明算子替换本身的开销趋势。在固定同一DeiT-S形状的代理基准"),
        (81, 672.12, "对比中，密捷的安全友好算子将通信压缩到外部基准算子的0.149倍，时间压缩到"),
        (81, 695.52, "0.528倍，说明算子替换本身具有正向收益。跨模型结构的异构结构代理基准只"),
        (81, 718.92, "用于说明结构尺度差异会放大绝对代价，不能与医疗全量验证链路混写。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_56(page: fitz.Page) -> None:
    redact(
        page,
        [
            (306.8, 269.9, 350.6, 283.8),
            (309.8, 301.3, 347.6, 315.3),
            (360.3, 328.3, 399.0, 346.0),
        ],
    )
    insert_centered(page, 328.68, 281.28, "15.3365", all_simsun=True)
    insert_centered(page, 328.68, 312.72, "8.1045", all_simsun=True)
    insert_mixed(page, 361.08, 341.64, "47.16%")


def patch_page_59(page: fitz.Page) -> None:
    # Runtime code uses int(init_n * ratio), i.e. floor for positive ratios.
    redact(page, [(233.8, 519.0, 369.0, 585.5)])
    font = fitz.Font(fontfile=str(CAMBRIA_PATH))
    formulas = [
        (532.44, "K₃ = ⌊196 × 0.7⌋ = 137,"),
        (555.84, "K₆ = ⌊196 × 0.49⌋ = 96,"),
        (579.24, "K₉ = ⌊196 × 0.343⌋ = 67."),
    ]
    for baseline, formula in formulas:
        x = 301.4 - font.text_length(formula, 12) / 2
        page.insert_text(
            (x, baseline),
            formula,
            fontfile=str(CAMBRIA_PATH),
            fontname="CambriaMathFull",
            fontsize=12,
            color=(0, 0, 0),
            overlay=True,
        )


def patch_page_61(page: fitz.Page) -> None:
    redact(
        page,
        [
            (262, 194, 353, 212),
            (270, 224, 345, 241),
            (378, 194, 515, 212),
            (401, 224, 492, 241),
        ],
    )
    insert_centered(page, 307.5, 207.36, "47.23 秒/样本", all_simsun=True)
    insert_centered(page, 307.5, 236.88, "15.45 GiB", all_simsun=True)
    insert_centered(page, 446.4, 207.36, "金融边界压力验证", all_simsun=True)
    insert_centered(page, 446.4, 236.88, "两方2PC通信量", all_simsun=True)


def patch_page_67(page: fitz.Page) -> None:
    # This 3.0534-point statement cites the Figure 4-8 ablation baseline
    # (89.6947%), not the Figure 4-2 static control (91.9847%).
    redact(page, [(104.5, 310.8, 213.5, 324.5)])
    insert_mixed(page, 105.0, 321.96, "静态安全消融基准线", all_simsun=True)


def patch_page_68(page: fitz.Page) -> None:
    redact(page, [(80, 425, 523, 538)])
    lines = [
        (105, 438.96, "实验证据方面，医疗交付场景在32条部署验证样本上的平均时延为30.13秒/样本，"),
        (81, 462.36, "批次通信量为42.57 GiB；阈值精度为93.75%，AUC为0.96484。服务端不接收明"),
        (81, 485.76, "文像素值、模型参数不以明文暴露，对外仅返回最终分类结果；金融边界压力样本"),
        (81, 509.16, "8/8与明文参考逐样本一致，平均时延为47.23秒/样本，批次通信量为15.45 GiB。"),
        (81, 532.56, "该层内容由第2.1、2.5、3.2、4.4与4.6节共同支撑。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_69(page: fitz.Page) -> None:
    # The original justified lines place two punctuation glyphs slightly past
    # x=522 pt, so clear through x=540 to avoid leaving detached marks.
    redact(page, [(80, 238, 540, 514)])
    lines = [
        (105, 251.76, "从可核验性角度看，本作品按实验口径组织模型精度、性能和鲁棒性证据。"),
        (81, 275.16, "医疗全量阈值精度与AUC对应results/final/中的冻结结果；协议与控制面验证"),
        (81, 298.56, "对应results/fuzzing/和results/guard_stress/中的正式记录；完整端到端SPU"),
        (81, 321.96, "时延与通信量则对应本报告数据来源文档及归档运行摘要。各类指标均标注样本"),
        (81, 345.36, "范围和统计方式，避免把展示界面、全量离线验证与部署批次性能混为一谈。报告"),
        (81, 368.76, "中的统计图由结构化数据重新生成，并可回指到相应证据文件。"),
        (105, 392.16, "具体而言，医疗阈值精度92.7481%和AUC 0.9639来自524条全量验证样本；"),
        (81, 415.56, "端到端平均时延30.13秒/样本和批次通信量42.57 GiB来自32条固定部署样"),
        (81, 438.96, "本的完整两方SPU运行；金融结果来自8条边界压力样本。通信量统一采用环回接"),
        (81, 462.36, "口TX增量单计，避免把TX与RX镜像流量重复相加。鲁棒性结论则由13类协议模"),
        (81, 485.76, "糊用例和4类控制面压力用例支撑。通过这种按口径索引的组织方式，评审可从报"),
        (81, 509.16, "告中的每一类核心数字回溯到对应记录，而不依赖页面中的演示值。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_88(page: fitz.Page) -> None:
    redact(page, [(80, 75, 523, 140)])
    lines = [
        (105, 87.96, "图C-2展示了管理员端总览看板。该页面集中呈现当前任务状态、正式主线"),
        (81, 111.36, "模型指标、VPS运行环境以及证据目录等关键信息，便于在展示过程中快速说明"),
        (81, 134.76, "系统运行状态与主线模型交付情况。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def patch_page_92(page: fitz.Page) -> None:
    redact(page, [(80, 323, 523, 388)])
    lines = [
        (105, 337.56, "图C-11展示了用户端报告详情页。该界面以标准化报告形式呈现样本信息、分"),
        (81, 360.96, "类结论、部署阈值、完整SPU运行证据与三阶段动态剪枝摘要，是本作品结果交付"),
        (81, 384.36, "链路在用户侧的最终承接页面。"),
    ]
    for x, baseline, value in lines:
        insert_mixed(page, x, baseline, value)


def main() -> None:
    require_inputs()
    doc = fitz.open(SOURCE)
    if doc.page_count != 93:
        raise ValueError(f"Expected 93 pages, found {doc.page_count}")

    build_report_figures(doc)
    patch_abstracts(doc)
    patch_page_29(doc[28])
    patch_page_30(doc[29])
    patch_page_51(doc[50])
    patch_page_53(doc[52])
    build_chart(doc)
    patch_page_54(doc[53])
    patch_page_55(doc[54])
    patch_page_56(doc[55])
    patch_page_59(doc[58])
    patch_page_61(doc[60])
    patch_page_67(doc[66])
    patch_page_68(doc[67])
    patch_page_69(doc[68])
    build_appendix_ui(doc)
    patch_page_88(doc[87])
    patch_page_92(doc[91])

    doc.save(OUTPUT, garbage=4, deflate=True, clean=True)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
