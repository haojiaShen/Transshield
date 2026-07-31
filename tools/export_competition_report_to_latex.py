#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCX = REPO_ROOT / "docs" / "密捷竞赛作品报告.docx"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "overleaf_report"
ASSET_SOURCE = REPO_ROOT / "showcase" / "public" / "report-assets"

FIGURE_MAP = {
    "2-1": "fig2-1-system-topology.png",
    "2-2": "fig2-2-software-sequence.png",
    "2-3": "fig2-3-pruning-rewrite.png",
    "3-1": "fig3-1-project-architecture.png",
    "3-2": "fig3-2-browser-worker-collaboration.png",
    "3-3": "fig3-3-control-plane-gates.jpeg",
    "4-1": "fig4-1-threshold-shift.png",
    "4-2": "fig4-2-guard-matrix.png",
    "5-1": "fig5-1-capability-matrix.jpeg",
}

EQUATION_MAP = {
    "3-1": r"\tau^{(l)}=\operatorname{TopKBoundary}\!\left(s^{(l)},K^{(l)}\right), \quad m_i^{(l)}=\mathbf{1}\!\left[\left(s_i^{(l)},i\right)\ge \tau^{(l)}\right]",
    "3-2": r"\tilde{h}_i^{(l)}=m_i^{(l)} \cdot h_i^{(l)}",
    "3-2a": r"e_i^{(l)}=s_i^{(l)}-\epsilon \cdot i",
    "3-2b": r"\pi^{(l)}=\operatorname{BitonicSortDesc}\!\left(e^{(l)}\right), \quad m_i^{(l)}=\mathbf{1}\!\left[i\in \pi^{(l)}_{1:K^{(l)}}\right]",
    "3-2c": r"K^{(l)}=\left\lceil \rho^{(l)} N^{(l)} \right\rceil, \quad N_{\text{out}}^{(l)}=\sum_{i=1}^{N^{(l)}} m_i^{(l)}",
    "3-2d": r"b_i^{(l)}=F_{\text{less}}\!\left(\tau^{(l)}, s_i^{(l)}\right)",
    "3-2e": r"\tilde{h}_i^{(l)}=F_{\text{mux}}\!\left(b_i^{(l)}, h_i^{(l)}, 0\right)",
    "3-3": r"x_{c,h,w}=\operatorname{clip}\!\left(\frac{\frac{p_{c,h,w}}{255}-\mu_c}{\sigma_c},-2,2\right)",
    "3-4": r"\operatorname{share0}_{c,h,w}=r_{c,h,w}, \quad r_{c,h,w}\sim U[-2,2], \quad \operatorname{share1}_{c,h,w}=x_{c,h,w}-\operatorname{share0}_{c,h,w}",
    "3-4a": r"x_{c,h,w}=\langle x_{c,h,w}\rangle_{0}+\langle x_{c,h,w}\rangle_{1}, \quad \langle x_{c,h,w}\rangle_{0}=\operatorname{share0}_{c,h,w}, \quad \langle x_{c,h,w}\rangle_{1}=\operatorname{share1}_{c,h,w}",
    "3-5": r"H_{\text{audit}}=\operatorname{SHA256}\!\left(v7 \parallel nonce \parallel H(src) \parallel H(x) \parallel H(\operatorname{share0}) \parallel H(\operatorname{share1})\right)",
    "3-6": r"X=\operatorname{share0}+\operatorname{share1}, \quad RGB=X\odot \sigma+\mu",
    "3-6a": r"z_c=\langle z_c\rangle_{0}+\langle z_c\rangle_{1}, \quad \hat{y}=\arg\max_c z_c",
    "3-7": r"L_i=0.299R_i+0.587G_i+0.114B_i, \quad \operatorname{OverExp}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left[L_i\ge 0.95\right]",
    "3-8": r"\operatorname{lap}(x,y)= -4L(x,y)+L(x-1,y)+L(x+1,y)+L(x,y-1)+L(x,y+1)",
    "3-9": r"\Delta_k=\left|q_k^{(\text{client})}-q_k^{(\text{server})}\right|, \quad \Delta_k \le 10^{-4}",
    "3-8a": r"\mu_{\text{lap}}=\frac{1}{HW}\sum_{x=1}^{H}\sum_{y=1}^{W}\operatorname{lap}(x,y), \quad \sigma_{\text{lap}}^{2}=\frac{1}{HW}\sum_{x=1}^{H}\sum_{y=1}^{W}\left(\operatorname{lap}(x,y)-\mu_{\text{lap}}\right)^{2}",
    "3-9a": r"F_{\text{payload}}=\operatorname{BLAKE2s}\!\left(H(\operatorname{share0}) \parallel \text{:} \parallel H(\operatorname{share1})\right)",
    "3-9b": r"n_{\text{ip}}(t;\Delta)=\sum_j \mathbf{1}\!\left[t-\Delta \le t_j \le t\right], \quad n_{\text{ip}}(t;\Delta)\le R_{\text{ip}}, \quad c_{\text{ip}}^{\text{inflight}}\le C_{\text{ip}}",
    "4-1": r"T_{\text{total}}=T_{\text{pre}}+T_{\text{guard}}+T_{\text{SPU}}+T_{\text{post}}",
    "4-2": r"V_{\text{total}}=\sum_{r}\left(V_{\text{send}}^{r}+V_{\text{recv}}^{r}\right), \quad \rho_V=\frac{V_{\mathrm{Transshield}}}{V_{\text{ref}}}, \quad \rho_T=\frac{T_{\mathrm{Transshield}}}{T_{\text{ref}}}",
}

PARAGRAPH_REPLACEMENTS = {
    "其中， 表示第 l 个剪枝 stage 的词元得分向量， 表示该 stage 允许保留的词元数， 表示由排序边界导出的保留阈值， 表示引入索引微扰后的编码键， 表示经编码键双调排序得到的索引排列， 表示第 i 个词元的保留掩码。报告正文中的安全选择原语  与安全比较原语  正是围绕这组关系完成的协议友好改写。":
        r"其中，$s^{(l)}$ 表示第 $l$ 个剪枝 stage 的词元得分向量，$K^{(l)}$ 表示该 stage 允许保留的词元数，$\tau^{(l)}$ 表示由排序边界导出的保留阈值，$e_i^{(l)}$ 表示引入索引微扰后的编码键，$\pi^{(l)}$ 表示经编码键双调排序得到的索引排列，$m_i^{(l)}$ 表示第 $i$ 个词元的保留掩码。报告正文中的安全选择原语 $F_{\text{mux}}$ 与安全比较原语 $F_{\text{less}}$ 正是围绕这组关系完成的协议友好改写。",
    "式（3-2c）进一步给出每个剪枝 stage 的保留预算约束；式（3-2d）与式（3-2e）则分别把保留判定与安全选择写为  和  原语。其中，表示第 l 个阶段的目标保留率，表示该阶段输入词元数，表示 keep-mask 作用后的实际保留词元数，表示第 i 个词元在该阶段的安全比较结果。":
        r"式（3-2c）进一步给出每个剪枝 stage 的保留预算约束；式（3-2d）与式（3-2e）则分别把保留判定与安全选择写为 $F_{\text{less}}$ 和 $F_{\text{mux}}$ 原语。其中，$\rho^{(l)}$ 表示第 $l$ 个阶段的目标保留率，$N^{(l)}$ 表示该阶段输入词元数，$N_{\text{out}}^{(l)}$ 表示 keep-mask 作用后的实际保留词元数，$b_i^{(l)}$ 表示第 $i$ 个词元在该阶段的安全比较结果。",
    "其中，表示裁剪后图像在通道 c、位置 (h,w) 的原始像素值，与分别对应 ImageNet 均值与标准差，为送入模型的归一化张量值；式（3-4a）中的与对应浏览器端生成的两份 additively split share，并在传输层分别落到 share0 与 share1。为避免浏览器端异常数值写入底层缓冲区，前端会先将归一化结果裁剪到 [-2, 2]，再以 little-endian Float32 方式写入 share 字节流；审计链则把源图哈希、归一化张量哈希与两份 share 哈希绑定到同一个 nonce 上。":
        r"其中，$p_{c,h,w}$ 表示裁剪后图像在通道 $c$、位置 $(h,w)$ 的原始像素值，$\mu_c$ 与 $\sigma_c$ 分别对应 ImageNet 均值与标准差，$x_{c,h,w}$ 为送入模型的归一化张量值；式（3-4a）中的 $\langle x_{c,h,w}\rangle_0$ 与 $\langle x_{c,h,w}\rangle_1$ 对应浏览器端生成的两份 additively split share，并在传输层分别落到 share0 与 share1。为避免浏览器端异常数值写入底层缓冲区，前端会先将归一化结果裁剪到 [-2, 2]，再以 little-endian Float32 方式写入 share 字节流；审计链则把源图哈希、归一化张量哈希与两份 share 哈希绑定到同一个 nonce 上。",
    "其中，服务端会先对 share 执行内存对齐复制，再将绝对值小于 的极小值刷新为零，以切断次正规数带来的微代码异常路径；随后检查 share 与重构张量是否存在非有限值，以及 share 幅值是否超过。对客户端与服务端分别得到的质量摘要 ，只在基础连续数值出现显著偏离时才标记可疑漂移，而不会仅凭离散风险标签不一致就直接判定篡改。":
        r"其中，服务端会先对 share 执行内存对齐复制，再将绝对值小于 $2^{-126}$ 的极小值刷新为零，以切断次正规数带来的微代码异常路径；随后检查 share 与重构张量是否存在非有限值，以及 share 幅值是否超过 $2$。对客户端与服务端分别得到的质量摘要 $q_k^{(\text{client})}$ 与 $q_k^{(\text{server})}$，只在基础连续数值出现显著偏离时才标记可疑漂移，而不会仅凭离散风险标签不一致就直接判定篡改。",
    "其中，由前端预处理、服务端控制面快检、SPU 密态执行与结果回传/审计收尾四部分组成；表示一次完整推理的双向总通信量，与分别表示相对通信量与相对时延比。":
        r"其中，$T_{\text{total}}$ 由前端预处理、服务端控制面快检、SPU 密态执行与结果回传/审计收尾四部分组成；$V_{\text{total}}$ 表示一次完整推理的双向总通信量，$\rho_V$ 与 $\rho_T$ 分别表示相对通信量与相对时延比。",
}

CHAPTER_MAP = {
    "第一章 作品概述": ("01_overview.tex", "作品概述"),
    "第二章 系统设计": ("02_design.tex", "系统设计"),
    "第三章 系统实现": ("03_implementation.tex", "系统实现"),
    "第四章 测试方案与结果分析": ("04_evaluation.tex", "测试方案与结果分析"),
    "第五章 创新性与局限性": ("05_innovation.tex", "创新性与局限性"),
    "第六章 复现与参赛声明": ("06_reproducibility.tex", "复现与参赛声明"),
}

SPECIAL_FILES = {
    "摘要": "00_abstract.tex",
    "参考文献": "90_references.tex",
    "附录A 关键代码实现": "99_appendix_a.tex",
}

SECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.+)$")
APPENDIX_SECTION_RE = re.compile(r"^(A\.\d+)\s+(.+)$")
REFERENCE_RE = re.compile(r"^\[(\d+)\]\s+(.+)$")
TABLE_RE = re.compile(r"^表\s*([0-9]+)\s*(.+)$")
FIGURE_CAPTION_RE = re.compile(r"^图([0-9]+-[0-9]+)\s*(.+)$")
EQUATION_TAG_RE = re.compile(r"^\(([0-9A-Za-z\-]+)\)$")


MAIN_TEMPLATE = r"""\documentclass[12pt,a4paper,UTF8]{ctexrep}
\usepackage{iftex}
\ifPDFTeX
  \errmessage{This project must be compiled with XeLaTeX. Please switch Overleaf compiler from pdfLaTeX to XeLaTeX.}
\fi

% Overleaf 免费版建议保持快速模式。
% 交稿前如需完整高保真编译，可把下面两项改为 false / true。
\newif\iffastdraft
\fastdrafttrue
\newif\ifusecustomfonts
\usecustomfontsfalse
\newif\ifincludeappendix
\includeappendixfalse

\iffastdraft
  \PassOptionsToPackage{draft}{graphicx}
\fi

\usepackage[left=2.54cm,right=2.54cm,top=2.54cm,bottom=2.54cm]{geometry}
\usepackage{fontspec}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\usepackage{amsmath,amssymb}
\usepackage{listings}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}

\ifusecustomfonts
  % 定稿模式：优先使用上传到 Overleaf 的宋体 / 黑体。
  \IfFontExistsTF{SimSun}{
    \setCJKmainfont{SimSun}
  }{
    \IfFileExists{fonts/simsun.ttc}{
      \setCJKmainfont{simsun.ttc}[Path=fonts/]
    }{
      \setCJKmainfont{FandolSong-Regular}
    }
  }
  \IfFontExistsTF{SimHei}{
    \setCJKsansfont{SimHei}
  }{
    \IfFileExists{fonts/simhei.ttf}{
      \setCJKsansfont{simhei.ttf}[Path=fonts/]
    }{
      \setCJKsansfont{FandolHei-Regular}
    }
  }
\else
  % 草稿模式：使用 TeX Live 自带中文字体，减少编译时间。
  \setCJKmainfont{FandolSong-Regular}
  \setCJKsansfont{FandolHei-Regular}
\fi
\IfFontExistsTF{Times New Roman}{
  \setmainfont{Times New Roman}
}{
  \setmainfont{TeX Gyre Termes}
}

\setstretch{1.5}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\captionsetup{font={normalsize},labelfont=bf}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  framerule=0.3pt,
  rulecolor=\color{black!30},
  backgroundcolor=\color{black!2},
  keepspaces=true
}

\ctexset{
  chapter = {
    name = {第,章},
    number = \chinese{chapter},
    format = \centering\heiti\zihao{-4},
    aftername = \quad,
    beforeskip = 0pt,
    afterskip = 6pt
  },
  section = {
    format = \heiti\zihao{-4},
    beforeskip = 6pt,
    afterskip = 6pt
  },
  appendix = {
    name = {附录},
    number = \Alph{chapter},
    format = \centering\heiti\zihao{-4},
    aftername = \quad
  }
}

\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

\newcommand{\reportequationplaceholder}[1]{
  \begin{equation}
  \tag{#1}
  \text{[请根据 Word 原稿补录该公式]}
  \end{equation}
}

\newcommand{\reportmissingfigure}[2]{
  \begin{figure}[htbp]
  \centering
  \fbox{\parbox[c][0.22\textheight][c]{0.82\textwidth}{\centering 图 #1 暂未找到可直接复用的图片资源}}
  \caption{#2}
  \end{figure}
}

\begin{document}
\zihao{-4}
\songti

\begin{titlepage}
\centering
{\heiti\zihao{2} 2026年全国大学生信息安全竞赛\par}
\vspace{1.2cm}
{\heiti\zihao{1} 作品报告\par}
\vspace{2.4cm}
\begin{flushleft}
\zihao{-4}
作品名称：\\[0.8cm]
电子邮箱：\\[0.8cm]
提交日期：\\[0.8cm]
\end{flushleft}
\vfill
\end{titlepage}

\tableofcontents
\clearpage

\chapter*{摘要}
\addcontentsline{toc}{chapter}{摘要}
\input{chapters/00_abstract}

\chapter{作品概述}
\input{chapters/01_overview}

\chapter{系统设计}
\input{chapters/02_design}

\chapter{系统实现}
\input{chapters/03_implementation}

\chapter{测试方案与结果分析}
\input{chapters/04_evaluation}

\chapter{创新性与局限性}
\input{chapters/05_innovation}

\chapter{复现与参赛声明}
\input{chapters/06_reproducibility}

\chapter*{参考文献}
\addcontentsline{toc}{chapter}{参考文献}
\input{chapters/90_references}

\ifincludeappendix
  \appendix
  \renewcommand{\thesection}{\Alph{chapter}.\arabic{section}}
  \chapter{关键代码实现}
  \input{chapters/99_appendix_a}
\fi

\end{document}
"""


README_TEMPLATE = r"""# Overleaf 报告工程

这个目录是从 `docs/密捷竞赛作品报告.docx` 自动导出的 LaTeX / Overleaf 工程骨架。

## 用法

1. 在 Overleaf 新建一个空项目。
2. 把这个目录里的全部文件上传到 Overleaf。
3. 将编译器切换为 `XeLaTeX`。
4. 默认是免费版快速模式：跳过图片实际渲染、使用 TeX Live 自带中文字体、暂不编译附录，通常更容易卡进免费编译时长。
5. 如果希望字体尽量贴近当前 Word 报告，请把 Windows 的 `simsun.ttc` 与 `simhei.ttf` 上传到 `fonts/` 目录，并把 `main.tex` 中的 `\usecustomfontsfalse` 改为 `\usecustomfontstrue`。
6. 如果需要完整附录，把 `\includeappendixfalse` 改为 `\includeappendixtrue`。
7. 如果需要完整图片渲染，把 `\fastdrafttrue` 改为 `\fastdraftfalse`。
8. 定稿前编译两次，让目录页码稳定。

## 当前状态

- 章节、正文、参考文献、附录代码节选已经批量迁移。
- 多数表格已转换为 `longtable`。
- 已存在的报告配图会复制到 `figures/` 并自动接入。
- Word 中的公式对象没有被逐项解析为 LaTeX 公式，当前以占位形式保留，后续需要手工补公式内容。
- 没有现成图片资源的图会生成占位框，后续可在 Overleaf 中替换。

## 重新导出

如果 Word 报告继续修改，可以在仓库根目录重新执行：

```bash
python3 tools/export_competition_report_to_latex.py
```
"""


def iter_block_items(document: DocumentObject) -> Iterable[Paragraph | Table]:
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def latex_escape(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + latex_escape_inline(m.group(1)) + "}", text)
    return text


def latex_escape_inline(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def wrap_paragraph(text: str) -> str:
    return latex_escape(text) + "\n"


def wrap_raw_paragraph(text: str) -> str:
    return text + "\n"


def render_table(caption: str, table: Table) -> str:
    rows = []
    max_cols = 0
    for row in table.rows:
        values = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        max_cols = max(max_cols, len(values))
        rows.append(values)

    if max_cols == 0:
        return ""

    padded_rows = []
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        padded_rows.append(padded)

    col_width = max(0.15, 0.96 / max_cols)
    col_spec = "".join(
        rf">{{\raggedright\arraybackslash}}p{{{col_width:.2f}\textwidth}}" for _ in range(max_cols)
    )

    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{" + latex_escape(caption) + r"}\\",
        r"\toprule",
    ]

    header = padded_rows[0]
    lines.append(" & ".join(r"\textbf{" + latex_escape(cell) + "}" for cell in header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(" & ".join(r"\textbf{" + latex_escape(cell) + "}" for cell in header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for row in padded_rows[1:]:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def render_figure(fig_no: str, caption: str) -> str:
    asset_name = FIGURE_MAP.get(fig_no)
    if asset_name:
        return "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.92\textwidth]{{figures/{asset_name}}}",
                r"\caption{" + latex_escape(caption) + r"}",
                r"\end{figure}",
                "",
            ]
        )

    return "\n".join(
        [
            r"\reportmissingfigure{" + latex_escape_inline(fig_no) + r"}{" + latex_escape(caption) + r"}",
            "",
        ]
    )


def render_equation(eq_no: str) -> str:
    body = EQUATION_MAP.get(eq_no)
    if body is None:
        return "\n".join(
            [
                r"\reportequationplaceholder{" + eq_no + "}",
                "",
            ]
        )
    return "\n".join(
        [
            r"\begin{equation}",
            body,
            rf"\tag{{{eq_no}}}",
            r"\end{equation}",
            "",
        ]
    )


def is_figure_caption(text: str) -> bool:
    match = FIGURE_CAPTION_RE.match(text)
    if not match:
        return False
    return len(text) <= 100


def render_reference_block(reference_lines: list[str]) -> str:
    lines = [r"\begin{thebibliography}{99}"]
    for ref in reference_lines:
        match = REFERENCE_RE.match(ref)
        if not match:
            continue
        ref_id = match.group(1)
        body = match.group(2)
        lines.append(rf"\bibitem{{ref{ref_id}}} {latex_escape(body)}")
    lines.append(r"\end{thebibliography}")
    lines.append("")
    return "\n".join(lines)


def render_code_block(text: str) -> str:
    return "\n".join(
        [
            r"\begin{lstlisting}",
            text.rstrip(),
            r"\end{lstlisting}",
            "",
        ]
    )


def ensure_directories(output_dir: Path) -> None:
    (output_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "fonts").mkdir(parents=True, exist_ok=True)


def copy_assets(output_dir: Path) -> None:
    for asset_name in FIGURE_MAP.values():
        src = ASSET_SOURCE / asset_name
        if src.exists():
            shutil.copy2(src, output_dir / "figures" / asset_name)


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def export(docx_path: Path, output_dir: Path) -> None:
    document = Document(docx_path)
    ensure_directories(output_dir)
    copy_assets(output_dir)

    chapter_buffers: dict[str, list[str]] = {SPECIAL_FILES["摘要"]: []}
    for value in CHAPTER_MAP.values():
        chapter_buffers[value[0]] = []
    chapter_buffers[SPECIAL_FILES["附录A 关键代码实现"]] = []
    references: list[str] = []

    current_file: str | None = None
    in_references = False
    pending_table_caption: str | None = None

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue

            if text == "摘要":
                current_file = SPECIAL_FILES["摘要"]
                in_references = False
                continue

            if text in CHAPTER_MAP:
                current_file = CHAPTER_MAP[text][0]
                in_references = False
                continue

            if text == "参考文献":
                current_file = None
                in_references = True
                continue

            if text == "附录A 关键代码实现":
                current_file = SPECIAL_FILES["附录A 关键代码实现"]
                in_references = False
                continue

            if in_references:
                if REFERENCE_RE.match(text):
                    references.append(text)
                continue

            if current_file is None:
                continue

            section_match = SECTION_RE.match(text)
            appendix_section_match = APPENDIX_SECTION_RE.match(text)
            table_match = TABLE_RE.match(text)
            figure_match = FIGURE_CAPTION_RE.match(text) if is_figure_caption(text) else None
            equation_match = EQUATION_TAG_RE.match(text)

            if section_match:
                chapter_buffers[current_file].append(r"\section{" + latex_escape(section_match.group(2)) + "}")
                chapter_buffers[current_file].append("")
                continue

            if appendix_section_match:
                chapter_buffers[current_file].append(r"\section{" + latex_escape(appendix_section_match.group(2)) + "}")
                chapter_buffers[current_file].append("")
                continue

            if table_match:
                pending_table_caption = f"表 {table_match.group(1)} {table_match.group(2)}".strip()
                continue

            if figure_match:
                fig_no = figure_match.group(1)
                caption = f"图{fig_no} {figure_match.group(2)}".strip()
                chapter_buffers[current_file].append(render_figure(fig_no, caption))
                continue

            if equation_match:
                chapter_buffers[current_file].append(render_equation(equation_match.group(1)))
                continue

            if text.startswith("注：") or text.startswith("注:"):
                chapter_buffers[current_file].append(r"\noindent\textit{" + latex_escape(text) + "}\n")
                continue

            if text.startswith("算法"):
                chapter_buffers[current_file].append(r"\noindent\textbf{" + latex_escape(text) + "}\n")
                continue

            if "\n" in block.text and (text.startswith("function ") or text.startswith("def ")):
                chapter_buffers[current_file].append(render_code_block(block.text))
                continue

            if text.startswith("function ") or text.startswith("def "):
                chapter_buffers[current_file].append(render_code_block(block.text))
                continue

            if text in PARAGRAPH_REPLACEMENTS:
                chapter_buffers[current_file].append(wrap_raw_paragraph(PARAGRAPH_REPLACEMENTS[text]))
            else:
                chapter_buffers[current_file].append(wrap_paragraph(text))

        else:
            if current_file is None:
                continue

            caption = pending_table_caption or "表格（标题待补）"
            chapter_buffers[current_file].append(render_table(caption, block))
            pending_table_caption = None

    for filename, lines in chapter_buffers.items():
        write_text(output_dir / "chapters" / filename, "\n".join(lines))

    write_text(output_dir / "chapters" / SPECIAL_FILES["参考文献"], render_reference_block(references))
    write_text(output_dir / "main.tex", MAIN_TEMPLATE)
    write_text(output_dir / "README.md", README_TEMPLATE)
    write_text(
        output_dir / ".gitignore",
        "\n".join(
            [
                "*.aux",
                "*.bbl",
                "*.bcf",
                "*.blg",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.run.xml",
                "*.synctex.gz",
                "*.toc",
            ]
        ),
    )
    write_text(output_dir / "fonts" / "README.txt", "请在 Overleaf 中上传 simsun.ttc 和 simhei.ttf 到此目录。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="将竞赛作品报告 docx 导出为 Overleaf / LaTeX 工程骨架。")
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX, help="Word 报告路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="导出目录")
    args = parser.parse_args()

    export(args.docx.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
