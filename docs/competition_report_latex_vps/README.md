# 作品报告 LaTeX（严格原格式 VPS 数据版）

本工程以用户提供的 93 页正式 PDF 为唯一版式基准。修订过程不再使用 LaTeX 重新排正文或表格，而是在原 PDF 的字符基线和单元格坐标内替换必要文字，再由 `pdfpages` 原尺寸封装全部页面。

## 严格版式约束

- 页面尺寸、页边距、页眉、页码及分页与原 PDF 一致。
- 正文保留原稿宋体 12 pt、23.4 pt 行距和 24 pt 首行缩进。
- 章节标题、表题、列宽、行高及单元格坐标沿用原稿；被文字遮盖的表格线按原坐标和原线宽恢复。
- 中文替换使用原稿同版 Windows SimSun，英文与数字使用原稿同版 Times New Roman。
- 表 4-5 的加粗单元格复用原稿“宋体填充 + 0.3432 pt 描边”的字重方式。
- 仅第 29、30、53、54、61、68 页包含数据或口径修订，其余页面不作排版变更。

## 内容边界

- 保持不变：算法、公式、模型结构、训练过程、离线全量指标、算子代理基准、鲁棒性证据、附录代码及展示代码。
- 当前 VPS 医疗复跑：原始代码、32 条医疗样本、两方 SPU colocated localhost、batch size 8。
- 金融性能数据：保留原报告历史记录，并明确不与当前 VPS 医疗结果作直接比例比较。
- 展示、推理、模型、配置和仓库正式结果文件均未修改。

详细数据来源与计数可比性限制见 `DATA_PROVENANCE.md`。

## 生成与编译

严格修补 PDF 已提供在：

```text
source/report_strict_vps.pdf
```

在具有原版 Windows 字体的 WSL 环境中可重新生成：

```bash
python3 build_strict_pdf.py
```

LaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf final_report.tex
```

最终文件：

```text
output/pdf/final_report.pdf
```

`main.tex` 与 `final_report.tex` 都只原尺寸引入严格修补后的 93 页 PDF，不会再次改变字体、缩进或表格几何。
