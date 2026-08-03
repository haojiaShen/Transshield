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
- 医疗交付场景：32 条医疗样本、两方 SPU colocated localhost、batch size 16。
- 金融边界压力验证场景：8 条固定样本，并记录端到端时延与双向通信量。
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

图 4-1 至图 4-8、图 5-1 均由代码在空白画布上重新生成：

- 图 4-4 的数据保存在 `performance_chart_data.json`，由
  `generate_performance_chart.py` 绘制。
- 其余统计图与评测矩阵的数据保存在 `report_figure_data.json`，由
  `generate_report_figures.py` 绘制。
- 图 4-3 直接读取仓库中的 524 条动态路径预测结果与 DenseNet121 验证集 CSV，
  重新统计直方分布，不从旧图反推柱高。

构建脚本会删除对应原图对象，再将完整生成结果按原页面矩形插入报告；整个流程
不读取、描摹或复用原图像素。附录界面截图与医学样本图仍保留原始证据，不做重绘。

LaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf final_report.tex
```

最终文件：

```text
output/pdf/final_report.pdf
```

`main.tex` 与 `final_report.tex` 都只原尺寸引入严格修补后的 93 页 PDF，不会再次改变字体、缩进或表格几何。
