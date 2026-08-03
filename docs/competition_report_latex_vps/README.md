# 作品报告 LaTeX（严格原格式 VPS 数据版）

本工程以用户提供的 93 页正式 PDF 为唯一版式基准。修订过程不再使用 LaTeX 重新排正文或表格，而是在原 PDF 的字符基线和单元格坐标内替换必要文字，再由 `pdfpages` 原尺寸封装全部页面。

## 严格版式约束

- 页面尺寸、页边距、页眉、页码及分页与原 PDF 一致。
- 正文保留原稿宋体 12 pt、23.4 pt 行距和 24 pt 首行缩进。
- 章节标题、表题、列宽、行高及单元格坐标沿用原稿；被文字遮盖的表格线按原坐标和原线宽恢复。
- 中文替换使用原稿同版 Windows SimSun，英文与数字使用原稿同版 Times New Roman。
- 表 4-5 的加粗单元格复用原稿“宋体填充 + 0.3432 pt 描边”的字重方式。
- 修订页只替换经审计确认需要更新的文字、数值或图像对象，其余页面对象保持原样。

## 内容边界

- 保持不变：算法、模型结构、训练过程、离线全量指标、鲁棒性证据、附录代码及展示代码。仅将 token 数量公式的取整符号纠正为与实现一致的向下取整。
- 医疗交付场景：32 条医疗样本、两方 SPU colocated localhost、batch size 16。
- 金融边界压力验证场景：8 条固定样本，并记录端到端时延与双向通信量。
- 展示、推理、模型、配置和仓库正式结果文件均未修改。

详细数据来源与计数可比性限制见 `DATA_PROVENANCE.md`。
报告所用 VPS 环境、医疗与金融运行字段保存在 `vps_report_data.json`；其中同时记录原始证据路径和 SHA-256。生成脚本不依赖未跟踪的 VPS 备份目录。

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
不读取、描摹或复用原图像素。附录中涉及配置、指标、剪枝数量或运行性能的界面
同样由 `generate_report_ui_snapshots.py` 根据正式 JSON 在空白画布上生成；不含
数值口径的登录、流程和医学样本界面保留原始证据。

LaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf final_report.tex
```

数据审计：

```bash
python3 audit_report_data.py
```

审计脚本会交叉检查医疗/金融完整运行、524 条全量验证、代理基准、图表数据、
token 计数、17 类鲁棒性记录及最终 PDF 中的关键数值，并拒绝已知旧口径残留；
若本机存在原始 VPS JSON，还会校验其 SHA-256 及摘录字段。

最终文件：

```text
output/pdf/final_report.pdf
```

`main.tex` 与 `final_report.tex` 都只原尺寸引入严格修补后的 93 页 PDF，不会再次改变字体、缩进或表格几何。
