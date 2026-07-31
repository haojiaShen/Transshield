# Overleaf 报告工程

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
