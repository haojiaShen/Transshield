# TransShield 复现说明

本文档用于复现当前正式交付口径中的计算与验证链路：医疗主线双向隐私演示、协议层异常输入验证，以及并发与重放守卫验证。

## 0. 最低可复现路径（10–20 分钟）

前提：模型目录和 Python 环境已经就位。

建议按以下最短路径验证核心能力：

1. 启动 Web 演示并访问页面，在浏览器端加载一张本地医疗样本图片。
2. 确认页面返回：
   - 分类结果
   - 质量保障结论
   - 审计摘要
   - 控制面耗时
3. 运行协议层异常输入验证脚本，确认生成 JSON 结果。
4. 运行一项控制面守卫检查，建议先跑 `duplicate_nonce`。

建议命令：

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=python3
export WEB_DEMO_HOST=127.0.0.1
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

另开终端执行：

```bash
cd /path/to/Transshield_final
python3 tools/web_demo_protocol_fuzz.py \
  --base-url http://127.0.0.1:7860
```

```bash
cd /path/to/Transshield_final
export WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC=1.5
python3 tools/web_demo_guard_stress.py \
  --base-url http://127.0.0.1:7860 \
  --checks duplicate_nonce \
  --window-reset-sec 0
```

## 1. 环境前提

- 仓库路径：`/path/to/Transshield_final`
- Python：建议 `Python 3.9+`
- 模型与数据：
  - 仓库默认不随 Git 分发全部模型权重与数据集
  - 需要按 `README.md` 中的说明补齐对应模型目录
## 2. 启动 Web 演示

本地启动：

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=python3
export WEB_DEMO_HOST=127.0.0.1
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

服务器启动：

```bash
cd /path/to/Transshield_final
export PYTHON_BIN=python3
export WEB_DEMO_HOST=0.0.0.0
export WEB_DEMO_PORT=7860
bash artifacts/server_inference_friendly_pack/run_web_demo.sh
```

预期结果：

- 控制台打印当前监听地址
- 浏览器可访问 `http://127.0.0.1:7860/` 或 `http://<server-ip>:7860/`

## 3. 运行医疗主线演示

- 打开页面后在浏览器端加载医疗样本图片
- 页面会触发浏览器工作线程预处理、本地质量评估、share 构造与后端安全推理
- 预期看到：
  - 分类结果
  - 质量保障结论
  - 审计摘要
  - 控制面耗时

## 4. 协议层异常输入验证

若要记录资源状态，先记下演示服务进程号，例如：

```bash
ps -ef | grep 'tools/transshield_chat_demo.py' | grep -v grep
```

运行模糊测试：

```bash
cd /path/to/Transshield_final
python3 tools/web_demo_protocol_fuzz.py \
  --base-url http://127.0.0.1:7860 \
  --server-pid <PID> \
  --out results/report_evidence/protocol_fuzz_evidence_custom.json
```

预期结果：

- 生成 `protocol_fuzz_evidence.json`、`protocol_fuzz_batch_*.json` 或指定输出文件
- 输出包含：
  - `interception_layer`
  - `fallback_layer`
  - `system_state`

## 5. 重放与并发守卫验证

建议先设置：

```bash
export WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC=1.5
```

然后运行：

```bash
cd /path/to/Transshield_final
python3 tools/web_demo_guard_stress.py \
  --base-url http://127.0.0.1:7860 \
  --server-pid <PID> \
  --out results/report_evidence/control_plane_guard_evidence_custom.json
```

预期结果：

- 生成 `control_plane_guard_evidence.json` 或指定输出文件
- 检查项包括：
  - 重复 nonce 并发重放
  - 相同载荷更换 nonce
  - 同 IP 并发占满
  - 短窗限频

## 6. 常见问题

- 如果 `run_web_demo.sh` 提示找不到模型目录，先补齐 `README.md` 中列出的模型文件。
- 如果需要核对后端参数，可运行：

```bash
python3 tools/transshield_chat_demo.py --help
```

- 如果需要重新核对第三方许可映射，检查：
  - `spu_vendored/LICENSE`
  - `spu_vendored/MODIFICATIONS.md`
  - `THIRD_PARTY.md`
