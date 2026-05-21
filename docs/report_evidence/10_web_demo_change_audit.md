# 网页展示与控制面闭环变更审计（2026-05-20）

## 1. 本次变更结论

- **展示叙事已收束**：医疗是唯一正式落地场景；金融只保留为边界压力测试区
- **医疗前端已补控制面**：浏览器 worker 负责本地 DQA、审计哈希链和分片生成
- **医疗后端已补权威快检**：share 哈希校验、位级检查、张量合法性检查、服务端 DQA、审计落盘
- **金融域已降级**：只保留内置压力样本和批量压力实测，不再写成第二条正式主线

## 2. 前端变更

### 2.1 `web_demo/index.html`

- 页面标题、Hero、副标题、金融区域标题、创新点文案全部改为新口径
- 医疗结果区新增：
  - `quality_assurance`
  - `audit`
  - `control_plane_metrics`
- 医疗运行链改为：
  - 主线程只负责文件选择、预览和请求编排
  - worker 负责预处理、DQA、哈希、分片
  - 任意时刻只允许一个活跃医疗 worker 和一个活跃医疗请求

### 2.2 `web_demo/control_plane_worker.js`

- 新增一次性 worker
- 新增图片头部尺寸嗅探
- 新增显式 little-endian 浮点序列化
- 新增本地质量摘要、审计哈希链与控制面耗时采集
- 新增 `bitmap.close()` 与 canvas 尺寸清零，防止隐性显存滞留

## 3. 后端变更

### 3.1 `tools/transshield_chat_demo.py`

- 移除 `cgi.FieldStorage`
- multipart 处理改为：
  - raw body 硬限制读取
  - boundary / header / 字段集预检
  - `email` 标准库做结构校验
  - share payload 直接从 `raw_body` 按偏移切原始字节
- 医疗控制面新增：
  - JSON 文本零拷贝长度门
  - 安全 `json.loads` 钩子
  - share 哈希校验
  - 次正规数检测
  - 对齐复制后再做 NumPy 运算
  - 服务端 DQA 与阻断 / 告警裁决
  - replay guard 与 per-IP / global inflight 限制
  - 审计日志 `audit_events.jsonl` / `audit_rejections.jsonl`

## 4. 已知局限

- 当前 demo 采用同步阻塞式 `ThreadingHTTPServer`
- 请求一旦通过前置校验并进入长耗时 SPU 计算，服务端**不能主动感知客户端中途断连并终止该计算**
- 因此当前原型的 inflight 限额只能作为 demo 级保护，不能视为生产级长任务抗 DoS 方案
- 生产环境必须迁移到异步框架或任务队列 / 轮询架构
- 对恶意超大流上传，系统以 **TCP 阻断优先**；客户端可能收不到应用层 `413`

## 5. 自动化黑盒验证入口

- 协议 / multipart 结构模糊测试：
  - `python tools/web_demo_protocol_fuzz.py --base-url http://127.0.0.1:7860`
- 并发守卫 / 重放 / inflight / 限频测试：
  - 推荐先以 `WEB_DEMO_TEST_ACCEPTED_SLEEP_SEC=1.5` 启动 demo server
  - 再运行 `python tools/web_demo_guard_stress.py --base-url http://127.0.0.1:7860`
- worker 前端自测页：
  - `http://127.0.0.1:7860/worker_selftest.html`
  - 可直接验证超大 PNG 头、畸形 JPEG、截断 WebP 和连续 worker 重建

## 6. 当前判断

- 前端展示内容与文档描述的“控制面闭环”已一致
- 医疗域现在能够显式展示：
  - 本地 DQA
  - 审计链摘要
  - 服务端权威快检
  - 控制面开销
- 金融域现在只承担边界压力验证职责，不再与医疗并列为正式落地主线
