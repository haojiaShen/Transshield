# 场景适配与部署档位（城市 / 县域 / 乡村）

## 1. 先给结论

- 当前不建议把“静态剪枝 / true static no-pruning”继续保留为第二条正式主线。
- 更合理的正式口径是：
  - 医疗：`dynamic secure pruning + public threshold calibration + full privacy`
  - 金融：`dynamic secure pruning + LRD rank192 + full privacy`
- 正式落地拓扑固定为两方：
  - 医院 / 银行等业务使用方侧服务器
  - AI 公司侧服务器
- 不同发展状况下允许调整的是阈值、batch、depth 和压缩配置，而不是把两方协同改成“三方托管”。
- 不同发展状况下的适配，优先靠以下四个公开可调节项完成，而不是再维护第二条正式静态主线：
  1. public threshold calibration
  2. batch size
  3. static depth limit
  4. bundle 压缩 / operator preset

## 2. 为什么不把静态线继续保留成正式主线

### 2.1 医疗

- 医疗域真正决定能否落地的，不是“静态 vs 动态”二选一，而是：
  - dynamic secure pruning 是否全隐私成立
  - dynamic path 是否有正确的公开阈值
- 2026-05-19 已明确：
  - dynamic secure pruning 全隐私成立
  - 只要用 dynamic-path threshold，全量验证集 `92.7481%`，`32` 样本复核集 `93.75%`
- 因此，医疗域不需要为了“适配弱资源场景”回退成静态主线；直接用 dynamic 主线，再调 batch / depth 即可。

### 2.2 金融

- 金融域 2026-05-19 dynamic vs static same-condition pair 结果：
  - dynamic：`105.16s/sample`
  - static：`103.64s/sample`
  - 预测完全一致：`argmax/threshold match = 1.0 / 1.0`
- 静态只快 `1.52s/sample`，收益过小，不足以支撑第二条正式主线。

结论：

- 静态线可以保留为 fallback
- 但不建议继续写成“正式并列主线”

## 3. 推荐的适配手段

### 3.1 手段 A：public threshold calibration

这是最重要的跨场景适配项。

原因：

- 医疗动态路径的概率边界与 static path 不同
- 不同医院 / 不同群体 / 不同抽样分布，也可能导致部署边界轻微偏移

建议做法：

- 每个部署域都保留一小块公开验证集
- 只在公开 logits 上重新搜索 threshold
- 不改变 secure graph，不改变隐私边界

当前已验证阈值：

| 领域 | 配置 | threshold | 全量验证集 / 复核配置结果 |
|---|---|---:|---:|
| 医疗 | depth12 dynamic | `0.6226428151` | 全量验证集 `92.7481%` |
| 医疗 | depth10 dynamic | `0.6619606018` | 全量验证集 `92.7481%` |
| 金融 | dynamic / static `8` 样本压力验证 | `0.5` | 当前 `100%` |

### 3.2 手段 B：batch size

- 适配对象：同一模型，不同资源档位
- 原则：
  - 高资源场景：增大 batch，压低 sec/sample
  - 低资源场景：减小 batch，保证不 OOM

当前证据：

- 医疗 `32` 样本复核集、批次规模 `12`、深度 `12` 已 OOM
- 医疗 `32` 样本复核集、批次规模 `8` 的 depth12 / depth10 配置已成功

所以当前服务器上的正式建议是：

- 不把 `batch12` 当成大样本部署默认值
- 以 `batch8` 作为稳态默认档

### 3.3 手段 C：static depth limit

这是真正适合“城市 / 县域 / 乡村”分档的手段。

推荐逻辑：

- 城市场景（资源较好）
  - 医疗：优先 `depth12 dynamic`
  - 金融：优先 `depth12 dynamic`
- 县域 / 中等资源
  - 医疗：`depth10 dynamic`
  - 金融：仍可用 `depth12 dynamic`，因为当前 rank192 LRD 已压缩参数
- 乡村 / 边缘资源
  - 医疗：`depth10 dynamic + smaller batch`
  - 金融：`depth12 dynamic + smaller batch`；若必须固定 shape，再考虑启用 static fallback

### 3.4 手段 D：bundle 压缩与算子 preset

- 医疗：
  - 主适配手段是 `depth10`
  - 不建议再切回 LRD 主线
- 金融：
  - 主适配手段是 `LRD rank192 merged`
  - 它已经把参数压到 `68.39%`

## 4. 推荐部署档位

## 4.1 城市 / 三甲 / 较强算力

### 医疗

- 主线：`depth12 dynamic secure pruning`
- 阈值：`0.6226428151`
- 理由：优先精度与语义完整度

### 金融

- 主线：`LRD rank192 + dynamic secure pruning`
- 阈值：`0.5`
- 理由：当前 dynamic 已全隐私成立，且不需要为了极小速度差再拆第二条主线

## 4.2 县域 / 中等算力

### 医疗

- 主线：`depth10 dynamic secure pruning`
- 阈值：`0.6619606018`
- 理由：当前服务器 `32` 样本正式复核配置已验证 `86.91s/sample` 且 `93.75%`

### 金融

- 主线：`LRD rank192 + dynamic secure pruning`
- 调参：优先减 batch，不优先切静态

## 4.3 乡村 / 边缘资源

### 医疗

- 主线仍建议保持 dynamic
- 先降 batch，再考虑进一步工程裁剪
- 不建议第一反应就切回静态主线

### 金融

- 默认仍建议保持 dynamic
- 只有在以下场景才启用 true static no-pruning fallback：
  - 必须严格固定 shape
  - 本地运维只接受最保守图结构
  - 对 `1~2s/sample` 的差异也非常敏感

## 5. 最终建议

- **正式报告**：两域都写成 dynamic secure pruning + full privacy
- **正式部署**：固定写成“业务使用方侧服务器 + AI 公司侧服务器”两方协同，不额外引入第三方可信服务器
- **工程附注**：
  - 医疗用 `depth10 / depth12` + threshold calibration 做分档
  - 金融用 `dynamic` 作为默认、`true static no-pruning` 作为 fallback
- **不建议**：
  - 再把静态线抬回正式并列主线
  - 再用“静态 / 动态双正式主线”增加报告复杂度
