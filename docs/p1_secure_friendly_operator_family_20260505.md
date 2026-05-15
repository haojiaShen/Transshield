# P1 第四项：Secure-Friendly Operator Family 轻量抽象

最后更新：`2026-05-05`

## 1. 这份文档的定位

这一项当前不是要新造一套“大而全”的方法族名字，也不是要把当前系统重新包装成另一条主线。

当前更合适的收口方式是：

- 把已经支撑当前 official line 的几个 secure-friendly 近似点抽象成一组统一设计原则；
- 说明它们为什么能共同服务当前 `masking -> F_mux ; F_less ; secure sidecar/replay` 主线；
- 固定它们在当前项目里的正式角色：**deployable approximation family**，而不是过渡调试拼凑件。

## 2. 当前 family 的正式成员

当前正式成员只有三项：

1. `uniform` attention
2. `fixed_square` activation
3. `public_calibrated` layer norm

它们对应的当前 official line 是：

- plaintext bundle：
  - `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- secret runtime：
  - `secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1`

## 3. 统一设计原则

当前这三个算子/策略之所以可以被归成一组，不是因为它们都“近似了原始 ViT”，而是因为它们共同满足下面四条约束。

### 3.1 优先服务 secret execution，而不是优先复刻原始算子

当前主线的目标不是 exact ViT secure 复现，而是：

- 让 secret runtime 真正可运行；
- 让边界决策链能被稳定承接；
- 让 plaintext / secure 能维持同策略可对齐。

因此，这组 family 的第一原则是：

- **先保证协议友好、部署友好、验证友好，再追求更像原始 ViT。**

### 3.2 尽量避免 secret 路径中高成本或高不稳定算子

这组 family 的第二原则是：

- **优先减少 secret 路径里昂贵、脆弱、难校准的非线性与归一化负担。**

具体表现为：

- `uniform` attention 避免把注意力精度压力继续压在 softmax-like score materialization 上；
- `fixed_square` 避免把 GELU 一类更复杂非线性直接带进当前 secret 主线；
- `public_calibrated` layer norm 把一部分归一化压力转成当前可控、可校准、可复用的策略。

### 3.3 保持 same-policy plaintext reference 可构造

这组 family 的第三原则是：

- **任何 secret-friendly 改写，都必须能在 plaintext 侧构造同策略 reference。**

否则：

- secret 路径就只能做 demo；
- compare / replay / fairness 都会失去统一口径；
- `P0` 的验证闭环就会断掉。

当前这条原则已经被实际系统承接在：

- modified plaintext full-val
- plaintext vs secure replay compare
- fairness report
- guarded secret eval

## 3.4 优先保持当前 DynamicViT 主线的结构连续性

这组 family 不是另起炉灶的新骨干，而是服务于当前官方 `DynamicViT + masking-friendly pruning` 主线。

因此第四原则是：

- **近似替换应尽量保持训练栈、bundle、threshold、replay、secret runtime 之间的结构连续性。**

这也是为什么当前 family 只轻量收束成：

- `uniform`
- `fixed_square`
- `public_calibrated`

而不把更多长期研究近似一起塞进当前正式口径。

## 4. 三个成员各自承担什么角色

### 4.1 `uniform` attention

当前角色：

- 作为 secret-friendly attention policy；
- 降低当前 secret 路径对 attention score 数值处理的依赖；
- 为 static-depth / same-policy plaintext reference / secret replay 提供统一口径。

当前不应表述成：

- “对原始 attention 的严格等价替代”
- “已经证明是最优 attention 近似”

它当前只应被表述为：

- **当前 official line 中可部署、可验证的 attention approximation 选择。**

### 4.2 `fixed_square` activation

当前角色：

- 作为 current secret/plaintext shared activation choice；
- 降低当前部署链对更复杂激活近似的依赖；
- 保持 bundle、training、secret runtime 的一致性。

当前不应表述成：

- “已经证明比所有 learnable activation 更好”

它当前只应被表述为：

- **当前 official line 中最稳、最统一、最容易维持 compare 口径的 activation 选择。**

### 4.3 `public_calibrated` layer norm

当前角色：

- 作为 current secret runtime 的 layer norm policy；
- 通过校准把 secret 路径的归一化处理收束到一个可重复使用的运行策略；
- 为 batch=1 / guarded / isolated secret eval 提供当前最稳定的最小运行口径。

当前不应表述成：

- “它已经完成 full-depth exact LN secure 复现”

它当前只应被表述为：

- **当前 official secret runtime 中最小可交付的 layer norm policy。**

## 5. 为什么它们是一组 family，而不是三块散件

若只看单个实现点，`uniform`、`fixed_square`、`public_calibrated` 确实像三块散件。

但在当前 official line 里，它们实际上共同承担的是同一个职责：

- 把原始 ViT 中对 secure execution 不友好的若干高成本环节，
- 改写成一组可以被训练、导出、reference、replay、secret runtime 和 fairness 共同承接的 deployable approximation 组合。

因此，当前最准确的说法是：

- 它们不是彼此独立的 patch；
- 它们组成了当前系统的 **secure-friendly operator family**；
- 这组 family 的价值首先在于“可交付”，其次才是未来可能的“方法泛化”。

## 6. 当前能支持的说法

当前已经能支持：

- `uniform + fixed_square + public_calibrated` 已经构成当前 official line 的 deployable approximation family；
- 这组 family 已经同时进入：
  - training/bundle
  - modified plaintext full-val
  - fairness
  - secret runtime
  - acceptance 口径
- 它们不是过渡调试线，而是当前正式交付线的一部分。

## 7. 当前还不能支持的说法

当前还不能支持：

- “这已经是一套完整泛化的 secure operator theory”
- “family 中每个成员都已被独立 ablation 完整证明最优”
- “它已经可以替代未来所有更深 secret 路径或更精细算子研究”

这些都超出了当前项目阶段。

## 8. 当前结论

截至 `2026-05-05`，`secure-friendly operator family` 最合理的轻量抽象是：

- 把 `uniform`、`fixed_square`、`public_calibrated` 固定为当前 official line 的三件套；
- 明确它们的共同目标是：
  - 降低 secret execution 成本与脆弱性；
  - 保持 same-policy plaintext reference 可构造；
  - 服务当前 pruning-boundary secure execution 主线；
- 暂不把它扩写成更大、更泛化、但会分散主线注意力的新体系。
