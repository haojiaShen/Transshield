# 协议候选评审（2026-05-19）

## 1. 本轮新增评审目标

本轮不再泛泛罗列“可能有帮助的底层协议”，而是只保留与当前正式主线直接相关、且能够映射到现有代码入口的三个候选：

1. **ABY2.0 式 mixed compare reactivation**
2. **Cheetah matmul packing policy 调整**
3. **positive-domain truncation scheduling**

评审标准固定为：

- 必须能映射到当前 `Transshield_final` 主仓已有运行链；
- 必须优先验证医疗 `dynamic secure pruning + full privacy` 代表路径；
- 若在代表路径下都不能形成稳定收益，不再继续扩到金融域；
- 结论优先看：
  - 是否能进入当前部署链；
  - 是否保持完全隐私；
  - 是否带来可重复的时间或通信收益；
  - 是否引入新的稳定性风险。

## 2. 候选一：ABY2.0 式 mixed compare reactivation

### 2.1 来源

- ABY2.0 论文：<https://www.usenix.org/system/files/sec21summer_patra.pdf>
- ABY2.0 slides：<https://www.usenix.org/system/files/sec21_slides_patra.pdf>

### 2.2 与当前仓的映射

当前仓里已经保留过一次 mixed compare 改造痕迹：

- `spu_vendored/libspu/spu.proto`
- `spu_vendored/libspu/mpc/cheetah/protocol.cc`
- 历史说明：
  - `docs/current_work_status.md`
  - `docs/transshield_modifications_improvements_log.md`
  - `docs/transshield_bumblebee_spu_modifications.md`

这说明它并不是一个“从 0 开始的新方向”，而是一个**曾经做过、但需要重新判断是否值得重新启用**的方向。

### 2.3 当前判断

结合仓内历史记录，mixed compare 的核心问题已经比较明确：

- 早期设计初衷是把 compare-heavy 子图从原生算术路径切到布尔路径；
- 但当前仓内历史记录已经确认：原生 `MsbA2B` 本身就是高效的 MSB-only 路径；
- 因此，ABY2.0 风格的 mixed compare 在当前 Cheetah 实现上，并没有带来先前设想中的数量级通信节省。

本轮重新评审 mixed compare 的意义，不是重新讲一遍理论，而是确认：

- 在 **2026-05-19 当前 packaged runtime** 下，它是否还能被直接启用；
- 如果还能启用，是否还有值得保留的真实收益。

## 3. 候选二：Cheetah matmul packing policy 调整

### 3.1 来源

- Cheetah 论文：<https://www.usenix.org/system/files/sec22-huang-zhicong.pdf>

### 3.2 与当前仓的映射

当前仓的 vendored Cheetah 已经不是“纯 OT matmul”状态，而是已经接入了 `CheetahDot` 路径：

- `spu_vendored/libspu/mpc/cheetah/protocol.cc`
- `spu_vendored/libspu/mpc/cheetah/state.h`
- `spu_vendored/libspu/mpc/cheetah/arith/cheetah_dot.cc`

因此，本轮不再把“是否使用 Cheetah 线性核”当成候选，而是把候选收束为：

- **是否保留当前默认的 packed matmul**
- **是否在当前 ViT / DynamicViT 形状下关闭 packing 反而更好**

对应开关：

- `CheetahConfig.disable_matmul_pack`

### 3.3 当前判断

这是一个可以直接落到当前运行链、且不需要改模型语义的候选：

- 优点：不影响双向隐私边界；
- 风险：如果当前 ViT 形状本来就依赖 packing 才能稳定运行，关闭后可能直接拖慢或失稳。

因此它适合做**服务器 triage**，不适合直接写进主线。

## 4. 候选三：positive-domain truncation scheduling

### 4.1 来源

这不是直接从某一篇论文照搬下来的结论，而是结合当前代码推出来的协议候选：

- `TruncA::proc` 已经显式接收 `SignType`
- `TruncateProtocol::Meta` 已经支持 `sign`
- 当前正式主线又大量使用 `fixed_square`

对应入口：

- `spu_vendored/libspu/mpc/cheetah/arithmetic.cc`
- `spu_vendored/libspu/mpc/cheetah/nonlinear/truncate_prot.h`
- `spu_vendored/libspu/mpc/cheetah/nonlinear/truncate_prot.cc`
- `models/dyvit.py`
- `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py`

### 4.2 候选设想

候选设想不是“全图所有 trunc 都强行当成正数”，而是：

- 若某些 trunc 位于明确非负的 square-family 子图后面；
- 那么这些 trunc 理论上可以绕过更保守的 unknown-sign 处理；
- 从而减少 wrap 判定或启发式补偿带来的协议开销。

### 4.3 当前判断

从方法角度看，这个方向**比 mixed compare 更值得继续研究**，原因有两点：

1. 它直接针对当前主线已经大量使用的 `fixed_square` 子图；
2. 它不是去重开一条历史分支，而是尝试在现有正式主线上做更细粒度的协议收缩。

但当前风险也很明确：

- 当前图里并不是所有 unknown-sign trunc 都真的可视为正域；
- 如果没有做到“只对确定非负的子图生效”，粗暴全局覆盖可能直接破坏稳定性；
- 因此它只能作为**受控实验钩子**，不能直接进默认配置。

## 5. 本轮候选优先级结论

按“与当前主线的贴合度 + 实验可落地性”排序，本轮优先级为：

1. `disable_matmul_pack`（最容易直接做服务器 triage）
2. `mixed_compare_mode`（已有历史代码，但要先确认 packaged runtime 是否还支持）
3. `positive-domain truncation scheduling`（研究价值高，但依赖可用的 SPU 重编链）

## 6. 本轮默认策略

- **直接做服务器 triage**：`disable_matmul_pack`
- **先尝试启用再决定是否值得重开**：`mixed_compare_mode`
- **只作为受控实验钩子，不默认计入主线**：`positive-domain truncation scheduling`
