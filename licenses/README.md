# licenses

本目录集中保存当前最终交付仓中需要单独保留的第三方许可证文本，便于评委、审计或后续归档时统一查阅。

## 当前保留文件

| 文件 | 适用对象 | 说明 |
|---|---|---|
| `licenses/DynamicViT-MIT.txt` | DynamicViT 相关来源代码 | 对应当前仓中保留的 DynamicViT 主干训练 / 推理代码与 `models/` 目录来源 |
| `licenses/OpenBumbleBee-Apache-2.0.txt` | OpenBumbleBee 相关来源代码 | 对应早期安全执行集成来源与部分协议层桥接代码来源 |
| `licenses/SPU-Apache-2.0.txt` | `spu_vendored/` | 为 `spu_vendored/LICENSE` 的镜像副本，便于在统一许可证目录中查阅 |

## 说明

- 当前仓库已经保留第三方来源总表：`THIRD_PARTY.md`。
- `spu_vendored/LICENSE` 仍是 vendored 子树内的原位许可证文件；`licenses/SPU-Apache-2.0.txt` 只是集中镜像，不替代原位文件。
- 当前仓库尚未单独声明整体项目的统一对外开源许可证，因此根目录暂不新增 `LICENSE`，避免对整仓再分发边界作出不准确声明。
