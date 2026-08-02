# SPU 实验补丁

本目录中的补丁不属于默认 vendored 源码，也不会被项目构建流程自动应用。

| 补丁 | 用途 | 状态 |
|---|---|---|
| `cheetah_parallel_rlwe_packing.patch` | 并行 CHEETAH 矩阵响应的独立 RLWE packing 分组 | 实验否决：性能通过，medical32 精度稳定性未通过 |
| `spu093_eigen_github_mirror.patch` | 将无法访问的 GitLab Eigen 归档切换到相同 commit 的官方 GitHub mirror | 仅用于复现实验构建 |

详细结果见 `docs/evidence/spu_mlp_parallel_packing.md`。除非重新完成完整精度验收，否则不要把这些补丁加入默认构建。
