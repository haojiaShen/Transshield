# SPU 实验补丁

本目录中的补丁不属于默认 vendored 源码，也不会被项目构建流程自动应用。

| 补丁 | 用途 | 状态 |
|---|---|---|
| `cheetah_parallel_rlwe_packing.patch` | 单分组保留内部并行，多分组并行 CHEETAH 矩阵响应的独立 RLWE packing | 已通过两次完整门槛；只作为隔离 runtime 的显式 batch16 吞吐补丁，不自动应用 |
| `spu093_eigen_github_mirror.patch` | 将无法访问的 GitLab Eigen 归档切换到相同 commit 的官方 GitHub mirror | 仅用于复现实验构建 |

最初候选见 `docs/evidence/spu_mlp_parallel_packing.md`，后续完整验收见 `docs/evidence/spu_hybrid_rlwe_public_alpha.md`。补丁不得覆盖原 SPU runtime，也不由默认构建自动应用。
