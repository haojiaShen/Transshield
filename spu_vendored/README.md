# spu_vendored — Transshield 专用 SPU 协议层源码

最后更新：`2026-05-13`

## 来源

从 `/home/yclcg/OpenBumbleBee` 仓库提取，用于在 Transshield 项目内独立维护和改造。

## 目录结构

```
spu_vendored/
├── spu_python/                    # SPU Python API（顶层接口）
│   ├── api.py                     # Runtime class（run/set_var/get_var）
│   ├── utils/                     # distributed 启动、ppd.device 等
│   ├── intrinsic/                 # 自定义 intrinsic 算子实现
│   └── ops/                       # groupby 等扩展算子
│
├── libspu/
│   ├── spu.proto                  # Protobuf 定义（RuntimeConfig、ExecutableProto 等）
│   ├── compiler/
│   │   └── tests/interpret/less.mlir   # less 操作的 MLIR 测试用例
│   ├── core/                      # 核心类型定义（暂留占位）
│   └── mpc/
│       ├── kernel.h               # 所有 MPC kernel 的基类定义
│       ├── cheetah/               # ★ 核心改造区域
│       │   ├── protocol.h/cc      # 协议注册入口
│       │   ├── arithmetic.h/cc    # Arithmetic share 算子族
│       │   ├── boolean.h/cc       # Boolean share 算子族
│       │   ├── conversion.h/cc    # A2B / B2A 转换
│       │   └── ...
│       ├── securenn/              # SecureNN 协议参考（header only）
│       └── spdz2k/                # SPDZ2k 协议参考（header only）
```

## 说明

此目录为 SPU v0.9 的协议层源码副本，供 Transshield 项目参考和调试使用。
Transshield 的核心改造（PredictorLG SPU 内部执行、bitonic sort、F_mux/F_less 算子）
均在 Python 层（JAX tracer 兼容方式）实现，不依赖 C++ 层修改。
