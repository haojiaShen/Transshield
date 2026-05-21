# TransShield - 双向隐私安全推理系统

## 项目简介

TransShield 是一套面向医疗影像隐私推理的双向隐私安全推理系统，基于动态视觉 Transformer（DynamicViT）实现，在不引入可信第三方的前提下完成两方安全计算（2PC）原型落地。

### 核心特性

- **双向隐私保护**：医院侧数据不出明文，AI公司侧模型参数不出明文
- **动态剪枝保留**：通过协议友好重写，在密态执行环境中保留动态词元剪枝能力
- **轻量控制面**：浏览器工作线程本地预处理 + 服务端权威快检 + 审计哈希链闭环
- **可验证交付**：17类协议层异常输入与控制面黑盒验证证据

### 正式交付指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 医疗阈值精度 | 92.7481% | 524张全量验证集 |
| 医疗AUC | 0.9639 | 受试者工作特征曲线下面积 |
| 端到端时延 | 89.06秒/样本 | 32条部署验证样本 |
| 双向通信量 | 84.47 GiB | 32条样本双向总通信 |
| 鲁棒性验证 | 17/17 通过 | 协议层异常输入与控制面守卫 |

## 仓库结构

```
Transshield_final/
├── web_demo/                              # 前端展示
│   ├── *.html                             # 各页面（设计、实现、结果、创新、演示）
│   ├── site.css                           # 全局样式
│   ├── site.js                            # 全局脚本
│   ├── control_plane_worker.js            # 浏览器Worker（报告附录A.1）
│   └── assets/                            # 图件资产
│       ├── system_trust_boundary_topology.png   # 图2-1 系统部署图
│       ├── software_flow_sequence.png           # 图2-2 软件流程图
│       ├── medical_threshold_calibration_shift.png  # 图4-1 阈值校准对比
│       └── robustness_guard_matrix.png          # 图4-2 鲁棒性验证矩阵
│
├── tools/                                 # 核心工具脚本
│   ├── transshield_chat_demo.py           # Web演示后端（报告附录A.3）
│   ├── web_demo_protocol_fuzz.py          # 协议层fuzz验证（报告附录A.4）
│   ├── web_demo_guard_stress.py           # 控制面守卫测试
│   ├── generate_competition_report.py     # 作品报告生成
│   ├── generate_report_figures.py         # 图件生成
│   └── README.md                          # 工具说明
│
├── integrations/                          # 安全计算集成
│   └── openbumblebee/
│       └── e2e_secure_vit/
│           └── spu_static_vit.py          # SPU安全推理（报告附录A.2）
│
├── spu_vendored/                          # SPU修改
│   ├── LICENSE                            # Apache 2.0许可证
│   ├── MODIFICATIONS.md                   # 修改说明
│   └── libspu/                            # SPU库修改
│
├── artifacts/                             # 交付资产
│   ├── server_inference_friendly_pack/    # 服务器推理配置
│   ├── web_demo_assets/                   # Web演示资产
│   └── frozen_bundle_*/                   # 冻结的模型包
│
├── models/                                # 模型定义
│   └── dyvit.py                           # DynamicViT实现
│
├── training_compat/                       # 训练兼容层
│   ├── main.py                            # 训练入口
│   ├── engine.py                          # 训练引擎
│   ├── losses.py                          # 损失函数
│   └── models/                            # 模型定义
│
├── configs/                               # 配置文件
│   └── openbumblebee/                     # OpenBumbleBee配置
│
├── scripts/                               # 辅助脚本
│   └── README.md                          # 脚本说明
│
├── docs/                                  # 文档
│   ├── 密捷竞赛作品报告.docx              # 最终作品报告
│   ├── README_REPRODUCE.md                # 复现说明（报告6.1节）
│   ├── transshield_innovation.md          # 创新点文档
│   ├── data_source_policy.md              # 数据来源政策
│   └── current_work_status.md             # 当前工作状态
│
├── .gitignore                             # Git忽略规则
├── AGENTS.md                              # 仓库规范
└── LICENSE                                # 项目许可证
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+（用于图件生成）
- 支持的浏览器：Chrome 90+, Firefox 88+, Safari 15+

### 启动Web演示

```bash
# 启动后端服务
python tools/transshield_chat_demo.py

# 访问演示页面
# http://localhost:7860/
```

### 运行协议层验证

```bash
# 运行协议层fuzz测试
python tools/web_demo_protocol_fuzz.py

# 运行控制面守卫测试
python tools/web_demo_guard_stress.py
```

### 最低可复现路径（10-20分钟）

详见 `README_REPRODUCE.md`，包含：
1. 启动Web演示并加载医疗样本
2. 运行协议层异常输入脚本
3. 验证控制面守卫检查

## 报告中的关键公式

### 式(3-1) - 词元保留边界
$$\tau_l = \text{TopKBoundary}(s_l, K_l)$$

### 式(3-2) - 安全选择原语
$$\tilde{h}'_i^l = m_i^l \cdot h_i^l$$

### 式(3-3) - 输入归一化
$$x'_{c,h,w} = \text{clip}\left(\frac{p_{c,h,w} - \mu_c}{\sigma_c}, -2, 2\right)$$

### 式(3-4) - 秘密分享
$$X = \text{share0} + \text{share1}, \quad r_{c,h,w} \sim \mathcal{U}(-2, 2)$$

### 式(3-5) - 审计哈希链
$$H_{\text{audit}} = \text{SHA256}(\text{v7} \parallel \text{nonce} \parallel \cdots)$$

## 安全边界说明

### 保护范围
- ✅ 用户输入数据（医疗影像）
- ✅ 模型参数（DynamicViT权重）
- ✅ 中间推理状态（词元得分、keep-mask）
- ✅ 最终分类结果（仅返回类别）

### 不覆盖范围
- ❌ 恶意参与方串谋
- ❌ 模型抽取攻击
- ❌ 输出侧反推风险
- ❌ 生产级DoS防护（当前为原型级）

## 第三方依赖

| 依赖 | 许可证 | 用途 |
|------|--------|------|
| SecretFlow SPU | Apache 2.0 | 安全计算底座 |
| OpenBumbleBee | Apache 2.0 | 两方安全推理框架 |
| JAX | Apache 2.0 | 数值计算 |
| NumPy | BSD-3 | 数组操作 |
| PyTorch | BSD-3 | 模型训练 |

## 参与方说明

- **数据方（医院）**：提供医疗影像，在本地浏览器完成预处理与秘密分享
- **模型方（AI公司）**：提供DynamicViT模型参数，参与SPU密态计算
- **服务端控制面**：负责协议预检、权威快检与审计落盘
- **SPU执行域**：完成双向隐私约束下的动态安全推理

## 相关论文

1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
2. Rao et al. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification" (NeurIPS 2021)
3. Zeng et al. "MPCViT: Searching for Accurate and Efficient MPC-Friendly Vision Transformer" (ICCV 2023)
4. Lu et al. "BumbleBee: Secure Two-party Inference Framework for Large Transformers" (NDSS 2025)
5. Ma et al. "SecretFlow-SPU: A Performant and User-Friendly Framework for Privacy-Preserving Machine Learning" (USENIX ATC 2023)

## 联系方式

- 作品报告：`docs/密捷竞赛作品报告.docx`
- 复现说明：`README_REPRODUCE.md`
- 创新点文档：`docs/transshield_innovation.md`

## 许可证

本项目基于 Apache License 2.0 许可证开源。详见 `LICENSE` 文件。

## 致谢

感谢 SecretFlow SPU 和 OpenBumbleBee 团队提供的安全计算基础设施。
