# GenSR-Cola: 用流匹配替换潜空间进化搜索

## 项目概述

将 GenSR 符号回归项目中的潜空间进化搜索（CMA-ES）替换为 ColaDLM 风格的 Flow Matching 推理，实现从噪声到方程潜在表示的直接映射。

## 背景

### 当前架构（GenSR）
- **预训练**：CVAE 将 (X, Y) 数据对编码到 512 维潜空间，解码回方程序列
- **推理**：CMA-ES 进化算法在潜空间迭代搜索最优潜在表示（120 次迭代，80 个体种群）
- **关键模块**：NumericalEmbedder → CVAEDE_SR → FeatureFusion → TransformerDecoder
- **关键文件**：`train.py`, `LSO_fit.py`, `cma_es_modular.py`, `model.py`

### 目标架构（ColaDLM 启发）
- **Stage 1（保持）**：CVAE 预训练，学习稳定的 latent-text 映射
- **Stage 2（新增）**：Flow Matching 模型学习条件先验 p(z₀ | data)
- **推理（替换 CMA-ES）**：噪声 → ODE 求解 → 潜在表示 → 解码 → 方程

### 核心映射关系

| ColaDLM 概念 | GenSR 对应 | 说明 |
|---|---|---|
| Text VAE | CVAE + FeatureFusion + Decoder | 保持不变 |
| Block-Causal DiT | 简化 MLP/Transformer | 我们的潜空间是 512 维向量，不需要 block 结构 |
| Flow Matching 先验 | 条件流匹配模型 | 输入：z_t + t + data_embedding，输出：向量场 |
| CMA-ES 搜索 | ODE 推理 | z₁ ~ N(0,I) → ODE 求解 → z₀ → 解码 |
| BFGS 常数优化 | 保持不变 | 流匹配输出后仍做 BFGS |

## 技术关键点

### Flow Matching 训练
- 数据来源：CVAE 训练时同时产出 prior_mu（数据条件）和 post_mu（真实后验）
- 目标：学习从 N(0,I) 到后验分布的条件传输
- 条件向量场：v_ψ(z_t, t, c)，其中 c = prior_mu（数据嵌入）
- 训练目标：L_FM = E[‖v_ψ(z_t, t, c) - u_t(z_0, z_1)‖²]

### Flow Matching 推理
- 采样 z₁ ~ N(0, I)
- Euler/Heun 求解 z₀ = Φ_{0←1}(z₁; c)
- 多次采样取最优（替代 CMA-ES 的种群搜索）
- 通过 FeatureFusion → Decoder 解码方程
- BFGS 常数优化

### 适配要点
1. 我们的 latent 是单个 512 维向量（非序列），Flow Matching 模型更简单
2. 条件信息 c = prior_mu 由 CVAE 编码器提供
3. 训练时需要收集 (prior_mu, post_mu) 对作为 Flow Matching 的训练数据
4. 推理时支持多次采样（N_sample 次）取最优 R²

## 约束
- 保持预训练 CVAE 不变（或仅微调），复用已有权重
- 保持 `experiments/pmlb/pmlb_batch_inference.py` 接口兼容
- 保持 BFGS 常数优化环节
- Flow Matching 推理应显著快于 CMA-ES（从 120 次迭代降到 ODE 求解 + 多次采样）
