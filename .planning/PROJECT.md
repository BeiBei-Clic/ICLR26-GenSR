# GenSR-DiT: 用 Flow Matching DiT 替换 CMA-ES 潜空间搜索

## What This Is

一个符号回归（Symbolic Regression）系统，在已有的 GenSR VAE 框架上，将推理阶段的 CMA-ES 进化搜索替换为 DiT（Diffusion Transformer）的 Flow Matching 推理。核心思路来自 Cola-DLM：在潜空间中用学习的先验传输替代无模型优化，实现更快、更准确的符号表达式发现。

现有 GenSR 的流程：数据 → CVAE 编码 → 潜向量 → CMA-ES 搜索 → 解码 → 表达式。替换后：数据 → CVAE 编码 → prior_mu → DiT 先验传输 → 更好的潜向量 → 解码 → 表达式。

## Core Value

用 DiT Flow Matching 替换 CMA-ES，在保持 VAE + Decoder 不变的前提下，将符号回归的推理从迭代进化搜索变为单次前向传输，大幅提升推理速度并可能提升表达式质量。

## Requirements

### Validated

- ✓ CVAE 编码器：将 (X, Y) 数据编码为 prior_mu（只看数据）和 post_mu（看数据 + GT 表达式） — existing
- ✓ 训练数据生成器：LatentPairDataset 在线生成 (prior_mu, post_mu) 对 — Validated in Phase 1
- ✓ FeatureFusion：将 (mu, logvar) 扩展为 (200, 512) 序列表示 — existing
- ✓ 自回归 Decoder：从潜向量生成符号表达式 token 序列 — existing
- ✓ gen2eq 评估管线：潜向量 → 解码 → BFGS 常数优化 → R² 评估 — existing
- ✓ CMA-ES 潜空间搜索：DiagonalCMAES 迭代优化潜向量 — existing（将被替换）

### Active

- [ ] DiT 模型：实现 AdaLN 条件化的 DiT，接收 (z_t, t, prior_mu)，预测速度场
- [ ] Flow Matching 训练：用 (prior_mu → post_mu) 对训练 DiT 学习速度场
- [ ] 推理管线：从 prior_mu 出发，经 DiT 先验传输得到优化后的潜向量，再走现有 Decoder
- ✓ 训练数据生成器：LatentPairDataset 在线生成 (prior_mu, post_mu) 对 — Validated in Phase 1
- [ ] 推理评估脚本：用 PMLB 数据集评估 DiT 方法 vs CMA-ES 方法的 R²、复杂度、推理时间
- [ ] DiT 的 checkpoint 保存与加载

### Out of Scope

- 重构 VAE 或 Decoder 架构 — 保持不动，只替换搜索过程
- 将 Decoder 改为非自回归 — 不在本次范围内
- 将潜空间从单向量改为序列 — 保持 512 维单向量
- 训练 Cola-DLM 本身 — 只迁移其 DiT + Flow Matching 的方法论

## Context

### Cola-DLM 的关键思路

Cola-DLM（Continuous Latent Diffusion Language Model, arXiv:2605.06548）的核心：
1. VAE 将文本映射到连续潜空间（潜序列 `z_0 ∈ R^{n×d}`）
2. DiT 学习潜空间上的 Flow Matching 先验 `p_psi(z_0)`
3. 推理时：噪声 → DiT 分块先验传输 → 干净潜向量 → VAE 解码回文本
4. DiT 架构：AdaLN 条件化（sinusoidal timestep）、SwiGLU FFN、RoPE、分块因果注意力
5. Flow Matching：Euler 积分 `x_{t-Δ} = x_t - Δ/T * v_psi(z_t, t)`

### GenSR 现有架构

- CVAE (`CVAEDE_SR`)：Transformer encoder → 池化 → bottleneck → prior/posterior (mu, logvar)
- Latent dim = 512，单向量
- FeatureFusion：sample_gaussian_multi → mean → expand_proj → (200, 512)
- Decoder：自回归 Transformer，cross-attention 到 src_enc（经 latent_proj 的潜向量）
- CMA-ES (`DiagonalCMAES`)：对角协方差 CMA-ES，50-100 代迭代，每代需解码 + BFGS + R² 评估

### 迁移方案

**训练**：
- 数据来源：现有训练集中的 (X, Y, GT 表达式) 三元组
- 对每个样本：CVAE 编码得到 prior_mu（只看数据）和 post_mu（看数据+表达式）
- Flow Matching 目标：`v_psi(z_t, t; prior_mu)` 预测从 prior_mu 到 post_mu 的速度场
- 插值路径：`z_t = (1-t) * prior_mu + t * post_mu`（或 OT 路径）

**推理**：
1. CVAE 编码 (X, Y) → prior_mu
2. DiT 先验传输：从 prior_mu 出发，Euler 积分若干步，得到优化后的潜向量 z_opt
3. FeatureFusion(z_opt, logvar) → Decoder → 符号表达式
4. 可选：对生成的表达式做 BFGS 常数优化（保留现有的 refine 管线）

**DiT 架构适配**：
- 输入：z_t (512-dim)，reshape 为 (1, 512) 或 patchify
- 条件：timestep t 通过 AdaLN，prior_mu 通过 cross-attention 或拼接
- 输出：速度预测 (512-dim)
- 使用 AdaLN-Zero 初始化、SwiGLU FFN 等 Cola 标准组件

## Constraints

- **VAE/Decoder 冻结**：训练 DiT 时，CVAE、FeatureFusion、Decoder 的权重全部冻结
- **潜空间不变**：DiT 必须在同一个 512 维潜空间内操作
- **PyTorch**：使用 PyTorch 实现，与现有代码库保持一致
- **单 GPU 训练**：DiT 规模不大（512 维输入），单卡足够
- **评估对比**：必须在 PMLB 数据集上与 CMA-ES 方法做定量对比

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 用 prior_mu → post_mu 作为 Flow 目标 | post_mu 是 CVAE 在看到 GT 表达式后的编码，代表"最优潜向量"；prior_mu 只看数据 | — Pending |
| 用 prior_mu 作为 DiT 的数据条件 | prior_mu 已编码数据信息，避免重复传入原始数据 | — Pending |
| 直接用 DiT 架构而非 MLP | 用户希望迁移 Cola 的 DiT 方法论，DiT 的 AdaLN 机制天然适合 Flow Matching | — Pending |
| 保持 Decoder + BFGS 管线 | 只替换搜索部分，最大化复用现有代码 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-09 after initialization*
