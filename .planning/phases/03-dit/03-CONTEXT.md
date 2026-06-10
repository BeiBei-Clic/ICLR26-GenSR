# Phase 3: DiT 模型定义 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

实现 AdaLN-Zero 条件化的 DiT 模型，参照 Cola-DLM-main 的架构。输入 (z_t, t, prior_mu)，输出 512 维速度预测 v_psi(z_t, t; prior_mu)。DiT 用于学习 prior_mu → post_mu 的 Flow Matching 速度场。

**范围内：**
- DiT 模型类定义（参照 Cola-DLM 架构）
- AdaLN-Zero timestep 条件化
- Cross-attention 注入 prior_mu 条件
- Patchification + un-patchification（512 维 ↔ patches）
- 配置化的模型参数（层数、隐藏维度、头数、patch 数等）
- 模型前向传播验证（随机输入不出错，输出 shape 正确）

**范围外：**
- Flow Matching 训练逻辑（Phase 4）
- Euler 积分推理（Phase 5）
- 端到端推理管线（Phase 6）

</domain>

<decisions>
## Implementation Decisions

### 参照架构
- **D-01:** 参照 Cola-DLM-main/cola_dlm/modeling_cola_dit.py 的 DiT 架构，适配到 GenSR 的单向量 512 维潜空间。

### Patchification
- **D-02:** Patch 数量和维度通过配置参数指定。默认 16 patches × 32 dim。通过线性投影映射到隐藏维度。

### 条件注入
- **D-03:** prior_mu 通过 cross-attention 注入。z_t patches 做 self-attention 后，cross-attend 到 prior_mu（作为 K/V）。

### 模型规模
- **D-04:** 所有参数通过配置字典调整（DIT-05）。默认配置：6 层、隐藏维度 512、8 头、16 patches × 32 dim。

### 组件选择
- **D-05:** FFN 用 GELU（按 Cola 实际代码），非 SwiGLU。
- **D-06:** 归一化用 LayerNorm（按 Cola 实际代码），非 RMSNorm。

### 不需要的组件（GenSR 无序列）
- **D-07:** 移除所有序列相关机制：RoPE、块因果 mask、KV cache、NA 布局。

### Claude's Discretion
- DiT block 的具体实现细节
- TimestepEmbedding 的 sinusoidal 编码维度
- AdaLN-Zero 的初始化方式
- Cross-attention 的具体参数
- 代码文件位置和命名

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cola-DLM DiT 架构（核心参考）
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py` — DiT 完整实现
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py:135-159` — TimestepEmbedding
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py:310-348` — AdaLN 条件化
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py:355-364` — MLP (GELU FFN)
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py:479-531` — ColaDiTBlock
- `Cola-DLM-main/cola_dlm/modeling_cola_dit.py:548-712` — ColaDiTModel 主模型
- `Cola-DLM-main/cola_dlm/configuration_cola_dit.py` — 配置参数

### GenSR 已有组件
- `dit_train/data/latent_dataset.py` — Phase 1 交付的数据生成器
- `symbolicregression/model/cvae.py` — CVAE（latent_dim=512）

### 项目文档
- `.planning/research/PITFALLS.md` — Pitfall 1: Self-attention degenerates on single-token
- `Cola-DLM-main/docs/architecture_zh.md` — Cola 架构中文文档

</canonical_refs>

<code_context>
## Existing Code Insights

### Cola-DLM DiT 核心组件（需保留）
- AdaLN: `norm(x) * (1 + scale) + shift` (in), `x * gate + residual` (out)
- TimestepEmbedding: sinusoidal → Linear → SiLU → Linear → Linear
- MLP: Linear(dim, dim*4) → GELU → Linear(dim*4, dim)
- ColaDiTBlock: AdaLN(in) → Attention → AdaLN(out) + AdaLN(in) → MLP → AdaLN(out)

### Cola-DLM 序列组件（需移除）
- PatchIn1D/PatchOut1D: 替换为简单的 Linear patchify
- TextRotaryEmbedding: 移除（无序列位置）
- create_na_block_causal_mask: 移除（无块因果约束）
- KV cache: 移除（无自回归生成）

### GenSR 适配要点
- 输入: z_t (B, 512) → patchify → (B, num_patches, hidden_dim)
- 条件: prior_mu (B, 512) → 线性投影 → cross-attention K/V
- 时间步: t (B,) → sinusoidal → AdaLN 调制
- 输出: (B, num_patches, hidden_dim) → un-patchify → (B, 512)

</code_context>

<specifics>
## Specific Ideas

- DiT block 参照 ColaDiTBlock 但简化：去掉 txt_shape 相关逻辑，直接用标准 (B, L, D) 张量
- Cross-attention: prior_mu 投影为 (B, 1, hidden_dim) 作为 K/V，z_t patches 作为 Q
- 默认配置提供合理的初始值，用户可通过 config dict 调整

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-dit*
*Context gathered: 2026-06-09*
