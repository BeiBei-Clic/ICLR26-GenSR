# Pitfalls Research

**Domain:** Flow Matching DiT for Symbolic Regression Latent Space Optimization
**Researched:** 2026-06-09
**Confidence:** HIGH (architecture/codebase) / MEDIUM (domain-specific flow matching traps)

## Critical Pitfalls

### Pitfall 1: Self-Attention Degenerates on Single-Token Input

**What goes wrong:**
DiT 的核心是多头自注意力，要求序列长度 >= 2 才有意义。GenSR 的潜空间是单个 512 维向量（prior_mu / post_mu）。如果将 512 维向量 reshape 为 (1, 512) 作为 1-token 序列输入 DiT，self-attention 退化为恒等映射（1 个 token 只能 attend to 自己），整个 Transformer 块变成 `AdaLN(x) + FFN(AdaLN(x))` —— 本质上是一个带条件归一化的 MLP。多层堆叠后，每层的注意力权重完全相同（attention collapse），模型的表达力被大幅浪费，训练可能看似收敛但泛化极差。

**Why it happens:**
DiT 是为 Cola-DLM 的潜序列（`z_0 ∈ R^{n x d}`，n 通常为数百个 patch token）设计的。注意力机制依赖 token 间的关系建模。当序列长度为 1 时，`softmax(QK^T/sqrt(d))` 在 1x1 矩阵上操作，结果恒为 1.0。RoPE 在单 token 上退化为恒等旋转（position 0）。这不是一个"可能发生"的问题，而是在 (1, 512) 输入下必然发生的数学事实。

**How to avoid:**
必须将 512 维单向量拆分为多个 token。推荐方案：
- **Patchify 拆分**：将 (512,) reshape 为 (16, 32) 或 (8, 64)，加上一个可学习的线性投影映射到 DiT 的 hidden_dim。每个 "patch" 是潜向量的一个子段，这样注意力有实际内容可以操作。
- **保留 AdaLN 条件化**：timestep 通过 AdaLN 注入（这是 DiT 的强项），prior_mu 作为 cross-attention 的 context 或直接拼接为额外 token。
- 验证方式：在训练初期打印 attention 权重矩阵，确认其不是恒等矩阵。如果序列长度 L=1 且 attention weights = 1.0，则确认退化。

**Warning signs:**
- 训练 loss 下降但在验证集上完全不泛化
- 不同 prior_mu 条件下 DiT 输出几乎相同（条件失效）
- Attention 权重矩阵恒为 [[1.0]]
- 训练后期 loss 停滞在一个平台，远高于预期

**Phase to address:**
Phase 1（DiT 模型实现）—— 这是最基础的架构决策，必须在第一版代码中正确处理。

---

### Pitfall 2: FeatureFusion 的 Sampling 随机性破坏 Flow Matching 的确定性

**What goes wrong:**
FeatureFusion 的 forward 方法执行 `sample_gaussian_multi(mu, logvar, n_samples=200)`，即从 N(mu, logvar) 中采样 200 次再取均值。这是一个随机操作。在 DiT 推理管线中，如果 z_opt（DiT 输出的优化潜向量）通过相同的 FeatureFusion 处理，每次采样的随机性会导致同样的 z_opt 产生不同的 Decoder 输入。这会引入噪声，使得 Flow Matching 的确定性传输路径失去意义。更严重的是，训练时的 post_mu 对应的 FeatureFusion 输出和推理时 z_opt 对应的 FeatureFusion 输出在统计分布上可能不一致。

**Why it happens:**
现有 CMA-ES 管线中，每个候选潜向量只走一次 FeatureFusion → Decoder 路径，CMA-ES 通过多代迭代来"平均掉"采样噪声。但 Flow Matching 的目标是学习一条确定性的速度场 v(z_t, t)，如果下游的 FeatureFusion 引入随机性，训练目标（post_mu）和推理输出之间的映射就不再是确定性的，flow matching 理论保证失效。

**How to avoid:**
方案一（推荐）：在 DiT 推理时，用 `mu` 直接替代 FeatureFusion 中的采样均值。FeatureFusion 中 `z = samples.mean(dim=1)` 实际上等价于 `z = mu`（因为 200 个采样的均值近似等于 mu）。所以直接用 `expand_proj(mu)` 跳过采样步骤。
方案二：将 FeatureFusion 改为确定性版本（`z = mu`），仅在原始 CVAE 训练时保持随机性。
验证：检查 GenSR 原始推理代码中 FeatureFusion 是否被调用、调用时的随机性如何处理。

**Warning signs:**
- 同一 z_opt 多次推理，生成的表达式不一致
- R² 方差极大（同一潜向量的结果波动剧烈）
- 训练 loss 很低但推理 R² 极不稳定

**Phase to address:**
Phase 2（推理管线集成）—— 需要在对接 FeatureFusion 时仔细分析其调用路径。

---

### Pitfall 3: post_mu 不是真正的"最优潜向量"—— 训练目标与实际目标的 Gap

**What goes wrong:**
DiT 的 Flow Matching 目标是学习从 prior_mu（只看数据）到 post_mu（看数据+GT表达式）的传输。但 post_mu 是 CVAE 编码器在看到 GT 表达式后的编码结果，它代表的是"编码器认为的含表达式信息的潜向量"，而非"解码器能产生最好表达式的潜向量"。这两个目标之间存在 gap：
1. CVAE 的 KL 正则化会将 post_mu 拉向 prior_mu（KL 散度惩罚），使得 post_mu 不完全是"最优"的。
2. CMA-ES 找到的最优潜向量可能与 post_mu 完全不同，因为 CMA-ES 直接优化 R²（经过 Decoder + BFGS），而 post_mu 只经过了编码器。
3. 如果 CVAE 存在 posterior collapse（KL 权重过高时），post_mu ≈ prior_mu，此时 Flow Matching 学习的是恒等映射。

**Why it happens:**
这是 VAE 后验与最优潜向量之间固有的 gap。在标准 VAE 中，encoder 的目标是 ELBO 而非下游任务的最优性能。Cola-DLM 在 Stage 2 中联合训练 VAE 和 DiT（VAE 保持可训练），避免了这个问题。但 GenSR 的约束是 VAE 冻结，这意味着 post_mu 的质量完全取决于已训练好的 CVAE。

**How to avoid:**
1. 训练前先做诊断：计算训练集上 prior_mu 和 post_mu 之间的平均 L2 距离。如果距离很小（<0.1），说明 posterior collapse，Flow Matching 学不到有用的东西。
2. 考虑用 CMA-ES 找到的最优潜向量替代 post_mu 作为训练目标（如果能从现有实验数据中提取）。
3. 如果只能用 post_mu，监控训练时速度场的范数 `||v_psi(z_t, t)||`。如果范数很小，说明 prior_mu → post_mu 的传输距离太短，模型可能在学噪声。
4. 使用 OT-CFM（Optimal Transport CFM）路径而非线性插值路径，可以更高效地学习短距离传输。

**Warning signs:**
- DiT 的训练 loss 很低但推理 R² 与直接用 prior_mu（不做传输）相比几乎没有提升
- 速度场预测值的 L2 范数在整个训练过程中一直很小（< 1e-3）
- post_mu 和 prior_mu 的余弦相似度 > 0.99

**Phase to address:**
Phase 1（数据准备阶段）—— 在开始 DiT 训练前必须先验证训练数据质量。

---

### Pitfall 4: Euler 积分步数不足导致潜向量偏移

**What goes wrong:**
Flow Matching 推理使用 Euler 积分：`z_{t-dt} = z_t - dt/T * v_psi(z_t, t)`。如果步数太少（如只用 5-10 步），积分误差会在 512 维空间中累积。虽然 512 维远小于图像的 100K+ 维，但潜空间的几何结构可能非常不规则（frozen VAE 的潜空间不一定平滑），使得速度场 v_psi 的曲率较大，需要更细的步长才能准确跟踪。

**Why it happens:**
Euler 方法是一阶方法，局部误差 O(dt^2)，全局误差 O(dt)。对于 512 维空间中的非线性速度场，步数不足会导致积分轨迹偏离真实的 OT 路径。Cola-DLM 使用 T=1000.0 和 16 步，但 Cola-DLM 的潜空间经过 Stage 2 联合训练的精心设计，较为平滑。GenSR 的 frozen VAE 潜空间平滑度未知。

**How to avoid:**
1. 从较多的步数开始（如 50-100 步），逐步减少并观察 R² 变化。
2. 实现更高阶的积分器（如 midpoint method 或 RK4），作为 Euler 的替代方案。参考论文 "From Euler to Dormand-Prince: ODE Solvers for Flow Matching"。
3. 监控积分过程中每步速度场的 L2 范数变化。如果范数剧烈变化，说明需要更细的步长。
4. 实现 adaptive step size 机制：当速度场范数变化超过阈值时自动细化步长。

**Warning signs:**
- 减少推理步数时 R² 急剧下降（而 Cola-DLM 对步数减少相对鲁棒）
- 积分轨迹上相邻步之间的变化量 `||z_{t-dt} - z_t||` 差异很大
- 增加步数后 R² 持续提升，没有平台效应

**Phase to address:**
Phase 2（推理管线）—— 积分步数是核心超参数，需要系统性消融实验。

---

### Pitfall 5: Frozen VAE 潜空间不平滑，Flow Matching 学到的速度场在"缝隙"中失效

**What goes wrong:**
EQ-VAE（ICML 2025）的研究表明，冻结的 VAE 潜空间可能存在不平滑区域：两个相邻的潜向量解码后可能产生完全不同的表达式。Flow Matching 假设数据流形是光滑的（可以在 prior_mu 和 post_mu 之间平滑插值），但 frozen VAE 的潜空间中可能存在"断裂带"。DiT 学到的速度场在这些断裂带附近会产生错误的传输方向，导致 z_opt 落入潜空间中解码出无效表达式的区域。

**Why it happens:**
GenSR 的 CVAE 训练目标是最小化重建误差 + KL 正则化，并没有显式地优化潜空间的平滑性。Symbolic expression 的离散性质（token 序列）使得解码过程对潜向量的微小扰动极其敏感——一个小的潜向量偏移可能导致第一个 token 从 "sin" 变成 "exp"，从而完全改变表达式结构。

**How to avoid:**
1. 在 DiT 训练数据准备阶段，做插值实验：对多个 (prior_mu, post_mu) 对，在 t=0.1, 0.3, 0.5, 0.7, 0.9 处做线性插值 z_t = (1-t)*prior + t*post，通过 Decoder 解码并计算 R²。如果中间点的 R² 急剧下降，说明潜空间不平滑。
2. 如果不平滑，考虑增加 VAE 的 KL 权重（但这意味着重新训练 VAE，可能超出 scope）。
3. 在推理时使用更小的步长来避免大步跳过断裂带。
4. 实现回退机制：如果 z_opt 解码出的表达式 R² < prior_mu 解码出的 R²，则回退到 prior_mu。

**Warning signs:**
- 线性插值实验中，中间点的 R² 远低于两个端点
- DiT 训练收敛后，推理 R² 的方差极大
- 某些测试样本 R² 极高，另一些极低（不稳定性）

**Phase to address:**
Phase 1（数据准备和诊断）—— 必须在开始 DiT 训练前评估潜空间质量。

---

### Pitfall 6: BFGS 常数优化掩盖 DiT 的实际精度问题

**What goes wrong:**
GenSR 的评估管线是：z_opt → FeatureFusion → Decoder → 表达式骨架 → BFGS 常数优化 → R² 评估。BFGS 会修改表达式中的常数项，使得即使骨架结构不太对的表达式也能通过常数优化获得不错的 R²。这会导致 DiT 看起来比实际上更好——一个不精确的 DiT 可能因为 BFGS 的"补救"而获得与 CMA-ES 相近的 R²，但在更困难的测试集上（需要更精确的骨架结构），差距就会暴露。

**Why it happens:**
符号表达式的骨架（如 `sin(a*x + b) + c*x^2`）和常数（a, b, c）是分开评估的。BFGS 只优化常数，但如果骨架本身就是对的（只是常数不同），BFGS 能轻松修正。问题在于：DiT 可能学到了一种"近似正确骨架 + 靠 BFGS 补救"的策略，而非真正精确的潜空间传输。

**How to avoid:**
1. 评估时同时报告有 BFGS 和无 BFGS 的 R²。无 BFGS 的 R²（使用 DiT 输出的常数直接计算）才是 DiT 真正质量的衡量。
2. 比较骨架匹配率：DiT 和 CMA-ES 生成的表达式骨架是否与 GT 骨架相同。
3. 在论文/报告中明确区分"骨架正确率"和"BFGS 后 R²"，避免仅用后者作为指标。
4. 考虑增加一个不经过 BFGS 的评估通道，直接用解码出的常数计算 R²。

**Warning signs:**
- 有 BFGS 时 R² 与 CMA-ES 相当，但无 BFGS 时 R² 明显更低
- 表达式骨架与 GT 不匹配但 R² 仍然较高（BFGS "作弊"）
- 复杂表达式（多个常数）上 R² 差距更大（BFGS 难以补救多个常数）

**Phase to address:**
Phase 3（评估阶段）—— 评估指标设计时必须包含无 BFGS 的基线。

---

### Pitfall 7: 512 维 Flow Matching 的训练数据不足导致过拟合

**What goes wrong:**
Flow Matching 在 512 维空间中学习一个条件速度场 `v_psi(z_t, t; prior_mu)`。如果训练样本数量（N 个 prior_mu-post_mu 对）相对于模型参数量太少，DiT 会过拟合到训练分布中的特定传输路径上，对新的 prior_mu 无法泛化。符号回归的训练数据通常只有数万到数十万样本，而 DiT（即使是小规模）也有数百万参数。对比 Cola-DLM 使用了海量文本数据训练。

**Why it happens:**
Flow Matching 的条件设定（每个 prior_mu 对应一个特定的速度场路径）增加了模型的输入空间维度。512 维 z_t + 1 维 t + 512 维 prior_mu 条件 = 1025 维输入空间。在有限样本下，速度场的很多区域是"未见过的"，DiT 必须靠归纳偏置（而非数据）来泛化。

**How to avoid:**
1. DiT 的规模要适度：不要用大模型。推荐 4-6 层、256-384 hidden dim、4-8 heads 的小型 DiT，参数量控制在 1M-5M。
2. 使用较强的正则化：dropout 0.1-0.2、weight decay、gradient clipping。
3. 监控训练集和验证集上的速度场预测误差。如果验证误差在训练集误差下降时反而上升，确认过拟合。
4. 数据增强：对 prior_mu 添加小量高斯噪声生成额外的训练对（需验证不影响传输质量）。
5. 使用 classifier-free guidance（CFG）：训练时随机 drop 条件（将 prior_mu 替换为零向量），推理时用 guidance_scale > 1 提升条件响应。

**Warning signs:**
- 训练集上速度场 MSE 持续下降，验证集上开始上升
- 同一训练样本多次出现"完美"传输，但新样本的传输结果差
- 小规模 DiT（<1M 参数）反而比大规模 DiT 效果好

**Phase to address:**
Phase 1（模型设计和训练）—— 模型规模和数据量的匹配从一开始就需要考虑。

---

### Pitfall 8: prior_mu 条件化方式选择错误导致信息丢失

**What goes wrong:**
DiT 需要以 prior_mu 作为条件来预测速度场。如果条件化方式不当，prior_mu 的信息可能无法有效传递给 DiT。常见错误：
- 仅通过 AdaLN 注入 prior_mu：AdaLN 本质上是全局缩放+偏移，无法捕捉 prior_mu 中的维度级细节。
- 将 prior_mu 拼接到 z_t 上（增大维度）：这改变了输入空间结构，需要修改模型架构。

**Why it happens:**
在 Cola-DLM 中，条件信息（前缀的 z_0^(<b)）通过 block-causal attention 的 KV 传递，天然支持丰富的条件交互。但 GenSR 的 prior_mu 是单个向量，不是序列，无法直接使用相同的 cross-attention 机制。

**How to avoid:**
推荐方案：将 prior_mu 扩展为额外的 token 加入到 patchified 序列中。例如，z_t reshape 为 (L, d)，prior_mu 作为一个额外的 (1, d) token 拼接到序列开头，得到 (L+1, d) 输入。prior_mu token 通过 self-attention 与所有 z_t patch 交互。这是最自然的条件化方式，也是 DiT 原论文中 in-context conditioning 的思路。
备选方案：cross-attention，z_t patches 作为 Q，prior_mu 扩展为 (1, d) 作为 KV。但增加了额外的 cross-attention 模块和参数。

**Warning signs:**
- DiT 输出对不同的 prior_mu 几乎不变（条件化失效）
- 用固定的 prior_mu 替换真实条件时，R² 没有显著下降
- prior_mu 的各维度对 DiT 输出的梯度很小

**Phase to address:**
Phase 1（DiT 架构设计）—— 条件化方式是架构的核心选择之一。

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| 用 (1, 512) 直接输入 DiT 不 patchify | 快速实现 prototype | 注意力退化，模型变成昂贵的 MLP | 仅在概念验证阶段（< 1 天），之后必须 patchify |
| 用线性插值路径而非 OT-CFM | 实现简单 | 传输路径可能穿越潜空间断裂带 | 可在初期使用，但 OT-CFM 应尽快替换 |
| 固定推理步数不做步数消融 | 节省实验时间 | 可能用了过多步数（慢）或过少（不准） | 绝不可接受——步数是核心超参数 |
| 只报告 BFGS 后的 R² | 数字好看 | 掩盖 DiT 真实质量，无法发现问题 | 绝不可接受——必须同时报告无 BFGS 指标 |
| 不做 prior_mu-post_mu 距离诊断 | 跳过数据分析 | 可能在无用数据上训练（post_collapse） | 绝不可接受——这是训练前的必要验证 |

## Integration Gotchas

| Integration Point | Common Mistake | Correct Approach |
|-------------------|----------------|------------------|
| FeatureFusion 接口 | 直接调用 FeatureFusion.forward(mu, logvar) 保留采样随机性 | 用确定性路径：expand_proj(mu) 跳过 sample_gaussian_multi |
| Decoder 接口 | 传入 z_opt 时形状不匹配 FeatureFusion 输出的 (200, 512) | z_opt 是 (512,) 向量，需先经过 FeatureFusion 的 expand_proj 转为 (200, dec_emb_dim) |
| CVAE 编码 | 训练时 post_mu 使用了 concat([x1, x2])，推理时只有 x1 | 推理只产出 prior_mu，DiT 从 prior_mu 出发传输 |
| gen2eq 评估 | 传入 generations 的 shape 为 (beam_size, seq_len) 而非 (1, seq_len) | 需要逐个 beam 调用或正确 batch |
| CMA-ES 对比基线 | 用不同数量的 BFGS 迭代/不同超参数做对比 | 严格匹配计算预算：相同 GPU 时间或相同评估次数 |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| 推理步数过多 | 推理时间 > CMA-ES 的 50 代迭代时间 | 系统性步数消融，找到精度-速度甜蜜点 | 推理时间 > 5 秒/样本 |
| DiT 过大导致推理慢 | 单次前向传播耗时 > 100ms | 控制模型规模：4-6 层，hidden_dim 256-384 | 模型参数 > 10M |
| FeatureFusion 的 200 次采样 | 推理时重复采样耗时 | 用确定性路径替代 | 每次推理 > 50ms 花在采样上 |
| 不必要的 CPU-GPU 数据传输 | gen2eq 中 tensor 频繁 .cpu().tolist() | 批量处理减少传输次数 | 每次 eval 额外 > 10ms 传输开销 |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| torch.load(weights_only=False) 加载检查点 | 任意代码执行 | 使用 weights_only=True 或验证检查点来源（已在 CONCERNS.md 中标记） |
| 不验证输入数据维度 | 静默错误传播，DiT 训练产生垃圾结果 | 在 DiT forward 开头添加 assert 检查输入 shape |

## "Looks Done But Isn't" Checklist

- [ ] **DiT 训练收敛:** 可能只是在学恒等映射（post_mu ≈ prior_mu）—— 验证 prior_mu 和 post_mu 的平均 L2 距离
- [ ] **推理 R² 匹配 CMA-ES:** 可能是 BFGS 在"补救"—— 验证无 BFGS 时的 R² 和骨架匹配率
- [ ] **注意力机制工作正常:** 可能退化为恒等（单 token）—— 打印 attention weights 验证
- [ ] **Flow Matching 积分正确:** 可能数值误差累积 —— 对比不同步数下的输出一致性
- [ ] **条件化有效:** 可能 DiT 忽略了 prior_mu 条件 —— 做 ablation：将 prior_mu 替换为零向量，检查 R² 变化
- [ ] **FeatureFusion 对接正确:** 可能采样随机性引入噪声 —— 对比确定性路径和采样路径的 R² 方差
- [ ] **评估管线完整:** 可能只测了 in-distribution 数据 —— 必须包含 OOD 测试集（不同变量数、不同复杂度）

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| 注意力退化（单 token） | LOW | 重构输入为 patchified 序列，改 1 行 reshape + 添加 linear proj |
| FeatureFusion 随机性 | LOW | 替换为确定性路径 expand_proj(mu) |
| post_mu 质量差（posterior collapse） | HIGH | 需要重新评估 CVAE 质量，可能需要重新训练 VAE 或用 CMA-ES 结果作为目标 |
| 积分步数错误 | LOW | 调整步数超参数，无需改代码 |
| 潜空间不平滑 | MEDIUM | 增加推理步数、添加回退机制、或考虑重新训练 VAE |
| DiT 过拟合 | MEDIUM | 缩小模型规模、增加正则化、增加数据增强 |
| BFGS 掩盖真实质量 | LOW | 添加无 BFGS 评估通道 |
| 条件化失效 | MEDIUM | 重构条件化方式（改为 in-context token 或 cross-attention） |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 注意力退化 | Phase 1 (架构设计) | 打印 attention weights，确认 L > 1 时不为恒等矩阵 |
| FeatureFusion 随机性 | Phase 2 (推理管线) | 对比确定性/随机路径的 R² 方差 |
| post_mu 质量 gap | Phase 1 (数据准备) | 计算 prior_mu-post_mu L2 距离分布 |
| Euler 步数不足 | Phase 2 (推理管线) | 步数消融实验，绘制步数-R² 曲线 |
| 潜空间不平滑 | Phase 1 (数据准备) | 插值实验：中间点的解码 R² 分布 |
| BFGS 掩盖精度 | Phase 3 (评估) | 同时报告有/无 BFGS 的指标 |
| 训练数据不足 | Phase 1 (模型设计) | 训练/验证 loss 曲线，参数量 vs 样本量比 |
| 条件化失效 | Phase 1 (架构设计) | 条件 ablation：替换条件为零向量，检查输出变化 |

## Sources

- Cola-DLM architecture documentation: https://github.com/ByteDance-Seed/Cola-DLM/blob/main/docs/architecture.md
- Cola-DLM paper: arXiv:2605.06548 (Continuous Latent Diffusion Language Model)
- AdaLN-Zero analysis: https://openreview.net/forum?id=E4roJSM9RM
- Attention rank collapse study: https://openreview.net/forum?id=gm5mkiTGOy (NeurIPS 2025)
- EQ-VAE (latent space smoothness for diffusion): https://arxiv.org/html/2502.09509v1 (ICML 2025)
- Smooth Diffusion (CVPR 2024): https://shi-labs.github.io/Smooth-Diffusion/
- ODE solvers for flow matching: https://arxiv.org/html/2605.00836v1
- "The Curse of Conditions" (OT in conditional FM): https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng (ICCV 2025)
- Conditional Flow Matching visual guide: https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/ (ICLR 2025)
- GenSR codebase analysis: CONCERNS.md and source code review

---
*Pitfalls research for: Flow Matching DiT for Symbolic Regression Latent Space Optimization*
*Researched: 2026-06-09*
