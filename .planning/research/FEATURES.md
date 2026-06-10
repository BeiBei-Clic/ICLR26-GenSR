# Feature Research

**Domain:** Flow Matching DiT for Symbolic Regression Latent Space Search
**Researched:** 2026-06-09
**Confidence:** MEDIUM-HIGH (architecture patterns well-established in Cola-DLM; domain-specific adaptation to symbolic regression requires validation)

## Feature Landscape

### Table Stakes (Must Have)

These are non-negotiable for the system to function. Without any of these, the DiT replacement of CMA-ES cannot be meaningfully evaluated.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| DiT backbone with AdaLN-Zero | Cola-DLM 标准架构; AdaLN-Zero 是 Flow Matching 条件化的核心机制，用 timestep embedding 调制 LayerNorm 参数 | MEDIUM | 每个 DiT block: AdaLN 调制 (scale, shift, gate) 6 个参数; residual 分支用 zero-init gate 初始化，保证训练初期输出恒等映射 |
| Flow Matching velocity prediction | DiT 必须预测速度场 v_psi(z_t, t)，而不是噪声 (eps-prediction) 或数据 (x-prediction) | LOW | 训练目标: L = E_{t,z_0,z_1}[||v_psi(z_t, t) - u_t(z_0, z_1)||^2]; z_0=prior_mu, z_1=post_mu |
| Conditional interpolation path | 在 prior_mu 和 post_mu 之间定义插值路径 z_t = (1-t)*prior_mu + t*post_mu | LOW | 线性 OT 路径作为起点; 速度场目标 u_t = post_mu - prior_mu (与 t 无关，最简形式) |
| Euler ODE inference | 从 prior_mu 出发，Euler 积分若干步得到优化后潜向量 | LOW | x_{t-dt} = x_t - dt/T * v_psi(z_t, t); 典型 5-20 步即可 |
| Training data extraction | 从 CVAE 训练集提取 (prior_mu, post_mu) 对作为 Flow Matching 训练数据 | MEDIUM | 需要遍历已有训练集，冻结 CVAE 参数，对每个样本计算 prior_mu (只看数据) 和 post_mu (看数据+GT表达式) |
| Checkpoint save/load | DiT 模型参数的序列化保存与恢复 | LOW | 标准 PyTorch state_dict; 需要保存 optimizer 状态用于断点续训 |
| PMLB evaluation pipeline | 在 PMLB 基准上与 CMA-ES 做定量对比 | MEDIUM | 复用现有 gen2eq 管线: DiT 输出 -> FeatureFusion -> Decoder -> BFGS 常数优化 -> R2 评估 |
| prior_mu conditioning | DiT 必须以 prior_mu 为条件输入，编码数据信息 | MEDIUM | prior_mu 已由 CVAE 编码了 (X, Y) 数据信息; 通过 cross-attention 或直接拼接作为 DiT 输入条件 |

### Differentiators (Competitive Advantage)

这些特性是 DiT 方法相对于 CMA-ES 的核心优势来源。不是必需的，但实现后显著提升论文贡献度。

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 推理速度 100x+ 加速 | CMA-ES 需要 50-100 代迭代，每代需解码+BFGS+R2 评估; DiT 仅需单次前向传输 (5-20 步) | LOW (inherent) | 这是方法论的天然优势，不是额外实现的功能。DiT 推理耗时 = 5-20 次 DiT 前向 + 1 次 Decoder 解码 + 1 次 BFGS |
| Multi-sample generation | 从 prior_mu 出发加不同噪声，一次生成多个候选潜向量，选最优 | LOW | 实现: z_init = prior_mu + sigma * randn(N, 512); 批量 DiT 传输; 对每个候选做 gen2eq 评估 |
| Configurable inference steps | 允许用户调整 Euler 积分步数，在速度和精度之间权衡 | LOW | 参数化 timestep_num (5, 10, 20, 50 步); 线性时间调度 linspace(T, 0, steps+1) |
| SwiGLU FFN | 相比标准 ReLU FFN 更强的表达能力 | LOW | Cola-DLM 标准组件: gate = xW1 * silu(xW2); hidden_dim = 8/3 * model_dim |
| RoPE positional encoding | 如果将 512-dim 潜向量 reshape 为序列 (如 1x512 或 patchify 为多个 token)，RoPE 提供位置信息 | MEDIUM | 当输入为单 512-dim 向量时 RoPE 退化为恒等; 但若 patchify 为 (4, 128) 或 (8, 64)，RoPE 有意义。需架构决策 |
| Training loss logging & visualization | 监控 Flow Matching loss 曲线、速度场范数、潜向量分布变化 | LOW | 接入 wandb (项目已使用); 记录: FM loss, velocity L2 norm, z_t 分布的 mean/std 随 t 变化 |
| Latent space analysis | 分析 DiT 输出的潜向量质量: 与 post_mu 的 L2 距离、解码后表达式的 R2 分布 | MEDIUM | 不是模型功能但极有价值; 可视化传输轨迹 (prior_mu -> z_t -> z_opt) 在 2D PCA/t-SNE 上的投影 |

### Anti-Features (Do NOT Build)

这些看起来合理但会引入不必要复杂度或偏离核心目标的功能。

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Classifier-Free Guidance (CFG) | Cola-DLM 使用了 CFG，看起来是标准做法 | 在 SR 场景下，"条件"就是 prior_mu (数据编码)。没有明确的"无条件"场景 -- 不像文本生成有 prompt/no-prompt 的区分。对 512-dim 单向量做 CFG 增加双倍推理开销，收益不确定 | MVP 先不做 CFG，观察 DiT 输出质量。如果发现生成多样性不足，再考虑用 prior_mu dropout 做 CFG |
| 高阶 ODE solver (DPM-Solver, RK45) | 更少步数达到同样精度 | 512-dim 速度场的 ODE 轨迹通常很平滑 (线性插值路径)，Euler 已足够。增加 solver 复杂度在 MVP 阶段是过度工程 | 先用 Euler。如果 20 步 Euler 还不够好，再考虑 mid-point 或 RK4。不要在 MVP 实现自适应步长 solver |
| OT coupling / Minibatch OT | OT 路径理论上比线性插值更直更高效 | 需要 batch 内做 Hungarian matching，增加训练复杂度。且本项目训练对 (prior_mu, post_mu) 是自然配对的 (同一个样本的两个编码)，OT coupling 意义不大 | 使用简单的线性插值 z_t = (1-t)*z_0 + t*z_1。这是 Cola-DLM 的默认做法，对自然配对数据足够 |
| Latent sequence / Patchify | Cola-DLM 在潜序列上操作 (n x d)，patchify 更有 Transformer 的感觉 | 现有 VAE 的潜空间是 512-dim 单向量，不是序列。强行 patchify 引入不必要的架构复杂度，且 Decoder 期望输入是单向量经 FeatureFusion 展开的 | 保持 512-dim 单向量输入。DiT 处理 (1, 512) 序列。这虽然看似"浪费"了 Transformer 的序列处理能力，但保持了与现有管线的兼容性 |
| DiT + VAE 联合训练 | 联合微调可能获得更好的端到端效果 | 项目约束明确: VAE/Decoder 冻结。联合训练会引入 latent drift (潜空间漂移)，需要 Cola-DLM 的 reference-encoder KL 正则化来防止，复杂度剧增 | 冻结 VAE，只训练 DiT。这是项目约束，也是合理的 MVP 范围 |
| Reinforcement Learning fine-tuning | 用 R2 作为奖励信号微调 DiT，类似 RLHF | 引入 RL 管线 (reward model, PPO, KL penalty) 复杂度极高。且符号回归的 reward (R2 after BFGS) 不可微且计算昂贵 | MVP 用纯 Flow Matching 监督学习。后续若需提升，可考虑 DPO 或直接 reward-weighted regression，但不在 MVP 范围 |
| Temperature / Sampling strategies | 推理时加随机性增加多样性 | Flow Matching 推理是确定性 ODE 积分，加 temperature 没有标准做法。随意引入噪声可能破坏传输轨迹的质量 | 多样性通过 multi-sample generation 实现 (不同初始噪声)。不在 ODE 积分中加 temperature |

## Feature Dependencies

```
[Training Data Extraction (prior_mu, post_mu pairs)]
    |
    v
[DiT Backbone with AdaLN-Zero]  <-- depends on architecture decision
    |
    v
[Flow Matching Velocity Prediction]
    |
    +--> [Euler ODE Inference]  --> [PMLB Evaluation Pipeline]
    |                                     |
    +--> [Multi-sample Generation] ------+
    |                                     |
    +--> [Configurable Inference Steps] --+

[prior_mu Conditioning]  ──integrated into──> [DiT Backbone]

[Checkpoint Save/Load] ──orthogonal to──> [all training features]

[SwiGLU FFN] ──part of──> [DiT Backbone]
[RoPE] ──part of──> [DiT Backbone] (but low value for single-vector input)

[Training Loss Logging] ──enhances──> [Flow Matching Velocity Prediction]

[Latent Space Analysis] ──enhances──> [PMLB Evaluation Pipeline]
```

### Dependency Notes

- **DiT Backbone requires prior_mu Conditioning:** DiT 的条件输入是 prior_mu。条件注入方式 (cross-attention vs 拼接 vs AdaLN) 是架构核心决策，必须在 DiT 设计时确定。推荐: 将 prior_mu 作为 cross-attention 的 K/V 注入， timestep t 通过 AdaLN 注入。

- **Flow Matching Training requires Training Data Extraction:** 必须先有 (prior_mu, post_mu) 对才能训练。数据提取是一次性的离线预处理步骤。

- **PMLB Evaluation requires Euler ODE Inference + gen2eq pipeline:** 推理管线必须先产出优化后的潜向量，再通过 FeatureFusion -> Decoder -> BFGS 得到可评估的表达式。

- **Multi-sample Generation depends on Euler ODE Inference:** 只是推理时的批量化，实现简单。

- **Configurable Inference Steps is orthogonal to Multi-sample:** 两者可独立调整，形成步数 x 样本数的超参矩阵。

- **Latent Space Analysis is optional but highly valuable:** 不影响模型功能，但对论文分析和 debugging 极有帮助。建议在评估阶段同步实现。

## MVP Definition

### Launch With (v1) -- 核心验证

这些是验证"DiT Flow Matching 能否替代 CMA-ES"所需的最小功能集。

- [ ] **Training data extraction** -- 从 CVAE 训练集提取 (prior_mu, post_mu) 对，存为 .pt 文件。必须首先完成，后续一切依赖此数据。
- [ ] **DiT backbone (minimal)** -- 4-8 层 DiT blocks，AdaLN-Zero 条件化 (timestep + prior_mu)，SwiGLU FFN，输入输出各 512 维。先不做 RoPE (单向量输入无意义)。
- [ ] **Flow Matching training loop** -- 线性插值路径 + velocity MSE loss + AdamW optimizer + cosine LR schedule。EPOCH-based training with wandb logging。
- [ ] **Euler ODE inference (10 steps)** -- 从 prior_mu 出发，10 步 Euler 积分得到 z_opt，走 gen2eq 管线。
- [ ] **Checkpoint save/load** -- 训练完成后保存 DiT state_dict，推理时加载。
- [ ] **PMLB evaluation script** -- 在至少 50 个 PMLB 数据集上运行 DiT 方法，记录 R2_fit, R2_predict, complexity, inference_time，与 CMA-ES 结果对比。

### Add After Validation (v1.x)

核心管线跑通后，添加这些来提升结果质量和论文贡献。

- [ ] **Multi-sample generation** -- 从 prior_mu 出发生成 N 个候选 (N=5, 10, 20)，选最优。触发条件: 单样本 R2 不够好。
- [ ] **Configurable inference steps** -- 测试 5, 10, 20, 50 步的效果-速度 tradeoff。触发条件: 需要确定最优推理步数。
- [ ] **Hyperparameter sweep** -- DiT 层数 (4, 8, 12), hidden dim, 学习率, 训练 epoch 数。触发条件: v1 结果有希望但不够好。
- [ ] **Training loss visualization** -- FM loss 曲线, velocity field norm, latent distribution。触发条件: 训练过程需要 debugging。
- [ ] **Latent space analysis** -- PCA/t-SNE 可视化传输轨迹，与 post_mu 的距离分析。触发条件: 论文需要深入分析。

### Future Consideration (v2+)

论文投稿后的扩展方向。

- [ ] **Classifier-Free Guidance** -- prior_mu dropout + CFG scale。触发条件: DiT 生成多样性不足或精度不够。
- [ ] **Higher-order ODE solver** -- Midpoint method 或 RK4。触发条件: Euler 积分精度不够且增加步数太慢。
- [ ] **Latent sequence architecture** -- 将 512-dim patchify 为序列，利用 Transformer 的序列建模能力。触发条件: 单向量架构遇到瓶颈。
- [ ] **Reward-weighted fine-tuning** -- 用 R2 信号做轻量级微调。触发条件: 纯 FM 监督学习的结果有上限。

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Training data extraction | HIGH | LOW | P1 |
| DiT backbone with AdaLN-Zero | HIGH | MEDIUM | P1 |
| Flow Matching training loop | HIGH | LOW | P1 |
| Euler ODE inference | HIGH | LOW | P1 |
| Checkpoint save/load | HIGH | LOW | P1 |
| PMLB evaluation pipeline | HIGH | MEDIUM | P1 |
| prior_mu conditioning (cross-attention) | HIGH | MEDIUM | P1 |
| SwiGLU FFN | MEDIUM | LOW | P1 (part of backbone) |
| Multi-sample generation | MEDIUM | LOW | P2 |
| Configurable inference steps | MEDIUM | LOW | P2 |
| Training loss logging (wandb) | MEDIUM | LOW | P2 |
| Latent space analysis | MEDIUM | MEDIUM | P2 |
| RoPE | LOW | MEDIUM | P3 |
| Classifier-Free Guidance | MEDIUM | MEDIUM | P3 |
| Higher-order ODE solver | LOW | LOW | P3 |
| Reward-weighted fine-tuning | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (core DiT pipeline)
- P2: Should have, add after basic validation
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | CMA-ES (Current) | DiffuSR (arXiv 2509.13136) | Discrete Diffusion SR (arXiv 2505.24776) | Our Approach (DiT FM) |
|---------|-------------------|---------------------------|------------------------------------------|----------------------|
| 搜索空间 | CVAE 512-dim 潜空间 | 直接生成表达式 token | Token masking diffusion | CVAE 512-dim 潜空间 |
| 推理方式 | 迭代进化 50-100 代 | 连续状态 diffusion | 离散 token denoising | 单次 Flow 传输 10-20 步 |
| 推理速度 | 慢 (~30s/sample) | 中等 | 中等 | 快 (~1s/sample) |
| 条件信息 | 通过 prior_mu 初始点 | 数据编码 | 数据编码 | prior_mu 通过 cross-attention |
| 表达式质量 | 依赖迭代优化 | 依赖预训练 | 依赖预训练 | 学习从 prior 到 posterior 的映射 |
| 常数优化 | 每代 BFGS | 后处理 BFGS | 后处理 BFGS | 后处理 BFGS (复用现有) |
| 可扩展性 | 受限于解码评估开销 | 受限于 token vocab | 受限于 token vocab | 潜空间操作，可扩展 |

### Key Differentiators vs Competition

1. **vs CMA-ES:** 速度快 10-100x (单次前向 vs 迭代进化)，且 DiT 学习的是数据驱动的先验传输，而 CMA-ES 是无模型优化。
2. **vs DiffuSR / Discrete Diffusion:** 在潜空间操作而非 token 空间，避开了 token 离散化的信息损失；且可以完全复用现有 VAE + Decoder，不改变下游管线。
3. **vs LLM-SR (GPT-fine-tune):** 不需要大规模语言模型预训练，训练数据直接来自现有 CVAE 的编码结果，数据效率高。

## Evaluation Metrics

### Primary Metrics (论文核心指标)

| Metric | Description | Target vs CMA-ES | Notes |
|--------|-------------|-------------------|-------|
| R2_fit (R2_zero) | 拟合集 R2 (max(0, R2)) | >= CMA-ES | 复用现有 compute_metrics("r2_zero") |
| R2_predict | 预测集 R2 | >= CMA-ES | 泛化能力指标 |
| Inference time | 单样本推理耗时 | < 1/10 CMA-ES | 核心优势指标 |
| Complexity | 表达式节点数 | <= CMA-ES | 复用 len(tree.prefix().split(",")) |

### Secondary Metrics (辅助分析)

| Metric | Description | Notes |
|--------|-------------|-------|
| Recovery rate | 恢复 GT 表达式结构的比例 | SRBench 标准指标，需定义"恢复"的等价标准 |
| Pareto front (R2 vs complexity) | R2-复杂度前沿 | SRBench 标准可视化 |
| FM training loss curve | Flow Matching loss 收敛曲线 | 判断训练是否充分 |
| Velocity field L2 norm | 速度场范数随 t 变化 | 判断传输轨迹的平滑度 |
| Latent distance | z_opt 与 post_mu 的 L2 距离 | 判断 DiT 是否学到了有用的传输 |

### Metrics from Existing Codebase

项目已有以下指标 (在 metrics.py 和 LSO_fit.py 中):

- `r2_zero`: max(0, R2) -- 主要指标
- `r2`: 标准 R2
- `accuracy_l1_biggio`: rtol=0.05 的点精度
- `accuracy_l1_1e-3`, `accuracy_l1_1e-2`, `accuracy_l1_1e-1`: 不同精度的准确率
- `_complexity`: 表达式节点数
- `_relative_complexity`: 预测 vs GT 的复杂度差
- `_mse`, `_nmse`, `_rmse`: 误差指标
- `mse_fit`, `mse_pred`: 标准化 MSE
- `time`: 推理耗时

## Sources

- [Cola-DLM Official Repository & Architecture Docs](https://github.com/ByteDance-Seed/Cola-DLM) -- HIGH confidence, primary architecture reference
- [Cola-DLM Paper (arXiv:2605.06548)](https://arxiv.org/abs/2605.06548) -- HIGH confidence, Flow Matching + DiT methodology
- [DiT Original Paper (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748) -- HIGH confidence, AdaLN-Zero architecture
- [AdaLN-Zero Analysis (OpenReview)](https://openreview.net/forum?id=E4roJSM9RM) -- HIGH confidence, why AdaLN-Zero works
- [OT-CFM (arXiv:2210.02747)](https://arxiv.org/abs/2210.02747) -- HIGH confidence, Flow Matching foundations
- [DiffuSR (arXiv:2509.13136)](https://arxiv.org/html/2509.13136v1) -- MEDIUM confidence, competitor approach
- [Discrete Diffusion SR (arXiv:2505.24776)](https://arxiv.org/html/2505.24776v1) -- MEDIUM confidence, competitor approach
- [SRBench / SRBench++](https://openreview.net/forum?id=xVQMrDLyGst) -- HIGH confidence, SR evaluation methodology
- [PMLB Benchmark Suite](https://epistasislab.github.io/pmlb/) -- HIGH confidence, standard SR datasets
- [MIT Flow Matching Lecture Notes 2026](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) -- HIGH confidence, Euler solver and guidance
- [Blockwise Flow Matching (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/118395) -- MEDIUM confidence, advanced FM techniques

---
*Feature research for: Flow Matching DiT for Symbolic Regression*
*Researched: 2026-06-09*
