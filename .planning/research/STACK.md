# Stack Research

**Domain:** Flow Matching DiT for Symbolic Regression (latent-space prior transport)
**Researched:** 2026-06-09
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyTorch | 2.10.0+cu128 (已安装) | 深度学习框架 | 已在环境中安装，支持 bf16 autocast、torch.compile、fused optimizers。Flow Matching 的核心操作（Euler 积分、速度场计算）都是标准张量运算，不需要特殊 CUDA 算子 |
| einops | 0.8.2 (已安装) | 张量 reshape 操作 | Cola-DLM DiT 代码大量使用 `rearrange`，保持一致性。比手写 reshape 更安全、更可读 |
| rotary-embedding-torch | >=0.8.6 (需安装) | RoPE 位置编码 | Cola-DLM DiT 使用此库的 `RotaryEmbedding(freqs_for="lang")` 模式。对于我们的简化 DiT（512-dim 单向量输入），位置编码的作用有限，但保留 RoPE 可最大程度复用 Cola 架构模式，也为将来 patchify 输入做准备 |
| wandb | >=0.14.0 (已安装) | 实验跟踪 | 已在项目中使用，保持一致。训练时跟踪 FM loss 曲线、Euler 步数 vs R²、推理时间对比 |

### Flow Matching 核心方案

| 方案 | 推荐 | 说明 |
|------|------|------|
| **自实现 Flow Matching 训练** | 推荐 (HIGH confidence) | 本项目的 Flow Matching 非常简单——一个条件速度场从 prior_mu 到 post_mu 的传输。训练 loss 就是 `MSE(v_psi(z_t, t, prior_mu), post_mu - prior_mu)`，推理就是 Euler 积分 `z_{t-dt} = z_t - dt * v_psi(...)`。不需要任何 FM 库 |
| torchcfm | 不推荐 | 面向通用 CNF 训练（图像、表格数据），提供的 `ConditionalFlowMatcher`、`ExactOptimalTransportConditionalFlowMatcher` 等类是为"从噪声到数据分布"的标准 FM 设计的。我们的场景是"从条件先验到条件后验"的配对传输，数据天然配对（同一个样本的 prior_mu 和 post_mu），不存在 batch OT matching 的需求 |
| facebookresearch/flow_matching | 不推荐 | 面向大规模图像/文本 FM 训练，提供 Path、Scheduler、Solver 等抽象。对 512-dim 单向量场景过重。且其 CC BY-NC 许可证对学术使用无碍但引入不必要的依赖 |

**自实现 Flow Matching 的核心代码量极小（约 20 行），比引入第三方库更清晰：**

```python
# 训练（OT-path / 线性插值）
t = torch.rand(batch_size, 1, device=device)
z_t = (1 - t) * prior_mu + t * post_mu          # 线性插值路径
target_velocity = post_mu - prior_mu              # OT-path 的目标速度场
predicted_velocity = dit(z_t, t, prior_mu)        # DiT 前向
loss = F.mse_loss(predicted_velocity, target_velocity)

# 推理（Euler 积分）
z = prior_mu.clone()
timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
    dt = t_curr - t_next
    v = dit(z, torch.full((B,), t_curr, device=device), prior_mu)
    z = z - dt * v
```

### DiT 架构方案

| 方案 | 推荐 | 说明 |
|------|------|------|
| **基于 Cola-DLM DiT 简化** | 推荐 (HIGH confidence) | 参考代码已在 `Cola-DLM-main/cola_dlm/modeling_cola_dit.py`。需要大幅简化：去掉 block-causal 逻辑（我们只有 1 个 block）、去掉 NA flatten-concat（我们用标准 batch 维度）、去掉 KV cache（推理是一次性前向）。保留的核心组件：AdaLN-Zero 条件化、Sinusoidal timestep embedding、RoPE（可选）、QK Norm |
| facebookresearch/dit (原版 DiT) | 不推荐 | 面向图像 latent patch，使用 class-conditional AdaLN。我们的条件是连续的 prior_mu 向量，不是离散的类别标签 |
| 从零实现 MLP-based velocity predictor | 备选 | 对于 512-dim 单向量，纯 MLP（若干 Linear + SiLU）也能 work。但项目要求"迁移 Cola 的 DiT 方法论"，且 DiT 的 AdaLN-Zero 对训练稳定性有实际好处 |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.24.0 (已安装) | 数据处理 | 训练数据准备、评估指标计算 |
| scipy | >=1.10.0 (已安装) | BFGS 常数优化 | 推理管线中保留的常数优化步骤 |
| scikit-learn | >=1.2.0 (已安装) | R² 评估指标 | gen2eq 评估管线中使用 |
| sympy | >=1.11.0 (已安装) | 符号表达式处理 | 解码后的表达式解析和评估 |
| sympytorch | >=0.1.0 (已安装) | SymPy-PyTorch 桥接 | 表达式的可微评估 |
| tqdm | >=4.65.0 (已安装) | 进度条 | 训练和推理循环 |
| matplotlib | >=3.7.0 (已安装) | 可视化 | 训练曲线、推理结果可视化 |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| PyTorch AdamW | 优化器 | DiT 训练的标准选择。初始 lr=1e-4，weight_decay=0.01 |
| PyTorch CosineAnnealingLR | 学习率调度 | Flow Matching 训练常用。配合 warmup（前 1000 步线性升温）|
| torch.amp (bf16) | 混合精度训练 | 已有环境支持 bf16。对 DiT 训练可加速约 2x，节省显存 |
| torch.compile | 可选的图编译加速 | PyTorch 2.10 对 compile 支持成熟。小模型可能收益有限 |

## Installation

```bash
# 新增依赖（相对于现有 requirements.txt）
pip install rotary-embedding-torch>=0.8.6

# 不需要安装的
# pip install torchcfm       # 不需要——自实现 FM 更简单
# pip install flow_matching  # 不需要——自实现 FM 更简单
# pip install diffusers      # 不需要——不是扩散模型
```

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| 自实现 FM (线性 OT-path) | torchcfm OT-CFM | torchcfm 的 OT-CFM 解决的是"batch 内无配对信息时做最优匹配"。我们天然有 (prior_mu, post_mu) 配对，不存在匹配问题。引入库只会增加不必要的抽象层 |
| 自实现 FM | facebookresearch/flow_matching | 过重。提供的 Path/Scheduler/Solver 抽象是为大规模图像/文本设计的。512-dim 单向量场景下，核心逻辑 20 行代码搞定。且 CC BY-NC 许可证 |
| 简化 DiT (AdaLN + Transformer blocks) | 纯 MLP velocity field | 512-dim 单向量确实可以用 MLP。但项目明确要求"迁移 Cola-DLM 的 DiT 方法论"。且 AdaLN-Zero 对训练稳定性的好处是经过大规模验证的。如果 DiT 表现不佳，可后续退化为 MLP |
| 线性 OT-path (affine interpolation) | 分离桥 (Schrödinger Bridge) | SB 引入额外的随机性（score function），增加训练复杂度。对于从确定性起点到确定性终点的配对传输，OT-path（直线）是最优的——路径最短，Euler 积分步数最少 |
| 线性 OT-path | Rectified Flow (reflow) | Reflow 需要多轮训练（先训练 flow，再用 flow 生成配对数据重训）。对于研究阶段来说过重。首轮 OT-path 训练若效果足够好则不需要 reflow |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| diffusers 库 | diffusers 是为 DDPM/DDIM 等噪声扩散设计的，其 Scheduler（PNDM, EulerDiscrete 等）是围绕 "噪声调度" 概念的。Flow Matching 不是扩散模型，没有 SNR、噪声调度等概念 | 自实现 Euler 积分器（5 行代码）|
| DDPM / DDIM / DPM-Solver | 这些都是噪声扩散的采样方法，假设正向过程是加噪。Flow Matching 是 OT 传输，从 prior_mu 出发（不是从纯噪声出发），数学框架完全不同 | Flow Matching 的 Euler / RK4 积分 |
| HuggingFace Diffusers DiT Pipeline | diffusers 中的 DiT Pipeline 是为 class-conditional 图像生成设计的，与我们的连续条件 prior_mu 不兼容 | 自实现 DiT，参考 Cola-DLM 的架构模式 |
| torchcfm / flow_matching 库 | 如上所述，512-dim 配对传输场景太简单，引入这些库增加依赖但无实际收益 | 自实现（~20 行核心代码）|
| PyTorch Lightning | 现有代码库使用 Lightning 2.0.5 做 VAE 训练。DiT 训练循环极其简单（单 GPU、无分布式需求），用 Lightning 反而增加抽象复杂度 | 纯 PyTorch 训练循环 |

## Stack Patterns by Variant

**标准情况（推荐）：DiT on 512-dim single vector**
- 输入: `z_t` shape `(B, 512)`，reshape 为 `(B, 1, 512)` 作为序列长度 1 的 token
- 条件注入: `prior_mu` 通过 cross-attention 或直接拼接后 linear projection
- Timestep: sinusoidal embedding -> AdaLN modulation
- 输出: 速度预测 `(B, 512)`
- 这是一个"1-token transformer"，本质上是 AdaLN-conditioned MLP

**如果后续想 patchify（扩展方案）：**
- 将 512-dim 切成 `P` 个 patch，每个 patch dim = `512/P`
- 这时 RoPE 和 self-attention 真正发挥作用
- Cola-DLM 的代码可以直接复用更多

**如果 DiT 训练不稳定（退化方案）：**
- 用 MLP (Linear -> SiLU -> Linear -> SiLU -> Linear) 替代 DiT
- 条件通过拼接 (z_t, t_embed, prior_mu) 注入
- 损失少量表达能力但大幅简化训练

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch 2.10.0 | rotary-embedding-torch>=0.8.6 | 已验证 Cola-DLM 代码在此版本下运行 |
| PyTorch 2.10.0 | einops 0.8.2 | 已安装且正常工作 |
| PyTorch 2.10.0 | CUDA 12.8 | 已安装 cu128 版本 |
| rotary-embedding-torch | einops | rotary-embedding-torch 的依赖，需要 einops |
| 整个 DiT stack | 现有 GenSR VAE | DiT 训练时 VAE 完全冻结，不需要梯度传播到 VAE。只需调用 VAE encoder 获取 (prior_mu, post_mu) 对 |

## 关键架构决策

### DiT 对 512-dim 单向量的适配

Cola-DLM 的 DiT 处理的是 `(L, d)` 序列（其中 `d=16`, `L` 为文本长度）。我们的场景是 `(1, 512)` —— 序列长度 1，维度 512。需要做以下调整：

1. **patch_size=1 不需要 patchify**：Cola 的 `PatchIn1D`/`PatchOut1D` 在 patch_size=1 时退化为 `nn.Linear(512, hidden_dim)`
2. **block-causal 退化为标准 self-attention**：只有一个 block，不需要 block-causal mask
3. **NA flatten-concat 不需要**：用标准 `(B, L, D)` batch 维度
4. **KV cache 不需要**：推理是一次性前向，不分块生成

### DiT hidden_dim 的选择

| 配置 | 参数量 | 适用场景 |
|------|--------|---------|
| hidden=256, layers=4, heads=4 | ~2M | 快速原型验证 |
| hidden=512, layers=8, heads=8 | ~15M | 推荐的起点 |
| hidden=768, layers=12, heads=12 | ~40M | 追求性能 |

建议从 `hidden=512, layers=8, heads=8` 开始，这是一个在表达能力和训练效率之间的良好平衡点。

### 条件注入方式

| 方式 | 优点 | 缺点 |
|------|------|------|
| **Cross-attention** (prior_mu 作为 K/V) | 最灵活，不改变输入维度 | 对于 1-token 输入有些浪费 |
| **拼接 + Linear** (concat z_t, prior_mu) | 最简单，等价于 MLP with conditioning | 增加输入维度到 1024 |
| **AdaLN (prior_mu 也走 timestep embedding)** | 最"DiT-native" | 需要修改 AdaLN 接受两路条件 |

推荐 **拼接方式**（最简单且有效）：将 `z_t` 和 `prior_mu` 拼接后通过输入层投影到 hidden_dim。这与 "1-token transformer" 的本质最匹配。

## Sources

- [Cola-DLM 官方代码库](https://github.com/ByteDance-Seed/Cola-DLM) -- DiT 架构实现、Flow Matching 推理管线、AdaLN-Zero 实现 (HIGH confidence)
- [Cola-DLM 论文](https://arxiv.org/abs/2605.06548) -- 数学公式 (Eq 2.1.2, 2.1.7, 2.2.3, 2.2.5) (HIGH confidence)
- [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) -- Flow Matching 参考实现，4.5k stars，PyPI 可安装 (HIGH confidence, 决定不用)
- [torchcfm](https://github.com/atong01/conditional-flow-matching) -- Conditional Flow Matching 库，MIT 许可 (HIGH confidence, 决定不用)
- [Meta Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) -- FM 数学综述，2024.12 (HIGH confidence, 理论参考)
- [PyTorch 2.10 Release](https://dev-discuss.pytorch.org/t/pytorch-release-2-10-key-dates-updated/3259) -- 2026.1.21 发布 (HIGH confidence)
- [DiT 原始论文](https://arxiv.org/abs/2212.09748) -- AdaLN-Zero 机制 (HIGH confidence, 理论参考)
- [AdaLN-Zero 分析论文](https://openreview.net/forum?id=E4roJSM9RM) -- 为什么 AdaLN-Zero 比标准 AdaLN 效果更好 (MEDIUM confidence)

---
*Stack research for: Flow Matching DiT for Symbolic Regression*
*Researched: 2026-06-09*
