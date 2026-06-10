# Phase 4: Flow Matching 训练 - Research

**Researched:** 2026-06-09
**Domain:** Flow Matching / DiT 训练循环
**Confidence:** HIGH

## Summary

Phase 4 实现 DiT 的 Flow Matching 训练脚本。核心任务是将 Cola-DLM 的 `cola_sft.py` 训练逻辑适配到 GenSR 的简化场景：512 维单向量（无序列、无 mask、无 2L trick）。训练数据来自 Phase 1 交付的 `LatentPairDataset`（在线生成 prior_mu/post_mu 对），模型来自 Phase 3 交付的 `GenSRDiT`。

GenSR 的 Flow Matching 训练比 Cola-DLM 简单一个数量级：Cola 的 `build_noisy_sample` 有 150+ 行处理序列角色、block boundary、loss mask，而 GenSR 只需 3 行向量操作（OT-path 插值 + 速度目标计算）。所有超参数（lr=1e-4, AdamW, cosine schedule, grad clip=1.0）和训练机制（EMA loss, gradient accumulation）均照搬 Cola-DLM 默认值。

**Primary recommendation:** 照搬 cola_sft.py 的训练循环骨架，将 flow_matching_step 简化为纯向量 OT-path 插值 + MSE loss，去掉所有序列/mask/2L 逻辑。预计核心训练循环 ~100 行。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 参照 cola_sft.py 写独立原生 PyTorch 脚本，手写 for loop + optimizer.step()。不用 PyTorch Lightning。
- **D-02:** 默认使用 Logit-normal 时间步采样（loc=0.0, scale=1.0），与 Cola-DLM 默认一致。也支持 uniform 作为可选项。
- **D-03:** AdamW lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01
- **D-04:** Cosine LR schedule: warmup 5% + warmdown 30% + final_lr_frac=0.0
- **D-05:** device_batch_size=4, grad_accum_steps=8 (effective batch=32)
- **D-06:** Gradient clipping max_norm=1.0
- **D-07:** EMA loss smoothing (beta=0.95) 用于日志显示
- **D-08:** 先用 print 日志（loss, lr, dt），后续再加 wandb
- **D-09:** 无 2L trick、无 NA layout、无 loss_mask、无 block-causal attention。GenSR 是 512 维单向量，所有位置都参与 loss 计算。
- **D-10:** 无 CFG（v2 feature），训练时无条件分支。

### Claude's Discretion
- 训练脚本的具体文件位置和命名
- Checkpoint 保存格式（state_dict）
- Eval 频率和方式
- 命令行参数的组织方式
- 数据加载与训练循环的具体实现细节

### Deferred Ideas (OUT OF SCOPE)
- Wandb 日志集成（后续添加，不影响训练脚本核心逻辑）
- CFG (Classifier-Free Guidance) 训练（v2 feature ADV-01）
- 多 GPU DDP 支持（当前单卡训练足够）
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FM-01 | OT-path Flow Matching: z_t = (1-t)*prior_mu + t*post_mu, velocity = post_mu - prior_mu | Cola cola_sft.py:451-534 的简化版；GenSR 无序列，纯向量操作 |
| FM-02 | Loss: MSE(v_psi(z_t, t; prior_mu), post_mu - prior_mu) | Cola cola_sft.py:530-532，去掉 loss_mask，直接 mean |
| FM-03 | CVAE/FeatureFusion/Decoder 全部冻结，只训练 DiT | LatentPairDataset 已实现冻结；DiT 独立实例化，无共享参数 |
| FM-04 | 训练日志（loss, lr, dt） | Cola cola_sft.py:673-680 的 EMA smoothing + print0 模式 |
| FM-05 | DiT checkpoint 保存与加载 | torch.save(state_dict) 格式；GenSRDiT 非 HF 模型，用原生 PyTorch 保存 |
</phase_requirements>
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | 训练框架 | 项目已有，与 CVAE/Decoder 共享 |
| argparse | stdlib | CLI 参数 | Cola-DLM 和 GenSR 统一风格 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.24+ | 随机种子 | params 初始化 |
| time | stdlib | 训练计时 | 每步 dt 日志 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手写 for loop | PyTorch Lightning | D-01 锁定为原生 PyTorch |
| print 日志 | wandb | D-08 锁定为 print，后续加 wandb |
| torch.save | HF save_pretrained | GenSRDiT 非 HF 模型，state_dict 更简洁 |

**Installation:** 无需安装新包，全部使用现有环境。

## Architecture Patterns

### Recommended Project Structure
```
dit_train/
├── __init__.py
├── model.py              # Phase 3 交付: GenSRDiT
├── data/
│   ├── __init__.py
│   └── latent_dataset.py # Phase 1 交付: LatentPairDataset
└── train_fm.py           # Phase 4 新增: Flow Matching 训练脚本
```

### Pattern 1: 原生 PyTorch 训练循环 (参照 cola_sft.py)
**What:** 手写 for loop + optimizer.step()，无框架依赖
**When to use:** 单 GPU 训练，需要完全控制训练过程
**Example:**
```python
# Cola cola_sft.py:633-691 的简化版（GenSR 适配）
for step in range(num_iterations):
    t0 = time.time()

    # --- Eval ---
    if step % eval_every == 0:
        val_loss = evaluate(dit, val_loader, eval_steps)
        print(f"Step {step:05d} | Val FM loss: {val_loss:.6f}")

    # --- Training step with gradient accumulation ---
    for micro_step in range(grad_accum_steps):
        prior_mu, post_mu = next(data_iter)
        loss = flow_matching_step(dit, prior_mu, post_mu)
        (loss / grad_accum_steps).backward()

    # LR schedule
    progress = step / max(num_iterations - 1, 1)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # EMA loss smoothing
    smooth_loss = 0.95 * smooth_loss + 0.05 * train_loss.item()
```

### Pattern 2: GenSR Flow Matching Step (极简版)
**What:** 3 行 OT-path 插值 + 1 行 MSE loss
**When to use:** 512 维单向量，无序列、无 mask
**Example:**
```python
def flow_matching_step(dit, prior_mu, post_mu):
    """GenSR FM step — cola_sft.py:451-534 的极简版。
    无序列、无 mask、无 2L trick，纯向量操作。
    """
    B = prior_mu.shape[0]
    # 时间步采样 (D-02)
    t = sample_timestep(B)  # (B,)

    # OT-path 插值 (FM-01)
    t_expand = t[:, None]  # (B, 1) 广播
    z_t = (1 - t_expand) * prior_mu + t_expand * post_mu

    # 速度目标 (FM-01)
    target = post_mu - prior_mu  # (B, 512)

    # 模型前向
    pred = dit(z_t, t, prior_mu)  # (B, 512)

    # MSE loss (FM-02)，全维度参与，无 mask
    loss = ((pred - target) ** 2).mean()
    return loss
```

### Pattern 3: Cosine LR Schedule (照搬 Cola)
**What:** warmup 5% + plateau + warmdown 30% linear decay
**When to use:** 所有 DiT 训练
**Example:**
```python
# cola_sft.py:189-196
def get_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.0):
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac
```

### Anti-Patterns to Avoid
- **使用 PyTorch Lightning:** D-01 明确锁定为原生 PyTorch
- **在 DataLoader 中使用 num_workers > 0:** LatentPairDataset 内部使用 CUDA，多进程会失败
- **对 DiT 使用 `torch.autocast("cuda", dtype=torch.bfloat16)`:** Cola-DLM 使用 bf16 autocast，但 GenSR 的小模型在 RTX 3090 上用 fp32 即可，避免 bf16 在 3090 上的性能问题
- **给 flow_matching_step 加 loss_mask:** D-09 明确不需要，全维度参与
- **保存 optimizer state 时忘了也保存 LR schedule 状态:** 如果要支持 resume 训练，需要保存 step 计数

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Timestep 采样 | 自己写分布采样 | 照搬 cola_sft.py:440-445 的 logit_normal 采样 | 数学正确性已验证 |
| LR schedule | 自己写 scheduler | 照搬 cola_sft.py:189-196 的 get_lr_multiplier | Cola-DLM 验证过的 schedule |
| EMA smoothing | 复杂的日志系统 | Cola 的 3 行 EMA: `smooth = beta * smooth + (1-beta) * loss` | 简单有效 |

**Key insight:** cola_sft.py 的核心训练机制（optimizer, schedule, accumulation, logging）都是通用组件，可以直接照搬。唯一需要简化的是 flow_matching_step 函数：从 150+ 行序列处理简化为 ~10 行向量操作。

## Common Pitfalls

### Pitfall 1: LatentPairDataset yield 的是单个样本 (512,)，不是 batch (B, 512)
**What goes wrong:** `LatentPairDataset.__iter__` 逐个 yield `(prior_mu[i], post_mu[i])`，每个是 `(512,)`。DataLoader 的 collate 会自动堆叠成 `(B, 512)`。但如果在训练脚本中手动取数据而不是用 DataLoader，shape 会出错。
**Why it happens:** LatentPairDataset 是 IterableDataset，yield 单个样本，DataLoader 负责 batching。
**How to avoid:** 使用 `create_latent_dataloader()` 工厂函数，直接得到 `(B, 512)` 的 batch。
**Warning signs:** `RuntimeError: The size of tensor a (512) must match the size of tensor b (B*512)`

### Pitfall 2: params 对象需要 max_input_dimension=10
**What goes wrong:** `parsers.py` 默认 `max_input_dimension=1`，但 `weights/checkpoint.pth` 是用 `max_input_dimension=10` 训练的（Phase 01 决策）。如果用默认值初始化，CVAE 权重形状不匹配。
**Why it happens:** NumericalEmbedder 的 total_dimension 依赖 max_input_dimension。
**How to avoid:** 训练脚本中初始化 params 后显式设置 `params.max_input_dimension = 10`。
**Warning signs:** `RuntimeError: size mismatch for cvae.encoder.layers.0.weight`

### Pitfall 3: CUDA 设置全局标志
**What goes wrong:** `symbolicregression.utils.CUDA` 默认 False，LatentPairDataset 构造时需要设为 True，否则 `to_cuda` 不会把张量移到 GPU。
**Why it happens:** LatentPairDataset 内部已经处理了这个问题（行 29），但如果训练脚本直接操作 CVAE 组件需要手动设置。
**How to avoid:** 使用 `create_latent_dataloader()`，它内部已经设置了 CUDA 标志。

### Pitfall 4: checkpoint 保存格式不兼容 HF
**What goes wrong:** GenSRDiT 不是 HuggingFace 模型，不能用 `save_pretrained()`。
**Why it happens:** Phase 3 实现的是原生 `nn.Module`，不是 `PreTrainedModel`。
**How to avoid:** 使用 `torch.save(dit.state_dict(), path)` 保存，`dit.load_state_dict(torch.load(path))` 加载。
**Warning signs:** `AttributeError: 'GenSRDiT' object has no attribute 'save_pretrained'`

### Pitfall 5: gradient accumulation 的 loss 除以 accum_steps
**What goes wrong:** Cola 的实现是 `(loss / grad_accum_steps).backward()`，这意味着 loss 值会被缩放。如果日志打印的是缩放后的 loss，会误导用户。
**Why it happens:** 需要用原始 loss 做日志，缩放后的 loss 做 backward。
**How to avoid:** `train_loss = loss.detach()` 在缩放之前获取，`loss / grad_accum_steps` 只用于 backward。

### Pitfall 6: IterableDataset 的无限迭代器 + num_iterations 不匹配
**What goes wrong:** LatentPairDataset 是 `while True` 无限迭代器，训练脚本需要自己控制何时停止。
**Why it happens:** IterableDataset 没有 `__len__`，DataLoader 不知道何时结束。
**How to avoid:** 用 `for step in range(num_iterations)` 控制训练步数，`next(data_iter)` 获取每个 batch。

## Code Examples

### GenSR Flow Matching 完整训练步骤 (对照 cola_sft.py 简化)
```python
# Source: cola_sft.py:451-534 的 GenSR 简化版

def sample_timestep(batch_size, dist="logit_normal", loc=0.0, scale=1.0):
    """照搬 cola_sft.py:440-445"""
    if dist == "uniform":
        return torch.rand(batch_size, device="cuda")
    u = torch.randn(batch_size, device="cuda")
    return torch.sigmoid(loc + scale * u)

def flow_matching_step(dit, prior_mu, post_mu, timestep_dist="logit_normal"):
    """GenSR FM step — cola_sft.py:451-534 的极简版"""
    B = prior_mu.shape[0]
    t = sample_timestep(B, dist=timestep_dist)
    z_t = (1 - t[:, None]) * prior_mu + t[:, None] * post_mu
    target = post_mu - prior_mu
    pred = dit(z_t, t, prior_mu)
    loss = ((pred - target) ** 2).mean()
    return loss, loss.detach()
```

### Checkpoint 保存与加载
```python
# Source: cola_sft.py:582-605 的 GenSR 适配版

def save_checkpoint(dit, optimizer, step, val_loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"fm_step_{step:06d}.pt")
    torch.save({
        "model_state_dict": dit.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

def load_checkpoint(dit, path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    dit.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf"))
```

### Params 初始化（照搬 Phase 1 验证脚本）
```python
# Source: tests/test_latent_dataset.py + dit_train/data/verify_dataset.py
from parsers import get_parser

params = get_parser().parse_args([])
params.device = "cuda"
params.max_input_dimension = 10  # Phase 01 决策：匹配 checkpoint 模型结构
```

### Eval 函数
```python
# Source: cola_sft.py:567-576 的简化版
@torch.no_grad()
def evaluate(dit, data_iter, eval_steps):
    dit.eval()
    losses = []
    for _ in range(eval_steps):
        prior_mu, post_mu = next(data_iter)
        loss, _ = flow_matching_step(dit, prior_mu, post_mu)
        losses.append(loss.item())
    dit.train()
    return sum(losses) / len(losses)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CMA-ES 进化搜索 | DiT Flow Matching | Cola-DLM 2025 | 从迭代优化变为单次前向传输 |
| Diffusion (DDPM) | Flow Matching (OT-path) | 2022-2023 Lipman et al. | 更简单的训练目标，更少的推理步 |
| Uniform timestep | Logit-normal timestep | Cola-DLM 默认 | 中间时间步更密集采样 |

**Deprecated/outdated:**
- PyTorch Lightning 训练框架：D-01 锁定为原生 PyTorch
- HF save_pretrained：GenSRDiT 是原生 nn.Module

## Open Questions

1. **训练总步数 (num_iterations)**
   - What we know: Cola-DLM 用 full epoch（数据集大小 / effective batch），但 GenSR 的数据是在线无限生成的
   - What's unclear: 合理的训练步数是多少
   - Recommendation: 设为 CLI 参数，默认 5000 步作为快速验证，10000-20000 步作为正式训练。可以通过观察 loss 收敛情况调整。

2. **Eval 数据来源**
   - What we know: Cola-DLM 有独立的 val_dataset，GenSR 的 LatentPairDataset 是在线生成的（无限流）
   - What's unclear: 是否需要独立的 eval 数据流
   - Recommendation: eval 使用同一个 LatentPairDataset（新实例），因为在线生成意味着每次都不同。eval_steps=20 就足够评估当前 loss 水平。

3. **Checkpoint 保存频率**
   - What we know: Cola-DLM 默认 save_every=-1（只在结束时保存）
   - What's unclear: GenSR 需要多频繁保存
   - Recommendation: 默认每 1000 步保存一次（save_every=1000），最后一步也保存。保留最近 3 个 checkpoint 以节省磁盘。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10 | 训练脚本 | True | 3.10.12 | -- |
| PyTorch | 模型训练 | True | 2.10.0+cu128 | -- |
| CUDA GPU | GPU 训练 | True | RTX 3090 (24GB) | -- |
| weights/checkpoint.pth | CVAE 冻结权重 | True | 671MB | -- |
| dit_train/model.py | GenSRDiT 模型 | True | Phase 3 交付 | -- |
| dit_train/data/latent_dataset.py | 数据生成器 | True | Phase 1 交付 | -- |

**Missing dependencies with no fallback:**
- 无

**Missing dependencies with fallback:**
- 无

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (已安装，无配置文件) |
| Config file | none |
| Quick run command | `python -m pytest tests/ -x -v` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FM-01 | OT-path 插值和速度目标正确 | unit | `python -m pytest tests/test_fm_train.py::test_ot_path_interpolation -x` | Wave 0 |
| FM-02 | MSE loss 计算正确 | unit | `python -m pytest tests/test_fm_train.py::test_fm_loss -x` | Wave 0 |
| FM-03 | 只有 DiT 参数有梯度 | unit | `python -m pytest tests/test_fm_train.py::test_only_dit_grad -x` | Wave 0 |
| FM-04 | 训练 loss 下降（短训练） | smoke | `python -m pytest tests/test_fm_train.py::test_loss_decreases -x` | Wave 0 |
| FM-05 | Checkpoint 保存和加载 | unit | `python -m pytest tests/test_fm_train.py::test_checkpoint_save_load -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_fm_train.py -x -v`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_fm_train.py` -- covers FM-01 through FM-05
- [ ] Framework install: none -- pytest already available

## Sources

### Primary (HIGH confidence)
- `Cola-DLM-main/scripts/cola_sft.py` -- 完整训练脚本，包含所有训练机制
- `dit_train/model.py` -- Phase 3 交付的 GenSRDiT 模型定义
- `dit_train/data/latent_dataset.py` -- Phase 1 交付的数据生成器
- `.planning/phases/04-flow-matching/04-CONTEXT.md` -- Phase 4 决策

### Secondary (MEDIUM confidence)
- `cola_sft.py:179-196` -- AdamW + cosine LR schedule (验证通过源码阅读)
- `cola_sft.py:440-445` -- Logit-normal 时间步采样 (验证通过源码阅读)
- `cola_sft.py:451-534` -- flow_matching_step (验证通过源码阅读)

### Tertiary (LOW confidence)
- 无

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- 全部使用已有环境，无新依赖
- Architecture: HIGH -- 照搬 Cola-DLM 验证过的模式，GenSR 简化版
- Pitfalls: HIGH -- 从 cola_sft.py 源码和 Phase 1-3 交付物中直接发现

**Research date:** 2026-06-09
**Valid until:** 2026-07-09 (stable, 无快速变化的外部依赖)
