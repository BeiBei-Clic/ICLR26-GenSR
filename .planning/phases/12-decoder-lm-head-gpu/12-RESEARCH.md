# Phase 12: Decoder lm_head 微调多 GPU 并行训练 - Research

**Researched:** 2026-06-10
**Domain:** PyTorch DDP 多卡训练改造
**Confidence:** HIGH

## Summary

将 `finetune_lm_head.py` 从单卡训练改造为 DDP 多卡训练，直接参照 `train_fm.py` 已验证的 DDP 模式。核心挑战是 DDP 包装策略：decoder 有 74.6M 参数，但只有 lm_head 的 5.3M 参数可训练（92.9% 冻结），不能包装整个 decoder。解决方案是只包装 `decoder.lm_head` 子模块为 DDP，这样 decoder 内部调用 `self.lm_head(x)` 时自动走 DDP 前向钩子。数据生成（`env.gen_expr`）是独立随机的，每个 rank 各自生成不重叠的数据，无需 DistributedSampler。验证和保存仅在 rank 0 执行。

**Primary recommendation:** 只 DDP 包装 `decoder.lm_head`，其余逻辑按 `train_fm.py` 模式加 `dist.init_process_group` / `rank` 判断 / `dist.barrier()`。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 用 PyTorch DDP + torchrun，跟 `train_fm.py` 的多卡模式一致。每个 GPU 独立运行模型副本，梯度自动 all-reduce 同步。
- **D-02:** 只在 rank 0 上跑验证循环和保存最优权重，跟 `train_fm.py` 一致。其他 rank 跳过验证。

### Claude's Discretion
- DDP wrapper 的具体实现细节（`DistributedDataParallel` 包装哪些模块）
- 数据并行的梯度累积适配
- 日志只在 rank 0 输出
- torchrun 启动命令的具体参数

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.distributed | PyTorch 内置 | DDP 多卡通信 | `train_fm.py` 已验证可用 |
| torch.nn.parallel.DistributedDataParallel | PyTorch 内置 | DDP 模型包装 | 标准 PyTorch DDP |
| torchrun | PyTorch 内置 | DDP 启动器 | `train_fm.py` 使用方式 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| NCCL backend | PyTorch 内置 | GPU 间梯度同步 | DDP 必需 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 只包装 lm_head | 包装整个 decoder + find_unused_parameters=True | 包装整个 decoder 会导致 92.9% 参数无梯度，DDP 需要开启 find_unused_parameters，性能差且可能出现同步问题 |

**Installation:**
无需额外安装，全部是 PyTorch 内置功能。

## Architecture Patterns

### DDP 包装策略：只包装 lm_head

**核心发现：** decoder 有 74,642,484 个参数，但 lm_head 只有 5,279,796 个参数（7.1%）。如果包装整个 decoder，需要 `find_unused_parameters=True`，DDP backward 时需逐参数检查梯度，性能差且可能出错。

**推荐策略：** 将 `decoder.lm_head` 替换为 DDP 包装版本：

```python
# 保存原始引用用于保存/加载
lm_head_raw = decoder.lm_head

# DDP 包装
if ddp:
    decoder.lm_head = DDP(decoder.lm_head, device_ids=[local_rank], output_device=local_rank)

# 训练时，decoder 内部调用 self.lm_head(x) 自动走 DDP 前向钩子
# decoder.predict() 中的 scores = self.lm_head(x) 正常工作

# 保存时用原始引用
torch.save(lm_head_raw.state_dict(), save_path)
```

### DDP 初始化模式（照搬 train_fm.py）

```python
# 来源: dit_train/train_fm.py:209-221
ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    local_rank = 0
    rank = 0
    world_size = 1

is_master = rank == 0
device = f"cuda:{local_rank}"
```

### 数据生成：每个 rank 独立

`finetune_lm_head.py` 的数据生成方式是 `env.gen_expr(train=True)`，每次调用随机生成新表达式。这意味着：
- 每个 rank 各自独立生成不同的训练数据
- 无需 DistributedSampler 或数据分片
- 等效于 batch_size * world_size 的有效 batch

### 验证循环：只在 rank 0

```python
# 来源: dit_train/train_fm.py:307-319 的模式
if (step + 1) % 50 == 0:
    if is_master:
        decoder.eval()
        # ... 验证逻辑 ...
        decoder.train()
        # 保存最优权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lm_head_raw.state_dict(), save_path)
    if ddp:
        dist.barrier()  # 等待 rank 0 完成验证
```

### 日志：只在 rank 0 输出

```python
if is_master:
    print(f"step {step:04d} | loss: {loss_val:.4f} | ...")
```

### 清理：训练结束销毁进程组

```python
# 来源: dit_train/train_fm.py:342-343
if ddp:
    dist.destroy_process_group()
```

### Anti-Patterns to Avoid
- **包装整个 decoder:** 92.9% 参数冻结，DDP 需 `find_unused_parameters=True`，性能差且可能导致梯度同步错误
- **忘记 `dist.barrier()` after validation:** 其他 rank 可能在 rank 0 还在验证时就进入下一步训练，导致死锁或不一致
- **忘记 `torch.cuda.set_device(local_rank)`:** 所有进程默认用 GPU 0，DDP 要求每个进程绑定自己的 GPU

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 梯度同步 | 手动 all-reduce | DDP 自动同步 | DDP 内部 bucket 化梯度、异步通信，比手动 all-reduce 快很多 |
| 进程间通信 | 自己写 socket/pipe | `torch.distributed` + torchrun | torchrun 自动设置 RANK/LOCAL_RANK/WORLD_SIZE 环境变量 |
| 分布式采样 | 手动分数据 | 每个 rank 独立生成 | 数据是随机生成的，天然不重叠 |

**Key insight:** 数据是 `env.gen_expr(train=True)` 随机生成的，不是从固定数据集读取，所以不需要 DistributedSampler。

## Common Pitfalls

### Pitfall 1: lm_head.weight 与 tok_embed.weight 权重共享

**What goes wrong:** `share_inout_emb=True` 导致 `lm_head.weight` 就是 `tok_embed.weight`。解冻 `lm_head.parameters()` 时，`tok_embed.weight` 也变为可训练。DDP 包装 `lm_head` 时，shared weight 只会被 bucket 一次并正确同步。
**Why it happens:** Python 对象引用共享，`nn.Linear.weight` 和 `nn.Embedding.weight` 指向同一个 tensor。
**How to avoid:** 无需特殊处理。DDP 按唯一 tensor identity 做 bucket，shared weight 自动只同步一次。保存时用 `lm_head_raw.state_dict()` 即可。
**Warning signs:** 如果 DDP 报 "found unused parameters" 警告，说明 `tok_embed.weight` 的梯度被 DDP 认为属于 "unused"（因为它不通过 DDP wrapper 前向），但实际上它被 lm_head 的 DDP wrapper 管了。

### Pitfall 2: DDP 的 "first forward must be training" 规则

**What goes wrong:** DDP 要求第一次前向传播是训练模式（不是 eval），否则内部 bucket 设置可能出错。
**Why it happens:** DDP 在第一次 forward 时设置梯度 bucket，如果首次 forward 是 eval 模式（no_grad），bucket 无法正确初始化。
**How to avoid:** 验证循环必须在训练步骤之后运行（当前代码已经是这个顺序，不需要改）。`train_fm.py` 注释也强调了这点："训练必须在 eval 之前，确保 DDP 的首次 forward 是训练而非 eval"。
**Warning signs:** DDP 报错 "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one."

### Pitfall 3: decoder.eval() / decoder.train() 与 DDP 的交互

**What goes wrong:** DDP 包装的是 `lm_head`，但 `decoder.eval()` 会把整个 decoder 设为 eval 模式，包括 lm_head。这会影响 BatchNorm/Dropout 等层的行为。
**Why it happens:** `decoder.eval()` 递归设置所有子模块为 eval，包括被 DDP 包装的 `lm_head`。
**How to avoid:** 这实际上是期望行为——验证时 decoder 确实应该在 eval 模式。DDP 包装不影响 eval/train 模式切换。只需要确保验证后恢复 `decoder.train()`。
**Warning signs:** 无特殊风险。

### Pitfall 4: torch.save 与 DDP state_dict

**What goes wrong:** DDP 包装后的 `lm_head` 的 `state_dict()` 键名有 `module.` 前缀（如 `module.weight` 而不是 `weight`），导致加载时键名不匹配。
**Why it happens:** DDP 将原始模块存为 `self.module`，`state_dict()` 自动加 `module.` 前缀。
**How to avoid:** 保存和加载都用 `lm_head_raw`（未包装的原始引用），不用 DDP wrapper 的 state_dict。`train_fm.py` 也是这么做的：`dit_raw = dit` 保存原始引用。
**Warning signs:** 加载权重时出现 "Missing key(s)" 或 "Unexpected key(s)" 错误。

### Pitfall 5: 多进程初始化 FunctionEnvironment

**What goes wrong:** `env.gen_expr()` 在 DDP 多进程中，如果使用 fork 模式且主进程已初始化 CUDA，会导致 CUDA re-initialization 错误。
**Why it happens:** `finetune_lm_head.py` 在主循环中直接调用 `env.gen_expr()`，不走 DataLoader worker，所以不涉及 spawn/fork 问题。
**How to avoid:** 当前代码每个 rank 在主进程中独立生成数据，不使用 DataLoader worker，没有 spawn/fork 问题。直接用 torchrun 启动多个进程即可。
**Warning signs:** 无特殊风险。

## Code Examples

### 完整的 DDP 改造点清单（对比 train_fm.py）

以下是 `finetune_lm_head.py` 需要修改的具体位置：

**1. 导入 DDP 相关模块（文件开头）**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

**2. DDP 初始化（替换 `device = "cuda"` 行）**
```python
# 照搬 train_fm.py:209-221
ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    local_rank = 0
    rank = 0
    world_size = 1

is_master = rank == 0
device = f"cuda:{local_rank}"
```

**3. DDP 包装 lm_head（在冻结/解冻逻辑之后）**
```python
lm_head_raw = decoder.lm_head  # 保存原始引用
if ddp:
    decoder.lm_head = DDP(decoder.lm_head, device_ids=[local_rank], output_device=local_rank)
```

**4. Optimizer 使用 DDP 包装后的参数**
```python
optimizer = torch.optim.AdamW(
    decoder.lm_head.parameters(),  # 通过 DDP wrapper 获取参数
    lr=args.learning_rate,
)
```

**5. 日志只在 rank 0 输出**
```python
if is_master:
    print(f"\n=== Decoder lm_head Fine-tuning ===")
    # ... 所有 print 语句加 is_master 判断
```

**6. 验证循环只在 rank 0 执行，加 dist.barrier()**
```python
if (step + 1) % 50 == 0:
    if is_master:
        decoder.eval()
        # ... 验证逻辑不变 ...
        decoder.train()
        # 保存最优权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lm_head_raw.state_dict(), save_path)
    if ddp:
        dist.barrier()
```

**7. 训练结束清理**
```python
if ddp:
    dist.destroy_process_group()
```

### torchrun 启动命令

```bash
# 单卡（向后兼容）
python dit_train/finetune_lm_head.py

# 4 卡 DDP
torchrun --nproc_per_node=4 dit_train/finetune_lm_head.py

# 2 卡 DDP + 自定义参数
torchrun --nproc_per_node=2 dit_train/finetune_lm_head.py --num-iterations 1000 --batch-size 8
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 无 DDP | train_fm.py 已有完整 DDP | Phase 4 | DDP 模式已在本项目中验证可用 |

**Deprecated/outdated:**
- 无

## Open Questions

1. **DDP 与逐样本 CE loss 的兼容性**
   - What we know: 当前代码逐样本计算 CE loss 再求和求平均。每个 rank 独立计算自己的 batch 内平均 loss，DDP 自动 all-reduce 梯度。等效于 world_size * batch_size 的有效 batch。
   - What's unclear: 无。这种模式在 `train_fm.py` 中已经验证可行。
   - Recommendation: 无需特殊处理，DDP 自动同步梯度。

2. **梯度累积是否需要调整**
   - What we know: 当前 `finetune_lm_head.py` 没有梯度累积。如果需要加入，参照 `train_fm.py` 的模式：`loss / grad_accum_steps` backward。
   - Recommendation: 本 phase 不加梯度累积，保持简单。如需要可后续添加。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch DDP | 多卡训练 | True | 2.5.1+cu124 | — |
| NCCL backend | DDP GPU 通信 | True | 内置 | — |
| torchrun | DDP 启动 | True | 内置 | — |
| NVIDIA GPU x4 | DDP 训练 | True | RTX 3090 x4 24GB | — |
| CUDA | GPU 运行时 | True | 12.4 (driver 595.71) | — |

**Missing dependencies with no fallback:**
- 无

**Missing dependencies with fallback:**
- 无

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | 无专用配置 |
| Quick run command | `python -m pytest tests/test_lm_head_finetune.py -x -v` |
| Full suite command | `python -m pytest tests/test_lm_head_finetune.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-01 | DDP 初始化 + torchrun 启动 | smoke | `torchrun --nproc_per_node=2 dit_train/finetune_lm_head.py --num-iterations 2 --batch-size 2` | False (Wave 0) |
| D-02 | 验证和保存仅在 rank 0 执行 | manual | 检查只有 rank 0 输出验证日志和保存权重 | False |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_lm_head_finetune.py -x -v`
- **Per wave merge:** `torchrun --nproc_per_node=2 dit_train/finetune_lm_head.py --num-iterations 5 --batch-size 2`
- **Phase gate:** 2 卡 DDP 跑 10 步不报错，rank 0 保存 lm_head_best.pt

### Wave 0 Gaps
- [ ] DDP smoke test — 验证 torchrun 启动后多进程不报错
- [ ] 确认 finetune_lm_head.py 的 print 语句全部加了 is_master 判断

## Sources

### Primary (HIGH confidence)
- `dit_train/train_fm.py` — 已验证的 DDP 参考实现，直接照搬模式
- `dit_train/finetune_lm_head.py` — 需要改造的目标文件
- `symbolicregression/model/transformer.py` — TransformerModel 中 lm_head 定义和 weight sharing 逻辑
- PyTorch DDP 官方文档 — DDP 包装子模块的行为

### Secondary (MEDIUM confidence)
- `symbolicregression/model/__init__.py` — 确认 `seq_decoder` 是 `TransformerModel` 类
- `parsers.py` — 确认 `share_inout_emb` 默认 True

### Tertiary (LOW confidence)
- 无

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch DDP，项目中已有验证过的实现
- Architecture: HIGH - 完全照搬 train_fm.py 模式，只调整 DDP 包装范围
- Pitfalls: HIGH - 通过实际代码分析和参数统计验证

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (stable technology)
