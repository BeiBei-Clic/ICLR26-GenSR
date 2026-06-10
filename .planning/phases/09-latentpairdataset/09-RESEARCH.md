# Phase 9: 优化 LatentPairDataset 数据生成性能 - Research

**Researched:** 2026-06-10
**Domain:** PyTorch DataLoader 多进程 + GPU 推理并行
**Confidence:** HIGH

## Summary

LatentPairDataset 当前以单进程（num_workers=0）串行生成数据，每次循环调用 gen_expr(CPU) + embedder_f(CPU+GPU) + CVAE forward(GPU)。实测耗时约 58.5ms/样本，吞吐量仅 14.9 samples/s。当 device_batch_size=20 时，仅数据生成就需要 ~1.3s/step，远跟不上 GPU 训练速度。

**瓶颈量化：**
- gen_expr (CPU 随机表达式树构建): 7.5ms (12.8%)
- embedder_f (CPU 编码 + GPU 嵌入): 8.7ms (14.9%)
- embedder_e (GPU 方程嵌入): 0.3ms (0.4%)
- CVAE forward (GPU Transformer 编码): 42.0ms (71.8%)
- **CVAE forward 是绝对瓶颈，占 71.8%**

**核心发现：D-03（fork 共享 CUDA context）方案不可行。** 实测验证：主进程已初始化 CUDA 后，fork 的子进程调用 `.cuda()` 会报 "Cannot re-initialize CUDA in forked subprocess"。这在 PyTorch 官方 issue #13883 中有记录。需要改为 spawn 模式，每个 worker 独立初始化 CUDA 并加载冻结模型。

**Primary recommendation:** 使用 spawn 模式 DataLoader + worker 内延迟初始化冻结模型 + yield 完整 batch 跳过 collate。每个 worker 独立持有 CVAE GPU 副本（~641MB/worker），通过 persistent_workers 避免重复创建开销。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 保持在线生成模式，不使用预生成缓存。使用 DataLoader 多进程 (`num_workers>0`) 并行生成数据
- **D-02:** 不使用预生成缓存方案的原因：CVAE 冻结后每次生成的数据统计上等价，但用户倾向保持数据多样性，在线生成更灵活
- **D-03:** 使用共享 CVAE GPU 权重方案：主进程加载 CVAE 到 GPU，worker 进程通过 fork 共享 CUDA context 访问同一份 GPU 权重
- **D-04:** Worker 数量由 Claude 根据实测结果决定（需考虑 DDP 下每卡 worker 数与显存占用的平衡）
- **D-05:** LatentPairDataset.__iter__ 直接 yield 完整 batch 的 (prior_mu, post_mu) 张量对，跳过 DataLoader 的逐样本 collate 开销
- **D-06:** batch_size 逻辑从 DataLoader 移到 Dataset 内部

### Claude's Discretion
- DataLoader worker 数量的具体值
- fork 共享 CUDA context 的具体实现方式（`torch.multiprocessing.set_start_method('fork')` + `mp.set_sharing_strategy('file_system')` 等）
- 是否需要 pin_memory、prefetch_factor 等 DataLoader 优化参数
- 训练循环中 `next(data_iter)` 的调用方式是否需要配合调整

### Deferred Ideas (OUT OF SCOPE)
None -- 讨论保持在 phase 范围内
</user_constraints>

<phase_requirements>
## Phase Requirements

本 phase 没有正式的 REQ-ID（不在 REQUIREMENTS.md 中），但 CONTEXT.md 定义了明确的需求：

| 需求 | 描述 | 研究支持 |
|------|------|----------|
| PERF-01 | 消除 GPU 训练时的数据供给瓶颈 | 性能基准实测：14.9 samples/s -> 目标 >100 samples/s |
| PERF-02 | 增大 batch_size 后 GPU 利用率不下降 | spawn 并行 + yield batch 跳过 collate |
| PERF-03 | 兼容 4 卡 DDP 训练 | 每个 rank 独立 DataLoader，spawn 模式无跨进程依赖 |
</phase_requirements>

## Critical Issue: D-03 不可行

### 实测验证

| 测试场景 | 结果 | 说明 |
|----------|------|------|
| fork + 主进程未初始化 CUDA | 通过 | 不适用 -- DiT 训练必须用 CUDA |
| fork + 主进程已初始化 CUDA | **失败** | RuntimeError: Cannot re-initialize CUDA in forked subprocess |
| spawn + 延迟初始化 CUDA | **通过** | 每个 worker 独立初始化 CUDA 和加载模型 |
| spawn + yield batch + batch_size=None | **通过** | DataLoader 直接透传 dataset yield 的完整 batch |

### D-03 替代方案

由于 D-03 的 fork 方案不可行，需要修正为 spawn 方案：

1. **multiprocessing_context='spawn'** -- DataLoader 参数指定 spawn 模式
2. **worker 内延迟初始化模型** -- `__init__` 只存配置参数，`__iter__` 首次调用时构建 env + modules + 加载权重到 GPU
3. **persistent_workers=True** -- 避免每个 step 重建 worker（模型加载只需一次）
4. **每个 worker 独立 GPU 显存** -- CVAE + embedders 约 641 MB/worker

### 显存预算

| 组件 | 显存占用 |
|------|----------|
| CVAE + embedders (冻结) | 641 MB |
| DiT 模型 (训练中) | ~100 MB |
| 单 worker 总计 | ~741 MB |

**DDP 场景 (4x RTX 3090, 24GB each):**
- 每张卡: 1 DiT + N workers x 641 MB
- N=2 workers: 1.4 GB (冻结模型) + DiT + 激活值 << 24 GB
- N=4 workers: 2.8 GB (冻结模型) + DiT + 激活值 < 24 GB

**推荐 num_workers=2**（保守），最多 num_workers=4（激进）。

## Standard Stack

### Core -- 无新依赖

本项目所有依赖已安装，此 phase 不引入新库。

| 库 | 版本 | 用途 |
|----|------|------|
| torch.utils.data.IterableDataset | PyTorch 2.10.0 内置 | 无限迭代数据集基类 |
| torch.utils.data.DataLoader | PyTorch 2.10.0 内置 | 多进程数据加载器 |

### 关键 DataLoader 参数

| 参数 | 值 | 原因 |
|------|-----|------|
| num_workers | 2-4 | CVAE forward 是 GPU 密集，过多 worker 无收益 |
| multiprocessing_context | 'spawn' | CUDA 安全，主进程已初始化 CUDA |
| persistent_workers | True | 避免重复加载模型（641MB/worker） |
| batch_size | None | Dataset 直接 yield 完整 batch，跳过 collate |
| collate_fn | 不需要 | batch_size=None 时 DataLoader 透传单个 yield |
| pin_memory | False | Dataset 直接 yield GPU -> CPU -> .to(device) 的张量，pin_memory 无意义 |
| prefetch_factor | 2 (默认) | 2 workers x 2 prefetch = 4 batch 预取足够 |

## Architecture Patterns

### 当前架构 (num_workers=0)

```
main process:
  train loop -> next(data_iter) -> Dataset.__iter__
    -> gen_expr() [CPU, 7.5ms]
    -> embedder_f() [CPU+GPU, 8.7ms]
    -> CVAE.forward() [GPU, 42ms]
    -> yield single (prior_mu, post_mu)
  -> DataLoader collate (batch_size=20)
  -> .to(device)
  -> DiT forward + backward
```

**问题:** 串行生成，GPU 利用率低。batch_size 越大等待越久。

### 目标架构 (num_workers>0, spawn)

```
main process:
  DataLoader(spawn, num_workers=2, batch_size=None, persistent_workers=True)
    |
    +-- Worker 0 (独立进程, 独立 CUDA context):
    |     __iter__ 首次调用:
    |       build_env() + build_modules() + reload_model() -> GPU
    |     循环:
    |       gen_expr() [CPU] -> embed [CPU+GPU] -> CVAE [GPU]
    |       yield (prior_mu_batch, post_mu_batch)  # 完整 batch, 已在 GPU
    |
    +-- Worker 1 (同上, 独立进程)

  train loop:
    prior_mu, post_mu = next(data_iter)  # 直接 GPU 张量
    # 跳过 .to(device) -- 已经在 GPU
    DiT forward + backward
```

### Pattern 1: Worker 内延迟初始化冻结模型

**What:** 不在 `__init__` 中加载 GPU 模型，在 `__iter__` 首次调用时初始化。
**When:** spawn 模式下避免 pickle 大型 GPU 对象，且需要 worker 独立 CUDA context。
**Example:**

```python
class LatentPairDataset(IterableDataset):
    def __init__(self, params, batch_size, device, checkpoint_path):
        # 只保存配置，不初始化 GPU 模型
        self.params = params
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._initialized = False

    def _lazy_init(self):
        """在 worker 进程中首次调用时初始化。"""
        params = self.params
        symbolicregression.utils.CUDA = not params.cpu

        if isinstance(params.tasks, list):
            params.tasks = ",".join(params.tasks)

        env = build_env(params)
        env.rng = np.random.RandomState()
        self.env = env

        modules = build_modules(env, params)
        reload_model(
            modules,
            modules_to_load=["cvae", "data_encoder", "token_embed"],
            path=self.checkpoint_path,
            requires_grad=False,
        )

        self.embedder_f = modules["data_encoder"]
        self.embedder_e = modules["token_embed"]
        self.vae_model = modules["cvae"]
        self.vae_model.eval()
        for module in [self.vae_model, self.embedder_f, self.embedder_e]:
            for param in module.parameters():
                param.requires_grad = False

        self._initialized = True

    def __iter__(self):
        if not self._initialized:
            self._lazy_init()
        while True:
            # 生成一个完整 batch
            with torch.no_grad():
                prior_list, post_list = [], []
                for _ in range(self.batch_size):
                    # ... gen_expr + embed + CVAE forward ...
                    prior_list.append(prior_mu_i)
                    post_list.append(post_mu_i)
                yield (torch.cat(prior_list), torch.cat(post_list))
```

### Pattern 2: yield 完整 batch 跳过 collate

**What:** Dataset `__iter__` 直接 yield `(B, 512)` 的 batch 张量，DataLoader 设 batch_size=None 透传。
**When:** IterableDataset 内部已经组装好 batch，不需要 DataLoader 再 collate。

```python
# Dataset side:
def __iter__(self):
    ...
    yield batch_prior, batch_post  # each is (B, 512) tensor

# DataLoader side:
loader = DataLoader(
    dataset,
    batch_size=None,        # 关键：禁用自动 batching
    num_workers=2,
    multiprocessing_context='spawn',
    persistent_workers=True,
)

# Training loop:
prior_mu, post_mu = next(data_iter)  # 直接是 (B, 512)
# 无需 .to(device) -- 已经在 worker 的 GPU 上
```

### Anti-Patterns to Avoid

- **在 `__init__` 中初始化 CUDA/GPU 模型 + spawn:** `__init__` 在主进程执行，GPU 对象 pickle 到 worker 会失败或复制整个 GPU 状态
- **fork + 主进程已初始化 CUDA:** 报 "Cannot re-initialize CUDA in forked subprocess"
- **yield GPU 张量依赖 collate:** DataLoader collate 无法处理 GPU 张量列表，会报错
- **num_workers 过多 (>=8):** CVAE forward 是 GPU bound，同一 GPU 上 8 个 worker 会竞争 CUDA 流，不会更快

## Don't Hand-Roll

| 问题 | 不要自建 | 使用替代 |
|------|----------|----------|
| DataLoader worker 进程管理 | 手写 multiprocessing.Process | DataLoader(num_workers=N, multiprocessing_context='spawn') |
| Worker 初始化钩子 | 手写 IPC 传递模型参数 | worker 内 `_lazy_init()` 模式 |
| batch collate | 手写 stack/cat 逻辑 | batch_size=None 跳过 collate |
| 进程间数据传输 | 手写 shared_memory / Queue | DataLoader 内部基于 index_queue + data_queue 的 IPC |

**Key insight:** PyTorch DataLoader 的 spawn + persistent_workers 已经处理了进程生命周期管理、数据序列化、错误恢复。自建多进程只会增加复杂度和 bug。

## Common Pitfalls

### Pitfall 1: spawn 模式要求 `__main__` 守卫
**What goes wrong:** spawn 模式会在子进程中重新导入主模块。如果 DataLoader 创建代码不在 `if __name__ == '__main__'` 内，会导致无限递归。
**Why it happens:** spawn 重新执行模块顶层代码来构建子进程环境。
**How to avoid:** 确保 `train_fm.py` 的 DataLoader 创建在 `main()` 函数内（已有 `if __name__ == "__main__"` 守卫）。
**Warning signs:** RuntimeError about recursive import or infinite worker spawning.
**状态:** train_fm.py 已经正确使用 `if __name__ == "__main__": main()` 模式，无需修改。

### Pitfall 2: IterableDataset + num_workers > 0 产生重复数据
**What goes wrong:** 每个 worker 都会独立执行完整的 `__iter__`，产生相同的数据序列。
**Why it happens:** IterableDataset 没有自动分片机制，所有 worker 看到相同的数据源。
**How to avoid:** 由于 gen_expr 是随机生成（每个样本独立），重复数据实际上不是问题 -- 每个 worker 的 RandomState 种子不同，产生的数据统计上等价但实例不同。
**Warning signs:** 训练 loss 不收敛（如果使用了确定性数据源）。
**状态:** 本项目的 gen_expr 使用 `np.random.RandomState()` 生成随机表达式，每个 worker 的随机种子由 numpy 内部管理，天然不重复。

### Pitfall 3: Worker 中 gen_expr 的 env.rng 非线程安全
**What goes wrong:** 如果多个线程/协程共享同一个 env.rng，会产生竞争条件。
**Why it happens:** numpy RandomState 不是线程安全的。
**How to avoid:** 每个 worker 是独立进程（spawn），有独立的地址空间，不存在线程安全问题。
**状态:** spawn 模式下每个 worker 有完整的进程隔离，安全。

### Pitfall 4: GPU 显存不足 (OOM)
**What goes wrong:** N 个 worker 各持有一份冻结模型 (641MB/worker)，加上 DiT 训练的激活值，超出 GPU 显存。
**Why it happens:** spawn 模式每个 worker 独立初始化 CUDA，不共享 GPU 显存。
**How to avoid:** 监控 GPU 显存使用，限制 num_workers。RTX 3090 (24GB) 上建议 num_workers=2 (额外 1.3GB)。
**Warning signs:** CUDA out of memory 错误，通常在 worker 初始化时出现。

### Pitfall 5: yield GPU 张量后未 detach / clone
**What goes wrong:** 如果 worker yield 的 GPU 张量仍关联着计算图，会导致 GPU 显存泄漏。
**Why it happens:** `torch.no_grad()` 上下文外 yield 时，可能意外保留了计算图引用。
**How to avoid:** 确保所有 yield 的张量都是 `.clone()` 过的（当前代码已做 clone），且在 `torch.no_grad()` 块内完成计算。
**状态:** 当前代码已经使用 `batch_prior = prior_mu.clone()`，但需要注意 clone 后的张量仍然在 GPU 上，通过 DataLoader IPC 传输到主进程时 PyTorch 会自动处理。

### Pitfall 6: DDP 下 eval 使用 data_iter 与 worker 竞争
**What goes wrong:** train_fm.py 的 evaluate() 也调用 `next(data_iter)`，与训练循环共享同一个迭代器。
**Why it happens:** evaluate 在训练循环中间调用，可能触发 worker 竞争。
**How to avoid:** evaluate 使用独立的 data_iter 或直接从同一个 iter 取数据（当前代码的方式，因为 IterableDataset 是无限生成，eval 取走几个 batch 不影响训练）。
**状态:** 当前设计可接受。evaluate 消耗几个 batch 不影响无限生成的数据流。

## Code Examples

### 完整的优化后 LatentPairDataset

```python
class LatentPairDataset(IterableDataset):
    def __init__(self, params, batch_size=16, device="cuda",
                 checkpoint_path="weights/checkpoint.pth"):
        # 只保存配置，不初始化 GPU 或加载模型
        self.params = params
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._initialized = False

    def _lazy_init(self):
        """在 worker 进程中延迟初始化。"""
        import symbolicregression.utils
        symbolicregression.utils.CUDA = not self.params.cpu

        params = self.params
        if isinstance(params.tasks, list):
            params.tasks = ",".join(params.tasks)

        env = build_env(params)
        env.rng = np.random.RandomState()
        self.env = env

        modules = build_modules(env, params)
        reload_model(
            modules,
            modules_to_load=["cvae", "data_encoder", "token_embed"],
            path=self.checkpoint_path,
            requires_grad=False,
        )

        self.embedder_f = modules["data_encoder"]
        self.embedder_e = modules["token_embed"]
        self.vae_model = modules["cvae"]
        self.vae_model.eval()
        for module in [self.vae_model, self.embedder_f, self.embedder_e]:
            for param in module.parameters():
                param.requires_grad = False
        self._initialized = True

    def __iter__(self):
        if not self._initialized:
            self._lazy_init()

        while True:
            with torch.no_grad():
                prior_list, post_list = [], []
                for _ in range(self.batch_size):
                    samples, _ = self.env.gen_expr(train=True)

                    x_to_fit = samples["X_to_fit"]
                    y_to_fit = samples["Y_to_fit"]
                    x1 = [[[x, y] for x, y in zip(xs, ys)]
                           for xs, ys in zip(x_to_fit, y_to_fit)]
                    x1, len1 = self.embedder_f(x1)

                    x2, len2 = self.env.batch_equations(
                        self.env.word_to_idx(
                            [samples["tree_encoded"]], float_input=False
                        )
                    )
                    x2, len2 = to_cuda(x2, len2)
                    x2_e = self.embedder_e(x2.transpose(0, 1)).transpose(0, 1)

                    prior_mu, _, post_mu, _, _, _, _ = self.vae_model(
                        x1, x2_e, len1, len2, mode="train"
                    )
                    prior_list.append(prior_mu.clone())
                    post_list.append(post_mu.clone())

                yield (torch.cat(prior_list, dim=0),
                       torch.cat(post_list, dim=0))


def create_latent_dataloader(params, batch_size=16, num_workers=2,
                              device="cuda",
                              checkpoint_path="weights/checkpoint.pth"):
    dataset = LatentPairDataset(
        params, batch_size=batch_size,
        device=device, checkpoint_path=checkpoint_path,
    )
    return DataLoader(
        dataset,
        batch_size=None,                # D-05: 跳过 collate
        num_workers=num_workers,
        multiprocessing_context='spawn', # CUDA 安全
        persistent_workers=True,         # 避免重复加载模型
        pin_memory=False,                # 已在 GPU 上，无需 pin
    )
```

### 训练循环调整

```python
# train_fm.py 中的改动
loader = create_latent_dataloader(
    params, batch_size=args.device_batch_size,
    num_workers=2,  # 新增参数
    checkpoint_path=args.checkpoint_path,
)
data_iter = iter(loader)

for step in range(start_step, num_iterations):
    for micro_step in range(args.grad_accum_steps):
        prior_mu, post_mu = next(data_iter)
        # 移除 .to(device) -- 已经在 worker 的 GPU 上
        # 但需要注意：worker 使用的 device 是 "cuda:0"
        # DDP 场景下 worker 用 local_rank 的 GPU
        loss, train_loss = flow_matching_step(dit, prior_mu, post_mu, args.timestep_dist)
        (loss / args.grad_accum_steps).backward()
```

## State of the Art

| 旧方案 | 当前方案 | 变更时间 | 影响 |
|--------|----------|----------|------|
| num_workers=0, 主进程串行 | num_workers>0, spawn 并行 | 本 phase | 数据生成并行化 |
| yield 逐样本 + DataLoader collate | yield 完整 batch + batch_size=None | 本 phase | 消除 collate 开销 |
| fork 共享 CUDA context (D-03) | spawn 独立 CUDA context | 本 phase | 修正不可行方案 |

**Deprecated/outdated:**
- D-03 fork 共享 CUDA context: 已验证不可行（主进程已初始化 CUDA 后 fork 子进程无法使用 CUDA）

## Open Questions

1. **Worker device 绑定**
   - What we know: DDP 场景下每个 rank 看到不同的 GPU (local_rank)。Worker 需要在正确的 GPU 上初始化模型。
   - What's unclear: spawn worker 继承的 CUDA_VISIBLE_DEVICES 是否与 local_rank 对应。
   - Recommendation: 在 `_lazy_init` 中使用 `torch.cuda.set_device(local_rank)` 或将 device 传递给 Dataset 构造函数。

2. **Worker 初始化耗时**
   - What we know: build_env + build_modules + reload_model 约 2-3 秒。persistent_workers=True 意味着只在首次 iter 时初始化。
   - What's unclear: 如果训练中断重启 worker，初始化延迟是否影响训练稳定性。
   - Recommendation: 使用 persistent_workers=True 避免重复初始化。如果 DataLoader 重建（如 epoch 边界），初始化开销是可接受的（一次性）。

3. **最优 num_workers 数值**
   - What we know: CVAE forward 42ms/样本，gen_expr 7.5ms/样本。理论上 2 个 worker 可以将数据生成吞吐量翻倍到 ~30 samples/s。但由于 GPU 竞争，线性加速不太可能。
   - What's unclear: 实际并行加速比。同一 GPU 上多个 CUDA context 是否真正并行。
   - Recommendation: 起始 num_workers=2，实测后可调整到 4。

## Environment Availability

| 依赖 | 用途 | 可用 | 版本 | 备选 |
|------|------|------|------|------|
| PyTorch | DataLoader + CUDA | 可用 | 2.10.0+cu128 | -- |
| CUDA 12.8 | GPU 计算 | 可用 | 12.8 | -- |
| GPU x4 | 训练 + 数据生成 | 可用 | RTX 3090 24GB | -- |
| CPU 32核 | gen_expr 并行 | 可用 | -- | -- |
| RAM 125GB | 模型加载 | 可用 | -- | -- |
| Python multiprocessing.spawn | DataLoader worker | 可用 | 内置 | -- |

**Missing dependencies with no fallback:** 无

**Missing dependencies with fallback:** 无

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (项目未配置，需新建) |
| Config file | none |
| Quick run command | `pytest tests/test_latent_dataset.py -x -v` |
| Full suite command | `pytest tests/test_latent_dataset.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-01 | spawn DataLoader 生成数据不报错 | smoke | `pytest tests/test_latent_dataset.py::test_spawn_dataloader -x` | Wave 0 |
| PERF-02 | yield batch 形状正确 (B, 512) | unit | `pytest tests/test_latent_dataset.py::test_batch_shape -x` | Wave 0 |
| PERF-03 | 多 worker 吞吐量 > 单 worker | smoke | `pytest tests/test_latent_dataset.py::test_throughput -x` | Wave 0 |
| PERF-04 | DDP 场景下每 rank 独立工作 | manual | 需要 torchrun 手动验证 | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/test_latent_dataset.py -x -v`
- **Per wave merge:** `pytest tests/test_latent_dataset.py -v`
- **Phase gate:** Full suite green + 手动 DDP 验证

### Wave 0 Gaps
- [ ] `tests/test_latent_dataset.py` -- covers PERF-01, PERF-02, PERF-03
- [ ] pytest install: `pip install pytest` (if not present)

## Sources

### Primary (HIGH confidence)
- PyTorch 2.12 官方文档 - DataLoader, IterableDataset, multiprocessing
- PyTorch 官方 multiprocessing best practices - https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
- PyTorch GitHub Issue #13883 - Cannot re-initialize CUDA in forked subprocess
- 实测验证：fork + 已初始化 CUDA = 失败；spawn + 延迟初始化 = 成功

### Secondary (MEDIUM confidence)
- PyTorch Forums - IterableDataset with multiple workers 自动分片行为
- PyTorch Forums - DataLoader memory usage with spawn vs fork
- StackOverflow - batch_size=None 跳过 collate 的正确用法

### Tertiary (LOW confidence)
- 无

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 无新依赖，全部基于已有 PyTorch API
- Architecture: HIGH - 实测验证了 spawn + 延迟初始化 + yield batch 的可行性
- Pitfalls: HIGH - 实测验证了 fork+CUDA 不可行，spawn 方案已验证通过
- D-03 修正: HIGH - 3 种方案均实测，结论可靠

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (PyTorch API 稳定)
