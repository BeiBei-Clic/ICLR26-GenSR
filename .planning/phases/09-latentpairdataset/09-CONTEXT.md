# Phase 9: 优化 LatentPairDataset 数据生成性能 - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

优化 LatentPairDataset 数据生成管线，消除 GPU 训练时的数据供给瓶颈。核心问题：增大 batch_size 后 GPU 利用率反而下降，说明数据生成速度跟不上。不改变训练逻辑和模型架构，只优化数据供给效率。

</domain>

<decisions>
## Implementation Decisions

### 数据生成策略
- **D-01:** 保持在线生成模式，不使用预生成缓存。使用 DataLoader 多进程 (`num_workers>0`) 并行生成数据
- **D-02:** 不使用预生成缓存方案的原因：CVAE 冻结后每次生成的数据统计上等价，但用户倾向保持数据多样性，在线生成更灵活

### 并行架构
- **D-03:** 使用共享 CVAE GPU 权重方案：主进程加载 CVAE 到 GPU，worker 进程通过 fork 共享 CUDA context 访问同一份 GPU 权重
- **D-04:** Worker 数量由 Claude 根据实测结果决定（需考虑 DDP 下每卡 worker 数与显存占用的平衡）

### Yield 模式
- **D-05:** LatentPairDataset.__iter__ 直接 yield 完整 batch 的 (prior_mu, post_mu) 张量对，跳过 DataLoader 的逐样本 collate 开销
- **D-06:** batch_size 逻辑从 DataLoader 移到 Dataset 内部

### Claude's Discretion
- DataLoader worker 数量的具体值
- fork 共享 CUDA context 的具体实现方式（`torch.multiprocessing.set_start_method('fork')` + `mp.set_sharing_strategy('file_system')` 等）
- 是否需要 pin_memory、prefetch_factor 等 DataLoader 优化参数
- 训练循环中 `next(data_iter)` 的调用方式是否需要配合调整

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 数据生成管线
- `dit_train/data/latent_dataset.py` — LatentPairDataset 当前实现，__iter__ 数据生成流程
- `dit_train/train_fm.py` — 训练循环中 data_iter 的使用方式（第 267-289 行）
- `symbolicregression/envs/generators.py` — gen_expr 方法，数据生成的 CPU 密集部分
- `symbolicregression/model/cvae.py` — CVAE forward mode="train"，GPU 密集部分
- `symbolicregression/model/embedders.py` — embedder_f 和 embedder_e 编码器

### DataLoader 相关
- `dit_train/data/latent_dataset.py` §90-97 — create_latent_dataloader 当前 num_workers=0, pin_memory=False

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LatentPairDataset.__init__` 中 CVAE 加载和冻结逻辑可复用
- `create_latent_dataloader` 函数可作为优化入口

### Established Patterns
- CVAE 冻结模式：`param.requires_grad = False` + `model.eval()` + `torch.no_grad()` 上下文
- IterableDataset 无限迭代器模式

### Integration Points
- `train_fm.py:271` — `data_iter = iter(loader)` 创建迭代器
- `train_fm.py:285-290` — 训练循环中 `next(data_iter)` + `.to(device)` 调用
- DDP 场景：每个 rank 独立创建自己的 DataLoader

</code_context>

<specifics>
## Specific Ideas

- 用户观察到 device_batch_size 从 4 改到 20 后 GPU 利用率反而下降，确认数据生成是瓶颈
- 主要瓶颈在 `env.gen_expr()`（CPU 密集的随机表达式树构建 + 数值计算）和 CVAE forward（GPU 密集的 Transformer 编码）
- 4 卡 DDP 训练时每张卡独立创建 DataLoader，优化需兼容 DDP

</specifics>

<deferred>
## Deferred Ideas

None — 讨论保持在 phase 范围内

</deferred>

---

*Phase: 09-latentpairdataset*
*Context gathered: 2026-06-10*
