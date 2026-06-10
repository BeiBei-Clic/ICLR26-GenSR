# Phase 9: 优化 LatentPairDataset 数据生成性能 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 09-latentpairdataset
**Areas discussed:** 数据生成策略, 并行架构, yield 模式优化

---

## 数据生成策略

| Option | Description | Selected |
|--------|-------------|----------|
| 预生成缓存 | 训练前一次性生成大量 latent pairs 存磁盘/内存，训练时直接读取 | |
| 在线多进程生成 | 保持 IterableDataset 在线生成，用 DataLoader num_workers>0 并行 | ✓ |
| 混合方案 | 预生成 buffer 池，训练时异步补充 | |

**User's choice:** 在线多进程生成
**Notes:** 用户倾向保持数据多样性，不使用静态缓存

---

## 并行架构

| Option | Description | Selected |
|--------|-------------|----------|
| 每 worker 独立 CVAE | 每个 worker 独立加载 CVAE 到 GPU，简单但显存占用高 | |
| CPU 推理 CVAE | worker 在 CPU 上跑 CVAE，省显存但速度慢 | |
| 共享 CVAE GPU 权重 | fork 共享 CUDA context，worker 共用同一份 GPU 权重 | ✓ |

**User's choice:** 共享 CVAE GPU 权重
**Notes:** 复杂度较高但显存效率最好

### Worker 数量

| Option | Description | Selected |
|--------|-------------|----------|
| 1 worker | 每个 GPU 一个 worker | |
| 2 workers | 每个 GPU 两个 worker | |
| Claude 决定 | 根据实测结果选择 | ✓ |

**User's choice:** Claude 决定
**Notes:** 由规划和执行阶段根据实测决定

---

## Yield 模式优化

| Option | Description | Selected |
|--------|-------------|----------|
| 直接 yield 整 batch | Dataset 直接 yield 完整 batch，跳过 DataLoader collate | ✓ |
| 保持逐样本 yield | 当前模式，由 DataLoader collate 打包 | |
| Claude 决定 | 根据性能测试选择 | |

**User's choice:** 直接 yield 整 batch

---

## Claude's Discretion

- DataLoader worker 数量
- fork 共享 CUDA context 的具体实现
- pin_memory、prefetch_factor 等参数
- 训练循环调用方式调整

## Deferred Ideas

None — 讨论保持在 phase 范围内
