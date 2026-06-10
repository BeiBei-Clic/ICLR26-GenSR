# Phase 4: Flow Matching 训练 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 4-Flow Matching 训练
**Areas discussed:** 训练脚本形式, 时间步采样策略, 训练超参数默认值, 日志和监控方式

---

## 训练脚本形式

| Option | Description | Selected |
|--------|-------------|----------|
| 原生 PyTorch | 参照 cola_sft.py 风格，手写 for loop + optimizer.step() | ✓ |
| PyTorch Lightning | LightningModule 包装，用 GenSR 已有的 Lightning Trainer | |

**User's choice:** 原生 PyTorch（推荐）

---

## 时间步采样策略

| Option | Description | Selected |
|--------|-------------|----------|
| Logit-normal | Cola-DLM 默认，中间时间步更密集，loc=0.0, scale=1.0 | ✓ |
| Uniform | 最简单，各时间步等概率 | |
| 两种都支持 | 命令行参数选择 | |

**User's choice:** Logit-normal（推荐）

---

## 训练超参数默认值

| Option | Description | Selected |
|--------|-------------|----------|
| 照搬 Cola-DLM | AdamW lr=1e-4, wd=0.01, warmup 5% + warmdown 30%, batch=4, grad_accum=8 | ✓ |
| GenSR 专用 | 调整默认值适应 512 维单向量 | |
| 命令行可配 + Cola 默认 | 全部参数可配，默认值照搬 Cola | |

**User's choice:** 照搬 Cola-DLM 默认值（推荐）

---

## 日志和监控方式

| Option | Description | Selected |
|--------|-------------|----------|
| Wandb + print | 与 GenSR/Cass 已有集成一致 | |
| 仅 print | 先简单 print，后续加 wandb | ✓ |

**User's choice:** 仅 print，后续加 wandb

---

## Claude's Discretion

- 训练脚本的具体文件位置和命名
- Checkpoint 保存格式
- Eval 频率和方式
- 命令行参数的组织方式
- 数据加载与训练循环的具体实现细节

## Deferred Ideas

- Wandb 日志集成（后续添加）
- CFG (Classifier-Free Guidance) 训练（v2 feature）
- 多 GPU DDP 支持
