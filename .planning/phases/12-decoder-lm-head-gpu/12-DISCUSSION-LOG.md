# Phase 12: Decoder lm_head 微调多 GPU 并行训练 - Discussion Log

> **Audit trail only.**

**Date:** 2026-06-10
**Phase:** 12-decoder-lm-head-gpu
**Areas discussed:** 并行方案选择, 验证循环适配

---

## 并行方案选择

| Option | Description | Selected |
|--------|-------------|----------|
| DDP + torchrun | 跟 train_fm.py 一致，项目已有成熟模式 | ✓ |
| DataParallel | 简单但效率低 | |

**User's choice:** DDP + torchrun

---

## 验证循环适配

| Option | Description | Selected |
|--------|-------------|----------|
| 只在 rank 0 验证 | 跟 train_fm.py 一致，简单直接 | ✓ |
| 所有 rank 验证 | all-reduce val loss | |

**User's choice:** 只在 rank 0 验证

---

## Claude's Discretion

- DDP wrapper 具体实现
- 梯度累积适配
- 日志 rank 判断
- torchrun 启动参数
