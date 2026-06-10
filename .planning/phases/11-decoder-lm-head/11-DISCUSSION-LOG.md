# Phase 11: Decoder lm_head 微调验证循环 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-06-10
**Phase:** 11-decoder-lm-head
**Areas discussed:** 验证数据来源, 验证频率

---

## 验证数据来源

| Option | Description | Selected |
|--------|-------------|----------|
| 在线生成 (gen_expr train=False) | 简单直接，跟训练数据方式一致 | ✓ |
| 预生成固定验证集 | 评估更稳定但复杂度高 | |

**User's choice:** 在线生成 (gen_expr train=False)

---

## 验证频率

| Option | Description | Selected |
|--------|-------------|----------|
| 每 50 步 | 跟 train_fm.py 一致 | ✓ |
| 自定义 | 用户指定步数 | |

**User's choice:** 每 50 步

---

## Claude's Discretion

- 验证时 batch_size
- 验证 loss 计算方式
- 验证结果日志格式
