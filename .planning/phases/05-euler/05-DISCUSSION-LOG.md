# Phase 5: Euler 积分推理 - Discussion Log

> **Audit trail only.**

**Date:** 2026-06-09
**Phase:** 5-Euler 积分推理
**Areas discussed:** 推理步数默认值, 推理接口形式, Batch 推理支持

---

## 推理步数默认值

| Option | Description | Selected |
|--------|-------------|----------|
| 16 步 | Cola-DLM 默认值，OT-path 上足够精细 | ✓ |
| 10 步 | 更快但精度可能下降 | |
| 20 步 | 更精细但更慢 | |

**User's choice:** 16 步（推荐）

---

## 推理接口形式

| Option | Description | Selected |
|--------|-------------|----------|
| 函数 + CLI | euler_inference 函数 + CLI 脚本 | |
| 仅函数 | euler_inference 函数供 Phase 6 调用 | ✓ |
| 仅 CLI | CLI 脚本 | |

**User's choice:** 仅函数

---

## Batch 推理支持

| Option | Description | Selected |
|--------|-------------|----------|
| 直接支持 batch | GenSRDiT 本身支持 batch 输入 | ✓ |
| 仅单个推理 | 单个 prior_mu → z_opt | |

**User's choice:** 直接支持 batch（推荐）

---

## Claude's Discretion

- euler_inference 函数的具体参数和返回值设计
- 文件位置和命名
- 测试的组织方式

## Deferred Ideas

- CLI 推理脚本
- CFG 推理（v2 feature）
- 自适应步长
