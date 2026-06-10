# Phase 3: DiT 模型定义 - Discussion Log

> **Audit trail only.**

**Date:** 2026-06-09
**Phase:** 3-DiT 模型定义
**Areas discussed:** Patchification, 条件注入, 模型规模, FFN 类型, 归一化, 默认配置

---

## Patchification

| Option | Description | Selected |
|--------|-------------|----------|
| 16 patches × 32 dim | 足够的 token 数，投影到隐藏维度 | |
| 8 patches × 64 dim | 更粗粒度 | |
| 参数配置 | 通过命令行参数配置 | ✓ |

**User's choice:** 参数配置

---

## 条件注入

| Option | Description | Selected |
|--------|-------------|----------|
| Cross-Attention | prior_mu 作为 K/V，z_t patches 做 Q | ✓ |
| 拼接为额外 token | prior_mu 加入序列 | |
| AdaLN 调制 | 与 timestep 共享调制通道 | |

**User's choice:** Cross-Attention

---

## 模型规模

| Option | Description | Selected |
|--------|-------------|----------|
| 小型（4/256/4） | 轻量级先跑通 | |
| 中型（6/512/8） | 中等规模 | |
| 参数配置（DIT-05） | 全部可调 | ✓ |

**User's choice:** 参数配置，默认中型（6/512/8）

---

## FFN 类型

| Option | Description | Selected |
|--------|-------------|----------|
| SwiGLU（按需求文档） | REQUIREMENTS DIT-01 | |
| GELU（按 Cola 代码） | Cola 实际实现 | ✓ |

**User's choice:** GELU（按 Cola 实际代码）

---

## 归一化

| Option | Description | Selected |
|--------|-------------|----------|
| RMSNorm（按需求文档） | REQUIREMENTS DIT-01 | |
| LayerNorm（按 Cola 代码） | Cola 实际实现 | ✓ |

**User's choice:** LayerNorm（按 Cola 实际代码）

---

## 默认配置

| Option | Description | Selected |
|--------|-------------|----------|
| 小型默认（4/256/4） | 先跑通 | |
| 中型默认（6/512/8） | 更强表达力 | ✓ |

**User's choice:** 中型默认（6 层、512 隐藏、8 头、16×32 patches）

## Deferred Ideas

None
