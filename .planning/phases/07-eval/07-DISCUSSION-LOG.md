# Phase 7: 评估对比实验 - Discussion Log

> **Audit trail only.**

**Date:** 2026-06-09
**Phase:** 7-评估对比实验
**Areas discussed:** 评估数据集范围, CMA-ES 基线来源, 评估脚本设计, DiT 推理配置

---

## 评估数据集范围

| Option | Description | Selected |
|--------|-------------|----------|
| 仅 Feynman | ~120 个物理公式数据集 | |
| Feynman + Strogatz | ~150 个数据集 | |
| 全部 PMLB 回归 | ~271 个数据集，覆盖 Feynman/Strogatz/黑盒 | ✓ |

**User's choice:** 全部 PMLB 回归数据集

---

## CMA-ES 基线来源

| Option | Description | Selected |
|--------|-------------|----------|
| 复用已有结果 | experiments/pmlb/GenSR_results/ 下 240 数据集完整 CSV | ✓ |
| 重新跑 CMA-ES | 保证环境一致但耗时几小时 | |

**User's choice:** 复用已有结果（推荐）

---

## 评估脚本设计

| Option | Description | Selected |
|--------|-------------|----------|
| 新建评估脚本 | dit_eval.py，复用 CSV 格式，结果放 GenSR_dit/ | ✓ |
| 改造现有脚本 | pmlb_batch_inference.py 加 method 参数 | |

**User's choice:** 新建评估脚本（推荐）

---

## DiT 推理配置

| Option | Description | Selected |
|--------|-------------|----------|
| 单次推理 num_steps=16 | 每个数据集跑一次，简单直接 | ✓ |
| 多次采样取最优 | 跑 K 次取最优 R^2 | |
| 多步数对比 | 8/16/32 步消融实验 | |

**User's choice:** 单次推理 num_steps=16（推荐）

---

## Claude's Discretion

- 评估脚本的具体参数和 CLI 接口
- 对比结果的具体输出格式
- 数据集加载和预处理的细节
- 断点续传是否需要

## Deferred Ideas

- 多步数消融实验
- 多次采样取最优
- 噪声鲁棒性实验
