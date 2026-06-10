# Phase 2: 训练数据验证 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 2-训练数据验证
**Areas discussed:** 验证方式, 验证内容, 样本量, 通过/不通过标准

---

## 验证方式

| Option | Description | Selected |
|--------|-------------|----------|
| 扩展 verify_dataset.py | 在 Phase 1 已有脚本上增加功能 | ✓ |
| 独立脚本 | 创建新的验证脚本 | |

**User's choice:** 扩展 verify_dataset.py

---

## 验证内容

| Option | Description | Selected |
|--------|-------------|----------|
| 分布可视化图 | prior_mu/post_mu 各维度直方图/KDE，矢量图保存 | ✓ |
| KL 散度统计 | 检测 posterior collapse | ✓ |
| 基础统计检查 | mean, std, NaN/Inf 检测 | ✓ |
| 通过/不通过判断 | 自动 PASS/FAIL | ✓ |

**User's choice:** 全部四项

---

## 样本量

| Option | Description | Selected |
|--------|-------------|----------|
| 1000 个样本 | 足够做分布可视化和 KL 估计，1-2 分钟内完成 | ✓ |
| 5000 个样本 | 更精确但耗时更长 | |
| 参数配置 | 通过命令行参数指定 | |

**User's choice:** 1000 个样本

---

## 通过/不通过标准

| Option | Description | Selected |
|--------|-------------|----------|
| 自动判定 | KL > 0.01, diff norm > 0.1, 无 NaN/Inf | ✓ |
| 仅输出统计 | 用户自己判断 | |
| 参数配置阈值 | 命令行参数指定阈值 | |

**User's choice:** 自动判定

---

## Claude's Discretion

- 具体可视化图的布局和数量
- KL 散度的计算方式
- 矢量图保存路径
- 脚本的输出格式

## Deferred Ideas

None
