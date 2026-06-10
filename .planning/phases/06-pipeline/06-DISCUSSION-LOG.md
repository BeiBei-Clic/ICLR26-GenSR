# Phase 6: 端到端推理管线 - Discussion Log

> **Audit trail only.**

**Date:** 2026-06-09
**Phase:** 6-端到端推理管线
**Areas discussed:** logvar 处理策略, 推理接口设计, 模型加载与初始化

---

## logvar 处理策略

| Option | Description | Selected |
|--------|-------------|----------|
| 用 prior_logvar | CVAE 编码时产生的，保持一致 | ✓ |
| 用 zeros | sigma=1 的标准正态 | |
| 直接传入不采样 | 跳过采样只做投影 | |

**User's choice:** 用 prior_logvar（推荐）

---

## 推理接口设计

| Option | Description | Selected |
|--------|-------------|----------|
| 简单端到端函数 | 输入 (X,Y) → 输出 (expression, R^2) | ✓ |
| 分步函数 | encode → dit → decode → bfgs | |

**User's choice:** 简单端到端函数（推荐）

---

## 模型加载与初始化

| Option | Description | Selected |
|--------|-------------|----------|
| 接受模型对象 | 调用方负责加载和初始化 | ✓ |
| 内部加载模型 | 函数内部加载所有模型 | |

**User's choice:** 接受模型对象（推荐）

---

## Claude's Discretion

- 推理函数的具体参数和返回值细节
- 文件位置和命名
- BFGS 调用的具体方式
- 测试的组织方式

## Deferred Ideas

- Beam search 支持
- 多候选表达式排序
- wandb 日志
