# Phase 1: 训练数据生成器 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 1-训练数据生成器
**Areas discussed:** 数据生成方式, 样本量配置, 数据格式, 检查点选择, 随机种子, Phase 定位修正

---

## 数据生成方式

| Option | Description | Selected |
|--------|-------------|----------|
| 预生成固定数据集 | 先生成 (X, Y, GT表达式) 保存到文件，再批量提取 | |
| 在线生成+提取 | 像 CVAE 训练时一样在线调用 FunctionEnvironment，生成后立刻提取 | ✓ |
| 两种都支持 | 脚本支持两种模式，通过参数切换 | |

**User's choice:** 在线生成+提取
**Notes:** 用户认为在线生成模式下不需要保存中间数据（X/Y/表达式），因为数据是随机的。

---

## 样本量配置

| Option | Description | Selected |
|--------|-------------|----------|
| 50,000 对 | 约等于训练时1个epoch的数据量 | |
| 100,000 对 | 更多数据，训练更充分 | |
| 参数配置 | 通过参数指定，灵活配置 | ✓ |

**User's choice:** 参数配置
**Notes:** 灵活性优先，用户自己决定每次生成多少。

---

## 数据格式

| Option | Description | Selected |
|--------|-------------|----------|
| 只保存 (prior_mu, post_mu) | .pt 文件包含两个 (N, 512) 张量 | |
| 加上 logvar | 同时保存 prior_logvar 和 post_logvar | |
| 加上表达式和 X/Y | 完整数据，便于调试但文件很大 | |

**User's choice:** 在线生成不保存文件
**Notes:** 用户纠正：既然是在线生成，就不需要保存到 .pt 文件。数据生成器直接作为 PyTorch Dataset 供 DiT 训练使用。

---

## 检查点选择

| Option | Description | Selected |
|--------|-------------|----------|
| checkpoint.pth (671MB) | 主要预训练模型 | ✓ |
| fm_best.pth (932MB) | 文件名像是 flow matching 相关 | |
| 参数配置 | 命令行指定，默认 checkpoint.pth | |

**User's choice:** checkpoint.pth
**Notes:** CVAE 已训练好，加载后完全冻结不动。

---

## 随机种子

| Option | Description | Selected |
|--------|-------------|----------|
| 固定种子 | --seed 参数，保证每次提取结果相同 | |
| 不固定种子 | 每次运行随机生成不同数据 | ✓ |

**User's choice:** 不固定种子

---

## Phase 定位修正（第二轮讨论）

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 1 做数据生成器 | 交付 PyTorch Dataset/DataLoader 组件，DiT 训练时直接 import 使用 | ✓ |
| Phase 1 合并到 Phase 4 | 取消 Phase 1，数据生成逻辑在 DiT 训练中实现 | |
| Phase 1 还是保存但可选 | 保留 .pt 保存功能，但 DiT 训练主要用在线生成 | |

**User's choice:** Phase 1 做数据生成器
**Notes:** 用户明确：在线生成不需要保存文件步骤，CVAE checkpoint 是已训练好的冻结模型。Phase 1 交付的是一个可复用的数据生成组件。

---

## Claude's Discretion

- Dataset 类的具体实现方式（IterableDataset vs Map-style Dataset）
- 批次处理的具体逻辑（collate_fn 等）
- train/val split
- 设备管理（CVAE 在哪个设备上运行）
- 组件文件位置和命名
- 是否需要独立的验证脚本（供 Phase 2 使用）

## Deferred Ideas

None
