# Phase 10: Decoder 微调 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 10-dit-vae-cola-dlm-dit-vae
**Areas discussed:** Decoder 解冻范围, 训练数据来源, 训练策略

---

## Decoder 解冻范围

| Option | Description | Selected |
|--------|-------------|----------|
| 只解冻 Decoder 输出层（lm_head） | 跟 Cola-DLM cola_vae_finetune.py 一致，最小改动 | ✓ |
| 解冻整个 Decoder | 让 Decoder 全面适应新输入，风险更高 | |
| 全部解冻（Encoder+FeatureFusion+Decoder） | 需要 reference-encoder KL 约束 | |

**User's choice:** 只解冻 Decoder 输出层（lm_head），跟 Cola-DLM 原型代码一致
**Notes:** 用户明确要求跟 Cola-DLM-main 原型代码保持一致

---

## 训练数据来源

| Option | Description | Selected |
|--------|-------------|----------|
| 用 CVAE 编码的 post_mu | 跟 Cola compute_loss 一致，但 Decoder 学到的跟推理时不匹配 | |
| 用 DiT 传输后的 z_opt | Decoder 直接适应推理时的实际输入分布 | ✓ |

**User's choice:** 用 DiT 传输后的 z_opt
**Notes:** 训练数据走完整推理路径：CVAE → DiT Euler → FeatureFusion → Decoder → CE loss

---

## 训练策略

| Option | Description | Selected |
|--------|-------------|----------|
| 照搬 Cola 参数 | AdamW lr=1e-4, batch_size=8, 500 iterations | ✓ |
| 自定义参数 | 用户指定超参数 | |

**User's choice:** 照搬 Cola 参数
**Notes:** 只微调一个 Linear 层，参数量很少，Cola 默认参数足够

---

## Claude's Discretion

- 训练脚本组织方式
- Checkpoint 保存策略
- 是否需要 eval 验证步骤

## Deferred Ideas

- 联合训练 DiT + VAE（Stage 2 全套）——留作未来阶段
- 解冻更多 Decoder 层——如果输出层微调效果不够再考虑
