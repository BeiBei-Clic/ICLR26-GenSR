# Phase 11: Decoder lm_head 微调验证循环 - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

给现有 `dit_train/finetune_lm_head.py` 训练脚本添加验证循环：每隔一定步数在验证集上评估 CE loss，跟踪 best val loss，自动保存最优权重。

前置条件：Phase 10 完成的 `finetune_lm_head.py` 训练脚本已可运行。

</domain>

<decisions>
## Implementation Decisions

### 验证数据来源
- **D-01:** 用 `gen_expr(train=False)` 在线生成验证数据，不做预生成缓存。跟训练数据生成方式一致（只是 train=False），简单直接。

### 验证频率
- **D-02:** 每 50 步验证一次，跟 `train_fm.py` 的验证频率一致。

### Claude's Discretion
- 验证时的 batch_size（可以用跟训练相同的 8）
- 验证 loss 的计算方式（逐样本 CE loss 平均，跟训练一致）
- 验证结果日志格式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 现有训练脚本（被修改的文件）
- `dit_train/finetune_lm_head.py` — Phase 10 完成的 lm_head 微调脚本，需要添加验证循环

### 参考验证模式
- `dit_train/train_fm.py` — FM 训练中的验证循环模式（best val loss 跟踪 + 最优权重保存）
- `Cola-DLM-main/scripts/cola_vae_finetune.py` — Cola 原型的 eval 验证模式

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `finetune_lm_head.py` 的训练数据生成逻辑（`env.gen_expr` + CVAE + DiT + FeatureFusion）：可直接复用，只改 `train=False`

### Established Patterns
- 逐样本 CE loss 计算然后平均（Phase 10 D-02）
- `lm_head_best.pt` checkpoint 保存格式（Phase 10）

### Integration Points
- 在 `finetune_lm_head.py` 的 `for step in range(...)` 循环中，每 50 步插入验证逻辑
- 验证时所有模块保持冻结 + eval 模式

</code_context>

<specifics>
## Specific Ideas

- 参照 `train_fm.py` 的验证循环模式：eval_every 参数 + best_val_loss 跟踪 + 保存最优权重

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-decoder-lm-head*
*Context gathered: 2026-06-10*
