# Phase 12: Decoder lm_head 微调多 GPU 并行训练 - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

将现有单卡 `finetune_lm_head.py` 改造为支持多 GPU DDP 并行训练。参照 `train_fm.py` 的 DDP 多卡模式，使用 torchrun 启动。

前置条件：Phase 10（lm_head 微调脚本）+ Phase 11（验证循环）已完成。

</domain>

<decisions>
## Implementation Decisions

### 并行方案
- **D-01:** 用 PyTorch DDP + torchrun，跟 `train_fm.py` 的多卡模式一致。每个 GPU 独立运行模型副本，梯度自动 all-reduce 同步。

### 验证循环适配
- **D-02:** 只在 rank 0 上跑验证循环和保存最优权重，跟 `train_fm.py` 一致。其他 rank 跳过验证。

### Claude's Discretion
- DDP wrapper 的具体实现细节（`DistributedDataParallel` 包装哪些模块）
- 数据并行的梯度累积适配
- 日志只在 rank 0 输出
- torchrun 启动命令的具体参数

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 被修改的文件
- `dit_train/finetune_lm_head.py` — Phase 10+11 完成的单卡微调+验证脚本，需要改造为 DDP

### DDP 参考实现
- `dit_train/train_fm.py` — 已有完整的 DDP 多卡训练实现，直接参照其模式

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `finetune_lm_head.py` 的完整训练+验证逻辑：只需要加 DDP wrapper 和 rank 判断
- `train_fm.py` 的 DDP 模式：`setup_ddp()`, `DistributedDataParallel` 包装, `rank/local_rank` 判断, `dist.barrier()` 等

### Established Patterns
- 单卡训练逻辑已完成（Phase 10+11）
- DDP 在 `train_fm.py` 中已验证可用

### Integration Points
- 只有 `lm_head` 需要被 DDP 包装（其他模块冻结不参与梯度计算）
- `optimizer` 需要用 DDP 包装后的参数

</code_context>

<specifics>
## Specific Ideas

- 参照 `train_fm.py` 的 DDP 模式，尽量保持一致的实现风格

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-decoder-lm-head-gpu*
*Context gathered: 2026-06-10*
