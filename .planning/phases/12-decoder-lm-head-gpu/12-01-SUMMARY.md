---
phase: 12-decoder-lm-head-gpu
plan: 01
subsystem: training
tags: [ddp, multi-gpu, pytorch, distributed-training, nccl]

# Dependency graph
requires:
  - phase: 11-decoder-lm-head
    provides: "finetune_lm_head.py 单卡训练脚本 + 验证循环 + best val loss 保存"
provides:
  - "finetune_lm_head.py 支持 torchrun DDP 多卡训练"
  - "lm_head DDP 包装模式（只包装 lm_head，不包装整个 decoder）"
  - "is_master 日志控制 + rank 0 验证 + dist.barrier 同步"
affects: []

# Tech tracking
tech-stack:
  added: [torch.distributed, DistributedDataParallel, nccl]
  patterns: [ddp-wrapper-on-submodule, is-master-log-control, rank0-eval-with-barrier]

key-files:
  created: []
  modified: ["dit_train/finetune_lm_head.py"]

key-decisions:
  - "只包装 lm_head 为 DDP，不包装整个 decoder（92.9% 参数冻结，包装整个 decoder 需 find_unused_parameters=True）"
  - "保存权重用 lm_head_raw（未包装原始引用），避免 DDP module. 前缀问题"
  - "验证循环仅在 rank 0 执行 + dist.barrier() 同步其他 rank"
  - "CUDA 标志判断改用 startswith(\"cuda\") 兼容 cuda:N 格式"

patterns-established:
  - "DDP sub-module wrapping: 只包装可训练子模块，不包装整个冻结模型"
  - "lm_head_raw 引用模式: 保存前保留未包装引用，避免 DDP state_dict 前缀问题"

requirements-completed: [DDP-01, DDP-02]

# Metrics
duration: 3min
completed: 2026-06-10
---

# Phase 12 Plan 01: Decoder lm_head DDP Multi-GPU Summary

**finetune_lm_head.py DDP 多卡改造：lm_head DDP 包装 + rank 0 验证 + is_master 日志控制 + torchrun 兼容**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-10T13:56:05Z
- **Completed:** 2026-06-10T13:59:08Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- finetune_lm_head.py 完整支持 torchrun DDP 多卡训练，单卡向后兼容
- 双卡 DDP smoke test 通过：两个 rank 独立生成数据、DDP all-reduce 梯度同步、无 CUDA/NCCL 错误、无死锁
- 只有 rank 0 输出日志和执行验证循环，其他 rank 在 dist.barrier() 处等待

## Task Commits

Each task was committed atomically:

1. **Task 1: 改造 finetune_lm_head.py 为 DDP 多卡训练** - `ff8b24c` (feat)
2. **Task 2: 单卡兼容性试运行 + 多卡 DDP smoke test** - `b358eba` (fix)

## Files Created/Modified
- `dit_train/finetune_lm_head.py` - DDP 多卡训练支持（7 个改造点：DDP 导入、初始化、lm_head 包装、optimizer、is_master 日志、rank 0 验证+barrier、清理）

## Decisions Made
- 只包装 lm_head 为 DDP，不包装整个 decoder：decoder 92.9% 参数冻结，DDP 包装整个 decoder 需要 find_unused_parameters=True，性能差且语义不匹配
- 保存权重用 lm_head_raw：DDP wrapper 的 state_dict() 会在 key 前加 "module." 前缀，导致加载不兼容
- 验证循环仅在 rank 0 执行 + dist.barrier() 同步：避免重复验证和竞态条件
- 每个 rank 独立生成数据（env.gen_expr 天然随机），无需 DistributedSampler

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] CUDA 标志判断 device == "cuda" 不兼容 cuda:N 格式**
- **Found during:** Task 2 (单卡兼容性测试)
- **Issue:** DDP 初始化后 device 变为 "cuda:0"，原先 device == "cuda" 判断失败，导致 CUDA 标志为 False，embedder 参数未搬到 GPU，触发 RuntimeError: Expected all tensors on same device
- **Fix:** 改为 device.startswith("cuda")，兼容 "cuda" 和 "cuda:N" 两种格式
- **Files modified:** dit_train/finetune_lm_head.py
- **Verification:** 单卡 5 步和双卡 2 步均正常运行
- **Committed in:** b358eba

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** 必要修复，DDP 初始化引入的 device 格式变更导致的兼容性问题。无 scope creep。

## Issues Encountered
None beyond the auto-fixed bug above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- finetune_lm_head.py 已支持单卡和 DDP 多卡训练，可直接用于生产训练
- Phase 12 全部完成（仅 1 个 plan）

## Self-Check: PASSED
- FOUND: dit_train/finetune_lm_head.py
- FOUND: ff8b24c (Task 1 commit)
- FOUND: b358eba (Task 2 commit)
- FOUND: 12-01-SUMMARY.md

---
*Phase: 12-decoder-lm-head-gpu*
*Completed: 2026-06-10*
