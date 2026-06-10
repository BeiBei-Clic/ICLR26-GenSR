---
phase: 04-flow-matching
plan: 01
subsystem: training
tags: [flow-matching, dit, pytorch, ot-path, adamw, cosine-lr]

# Dependency graph
requires:
  - phase: 01-train-data-extract
    provides: LatentPairDataset online (prior_mu, post_mu) pair generation
  - phase: 03-dit
    provides: GenSRDiT model with AdaLN-Zero conditioned DiT
provides:
  - "Flow Matching training script (dit_train/train_fm.py)"
  - "OT-path interpolation + MSE loss training step"
  - "Full training loop with gradient accumulation, LR schedule, checkpoint save/load"
  - "7 unit tests covering FM-01 through FM-05"
affects: [05-euler-inference, 06-e2e-pipeline, 07-experiments]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "OT-path Flow Matching: z_t = (1-t)*prior_mu + t*post_mu, target = post_mu - prior_mu"
    - "Logit-normal timestep sampling (cola_sft.py:440-445)"
    - "Cosine LR schedule: warmup 5% + plateau + warmdown 30%"
    - "EMA loss smoothing (beta=0.95) for log display"

key-files:
  created:
    - dit_train/train_fm.py
    - tests/test_fm_train.py
  modified: []

key-decisions:
  - "Task 1 and Task 2 implemented together in single TDD cycle — core functions and training loop are in the same commit"
  - "train_loss obtained from flow_matching_step's second return value (loss.detach()) before gradient scaling"

patterns-established:
  - "FM training step: sample_timestep -> OT interpolation -> DiT forward -> MSE loss"
  - "Checkpoint format: torch.save dict with model_state_dict, optimizer_state_dict, step, val_loss"

requirements-completed: [FM-01, FM-02, FM-03, FM-04, FM-05]

# Metrics
duration: 3min
completed: 2026-06-09
---

# Phase 04 Plan 01: Flow Matching Training Script Summary

**OT-path Flow Matching 训练脚本：sample_timestep (logit-normal/uniform) + flow_matching_step (512 维 OT 插值 + MSE loss) + cosine LR schedule + gradient accumulation + EMA smoothing + checkpoint save/load，7 个测试全部通过**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-09T13:28:26Z
- **Completed:** 2026-06-09T13:31:56Z
- **Tasks:** 2
- **Files modified:** 2 (created)

## Accomplishments
- FM 训练核心函数：sample_timestep, flow_matching_step, get_lr_multiplier, save_checkpoint, load_checkpoint, evaluate
- 完整训练循环：CLI 参数 + LatentPairDataset 数据加载 + gradient accumulation + LR schedule + gradient clipping + EMA smoothing
- 7 个单元测试全部通过，覆盖 OT-path 插值、MSE loss、DiT 梯度隔离、checkpoint 保存加载

## Task Commits

Each task was committed atomically:

1. **Task 1+2: FM training core functions + training loop** - `5794be7` (test)

_Note: Task 1 (TDD) and Task 2 (training loop) were implemented together in a single TDD cycle since the functions are in the same file._

## Files Created/Modified
- `dit_train/train_fm.py` - Flow Matching 训练脚本：6 个核心函数 + CLI 参数 + 完整训练循环
- `tests/test_fm_train.py` - 7 个单元测试：timestep 采样、LR schedule、OT-path 插值、FM loss、梯度隔离、checkpoint、evaluate

## Decisions Made
- Task 1 和 Task 2 在同一 TDD 周期中实现 — train_fm.py 同时包含核心函数和训练循环
- train_loss 使用 flow_matching_step 的第二个返回值 (loss.detach())，在 gradient scaling 之前获取

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Flow Matching 训练脚本就绪，可直接运行 `python3 dit_train/train_fm.py --num-iterations 10` 验证
- Phase 05 (Euler 推理) 可使用 GenSRDiT 模型 + checkpoint 进行前向传输推理
- Phase 07 (Experiments) 可使用训练好的 DiT checkpoint 进行 PMLB 评估

## Self-Check: PASSED

- dit_train/train_fm.py: FOUND
- tests/test_fm_train.py: FOUND
- Commit 5794be7: FOUND

---
*Phase: 04-flow-matching*
*Completed: 2026-06-09*
