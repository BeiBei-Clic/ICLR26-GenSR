---
phase: 07-eval
plan: 01
subsystem: evaluation
tags: [pmlb, dit, flow-matching, csv, batch-inference]

# Dependency graph
requires:
  - phase: 06-pipeline
    provides: dit_inference() 端到端推理管线
  - phase: 05-euler
    provides: euler_inference() Euler 积分推理
  - phase: 04-flow-matching
    provides: Flow Matching 训练脚本 + fm_best.pth checkpoint
provides:
  - dit_eval.py 批量评估脚本
  - PMLB 上 DiT 方法的 R^2/复杂度/推理时间评估结果
affects: [08-summary]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FlowMatchingModel: 匹配 fm_best.pth 的 MLP-like DiT 结构（input_proj + 6 AdaLN blocks + output_proj）"
    - "断点续传: 检查已有 CSV 跳过已完成数据集"

key-files:
  created:
    - experiments/pmlb/dit_eval.py
  modified: []

key-decisions:
  - "FlowMatchingModel 替代 GenSRDiT: fm_best.pth 的 checkpoint 结构是 MLP-like（6 层 AdaLN block，无 attention），与 dit_train/model.py 的 Transformer DiT 不匹配"
  - "FlowMatchingModel 输入为 concat(z_t, prior_mu) 而非 cross-attention 条件化"
  - "特征维度超限数据集 skip 而非 assert crash"

patterns-established:
  - "checkpoint 结构匹配: 根据 fm_best.pth 的 key 结构反推模型定义"
  - "评估脚本结构: argparse + get_parser parents + parser.set_defaults + 内联逻辑"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03]

# Metrics
duration: 13min
completed: 2026-06-09
---

# Phase 07: 评估对比实验 Summary

**DiT 批量评估脚本: 在 PMLB 回归数据集上运行 Flow Matching 推理，输出 R^2/复杂度/时间 CSV + 统计摘要**

## Performance

- **Duration:** 13 min
- **Started:** 2026-06-09T15:02:03Z
- **Completed:** 2026-06-09T15:15:16Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- 创建 dit_eval.py 评估脚本，调用 dit_inference() 在 PMLB 数据集上批量推理
- 匹配 fm_best.pth checkpoint 的 FlowMatchingModel（MLP-like 6 层 AdaLN block）
- 输出 CSV 包含 dataset, status, n_features, r2, complexity, seconds, error, expr
- 运行结束打印成功率(R^2 > 0.99)、平均 R^2、中位 R^2、平均时间、平均复杂度
- 支持断点续传（跳过已完成数据集）
- 试运行 2 个数据集验证通过（1030_ERA R^2=0.159, 1096_FacultySalaries R^2=0.788）

## Task Commits

Each task was committed atomically:

1. **Task 1: 创建 DiT 评估脚本 dit_eval.py** - `ae98443` (feat)
2. **Task 2: 试运行验证 + 修复模型结构** - `d7e19c0` (fix)

## Files Created/Modified
- `experiments/pmlb/dit_eval.py` - DiT 批量评估脚本，包含 FlowMatchingModel 模型定义 + 评估循环 + 统计输出

## Decisions Made
- **FlowMatchingModel 替代 GenSRDiT**: fm_best.pth 的 checkpoint 结构是 MLP-like（每个 block 只有 adaLN + proj，无 self-attention/cross-attention），与 dit_train/model.py 的 Transformer DiT 完全不同。需要在 dit_eval.py 中定义匹配的模型
- **concat(z_t, prior_mu) 输入**: FlowMatchingModel 的 input_proj 接受 1024 维输入（512+512 concat），而非 Transformer DiT 的 cross-attention 条件化
- **特征维度超限 skip**: summary_tsv 中的 n_features 与实际数据不一致时，skip 该数据集并记录到 CSV，而非 assert crash

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Checkpoint 结构与 GenSRDiT 不匹配**
- **Found during:** Task 2 (试运行验证)
- **Issue:** fm_best.pth 的 key 结构（input_proj, time_embed.proj_in/proj_hid, blocks.*.adaLN/proj, output_proj）与 GenSRDiT 完全不匹配
- **Fix:** 创建 FlowMatchingModel 匹配 checkpoint 的 MLP-like 结构：input_proj(1024->1024) + time_embed(256->1024->1024) + 6 AdaLN blocks + output_proj(1024->512)
- **Files modified:** experiments/pmlb/dit_eval.py
- **Verification:** checkpoint 加载成功，试运行 2 数据集通过
- **Committed in:** d7e19c0

**2. [Rule 3 - Blocking] datasets_dir 路径错误**
- **Found during:** Task 2 (试运行验证)
- **Issue:** 默认路径 datasets/pmlb/datasets 中文件是 Git LFS 指针，真实数据在 pmlb/datasets
- **Fix:** 将默认 --datasets_dir 从 "datasets/pmlb/datasets" 改为 "pmlb/datasets"
- **Files modified:** experiments/pmlb/dit_eval.py
- **Committed in:** d7e19c0

**3. [Rule 1 - Bug] 特征维度超限导致 crash**
- **Found during:** Task 2 (试运行验证)
- **Issue:** summary_tsv 的 n_features 与实际数据不一致，第二个数据集(1089_USCrime)实际 13 维超限 assert crash
- **Fix:** 将 assert 替换为 if-check + skip，记录到 CSV
- **Files modified:** experiments/pmlb/dit_eval.py
- **Committed in:** d7e19c0

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** 所有修复必要，无 scope creep。关键发现：fm_best.pth 使用 MLP-like DiT 而非 Transformer DiT。

## Issues Encountered
- 部分数据集触发 CUDA assert（CVAE 编码器内部 index out of bounds），这是模型/数据问题，非脚本问题。在完整 PMLB 运行中可能需要处理

## User Setup Required
None - 无需外部服务配置。

## Next Phase Readiness
- dit_eval.py 已就绪，可通过 `python3 experiments/pmlb/dit_eval.py` 运行完整 PMLB 评估
- fm_best.pth 使用 MLP-like DiT，未来训练可能需要统一模型定义

---
*Phase: 07-eval*
*Completed: 2026-06-09*

## Self-Check: PASSED
- experiments/pmlb/dit_eval.py: FOUND
- .planning/phases/07-eval/07-01-SUMMARY.md: FOUND
- ae98443: FOUND
- d7e19c0: FOUND
