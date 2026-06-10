---
phase: 09-latentpairdataset
plan: 01
subsystem: data
tags: [pytorch, dataloader, spawn, multiprocessing, gpu, iterable-dataset]

# Dependency graph
requires:
  - phase: 01
    provides: "LatentPairDataset 原始实现和 CVAE 冻结模式"
provides:
  - "spawn 多进程 LatentPairDataset (延迟初始化 + yield batch)"
  - "create_latent_dataloader 支持 num_workers 和 spawn 模式"
  - "训练循环适配 spawn DataLoader"
affects: [训练管线]

# Tech tracking
tech-stack:
  added: []
  patterns: ["spawn DataLoader + _lazy_init 延迟初始化模式", "batch_size=None 跳过 collate 模式"]

key-files:
  created: []
  modified:
    - "dit_train/data/latent_dataset.py"
    - "dit_train/train_fm.py"
    - "tests/test_latent_dataset.py"

key-decisions:
  - "spawn 模式替代 fork（fork 在主进程已初始化 CUDA 时不可用）"
  - "num_workers=2 默认值（RTX 3090 上 2 worker 额外 1.3GB 显存，安全）"
  - "yield 完整 batch 跳过 DataLoader collate 开销"

patterns-established:
  - "Worker 延迟初始化: __init__ 只保存配置，_lazy_init 在首次 __iter__ 时构建模型"
  - "batch_size=None DataLoader: Dataset 内部组装 batch，DataLoader 透传不做 collate"

requirements-completed: [PERF-01, PERF-02, PERF-03]

# Metrics
duration: 3min
completed: 2026-06-10
---

# Phase 09 Plan 01: Spawn 多进程 LatentPairDataset Summary

**spawn DataLoader 多进程并行生成数据，_lazy_init 延迟初始化 CVAE 到 worker GPU，yield 完整 batch 跳过 collate**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-10T04:44:46Z
- **Completed:** 2026-06-10T04:48:35Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- LatentPairDataset 支持 spawn 多进程并行数据生成，每个 worker 独立持有冻结 CVAE（~641MB/worker）
- Dataset yield 完整 batch (B, 512)，DataLoader batch_size=None 跳过 collate 开销
- 训练循环移除 .to(device)，spawn worker 生成的数据已在 GPU 上
- 7/7 测试全部通过（包括 spawn 多 worker 测试和吞吐量对比测试）

## Task Commits

Each task was committed atomically:

1. **Task 1: 重写 LatentPairDataset 支持 spawn 多进程 + yield batch** - `bf58b43` (feat)
2. **Task 2: 适配训练循环 + 更新测试** - `c65b94e` (feat)

## Files Created/Modified
- `dit_train/data/latent_dataset.py` - 重写: __init__ 只保存配置，_lazy_init 延迟初始化，__iter__ yield 完整 batch，create_latent_dataloader 支持 spawn
- `dit_train/train_fm.py` - DataLoader 创建传入 num_workers=2，训练循环移除 .to(device)
- `tests/test_latent_dataset.py` - 适配 yield batch，新增 test_spawn_dataloader 和 test_throughput_comparison

## Decisions Made
- spawn 模式替代 fork: 实测验证 fork 在主进程已初始化 CUDA 后报 "Cannot re-initialize CUDA in forked subprocess"，spawn 是唯一可行方案
- num_workers=2 默认值: RTX 3090 上每个 worker 额外占用约 641MB 显存（冻结 CVAE），2 worker 总额外 1.3GB 安全
- persistent_workers=True: 避免每个 step 重建 worker 和重新加载模型（641MB/worker 加载需 2-3 秒）

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- spawn DataLoader 已就绪，训练循环已适配
- 后续可实测不同 num_workers（2/4）对实际训练吞吐量的影响
- DDP 场景下每张卡独立创建 DataLoader 和 worker 已设计兼容

## Self-Check: PASSED

All files exist: dit_train/data/latent_dataset.py, dit_train/train_fm.py, tests/test_latent_dataset.py
All commits found: bf58b43, c65b94e

---
*Phase: 09-latentpairdataset*
*Completed: 2026-06-10*
