---
phase: 05-cython
plan: 02
subsystem: performance
tags: [cython, integration, compile-cache, hot-path, data-generation]

# Dependency graph
requires:
  - phase: 05-01
    provides: cython_eval.pyx 16 操作码栈式求值器 + build_cython.py
provides:
  - generators.py 中 Cython 求值路径替代 Python Node.val() 热路径
  - 25 种操作码全覆盖，无 fallback
  - 编译缓存避免重复编译
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [compile-cache-in-NodeList, no-fallback-direct-cython]

key-files:
  created:
    - scripts/benchmark_cython.py
  modified:
    - symbolicregression/envs/cython_eval.pyx
    - symbolicregression/envs/generators.py
    - tests/test_cython_eval.py

key-decisions:
  - "USER OVERRIDE: 不做 fallback，所有训练操作符直接在 Cython 中实现"
  - "新增 9 个操作码 (16-24): tan, arctan, arcsin, arccos, pow, max, min, sign, step"
  - "val_cython 使用 _compiled_cache 缓存编译结果，避免每次调用重复编译"
  - "arcsin/arccos 对越界输入返回 NaN 匹配 Python np.arcsin/np.arccos 行为"

patterns-established:
  - "编译缓存模式：NodeList._compiled_cache 首次编译后缓存，后续直接使用"
  - "全覆盖模式：所有 operators_real + operators_extra 中的操作符全部有 Cython 实现"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-06-09
---

# Phase 5 Plan 02: Cython 集成到训练热路径 Summary

**25 种操作符全覆盖的 Cython 栈式求值器替代 Python Node.val()，编译缓存实现端到端 2.0x 加速，数据生成管线验证通过**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-09T02:07:48Z
- **Completed:** 2026-06-09T02:12:04Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- cython_eval.pyx 从 16 种操作码扩展到 25 种，新增 tan/atan/asin/acos/pow/max/min/sign/step
- generators.py 集成 Cython 路径：compile_to_instructions + val_cython + 编译缓存
- _generate_datapoints 热路径已替换为 tree.val_cython(input)
- 端到端加速比 2.0x（Python 22.4 us/call vs Cython e2e 11.3 us/call）

## Task Commits

Each task was committed atomically:

1. **Task 1: 扩展 Cython 求值器支持 25 种操作符** - `cfc3f3f` (feat)
2. **Task 2: 集成 Cython 求值路径到 generators.py 热路径** - `241ad3d` (feat)
3. **Task 3: 添加编译缓存和基准测试脚本** - `cc9510d` (feat)

## Files Created/Modified
- `symbolicregression/envs/cython_eval.pyx` - 新增 9 个操作码 (16-24) 的求值逻辑，cimport tan/atan/asin/acos/pow/fmax/fmin
- `symbolicregression/envs/generators.py` - 添加操作码常量、_OP_MAP、compile_to_instructions、val_cython（含编译缓存）、热路径替换
- `tests/test_cython_eval.py` - 新增 test_new_operators 测试 9 个新操作符
- `scripts/benchmark_cython.py` - 性能基准测试脚本

## Decisions Made
- USER OVERRIDE 决定不做 fallback，所有训练操作符直接在 Cython 中实现，编译时遇到未知操作符直接报错
- 编译缓存（_compiled_cache）存储在 NodeList 实例上，首次编译后续复用
- step 操作符的 Python 实现有向量化 bug（标量比较数组），Cython 实现是正确的

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] step 测试中 Python Node.val() 向量化 bug**
- **Found during:** Task 1 (test_new_operators)
- **Issue:** Python 的 step 实现 `return x if x > 0 else 0` 使用标量比较处理数组，触发 ValueError
- **Fix:** 测试中使用 np.maximum(x, 0) 作为正确性基准而非 Python 实现
- **Files modified:** tests/test_cython_eval.py
- **Verification:** step 测试通过
- **Committed in:** cfc3f3f

**2. [Rule 2 - Missing Critical] 添加编译缓存**
- **Found during:** Task 3 (benchmark)
- **Issue:** val_cython 每次调用都重新编译树 + 构造 numpy 数组，端到端加速比仅 1.1x
- **Fix:** 在 NodeList 上添加 _compiled_cache，首次编译后缓存
- **Files modified:** symbolicregression/envs/generators.py
- **Verification:** 端到端加速比从 1.1x 提升到 2.0x
- **Committed in:** cc9510d

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** 两个修复都是必要的，无范围蔓延。

## Issues Encountered
- Python step 实现的向量化 bug 是已存在的，Cython 实现正确处理了向量情况

## User Setup Required
None - 无需外部服务配置。

## Next Phase Readiness
- Phase 5 全部完成，Cython 求值器已集成到训练热路径
- 如需进一步优化，可考虑批量编译多个 NodeList 或 Cython 端直接生成数据

---
*Phase: 05-cython*
*Completed: 2026-06-09*

## Self-Check: PASSED

All files verified present. All commits verified in git log.
