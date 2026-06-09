---
phase: 05-cython
plan: 01
subsystem: performance
tags: [cython, numpy, typed-memoryview, stack-evaluator, c-extension]

# Dependency graph
requires:
  - phase: 01-flow-matching
    provides: generators.py 中的 Node.val() Python 实现作为正确性基准
provides:
  - cython_eval.pyx: 16 操作码栈式求值器（C 编译后 3.8x 加速）
  - build_cython.py: setuptools 编译脚本
  - compile_tree(): Python 端树编译函数（测试文件中）
affects: [05-02-integration]

# Tech tracking
tech-stack:
  added: [Cython-3.2.4, gcc-12.3.0-O3]
  patterns: [flat-instruction-stack-evaluator, post-order-tree-compilation, typed-memoryview]

key-files:
  created:
    - symbolicregression/envs/cython_eval.pyx
    - symbolicregression/envs/build_cython.py
    - tests/test_cython_eval.py
  modified:
    - .gitignore

key-decisions:
  - "OP_DIV 分母为 0 时显式置 NaN 以匹配 Python Node.val() 行为（cdivision=True 下 1/0=inf 而非 NaN）"
  - "移除 -ffast-math 编译选项，避免与 numpy 2.x SIMD 内部函数冲突导致 undefined symbol"
  - "测试文件支持 python 直接执行和 pytest 两种运行方式"

patterns-established:
  - "栈式求值器模式：后序遍历编译为扁平指令列表，C 端用栈执行"
  - "编译缓存模式：compile_tree 一次、eval_tree_flat 多次"

requirements-completed: []

# Metrics
duration: 8min
completed: 2026-06-09
---

# Phase 5 Plan 01: Cython 栈式求值器 Summary

**Cython 栈式求值器 eval_tree_flat 覆盖 16 种操作符，cdivision=True + typed memoryview 实现 3.8x 加速，6 个正确性测试全部通过**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-09T01:56:40Z
- **Completed:** 2026-06-09T02:04:57Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Cython 栈式求值器 eval_tree_flat 支持全部 16 种操作符，编译后 .so 可正常 import
- 正确性与 Python Node.val() 完全一致，包括 NaN 边界条件（div-by-zero、log(-1)、sqrt(-1)、inv(0)、inf*0）
- 速度对比：Python 33.8 us/call vs Cython 8.8 us/call，3.8x 加速

## Task Commits

Each task was committed atomically:

1. **Task 1: 创建 Cython 栈式求值器和编译脚本** - `1ee0434` (feat)
2. **Task 2: 创建正确性对比测试脚本** - `ef2bfd7` (test)

## Files Created/Modified
- `symbolicregression/envs/cython_eval.pyx` - Cython 栈式求值器核心，16 操作码，typed memoryview
- `symbolicregression/envs/build_cython.py` - setuptools 编译配置
- `tests/test_cython_eval.py` - 6 个正确性测试 + 速度对比
- `tests/__init__.py` - tests 包初始化
- `.gitignore` - 排除 .so 和 .c 编译产物

## Decisions Made
- OP_DIV 分母为 0 时显式置 NaN：cdivision=True 下 C 的除零返回 inf，但 Python 端先置 NaN 再除，两端行为需一致
- 移除 -ffast-math：编译时 -ffast-math 导致 numpy 2.x 的 `_ZGVbN2v_exp` undefined symbol，改用纯 -O3
- 测试文件双模式运行：因 pytest 缺少 py 模块依赖（pip SSL 问题），测试文件支持 `python tests/test_cython_eval.py` 直接执行

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 修复 OP_DIV 除零语义不匹配**
- **Found during:** Task 2 (test_nan_handling)
- **Issue:** cdivision=True 下 1.0/0.0 返回 inf，但 Python Node.val() 中先将 denominator==0 置 NaN 再除，结果为 NaN
- **Fix:** OP_DIV 中先检查 sv[n_stack, j] == 0.0，如果是则直接置 NAN
- **Files modified:** symbolicregression/envs/cython_eval.pyx
- **Verification:** test_nan_handling div-by-zero 通过
- **Committed in:** ef2bfd7 (Task 2 commit)

**2. [Rule 3 - Blocking] 移除 -ffast-math 编译选项**
- **Found during:** Task 1 (编译验证)
- **Issue:** -ffast-math 导致 undefined symbol `_ZGVbN2v_exp`，import 失败
- **Fix:** build_cython.py 中 extra_compile_args 只保留 ["-O3"]
- **Files modified:** symbolicregression/envs/build_cython.py
- **Verification:** 编译成功，import 无报错
- **Committed in:** 1ee0434 (Task 1 commit)

**3. [Rule 1 - Bug] 修正 inf*0 测试用例中的输入值**
- **Found during:** Task 2 (test_nan_handling)
- **Issue:** Plan 中使用 x_0=700 构造 inf，但 np.exp(700) 在 float64 下为有限大数（非 inf），实际需要 x_0=710
- **Fix:** 将测试输入从 700.0 改为 710.0
- **Files modified:** tests/test_cython_eval.py
- **Verification:** Cython 和 Python 两端均返回 NaN，一致
- **Committed in:** ef2bfd7 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bug, 1 blocking)
**Impact on plan:** 所有修复为保证正确性和编译通过所必需，无范围蔓延。

## Issues Encountered
- pytest 安装依赖链不完整（缺少 py 模块），测试文件改为支持 `python` 直接执行模式

## User Setup Required
None - 无需外部服务配置。

## Next Phase Readiness
- eval_tree_flat 已就绪，05-02-plan 可将 compile_tree 集成到 generators.py 中
- 编译缓存模式已在测试中验证，05-02 可直接复用 compile_tree + eval_tree_flat 模式
- build_cython.py 编译命令已验证可用

---
*Phase: 05-cython*
*Completed: 2026-06-09*

## Self-Check: PASSED

All files verified present. All commits verified in git log.
