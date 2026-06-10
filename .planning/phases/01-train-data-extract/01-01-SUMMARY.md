---
phase: 01-train-data-extract
plan: 01
subsystem: data
tags: [pytorch, iterable-dataset, cvae, latent-space, flow-matching]

# Dependency graph
requires:
  - phase: existing-gensr
    provides: "冻结的 CVAE 模型权重 (weights/checkpoint.pth), GenSR 组件 (build_env, build_modules, reload_model)"
provides:
  - "LatentPairDataset: IterableDataset 在线生成 (prior_mu, post_mu) 训练对"
  - "create_latent_dataloader: 工厂函数创建 DataLoader"
  - "verify_dataset.py: 快速数据质量验证脚本"
affects: [02-dit-model, 04-training-loop]

# Tech tracking
tech-stack:
  added: [pytest]
  patterns: ["IterableDataset + DataLoader 在线数据生成模式", "冻结 CVAE 并复用其前向传播提取潜空间表示"]

key-files:
  created:
    - dit_train/data/latent_dataset.py
    - dit_train/data/verify_dataset.py
    - tests/test_latent_dataset.py
    - dit_train/__init__.py
    - dit_train/data/__init__.py
    - tests/__init__.py
  modified: []

key-decisions:
  - "使用 --max_input_dimension 10 匹配 checkpoint 模型结构（默认值 1 会导致 shape 不匹配）"
  - "显式 param.requires_grad=False 冻结参数（reload_model 的 requires_grad 仅设模块属性）"
  - "tree_encoded 包在 [list] 中传给 word_to_idx（gen_expr 返回单样本 list of strings）"
  - "build_env 会修改 params.tasks 从字符串变 list，需要用前检查并恢复"

patterns-established:
  - "在线数据生成模式：IterableDataset + DataLoader，不保存磁盘文件"
  - "冻结 CVAE 复用模式：build_env -> build_modules -> reload_model -> eval()"

requirements-completed: [DATA-01, DATA-02]

# Metrics
duration: 15min
completed: 2026-06-09
---

# Phase 01 Plan 01: LatentPairDataset Summary

**IterableDataset 在线生成 (prior_mu, post_mu) 潜空间训练对，复用冻结 CVAE 前向传播，供 DiT Flow Matching 训练**

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-09T07:46:58Z
- **Completed:** 2026-06-09T08:02:11Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- LatentPairDataset 实现，每次迭代调用 CVAE forward 在线生成 (prior_mu, post_mu) 对
- 5 个自动化测试全部通过（形状、非平凡性、DataLoader 迭代、batch_size、CVAE 冻结）
- 验证脚本输出统计信息：prior_mu std~0.65, post_mu std~0.63, diff norm~0.72

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for LatentPairDataset** - `72101d1` (test)
2. **Task 1 (GREEN): Implement LatentPairDataset and create_latent_dataloader** - `a9e9b43` (feat)
3. **Task 2: verify_dataset.py for data quality validation** - `faa30e0` (feat)

## Files Created/Modified
- `dit_train/data/latent_dataset.py` - LatentPairDataset 类和 create_latent_dataloader 工厂函数
- `dit_train/data/verify_dataset.py` - 快速验证脚本，输出数据统计
- `tests/test_latent_dataset.py` - 5 个自动化测试
- `dit_train/__init__.py` - 包初始化
- `dit_train/data/__init__.py` - 子包初始化
- `tests/__init__.py` - 测试包初始化

## Decisions Made
- **--max_input_dimension 10**: checkpoint 训练时使用 max_input_dimension=10，默认值 1 导致 NumericalEmbedder shape 不匹配（384 vs 2112）
- **显式 param.requires_grad=False**: reload_model 中 `v.requires_grad=False` 仅设置模块属性，不冻结参数，需遍历参数显式设置
- **tree_encoded 包一层 list**: gen_expr(train=True) 返回单个 tree_encoded (list of strings)，word_to_idx 期望 list of lists
- **params.tasks list 恢复**: build_env 会将 params.tasks 从字符串改为 list，后续调用需恢复

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] checkpoint 模型结构与默认参数不匹配**
- **Found during:** Task 1 (LatentPairDataset 实现)
- **Issue:** get_parser().parse_args([]) 默认 max_input_dimension=1，但 checkpoint 用 10 训练，导致 NumericalEmbedder mlp_layers shape 不匹配 (384,384) vs (2112,2112)
- **Fix:** 测试和验证脚本中使用 --max_input_dimension 10
- **Files modified:** tests/test_latent_dataset.py, dit_train/data/verify_dataset.py
- **Verification:** reload_model 成功加载所有模块
- **Committed in:** a9e9b43 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] reload_model requires_grad 不冻结参数**
- **Found during:** Task 1 (test_cva_frozen 失败)
- **Issue:** reload_model 中 `v.requires_grad = requires_grad` 只设模块布尔属性，不递归设置参数的 requires_grad
- **Fix:** 在 LatentPairDataset.__init__ 中显式遍历 vae_model、embedder_f、embedder_e 的所有参数设 requires_grad=False
- **Files modified:** dit_train/data/latent_dataset.py
- **Verification:** test_cva_frozen 通过
- **Committed in:** a9e9b43 (Task 1 GREEN commit)

**3. [Rule 3 - Blocking] gen_expr tree_encoded 格式与 word_to_idx 不兼容**
- **Found during:** Task 1 (KeyError: 's')
- **Issue:** gen_expr(train=True) 返回 tree_encoded 为 list of strings，但 word_to_idx 期望 list of lists（遍历外层为 eq，内层为 token）
- **Fix:** 用 [samples["tree_encoded"]] 包一层
- **Files modified:** dit_train/data/latent_dataset.py
- **Verification:** test_latent_pair_shapes 通过
- **Committed in:** a9e9b43 (Task 1 GREEN commit)

**4. [Rule 3 - Blocking] build_env 修改 params.tasks 导致二次调用失败**
- **Found during:** Task 1 (test_dataloader_iteration 失败)
- **Issue:** build_env 将 params.tasks 从字符串改为 list，module-scoped fixture 第二次调用 build_env 时 .split() 报错
- **Fix:** 在 LatentPairDataset.__init__ 中检查 params.tasks 类型，如果是 list 则恢复为逗号分隔字符串
- **Files modified:** dit_train/data/latent_dataset.py
- **Verification:** test_dataloader_iteration 通过
- **Committed in:** a9e9b43 (Task 1 GREEN commit)

---

**Total deviations:** 4 auto-fixed (3 blocking, 1 bug)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- pip install py 安装到了系统 Python 而非 venv，需用 --target 修复

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LatentPairDataset 可直接被 DiT 训练循环 import 使用
- 使用时需传 params 并设 --max_input_dimension 10
- create_latent_dataloader(params, batch_size=N) 即可获取 DataLoader

---
*Phase: 01-train-data-extract*
*Completed: 2026-06-09*

## Self-Check: PASSED

All 7 files verified to exist on disk. All 3 task commits verified in git log (72101d1, a9e9b43, faa30e0).
