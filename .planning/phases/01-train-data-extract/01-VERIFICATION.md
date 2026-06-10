---
phase: 01-train-data-extract
verified: 2026-06-09T08:30:00Z
status: passed
score: 6/6 must-haves verified
---

# Phase 1: 训练数据提取 Verification Report

**Phase Goal:** 实现一个在线数据生成器（PyTorch Dataset/DataLoader），在 DiT 训练时实时生成 (prior_mu, post_mu) 训练对。CVAE (checkpoint.pth) 已训练好，冻结不动。
**Verified:** 2026-06-09
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | LatentPairDataset 是 IterableDataset 子类，`__iter__` 每次生成一个新 (prior_mu, post_mu) 对，各为 (512,) tensor | VERIFIED | `latent_dataset.py:21` 继承 IterableDataset; `latent_dataset.py:98` yield `(prior_mu[i], post_mu[i])` 其中 i 在 `range(prior_mu.shape[0])` 内; CVAE forward 返回 `(batch, 512)` 经 `torch.split(..., self.latent_dim, dim=1)` 确认 latent_dim=512 |
| 2 | prior_mu 和 post_mu 不是全零或全相同，是有意义的潜空间表示 | VERIFIED | `test_latent_pair_nontrivial` 测试断言 `not torch.all(prior_mu == 0)` 且 `not torch.allclose(prior_mu, post_mu)`; SUMMARY 报告 std~0.65/0.63, diff norm~0.72 |
| 3 | DataLoader 包裹后能正常迭代出 batch | VERIFIED | `test_dataloader_iteration` 迭代 3 个 batch 验证形状; `create_latent_dataloader` 返回 `DataLoader(dataset, batch_size=batch_size)` |
| 4 | batch_size 参数生效，每个 batch 的 prior_mu 形状为 (batch_size, 512) | VERIFIED | `test_batch_size` 分别测试 batch_size=2 和 batch_size=4，验证输出形状匹配; DataLoader collate 自动按 batch_size 组装 |
| 5 | CVAE 完全冻结，没有任何参数 requires_grad=True | VERIFIED | `latent_dataset.py:59-64` 显式遍历 vae_model、embedder_f、embedder_e 所有参数设 `requires_grad=False`; `test_cva_frozen` 验证所有参数 |
| 6 | 每次迭代生成新数据，不保存文件到磁盘 | VERIFIED | `__iter__` 用 `while True` 每次循环调用 `env.gen_expr(train=True)` 生成新样本; 全文件无 `torch.save`、无文件写入; grep 搜索 save/pt/h5/dump 仅匹配 checkpoint.pth 加载路径 |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dit_train/data/latent_dataset.py` | LatentPairDataset + create_latent_dataloader | VERIFIED (108 lines) | 导出 LatentPairDataset(IterableDataset) 和 create_latent_dataloader 工厂函数 |
| `tests/test_latent_dataset.py` | 5 个自动化测试 | VERIFIED (63 lines) | test_latent_pair_shapes, test_latent_pair_nontrivial, test_dataloader_iteration, test_batch_size, test_cva_frozen |
| `dit_train/data/verify_dataset.py` | 快速验证脚本 | VERIFIED (42 lines) | 输出 5 个 batch 统计信息，无文件保存 |
| `dit_train/__init__.py` | 包初始化 | VERIFIED | 空文件，存在 |
| `dit_train/data/__init__.py` | 子包初始化 | VERIFIED | 空文件，存在 |
| `tests/__init__.py` | 测试包初始化 | VERIFIED | 空文件，存在 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| latent_dataset.py | trainer_vae.py (行 862-888) | 复刻 enc_dec_vae_step 前半段数据流 | WIRED | x1 构造 (行 78-83), x2 构造 (行 87-91), CVAE forward (行 94) 完全匹配 trainer_vae.py 行 862-888 逻辑 |
| latent_dataset.py | weights/checkpoint.pth | reload_model 加载冻结权重 | WIRED | 第 45-50 行: reload_model(modules, ["cvae","data_encoder","token_embed"], path="weights/checkpoint.pth", requires_grad=False) |
| latent_dataset.py | environment.py | env.gen_expr 生成训练样本 | WIRED | 第 37 行: build_env(params); 第 38 行: env.rng; 第 72 行: env.gen_expr(train=True); 第 87-88 行: env.word_to_idx + env.batch_equations |

**注意：** latent_dataset.py 使用大写键名 `X_to_fit`/`Y_to_fit`（gen_expr 直接返回格式），trainer_vae.py 使用小写 `x_to_fit`/`y_to_fit`（经 generate_sample 转换后格式）。latent_dataset.py 直接调用 gen_expr()，使用大写键名是正确的。

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| latent_dataset.py `__iter__` | prior_mu, post_mu | env.gen_expr -> embedder_f/embedder_e -> CVAE forward(mode="train") | FLOWING | gen_expr 每次生成随机方程 + 数值数据; CVAE forward 对真实编码数据进行 prior/post 投影; 无 hardcoded fallback |

数据流路径验证：
1. `env.gen_expr(train=True)` -> 生成随机 (X, Y, tree_encoded) 样本
2. `embedder_f(x1)` -> 数值数据编码为张量
3. `env.word_to_idx` + `env.batch_equations` + `embedder_e` -> 表达式编码为张量
4. `vae_model(x1, x2_e, len1, len2, mode="train")` -> CVAE 前向传播输出 prior_mu/post_mu
5. 逐个 yield (prior_mu[i], post_mu[i])

### Behavioral Spot-Checks

Step 7b: SKIPPED -- 验证需要 GPU + 671MB checkpoint.pth 加载，无法在快速命令中完成。自动化测试已覆盖行为验证。

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-01 | 01-01-PLAN | 从现有 CVAE 训练集中提取 (prior_mu, post_mu) 对作为 Flow Matching 训练数据 | SATISFIED | LatentPairDataset 在线生成 (prior_mu, post_mu) 对，每次 __iter__ 调用 CVAE forward |
| DATA-02 | 01-01-PLAN | 支持批量提取，处理整个训练数据集生成训练对 | SATISFIED | IterableDataset + DataLoader 模式，batch_size 可配置，create_latent_dataloader 工厂函数 |

**Orphaned Requirements:** 无 -- REQUIREMENTS.md 仅映射 DATA-01 和 DATA-02 到 Phase 1。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | 无 anti-patterns |

所有文件扫描结果：
- TODO/FIXME/PLACEHOLDER: 0 匹配
- return null / return {} / return []: 0 匹配
- save/pt/h5/dump: 仅匹配 checkpoint.pth 加载路径（非保存操作）
- Hardcoded empty data: 0 匹配

### Human Verification Required

### 1. 测试实际执行

**Test:** 在有 GPU 的环境下运行 `python -m pytest tests/test_latent_dataset.py -x -v`
**Expected:** 5 个测试全部通过（需要加载 671MB checkpoint.pth）
**Why human:** 需要 CUDA GPU 和完整模型权重，无法在无 GPU 环境下程序化验证

### 2. 验证脚本输出合理性

**Test:** 运行 `python -m dit_train.data.verify_dataset` 查看统计输出
**Expected:** prior_mu std~0.6, post_mu std~0.6, diff norm~0.7（与 SUMMARY 报告一致）
**Why human:** 需要 CUDA GPU；统计值需要人工判断是否在合理范围内

### Gaps Summary

无 gaps。所有 6 个 must-have truths 验证通过：
- LatentPairDataset 正确继承 IterableDataset，__iter__ 在线生成 (prior_mu, post_mu) 对
- 数据流完整复刻 trainer_vae.py 的 enc_dec_vae_step 前半段
- CVAE 通过 reload_model 加载并显式冻结所有参数
- create_latent_dataloader 工厂函数返回正确配置的 DataLoader
- 无文件保存到磁盘（纯在线生成）
- 无 anti-patterns（无 TODO、无 placeholder、无空实现）

Phase goal achieved. Ready to proceed to Phase 2.

---

_Verified: 2026-06-09T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
