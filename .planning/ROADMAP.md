# Roadmap: GenSR-Cola 流匹配推理

## Phase 1: Flow Matching 核心模块实现
**目标**：实现 Flow Matching 模型、训练 loss 和 ODE 推理器

### Tasks
1. **实现 Flow Matching 模型** (`symbolicregression/model/flow_matching.py`)
   - ConditionalFlowMatcher 类：MLP 网络，输入 (z_t, t, condition)，输出向量场 v
   - 时间嵌入（sinusoidal 或 learned）
   - 条件注入（concat 或 cross-attention）

2. **实现 ODE 推理器** (`symbolicregression/model/flow_matching.py`)
   - Euler 求解器
   - Heun 求解器（二阶）
   - 支持多次采样 + 选取最优

3. **实现 Flow Matching 训练 loss**
   - OT-CFM (Optimal Transport Conditional Flow Matching)
   - 从 (z₀=post_mu, z₁=noise) 构造插值路径 z_t = (1-t)z₀ + t·z₁
   - 条件向量场目标 u_t = z₁ - z₀

4. **更新 build_modules 和 parsers**
   - 新增 `--fm_hidden_dim`, `--fm_n_layers`, `--fm_n_samples`, `--fm_ode_steps` 等参数
   - 在 `build_modules` 中注册 Flow Matching 模型

**验证**：Flow Matching 模型可以实例化、前向传播、计算 loss

---

## Phase 2: Stage 2 训练流程
**目标**：实现 Flow Matching 的训练脚本，可从 CVAE checkpoint 启动训练

### Tasks
1. **实现 Flow Matching 训练 step** (`symbolicregression/trainer_vae.py` 扩展)
   - `fm_train_step`: 从 CVAE 获取 (prior_mu, post_mu) 对
   - 冻结 CVAE（可选微调），只训练 Flow Matching 模型
   - 记录 FM loss 到 WandB

2. **实现训练脚本** (`train_fm.py`)
   - 加载 CVAE checkpoint
   - 遍历训练数据，收集 latent 对，训练 Flow Matching
   - 定期评估和保存 checkpoint

3. **评估集成**
   - 在训练过程中定期用 Flow Matching 推理评估 Feynman 数据集
   - 对比 direct decode 和 FM decode 的 R²

**验证**：Flow Matching loss 收敛，能从噪声生成合理的潜在表示

---

## Phase 3: 推理替换 & PMLB 批量测试
**目标**：替换 CMA-ES 推理，跑通 PMLB 批量测试

### Tasks
1. **实现 Flow Matching 推理函数** (`fm_inference.py` 或 `LSO_fit.py` 扩展)
   - `fm_fit`: data → CVAE encoder → prior_mu → FM ODE → z₀ → decode → BFGS
   - 替代 `lso_fit_es_covfromvae_fit` 的接口

2. **修改 pmlb_batch_inference.py**
   - 新增 `--inference_mode` 参数：`cma_es` / `flow_matching`
   - Flow Matching 模式：加载 FM checkpoint，直接推理
   - 输出格式与 CMA-ES 模式兼容

3. **端到端测试**
   - 先在 2 个 Feynman 数据集上验证
   - 跑通 PMLB 批量推理
   - 汇总结果

**验证**：`pmlb_batch_inference.py --inference_mode flow_matching` 成功运行并输出结果 CSV

---

## Phase 4: Flow Matching 多 GPU 训练
**目标**：为 `train_fm.py` 添加 DDP 多卡并行支持，大幅加速 FM 训练
**Plans:** 1 plan

Plans:
- [x] 04-01-PLAN.md — DDP 多卡训练支持（train_fm.py + train_fm.sh） -- COMPLETED 2026-06-07

**Requirements:** FR-5.1, FR-5.2, FR-5.3, FR-5.4, FR-5.5

---

## Phase 5: Cython 重写表达式求值热路径
**目标**：用 Cython 重写 `Node.val(x)` 表达式树求值热路径，消除 Python 解释器开销，提升数据生成吞吐量，从而提高 GPU 训练利用率
**Depends on:** Phase 4
**Plans:** 2 plans

Plans:
- [x] 05-01-PLAN.md -- Cython 栈式求值器实现 + 正确性测试 -- COMPLETED 2026-06-09
- [x] 05-02-PLAN.md -- generators.py 集成 + 端到端验证 + 性能基准 -- COMPLETED 2026-06-09

---

## 依赖关系
- Phase 2 依赖 Phase 1（模型定义）
- Phase 3 依赖 Phase 2（训练好的模型）
- Phase 4 依赖 Phase 2（在 train_fm.py 基础上改）
- Phase 5 依赖 Phase 4（在优化后的训练管线基础上进一步提升）
- Phase 1 内部任务可部分并行
