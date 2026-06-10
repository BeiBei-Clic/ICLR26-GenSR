# Roadmap: GenSR-DiT

## Overview

将 GenSR 符号回归系统的 CMA-ES 进化搜索替换为 DiT Flow Matching 先验传输。路线从训练数据提取开始，经过 DiT 模型构建和 Flow Matching 训练，到推理管线集成，最终在 PMLB 数据集上评估 DiT 方法。每个阶段交付一个可独立验证的能力。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: 训练数据提取** - 从 CVAE 在线生成 (prior_mu, post_mu) 训练对
- [x] **Phase 2: 训练数据验证** - 验证训练对的分布质量
- [x] **Phase 3: DiT 模型定义** - 实现 AdaLN-Zero 条件化 DiT 架构
- [x] **Phase 4: Flow Matching 训练** - 训练 DiT 学习 prior_mu -> post_mu 速度场
- [x] **Phase 5: Euler 积分推理** - 实现 DiT 先验传输推理核心
- [x] **Phase 6: 端到端推理管线** - 接入 Decoder + BFGS 完成推理闭环
- [ ] **Phase 7: 评估对比实验** - PMLB 上 DiT 评估实验
- [ ] **Phase 8: 结果记录与分析** - CSV/wandb 日志与结果汇总

## Phase Details

### Phase 1: 训练数据提取
**Goal**: 实现在线数据生成器（LatentPairDataset），实时生成 (prior_mu, post_mu) 训练对供 DiT 训练使用
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02
**Success Criteria** (what must be TRUE):
  1. LatentPairDataset 能在线生成 (prior_mu, post_mu) 对，形状各为 (batch_size, 512)
  2. CVAE 完全冻结（requires_grad=False），每次生成新数据不保存文件
  3. DataLoader 正常迭代，batch_size 参数生效
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — LatentPairDataset IterableDataset + 测试 + 验证脚本

### Phase 2: 训练数据验证
**Goal**: 用户可以验证提取的训练数据质量，确认 prior_mu 和 post_mu 分布合理
**Depends on**: Phase 1
**Requirements**: DATA-03
**Success Criteria** (what must be TRUE):
  1. 能生成 prior_mu 和 post_mu 的分布可视化图（直方图或 KDE）
  2. 能计算并输出 prior_mu 与 post_mu 之间的 KL 散度统计量
  3. 验证脚本执行后输出"通过/不通过"的质量判断
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md — 扩展 verify_dataset.py：1000 样本收集 + KL 散度 + 基础统计 + NaN/Inf 检测 + PASS/FAIL + 分布可视化 PDF

### Phase 3: DiT 模型定义
**Goal**: 用户可以实例化一个配置化的 DiT 模型，输入 (z_t, t, prior_mu) 输出 512 维速度预测
**Depends on**: Phase 2 (训练数据就绪后才能训练，但模型定义可并行)
**Requirements**: DIT-01, DIT-02, DIT-03, DIT-04, DIT-05
**Success Criteria** (what must be TRUE):
  1. DiT block 包含 LayerNorm + GELU + AdaLN-Zero 调制，可独立实例化
  2. DiT 接受 z_t (B, 512) patchify 后的 (B, 16, hidden_dim)，t 通过 AdaLN 条件化，prior_mu 通过 cross-attention 注入
  3. DiT 输出 (B, 512) 的速度预测 v_psi
  4. 可通过配置字典调整层数、注意力头数、隐藏维度，模型参数量随之变化
  5. 模型前向传播能跑通（随机输入不出错），输出 shape 正确
**Plans**: 1 plan

Plans:
- [x] 03-01-PLAN.md — GenSRDiT 模型定义：基础组件 + DiTBlock + 顶层模型 + shape 验证测试

### Phase 4: Flow Matching 训练
**Goal**: 用户可以用 Phase 1 提取的训练对训练 DiT 学会 prior_mu -> post_mu 的速度场
**Depends on**: Phase 2, Phase 3
**Requirements**: FM-01, FM-02, FM-03, FM-04, FM-05
**Success Criteria** (what must be TRUE):
  1. 训练循环实现 OT-path Flow Matching：z_t = (1-t)*prior_mu + t*post_mu，损失为速度预测 MSE
  2. CVAE/FeatureFusion/Decoder 参数冻结，只有 DiT 参数更新
  3. 训练过程中 loss 曲线在 wandb 或本地日志中持续下降
  4. checkpoint 按配置保存到磁盘，并能完整加载恢复训练
**Plans**: 1 plan

Plans:
- [x] 04-01-PLAN.md — FM 训练核心函数 + 测试 + 训练循环主体

### Phase 5: Euler 积分推理
**Goal**: 用户可以用训练好的 DiT 从 prior_mu 出发，经 N 步 Euler 积分得到优化后的潜向量 z_opt
**Depends on**: Phase 4
**Requirements**: INF-01, INF-04
**Success Criteria** (what must be TRUE):
  1. 给定 prior_mu 和 DiT checkpoint，Euler 积分能输出 z_opt (512-dim)
  2. 推理步数 timestep_num 可配置（如 5, 10, 20 步），步数越多结果越稳定
  3. 单次推理时间在毫秒级（不含 CVAE 编码和解码），显著快于 CMA-ES 50 代迭代
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md — euler_inference 函数 + 6 个单元测试

### Phase 6: 端到端推理管线
**Goal**: 用户可以输入 (X, Y) 数据，经过 CVAE 编码 -> DiT 传输 -> Decoder 解码 -> BFGS 优化，得到符号表达式
**Depends on**: Phase 5
**Requirements**: INF-02, INF-03
**Success Criteria** (what must be TRUE):
  1. z_opt 能正确传入 FeatureFusion + Decoder，生成合法的符号表达式 token 序列
  2. 生成的表达式经过 BFGS 常数优化后 R^2 有意义（非零非 NaN）
  3. 端到端管线可以作为函数调用：输入 (X, Y) → 输出 (expression, R^2)
**Plans**: 1 plan

Plans:
- [x] 06-01-PLAN.md — dit_inference 端到端管线函数 + 6 个 mock 测试

### Phase 7: 评估对比实验
**Goal**: 在全部 PMLB 回归数据集上运行 DiT 推理，输出评估结果 CSV（R^2、复杂度、推理时间、成功率）
**Depends on**: Phase 6
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. 评估脚本在全部 PMLB 回归数据集上运行 DiT 推理管线，输出每个数据集的 R^2 和表达式复杂度
  2. 推理时间数据可用，运行结束打印平均推理时间
  3. 成功率统计（R^2 > 0.99 比例）在运行结束后输出
**Plans**: 1 plan

Plans:
- [x] 07-01-PLAN.md — DiT 评估脚本 dit_eval.py + 试运行验证

### Phase 8: 结果记录与分析
**Goal**: 评估结果以结构化格式保存，支持后续分析和 wandb 追踪
**Depends on**: Phase 7
**Requirements**: EVAL-04
**Success Criteria** (what must be TRUE):
  1. 所有评估结果自动保存为 CSV 文件，包含数据集名、R^2、复杂度、推理时间、方法名
  2. wandb 日志集成，训练和评估指标可在 wandb dashboard 查看
  3. 结果 CSV 可直接用于后续论文表格和图表生成
**Plans**: 1 plan

Plans:
- [ ] 08-01-PLAN.md — (待规划)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 训练数据提取 | 1/1 | Complete | 2026-06-09 |
| 2. 训练数据验证 | 1/1 | Complete | 2026-06-09 |
| 3. DiT 模型定义 | 1/1 | Complete | 2026-06-09 |
| 4. Flow Matching 训练 | 1/1 | Complete | 2026-06-09 |
| 5. Euler 积分推理 | 1/1 | Complete | 2026-06-09 |
| 6. 端到端推理管线 | 1/1 | Complete | 2026-06-09 |
| 7. 评估对比实验 | 1/1 | Complete | 2026-06-09 |
| 8. 结果记录与分析 | 0/1 | Not started | - |
