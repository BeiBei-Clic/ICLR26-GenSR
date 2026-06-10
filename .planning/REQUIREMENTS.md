# Requirements: GenSR-DiT

**Defined:** 2026-06-09
**Core Value:** 用 DiT Flow Matching 替换 CMA-ES，将符号回归推理从迭代进化搜索变为单次前向传输

## v1 Requirements

Requirements for initial working system. Each maps to roadmap phases.

### Training Data

- [x] **DATA-01**: 从现有 CVAE 训练集中提取 (prior_mu, post_mu) 对作为 Flow Matching 训练数据
- [x] **DATA-02**: 支持批量提取，处理整个训练数据集生成训练对文件（如 .pt 或 .h5）
- [x] **DATA-03**: 验证提取的 (prior_mu, post_mu) 对的质量（分布可视化、KL 散度统计）

### DiT Model

- [] **DIT-01**: 实现 AdaLN-Zero 条件化的 DiT block（RMSNorm + SwiGLU + AdaLN 调制）
- [] **DIT-02**: DiT 接受输入 (z_t, t, prior_mu)，其中 t 通过 AdaLN 条件化，prior_mu 通过 cross-attention 或拼接作为条件
- [x] **DIT-03**: 输出 512 维速度预测 v_psi(z_t, t; prior_mu)
- [x] **DIT-04**: 支持 patchify 512 维向量（如 16 patches × 32 dim）以利用注意力机制
- [x] **DIT-05**: DiT 配置可调（层数、注意力头数、隐藏维度）

### Flow Matching Training

- [x] **FM-01**: 实现 OT-path Flow Matching 训练：z_t = (1-t)*prior_mu + t*post_mu，速度目标 u_t = post_mu - prior_mu
- [x] **FM-02**: 训练损失：L = E_{t, prior_mu, post_mu} || v_psi(z_t, t; prior_mu) - (post_mu - prior_mu) ||^2
- [x] **FM-03**: 训练时 CVAE、FeatureFusion、Decoder 全部冻结，只训练 DiT
- [x] **FM-04**: 训练日志记录（loss 曲线、速度预测误差）
- [x] **FM-05**: DiT checkpoint 保存与加载

### Inference Pipeline

- [x] **INF-01**: 实现 Euler 积分推理：从 prior_mu 出发，经 N 步 DiT 传输得到 z_opt
- [x] **INF-02**: z_opt 传入 FeatureFusion → Decoder → 符号表达式
- [x] **INF-03**: 生成的表达式经 BFGS 常数优化（复用现有 refine 管线）
- [x] **INF-04**: 支持可调推理步数（timestep_num 参数）

### Evaluation

- [x] **EVAL-01**: 在 PMLB Feynman 数据集上评估 DiT 方法 vs CMA-ES 方法的 R^2、复杂度
- [x] **EVAL-02**: 对比推理时间（DiT 单次前向 vs CMA-ES 50-100 代迭代）
- [x] **EVAL-03**: 统计成功率（R^2 > 0.99 的比例）
- [ ] **EVAL-04**: 结果保存为 CSV，支持 wandb 日志

## v2 Requirements

### Advanced Features

- **ADV-01**: Classifier-Free Guidance 支持（无条件 + 条件 DiT 前向融合）
- **ADV-02**: 多步 refinement（多次 prior_mu -> z_opt 循环，逐步优化）
- **ADV-03**: 支持 beam search（DiT 输出多个候选潜向量，选最优）
- **ADV-04**: 在 Strogatz 和其他黑盒数据集上评估

## Out of Scope

| Feature | Reason |
|---------|--------|
| 重构 VAE 或 Decoder | 保持现有架构不动，只替换搜索过程 |
| 将 Decoder 改为非自回归 | 非本次目标，需要完全重新训练 |
| 将潜空间改为序列 | 保持 512 维单向量，最小化改动 |
| 训练 Cola-DLM 本身 | 只迁移方法论，不训练文本模型 |
| 多 GPU DiT 训练 | 512 维输入，DiT 规模小，单卡足够 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1: 训练数据提取 | Complete |
| DATA-02 | Phase 1: 训练数据提取 | Complete |
| DATA-03 | Phase 2: 训练数据验证 | Complete |
| DIT-01 | Phase 3: DiT 模型定义 | Complete |
| DIT-02 | Phase 3: DiT 模型定义 | Complete |
| DIT-03 | Phase 3: DiT 模型定义 | Complete |
| DIT-04 | Phase 3: DiT 模型定义 | Complete |
| DIT-05 | Phase 3: DiT 模型定义 | Complete |
| FM-01 | Phase 4: Flow Matching 训练 | Complete |
| FM-02 | Phase 4: Flow Matching 训练 | Complete |
| FM-03 | Phase 4: Flow Matching 训练 | Complete |
| FM-04 | Phase 4: Flow Matching 训练 | Complete |
| FM-05 | Phase 4: Flow Matching 训练 | Complete |
| INF-01 | Phase 5: Euler 积分推理 | Complete |
| INF-04 | Phase 5: Euler 积分推理 | Complete |
| INF-02 | Phase 6: 端到端推理管线 | Complete |
| INF-03 | Phase 6: 端到端推理管线 | Complete |
| EVAL-01 | Phase 7: 评估对比实验 | Complete |
| EVAL-02 | Phase 7: 评估对比实验 | Complete |
| EVAL-03 | Phase 7: 评估对比实验 | Complete |
| EVAL-04 | Phase 8: 结果记录与分析 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-06-09*
*Last updated: 2026-06-09 after roadmap creation*
