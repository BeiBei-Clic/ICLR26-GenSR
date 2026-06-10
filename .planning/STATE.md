---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 12-01-PLAN.md
last_updated: "2026-06-10T14:04:28.968Z"
last_activity: 2026-06-10
progress:
  total_phases: 12
  completed_phases: 11
  total_plans: 11
  completed_plans: 11
  percent: 87
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-09)

**Core value:** 用 DiT Flow Matching 替换 CMA-ES，将符号回归推理从迭代进化搜索变为单次前向传输
**Current focus:** Phase 12 — decoder-lm-head-gpu

## Current Position

Phase: 12
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-06-10

Progress: [████████░░] 87%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01 P01 | 15m | 2 tasks | 6 files |
| Phase 02 P01 | 4min | 2 tasks | 2 files |
| Phase 03-dit P01 | 3min | 2 tasks | 2 files |
| Phase 04-flow-matching P01 | 3min | 2 tasks | 2 files |
| Phase 05-euler P01 | 3min | 1 tasks | 2 files |
| Phase 06 P01 | 2min | 1 tasks | 2 files |
| Phase 07-eval P01 | 13min | 2 tasks | 1 files |
| Phase 09 P01 | 3min | 2 tasks | 3 files |
| Phase 10 P01 | 8min | 2 tasks | 2 files |
| Phase 11 P01 | 4min | 2 tasks | 1 files |
| Phase 12 P01 | 3min | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 09]: spawn 模式替代 fork: fork 在主进程已初始化 CUDA 后不可用，spawn 是唯一可行方案
- [Phase 09]: num_workers=2 默认值: RTX 3090 上每个 worker 额外 641MB 显存，2 worker 总额外 1.3GB 安全
- [Phase 09]: yield 完整 batch 跳过 DataLoader collate: Dataset 内部组装 batch，DataLoader batch_size=None 透传
- [Phase 10]: lm_head.weight 与 tok_embed.weight 权重共享 (share_inout_emb=True)，解冻 lm_head 时 tok_embed 也变为可训练
- [Phase 10]: 逐样本 teacher-forcing CE loss 然后平均，因为方程长度不同无法直接 batch
- [Phase 11]: 验证循环内联在训练循环中，每 50 步用 gen_expr(train=False) 独立验证，best_val_loss 跟踪保存最优权重
- [Phase 12]: 只包装 lm_head 为 DDP，不包装整个 decoder（92.9% 参数冻结，find_unused_parameters 性能差）
- [Phase 12]: 保存权重用 lm_head_raw（未包装原始引用），避免 DDP module. 前缀问题
- [Phase 12]: 验证循环仅在 rank 0 执行 + dist.barrier() 同步其他 rank

### Roadmap Evolution

- Phase 12 added: Decoder lm_head 微调多 GPU 并行训练
- Phase 11 added: Decoder lm_head 微调验证循环：每隔一定步数在验证集上评估，保存最优权重
- Phase 10 added: 联合训练 DiT 与 VAE：参考 Cola-DLM 微调方法，将解码器交叉熵损失纳入训练，实现 DiT 与 VAE 联合端到端训练
- Phase 9 added: 优化 LatentPairDataset 数据生成性能

- [Roadmap]: 5 个粗阶段拆分为 8 个细阶段（fine 粒度），DATA 拆为提取+验证，INF 拆为推理核心+端到端管线，EVAL 拆为实验+日志
- [Phase 01]: 使用 --max_input_dimension 10 匹配 checkpoint 模型结构
- [Phase 01]: 显式 param.requires_grad=False 冻结参数（reload_model 仅设模块属性）
- [Phase 01]: tree_encoded 包 [list] 传给 word_to_idx（gen_expr 单样本格式差异）
- [Phase 02]: KL 散度阈值 0.01、diff norm 阈值 0.1 作为训练数据质量门控；latent_distribution.pdf 为生成文件加入 .gitignore
- [Phase 03-dit]: 每个子块使用独立 AdaLN 实例（ada_sa, ada_ca, ada_ff）简化接口
- [Phase 03-dit]: 手写缩放点积注意力保持实现透明
- [Phase 04-flow-matching]: Task 1+2 在同一 TDD 周期实现 — train_fm.py 同时包含核心函数和训练循环
- [Phase 05-euler]: Test 3 (adjustable steps) uses DirectedVelocityDiT — untrained GenSRDiT outputs zeros due to AdaLN-Zero init
- [Phase 05-euler]: Test 6 (direction) verifies convergence direction rather than absolute precision — v=target-z ODE has exponential convergence
- [Phase 06]: z_opt as mu, prior_logvar as logvar into FeatureFusion — keeps sampling distribution consistent with CVAE
- [Phase 07-eval]: FlowMatchingModel 替代 GenSRDiT: fm_best.pth 使用 MLP-like 结构（6 层 AdaLN block，无 attention），与 dit_train/model.py 的 Transformer DiT 不匹配

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-06-10T14:00:28.676Z
Stopped at: Completed 12-01-PLAN.md
Resume file: None
