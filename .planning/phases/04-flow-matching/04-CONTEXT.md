# Phase 4: Flow Matching 训练 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

实现 DiT 的 Flow Matching 训练循环，让模型学会 prior_mu → post_mu 的速度场。训练数据来自 Phase 1 的 LatentPairDataset 在线生成，DiT 模型来自 Phase 3 的 GenSRDiT。

**范围内：**
- Flow Matching 训练脚本（参照 cola_sft.py 原生 PyTorch 风格）
- OT-path 插值：z_t = (1-t)*prior_mu + t*post_mu，速度目标 u = post_mu - prior_mu
- Logit-normal 时间步采样
- AdamW 优化器 + cosine LR schedule（warmup + warmdown）
- Gradient clipping
- 训练过程中的 eval 和 checkpoint 保存
- 简单 print 日志（后续加 wandb）
- 训练循环验证（loss 能正常下降）

**范围外：**
- Euler 积分推理（Phase 5）
- 端到端推理管线（Phase 6）
- CFG（v2 feature ADV-01）
- Wandb 日志（后续添加）
- 多 GPU DDP（单卡训练）

</domain>

<decisions>
## Implementation Decisions

### 训练脚本形式
- **D-01:** 参照 cola_sft.py 写独立原生 PyTorch 脚本，手写 for loop + optimizer.step()。不用 PyTorch Lightning。

### 时间步采样
- **D-02:** 默认使用 Logit-normal 时间步采样（loc=0.0, scale=1.0），与 Cola-DLM 默认一致。也支持 uniform 作为可选项。

### 训练超参数（照搬 Cola-DLM）
- **D-03:** AdamW lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01
- **D-04:** Cosine LR schedule: warmup 5% + warmdown 30% + final_lr_frac=0.0
- **D-05:** device_batch_size=4, grad_accum_steps=8 (effective batch=32)
- **D-06:** Gradient clipping max_norm=1.0
- **D-07:** EMA loss smoothing (beta=0.95) 用于日志显示

### 日志
- **D-08:** 先用 print 日志（loss, lr, dt），后续再加 wandb。

### 不需要的 Cola-DLM 机制（已在 Phase 3 决定）
- **D-09:** 无 2L trick、无 NA layout、无 loss_mask、无 block-causal attention。GenSR 是 512 维单向量，所有位置都参与 loss 计算。
- **D-10:** 无 CFG（v2 feature），训练时无条件分支。

### Claude's Discretion
- 训练脚本的具体文件位置和命名
- Checkpoint 保存格式（state_dict）
- Eval 频率和方式
- 命令行参数的组织方式
- 数据加载与训练循环的具体实现细节

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cola-DLM 训练逻辑（核心参考）
- `Cola-DLM-main/scripts/cola_sft.py` — Cola-DLM 训练脚本完整实现
- `Cola-DLM-main/scripts/cola_sft.py:41-80` — CLI 参数定义
- `Cola-DLM-main/scripts/cola_sft.py:179-196` — AdamW 优化器 + LR schedule
- `Cola-DLM-main/scripts/cola_sft.py:440-445` — 时间步采样（logit-normal / uniform）
- `Cola-DLM-main/scripts/cola_sft.py:451-534` — flow_matching_step 核心训练逻辑
- `Cola-DLM-main/scripts/cola_sft.py:633-691` — 训练循环主体

### GenSR 已有组件
- `dit_train/data/latent_dataset.py` — Phase 1 交付的在线数据生成器
- `dit_train/model.py` — Phase 3 交付的 GenSRDiT 模型
- `dit_train/model.py:22-31` — dit_default_config 默认配置

### 项目文档
- `.planning/phases/03-dit/03-CONTEXT.md` — Phase 3 决策（模型架构、D-01 到 D-07）

</canonical_refs>

<code_context>
## Existing Code Insights

### Cola-DLM 训练核心逻辑（需保留/适配）
- OT-path: z_noisy = (1-t)*z_0 + t*z_1, target = z_1 - z_0
- Loss: MSE with optional loss_mask (GenSR 不需要 mask，全维度参与)
- Timestep: logit_normal → Sigmoid(loc + scale * N(0,1))
- LR schedule: warmup 5% → plateau → warmdown 30% linear decay to 0
- Eval: 每 N 步跑 eval_steps 个 batch 平均 loss
- Checkpoint: save_pretrained + optimizer state_dict + meta.json
- EMA smoothing: beta=0.95 用于日志显示

### Cola-DLM 序列机制（GenSR 不需要）
- 2L trick: clean + noisy copies 拼接 → 不需要
- NA layout: txt_shape, txt_q_shape → 不需要
- Block-causal mask: create_2l_block_causal_mask → 不需要
- Loss mask: 按 token 角色 (P/R) 分配 → 不需要，全维度参与
- Position IDs: k_position_ids, q_position_ids → 不需要

### GenSR Flow Matching 适配要点
- 输入: prior_mu (B, 512) 和 post_mu (B, 512) 从 LatentPairDataset 在线获取
- 插值: z_t = (1-t)*prior_mu + t*post_mu（向量化，无需逐 token 循环）
- 目标: velocity = post_mu - prior_mu（简单差值）
- 模型: GenSRDiT(z_t, t, prior_mu) → predicted_velocity (B, 512)
- Loss: MSE(predicted_velocity, target_velocity)，无 mask
- 冻结: CVAE/FeatureFusion/Decoder 全部冻结，只训练 DiT 参数

</code_context>

<specifics>
## Specific Ideas

- 训练脚本参照 cola_sft.py 的结构：CLI 参数 → 模型加载 → 数据加载 → 训练循环 → eval → checkpoint
- GenSR 的 FM step 比 Cola 简单得多：无序列、无 mask、无 2L trick，核心就是 z_t 插值 + MSE loss
- 关键简化：Cola 的 build_noisy_sample 是 150+ 行（处理序列角色），GenSR 只需 3 行向量操作
- LR schedule 函数可以照搬 get_lr_multiplier
- Checkpoint 保存：torch.save(state_dict) 格式，因为 GenSRDiT 不是 HuggingFace 模型

</specifics>

<deferred>
## Deferred Ideas

- Wandb 日志集成（后续添加，不影响训练脚本核心逻辑）
- CFG (Classifier-Free Guidance) 训练（v2 feature ADV-01）
- 多 GPU DDP 支持（当前单卡训练足够）

</deferred>

---

*Phase: 04-flow-matching*
*Context gathered: 2026-06-09*
