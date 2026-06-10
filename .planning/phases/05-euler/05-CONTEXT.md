# Phase 5: Euler 积分推理 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

实现 DiT 的 Euler 积分推理核心：给定 prior_mu 和训练好的 DiT checkpoint，经 N 步 Euler 积分得到优化后的潜向量 z_opt。

**范围内：**
- Euler 积分函数：从 z_0 = prior_mu 出发，t 从 0 到 1，每步 z_{t+dt} = z_t + dt * v_psi(z_t, t; prior_mu)
- 默认 16 步，步数可配置
- 加载 DiT checkpoint 并执行推理
- 支持 batch 推理（多个 prior_mu 同时积分）
- 单元测试验证积分正确性

**范围外：**
- 端到端推理管线（Phase 6：z_opt → FeatureFusion → Decoder → BFGS）
- 评估对比实验（Phase 7）
- CFG 推理（v2 feature）
- CLI 推理脚本（仅函数接口）

</domain>

<decisions>
## Implementation Decisions

### 推理步数
- **D-01:** 默认 16 步 Euler 积分，步数通过参数配置。与 Cola-DLM 默认一致。

### 接口形式
- **D-02:** 仅提供函数接口（euler_inference），不提供 CLI 脚本。Phase 6 和 Phase 7 通过函数调用。

### Batch 支持
- **D-03:** 直接支持 batch 推理。GenSRDiT.forward 本身接受 (B, 512)，Euler 积分自然支持 batch。

### 积分方向
- **D-04:** 从 t=0 到 t=1 积分（训练时 z_0 = prior_mu，z_1 ≈ post_mu）。均匀步长 dt = 1/num_steps。

### Claude's Discretion
- euler_inference 函数的具体参数和返回值设计
- 文件位置和命名
- 测试的组织方式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cola-DLM 推理逻辑（核心参考）
- `Cola-DLM-main/cola_dlm/inference.py:294` — timestep_num 参数
- `Cola-DLM-main/cola_dlm/inference.py:356-357` — _diffusion_dt 计算
- `Cola-DLM-main/cola_dlm/inference.py:478` — timesteps linspace 生成
- `Cola-DLM-main/cola_dlm/inference.py:608-649` — Euler 积分循环

### GenSR 已有组件
- `dit_train/model.py` — GenSRDiT 模型，forward(z_t, t, prior_mu) → v_psi (B, 512)
- `dit_train/train_fm.py` — Phase 4 交付的 load_checkpoint 函数
- `dit_train/train_fm.py:124-137` — load_checkpoint(dit, path, device) → (step, val_loss, ckpt)

### 项目文档
- `.planning/phases/04-flow-matching/04-CONTEXT.md` — Phase 4 决策
- `.planning/phases/03-dit/03-CONTEXT.md` — Phase 3 模型架构决策

</canonical_refs>

<code_context>
## Existing Code Insights

### Cola-DLM Euler 积分（需简化）
- T=1000.0, timestep_num=16, timesteps = linspace(T, 0, timestep_num+1)
- dt = (t_curr - t_next) / T
- 更新: z_{t-Δ} = z_t - Δ/T * v_psi(...)  (Cola 方向: T→0)
- Cola 的积分方向是从 T=1000 到 0（从噪声到数据）

### GenSR 适配（极简）
- 积分方向: t 从 0 到 1（从 prior_mu 到 post_mu）
- 时间归一化: 训练时 t ∈ [0,1]，推理也在 [0,1]
- 步长: dt = 1 / num_steps（均匀步长）
- 更新: z_{t+dt} = z_t + dt * v_psi(z_t, t; prior_mu)
- 初始: z_0 = prior_mu
- 终止: z_1 = z_opt

### GenSR Euler 积分核心（约 15 行）
```python
z = prior_mu.clone()
dt = 1.0 / num_steps
timesteps = torch.linspace(0, 1, num_steps + 1)  # [0, dt, 2*dt, ..., 1]
for i in range(num_steps):
    t_val = timesteps[i]
    t = torch.full((B,), t_val, device=device)
    v = dit(z, t, prior_mu)
    z = z + dt * v
return z  # z_opt
```

### 无需 Cola 的复杂机制
- 无 CFG（v1 不需要）
- 无 KV cache（无序列）
- 无 block-wise 生成（无序列）
- 无 clean-guidance（无序列）
- 无 per-sample noise seed（无随机噪声）

</code_context>

<specifics>
## Specific Ideas

- euler_inference 函数放在 dit_train/inference.py 中，与 train_fm.py 平级
- 函数签名: euler_inference(dit, prior_mu, num_steps=16) → z_opt
- prior_mu 可以是 (B, 512) 或 (512,)，函数内部统一处理
- 测试验证: 手动构造一个简单速度场（如常数速度），验证 Euler 积分结果正确
- 可选: 返回中间轨迹用于调试和可视化

</specifics>

<deferred>
## Deferred Ideas

- CLI 推理脚本（可通过 Phase 6 的端到端管线间接使用）
- CFG 推理（v2 feature ADV-01）
- 自适应步长（当前均匀步长足够）

</deferred>

---

*Phase: 05-euler*
*Context gathered: 2026-06-09*
