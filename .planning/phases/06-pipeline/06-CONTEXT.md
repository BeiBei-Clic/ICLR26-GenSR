# Phase 6: 端到端推理管线 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

实现端到端推理管线函数：输入 (X, Y) 数据，经 CVAE 编码 → DiT Euler 积分 → FeatureFusion → Decoder 解码 → BFGS 常数优化 → 输出 (expression, R^2)。

**范围内：**
- 端到端推理函数 `dit_inference(X, Y, models, ...)`
- CVAE 编码获取 (prior_mu, prior_logvar)
- DiT Euler 积分得到 z_opt
- FeatureFusion 转换为 Decoder 输入
- Decoder 自回归生成表达式
- BFGS 常数优化
- 单元测试验证管线正确性

**范围外：**
- 评估对比实验（Phase 7）
- 结果记录与分析（Phase 8）
- CFG 推理（v2 feature）
- Beam search（v2 feature）

</domain>

<decisions>
## Implementation Decisions

### logvar 处理
- **D-01:** z_opt 作为 mu，logvar 使用 CVAE 编码时产生的 prior_logvar。保持与训练时一致的采样分布。

### 推理接口
- **D-02:** 简单端到端函数。输入 (X, Y) → 输出 (best_expression, R^2)。支持 batch。
- **D-03:** 函数接受已加载的模型对象（CVAE, DiT, FeatureFusion, Decoder, env, params）。调用方负责加载和初始化。

### 管线流程
- **D-04:** 管线步骤：
  1. 数据预处理：X, Y → sample_to_learn dict
  2. CVAE 编码：sample_to_learn → (prior_mu, prior_logvar)
  3. DiEuler 积分：euler_inference(dit, prior_mu) → z_opt
  4. FeatureFusion：(z_opt, prior_logvar) → src_enc (B, 200, 512)
  5. Decoder 解码：src_enc → token 序列 → 表达式字符串
  6. BFGS 优化：表达式 → 常数优化 → 最终表达式 + R^2

### Claude's Discretion
- 推理函数的具体参数和返回值细节
- 文件位置和命名
- BFGS 调用的具体方式
- 测试的组织方式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GenSR 推理管线（核心参考）
- `LSO_fit.py:217-625` — `lso_fit_es_covfromvae_fit()` 现有推理主函数
- `LSO_fit.py:gen2eq()` — token 序列 → 表达式树 → 评估
- `model.py` — VAESymbolicRegressor 接口
- `model.py:reload_model()` — 模型加载

### GenSR 组件
- `symbolicregression/model/feature_fusion.py` — FeatureFusion(mu, logvar) → src_enc
- `symbolicregression/model/cvae.py:233` — TransformerModel_VAE.generate_from_latent()
- `const_opt.py` — BFGS 常数优化 refine()

### DiT 已有组件
- `dit_train/inference.py` — euler_inference(dit, prior_mu, num_steps) → z_opt
- `dit_train/train_fm.py:load_checkpoint()` — 加载 DiT checkpoint
- `dit_train/data/latent_dataset.py` — CVAE 编码参考（如何初始化 params 和模块）

### 项目文档
- `.planning/phases/05-euler/05-CONTEXT.md` — Phase 5 Euler 积分决策

</canonical_refs>

<code_context>
## Existing Code Insights

### 现有推理管线关键步骤（LSO_fit.py）
1. `model(sample, return_logvar=True)` → (prior_mu, generations, gen_len, prior_logvar)
2. CMA-ES 迭代搜索更好的 prior_mu
3. `model.generate_from_latent_sampling(src_enc)` → token 序列
4. `gen2eq()` → 表达式树 → 评估
5. `const_opt.refine()` → BFGS 常数优化

### DiT 替换点
- 替换步骤 2（CMA-ES 搜索）为 euler_inference
- 步骤 1, 3, 4, 5 保持不变
- 关键差异：CMA-ES 搜索 prior_mu 空间（512 维），DiEuler 积分也是 prior_mu → z_opt

### VAESymbolicRegressor 关键方法
- `encode_only(sample)` → (prior_mu, prior_logvar)
- `prepare_latent_for_decoder(mu, logvar)` → src_enc（调用 FeatureFusion）
- `generate_from_latent_sampling(src_enc)` → (generations, gen_len)

### 模型初始化模板（参照 latent_dataset.py）
```python
from parsers import get_parser
from symbolicregression.envs.build_env import build_env
from symbolicregression.model.build_modules import build_modules
from symbolicregression.model.utils import reload_model

params = get_parser().parse_args(["--max_input_dimension", "10"])
params.device = "cuda"
env = build_env(params)
modules = build_modules(env, params)
modules = reload_model(modules, params.reload_from)
```

### BFGS 调用方式
```python
from const_opt import refine
refined = refine(env, X, y, candidates, verbose=False)
```

</code_context>

<specifics>
## Specific Ideas

- 推理函数放在 `dit_train/pipeline.py`
- 函数签名: `dit_inference(X, Y, env, params, modules, dit, num_steps=16, beam_size=1, verbose=False)`
- 调用方先用 build_env + build_modules + reload_model 初始化，再传入
- DiT 用 load_checkpoint 加载
- 测试：构造简单数据 → 端到端推理 → 验证返回非空表达式和有限 R^2

</specifics>

<deferred>
## Deferred Ideas

- Beam search 支持（v2 feature）
- 多候选表达式排序
- wandb 日志

</deferred>

---

*Phase: 06-pipeline*
*Context gathered: 2026-06-09*
