# Phase 7: 评估对比实验 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

在 PMLB 全部回归数据集上运行 DiT 推理管线，与已有 CMA-ES 基线做定量对比。输出每个数据集的 R^2、表达式复杂度、推理时间，并统计成功率。

**范围内：**
- 新建评估脚本，调用 dit_inference() 对全部 PMLB 回归数据集做推理
- 复用已有 CMA-ES 基线结果 CSV（experiments/pmlb/GenSR_results/）
- 对比指标：R^2、复杂度、推理时间、成功率（R^2 > 0.99）
- 结果保存为 CSV

**范围外：**
- wandb 日志集成（Phase 8）
- 结果可视化与分析（Phase 8）
- 多步数消融实验（v2 feature）
- CFG 推理（v2 feature）

</domain>

<decisions>
## Implementation Decisions

### 评估数据集范围
- **D-01:** 覆盖全部 PMLB 回归数据集（约 271 个，含 Feynman、Strogatz、黑盒）。

### CMA-ES 基线
- **D-02:** 复用已有 CMA-ES 基线结果（experiments/pmlb/GenSR_results/pmlb_batch_inference_noise_0.csv），不重新跑。

### 评估脚本
- **D-03:** 新建评估脚本 dit_eval.py，放在 experiments/pmlb/ 下。复用 pmlb_batch_inference.py 的 CSV 格式和评估框架，但直接调用 dit_inference()。结果放在 experiments/pmlb/GenSR_dit/。

### DiT 推理配置
- **D-04:** 每个数据集跑一次 dit_inference()，num_steps=16，单次推理取最优。

### Claude's Discretion
- 评估脚本的具体参数和 CLI 接口
- 对比结果的具体输出格式
- 数据集加载和预处理的细节
- 断点续传是否需要

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### DiT 推理管线（Phase 6 交付）
- `dit_train/pipeline.py` — dit_inference(X, y, env, params, model, dit, num_steps=16)
- `dit_train/inference.py` — euler_inference() Euler 积分核心
- `dit_train/train_fm.py:load_checkpoint()` — 加载 DiT checkpoint

### 现有评估框架（复用参考）
- `experiments/pmlb/pmlb_batch_inference.py` — CMA-ES 批量推理脚本（复用 CSV 格式和框架）
- `LSO_eval.py` — evaluate_pmlb_lso() 评估主函数
- `LSO_eval.py:read_file()` — PMLB 数据集加载
- `symbolicregression/metrics.py` — compute_metrics() 指标计算

### 已有 CMA-ES 基线结果
- `experiments/pmlb/GenSR_results/pmlb_batch_inference_noise_0.csv` — CMA-ES 基线（240 数据集）

### 模型初始化模板
- `dit_train/data/latent_dataset.py` — CVAE 编码参考（如何初始化 params 和模块）
- `model.py:reload_model()` — 模型加载

### 项目文档
- `.planning/phases/06-pipeline/06-CONTEXT.md` — Phase 6 管线决策

</canonical_refs>

<code_context>
## Existing Code Insights

### 已有评估脚本结构（pmlb_batch_inference.py）
- CLI 入口，支持 --gpu 选择 GPU
- 遍历 datasets/pmlb/datasets/ 下所有数据集
- read_file() 加载数据 → train_test_split → 构造 sample_to_learn → 调用推理 → 记录结果
- CSV 输出：dataset, status, n_features, refinement_type, r2, rmse, complexity, seconds, error, noise_strength, expr
- 支持断点续传（检查已有 CSV 中 dataset 是否已评估）

### DiT 推理调用方式
```python
from dit_train.pipeline import dit_inference
result = dit_inference(X, y, env, params, model, dit, num_steps=16)
# result: {"success": bool, "expression": str, "r2": float, "complexity": int, "tree": object}
```

### 模型初始化（参照 latent_dataset.py）
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

### DiT 加载
```python
from dit_train.model import GenSRDiT
from dit_train.train_fm import load_checkpoint

dit = GenSRDiT().to("cuda")
step, val_loss, ckpt = load_checkpoint(dit, "weights/fm_best.pth")
```

### 指标计算（已有）
- R^2: `compute_metrics(infos, metrics="r2")` → r2_score
- 复杂度: 表达式树 prefix().split(",") 的长度
- 推理时间: time.time() 差值

</code_context>

<specifics>
## Specific Ideas

- 新建 `experiments/pmlb/dit_eval.py`
- 复用 pmlb_batch_inference.py 的 CSV 格式（dataset, status, r2, complexity, seconds, expr 等）
- 结果放在 `experiments/pmlb/GenSR_dit/pmlb_dit_results.csv`
- 评估循环：初始化模型 → 遍历数据集 → dit_inference() → 记录结果
- 对比脚本：读取两个 CSV → 按数据集名 join → 输出对比表

</specifics>

<deferred>
## Deferred Ideas

- 多步数消融实验（8/16/32 步对比）
- 多次采样取最优
- wandb 日志集成（Phase 8）
- 结果可视化（Phase 8）
- 噪声鲁棒性实验

</deferred>

---

*Phase: 07-eval*
*Context gathered: 2026-06-09*
