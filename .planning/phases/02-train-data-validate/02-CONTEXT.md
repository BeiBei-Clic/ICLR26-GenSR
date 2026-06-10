# Phase 2: 训练数据验证 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

验证 LatentPairDataset 生成的 (prior_mu, post_mu) 训练对的分布质量。扩展 Phase 1 已有的 verify_dataset.py，增加分布可视化、KL 散度统计、基础统计检查和自动通过/不通过判断。

**范围内：**
- 扩展 dit_train/data/verify_dataset.py 增加完整验证功能
- 生成 prior_mu 和 post_mu 的分布可视化图（矢量图 PDF/SVG）
- 计算 KL 散度统计量检测 posterior collapse
- 基础统计检查（mean, std, NaN/Inf）
- 自动 PASS/FAIL 判断并输出

**范围外：**
- DiT 模型定义或训练（Phase 3-4）
- 推理管线（Phase 5-6）
- 评估对比实验（Phase 7-8）

</domain>

<decisions>
## Implementation Decisions

### 验证方式
- **D-01:** 扩展 Phase 1 已有的 `dit_train/data/verify_dataset.py`，不创建新文件。
- **D-02:** 验证时收集 1000 个 (prior_mu, post_mu) 样本做统计分析。

### 验证内容
- **D-03:** 四项验证全部实施：
  1. 分布可视化图（prior_mu/post_mu 各维度直方图/KDE，diff norm 分布）— 矢量图保存
  2. KL 散度统计（检测 posterior collapse）
  3. 基础统计检查（mean, std, NaN/Inf 检测）
  4. 自动 PASS/FAIL 判断

### 通过/不通过标准
- **D-04:** 自动判定，条件：
  - KL 散度均值 > 0.01（posterior 与 prior 有显著差异，非 posterior collapse）
  - diff norm 均值 > 0.1（prior 与 post 有实际差距，Flow Matching 有东西可学）
  - 无 NaN/Inf（数值稳定性）
- **D-05:** 阈值由脚本内部定义（基于训练时 kl_limits=0.2 推算），不通过命令行参数配置。

### 可视化
- **D-06:** 使用矢量图格式（PDF 或 SVG），符合 CLAUDE.md 要求"导出的图片一律使用矢量图，禁止使用位图"。

### Claude's Discretion
- 具体可视化图的布局和数量
- KL 散度的计算方式（逐维度 vs 整体）
- 矢量图保存路径和命名
- 脚本的输出格式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 交付物
- `dit_train/data/latent_dataset.py` — LatentPairDataset + create_latent_dataloader
- `dit_train/data/verify_dataset.py` — Phase 1 的基础验证脚本（要被扩展）
- `tests/test_latent_dataset.py` — 现有测试

### CVAE 训练参数
- `parsers.py` — kl_limits 默认 0.2，影响 posterior collapse 阈值
- `symbolicregression/trainer_vae.py` — 训练循环中 KL 退火的实现

### Pitfall 参考
- `.planning/research/PITFALLS.md` — Pitfall 3: posterior collapse 风险分析

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `verify_dataset.py`: 已有数据收集逻辑（从 DataLoader 收集 batch），只需扩展统计和可视化
- `LatentPairDataset` + `create_latent_dataloader`: Phase 1 交付的数据生成器

### Established Patterns
- 验证脚本通过 `python -m dit_train.data.verify_dataset` 运行
- 使用 matplotlib 做可视化（CLAUDE.md 要求矢量图）
- 参数通过 `get_parser()` 获取

</code_context>

<specifics>
## Specific Ideas

无特殊要求。

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-train-data-validate*
*Context gathered: 2026-06-09*
