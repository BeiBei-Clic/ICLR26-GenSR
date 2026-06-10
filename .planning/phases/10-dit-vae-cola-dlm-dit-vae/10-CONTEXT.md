# Phase 10: Decoder 微调 - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

冻结已训练好的 CVAE + DiT（FM），只微调 Decoder 的输出投影层（`lm_head`），让 Decoder 适应 DiT 传输后的潜向量分布，提升表达式解码质量。

前置条件：VAE 预训练完成（`weights/checkpoint.pth`）、DiT FM 训练完成（`weights/fm_best.pth`）。

</domain>

<decisions>
## Implementation Decisions

### Decoder 解冻范围
- **D-01:** 只解冻 Decoder 的 `lm_head`（`nn.Linear` 输出投影层），跟 Cola-DLM `cola_vae_finetune.py` 一致。其他所有参数（CVAE、FeatureFusion、Decoder Transformer blocks、DiT）全部冻结。

### 训练数据来源
- **D-02:** 用 DiT 传输后的 z_opt 作为 Decoder 训练输入，而非 CVAE 编码的 post_mu。训练数据流程：
  1. 在线随机生成 (X, Y, GT表达式) 样本（复用 FunctionEnvironment）
  2. 冻结 CVAE 编码 → prior_mu
  3. 冻结 DiT Euler 积分 → z_opt
  4. FeatureFusion(z_opt, prior_logvar) → src_enc
  5. Decoder(src_enc) → logits → CE loss vs GT 表达式 tokens

### 训练策略
- **D-03:** 照搬 Cola `cola_vae_finetune.py` 参数：AdamW lr=1e-4, batch_size=8, 500 iterations。

### Claude's Discretion
- 具体训练脚本的组织方式（独立脚本 vs 扩展现有脚本）
- checkpoint 保存策略
- 是否需要 eval 验证步骤

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cola-DLM VAE 微调原型
- `Cola-DLM-main/scripts/cola_vae_finetune.py` — Cola Decoder 微调的完整实现：冻结策略、CE 重建损失、训练循环
- `Cola-DLM-main/docs/architecture.md` §9 — Stage 1/2 训练理论框架，损失函数公式

### GenSR 现有训练代码
- `dit_train/train_fm.py` — 现有 FM 训练循环，包含 LatentPairDataset 使用方式
- `dit_train/data/latent_pair_dataset.py` — LatentPairDataset 在线生成 (prior_mu, post_mu) 对
- `dit_train/inference/euler_inference.py` — Euler 积分推理函数，用于生成 z_opt
- `symbolicregression/model/transformer.py` — TransformerModel_VAE Decoder，`lm_head` 输出层定义
- `symbolicregression/model/feature_fusion.py` — FeatureFusion 将 (mu, logvar) 转为 Decoder 输入
- `symbolicregression/model/cvae.py` — CVAE 编码器

### 推理管线参考
- `dit_train/inference/dit_inference.py` — 端到端推理管线，展示 CVAE → DiT → FeatureFusion → Decoder 完整流程

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LatentPairDataset`: 在线生成训练数据，可复用其 FunctionEnvironment 和 CVAE 编码逻辑
- `euler_inference`: 已实现 Euler 积分，可直接调用生成 z_opt
- `dit_inference`: 端到端管线中 FeatureFusion → Decoder 的调用方式可直接参考

### Established Patterns
- 原生 PyTorch 训练循环（不用 PyTorch Lightning）— Phase 4 D-01
- CVAE 冻结模式：`param.requires_grad = False` + `model.eval()` + `torch.no_grad()`
- 在线数据生成（不保存文件）— Phase 1 D-01

### Integration Points
- 训练脚本需同时加载 CVAE checkpoint + DiT FM checkpoint + Decoder
- Decoder `lm_head` 的输出维度为 `n_words`（词表大小）
- FeatureFusion 输入格式：(mu, logvar)，输出 (B, max_len, 512) 给 Decoder

</code_context>

<specifics>
## Specific Ideas

- 跟 Cola-DLM 原型代码 `cola_vae_finetune.py` 保持一致的冻结和训练策略
- 训练数据走完整推理路径（CVAE → DiT → Decoder），让 Decoder 适应实际推理时的输入分布

</specifics>

<deferred>
## Deferred Ideas

- 联合训练 DiT + VAE（Stage 2 全套）——需要 reference-encoder KL 等复杂正则化，留作未来阶段
- 解冻更多 Decoder 层——如果只微调输出层效果不够，再考虑扩大解冻范围

</deferred>

---

*Phase: 10-dit-vae-cola-dlm-dit-vae*
*Context gathered: 2026-06-10*
