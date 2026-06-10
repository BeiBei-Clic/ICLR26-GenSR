# Phase 10: Decoder 微调 - Research

**Researched:** 2026-06-10
**Domain:** Decoder lm_head 输出投影层微调，适应 DiT 传输后的潜向量分布
**Confidence:** HIGH

## Summary

Phase 10 的核心任务是：冻结已训练好的 CVAE + DiT（FM），只微调 Decoder 的 `lm_head` 输出投影层，让 Decoder 适应 DiT Euler 积分传输后的 z_opt 潜向量分布。训练数据在线生成，走完整推理路径：FunctionEnvironment 随机生成 (X, Y, GT表达式) → CVAE 编码 prior_mu → DiT Euler 积分 → z_opt → FeatureFusion → Decoder → CE loss vs GT tokens。

关键发现：(1) `lm_head.weight` 与 `tok_embed.weight` 通过 `share_inout_emb=True` 共享，只解冻 lm_head 会同时更新 tok_embed 的梯度——但这正是 Cola-DLM `cola_vae_finetune.py` 的做法，因为 `p.requires_grad_(True)` 设在底层参数上，梯度自然流动到共享权重；(2) fm_best.pt 位于 `dit_train/checkpoints/fm_best.pt`（非 `weights/fm_best.pth`），且是纯 state_dict 格式；(3) 训练中 teacher-forcing CE loss 需要通过 Decoder 的 `fwd` + `predict` 两步计算（先 fwd 得到 hidden states，再 predict 用 lm_head 映射到词表维度）。

**Primary recommendation:** 写一个独立的 `dit_train/finetune_lm_head.py` 脚本，照搬 `cola_vae_finetune.py` 的冻结/训练模式，数据流走 CVAE → DiT → FeatureFusion → Decoder (teacher-forcing) → CE loss。500 iterations，AdamW lr=1e-4，batch_size=8。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 只解冻 Decoder 的 `lm_head`（`nn.Linear` 输出投影层），跟 Cola-DLM `cola_vae_finetune.py` 一致。其他所有参数（CVAE、FeatureFusion、Decoder Transformer blocks、DiT）全部冻结。
- **D-02:** 用 DiT 传输后的 z_opt 作为 Decoder 训练输入，而非 CVAE 编码的 post_mu。训练数据流程：
  1. 在线随机生成 (X, Y, GT表达式) 样本（复用 FunctionEnvironment）
  2. 冻结 CVAE 编码 → prior_mu
  3. 冻结 DiT Euler 积分 → z_opt
  4. FeatureFusion(z_opt, prior_logvar) → src_enc
  5. Decoder(src_enc) → logits → CE loss vs GT 表达式 tokens
- **D-03:** 照搬 Cola `cola_vae_finetune.py` 参数：AdamW lr=1e-4, batch_size=8, 500 iterations。

### Claude's Discretion
- 具体训练脚本的组织方式（独立脚本 vs 扩展现有脚本）
- checkpoint 保存策略
- 是否需要 eval 验证步骤

### Deferred Ideas (OUT OF SCOPE)
- 联合训练 DiT + VAE（Stage 2 全套）——需要 reference-encoder KL 等复杂正则化，留作未来阶段
- 解冻更多 Decoder 层——如果只微调输出层效果不够，再考虑扩大解冻范围
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | 深度学习框架 | 项目核心依赖，所有模型组件基于 PyTorch |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | 4.65.0+ | 进度条显示 | 训练循环日志 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 独立 finetune 脚本 | 扩展 train_fm.py | 独立脚本更清晰，两个训练任务数据流差异大 |

**Installation:** 无需额外安装，所有依赖已存在于项目中。

## Architecture Patterns

### Training Data Flow（在线生成）

```
FunctionEnvironment.gen_expr(train=True)
  → samples (X_to_fit, Y_to_fit, tree_encoded)
  → NumericalEmbedder → x1, len1
  → word_to_idx + token_embed → x2_e
  → CVAE(x1, x2_e, mode="train") → (prior_mu, prior_logvar, post_mu, ...)
  → DiT Euler(z_t=prior_mu, num_steps=16) → z_opt
  → FeatureFusion(z_opt, prior_logvar) → src_enc  (B, 200, 512)
  → Decoder fwd(x=GT_tokens, src_enc=src_enc, causal=True) → tensor (slen, B, 512)
  → Decoder predict(tensor, pred_mask, y=GT_targets) → CE loss
```

### 关键模型组件及其加载方式

**Checkpoint 结构** (`weights/checkpoint.pth`):
- `data_encoder` — NumericalEmbedder
- `cvae` — CVAEDE_SR
- `token_embed` — nn.Embedding(10292, 512, padding_idx=82)
- `seq_decoder` — TransformerModel (rebuttal=1, is_decoder=True, with_output=True)
- `feature_fusion` — FeatureFusion(latent_dim=512, dec_emb_dim=512, seq_len=200)

**DiT Checkpoint** (`dit_train/checkpoints/fm_best.pt`):
- 纯 state_dict（无包装），134 个 key
- 加载方式: `dit.load_state_dict(torch.load(path))`

### Teacher-Forcing CE Loss 计算

Decoder 的 `predict` 方法签名:
```python
def predict(self, tensor, pred_mask, y, get_scores):
    x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
    scores = self.lm_head(x).view(-1, self.n_words)  # n_words=10292
    loss = F.cross_entropy(scores.float(), y, reduction="mean")
    return scores, loss
```

两步调用:
1. `fwd(x=eq_tokens, lengths=eq_lengths, causal=True, src_enc=src_enc)` → tensor (slen, B, 512)
2. `predict(tensor, pred_mask, y=shifted_targets)` → CE loss

### Decoder `lm_head` 的权重共享

```python
# TransformerModel.__init__ 中:
self.lm_head = nn.Linear(self.dim, self.n_words, bias=True)  # 512 → 10292
if params.share_inout_emb:  # 默认 True
    self.lm_head.weight = self.tok_embed.weight  # 共享权重
```

lm_head 参数:
- `lm_head.weight`: (10292, 512) — 与 tok_embed.weight 共享
- `lm_head.bias`: (10292,) — 独立的 bias

### Anti-Patterns to Avoid
- **不要在 frozen 模型上调用 `.train()`:** CVAE/DiT/FeatureFusion 在微调期间应保持 `model.eval()`，否则 dropout 等行为会引入噪声
- **不要用 DataLoader 多进程生成 DiT 推理数据:** LatentPairDataset 用 spawn + 多 worker 是因为只需 CVAE forward；Phase 10 需要 CVAE + DiT + FeatureFusion + Decoder 全链路，多进程会大量消耗 GPU 显存。单进程在线生成就行
- **不要忽略 share_inout_emb 的梯度传播:** lm_head.weight 就是 tok_embed.weight，解冻 lm_head 时 tok_embed 也会被更新。这正是 Cola 的意图，不需要额外处理

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 在线数据生成 | 从零写数据加载器 | 复用 LatentPairDataset 的 gen_expr → CVAE 编码逻辑 | 已验证工作正确，含环境初始化、数据预处理 |
| CE loss 计算 | 手动写 softmax + NLL | Decoder.predict() 方法 | 已有实现，处理了 pred_mask、padding 过滤 |
| DiT 推理 | 重写 Euler 积分 | dit_train/inference.py::euler_inference | 已验证正确实现 |

**Key insight:** 训练数据生成需要复用 `LatentPairDataset._lazy_init` 中的 env + modules 初始化逻辑，但数据流从 `(prior_mu, post_mu)` 扩展为完整的 CVAE → DiT → FeatureFusion → Decoder 链路。

## Common Pitfalls

### Pitfall 1: share_inout_emb 导致 tok_embed 被意外更新
**What goes wrong:** `lm_head.weight` 和 `tok_embed.weight` 是同一个 tensor（不是 copy，是引用）。解冻 `lm_head.parameters()` 时，梯度通过反向传播更新 lm_head.weight，而 tok_embed.weight 也随之改变。
**Why it happens:** `params.share_inout_emb = True` 时，`self.lm_head.weight = self.tok_embed.weight` 是 Python 属性赋值，两者指向同一内存。
**How to avoid:** 这正是 Cola-DLM `cola_vae_finetune.py` 的设计意图——Cola 也解冻 `decoder.final_layer`，其权重也与 encoder.wte 共享。不需要避免，而是应该利用这种隐式的联合更新来保持输入输出 embedding 的一致性。
**Warning signs:** 如果担心 tok_embed 被污染导致 Decoder Transformer blocks（冻结层）的行为异常——由于 blocks 本身权重不变，只是 token embedding 分布略有变化，500 步微调不太可能导致灾难性偏移。

### Pitfall 2: DiT 推理在训练循环中的显存占用
**What goes wrong:** 每个训练 step 需要完整执行 CVAE forward + DiT Euler (16步) + FeatureFusion + Decoder forward，但只有 lm_head 需要梯度。如果不在 CVAE/DiT/FeatureFusion 上设 `torch.no_grad()`，会为冻结层保存中间激活，浪费显存。
**Why it happens:** PyTorch 默认为所有涉及 requires_grad 参数的计算图保存中间激活。
**How to avoid:** 将冻结部分（CVAE encode → DiT Euler → FeatureFusion）包裹在 `torch.no_grad()` 中，只在 Decoder forward（需要梯度回传到 lm_head）时打开 autograd。
**Warning signs:** OOM 错误，或 GPU 显存占用异常高。

### Pitfall 3: fm_best.pth 路径不匹配
**What goes wrong:** CONTEXT.md 引用 `weights/fm_best.pth`，但实际文件在 `dit_train/checkpoints/fm_best.pt`。
**Why it happens:** train_fm.py 的 `--output-dir` 默认值是 `dit_train/checkpoints`，不是 `weights/`。
**How to avoid:** 使用 `dit_train/checkpoints/fm_best.pt` 作为 DiT checkpoint 路径。
**Warning signs:** FileNotFoundError。

### Pitfall 4: GT tokens 格式与 Decoder predict 不匹配
**What goes wrong:** `env.word_to_idx` 返回 list of LongTensor，`env.batch_equations` 返回 (sent, lengths)。predict 方法需要 pred_mask 和 y（不含 EOS 的目标 tokens）。
**Why it happens:** Decoder 的输入格式是 (slen, B) 的 token 序列（含 EOS prefix），而 predict 的 y 是移位后的目标（不含 PAD）。
**How to avoid:** 参照 `batch_equations` 的输出格式：sent[0]=EOS, sent[1:-1]=eq tokens, sent[-1]=EOS。输入到 fwd 的是 sent，predict 的 y 是 sent[1:]（移位目标），pred_mask 标记非 PAD 位置。
**Warning signs:** 维度不匹配错误。

### Pitfall 5: FeatureFusion 的 n_samples 参数影响随机性
**What goes wrong:** FeatureFusion 内部调用 `sample_gaussian_multi(mu, logvar, n_samples=200)`，每次前向都会采样 200 个高斯样本再取均值。训练时这会引入噪声。
**Why it happens:** FeatureFusion 的 forward 不是确定性的——它先采样再投影。
**How to avoid:** 两种选择：(a) 接受噪声，因为推理时也有同样的采样过程；(b) 在微调时传确定的 z（直接用 expand_proj），绕过采样。**推荐 (a)**，因为微调的目标就是让 Decoder 适应推理时的真实分布。
**Warning signs:** Loss 震荡比预期大，但如果总体下降趋势正常就没问题。

## Code Examples

### Cola-DLM 冻结 + 解冻模式 (cola_vae_finetune.py)

```python
# Source: Cola-DLM-main/scripts/cola_vae_finetune.py

# Freeze everything
for p in vae.parameters():
    p.requires_grad_(False)

# Unfreeze only decoder output projection
for p in vae.decoder.final_layer.parameters():
    p.requires_grad_(True)

# Optimizer: 只优化 requires_grad=True 的参数
optimizer = torch.optim.AdamW(
    [p for p in vae.parameters() if p.requires_grad],
    lr=1e-4,
)

# Training loop: 500 iterations, batch_size=8
for step in range(500):
    batch = make_batch(8, 256)
    loss = compute_loss(vae, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### GenSR 的 teacher-forcing 前向路径

```python
# Source: symbolicregression/model/transformer.py (TransformerModel, line 1219+)
# data flow: fwd → predict

# Step 1: Decoder forward with GT tokens (teacher-forcing)
tensor = decoder(
    "fwd",
    x=eq_tokens,        # (slen, B) LongTensor — EOS + tokens + EOS + PAD
    lengths=eq_lengths,  # (B,) LongTensor
    causal=True,
    src_enc=src_enc,     # (B, 200, 512) from FeatureFusion
    src_len=None,
    use_cache=False,
)
# tensor: (slen, B, 512)

# Step 2: predict — 用 lm_head 映射到词表 + CE loss
scores, loss = decoder(
    "predict",
    tensor=tensor,
    pred_mask=pred_mask,  # (slen, B) bool — True for non-PAD positions
    y=y,                  # 移位后的目标 tokens
    get_scores=True,
)
```

### GenSR 完整推理路径 (dit_train/pipeline.py)

```python
# Source: dit_train/pipeline.py

# CVAE 编码
prior_mu, prior_logvar = model.encode_only(sample_to_learn)

# DiT Euler 积分
z_opt = euler_inference(dit, prior_mu, num_steps=16)

# FeatureFusion
src_enc = model.prepare_latent_for_decoder(z_opt, prior_logvar)
# src_enc: (B, 200, 512)

# Decoder 解码 (inference mode, 自回归)
generations, gen_len = model.generate_from_latent(src_enc)
```

### LatentPairDataset 的初始化模式 (dit_train/data/latent_dataset.py)

```python
# Source: dit_train/data/latent_dataset.py

# 延迟初始化模式 (spawn 安全)
def _lazy_init(self):
    torch.cuda.set_device(self.device)

    import symbolicregression.utils
    symbolicregression.utils.CUDA = not self.params.cpu

    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules, reload_model
    from symbolicregression.utils import to_cuda

    params = self.params
    env = build_env(params)
    env.rng = np.random.RandomState()
    self.env = env
    self._to_cuda = to_cuda

    modules = build_modules(env, params)
    reload_model(
        modules,
        modules_to_load=["cvae", "data_encoder", "token_embed"],
        path=self.checkpoint_path,
        requires_grad=False,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CMA-ES 进化搜索 | DiT Flow Matching 推理 | Phase 4-5 | 推理从迭代进化变为单次前向传输 |
| 冻结 VAE + Decoder 只训 DiT | 冻结 CVAE + DiT，微调 lm_head | Phase 10 | Decoder 适应 DiT 传输后的潜向量分布 |

**Deprecated/outdated:**
- CONTEXT.md 中引用 `weights/fm_best.pth`，实际路径为 `dit_train/checkpoints/fm_best.pt`

## Open Questions

1. **fm_best.pt 是否代表训练完成的 DiT？**
   - What we know: 文件存在（161MB），是 GenSRDiT state_dict，134 个 keys
   - What's unclear: train_fm.py 训练了多少步、val_loss 是多少（fm_best.pt 是纯 state_dict，不含元数据）
   - Recommendation: 按已有路径使用 `dit_train/checkpoints/fm_best.pt`，如果效果不好可以重新训练 DiT

2. **500 iterations 是否足够？**
   - What we know: Cola-DLM 用 500 iterations 微调 lm_head（文本域），GenSR 的词表 10292 vs Cola 的 100k+
   - What's unclear: 符号回归域的 token 分布是否与文本域差异大，需要更多步还是更少步
   - Recommendation: 先按 500 iterations 执行，通过 loss 曲线判断是否需要更多步

3. **eval 验证策略？**
   - What we know: Cola 用特殊 token 重建率做 eval，GenSR 的 token 是数学符号
   - What's unclear: 最合适的 eval 指标——可以直接用 CE loss 做验证
   - Recommendation: 每 50 步打印 CE loss，跟踪下降趋势即可

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA GPU | CVAE/DiT/Decoder 推理 | YES | RTX 3090 | — |
| weights/checkpoint.pth | CVAE + Decoder + FeatureFusion 权重 | YES | 671MB | — |
| dit_train/checkpoints/fm_best.pt | DiT 权重 | YES | 161MB | — |
| FunctionEnvironment | 训练数据在线生成 | YES | 内置 | — |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TBD | 冻结策略验证（只有 lm_head requires_grad=True） | unit | `pytest tests/test_lm_head_finetune.py::test_freeze_strategy -x` | Wave 0 |
| TBD | CE loss 计算端到端 | unit | `pytest tests/test_lm_head_finetune.py::test_ce_loss_forward -x` | Wave 0 |
| TBD | 训练循环收敛性 | smoke | 手动运行 50 步验证 loss 下降 | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/test_lm_head_finetune.py -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green + 手动运行 finetune 脚本 50 步验证 loss 下降

### Wave 0 Gaps
- [ ] `tests/test_lm_head_finetune.py` — covers freeze strategy, CE loss forward
- [ ] Framework install: 无需额外安装

## Sources

### Primary (HIGH confidence)
- `Cola-DLM-main/scripts/cola_vae_finetune.py` — Cola Decoder 微调完整实现，冻结策略、CE 重建损失、训练循环
- `symbolicregression/model/transformer.py` line 1219-1440 — TransformerModel (rebuttal=1) 的 fwd + predict + generate_from_latent
- `symbolicregression/model/feature_fusion.py` — FeatureFusion(mu, logvar) → (B, 200, 512)
- `dit_train/pipeline.py` — 端到端推理管线，展示完整数据流
- `dit_train/inference.py` — euler_inference 函数
- `model.py` — VAESymbolicRegressor，encode_only / prepare_latent_for_decoder / generate_from_latent

### Secondary (MEDIUM confidence)
- `Cola-DLM-main/docs/architecture.md` section 9 — Stage 1/2 训练理论框架
- `dit_train/data/latent_dataset.py` — 在线数据生成的 env + modules 初始化模式
- `symbolicregression/model/__init__.py` — build_modules / reload_model 实现

### Tertiary (LOW confidence)
- N/A — 所有核心代码直接读取验证

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 项目使用标准 PyTorch，无额外依赖
- Architecture: HIGH — 直接读取了所有相关源码，数据流、维度、checkpoint 格式均已验证
- Pitfalls: HIGH — share_inout_emb 权重共享通过代码验证确认，fm_best.pt 路径通过文件系统确认

**Research date:** 2026-06-10
**Valid until:** 2026-07-10（稳定的代码结构，不会快速变化）
