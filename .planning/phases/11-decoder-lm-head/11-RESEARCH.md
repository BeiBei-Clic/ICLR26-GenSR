# Phase 11: Decoder lm_head 微调验证循环 - Research

**Researched:** 2026-06-10
**Domain:** 训练循环验证逻辑（PyTorch 手写训练循环）
**Confidence:** HIGH

## Summary

Phase 11 需要在现有的 `finetune_lm_head.py` 训练脚本中添加验证循环。当前脚本已有基于 training loss 的 `best_loss` 跟踪和 `lm_head_best.pt` 保存，但缺少在独立验证集上的评估。研究目标是在每 50 步时用 `env.gen_expr(train=False)` 在线生成验证数据，计算验证 CE loss，跟踪 `best_val_loss`，并在验证 loss 改善时自动保存最优权重。

核心参考模式来自 `train_fm.py` 的验证循环（`evaluate()` 函数 + `best_val_loss` 跟踪）和 `cola_vae_finetune.py` 的 `eval_every=50` 验证频率。需要改造的关键点：将现有的 `best_loss`（基于 training loss）替换为 `best_val_loss`（基于 validation loss），并在训练循环中每 50 步插入验证数据生成 + CE loss 计算。

**Primary recommendation:** 参照 `train_fm.py` 的 `evaluate()` 模式，在 `finetune_lm_head.py` 中添加一个内联验证块（不需抽取函数），每 50 步生成验证数据、计算 val loss、更新 `best_val_loss` 并保存最优权重。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 验证数据用 `gen_expr(train=False)` 在线生成，不做预生成缓存。跟训练数据生成方式一致（只是 train=False），简单直接。
- **D-02:** 每 50 步验证一次，跟 `train_fm.py` 的验证频率一致。

### Claude's Discretion
- 验证时的 batch_size（可以用跟训练相同的 8）
- 验证 loss 的计算方式（逐样本 CE loss 平均，跟训练一致）
- 验证结果日志格式

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-11-01 | 每 50 步在验证集上评估 CE loss | 用 `gen_expr(train=False)` 生成验证数据，复用训练循环中已有的逐样本 CE loss 计算逻辑（L155-188），加 `torch.no_grad()` 包裹 |
| REQ-11-02 | 跟踪 best val loss | 参照 `train_fm.py` L312-317 的 `best_val_loss` 跟踪模式，将现有 `best_loss`（training loss）替换为 `best_val_loss`（validation loss） |
| REQ-11-03 | 自动保存最优权重 | 参照 `train_fm.py` L314-317 的保存模式，当 val loss < best_val_loss 时保存 `lm_head_best.pt`。当前脚本 L202-205 已有类似逻辑但基于 training loss |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | 训练循环、梯度计算、模型保存 | 项目标准框架 |
| NumPy | 1.24.0+ | 随机数种子管理 | 项目标准依赖 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | 4.65.0+ | 进度条 | 当前脚本已导入但未使用，验证阶段无需新增依赖 |

**Installation:** 无需新增依赖，所有需要的包已在项目中。

## Architecture Patterns

### 现有训练循环结构 (finetune_lm_head.py)

```
main()
  ├── 1. 初始化 env + modules (L42-69)
  ├── 2. 加载 DiT (L72-78)
  ├── 3. 冻结策略 (L80-98)
  ├── 4. Optimizer (L101-105)
  ├── 5. 训练循环 (L107-211)
  │     for step in range(num_iterations):
  │       ├── 在线生成训练数据 gen_expr(train=True) (L120-153)
  │       ├── 逐样本 CE loss 计算 (L155-188)
  │       ├── backward + optimizer step (L193-196)
  │       ├── 保存 best checkpoint (基于 training loss) (L200-205)
  │       └── 日志打印 (L207-208)
  └── 完成信息打印 (L210-211)
```

### 推荐的验证循环插入位置

```python
# 在训练循环内，optimizer step 之后、日志打印之前插入
for step in range(args.num_iterations):
    # ... 现有训练代码 (L117-196) ...

    # ===== 验证循环 (新增) =====
    eval_every = 50
    do_eval = (step + 1) % eval_every == 0 or step == 0
    if do_eval:
        decoder.eval()  # lm_head 进入 eval 模式
        with torch.no_grad():
            # 用 gen_expr(train=False) 生成验证数据
            val_src_enc_list = []
            val_x2_list = []
            val_len2_list = []
            for _ in range(args.batch_size):
                # ... 同训练数据生成，但 train=False ...

            # 逐样本计算 CE loss
            val_total_loss = 0.0
            val_count = 0
            for i in range(args.batch_size):
                # ... 同训练的 CE loss 计算 ...

            val_loss = val_total_loss / val_count

        decoder.train()  # 恢复训练模式

        # 更新 best_val_loss 并保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.lm_head.state_dict(),
                       os.path.join(args.output_dir, "lm_head_best.pt"))

        print(f"step {step:04d} | val_loss: {val_loss:.4f} | best_val: {best_val_loss:.4f}")
    # ===== 验证循环结束 =====
```

### Pattern: train_fm.py 的验证循环 (参考)

```python
# train_fm.py L306-319
do_eval = step == 0 or last_step or (args.eval_every > 0 and step % args.eval_every == 0)
if do_eval:
    val_loss = evaluate(dit_raw, data_iter, args.eval_steps)
    if is_master:
        print(f"Step {step:05d} | Val FM loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "fm_best.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(dit_raw.state_dict(), best_path)
            print(f"New best val loss: {best_val_loss:.6f} -> {best_path}")
```

### Pattern: cola_vae_finetune.py 的验证循环 (参考)

```python
# cola_vae_finetune.py L188-193
if step == 0 or (step + 1) % args.eval_every == 0:
    vae.eval()
    correct, total, sp_correct, sp_total = eval_special_tokens(vae)
    print(f"  Eval: {correct}/{total} tokens correct ...")
    vae.train()
```

### Anti-Patterns to Avoid
- **不要将 `best_loss` 和 `best_val_loss` 混用:** 当前脚本的 `best_loss`（L113, L202-205）基于 training loss。添加验证循环后，应该用 `best_val_loss` 替换 `best_loss` 来判断是否保存最优权重。training loss 波动大，不适合做保存决策。
- **不要在验证时忘记 `decoder.train()` 恢复:** 虽然 lm_head 没有 dropout/BN，但保持 train/eval 切换一致性是好习惯。
- **不要在验证时计算梯度:** 必须用 `torch.no_grad()` 包裹验证块，避免不必要的显存消耗。

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 无 | 本阶段无新库需求 | — | 所有需要的逻辑已在训练循环中存在，只需复制+修改 |

**Key insight:** 验证数据生成和 CE loss 计算逻辑与训练完全一致（只是 `train=False`），直接复制训练数据生成块即可，不需要抽象为函数。

## Common Pitfalls

### Pitfall 1: 验证数据生成遗漏 train=False
**What goes wrong:** 验证时仍然用 `gen_expr(train=True)`，导致验证 loss 与训练 loss 无差异，无法检测过拟合。
**Why it happens:** 复制训练数据生成代码后忘记修改参数。
**How to avoid:** 明确将 `train=False` 作为关键修改点，并在验证日志中标注 "val"。
**Warning signs:** val loss 和 train loss 几乎相同。

### Pitfall 2: 忘记 decoder.eval()/decoder.train() 切换
**What goes wrong:** 验证时 lm_head 仍处于 train 模式（虽然 lm_head 是线性层无 dropout，影响不大，但不符合规范）。
**Why it happens:** 参照 cola_vae_finetune.py 的模式，需要在验证前后切换。
**How to avoid:** 在验证块前后添加 `decoder.eval()` 和 `decoder.train()`。

### Pitfall 3: best_val_loss 初始值设为 0 而非 inf
**What goes wrong:** `best_val_loss = 0.0` 导致第一个验证 loss 永远不会被认为更好（CE loss > 0），永远不会保存最优权重。
**Why it happens:** 笔误或误解。
**How to avoid:** `best_val_loss = float("inf")`，与 `train_fm.py` L261 一致。

### Pitfall 4: 验证时 val_count 可能为 0
**What goes wrong:** 如果所有验证样本的 `y.numel() == 0`（极端情况），`val_loss = val_total_loss / 0` 会报错。
**Why it happens:** 与训练循环 L190-191 的 `if count == 0: continue` 相同的逻辑，但验证时如果跳过则无法计算 val_loss。
**How to avoid:** 在验证块中添加 `if val_count == 0: continue` 并跳过本次验证评估。

### Pitfall 5: eval_every 判断条件不一致
**What goes wrong:** 用 `step % 50 == 0` 时 step=0 也会触发验证，但此时模型还没训练，val_loss 无意义。用 `step % 50 == 49` 或 `(step + 1) % 50 == 0` 更合理。
**Why it happens:** 不同脚本的判断条件不同：`train_fm.py` 用 `step % eval_every == 0`（包含 step=0），`cola_vae_finetune.py` 用 `(step + 1) % eval_every == 0`（不包含 step=0 除非显式加 `step == 0`）。
**How to avoid:** 参照 `cola_vae_finetune.py` 的模式：`step == 0 or (step + 1) % 50 == 0`。或者简化为 `(step + 1) % 50 == 0`，只在训练了 50 步后才首次验证。

## Code Examples

### 现有训练数据生成逻辑 (finetune_lm_head.py L120-153)

```python
# 验证时复用此逻辑，只改 train=True -> train=False
with torch.no_grad():
    for _ in range(args.batch_size):
        samples, _ = env.gen_expr(train=True)  # <- 改为 False

        x_to_fit = samples["X_to_fit"]
        y_to_fit = samples["Y_to_fit"]
        x1 = [[[x, y] for x, y in zip(xs, ys)]
               for xs, ys in zip(x_to_fit, y_to_fit)]
        x1_single, len1_single = embedder_f(x1)

        x2_single, len2_single = env.batch_equations(
            env.word_to_idx([samples["tree_encoded"]], float_input=False)
        )
        x2_single, len2_single = to_cuda(x2_single, len2_single)
        x2_e_single = embedder_e(x2_single.transpose(0, 1)).transpose(0, 1)

        prior_mu, prior_logvar, _, _, _, _, _ = vae_model(
            x1_single, x2_e_single, len1_single, len2_single, mode="train"
        )

        z_opt = euler_inference(dit, prior_mu, num_steps=args.dit_num_steps)
        src_enc = feature_fusion(z_opt, prior_logvar)

        src_enc_list.append(src_enc)
        x2_list.append(x2_single)
        len2_list.append(len2_single)
```

### 现有逐样本 CE loss 计算 (finetune_lm_head.py L155-188)

```python
# 验证时复用此逻辑，但在 torch.no_grad() 下执行（不需要 backward）
total_loss = 0.0
count = 0
for i in range(args.batch_size):
    src_enc = src_enc_list[i]
    eq_tokens = x2_list[i]
    eq_len = len2_list[i]

    alen = torch.arange(params.max_src_len, dtype=torch.long, device=device)
    pred_mask = (alen[:, None] < eq_len[None] - 1)
    y = eq_tokens[1:].masked_select(pred_mask[:-1])

    if y.numel() == 0:
        continue

    tensor = decoder(
        "fwd", x=eq_tokens, lengths=eq_len, causal=True,
        src_enc=src_enc, src_len=None, use_cache=False,
    )
    scores, loss = decoder(
        "predict", tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True,
    )
    total_loss += loss
    count += 1
```

### train_fm.py best_val_loss 跟踪模式 (L312-317)

```python
# Source: dit_train/train_fm.py
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_path = os.path.join(args.output_dir, "fm_best.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dit_raw.state_dict(), best_path)
    print(f"New best val loss: {best_val_loss:.6f} -> {best_path}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| N/A | N/A | N/A | 本阶段是简单的训练循环修改，无 SOTA 变化 |

## Open Questions

1. **验证数据量是否足够？**
   - What we know: 每次验证用 batch_size=8 个样本计算 val loss
   - What's unclear: 8 个样本的 val loss 方差是否太大，导致 best_val_loss 判断不稳定
   - Recommendation: 先用 8 样本，如果 val loss 波动大再增加。与训练 batch_size 一致是合理的起点。

2. **是否需要替换现有的 best_loss 保存逻辑？**
   - What we know: 现有 L200-205 基于 training loss 保存 `lm_head_best.pt`
   - What's unclear: 是否应完全替换为 val_loss 触发保存
   - Recommendation: 完全替换。删除 L200-205 的 training loss 保存逻辑，改为只在验证时保存。避免保存基于 training loss 的非最优权重。

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | 手动运行验证（无独立测试框架） |
| Config file | none |
| Quick run command | `python dit_train/finetune_lm_head.py --num-iterations 100` |
| Full suite command | N/A（单脚本验证） |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-11-01 | 每 50 步验证 CE loss | smoke | `python dit_train/finetune_lm_head.py --num-iterations 100` | N/A |
| REQ-11-02 | 跟踪 best val loss | smoke | 同上，检查日志输出 | N/A |
| REQ-11-03 | 自动保存最优权重 | smoke | 同上，检查 `lm_head_best.pt` 文件 | N/A |

### Sampling Rate
- **Per task commit:** `python dit_train/finetune_lm_head.py --num-iterations 100`
- **Per wave merge:** N/A
- **Phase gate:** 运行 100 步确认验证循环正常工作

### Wave 0 Gaps
- None — 本阶段是对现有脚本的增量修改，无需额外测试基础设施

## Sources

### Primary (HIGH confidence)
- `dit_train/finetune_lm_head.py` — 完整的训练脚本，需添加验证循环的目标文件
- `dit_train/train_fm.py` — 验证循环参考模式（evaluate + best_val_loss + save best）
- `Cola-DLM-main/scripts/cola_vae_finetune.py` — eval_every=50 验证频率参考

### Secondary (MEDIUM confidence)
- N/A

### Tertiary (LOW confidence)
- N/A

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 无新依赖，全部复用现有代码
- Architecture: HIGH — 验证循环模式已在 train_fm.py 和 cola_vae_finetune.py 中验证
- Pitfalls: HIGH — 基于 read 代码发现的实际问题

**Research date:** 2026-06-10
**Valid until:** 2026-07-10（稳定，训练循环模式不会变化）
