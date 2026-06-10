# Phase 1: 训练数据生成器 - Research

**Researched:** 2026-06-09
**Domain:** PyTorch DataLoader + GenSR CVAE 数据管线
**Confidence:** HIGH

## Summary

本阶段需要实现一个在线数据生成器，在 DiT 训练时实时生成 (prior_mu, post_mu) 训练对。核心管线是：FunctionEnvironment 在线生成随机表达式 + 数据点 -> NumericalEmbedder 编码数值数据 -> batch_equations 编码表达式 -> CVAE forward(mode="train") -> 输出 (prior_mu, post_mu)。

**关键发现：** GenSR 已有完整的数据生成管线，从 FunctionEnvironment.gen_expr() 到 CVAE forward(mode="train") 的每一步都可以直接复用。现有代码中 `trainer_vae.py:enc_dec_vae_step()` 方法包含了完整的端到端数据流，我们的 Dataset 只需要复刻这个流程的前半段（到 CVAE 输出 prior_mu/post_mu 为止）。

**Primary recommendation:** 使用 IterableDataset（因为数据无限生成、无固定长度），内部持有冻结的 CVAE + embedder + FunctionEnvironment，在 `__iter__` 中批量生成数据。DataLoader 的 num_workers 设为 0（因为 CVAE 在 GPU 上，多进程无法共享 GPU 张量）。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 在线生成，不保存到文件。使用 FunctionEnvironment 在线随机生成 (X, Y, GT表达式) 样本，立刻过冻结的 CVAE 提取 (prior_mu, post_mu)。每次调用生成新数据，无需持久化。
- **D-02:** 不固定随机种子，每次运行生成不同数据。
- **D-03:** Phase 1 交付的是一个 PyTorch 数据生成组件（Dataset 类），不是独立脚本。Phase 4 的 DiT 训练循环直接 import 这个组件使用。
- **D-04:** CVAE 使用 `weights/checkpoint.pth`（671MB），加载后完全冻结（requires_grad=False），不参与 DiT 训练。
- **D-05:** 生成器每次 __getitem__ 返回 (prior_mu, post_mu) 对，各为 (512,) 张量。不返回 logvar、原始表达式、X/Y 数据等。
- **D-06:** batch_size、num_workers 等通过构造参数配置，方便 DiT 训练时灵活调整。

### Claude's Discretion
- Dataset 类的具体实现方式（IterableDataset vs Map-style Dataset）
- 批次处理的具体逻辑（collate_fn 等）
- 是否需要 train/val split 及划分方式
- 设备管理（CVAE 在哪个设备上运行）
- 组件文件位置和命名
- 是否需要独立的验证脚本（供 Phase 2 使用）

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | 从现有 CVAE 训练集中提取 (prior_mu, post_mu) 对作为 Flow Matching 训练数据 | 通过复用 FunctionEnvironment.gen_expr() + CVAE forward(mode="train") 实现在线生成 |
| DATA-02 | 支持批量提取，处理整个训练数据集生成训练对文件（如 .pt 或 .h5） | 通过 PyTorch DataLoader 的 batch_size 参数控制批量，在线生成无需保存文件 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | Dataset/DataLoader 基础设施 | 项目已有，无额外依赖 |
| NumPy | 1.24.0+ | FunctionEnvironment 内部使用 | 已有依赖 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.data.IterableDataset | 2.10.0 | 无限数据流 Dataset | 在线生成场景，数据无固定长度 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| IterableDataset | Map-style Dataset (size=2^60) | Map-style 可复用现有 EnvDataset 模式，但 IterableDataset 语义更准确 |
| 在线生成 | 预生成 .pt 文件 | 用户决定在线生成 (D-01)，预生成速度快但占磁盘 |

**Installation:**
无额外安装。全部使用现有依赖。

## Architecture Patterns

### Recommended Project Structure
```
dit_train/                      # DiT 训练相关代码
├── data/                       # 数据生成组件
│   └── latent_dataset.py       # LatentPairDataset 类
└── ...
```

### Pattern 1: IterableDataset 在线生成

**What:** 使用 `torch.utils.data.IterableDataset` 包装完整的数据生成管线，每次迭代在线生成新的 (prior_mu, post_mu) 对。

**When to use:** 数据无限生成、无固定长度的场景。

**Example（核心流程伪代码）：**
```python
class LatentPairDataset(IterableDataset):
    def __init__(self, params, env, modules, checkpoint_path, batch_size, device):
        # 加载冻结的 CVAE
        # modules 包含 cvae, data_encoder, token_embed
        # 从 checkpoint_path 加载权重并冻结

    def __iter__(self):
        while True:
            # 1. env.gen_expr(train=True) 生成一个样本
            # 2. 构造 x1 (数值数据) 和 x2 (表达式 token)
            # 3. embedder_f(x1) -> 编码数值数据
            # 4. batch_equations + embedder_e -> 编码表达式
            # 5. cvae(x1, x2_e, len1, len2, mode="train") -> prior_mu, post_mu
            yield prior_mu, post_mu
```

### Pattern 2: 批量生成后逐个 yield

**What:** 为了效率，每次调用 gen_expr 生成 batch_size 个样本，批量通过 CVAE，然后逐个 yield 出来。

**When to use:** CVAE 前向传播有 GPU 开销，批量处理比逐个处理更高效。

### Anti-Patterns to Avoid
- **在 DataLoader worker 中运行 GPU 操作：** CVAE 在 GPU 上运行，DataLoader 的 num_workers > 0 时 worker 进程无法共享 CUDA context。必须设 num_workers=0，或在 Dataset.__iter__ 内部完成所有 GPU 操作。
- **在 Dataset 中保留梯度计算：** 整个 CVAE 冻结后应使用 `torch.no_grad()` 包裹前向传播，避免显存泄漏。
- **复用 trainer_vae.py 中的 collate 逻辑：** 现有 collate 逻辑处理的是原始样本字典，我们的 Dataset 直接返回张量对，需要简单的 default_collate。

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 数值数据编码 | 自己写 float encoding + padding | `NumericalEmbedder.forward()` | 内部处理了 float precision、padding、维度对齐 |
| 表达式编码 | 自己写 token encoding | `env.batch_equations()` + `embedder_e()` | 内部处理了 EOS、PAD、length 计算 |
| CVAE 权重加载 | 自己写 state_dict 加载 | `reload_model()` (symbolicregression/model/__init__.py) | 内部处理了 DDP wrapper key 前缀问题 |
| 随机表达式生成 | 自己写方程采样 | `FunctionEnvironment.gen_expr()` | 内部有完善的采样、重试、噪声注入 |
| batch padding | 自己写序列 padding | `env.batch_equations()` | 内部处理了变长序列对齐 |

**Key insight:** 现有 `enc_dec_vae_step()` 方法（trainer_vae.py:836-953）包含了完整的端到端数据流。我们的 Dataset 只需复刻到第 888 行 `vae_model(x1, x2_e, len1, len2, mode="train")` 为止，不需要后续的 feature_fusion、decoder 和 loss 计算。

## Common Pitfalls

### Pitfall 1: DataLoader num_workers 与 CUDA 冲突
**What goes wrong:** 设置 num_workers > 0 时，worker 子进程无法访问 CUDA，导致 RuntimeError。
**Why it happens:** PyTorch DataLoader 在独立进程中运行 worker，CUDA context 不能跨进程共享。
**How to avoid:** 设置 num_workers=0。由于 CVAE 前向传播在 Dataset.__iter__ 中完成，所有 GPU 操作必须在主进程。
**Warning signs:** RuntimeError: Cannot re-initialize CUDA in forked subprocess.

### Pitfall 2: 忘记 torch.no_grad() 导致显存泄漏
**What goes wrong:** CVAE 前向传播记录计算图，每个 batch 都消耗显存，训练久了 OOM。
**Why it happens:** 即使 requires_grad=False，不带 no_grad() 的前向传播仍会记录中间激活。
**How to avoid:** 在 `__iter__` 的 CVAE 前向传播外包裹 `with torch.no_grad():`。
**Warning signs:** 训练过程中显存持续增长。

### Pitfall 3: gen_expr 的 rng 未初始化
**What goes wrong:** FunctionEnvironment.rng 为 None 时调用 gen_expr() 会崩溃。
**Why it happens:** 现有 EnvDataset 在 worker_init_fn 中通过 init_rng() 初始化 rng，而我们的 IterableDataset 需要自行初始化。
**How to avoid:** 在 Dataset 构造时或 __iter__ 首次调用时初始化 `env.rng = np.random.RandomState()`。注意 D-02 决定不固定种子，直接用 `np.random.RandomState()` 即可。
**Warning signs:** AttributeError: 'NoneType' object has no attribute 'randint'.

### Pitfall 4: gen_expr 可能失败无限重试
**What goes wrong:** gen_expr 内部有 while True 循环，某些极端参数下可能持续生成无效样本。
**Why it happens:** gen_expr 的外层循环在 except 后 continue，没有退出机制。
**How to avoid:** 实际上 gen_expr 有正常的错误处理和重试机制，通常不会卡住。但应监控生成速度，如果极慢则排查参数。
**Warning signs:** 训练循环停滞不前，无数据产出。

### Pitfall 5: batch_equations 的输入格式要求
**What goes wrong:** batch_equations 期望 `List[LongTensor]`，如果传入格式不对会崩溃。
**Why it happens:** gen_expr 返回的 tree_encoded 是 `List[str]`（字符串列表），需要先通过 `env.word_to_idx()` 转换为 `List[LongTensor]`。
**How to avoid:** 严格遵循 enc_dec_vae_step 中的格式：先 `env.word_to_idx(samples["tree_encoded"], float_input=False)` 得到 `List[LongTensor]`，再传给 `env.batch_equations()`。

## Code Examples

### 完整数据管线参考（来自 trainer_vae.py:enc_dec_vae_step，行 854-888）

```python
# === 第一步：获取原始样本 ===
# env.gen_expr(train=True) 返回 expr 字典，包含：
#   X_to_fit, Y_to_fit: List[np.ndarray] 数值数据点
#   tree_encoded: List[str] 符号表达式的 token 序列
#   skeleton_tree_encoded: List[str] 骨架表达式 token 序列
#   infos: dict 包含 n_input_points 等元数据

# === 第二步：构造 x1（数值数据编码输入）===
# 来自 trainer_vae.py:862-871
x_to_fit = samples["x_to_fit"]   # list of ndarray
y_to_fit = samples["y_to_fit"]   # list of ndarray
x1 = []
for seq_id in range(len(x_to_fit)):
    x1.append([])
    for seq_l in range(len(x_to_fit[seq_id])):
        x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
x1, len1 = embedder_f(x1)  # NumericalEmbedder.forward()

# === 第三步：构造 x2（表达式 token 编码输入）===
# 来自 trainer_vae.py:880-886
x2, len2 = env.batch_equations(
    env.word_to_idx(samples["tree_encoded"], float_input=False)
)
x2, len2 = to_cuda(x2, len2)
x2_e = embedder_e(x2.transpose(0, 1)).transpose(0, 1)

# === 第四步：CVAE 前向传播 ===
# 来自 trainer_vae.py:888
prior_mu, prior_logvar, post_mu, post_logvar, kl_loss, kl_weights, kld = \
    vae_model(x1, x2_e, len1, len2, mode="train")
# prior_mu: (batch, 512), post_mu: (batch, 512)
```

### CVAE forward(mode="train") 签名和返回值

```python
# 来自 symbolicregression/model/cvae.py:807-833
def forward(self, x1, x2, len1, len2, mode: str = "train"):
    """
    参数:
        x1: 数值数据嵌入, shape (max_seq_len, batch, emb_dim)
        x2: 表达式 token 嵌入, shape (max_eq_len, batch, emb_dim)
        len1: 数值数据序列长度, shape (batch,)
        len2: 表达式 token 序列长度, shape (batch,)
        mode: "train" 返回全部, 其他只返回 prior

    返回 (mode="train"):
        prior_mu: (batch, 512)
        prior_logvar: (batch, 512)
        post_mu: (batch, 512)
        post_logvar: (batch, 512)
        kl_loss: scalar
        kl_weights: scalar
        kld: (batch,)
    """
```

### 现有模块初始化和加载流程

```python
# 来自 train.py:218-222 和 symbolicregression/model/__init__.py
# 1. 构建 env
env = build_env(params)  # 需要 params 有 env_name, tasks 等参数

# 2. 构建 modules
modules = build_modules(env, params)  # 返回 dict: data_encoder, cvae, token_embed, seq_decoder, feature_fusion

# 3. 加载权重（冻结）
from symbolicregression.model import reload_model
modules = reload_model(
    modules,
    modules_to_load=["cvae", "data_encoder", "token_embed"],
    path="weights/checkpoint.pth",
    requires_grad=False
)
```

### FunctionEnvironment.gen_expr 返回格式

```python
# 来自 symbolicregression/envs/environment.py:390-402
expr = {
    "X_to_fit": [np.ndarray],      # list of arrays, 每个形状 (n_points, input_dim)
    "Y_to_fit": [np.ndarray],      # list of arrays, 每个形状 (n_points, output_dim)
    "tree_encoded": [str, ...],    # 表达式 token 序列 (word list)
    "skeleton_tree_encoded": [str, ...],  # 骨架 token 序列
    "tree": tree_object,           # 原始树对象
    "skeleton_tree": tree_object,  # 骨架树对象
    "infos": {
        "n_input_points": [int],
        "n_unary_ops": [int],
        "n_binary_ops": [int],
        "d_in": [int],
        "d_out": [int],
        "input_distribution_type": [int],  # 0=gaussian, 1=uniform
        "n_centroids": [int],
    },
    "x_to_predict": np.ndarray,    # 预测数据点
    "y_to_predict": np.ndarray,    # 预测标签
}
```

### EnvDataset.generate_sample 参考

```python
# 来自 symbolicregression/envs/environment.py:1036-1063
def generate_sample(self):
    if self.remaining_data == 0:
        gen_args = {'train': self.train, 'input_length_modulo': self.input_length_modulo}
        self.expr, errors = self.env.gen_expr(**gen_args)
        self.remaining_data = len(self.expr["X_to_fit"])

    self.remaining_data -= 1
    x_to_fit = self.expr["X_to_fit"][-self.remaining_data]
    y_to_fit = self.expr["Y_to_fit"][-self.remaining_data]
    sample = copy.deepcopy(self.expr)
    sample["x_to_fit"] = x_to_fit
    sample["y_to_fit"] = y_to_fit
    del sample["X_to_fit"]
    del sample["Y_to_fit"]
    return sample
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| EnvDataset (Map-style, size=2^60) | IterableDataset | 设计阶段 | 语义更准确，代码更清晰 |
| 逐样本生成 | 批量生成后 yield | 设计阶段 | CVAE 前向传播更高效 |

**Deprecated/outdated:**
- 无。所有使用的组件都是当前最新版本。

## Open Questions

1. **组件放置位置**
   - What we know: 需要 Phase 4 能直接 import。现有代码没有 dit_train/ 目录。
   - What's unclear: 放在项目根目录还是创建新目录。
   - Recommendation: 创建 `dit_train/data/latent_dataset.py`，保持与现有 `symbolicregression/` 包的结构一致。

2. **train/val split**
   - What we know: 数据是无限生成的，传统 split 不适用。
   - What's unclear: 是否需要验证集生成器。
   - Recommendation: 不做 split。DiT 训练时可以用不同的 FunctionEnvironment 参数（如不同的 noise_gamma）作为验证，但这属于 Phase 4 的设计。

3. **params 对象的构建**
   - What we know: 现有代码依赖 argparse 生成的 params (Namespace) 对象，含大量参数。
   - What's unclear: Dataset 如何获取 params。
   - Recommendation: Dataset 构造函数接受 params 对象，或者提取必要的参数子集。Phase 4 训练脚本需要用 parsers.py 构建完整 params。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch + CUDA | CVAE 前向传播 | OK | 2.10.0+cu128 | - |
| NVIDIA RTX 3090 | GPU 推理 | OK | 24GB VRAM | - |
| weights/checkpoint.pth | CVAE 权重 | OK | 641MB | - |
| Python 3.10 | 运行时 | OK | 3.10.12 | - |

**Missing dependencies with no fallback:**
- 无。所有依赖均已就绪。

**Missing dependencies with fallback:**
- 无。

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (需安装) |
| Config file | 无 - 需 Wave 0 创建 |
| Quick run command | `python -m pytest tests/test_latent_dataset.py -x -v` |
| Full suite command | `python -m pytest tests/test_latent_dataset.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Dataset 在线生成 (prior_mu, post_mu) 对，形状正确 | unit | `pytest tests/test_latent_dataset.py::test_latent_pair_shapes -x` | Wave 0 |
| DATA-01 | prior_mu 和 post_mu 不是全零或全相同 | unit | `pytest tests/test_latent_dataset.py::test_latent_pair_nontrivial -x` | Wave 0 |
| DATA-02 | DataLoader 能正常迭代出 batch | unit | `pytest tests/test_latent_dataset.py::test_dataloader_iteration -x` | Wave 0 |
| DATA-02 | batch_size 参数生效 | unit | `pytest tests/test_latent_dataset.py::test_batch_size -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_latent_dataset.py -x`
- **Per wave merge:** `pytest tests/test_latent_dataset.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_latent_dataset.py` - covers DATA-01, DATA-02
- [ ] Framework install: `pip install pytest` - if none detected
- [ ] `tests/__init__.py` - package init

## Sources

### Primary (HIGH confidence)
- `symbolicregression/model/cvae.py:807-833` - CVAE forward(mode="train") 签名和返回值
- `symbolicregression/trainer_vae.py:836-953` - enc_dec_vae_step 完整数据流
- `symbolicregression/envs/environment.py:194-402` - gen_expr 生成逻辑和返回格式
- `symbolicregression/envs/environment.py:713-1063` - EnvDataset 现有实现
- `symbolicregression/model/__init__.py:191-228` - build_modules 模块构建
- `symbolicregression/model/embedders.py` - NumericalEmbedder 完整实现
- `symbolicregression/model/feature_fusion.py` - FeatureFusion 实现
- `train.py:154-282` - 训练入口和模块初始化流程
- `model.py:1-108` - VAESymbolicRegressor 推理时使用模式
- `parsers.py` - 完整参数定义

### Secondary (MEDIUM confidence)
- 无外部搜索依赖，所有信息来自源码直接阅读

### Tertiary (LOW confidence)
- 无

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 无新依赖，全部复用现有组件
- Architecture: HIGH - 数据管线完全清晰，enc_dec_vae_step 提供了完整参考
- Pitfalls: HIGH - 从源码中直接识别，特别是 CUDA 多进程和 rng 初始化问题
- Component reuse: HIGH - 已精确识别每个可复用组件及其调用方式

**Research date:** 2026-06-09
**Valid until:** 2026-07-09 (稳定，代码库不变则长期有效)
