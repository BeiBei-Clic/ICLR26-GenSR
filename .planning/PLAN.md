# Phase 1 计划：Flow Matching 核心模块实现

## 目标
实现条件 Flow Matching 模型、训练 loss 和 ODE 推理器，使模型可以在实例化后前向传播并计算 loss。

## 背景

### ColaDLM 关键实现（已研究）
ColaDLM 源码 (`resources/Cola-DLM-main/cola_dlm/`) 的 Flow Matching 实现要点：
- **时间嵌入**：正弦嵌入 + MLP 投影（`_get_sinusoidal_embedding` → `TimestepEmbedding`）
- **AdaLN**：通过时间嵌入生成 scale/shift 调制 Transformer 层
- **ODE 推理**：Euler 积分，从 t=T(1000) 到 t=0，默认 16 步
- **条件注入**：通过 KV 缓存将前缀编码注入 DiT 注意力
- **损失**：条件 Flow Matching，目标向量场 u_t = z_1 - z_0，插值 z_t = (1-t)z_0 + t*z_1

### 我们的简化适配
ColaDLM 的 DiT 处理的是**序列潜在**（多个 token 块），需要 block-causal 注意力和 KV 缓存。
我们的 CVAE 潜在是**单个 512 维向量**，因此：
- 不需要 block-causal 注意力、KV 缓存、序列位置编码
- 用简单 MLP + 时间 AdaLN 即可
- 条件注入直接 concat prior_mu 即可（无需 cross-attention）

---

## Task 1: 实现 Flow Matching 模型

### 文件: `symbolicregression/model/flow_matching.py`（新建）

#### 1.1 时间嵌入 `TimestepEmbedding`
参考 ColaDLM 的 `_get_sinusoidal_embedding` + `TimestepEmbedding`：
- 输入：标量 t（0 到 1，已归一化）
- 正弦嵌入 → Linear → SiLU → Linear
- 输出维度：hidden_dim

```python
class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim):
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, t):
        # t: (B,) 标量时间步
        emb = sinusoidal_embedding(t, self.sinusoidal_dim)  # (B, sinusoidal_dim)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        return emb  # (B, hidden_dim)
```

#### 1.2 Flow Matching 速度网络 `FlowMatchingNet`
参考 ColaDLM 的 `ColaDiTBlock` 的 AdaLN 调制思想，但用 MLP 替代 Transformer：
- 输入：z_t (B, latent_dim=512) + t (B,) + condition (B, latent_dim=512)
- 时间嵌入调制：AdaLN 风格，每层用 time_emb 生成 (scale, shift)
- 条件注入：concat z_t 和 condition 作为网络输入
- 输出：v (B, latent_dim=512) 预测向量场

```python
class FlowMatchingNet(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024, n_layers=6):
        # 时间嵌入
        self.time_embed = TimestepEmbedding(sinusoidal_dim=256, hidden_dim=hidden_dim)
        # 输入投影: concat(z_t, condition) → hidden_dim
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        # N 个残差块，每块有 AdaLN 调制
        self.blocks = nn.ModuleList([FMBlock(hidden_dim) for _ in range(n_layers)])
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, condition):
        # z_t: (B, D), t: (B,), condition: (B, D)
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        h = self.input_proj(torch.cat([z_t, condition], dim=-1))  # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)  # AdaLN 调制
        v = self.output_proj(h)  # (B, D)
        return v
```

#### 1.3 FMBlock（带 AdaLN 的残差 MLP 块）
```python
class FMBlock(nn.Module):
    def __init__(self, hidden_dim):
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN = nn.Linear(hidden_dim, hidden_dim * 2)  # scale, shift
        self.act = nn.SiLU()

    def forward(self, h, t_emb):
        # AdaLN: scale 和 shift 由时间嵌入控制
        scale, shift = self.adaLN(t_emb).chunk(2, dim=-1)
        h_norm = self.norm(h) * (1 + scale) + shift
        h = h + self.proj(self.act(h_norm))  # 残差连接
        return h
```

---

## Task 2: 实现 ODE 推理器

### 同一文件: `symbolicregression/model/flow_matching.py`

#### 2.1 Euler 求解器
参考 ColaDLM 的 `generate_task_repaint_inference` 中的 Euler 积分：
- 从 t=1（纯噪声）到 t=0（目标潜在）
- 固定步数（默认 10-20 步）

```python
def euler_solve(fm_net, z_1, condition, n_steps=10):
    dt = 1.0 / n_steps
    z = z_1
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device)
        v = fm_net(z, t, condition)  # 预测向量场
        z = z - v * dt  # Euler 更新: z_{t-dt} = z_t - v * dt
    return z  # z_0
```

#### 2.2 Heun 求解器（二阶，可选更高质量）
```python
def heun_solve(fm_net, z_1, condition, n_steps=10):
    dt = 1.0 / n_steps
    z = z_1
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device)
        v1 = fm_net(z, t, condition)
        z_next = z - v1 * dt
        t_next = torch.full((z.shape[0],), 1.0 - (i + 1) * dt, device=z.device)
        v2 = fm_net(z_next, t_next, condition)
        z = z - (v1 + v2) * dt / 2  # Heun 校正
    return z
```

#### 2.3 多次采样推理
```python
def fm_sample(fm_net, condition, latent_dim, n_samples=16, n_steps=10, solver='euler'):
    # condition: (B, D) → 扩展为 (B*n_samples, D)
    cond_expanded = condition.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, latent_dim)
    z_1 = torch.randn(cond_expanded.shape[0], latent_dim, device=condition.device)

    if solver == 'heun':
        z_0 = heun_solve(fm_net, z_1, cond_expanded, n_steps)
    else:
        z_0 = euler_solve(fm_net, z_1, cond_expanded, n_steps)

    # reshape 回 (B, n_samples, D)
    B = condition.shape[0]
    z_0 = z_0.reshape(B, n_samples, latent_dim)
    return z_0  # (B, n_samples, D)
```

---

## Task 3: 实现 Flow Matching 训练 loss

### 同一文件: `symbolicregression/model/flow_matching.py`

#### 3.1 OT-CFM Loss
参考 ColaDLM 论文公式 3.7 和实际实现：
- z_0 = post_mu（CVAE 后验，真实潜在）
- z_1 = noise ~ N(0, I)
- z_t = (1-t) * z_0 + t * z_1（线性插值）
- 目标向量场：u_t = z_1 - z_0
- 预测向量场：v = FlowMatchingNet(z_t, t, condition=prior_mu)
- Loss = MSE(v, u_t)

```python
def compute_fm_loss(fm_net, z_0, condition):
    """
    z_0: (B, D) 真实潜在（post_mu）
    condition: (B, D) 数据条件（prior_mu）
    """
    B, D = z_0.shape
    z_1 = torch.randn_like(z_0)  # 噪声
    t = torch.rand(B, device=z_0.device)  # 随机时间步 [0, 1]

    # 线性插值
    t_expand = t.unsqueeze(-1)  # (B, 1)
    z_t = (1 - t_expand) * z_0 + t_expand * z_1

    # 目标向量场
    u_t = z_1 - z_0

    # 预测向量场
    v_pred = fm_net(z_t, t, condition)

    # MSE loss
    loss = F.mse_loss(v_pred, u_t)
    return loss
```

---

## Task 4: 更新 parsers.py 和 build_modules

### 4.1 新增参数 (`parsers.py`)
```python
parser.add_argument("--fm_hidden_dim", type=int, default=1024, help="Flow Matching hidden dim")
parser.add_argument("--fm_n_layers", type=int, default=6, help="Flow Matching network layers")
parser.add_argument("--fm_n_samples", type=int, default=16, help="Number of FM samples at inference")
parser.add_argument("--fm_ode_steps", type=int, default=10, help="ODE solver steps")
parser.add_argument("--fm_solver", type=str, default="euler", help="ODE solver: euler or heun")
parser.add_argument("--fm_time_dim", type=int, default=256, help="Sinusoidal time embedding dim")
```

### 4.2 注册模块 (`symbolicregression/model/__init__.py`)
在 `build_modules` 中添加：
```python
from .flow_matching import FlowMatchingNet
modules["flow_matching"] = FlowMatchingNet(
    latent_dim=params.latent_dim,
    hidden_dim=params.fm_hidden_dim,
    n_layers=params.fm_n_layers,
)
```

### 4.3 更新 model.py
在 `VAESymbolicRegressor.__init__` 中添加：
```python
self.fm_net = self.modules.get("flow_matching", None)
```

---

## 依赖分析
- Task 1, 2, 3 全部在同一个新文件 `flow_matching.py` 中，顺序实现
- Task 4 依赖 Task 1 完成后才能 import
- 实际执行顺序：Task 1 → Task 2 → Task 3 → Task 4

## 验证标准
1. `FlowMatchingNet` 可实例化，前向传播输出正确形状
2. `compute_fm_loss` 可计算 loss，可反向传播
3. `euler_solve` 和 `heun_solve` 可从噪声生成潜在
4. `build_modules` 可正确构建包含 FM 模型的完整模块字典
5. 加载已有 CVAE checkpoint 不报错（新增的 FM 模块不在 checkpoint 中应被忽略）

## 不做的事
- 不修改现有 CVAE 架构
- 不修改训练循环
- 不实现完整推理管线（Phase 3 的事）
- 不实现 Stage 2 训练流程（Phase 2 的事）
- 不引入新的外部依赖
