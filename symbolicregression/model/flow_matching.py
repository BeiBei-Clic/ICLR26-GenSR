import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_sinusoidal_embedding(timesteps, embedding_dim):
    """正弦时间步嵌入，参考 ColaDLM 的 TimestepEmbedding。"""
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        0, half_dim, device=timesteps.device, dtype=torch.float32
    ) / half_dim
    emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, t):
        emb = _get_sinusoidal_embedding(t, self.sinusoidal_dim)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        return emb


class FMBlock(nn.Module):
    """带 AdaLN 时间调制的残差 MLP 块，参考 ColaDLM 的 AdaLN 机制。"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.SiLU()

    def forward(self, h, t_emb):
        scale, shift = self.adaLN(t_emb).chunk(2, dim=-1)
        h_norm = self.norm(h) * (1 + scale) + shift
        return h + self.proj(self.act(h_norm))


class FlowMatchingNet(nn.Module):
    """条件 Flow Matching 速度预测网络。

    输入: z_t (B, latent_dim) + t (B,) + condition (B, latent_dim)
    输出: v (B, latent_dim) 预测向量场
    """

    def __init__(self, latent_dim=512, hidden_dim=1024, n_layers=6, time_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimestepEmbedding(time_dim, hidden_dim)
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([FMBlock(hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, condition):
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([z_t, condition], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


# ── ODE 求解器 ──────────────────────────────────────────────────


def euler_solve(fm_net, z_1, condition, n_steps=10):
    """Euler 方法从 t=1 积分到 t=0。"""
    dt = 1.0 / n_steps
    z = z_1
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device)
        v = fm_net(z, t, condition)
        z = z - v * dt
    return z


def heun_solve(fm_net, z_1, condition, n_steps=10):
    """Heun 方法（二阶）从 t=1 积分到 t=0。"""
    dt = 1.0 / n_steps
    z = z_1
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device)
        v1 = fm_net(z, t, condition)
        z_next = z - v1 * dt
        t_next = torch.full((z.shape[0],), max(0.0, 1.0 - (i + 1) * dt), device=z.device)
        v2 = fm_net(z_next, t_next, condition)
        z = z - (v1 + v2) * dt / 2
    return z


# ── 训练 loss ──────────────────────────────────────────────────


def compute_fm_loss(fm_net, z_0, condition):
    """OT-CFM loss。

    z_0: (B, D) 真实潜在（CVAE 的 post_mu）
    condition: (B, D) 数据条件（CVAE 的 prior_mu）
    """
    B, D = z_0.shape
    z_1 = torch.randn_like(z_0)
    t = torch.rand(B, device=z_0.device)

    t_expand = t.unsqueeze(-1)
    z_t = (1 - t_expand) * z_0 + t_expand * z_1

    u_t = z_1 - z_0
    v_pred = fm_net(z_t, t, condition)

    return F.mse_loss(v_pred, u_t)
