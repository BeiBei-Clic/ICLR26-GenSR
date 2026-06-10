"""GenSR DiT 模型定义 — AdaLN-Zero 条件化的 Diffusion Transformer

参照 Cola-DLM-main/cola_dlm/modeling_cola_dit.py 架构，适配到 512 维单向量潜空间。
输入: z_t (B, 512), t (B,), prior_mu (B, 512)
输出: 速度预测 v_psi (B, 512)

关键适配（按 D-01 到 D-07 决策）:
- D-02: 16 patches x 32 dim patchification
- D-03: prior_mu 通过 cross-attention 注入
- D-05: GELU FFN（非 SwiGLU）
- D-06: LayerNorm（非 RMSNorm）
- D-07: 移除 RoPE、块因果 mask、KV cache
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


dit_default_config = {
    "latent_dim": 512,
    "num_patches": 16,
    "patch_dim": 32,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "ffn_expand_ratio": 4,
    "timestep_embed_dim": 256,
}


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------


def _get_sinusoidal_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embedding，照搬 Cola 实现（downscale_freq_shift=0）。

    half_dim = embedding_dim // 2
    exponent = -log(10000) * arange(half_dim) / half_dim
    emb = t[:, None] * exp(exponent)[None, :]
    cat([sin(emb), cos(emb)], dim=-1)
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Timestep 编码: sinusoidal → Linear → SiLU → Linear → SiLU → Linear"""

    def __init__(self, sinusoidal_dim, hidden_dim, output_dim):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()
        # Cola 模式初始化
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_hid.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    def forward(self, timestep, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = _get_sinusoidal_embedding(timestep, self.sinusoidal_dim).to(dtype)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        emb = self.proj_out(emb)
        return emb


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN) — 简化版
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    """AdaLN-Zero 条件调制。

    无 residual 时: "in" 模式 — LayerNorm(x) * (1 + scale) + shift
    有 residual 时: "out" 模式 — x * gate + residual
    """

    def __init__(self, dim, emb_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 3 * dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # AdaLN-Zero: proj 最后一层初始化为 0
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, x, emb, residual=None):
        shift, scale, gate = self.proj(emb).chunk(3, dim=-1)
        # 处理 batch 维度广播: emb (B, dim) → (B, 1, dim) 以匹配 x (B, L, dim)
        if x.ndim == 3 and emb.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        if residual is None:
            # "in" 模式: 归一化 + 调制
            return self.norm(x) * (1 + scale) + shift
        # "out" 模式: 门控 + 残差
        return x * gate + residual


# ---------------------------------------------------------------------------
# MLP (GELU FFN)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """GELU FFN: Linear(dim, dim*expand_ratio) → GELU(tanh) → Linear(dim*expand_ratio, dim)"""

    def __init__(self, dim, expand_ratio):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x):
        return self.proj_out(self.act(self.proj_in(x)))


# ---------------------------------------------------------------------------
# DiT Block (Self-Attention + Cross-Attention + FFN, AdaLN-Zero 调制)
# ---------------------------------------------------------------------------


class GenSRDiTBlock(nn.Module):
    """参照 ColaDiTBlock，添加 cross-attention 注入 prior_mu。

    三个子块，每个都有 AdaLN in/out 配对:
    1. Self-attention on z_t patches
    2. Cross-attention: Q=z_t patches, K=V=prior_mu
    3. GELU FFN

    无 RoPE (D-07)、无 causal mask (D-07)、无 KV cache (D-07)。
    """

    def __init__(self, hidden_dim, num_heads, emb_dim, ffn_expand_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.self_attn_out = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention: Q from z_t patches, K/V from prior_mu
        self.cross_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_attn_out = nn.Linear(hidden_dim, hidden_dim)

        # FFN
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = MLP(hidden_dim, ffn_expand_ratio)

        # 共享 AdaLN（按 Cola 模式，每个子块有独立 AdaLN 但这里简化为 3 个独立实例）
        self.ada_sa = AdaLN(hidden_dim, emb_dim)
        self.ada_ca = AdaLN(hidden_dim, emb_dim)
        self.ada_ff = AdaLN(hidden_dim, emb_dim)

    def _multi_head_attention(self, q, k, v, proj_out):
        """标准缩放点积注意力，无 RoPE/无 causal mask。

        q: (B*num_heads, Lq, head_dim)
        k: (B*num_heads, Lk, head_dim)
        v: (B*num_heads, Lk, head_dim)
        """
        d_head = q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn = q.mul(scale) @ k.transpose(-2, -1)
        attn_weight = attn.softmax(dim=-1)
        out = attn_weight @ v
        # (B*num_heads, Lq, head_dim) → (B, Lq, hidden_dim)
        B_multi, Lq, _ = out.shape
        out = out.reshape(B_multi // self.num_heads, self.num_heads, Lq, self.head_dim)
        out = out.transpose(1, 2).reshape(B_multi // self.num_heads, Lq, -1)
        return proj_out(out)

    def forward(self, x, emb, cond):
        """
        x:    (B, num_patches, hidden_dim) — z_t patches
        emb:  (B, hidden_dim) — timestep embedding
        cond: (B, 1, hidden_dim) — prior_mu 投影后的条件
        """
        B, L, D = x.shape

        # --- Self-attention sub-block ---
        h = self.ada_sa(x, emb)
        qkv = self.self_attn_qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # reshape to (B*heads, L, head_dim)
        q = q.reshape(B * self.num_heads, L, self.head_dim)
        k = k.reshape(B * self.num_heads, L, self.head_dim)
        v = v.reshape(B * self.num_heads, L, self.head_dim)
        h = self._multi_head_attention(q, k, v, self.self_attn_out)
        x = self.ada_sa(h, emb, residual=x)

        # --- Cross-attention sub-block (D-03) ---
        h = self.ada_ca(x, emb)
        # Q from z_t patches
        q = self.cross_attn_q(h).reshape(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).reshape(B * self.num_heads, L, self.head_dim)
        # K/V from prior_mu cond (B, 1, hidden_dim)
        kv = self.cross_attn_kv(cond.squeeze(1)).reshape(B, 1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        ck, cv = kv[0], kv[1]  # (B, heads, 1, head_dim)
        ck = ck.reshape(B * self.num_heads, 1, self.head_dim)
        cv = cv.reshape(B * self.num_heads, 1, self.head_dim)
        h = self._multi_head_attention(q, ck, cv, self.cross_attn_out)
        x = self.ada_ca(h, emb, residual=x)

        # --- FFN sub-block ---
        h = self.ada_ff(x, emb)
        h = self.mlp(h)
        x = self.ada_ff(h, emb, residual=x)

        return x


# ---------------------------------------------------------------------------
# GenSR DiT 顶层模型
# ---------------------------------------------------------------------------


class GenSRDiT(nn.Module):
    """GenSR DiT: 学习 prior_mu → post_mu 的 Flow Matching 速度场。

    输入: z_t (B, latent_dim), t (B,), prior_mu (B, latent_dim)
    输出: 速度预测 (B, latent_dim)

    架构:
    1. z_t patchify: (B, latent_dim) → (B, num_patches, patch_dim) → Linear → (B, num_patches, hidden_dim)
    2. prior_mu 投影: (B, latent_dim) → Linear → (B, 1, hidden_dim)
    3. Timestep 编码: (B,) → sinusoidal → (B, hidden_dim)
    4. N x DiTBlock(self_attn + cross_attn + ffn, AdaLN-Zero 调制)
    5. LayerNorm + Linear → (B, num_patches, patch_dim) → unpatchify → (B, latent_dim)
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or dit_default_config
        latent_dim = config["latent_dim"]
        num_patches = config["num_patches"]
        patch_dim = config["patch_dim"]
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        ffn_expand_ratio = config["ffn_expand_ratio"]
        timestep_embed_dim = config["timestep_embed_dim"]

        assert latent_dim == num_patches * patch_dim, \
            f"latent_dim ({latent_dim}) != num_patches ({num_patches}) * patch_dim ({patch_dim})"

        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim

        # Patchify 投影
        self.patch_proj_in = nn.Linear(patch_dim, hidden_dim)
        # prior_mu 条件投影
        self.cond_proj = nn.Linear(latent_dim, hidden_dim)
        # Timestep 编码
        self.timestep_emb = TimestepEmbedding(
            sinusoidal_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        # DiT blocks
        self.blocks = nn.ModuleList([
            GenSRDiTBlock(hidden_dim, num_heads, hidden_dim, ffn_expand_ratio)
            for _ in range(num_layers)
        ])
        # 最终归一化 + 输出投影
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.patch_proj_out = nn.Linear(hidden_dim, patch_dim)
        # AdaLN-Zero 输出初始化（per Cola PatchOut1D）
        nn.init.constant_(self.patch_proj_out.weight, 0)
        nn.init.constant_(self.patch_proj_out.bias, 0)

    def forward(self, z_t, t, prior_mu):
        """
        z_t:      (B, latent_dim) — 噪声潜向量
        t:        (B,) — Flow Matching 时间步
        prior_mu: (B, latent_dim) — CVAE 先验均值，条件信号
        返回:     (B, latent_dim) — 预测速度场
        """
        B = z_t.shape[0]

        # Patchify: (B, latent_dim) → (B, num_patches, patch_dim) → (B, num_patches, hidden_dim)
        x = z_t.reshape(B, self.num_patches, self.patch_dim)
        x = self.patch_proj_in(x)

        # 条件投影: (B, latent_dim) → (B, hidden_dim) → (B, 1, hidden_dim)
        cond = self.cond_proj(prior_mu).unsqueeze(1)

        # Timestep 编码: (B,) → (B, hidden_dim)
        emb = self.timestep_emb(t, device=z_t.device, dtype=z_t.dtype)

        # DiT blocks
        for block in self.blocks:
            x = block(x, emb, cond)

        # 输出: (B, num_patches, hidden_dim) → LayerNorm → Linear → (B, num_patches, patch_dim)
        x = self.final_norm(x)
        x = self.patch_proj_out(x)

        # Unpatchify: (B, num_patches, patch_dim) → (B, latent_dim)
        output = x.reshape(B, -1)
        return output
