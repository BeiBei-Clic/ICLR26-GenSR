"""Euler 积分推理 — 从 prior_mu 出发，经 N 步 DiT 传输得到 z_opt

积分方向: t=0 → t=1（从 prior_mu 到 post_mu）
步长: dt = 1.0 / num_steps（均匀步长）
更新: z_{t+dt} = z_t + dt * v_psi(z_t, t; prior_mu)
"""

import torch


def euler_inference(dit, prior_mu, num_steps=16):
    """Euler 积分推理：从 prior_mu 出发，经 num_steps 步 DiT 传输得到 z_opt。

    Args:
        dit: GenSRDiT 模型，forward(z_t, t, prior_mu) → v_psi (B, 512)
        prior_mu: (B, 512) 或 (512,) CVAE 先验均值
        num_steps: Euler 积分步数，默认 16
    Returns:
        z_opt: 与输入 prior_mu 同形状的优化潜向量
    """
    squeeze_output = False
    if prior_mu.ndim == 1:
        prior_mu = prior_mu.unsqueeze(0)
        squeeze_output = True

    B = prior_mu.shape[0]
    z = prior_mu
    dt = 1.0 / num_steps

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=z.device)
            v = dit(z, t, prior_mu)
            z = z + dt * v

    if squeeze_output:
        z = z.squeeze(0)

    return z
