"""Euler 积分推理单元测试 — 验证 euler_inference 函数

覆盖:
1. 常数速度场精确解（1 步 Euler）
2. 默认 16 步，输出 shape 和无 NaN
3. 可调步数 (5 vs 20)，结果不同
4. 单样本输入 (512,) 自动 unsqueeze
5. no_grad 验证，参数 grad 全 None
6. 积分方向 t=0→1，z 趋近 target
"""

import torch
import pytest

from dit_train.model import GenSRDiT
from dit_train.inference import euler_inference


# ---------------------------------------------------------------------------
# 辅助: Mock DiT
# ---------------------------------------------------------------------------

class ConstantVelocityDiT(GenSRDiT):
    """DiT 子类，forward 恒返回 delta（常数速度场）。"""

    def __init__(self):
        super().__init__()
        self._delta = None

    def forward(self, z_t, t, prior_mu):
        assert self._delta is not None, "Call set_delta() first"
        return self._delta.expand_as(z_t)

    def set_delta(self, delta):
        self._delta = delta


class DirectedVelocityDiT(GenSRDiT):
    """DiT 子类，forward 返回 v = target - z_t（指向 target 的速度场）。"""

    def __init__(self):
        super().__init__()
        self._target = None

    def forward(self, z_t, t, prior_mu):
        assert self._target is not None, "Call set_target() first"
        return self._target.expand_as(z_t) - z_t

    def set_target(self, target):
        self._target = target


# ---------------------------------------------------------------------------
# Test 1: 常数速度场精确解
# ---------------------------------------------------------------------------

def test_euler_constant_velocity_exact():
    """Euler 1 步 + 常数速度场 v = post_mu - prior_mu → z_opt 应精确等于 post_mu。"""
    dit = ConstantVelocityDiT()
    dit.eval()

    prior_mu = torch.randn(4, 512)
    post_mu = torch.randn(4, 512)
    dit.set_delta(post_mu - prior_mu)

    z_opt = euler_inference(dit, prior_mu, num_steps=1)
    assert z_opt.shape == (4, 512)
    assert torch.allclose(z_opt, post_mu, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: 默认步数 16，shape 正确、无 NaN
# ---------------------------------------------------------------------------

def test_euler_default_steps():
    """默认 num_steps=16，使用真实 GenSRDiT，输出 (4, 512) 无 NaN。"""
    dit = GenSRDiT()
    dit.eval()

    prior_mu = torch.randn(4, 512)
    z_opt = euler_inference(dit, prior_mu)

    assert z_opt.shape == (4, 512)
    assert not torch.isnan(z_opt).any()


# ---------------------------------------------------------------------------
# Test 3: 可调步数，5 步和 20 步结果不同
# ---------------------------------------------------------------------------

def test_euler_adjustable_steps():
    """num_steps=5 和 num_steps=20 结果不同，说明步数影响积分路径。"""
    dit = DirectedVelocityDiT()
    dit.eval()

    prior_mu = torch.randn(2, 512)
    dit.set_target(torch.randn(2, 512))

    z5 = euler_inference(dit, prior_mu, num_steps=5)
    z20 = euler_inference(dit, prior_mu, num_steps=20)

    assert z5.shape == (2, 512)
    assert z20.shape == (2, 512)
    assert not torch.equal(z5, z20), "5 步和 20 步结果应不同"


# ---------------------------------------------------------------------------
# Test 4: 单样本输入 (512,)
# ---------------------------------------------------------------------------

def test_euler_single_sample():
    """prior_mu shape (512,) → 输出 shape (512,)。"""
    dit = GenSRDiT()
    dit.eval()

    prior_mu = torch.randn(512)
    z_opt = euler_inference(dit, prior_mu, num_steps=4)

    assert z_opt.shape == (512,), f"期望 (512,)，得到 {z_opt.shape}"
    assert not torch.isnan(z_opt).any()


# ---------------------------------------------------------------------------
# Test 5: no_grad 验证
# ---------------------------------------------------------------------------

def test_euler_no_grad():
    """euler_inference 内部不记录梯度，dit 参数的 grad 全部为 None。"""
    dit = GenSRDiT()
    dit.eval()

    prior_mu = torch.randn(2, 512)
    euler_inference(dit, prior_mu, num_steps=4)

    for name, p in dit.named_parameters():
        assert p.grad is None, f"参数 {name} 的 grad 应为 None"


# ---------------------------------------------------------------------------
# Test 6: 积分方向 t=0→1，z 趋近 target
# ---------------------------------------------------------------------------

def test_euler_direction_convergence():
    """速度场 v = target - z_t，Euler 积分后 z 应比 prior_mu 更接近 target。"""
    dit = DirectedVelocityDiT()
    dit.eval()

    prior_mu = torch.randn(3, 512) * 2.0
    target = torch.randn(3, 512)
    dit.set_target(target)

    z_opt = euler_inference(dit, prior_mu, num_steps=100)

    dist_before = (prior_mu - target).norm(dim=-1).mean().item()
    dist_after = (z_opt - target).norm(dim=-1).mean().item()
    assert dist_after < dist_before, \
        f"Euler 积分后应更接近 target: {dist_after:.4f} < {dist_before:.4f}"
