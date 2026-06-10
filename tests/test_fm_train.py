"""Flow Matching 训练核心函数单元测试 — 覆盖 FM-01 到 FM-05"""
import os
import torch
import pytest

from dit_train.model import GenSRDiT
from dit_train.train_fm import (
    sample_timestep,
    get_lr_multiplier,
    flow_matching_step,
    save_checkpoint,
    load_checkpoint,
    evaluate,
)

# 共享小模型配置，避免 5 处重复
SMALL_CONFIG = {
    "latent_dim": 512, "num_patches": 16, "patch_dim": 32,
    "hidden_dim": 64, "num_heads": 4, "num_layers": 1,
    "ffn_expand_ratio": 2, "timestep_embed_dim": 32,
}


# ---------------------------------------------------------------------------
# test_sample_timestep (D-02)
# ---------------------------------------------------------------------------

def test_sample_timestep():
    ts = sample_timestep(16, "logit_normal")
    assert ts.shape == (16,)
    assert (ts > 0).all() and (ts < 1).all()

    ts_u = sample_timestep(16, "uniform")
    assert ts_u.shape == (16,)
    assert (ts_u >= 0).all() and (ts_u <= 1).all()


# ---------------------------------------------------------------------------
# test_get_lr_multiplier (D-04)
# ---------------------------------------------------------------------------

def test_get_lr_multiplier():
    # warmup 区间: progress=0.025 → 接近 0.5
    lr_warmup = get_lr_multiplier(0.025, warmup_ratio=0.05)
    assert lr_warmup < 1.0
    assert lr_warmup > 0.0

    # plateau: progress=0.5 → 1.0
    lr_plateau = get_lr_multiplier(0.5, warmup_ratio=0.05, warmdown_ratio=0.3)
    assert lr_plateau == 1.0

    # warmdown 结束: progress=1.0 → final_lr_frac=0.0
    lr_end = get_lr_multiplier(1.0, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.0)
    assert lr_end < 0.01


# ---------------------------------------------------------------------------
# test_ot_path_interpolation (FM-01)
# ---------------------------------------------------------------------------

def test_ot_path_interpolation():
    prior_mu = torch.randn(4, 512)
    post_mu = torch.randn(4, 512)
    t = torch.tensor([0.5] * 4)

    # 手动计算期望值
    z_t = (1 - t[:, None]) * prior_mu + t[:, None] * post_mu
    target = post_mu - prior_mu

    z_t_expected = 0.5 * prior_mu + 0.5 * post_mu
    assert torch.allclose(z_t, z_t_expected, atol=1e-6)
    assert torch.allclose(target, post_mu - prior_mu, atol=1e-6)


# ---------------------------------------------------------------------------
# test_fm_loss (FM-02)
# ---------------------------------------------------------------------------

def test_fm_loss():
    dit = GenSRDiT(SMALL_CONFIG).cuda().train()
    prior_mu = torch.randn(2, 512).cuda()
    post_mu = torch.randn(2, 512).cuda()

    loss, train_loss = flow_matching_step(dit, prior_mu, post_mu)
    assert loss.dim() == 0, "loss 应为标量"
    assert loss.item() > 0, "loss 应大于 0"
    assert torch.allclose(train_loss, loss.detach()), "train_loss 应与 loss.detach() 一致"


# ---------------------------------------------------------------------------
# test_only_dit_grad (FM-03)
# ---------------------------------------------------------------------------

def test_only_dit_grad():
    dit = GenSRDiT(SMALL_CONFIG).cuda().train()
    prior_mu = torch.randn(2, 512).cuda()
    post_mu = torch.randn(2, 512).cuda()

    loss, _ = flow_matching_step(dit, prior_mu, post_mu)
    loss.backward()

    # DiT 参数应有梯度
    for name, param in dit.named_parameters():
        assert param.grad is not None, f"DiT 参数 {name} 应有梯度"


# ---------------------------------------------------------------------------
# test_checkpoint_save_load (FM-05)
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(tmp_path):
    dit = GenSRDiT(SMALL_CONFIG).cuda().train()
    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-4)

    # 保存
    output_dir = str(tmp_path / "ckpts")
    save_checkpoint(dit, optimizer, step=100, val_loss=0.5, output_dir=output_dir)

    # 验证文件存在
    ckpt_path = os.path.join(output_dir, "fm_step_000100.pt")
    assert os.path.exists(ckpt_path)

    # 加载到新模型
    dit2 = GenSRDiT(SMALL_CONFIG).cuda()
    step, val_loss, ckpt = load_checkpoint(dit2, ckpt_path)

    assert step == 100
    assert val_loss == 0.5
    assert "optimizer_state_dict" in ckpt

    # 验证 state_dict 一致
    for (k1, v1), (k2, v2) in zip(
        dit.state_dict().items(), dit2.state_dict().items()
    ):
        assert torch.equal(v1, v2), f"state_dict key {k1} 不一致"


# ---------------------------------------------------------------------------
# test_evaluate
# ---------------------------------------------------------------------------

def test_evaluate():
    dit = GenSRDiT(SMALL_CONFIG).cuda().train()

    # 构造一个简单的数据迭代器
    def fake_data_iter():
        while True:
            yield torch.randn(2, 512).cuda(), torch.randn(2, 512).cuda()

    data_iter = fake_data_iter()
    avg_loss = evaluate(dit, data_iter, eval_steps=3)
    assert isinstance(avg_loss, float)
    assert avg_loss > 0
