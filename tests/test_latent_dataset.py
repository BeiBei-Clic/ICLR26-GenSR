import pytest
import torch
import numpy as np
from parsers import get_parser
from dit_train.data.latent_dataset import LatentPairDataset, create_latent_dataloader


@pytest.fixture(scope="module")
def params():
    p = get_parser().parse_args(["--max_input_dimension", "10"])
    p.device = "cuda"
    return p


@pytest.fixture(scope="module")
def dataset(params):
    return LatentPairDataset(params, batch_size=4, device="cuda")


@pytest.mark.slow
def test_latent_pair_shapes(dataset):
    """DATA-01: 每次 __iter__ yield (prior_mu, post_mu) 对，形状各为 (512,)"""
    prior_mu, post_mu = next(iter(dataset))
    assert prior_mu.shape == (512,), f"prior_mu shape: {prior_mu.shape}"
    assert post_mu.shape == (512,), f"post_mu shape: {post_mu.shape}"


@pytest.mark.slow
def test_latent_pair_nontrivial(dataset):
    """DATA-02: prior_mu 和 post_mu 不是全零，且两者不相等"""
    prior_mu, post_mu = next(iter(dataset))
    assert not torch.all(prior_mu == 0), "prior_mu is all zeros"
    assert not torch.all(post_mu == 0), "post_mu is all zeros"
    assert not torch.allclose(prior_mu, post_mu), "prior_mu == post_mu (trivial)"


@pytest.mark.slow
def test_dataloader_iteration(params):
    """DataLoader 包裹后能正常迭代 3 个 batch"""
    dataloader = create_latent_dataloader(params, batch_size=4)
    for i, (prior_mu, post_mu) in enumerate(dataloader):
        assert prior_mu.shape == (4, 512), f"batch {i} prior_mu shape: {prior_mu.shape}"
        assert post_mu.shape == (4, 512), f"batch {i} post_mu shape: {post_mu.shape}"
        if i >= 2:
            break


@pytest.mark.slow
def test_batch_size(params):
    """batch_size 参数生效：batch_size=2 和 batch_size=4 各产出正确形状"""
    for bs in [2, 4]:
        dataloader = create_latent_dataloader(params, batch_size=bs)
        prior_mu, post_mu = next(iter(dataloader))
        assert prior_mu.shape == (bs, 512), f"batch_size={bs} prior_mu shape: {prior_mu.shape}"
        assert post_mu.shape == (bs, 512), f"batch_size={bs} post_mu shape: {post_mu.shape}"


@pytest.mark.slow
def test_cva_frozen(dataset):
    """CVAE 完全冻结，没有任何参数 requires_grad=True"""
    for name, param in dataset.vae_model.named_parameters():
        assert not param.requires_grad, f"CVAE param {name} still requires grad"
