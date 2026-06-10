"""GenSR DiT 模型前向传播 shape 验证测试"""

import torch
import sys
sys.path.insert(0, ".")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_forward_shape():
    from dit_train.model import GenSRDiT

    model = GenSRDiT()
    z_t = torch.randn(4, 512)
    t = torch.rand(4)
    prior_mu = torch.randn(4, 512)

    output = model(z_t, t, prior_mu)
    assert output.shape == (4, 512), f"Output shape wrong: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print(f"test_forward_shape passed: output shape {output.shape}")


def test_config_flexibility():
    from dit_train.model import GenSRDiT

    config = {
        "num_layers": 3,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_patches": 8,
        "patch_dim": 64,
        "latent_dim": 512,
        "ffn_expand_ratio": 4,
        "timestep_embed_dim": 256,
    }
    model = GenSRDiT(config)
    z_t = torch.randn(2, 512)
    t = torch.rand(2)
    prior_mu = torch.randn(2, 512)

    output = model(z_t, t, prior_mu)
    assert output.shape == (2, 512), f"Output shape wrong: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print(f"test_config_flexibility passed: output shape {output.shape}")


def test_parameter_count_changes():
    from dit_train.model import GenSRDiT

    model_default = GenSRDiT()
    model_small = GenSRDiT({
        "latent_dim": 512, "num_patches": 8, "patch_dim": 64,
        "hidden_dim": 128, "num_heads": 4, "num_layers": 2,
        "ffn_expand_ratio": 4, "timestep_embed_dim": 256,
    })

    count_default = count_parameters(model_default)
    count_small = count_parameters(model_small)
    assert count_default > count_small, \
        f"Default params ({count_default}) should > small params ({count_small})"
    print(f"test_parameter_count_changes passed: default={count_default}, small={count_small}")


def test_output_zero_init():
    """验证 AdaLN-Zero + 输出层零初始化使初始输出接近 0"""
    from dit_train.model import GenSRDiT

    model = GenSRDiT()
    z_t = torch.randn(2, 512)
    t = torch.rand(2)
    prior_mu = torch.randn(2, 512)

    output = model(z_t, t, prior_mu)
    max_abs = output.abs().max().item()
    assert max_abs < 0.01, f"Output not near zero: max abs = {max_abs}"
    print(f"test_output_zero_init passed: max abs = {max_abs}")


if __name__ == "__main__":
    test_forward_shape()
    test_config_flexibility()
    test_parameter_count_changes()
    test_output_zero_init()
    print("\nAll 4 tests passed!")
