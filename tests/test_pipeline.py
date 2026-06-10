"""端到端 DiT 推理管线 dit_inference 的单元测试。

用 mock 对象模拟 CVAE 编码、DiT Euler 积分、FeatureFusion、Decoder、gen2eq 各组件，
验证 dit_inference 函数正确串联所有步骤。
"""

import numpy as np
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_mock_model():
    """构造 mock VAESymbolicRegressor。"""
    model = MagicMock()

    prior_mu = torch.randn(1, 512)
    prior_logvar = torch.randn(1, 512)
    model.encode_only.return_value = (prior_mu, prior_logvar)

    src_enc = torch.randn(1, 200, 512)
    model.prepare_latent_for_decoder.return_value = src_enc

    max_len = 30
    generations = torch.randint(0, 50, (max_len, 1))
    gen_len = torch.tensor([max_len])
    model.generate_from_latent.return_value = (generations, gen_len)

    return model


def _make_mock_dit():
    """构造 mock GenSRDiT。"""
    dit = MagicMock()
    dit.return_value = torch.randn(1, 512)
    return dit


def _make_sample_data():
    """构造 (X, y) 测试数据。"""
    X = np.random.randn(50, 2).astype(np.float32)
    y = np.random.randn(50, 1).astype(np.float32)
    return X, y


def _make_mock_env_params():
    """构造 mock env 和 params。"""
    env = MagicMock()
    params = MagicMock()
    params.max_complexity = -1
    return env, params


def _patch_gen2eq_success():
    """构造 gen2eq 成功返回值的 patch。"""
    mock_tree = MagicMock()
    mock_tree.infix.return_value = "x_0 + sin(x_1)"

    return (True, "skeleton_stub", mock_tree, 5,
            np.random.randn(50, 1), np.random.randn(50, 1),
            0.01, 0.02,
            {"r2_zero": [0.95]}, {"r2_zero": [0.93]})


# --- Test 1: 返回 dict 包含 expression 和 r2 键 ---
def test_dit_inference_returns_dict_with_required_keys():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        result = dit_inference(X, y, env, params, model, dit)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "expression" in result, "Missing 'expression' key"
    assert "r2" in result, "Missing 'r2' key"


# --- Test 2: 返回的 expression 是非空字符串 ---
def test_dit_inference_expression_nonempty_string():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        result = dit_inference(X, y, env, params, model, dit)

    assert isinstance(result["expression"], str), \
        f"Expected str, got {type(result['expression'])}"
    assert len(result["expression"]) > 0, "Expression should not be empty"


# --- Test 3: 返回的 r2 是有限浮点数 ---
def test_dit_inference_r2_finite_float():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        result = dit_inference(X, y, env, params, model, dit)

    assert isinstance(result["r2"], float), f"Expected float, got {type(result['r2'])}"
    assert np.isfinite(result["r2"]), f"r2 should be finite, got {result['r2']}"


# --- Test 4: 函数调用 model.encode_only 并传入正确的 sample_to_learn 格式 ---
def test_dit_inference_calls_encode_only_with_correct_format():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        dit_inference(X, y, env, params, model, dit)

    model.encode_only.assert_called_once()
    call_args = model.encode_only.call_args[0][0]
    assert isinstance(call_args, dict), "encode_only should receive a dict"
    assert "X_scaled_to_fit" in call_args, "Missing 'X_scaled_to_fit' key"
    assert "Y_scaled_to_fit" in call_args, "Missing 'Y_scaled_to_fit' key"
    np.testing.assert_array_equal(call_args["X_scaled_to_fit"][0], X)
    np.testing.assert_array_equal(call_args["Y_scaled_to_fit"][0], y)


# --- Test 5: 函数调用 euler_inference，prior_mu 形状正确 ---
def test_dit_inference_calls_euler_inference():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        with patch("dit_train.pipeline.euler_inference",
                   return_value=torch.randn(1, 512)) as mock_euler:
            dit_inference(X, y, env, params, model, dit, num_steps=8)

    mock_euler.assert_called_once()
    call_args = mock_euler.call_args
    # euler_inference(dit, prior_mu, num_steps=8)
    assert call_args[0][0] is dit, "First arg should be dit"
    prior_mu_arg = call_args[0][1]
    assert prior_mu_arg.shape == (1, 512), \
        f"prior_mu shape should be (1, 512), got {prior_mu_arg.shape}"
    assert call_args[1]["num_steps"] == 8, "num_steps should be 8"


# --- Test 6: 函数调用 prepare_latent_for_decoder 使用 z_opt 和 prior_logvar ---
def test_dit_inference_calls_prepare_latent_with_zopt_and_logvar():
    from dit_train.pipeline import dit_inference

    X, y = _make_sample_data()
    model = _make_mock_model()
    dit = _make_mock_dit()
    env, params = _make_mock_env_params()

    z_opt_expected = torch.randn(1, 512)
    prior_logvar_expected = model.encode_only.return_value[1]

    with patch("dit_train.pipeline.gen2eq", return_value=_patch_gen2eq_success()):
        with patch("dit_train.pipeline.euler_inference", return_value=z_opt_expected):
            dit_inference(X, y, env, params, model, dit)

    model.prepare_latent_for_decoder.assert_called_once()
    call_args = model.prepare_latent_for_decoder.call_args[0]
    z_arg = call_args[0]
    logvar_arg = call_args[1]
    torch.testing.assert_close(z_arg, z_opt_expected)
    torch.testing.assert_close(logvar_arg, prior_logvar_expected)
