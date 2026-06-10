"""端到端 DiT 推理管线。

输入 (X, Y) → CVAE 编码 → DiT Euler 积分 → FeatureFusion → Decoder 解码 → BFGS 常数优化 → 输出 (expression, R^2)
"""

import torch

from dit_train.inference import euler_inference
from LSO_fit import gen2eq


def dit_inference(X, y, env, params, model, dit, num_steps=16):
    """端到端 DiT 推理管线。

    Args:
        X: (N, D) numpy array，输入特征
        y: (N, 1) numpy array，目标值
        env: 符号回归环境
        params: 参数对象
        model: VAESymbolicRegressor 实例（已加载权重）
        dit: GenSRDiT 实例（已加载 checkpoint）
        num_steps: Euler 积分步数，默认 16
    Returns:
        dict: {
            "success": bool,
            "expression": str,  # 表达式字符串
            "r2": float,        # R^2 拟合值
            "complexity": int,  # 表达式复杂度
            "tree": object,     # 表达式树
        }
    """
    # Step 1 - 数据预处理
    sample_to_learn = {
        "X_scaled_to_fit": [X],
        "Y_scaled_to_fit": [y],
        "x_to_fit": [X],
        "y_to_fit": [y],
        "x_to_predict": [X],
        "y_to_predict": [y],
    }

    with torch.no_grad():
        # Step 2 - CVAE 编码
        prior_mu, prior_logvar = model.encode_only(sample_to_learn)

        # Step 3 - Euler 积分
        z_opt = euler_inference(dit, prior_mu, num_steps=num_steps)

        # Step 4 - FeatureFusion（z_opt 作为 mu，logvar 使用 prior_logvar）
        src_enc = model.prepare_latent_for_decoder(z_opt, prior_logvar)

        # Step 5 - Decoder 解码
        generations, gen_len = model.generate_from_latent(src_enc)

    # Step 6 - gen2eq 表达式生成 + BFGS 优化
    eq_outputs = gen2eq(env, params, z_opt, generations,
                        sample_to_learn, set())
    success, _, tree, complexity, _, _, _, _, results_fit, _ = eq_outputs

    if success:
        return {
            "success": True,
            "expression": tree.infix(),
            "r2": float(results_fit["r2_zero"][0]),
            "complexity": complexity,
            "tree": tree,
        }
    else:
        return {
            "success": False,
            "expression": "",
            "r2": 0.0,
            "complexity": -1,
            "tree": None,
        }
