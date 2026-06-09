"""Flow Matching 推理：替代 CMA-ES 的 latent space 搜索。

接口与 lso_fit_es_covfromvae_fit 兼容，返回相同格式的 batch_results。
"""

import time

import numpy as np
import torch

from const_opt import compute_metrics
from LSO_fit import gen2eq
from symbolicregression.model.flow_matching import euler_solve, heun_solve


def fm_fit(sample_to_learn, env, params, model, batch_results, bag_number):
    """Flow Matching 推理：data → CVAE encoder → FM ODE → decode → BFGS → best。

    与 lso_fit_es_covfromvae_fit 相同的接口签名。
    """
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1, dimension)
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1, 1)
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1, dimension)
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1, 1)

    stored_skeletons = set()
    global_best = {
        'max_fitness': -float('inf'),
        'min_mse': float('inf'),
        'best_eq': 'NaN',
        'best_r2': 0,
        'best_complexity': -1,
    }

    start_time = time.time()

    # 1. CVAE 编码 → prior_mu (数据条件)
    with torch.no_grad():
        prior_mu, prior_logvar = model.encode_only({
            "X_scaled_to_fit": sample_to_learn["X_scaled_to_fit"],
            "Y_scaled_to_fit": sample_to_learn["Y_scaled_to_fit"],
        })

    fm_net = model.fm_net
    n_samples = params.fm_n_samples
    n_steps = params.fm_ode_steps
    latent_dim = prior_mu.shape[-1]

    # 2. FM 采样：noise → ODE → z_0
    with torch.no_grad():
        condition = prior_mu.expand(n_samples, latent_dim)
        z_1 = torch.randn(n_samples, latent_dim, device=prior_mu.device)

        if params.fm_solver == "heun":
            z_0_all = heun_solve(fm_net, z_1, condition, n_steps)
        else:
            z_0_all = euler_solve(fm_net, z_1, condition, n_steps)

    # 3. 逐个 decode + BFGS + 评估（复用 LSO_fit.gen2eq）
    logvar_expanded = prior_logvar.expand(n_samples, -1)
    dummy_latent = torch.zeros(1, 1)

    for i in range(n_samples):
        z_0_i = z_0_all[i:i+1]  # (1, D)

        with torch.no_grad():
            src_enc = model.prepare_latent_for_decoder(z_0_i, logvar_expanded[i:i+1])
            generations = model.generate_from_latent_sampling(src_enc)  # (beam_size, seq_len)

        generations = generations.cpu()

        for b in range(generations.shape[0]):
            gen_data = generations[b:b+1]  # (1, seq_len) — 保持 2D

            eq_outputs = gen2eq(env, params, dummy_latent, gen_data,
                                sample_to_learn, stored_skeletons)
            success, skeleton_candidate, tree, complexity, y, y_pred, \
                mse_fit, mse_pred, results_fit, results_predict = eq_outputs

            if not success:
                continue

            r2 = results_fit['r2_zero'][0]
            fitness = r2 - complexity / params.com_weight if r2 >= 0.5 else r2

            if fitness > global_best['max_fitness']:
                global_best['max_fitness'] = fitness
                global_best['min_mse'] = mse_fit
                global_best['best_eq'] = tree
                global_best['best_r2'] = r2
                global_best['best_complexity'] = complexity

            if skeleton_candidate is not None:
                stored_skeletons.add(skeleton_candidate.infix())

            # 早停
            if global_best['best_r2'] > params.lso_stop_r2:
                break

        if global_best['best_r2'] > params.lso_stop_r2:
            break

    optimization_duration = time.time() - start_time

    # 4. 组装 batch_results
    batch_results["final_predicted_tree"].extend([global_best['best_eq']])
    batch_results["final_fit_mse"].extend([global_best['min_mse']])
    batch_results["time"].extend([optimization_duration])

    best_eq = global_best['best_eq']
    if best_eq == 'NaN':
        metrics_keys = [
            'r2_zero', 'r2', 'accuracy_l1_biggio', 'accuracy_l1_1e-3',
            'accuracy_l1_1e-2', 'accuracy_l1_1e-1', '_complexity',
        ]
        for k in metrics_keys:
            batch_results[k + "_final_fit"] = [0.0]
            batch_results[k + "_final_predict"] = [0.0]
        return batch_results

    numexpr_fn = env.simplifier.tree_to_numexpr_fn(best_eq)
    y = numexpr_fn(x_gt)[:, 0].reshape(-1, 1)
    y_pred = numexpr_fn(x_gt_pred)[:, 0].reshape(-1, 1)
    y = np.clip(y, -1e150, 1e150)
    y_pred = np.clip(y_pred, -1e150, 1e150)

    complexity = len(best_eq.prefix().split(','))
    if params.max_complexity != -1 and complexity > params.max_complexity:
        raise ValueError(f"Complexity {complexity} > max {params.max_complexity}")

    results_fit = compute_metrics(
        {"true": [y_gt], "predicted": [y], "predicted_tree": [best_eq]},
        metrics=params.validation_metrics,
    )
    results_predict = compute_metrics(
        {"true": [y_gt_pred], "predicted": [y_pred], "predicted_tree": [best_eq]},
        metrics=params.validation_metrics,
    )

    for k, v in results_fit.items():
        batch_results[k + "_final_fit"].extend(v)
    for k, v in results_predict.items():
        batch_results[k + "_final_predict"].extend(v)

    return batch_results
