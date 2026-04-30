"""
隐空间 L2 范数与成对距离分布直方图

使用训练数据生成逻辑 (env.gen_expr) 生成 1000 个样本，
通过 CVAE 编码器映射到 512 维隐空间，统计 L2 范数分布和两两 L2 距离分布。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import chi2, kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 将项目根目录加入 sys.path，确保能导入根目录模块
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import symbolicregression
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer_vae import Trainer
from LSO_eval import reload_model
from model import VAESymbolicRegressor


if __name__ == "__main__":

    # ===== 参数设置（与 pmlb_batch_inference.py 一致） =====
    parser = argparse.ArgumentParser(
        description="Latent space distribution analysis",
        parents=[get_parser()],
    )
    parser.set_defaults(
        beam_size=2,
        model_type="vae",
        lso_optimizer="es_fromvae_fit",
        lso_stop_r2=0.95,
        compre_num=128,
        lso_stop_r2_abandon_lower=0,
        com_weight=400.0,
        ev_sigma=1.3,
        warmup_iteration=15,
        pop_init_index="0*1*1",
        pop_num=80,
        mu_num=16,
        lso_max_iteration=120,
        max_complexity=-1,
        n_trees_to_refine=2,
        max_input_points=200,
        wandb_disabled=True,
    )
    parser.add_argument("--model_path", type=str, default="weights/checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default="experiments/latent_space")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # ===== 设备设置 =====
    if not args.device.startswith("cuda"):
        raise ValueError(f"Requires a CUDA device, got {args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if args.device == "cuda":
        torch.cuda.set_device(0)
        args.device = "cuda:0"
    elif ":" in args.device:
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.reload_model_dir = str(model_path.parent)
    args.reload_model = model_path.name
    args.batch_size = 1
    args.batch_size_eval = args.batch_size_eval or int(1.5 * args.batch_size)
    args.n_steps_per_epoch = 100
    args.max_input_dimension = 10
    args.env_base_seed = 2023
    args.n_dec_layers = 16
    args.local_rank = 0
    args.master_port = -1
    args.random_state = 14423
    args.max_number_bags = 10
    args.eval_verbose_print = True
    args.rescale = True
    args.cpu = False
    args.device = torch.device(args.device)
    args.num_workers = 1
    args.eval_only = True
    args.is_slurm_job = False
    args.n_nodes = 1
    args.node_id = 0
    args.global_rank = 0
    args.world_size = 1
    args.n_gpu_per_node = 1
    args.is_master = True
    args.multi_node = False
    args.multi_gpu = False

    use_sample_pop, use_y_noise_pop, use_latent_noise_pop = args.pop_init_index.split("*")
    args.use_sample_pop = int(use_sample_pop)
    args.use_y_noise_pop = int(use_y_noise_pop)
    args.use_latent_noise_pop = int(use_latent_noise_pop)

    if "_rmse" not in args.validation_metrics.split(","):
        args.validation_metrics = args.validation_metrics + ",_rmse"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    symbolicregression.utils.CUDA = True

    # ===== 构建环境和模型 =====
    env = build_env(args)
    env.rng = np.random.RandomState(0)

    if args.eval_in_train_mode:
        modules = build_modules(env, args, mode="train")
    else:
        modules = build_modules(env, args, mode="eval")

    trainer = Trainer(modules, env, args)
    reload_model(trainer.modules, str(model_path))

    model = VAESymbolicRegressor(params=args, env=env, modules=trainer.modules)
    model.to(args.device)

    # ===== 生成样本并编码到隐空间 =====
    n_samples = args.n_samples
    all_latent_vectors = []

    for i in range(n_samples):
        expr, _ = env.gen_expr(train=True)

        # gen_expr 返回 "X_to_fit" / "Y_to_fit"，encode_only 期望 "X_scaled_to_fit" / "Y_scaled_to_fit"
        # 训练数据无需 StandardScaler 缩放，直接传递即可
        with torch.no_grad():
            prior_mu, prior_logvar = model.encode_only({
                "X_scaled_to_fit": expr["X_to_fit"],
                "Y_scaled_to_fit": expr["Y_to_fit"],
            })

        all_latent_vectors.append(prior_mu.squeeze(0).cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1}/{n_samples} samples")

    Z = np.stack(all_latent_vectors)  # shape [n_samples, 512]
    print(f"\nLatent matrix shape: {Z.shape}")

    # ===== 统计计算 =====
    d = Z.shape[1]  # 隐空间维度 512
    l2_norms = np.linalg.norm(Z, ord=2, axis=1)
    pairwise_distances = pdist(Z, metric="euclidean")

    # 估计各维度的平均方差，用于缩放理论分布
    # 若 Z ~ N(0, sigma^2 * I_d)，则 ||Z||^2 / sigma^2 ~ chi2(d)
    sigma_sq = np.mean(np.var(Z, axis=0, ddof=1))
    sigma = np.sqrt(sigma_sq)

    print(f"\n--- Latent Space Global Statistics ---")
    print(f"  Dimension:          {d}")
    print(f"  Estimated sigma^2:  {sigma_sq:.4f}")
    print(f"  Estimated sigma:    {sigma:.4f}")
    print(f"  E[||Z||^2] actual:  {np.mean(l2_norms ** 2):.4f}")
    print(f"  E[||Z||^2] theory:  {d * sigma_sq:.4f}  (d * sigma^2)")

    print(f"\n--- L2 Norm Statistics ---")
    print(f"  Mean:   {l2_norms.mean():.4f}  (theory: sigma*sqrt(2)*Gamma((d+1)/2)/Gamma(d/2) ≈ {sigma * np.sqrt(d - 0.5):.4f})")
    print(f"  Median: {np.median(l2_norms):.4f}")
    print(f"  Std:    {l2_norms.std():.4f}")
    print(f"  Min:    {l2_norms.min():.4f}")
    print(f"  Max:    {l2_norms.max():.4f}")

    print(f"\n--- Pairwise L2 Distance Statistics ---")
    print(f"  Number of pairs: {len(pairwise_distances)}")
    print(f"  Mean:   {pairwise_distances.mean():.4f}  (theory: sigma*sqrt(2)*sqrt(2)*Gamma((d+1)/2)/Gamma(d/2) ≈ {sigma * np.sqrt(2) * np.sqrt(d - 0.5):.4f})")
    print(f"  Median: {np.median(pairwise_distances):.4f}")
    print(f"  Std:    {pairwise_distances.std():.4f}")
    print(f"  Min:    {pairwise_distances.min():.4f}")
    print(f"  Max:    {pairwise_distances.max():.4f}")

    # ===== 卡方分布拟合检验 =====
    # 若 Z ~ N(0, sigma^2 * I_d)，则 ||Z||^2 / sigma^2 ~ chi2(d)
    scaled_norm_sq = l2_norms ** 2 / sigma_sq
    ks_norm, p_norm = kstest(scaled_norm_sq, chi2(df=d).cdf)
    print(f"\n--- Chi-squared Goodness-of-fit (||Z||^2/sigma^2 vs chi2({d})) ---")
    print(f"  KS statistic: {ks_norm:.6f}")
    print(f"  p-value:      {p_norm:.6e}")
    print(f"  Verdict:      {'Fail to reject H0 (consistent with scaled chi2)' if p_norm > 0.05 else 'Reject H0 (NOT consistent with scaled chi2)'}")

    # 若 Z_i, Z_j ~ N(0, sigma^2 * I_d)，则 ||Z_i - Z_j||^2 / (2*sigma^2) ~ chi2(d)
    scaled_pair_sq = pairwise_distances ** 2 / (2 * sigma_sq)
    ks_pair, p_pair = kstest(scaled_pair_sq, chi2(df=d).cdf)
    print(f"\n--- Chi-squared Goodness-of-fit (||Zi-Zj||^2/(2*sigma^2) vs chi2({d})) ---")
    print(f"  KS statistic: {ks_pair:.6f}")
    print(f"  p-value:      {p_pair:.6e}")
    print(f"  Verdict:      {'Fail to reject H0 (consistent with scaled chi2)' if p_pair > 0.05 else 'Reject H0 (NOT consistent with scaled chi2)'}")

    # ===== 绘图 =====
    # 图1：L2 范数分布 + 缩放 chi 分布理论曲线
    # 若 Z ~ N(0, sigma^2 * I_d)，则 ||Z|| ~ sigma * chi(d)
    # PDF: f(x) = (2*x/sigma^2) * chi2.pdf(x^2/sigma^2, d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(l2_norms, bins=50, density=True, edgecolor="black", alpha=0.75, label="Empirical")
    x_max = max(l2_norms.max(), np.sqrt(d) + 3) * 1.1
    x_theory = np.linspace(0, x_max, 300)
    chi_pdf = 2 * x_theory * chi2.pdf(x_theory ** 2, df=d)
    ax.plot(x_theory, chi_pdf, "r-", linewidth=2, label=f"Chi({d}) theoretical (prior N(0,I))")
    mean_val = l2_norms.mean()
    median_val = np.median(l2_norms)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median = {median_val:.2f}")
    ax.set_xlabel("L2 Norm", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"L2 Norm Distribution vs σ·Chi({d})", fontsize=14)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "l2_norm_distribution.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'l2_norm_distribution.pdf'}")

    # 图2：成对 L2 距离分布 + 理论曲线（缩放 chi 分布）
    # ||Z_i - Z_j|| ~ sigma * sqrt(2) * chi(d)
    # PDF: f(x) = (x / (sigma^2)) * chi2.pdf(x^2 / (2*sigma^2), d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairwise_distances, bins=50, density=True, edgecolor="black", alpha=0.75, label="Empirical")
    x_theory = np.linspace(pairwise_distances.min(), pairwise_distances.max(), 300)
    pair_pdf = (x_theory / sigma_sq) * chi2.pdf(x_theory ** 2 / (2 * sigma_sq), df=d)
    ax.plot(x_theory, pair_pdf, "r-", linewidth=2, label=f"σ√2·Chi({d}) theoretical (σ={sigma:.3f})")
    mean_val = pairwise_distances.mean()
    median_val = np.median(pairwise_distances)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median = {median_val:.2f}")
    ax.set_xlabel("Pairwise L2 Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Pairwise L2 Distance vs σ√2·Chi({d})", fontsize=14)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_l2_distance_distribution.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'pairwise_l2_distance_distribution.pdf'}")
