"""训练数据质量验证脚本：收集 1000 样本 (prior_mu, post_mu)，计算基础统计、KL 散度、diff norm，自动 PASS/FAIL 判断。
用法: python -m dit_train.data.verify_dataset
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parsers import get_parser
from dit_train.data.latent_dataset import create_latent_dataloader


def main():
    # ===== 阈值常量 =====
    KL_THRESHOLD = 0.01
    DIFF_NORM_THRESHOLD = 0.1
    n_samples = 1000
    batch_size = 32

    # ===== 初始化 =====
    params = get_parser().parse_args(["--max_input_dimension", "10"])
    params.device = "cuda"
    dataloader = create_latent_dataloader(params, batch_size=batch_size)

    # ===== 数据收集 =====
    prior_list = []
    post_list = []
    collected = 0
    for prior_mu, post_mu in dataloader:
        batch_n = prior_mu.shape[0]
        remaining = n_samples - collected
        take = min(batch_n, remaining)
        prior_list.append(prior_mu[:take].cpu().numpy())
        post_list.append(post_mu[:take].cpu().numpy())
        collected += take
        print(f"  收集 {collected}/{n_samples}")
        if collected >= n_samples:
            break

    prior_all = np.concatenate(prior_list, axis=0)
    post_all = np.concatenate(post_list, axis=0)
    print(f"\n共收集 {len(prior_all)} 个样本，形状 {prior_all.shape}")

    # ===== 分布可视化 =====
    diff = post_all - prior_all
    diff_norms = np.linalg.norm(diff, axis=1)

    # 逐维度统计（同时用于可视化和 KL 计算）
    prior_dim_mean = prior_all.mean(axis=0)  # (512,)
    prior_dim_std = prior_all.std(axis=0)
    post_dim_mean = post_all.mean(axis=0)
    post_dim_std = post_all.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Latent Distribution Validation (n=1000)")

    axes[0].hist(prior_dim_mean, bins=50)
    axes[0].set_title("prior_mu dimension means")
    axes[0].set_xlabel("mean value")
    axes[0].set_ylabel("count")

    axes[1].hist(post_dim_mean, bins=50)
    axes[1].set_title("post_mu dimension means")
    axes[1].set_xlabel("mean value")
    axes[1].set_ylabel("count")

    # 子图 3: diff norm 直方图
    axes[2].hist(diff_norms, bins=50)
    axes[2].set_title("||post_mu - prior_mu|| distribution")
    axes[2].set_xlabel("L2 norm")
    axes[2].set_ylabel("count")

    plt.tight_layout()
    pdf_path = os.path.join(os.path.dirname(__file__), "latent_distribution.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  分布可视化已保存: {pdf_path}")

    # ===== NaN / Inf 检测 =====
    has_nan_prior = np.any(np.isnan(prior_all))
    has_nan_post = np.any(np.isnan(post_all))
    has_inf_prior = np.any(np.isinf(prior_all))
    has_inf_post = np.any(np.isinf(post_all))
    has_nan_or_inf = has_nan_prior or has_nan_post or has_inf_prior or has_inf_post

    print("\n===== NaN / Inf 检测 =====")
    print(f"  prior_mu NaN: {has_nan_prior}, Inf: {has_inf_prior}")
    print(f"  post_mu  NaN: {has_nan_post}, Inf: {has_inf_post}")

    # ===== 基础统计 =====
    print("\n===== 基础统计 =====")
    print(f"  prior_mu  mean={prior_all.mean():.6f}, std={prior_all.std():.6f}, min={prior_all.min():.6f}, max={prior_all.max():.6f}")
    print(f"  post_mu   mean={post_all.mean():.6f}, std={post_all.std():.6f}, min={post_all.min():.6f}, max={post_all.max():.6f}")

    # ===== 逐维度 KL 散度（KL(posterior || prior)） =====
    # prior_dim_mean/std, post_dim_mean/std 已在上面计算
    # KL(N1||N2) = log(std2/std1) + (std1^2 + (mean1-mean2)^2) / (2*std2^2) - 0.5
    # N1 = posterior 每维分布, N2 = prior 每维分布
    kl_per_dim = np.log(prior_dim_std / post_dim_std) + \
                 (post_dim_std**2 + (post_dim_mean - prior_dim_mean)**2) / (2 * prior_dim_std**2) - 0.5
    kl_mean = kl_per_dim.mean()
    kl_std = kl_per_dim.std()

    print("\n===== KL 散度统计（逐维度 KL(posterior || prior)） =====")
    print(f"  KL 散度均值: {kl_mean:.6f}")
    print(f"  KL 散度标准差: {kl_std:.6f}")
    print(f"  KL 散度 min: {kl_per_dim.min():.6f}, max: {kl_per_dim.max():.6f}")

    # ===== diff norm 统计 =====
    # diff 和 diff_norms 已在可视化部分计算
    diff_norm_mean = diff_norms.mean()
    diff_norm_std = diff_norms.std()

    print("\n===== diff norm 统计 =====")
    print(f"  ||post_mu - prior_mu|| 均值: {diff_norm_mean:.6f}")
    print(f"  ||post_mu - prior_mu|| 标准差: {diff_norm_std:.6f}")
    print(f"  ||post_mu - prior_mu|| min: {diff_norms.min():.6f}, max: {diff_norms.max():.6f}")

    # ===== 自动 PASS / FAIL 判断 =====
    print("\n===== 验证结果 =====")
    all_passed = True

    if has_nan_or_inf:
        print("  [FAIL] 存在 NaN 或 Inf 值")
        all_passed = False
    else:
        print("  [OK] 无 NaN / Inf 值")

    if kl_mean > KL_THRESHOLD:
        print(f"  [OK] KL 散度均值 {kl_mean:.6f} > {KL_THRESHOLD}")
    else:
        print(f"  [FAIL] KL 散度均值 {kl_mean:.6f} <= {KL_THRESHOLD}（可能 posterior collapse）")
        all_passed = False

    if diff_norm_mean > DIFF_NORM_THRESHOLD:
        print(f"  [OK] diff norm 均值 {diff_norm_mean:.6f} > {DIFF_NORM_THRESHOLD}")
    else:
        print(f"  [FAIL] diff norm 均值 {diff_norm_mean:.6f} <= {DIFF_NORM_THRESHOLD}（prior 和 posterior 差异太小）")
        all_passed = False

    if all_passed:
        print("\n>>> PASS: 训练数据质量验证通过 <<<")
    else:
        print("\n>>> FAIL: 训练数据质量验证未通过，请检查上述失败项 <<<")


if __name__ == "__main__":
    main()
