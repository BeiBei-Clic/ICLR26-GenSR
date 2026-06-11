"""DiT 批量评估脚本 — 在 PMLB 回归数据集上运行 dit_inference()，输出评估结果 CSV。"""

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import symbolicregression
import symbolicregression.model.utils_wrapper as utils_wrapper
from dit_train.pipeline import dit_inference
from LSO_eval import read_file, reload_model
from model import VAESymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiT evaluation on PMLB datasets",
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
    parser.add_argument("--datasets_dir", type=str, default="pmlb/datasets")
    parser.add_argument("--summary_tsv", type=str, default="datasets/pmlb/pmlb/all_summary_stats.tsv")
    parser.add_argument("--model_path", type=str, default="weights/checkpoint.pth")
    parser.add_argument("--dit_checkpoint", type=str, default="weights/fm_best.pt",
                        help="Path to DiT checkpoint (output of train_fm.py)")
    parser.add_argument("--output_csv", type=str, default="experiments/pmlb/GenSR_dit/pmlb_dit_results.csv")
    parser.add_argument("--dataset_limit", type=int, default=-1)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--lm_head_checkpoint", type=str, default="",
                        help="Path to fine-tuned lm_head checkpoint (optional)")
    args = parser.parse_args()

    # ---- 设备检查 ----
    if not args.device.startswith("cuda"):
        raise ValueError(f"DiT evaluation requires a CUDA device, got {args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device} requested, but torch.cuda.is_available() is False")

    if args.device == "cuda":
        torch.cuda.set_device(0)
        args.device = "cuda:0"
    elif ":" in args.device:
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    # ---- 路径检查 ----
    model_path = Path(args.model_path).resolve()
    assert model_path.is_file(), f"CVAE checkpoint not found: {model_path}"

    dit_path = Path(args.dit_checkpoint).resolve()
    assert dit_path.is_file(), f"DiT checkpoint not found: {dit_path}"

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- 参数设置（参照 pmlb_batch_inference.py） ----
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

    # ---- 构建模型 ----
    env = build_env(args)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, args, mode="eval")
    reload_model(modules, str(model_path))
    model = VAESymbolicRegressor(params=args, env=env, modules=modules)
    model.to(args.device)

    # 加载 DiT 模型（GenSRDiT，由 train_fm.py 训练）
    from dit_train.model import GenSRDiT
    from dit_train.train_fm import load_checkpoint
    dit = GenSRDiT().to(args.device)
    load_checkpoint(dit, str(dit_path), device=str(args.device))
    dit.eval()

    # 加载微调后的 lm_head 权重（可选）
    if args.lm_head_checkpoint:
        lm_head_path = Path(args.lm_head_checkpoint).resolve()
        assert lm_head_path.is_file(), f"lm_head checkpoint not found: {lm_head_path}"
        decoder = modules["seq_decoder"]
        decoder.lm_head.load_state_dict(torch.load(str(lm_head_path), map_location=args.device))
        print(f"Loaded fine-tuned lm_head from {lm_head_path}")

    # ---- 数据集发现 ----
    summary_df = pd.read_csv(args.summary_tsv, sep="\t")
    summary_df = summary_df[
        (summary_df["task"] == "regression")
        & (summary_df["n_categorical_features"] == 0)
    ]
    eligible_datasets = set(summary_df["dataset"].tolist())
    summary_task_map = dict(zip(summary_df["dataset"], summary_df["task"]))

    problem_names = []
    for dataset_dir in sorted(Path(args.datasets_dir).iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        if dataset_name not in eligible_datasets:
            continue
        metadata_path = dataset_dir / "metadata.yaml"
        metadata_task = None
        if metadata_path.is_file():
            for line in metadata_path.read_text().splitlines():
                if line.startswith("task:"):
                    metadata_task = line.split(":", 1)[1].strip()
                    break
        if metadata_task is None:
            metadata_task = summary_task_map.get(dataset_name)
        if metadata_task != "regression":
            continue
        problem_names.append(dataset_name)

    if args.dataset_limit != -1:
        problem_names = problem_names[:args.dataset_limit]

    # ---- 断点续传：检查已处理的数据集 ----
    processed_datasets = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        processed_df = pd.read_csv(output_csv, usecols=["dataset", "status"])
        processed_datasets = set(processed_df["dataset"].astype(str).tolist())

    # ---- CSV 字段 ----
    fieldnames = [
        "dataset",
        "status",
        "n_features",
        "r2",
        "complexity",
        "seconds",
        "error",
        "expr",
    ]

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    with output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            handle.flush()

        for problem_name in problem_names:
            if problem_name in processed_datasets:
                print(f"skip finished dataset: {problem_name}")
                continue

            row = {
                "dataset": problem_name,
                "status": "failed",
                "n_features": 0,
                "r2": np.nan,
                "complexity": np.nan,
                "seconds": np.nan,
                "error": "",
                "expr": "",
            }

            X, y, _ = read_file(
                str(Path(args.datasets_dir) / problem_name / f"{problem_name}.tsv.gz")
            )
            if args.max_rows != -1:
                X = X[:args.max_rows]
                y = y[:args.max_rows]
            if X.shape[1] > args.max_input_dimension:
                row["n_features"] = int(X.shape[1])
                row["error"] = f"Input dimension {X.shape[1]} exceeds max_input_dimension {args.max_input_dimension}"
                writer.writerow(row)
                handle.flush()
                processed_datasets.add(problem_name)
                print(f"{problem_name}: skipped (dim={X.shape[1]} > {args.max_input_dimension})")
                continue
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=args.random_state,
            )

            scaler = utils_wrapper.StandardScaler()
            X_scaled_to_fit = scaler.fit_transform(x_to_fit)

            t0 = time.time()
            result = dit_inference(
                X_scaled_to_fit, y_to_fit, env, args, model, dit, num_steps=args.num_steps
            )
            elapsed = time.time() - t0

            row["n_features"] = int(X.shape[1])
            row["seconds"] = round(elapsed, 3)

            if result["success"]:
                row["status"] = "success"
                row["r2"] = result["r2"]
                row["complexity"] = result["complexity"]
                row["expr"] = result["expression"]
            else:
                row["error"] = "dit_inference returned success=False"

            writer.writerow(row)
            handle.flush()
            processed_datasets.add(problem_name)
            print(f"{problem_name}: {row['status']} r2={row['r2']} complexity={row['complexity']} time={row['seconds']}s")

    # ---- 运行结束统计 ----
    result_df = pd.read_csv(output_csv)
    total = len(result_df)
    success_df = result_df[result_df["status"] == "success"]
    success_count = len(success_df)
    fail_count = total - success_count

    print("\n===== DiT Evaluation Summary =====")
    print(f"Total datasets: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    if success_count > 0:
        r2_values = success_df["r2"].dropna()
        complexity_values = success_df["complexity"].dropna()
        seconds_values = success_df["seconds"].dropna()

        avg_r2 = float(r2_values.mean())
        median_r2 = float(r2_values.median())
        success_rate_099 = float((r2_values > 0.99).sum()) / len(r2_values)
        avg_complexity = float(complexity_values.mean())
        avg_seconds = float(seconds_values.mean())

        print(f"Average R^2: {avg_r2:.6f}")
        print(f"Median R^2: {median_r2:.6f}")
        print(f"Success rate (R^2 > 0.99): {success_rate_099:.2%} ({int((r2_values > 0.99).sum())}/{len(r2_values)})")
        print(f"Average complexity: {avg_complexity:.1f}")
        print(f"Average inference time: {avg_seconds:.3f}s")
    else:
        print("No successful evaluations to summarize.")
    print("===================================")
