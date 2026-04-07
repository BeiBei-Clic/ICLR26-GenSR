import argparse
import csv
import os
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import symbolicregression
import symbolicregression.model.utils_wrapper as utils_wrapper
from LSO_eval import read_file, reload_model
from LSO_fit import lso_fit_es_covfromvae_fit
from model import VAESymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer_vae import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference on local PMLB datasets",
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
        max_complexity=50,
        n_trees_to_refine=2,
        max_input_points=200,
        wandb_disabled=True,
    )
    parser.add_argument("--datasets_dir", type=str, default="pmlb/datasets")
    parser.add_argument("--summary_tsv", type=str, default="datasets/pmlb/pmlb/all_summary_stats.tsv")
    parser.add_argument("--model_path", type=str, default="weights/checkpoint.pth")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--dataset_limit", type=int, default=-1)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    args = parser.parse_args()

    if args.noise_strength < 0:
        raise ValueError(f"noise_strength must be non-negative, got {args.noise_strength}")

    if not args.device.startswith("cuda"):
        raise ValueError(f"Batch inference requires a CUDA device, got {args.device}")

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device} requested, but torch.cuda.is_available() is False")

    if args.device == "cuda":
        torch.cuda.set_device(0)
        args.device = "cuda:0"
    elif ":" in args.device:
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    else:
        raise ValueError(f"Unsupported device format: {args.device}")

    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_csv = Path(args.output_csv) if args.output_csv else Path(
        f"experiments/pmlb/results/pmlb_batch_inference_noise_{args.noise_strength:g}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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

    processed_datasets = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        processed_df = pd.read_csv(output_csv, usecols=["dataset", "status"])
        processed_datasets = set(processed_df["dataset"].astype(str).tolist())

    fieldnames = [
        "dataset",
        "status",
        "n_features",
        "refinement_type",
        "r2",
        "rmse",
        "complexity",
        "seconds",
        "error",
        "noise_strength",
        "expr",
    ]

    with output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if output_csv.stat().st_size == 0:
            writer.writeheader()
            handle.flush()

        for dataset_index, problem_name in enumerate(problem_names):
            if problem_name in processed_datasets:
                print(f"skip finished dataset: {problem_name}")
                continue

            row = {
                "dataset": problem_name,
                "status": "failed",
                "n_features": 0,
                "refinement_type": "lso",
                "r2": np.nan,
                "rmse": np.nan,
                "complexity": np.nan,
                "seconds": np.nan,
                "error": "",
                "noise_strength": args.noise_strength,
                "expr": "",
            }

            try:
                X, y, _ = read_file(
                    str(Path(args.datasets_dir) / problem_name / f"{problem_name}.tsv.gz")
                )
                if args.max_rows != -1:
                    X = X[:args.max_rows]
                    y = y[:args.max_rows]
                if X.shape[1] > args.max_input_dimension:
                    raise ValueError(
                        f"Input dimension {X.shape[1]} exceeds max_input_dimension {args.max_input_dimension}"
                    )
                y = np.expand_dims(y, -1)

                x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    shuffle=True,
                    random_state=args.random_state,
                )

                noise_rng = np.random.RandomState(args.noise_seed + dataset_index)
                noise = noise_rng.normal(0.0, args.noise_strength, size=y_to_fit.shape)
                y_to_fit = y_to_fit * (1.0 + noise)

                scaler = utils_wrapper.StandardScaler()
                X_scaled_to_fit = scaler.fit_transform(x_to_fit)

                sample_to_learn = {
                    "X_scaled_to_fit": [X_scaled_to_fit],
                    "Y_scaled_to_fit": [y_to_fit],
                    "x_to_fit": [x_to_fit],
                    "y_to_fit": [y_to_fit],
                    "x_to_predict": [x_to_predict],
                    "y_to_predict": [y_to_predict],
                    "eq_gt": [problem_name],
                }

                with torch.no_grad():
                    batch_results = lso_fit_es_covfromvae_fit(
                        sample_to_learn,
                        env,
                        args,
                        model,
                        defaultdict(list),
                        1,
                        es_strategy=args.es_strategy,
                    )

                result_df = pd.DataFrame.from_dict(
                    {k: v for k, v in batch_results.items() if k != "all_iteration_times"}
                )
                if result_df.empty:
                    raise ValueError(f"No result returned for dataset {problem_name}")

                expr = result_df.loc[0, "final_predicted_tree"]
                if pd.isna(expr):
                    raise ValueError(f"Final expression is NaN for dataset {problem_name}")

                numexpr_fn = env.simplifier.tree_to_numexpr_fn(expr)
                y_pred = numexpr_fn(x_to_predict)[:, 0].reshape(-1, 1)
                rmse = float(np.sqrt(np.mean((y_pred - y_to_predict) ** 2)))

                row["status"] = "success"
                row["n_features"] = int(X.shape[1])
                row["r2"] = float(result_df.loc[0, "r2_final_predict"])
                row["rmse"] = float(rmse)
                row["complexity"] = int(result_df.loc[0, "_complexity_final_predict"])
                row["seconds"] = float(result_df.loc[0, "time"])
                row["expr"] = str(expr)
            except Exception as error:
                row["error"] = str(error)

            writer.writerow(row)
            handle.flush()
            processed_datasets.add(problem_name)
            print(f"{problem_name}: {row['status']}")
