import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import symbolicregression
import symbolicregression.model.utils_wrapper as utils_wrapper
from model import VAESymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.metrics import compute_metrics
from symbolicregression.model import build_modules
from symbolicregression.slurm import init_distributed_mode, init_signal_handler


def reload_model(modules, path):
    assert os.path.isfile(path)

    if torch.cuda.is_available():
        data = torch.load(path, weights_only=False)
    else:
        data = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    for key, module in modules.items():
        if key not in data:
            if key == "feature_fusion" and "latent_bridge" in data:
                weights = data["latent_bridge"]
            elif key == "feature_fusion" and "mapper" in data:
                weights = data["mapper"]
            else:
                continue
        else:
            weights = data[key]

        try:
            module.load_state_dict(weights, strict=False)
        except RuntimeError:
            stripped = {
                name[len("module."):] if name.startswith("module.") else name: value
                for name, value in weights.items()
            }
            module.load_state_dict(stripped, strict=False)

        for parameter in module.parameters():
            parameter.requires_grad_(False)


def read_file(filename, label="target", sep=None):
    compression = "gzip" if filename.endswith("gz") else None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()

    params.batch_size = 1
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    params.n_steps_per_epoch = 100
    params.max_input_dimension = 10
    params.env_base_seed = 2023
    params.n_dec_layers = 16
    params.local_rank = -1
    params.master_port = -1
    params.random_state = 14423
    params.max_number_bags = 10
    params.eval_verbose_print = True
    params.rescale = True
    params.num_workers = 1
    params.eval_only = True

    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    if not params.cpu:
        assert torch.cuda.is_available()
        params.device = "cuda"
    else:
        params.device = "cpu"
    symbolicregression.utils.CUDA = not params.cpu

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params, mode="eval")

    model_path = os.path.join(params.reload_model_dir, params.reload_model)
    reload_model(modules, model_path)
    model = VAESymbolicRegressor(params=params, env=env, modules=modules)
    model.to(params.device)
    model.eval()

    all_datasets = pd.read_csv("./datasets/pmlb/pmlb/all_summary_stats.tsv", sep="\t")
    regression_datasets = all_datasets[all_datasets["task"] == "regression"]
    regression_datasets = regression_datasets[
        regression_datasets["n_categorical_features"] == 0
    ]

    if params.pmlb_data_type == "feynman":
        problems = regression_datasets[regression_datasets["dataset"].str.contains("feynman")]
    elif params.pmlb_data_type == "strogatz":
        problems = regression_datasets[regression_datasets["dataset"].str.contains("strogatz")]
    else:
        problems = regression_datasets[
            ~(
                regression_datasets["dataset"].str.contains("strogatz")
                | regression_datasets["dataset"].str.contains("feynman")
            )
        ]

    problems = problems.loc[problems["n_features"] < 11]
    problem_names = problems["dataset"].values.tolist()

    if params.feynman_sel_equs_num != -1:
        problem_names = problem_names[:params.feynman_sel_equs_num]

    feynman_formulas = {}
    pmlb_path = "./datasets/pmlb/datasets/"
    for metadata_path in Path(pmlb_path).glob("feynman_*/metadata.yaml"):
        formula = None
        with open(metadata_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped:
                    formula = stripped
                    break
        if formula is not None:
            feynman_formulas[metadata_path.parent.name] = formula

    wandb.init(mode=os.environ.get("WANDB_MODE", "disabled"))

    rng = np.random.RandomState(params.random_state)
    rows = []
    pbar = tqdm(total=len(problem_names))
    for counter, problem_name in enumerate(problem_names, 1):
        formula = feynman_formulas.get(problem_name, "???")
        print("Sample: ", counter)
        print("GT equation : ", formula)
        print("EQ: ", problem_name)

        X, y, _ = read_file(f"{pmlb_path}{problem_name}/{problem_name}.tsv.gz")
        y = np.expand_dims(y, -1)

        x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=params.random_state
        )

        scale = params.target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
        noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
        y_to_fit += noise

        scaler = utils_wrapper.StandardScaler() if params.rescale else None
        if scaler is not None:
            X_scaled_to_fit = scaler.fit_transform(x_to_fit)
            Y_scaled_to_fit = y_to_fit
        else:
            X_scaled_to_fit = x_to_fit
            Y_scaled_to_fit = y_to_fit

        if len(X_scaled_to_fit) >= params.max_input_points:
            fit_indices = rng.choice(len(X_scaled_to_fit), size=params.max_input_points, replace=False)
            X_scaled_for_model = X_scaled_to_fit[fit_indices]
            Y_scaled_for_model = Y_scaled_to_fit[fit_indices]
        else:
            X_scaled_for_model = X_scaled_to_fit
            Y_scaled_for_model = Y_scaled_to_fit

        sample_to_learn = {
            "X_scaled_to_fit": [X_scaled_for_model],
            "Y_scaled_to_fit": [Y_scaled_for_model],
            "x_to_fit": [x_to_fit],
            "y_to_fit": [y_to_fit],
            "x_to_predict": [x_to_predict],
            "y_to_predict": [y_to_predict],
            "eq_gt": [formula],
        }

        with torch.no_grad():
            _, generations, _ = model(sample_to_learn, params.max_generated_output_len)

        candidate_trees = []
        for generation in generations.cpu().tolist():
            candidate = env.idx_to_infix(generation[1:-1], is_float=False, str_array=False)
            if candidate is not None:
                candidate_trees.append(candidate)

        assert candidate_trees
        predicted_tree = candidate_trees[0]
        numexpr_fn = env.simplifier.tree_to_numexpr_fn(predicted_tree)
        y_fit = numexpr_fn(x_to_fit)[:, 0].reshape(-1, 1)
        y_predict = numexpr_fn(x_to_predict)[:, 0].reshape(-1, 1)

        results_fit = compute_metrics(
            {
                "true": [y_to_fit],
                "predicted": [y_fit],
                "predicted_tree": [predicted_tree],
            },
            metrics=params.validation_metrics,
        )
        results_predict = compute_metrics(
            {
                "true": [y_to_predict],
                "predicted": [y_predict],
                "predicted_tree": [predicted_tree],
            },
            metrics=params.validation_metrics,
        )

        row = {
            "problem": problem_name,
            "formula": formula,
            "generated_equation": predicted_tree.infix(),
            "r2_fit": results_fit["r2"][0],
            "r2_zero_fit": results_fit["r2_zero"][0],
            "r2_predict": results_predict["r2"][0],
            "r2_zero_predict": results_predict["r2_zero"][0],
            "complexity": len(predicted_tree.prefix().split(",")),
        }
        rows.append(row)
        print("Direct equation: ", row["generated_equation"])
        print("R2 zero fit: ", row["r2_zero_fit"])
        print("R2 zero predict: ", row["r2_zero_predict"])

        wandb.log(row, step=counter)
        pbar.update(1)

    output_path = "./eval_result/eval_pmlb_direct.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    pbar.close()
    wandb.finish()
