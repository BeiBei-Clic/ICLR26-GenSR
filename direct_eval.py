import datetime
import os

import numpy as np
import torch
import wandb

import symbolicregression
from LSO_eval import evaluate_pmlb_lso, extract_info
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.slurm import init_distributed_mode, init_signal_handler
from symbolicregression.trainer_vae import Trainer


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()

    use_sample_pop, use_y_noise_pop, use_latent_noise_pop = params.pop_init_index.split("*")
    params.use_sample_pop = int(use_sample_pop)
    params.use_y_noise_pop = int(use_y_noise_pop)
    params.use_latent_noise_pop = int(use_latent_noise_pop)

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
    params.n_trees_to_refine = params.beam_size

    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    params.num_workers = 1

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    if params.eval_in_train_mode:
        modules = build_modules(env, params, mode="train")
    else:
        modules = build_modules(env, params, mode="eval")

    trainer = Trainer(modules, env, params)

    target_noise = params.target_noise
    random_state = params.random_state
    data_type = params.pmlb_data_type
    save = params.save_results

    if data_type == "feynman":
        filter_fn = lambda x: x["dataset"].str.contains("feynman")
    elif data_type == "strogatz":
        print("Strogatz data")
        filter_fn = lambda x: x["dataset"].str.contains("strogatz")
    else:
        filter_fn = lambda x: ~(
            x["dataset"].str.contains("strogatz")
            | x["dataset"].str.contains("feynman")
        )

    group_name = "{}_{}_be{}_it{}_st{}_pn{}_mn{}_s{}_n{}".format(
        params.lso_optimizer,
        params.model_type,
        params.beam_size,
        params.lso_max_iteration,
        params.lso_stop_r2,
        params.pop_num,
        params.mu_num,
        params.ev_sigma,
        params.target_noise,
    )

    params.train_info, params.train_period = extract_info(params.reload_model)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_group_name = getattr(params, "wandb_group_name", None) or f"{group_name}"
    wandb_run_name = getattr(params, "wandb_run_name", None) or f"eval-{params.pmlb_data_type}-seed{params.seed}-{current_time}"
    wandb_project = getattr(params, "wandb_project", None) or "symbolic-regression-lso"

    if params.wandb_disabled:
        wandb.init(mode="disabled")
    elif getattr(params, "wandb_resume_id", ""):
        wandb.init(
            project=wandb_project,
            id=params.wandb_resume_id,
            resume="must",
            config=params,
        )
    else:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group_name,
            config=params,
        )

    save_dir = f"./eval_result/noise/res_{wandb_group_name}/{wandb_run_name}.csv"

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    evaluate_pmlb_lso(
        trainer,
        params,
        target_noise=target_noise,
        verbose=params.eval_verbose_print,
        random_state=random_state,
        save=save,
        filter_fn=filter_fn,
        save_file=None,
        save_suffix=save_dir,
    )

    wandb.finish()
