"""Flow Matching Stage 2 训练脚本。

用法:
    python train_fm.py \
        --cvae_checkpoint weights/checkpoint.pth \
        --fm_epochs 50 --fm_lr 1e-4 \
        --batch_size 64 --n_steps_per_epoch 2000 \
        --fm_save_periodic 5
"""

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import wandb

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.trainer_vae import Trainer
from symbolicregression.slurm import init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.optim import get_optimizer
from parsers import get_parser
from LSO_fit import reload_model


def save_fm_checkpoint(modules, fm_optimizer, epoch, loss, path, fm_module_raw=None):
    fm_state = fm_module_raw.state_dict() if fm_module_raw is not None else modules["flow_matching"].state_dict()
    data = {
        "flow_matching": fm_state,
        "fm_optimizer": fm_optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    # 保存 CVAE 模块以便独立加载
    for k in ["data_encoder", "cvae", "token_embed", "seq_decoder", "feature_fusion"]:
        if k in modules:
            data[k] = modules[k].state_dict()
    torch.save(data, path)
    print(f"FM checkpoint saved to {path} (epoch {epoch}, loss {loss:.6f})")


def load_fm_checkpoint(modules, fm_optimizer, path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    modules["flow_matching"].load_state_dict(data["flow_matching"])
    fm_optimizer.load_state_dict(data["fm_optimizer"])
    start_epoch = data.get("epoch", 0) + 1
    print(f"Resumed FM checkpoint from {path}, starting epoch {start_epoch}")
    return start_epoch


def main(params):
    init_distributed_mode(params)
    logger = initialize_exp(params)

    if not params.cpu:
        params.device = "cuda"
        assert torch.cuda.is_available()
    else:
        params.device = "cpu"
    symbolicregression.utils.CUDA = not params.cpu

    # Adjust batch size per GPU for DDP
    if params.multi_gpu:
        assert params.batch_size % params.world_size == 0, \
            f"batch_size ({params.batch_size}) must be divisible by world_size ({params.world_size})"
        params.batch_size = params.batch_size // params.world_size

    if params.wandb_disabled or not params.is_master:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=getattr(params, "wandb_project", "symbolic-regression-training"),
            name=getattr(params, "wandb_run_name", "fm-training"),
            group=getattr(params, "wandb_group_name", "fm"),
            config=vars(params),
        )

    env = build_env(params)
    modules = build_modules(env, params)

    # 加载 CVAE checkpoint（FM 模块会被跳过）
    cvae_ckpt = getattr(params, "cvae_checkpoint", "")
    if cvae_ckpt and os.path.isfile(cvae_ckpt):
        reload_model(modules, cvae_ckpt)
        print(f"Loaded CVAE from {cvae_ckpt}")
    elif params.reload_model and os.path.isfile(
        os.path.join(params.reload_model_dir, params.reload_model)
    ):
        ckpt_path = os.path.join(params.reload_model_dir, params.reload_model)
        reload_model(modules, ckpt_path)
        print(f"Loaded CVAE from {ckpt_path}")

    # 冻结 CVAE 相关模块
    for key in ["data_encoder", "cvae", "token_embed", "seq_decoder", "feature_fusion"]:
        for p in modules[key].parameters():
            p.requires_grad = False

    fm_params = list(modules["flow_matching"].parameters())
    fm_optimizer = get_optimizer(fm_params, params.fm_lr, params.optimizer)

    # DDP 包装
    fm_module_raw = modules["flow_matching"]  # 保存原始引用，用于保存 state_dict
    if params.multi_gpu:
        from torch.nn.parallel import DistributedDataParallel as DDP
        modules["flow_matching"] = DDP(
            modules["flow_matching"],
            device_ids=[params.local_rank],
            output_device=params.local_rank,
        )

    n_fm_params = sum(p.numel() for p in fm_params) / 1e6
    if params.is_master:
        print(f"Flow Matching trainable parameters: {n_fm_params:.4f}M")
        print(f"CVAE frozen. FM lr={params.fm_lr}, epochs={params.fm_epochs}")

    # 创建 Trainer 以复用数据管线
    trainer = Trainer(modules, env, params)

    # 恢复 FM checkpoint（如果存在）
    fm_ckpt_dir = Path(params.dump_path) / "fm_checkpoints"
    fm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    latest_ckpt = fm_ckpt_dir / "fm_latest.pth"
    if latest_ckpt.exists() and not params.fm_restart:
        start_epoch = load_fm_checkpoint(modules, fm_optimizer, str(latest_ckpt), params.device)

    task = params.tasks[0] if isinstance(params.tasks, (list, tuple)) else params.tasks

    best_loss = float("inf")

    fm_net_ddp = modules["flow_matching"]

    for epoch in range(start_epoch, params.fm_epochs):
        epoch_loss = 0.0
        n_steps = 0

        for step in range(params.n_steps_per_epoch):
            # 梯度累积：仅在同步步同步，其余步用 no_sync 跳过 all-reduce
            is_sync_step = (step + 1) % params.accumulate_gradients == 0
            if params.multi_gpu and not is_sync_step:
                context = fm_net_ddp.no_sync()
            else:
                context = nullcontext()

            with context:
                fm_loss = trainer.fm_train_step(task)
                if params.accumulate_gradients > 1:
                    fm_loss = fm_loss / params.accumulate_gradients
                fm_loss.backward()

            if is_sync_step:
                fm_optimizer.step()
                fm_optimizer.zero_grad()

            loss_val = fm_loss.item() * params.accumulate_gradients  # 恢复原始 loss 值
            epoch_loss += loss_val
            n_steps += 1

            if params.is_master and (step + 1) % params.print_freq == 0:
                avg_loss = epoch_loss / n_steps
                print(f"  Epoch {epoch} Step {step+1}/{params.n_steps_per_epoch}  "
                      f"FM loss: {loss_val:.6f}  avg: {avg_loss:.6f}")

        # 处理尾部不完整的累积步
        if params.n_steps_per_epoch % params.accumulate_gradients != 0:
            fm_optimizer.step()
            fm_optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / max(n_steps, 1)
        if params.is_master:
            print(f"Epoch {epoch} finished. Avg FM loss: {avg_epoch_loss:.6f}")

        if params.is_master:
            wandb.log({
                "fm/epoch": epoch,
                "fm/avg_loss": avg_epoch_loss,
                "fm/lr": fm_optimizer.param_groups[0]["lr"],
            })

        # 保存最新 checkpoint
        if params.is_master:
            save_fm_checkpoint(modules, fm_optimizer, epoch, avg_epoch_loss, str(latest_ckpt), fm_module_raw=fm_module_raw)

        # 定期保存
        if params.is_master and params.fm_save_periodic > 0 and (epoch + 1) % params.fm_save_periodic == 0:
            save_path = fm_ckpt_dir / f"fm_epoch{epoch:04d}.pth"
            save_fm_checkpoint(modules, fm_optimizer, epoch, avg_epoch_loss, str(save_path), fm_module_raw=fm_module_raw)

        # 保存 best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if params.is_master:
                best_path = fm_ckpt_dir / "fm_best.pth"
                save_fm_checkpoint(modules, fm_optimizer, epoch, avg_epoch_loss, str(best_path), fm_module_raw=fm_module_raw)

    wandb.finish()
    if params.is_master:
        print(f"\nTraining done. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--cvae_checkpoint", type=str, default="", help="Path to CVAE checkpoint")
    parser.add_argument("--fm_restart", action="store_true", help="Restart FM training from scratch")
    params = parser.parse_args()
    check_model_params(params)
    main(params)
