#!/bin/bash

set -euo pipefail

# GPU 数量，默认单卡。使用方式: N_GPU=2 bash scripts/train_fm.sh
N_GPU=${N_GPU:-1}
GPUS=$(seq -s, 0 $((N_GPU - 1)))

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --standalone \
    --nproc_per_node=$N_GPU \
    ./train_fm.py \
    --cvae_checkpoint weights/checkpoint.pth \
    --max_input_dimension 10 \
    --batch_size 128 \
    --num_workers 8 \
    --accumulate_gradients 2 \
    --amp 0 \
    --dump_path ./dump \
    --exp_name fm \
    --exp_id run-fm \
    --latent_dim 512 \
    --d_model 512 \
    --d_input 512 \
    --enc_emb_dim 512 \
    --dec_emb_dim 512 \
    --n_enc_layers 8 \
    --n_dec_layers 16 \
    --n_enc_heads 16 \
    --n_dec_heads 16 \
    --n_steps_per_epoch 2000 \
    --max_epoch 1 \
    --kl_limits 1.0 \
    --model_type vae \
    --lr 1e-3 \
    --fm_lr 1e-4 \
    --fm_epochs 50 \
    --fm_hidden_dim 1024 \
    --fm_n_layers 6 \
    --fm_n_samples 16 \
    --fm_ode_steps 10 \
    --fm_save_periodic 5 \
    --print_freq 100 \
    --save_periodic 25 \
    --wandb_project "symbolic-regression-training" \
    --wandb_group_name "fm-training" \
    --wandb_run_name "fm-stage2"
