#!/bin/bash

set -euo pipefail

# Flow Matching 批量推理脚本
# 用法: bash scripts/fm_batch_inference.sh [GPU_ID] [FM_CHECKPOINT]

GPU_ID="${1:-0}"
FM_CKPT="${2:-weights/fm_best.pth}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u experiments/pmlb/pmlb_batch_inference.py \
    --inference_mode flow_matching \
    --fm_checkpoint "${FM_CKPT}" \
    --model_path weights/checkpoint.pth \
    --device "cuda:${GPU_ID}" \
    --datasets_dir pmlb/datasets \
    --max_input_dimension 10 \
    --beam_size 2 \
    --model_type vae \
    --com_weight 400 \
    --lso_stop_r2 0.95 \
    --fm_n_samples 16 \
    --fm_ode_steps 10 \
    --fm_solver euler \
    --wandb_disabled \
    --output_csv "experiments/pmlb/results/pmlb_flow_matching.csv"
