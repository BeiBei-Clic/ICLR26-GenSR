#!/bin/bash

MODEL_DIR="./weights"
MODEL_CHECKPOINT="checkpoint.pth"

DATA_TYPE="feynman"

CUDA_VISIBLE_DEVICES=0 python -u ./direct_eval.py --reload_model_dir "${MODEL_DIR}" --reload_model "${MODEL_CHECKPOINT}" --pmlb_data_type "${DATA_TYPE}" --target_noise 0.0 --max_input_points 200 --beam_size 2 --model_type vae --feynman_sel_equs_num -1
