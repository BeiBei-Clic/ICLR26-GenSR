# 运行指南

## PMLB 批量推理

`pmlb_batch_inference.py` 的默认值按 `main` 分支 `scripts/eval.sh` 的 PMLB 评测配置设置。下表和下面的显式运行命令逐项一致。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `weights/checkpoint.pth` | 预训练权重路径 |
| `beam_size` | `2` | beam search 宽度 |
| `model_type` | `vae` | 模型类型 |
| `lso_optimizer` | `es_fromvae_fit` | LSO 优化器 |
| `lso_stop_r2` | `0.95` | R² 达到此阈值提前停止 |
| `lso_stop_r2_abandon_lower` | `0` | 低于此 R² 放弃优化 |
| `compre_num` | `128` | 压缩维度 |
| `com_weight` | `400.0` | 复杂度惩罚权重 |
| `ev_sigma` | `1.3` | 进化策略 sigma |
| `warmup_iteration` | `15` | 预热迭代数 |
| `pop_init_index` | `0*1*1` | 种群初始化索引（sample/y_noise/latent_noise） |
| `pop_num` | `80` | 种群大小 |
| `mu_num` | `16` | 父代数量 |
| `lso_max_iteration` | `120` | LSO 最大迭代数 |
| `max_complexity` | `-1` | 实验分支新增参数；关闭硬复杂度限制以对齐 `main` |
| `n_trees_to_refine` | `2` | `main` 中由 `beam_size` 推导得到 |
| `max_input_points` | `200` | 单次评估最大数据点数 |
| `device` | `cuda:0` | 运行设备 |
| `noise_strength` | `0` | 输出噪声强度，对应 `main` 的 `target_noise=0.0` |

使用 `main` 分支评测默认参数：

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --model_path weights/checkpoint.pth \
  --beam_size 2 \
  --model_type vae \
  --lso_optimizer es_fromvae_fit \
  --lso_stop_r2 0.9 \
  --lso_stop_r2_abandon_lower 0 \
  --compre_num 128 \
  --com_weight 400.0 \
  --ev_sigma 1.3 \
  --warmup_iteration 15 \
  --pop_init_index 0*1*1 \
  --pop_num 80 \
  --mu_num 16 \
  --lso_max_iteration 50 \
  --max_complexity -1 \
  --n_trees_to_refine 2 \
  --max_input_points 200 \
  --device cuda:0 \
  --noise_strength 0
```

上面命令显式暴露了批量推理的主要参数接口；这些取值与 `pmlb_batch_inference.py` 中写入的默认值保持一致，并对齐 `main` 分支 `scripts/eval.sh` 的 PMLB 评测配置。

以下命令是覆盖默认值的自定义运行示例。

快速运行（低搜索强度）：

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --model_path weights/checkpoint.pth \
  --max_rows 200 \
  --max_input_points 200 \
  --pop_num 24 \
  --mu_num 6 \
  --beam_size 10 \
  --lso_max_iteration 5 \
  --max_complexity 50 \
  --device cuda:0 \
  --noise_strength 0.1
```

提高搜索强度（结果偏弱时使用）：

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --model_path weights/checkpoint.pth \
  --max_rows 200 \
  --max_input_points 200 \
  --pop_num 15 \
  --mu_num 2 \
  --beam_size 10 \
  --lso_max_iteration 10 \
  --device cuda:0 \
  --noise_strength 0
```

## 预训练（CVAE Stage 1）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size 64 \
  --accumulate_gradients 2 \
  --amp 0 \
  --dump_path ./dump \
  --max_input_dimension 10 \
  --exp_name vae \
  --exp_id run-train \
  --lr 1e-3 \
  --latent_dim 512 \
  --save_periodic 10 \
  --n_steps_per_epoch 2000 \
  --max_epoch 500 \
  --kl_limits 1.0 \
  --model_type vae \
  --wandb_disabled \
  --print_freq 100
```

## Flow Matching 训练（Stage 2）

冻结 CVAE，只训练 Flow Matching 先验网络。需先完成预训练。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
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
  --fm_val_freq 5 \
  --fm_val_steps 200 \
  --print_freq 100 \
  --save_periodic 25 \
  --wandb_disabled
```

从已有 checkpoint 接续训练（不加 `--fm_restart` 即自动从 `fm_latest.pth` 恢复）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
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
  --fm_val_freq 5 \
  --fm_val_steps 200 \
  --print_freq 100 \
  --save_periodic 25 \
  --wandb_disabled
```

或使用封装好的脚本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_fm.sh
```

低资源快速验证（CPU / 少步数）：

```bash
python train_fm.py \
  --cvae_checkpoint weights/checkpoint.pth \
  --cpu True \
  --wandb_disabled \
  --env_name functions \
  --max_input_dimension 10 \
  --fm_epochs 1 \
  --fm_lr 1e-4 \
  --n_steps_per_epoch 10 \
  --print_freq 1 \
  --batch_size 4 \
  --latent_dim 512 \
  --d_model 512 \
  --d_input 512 \
  --enc_emb_dim 512 \
  --dec_emb_dim 512 \
  --n_enc_layers 8 \
  --n_dec_layers 16 \
  --n_enc_heads 16 \
  --n_dec_heads 16 \
  --max_epoch 1 \
  --dump_path ./dump \
  --fm_restart
```

## Flow Matching 批量推理

需先完成 FM 训练并得到 `fm_best.pth`。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pmlb/pmlb_batch_inference.py \
  --inference_mode flow_matching \
  --fm_checkpoint weights/fm_best.pth \
  --model_path weights/checkpoint.pth \
  --device cuda:0 \
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
  --output_csv experiments/pmlb/results/pmlb_flow_matching.csv
```

或使用脚本：

```bash
bash scripts/fm_batch_inference.sh 0 weights/fm_best.pth
```

## 隐空间分布分析

```bash
python experiments/latent_space/latent_distribution.py
```
