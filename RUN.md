# 运行指南

## VAE 训练

### 单卡训练

```bash
bash scripts/train.sh
```

### 多卡训练（单机多卡，DDP）

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py \
    --batch_size 64 \
    --accumulate_gradients 2 \
    --amp 0 \
    --dump_path ./dump \
    --max_input_dimension 10 \
    --exp_name vae \
    --exp_id run-train-multigpu \
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

> - `--nproc_per_node=4` 表示使用 4 张 GPU，根据实际 GPU 数量修改
> - batch_size 64 会自动均分到每张卡上（每卡 16）
> - `--master_port=29500` 可按需修改避免端口冲突

## DiT Flow Matching 训练

训练 GenSRDiT 学习 prior_mu → post_mu 速度场。训练数据由 CVAE 在线生成，CVAE/Decoder 全部冻结。

单卡训练：

```bash
python dit_train/train_fm.py \
  --num-iterations 100000 \
  --device-batch-size 4 \
  --grad-accum-steps 8 \
  --learning-rate 1e-4 \
  --output-dir dit_train/checkpoints \
  --eval-every 5000 \
  --save-every 10000 \
  --checkpoint-path weights/checkpoint.pth
```

多卡训练（单机多卡，DDP）：

```bash
torchrun --nproc_per_node=4 --master_port=29500 dit_train/train_fm.py \
  --num-iterations 100000 \
  --device-batch-size 20 \
  --grad-accum-steps 8 \
  --learning-rate 1e-4 \
  --output-dir dit_train/checkpoints \
  --eval-every 100 \
  --save-every 100 \
  --checkpoint-path weights/checkpoint.pth
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_iterations` | 5000 | 训练迭代数 |
| `device_batch_size` | 4 | 单卡 batch size |
| `grad_accum_steps` | 8 | 梯度累积步数（有效 batch = 4×8 = 32） |
| `learning_rate` | 1e-4 | AdamW 学习率 |
| `weight_decay` | 0.01 | AdamW 权重衰减 |
| `warmup_ratio` | 0.05 | 预热占比 |
| `warmdown_ratio` | 0.3 | 末尾衰减占比 |
| `timestep_dist` | logit_normal | 时间步采样分布 |
| `output_dir` | dit_train/checkpoints | checkpoint 保存目录 |
| `eval_every` | 500 | 评估间隔 |
| `eval_steps` | 20 | 评估步数 |
| `save_every` | 1000 | 保存间隔 |
| `checkpoint_path` | weights/checkpoint.pth | CVAE 预训练权重 |
| `resume` | (空) | 从 checkpoint 恢复训练 |

快速验证（调试用）：

```bash
python dit_train/train_fm.py \
  --num-iterations 100 \
  --device-batch-size 2 \
  --eval-every 50 \
  --save-every 100 \
  --checkpoint-path weights/checkpoint.pth
```

## PMLB 批量推理（CMA-ES 基线）

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

## DiT 评估（PMLB 数据集）

训练完成后，用 `dit_eval.py` 在全部 PMLB 回归数据集上运行 DiT 推理评估。

```bash
python experiments/pmlb/dit_eval.py \
  --dit_checkpoint dit_train/checkpoints/fm_best.pt \
  --model_path weights/checkpoint.pth \
  --output_csv experiments/pmlb/GenSR_dit/pmlb_dit_results.csv \
  --num_steps 16 \
  --device cuda:0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dit_checkpoint` | weights/fm_best.pt | DiT checkpoint 路径 |
| `model_path` | weights/checkpoint.pth | CVAE 预训练权重 |
| `output_csv` | experiments/pmlb/GenSR_dit/pmlb_dit_results.csv | 结果 CSV 路径 |
| `num_steps` | 16 | Euler 积分步数 |
| `dataset_limit` | -1 | 限制数据集数量（-1=全部） |
| `max_rows` | -1 | 限制每数据集行数 |
| `device` | cuda:0 | 运行设备 |

快速验证（2 个数据集）：

```bash
python experiments/pmlb/dit_eval.py \
  --dit_checkpoint dit_train/checkpoints/fm_best.pt \
  --dataset_limit 2 \
  --device cuda:0
```

## 隐空间分布分析

```bash
python experiments/latent_space/latent_distribution.py
```

## Decoder lm_head 微调

冻结 CVAE + DiT + FeatureFusion + Decoder blocks，只解冻 lm_head 输出投影层。用 DiT 传输后的 z_opt 作为 Decoder 输入，teacher-forcing CE loss 训练。每 50 步在验证集上评估，自动保存最优权重。

```bash
python dit_train/finetune_lm_head.py \
  --num-iterations 500 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --dit-checkpoint dit_train/checkpoints/fm_best.pt \
  --vae-checkpoint weights/checkpoint.pth \
  --dit-num-steps 16 \
  --num-workers 2 \
  --log-every 50
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_iterations` | 500 | 训练迭代数 |
| `batch_size` | 8 | 训练/验证 batch size |
| `learning_rate` | 1e-4 | AdamW 学习率 |
| `dit_checkpoint` | dit_train/checkpoints/fm_best.pt | DiT FM checkpoint |
| `vae_checkpoint` | weights/checkpoint.pth | CVAE 预训练权重 |
| `output_dir` | dit_train/checkpoints | 最优权重保存目录 |
| `dit_num_steps` | 16 | DiT Euler 积分步数 |
| `num_workers` | 2 | 数据预加载 worker 数 |
| `log_every` | 50 | 训练 loss 日志间隔（也是验证间隔） |

多卡训练（单机多卡，DDP）：

```bash
torchrun --nproc_per_node=4 --master_port=29500 dit_train/finetune_lm_head.py \
  --num-iterations 500 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --dit-checkpoint dit_train/checkpoints/fm_best.pt \
  --vae-checkpoint weights/checkpoint.pth \
  --dit-num-steps 16 \
  --num-workers 2 \
  --log-every 50
```

> - `--nproc_per_node=4` 表示使用 4 张 GPU，根据实际 GPU 数量修改
> - 验证循环和权重保存只在 rank 0 执行，其他 rank 通过 barrier 同步
> - 最优权重保存到 `dit_train/checkpoints/lm_head_best.pt`

快速验证（100 步）：

```bash
python dit_train/finetune_lm_head.py \
  --num-iterations 100 \
  --batch-size 8 \
  --log-every 10
```
