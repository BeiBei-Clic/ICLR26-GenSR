# Run

## Training

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
