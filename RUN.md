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

## 隐空间分布分析

```bash
python experiments/latent_space/latent_distribution.py
```
