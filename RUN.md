# 运行指南

## 项目默认超参数（`parsers.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beam_size` | `10` | beam search 宽度 |
| `lso_max_iteration` | `50` | LSO 最大迭代数 |
| `warmup_iteration` | `11` | 预热迭代数 |
| `lso_stop_r2` | `0.995` | R² 达到此阈值提前停止 |
| `lso_stop_r2_abandon_lower` | `0` | 低于此 R² 放弃优化 |
| `lso_min_iteration` | `1` | LSO 最小迭代数 |
| `ev_sigma` | `0.01` | 进化策略 sigma |
| `pop_num` | `15` | 种群大小 |
| `mu_num` | `2` | 父代数量 |
| `max_complexity` | `50` | 复杂度上界（-1 不限制） |
| `com_weight` | `100.0` | 复杂度惩罚权重 |
| `compre_num` | `128` | 压缩维度 |
| `n_trees_to_refine` | `10` | 精炼树数量 |
| `max_input_points` | `200` | 单次评估最大数据点数 |
| `pop_init_index` | `1*0*0` | 种群初始化索引（sample/y_noise/latent_noise） |
| `model_type` | `vae` | 模型类型 |
| `lso_optimizer` | `es_fromvae_fit` | LSO 优化器 |
| `es_strategy` | `adaptive_gaussian` | 进化策略变体 |

## PMLB 批量推理

实验脚本（`pmlb_batch_inference.py`）在项目默认基础上做了覆盖：

| 参数 | 项目默认 | 实验脚本值 |
|------|----------|-----------|
| `beam_size` | 10 | **2** |
| `lso_max_iteration` | 50 | **120** |
| `warmup_iteration` | 11 | **15** |
| `lso_stop_r2` | 0.995 | **0.95** |
| `ev_sigma` | 0.01 | **1.3** |
| `pop_num` | 15 | **80** |
| `mu_num` | 2 | **16** |
| `max_complexity` | 50 | **-1** |
| `com_weight` | 100.0 | **400.0** |
| `n_trees_to_refine` | 10 | **2** |
| `pop_init_index` | `1*0*0` | **`0*1*1`** |

使用项目默认参数：

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --model_path weights/checkpoint.pth \
  --beam_size 10 \
  --lso_max_iteration 50 \
  --warmup_iteration 11 \
  --lso_stop_r2 0.9 \
  --ev_sigma 0.01 \
  --pop_num 15 \
  --mu_num 2 \
  --max_complexity 50 \
  --com_weight 100.0 \
  --n_trees_to_refine 10 \
  --pop_init_index 1*0*0 \
  --device cuda:0 \
  --noise_strength 0
```

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
