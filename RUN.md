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

这组参数比 `--lso_max_iteration 2` 更有搜索能力，但没有直接恢复项目默认的重配置。

- `lso_max_iteration` 提到 `8`，先补足搜索轮数
- `pop_num 24`、`mu_num 6`，控制每轮搜索宽度，避免时间暴涨
- `beam_size 1`，减少每个候选的生成开销
- `max_input_points 128`，进一步压住单数据集成本
- `max_complexitiy 50`，复杂度限制

如果这组结果还是偏弱，优先继续试：

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
