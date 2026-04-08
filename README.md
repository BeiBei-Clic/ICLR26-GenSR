<div align="center">
  <h2><b> (ICLR'26) GenSR: Symbolic Regression Based on Equation Generative Space </b></h2>
</div>

The official implementation of our ICLR-2026 paper "**GenSR: Symbolic Regression Based on Equation Generative Space**" [[OpenReview](https://openreview.net/forum?id=8emIjwUQZg)] [[WandB](https://wandb.ai/yuxiao-hu-the-hong-kong-polytechnic-university/ICLR26-GenSR)].

```
@inproceedings{
li2026gensr,
title={Gen{SR}: Symbolic regression based on equation generative space},
author={Qian Li and Yuxiao Hu and Juncheng Liu and Yuntian Chen},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=8emIjwUQZg}
}
```

## Introduction

<p align="center">
<img src="./framework.png" height = "360" alt="" align=center />
</p>

GenSR constructs an **equation generative space** via a Conditional Variational Autoencoder (CVAE), where each point in the latent space maps to a symbolic equation. At inference time, a **degraded CMA-ES** searches this space to find equations that best fit the observed data, with constants refined via BFGS. Specifically, GenSR first pretrains a dual-branch Conditional Variational Autoencoder (CVAE) to reparameterize symbolic equations into a generative latent space with symbolic continuity and local numerical smoothness. At inference, the CVAE coarsely localizes the input data to promising regions in the latent space. Then, a modified CMA-ES refines the candi- date region, leveraging smooth latent gradients.

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Pretrained Weights

Download the checkpoint from [Google Drive](https://drive.google.com/file/d/1TbcRSzO3rGQBuJIPN5P6fQpYYEv4__K4/view?usp=sharing) and place it in `weights/`:

```
weights/checkpoint.pth
```

Or bootstrap it directly inside the repo:

```bash
bash scripts/bootstrap_pretrained.sh
```

### Training

```bash
bash scripts/train.sh
```

### Evaluation

```bash
bash scripts/eval.sh
```

Change `DATA_TYPE` in the script to evaluate on different SRBench datasets: `feynman`, `strogatz`, or `blackbox`.

You can also run evaluation directly from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python -u ./direct_eval.py \
  --reload_model_dir "./weights" \
  --reload_model "checkpoint.pth" \
  --eval_lso_on_pmlb True \
  --pmlb_data_type "feynman" \
  --target_noise 0.0 \
  --max_input_points 200 \
  --lso_optimizer es_fromvae_fit \
  --beam_size 2 \
  --model_type vae \
  --lso_stop_r2 0.95 \
  --compre_num 128 \
  --lso_stop_r2_abandon_lower 0 \
  --feynman_sel_equs_num -1 \
  --com_weight 400 \
  --ev_sigma 1.3 \
  --warmup_iteration 15 \
  --pop_init_index 0*1*1 \
  --num_eval_workers 16 \
  --wandb_project symbolic-regression-example \
  --wandb_group_name "eval-feynman" \
  --wandb_run_name "pop80_iter120" \
  --pop_num 80 \
  --mu_num 16 \
  --lso_max_iteration 120
```

By default, inference uses a hard complexity upper bound of `50`. To override it, add:

```bash
--max_complexity 40
```

In this project, complexity is computed as:

```python
len(tree.prefix().split(","))
```

That is, the number of tokens / nodes in the prefix representation of the expression tree. Candidates with complexity larger than `max_complexity` are discarded directly during inference.

You can also batch-run the local PMLB datasets with the same LSO pipeline on GPU:

无噪声 smoke 测试，先快速确认整条批量链路和 CSV 落盘正常。

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:0 \
  --model_path weights/checkpoint.pth \
  --data_type feynman \
  --dataset_limit 2 \
  --max_rows 256 \
  --max_input_points 128 \
  --pop_num 20 \
  --mu_num 4 \
  --lso_max_iteration 30
```

默认全量无噪声批量实验，结果写到 `experiments/pmlb/results/`。

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:0 \
  --model_path weights/checkpoint.pth \
  --data_type feynman
```

带乘性高斯噪声的批量实验，只额外指定噪声强度即可。

```bash
python experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:0 \
  --model_path weights/checkpoint.pth \
  --data_type feynman \
  --noise_strength 0.1
```

汇总不同噪声强度下的 PMLB 结果。

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_results_summary.py --input_csvs experiments/pmlb/results/pmlb_batch_inference_noise_0.csv experiments/pmlb/results/pmlb_batch_inference_noise_0.1.csv --output_csv experiments/pmlb/results/pmlb_results_summary.csv
```

## Acknowledge

We appreciate the following repos for their valuable code:

[[Multimodal-Math-Pretraining](https://github.com/deep-symbolic-mathematics/Multimodal-Math-Pretraining)] [[End-to-end Symbolic Regression](https://github.com/facebookresearch/symbolicregression)]


## License

This repository is licensed under the MIT License.
