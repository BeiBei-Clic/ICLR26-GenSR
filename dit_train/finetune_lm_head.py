"""Decoder lm_head 微调训练脚本

冻结 CVAE + DiT + FeatureFusion + Decoder Transformer blocks，
只解冻 Decoder 的 lm_head 输出投影层，用 DiT 传输后的 z_opt 做 teacher-forcing CE loss 训练。

照搬 Cola-DLM cola_vae_finetune.py 冻结/训练策略 (D-01, D-03)。
数据流：FunctionEnvironment -> CVAE -> DiT Euler -> FeatureFusion -> Decoder fwd+predict -> CE loss (D-02)。

Usage:
    python dit_train/finetune_lm_head.py
    python dit_train/finetune_lm_head.py --num-iterations 500 --batch-size 8 --learning-rate 1e-4
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GenSR Decoder lm_head 微调")
    parser.add_argument("--num-iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dit-checkpoint", type=str, default="dit_train/checkpoints/fm_best.pt")
    parser.add_argument("--vae-checkpoint", type=str, default="weights/checkpoint.pth")
    parser.add_argument("--output-dir", type=str, default="dit_train/checkpoints")
    parser.add_argument("--dit-num-steps", type=int, default=16, help="DiT Euler 积分步数")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 1. 初始化 env + modules ==========
    import symbolicregression.utils
    symbolicregression.utils.CUDA = device == "cuda"

    from parsers import get_parser
    params = get_parser().parse_args([])
    params.max_input_dimension = 10
    params.device = device

    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules, reload_model
    from symbolicregression.utils import to_cuda

    env = build_env(params)
    env.rng = np.random.RandomState()

    modules = build_modules(env, params)
    reload_model(
        modules,
        modules_to_load=["cvae", "data_encoder", "token_embed", "seq_decoder", "feature_fusion"],
        path=args.vae_checkpoint,
        requires_grad=False,
    )

    embedder_f = modules["data_encoder"]
    embedder_e = modules["token_embed"]
    vae_model = modules["cvae"]
    decoder = modules["seq_decoder"]
    feature_fusion = modules["feature_fusion"]

    # ========== 2. 加载 DiT ==========
    from dit_train.model import GenSRDiT
    from dit_train.inference import euler_inference

    dit = GenSRDiT().to(device)
    dit.load_state_dict(torch.load(args.dit_checkpoint, map_location=device))
    dit.eval()

    # ========== 3. 冻结策略 (D-01) ==========
    # 冻结所有模块
    for mod in modules.values():
        for p in mod.parameters():
            p.requires_grad = False
    for p in dit.parameters():
        p.requires_grad = False

    # 解冻 lm_head (per D-01, 与 Cola cola_vae_finetune.py 一致)
    for p in decoder.lm_head.parameters():
        p.requires_grad = True

    # 设置 eval 模式
    vae_model.eval()
    dit.eval()
    feature_fusion.eval()

    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder: {total_params:,} total, {trainable_params:,} trainable ({100*trainable_params/total_params:.2f}%)")

    # ========== 4. Optimizer (D-03, 照搬 cola_vae_finetune.py) ==========
    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    # ========== 5. 训练循环 ==========
    print(f"\n=== Decoder lm_head Fine-tuning ===")
    print(f"iterations: {args.num_iterations}, batch_size: {args.batch_size}, lr: {args.learning_rate}")
    print(f"dit_num_steps: {args.dit_num_steps}")
    print()

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for step in range(args.num_iterations):
        t0 = time.time()

        # --- 在线生成训练数据 (D-02) ---
        src_enc_list = []
        x2_list = []
        len2_list = []

        with torch.no_grad():
            for _ in range(args.batch_size):
                samples, _ = env.gen_expr(train=True)

                x_to_fit = samples["X_to_fit"]
                y_to_fit = samples["Y_to_fit"]
                x1 = [[[x, y] for x, y in zip(xs, ys)]
                       for xs, ys in zip(x_to_fit, y_to_fit)]
                x1_single, len1_single = embedder_f(x1)

                x2_single, len2_single = env.batch_equations(
                    env.word_to_idx([samples["tree_encoded"]], float_input=False)
                )
                x2_single, len2_single = to_cuda(x2_single, len2_single)
                x2_e_single = embedder_e(x2_single.transpose(0, 1)).transpose(0, 1)

                # CVAE 编码
                prior_mu, prior_logvar, _, _, _, _, _ = vae_model(
                    x1_single, x2_e_single, len1_single, len2_single, mode="train"
                )

                # DiT Euler 积分 -> z_opt
                z_opt = euler_inference(dit, prior_mu, num_steps=args.dit_num_steps)

                # FeatureFusion -> src_enc
                src_enc = feature_fusion(z_opt, prior_logvar)

                src_enc_list.append(src_enc)
                x2_list.append(x2_single)
                len2_list.append(len2_single)

        # 逐样本计算 CE loss 然后平均（方程长度不同无法直接 batch）
        total_loss = 0.0
        count = 0
        for i in range(args.batch_size):
            src_enc = src_enc_list[i]  # (1, 200, 512) from FeatureFusion
            eq_tokens = x2_list[i]     # (slen, 1)
            eq_len = len2_list[i]      # (1,)

            alen = torch.arange(params.max_src_len, dtype=torch.long, device=device)
            pred_mask = (alen[:, None] < eq_len[None] - 1)
            y = eq_tokens[1:].masked_select(pred_mask[:-1])

            if y.numel() == 0:
                continue

            # Decoder forward (teacher-forcing) — 需要梯度流到 lm_head
            tensor = decoder(
                "fwd",
                x=eq_tokens,
                lengths=eq_len,
                causal=True,
                src_enc=src_enc,
                src_len=None,
                use_cache=False,
            )
            scores, loss = decoder(
                "predict",
                tensor=tensor,
                pred_mask=pred_mask,
                y=y,
                get_scores=True,
            )
            total_loss += loss
            count += 1

        if count == 0:
            continue

        avg_loss = total_loss / count
        avg_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = time.time() - t0

        # --- 验证循环 (每 50 步) ---
        if (step + 1) % 50 == 0:
            decoder.eval()
            with torch.no_grad():
                val_src_enc_list = []
                val_x2_list = []
                val_len2_list = []
                for _ in range(args.batch_size):
                    samples, _ = env.gen_expr(train=False)

                    x_to_fit = samples["X_to_fit"]
                    y_to_fit = samples["Y_to_fit"]
                    x1 = [[[x, y] for x, y in zip(xs, ys)]
                           for xs, ys in zip(x_to_fit, y_to_fit)]
                    x1_single, len1_single = embedder_f(x1)

                    x2_single, len2_single = env.batch_equations(
                        env.word_to_idx([samples["tree_encoded"]], float_input=False)
                    )
                    x2_single, len2_single = to_cuda(x2_single, len2_single)
                    x2_e_single = embedder_e(x2_single.transpose(0, 1)).transpose(0, 1)

                    prior_mu, prior_logvar, _, _, _, _, _ = vae_model(
                        x1_single, x2_e_single, len1_single, len2_single, mode="train"
                    )
                    z_opt = euler_inference(dit, prior_mu, num_steps=args.dit_num_steps)
                    src_enc = feature_fusion(z_opt, prior_logvar)

                    val_src_enc_list.append(src_enc)
                    val_x2_list.append(x2_single)
                    val_len2_list.append(len2_single)

                val_total_loss = 0.0
                val_count = 0
                for i in range(args.batch_size):
                    src_enc = val_src_enc_list[i]
                    eq_tokens = val_x2_list[i]
                    eq_len = val_len2_list[i]

                    alen = torch.arange(params.max_src_len, dtype=torch.long, device=device)
                    pred_mask = (alen[:, None] < eq_len[None] - 1)
                    y = eq_tokens[1:].masked_select(pred_mask[:-1])

                    if y.numel() == 0:
                        continue

                    tensor = decoder(
                        "fwd",
                        x=eq_tokens,
                        lengths=eq_len,
                        causal=True,
                        src_enc=src_enc,
                        src_len=None,
                        use_cache=False,
                    )
                    scores, loss = decoder(
                        "predict",
                        tensor=tensor,
                        pred_mask=pred_mask,
                        y=y,
                        get_scores=True,
                    )
                    val_total_loss += loss.item()
                    val_count += 1

            decoder.train()

            if val_count == 0:
                continue

            val_loss = val_total_loss / val_count
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, "lm_head_best.pt")
                torch.save(decoder.lm_head.state_dict(), save_path)
                print(f"  New best val loss: {best_val_loss:.4f} -> {save_path}")

            print(f"  val_loss: {val_loss:.4f} | best_val: {best_val_loss:.4f}")

        loss_val = avg_loss.item()
        if step % args.log_every == 0 or step == args.num_iterations - 1:
            print(f"step {step:04d} | loss: {loss_val:.4f} | best_val: {best_val_loss:.4f} | dt: {dt*1000:.0f}ms")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved best lm_head to: {os.path.join(args.output_dir, 'lm_head_best.pt')}")


if __name__ == "__main__":
    main()
