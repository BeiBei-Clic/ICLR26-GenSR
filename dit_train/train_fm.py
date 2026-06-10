"""Flow Matching 训练脚本 — 学习 prior_mu -> post_mu 的速度场

参照 Cola-DLM-main/scripts/cola_sft.py 训练逻辑，适配到 GenSR 的简化场景:
- 512 维单向量，无序列/mask/2L trick (D-09)
- 原生 PyTorch 手写循环 (D-01)
- OT-path 插值: z_t = (1-t)*prior_mu + t*post_mu (FM-01)
- MSE loss 全维度参与无 mask (FM-02)

支持单卡和多卡 DDP 训练：
  单卡: python dit_train/train_fm.py [args]
  多卡: torchrun --nproc_per_node=N dit_train/train_fm.py [args]
"""
import os
import sys
import time

# 确保项目根目录在 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dit_train.model import GenSRDiT, dit_default_config
from dit_train.data.latent_dataset import create_latent_dataloader


# ---------------------------------------------------------------------------
# 时间步采样 (D-02, 照搬 cola_sft.py:440-445)
# ---------------------------------------------------------------------------

def sample_timestep(batch_size, device="cuda", dist="logit_normal", loc=0.0, scale=1.0):
    """采样 Flow Matching 时间步。

    Args:
        batch_size: batch 大小
        device: 设备（DDP 多卡时传入正确设备）
        dist: "logit_normal" 或 "uniform"
        loc: logit-normal 的位置参数
        scale: logit-normal 的尺度参数
    Returns:
        (batch_size,) 时间步张量，值在 (0, 1) 范围内
    """
    if dist == "uniform":
        return torch.rand(batch_size, device=device)
    u = torch.randn(batch_size, device=device)
    return torch.sigmoid(loc + scale * u)


# ---------------------------------------------------------------------------
# Flow Matching Step (FM-01, FM-02, 照搬 cola_sft.py:451-534 极简版)
# ---------------------------------------------------------------------------

def flow_matching_step(dit, prior_mu, post_mu, timestep_dist="logit_normal"):
    """单步 Flow Matching 训练。

    Args:
        dit: GenSRDiT 模型
        prior_mu: (B, 512) CVAE 先验均值
        post_mu: (B, 512) CVAE 后验均值
        timestep_dist: 时间步采样分布
    Returns:
        (loss, loss.detach()) — loss 用于 backward，train_loss 用于日志
    """
    B = prior_mu.shape[0]

    # 时间步采样 (D-02)
    device = prior_mu.device
    t = sample_timestep(B, device=device, dist=timestep_dist)  # (B,)

    # OT-path 插值 (FM-01)
    t_expand = t[:, None]  # (B, 1) 广播
    z_t = (1 - t_expand) * prior_mu + t_expand * post_mu  # (B, 512)

    # 速度目标 (FM-01)
    target = post_mu - prior_mu  # (B, 512)

    # 模型前向
    pred = dit(z_t, t, prior_mu)  # (B, 512)

    # MSE loss (FM-02)，全维度参与，无 mask
    loss = ((pred - target) ** 2).mean()

    return loss, loss.detach()


# ---------------------------------------------------------------------------
# LR Schedule (D-04, 照搬 cola_sft.py:189-196)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.0):
    """Cosine LR schedule: warmup -> plateau -> warmdown。

    Args:
        progress: 训练进度 [0, 1]
        warmup_ratio: warmup 比例
        warmdown_ratio: warmdown 比例
        final_lr_frac: 最终学习率比例
    Returns:
        LR 乘数
    """
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac


# ---------------------------------------------------------------------------
# Checkpoint 保存与加载 (FM-05, per Pitfall 4 使用 torch.save)
# ---------------------------------------------------------------------------

def save_checkpoint(dit, optimizer, step, val_loss, best_val_loss, output_dir):
    """保存训练 checkpoint。

    Args:
        dit: GenSRDiT 模型
        optimizer: 优化器
        step: 当前步数
        val_loss: 验证 loss
        best_val_loss: 历史最优 val loss
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"fm_step_{step:06d}.pt")
    torch.save({
        "model_state_dict": dit.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


def load_checkpoint(dit, path, device="cuda"):
    """加载训练 checkpoint。

    Args:
        dit: GenSRDiT 模型
        path: checkpoint 路径
        device: 设备
    Returns:
        (step, val_loss, best_val_loss, ckpt_dict) — ckpt_dict 包含 optimizer_state_dict 等
    """
    ckpt = torch.load(path, map_location=device)
    dit.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf")), ckpt.get("best_val_loss", float("inf")), ckpt


# ---------------------------------------------------------------------------
# Evaluate (照搬 cola_sft.py:567-576 简化版)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(dit, data_iter, eval_steps):
    """在数据流上评估平均 loss。

    Args:
        dit: GenSRDiT 模型
        data_iter: 数据迭代器，yield (prior_mu, post_mu)
        eval_steps: 评估步数
    Returns:
        平均 loss
    """
    dit.eval()
    losses = []
    for _ in range(eval_steps):
        prior_mu, post_mu = next(data_iter)
        loss, _ = flow_matching_step(dit, prior_mu, post_mu)
        losses.append(loss.item())
    dit.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# 训练循环主体 — CLI + 数据加载 + 训练 (Task 2 将在此处添加)
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GenSR DiT Flow Matching 训练")
    parser.add_argument("--num-iterations", type=int, default=5000)
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmdown-ratio", type=float, default=0.3)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--timestep-dist", type=str, default="logit_normal",
                        choices=["logit_normal", "uniform"])
    parser.add_argument("--logit-normal-loc", type=float, default=0.0)
    parser.add_argument("--logit-normal-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="dit_train/checkpoints")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-path", type=str, default="weights/checkpoint.pth")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    # ---- DDP 初始化 ----
    ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_master = rank == 0
    device = f"cuda:{local_rank}"

    if is_master:
        print(f"=== GenSR DiT Flow Matching Training ===")
        print(f"num_iterations: {args.num_iterations}")
        print(f"batch: {args.device_batch_size} x {args.grad_accum_steps} x {world_size} = "
              f"{args.device_batch_size * args.grad_accum_steps * world_size} effective")
        print(f"lr: {args.learning_rate}, warmup: {args.warmup_ratio}, "
              f"warmdown: {args.warmdown_ratio}")
        print(f"timestep_dist: {args.timestep_dist}")
        print(f"DDP: {ddp}, world_size: {world_size}")

    # 初始化 params (per Pitfall 2: max_input_dimension=10)
    from parsers import get_parser
    params = get_parser().parse_args([])
    params.device = device
    params.max_input_dimension = 10

    # 创建 DiT
    dit = GenSRDiT().to(device).train()
    dit_raw = dit  # 保留原始引用用于 save/load
    if is_master:
        print(f"DiT: {sum(p.numel() for p in dit.parameters()):,} params")

    # DDP 包裹（所有参数每次 forward 都使用，无需 find_unused_parameters）
    if ddp:
        dit = DDP(dit, device_ids=[local_rank], output_device=local_rank)

    # 优化器 (D-03)
    optimizer = torch.optim.AdamW(
        dit.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Resume
    start_step = 0
    val_loss = float("nan")
    best_val_loss = float("inf")
    if args.resume:
        step, val_loss, best_val_loss, ckpt = load_checkpoint(dit_raw, args.resume, device=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = step + 1
        if is_master:
            print(f"Resumed from step {step}, val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f}")

    # 数据加载（共享同一个 loader，因为 CVAE 冻结后每次生成的数据统计上等价）
    loader = create_latent_dataloader(
        params, batch_size=args.device_batch_size,
        checkpoint_path=args.checkpoint_path,
    )
    data_iter = iter(loader)

    # 训练循环 (照搬 cola_sft.py:633-691)
    smooth_loss = 0.0
    ema_beta = 0.95  # D-07
    num_iterations = args.num_iterations

    for step in range(start_step, num_iterations):
        t0 = time.time()
        last_step = step == num_iterations - 1

        # --- Training step with gradient accumulation (D-05) ---
        # 训练必须在 eval 之前，确保 DDP 的首次 forward 是训练而非 eval
        for micro_step in range(args.grad_accum_steps):
            prior_mu, post_mu = next(data_iter)
            prior_mu = prior_mu.to(device)
            post_mu = post_mu.to(device)
            loss, train_loss = flow_matching_step(dit, prior_mu, post_mu, args.timestep_dist)
            (loss / args.grad_accum_steps).backward()  # per Pitfall 5

        # LR schedule (D-04)
        progress = step / max(num_iterations - 1, 1)
        lrm = get_lr_multiplier(progress, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        # Optimizer step + gradient clipping (D-06)
        torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # --- Eval (训练之后，仅 master) ---
        do_eval = step == 0 or last_step or (args.eval_every > 0 and step % args.eval_every == 0)
        if do_eval:
            val_loss = evaluate(dit_raw, data_iter, args.eval_steps)
            if is_master:
                print(f"Step {step:05d} | Val FM loss: {val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.output_dir, "fm_best.pt")
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(dit_raw.state_dict(), best_path)
                    print(f"New best val loss: {best_val_loss:.6f} -> {best_path}")
            if ddp:
                dist.barrier()

        # --- Save (master only，然后 barrier) ---
        do_save = last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0)
        if is_master and do_save:
            save_checkpoint(dit_raw, optimizer, step, val_loss, best_val_loss,
                            os.path.join(args.output_dir, "run"))
        if ddp and do_save:
            dist.barrier()

        dt = time.time() - t0

        # EMA smoothing (D-07)
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
        debiased = smooth_loss / (1 - ema_beta ** (step - start_step + 1))

        if is_master and (step % 10 == 0 or last_step):
            print(f"step {step:05d} | loss: {debiased:.6f} | "
                  f"lr: {lrm * args.learning_rate:.2e} | dt: {dt * 1000:.0f}ms")

    if is_master:
        print("Training complete.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
