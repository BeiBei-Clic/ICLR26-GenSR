"""LatentPairDataset: 在线生成 (prior_mu, post_mu) 训练对供 DiT Flow Matching 训练。

复用 GenSR 已有组件：
- FunctionEnvironment.gen_expr() 生成训练样本
- NumericalEmbedder 编码数值数据
- CVAEDE_SR.forward(mode="train") 获取 prior_mu/post_mu
- reload_model() 加载冻结权重

数据流复刻 trainer_vae.py:enc_dec_vae_step (行 862-888)。
"""
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from symbolicregression.envs import build_env
from symbolicregression.model import build_modules, reload_model
from symbolicregression.utils import to_cuda
import symbolicregression.utils


class LatentPairDataset(IterableDataset):

    def __init__(self, params, batch_size=16, device="cuda", checkpoint_path="weights/checkpoint.pth"):
        self.params = params
        self.batch_size = batch_size
        self.device = device

        # 设置 CUDA 全局标志，否则 to_cuda 不会生效
        symbolicregression.utils.CUDA = not params.cpu

        # build_env 会把 params.tasks 从 "a,b" 变成 ["a", "b"]，
        # 如果已经是 list（之前调用过 build_env），先恢复为字符串
        if isinstance(params.tasks, list):
            params.tasks = ",".join(params.tasks)

        # 1. 创建 env
        env = build_env(params)
        env.rng = np.random.RandomState()
        self.env = env

        # 2. 创建 modules
        modules = build_modules(env, params)

        # 3. 加载冻结权重
        reload_model(
            modules,
            modules_to_load=["cvae", "data_encoder", "token_embed"],
            path=checkpoint_path,
            requires_grad=False,
        )

        # 4. 保存引用并冻结所有参数
        self.embedder_f = modules["data_encoder"]
        self.embedder_e = modules["token_embed"]
        self.vae_model = modules["cvae"]
        self.vae_model.eval()
        # reload_model 的 requires_grad=False 只设模块属性，不冻结参数
        for module in [self.vae_model, self.embedder_f, self.embedder_e]:
            for param in module.parameters():
                param.requires_grad = False

    def __iter__(self):
        while True:
            # 数据生成在 no_grad 内完成，但 yield 必须在 with 块外面，
            # 否则 Python 生成器 yield 时不会退出 with，导致 no_grad 泄漏到训练循环
            with torch.no_grad():
                samples, _ = self.env.gen_expr(train=True)

                x_to_fit = samples["X_to_fit"]
                y_to_fit = samples["Y_to_fit"]

                x1 = [[[x, y] for x, y in zip(xs, ys)] for xs, ys in zip(x_to_fit, y_to_fit)]
                x1, len1 = self.embedder_f(x1)

                x2, len2 = self.env.batch_equations(
                    self.env.word_to_idx([samples["tree_encoded"]], float_input=False)
                )
                x2, len2 = to_cuda(x2, len2)
                x2_e = self.embedder_e(x2.transpose(0, 1)).transpose(0, 1)

                prior_mu, _, post_mu, _, _, _, _ = self.vae_model(x1, x2_e, len1, len2, mode="train")

                batch_prior = prior_mu.clone()
                batch_post = post_mu.clone()

            for i in range(batch_prior.shape[0]):
                yield batch_prior[i], batch_post[i]


def create_latent_dataloader(params, batch_size=16, num_workers=0, device="cuda", checkpoint_path="weights/checkpoint.pth"):
    dataset = LatentPairDataset(params, batch_size=batch_size, device=device, checkpoint_path=checkpoint_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
