"""LatentPairDataset: spawn 多进程并行生成 (prior_mu, post_mu) 训练对。

架构:
- __init__ 只保存配置参数，不初始化 GPU 模型（spawn 安全）
- _lazy_init 在 worker 进程内首次 __iter__ 时延迟初始化 env + modules + 冻结权重
- __iter__ yield 完整 batch (B, 512) 张量对，DataLoader batch_size=None 跳过 collate
- create_latent_dataloader 使用 multiprocessing_context='spawn' 确保 CUDA 安全

复用 GenSR 已有组件：
- FunctionEnvironment.gen_expr() 生成训练样本
- NumericalEmbedder 编码数值数据
- CVAEDE_SR.forward(mode="train") 获取 prior_mu/post_mu
- reload_model() 加载冻结权重
"""
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class LatentPairDataset(IterableDataset):

    def __init__(self, params, batch_size=16, device="cuda", checkpoint_path="weights/checkpoint.pth"):
        self.params = params
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._initialized = False
        # build_env 要求 tasks 是逗号分隔字符串，提前转换避免 _lazy_init 有副作用
        if isinstance(params.tasks, list):
            params.tasks = ",".join(params.tasks)

    def _lazy_init(self):
        """在 worker 进程中首次调用 __iter__ 时延迟初始化。"""
        torch.cuda.set_device(self.device)

        import symbolicregression.utils
        symbolicregression.utils.CUDA = not self.params.cpu

        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules, reload_model
        from symbolicregression.utils import to_cuda

        params = self.params
        env = build_env(params)
        env.rng = np.random.RandomState()
        self.env = env
        self._to_cuda = to_cuda

        modules = build_modules(env, params)
        reload_model(
            modules,
            modules_to_load=["cvae", "data_encoder", "token_embed"],
            path=self.checkpoint_path,
            requires_grad=False,
        )

        self.embedder_f = modules["data_encoder"]
        self.embedder_e = modules["token_embed"]
        self.vae_model = modules["cvae"]
        self.vae_model.eval()
        for module in [self.vae_model, self.embedder_f, self.embedder_e]:
            for param in module.parameters():
                param.requires_grad = False
        self._initialized = True

    def __iter__(self):
        if not self._initialized:
            self._lazy_init()

        while True:
            with torch.no_grad():
                prior_list, post_list = [], []
                for _ in range(self.batch_size):
                    samples, _ = self.env.gen_expr(train=True)

                    x_to_fit = samples["X_to_fit"]
                    y_to_fit = samples["Y_to_fit"]
                    x1 = [[[x, y] for x, y in zip(xs, ys)]
                           for xs, ys in zip(x_to_fit, y_to_fit)]
                    x1, len1 = self.embedder_f(x1)

                    x2, len2 = self.env.batch_equations(
                        self.env.word_to_idx([samples["tree_encoded"]], float_input=False)
                    )
                    x2, len2 = self._to_cuda(x2, len2)
                    x2_e = self.embedder_e(x2.transpose(0, 1)).transpose(0, 1)

                    prior_mu, _, post_mu, _, _, _, _ = self.vae_model(
                        x1, x2_e, len1, len2, mode="train"
                    )
                    prior_list.append(prior_mu.clone())
                    post_list.append(post_mu.clone())

            yield (torch.cat(prior_list, dim=0), torch.cat(post_list, dim=0))


def create_latent_dataloader(params, batch_size=16, num_workers=2, device="cuda", checkpoint_path="weights/checkpoint.pth"):
    dataset = LatentPairDataset(
        params, batch_size=batch_size,
        device=device, checkpoint_path=checkpoint_path,
    )
    kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True)
    return DataLoader(dataset, **kwargs)
