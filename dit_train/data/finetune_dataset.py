"""FineTuneDataset: spawn 多进程预加载 lm_head 微调训练数据。

每个 worker 独立持有 env + CVAE + DiT + FeatureFusion（全部冻结），
生成 (src_enc, x2_tokens, len2) 三元组，主进程只做 decoder forward + CE loss。
"""
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class FineTuneDataset(IterableDataset):

    def __init__(self, params, batch_size=8, device="cuda",
                 vae_checkpoint="weights/checkpoint.pth",
                 dit_checkpoint="dit_train/checkpoints/fm_best.pt",
                 dit_num_steps=16):
        self.params = params
        self.batch_size = batch_size
        self.device = device
        self.vae_checkpoint = vae_checkpoint
        self.dit_checkpoint = dit_checkpoint
        self.dit_num_steps = dit_num_steps
        self._initialized = False
        if isinstance(params.tasks, list):
            params.tasks = ",".join(params.tasks)

    def _lazy_init(self):
        torch.cuda.set_device(self.device)

        import symbolicregression.utils
        symbolicregression.utils.CUDA = not self.params.cpu

        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules, reload_model
        from symbolicregression.utils import to_cuda
        from dit_train.model import GenSRDiT
        from dit_train.inference import euler_inference

        params = self.params
        env = build_env(params)
        env.rng = np.random.RandomState()
        self.env = env
        self._to_cuda = to_cuda

        modules = build_modules(env, params)
        reload_model(
            modules,
            modules_to_load=["cvae", "data_encoder", "token_embed", "feature_fusion"],
            path=self.vae_checkpoint,
            requires_grad=False,
        )

        self.embedder_f = modules["data_encoder"]
        self.embedder_e = modules["token_embed"]
        self.vae_model = modules["cvae"]
        self.feature_fusion = modules["feature_fusion"]

        for mod in [self.vae_model, self.embedder_f, self.embedder_e, self.feature_fusion]:
            for p in mod.parameters():
                p.requires_grad = False
        self.vae_model.eval()
        self.feature_fusion.eval()

        dit = GenSRDiT().to(self.device)
        dit.load_state_dict(torch.load(self.dit_checkpoint, map_location=self.device))
        dit.eval()
        for p in dit.parameters():
            p.requires_grad = False
        self.dit = dit
        self._euler_inference = euler_inference

        self._initialized = True

    def __iter__(self):
        if not self._initialized:
            self._lazy_init()

        while True:
            src_enc_list, x2_list, len2_list = [], [], []
            with torch.no_grad():
                for _ in range(self.batch_size):
                    samples, _ = self.env.gen_expr(train=True)

                    x_to_fit = samples["X_to_fit"]
                    y_to_fit = samples["Y_to_fit"]
                    x1 = [[[x, y] for x, y in zip(xs, ys)]
                           for xs, ys in zip(x_to_fit, y_to_fit)]
                    x1_single, len1_single = self.embedder_f(x1)

                    x2_single, len2_single = self.env.batch_equations(
                        self.env.word_to_idx([samples["tree_encoded"]], float_input=False)
                    )
                    x2_single, len2_single = self._to_cuda(x2_single, len2_single)
                    x2_e_single = self.embedder_e(x2_single.transpose(0, 1)).transpose(0, 1)

                    prior_mu, prior_logvar, _, _, _, _, _ = self.vae_model(
                        x1_single, x2_e_single, len1_single, len2_single, mode="train"
                    )
                    z_opt = self._euler_inference(self.dit, prior_mu, num_steps=self.dit_num_steps)
                    src_enc = self.feature_fusion(z_opt, prior_logvar)

                    src_enc_list.append(src_enc)
                    x2_list.append(x2_single)
                    len2_list.append(len2_single)

            yield (torch.cat(src_enc_list, dim=0), x2_list, len2_list)


def create_finetune_dataloader(params, batch_size=8, num_workers=2, device="cuda",
                                vae_checkpoint="weights/checkpoint.pth",
                                dit_checkpoint="dit_train/checkpoints/fm_best.pt",
                                dit_num_steps=16):
    dataset = FineTuneDataset(
        params, batch_size=batch_size, device=device,
        vae_checkpoint=vae_checkpoint,
        dit_checkpoint=dit_checkpoint,
        dit_num_steps=dit_num_steps,
    )
    kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True)
    return DataLoader(dataset, **kwargs)
