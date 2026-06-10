import torch
import pytest
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope="module")
def modules_and_env():
    """初始化 env + 所有模块（冻结 + 加载 checkpoint），只在模块级别初始化一次。"""
    from parsers import get_parser
    params = get_parser().parse_args([])
    params.max_input_dimension = 10
    params.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    import symbolicregression.utils
    symbolicregression.utils.CUDA = not params.cpu

    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules, reload_model

    env = build_env(params)
    modules = build_modules(env, params)

    # 加载 CVAE + data_encoder + token_embed + seq_decoder + feature_fusion
    reload_model(modules,
        modules_to_load=["cvae", "data_encoder", "token_embed", "seq_decoder", "feature_fusion"],
        path="weights/checkpoint.pth",
        requires_grad=False)

    return modules, env, params


def test_freeze_strategy(modules_and_env):
    """D-01: 只有 lm_head 参数 requires_grad=True，其他全部冻结。"""
    modules, env, params = modules_and_env
    decoder = modules["seq_decoder"]

    # 冻结所有模块
    for mod in modules.values():
        for p in mod.parameters():
            p.requires_grad = False

    # 解冻 lm_head (per D-01)
    for p in decoder.lm_head.parameters():
        p.requires_grad = True

    # 验证：只有 lm_head 参数可训练
    trainable_names = []
    frozen_count = 0
    for name, p in decoder.named_parameters():
        if p.requires_grad:
            trainable_names.append(name)
        else:
            frozen_count += 1

    assert len(trainable_names) > 0, "应该有可训练参数"
    for name in trainable_names:
        # lm_head.weight 与 tok_embed.weight 权重共享 (share_inout_emb=True)
        assert "lm_head" in name or "tok_embed" in name, \
            f"只有 lm_head/tok_embed 应该可训练，但 {name} 是可训练的"
    assert frozen_count > 0, "应该有冻结的参数"


def test_ce_loss_forward(modules_and_env):
    """完整链路 CVAE -> DiT Euler -> FeatureFusion -> Decoder fwd+predict 产生有效 CE loss。"""
    modules, env, params = modules_and_env
    device = params.device

    # 冻结策略 (D-01)
    for mod in modules.values():
        for p in mod.parameters():
            p.requires_grad = False

    # 加载 DiT
    from dit_train.model import GenSRDiT
    dit = GenSRDiT().to(device)
    dit.load_state_dict(torch.load("dit_train/checkpoints/fm_best.pt", map_location=device))
    dit.eval()
    for p in dit.parameters():
        p.requires_grad = False

    # 解冻 lm_head (D-01)
    decoder = modules["seq_decoder"]
    for p in decoder.lm_head.parameters():
        p.requires_grad = True

    embedder_f = modules["data_encoder"]
    embedder_e = modules["token_embed"]
    vae_model = modules["cvae"]
    feature_fusion = modules["feature_fusion"]

    # 生成一个 batch 的训练数据
    import numpy as np
    env.rng = np.random.RandomState(42)

    samples, _ = env.gen_expr(train=True)
    x_to_fit = samples["X_to_fit"]
    y_to_fit = samples["Y_to_fit"]
    x1 = [[[x, y] for x, y in zip(xs, ys)] for xs, ys in zip(x_to_fit, y_to_fit)]
    x1, len1 = embedder_f(x1)

    x2, len2 = env.batch_equations(
        env.word_to_idx([samples["tree_encoded"]], float_input=False)
    )
    from symbolicregression.utils import to_cuda
    x2, len2 = to_cuda(x2, len2)
    x2_e = embedder_e(x2.transpose(0, 1)).transpose(0, 1)

    # Step 1: CVAE 编码 -> prior_mu, prior_logvar
    with torch.no_grad():
        prior_mu, prior_logvar, _, _, _, _, _ = vae_model(x1, x2_e, len1, len2, mode="train")

    # Step 2: DiT Euler 积分 -> z_opt
    from dit_train.inference import euler_inference
    with torch.no_grad():
        z_opt = euler_inference(dit, prior_mu, num_steps=16)

    # Step 3: FeatureFusion -> src_enc
    with torch.no_grad():
        src_enc = feature_fusion(z_opt, prior_logvar)

    # Step 4: Decoder teacher-forcing -> CE loss
    alen = torch.arange(params.max_src_len, dtype=torch.long, device=len2.device)
    pred_mask = (alen[:, None] < len2[None] - 1)
    y = x2[1:].masked_select(pred_mask[:-1])

    tensor = decoder("fwd", x=x2, lengths=len2, causal=True, src_enc=src_enc, src_len=None, use_cache=False)
    scores, loss = decoder("predict", tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

    assert loss.dim() == 0, f"loss 应该是标量，实际 dim={loss.dim()}"
    assert torch.isfinite(loss), f"loss 应该是有限值，实际={loss.item()}"
    assert loss.item() > 0, f"CE loss 应该 > 0，实际={loss.item()}"

    # 验证 loss 可 backward（梯度流到 lm_head）
    loss.backward()
    assert decoder.lm_head.weight.grad is not None, "lm_head.weight 应该有梯度"
    assert decoder.lm_head.bias.grad is not None, "lm_head.bias 应该有梯度"
