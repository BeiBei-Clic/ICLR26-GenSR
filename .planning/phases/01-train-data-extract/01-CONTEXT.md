# Phase 1: 训练数据生成器 - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

实现一个在线数据生成器（PyTorch Dataset/DataLoader），在 DiT 训练时实时生成 (prior_mu, post_mu) 训练对。CVAE (checkpoint.pth) 已训练好，冻结不动。生成器流程：FunctionEnvironment 在线生成 (X, Y, GT表达式) → 编码 → CVAE forward(mode="train") → 输出 (prior_mu, post_mu) 对。不保存中间数据到文件。

**范围内：**
- 实现一个可复用的 PyTorch Dataset，冻结 CVAE，在线生成 (prior_mu, post_mu) 对
- 支持 batch size、样本量等参数配置
- 生成器可作为标准 DataLoader 被 Phase 4（DiT 训练）直接使用
- Phase 2 用这个生成器的输出做数据质量验证

**范围外：**
- 保存 .pt 文件到磁盘（在线生成不需要）
- 数据质量验证（Phase 2）
- DiT 模型定义或训练（Phase 3-4）
- 推理管线（Phase 5-6）

</domain>

<decisions>
## Implementation Decisions

### 数据生成方式
- **D-01:** 在线生成，不保存到文件。使用 FunctionEnvironment 在线随机生成 (X, Y, GT表达式) 样本，立刻过冻结的 CVAE 提取 (prior_mu, post_mu)。每次调用生成新数据，无需持久化。
- **D-02:** 不固定随机种子，每次运行生成不同数据。

### 组件定位
- **D-03:** Phase 1 交付的是一个 PyTorch 数据生成组件（Dataset 类），不是独立脚本。Phase 4 的 DiT 训练循环直接 import 这个组件使用。
- **D-04:** CVAE 使用 `weights/checkpoint.pth`（671MB），加载后完全冻结（requires_grad=False），不参与 DiT 训练。

### 数据格式
- **D-05:** 生成器每次 __getitem__ 返回 (prior_mu, post_mu) 对，各为 (512,) 张量。不返回 logvar、原始表达式、X/Y 数据等。

### 配置
- **D-06:** batch_size、num_workers 等通过构造参数配置，方便 DiT 训练时灵活调整。

### Claude's Discretion
- Dataset 类的具体实现方式（IterableDataset vs Map-style Dataset）
- 批次处理的具体逻辑（collate_fn 等）
- 是否需要 train/val split 及划分方式
- 设备管理（CVAE 在哪个设备上运行）
- 组件文件位置和命名
- 是否需要独立的验证脚本（供 Phase 2 使用）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CVAE 模型架构
- `symbolicregression/model/cvae.py` — CVAEDE_SR 类，forward 方法 mode="train" 返回 prior_mu 和 post_mu
- `symbolicregression/model/cvae.py:777-778` — post_fc 和 prior_fc 投影层定义
- `symbolicregression/model/cvae.py:807-833` — forward 方法训练模式分支

### 训练数据生成
- `symbolicregression/envs/environment.py` — FunctionEnvironment.gen_expr() 生成训练样本
- `symbolicregression/trainer_vae.py:836-953` — enc_dec_vae_step 展示完整的训练数据流

### 模型加载
- `model.py:701-728` — reload_model 函数，加载 checkpoint 中各模块权重
- `model.py` — VAESymbolicRegressor 类，modules 字典管理各组件

### 训练入口
- `train.py` — 训练脚本入口，展示如何初始化 env、modules、trainer

### 配置与参数
- `parsers.py` — 命令行参数定义，包括 batch_size、n_steps_per_epoch 等

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FunctionEnvironment`: 已有完整的在线训练数据生成器，生成随机方程 + 数值数据点
- `reload_model()`: 已有检查点加载逻辑，直接复用
- `CVAEDE_SR.forward(mode="train")`: 已有同时返回 prior_mu 和 post_mu 的接口
- `embedder_f / embedder_e`: 已有的数据和表达式编码器

### Established Patterns
- 模型通过 `modules` 字典管理（cvae, seq_decoder, data_encoder 等）
- 检查点格式为 dict，key 为模块名，value 为 state_dict
- 设备管理使用 `params.device`，通常为 "cuda"
- 训练循环在 trainer_vae.py 中，数据通过 get_batch() 获取

### Integration Points
- Dataset 需要持有冻结的 CVAE、embedder 和 FunctionEnvironment 实例
- 需要 import symbolicregression 包
- 需要通过 build_env() 创建 FunctionEnvironment
- Phase 4 DiT 训练直接 import 这个 Dataset 使用

</code_context>

<specifics>
## Specific Ideas

无特殊要求 — 标准的 PyTorch Dataset 组件。

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-train-data-extract*
*Context gathered: 2026-06-09*
