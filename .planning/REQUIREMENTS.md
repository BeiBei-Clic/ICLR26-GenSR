# 需求文档：GenSR-Cola 流匹配推理

## 功能需求

### FR-1: Flow Matching 模型
- **FR-1.1**: 实现条件 Flow Matching 模型，输入 (z_t, t, data_embedding)，输出向量场 v
- **FR-1.2**: 模型架构：MLP 或小型 Transformer，适配 512 维潜空间
- **FR-1.3**: 支持 ODE 求解（Euler/Heun 方法），从 z₁ ~ N(0,I) 传输到 z₀
- **FR-1.4**: 支持多次采样推理，取最优结果

### FR-2: Stage 2 训练流程
- **FR-2.1**: 收集 CVAE 训练时的 (prior_mu, post_mu) 对作为 Flow Matching 训练数据
- **FR-2.2**: 实现 Flow Matching 训练 loss：L_FM = E[‖v_ψ(z_t, t, c) - u_t(z_0, z_1)‖²]
- **FR-2.3**: 支持从已有 CVAE checkpoint 加载并冻结/微调
- **FR-2.4**: 训练脚本支持 WandB 日志

### FR-3: 推理替换
- **FR-3.1**: 新推理流程：data → CVAE encoder → prior_mu → Flow Matching → z₀ → FeatureFusion → Decoder → 方程
- **FR-3.2**: 保持 BFGS 常数优化环节
- **FR-3.3**: 保持与 `pmlb_batch_inference.py` 的接口兼容

### FR-4: PMLB 批量测试
- **FR-4.1**: 修改 `pmlb_batch_inference.py` 支持新的推理方式
- **FR-4.2**: 结果输出格式与原版兼容（CSV，相同字段）
- **FR-4.3**: 支持通过参数切换 CMA-ES / Flow Matching 推理模式

## 非功能需求

### NFR-1: 性能
- 推理速度应显著快于 CMA-ES（无迭代进化搜索）
- 支持 GPU 加速 ODE 求解

### NFR-2: 兼容性
- 保持所有现有参数的默认行为
- 新增参数不破坏已有脚本

### NFR-3: 可复现
- 固定随机种子可复现结果
- 清晰的检查点保存/加载

## 依赖
- 现有 CVAE 预训练权重 (`weights/checkpoint.pth`)
- 现有环境依赖（PyTorch, numpy, scipy 等）
- 不引入额外大型依赖

## 验收标准
1. Flow Matching 模型可训练且 loss 收敛
2. Flow Matching 推理在 Feynman 数据集上 R² > 0.5（初步目标）
3. `pmlb_batch_inference.py` 使用 Flow Mapping 模式可运行完成
4. 推理速度相比 CMA-ES 有明显提升
