<!-- GSD:project-start source:PROJECT.md -->
## Project

**GenSR-DiT: 用 Flow Matching DiT 替换 CMA-ES 潜空间搜索**

一个符号回归（Symbolic Regression）系统，在已有的 GenSR VAE 框架上，将推理阶段的 CMA-ES 进化搜索替换为 DiT（Diffusion Transformer）的 Flow Matching 推理。核心思路来自 Cola-DLM：在潜空间中用学习的先验传输替代无模型优化，实现更快、更准确的符号表达式发现。

现有 GenSR 的流程：数据 → CVAE 编码 → 潜向量 → CMA-ES 搜索 → 解码 → 表达式。替换后：数据 → CVAE 编码 → prior_mu → DiT 先验传输 → 更好的潜向量 → 解码 → 表达式。

**Core Value:** 用 DiT Flow Matching 替换 CMA-ES，在保持 VAE + Decoder 不变的前提下，将符号回归的推理从迭代进化搜索变为单次前向传输，大幅提升推理速度并可能提升表达式质量。

### Constraints

- **VAE/Decoder 冻结**：训练 DiT 时，CVAE、FeatureFusion、Decoder 的权重全部冻结
- **潜空间不变**：DiT 必须在同一个 512 维潜空间内操作
- **PyTorch**：使用 PyTorch 实现，与现有代码库保持一致
- **单 GPU 训练**：DiT 规模不大（512 维输入），单卡足够
- **评估对比**：必须在 PMLB 数据集上与 CMA-ES 方法做定量对比
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.10.12 - 核心开发语言，用于模型训练、推理和评估
- Bash - 脚本自动化和实验管理
- YAML - 配置文件和元数据
## Runtime
- Python 3.10+ (当前环境: Python 3.10.12)
- CUDA 12.8 (PyTorch 2.10.0+cu128)
- NVIDIA Driver 595.71.05
- Conda (通过 `environment.yml`)
- pip (通过 `requirements.txt`)
- Lockfile: `environment.yml` (conda lockfile), `requirements.txt` (pip lower bounds)
## Frameworks
- PyTorch 2.0.0+ (当前环境: 2.10.0) - 深度学习框架，用于模型训练和推理
- PyTorch Lightning 2.0.5 - 训练循环管理和分布式训练
- Transformers 4.40+ - Hugging Face Transformers 库 (Cola-DLM 使用)
- SymPy 1.11.0+ - 符号数学计算
- SymPyTorch 0.1.0+ - SymPy 和 PyTorch 桥接
- gplearn 0.4.2 - 遗传规划用于符号回归
- NumPy 1.24.0+ - 数值计算
- SciPy 1.10.0+ - 科学计算和优化
- pandas 2.0.0+ - 数据处理
- scikit-learn 1.2.0+ - 机器学习工具和评估指标
- numexpr 2.8.0+ - 快速数值表达式评估
- XGBoost 1.7.5 - 梯度提升 (可选使用)
- CMA-ES (自定义实现 `cma_es_modular.py`) - 协方差矩阵自适应进化策略
- BFGS (通过 SciPy) - 常数优化
- matplotlib 3.7.0+ - 绘图和可视化
- seaborn 0.12.0+ - 统计可视化
- Weights & Biases (wandb) 0.14.0+ - 实验跟踪和日志记录
- tqdm 4.65.0+ - 进度条显示
- PyYAML 6.0+ - YAML 配置文件解析
- requests 2.28.0+ - HTTP 请求
## Key Dependencies
- torch>=2.0.0 - 核心深度学习框架
- transformers>=4.40 - Cola-DLM 语言模型依赖
- sympytorch>=0.1.0 - 符号表达式与张量转换
- pytorch-lightning==2.0.5 - 训练框架
- wandb>=0.14.0 - 实验跟踪
- NVIDIA CUDA 库 (cublas, cudnn, nccl, nvrtc, curand, cusolver, cusparse, cufft) - GPU 加速
- triton==2.0.0 - GPU 内核编译
- 本地文件系统 (无数据库依赖)
## Configuration
- Conda 环境配置: `environment.yml`
- pip 依赖: `requirements.txt`
- Python 虚拟环境: `.venv/` (Python 3.10.12)
- 数据集元数据: `datasets/pmlb/datasets/*/metadata.yaml`
- 无需编译 (纯 Python)
- 代码直接运行，无构建步骤
- 配置: `Cola-DLM-main/pyproject.toml`
- 依赖: `Cola-DLM-main/requirements.txt`
## Platform Requirements
- Python 3.10+
- CUDA 12.x (推荐 CUDA 12.8)
- NVIDIA GPU (推荐显存 >= 16GB)
- Linux (测试环境: Ubuntu 20.04+)
- 单机 GPU 训练/推理
- SLURM 集群支持 (通过 `symbolicregression/slurm.py`)
- 分布式训练支持 (PyTorch DDP)
- 模型权重: `weights/checkpoint.pth` (671MB), `weights/fm_best.pth` (932MB)
- 数据集: `datasets/` 目录 (PMLB + Feynman)
- 实验输出: `dump/`, `wandb/`, `.planning/`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- 使用小写字母和下划线命名，如 `train.py`, `LSO_eval.py`, `const_opt.py`
- 模块文件使用描述性名称，如 `parsers.py`, `model.py`
- 测试文件使用 `test_*.py` 模式
- 使用小写字母和下划线命名，如 `evaluate_pmlb_lso`, `reload_model`, `read_file`
- 内部函数使用前导下划线，如 `_mp_eval_one`, `_update_params`
- 函数名应具有描述性，清楚表达其功能
- 使用小写字母和下划线命名，如 `batch_results`, `problem_name`, `target_noise`
- 布尔变量使用 `is_`, `has_` 前缀，如 `is_master`, `hasattr`
- 缩写使用全大写，如 `X`, `y`（数学变量），`mse`, `r2`
- 使用驼峰命名法（PascalCase），如 `VAESymbolicRegressor`, `DiagonalCMAES`
- 类名应该是名词，描述其代表的实体
## Code Style
- 没有使用自动格式化工具（未检测到 `.prettierrc`, `black`, `yapf` 配置）
- 使用4空格缩进（Python标准）
- 行长度未强制限制，但保持可读性
- 在二元运算符周围使用空格
- 未检测到 linting 配置（无 `.flake8`, `.pylintrc`, `pyproject.toml`）
- 代码质量依赖人工审查
## Import Organization
- 未使用路径别名
- 使用相对导入和绝对导入混合
## Error Handling
- 优先使用断言进行参数验证，如 `assert torch.cuda.is_available()`
- 使用 try-except 块捕获特定异常
- 多进程worker中使用 try-except 并返回失败状态
- 异常处理示例（来自 `LSO_fit.py`）：
- 在关键路径上让异常向上传播
- 在评估函数中捕获异常并提供默认值
- 使用 `logger.warning()` 记录非致命错误
## Logging
- 使用不同级别：`logger.info()`, `logger.warning()`, `print()`
- Wandb用于实验跟踪和指标记录
- 重要操作使用日志记录，如模型加载、评估开始
- 示例（来自 `train.py`）：
- 使用 `tqdm` 进度条进行长时间运行的任务
- Wandb日志用于训练和评估指标
## Comments
- 在函数开头使用docstring描述功能
- 复杂逻辑处添加行内注释
- 模块分隔使用注释块
- 示例（来自 `LSO_fit.py`）：
- 使用Python docstring（三引号）
- 简短描述函数目的
- 参数和返回值类型通常不在docstring中注明
## Function Design
- 函数长度变化很大，从10行到300+行
- 关键函数如 `evaluate_pmlb_lso` 较长（400+行）
- 优先功能完整性而非函数长度限制
- 使用位置参数和关键字参数混合
- 复杂函数使用参数对象（`params`, `env`）
- 示例（来自 `const_opt.py`）：
- 返回单个值或元组
- 多返回值使用元组解包
- 成功/失败状态通过返回值或异常传达
- 示例（来自 `model.py`）：
## Module Design
- 使用 `__all__` 明确导出符号，如 `model.py`:
- 未大量使用barrel文件（`__init__.py`汇总导入）
- 主要模块直接导入和使用
- 通过延迟导入避免循环依赖
- 在函数内部导入有时用于打破依赖循环
## Special Patterns
- 使用全局变量在worker间传递数据
- Worker函数设计为接受参数并返回结果
- 示例（来自 `LSO_fit.py`）：
- 模块字典用于管理多个模型组件
- 模型加载使用 `reload_model` 函数处理版本兼容性
- 示例（来自 `model.py`）：
- 显式设备管理：`.to(params.device)`
- CUDA可用性检查：`assert torch.cuda.is_available()`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Latent space-based equation generation using Conditional Variational Autoencoder (CVAE)
- Dual-branch architecture for numerical data and symbolic equations
- Evolution Strategy (CMA-ES) optimization over latent space
- BFGS refinement for constant optimization
- Multi-stage pipeline: training → latent encoding → evolutionary search → refinement
## Layers
- Purpose: Encode numerical (X, Y) data points into fixed-length representations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/embedders.py`
- Contains: `NumericalEmbedder` class with float encoding logic
- Depends on: Environment configuration, float encoder
- Used by: CVAE encoder during training and inference
- Purpose: Encode symbolic equations (trees) into token embeddings
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/embedders.py`
- Contains: Token embedding layer, positional embeddings
- Depends on: Equation vocabulary from environment
- Used by: CVAE encoder during training
- Purpose: Learn latent representations of (data, equation) pairs
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/cvae.py`
- Contains: `CVAEDE_SR` class with Transformer encoder and latent projection
- Depends on: Data encoder, equation encoder, Transformer layers
- Used by: Trainer for VAE training, inference pipeline for encoding
- Purpose: Fuse latent (μ, logvar) into decoder-ready representation
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/feature_fusion.py`
- Contains: `FeatureFusion` class
- Depends on: Latent dimension, decoder embedding dimension
- Used by: Decoder for equation generation
- Purpose: Generate equation tokens from latent representation
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/transformer.py`
- Contains: `TransformerModel_VAE` decoder with cross-attention
- Depends on: Fused latent representation, token embeddings
- Used by: Training loss computation, inference generation
- Purpose: Search latent space for optimal equations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py`
- Contains: CMA-ES implementation, population initialization, candidate evaluation
- Depends on: Pretrained CVAE model, environment for equation conversion
- Used by: Inference pipeline for final equation discovery
- Purpose: Optimize constants in discovered equations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/const_opt.py`
- Contains: BFGS-based constant refinement logic
- Depends on: Environment for equation-to-function conversion
- Used by: LSO optimization pipeline
## Data Flow
- Training: VAE parameters, optimizer state, learning rate schedule
- Inference: Best equation skeleton, constants, fitness tracking, skeleton deduplication
## Key Abstractions
- Purpose: Continuous representation space where each point maps to an equation
- Examples: `prior_mu`, `post_mu` tensors of shape (batch, latent_dim)
- Pattern: Gaussian distribution with (μ, logvar) parameters
- Purpose: Template of equation structure without constant values
- Examples: `skeleton_candidate` in `LSO_fit.py:gen2eq()`
- Pattern: Sympy expression tree with placeholder constants
- Purpose: Set of candidate solutions evolving over iterations
- Examples: `pop` tensor in `lso_fit_es_covfromvae_fit()`
- Pattern: (N, latent_dim) tensor sampled from Gaussian distribution
- Purpose: Generate multiple equation candidates from single latent
- Examples: `beam_size` parameter in decoder generation
- Pattern: Parallel decoding with top-K hypothesis tracking
## Entry Points
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/train.py:main()`
- Triggers: Direct script execution
- Responsibilities: Initialize environment, modules, trainer, run training epochs, handle checkpointing
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/direct_eval.py:__main__()`
- Triggers: Script execution with evaluation parameters
- Responsibilities: Load pretrained model, run PMLB evaluation, save results
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/experiments/pmlb/pmlb_batch_inference.py`
- Triggers: Script execution for batch processing
- Responsibilities: Process multiple datasets, generate CSV results
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py:lso_fit_es_covfromvae_fit()`
- Triggers: Called during inference for each dataset
- Responsibilities: Execute evolutionary search, return best equation
## Error Handling
- **NaN Detection:** Training checks for NaN loss and logs warnings
- **Empty Results:** Inference handles failed candidates with placeholder zeros
- **Checkpoint Recovery:** Model loading handles missing keys with strict=False
- **Skeleton Deduplication:** Prevents redundant evaluation of same equation structures
- Try-except in equation evaluation with fallback to failed status
- Timeout handling for equation simplification
- Gradient clipping for training stability
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
