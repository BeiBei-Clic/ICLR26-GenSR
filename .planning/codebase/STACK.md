# Technology Stack

**Analysis Date:** 2026-06-09

## Languages

**Primary:**
- Python 3.10.12 - 核心开发语言，用于模型训练、推理和评估

**Secondary:**
- Bash - 脚本自动化和实验管理
- YAML - 配置文件和元数据

## Runtime

**Environment:**
- Python 3.10+ (当前环境: Python 3.10.12)
- CUDA 12.8 (PyTorch 2.10.0+cu128)
- NVIDIA Driver 595.71.05

**Package Manager:**
- Conda (通过 `environment.yml`)
- pip (通过 `requirements.txt`)
- Lockfile: `environment.yml` (conda lockfile), `requirements.txt` (pip lower bounds)

## Frameworks

**Core:**
- PyTorch 2.0.0+ (当前环境: 2.10.0) - 深度学习框架，用于模型训练和推理
- PyTorch Lightning 2.0.5 - 训练循环管理和分布式训练
- Transformers 4.40+ - Hugging Face Transformers 库 (Cola-DLM 使用)

**Symbolic Regression:**
- SymPy 1.11.0+ - 符号数学计算
- SymPyTorch 0.1.0+ - SymPy 和 PyTorch 桥接
- gplearn 0.4.2 - 遗传规划用于符号回归

**Scientific Computing:**
- NumPy 1.24.0+ - 数值计算
- SciPy 1.10.0+ - 科学计算和优化
- pandas 2.0.0+ - 数据处理
- scikit-learn 1.2.0+ - 机器学习工具和评估指标
- numexpr 2.8.0+ - 快速数值表达式评估
- XGBoost 1.7.5 - 梯度提升 (可选使用)

**优化与算法:**
- CMA-ES (自定义实现 `cma_es_modular.py`) - 协方差矩阵自适应进化策略
- BFGS (通过 SciPy) - 常数优化

**Visualization:**
- matplotlib 3.7.0+ - 绘图和可视化
- seaborn 0.12.0+ - 统计可视化

**Training & Monitoring:**
- Weights & Biases (wandb) 0.14.0+ - 实验跟踪和日志记录
- tqdm 4.65.0+ - 进度条显示

**开发工具:**
- PyYAML 6.0+ - YAML 配置文件解析
- requests 2.28.0+ - HTTP 请求

## Key Dependencies

**Critical:**
- torch>=2.0.0 - 核心深度学习框架
- transformers>=4.40 - Cola-DLM 语言模型依赖
- sympytorch>=0.1.0 - 符号表达式与张量转换
- pytorch-lightning==2.0.5 - 训练框架
- wandb>=0.14.0 - 实验跟踪

**Infrastructure:**
- NVIDIA CUDA 库 (cublas, cudnn, nccl, nvrtc, curand, cusolver, cusparse, cufft) - GPU 加速
- triton==2.0.0 - GPU 内核编译

**数据存储:**
- 本地文件系统 (无数据库依赖)

## Configuration

**Environment:**
- Conda 环境配置: `environment.yml`
- pip 依赖: `requirements.txt`
- Python 虚拟环境: `.venv/` (Python 3.10.12)
- 数据集元数据: `datasets/pmlb/datasets/*/metadata.yaml`

**Build:**
- 无需编译 (纯 Python)
- 代码直接运行，无构建步骤

**Cola-DLM 子模块:**
- 配置: `Cola-DLM-main/pyproject.toml`
- 依赖: `Cola-DLM-main/requirements.txt`

## Platform Requirements

**Development:**
- Python 3.10+
- CUDA 12.x (推荐 CUDA 12.8)
- NVIDIA GPU (推荐显存 >= 16GB)
- Linux (测试环境: Ubuntu 20.04+)

**Production:**
- 单机 GPU 训练/推理
- SLURM 集群支持 (通过 `symbolicregression/slurm.py`)
- 分布式训练支持 (PyTorch DDP)

**存储需求:**
- 模型权重: `weights/checkpoint.pth` (671MB), `weights/fm_best.pth` (932MB)
- 数据集: `datasets/` 目录 (PMLB + Feynman)
- 实验输出: `dump/`, `wandb/`, `.planning/`

---

*Stack analysis: 2026-06-09*
