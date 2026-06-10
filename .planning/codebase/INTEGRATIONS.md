# External Integrations

**Analysis Date:** 2026-06-09

## APIs & External Services

**实验跟踪与监控:**
- Weights & Biases (wandb) - 实验跟踪、日志记录和可视化
  - SDK/Client: `wandb>=0.14.0`
  - Auth: 环境变量 (默认: `WANDB_API_KEY`)
  - 使用位置: `train.py`, `LSO_eval.py`, `direct_eval.py`
  - 项目: `symbolic-regression-training` (训练), `symbolic-regression-example` (评估)
  - 功能: 记录训练损失、评估指标 (R2 score)、学习率、模型检查点

**预训练模型下载:**
- Google Drive - 预训练权重分发
  - 位置: `weights/checkpoint.pth` (671MB)
  - 下载脚本: `scripts/bootstrap_pretrained.sh`
  - 无需认证 (公开链接)

## Data Storage

**数据集:**
- 本地文件系统
  - 位置: `datasets/pmlb/datasets/`, `datasets/feynman/`
  - 格式: TSV.GZ (压缩制表符分隔值), CSV
  - 元数据: `metadata.yaml` (每个数据集)
  - 元数据 Schema: `datasets/pmlb/metadata.schema.json`

**模型权重:**
- 本地文件系统
  - 位置: `weights/` 目录
  - 文件: `checkpoint.pth` (预训练 CVAE), `fm_best.pth` (微调模型)
  - 大小: ~1.6GB 总计

**实验输出:**
- 本地文件系统
  - 训练检查点: `dump/` (由 `--dump_path` 参数指定)
  - 评估结果: `experiments/pmlb/results/`
  - WandB 本地缓存: `.wandb/`

**文件存储:**
- 无外部对象存储 (S3, GCS 等)
- 无数据库依赖

## Caching & Performance

**计算缓存:**
- Python `pickle` 序列化 - 模型状态保存
- `numpy.savez` - 数组数据缓存

**数值计算优化:**
- numexpr 2.8.0+ - 快速表达式评估
- Triton 2.0.0 - GPU 内核优化

## Authentication & Identity

**认证提供方:**
- 无集中认证服务
- 本地开发: 无需认证
- WandB: 可选 API Key (通过环境变量或 `wandb.login()`)

**权限管理:**
- 文件系统权限 (本地用户)
- 无用户/角色系统

## Monitoring & Observability

**错误跟踪:**
- Sentry SDK 1.19.1 - 错误监控 (集成在环境中)
  - 默认配置，但可能未激活

**日志:**
- Python `logging` 模块 - 结构化日志
- 自定义日志: `symbolicregression/logger.py`
- WandB - 实验指标和可视化
- 控制台输出 - tqdm 进度条

**指标:**
- R² score - 回归性能评估
- MSE (均方误差) - 拟合质量
- 复杂度 (表达式节点数) - 模型简洁性
- 准确度 (多个阈值: 1e-3, 1e-2, 1e-1, l1_biggio)

## CI/CD & Deployment

**托管:**
- 无云端部署
- 本地 GPU 训练
- SLURM 集群支持 (通过 `symbolicregression/slurm.py`)

**CI Pipeline:**
- 无 GitHub Actions / CI/CD
- 手动执行脚本: `scripts/train.sh`, `scripts/eval.sh`

**版本控制:**
- Git (当前分支: `cola_v2`)
- 主分支: `main`

## Environment Configuration

**必需环境变量:**
- `CUDA_VISIBLE_DEVICES` - GPU 设备选择
- `WANDB_API_KEY` - WandB 认证 (可选)
- `WANDB_DISABLED` - 禁用 WandB (可选)

**可选环境变量:**
- `PYTHONPATH` - Python 模块搜索路径
- `SLURM_*` - SLURM 集群变量

**Secrets 位置:**
- 无集中 secrets 管理
- 环境变量或本地配置

**配置文件:**
- `environment.yml` - Conda 环境配置
- `requirements.txt` - pip 依赖
- `parsers.py` - 命令行参数定义

## Webhooks & Callbacks

**Incoming:**
- 无 Webhook 端点

**Outgoing:**
- WandB SDK 上传 - 实验指标和检查点
- 无主动 HTTP 调用

**回调:**
- SLURM 信号处理 - `symbolicregression/slurm.py::init_signal_handler()`
- 中断信号捕获 - 优雅的训练停止

## External Dependencies

**PMLB (Penn Machine Learning Benchmark):**
- 数据集仓库: `datasets/pmlb/`
- 元数据 Schema: JSON Schema 验证
- 来源: Epistasis Lab

**Feynman Symbolic Regression Database:**
- 数据集: `datasets/feynman/`
- 来源: MIT Space (https://space.mit.edu/home/tegmark/aifeynman.html)
- 参考: AI Feynman (Udrescu & Tegmark, 2019)

**代码引用:**
- Multimodal Math Pretraining - 数学表达式编码
- Facebook Symbolic Regression - CVAE 架构基础

## Compute Infrastructure

**GPU 加速:**
- NVIDIA CUDA 12.8
- cuDNN 8.5.0
- NCCL 2.14.3 (多 GPU 通信)
- 支持架构: CUDA-capable GPUs

**分布式训练:**
- PyTorch DistributedDataParallel (DDP)
- 多 GPU 单机
- SLURM 集群支持

**CPU 计算:**
- multiprocessing - 并行候选评估
- joblib - scikit-learn 并行后端

---

*Integration audit: 2026-06-09*
