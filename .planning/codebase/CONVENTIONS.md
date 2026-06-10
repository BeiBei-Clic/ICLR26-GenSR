# Coding Conventions

**Analysis Date:** 2026-06-09

## Naming Patterns

**Files:**
- 使用小写字母和下划线命名，如 `train.py`, `LSO_eval.py`, `const_opt.py`
- 模块文件使用描述性名称，如 `parsers.py`, `model.py`
- 测试文件使用 `test_*.py` 模式

**Functions:**
- 使用小写字母和下划线命名，如 `evaluate_pmlb_lso`, `reload_model`, `read_file`
- 内部函数使用前导下划线，如 `_mp_eval_one`, `_update_params`
- 函数名应具有描述性，清楚表达其功能

**Variables:**
- 使用小写字母和下划线命名，如 `batch_results`, `problem_name`, `target_noise`
- 布尔变量使用 `is_`, `has_` 前缀，如 `is_master`, `hasattr`
- 缩写使用全大写，如 `X`, `y`（数学变量），`mse`, `r2`

**Types/Classes:**
- 使用驼峰命名法（PascalCase），如 `VAESymbolicRegressor`, `DiagonalCMAES`
- 类名应该是名词，描述其代表的实体

## Code Style

**Formatting:**
- 没有使用自动格式化工具（未检测到 `.prettierrc`, `black`, `yapf` 配置）
- 使用4空格缩进（Python标准）
- 行长度未强制限制，但保持可读性
- 在二元运算符周围使用空格

**Linting:**
- 未检测到 linting 配置（无 `.flake8`, `.pylintrc`, `pyproject.toml`）
- 代码质量依赖人工审查

## Import Organization

**Order:**
1. 标准库导入（`os`, `sys`, `time`, `random`）
2. 第三方库导入（`numpy`, `torch`, `pandas`, `wandb`）
3. 本地模块导入（`symbolicregression.*`, 相对导入）

**示例（来自 `train.py`）：**
```python
import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import copy
import wandb

import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from parsers import get_parser
```

**Path Aliases:**
- 未使用路径别名
- 使用相对导入和绝对导入混合

## Error Handling

**Patterns:**
- 优先使用断言进行参数验证，如 `assert torch.cuda.is_available()`
- 使用 try-except 块捕获特定异常
- 多进程worker中使用 try-except 并返回失败状态
- 异常处理示例（来自 `LSO_fit.py`）：
```python
try:
    eq_outputs = gen2eq(_mp_env, _mp_params, dummy_latent, gen_tensor,
                        _mp_sample, skeleton_snapshot)
    # 处理结果
except Exception:
    pass
return idx, {'success': False, 'r2': 0, 'fitness': 0}
```

**错误传播:**
- 在关键路径上让异常向上传播
- 在评估函数中捕获异常并提供默认值
- 使用 `logger.warning()` 记录非致命错误

## Logging

**Framework:** `symbolicregression.logger`（自定义logger）和 `wandb`

**Patterns:**
- 使用不同级别：`logger.info()`, `logger.warning()`, `print()`
- Wandb用于实验跟踪和指标记录
- 重要操作使用日志记录，如模型加载、评估开始
- 示例（来自 `train.py`）：
```python
logger.info(f"Starting epoch {epoch} evaluation on {len(eval_problems)} Feynman problems...")
logger.info(f"Problem {problem_name}: R2 = {r2_score:.4f}")
logger.warning(f"Error evaluating {problem_name}: {e}")
```

**进度跟踪:**
- 使用 `tqdm` 进度条进行长时间运行的任务
- Wandb日志用于训练和评估指标

## Comments

**When to Comment:**
- 在函数开头使用docstring描述功能
- 复杂逻辑处添加行内注释
- 模块分隔使用注释块
- 示例（来自 `LSO_fit.py`）：
```python
# ── Multiprocessing worker globals (inherited via fork) ──────────────
_mp_env = None
_mp_params = None
_mp_sample = None
```

**JSDoc/TSDoc:**
- 使用Python docstring（三引号）
- 简短描述函数目的
- 参数和返回值类型通常不在docstring中注明

## Function Design

**Size:**
- 函数长度变化很大，从10行到300+行
- 关键函数如 `evaluate_pmlb_lso` 较长（400+行）
- 优先功能完整性而非函数长度限制

**Parameters:**
- 使用位置参数和关键字参数混合
- 复杂函数使用参数对象（`params`, `env`）
- 示例（来自 `const_opt.py`）：
```python
def evaluate_tree(env, tree, X, y, metric):
    # 实现
```

**Return Values:**
- 返回单个值或元组
- 多返回值使用元组解包
- 成功/失败状态通过返回值或异常传达
- 示例（来自 `model.py`）：
```python
def forward(self, samples, max_len, return_logvar=False):
    prior_mu, prior_logvar = self._encode(samples)
    latent_repr = self.feature_fusion(prior_mu, prior_logvar)
    generations, gen_len = self.generate_from_latent_direct(latent_repr)
    outputs = (prior_mu, generations, gen_len)
    if return_logvar:
        outputs = (prior_mu, generations, gen_len, prior_logvar)
    return outputs
```

## Module Design

**Exports:**
- 使用 `__all__` 明确导出符号，如 `model.py`:
```python
__all__ = ['VAESymbolicRegressor']
```

**Barrel Files:**
- 未大量使用barrel文件（`__init__.py`汇总导入）
- 主要模块直接导入和使用

**Circular Dependencies:**
- 通过延迟导入避免循环依赖
- 在函数内部导入有时用于打破依赖循环

## Special Patterns

**Multiprocessing:**
- 使用全局变量在worker间传递数据
- Worker函数设计为接受参数并返回结果
- 示例（来自 `LSO_fit.py`）：
```python
_mp_env = None
_mp_params = None
_mp_sample = None

def _mp_eval_one(args):
    # 使用全局变量
```

**Model Management:**
- 模块字典用于管理多个模型组件
- 模型加载使用 `reload_model` 函数处理版本兼容性
- 示例（来自 `model.py`）：
```python
self.modules = modules
self.vae_model = self.modules["cvae"]
self.decoder = self.modules["seq_decoder"]
```

**Device Management:**
- 显式设备管理：`.to(params.device)`
- CUDA可用性检查：`assert torch.cuda.is_available()`

---

*Convention analysis: 2026-06-09*
