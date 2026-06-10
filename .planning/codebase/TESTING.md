# Testing Patterns

**Analysis Date:** 2026-06-09

## Test Framework

**Runner:**
- `nose` - 测试框架
- `parameterized` - 参数化测试
- 配置：未找到独立配置文件

**Assertion Library:**
- 标准 `assert` 语句
- `nose.tools.assert_raises` 用于异常测试

**Run Commands:**
```bash
# 运行所有测试（推测）
nosetests datasets/pmlb/tests/

# 运行特定测试文件
nosetests datasets/pmlb/tests/test_pmlb.py

# 运行特定测试
nosetests datasets/pmlb/tests/test_pmlb.py:test_fetch_data_1
```

**Coverage:**
- 未强制执行覆盖率要求
- 未找到覆盖率配置文件

## Test File Organization

**Location:**
- 测试与源代码分离（不在同一目录）
- 项目测试：`./datasets/pmlb/tests/`

**Naming:**
- 测试文件：`test_*.py` 模式
- 测试函数：`test_*` 模式
- 示例：`test_pmlb.py`, `test_metadata.py`, `test_fetch_nearest_dataset_names.py`

**Structure:**
```
datasets/pmlb/
├── pmlb/
│   ├── __init__.py
│   ├── pmlb.py
│   └── ...
└── tests/
    ├── test_pmlb.py
    ├── test_metadata.py
    └── test_fetch_nearest_dataset_names.py
```

## Test Structure

**Suite Organization:**
```python
# 简单测试函数
def test_fetch_data_1():
    """Test fetch_data can fetch data from GitHub."""
    mushroom = fetch_data('mushroom')
    assert not mushroom.empty
    assert not mushroom.isnull().values.any()

# 参数化测试
@parameterized.expand(all_yfs)
def test_all_yaml_files(yf):
    "Check basic information in yaml files."
    print("\nTesting {}".format(yf))
    folder_name = yf.split("/")[-2]
    with open(yf) as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    assert metadata['dataset'] == folder_name
    assert metadata['task'] in ["classification", "regression"]
```

**Patterns:**
- **Setup**: 在测试函数内部进行设置
- **Teardown**: 使用临时目录并在测试后清理
- **Assertion**: 直接使用 `assert` 语句
- **示例（来自 `test_pmlb.py`）**：
```python
def test_fetch_data_5():
    """Test fetch_data can fetch data from local cache
     but the dataset is not available in local cache"""
    cachedir = mkdtemp()
    dataset_name = 'mushroom'
    mushroom = fetch_data(dataset_name, local_cache_dir=cachedir)
    out_cache_data = path.join(cachedir, dataset_name,
                                dataset_name+'.tsv.gz')
    assert not mushroom.empty
    assert path.isfile(out_cache_data)
    rmtree(cachedir)  # 清理
```

## Mocking

**Framework:** 未检测到专门的mocking框架

**Patterns:**
- 测试使用真实数据（GitHub数据集）
- 临时文件系统用于测试文件操作
- 示例（来自 `test_pmlb.py`）：
```python
cachedir = mkdtemp()  # 创建临时目录
# ... 测试代码 ...
rmtree(cachedir)      # 清理临时目录
```

**What to Mock:**
- 外部API调用（在真实测试中未mock）
- 文件系统操作（使用临时目录）

**What NOT to Mock:**
- 核心业务逻辑
- 数据验证逻辑

## Fixtures and Factories

**Test Data:**
```python
# 使用真实数据集
mushroom = fetch_data('mushroom')

# 使用参数化测试多个数据集
yaml_files = glob.glob('datasets/*/*.yaml')
all_yfs = [(yf,) for yf in yaml_files]
```

**Location:**
- 测试数据从GitHub或本地缓存加载
- 未使用专门的fixture目录

## Coverage

**Requirements:** 未强制执行

**View Coverage:**
```bash
# 可能的命令（未验证）
nosetests --with-coverage
```

**当前状态:**
- 覆盖率工具未配置
- 测试仅覆盖PMLB数据集功能
- 核心模型代码（`model.py`, `train.py`等）无测试

## Test Types

**Unit Tests:**
- 范围：数据获取、元数据验证
- 方法：隔离函数测试
- 示例：`test_fetch_data_1()` 测试数据获取功能

**Integration Tests:**
- 范围：完整数据集加载和验证流程
- 方法：测试数据集文件和元数据一致性
- 示例：`test_dataset()` 验证YAML元数据与实际数据集文件

**E2E Tests:**
- **未使用**

## Common Patterns

**Async Testing:**
- **不适用**（项目主要为同步代码）

**Error Testing:**
```python
# 测试预期的异常
def test_fetch_data_3():
    """Test fetch_data can not fetch data with incorrect dataset name."""
    assert_raises(ValueError, fetch_data, "musroom")

def test_get_dataset_url_2():
    """Test get_dataset_url can not fetch data from GitHub with
    incorrect dataset name."""
    dataset_name = 'mushrom'
    assert_raises(ValueError, get_dataset_url,
                                GITHUB_URL,
                                dataset_name,
                                suffix)
```

**参数化测试:**
```python
@parameterized.expand(all_yfs)
def test_all_yaml_files(yf):
    # 对所有yaml文件运行相同测试
```

**数据验证测试:**
```python
def test_dataset(yf):
    "Check basic information in dataset files."
    # 验证列名一致性
    cols = set(list(dataset.columns))
    exported_cols = set([str(s['name']) for s in metadata['features']]
            +['target'])
    assert exported_cols == cols
```

## 测试覆盖缺口

**未测试的关键组件:**
- `model.py` - 核心VAESymbolicRegressor类
- `train.py` - 训练流程
- `LSO_fit.py` - 演化策略优化
- `LSO_eval.py` - 评估逻辑
- `const_opt.py` - 常数优化
- `cma_es_modular.py` - CMA-ES实现

**建议添加测试:**
1. **模型测试**：验证前向传播、编码/解码功能
2. **优化器测试**：验证CMA-ES更新逻辑
3. **评估测试**：验证R2计算等指标
4. **集成测试**：端到端训练流程

---

*Testing analysis: 2026-06-09*
