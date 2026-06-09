"""Cython 求值器性能基准测试。

对比 Python Node.val() 和 Cython eval_tree_flat 在典型训练配置下的性能。
"""
import numpy as np
import time
import sys

sys.path.insert(0, '.')

from symbolicregression.envs.generators import Node, NodeList
from symbolicregression.envs.cython_eval import eval_tree_flat


class FakeParams:
    use_abs = False


def build_tree():
    """构造一棵典型复杂度的表达式树（约 15-20 节点）。"""
    params = FakeParams()
    # mul(sin(x_0), add(x_0, cos(mul(x_0, 2.5))))
    x0 = Node('x_0', params)
    sin_x0 = Node('sin', params, [x0])

    x0b = Node('x_0', params)
    const_25 = Node('2.5', params)
    mul_inner = Node('mul', params, [x0b, const_25])
    cos_inner = Node('cos', params, [mul_inner])
    x0c = Node('x_0', params)
    add_inner = Node('add', params, [x0c, cos_inner])

    root = Node('mul', params, [sin_x0, add_inner])
    return NodeList([root])


# 构建测试数据
tree = build_tree()
compiled = []
for node in tree.nodes:
    oc, co, vd = node.compile_to_instructions()
    compiled.append((
        np.array(oc, dtype=np.int32),
        np.array(co, dtype=np.float64),
        np.array(vd, dtype=np.int32),
    ))

n_samples = 200
x = np.random.randn(n_samples, 1).astype(np.float64)

# Warmup
for _ in range(100):
    tree.val(x)
for _ in range(100):
    eval_tree_flat(np.ascontiguousarray(x), compiled[0][0], compiled[0][1], compiled[0][2])

# Benchmark Python
n_runs = 5000
t0 = time.perf_counter()
for _ in range(n_runs):
    tree.val(x)
t_python = (time.perf_counter() - t0) / n_runs * 1e6  # us

# Benchmark Cython
t0 = time.perf_counter()
for _ in range(n_runs):
    for opcodes, constants, var_dims in compiled:
        eval_tree_flat(np.ascontiguousarray(x), opcodes, constants, var_dims)
t_cython = (time.perf_counter() - t0) / n_runs * 1e6  # us

print(f"Python Node.val(): {t_python:.1f} us/call")
print(f"Cython eval_tree_flat: {t_cython:.1f} us/call")
print(f"Speedup: {t_python / t_cython:.1f}x")

# Benchmark val_cython (end-to-end)
t0 = time.perf_counter()
for _ in range(n_runs):
    tree.val_cython(x)
t_cython_e2e = (time.perf_counter() - t0) / n_runs * 1e6  # us

print(f"Cython val_cython (e2e): {t_cython_e2e:.1f} us/call")
print(f"E2E Speedup: {t_python / t_cython_e2e:.1f}x")
