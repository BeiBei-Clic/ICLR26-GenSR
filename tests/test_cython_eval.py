"""
Cython 栈式求值器正确性测试。

运行方式：
    python tests/test_cython_eval.py        # 直接执行
    pytest tests/test_cython_eval.py -x -v  # pytest 执行（需要完整 pytest 依赖）
"""
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from symbolicregression.envs.cython_eval import eval_tree_flat
from symbolicregression.envs.generators import Node, NodeList, RandomFunctions

# ============================================================================
# 操作码定义（与 cython_eval.pyx 严格对应）
# ============================================================================
OP_CONST, OP_VAR, OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3, 4, 5
OP_SIN, OP_COS, OP_EXP, OP_LOG, OP_SQRT = 6, 7, 8, 9, 10
OP_ABS, OP_POW2, OP_POW3, OP_INV, OP_NEG = 11, 12, 13, 14, 15
OP_TAN, OP_ATAN, OP_ASIN, OP_ACOS = 16, 17, 18, 19
OP_POW, OP_MAX, OP_MIN, OP_SIGN, OP_STEP = 20, 21, 22, 23, 24

OP_MAP = {
    'add': OP_ADD, 'sub': OP_SUB, 'mul': OP_MUL, 'div': OP_DIV,
    'sin': OP_SIN, 'cos': OP_COS, 'exp': OP_EXP, 'log': OP_LOG,
    'sqrt': OP_SQRT, 'abs': OP_ABS, 'pow2': OP_POW2, 'pow3': OP_POW3,
    'inv': OP_INV,
    'tan': OP_TAN, 'arctan': OP_ATAN, 'arcsin': OP_ASIN, 'arccos': OP_ACOS,
    'pow': OP_POW, 'max': OP_MAX, 'min': OP_MIN, 'sign': OP_SIGN, 'step': OP_STEP,
}

MATH_CONSTANTS = {'e': np.e, 'pi': np.pi, 'euler_gamma': np.euler_gamma}


def compile_tree(node):
    """后序遍历 Node 树，生成 (opcodes, constants, var_dims) 三个列表。"""
    opcodes, constants, var_dims = [], [], []
    for child in node.children:
        oc, co, vd = compile_tree(child)
        opcodes.extend(oc)
        constants.extend(co)
        var_dims.extend(vd)
    val = node.value
    if val.startswith('x_'):
        dim = int(val.split('_')[1])
        opcodes.append(OP_VAR)
        var_dims.append(dim)
    elif val in OP_MAP:
        opcodes.append(OP_MAP[val])
    else:
        opcodes.append(OP_CONST)
        if val in MATH_CONSTANTS:
            constants.append(MATH_CONSTANTS[val])
        else:
            constants.append(float(val))
    return opcodes, constants, var_dims


def eval_node_cython(node, x):
    """用 Cython 求值单个 Node 树。"""
    oc, co, vd = compile_tree(node)
    return eval_tree_flat(
        np.ascontiguousarray(x, dtype=np.float64),
        np.array(oc, dtype=np.int32),
        np.array(co, dtype=np.float64),
        np.array(vd, dtype=np.int32),
    )


class MockParams:
    use_abs = False


def _make_node(value, children=None):
    return Node(value, MockParams(), children)


# ============================================================================
# 测试用例
# ============================================================================


def test_correctness_basic():
    """测试 mul(add(x_0, 1.0), sin(x_0)) 的正确性。"""
    x0 = _make_node('x_0')
    c1 = _make_node('1.0')
    add = _make_node('add', [x0, c1])
    sin_x0 = _make_node('sin', [_make_node('x_0')])
    tree = _make_node('mul', [add, sin_x0])

    x = np.array([[0.5], [1.0], [2.0], [-0.3]])

    py_result = tree.val(x)
    cy_result = eval_node_cython(tree, x)

    assert np.allclose(py_result, cy_result, rtol=1e-12, atol=1e-12, equal_nan=True), \
        f"Basic test failed: Python={py_result}, Cython={cy_result}"
    print("  PASSED")


def test_correctness_all_operators():
    """对每种操作符构造简单表达式树，对比 Node.val() vs eval_tree_flat。"""
    x = np.array([[0.5], [1.0], [2.0], [0.3]])

    tests = {
        'add': _make_node('add', [_make_node('x_0'), _make_node('x_0')]),
        'sub': _make_node('sub', [_make_node('x_0'), _make_node('x_0')]),
        'mul': _make_node('mul', [_make_node('x_0'), _make_node('x_0')]),
        'div': _make_node('div', [_make_node('x_0'), _make_node('2.0')]),
        'sin': _make_node('sin', [_make_node('x_0')]),
        'cos': _make_node('cos', [_make_node('x_0')]),
        'exp': _make_node('exp', [_make_node('x_0')]),
        'abs': _make_node('abs', [_make_node('x_0')]),
        'pow2': _make_node('pow2', [_make_node('x_0')]),
        'pow3': _make_node('pow3', [_make_node('x_0')]),
        'inv': _make_node('inv', [_make_node('x_0')]),
    }

    # log 和 sqrt 需要正数输入
    x_pos = np.array([[0.5], [1.0], [2.0], [0.3]])
    tests_pos = {
        'log': _make_node('log', [_make_node('x_0')]),
        'sqrt': _make_node('sqrt', [_make_node('x_0')]),
    }

    for name, tree in tests.items():
        py = tree.val(x)
        cy = eval_node_cython(tree, x)
        assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), \
            f"{name} failed: Python={py}, Cython={cy}"
        print(f"  {name}: PASSED")

    for name, tree in tests_pos.items():
        py = tree.val(x_pos)
        cy = eval_node_cython(tree, x_pos)
        assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), \
            f"{name} failed: Python={py}, Cython={cy}"
        print(f"  {name}: PASSED")


def test_new_operators():
    """测试新增的 9 个操作符：tan, arctan, arcsin, arccos, pow, max, min, sign, step。"""
    x = np.array([[0.5], [1.0], [-0.5], [0.3]])

    # tan
    tree = _make_node('tan', [_make_node('x_0')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"tan: py={py}, cy={cy}"
    print("  tan: PASSED")

    # arctan
    tree = _make_node('arctan', [_make_node('x_0')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"arctan: py={py}, cy={cy}"
    print("  arctan: PASSED")

    # arcsin -- 需要 [-1, 1] 输入
    x_bounded = np.array([[0.5], [-0.3], [1.0], [-1.0]])
    tree = _make_node('arcsin', [_make_node('x_0')])
    py = tree.val(x_bounded)
    cy = eval_node_cython(tree, x_bounded)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"arcsin: py={py}, cy={cy}"
    print("  arcsin: PASSED")

    # arcsin 越界 -> NaN
    x_oob = np.array([[2.0], [-2.0]])
    py = tree.val(x_oob)
    cy = eval_node_cython(tree, x_oob)
    assert np.allclose(py, cy, equal_nan=True), f"arcsin oob: py={py}, cy={cy}"
    print("  arcsin(oob): PASSED")

    # arccos
    tree = _make_node('arccos', [_make_node('x_0')])
    py = tree.val(x_bounded)
    cy = eval_node_cython(tree, x_bounded)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"arccos: py={py}, cy={cy}"
    print("  arccos: PASSED")

    # pow -- 二元操作符
    tree = _make_node('pow', [_make_node('x_0'), _make_node('2.0')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"pow: py={py}, cy={cy}"
    print("  pow: PASSED")

    # max
    tree = _make_node('max', [_make_node('x_0'), _make_node('0.5')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"max: py={py}, cy={cy}"
    print("  max: PASSED")

    # min
    tree = _make_node('min', [_make_node('x_0'), _make_node('0.5')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"min: py={py}, cy={cy}"
    print("  min: PASSED")

    # sign
    tree = _make_node('sign', [_make_node('x_0')])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.allclose(py, cy, rtol=1e-12, equal_nan=True), f"sign: py={py}, cy={cy}"
    print("  sign: PASSED")

    # step -- Python Node.val() 中 step 实现有向量化的 bug，
    # 所以直接用 np.maximum(x, 0) 作为正确基准
    tree = _make_node('step', [_make_node('x_0')])
    cy = eval_node_cython(tree, x)
    expected = np.maximum(x[:, 0], 0.0)
    assert np.allclose(cy, expected, rtol=1e-12), f"step: cy={cy}, expected={expected}"
    print("  step: PASSED")


def test_nan_handling():
    """测试 NaN 边界条件：div-by-zero、log(负数)、sqrt(负数)、inv(0)、inf*0。"""
    # div(x_0, 0.0) -- x_0=1.0，分母为 0
    tree = _make_node('div', [_make_node('x_0'), _make_node('0.0')])
    x = np.array([[1.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.isnan(py[0]) and np.isnan(cy[0]), f"div by zero: py={py}, cy={cy}"
    print("  div-by-zero: PASSED")

    # log(x_0) -- x_0=-1.0
    tree = _make_node('log', [_make_node('x_0')])
    x = np.array([[-1.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.isnan(py[0]) and np.isnan(cy[0]), f"log(negative): py={py}, cy={cy}"
    print("  log(negative): PASSED")

    # sqrt(x_0) -- x_0=-1.0
    tree = _make_node('sqrt', [_make_node('x_0')])
    x = np.array([[-1.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.isnan(py[0]) and np.isnan(cy[0]), f"sqrt(negative): py={py}, cy={cy}"
    print("  sqrt(negative): PASSED")

    # inv(x_0) -- x_0=0.0
    tree = _make_node('inv', [_make_node('x_0')])
    x = np.array([[0.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.isnan(py[0]) and np.isnan(cy[0]), f"inv(0): py={py}, cy={cy}"
    print("  inv(0): PASSED")

    # mul(inf, 0.0) -- 构造 mul(exp(x_0), 0.0), x_0=710 使得 exp(710)=inf
    tree = _make_node('mul', [
        _make_node('exp', [_make_node('x_0')]),
        _make_node('0.0'),
    ])
    x = np.array([[710.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert np.isnan(py[0]) and np.isnan(cy[0]), \
        f"inf*0: py={py}, cy={cy}"
    print("  inf*0: PASSED")

    # 混合 NaN/有效值场景
    tree = _make_node('div', [_make_node('x_0'), _make_node('x_0')])
    x = np.array([[2.0], [0.0], [4.0]])
    py = tree.val(x)
    cy = eval_node_cython(tree, x)
    assert py[0] == cy[0] == 1.0, "div valid(0)"
    assert np.isnan(py[1]) and np.isnan(cy[1]), f"div 0/0 NaN: py={py[1]}, cy={cy[1]}"
    assert py[2] == cy[2] == 1.0, "div valid(2)"
    print("  mixed NaN/valid: PASSED")


def test_nodelist():
    """使用 RandomFunctions 生成随机 NodeList，对比求值结果。"""
    from parsers import get_parser
    from symbolicregression.envs.environment import SPECIAL_WORDS

    parser = get_parser()
    params = parser.parse_args([])
    generator = RandomFunctions(params, SPECIAL_WORDS)
    rng = np.random.RandomState(42)

    tree, input_dim, output_dim, _, _ = generator.generate_multi_dimensional_tree(
        rng, input_dimension=1, output_dimension=1,
    )

    x = rng.randn(50, input_dim)
    x = np.ascontiguousarray(x, dtype=np.float64)

    py_result = tree.val(np.copy(x))

    cy_results = []
    for node in tree.nodes:
        oc, co, vd = compile_tree(node)
        r = eval_tree_flat(
            x,
            np.array(oc, dtype=np.int32),
            np.array(co, dtype=np.float64),
            np.array(vd, dtype=np.int32),
        )
        cy_results.append(np.expand_dims(r, -1))
    cy_result = np.concatenate(cy_results, axis=-1)

    for col in range(output_dim):
        py_col = py_result[:, col]
        cy_col = cy_result[:, col]
        both_valid = ~(np.isnan(py_col) | np.isnan(cy_col))
        if both_valid.sum() > 0:
            assert np.allclose(py_col[both_valid], cy_col[both_valid], rtol=1e-10, atol=1e-10), \
                f"NodeList col={col} valid mismatch"
        py_nan = np.isnan(py_col)
        cy_nan = np.isnan(cy_col)
        assert np.array_equal(py_nan, cy_nan), \
            f"NodeList col={col} NaN positions differ: py_nan={np.where(py_nan)}, cy_nan={np.where(cy_nan)}"
    print("  PASSED")


def test_compilation_cache():
    """验证同一棵树编译一次、用不同输入求值两次。"""
    # 构造 sin(exp(x_0 + 2.0))
    tree = _make_node('sin', [
        _make_node('exp', [
            _make_node('add', [_make_node('x_0'), _make_node('2.0')])
        ])
    ])

    oc, co, vd = compile_tree(tree)
    opcodes = np.array(oc, dtype=np.int32)
    constants = np.array(co, dtype=np.float64)
    var_dims = np.array(vd, dtype=np.int32)

    x1 = np.ascontiguousarray(np.array([[0.5], [1.0], [2.0]]), dtype=np.float64)
    x2 = np.ascontiguousarray(np.array([[-1.0], [0.0], [3.0]]), dtype=np.float64)

    cy1 = eval_tree_flat(x1, opcodes, constants, var_dims)
    cy2 = eval_tree_flat(x2, opcodes, constants, var_dims)

    py1 = tree.val(x1)
    py2 = tree.val(x2)

    assert np.allclose(py1, cy1, rtol=1e-12, equal_nan=True), "cache x1 mismatch"
    assert np.allclose(py2, cy2, rtol=1e-12, equal_nan=True), "cache x2 mismatch"
    print("  PASSED")


def test_speedup():
    """构造深度 5 的树，对比 Node.val() 和 eval_tree_flat 的速度。"""
    # 构造 add(mul(sin(x_0), cos(x_0)), mul(exp(x_0), log(add(x_0, 2.0))))
    # 再包一层 sqrt(abs(...))
    inner = _make_node('add', [
        _make_node('mul', [
            _make_node('sin', [_make_node('x_0')]),
            _make_node('cos', [_make_node('x_0')]),
        ]),
        _make_node('mul', [
            _make_node('exp', [_make_node('x_0')]),
            _make_node('log', [_make_node('add', [_make_node('x_0'), _make_node('2.0')])]),
        ]),
    ])
    tree = _make_node('sqrt', [_make_node('abs', [inner])])

    x = np.ascontiguousarray(np.random.randn(200, 1), dtype=np.float64)

    oc, co, vd = compile_tree(tree)
    opcodes = np.array(oc, dtype=np.int32)
    constants = np.array(co, dtype=np.float64)
    var_dims = np.array(vd, dtype=np.int32)

    # 先验证正确性
    py = tree.val(x)
    cy = eval_tree_flat(x, opcodes, constants, var_dims)
    assert np.allclose(py, cy, rtol=1e-10, equal_nan=True), "speedup test correctness failed"

    # Python 基准
    n_iter = 1000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        tree.val(x)
    t_py = time.perf_counter() - t0

    # Cython 基准
    t0 = time.perf_counter()
    for _ in range(n_iter):
        eval_tree_flat(x, opcodes, constants, var_dims)
    t_cy = time.perf_counter() - t0

    print(f"  Python: {t_py * 1e6 / n_iter:.1f} us/call")
    print(f"  Cython: {t_cy * 1e6 / n_iter:.1f} us/call")
    print(f"  Speedup: {t_py / t_cy:.1f}x")


# ============================================================================
# 直接执行入口
# ============================================================================
if __name__ == '__main__':
    tests = [
        ('test_correctness_basic', test_correctness_basic),
        ('test_correctness_all_operators', test_correctness_all_operators),
        ('test_new_operators', test_new_operators),
        ('test_nan_handling', test_nan_handling),
        ('test_nodelist', test_nodelist),
        ('test_compilation_cache', test_compilation_cache),
        ('test_speedup', test_speedup),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}:")
        fn()
        passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
