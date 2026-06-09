# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, exp, log, sqrt, fabs, tan, atan, asin, acos, pow, fmax, fmin, NAN

np.import_array()

# 操作码常量
cdef int OP_CONST = 0
cdef int OP_VAR   = 1
cdef int OP_ADD   = 2
cdef int OP_SUB   = 3
cdef int OP_MUL   = 4
cdef int OP_DIV   = 5
cdef int OP_SIN   = 6
cdef int OP_COS   = 7
cdef int OP_EXP   = 8
cdef int OP_LOG   = 9
cdef int OP_SQRT  = 10
cdef int OP_ABS   = 11
cdef int OP_POW2  = 12
cdef int OP_POW3  = 13
cdef int OP_INV   = 14
cdef int OP_NEG   = 15
cdef int OP_TAN   = 16
cdef int OP_ATAN  = 17
cdef int OP_ASIN  = 18
cdef int OP_ACOS  = 19
cdef int OP_POW   = 20
cdef int OP_MAX   = 21
cdef int OP_MIN   = 22
cdef int OP_SIGN  = 23
cdef int OP_STEP  = 24


def eval_tree_flat(double[:, ::1] x, int[:] opcodes, double[:] constants, int[:] var_dims):
    """栈式求值器：对扁平指令列表执行表达式求值。

    参数:
        x: (n_samples, n_features) C-contiguous float64 输入数组
        opcodes: 指令序列（后序遍历）
        constants: 常量池
        var_dims: 变量维度池
    返回:
        result: (n_samples,) float64 求值结果
    """
    cdef int n = x.shape[0]
    cdef int n_instr = opcodes.shape[0]
    cdef int n_stack = 0
    cdef int const_idx = 0
    cdef int var_idx = 0

    cdef np.ndarray[np.float64_t, ndim=2] stack_arr = np.empty((n_instr, n), dtype=np.float64)
    cdef double[:, ::1] sv = stack_arr

    cdef int i, j, op, dim
    cdef double val

    for i in range(n_instr):
        op = opcodes[i]

        if op == OP_CONST:
            val = constants[const_idx]
            const_idx += 1
            for j in range(n):
                sv[n_stack, j] = val
            n_stack += 1

        elif op == OP_VAR:
            dim = var_dims[var_idx]
            var_idx += 1
            for j in range(n):
                sv[n_stack, j] = x[j, dim]
            n_stack += 1

        elif op == OP_ADD:
            n_stack -= 1
            for j in range(n):
                sv[n_stack - 1, j] += sv[n_stack, j]

        elif op == OP_SUB:
            n_stack -= 1
            for j in range(n):
                sv[n_stack - 1, j] -= sv[n_stack, j]

        elif op == OP_MUL:
            n_stack -= 1
            for j in range(n):
                val = sv[n_stack - 1, j] * sv[n_stack, j]
                sv[n_stack - 1, j] = val

        elif op == OP_DIV:
            n_stack -= 1
            for j in range(n):
                if sv[n_stack, j] == 0.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = sv[n_stack - 1, j] / sv[n_stack, j]

        elif op == OP_SIN:
            for j in range(n):
                sv[n_stack - 1, j] = sin(sv[n_stack - 1, j])

        elif op == OP_COS:
            for j in range(n):
                sv[n_stack - 1, j] = cos(sv[n_stack - 1, j])

        elif op == OP_EXP:
            for j in range(n):
                sv[n_stack - 1, j] = exp(sv[n_stack - 1, j])

        elif op == OP_LOG:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val <= 0.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = log(val)

        elif op == OP_SQRT:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val < 0.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = sqrt(val)

        elif op == OP_ABS:
            for j in range(n):
                sv[n_stack - 1, j] = fabs(sv[n_stack - 1, j])

        elif op == OP_POW2:
            for j in range(n):
                val = sv[n_stack - 1, j]
                sv[n_stack - 1, j] = val * val

        elif op == OP_POW3:
            for j in range(n):
                val = sv[n_stack - 1, j]
                sv[n_stack - 1, j] = val * val * val

        elif op == OP_INV:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val == 0.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = 1.0 / val

        elif op == OP_NEG:
            for j in range(n):
                sv[n_stack - 1, j] = -sv[n_stack - 1, j]

        elif op == OP_TAN:
            for j in range(n):
                sv[n_stack - 1, j] = tan(sv[n_stack - 1, j])

        elif op == OP_ATAN:
            for j in range(n):
                sv[n_stack - 1, j] = atan(sv[n_stack - 1, j])

        elif op == OP_ASIN:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val < -1.0 or val > 1.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = asin(val)

        elif op == OP_ACOS:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val < -1.0 or val > 1.0:
                    sv[n_stack - 1, j] = NAN
                else:
                    sv[n_stack - 1, j] = acos(val)

        elif op == OP_POW:
            n_stack -= 1
            for j in range(n):
                sv[n_stack - 1, j] = pow(sv[n_stack - 1, j], sv[n_stack, j])

        elif op == OP_MAX:
            n_stack -= 1
            for j in range(n):
                sv[n_stack - 1, j] = fmax(sv[n_stack - 1, j], sv[n_stack, j])

        elif op == OP_MIN:
            n_stack -= 1
            for j in range(n):
                sv[n_stack - 1, j] = fmin(sv[n_stack - 1, j], sv[n_stack, j])

        elif op == OP_SIGN:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val >= 0.0:
                    sv[n_stack - 1, j] = 1.0
                else:
                    sv[n_stack - 1, j] = -1.0

        elif op == OP_STEP:
            for j in range(n):
                val = sv[n_stack - 1, j]
                if val > 0.0:
                    sv[n_stack - 1, j] = val
                else:
                    sv[n_stack - 1, j] = 0.0

    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    for j in range(n):
        result[j] = sv[0, j]
    return result
