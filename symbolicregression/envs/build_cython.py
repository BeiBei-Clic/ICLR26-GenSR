from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "symbolicregression.envs.cython_eval",
    sources=["symbolicregression/envs/cython_eval.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
)
