"""
setup.py — Build the lindblad_bench Python C extension.

Links against the static library built by CMake.
Run: pip install -e .  (from the python/ directory)
"""

import os
from setuptools import setup, Extension

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The static library location is overridable so CI / out-of-tree builds can
# point at their own build directory (e.g. LINDBLAD_BUILD_DIR=build-audit).
build_dir = os.environ.get("LINDBLAD_BUILD_DIR", os.path.join(root, "build"))

ext = Extension(
    "lindblad_bench._lindblad_ext",
    sources=["lindblad_ext.c"],
    include_dirs=[os.path.join(root, "include"), np.get_include()],
    library_dirs=[build_dir],
    libraries=["lindblad_bench", "m"],
    extra_compile_args=["-O3", "-march=native", "-std=c11"],
)

setup(
    name="lindblad_bench",
    version="0.1.0.dev0",
    description="Python bindings for lindblad-bench C library",
    packages=["lindblad_bench"],
    ext_modules=[ext],
    python_requires=">=3.10",
)
