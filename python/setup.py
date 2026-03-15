"""
setup.py — Build the lindblad_bench Python C extension.

Links against the static library built by CMake.
Run: pip install -e .  (from the python/ directory)
"""

import os
import sys
from setuptools import setup, Extension

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(root, "build")

ext = Extension(
    "lindblad_bench._lindblad_ext",
    sources=["lindblad_ext.c"],
    include_dirs=[os.path.join(root, "include")],
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
