import os
import sys

from setuptools import Extension, find_packages, setup

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required to build lsl._sparse_native") from exc


extra_compile_args = []
extra_link_args = []
enable_simd = os.environ.get("LSL_DISABLE_SIMD", "0").strip() != "1"
if os.name == "nt":
    extra_compile_args.extend(["/O2"])
    if enable_simd:
        extra_compile_args.extend(["/arch:AVX2"])
else:
    extra_compile_args.extend(["-O3"])
    if sys.platform != "darwin":
        extra_compile_args.extend(["-fno-math-errno"])
    if enable_simd:
        extra_compile_args.extend(["-mavx2", "-mfma"])


setup(
    name="living-synapse-language-model",
    version="0.1.0",
    packages=find_packages(include=["lsl", "lsl.*", "benchmarks", "benchmarks.*"]),
    ext_modules=[
        Extension(
            "lsl._sparse_native",
            sources=["lsl/_sparse_native.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
