from setuptools import Extension, find_packages, setup

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required to build lsl._sparse_native") from exc


setup(
    name="living-synapse-language-model",
    version="0.1.0",
    packages=find_packages(include=["lsl", "lsl.*", "benchmarks", "benchmarks.*"]),
    ext_modules=[
        Extension(
            "lsl._sparse_native",
            sources=["lsl/_sparse_native.c"],
            include_dirs=[np.get_include()],
        )
    ],
)
