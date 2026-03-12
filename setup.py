from setuptools import setup, find_packages

setup(
    name="baht-benchmark",
    version="0.1.0",
    description="Unified benchmark suite for Byzantine Ad Hoc Teamwork",
    author="Lain Mustafaoglu",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pyyaml>=6.0",
        "torch>=2.0",
    ],
    extras_require={
        "dsse": ["DSSE"],
        "lbf": ["lbforaging"],
        "overcooked": ["overcooked-ai"],
        "smac": ["pysc2", "s2clientprotocol"],
        "matrix": ["matrix-games"],
        "all": ["DSSE", "lbforaging", "overcooked-ai", "pysc2", "s2clientprotocol", "matrix-games"],
    },
)
