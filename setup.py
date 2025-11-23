"""
Setup script for PyTorch Lucid package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("pytorch_lucid/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pytorch-lucid",
    version="0.1.0",
    author="PyTorch Lucid Contributors",
    author_email="",
    description="A PyTorch implementation of the Lucid library for neural network interpretability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pytorch-lucid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "examples": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "pytorch-lucid=pytorch_lucid.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "pytorch",
        "neural-networks",
        "visualization",
        "interpretability",
        "feature-visualization",
        "style-transfer",
        "deepdream",
        "lucid",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/pytorch-lucid/issues",
        "Source": "https://github.com/your-username/pytorch-lucid",
        "Documentation": "https://pytorch-lucid.readthedocs.io/",
    },
)