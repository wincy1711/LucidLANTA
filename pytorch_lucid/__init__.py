"""
PyTorch Lucid: A collection of infrastructure and tools for research in neural network interpretability.

This is a PyTorch port of the original TensorFlow Lucid library, providing
tools for neural network visualization and interpretability research.

Main components:
- optvis: Optimization-based feature visualization framework
- modelzoo: Model loading and activation extraction utilities
- misc: Miscellaneous utilities and helper functions
- recipes: Pre-built visualization recipes and examples
"""

__version__ = "0.1.0"
__author__ = "PyTorch Lucid Contributors"

from . import optvis
from . import modelzoo
from . import misc
from . import recipes

__all__ = [
    "optvis",
    "modelzoo", 
    "misc",
    "recipes"
]