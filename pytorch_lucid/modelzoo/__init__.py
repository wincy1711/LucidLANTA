"""
Model zoo utilities for loading and working with pre-trained models.

This module provides utilities for loading popular pre-trained models and
extracting their activations for visualization purposes.
"""

from .vision_models import load_model, get_model_layers, extract_activations
from .vision_base import ModelWrapper

__all__ = [
    "load_model",
    "get_model_layers", 
    "extract_activations",
    "ModelWrapper"
]