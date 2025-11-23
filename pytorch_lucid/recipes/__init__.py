"""
Pre-built visualization recipes for PyTorch Lucid.

This module provides ready-to-use recipes for common visualization tasks,
making it easier to get started with neural network interpretability.
"""

from .basic_visualizations import visualize_neurons, visualize_channels, visualize_layers
from .deepdream import deepdream_visualization
from .style_transfer import style_transfer_visualization
from .feature_visualization import feature_visualization_pipeline

__all__ = [
    "visualize_neurons",
    "visualize_channels", 
    "visualize_layers",
    "deepdream_visualization",
    "style_transfer_visualization",
    "feature_visualization_pipeline"
]