"""
Style transfer utilities for neural network visualization.

This module provides functionality for style transfer objectives, which can be
used to create artistic visualizations or for studying style representations
in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix of input features.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Gram matrix of shape (batch_size, channels, channels)
        """
        batch_size, channels, height, width = x.size()
        
        # Reshape to (batch_size, channels, height * width)
        features = x.view(batch_size, channels, height * width)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (channels * height * width)
        
        return gram


class StyleLoss(nn.Module):
    """
    Style loss based on Gram matrix matching.
    
    This loss measures the difference in style between two images by comparing
    the Gram matrices of their feature representations at multiple layers.
    """
    
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        """
        Initialize style loss.
        
        Args:
            layer_weights: Dictionary mapping layer names to weights
        """
        super().__init__()
        self.gram_matrix = GramMatrix()
        
        # Default layer weights (similar to original style transfer paper)
        if layer_weights is None:
            self.layer_weights = {
                'conv1_1': 1.0,
                'conv2_1': 0.75,
                'conv3_1': 0.2,
                'conv4_1': 0.2,
                'conv5_1': 0.2
            }
        else:
            self.layer_weights = layer_weights
        
        self.target_grams = {}
        self.mse_loss = nn.MSELoss()
    
    def set_target(self, activations: Dict[str, torch.Tensor]):
        """
        Set target style from activations.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
        """
        self.target_grams = {}
        for layer_name, activation in activations.items():
            if layer_name in self.layer_weights:
                with torch.no_grad():
                    gram = self.gram_matrix(activation)
                    self.target_grams[layer_name] = gram.clone()
    
    def forward(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute style loss.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            
        Returns:
            Style loss value
        """
        total_loss = 0.0
        
        for layer_name, activation in activations.items():
            if layer_name in self.layer_weights and layer_name in self.target_grams:
                # Compute Gram matrix for current activations
                current_gram = self.gram_matrix(activation)
                
                # Compute loss for this layer
                layer_loss = self.mse_loss(current_gram, self.target_grams[layer_name])
                
                # Weight by layer importance
                weighted_loss = self.layer_weights[layer_name] * layer_loss
                total_loss += weighted_loss
        
        return total_loss


class ContentLoss(nn.Module):
    """
    Content loss for preserving content during style transfer.
    
    This loss measures the difference in content by comparing feature
    representations directly (without Gram matrix transformation).
    """
    
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        """
        Initialize content loss.
        
        Args:
            layer_weights: Dictionary mapping layer names to weights
        """
        super().__init__()
        
        if layer_weights is None:
            self.layer_weights = {'conv4_2': 1.0}  # Default from style transfer paper
        else:
            self.layer_weights = layer_weights
        
        self.target_features = {}
        self.mse_loss = nn.MSELoss()
    
    def set_target(self, activations: Dict[str, torch.Tensor]):
        """
        Set target content from activations.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
        """
        self.target_features = {}
        for layer_name, activation in activations.items():
            if layer_name in self.layer_weights:
                with torch.no_grad():
                    self.target_features[layer_name] = activation.clone()
    
    def forward(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute content loss.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            
        Returns:
            Content loss value
        """
        total_loss = 0.0
        
        for layer_name, activation in activations.items():
            if layer_name in self.layer_weights and layer_name in self.target_features:
                # Compute loss for this layer
                layer_loss = self.mse_loss(activation, self.target_features[layer_name])
                
                # Weight by layer importance
                weighted_loss = self.layer_weights[layer_name] * layer_loss
                total_loss += weighted_loss
        
        return total_loss


class StyleTransferObjective:
    """
    Combined style and content objective for style transfer.
    
    This objective balances style similarity with content preservation.
    """
    
    def __init__(self, style_weight: float = 1e6, content_weight: float = 1.0,
                 style_layers: Optional[Dict[str, float]] = None,
                 content_layers: Optional[Dict[str, float]] = None):
        """
        Initialize style transfer objective.
        
        Args:
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            style_layers: Dictionary mapping layer names to style weights
            content_layers: Dictionary mapping layer names to content weights
        """
        self.style_loss = StyleLoss(style_layers)
        self.content_loss = ContentLoss(content_layers)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.description = f"StyleTransfer(style_w={style_weight}, content_w={content_weight})"
    
    def set_style_target(self, style_activations: Dict[str, torch.Tensor]):
        """Set the target style from style image activations."""
        self.style_loss.set_target(style_activations)
    
    def set_content_target(self, content_activations: Dict[str, torch.Tensor]):
        """Set the target content from content image activations."""
        self.content_loss.set_target(content_activations)
    
    def __call__(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute combined style and content loss.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            
        Returns:
            Combined loss value
        """
        style_loss_val = self.style_loss(activations)
        content_loss_val = self.content_loss(activations)
        
        total_loss = (self.style_weight * style_loss_val + 
                     self.content_weight * content_loss_val)
        
        return total_loss


def create_style_objective(style_image: torch.Tensor, content_image: torch.Tensor,
                          model: nn.Module, layer_names: List[str],
                          style_weight: float = 1e6, content_weight: float = 1.0,
                          device: str = 'cuda') -> StyleTransferObjective:
    """
    Create a style transfer objective from images.
    
    Args:
        style_image: Style reference image
        content_image: Content reference image
        model: Model to use for feature extraction
        layer_names: Names of layers to use
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        device: Device to run on
        
    Returns:
        StyleTransferObjective configured with the given images
    """
    # Create objective
    objective = StyleTransferObjective(style_weight, content_weight)
    
    # Extract activations for style image
    with torch.no_grad():
        style_activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    style_activations[name] = output[0]
                else:
                    style_activations[name] = output
            return hook
        
        # Register hooks
        for name in layer_names:
            try:
                module = dict(model.named_modules())[name]
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
            except KeyError:
                continue
        
        # Forward pass for style
        _ = model(style_image.to(device))
        objective.set_style_target(style_activations)
        
        # Clear activations
        style_activations.clear()
        
        # Forward pass for content
        _ = model(content_image.to(device))
        objective.set_content_target(style_activations)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return objective