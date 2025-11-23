"""
Base classes for model handling in PyTorch Lucid.

This module provides base classes and utilities for working with different
neural network architectures in a unified way.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    This class provides a unified interface for working with different
    model architectures, making it easier to extract activations and
    perform visualizations.
    """
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        """
        Initialize model wrapper.
        
        Args:
            model: PyTorch model
            input_size: Expected input size (channels, height, width)
        """
        self.model = model
        self.input_size = input_size
        self._layer_names = None
        self._layer_info = None
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get names of all layers that can be visualized."""
        pass
    
    @abstractmethod
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get information about a specific layer."""
        pass
    
    def get_activations(self, input_tensor: torch.Tensor, 
                       layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract activations from specified layers.
        
        Args:
            input_tensor: Input tensor to the model
            layer_names: Names of layers to extract activations from
                        If None, extracts from all available layers
                        
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if layer_names is None:
            layer_names = self.get_layer_names()
        
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    # Handle multiple outputs - take the first one
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks
        for name in layer_names:
            try:
                module = dict(self.model.named_modules())[name]
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
            except KeyError:
                continue
        
        # Forward pass
        with torch.no_grad():
            try:
                _ = self.model(input_tensor)
            except Exception as e:
                print(f"Error during forward pass: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.model.to(device)
        return self


class StandardModelWrapper(ModelWrapper):
    """
    Standard wrapper for common PyTorch models.
    
    This wrapper works with models that use standard PyTorch layer naming
    conventions and have a straightforward architecture.
    """
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__(model, input_size)
        self._discover_layers()
    
    def _discover_layers(self):
        """Discover all layers in the model."""
        self._layer_names = []
        self._layer_info = {}
        
        for name, module in self.model.named_modules():
            # Skip the root module and containers
            if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            
            # Only include layers that have parameters or are activation layers
            if (len(list(module.children())) == 0 or 
                isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.LeakyReLU,
                                  nn.BatchNorm2d, nn.Dropout, nn.MaxPool2d,
                                  nn.AdaptiveAvgPool2d, nn.AvgPool2d))):
                
                self._layer_names.append(name)
                
                # Get layer information
                info = {
                    'type': type(module).__name__,
                    'class': module.__class__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'trainable': any(p.requires_grad for p in module.parameters())
                }
                
                # Add specific information based on layer type
                if isinstance(module, nn.Conv2d):
                    info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                        'dilation': module.dilation,
                        'groups': module.groups
                    })
                elif isinstance(module, nn.Linear):
                    info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    info.update({
                        'num_features': module.num_features,
                        'eps': module.eps,
                        'momentum': module.momentum
                    })
                
                self._layer_info[name] = info
    
    def get_layer_names(self) -> List[str]:
        """Get names of all layers."""
        return self._layer_names
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get information about a specific layer."""
        if layer_name not in self._layer_info:
            raise KeyError(f"Layer '{layer_name}' not found")
        return self._layer_info[layer_name]
    
    def get_conv_layers(self) -> List[str]:
        """Get names of all convolutional layers."""
        return [name for name, info in self._layer_info.items() 
                if info['type'] == 'Conv2d']
    
    def get_linear_layers(self) -> List[str]:
        """Get names of all linear layers."""
        return [name for name, info in self._layer_info.items() 
                if info['type'] == 'Linear']
    
    def get_layer_output_shape(self, layer_name: str, input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        """
        Get the output shape of a layer.
        
        Args:
            layer_name: Name of the layer
            input_shape: Input shape to use (defaults to model's input size)
            
        Returns:
            Output shape tuple
        """
        if input_shape is None:
            input_shape = (1,) + self.input_size  # Add batch dimension
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Hook to capture output
        output_shape = {}
        
        def get_output_shape(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output_shape[name] = tuple(output[0].shape)
                else:
                    output_shape[name] = tuple(output.shape)
            return hook
        
        # Register hook
        module = dict(self.model.named_modules())[layer_name]
        hook = module.register_forward_hook(get_output_shape(layer_name))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Remove hook
        hook.remove()
        
        return output_shape[layer_name]


class VGGWrapper(StandardModelWrapper):
    """
    Specialized wrapper for VGG models.
    
    Provides additional utilities specific to VGG architectures.
    """
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__(model, input_size)
    
    def get_feature_layers(self) -> List[str]:
        """Get layers commonly used for feature visualization."""
        # VGG layers that are typically useful for visualization
        feature_layers = []
        for name in self.get_conv_layers():
            # Include conv layers before pooling
            if 'conv' in name and not any(x in name for x in ['_2', '_4', '_6']):
                feature_layers.append(name)
        return feature_layers


class ResNetWrapper(StandardModelWrapper):
    """
    Specialized wrapper for ResNet models.
    
    Handles the special architecture of ResNet models.
    """
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__(model, input_size)
    
    def get_feature_layers(self) -> List[str]:
        """Get layers commonly used for feature visualization."""
        # ResNet layers that are typically useful for visualization
        feature_layers = []
        for name in self.get_conv_layers():
            # Focus on later layers for better visualization
            if any(x in name for x in ['layer2', 'layer3', 'layer4']):
                feature_layers.append(name)
        return feature_layers


def create_model_wrapper(model: nn.Module, model_type: str = 'standard',
                        input_size: Tuple[int, int, int] = (3, 224, 224)) -> ModelWrapper:
    """
    Create appropriate model wrapper for a given model.
    
    Args:
        model: PyTorch model
        model_type: Type of model ('standard', 'vgg', 'resnet', etc.)
        input_size: Expected input size
        
    Returns:
        Appropriate model wrapper
    """
    if model_type.lower() == 'vgg':
        return VGGWrapper(model, input_size)
    elif model_type.lower() == 'resnet':
        return ResNetWrapper(model, input_size)
    else:
        return StandardModelWrapper(model, input_size)