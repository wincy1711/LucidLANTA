"""
Vision model utilities for PyTorch Lucid.

This module provides utilities for loading popular vision models and
extracting their activations for visualization purposes.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import torchvision.models as models
from .vision_base import ModelWrapper, create_model_wrapper


def load_model(model_name: str, pretrained: bool = True, 
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
               **kwargs) -> ModelWrapper:
    """
    Load a pre-trained vision model.
    
    Args:
        model_name: Name of the model (e.g., 'vgg16', 'resnet50', 'alexnet')
        pretrained: Whether to load pre-trained weights
        device: Device to load the model on
        **kwargs: Additional arguments for model loading
        
    Returns:
        ModelWrapper instance
        
    Raises:
        ValueError: If model name is not supported
    """
    model_name = model_name.lower()
    
    try:
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'vgg')
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'vgg')
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'resnet')
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'resnet')
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'resnet')
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'resnet')
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'resnet')
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained, **kwargs)
            wrapper = create_model_wrapper(model, 'standard')
        else:
            raise ValueError(f"Model '{model_name}' is not supported. "
                           f"Supported models: vgg16, vgg19, resnet18/34/50/101/152, "
                           f"alexnet, densenet121/169/201, inception_v3, googlenet, mobilenet_v2")
    
    # Move to device and set to eval mode
    wrapper.to(device)
    wrapper.eval()
    
    return wrapper


def get_model_layers(model_wrapper: ModelWrapper, layer_type: Optional[str] = None) -> List[str]:
    """
    Get layer names from a model wrapper.
    
    Args:
        model_wrapper: ModelWrapper instance
        layer_type: Type of layers to return ('conv', 'linear', or None for all)
        
    Returns:
        List of layer names
    """
    if layer_type is None:
        return model_wrapper.get_layer_names()
    elif layer_type.lower() == 'conv':
        return model_wrapper.get_conv_layers()
    elif layer_type.lower() == 'linear':
        return model_wrapper.get_linear_layers()
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def extract_activations(model_wrapper: ModelWrapper, input_tensor: torch.Tensor,
                       layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Extract activations from a model.
    
    Args:
        model_wrapper: ModelWrapper instance
        input_tensor: Input tensor to the model
        layer_names: Names of layers to extract activations from
                    If None, extracts from all available layers
                    
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    return model_wrapper.get_activations(input_tensor, layer_names)


def get_feature_layers(model_wrapper: ModelWrapper) -> List[str]:
    """
    Get layers that are typically good for feature visualization.
    
    Args:
        model_wrapper: ModelWrapper instance
        
    Returns:
        List of recommended layer names for visualization
    """
    if hasattr(model_wrapper, 'get_feature_layers'):
        return model_wrapper.get_feature_layers()
    else:
        # Default: return conv layers from later parts of the network
        conv_layers = model_wrapper.get_conv_layers()
        # Take the last half of conv layers
        return conv_layers[len(conv_layers)//2:]


def print_model_summary(model_wrapper: ModelWrapper, input_shape: Optional[Tuple[int, ...]] = None):
    """
    Print a summary of the model structure.
    
    Args:
        model_wrapper: ModelWrapper instance
        input_shape: Input shape to use for output shape calculation
    """
    print(f"Model: {model_wrapper.model.__class__.__name__}")
    print(f"Input size: {model_wrapper.input_size}")
    print(f"Total layers: {len(model_wrapper.get_layer_names())}")
    print(f"Convolutional layers: {len(model_wrapper.get_conv_layers())}")
    print(f"Linear layers: {len(model_wrapper.get_linear_layers())}")
    print("\nLayer Information:")
    print("-" * 80)
    
    for layer_name in model_wrapper.get_layer_names()[:20]:  # Show first 20 layers
        info = model_wrapper.get_layer_info(layer_name)
        layer_type = info['type']
        params = info['parameters']
        
        if input_shape:
            try:
                output_shape = model_wrapper.get_layer_output_shape(layer_name, input_shape)
                shape_str = f" → {output_shape}"
            except:
                shape_str = ""
        else:
            shape_str = ""
        
        print(f"{layer_name:30} {layer_type:15} {params:8,} params{shape_str}")
    
    if len(model_wrapper.get_layer_names()) > 20:
        print(f"... and {len(model_wrapper.get_layer_names()) - 20} more layers")


def create_random_input(batch_size: int = 1, 
                       input_size: Tuple[int, int, int] = (3, 224, 224),
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    """
    Create random input tensor for testing.
    
    Args:
        batch_size: Batch size
        input_size: Input size (channels, height, width)
        device: Device to create tensor on
        
    Returns:
        Random input tensor
    """
    shape = (batch_size,) + input_size
    return torch.randn(*shape, device=device)