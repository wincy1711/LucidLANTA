"""
Basic visualization recipes for PyTorch Lucid.

This module provides simple, ready-to-use functions for common visualization
tasks like visualizing neurons, channels, and layers.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union
from ..modelzoo import ModelWrapper
from ..optvis import render_vis, visualize_neuron, visualize_channel, visualize_layer
from ..optvis.objectives import channel, neuron, layer
from ..optvis.param import image
from ..optvis.transform import standard_transforms


def visualize_neurons(model_wrapper: ModelWrapper,
                     layer_name: str,
                     neuron_indices: List[int],
                     num_steps: int = 512,
                     image_size: tuple = (224, 224),
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                     **kwargs) -> Dict[int, np.ndarray]:
    """
    Visualize multiple neurons from a specific layer.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        neuron_indices: List of neuron indices to visualize
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping neuron indices to visualization images
    """
    results = {}
    
    for neuron_idx in neuron_indices:
        print(f"Visualizing neuron {neuron_idx} in layer {layer_name}...")
        
        try:
            # Create visualization
            images = visualize_neuron(
                model_wrapper.model,
                layer_name,
                neuron_idx,
                thresholds=(num_steps,),
                device=device,
                **kwargs
            )
            
            if images:
                results[neuron_idx] = images[-1]  # Get final image
                
        except Exception as e:
            print(f"Error visualizing neuron {neuron_idx}: {e}")
            continue
    
    return results


def visualize_channels(model_wrapper: ModelWrapper,
                      layer_name: str,
                      channel_indices: List[int],
                      num_steps: int = 512,
                      image_size: tuple = (224, 224),
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                      **kwargs) -> Dict[int, np.ndarray]:
    """
    Visualize multiple channels from a specific layer.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        channel_indices: List of channel indices to visualize
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping channel indices to visualization images
    """
    results = {}
    
    for channel_idx in channel_indices:
        print(f"Visualizing channel {channel_idx} in layer {layer_name}...")
        
        try:
            # Create visualization
            images = visualize_channel(
                model_wrapper.model,
                layer_name,
                channel_idx,
                thresholds=(num_steps,),
                device=device,
                **kwargs
            )
            
            if images:
                results[channel_idx] = images[-1]  # Get final image
                
        except Exception as e:
            print(f"Error visualizing channel {channel_idx}: {e}")
            continue
    
    return results


def visualize_layers(model_wrapper: ModelWrapper,
                    layer_names: List[str],
                    num_steps: int = 512,
                    image_size: tuple = (224, 224),
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    **kwargs) -> Dict[str, np.ndarray]:
    """
    Visualize multiple layers in the model.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_names: List of layer names to visualize
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping layer names to visualization images
    """
    results = {}
    
    for layer_name in layer_names:
        print(f"Visualizing layer {layer_name}...")
        
        try:
            # Create visualization
            images = visualize_layer(
                model_wrapper.model,
                layer_name,
                thresholds=(num_steps,),
                device=device,
                **kwargs
            )
            
            if images:
                results[layer_name] = images[-1]  # Get final image
                
        except Exception as e:
            print(f"Error visualizing layer {layer_name}: {e}")
            continue
    
    return results


def visualize_layer_evolution(model_wrapper: ModelWrapper,
                             layer_name: str,
                             channel_idx: int,
                             num_steps: int = 512,
                             save_intervals: List[int] = [64, 128, 256, 512],
                             image_size: tuple = (224, 224),
                             device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                             **kwargs) -> Dict[int, np.ndarray]:
    """
    Visualize the evolution of a channel during optimization.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        channel_idx: Channel index to visualize
        num_steps: Total number of optimization steps
        save_intervals: List of steps to save intermediate results
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping step numbers to visualization images
    """
    print(f"Visualizing evolution of channel {channel_idx} in layer {layer_name}...")
    
    try:
        # Create visualization with multiple thresholds
        images = visualize_channel(
            model_wrapper.model,
            layer_name,
            channel_idx,
            thresholds=save_intervals,
            device=device,
            **kwargs
        )
        
        # Create result dictionary
        results = {}
        for i, step in enumerate(save_intervals):
            if i < len(images):
                results[step] = images[i]
        
        return results
        
    except Exception as e:
        print(f"Error visualizing layer evolution: {e}")
        return {}


def create_feature_visualization_grid(model_wrapper: ModelWrapper,
                                     layer_name: str,
                                     num_channels: int = 9,
                                     grid_size: tuple = (3, 3),
                                     num_steps: int = 512,
                                     image_size: tuple = (224, 224),
                                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                     **kwargs) -> np.ndarray:
    """
    Create a grid visualization of multiple channels from a layer.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        num_channels: Number of channels to visualize
        grid_size: Grid layout (rows, cols)
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Grid image as numpy array
    """
    from ..misc.io import create_image_grid
    
    # Get layer info to determine available channels
    try:
        layer_info = model_wrapper.get_layer_info(layer_name)
        if 'out_channels' in layer_info:
            max_channels = layer_info['out_channels']
            channel_indices = np.linspace(0, max_channels - 1, num_channels, dtype=int)
        else:
            channel_indices = list(range(num_channels))
    except:
        channel_indices = list(range(num_channels))
    
    # Visualize channels
    channel_images = []
    for channel_idx in channel_indices:
        try:
            images = visualize_channel(
                model_wrapper.model,
                layer_name,
                channel_idx,
                thresholds=(num_steps,),
                device=device,
                **kwargs
            )
            
            if images:
                channel_images.append(images[-1])
                
        except Exception as e:
            print(f"Error visualizing channel {channel_idx}: {e}")
            # Add blank image for failed visualizations
            channel_images.append(np.zeros((image_size[0], image_size[1], 3)))
    
    # Create grid
    if channel_images:
        grid = create_image_grid(channel_images, grid_size)
        return grid.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
    else:
        return np.zeros((image_size[0] * grid_size[0], image_size[1] * grid_size[1], 3))


def compare_layer_representations(model_wrapper: ModelWrapper,
                                 layer_names: List[str],
                                 num_channels: int = 4,
                                 num_steps: int = 512,
                                 image_size: tuple = (224, 224),
                                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                 **kwargs) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Compare feature representations across different layers.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_names: List of layer names to compare
        num_channels: Number of channels to visualize per layer
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Nested dictionary: {layer_name: {channel_idx: image}}
    """
    results = {}
    
    for layer_name in layer_names:
        print(f"Visualizing layer {layer_name}...")
        
        # Get layer info
        try:
            layer_info = model_wrapper.get_layer_info(layer_name)
            if 'out_channels' in layer_info:
                max_channels = layer_info['out_channels']
                channel_indices = np.linspace(0, max_channels - 1, num_channels, dtype=int)
            else:
                channel_indices = list(range(num_channels))
        except:
            channel_indices = list(range(num_channels))
        
        # Visualize channels for this layer
        layer_results = {}
        for channel_idx in channel_indices:
            try:
                images = visualize_channel(
                    model_wrapper.model,
                    layer_name,
                    channel_idx,
                    thresholds=(num_steps,),
                    device=device,
                    **kwargs
                )
                
                if images:
                    layer_results[channel_idx] = images[-1]
                    
            except Exception as e:
                print(f"Error visualizing channel {channel_idx} in layer {layer_name}: {e}")
                continue
        
        results[layer_name] = layer_results
    
    return results


def visualize_model_architecture(model_wrapper: ModelWrapper,
                                num_visualizations: int = 3,
                                num_steps: int = 256,
                                image_size: tuple = (224, 224),
                                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                **kwargs) -> Dict[str, np.ndarray]:
    """
    Create visualizations that represent the model's architecture.
    
    Args:
        model_wrapper: Model wrapper instance
        num_visualizations: Number of visualizations to create
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping layer names to representative visualizations
    """
    # Get feature layers
    feature_layers = model_wrapper.get_feature_layers()
    
    # Select representative layers
    if len(feature_layers) <= num_visualizations:
        selected_layers = feature_layers
    else:
        # Select evenly spaced layers
        indices = np.linspace(0, len(feature_layers) - 1, num_visualizations, dtype=int)
        selected_layers = [feature_layers[i] for i in indices]
    
    # Visualize selected layers
    results = {}
    for layer_name in selected_layers:
        print(f"Creating architectural visualization for layer {layer_name}...")
        
        try:
            # Use layer objective for architectural representation
            from ..optvis.objectives import layer
            
            images = render_vis(
                model_wrapper.model,
                layer(layer_name),
                thresholds=(num_steps,),
                device=device,
                **kwargs
            )
            
            if images:
                results[layer_name] = images[-1]
                
        except Exception as e:
            print(f"Error visualizing layer {layer_name}: {e}")
            continue
    
    return results