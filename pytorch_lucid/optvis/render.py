"""
Rendering utilities for optimization-based feature visualization.

This module provides the core render_vis function and supporting utilities
for actually performing the optimization and rendering visualizations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from .objectives import Objective
from .param import ImageParam, FFTImageParam
from .transform import Transform, standard_transforms


def _extract_activations(model: nn.Module, input_tensor: torch.Tensor, 
                        layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Extract activations from specified layers of a model.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor to the model
        layer_names: Names of layers to extract activations from
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                # Handle multiple outputs
                activations[name] = output[0]
            else:
                activations[name] = output
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def _get_layer_names(model: nn.Module, layer_type: type = nn.Conv2d) -> List[str]:
    """Get names of all layers of a specific type."""
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            layer_names.append(name)
    return layer_names


def render_vis(model: nn.Module, 
               objective_f: Union[Objective, Callable[[Dict[str, torch.Tensor]], torch.Tensor]],
               param_f: Optional[nn.Module] = None,
               optimizer: Optional[optim.Optimizer] = None,
               transforms: Optional[Transform] = None,
               thresholds: Tuple[int, ...] = (512,),
               print_objectives: Optional[Callable] = None,
               verbose: bool = True,
               use_fixed_seed: bool = False,
               device: Optional[str] = None) -> List[np.ndarray]:
    """
    Flexible optimization-based feature visualization.
    
    This is the main function for creating feature visualizations through
    optimization. It can handle a wide variety of objectives, parameterizations,
    and transformations.
    
    Args:
        model: PyTorch model to visualize
        objective_f: Objective function to optimize
        param_f: Parameterization function/module (creates the image)
        optimizer: Optimizer to use for optimization
        transforms: Image transformations to apply during optimization
        thresholds: Number of optimization steps to run
        print_objectives: Function to print objective values
        verbose: Whether to show progress bar
        use_fixed_seed: Whether to use fixed random seed
        device: Device to run optimization on
        
    Returns:
        List of images at each threshold
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if use_fixed_seed:
        torch.manual_seed(0)
        np.random.seed(0)
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Default parameterization if not provided
    if param_f is None:
        param_f = ImageParam((1, 3, 224, 224), decorrelate=True, sigmoid=True)
    
    param_f.to(device)
    
    # Default optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(param_f.parameters(), lr=0.05)
    
    # Default transforms if not provided
    if transforms is None:
        transforms = standard_transforms()
    
    # Prepare objective function
    if not isinstance(objective_f, Objective):
        # If it's a regular function, wrap it
        def wrapped_objective(activations):
            return objective_f(activations)
        wrapped_objective.description = "Custom Objective"
        objective_f = wrapped_objective
    
    # Get layer names that the objective might need
    # This is a simplified approach - in practice, you'd parse the objective
    layer_names = _get_layer_names(model)
    
    # Storage for results
    results = []
    current_threshold_idx = 0
    
    # Optimization loop
    total_steps = max(thresholds) if thresholds else 512
    
    if verbose:
        pbar = tqdm(range(total_steps), desc="Optimizing")
    else:
        pbar = range(total_steps)
    
    for step in pbar:
        # Zero gradients
        optimizer.zero_grad()
        
        # Generate image
        img = param_f()
        
        # Apply transformations
        img_transformed = transforms(img)
        
        # Forward pass through model
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0]
                else:
                    activations[name] = output
            return hook
        
        # Register hooks
        for name in layer_names:
            try:
                module = dict(model.named_modules())[name]
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
            except KeyError:
                continue
        
        # Forward pass
        try:
            _ = model(img_transformed)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute objective
        try:
            loss = objective_f(activations)
            
            # Add image regularization if needed
            if hasattr(param_f, 'regularization'):
                loss = loss + param_f.regularization(img)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Print objective if requested
            if print_objectives is not None and step % 50 == 0:
                print_objectives(loss.item())
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Check if we've reached a threshold
            if current_threshold_idx < len(thresholds) and step >= thresholds[current_threshold_idx] - 1:
                # Store current image
                with torch.no_grad():
                    final_img = param_f()
                    # Convert to numpy and denormalize
                    img_np = final_img.detach().cpu().numpy()
                    if img_np.shape[1] == 3:  # RGB
                        img_np = img_np.transpose(0, 2, 3, 1)  # NHWC
                    img_np = np.clip(img_np[0], 0, 1)  # Remove batch dim and clip
                    results.append(img_np)
                
                current_threshold_idx += 1
                
        except Exception as e:
            print(f"Error during optimization step {step}: {e}")
            continue
    
    return results


def visualize_neuron(model: nn.Module, layer_name: str, channel_idx: int,
                    x: Optional[int] = None, y: Optional[int] = None,
                    **kwargs) -> List[np.ndarray]:
    """
    Convenience function to visualize a specific neuron.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        channel_idx: Channel index to visualize
        x: X coordinate (optional)
        y: Y coordinate (optional)
        **kwargs: Additional arguments for render_vis
        
    Returns:
        List of visualization images
    """
    from .objectives import neuron
    
    objective = neuron(layer_name, channel_idx, x, y)
    return render_vis(model, objective, **kwargs)


def visualize_channel(model: nn.Module, layer_name: str, channel_idx: int,
                     **kwargs) -> List[np.ndarray]:
    """
    Convenience function to visualize a specific channel.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        channel_idx: Channel index to visualize
        **kwargs: Additional arguments for render_vis
        
    Returns:
        List of visualization images
    """
    from .objectives import channel
    
    objective = channel(layer_name, channel_idx)
    return render_vis(model, objective, **kwargs)


def visualize_layer(model: nn.Module, layer_name: str, **kwargs) -> List[np.ndarray]:
    """
    Convenience function to visualize a specific layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        **kwargs: Additional arguments for render_vis
        
    Returns:
        List of visualization images
    """
    from .objectives import layer
    
    objective = layer(layer_name)
    return render_vis(model, objective, **kwargs)


def deepdream_visualization(model: nn.Module, layer_name: str, **kwargs) -> List[np.ndarray]:
    """
    Convenience function for DeepDream-style visualization.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        **kwargs: Additional arguments for render_vis
        
    Returns:
        List of visualization images
    """
    from .objectives import deepdream
    
    objective = deepdream(layer_name)
    return render_vis(model, objective, **kwargs)