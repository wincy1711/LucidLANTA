"""
DeepDream visualization recipes for PyTorch Lucid.

This module provides ready-to-use functions for creating DeepDream-style
visualizations, which maximize interesting features across multiple layers.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union
from ..modelzoo import ModelWrapper
from ..optvis import render_vis, deepdream_visualization
from ..optvis.objectives import deepdream
from ..optvis.param import image


def deepdream_multilayer(model_wrapper: ModelWrapper,
                        layer_names: List[str],
                        weights: Optional[List[float]] = None,
                        num_steps: int = 512,
                        image_size: tuple = (224, 224),
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        **kwargs) -> np.ndarray:
    """
    Create DeepDream visualization using multiple layers.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_names: List of layer names to maximize
        weights: Weights for each layer (default: equal weights)
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        DeepDream visualization image
    """
    if weights is None:
        weights = [1.0] * len(layer_names)
    
    if len(weights) != len(layer_names):
        raise ValueError("Number of weights must match number of layers")
    
    # Create combined objective
    total_objective = None
    for layer_name, weight in zip(layer_names, weights):
        layer_objective = deepdream(layer_name)
        weighted_objective = weight * layer_objective
        
        if total_objective is None:
            total_objective = weighted_objective
        else:
            total_objective = total_objective + weighted_objective
    
    # Perform optimization
    images = render_vis(
        model_wrapper.model,
        total_objective,
        thresholds=(num_steps,),
        device=device,
        **kwargs
    )
    
    return images[-1] if images else None


def deepdream_evolution(model_wrapper: ModelWrapper,
                       layer_name: str,
                       evolution_steps: List[int] = [64, 128, 256, 512],
                       image_size: tuple = (224, 224),
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                       **kwargs) -> Dict[int, np.ndarray]:
    """
    Create DeepDream visualization showing evolution over time.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to maximize
        evolution_steps: List of optimization steps to save
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Dictionary mapping step numbers to visualization images
    """
    # Create DeepDream objective
    objective = deepdream(layer_name)
    
    # Perform optimization with multiple thresholds
    images = render_vis(
        model_wrapper.model,
        objective,
        thresholds=evolution_steps,
        device=device,
        **kwargs
    )
    
    # Create result dictionary
    results = {}
    for i, step in enumerate(evolution_steps):
        if i < len(images):
            results[step] = images[i]
    
    return results


def deepdream_octaves(model_wrapper: ModelWrapper,
                     layer_name: str,
                     num_octaves: int = 4,
                     octave_scale: float = 1.4,
                     num_steps: int = 200,
                     image_size: tuple = (224, 224),
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                     **kwargs) -> np.ndarray:
    """
    Create DeepDream visualization using octave-based processing.
    
    This technique processes the image at multiple scales for more detailed results.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to maximize
        num_octaves: Number of octaves (scales) to process
        octave_scale: Scale factor between octaves
        num_steps: Number of optimization steps per octave
        image_size: Base image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        DeepDream visualization image
    """
    from ..misc.image_utils import resize_image
    
    # Start with base image size
    current_size = image_size
    base_image = torch.randn(1, 3, *image_size, device=device) * 0.1
    
    # Process each octave
    for octave in range(num_octaves):
        print(f"Processing octave {octave + 1}/{num_octaves} at size {current_size}")
        
        # Resize base image to current size
        if octave > 0:
            base_image = resize_image(base_image, current_size)
        
        # Create parameterization starting from base image
        param = optvis.param.image((1, 3, *current_size), decorrelate=True, sigmoid=True)
        param.param.data = base_image.squeeze(0).data
        
        # Create objective
        objective = deepdream(layer_name)
        
        # Optimize for this octave
        images = render_vis(
            model_wrapper.model,
            objective,
            param_f=param,
            thresholds=(num_steps,),
            device=device,
            **kwargs
        )
        
        if images:
            # Update base image for next octave
            base_image = torch.from_numpy(images[-1]).permute(2, 0, 1).unsqueeze(0)
        
        # Increase size for next octave
        current_size = (int(current_size[0] * octave_scale),
                       int(current_size[1] * octave_scale))
    
    # Return final result
    return images[-1] if images else None


def deepdream_guided(model_wrapper: ModelWrapper,
                    layer_name: str,
                    guide_image: np.ndarray,
                    num_steps: int = 512,
                    image_size: tuple = (224, 224),
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    **kwargs) -> np.ndarray:
    """
    Create guided DeepDream visualization that matches a guide image.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to maximize
        guide_image: Guide image to match patterns from
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Guided DeepDream visualization image
    """
    # Convert guide image to tensor
    guide_tensor = torch.from_numpy(guide_image).permute(2, 0, 1).unsqueeze(0)
    guide_tensor = guide_tensor.to(device)
    
    # Extract guide features
    with torch.no_grad():
        guide_activations = model_wrapper.get_activations(guide_tensor, [layer_name])
    
    # Create guided objective that maximizes correlation with guide features
    def guided_objective(activations):
        current_act = activations[layer_name]
        guide_act = guide_activations[layer_name]
        
        # Normalize activations
        current_act_norm = F.normalize(current_act.flatten(1), dim=1)
        guide_act_norm = F.normalize(guide_act.flatten(1), dim=1)
        
        # Maximize correlation
        correlation = (current_act_norm * guide_act_norm).sum()
        return correlation
    
    # Perform optimization
    images = render_vis(
        model_wrapper.model,
        guided_objective,
        thresholds=(num_steps,),
        device=device,
        **kwargs
    )
    
    return images[-1] if images else None


def deepdream_class_visualization(model_wrapper: ModelWrapper,
                                 target_class: int,
                                 num_steps: int = 512,
                                 image_size: tuple = (224, 224),
                                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                 **kwargs) -> np.ndarray:
    """
    Create DeepDream-style visualization for a specific class.
    
    Args:
        model_wrapper: Model wrapper instance
        target_class: Target class index to visualize
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Class visualization image
    """
    # Create class objective
    def class_objective(activations):
        # Get the model output (assuming it's the last activation)
        output_key = list(activations.keys())[-1]
        output = activations[output_key]
        
        # Maximize the target class
        return output[:, target_class].mean()
    
    # Perform optimization
    images = render_vis(
        model_wrapper.model,
        class_objective,
        thresholds=(num_steps,),
        device=device,
        **kwargs
    )
    
    return images[-1] if images else None


def create_deepdream_video_frames(model_wrapper: ModelWrapper,
                                 layer_name: str,
                                 base_image: np.ndarray,
                                 num_frames: int = 60,
                                 zoom_speed: float = 0.01,
                                 rotation_speed: float = 0.5,
                                 num_steps_per_frame: int = 50,
                                 image_size: tuple = (224, 224),
                                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                 **kwargs) -> List[np.ndarray]:
    """
    Create frames for a DeepDream zoom video.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to maximize
        base_image: Starting image
        num_frames: Number of frames to generate
        zoom_speed: Zoom speed per frame
        rotation_speed: Rotation speed per frame (degrees)
        num_steps_per_frame: Optimization steps per frame
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for render_vis
        
    Returns:
        List of video frames as numpy arrays
    """
    from ..misc.image_utils import resize_image, random_rotate
    
    frames = []
    current_image = torch.from_numpy(base_image).permute(2, 0, 1).unsqueeze(0)
    current_image = current_image.to(device)
    
    for frame in range(num_frames):
        print(f"Generating frame {frame + 1}/{num_frames}")
        
        # Apply zoom and rotation
        current_size = current_image.shape[2:]
        new_size = (int(current_size[0] * (1 + zoom_speed)),
                   int(current_size[1] * (1 + zoom_speed)))
        
        # Resize and rotate
        current_image = resize_image(current_image, new_size)
        current_image = random_rotate(current_image, angle_range=(rotation_speed, rotation_speed))
        
        # Center crop back to original size
        if current_image.shape[2:] != image_size:
            from ..misc.image_utils import center_crop
            current_image = center_crop(current_image, image_size)
        
        # Create parameterization from current image
        param = optvis.param.image((1, 3, *image_size), decorrelate=True, sigmoid=True)
        param.param.data = current_image.squeeze(0).data
        
        # Optimize for DeepDream
        objective = deepdream(layer_name)
        
        images = render_vis(
            model_wrapper.model,
            objective,
            param_f=param,
            thresholds=(num_steps_per_frame,),
            device=device,
            **kwargs
        )
        
        if images:
            current_image = torch.from_numpy(images[-1]).permute(2, 0, 1).unsqueeze(0)
            current_image = current_image.to(device)
            frames.append(images[-1])
        else:
            # If optimization failed, use current image
            frames.append(current_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    
    return frames