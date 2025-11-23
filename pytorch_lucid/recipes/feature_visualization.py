"""
Feature visualization pipeline for PyTorch Lucid.

This module provides a comprehensive pipeline for creating high-quality
feature visualizations with sensible defaults and best practices.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from ..modelzoo import ModelWrapper
from ..optvis import render_vis
from ..optvis.objectives import channel, neuron, layer, total_variation
from ..optvis.param import fft_image
from ..optvis.transform import standard_transforms


def feature_visualization_pipeline(model_wrapper: ModelWrapper,
                                  layer_name: str,
                                  feature_type: str = 'channel',
                                  feature_idx: int = 0,
                                  neuron_coords: Optional[Tuple[int, int]] = None,
                                  num_steps: int = 1024,
                                  image_size: tuple = (224, 224),
                                  use_fft_param: bool = True,
                                  decorrelate: bool = True,
                                  regularization_weight: float = 0.1,
                                  transform_config: Optional[Dict[str, Any]] = None,
                                  optimizer_config: Optional[Dict[str, Any]] = None,
                                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                  save_intermediates: bool = False,
                                  intermediate_steps: Optional[List[int]] = None,
                                  **kwargs) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Comprehensive feature visualization pipeline with sensible defaults.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        feature_type: Type of feature ('channel', 'neuron', or 'layer')
        feature_idx: Index of the feature to visualize
        neuron_coords: (x, y) coordinates for neuron visualization (required if feature_type='neuron')
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        use_fft_param: Whether to use FFT parameterization
        decorrelate: Whether to use decorrelated color space
        regularization_weight: Weight for total variation regularization
        transform_config: Configuration for transforms
        optimizer_config: Configuration for optimizer
        device: Device to run on
        save_intermediates: Whether to save intermediate results
        intermediate_steps: List of steps to save (default: logarithmic spacing)
        **kwargs: Additional arguments for render_vis
        
    Returns:
        Final visualization image, or dictionary of images if save_intermediates=True
    """
    print(f"Starting feature visualization pipeline for {feature_type} {feature_idx} in layer {layer_name}")
    
    # Create objective based on feature type
    if feature_type == 'channel':
        objective = channel(layer_name, feature_idx)
    elif feature_type == 'neuron':
        if neuron_coords is None:
            raise ValueError("neuron_coords must be provided for neuron visualization")
        objective = neuron(layer_name, feature_idx, neuron_coords[0], neuron_coords[1])
    elif feature_type == 'layer':
        objective = layer(layer_name)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # Add regularization
    if regularization_weight > 0:
        objective = objective - regularization_weight * total_variation()
    
    # Create parameterization
    if use_fft_param:
        param_f = fft_image((1, 3, *image_size))
    else:
        from ..optvis.param import image
        param_f = image((1, 3, *image_size), decorrelate=decorrelate, sigmoid=True)
    
    # Create transforms
    if transform_config is None:
        transforms = standard_transforms(
            jitter=16,
            scale=(0.9, 1.1),
            rotate=(-5, 5),
            pad=16
        )
    else:
        transforms = standard_transforms(**transform_config)
    
    # Create optimizer
    if optimizer_config is None:
        optimizer = torch.optim.Adam(param_f.parameters(), lr=0.05)
    else:
        opt_class = getattr(torch.optim, optimizer_config.get('class', 'Adam'))
        opt_params = {k: v for k, v in optimizer_config.items() if k != 'class'}
        optimizer = opt_class(param_f.parameters(), **opt_params)
    
    # Determine intermediate steps
    if save_intermediates:
        if intermediate_steps is None:
            # Use logarithmic spacing
            intermediate_steps = [2**i for i in range(2, int(np.log2(num_steps)) + 1)]
        thresholds = intermediate_steps
    else:
        thresholds = (num_steps,)
    
    # Perform optimization
    images = render_vis(
        model_wrapper.model,
        objective,
        param_f=param_f,
        optimizer=optimizer,
        transforms=transforms,
        thresholds=thresholds,
        device=device,
        **kwargs
    )
    
    if save_intermediates:
        # Return dictionary of images
        results = {}
        for i, step in enumerate(intermediate_steps):
            if i < len(images):
                results[step] = images[i]
        return results
    else:
        # Return final image
        return images[-1] if images else None


def batch_visualize_features(model_wrapper: ModelWrapper,
                            layer_name: str,
                            feature_indices: List[int],
                            feature_type: str = 'channel',
                            output_dir: str = 'visualizations',
                            num_steps: int = 512,
                            image_size: tuple = (224, 224),
                            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                            **kwargs) -> Dict[int, np.ndarray]:
    """
    Batch visualize multiple features from the same layer.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        feature_indices: List of feature indices to visualize
        feature_type: Type of feature ('channel', 'neuron', or 'layer')
        output_dir: Directory to save results
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for feature_visualization_pipeline
        
    Returns:
        Dictionary mapping feature indices to visualization images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for feature_idx in feature_indices:
        print(f"\nVisualizing {feature_type} {feature_idx} in layer {layer_name}")
        
        try:
            # Create visualization
            image = feature_visualization_pipeline(
                model_wrapper,
                layer_name,
                feature_type=feature_type,
                feature_idx=feature_idx,
                num_steps=num_steps,
                image_size=image_size,
                device=device,
                **kwargs
            )
            
            if image is not None:
                results[feature_idx] = image
                
                # Save individual image
                from ..misc.io import save_image
                save_path = os.path.join(output_dir, f'{layer_name}_{feature_type}_{feature_idx}.png')
                save_image(
                    torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0),
                    save_path,
                    denormalize=False
                )
                print(f"Saved visualization to {save_path}")
            
        except Exception as e:
            print(f"Error visualizing {feature_type} {feature_idx}: {e}")
            continue
    
    return results


def visualize_layer_progression(model_wrapper: ModelWrapper,
                               layer_names: List[str],
                               feature_idx: int = 0,
                               feature_type: str = 'channel',
                               num_steps: int = 512,
                               image_size: tuple = (224, 224),
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                               **kwargs) -> Dict[str, np.ndarray]:
    """
    Visualize how feature representations change across layers.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_names: List of layer names to visualize
        feature_idx: Index of the feature to visualize
        feature_type: Type of feature ('channel', 'neuron', or 'layer')
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for feature_visualization_pipeline
        
    Returns:
        Dictionary mapping layer names to visualization images
    """
    results = {}
    
    for layer_name in layer_names:
        print(f"\nVisualizing {feature_type} {feature_idx} in layer {layer_name}")
        
        try:
            image = feature_visualization_pipeline(
                model_wrapper,
                layer_name,
                feature_type=feature_type,
                feature_idx=feature_idx,
                num_steps=num_steps,
                image_size=image_size,
                device=device,
                **kwargs
            )
            
            if image is not None:
                results[layer_name] = image
                
        except Exception as e:
            print(f"Error visualizing layer {layer_name}: {e}")
            continue
    
    return results


def create_comparison_grid(model_wrapper: ModelWrapper,
                          comparisons: Dict[str, Dict[str, Any]],
                          grid_size: tuple = (3, 3),
                          num_steps: int = 256,
                          image_size: tuple = (128, 128),
                          device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                          **kwargs) -> np.ndarray:
    """
    Create a comparison grid of different visualizations.
    
    Args:
        model_wrapper: Model wrapper instance
        comparisons: Dictionary of comparison configurations
        grid_size: Grid layout (rows, cols)
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for feature_visualization_pipeline
        
    Returns:
        Comparison grid as numpy array
    """
    from ..misc.io import create_image_grid
    
    images = []
    labels = []
    
    for name, config in comparisons.items():
        if len(images) >= grid_size[0] * grid_size[1]:
            break
            
        try:
            image = feature_visualization_pipeline(
                model_wrapper,
                config['layer_name'],
                feature_type=config.get('feature_type', 'channel'),
                feature_idx=config.get('feature_idx', 0),
                num_steps=num_steps,
                image_size=image_size,
                device=device,
                **kwargs
            )
            
            if image is not None:
                images.append(image)
                labels.append(name)
                
        except Exception as e:
            print(f"Error creating visualization for {name}: {e}")
            continue
    
    # Create grid
    if images:
        grid = create_image_grid(images, grid_size)
        return grid.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        return np.zeros((image_size[0] * grid_size[0], image_size[1] * grid_size[1], 3))


def analyze_feature_diversity(model_wrapper: ModelWrapper,
                             layer_name: str,
                             num_features: int = 16,
                             num_steps: int = 256,
                             image_size: tuple = (128, 128),
                             device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                             **kwargs) -> Dict[str, Any]:
    """
    Analyze the diversity of features in a layer.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to analyze
        num_features: Number of features to visualize
        num_steps: Number of optimization steps
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for feature_visualization_pipeline
        
    Returns:
        Dictionary containing visualizations and analysis results
    """
    # Get layer information
    try:
        layer_info = model_wrapper.get_layer_info(layer_name)
        max_features = layer_info.get('out_channels', num_features)
        feature_indices = np.linspace(0, max_features - 1, num_features, dtype=int)
    except:
        feature_indices = list(range(num_features))
    
    # Visualize features
    visualizations = batch_visualize_features(
        model_wrapper,
        layer_name,
        feature_indices,
        num_steps=num_steps,
        image_size=image_size,
        device=device,
        **kwargs
    )
    
    # Create diversity grid
    from ..misc.io import create_image_grid
    grid_size = (int(np.sqrt(num_features)), int(np.sqrt(num_features)))
    
    images = [visualizations[idx] for idx in feature_indices if idx in visualizations]
    if images:
        diversity_grid = create_image_grid(images, grid_size)
        diversity_grid = diversity_grid.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        diversity_grid = None
    
    return {
        'layer_name': layer_name,
        'visualizations': visualizations,
        'diversity_grid': diversity_grid,
        'num_features': len(visualizations),
        'feature_indices': [idx for idx in feature_indices if idx in visualizations]
    }


def optimize_visualization_quality(model_wrapper: ModelWrapper,
                                  layer_name: str,
                                  feature_idx: int,
                                  feature_type: str = 'channel',
                                  quality_levels: List[int] = [256, 512, 1024],
                                  image_size: tuple = (224, 224),
                                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                                  **kwargs) -> Dict[int, np.ndarray]:
    """
    Optimize visualization quality by testing different step counts.
    
    Args:
        model_wrapper: Model wrapper instance
        layer_name: Name of the layer to visualize
        feature_idx: Index of the feature to visualize
        feature_type: Type of feature ('channel', 'neuron', or 'layer')
        quality_levels: List of step counts to test
        image_size: Output image size (height, width)
        device: Device to run on
        **kwargs: Additional arguments for feature_visualization_pipeline
        
    Returns:
        Dictionary mapping step counts to visualization images
    """
    results = {}
    
    for num_steps in quality_levels:
        print(f"\nTesting quality with {num_steps} steps")
        
        try:
            image = feature_visualization_pipeline(
                model_wrapper,
                layer_name,
                feature_type=feature_type,
                feature_idx=feature_idx,
                num_steps=num_steps,
                image_size=image_size,
                device=device,
                **kwargs
            )
            
            if image is not None:
                results[num_steps] = image
                
        except Exception as e:
            print(f"Error with {num_steps} steps: {e}")
            continue
    
    return results