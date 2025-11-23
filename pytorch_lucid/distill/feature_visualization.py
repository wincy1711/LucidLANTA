"""
Feature Visualization implementations from the Distill paper.

This module implements the techniques described in:
"Feature Visualization" by Olah, Mordvintsev, and Schubert (2017)
https://distill.pub/2017/feature-visualization/

Key techniques:
- Parametric optimization for feature visualization
- Preconditioning and transformation robustness
- Regularization techniques (frequency and TV)
- Diversity and interpolation methods
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from ..optvis import render_vis, objectives
from ..optvis.param import fft_image, image
from ..optvis.transform import standard_transforms
from ..misc.io import save_image, create_image_grid


class FeatureVisualization:
    """
    Implementation of feature visualization techniques from the Distill paper.
    
    This class provides methods for creating high-quality feature visualizations
    using the techniques described in the paper, including preconditioning,
    transformation robustness, and regularization.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize feature visualization.
        
        Args:
            model: PyTorch model to visualize
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def visualize_neuron(self, layer_name: str, neuron_idx: int,
                        x: Optional[int] = None, y: Optional[int] = None,
                        num_steps: int = 1024, image_size: Tuple[int, int] = (224, 224),
                        use_fft: bool = True, **kwargs) -> np.ndarray:
        """
        Visualize a specific neuron.
        
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron to visualize
            x: X coordinate for spatial neuron (optional)
            y: Y coordinate for spatial neuron (optional)
            num_steps: Number of optimization steps
            image_size: Output image size
            use_fft: Whether to use FFT parameterization
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Visualization image as numpy array
        """
        obj = objectives.neuron(layer_name, neuron_idx, x, y)
        return self._visualize_objective(obj, num_steps, image_size, use_fft, **kwargs)
    
    def visualize_channel(self, layer_name: str, channel_idx: int,
                         num_steps: int = 1024, image_size: Tuple[int, int] = (224, 224),
                         use_fft: bool = True, **kwargs) -> np.ndarray:
        """
        Visualize a specific channel.
        
        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel to visualize
            num_steps: Number of optimization steps
            image_size: Output image size
            use_fft: Whether to use FFT parameterization
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Visualization image as numpy array
        """
        obj = objectives.channel(layer_name, channel_idx)
        return self._visualize_objective(obj, num_steps, image_size, use_fft, **kwargs)
    
    def visualize_layer(self, layer_name: str,
                       num_steps: int = 1024, image_size: Tuple[int, int] = (224, 224),
                       use_fft: bool = True, **kwargs) -> np.ndarray:
        """
        Visualize an entire layer.
        
        Args:
            layer_name: Name of the layer
            num_steps: Number of optimization steps
            image_size: Output image size
            use_fft: Whether to use FFT parameterization
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Visualization image as numpy array
        """
        obj = objectives.layer(layer_name)
        return self._visualize_objective(obj, num_steps, image_size, use_fft, **kwargs)
    
    def visualize_deepdream(self, layer_name: str,
                           num_steps: int = 512, image_size: Tuple[int, int] = (224, 224),
                           use_fft: bool = True, **kwargs) -> np.ndarray:
        """
        Create DeepDream-style visualization.
        
        Args:
            layer_name: Name of the layer
            num_steps: Number of optimization steps
            image_size: Output image size
            use_fft: Whether to use FFT parameterization
            **kwargs: Additional arguments for render_vis
            
        Returns:
            DeepDream visualization image
        """
        obj = objectives.deepdream(layer_name)
        return self._visualize_objective(obj, num_steps, image_size, use_fft, **kwargs)
    
    def _visualize_objective(self, objective, num_steps: int, image_size: Tuple[int, int],
                           use_fft: bool, **kwargs) -> np.ndarray:
        """
        Internal method to visualize an objective.
        
        Args:
            objective: Objective function to visualize
            num_steps: Number of optimization steps
            image_size: Output image size
            use_fft: Whether to use FFT parameterization
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Visualization image
        """
        # Create parameterization
        if use_fft:
            param_f = fft_image((1, 3, *image_size))
        else:
            param_f = image((1, 3, *image_size), decorrelate=True, sigmoid=True)
        
        # Create transforms with robustness (from the paper)
        transforms = standard_transforms(
            jitter=16,
            scale=(0.9, 1.1),
            rotate=(-5, 5),
            pad=16
        )
        
        # Add regularization (from the paper)
        regularized_obj = objective - 0.1 * objectives.total_variation()
        
        # Render visualization
        images = render_vis(
            self.model,
            regularized_obj,
            param_f=param_f,
            transforms=transforms,
            thresholds=(num_steps,),
            device=self.device,
            **kwargs
        )
        
        return images[-1] if images else None
    
    def visualize_neuron_evolution(self, layer_name: str, neuron_idx: int,
                                  evolution_steps: List[int] = [64, 128, 256, 512, 1024],
                                  image_size: Tuple[int, int] = (224, 224),
                                  **kwargs) -> Dict[int, np.ndarray]:
        """
        Visualize the evolution of a neuron during optimization.
        
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron
            evolution_steps: List of optimization steps to save
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Dictionary mapping step numbers to images
        """
        obj = objectives.neuron(layer_name, neuron_idx)
        
        # Create parameterization
        param_f = fft_image((1, 3, *image_size))
        transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
        regularized_obj = obj - 0.1 * objectives.total_variation()
        
        # Render with multiple thresholds
        images = render_vis(
            self.model,
            regularized_obj,
            param_f=param_f,
            transforms=transforms,
            thresholds=evolution_steps,
            device=self.device,
            **kwargs
        )
        
        # Create evolution dictionary
        evolution = {}
        for i, step in enumerate(evolution_steps):
            if i < len(images):
                evolution[step] = images[i]
        
        return evolution
    
    def create_diversity_visualization(self, layer_name: str, channel_idx: int,
                                      num_diverse: int = 6,
                                      num_steps: int = 512,
                                      image_size: Tuple[int, int] = (128, 128),
                                      **kwargs) -> List[np.ndarray]:
        """
        Create diverse visualizations of the same feature (from the paper).
        
        This technique creates multiple diverse visualizations by using different
        random seeds and optimization paths.
        
        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel
            num_diverse: Number of diverse visualizations to create
            num_steps: Number of optimization steps per visualization
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            List of diverse visualization images
        """
        obj = objectives.channel(layer_name, channel_idx)
        diverse_images = []
        
        for i in range(num_diverse):
            # Use different random seeds for diversity
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            
            # Create visualization with different initialization
            param_f = fft_image((1, 3, *image_size))
            transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
            regularized_obj = obj - 0.1 * objectives.total_variation()
            
            images = render_vis(
                self.model,
                regularized_obj,
                param_f=param_f,
                transforms=transforms,
                thresholds=(num_steps,),
                device=self.device,
                use_fixed_seed=False,
                **kwargs
            )
            
            if images:
                diverse_images.append(images[-1])
        
        return diverse_images
    
    def create_interpolation_visualization(self, layer_name: str,
                                          channel_idx1: int, channel_idx2: int,
                                          num_steps: int = 512,
                                          num_interps: int = 5,
                                          image_size: Tuple[int, int] = (128, 128),
                                          **kwargs) -> List[np.ndarray]:
        """
        Create interpolation between two channels (from the paper).
        
        Args:
            layer_name: Name of the layer
            channel_idx1: First channel index
            channel_idx2: Second channel index
            num_steps: Number of optimization steps
            num_interps: Number of interpolation steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            List of interpolation images
        """
        interpolations = []
        
        for i in range(num_interps + 1):
            # Create weighted combination of objectives
            weight1 = 1.0 - (i / num_interps)
            weight2 = i / num_interps
            
            obj1 = objectives.channel(layer_name, channel_idx1) * weight1
            obj2 = objectives.channel(layer_name, channel_idx2) * weight2
            combined_obj = obj1 + obj2
            
            # Regularize
            regularized_obj = combined_obj - 0.1 * objectives.total_variation()
            
            # Create visualization
            param_f = fft_image((1, 3, *image_size))
            transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
            
            images = render_vis(
                self.model,
                regularized_obj,
                param_f=param_f,
                transforms=transforms,
                thresholds=(num_steps,),
                device=self.device,
                **kwargs
            )
            
            if images:
                interpolations.append(images[-1])
        
        return interpolations
    
    def create_preconditioned_visualization(self, layer_name: str, channel_idx: int,
                                           preconditioning_strength: float = 0.1,
                                           num_steps: int = 1024,
                                           image_size: Tuple[int, int] = (224, 224),
                                           **kwargs) -> np.ndarray:
        """
        Create visualization with preconditioning (from the paper).
        
        Preconditioning helps optimization by adapting the learning rate
        based on the gradient history.
        
        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel
            preconditioning_strength: Strength of preconditioning
            num_steps: Number of optimization steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Preconditioned visualization image
        """
        obj = objectives.channel(layer_name, channel_idx)
        regularized_obj = obj - 0.1 * objectives.total_variation()
        
        # Use RMSprop for preconditioning effect
        import torch.optim as optim
        
        param_f = fft_image((1, 3, *image_size))
        optimizer = optim.RMSprop(param_f.parameters(), lr=0.05, alpha=0.9)
        transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
        
        images = render_vis(
            self.model,
            regularized_obj,
            param_f=param_f,
            optimizer=optimizer,
            transforms=transforms,
            thresholds=(num_steps,),
            device=self.device,
            **kwargs
        )
        
        return images[-1] if images else None
    
    def create_frequency_regularized_visualization(self, layer_name: str, channel_idx: int,
                                                  frequency_penalty: float = 1.0,
                                                  num_steps: int = 1024,
                                                  image_size: Tuple[int, int] = (224, 224),
                                                  **kwargs) -> np.ndarray:
        """
        Create visualization with frequency regularization (from the paper).
        
        Frequency regularization penalizes high-frequency content to create
        smoother, more natural-looking visualizations.
        
        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel
            frequency_penalty: Penalty strength for high frequencies
            num_steps: Number of optimization steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Frequency-regularized visualization image
        """
        # Create custom objective with frequency regularization
        def frequency_regularized_objective(activations):
            # Main objective
            main_obj = objectives.channel(layer_name, channel_idx)(activations)
            
            # Get the input image
            input_img = activations.get('input', None)
            if input_img is not None:
                # Calculate frequency content
                fft = torch.fft.fft2(input_img)
                magnitude = torch.abs(fft)
                
                # Create frequency penalty (higher frequencies get higher penalty)
                batch, channels, height, width = input_img.shape
                freqs_h = torch.fft.fftfreq(height, device=input_img.device)
                freqs_w = torch.fft.fftfreq(width, device=input_img.device)
                
                # Create frequency weight matrix
                freq_weights = torch.sqrt(freqs_h[:, None]**2 + freqs_w[None, :]**2)
                freq_weights = freq_weights.unsqueeze(0).unsqueeze(0)
                
                # Apply penalty
                freq_penalty = torch.mean(magnitude * freq_weights)
                
                return main_obj - frequency_penalty * frequency_penalty
            else:
                return main_obj
        
        param_f = fft_image((1, 3, *image_size))
        transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
        
        images = render_vis(
            self.model,
            frequency_regularized_objective,
            param_f=param_f,
            transforms=transforms,
            thresholds=(num_steps,),
            device=self.device,
            **kwargs
        )
        
        return images[-1] if images else None
    
    def create_visualization_comparison(self, layer_name: str, channel_idx: int,
                                       methods: List[str] = ['basic', 'fft', 'regularized', 'preconditioned'],
                                       output_dir: str = 'visualizations',
                                       **kwargs) -> Dict[str, np.ndarray]:
        """
        Create comparison of different visualization methods.
        
        Args:
            layer_name: Name of the layer
            channel_idx: Index of the channel
            methods: List of methods to compare
            output_dir: Directory to save results
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping method names to images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        method_functions = {
            'basic': lambda: self.visualize_channel(layer_name, channel_idx, use_fft=False, **kwargs),
            'fft': lambda: self.visualize_channel(layer_name, channel_idx, use_fft=True, **kwargs),
            'regularized': lambda: self.create_frequency_regularized_visualization(layer_name, channel_idx, **kwargs),
            'preconditioned': lambda: self.create_preconditioned_visualization(layer_name, channel_idx, **kwargs),
            'diverse': lambda: self.create_diversity_visualization(layer_name, channel_idx, num_diverse=1, **kwargs)[0] if self.create_diversity_visualization(layer_name, channel_idx, num_diverse=1, **kwargs) else None
        }
        
        for method in methods:
            if method in method_functions:
                try:
                    image = method_functions[method]()
                    if image is not None:
                        results[method] = image
                        
                        # Save individual image
                        save_path = os.path.join(output_dir, f'{layer_name}_ch{channel_idx}_{method}.png')
                        save_image(
                            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0),
                            save_path,
                            denormalize=False
                        )
                        print(f"Saved {method} visualization to {save_path}")
                        
                except Exception as e:
                    print(f"Error with {method} visualization: {e}")
                    continue
        
        return results
    
    def create_feature_visualization_grid(self, layer_name: str, 
                                        channel_indices: List[int],
                                          grid_size: Optional[Tuple[int, int]] = None,
                                          num_steps: int = 512,
                                          image_size: Tuple[int, int] = (128, 128),
                                          **kwargs) -> np.ndarray:
        """
        Create a grid of feature visualizations.
        
        Args:
            layer_name: Name of the layer
            channel_indices: List of channel indices to visualize
            grid_size: Grid size (rows, cols). If None, creates square grid
            num_steps: Number of optimization steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Grid image as numpy array
        """
        if grid_size is None:
            grid_size = (int(np.ceil(np.sqrt(len(channel_indices)))),) * 2
        
        images = []
        for channel_idx in channel_indices:
            try:
                image = self.visualize_channel(
                    layer_name, channel_idx,
                    num_steps=num_steps,
                    image_size=image_size,
                    **kwargs
                )
                if image is not None:
                    images.append(image)
            except Exception as e:
                print(f"Error visualizing channel {channel_idx}: {e}")
                # Add blank image for failed visualizations
                images.append(np.zeros((*image_size, 3)))
        
        if images:
            grid = create_image_grid(images, grid_size)
            return grid.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            return np.zeros((image_size[0] * grid_size[0], image_size[1] * grid_size[1], 3))