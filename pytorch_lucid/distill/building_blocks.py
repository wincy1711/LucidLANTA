"""
Building Blocks of Interpretability implementations from the Distill paper.

This module implements the techniques described in:
"The Building Blocks of Interpretability" by Olah, Satyanarayan, Johnson, Carter, 
Schubert, Ye, and Mordvintsev (2018)
https://distill.pub/2018/building-blocks/

Key techniques:
- Combined interpretability interfaces
- Feature visualization with attribution
- Semantic dictionaries
- Activation atlases concepts
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from ..optvis import render_vis, objectives
from ..optvis.param import fft_image, image
from ..optvis.transform import standard_transforms
from ..misc.io import save_image, create_image_grid


class BuildingBlocks:
    """
    Implementation of Building Blocks of Interpretability techniques.
    
    This class provides methods for creating the combined interpretability
    interfaces described in the paper, including feature visualization with
    attribution, semantic dictionaries, and activation-based analysis.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize building blocks.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def create_semantic_dictionary(self, layer_name: str,
                                 top_k: int = 12,
                                 num_steps: int = 512,
                                 image_size: Tuple[int, int] = (128, 128),
                                 **kwargs) -> Dict[str, Any]:
        """
        Create a semantic dictionary for a layer (from the paper).
        
        A semantic dictionary visualizes the most important neurons/channels
        in a layer to understand what features the layer detects.
        
        Args:
            layer_name: Name of the layer
            top_k: Number of top features to visualize
            num_steps: Number of optimization steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Dictionary containing semantic dictionary information
        """
        # Get layer information
        layer_info = self._get_layer_info(layer_name)
        if not layer_info:
            return {}
        
        # Determine what to visualize (neurons or channels)
        if 'out_channels' in layer_info:
            # Convolutional layer - visualize channels
            num_features = layer_info['out_channels']
            feature_type = 'channel'
            feature_indices = list(range(min(top_k, num_features)))
        else:
            # Other layer - try to visualize neurons
            feature_type = 'neuron'
            feature_indices = list(range(min(top_k, 64)))  # Default to 64 neurons
        
        # Create visualizations
        visualizations = {}
        for idx in feature_indices:
            try:
                if feature_type == 'channel':
                    image = self._visualize_channel(layer_name, idx, num_steps, image_size, **kwargs)
                else:
                    image = self._visualize_neuron(layer_name, idx, num_steps, image_size, **kwargs)
                
                if image is not None:
                    visualizations[f'{feature_type}_{idx}'] = image
                    
            except Exception as e:
                print(f"Error visualizing {feature_type} {idx}: {e}")
                continue
        
        # Create summary grid
        if visualizations:
            images = list(visualizations.values())
            grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2
            grid = create_image_grid(images, grid_size)
            grid_array = grid.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            grid_array = None
        
        return {
            'layer_name': layer_name,
            'feature_type': feature_type,
            'visualizations': visualizations,
            'semantic_grid': grid_array,
            'num_features': len(visualizations)
        }
    
    def create_attribution_visualization(self, layer_name: str, channel_idx: int,
                                        attribution_layer: str = 'output',
                                        class_idx: Optional[int] = None,
                                        num_steps: int = 512,
                                        image_size: Tuple[int, int] = (224, 224),
                                        **kwargs) -> Dict[str, Any]:
        """
        Create visualization with attribution (from the paper).
        
        This combines feature visualization with attribution to understand
        how a feature contributes to the final output.
        
        Args:
            layer_name: Name of the layer to visualize
            channel_idx: Index of the channel to visualize
            attribution_layer: Layer to calculate attribution to
            class_idx: Specific class to attribute to (if None, uses predicted class)
            num_steps: Number of optimization steps
            image_size: Output image size
            **kwargs: Additional arguments for render_vis
            
        Returns:
            Dictionary containing visualization and attribution information
        """
        # Create feature visualization
        feature_obj = objectives.channel(layer_name, channel_idx)
        feature_image = self._visualize_objective(
            feature_obj, num_steps, image_size, **kwargs
        )
        
        if feature_image is None:
            return {}
        
        # Convert to tensor for attribution
        feature_tensor = torch.from_numpy(feature_image).permute(2, 0, 1).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)
        
        # Calculate attribution
        attribution = self._calculate_attribution(
            feature_tensor, attribution_layer, class_idx
        )
        
        return {
            'feature_visualization': feature_image,
            'attribution_map': attribution,
            'layer_name': layer_name,
            'channel_idx': channel_idx,
            'attribution_layer': attribution_layer
        }
    
    def create_combined_interface(self, input_image: np.ndarray,
                                 layer_name: str,
                                 class_idx: Optional[int] = None,
                                 top_k: int = 5,
                                 **kwargs) -> Dict[str, Any]:
        """
        Create the combined interpretability interface from the paper.
        
        This creates a comprehensive interface showing:
        1. Feature visualizations for top channels
        2. Attribution to the output
        3. Spatial activation heatmaps
        
        Args:
            input_image: Input image to analyze
            layer_name: Name of the layer to analyze
            class_idx: Target class (if None, uses predicted class)
            top_k: Number of top channels to show
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing all interface components
        """
        # Convert input image to tensor
        if len(input_image.shape) == 3:
            input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(input_image).unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Get model predictions and activations
        with torch.no_grad():
            # Forward pass with hooks to get activations
            activations = {}
            hooks = []
            
            def get_activation(name):
                def hook(model, input, output):
                    if isinstance(output, tuple):
                        activations[name] = output[0]
                    else:
                        activations[name] = output
                return hook
            
            # Register hook for target layer
            for name, module in self.model.named_modules():
                if name == layer_name:
                    hook = module.register_forward_hook(get_activation(name))
                    hooks.append(hook)
                    break
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Determine target class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Get top channels by attribution
        layer_activations = activations.get(layer_name, None)
        if layer_activations is not None:
            # Calculate attribution scores
            attribution_scores = self._calculate_channel_attribution(
                layer_activations, output, class_idx
            )
            
            # Get top channels
            top_channels = torch.topk(attribution_scores, min(top_k, len(attribution_scores)))[1]
            top_channels = top_channels.cpu().numpy()
        else:
            # Fallback to first few channels
            top_channels = list(range(min(top_k, 64)))
        
        # Create feature visualizations for top channels
        feature_visualizations = {}
        for channel_idx in top_channels:
            try:
                viz = self._visualize_channel(layer_name, channel_idx, **kwargs)
                if viz is not None:
                    feature_visualizations[f'channel_{channel_idx}'] = viz
            except Exception as e:
                print(f"Error visualizing channel {channel_idx}: {e}")
                continue
        
        # Create spatial activation maps
        spatial_activations = {}
        if layer_activations is not None:
            # Get spatial activations for top channels
            for channel_idx in top_channels[:3]:  # Limit for visualization
                if channel_idx < layer_activations.shape[1]:
                    spatial_act = layer_activations[0, channel_idx].cpu().numpy()
                    spatial_activations[f'channel_{channel_idx}'] = spatial_act
        
        # Create attribution visualization
        attribution_map = self._calculate_attribution(input_tensor, 'output', class_idx)
        
        return {
            'input_image': input_image,
            'predicted_class': class_idx,
            'confidence': torch.softmax(output, dim=1)[0, class_idx].item(),
            'feature_visualizations': feature_visualizations,
            'spatial_activations': spatial_activations,
            'attribution_map': attribution_map,
            'top_channels': top_channels
        }
    
    def create_activation_atlas_concept(self, layer_name: str,
                                       num_samples: int = 1000,
                                       num_clusters: int = 16,
                                       **kwargs) -> Dict[str, Any]:
        """
        Create activation atlas concept (simplified version).
        
        This creates a simplified version of the activation atlas technique,
        clustering activations and visualizing cluster centers.
        
        Args:
            layer_name: Name of the layer
            num_samples: Number of sample images to use
            num_clusters: Number of clusters to create
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing atlas information
        """
        # Generate random samples (in practice, you'd use a real dataset)
        sample_activations = []
        
        for i in range(num_samples):
            # Create random input
            random_input = torch.randn(1, 3, 224, 224, device=self.device)
            
            # Get activations
            with torch.no_grad():
                activations = {}
                hooks = []
                
                def get_activation(name):
                    def hook(model, input, output):
                        if isinstance(output, tuple):
                            activations[name] = output[0]
                        else:
                            activations[name] = output
                    return hook
                
                # Register hook
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        hook = module.register_forward_hook(get_activation(name))
                        hooks.append(hook)
                        break
                
                _ = self.model(random_input)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Store activation
                if layer_name in activations:
                    # Flatten spatial dimensions
                    act_flat = activations[layer_name].flatten(2).mean(dim=2)  # Average pool spatial
                    sample_activations.append(act_flat.cpu().numpy())
        
        if not sample_activations:
            return {}
        
        # Stack activations
        activations_array = np.concatenate(sample_activations, axis=0)
        
        # Cluster activations
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(activations_array)
        cluster_centers = kmeans.cluster_centers_
        
        # Visualize cluster centers
        cluster_visualizations = {}
        for i in range(num_clusters):
            # Create objective that maximizes similarity to cluster center
            def cluster_objective(activations):
                current_act = activations[layer_name].flatten(2).mean(dim=2)
                target_act = torch.from_numpy(cluster_centers[i:i+1]).to(current_act.device)
                
                # Maximize cosine similarity
                similarity = F.cosine_similarity(current_act, target_act, dim=1)
                return similarity.mean()
            
            # Create visualization
            param_f = fft_image((1, 3, 224, 224))
            transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
            
            images = render_vis(
                self.model,
                cluster_objective,
                param_f=param_f,
                transforms=transforms,
                thresholds=(512,),
                device=self.device,
                **kwargs
            )
            
            if images:
                cluster_visualizations[f'cluster_{i}'] = images[-1]
        
        return {
            'layer_name': layer_name,
            'cluster_centers': cluster_centers,
            'cluster_labels': cluster_labels,
            'cluster_visualizations': cluster_visualizations,
            'num_clusters': num_clusters,
            'num_samples': num_samples
        }
    
    def _visualize_channel(self, layer_name: str, channel_idx: int,
                          num_steps: int = 512,
                          image_size: Tuple[int, int] = (128, 128),
                          **kwargs) -> Optional[np.ndarray]:
        """Helper to visualize a channel."""
        obj = objectives.channel(layer_name, channel_idx)
        return self._visualize_objective(obj, num_steps, image_size, **kwargs)
    
    def _visualize_neuron(self, layer_name: str, neuron_idx: int,
                         num_steps: int = 512,
                         image_size: Tuple[int, int] = (128, 128),
                         **kwargs) -> Optional[np.ndarray]:
        """Helper to visualize a neuron."""
        obj = objectives.neuron(layer_name, neuron_idx)
        return self._visualize_objective(obj, num_steps, image_size, **kwargs)
    
    def _visualize_objective(self, objective, num_steps: int, image_size: Tuple[int, int],
                           **kwargs) -> Optional[np.ndarray]:
        """Helper to visualize an objective."""
        param_f = fft_image((1, 3, *image_size))
        transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))
        regularized_obj = objective - 0.1 * objectives.total_variation()
        
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
    
    def _get_layer_info(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a layer."""
        try:
            # Create dummy input to get layer info
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            
            layer_info = {}
            hooks = []
            
            def get_info_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out_tensor = output[0]
                    else:
                        out_tensor = output
                    
                    layer_info['shape'] = tuple(out_tensor.shape)
                    if len(out_tensor.shape) == 4:  # Conv layer
                        layer_info['out_channels'] = out_tensor.shape[1]
                    layer_info['type'] = type(module).__name__
                
                return hook
            
            # Register hook
            for name, module in self.model.named_modules():
                if name == layer_name:
                    hook = module.register_forward_hook(get_info_hook(name))
                    hooks.append(hook)
                    break
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return layer_info
            
        except Exception as e:
            print(f"Error getting layer info for {layer_name}: {e}")
            return None
    
    def _calculate_attribution(self, input_tensor: torch.Tensor,
                              attribution_layer: str = 'output',
                              class_idx: Optional[int] = None) -> np.ndarray:
        """Calculate attribution map."""
        try:
            # Simple gradient-based attribution
            input_tensor.requires_grad = True
            
            # Forward pass
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            # Calculate gradient
            target_output = output[0, class_idx]
            target_output.backward()
            
            # Get gradient as attribution
            attribution = input_tensor.grad[0].detach().cpu().numpy()
            attribution = np.abs(attribution).mean(axis=0)  # Average over channels
            
            # Normalize
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
            
            return attribution
            
        except Exception as e:
            print(f"Error calculating attribution: {e}")
            return np.zeros((224, 224))
    
    def _calculate_channel_attribution(self, layer_activations: torch.Tensor,
                                      output: torch.Tensor,
                                      class_idx: int) -> torch.Tensor:
        """Calculate attribution scores for channels."""
        try:
            # Simple gradient-based channel attribution
            batch_size, num_channels = layer_activations.shape[:2]
            attribution_scores = torch.zeros(num_channels, device=layer_activations.device)
            
            # Calculate gradient of output with respect to layer activations
            target_output = output[0, class_idx]
            grad = torch.autograd.grad(target_output, layer_activations, retain_graph=True)[0]
            
            # Average over spatial dimensions
            if len(grad.shape) == 4:
                grad = grad.mean(dim=(2, 3))
            
            # Take absolute value and average over batch
            attribution_scores = torch.abs(grad[0])
            
            return attribution_scores
            
        except Exception as e:
            print(f"Error calculating channel attribution: {e}")
            # Fallback to random scores
            return torch.rand(layer_activations.shape[1])