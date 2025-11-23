"""
Thread: Circuits implementations from the Distill paper.

This module implements the techniques described in:
"Thread: Circuits" by Cammarata, Carter, Goh, Olah, Petrov, Schubert, Voss, Egan, and Lim (2020)
https://distill.pub/2020/circuits/

Key techniques:
- Circuit analysis and reverse engineering
- Curve detector and high-low frequency detector analysis
- Weight visualization and banding analysis
- Branch specialization detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ..optvis import render_vis, objectives
from ..optvis.param import fft_image, image
from ..optvis.transform import standard_transforms
from ..misc.io import save_image, create_image_grid


class Circuits:
    """
    Implementation of Thread: Circuits analysis techniques.
    
    This class provides methods for analyzing neural network circuits,
    including feature families, weight patterns, and learned algorithms.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize circuits analysis.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def analyze_curve_detectors(self, layer_name: str,
                               num_samples: int = 100,
                               angle_range: Tuple[float, float] = (0, 180),
                               **kwargs) -> Dict[str, Any]:
        """
        Analyze curve detectors in a layer (from the paper).
        
        This technique identifies neurons that respond to curves and
        analyzes their properties like orientation and curvature.
        
        Args:
            layer_name: Name of the layer to analyze
            num_samples: Number of sample images to test
            angle_range: Range of angles to test (degrees)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing curve detector analysis
        """
        # Generate test patterns for curves
        curve_responses = {}
        
        for angle in np.linspace(angle_range[0], angle_range[1], 18):  # 10 degree increments
            # Create curved test pattern
            test_pattern = self._create_curve_pattern(angle, size=64)
            test_tensor = torch.from_numpy(test_pattern).permute(2, 0, 1).unsqueeze(0)
            test_tensor = test_tensor.to(self.device)
            
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
                
                _ = self.model(test_tensor)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Store responses
                if layer_name in activations:
                    layer_acts = activations[layer_name]
                    if len(layer_acts.shape) == 4:
                        # Average over spatial dimensions
                        responses = layer_acts.mean(dim=(2, 3)).squeeze(0)
                        curve_responses[angle] = responses.cpu().numpy()
        
        # Find neurons that respond strongly to curves
        curve_neurons = []
        if curve_responses:
            # Calculate response variance across angles
            all_responses = np.array(list(curve_responses.values()))
            response_variance = np.var(all_responses, axis=0)
            
            # Find neurons with high variance (curve selective)
            threshold = np.percentile(response_variance, 90)
            curve_neuron_indices = np.where(response_variance > threshold)[0]
            
            for neuron_idx in curve_neuron_indices[:10]:  # Top 10
                # Get preferred orientation
                neuron_responses = all_responses[:, neuron_idx]
                preferred_angle = list(curve_responses.keys())[np.argmax(neuron_responses)]
                
                curve_neurons.append({
                    'neuron_idx': neuron_idx,
                    'preferred_angle': preferred_angle,
                    'response_variance': response_variance[neuron_idx],
                    'max_response': np.max(neuron_responses)
                })
        
        # Create visualizations for top curve detectors
        curve_visualizations = {}
        for curve_info in curve_neurons[:5]:  # Top 5
            neuron_idx = curve_info['neuron_idx']
            try:
                viz = self._visualize_neuron(layer_name, neuron_idx, **kwargs)
                if viz is not None:
                    curve_visualizations[f'neuron_{neuron_idx}'] = viz
            except Exception as e:
                print(f"Error visualizing curve neuron {neuron_idx}: {e}")
                continue
        
        return {
            'layer_name': layer_name,
            'curve_neurons': curve_neurons,
            'curve_responses': curve_responses,
            'curve_visualizations': curve_visualizations,
            'num_curve_detectors': len(curve_neurons)
        }
    
    def analyze_high_low_frequency_detectors(self, layer_name: str,
                                           num_samples: int = 50,
                                           **kwargs) -> Dict[str, Any]:
        """
        Analyze high-low frequency detectors (from the paper).
        
        These neurons detect transitions from high to low frequency content.
        
        Args:
            layer_name: Name of the layer to analyze
            num_samples: Number of test patterns
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing high-low frequency detector analysis
        """
        # Create test patterns with frequency transitions
        frequency_responses = {}
        
        for i in range(num_samples):
            # Create pattern with high-low frequency transition
            test_pattern = self._create_frequency_transition_pattern(i, size=64)
            test_tensor = torch.from_numpy(test_pattern).permute(2, 0, 1).unsqueeze(0)
            test_tensor = test_tensor.to(self.device)
            
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
                
                _ = self.model(test_tensor)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Store responses
                if layer_name in activations:
                    layer_acts = activations[layer_name]
                    if len(layer_acts.shape) == 4:
                        responses = layer_acts.mean(dim=(2, 3)).squeeze(0)
                        frequency_responses[f'pattern_{i}'] = responses.cpu().numpy()
        
        # Find high-low frequency detectors
        hl_neurons = []
        if frequency_responses:
            all_responses = np.array(list(frequency_responses.values()))
            
            # Look for neurons that respond strongly to frequency transitions
            response_mean = np.mean(all_responses, axis=0)
            response_std = np.std(all_responses, axis=0)
            
            # Find neurons with high mean response and moderate variance
            # (consistent but not uniform response)
            scores = response_mean * (response_std / (response_mean + 1e-8))
            
            threshold = np.percentile(scores, 95)
            hl_neuron_indices = np.where(scores > threshold)[0]
            
            for neuron_idx in hl_neuron_indices[:10]:
                hl_neurons.append({
                    'neuron_idx': neuron_idx,
                    'mean_response': response_mean[neuron_idx],
                    'response_std': response_std[neuron_idx],
                    'score': scores[neuron_idx]
                })
        
        # Create visualizations
        hl_visualizations = {}
        for hl_info in hl_neurons[:5]:
            neuron_idx = hl_info['neuron_idx']
            try:
                viz = self._visualize_neuron(layer_name, neuron_idx, **kwargs)
                if viz is not None:
                    hl_visualizations[f'neuron_{neuron_idx}'] = viz
            except Exception as e:
                print(f"Error visualizing HL neuron {neuron_idx}: {e}")
                continue
        
        return {
            'layer_name': layer_name,
            'hl_neurons': hl_neurons,
            'frequency_responses': frequency_responses,
            'hl_visualizations': hl_visualizations,
            'num_hl_detectors': len(hl_neurons)
        }
    
    def visualize_weights(self, layer_name: str,
                         max_channels: int = 16,
                         **kwargs) -> Dict[str, Any]:
        """
        Visualize weights of a layer (from the paper).
        
        This technique visualizes the learned weights to understand
        what patterns the layer is looking for.
        
        Args:
            layer_name: Name of the layer
            max_channels: Maximum number of channels to visualize
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing weight visualizations
        """
        weight_visualizations = {}
        
        try:
            # Get the layer
            layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer = module
                    break
            
            if layer is None or not hasattr(layer, 'weight'):
                print(f"Layer {layer_name} not found or has no weights")
                return {}
            
            weights = layer.weight.data
            
            # For convolutional layers, visualize filters
            if len(weights.shape) == 4:  # Conv layer: (out_channels, in_channels, H, W)
                out_channels = min(max_channels, weights.shape[0])
                
                for i in range(out_channels):
                    # Get filter weights
                    filter_weights = weights[i].cpu().numpy()
                    
                    # Normalize for visualization
                    filter_norm = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min() + 1e-8)
                    
                    # Convert to RGB if needed
                    if filter_norm.shape[0] == 1:  # Grayscale
                        filter_rgb = np.repeat(filter_norm, 3, axis=0)
                    elif filter_norm.shape[0] == 3:  # RGB
                        filter_rgb = filter_norm
                    else:  # More channels, take first 3
                        filter_rgb = filter_norm[:3]
                    
                    # Transpose to HWC format
                    filter_vis = filter_rgb.transpose(1, 2, 0)
                    
                    weight_visualizations[f'filter_{i}'] = filter_vis
            
            # Create weight grid
            if weight_visualizations:
                images = list(weight_visualizations.values())
                grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2
                grid = create_image_grid(images, grid_size)
                weight_grid = grid.detach().cpu().numpy().transpose(1, 2, 0)
            else:
                weight_grid = None
            
            return {
                'layer_name': layer_name,
                'weight_visualizations': weight_visualizations,
                'weight_grid': weight_grid,
                'num_filters': len(weight_visualizations),
                'weight_shape': tuple(weights.shape)
            }
            
        except Exception as e:
            print(f"Error visualizing weights for {layer_name}: {e}")
            return {}
    
    def analyze_weight_banding(self, layer_name: str,
                              num_bands: int = 8,
                              **kwargs) -> Dict[str, Any]:
        """
        Analyze weight banding patterns (from the paper).
        
        Weight banding is a phenomenon where weights are organized
        into distinct bands or groups.
        
        Args:
            layer_name: Name of the layer
            num_bands: Number of bands to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing banding analysis
        """
        try:
            # Get the layer
            layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer = module
                    break
            
            if layer is None or not hasattr(layer, 'weight'):
                return {}
            
            weights = layer.weight.data.cpu().numpy()
            
            # Analyze banding in convolutional weights
            if len(weights.shape) == 4:
                # Flatten spatial dimensions
                weights_flat = weights.reshape(weights.shape[0], -1)
                
                # Apply PCA to find banding patterns
                pca = PCA(n_components=2)
                weights_pca = pca.fit_transform(weights_flat)
                
                # Cluster to find bands
                kmeans = KMeans(n_clusters=num_bands, random_state=42)
                bands = kmeans.fit_predict(weights_pca)
                
                # Calculate band statistics
                band_stats = {}
                for i in range(num_bands):
                    band_mask = bands == i
                    if band_mask.sum() > 0:
                        band_weights = weights_flat[band_mask]
                        band_stats[f'band_{i}'] = {
                            'size': band_mask.sum(),
                            'mean_weight': band_weights.mean(),
                            'std_weight': band_weights.std(),
                            'channel_indices': np.where(band_mask)[0].tolist()
                        }
                
                return {
                    'layer_name': layer_name,
                    'weight_shape': tuple(weights.shape),
                    'bands': bands,
                    'band_stats': band_stats,
                    'pca_components': weights_pca,
                    'num_bands': num_bands,
                    'explained_variance': pca.explained_variance_ratio_
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error analyzing weight banding for {layer_name}: {e}")
            return {}
    
    def detect_branch_specialization(self, layer_name: str,
                                   num_samples: int = 100,
                                   **kwargs) -> Dict[str, Any]:
        """
        Detect branch specialization (from the paper).
        
        When a layer is divided into multiple branches, neurons
        self-organize into coherent groupings.
        
        Args:
            layer_name: Name of the layer
            num_samples: Number of sample images to test
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing specialization analysis
        """
        try:
            # This is a simplified implementation
            # In practice, you'd need to know the branch structure
            
            # Get the layer
            layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer = module
                    break
            
            if layer is None:
                return {}
            
            # Test with random inputs
            specialization_patterns = []
            
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
                    
                    # Store activation pattern
                    if layer_name in activations:
                        layer_acts = activations[layer_name]
                        if len(layer_acts.shape) == 4:
                            # Average spatial dimensions
                            pattern = layer_acts.mean(dim=(2, 3)).squeeze(0)
                            specialization_patterns.append(pattern.cpu().numpy())
            
            if specialization_patterns:
                patterns_array = np.array(specialization_patterns)
                
                # Analyze specialization using PCA
                pca = PCA(n_components=min(5, patterns_array.shape[1]))
                patterns_pca = pca.fit_transform(patterns_array)
                
                # Find specialized groups
                n_groups = min(4, patterns_array.shape[1])
                kmeans = KMeans(n_clusters=n_groups, random_state=42)
                groups = kmeans.fit_predict(patterns_pca)
                
                # Calculate group statistics
                group_stats = {}
                for i in range(n_groups):
                    group_mask = groups == i
                    if group_mask.sum() > 0:
                        group_patterns = patterns_array[group_mask]
                        group_stats[f'group_{i}'] = {
                            'size': group_mask.sum(),
                            'mean_activation': group_patterns.mean(axis=0),
                            'std_activation': group_patterns.std(axis=0),
                            'specialization_strength': np.std(group_patterns.mean(axis=0))
                        }
                
                return {
                    'layer_name': layer_name,
                    'num_groups': n_groups,
                    'group_assignments': groups,
                    'group_stats': group_stats,
                    'pca_components': patterns_pca,
                    'explained_variance': pca.explained_variance_ratio_,
                    'specialization_score': np.mean([stats['specialization_strength'] 
                                                   for stats in group_stats.values()])
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error detecting branch specialization for {layer_name}: {e}")
            return {}
    
    def reverse_engineer_curve_algorithm(self, layer_name: str,
                                        curve_neuron_idx: int,
                                        **kwargs) -> Dict[str, Any]:
        """
        Reverse engineer curve detection algorithm (from the paper).
        
        This attempts to understand how a curve detector neuron works
        by analyzing its weights and responses.
        
        Args:
            layer_name: Name of the layer
            curve_neuron_idx: Index of curve detector neuron
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing algorithm analysis
        """
        try:
            # Get the neuron
            layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer = module
                    break
            
            if layer is None:
                return {}
            
            # Analyze the neuron's weights
            if hasattr(layer, 'weight'):
                weights = layer.weight.data.cpu().numpy()
                
                if len(weights.shape) >= 2 and curve_neuron_idx < weights.shape[0]:
                    neuron_weights = weights[curve_neuron_idx]
                    
                    # Analyze weight patterns
                    weight_analysis = self._analyze_curve_weights(neuron_weights)
                    
                    # Test with different curve patterns
                    curve_tests = self._test_curve_responses(layer_name, curve_neuron_idx)
                    
                    return {
                        'layer_name': layer_name,
                        'neuron_idx': curve_neuron_idx,
                        'weight_analysis': weight_analysis,
                        'curve_tests': curve_tests,
                        'algorithm_hypothesis': self._generate_curve_hypothesis(weight_analysis, curve_tests)
                    }
            
            return {}
            
        except Exception as e:
            print(f"Error reverse engineering curve algorithm: {e}")
            return {}
    
    def _create_curve_pattern(self, angle: float, size: int = 64, curvature: float = 0.1) -> np.ndarray:
        """Create a curved test pattern."""
        # Create coordinate grid
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create curve based on angle and curvature
        angle_rad = np.radians(angle)
        
        # Rotate coordinates
        X_rot = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
        Y_rot = X * np.sin(angle_rad) + Y * np.cos(angle_rad)
        
        # Create curved pattern
        curve = np.exp(-((X_rot - curvature * Y_rot**2)**2) / 0.1)
        
        # Convert to RGB
        pattern = np.stack([curve] * 3, axis=-1)
        return (pattern * 255).astype(np.uint8)
    
    def _create_frequency_transition_pattern(self, pattern_idx: int, size: int = 64) -> np.ndarray:
        """Create a pattern with frequency transition."""
        # Create coordinate grid
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # High frequency on left, low frequency on right
        high_freq = np.sin(X * 8) * np.sin(Y * 8)
        low_freq = np.sin(X * 2) * np.sin(Y * 2)
        
        # Smooth transition
        transition = (np.tanh((X - np.pi) * 2) + 1) / 2
        pattern = high_freq * (1 - transition) + low_freq * transition
        
        # Convert to RGB
        pattern_rgb = np.stack([pattern] * 3, axis=-1)
        pattern_norm = (pattern_rgb - pattern_rgb.min()) / (pattern_rgb.max() - pattern_rgb.min() + 1e-8)
        return (pattern_norm * 255).astype(np.uint8)
    
    def _analyze_curve_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        """Analyze curve detector weights."""
        analysis = {}
        
        if len(weights.shape) == 3:  # (in_channels, H, W)
            # Analyze spatial patterns
            h, w = weights.shape[1:]
            
            # Look for curved patterns in weights
            weight_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
            # Calculate curvature-like measures
            center_h, center_w = h // 2, w // 2
            
            # Fit to curve model
            y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')
            
            analysis = {
                'weight_shape': weights.shape,
                'spatial_extent': (h, w),
                'weight_statistics': {
                    'mean': float(weights.mean()),
                    'std': float(weights.std()),
                    'min': float(weights.min()),
                    'max': float(weights.max())
                },
                'has_curved_pattern': self._detect_curved_pattern(weights)
            }
        
        return analysis
    
    def _detect_curved_pattern(self, weights: np.ndarray) -> bool:
        """Detect if weights contain curved patterns."""
        # Simple heuristic: look for non-linear spatial patterns
        if len(weights.shape) >= 2:
            # Calculate spatial derivatives
            spatial_weights = weights.reshape(weights.shape[0], -1)
            
            # Look for patterns that suggest curvature
            # This is a simplified check
            weight_std = spatial_weights.std(axis=1).mean()
            return weight_std > 0.1  # Threshold for meaningful pattern
        
        return False
    
    def _test_curve_responses(self, layer_name: str, neuron_idx: int) -> Dict[str, Any]:
        """Test curve neuron responses to different patterns."""
        tests = {}
        
        try:
            # Test with different curve patterns
            curve_angles = [0, 30, 60, 90, 120, 150]
            responses = []
            
            for angle in curve_angles:
                pattern = self._create_curve_pattern(angle, size=32)
                pattern_tensor = torch.from_numpy(pattern).permute(2, 0, 1).unsqueeze(0)
                pattern_tensor = pattern_tensor.to(self.device)
                
                # Get neuron response
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
                    
                    # Register hook for specific neuron
                    for name, module in self.model.named_modules():
                        if name == layer_name:
                            hook = module.register_forward_hook(get_activation(name))
                            hooks.append(hook)
                            break
                    
                    _ = self.model(pattern_tensor)
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
                    if layer_name in activations:
                        layer_acts = activations[layer_name]
                        if len(layer_acts.shape) == 4:
                            # Get specific neuron response
                            if neuron_idx < layer_acts.shape[1]:
                                neuron_response = layer_acts[0, neuron_idx].mean().item()
                                responses.append(neuron_response)
            
            tests = {
                'curve_orientations': curve_angles,
                'curve_responses': responses if responses else [0] * len(curve_angles),
                'preferred_orientation': curve_angles[np.argmax(responses)] if responses else 0,
                'tuning_width': np.std(responses) if responses else 0
            }
            
        except Exception as e:
            print(f"Error testing curve responses: {e}")
            tests = {'error': str(e)}
        
        return tests
    
    def _generate_curve_hypothesis(self, weight_analysis: Dict[str, Any],
                                  curve_tests: Dict[str, Any]) -> str:
        """Generate hypothesis about curve detection algorithm."""
        hypothesis = "Based on analysis: "
        
        if weight_analysis.get('has_curved_pattern', False):
            hypothesis += "Weights show curved spatial patterns. "
        
        if 'preferred_orientation' in curve_tests:
            hypothesis += f"Neuron prefers {curve_tests['preferred_orientation']}° orientation. "
        
        if 'tuning_width' in curve_tests and curve_tests['tuning_width'] > 0:
            hypothesis += f"Tuning width suggests {curve_tests['tuning_width']:.2f} selectivity. "
        
        hypothesis += "Algorithm likely involves weighted combination of input pixels with curved spatial arrangement."
        
        return hypothesis
    
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