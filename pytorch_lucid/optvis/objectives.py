"""
Objective functions for visualizing neural networks in PyTorch.

This module implements the core objective functions used for feature visualization.
Objectives describe what you want to optimize, whether a single neuron, channel,
layer activation, or more complex objectives like deepdream or style transfer.
"""

import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union, Optional, Dict, Any


def _dot(x, y):
    """Compute dot product between two tensors."""
    return torch.sum(x * y)


def _dot_cossim(x, y, cossim_pow=1.0):
    """Compute dot product with cosine similarity weighting."""
    dot = torch.sum(x * y)
    mag = torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
    cossim = dot / (mag + 1e-8)
    return dot * cossim ** cossim_pow


def _extract_act_pos(acts, x=None, y=None):
    """Extract activation at specific position."""
    if x is None and y is None:
        return acts
    elif x is not None and y is not None:
        return acts[:, :, y:y+1, x:x+1]
    else:
        raise ValueError("Must specify both x and y or neither")


def _make_arg_str(*args, **kwargs):
    """Create string representation of arguments."""
    arg_strs = [str(arg) for arg in args]
    kwarg_strs = [f"{k}={v}" for k, v in kwargs.items()]
    return ", ".join(arg_strs + kwarg_strs)


def _handle_batch(x, batch=None):
    """Handle batch dimension for activations."""
    if batch is not None:
        return x[batch:batch+1]
    return x


class Objective(ABC):
    """
    Abstract base class for objectives.
    
    Objectives are functions that take a model's layer activations and return
    a scalar loss value to be optimized.
    """
    
    def __init__(self, description: str = "Objective"):
        self.description = description
    
    @abstractmethod
    def __call__(self, layer_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the objective value from layer activations.
        
        Args:
            layer_activations: Dictionary mapping layer names to activation tensors
            
        Returns:
            Scalar tensor representing the objective value
        """
        pass
    
    def __add__(self, other):
        """Combine objectives by addition."""
        return CombinedObjective([self, other], "add")
    
    def __mul__(self, scalar):
        """Scale objective by a scalar."""
        return ScaledObjective(self, scalar)
    
    def __rmul__(self, scalar):
        """Scale objective by a scalar (reverse order)."""
        return ScaledObjective(self, scalar)
    
    def __sub__(self, other):
        """Subtract objectives."""
        return CombinedObjective([self, -other], "add")
    
    def __neg__(self):
        """Negate objective."""
        return ScaledObjective(self, -1.0)


class CombinedObjective(Objective):
    """Combine multiple objectives."""
    
    def __init__(self, objectives, operation="add"):
        self.objectives = objectives
        self.operation = operation
        desc = f"Combined({operation}): " + " + ".join([obj.description for obj in objectives])
        super().__init__(desc)
    
    def __call__(self, layer_activations):
        values = [obj(layer_activations) for obj in self.objectives]
        
        if self.operation == "add":
            return sum(values)
        elif self.operation == "mul":
            result = values[0]
            for v in values[1:]:
                result = result * v
            return result
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class ScaledObjective(Objective):
    """Scale an objective by a scalar factor."""
    
    def __init__(self, objective, scale):
        self.objective = objective
        self.scale = scale
        super().__init__(f"{scale} * {objective.description}")
    
    def __call__(self, layer_activations):
        return self.scale * self.objective(layer_activations)


class ChannelObjective(Objective):
    """Objective for maximizing a specific channel's activation."""
    
    def __init__(self, layer_name: str, channel_idx: int, batch: Optional[int] = None):
        self.layer_name = layer_name
        self.channel_idx = channel_idx
        self.batch = batch
        super().__init__(f"channel({layer_name}, {channel_idx})")
    
    def __call__(self, layer_activations):
        if self.layer_name not in layer_activations:
            raise KeyError(f"Layer '{self.layer_name}' not found. Available layers: {list(layer_activations.keys())}")
        
        acts = layer_activations[self.layer_name]
        acts = _handle_batch(acts, self.batch)
        
        # Handle different tensor formats (NCHW or NHWC)
        if len(acts.shape) == 4:
            if acts.shape[1] == self.channel_idx + 1:  # NCHW format
                channel_acts = acts[:, self.channel_idx]
            else:  # NHWC format
                channel_acts = acts[:, :, :, self.channel_idx]
        else:
            raise ValueError(f"Expected 4D tensor, got {len(acts.shape)}D")
        
        return torch.mean(channel_acts)


class NeuronObjective(Objective):
    """Objective for maximizing a specific neuron's activation."""
    
    def __init__(self, layer_name: str, channel_idx: int, x: Optional[int] = None, 
                 y: Optional[int] = None, batch: Optional[int] = None):
        self.layer_name = layer_name
        self.channel_idx = channel_idx
        self.x = x
        self.y = y
        self.batch = batch
        super().__init__(f"neuron({layer_name}, {channel_idx}, x={x}, y={y})")
    
    def __call__(self, layer_activations):
        if self.layer_name not in layer_activations:
            raise KeyError(f"Layer '{self.layer_name}' not found. Available layers: {list(layer_activations.keys())}")
        
        acts = layer_activations[self.layer_name]
        acts = _handle_batch(acts, self.batch)
        
        # Extract specific channel
        if len(acts.shape) == 4:
            if acts.shape[1] == self.channel_idx + 1:  # NCHW format
                channel_acts = acts[:, self.channel_idx: self.channel_idx + 1]
            else:  # NHWC format
                channel_acts = acts[:, :, :, self.channel_idx: self.channel_idx + 1]
        else:
            raise ValueError(f"Expected 4D tensor, got {len(acts.shape)}D")
        
        # Extract specific spatial position if specified
        if self.x is not None and self.y is not None:
            channel_acts = _extract_act_pos(channel_acts, self.x, self.y)
        
        return torch.mean(channel_acts)


class LayerObjective(Objective):
    """Objective for maximizing a layer's overall activation."""
    
    def __init__(self, layer_name: str, batch: Optional[int] = None):
        self.layer_name = layer_name
        self.batch = batch
        super().__init__(f"layer({layer_name})")
    
    def __call__(self, layer_activations):
        if self.layer_name not in layer_activations:
            raise KeyError(f"Layer '{self.layer_name}' not found. Available layers: {list(layer_activations.keys())}")
        
        acts = layer_activations[self.layer_name]
        acts = _handle_batch(acts, self.batch)
        
        return torch.mean(acts**2)


class DeepDreamObjective(Objective):
    """DeepDream-style objective for maximizing interesting features."""
    
    def __init__(self, layer_name: str, batch: Optional[int] = None):
        self.layer_name = layer_name
        self.batch = batch
        super().__init__(f"deepdream({layer_name})")
    
    def __call__(self, layer_activations):
        if self.layer_name not in layer_activations:
            raise KeyError(f"Layer '{self.layer_name}' not found. Available layers: {list(layer_activations.keys())}")
        
        acts = layer_activations[self.layer_name]
        acts = _handle_batch(acts, self.batch)
        
        # DeepDream objective: maximize squared activations
        return torch.mean(acts**2)


class TotalVariationObjective(Objective):
    """Total variation regularization objective."""
    
    def __init__(self, layer_name: str = "input", beta: float = 2.0):
        self.layer_name = layer_name
        self.beta = beta
        super().__init__(f"total_variation({layer_name}, beta={beta})")
    
    def __call__(self, layer_activations):
        if self.layer_name not in layer_activations:
            raise KeyError(f"Layer '{self.layer_name}' not found. Available layers: {list(layer_activations.keys())}")
        
        x = layer_activations[self.layer_name]
        
        # Calculate total variation
        if len(x.shape) == 4:  # NCHW format
            diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
            diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        else:  # NHWC format
            diff_h = x[:, 1:, :, :] - x[:, :-1, :, :]
            diff_w = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        tv = torch.sum(torch.sqrt(diff_h**2 + diff_w**2 + 1e-8))
        return tv


# Convenience functions for creating objectives
def channel(layer_name: str, channel_idx: int, batch: Optional[int] = None) -> Objective:
    """Create a channel objective."""
    return ChannelObjective(layer_name, channel_idx, batch)


def neuron(layer_name: str, channel_idx: int, x: Optional[int] = None, 
           y: Optional[int] = None, batch: Optional[int] = None) -> Objective:
    """Create a neuron objective."""
    return NeuronObjective(layer_name, channel_idx, x, y, batch)


def layer(layer_name: str, batch: Optional[int] = None) -> Objective:
    """Create a layer objective."""
    return LayerObjective(layer_name, batch)


def deepdream(layer_name: str, batch: Optional[int] = None) -> Objective:
    """Create a DeepDream objective."""
    return DeepDreamObjective(layer_name, batch)


def total_variation(layer_name: str = "input", beta: float = 2.0) -> Objective:
    """Create a total variation regularization objective."""
    return TotalVariationObjective(layer_name, beta)