"""
Gradient manipulation utilities for PyTorch Lucid.

This module provides utilities for modifying gradients during optimization,
which can improve visualization quality and optimization behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
import contextlib


def gradient_override(module: nn.Module, grad_input: torch.Tensor, 
                     grad_output: torch.Tensor) -> torch.Tensor:
    """
    Override gradient computation for a module.
    
    This function can be used to modify gradients during backpropagation,
    which is useful for techniques like gradient clipping or custom
    gradient flows.
    
    Args:
        module: Module to override gradients for
        grad_input: Input gradient
        grad_output: Output gradient
        
    Returns:
        Modified gradient
    """
    # Default implementation: pass through unchanged
    return grad_output


def redirect_relu_grad(module: nn.Module, grad_input: torch.Tensor,
                      grad_output: torch.Tensor) -> torch.Tensor:
    """
    Redirect ReLU gradients to improve optimization.
    
    This technique helps avoid dead neurons during optimization by
    ensuring gradients flow even when activations are zero.
    
    Args:
        module: ReLU module
        grad_input: Input gradient
        grad_output: Output gradient
        
    Returns:
        Modified gradient with redirected flow
    """
    # For ReLU, normally grad_input would be zero where input < 0
    # We redirect some gradient to flow through anyway
    if hasattr(module, 'last_input'):
        input_tensor = module.last_input
        # Create mask where we want to redirect gradient
        mask = (input_tensor < 0).float() * 0.1  # Small gradient for negative inputs
        redirected_grad = grad_output * (1 + mask)
        return redirected_grad
    return grad_output


class GradientOverrideContext:
    """
    Context manager for temporarily overriding gradients.
    
    Usage:
        with GradientOverrideContext(model, override_fn):
            # Perform operations with overridden gradients
            pass
    """
    
    def __init__(self, model: nn.Module, override_fn: Callable):
        self.model = model
        self.override_fn = override_fn
        self.hooks = []
    
    def __enter__(self):
        """Register gradient override hooks."""
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                hook = module.register_backward_hook(self.override_fn)
                self.hooks.append(hook)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove gradient override hooks."""
        for hook in self.hooks:
            hook.remove()


class RedirectReLUContext(GradientOverrideContext):
    """
    Context manager for redirecting ReLU gradients.
    
    This is a specialized version of GradientOverrideContext specifically
    for redirecting ReLU gradients to improve optimization.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__(model, redirect_relu_grad)
        
        # Store inputs for gradient redirection
        def store_input_hook(module, input, output):
            module.last_input = input[0].detach()
        
        self.input_hooks = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(store_input_hook)
                self.input_hooks.append(hook)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks and clean up."""
        super().__exit__(exc_type, exc_val, exc_tb)
        
        # Remove input storage hooks
        for hook in self.input_hooks:
            hook.remove()
        
        # Clean up stored inputs
        for module in self.model.modules():
            if hasattr(module, 'last_input'):
                delattr(module, 'last_input')


@contextlib.contextmanager
def gradient_override_map(model: nn.Module, override_dict: Dict[type, Callable]):
    """
    Context manager for overriding gradients based on module type.
    
    Args:
        model: Model to override gradients for
        override_dict: Dictionary mapping module types to override functions
        
    Example:
        with gradient_override_map(model, {nn.ReLU: redirect_relu_grad}):
            # Perform operations with overridden ReLU gradients
            pass
    """
    hooks = []
    
    def create_override_hook(override_fn):
        def override_hook(module, grad_input, grad_output):
            return override_fn(module, grad_input, grad_output)
        return override_hook
    
    try:
        # Register hooks
        for module_type, override_fn in override_dict.items():
            for module in model.modules():
                if isinstance(module, module_type):
                    hook = module.register_backward_hook(create_override_hook(override_fn))
                    hooks.append(hook)
        
        yield
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> None:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def clamp_gradients(model: nn.Module, min_val: float = -1.0, max_val: float = 1.0) -> None:
    """
    Clamp individual gradient values to a range.
    
    Args:
        model: Model to clamp gradients for
        min_val: Minimum gradient value
        max_val: Maximum gradient value
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(min_val, max_val)


class GradientNormalizer:
    """
    Utility class for normalizing gradients during optimization.
    
    This can help stabilize training and prevent gradient explosion.
    """
    
    def __init__(self, method: str = 'norm', value: float = 1.0):
        """
        Initialize gradient normalizer.
        
        Args:
            method: Normalization method ('norm', 'clamp', 'scale')
            value: Value to use for normalization (max norm, clamp range, or scale factor)
        """
        self.method = method
        self.value = value
    
    def __call__(self, model: nn.Module) -> None:
        """Apply gradient normalization to model."""
        if self.method == 'norm':
            clip_gradients(model, self.value)
        elif self.method == 'clamp':
            clamp_range = self.value
            clamp_gradients(model, -clamp_range, clamp_range)
        elif self.method == 'scale':
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(self.value)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


def create_gradient_hook(gradient_fn: Callable) -> Callable:
    """
    Create a gradient hook function that can be registered with modules.
    
    Args:
        gradient_fn: Function to apply to gradients
        
    Returns:
        Hook function that can be registered with register_backward_hook
    """
    def hook(module, grad_input, grad_output):
        return gradient_fn(module, grad_input, grad_output)
    return hook