"""
Input/Output utilities for images and visualizations.

This module provides functions for loading, saving, and displaying images.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
import os


def load_image(path: str, size: Optional[Tuple[int, int]] = None,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    """
    Load an image from file and convert to tensor.
    
    Args:
        path: Path to the image file
        size: Optional size to resize the image to (height, width)
        device: Device to load the tensor on
        
    Returns:
        Image tensor of shape (1, channels, height, width)
    """
    try:
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Resize if requested
        if size is not None:
            img = img.resize((size[1], size[0]), Image.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor (HWC to CHW)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(device)
        
    except Exception as e:
        raise ValueError(f"Error loading image from {path}: {e}")


def save_image(tensor: torch.Tensor, path: str, 
               denormalize: bool = True) -> None:
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor of shape (1, channels, height, width) or (channels, height, width)
        path: Path to save the image
        denormalize: Whether to denormalize the tensor (assuming ImageNet normalization)
    """
    try:
        # Handle different tensor shapes
        if len(tensor.shape) == 4:
            # Remove batch dimension
            tensor = tensor.squeeze(0)
        
        if len(tensor.shape) != 3:
            raise ValueError(f"Expected 3D tensor (CHW), got {tensor.shape}")
        
        # Denormalize if requested
        if denormalize:
            tensor = denormalize_image(tensor)
        
        # Ensure values are in [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        img_array = tensor.detach().cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert CHW to HWC
        img_array = img_array.transpose(1, 2, 0)
        
        # Handle different channel configurations
        if img_array.shape[2] == 1:
            # Grayscale
            img_array = img_array.squeeze(2)
            img = Image.fromarray(img_array, mode='L')
        elif img_array.shape[2] == 3:
            # RGB
            img = Image.fromarray(img_array, mode='RGB')
        elif img_array.shape[2] == 4:
            # RGBA
            img = Image.fromarray(img_array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save image
        img.save(path)
        
    except Exception as e:
        raise ValueError(f"Error saving image to {path}: {e}")


def show_image(tensor: torch.Tensor, title: Optional[str] = None,
               figsize: Tuple[int, int] = (8, 8), denormalize: bool = True) -> None:
    """
    Display an image tensor using matplotlib.
    
    Args:
        tensor: Image tensor of shape (1, channels, height, width) or (channels, height, width)
        title: Optional title for the plot
        figsize: Figure size (width, height)
        denormalize: Whether to denormalize the tensor
    """
    try:
        # Handle different tensor shapes
        if len(tensor.shape) == 4:
            # Remove batch dimension
            tensor = tensor.squeeze(0)
        
        if len(tensor.shape) != 3:
            raise ValueError(f"Expected 3D tensor (CHW), got {tensor.shape}")
        
        # Denormalize if requested
        if denormalize:
            tensor = denormalize_image(tensor)
        
        # Ensure values are in [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        img_array = tensor.detach().cpu().numpy()
        
        # Convert CHW to HWC
        img_array = img_array.transpose(1, 2, 0)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Handle different channel configurations
        if img_array.shape[2] == 1:
            # Grayscale
            plt.imshow(img_array.squeeze(2), cmap='gray')
        elif img_array.shape[2] == 3:
            # RGB
            plt.imshow(img_array)
        elif img_array.shape[2] == 4:
            # RGBA
            plt.imshow(img_array)
        else:
            raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
        
        if title is not None:
            plt.title(title)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {e}")


def denormalize_image(tensor: torch.Tensor, mean: Optional[torch.Tensor] = None,
                     std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Denormalize an image tensor (reverse ImageNet normalization).
    
    Args:
        tensor: Normalized image tensor
        mean: Mean values for normalization (default: ImageNet mean)
        std: Standard deviation values for normalization (default: ImageNet std)
        
    Returns:
        Denormalized image tensor
    """
    # Default ImageNet normalization values
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    # Handle device
    device = tensor.device
    mean = mean.to(device)
    std = std.to(device)
    
    # Handle different tensor shapes
    if len(tensor.shape) == 4:
        # Batch dimension present
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Denormalize
    denormalized = tensor * std + mean
    
    return denormalized


def normalize_image(tensor: torch.Tensor, mean: Optional[torch.Tensor] = None,
                   std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Normalize an image tensor (ImageNet normalization).
    
    Args:
        tensor: Image tensor with values in [0, 1]
        mean: Mean values for normalization (default: ImageNet mean)
        std: Standard deviation values for normalization (default: ImageNet std)
        
    Returns:
        Normalized image tensor
    """
    # Default ImageNet normalization values
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    # Handle device
    device = tensor.device
    mean = mean.to(device)
    std = std.to(device)
    
    # Handle different tensor shapes
    if len(tensor.shape) == 4:
        # Batch dimension present
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Normalize
    normalized = (tensor - mean) / std
    
    return normalized


def create_image_grid(images: list, grid_size: Optional[Tuple[int, int]] = None,
                     padding: int = 2, normalize: bool = True) -> torch.Tensor:
    """
    Create a grid of images from a list of tensors.
    
    Args:
        images: List of image tensors
        grid_size: Grid size (rows, cols). If None, creates square grid
        padding: Padding between images
        normalize: Whether to normalize images to [0, 1]
        
    Returns:
        Grid image tensor
    """
    from torchvision.utils import make_grid
    
    # Ensure all images have the same shape
    if not images:
        raise ValueError("Empty list of images")
    
    # Handle different input formats
    processed_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        if len(img.shape) == 4:
            img = img.squeeze(0)  # Remove batch dimension
        
        if normalize:
            img = torch.clamp(img, 0, 1)
        
        processed_images.append(img)
    
    # Create grid
    grid = make_grid(processed_images, nrow=grid_size[0] if grid_size else None,
                    padding=padding, normalize=False)
    
    return grid


def save_visualization_grid(images: list, path: str, 
                           grid_size: Optional[Tuple[int, int]] = None,
                           padding: int = 2, **kwargs) -> None:
    """
    Save a grid of visualization images.
    
    Args:
        images: List of image tensors or numpy arrays
        path: Path to save the grid
        grid_size: Grid size (rows, cols)
        padding: Padding between images
        **kwargs: Additional arguments for save_image
    """
    try:
        grid = create_image_grid(images, grid_size, padding, normalize=True)
        save_image(grid, path, denormalize=False, **kwargs)
    except Exception as e:
        print(f"Error saving visualization grid: {e}")