"""
Image transformations for optimization-based visualization.

This module provides various image transformations that can be applied during
optimization to make visualizations more robust and natural-looking.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from torchvision import transforms
import random


def _interpolate_bilinear(x, y, img):
    """Bilinear interpolation for sampling pixels."""
    # Get image dimensions
    _, _, height, width = img.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * x / (width - 1) - 1.0
    y_norm = 2.0 * y / (height - 1) - 1.0
    
    # Create grid
    grid = torch.stack([x_norm, y_norm], dim=-1)
    
    # Sample using grid_sample
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection')


def _random_scale(img, scale_range=(0.9, 1.1)):
    """Randomly scale an image."""
    scale = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
    
    # Get current size
    _, _, height, width = img.shape
    
    # Calculate new size
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize
    img_scaled = F.interpolate(img, size=(new_height, new_width), mode='bilinear', align_corners=False)
    
    # Crop or pad to original size
    if scale > 1.0:
        # Crop center
        start_h = (new_height - height) // 2
        start_w = (new_width - width) // 2
        img_scaled = img_scaled[:, :, start_h:start_h + height, start_w:start_w + width]
    else:
        # Pad
        pad_h = (height - new_height) // 2
        pad_w = (width - new_width) // 2
        img_scaled = F.pad(img_scaled, (pad_w, pad_w, pad_h, pad_h))
    
    return img_scaled


def _random_rotate(img, angle_range=(-5, 5)):
    """Randomly rotate an image."""
    angle = torch.rand(1).item() * (angle_range[1] - angle_range[0]) + angle_range[0]
    
    # Get rotation matrix
    theta = torch.deg2rad(torch.tensor(angle))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Create affine transformation matrix
    rotation_matrix = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0]
    ], dtype=torch.float32).unsqueeze(0)
    
    # Create grid and apply transformation
    grid = F.affine_grid(rotation_matrix, img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False)


def _random_jitter(img, max_jitter=16):
    """Apply random jitter (translation) to an image."""
    jitter_h = random.randint(-max_jitter, max_jitter)
    jitter_w = random.randint(-max_jitter, max_jitter)
    
    # Roll the image
    img_jittered = torch.roll(img, shifts=(jitter_h, jitter_w), dims=(-2, -1))
    
    return img_jittered


def _pad_crop(img, pad_size=16):
    """Pad and then crop an image."""
    # Pad the image
    img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Random crop back to original size
    _, _, height, width = img.shape
    top = random.randint(0, pad_size * 2)
    left = random.randint(0, pad_size * 2)
    
    img_cropped = img_padded[:, :, top:top + height, left:left + width]
    
    return img_cropped


def _random_noise(img, noise_std=0.1):
    """Add random noise to an image."""
    noise = torch.randn_like(img) * noise_std
    return img + noise


class Transform:
    """Base class for image transformations."""
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply transformation to image."""
        raise NotImplementedError


class Jitter(Transform):
    """Random jitter transformation."""
    
    def __init__(self, max_jitter: int = 16):
        self.max_jitter = max_jitter
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return _random_jitter(img, self.max_jitter)


class Scale(Transform):
    """Random scale transformation."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return _random_scale(img, self.scale_range)


class Rotate(Transform):
    """Random rotation transformation."""
    
    def __init__(self, angle_range: Tuple[float, float] = (-5, 5)):
        self.angle_range = angle_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return _random_rotate(img, self.angle_range)


class PadCrop(Transform):
    """Pad and crop transformation."""
    
    def __init__(self, pad_size: int = 16):
        self.pad_size = pad_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return _pad_crop(img, self.pad_size)


class RandomNoise(Transform):
    """Random noise transformation."""
    
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return _random_noise(img, self.noise_std)


class Compose(Transform):
    """Compose multiple transformations."""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img


def standard_transforms(jitter: int = 16, scale: Tuple[float, float] = (0.9, 1.1), 
                       rotate: Tuple[float, float] = (-5, 5), pad: int = 16,
                       noise: float = 0.0) -> Compose:
    """
    Create a standard set of transformations for feature visualization.
    
    Args:
        jitter: Maximum jitter in pixels
        scale: Range for random scaling
        rotate: Range for random rotation in degrees
        pad: Padding size for pad/crop
        noise: Standard deviation for random noise
        
    Returns:
        Composed transformation
    """
    transforms_list = []
    
    if jitter > 0:
        transforms_list.append(Jitter(jitter))
    
    if scale != (1.0, 1.0):
        transforms_list.append(Scale(scale))
    
    if rotate != (0.0, 0.0):
        transforms_list.append(Rotate(rotate))
    
    if pad > 0:
        transforms_list.append(PadCrop(pad))
    
    if noise > 0:
        transforms_list.append(RandomNoise(noise))
    
    return Compose(transforms_list)