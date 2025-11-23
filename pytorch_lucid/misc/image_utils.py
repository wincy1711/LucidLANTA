"""
Image processing utilities for PyTorch Lucid.

This module provides functions for preprocessing and postprocessing images
for use with neural networks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import torchvision.transforms as transforms


def preprocess_image(image: Union[torch.Tensor, np.ndarray, Image.Image],
                    target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = True) -> torch.Tensor:
    """
    Preprocess an image for neural network input.
    
    Args:
        image: Input image (tensor, numpy array, or PIL Image)
        target_size: Target size (height, width) for resizing
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to PIL Image if needed
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)  # CHW to HWC
        image = image.detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        # Convert numpy to PIL
        if len(image.shape) == 4:
            image = image.squeeze(0)
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if requested
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert to tensor
    tensor = transforms.ToTensor()(image)
    
    # Normalize if requested
    if normalize:
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(tensor)
    
    return tensor.unsqueeze(0)  # Add batch dimension


def postprocess_image(tensor: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
    """
    Postprocess a network output tensor to a displayable image.
    
    Args:
        tensor: Output tensor from network
        denormalize: Whether to reverse ImageNet normalization
        
    Returns:
        Postprocessed image tensor in [0, 1] range
    """
    # Handle different tensor shapes
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    if len(tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor (CHW), got {tensor.shape}")
    
    # Denormalize if requested
    if denormalize:
        from .io import denormalize_image
        tensor = denormalize_image(tensor)
    
    # Ensure values are in [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def resize_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize an image tensor.
    
    Args:
        image: Image tensor
        size: Target size (height, width)
        
    Returns:
        Resized image tensor
    """
    if len(image.shape) == 4:
        # Batch dimension present
        return F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    elif len(image.shape) == 3:
        # No batch dimension
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        return image.squeeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(image.shape)}D")


def crop_image(image: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """
    Crop an image tensor.
    
    Args:
        image: Image tensor
        top: Top coordinate
        left: Left coordinate
        height: Crop height
        width: Crop width
        
    Returns:
        Cropped image tensor
    """
    if len(image.shape) == 4:
        # Batch dimension present
        return image[:, :, top:top + height, left:left + width]
    elif len(image.shape) == 3:
        # No batch dimension
        return image[:, top:top + height, left:left + width]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(image.shape)}D")


def pad_image(image: torch.Tensor, padding: Union[int, Tuple[int, int, int, int]],
             mode: str = 'reflect') -> torch.Tensor:
    """
    Pad an image tensor.
    
    Args:
        image: Image tensor
        padding: Padding size (int for all sides, or (left, right, top, bottom))
        mode: Padding mode ('reflect', 'replicate', 'constant')
        
    Returns:
        Padded image tensor
    """
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[1], padding[1], padding[0], padding[0])  # (left, right, top, bottom)
    elif len(padding) != 4:
        raise ValueError("Padding must be int, (height, width), or (left, right, top, bottom)")
    
    return F.pad(image, padding, mode=mode)


def apply_jitter(image: torch.Tensor, jitter: int) -> torch.Tensor:
    """
    Apply random jitter to an image.
    
    Args:
        image: Image tensor
        jitter: Maximum jitter in pixels
        
    Returns:
        Jittered image tensor
    """
    if jitter == 0:
        return image
    
    # Random jitter
    dx = torch.randint(-jitter, jitter + 1, (1,)).item()
    dy = torch.randint(-jitter, jitter + 1, (1,)).item()
    
    if len(image.shape) == 4:
        # Batch dimension present
        return torch.roll(image, shifts=(dy, dx), dims=(2, 3))
    else:
        # No batch dimension
        return torch.roll(image, shifts=(dy, dx), dims=(1, 2))


def random_crop(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Randomly crop an image to the specified size.
    
    Args:
        image: Image tensor
        size: Crop size (height, width)
        
    Returns:
        Cropped image tensor
    """
    if len(image.shape) == 4:
        _, _, height, width = image.shape
    else:
        _, height, width = image.shape
    
    crop_height, crop_width = size
    
    if crop_height > height or crop_width > width:
        raise ValueError(f"Crop size {size} is larger than image size {(height, width)}")
    
    # Random crop coordinates
    top = torch.randint(0, height - crop_height + 1, (1,)).item()
    left = torch.randint(0, width - crop_width + 1, (1,)).item()
    
    return crop_image(image, top, left, crop_height, crop_width)


def center_crop(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crop an image to the specified size.
    
    Args:
        image: Image tensor
        size: Crop size (height, width)
        
    Returns:
        Center cropped image tensor
    """
    if len(image.shape) == 4:
        _, _, height, width = image.shape
    else:
        _, height, width = image.shape
    
    crop_height, crop_width = size
    
    if crop_height > height or crop_width > width:
        raise ValueError(f"Crop size {size} is larger than image size {(height, width)}")
    
    # Center crop coordinates
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    
    return crop_image(image, top, left, crop_height, crop_width)


def random_horizontal_flip(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Randomly flip an image horizontally.
    
    Args:
        image: Image tensor
        p: Probability of flipping
        
    Returns:
        Possibly flipped image tensor
    """
    if torch.rand(1).item() < p:
        if len(image.shape) == 4:
            return torch.flip(image, dims=[3])  # Flip width dimension
        else:
            return torch.flip(image, dims=[2])  # Flip width dimension
    return image


def color_jitter(image: torch.Tensor, brightness: float = 0.0, contrast: float = 0.0,
                saturation: float = 0.0, hue: float = 0.0) -> torch.Tensor:
    """
    Apply color jittering to an image.
    
    Args:
        image: Image tensor
        brightness: Brightness jitter strength
        contrast: Contrast jitter strength
        saturation: Saturation jitter strength
        hue: Hue jitter strength
        
    Returns:
        Color jittered image tensor
    """
    # Convert to PIL for torchvision transforms
    if len(image.shape) == 4:
        image = image.squeeze(0)  # Remove batch dimension
    
    # Convert to PIL
    img_array = image.detach().cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array.transpose(1, 2, 0)  # CHW to HWC
    pil_image = Image.fromarray(img_array)
    
    # Apply color jitter
    transform = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )
    
    jittered_pil = transform(pil_image)
    
    # Convert back to tensor
    jittered_tensor = transforms.ToTensor()(jittered_pil)
    
    return jittered_tensor.unsqueeze(0)  # Add batch dimension


def gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Image tensor
        kernel_size: Gaussian kernel size
        sigma: Gaussian standard deviation
        
    Returns:
        Blurred image tensor
    """
    # Create Gaussian kernel
    channels = image.shape[1] if len(image.shape) == 4 else image.shape[0]
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size))
    
    # Fill kernel with Gaussian values
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[:, :, x, y] = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Normalize kernel
    kernel = kernel / kernel.sum(dim=(2, 3), keepdim=True)
    
    # Apply convolution
    if len(image.shape) == 4:
        return F.conv2d(image, kernel, groups=channels, padding=center)
    else:
        image = image.unsqueeze(0)
        blurred = F.conv2d(image, kernel, groups=channels, padding=center)
        return blurred.squeeze(0)