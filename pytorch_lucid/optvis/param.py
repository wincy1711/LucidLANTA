"""
Parameterization utilities for optimization-based visualization.

This module provides different ways to parameterize images for optimization,
including direct pixel parameterization and Fourier-space parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


def _rfft2d(x):
    """2D real FFT."""
    return torch.fft.rfft2(x)


def _irfft2d(x, s):
    """Inverse 2D real FFT."""
    return torch.fft.irfft2(x, s=s)


def _fft2d(x):
    """2D complex FFT."""
    return torch.fft.fft2(x)


def _ifft2d(x):
    """Inverse 2D complex FFT."""
    return torch.fft.ifft2(x)


def _roll2d(x, shift):
    """Roll 2D tensor."""
    return torch.roll(x, shifts=shift, dims=(-2, -1))


def _symmetrize_image(x):
    """Symmetrize image in frequency domain."""
    # This ensures the inverse FFT produces a real image
    batch_size, channels, height, width = x.shape
    
    # Create symmetric frequency representation
    x_symm = torch.zeros(batch_size, channels, height, width // 2 + 1, 
                        dtype=torch.complex64, device=x.device)
    
    # Fill in the symmetric parts
    x_symm[:, :, :height//2, :] = x[:, :, :height//2, :width//2 + 1]
    
    return x_symm


class ImageParam(nn.Module):
    """
    Direct pixel parameterization of an image.
    
    This parameterization directly optimizes pixel values, typically with
    some preprocessing to make optimization easier.
    """
    
    def __init__(self, shape: Tuple[int, int, int, int], 
                 decorrelate: bool = True,
                 sigmoid: bool = True,
                 fft: bool = False,
                 alpha: bool = False):
        """
        Initialize image parameterization.
        
        Args:
            shape: (batch_size, channels, height, width)
            decorrelate: Whether to use decorrelated color space
            sigmoid: Whether to apply sigmoid to constrain pixel values
            fft: Whether to parameterize in Fourier space
            alpha: Whether to include alpha channel
        """
        super().__init__()
        self.shape = shape
        self.decorrelate = decorrelate
        self.sigmoid = sigmoid
        self.fft = fft
        self.alpha = alpha
        
        batch_size, channels, height, width = shape
        
        # Adjust channels for alpha
        if alpha:
            param_channels = channels + 1
        else:
            param_channels = channels
            
        # Create parameter tensor
        if fft:
            # Parameterize in Fourier space
            self.param_shape = (batch_size, param_channels, height, width // 2 + 1, 2)
            self.param = nn.Parameter(torch.randn(*self.param_shape) * 0.01)
        else:
            # Direct pixel parameterization
            self.param = nn.Parameter(torch.randn(batch_size, param_channels, height, width) * 0.01)
        
        # Color decorrelation matrix (RGB to decorrelated space)
        if decorrelate and channels == 3:
            self.register_buffer('color_correlation', self._color_correlation_matrix())
    
    def _color_correlation_matrix(self):
        """Create color correlation matrix for decorrelation."""
        # Standard RGB to YUV-like decorrelation
        corr_matrix = torch.tensor([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03]
        ], dtype=torch.float32)
        return corr_matrix
    
    def forward(self) -> torch.Tensor:
        """Generate the parameterized image."""
        if self.fft:
            # Convert from frequency domain to spatial domain
            # param is (batch, channels, height, width//2+1, 2) for real/imag parts
            real_imag = self.param
            complex_fft = torch.complex(real_imag[..., 0], real_imag[..., 1])
            
            # Ensure conjugate symmetry for real output
            height, width_half = complex_fft.shape[-2:]
            full_width = (width_half - 1) * 2
            
            # Create full frequency representation
            full_fft = torch.zeros(*complex_fft.shape[:-1], full_width, 
                                 dtype=torch.complex64, device=complex_fft.device)
            full_fft[..., :width_half] = complex_fft
            
            # Add conjugate symmetry
            for i in range(1, width_half - 1):
                full_fft[..., full_width - i] = torch.conj(complex_fft[..., i])
            
            # Inverse FFT
            image = _irfft2d(full_fft, s=full_fft.shape[-2:])
        else:
            # Direct pixel parameterization
            image = self.param
        
        # Apply sigmoid to constrain to [0,1] if requested
        if self.sigmoid:
            image = torch.sigmoid(image)
        
        # Color decorrelation
        if self.decorrelate and image.shape[1] >= 3:
            # Reshape for matrix multiplication
            batch_size, channels, height, width = image.shape
            pixels = image[:, :3].reshape(batch_size, 3, -1)
            
            # Apply decorrelation
            decorrelated = torch.matmul(self.color_correlation.T, pixels)
            
            # Reshape back and concatenate with alpha if present
            decorrelated = decorrelated.reshape(batch_size, 3, height, width)
            
            if self.alpha and image.shape[1] > 3:
                alpha = image[:, 3:]
                image = torch.cat([decorrelated, alpha], dim=1)
            else:
                image = decorrelated
        
        return image


class FFTImageParam(nn.Module):
    """
    Fourier-space parameterization of an image.
    
    This parameterization optimizes in the frequency domain, which can lead
    to more natural-looking images and better optimization behavior.
    """
    
    def __init__(self, shape: Tuple[int, int, int, int], 
                 decay_power: float = 1.0):
        """
        Initialize FFT parameterization.
        
        Args:
            shape: (batch_size, channels, height, width)
            decay_power: Power for frequency decay scaling
        """
        super().__init__()
        self.shape = shape
        self.decay_power = decay_power
        
        batch_size, channels, height, width = shape
        
        # Parameterize the frequency spectrum
        # Only need to parameterize half the spectrum due to symmetry
        self.freq_shape = (batch_size, channels, height, width // 2 + 1)
        
        # Real and imaginary parts for complex frequency coefficients
        self.freq_real = nn.Parameter(torch.randn(*self.freq_shape) * 0.01)
        self.freq_imag = nn.Parameter(torch.randn(*self.freq_shape) * 0.01)
        
        # Frequency scaling for natural image statistics
        self._create_frequency_scales(height, width)
    
    def _create_frequency_scales(self, height: int, width: int):
        """Create frequency scaling factors."""
        # Create frequency grid
        freqs_h = torch.fft.fftfreq(height)
        freqs_w = torch.fft.rfftfreq(width)
        
        # Compute frequency magnitudes
        freq_magnitudes = torch.sqrt(freqs_h[:, None]**2 + freqs_w[None, :]**2)
        
        # Apply decay scaling
        self.register_buffer('frequency_scales', 
                           freq_magnitudes ** self.decay_power + 1e-8)
    
    def forward(self) -> torch.Tensor:
        """Generate the parameterized image."""
        # Combine real and imaginary parts
        freq_spectrum = torch.complex(self.freq_real, self.freq_imag)
        
        # Apply frequency scaling
        freq_spectrum = freq_spectrum / self.frequency_scales
        
        # Create full frequency spectrum with conjugate symmetry
        batch_size, channels, height, width_half = freq_spectrum.shape
        width = (width_half - 1) * 2
        
        full_spectrum = torch.zeros(batch_size, channels, height, width, 
                                  dtype=torch.complex64, device=freq_spectrum.device)
        full_spectrum[:, :, :, :width_half] = freq_spectrum
        
        # Add conjugate symmetry for real output
        for i in range(1, width_half - 1):
            full_spectrum[:, :, :, width - i] = torch.conj(freq_spectrum[:, :, :, i])
        
        # Inverse FFT to get spatial domain image
        image = torch.fft.ifft2(full_spectrum).real
        
        # Normalize to reasonable range
        image = torch.sigmoid(image * 0.5)
        
        return image


def image(shape: Tuple[int, int, int, int], 
          decorrelate: bool = True,
          sigmoid: bool = True,
          fft: bool = False,
          alpha: bool = False) -> nn.Module:
    """
    Create direct pixel parameterization.
    
    Args:
        shape: (batch_size, channels, height, width)
        decorrelate: Whether to use decorrelated color space
        sigmoid: Whether to apply sigmoid to constrain pixel values
        fft: Whether to parameterize in Fourier space (uses FFTImageParam)
        alpha: Whether to include alpha channel
        
    Returns:
        Parameterized image module
    """
    if fft:
        return FFTImageParam(shape, decay_power=1.0)
    else:
        return ImageParam(shape, decorrelate, sigmoid, fft, alpha)


def fft_image(shape: Tuple[int, int, int, int], 
              decay_power: float = 1.0) -> nn.Module:
    """
    Create Fourier-space parameterization.
    
    Args:
        shape: (batch_size, channels, height, width)
        decay_power: Power for frequency decay scaling
        
    Returns:
        Parameterized image module using FFT
    """
    return FFTImageParam(shape, decay_power)