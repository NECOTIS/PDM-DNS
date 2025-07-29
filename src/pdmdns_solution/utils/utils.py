# -*- coding: utf-8 -*-
"""
Author: Arnaud Yarga

This file contains utility functions for neuromorphic speech denoising, including:
- PDMEncodeur: Pulse Density Modulation encoder for audio signal processing
- SpectralTransformer: Handles spectral domain transformations (DCT, STFT, OLA)
- Audio processing utilities: waveform rescaling, RMS calculation and setting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class PDMEncodeur(torch.nn.Module):
    """
    Pulse Density Modulation (PDM) encoder for audio signal processing.
    
    PDM is a modulation technique that converts analog signals into digital
    pulse streams. This encoder is used in neuromorphic systems to convert
    audio waveforms into spike-like representations suitable for spiking
    neural networks.
    
    Attributes:
        pdm_oversampling: Oversampling factor for PDM encoding
        pdm_normalize: Whether to normalize input waveforms
        pdm_negative: Whether to allow {-1, 1} spikes instead of {0, 1}
        upsampler: Upsampling layer for oversampling
        th: Threshold for spike generation
    """
    
    def __init__(self, pdm_oversampling=10, pdm_normalize=True, pdm_negative=True):
        """
        Initialize PDM encoder.
        
        Args:
            pdm_oversampling: Oversampling factor (default: 10)
            pdm_normalize: Normalize input waveforms (default: True)
            pdm_negative: Allow {-1, 1} spikes instead of {0, 1} (default: True)
        """
        super().__init__()
        self.pdm_oversampling = pdm_oversampling
        self.pdm_normalize = pdm_normalize
        self.pdm_negative = pdm_negative
        self.upsampler = nn.Upsample(scale_factor=pdm_oversampling, mode='nearest')
        self.th = 1.  # Threshold for spike generation
        
    def to(self, dest):
        """
        Move the encoder to specified device.
        
        Args:
            dest: Target device (CPU/GPU)
            
        Returns:
            Self for chaining
        """
        super().to(dest)
        self.upsampler = self.upsampler.to(dest)
        return self
    
    def forward(self, waveform):
        """
        Encode audio waveform into PDM spike representation.
        
        The encoding process involves:
        1. Normalization (optional)
        2. Upsampling (if oversampling > 1)
        3. Conversion to PDM format
        4. Spike generation based on cumulative sum
        
        Args:
            waveform: Input audio waveform tensor
            
        Returns:
            PDM spike representation tensor
        """
        # Normalize waveform if requested
        if self.pdm_normalize:
            maxx = waveform.abs().amax(-1).unsqueeze(1) + 1e-7
            waveform = waveform / maxx

        # Apply oversampling if needed
        if self.pdm_oversampling != 1: 
            if len(waveform.shape) == 1: 
                waveform = self.upsampler(waveform[None, None, :]).squeeze(0, 1)
            elif len(waveform.shape) == 2: 
                waveform = self.upsampler(waveform[:, None, :]).squeeze(1)
            else: 
                waveform = self.upsampler(waveform)
                
        # Initialize spike tensor
        spikes = torch.zeros_like(waveform)
        if self.pdm_negative: 
            spikes -= 1  # Initialize with negative spikes if allowed
            
        # Convert to float64 for precision in PDM calculation
        waveform = waveform.to(torch.float64)
        waveform = waveform / 2 + 0.5  # Scale to [0, 1] range
        
        # Generate spikes using cumulative sum method
        waveform_cumsum = torch.cumsum(waveform.to(torch.float64), dim=-1)
        waveform_div = waveform_cumsum // self.th
        waveform_div_diff = torch.diff(waveform_div, n=1, prepend=torch.zeros_like(waveform_div[..., :1]))
        spikes[waveform_div_diff > 0] = 1.  # Generate positive spikes
        
        return spikes


class SpectralTransformer():
    """
    Spectral transformation utility for audio processing.
    
    This class handles various spectral domain transformations including:
    - OLA (Overlap-Add) framing
    
    These transformations are used to convert time-domain audio signals
    into spectral representations suitable for neural network processing.
    
    Attributes:
        mode: Transformation mode ('ola')
        n_chan: Number of frequency channels
        upsample: Upsampling factor
        in_features: Number of input features
        out_features: Number of output features
        params: Dictionary to store transformation parameters
    """
    
    def __init__(self, mode='ola', n_chan=512, upsample=1):
        """
        Initialize spectral transformer.
        
        Args:
            mode: Transformation mode ('dct', 'stft', 'ola')
            n_chan: Number of frequency channels
            upsample: Upsampling factor
        """
        self.mode = mode
        self.n_chan = n_chan
        self.upsample = upsample
        
        # Calculate input and output feature dimensions for each mode
        in_features = {
            "ola": self.n_chan * self.upsample
        }
        out_features = {
            "ola": self.n_chan
        }
        self.in_features = in_features.get(self.mode, 1)
        self.out_features = out_features.get(self.mode, 1)
        self.params = {}  # Store transformation parameters
    
    def splitter(self, inputs):
        """
        Split audio
        
        Args:
            inputs: Input audio tensor
            
        Returns:
            Spectral frames tensor
        """
        if self.mode == 'ola':
            return self.framing(inputs, length=self.n_chan * self.upsample)
        return inputs.unsqueeze(1)
        
    def mixer(self, inputs):
        """
        Reconstruct audio
        
        Args:
            inputs: Spectral frames tensor
            
        Returns:
            Reconstructed audio tensor
        """
        if self.mode == 'ola':
            return self.unframing(inputs, length=self.n_chan, 
                                out_len=int(self.params['audio.shape'][-1] / self.upsample))
        return inputs.squeeze(1)
        
    def framing(self, waveform, length, overlapp=0.5, window_mode="None"):
        """
        Frame audio signal using overlap-add method.
        
        Args:
            waveform: Input audio waveform
            length: Frame length
            overlapp: Overlap ratio between frames
            window_mode: Window application mode
            
        Returns:
            Framed audio tensor
        """
        self.params['audio.shape'] = waveform.shape
        stride = int(length * (1 - overlapp))
        
        with torch.no_grad():
            # Pad waveform to ensure complete frames
            pad = stride - (waveform.shape[-1] - length) % stride
            waveform_frames = F.pad(waveform, (0, pad)).unfold(-1, length, stride)
            
            # Apply window if requested
            if window_mode in ["framing", "both"]: 
                waveform_frames = waveform_frames * torch.hann_window(length, device=waveform_frames.device)
                
        return waveform_frames.permute(0, 2, 1)
    
    def unframing(self, waveform_frames, length, overlapp=0.5, window_mode="unframing", out_len=-1):
        """
        Reconstruct audio from frames using overlap-add method.
        
        Args:
            waveform_frames: Framed audio tensor
            length: Frame length
            overlapp: Overlap ratio between frames
            window_mode: Window application mode
            out_len: Output length (if specified)
            
        Returns:
            Reconstructed audio waveform
        """
        waveform_frames = waveform_frames.permute(0, 2, 1)
        stride = int(length * (1 - overlapp))
        
        # Apply window if requested
        if window_mode in ["unframing", "both"]: 
            waveform_frames = waveform_frames * torch.hann_window(length, device=waveform_frames.device)
            
        # Calculate final length and overlap
        final_len = length + stride * (waveform_frames.shape[1] - 1)
        overlapp_n = length - stride
        
        # Create overlap mask for proper reconstruction
        msk = torch.ones_like(waveform_frames[0])
        if window_mode not in ["framing", "unframing", "both"]:
            msk[1:, :overlapp_n] *= 0.5
            msk[:-1, -overlapp_n:] *= 0.5
            
        # Reconstruct using fold operation
        waveform_output = F.fold(
            (waveform_frames * msk).permute(0, 2, 1), 
            (1, final_len), (1, length), stride=(1, stride)
        )
        waveform_output = waveform_output.squeeze(1, 2)
        
        # Truncate to specified output length
        if out_len > 0:
            waveform_output = waveform_output[..., :out_len]
            
        return waveform_output

    def flush(self):
        """
        Clear stored parameters.
        """
        self.params = {}


# Small constant to prevent division by zero
epsilon = 1e-8


def rescale_waveform(waveform):
    """
    Rescale waveform to unit amplitude.
    
    Normalizes the waveform by dividing by its maximum absolute value
    to ensure the amplitude is in the range [-1, 1].
    
    Args:
        waveform: Input audio waveform
        
    Returns:
        Rescaled waveform with unit amplitude
        
    Raises:
        Warning: If NaN values are detected
    """
    waveform = waveform / (waveform.abs().amax(-1).unsqueeze(1) + epsilon)
    if torch.isnan(waveform).any(): 
        warnings.warn("NAN detected in rescale_waveform function")
    return waveform


def get_rms(x):
    """
    Calculate Root Mean Square (RMS) of audio signal.
    
    RMS is a measure of the average power of the signal.
    
    Args:
        x: Input audio tensor
        
    Returns:
        RMS values with minimum bound to prevent division by zero
    """
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)).clamp_min(epsilon)


def set_rms(x, rms_target):
    """
    Set the RMS of audio signal to a target value.
    
    Scales the audio signal to have the specified RMS value
    while preserving its shape.
    
    Args:
        x: Input audio tensor
        rms_target: Target RMS value
        
    Returns:
        Audio tensor with target RMS value
    """
    origin_rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)).clamp_min(epsilon)
    return x * (rms_target / origin_rms) 

