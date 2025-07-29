# -*- coding: utf-8 -*-
"""
Author: Arnaud Yarga

This file contains loss functions and metrics for neuromorphic speech denoising, including:
- wrapped_phase_loss: Phase-aware loss function for spectral domain
- complex_spectral_loss: Multi-scale spectral loss combining magnitude and phase
- TemporalSpectralLoss: Combined temporal and spectral loss for training
- MOSMetric: Mean Opinion Score metric using DNSMOS model
"""

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


def wrapped_phase_loss(clean_phase, denoised_phase):
    """
    Compute wrapped phase loss between clean and denoised signals.
    
    This function handles the circular nature of phase values by computing
    the wrapped difference between phases, ensuring the loss is continuous
    across the phase circle (e.g., difference between 359° and 1° is 2°).
    
    Args:
        clean_phase: Phase of clean signal
        denoised_phase: Phase of denoised signal
        
    Returns:
        L1 loss between wrapped phase differences and zero
    """
    # Compute wrapped phase difference using atan2 for proper circular handling
    phase_diff = torch.atan2(torch.sin(clean_phase - denoised_phase), 
                            torch.cos(clean_phase - denoised_phase))
    return l1_loss(phase_diff, torch.zeros_like(phase_diff))


def complex_spectral_loss(clean_signal, denoised_signal, n_fft_list=[256, 512, 1024], 
                         lambda_magnitude=1.0, lambda_phase=1.0, alpha=0.3):
    """
    Compute multi-scale complex spectral loss.
    
    This loss function evaluates the quality of denoised audio in the spectral domain
    using multiple FFT sizes to capture different temporal resolutions. It combines
    magnitude and phase losses with configurable weights.
    
    Args:
        clean_signal: Clean reference audio signal
        denoised_signal: Denoised audio signal to evaluate
        n_fft_list: List of FFT sizes for multi-scale analysis
        lambda_magnitude: Weight for magnitude loss component
        lambda_phase: Weight for phase loss component
        alpha: Power scaling factor for magnitude (alpha < 1 emphasizes lower magnitudes)
        
    Returns:
        Average spectral loss across all FFT sizes
    """
    epsilon = 1e-6  # Small constant to prevent division by zero
    total_loss = 0
    
    # Compute loss for each FFT size
    for n_fft in n_fft_list:
        # Compute STFT for both signals
        clean_spec = torch.stft(clean_signal, n_fft, 
                               window=torch.hann_window(n_fft, device=clean_signal.device), 
                               return_complex=True, normalized=True)
        denoised_spec = torch.stft(denoised_signal, n_fft, 
                                  window=torch.hann_window(n_fft, device=denoised_signal.device), 
                                  return_complex=True, normalized=True)

        # Extract magnitudes with minimum bound
        clean_mag = clean_spec.abs().clamp_min(epsilon)
        denoised_mag = denoised_spec.abs().clamp_min(epsilon)
        
        # Apply power scaling if alpha != 1 (emphasizes lower magnitudes)
        if alpha != 1:
            clean_mag = clean_mag.pow(alpha)
            denoised_mag = denoised_mag.pow(alpha)

        # Compute magnitude and phase losses
        magnitude_loss = l1_loss(clean_mag, denoised_mag)
        phase_loss = wrapped_phase_loss(torch.angle(clean_spec), torch.angle(denoised_spec))
        
        # Combine losses with weights
        total_loss += lambda_magnitude * magnitude_loss + lambda_phase * phase_loss

    # Return average loss across all FFT sizes
    return total_loss / len(n_fft_list)


class TemporalSpectralLoss(nn.Module):
    """
    Combined temporal and spectral loss for speech denoising training.
    
    This loss function combines:
    1. Scale-Invariant Signal-to-Noise Ratio (SI-SNR) for temporal domain
    2. Multi-scale complex spectral loss for frequency domain
    
    The combination provides a comprehensive evaluation of denoising quality
    in both time and frequency domains.
    
    Attributes:
        lamda_temporal: Weight for temporal loss component
        n_fft_list: List of FFT sizes for spectral analysis
        lambda_magnitude: Weight for magnitude loss in spectral component
        lambda_phase: Weight for phase loss in spectral component
        alpha: Power scaling factor for magnitude
    """
    
    def __init__(self, args):
        """
        Initialize temporal-spectral loss function.
        
        Args:
            args: Dictionary containing loss configuration parameters
        """
        super(TemporalSpectralLoss, self).__init__()
        
        # Extract parameters from args dictionary with defaults
        self.lamda_temporal = args.get("lamda_temporal", 1.0)  # Weight for temporal loss
        self.n_fft_list = args.get("n_fft_list", [256, 512, 1024])  # FFT sizes for spectral analysis
        
        # Ensure n_fft_list is always a list/tuple
        if not isinstance(self.n_fft_list, (list, tuple)):
            self.n_fft_list = [self.n_fft_list]
            
        self.lambda_magnitude = args.get("lambda_magnitude", 0.5)  # Weight for magnitude loss
        self.lambda_phase = args.get("lambda_phase", 0.5)  # Weight for phase loss
        self.alpha = args.get("alpha", 0.2)  # Power scaling factor for magnitude

    def forward(self, denoised_signal, clean_signal, noisy_signal=None, noise_signal=None):
        """
        Compute combined temporal-spectral loss.
        
        Args:
            denoised_signal: Denoised audio signal
            clean_signal: Clean reference audio signal
            noisy_signal: Noisy input signal (unused, kept for interface compatibility)
            noise_signal: Noise signal (unused, kept for interface compatibility)
            
        Returns:
            Combined temporal and spectral loss
        """
        # Compute spectral loss using complex spectral loss function
        spectral_loss = complex_spectral_loss(clean_signal, denoised_signal, 
                                             self.n_fft_list, self.lambda_magnitude, 
                                             self.lambda_phase, self.alpha)
        
        # Compute SI-SNR loss (inverted and normalized)
        # SI-SNR is typically positive, so we use (1 - si_snr/100) to convert to loss
        si_snr_loss = 1 - scale_invariant_signal_noise_ratio(clean_signal, denoised_signal).mean() / 100
        
        # Combine losses with temporal weight
        return self.lamda_temporal * si_snr_loss + spectral_loss


# Import required modules for MOS metric
from concurrent.futures import ThreadPoolExecutor
from intel_code.dnsmos import DNSMOS
from torchmetrics import Metric


class MOSMetric(Metric):
    """
    Mean Opinion Score (MOS) metric using DNSMOS model.
    
    This metric evaluates the perceptual quality of denoised audio using
    the DNSMOS model, which predicts human-like quality scores. The metric
    supports parallel processing for efficient batch evaluation.
    
    Attributes:
        mos: DNSMOS model instance for quality prediction
        score: Accumulated MOS scores (SIG, BAK, OVR)
        nb_samples: Number of processed samples
    """
    
    def __init__(self, dns_model_path="../microsoft_dns/DNSMOS/DNSMOS/sig_bak_ovr.onnx", 
                 sampling_rate=16000):
        """
        Initialize MOS metric.
        
        Args:
            dns_model_path: Path to DNSMOS ONNX model
            sampling_rate: Audio sampling rate (default: 16000 Hz)
        """
        super().__init__()
        
        # Initialize DNSMOS model for quality prediction
        self.mos = DNSMOS(dns_model_path, sampling_rate=sampling_rate)
        
        # Initialize metric states for distributed training
        self.add_state("score", torch.zeros(3), dist_reduce_fx="sum")  # SIG, BAK, OVR scores
        self.add_state("nb_samples", torch.tensor(0), dist_reduce_fx="sum")  # Sample count

    def _process_sample(self, denoised_sample):
        """
        Process a single audio sample and compute its MOS score.
        
        Args:
            denoised_sample: Single denoised audio sample as numpy array
            
        Returns:
            MOS score tensor (SIG, BAK, OVR)
        """
        # Process each sample and get its MOS score
        mos_score = self.mos(denoised_sample)
        return torch.tensor(mos_score, device=self.device)

    def update(self, denoised_batch):
        """
        Update metric with denoised batch (parallel processing).
        
        This method uses ThreadPoolExecutor for parallel processing
        of the batch to improve efficiency.
        
        Args:
            denoised_batch: Batch of denoised audio samples
        """
        # Parallel processing of the batch using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Map the samples in the batch to the processing function
            results = list(executor.map(self._process_sample, denoised_batch.cpu().numpy()))
            
        # Accumulate the results into the score and update sample count
        self.score += torch.stack(results, dim=0).sum(0)
        self.nb_samples += len(results)

    def compute(self):
        """
        Compute the final average MOS score.
        
        Returns:
            Average MOS scores (SIG, BAK, OVR) across all processed samples
        """
        # Compute the final average MOS score
        return self.score / self.nb_samples

