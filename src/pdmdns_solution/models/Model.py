"""
Author: Arnaud Yarga

Main model implementation for neuromorphic speech denoising.

This module contains the primary Model class that orchestrates the entire
speech denoising pipeline, including data preprocessing, model inference,
training, validation, and evaluation with comprehensive logging.
"""

import os
import random
import numpy as np
from numpy import linspace
import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from .registration import ModelBank
from argparse import ArgumentParser

import torchaudio
import torchaudio.transforms as Taudio
import matplotlib.pyplot as plt

from .networks import DenoisingNet
from utils.loss import TemporalSpectralLoss, MOSMetric
from utils.utils import PDMEncodeur, SpectralTransformer, rescale_waveform, get_rms, set_rms
import gc
import time


class Model(LightningModule):
    """
    Main model class for neuromorphic speech denoising.
    
    This class implements a complete speech denoising pipeline using PyTorch Lightning,
    including data preprocessing, model inference, training, validation, and evaluation.
    It supports both spiking and non-spiking neural networks with comprehensive logging.
    """
    
    def __init__(self, mode, n_chan, archi="DenoisingNet", pdm_oversampling=0, dropout=0.,
                 lr=1e-3, optimizer="Adam", scheduler="None", not_spiking=False, 
                 neuron="ParaLIF-T", tau_mem=1e-3, tau_syn=1e-3, spk_threshold=1., multi_threshold=False,
                 multi_time_constant=False, population=1, loss_args={}, archi_args={},
                 check_grads_strategy=None, quick_eval=False, **_):
        """
        Initialize the Model.
        
        Args:
            mode: Spectral transformation mode ('stft' or 'dct')
            n_chan: Number of frequency channels
            archi: Architecture type (currently only 'DenoisingNet' supported)
            pdm_oversampling: PDM oversampling factor for high-resolution processing
            dropout: Dropout rate for regularization
            lr: Learning rate
            optimizer: Optimizer type ('Adam', 'Adamax', 'Adagrad', 'AdamW')
            scheduler: Learning rate scheduler type
            not_spiking: Whether to use non-spiking neurons (ReLU instead of LIF/ParaLIF)
            neuron: Neuron type ('ParaLIF-T', 'LIF', etc.)
            tau_mem: Membrane time constant for spiking neurons
            tau_syn: Synaptic time constant for spiking neurons
            spk_threshold: Spiking threshold
            multi_threshold: Whether to use multiple thresholds per neuron
            multi_time_constant: Whether to use multiple time constants per neuron
            population: Population size for population encoding
            loss_args: Loss function configuration dictionary
            archi_args: Architecture-specific configuration dictionary
            check_grads_strategy: Strategy for handling problematic gradients
            quick_eval: Whether to skip expensive metrics for faster evaluation
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration parameters
        self.mode = mode
        self.n_chan = n_chan
        self.archi = archi
        self.pdm_oversampling = pdm_oversampling
        self.dropout = dropout
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fs = 16000  # Sampling frequency in Hz
        self.not_spiking = not_spiking
        self.multi_threshold = multi_threshold
        self.multi_time_constant = multi_time_constant
        self.archi_args = archi_args
        
        # Configure neuron parameters
        self.neuron_args = {
            'not_spiking': not_spiking,
            'neuron': neuron,
            'recurrent': False,
            'learn_threshold': True,
            'tau_mem': tau_mem,
            'tau_syn': tau_syn,
            'spk_threshold': spk_threshold,
            'learn_tau': False,
            'surrogate_mode': "atan",
            'multi_threshold': multi_threshold,
            'multi_time_constant': multi_time_constant,
            'population': population
        }
        
        # Logging configuration
        self.log_spike_rate = not not_spiking
        self.log_synops = True
        self.log_weights_grads = False
        self.log_image_audio = True
        self.check_grads_strategy = check_grads_strategy
        self.quick_eval = quick_eval

        # Set up upsampling factor
        ups_ = 1
        if self.pdm_oversampling > 0: 
            self.pdm_encodeur = PDMEncodeur(pdm_oversampling=self.pdm_oversampling)
            ups_ = self.pdm_oversampling
            
        # Initialize spectral transformer for time-frequency conversion
        self.spectral_transformer = SpectralTransformer(mode=self.mode, n_chan=self.n_chan, upsample=ups_)
        
        # Build the neural network architecture
        if self.archi == "DenoisingNet":
            # Extract architecture-specific parameters
            delay_max = self.archi_args.get("delay_max", 0.5)
            ks = self.archi_args.get("ks", 12)
            ks1 = self.archi_args.get("ks1", 6)
            n_hidden = self.archi_args.get("n_hidden", 64)
            n_embedding = self.archi_args.get("n_embedding", 16)
            plus = self.archi_args.get("plus", 16)
            nb_blocks = self.archi_args.get("nb_blocks", 6)
            n_out = self.archi_args.get("n_out", 256)
            dilation = self.archi_args.get("dilation", 2)
            add_interframe = self.archi_args.get("add_interframe", True)
            causal = self.archi_args.get("causal", True)
            n_out_interframe = self.archi_args.get("n_out_interframe", None)
            groups = self.archi_args.get("groups", 2)
            ks_interframe = self.archi_args.get("ks_interframe", 12)
            no_sub_block = self.archi_args.get("no_sub_block", False)
            
            # Convert delay_max from milliseconds to samples
            # TODO: check if pdm oversampling is 0
            delay_max = int(delay_max * (self.fs / 1000) * self.pdm_oversampling)
            
            # Create the DenoisingNet model
            self.model = DenoisingNet(self.spectral_transformer.out_features, n_embedding,
                                     nb_blocks=nb_blocks, dropRate=self.dropout, neuron_args=self.neuron_args, 
                                     pdm_oversampling=self.pdm_oversampling, ks=ks, ks1=ks1, 
                                     n_hidden=n_hidden, plus=plus, delay_max=delay_max, n_out=n_out, 
                                     dilation=dilation, add_interframe=add_interframe, causal=causal,
                                     n_out_interframe=n_out_interframe, groups=groups, 
                                     ks_interframe=ks_interframe, no_sub_block=no_sub_block)
        else:
            raise Exception(f"Model '{self.archi}' is not defined.")
            
        print(self.model)
        
        # Initialize audio quality metrics
        self.si_snr_train = ScaleInvariantSignalNoiseRatio()
        self.si_snr_val = ScaleInvariantSignalNoiseRatio()
        self.si_snr_noisy = ScaleInvariantSignalNoiseRatio()
        self.si_snr_test = ScaleInvariantSignalNoiseRatio()
        self.stoi_val = ShortTimeObjectiveIntelligibility(self.fs, False)
        self.stoi_test = ShortTimeObjectiveIntelligibility(self.fs, False)
        self.spectrogram = Taudio.Spectrogram(n_fft=512, hop_length=128, center=False)
        self.pesq_test = PerceptualEvaluationSpeechQuality(self.fs, 'wb')
        self.mos_test = MOSMetric(sampling_rate=self.fs)
        
        # Initialize loss function
        self.loss_fn = TemporalSpectralLoss(loss_args)
        self.regularize_factor = loss_args.get("regularize_factor", 0.)

        # Tracking variables
        self.last_res = []
        self.MAX_SI_SNR = 30
        self.MAX_STOI = 1.
        self.MAX_PESQ = 4.5
        self.last_si_snr_val = 0
        self.last_stoi_val = 0
        self.example_input_array = torch.rand(10, 16000)

    def forward(self, noisy):
        """
        Forward pass through the speech denoising pipeline.
        
        Args:
            noisy: Noisy speech signal tensor
            
        Returns:
            Denoised speech signal tensor
        """
        # Get RMS of noisy signal for later restoration
        noisy_rms = get_rms(noisy)
        
        # Rescale waveform to normalized range
        noisy = rescale_waveform(noisy)
        
        # Apply PDM encoding if oversampling is enabled
        if self.pdm_oversampling > 0: 
            noisy = self.pdm_encodeur(noisy)
        
        # Convert time-domain signal to frequency-domain frames
        noisy_frames = self.spectral_transformer.splitter(noisy)
        
        # Process through the neural network (add channel dimension and permute)
        denoised_frames = self.model(noisy_frames.permute(0, 2, 1).unsqueeze(1)).squeeze(1).permute(0, 2, 1)
        
        # Convert back to time domain
        denoised = self.spectral_transformer.mixer(denoised_frames)
        
        # Apply tanh activation to constrain output range
        denoised = F.tanh(denoised)

        # Restore original RMS level
        denoised = set_rms(denoised, noisy_rms)
        
        return denoised

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer configuration dictionary
        """
        # Select optimizer based on configuration
        if self.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise Exception(f"Optimizer: '{self.optimizer}' is not implemented")
        
        frequency = 1.
        
        # Configure learning rate scheduler
        if self.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        elif self.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        elif self.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        elif self.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, threshold=0.011)
            frequency = 5.
        else:
            return optimizer
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': frequency, "monitor": "val_si_snr"}
        }

    def common_step(self, batch, si_snr_metric, stoi_metric=None, si_snr_noisy_metric=None, validation=False):
        """
        Common processing step for training, validation, and testing.
        
        Args:
            batch: Input batch containing audio data
            si_snr_metric: SI-SNR metric for evaluation
            stoi_metric: STOI metric for evaluation (optional)
            si_snr_noisy_metric: SI-SNR metric for noisy signal (optional)
            validation: Whether this is a validation step
            
        Returns:
            Tuple of (loss, denoised_audio)
        """
        # Unpack batch data
        if len(batch) == 4:
            clean, noise, noisy, _ = batch
        else:
            noisy, clean, noise = batch

        # Generate denoised output
        denoised = self(noisy)
        
        # Calculate SI-SNR metric
        si_snr_metric(denoised, clean)
        
        # Calculate loss
        loss = self.loss_fn(denoised, clean, noisy, noise)
        
        # Add regularization term if specified
        if self.regularize_factor > 0:
            loss += self.model.get_spike_sum().mean() * self.regularize_factor
            
        # Calculate additional metrics if provided
        if stoi_metric is not None: 
            stoi_metric(denoised, clean)
        if si_snr_noisy_metric is not None: 
            si_snr_noisy_metric(clean, noisy)
        
        # Store validation samples for visualization
        if validation and self.log_image_audio:
            self.validation_clean = clean
            self.validation_noisy = noisy
            self.validation_denoised = denoised
            
        # Clear GPU cache
        torch.cuda.empty_cache()

        return loss, denoised

    def on_train_epoch_start(self):
        self.training_start = time.time()
        
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        loss, _ = self.common_step(batch, self.si_snr_train)
        
        # Log training metrics
        self.log_dict(
            {
                "train_loss": loss,
                "train_si_snr": self.si_snr_train
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return loss
    
    def on_train_epoch_end(self):
        duration = time.time() - self.training_start
        self.log('training_duration', duration, on_epoch=True, prog_bar=False)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        loss, denoised = self.common_step(batch, self.si_snr_val, self.stoi_val, self.si_snr_noisy, validation=True)
        
        # Log validation metrics
        self.log_dict(
            {
                "val_loss": loss,
                "val_si_snr": self.si_snr_val,
                "val_si_snr_noisy": self.si_snr_noisy,
                "val_si_snr_improve": self.si_snr_val.compute() - self.si_snr_noisy.compute(),
                "val_stoi": self.stoi_val,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        # Log spike rates if enabled
        if self.log_spike_rate and getattr(self.model, 'log_spike_rate', False):
            self.log_dict(
                self.model.log_spike_rate(),
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )

    def test_step(self, batch, batch_idx, return_denoised=False):
        """
        Test step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            return_denoised: Whether to return denoised audio
            
        Returns:
            Denoised audio if return_denoised=True, otherwise None
        """
        # Unpack batch data
        if len(batch) == 4:
            clean, noise, noisy, _ = batch
        else:
            noisy, clean, noise = batch

        # Generate denoised output
        denoised = self(noisy)

        # Calculate basic metrics
        self.si_snr_test(denoised, clean)
        self.stoi_test(denoised, clean)
        
        # Calculate expensive metrics only if not in quick evaluation mode
        if not self.quick_eval:
            self.pesq_test(denoised, clean)
            self.mos_test(denoised)
            self.log('test_pesq', self.pesq_test, on_epoch=True, prog_bar=True)
        
        # Log test metrics
        self.log_dict(
            {
                "test_si_snr": self.si_snr_test,
                "test_stoi": self.stoi_test,
            },
            on_epoch=True,
            prog_bar=True
        )
        
        # Log spike rates if enabled
        if self.log_spike_rate and getattr(self.model, 'log_spike_rate', False):
            self.log_dict(
                self.model.log_spike_rate(),
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )
            
        # Log synaptic operations if enabled
        if self.log_synops:
            self.log_dict(
                {f"synops/{i}": synops for (i, synops) in enumerate(self.model.get_synops())},
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )
            
        if return_denoised: 
            return denoised
        
    def on_test_epoch_end(self):
        """Called at the end of test epoch - computes final metrics and logs results."""
        # Calculate combined metric based on available metrics
        combine_metric_test = [self.si_snr_test.compute().detach().item() / self.MAX_SI_SNR, 
                              self.stoi_test.compute().detach().item() / self.MAX_STOI]
        
        # Add expensive metrics if not in quick evaluation mode
        if not self.quick_eval:
            combine_metric_test += [self.pesq_test.compute().detach().item() / self.MAX_PESQ]
            
            # Compute and log the MOS score components
            mos_score = self.mos_test.compute()
            self.log('MOS/OVRL', mos_score[0])  # Overall quality
            self.log('MOS/SIG', mos_score[1])   # Signal quality
            self.log('MOS/BAK', mos_score[2])   # Background quality
            
        # Calculate and log the final combined metric
        self.combine_metric_test = torch.tensor(combine_metric_test).mean()
        self.log('test_combine_metric', self.combine_metric_test, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch - computes metrics and creates visualizations."""
        # Calculate combined validation metric
        self.combine_metric_val = torch.tensor([self.si_snr_val.compute().detach().item() / self.MAX_SI_SNR, 
                                               self.stoi_val.compute().detach().item() / self.MAX_STOI]).mean()
        self.log('val_combine_metric', self.combine_metric_val, on_epoch=True, prog_bar=True)
        self.last_res.append(self.combine_metric_val.item()) # for optuna
        if not self.log_image_audio: return 
        
        i = np.random.randint(0,self.validation_denoised.shape[0])
        denoised = self.validation_denoised[i]
        clean = self.validation_clean[i]
        noisy = self.validation_noisy[i]
        timesteps = linspace(0, len(denoised) / self.fs, len(denoised))
        
        # Create figure with spectrograms and waveforms
        fig = plt.figure(figsize=(14, 4))
        
        # Spectrograms
        plt.subplot(2, 3, 1)
        plt.title("Clean")
        plt.pcolormesh((self.spectrogram(clean) + 1e-7).log().cpu())
        plt.xticks([])
        
        plt.subplot(2, 3, 2)
        plt.title(f"Noisy - SISNR: {scale_invariant_signal_noise_ratio(clean, noisy).cpu().item():.2f}")
        plt.pcolormesh((self.spectrogram(noisy) + 1e-7).log().cpu())
        plt.xticks([])
        
        plt.subplot(2, 3, 3)
        plt.title(f"Denoised - SISNR: {scale_invariant_signal_noise_ratio(clean, denoised).cpu().item():.2f}")
        plt.pcolormesh((self.spectrogram(denoised) + 1e-7).log().cpu())
        plt.xticks([])
        
        # Waveforms
        plt.subplot(2, 3, 4)
        plt.plot(timesteps, clean.cpu())
        plt.xlabel('Time (s)')
        
        plt.subplot(2, 3, 5)
        plt.plot(timesteps, noisy.cpu())
        plt.xlabel('Time (s)')
        
        plt.subplot(2, 3, 6)
        plt.plot(timesteps, denoised.cpu())
        plt.xlabel('Time (s)')
        
        # Log figure to tensorboard
        if not self.trainer.sanity_checking:
            self.logger.experiment.add_figure("spectrograme_waveform", fig, self.current_epoch)
        plt.close(fig)
        
        # Save audio files
        audio_dir = self.logger.log_dir
        audio_dir = os.path.join(audio_dir, 'audio_dir')
        os.makedirs(audio_dir, exist_ok=True)
        torchaudio.save(os.path.join(audio_dir, 'clean.wav'), clean.unsqueeze(0).cpu(), self.fs)
        torchaudio.save(os.path.join(audio_dir, 'noisy.wav'), noisy.unsqueeze(0).cpu(), self.fs)
        torchaudio.save(os.path.join(audio_dir, 'denoised.wav'), denoised.unsqueeze(0).cpu(), self.fs)
        
        # Log audio to tensorboard
        if not self.trainer.sanity_checking:
            self.logger.experiment.add_audio('clean', clean.unsqueeze(0).cpu(), sample_rate=self.fs, global_step=self.current_epoch)
            self.logger.experiment.add_audio('noisy', noisy.unsqueeze(0).cpu(), sample_rate=self.fs, global_step=self.current_epoch)
            self.logger.experiment.add_audio('denoised', denoised.unsqueeze(0).cpu(), sample_rate=self.fs, global_step=self.current_epoch)
            
        # Clean up memory
        del clean, noisy, denoised
        torch.cuda.empty_cache(), gc.collect()

    def on_after_backward(self):
        """Called after backward pass - handles gradient checking and logging."""
        # Handle problematic gradients if strategy is specified
        if self.check_grads_strategy:
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isfinite(param.grad).logical_not().any():
                    print(f"[Warning] Problematic gradient detected! - {name}")
                    if self.check_grads_strategy == 'zero':
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    elif self.check_grads_strategy == 'random':
                        param.grad = torch.nan_to_num(param.grad, nan=random.uniform(-1e-6, 1e-6), 
                                                    posinf=random.uniform(0, 1e-6), neginf=random.uniform(-1e-6, 0))
                    elif self.check_grads_strategy == 'skip':
                        self.zero_grad()
                        break
                        
        # Log weights and gradients if enabled
        if not self.log_weights_grads: 
            return
            
        if self.trainer.global_step > 0 and self.trainer.global_step % 100 == 0:
            all_grads = []
            all_weights = []
            
            for name, param in self.named_parameters():
                if param.grad is not None:  # Make sure the parameter has a gradient
                    grads = param.grad
                    if "weight" in name: 
                        all_grads.append(grads.flatten())
                        all_weights.append(param.flatten())
                    if torch.isfinite(grads).logical_not().any() or grads.abs().max() > 10:
                        self.logger.experiment.add_histogram(tag=f"invalid_grad/{name}", values=grads,
                                                           global_step=self.trainer.global_step)
                                                            
            if self.trainer.global_step % 1000 == 0:
                if len(all_grads) > 0: 
                    self.logger.experiment.add_histogram(tag="Grads", values=torch.concat(all_grads), 
                                                       global_step=self.trainer.global_step)
                if len(all_weights) > 0: 
                    self.logger.experiment.add_histogram(tag="Weigths", values=torch.concat(all_weights), 
                                                       global_step=self.trainer.global_step)


def setup_args(parser: ArgumentParser) -> None:
    """
    Setup command line arguments for the Model.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument("--mode", type=str, default='ola', 
                       help="Spectral transformation mode ('stft', 'dct', 'ola')")
    parser.add_argument("--n_chan", type=int, default=512, 
                       help="Number of frequency channels")
    parser.add_argument("--archi", type=str, default='DenoisingNet', 
                       help="Neural network architecture")
    parser.add_argument("--pdm_oversampling", type=int, default=0, 
                       help="PDM oversampling factor")
    parser.add_argument("--dropout", type=float, default=0., 
                       help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adamax", 
                       help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="None", 
                       help="Learning rate scheduler")
    parser.add_argument("--not_spiking", action="store_true", default=False,
                       help="Use non-spiking neurons (ReLU)")
    parser.add_argument("--neuron", type=str, default='ParaLIF-T',
                       help="Neuron type")
    parser.add_argument("--tau_mem", type=float, default=1e-3, 
                       help="Membrane time constant")
    parser.add_argument("--tau_syn", type=float, default=1e-3, 
                       help="Synaptic time constant")
    parser.add_argument("--spk_threshold", type=float, default=1e-1, 
                       help="Spiking threshold")
    parser.add_argument("--multi_threshold", action="store_true", default=False,
                       help="Use multiple thresholds per neuron")
    parser.add_argument("--multi_time_constant", action="store_true", default=False,
                       help="Use multiple time constants per neuron")


# Register the Model in the model bank
ModelBank.register("Model", Model, setup_args)
