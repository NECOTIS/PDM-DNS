"""
Author: Arnaud Yarga

Evaluation script for neuromorphic speech denoising models.

This script provides functionality to evaluate trained models on the DNS dataset,
including model testing, audio visualization, and metric computation.
"""
import os

# Setup path for imports
exec(open(os.path.join(os.path.dirname(__file__), "..", "utils", "setup_path.py")).read())

import argparse
import pytorch_lightning as pl
import torch
from models.registration import ModelBank
from torch.utils.data import DataLoader
import glob
import numpy as np
from intel_code.audio_dataloader import DNSAudio
from tqdm import tqdm
import torchaudio
import matplotlib.pyplot as plt


def save_audio_batch(noisy_batch, clean_batch, denoised_batch, spectrogram_fn, fs, save_dir=""):
    """
    Save audio examples for an entire batch.
    
    Args:
        noisy_batch: Batch of noisy audio samples
        clean_batch: Batch of clean audio samples
        denoised_batch: Batch of denoised audio samples
        spectrogram_fn: Function to compute spectrograms
        fs: Sampling frequency
        save_dir: Directory to save audio files
    """
    for sample in range(len(clean_batch)):
        denoised = denoised_batch[sample]
        clean = clean_batch[sample]
        noisy = noisy_batch[sample]
        save_audio(sample, noisy, clean, denoised, spectrogram_fn, fs, save_dir=save_dir)

    
def save_audio(sample, noisy, clean, denoised, spectrogram_fn, fs, save_dir=""):
    """
    Save individual audio sample with visualization.
    
    Creates a comprehensive visualization including spectrograms and waveforms
    for clean, noisy, and denoised audio, and saves the audio files.
    
    Args:
        sample: Sample index for naming files
        noisy: Noisy audio signal
        clean: Clean audio signal
        denoised: Denoised audio signal
        spectrogram_fn: Function to compute spectrograms
        fs: Sampling frequency
        save_dir: Directory to save files
    """
    # Create examples directory
    example_dir = os.path.join(save_dir, "examples/")
    os.makedirs(example_dir, exist_ok=True)
    
    # Create visualization if spectrogram function is provided
    if spectrogram_fn is not None:
        # Create time axis for waveform plots
        timesteps = np.linspace(0, len(denoised)/fs, len(denoised))
        
        # Create figure with 2x3 subplots
        fig = plt.figure(figsize=(10, 4))
        
        # Plot spectrograms (top row)
        plt.subplot(2, 3, 1)
        plt.title("Clean")
        plt.pcolormesh((spectrogram_fn(clean) + 1e-7).log().cpu())
        plt.xticks([])
        
        plt.subplot(2, 3, 2)
        plt.title("Noisy")
        plt.pcolormesh((spectrogram_fn(noisy) + 1e-7).log().cpu())
        plt.xticks([])
        
        plt.subplot(2, 3, 3)
        plt.title("Denoised")
        plt.pcolormesh((spectrogram_fn(denoised) + 1e-7).log().cpu())
        plt.xticks([])
        
        # Plot waveforms (bottom row)
        plt.subplot(2, 3, 4)
        plt.plot(timesteps, clean.cpu())
        plt.xlabel('Time (s)')
        
        plt.subplot(2, 3, 5)
        plt.plot(timesteps, noisy.cpu())
        plt.xlabel('Time (s)')
        
        plt.subplot(2, 3, 6)
        plt.plot(timesteps, denoised.cpu())
        plt.xlabel('Time (s)')
        
        # Save figure
        plt.savefig(os.path.join(example_dir, f"{sample}_fig.pdf"))
        plt.close(fig)
    
    # Save audio files
    torchaudio.save(os.path.join(example_dir, f'{sample}_clean.wav'), clean.unsqueeze(0).cpu(), fs)
    torchaudio.save(os.path.join(example_dir, f'{sample}_noisy.wav'), noisy.unsqueeze(0).cpu(), fs)
    torchaudio.save(os.path.join(example_dir, f'{sample}_denoised.wav'), denoised.unsqueeze(0).cpu(), fs)
    
   
def define_config_final(args):
    """
    Apply final configuration based on SLURM ID for different experiment variants.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Modified arguments with experiment-specific configurations
    """
    if args.slurm_id == 0:
        args.experiment_name += "/base"  # Base configuration
    elif args.slurm_id == 1:
        args.tau_mem = 1e-3
        args.experiment_name += "/stateful"  # Stateful configuration
    elif args.slurm_id == 2:
        args.archi_args["causal"] = False
        args.experiment_name += "/non_causal"  # Non-causal configuration
    return args


def define_config(args):
    """
    Define the default configuration for neuromorphic speech denoising evaluation.
    
    Sets up all the default parameters for the model, loss function,
    and architecture configuration, then automatically finds the best checkpoint.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Modified arguments with default configurations and checkpoint path
    """
    # Basic model configuration
    args.mode = 'ola'  # Overlap-add mode for spectral processing
    args.n_chan = 512  # Number of frequency channels
    args.population = 5  # Population size for population encoding
    args.spk_threshold = 1.  # Spiking threshold
    args.neuron = "ParaLIF-T"  # Neuron type
    args.tau_mem = None  # Membrane time constant (None for default)
    args.tau_syn = None  # Synaptic time constant (None for default)
    
    # Loss function configuration
    loss_args = {}
    loss_args["lamda_temporal"] = 0.72  # Temporal loss weight
    loss_args["n_fft_list"] = 512  # FFT size for spectral loss
    loss_args["lambda_magnitude"] = 0.001  # Magnitude loss weight
    loss_args["lambda_phase"] = 0.014  # Phase loss weight
    loss_args["alpha"] = 0.5  # Alpha parameter for loss combination
    args.loss_args = loss_args

    # Architecture configuration
    archi_args = {}
    archi_args["ks1"] = 3  # Kernel size for sub-blocks
    archi_args["delay_max"] = 0.5  # Maximum delay in milliseconds
    archi_args["n_embedding"] = 16  # Number of embedding dimensions
    archi_args["nb_blocks"] = 5  # Number of processing blocks
    archi_args["n_hidden"] = 80  # Number of hidden units
    archi_args["plus"] = 10  # Additional units per block
    archi_args["dilation"] = 4  # Dilation factor
    archi_args["n_out"] = 128  # Number of output channels
    archi_args["ks"] = 7  # Kernel size for main convolutions
    archi_args["groups"] = 1  # Number of groups for grouped convolution
    archi_args["ks_interframe"] = 11  # Kernel size for interframe processing
    args.archi_args = archi_args
    args.dropout = 0.1  # Dropout rate

    # Neuron configuration
    args.multi_threshold = True  # Use one threshold per neuron
    args.multi_time_constant = False  # Use single time constant
    args.optimizer = "Adam"  # Optimizer type
    args.check_grads_strategy = 'random'  # Gradient checking strategy
    
    # Apply final configuration
    args = define_config_final(args)
   
    # Automatically find best checkpoint if not specified
    if args.from_checkpoint is None:
        pattern = f"./lightning_logs/{args.experiment_name}/*/checkpoints/checkpoint_epoch=*.ckpt"
        files = glob.glob(pattern)
        if files:
            # Find checkpoint with highest validation SI-SNR
            val_si_snr = [float(f.split("=")[-1].split("_")[0].split(".ckpt")[0]) for f in files]
            idx = np.argmax(val_si_snr)
            print(f"Checkpoint to be used: {files[idx]}")
            args.from_checkpoint = files[idx]
        else:
            raise Exception(f"Training cant be resumed. No checkpoint at {pattern}")
            
    # Set evaluation experiment name
    args.experiment_name = "eval/" + args.experiment_name
    return args


def main():
    """
    Main evaluation function.
    
    Sets up the complete evaluation pipeline including argument parsing,
    model loading, data preparation, and testing with optional audio saving.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Script to evaluate a model on the DNS dataset.")
    
    # Data and evaluation arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="dataset directory",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--fs", type=int, default=16000, help="Audio sampling frequency")
    parser.add_argument("--data_length", type=int, default=5, help="Audio data length per sample in seconds")
    
    # Model loading arguments
    parser.add_argument("--from_checkpoint", type=str, help="Path to a checkpoint to load from")
    parser.add_argument("--weigth_only", action="store_true", help="Use weight only from checkpoint")
    
    # DNS MOS evaluation arguments
    parser.add_argument(
        "--dns_model_path",
        type=str,
        default="../microsoft_dns/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        help="Path to the DNS MOS Model",
    )
    parser.add_argument(
        "--no_mos",
        action="store_true",
        help="Disable MOS evaluation",
    )
    parser.add_argument(
        "--max_mos_eval",
        type=int,
        default=-1,
        help="Number of batches for MOS test",
    )
    
    # General arguments
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize dataset",
    )
    parser.add_argument("--experiment_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--no_save_audio", action="store_true", help="Disable audio saving")
    parser.add_argument("--slurm_id", type=int, help="Slurm id")
    parser.add_argument("--config", type=str, default="", help="Experiment config")
    parser.add_argument("--quick_eval", action="store_true", help="Enable quick evaluation mode")

    # Add model selection subparser
    subparser = parser.add_subparsers(title="Model selection", required=True, dest="model")
    ModelBank.setup_args(subparser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply configuration if SLURM ID is provided
    if args.slurm_id is not None: 
        args = define_config(args)
    print(args)

    # Set up logger
    logger = None
    name = None if args.experiment_name == "default" else args.experiment_name
    log_root = os.path.join(os.getcwd(), "lightning_logs")
    
    # Try to set up TensorBoard logger
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=log_root, name=name)
    except ImportError:
        pass

    # Get model class and load model
    model_class = ModelBank.get_model_class(args.model)
    
    if args.from_checkpoint is not None and not args.weigth_only:
        # Load complete model state including optimizer state
        model = model_class.load_from_checkpoint(args.from_checkpoint)
        model.quick_eval = args.quick_eval
    else:
        # Create new model
        model = model_class(**vars(args))
        if args.from_checkpoint is not None:
            # Load only weights from checkpoint
            state_dict = torch.load(args.from_checkpoint)["state_dict"]
            model.load_state_dict(state_dict)
        
    # Print model information
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params} params")
    if logger:
        logger.experiment.add_scalar('num_params', n_params)
    print(f"Size of the model: {sum(p.element_size() * p.numel() for p in model.parameters()) // 1024} KB")

    # Set up test dataset and data loader
    testset = DNSAudio(args.dataset_path, normalize=args.normalize)
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=8,
        persistent_workers=True,
        shuffle=False
    )

    # Create PyTorch Lightning trainer for evaluation
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=32,
        val_check_interval=1.0,
        logger=logger,
        enable_checkpointing=False,
        limit_test_batches=args.max_mos_eval if args.max_mos_eval > 0 else 1.0,
        inference_mode=True,  # Enable inference mode for faster evaluation
    )

    # Run evaluation
    trainer.test(model, test_loader)
    
    # Save audio examples if requested
    if not args.no_save_audio:
        print("Save Examples ...")
        with torch.inference_mode():
            device = torch.device("cuda")
            model = model.eval().to(device)

            spectrogram_fn = None  # No spectrogram function for quick evaluation
            nb_examples = 20  # Number of examples to save

            # Process examples one by one
            for i in tqdm(range(nb_examples), desc="Saving:"):
                batch = testset[i]
                
                # Handle different batch formats
                if len(batch) == 4:
                    clean, noise, noisy, _ = batch
                else:
                    noisy, clean, noise = batch
                    
                # Convert to tensors
                noisy, clean, noise = torch.tensor(noisy), torch.tensor(clean), torch.tensor(noise)
                
                # Run inference
                denoised_batch = model(noisy.to(device).unsqueeze(0)).squeeze(0)
                
                # Save audio example
                save_audio(i, noisy, clean, denoised_batch.cpu(), spectrogram_fn=spectrogram_fn, fs=args.fs, 
                           save_dir=logger.log_dir if logger else "")


if __name__ == "__main__":
    main()
