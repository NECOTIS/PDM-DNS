"""
Author: Arnaud Yarga

Training script for neuromorphic speech denoising models.

This script provides a complete training pipeline using PyTorch Lightning for
neuromorphic speech denoising models. It includes configuration management,
data loading, model setup, and comprehensive logging with support for
distributed training and experiment tracking.
"""

import os

# Setup path for imports
exec(open(os.path.join(os.path.dirname(__file__), "..", "utils", "setup_path.py")).read())
import numpy as np
import argparse
import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.environments import SLURMEnvironment
from models.registration import ModelBank
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from intel_code.audio_dataloader import DNSAudio
from utils.audio_dataloader import H5DNSAudio
import models  # Registers all models
import glob


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
    Define the default configuration for neuromorphic speech denoising.
    
    Sets up all the default parameters for the model, loss function,
    and architecture configuration.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Modified arguments with default configurations
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
    
    return define_config_final(args)


def main():
    """
    Main training function.
    
    Sets up the complete training pipeline including argument parsing,
    model configuration, data loading, and PyTorch Lightning trainer setup.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Script to train a model on the subset file of the DNS dataset.")
    
    # Data and training arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="subset file path",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--norm_clip", type=float, default=-1, help="gradient norm-clipping limit, set to negative to disable"
    )
    parser.add_argument("--fs", type=int, default=16000, help="Audio sampling frequency")
    parser.add_argument("--data_length", type=int, default=5, help="Audio data length per sample in seconds")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    
    # Checkpoint and logging arguments
    parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to a checkpoint to load from")
    parser.add_argument("--weigth_only", action="store_true", help="Use weight only from checkpoint")
    parser.add_argument("--no_log", action="store_true", help="Disable logging")
    parser.add_argument("--experiment_name", type=str, default="default", help="Experiment name")
    
    # Training control arguments
    parser.add_argument("--profiler", action="store_true", help="Enable profiler")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable data parallel")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--slurm_id", type=int, help="Slurm id")
    parser.add_argument("--config", type=str, default="", help="Experiment config")
    parser.add_argument("--resume", action="store_true", help="Enable debug mode")
    parser.add_argument("--dataset_seed", type=int, default=27, help="Dataset random seed")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5, help="check_val_every_n_epoch")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early_stopping")
    parser.add_argument("--no_grad_scheduler", action="store_true", help="Disable GradientAccumulationScheduler")

    # Add model selection subparser
    subparser = parser.add_subparsers(title="Model selection", required=True, dest="model")
    ModelBank.setup_args(subparser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply configuration if SLURM ID is provided
    if args.slurm_id is not None: 
        args = define_config(args)
    print(args)

    # Set high precision for matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Get model class from model bank
    model_class = ModelBank.get_model_class(args.model)
    
    # Handle training resumption
    if args.resume:
        pattern = f"./lightning_logs/{args.experiment_name}/*/checkpoints/checkpoint_last_epoch*.ckpt"
        files = glob.glob(pattern)
        if files:
            # Find the latest checkpoint
            epochs = [int(f.split("=")[-1].split("_")[0].split(".ckpt")[0]) for f in files]
            idx = np.argmax(epochs)
            print(f"Checkpoint to be used: {files[idx]}")
            args.from_checkpoint = files[idx]
            args.dataset_seed += epochs[idx]  # change dataset seed to avoid retraining with same data
        else:
            raise Exception(f"Training cant be resumed. No checkpoint at {pattern}")

    # Load model from checkpoint or create new model
    if args.from_checkpoint is not None and not args.weigth_only:
        # Load complete model state including optimizer state
        model = model_class.load_from_checkpoint(args.from_checkpoint, **vars(args))
    else:
        # Create new model
        model = model_class(**vars(args))
        if args.from_checkpoint is not None:
            # Load only weights from checkpoint
            state_dict = torch.load(args.from_checkpoint)["state_dict"]
            model.load_state_dict(state_dict)
            
    # Print model parameter count
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1000}k")
    
    # Set up data loaders based on dataset type
    if args.dataset_path.endswith("h5"):
        # HDF5 dataset format
        trainset = H5DNSAudio(root=args.dataset_path, is_validation=False, 
                             audio_length=args.data_length, max_nb_files=100000, seed=args.dataset_seed)
        valset = H5DNSAudio(root=args.dataset_path, is_validation=True, audio_length=5)
        
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=False,
            persistent_workers=True,
            shuffle=True,
        )
        val_loader = DataLoader(
            valset,
            batch_size=args.batch_size,
            num_workers=8,
            persistent_workers=True,
            shuffle=False,
        )
        kwargs = {
            "max_epochs": args.max_epochs,  # Number of epochs
            "limit_train_batches": 3000,  # Number of batch per epoch
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
        }
    else:
        # Standard dataset format
        valset = DNSAudio(args.dataset_path, normalize=True)
        val_loader = DataLoader(
            valset,
            batch_size=args.batch_size,
            num_workers=8,
            persistent_workers=True,
            shuffle=False,
        )
        kwargs = {}

    # Set up callbacks for training
    callbacks = []
    if not args.debug:
        arr_nb = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        arr_str = f"_{arr_nb}" if arr_nb is not None else ""
        
        callbacks = [
            # Model checkpoint callback - save best models
            ModelCheckpoint(
                save_top_k=2,  # Save top 2 model based on metric below
                monitor="val_si_snr",
                mode="max",
                filename="checkpoint_{epoch:04d}_{val_si_snr:.4f}" + arr_str,
            ),
            # Model checkpoint callback - save last model
            ModelCheckpoint(
                filename="checkpoint_last_{epoch:04d}" + arr_str,
                save_on_train_epoch_end=True
            ),
            # Learning rate monitor
            LearningRateMonitor(),
        ]
        
        # Add early stopping if enabled
        if not args.no_early_stopping:
            callbacks += [
                EarlyStopping(
                    monitor="val_si_snr", 
                    min_delta=0.01, 
                    patience=3, 
                    verbose=True, 
                    mode="max",
                    check_finite=True,
                    divergence_threshold=-10,
                    check_on_train_epoch_end=False
                )
            ]
            
        # Add gradient accumulation scheduler if enabled
        if not args.no_grad_scheduler:
            callbacks += [
                GradientAccumulationScheduler(scheduling={0: 8, 9: 4, 19: 2, 29: 1})
            ]

    # Set up loggers
    loggers = []
    log_root = os.path.join(os.environ.get("SLURM_TMPDIR", os.getcwd()), "lightning_logs")
    name = None if args.experiment_name == "default" else args.experiment_name
    
    if not args.no_log:
        slurm_id = SLURMEnvironment.job_id()
        
        # Try to set up WandB logger
        try:
            from pytorch_lightning.loggers import WandbLogger
            loggers.append(WandbLogger(name=name))
            if slurm_id is not None:
                loggers[-1].experiment.notes = f"Slurm ID: {SLURMEnvironment.job_id()}"
        except ImportError:
            pass

        # Try to set up TensorBoard logger
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            if slurm_id is not None:
                version = str(slurm_id) + arr_str
            else:
                version = None
            loggers.append(TensorBoardLogger(save_dir=log_root, version=version, name=name, default_hp_metric=False))
        except ImportError:
            pass
            
    # Log model information
    loggers[-1].experiment.add_text("Model", str(model))
    loggers[-1].experiment.add_text("ModelSummary", str(ModelSummary(model, max_depth=-1)))
    loggers[-1].experiment.add_scalar('num_params', sum(p.numel() for p in model.parameters()))
    
    # Configure debug mode settings
    if args.debug:
        kwargs = {
            "max_epochs": args.max_epochs,  # Number of epochs
            "limit_train_batches": 1000,  # Number of batch per epoch
            "limit_val_batches": 200,  # Number of batch per epoch
            "limit_test_batches": 200,  # Number of batch per epoch
            "val_check_interval": 1.
        }
        
    # Configure profiler settings
    if args.profiler:
        kwargs["profiler"] = PyTorchProfiler(filename="profiler_trace")
        kwargs["max_epochs"] = 1
        kwargs["limit_train_batches"] = 100
        kwargs["limit_val_batches"] = 10.
        kwargs["val_check_interval"] = 1

    # Configure gradient clipping
    if args.norm_clip > 0:
        kwargs["gradient_clip_val"] = args.norm_clip
        kwargs["gradient_clip_algorithm"] = "norm"

    # Configure multi-GPU training
    if args.multi_gpu:
        kwargs["strategy"] = "ddp_find_unused_parameters_true"  # "ddp"

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=loggers[::-1],  # Reverse order to prioritize TensorBoard
        log_every_n_steps=25,
        enable_checkpointing=True,
        callbacks=callbacks,
        **kwargs,
    )

    # Start training
    if args.from_checkpoint is not None and not args.weigth_only:
        # Resume training from checkpoint
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.from_checkpoint)
    else:
        # Start new training
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
