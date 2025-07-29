# -*- coding: utf-8 -*-
"""
Author: Arnaud Yarga

This module implements neuromorphic neural networks for speech denoising, inspired by:
- DenseNet architecture: https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
- Paper: https://ieeexplore.ieee.org/abstract/document/9437806

The networks use spiking neurons (LIF, ParaLIF) and population encoding for efficient
neuromorphic computing on event-driven hardware.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ParaLIF import LIF, ParaLIF


class HRNeuron(torch.nn.Module):
    """
    High Resolution neuron with population encoding capabilities.
    
    This class implements a population of neurons that can process high-resolution
    signals by using multiple neurons per input with rolling mechanisms for diversity.
    """
    
    def __init__(self, n_neuron, n_population, neurons, normalize=True, weight_mode="before", device=None):
        """
        Initialize HRNeuron.
        
        Args:
            n_neuron: Number of neurons in the population
            n_population: Number of population encodings
            neurons: Base neuron module (LIF, ParaLIF, etc.)
            normalize: Whether to normalize outputs by population size
            weight_mode: Weight application mode ("before", "after", "sign")
            device: Device to place parameters on
        """
        super().__init__()
        self.n_population = n_population
        
        self.neurons = neurons
        self.n_neuron = n_neuron
        self.weight_mode = weight_mode
        self.normalize = normalize
        
        # Define weight shape for population encoding
        shape = (n_population, self.n_neuron)
        
        # Initialize weights based on weight mode
        if weight_mode in ["before", "after"] and self.n_neuron > 1:
            self.w = torch.nn.parameter.Parameter(data=torch.randn(shape, device=device))
        if weight_mode == "sign" and self.n_neuron > 1:
            # Use alternating signs for sign mode
            self.register_buffer('w', (torch.arange(1, n_population + 1, device=device) % 2 - 0.5).sign().repeat(self.n_neuron).reshape(shape))

    def forward(self, x):
        """
        Forward pass through the HR neuron with population encoding.
        
        Args:
            x: Input tensor of shape (batch, channels, time, features)
            
        Returns:
            Processed output tensor
        """
        if self.n_neuron > 1:
            outputs = torch.zeros_like(x)
            
            # Population encoding with rolling for diversity
            # Allows the reuse of neurons for population encoding by rolling them so that each input is processed each time by a different neuron
            for i in range(self.n_population):
                # Roll input to create different perspectives
                x_rolled = torch.roll(x, i, dims=-1)
                
                # Apply weights before or after neuron processing
                if self.weight_mode in ["before", "sign"]: 
                    x_rolled = x_rolled * self.w[i]
                x_rolled = self.neurons(x_rolled)
                if self.weight_mode in ["after"]: 
                    x_rolled = x_rolled * self.w[i]
                
                # Roll back and accumulate
                x_rolled = torch.roll(x_rolled, -i, dims=-1)
                outputs += x_rolled
                
            # Normalize by population size if requested
            if self.normalize: 
                outputs = outputs / self.n_population
        else:
            # Single neuron case
            outputs = self.neurons(x)
            
        return outputs
    
    def get_spike_rate(self):
        """Get spike rate from the underlying neurons."""
        return self.neurons.nb_spike_per_neuron


class NeuronClass(nn.Module):
    """
    Wrapper class for different types of neurons with flexible processing modes.
    
    This class provides a unified interface for different neuron types (LIF, ParaLIF, ReLU)
    and handles various processing modes including interframe and sample-wise processing.
    """
    
    def __init__(self, n_neuron, neuron_args, interframe=False, sample_wise=False):
        """
        Initialize NeuronClass.
        
        Args:
            n_neuron: Number of neurons
            neuron_args: Dictionary containing neuron configuration
            interframe: Whether to process across frames
            sample_wise: Whether to process sample-wise
        """
        super().__init__()
        self.neuron = self.create_neuron(n_neuron, neuron_args)
        self.interframe = interframe
        self.sample_wise = sample_wise

    def create_neuron(self, n_neuron, neuron_args):
        """
        Create the appropriate neuron based on configuration.
        
        Args:
            n_neuron: Number of neurons
            neuron_args: Neuron configuration dictionary
            
        Returns:
            Configured neuron module
        """
        neuron_args = neuron_args.copy()
        
        # Check if using non-spiking neurons (ReLU)
        self.not_spiking = neuron_args.pop("not_spiking")
        if self.not_spiking: 
            return nn.ReLU()
            
        # Extract neuron type and parameters
        neuron = neuron_args.pop("neuron")
        population = neuron_args.pop("population") if "population" in neuron_args.keys() else 1
        multi_threshold = neuron_args.pop("multi_threshold")
        
        # Set up multi-threshold if enabled
        if multi_threshold:
            neuron_args['spk_threshold'] = np.random.uniform(
                low=neuron_args['spk_threshold'] * 0.5,
                high=neuron_args['spk_threshold'] * 1.5,
                size=(n_neuron,)
            ).tolist()
            
        # Set up multi-time constants if enabled
        if neuron_args.pop("multi_time_constant"):
            if neuron_args['tau_mem'] is not None:
                neuron_args['tau_mem'] = np.random.uniform(
                    low=neuron_args['tau_mem'] * 0.5,
                    high=neuron_args['tau_mem'] * 1.5,
                    size=(n_neuron,)
                ).tolist()
            if neuron_args['tau_syn'] is not None:
                neuron_args['tau_syn'] = np.random.uniform(
                    low=neuron_args['tau_syn'] * 0.5,
                    high=neuron_args['tau_syn'] * 1.5,
                    size=(n_neuron,)
                ).tolist()
                
        # Determine neuron type (ParaLIF or LIF)
        spike_mode = neuron.split('-')[-1] if "ParaLIF" in neuron else ""
        neuron_type = ParaLIF if spike_mode else LIF
        if spike_mode: 
            neuron_args['spike_mode'] = spike_mode
            
        # Create base neuron layer
        layer = neuron_type(n_neuron, **neuron_args)
        
        # Wrap in HRNeuron if population encoding is used
        if population > 1:
            layer = HRNeuron(n_neuron, population, layer, weight_mode=None)
            
        return layer

    def forward(self, x):
        """
        Forward pass with flexible processing modes.
        
        Args:
            x: Input tensor of shape (batch, channels, time, features)
            
        Returns:
            Processed output tensor
        """
        if self.not_spiking: 
            return self.neuron(x)
            
        b, c, t, f = x.shape
        
        # Different processing modes for spiking neurons
        if self.sample_wise:
            # Process each sample independently
            x = self.neuron(x.permute(0, 2, 3, 1).reshape(b, t, f * c)).reshape(b, t, f, c).permute(0, 3, 1, 2)
        elif self.interframe:
            # Process across frames
            x = self.neuron(x.permute(0, 3, 2, 1).reshape(b * f, t, c)).reshape(b, f, t, c).permute(0, 3, 2, 1)
        else:
            # Standard processing across time
            x = self.neuron(x.permute(0, 2, 3, 1).reshape(b * t, f, c)).reshape(b, t, f, c).permute(0, 3, 1, 2)
            
        return x

    def get_spike_rate(self):
        """Get spike rate from the neuron."""
        if isinstance(self.neuron, (LIF, ParaLIF)):
            return self.neuron.nb_spike_per_neuron
        if isinstance(self.neuron, HRNeuron):
            return self.neuron.get_spike_rate()
        return torch.tensor(0)


class DELAYAvg(nn.Module):
    """
    Delay and averaging module for temporal processing.
    
    This module introduces delays to different input channels and optionally
    applies averaging pooling for temporal smoothing.
    """
    
    def __init__(self, n_input, delay_max=10, avgpooling=None, device=None, rand=False):
        """
        Initialize DELAYAvg.
        
        Args:
            n_input: Number of input channels
            delay_max: Maximum delay value
            avgpooling: Optional averaging pooling layer
            device: Device to place delays on
            rand: Whether to use random delays or linear spacing
        """
        super().__init__()
        self.n_input = n_input
        self.rand = rand
        self.avgpooling = avgpooling if avgpooling else nn.Identity()
        self.register_buffer('delay_max', torch.tensor(delay_max))
        
        # Set up delays
        if rand:
            # Random delays
            self.register_buffer('delays', torch.randint(0, delay_max + 1, (n_input,), device=device).to(torch.long))
        else: 
            # Linearly spaced delays
            self.register_buffer('delays', torch.linspace(0, delay_max, n_input, device=device).to(torch.long))
        self.device = device

    def forward(self, x):
        """
        Apply delays and averaging to input.
        
        Args:
            x: Input tensor of shape (batch, channels, time, features)
            
        Returns:
            Delayed and averaged output tensor
        """
        b, c, t, f = x.shape
        
        # Pad input to accommodate delays
        x = F.pad(x, (self.delay_max, 0), "constant", 0)
        out = []
        
        # Apply different delays to each channel
        for i, delay in enumerate(self.delays):
            offset = self.delay_max - delay
            out.append(self.avgpooling(x[..., offset:f + offset]))
            
        # Concatenate delayed outputs
        x = torch.cat(out, dim=1)
        return x
        
    def __repr__(self):
        return f"DELAYAvg(n_input={self.n_input}, delay_max={self.delay_max.item()}, rand={self.rand}, avgpooling={str(self.avgpooling)}"


class SubBlock(nn.Module):
    """
    Sub-block component with convolution and activation.
    
    This is a building block that applies convolution followed by activation
    and concatenates the output with the input (residual connection).
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, neuron, pad=None):
        """
        Initialize SubBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            neuron: Neuron activation function
            pad: Optional padding configuration
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding="same" if pad is None else 0, 
                             dilation=(1, 1), bias=False)
        self.activation = neuron
        self.pad = pad
        self.synapses = np.prod(self.conv.weight.data.shape)
        self.input_spike_rate = 0
        
    def forward(self, x):
        """
        Forward pass through sub-block.
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated input and processed output
        """
        # Calculate input spike rate for monitoring
        self.input_spike_rate = torch.mean((torch.abs(x) > 0).to(x.dtype)).item()
        
        # Apply convolution and activation
        out = self.conv(x)
        out = self.activation(out) 
        
        # Apply padding if specified
        if self.pad is not None: 
            out = F.pad(out, self.pad)
            
        # Concatenate with input (residual connection)
        return torch.concat((x, out), 1)
        
    def get_spike_rate(self):
        """Get spike rate from activation."""
        return self.activation.get_spike_rate()
        
    def get_synops(self):
        """Get synaptic operations count."""
        return self.synapses * self.input_spike_rate


class TransBlock(nn.Module):
    """
    Transition block with dual convolution paths and normalization.
    
    This block processes two input streams through separate convolution paths,
    applies normalization and activation, then combines the results.
    """
    
    def __init__(self, in_channels1, in_channels2, out_channels, n_features, 
                 kernel_size1, padding1, kernel_size2, stride2, padding2, 
                 dilation, neuron, norm="layer", groups=2, stride1=1):
        """
        Initialize TransBlock.
        
        Args:
            in_channels1: Number of channels for first input
            in_channels2: Number of channels for second input
            out_channels: Number of output channels
            n_features: Number of features for normalization
            kernel_size1: Kernel size for first convolution
            padding1: Padding for first convolution
            kernel_size2: Kernel size for second convolution
            stride2: Stride for second convolution
            padding2: Padding for second convolution
            dilation: Dilation factor
            neuron: Neuron activation function
            norm: Normalization type ("layer" or "batch")
            groups: Number of groups for grouped convolution
            stride1: Stride for first convolution
        """
        super().__init__()
        
        # Dual convolution paths
        self.conv1 = nn.Conv2d(in_channels1, out_channels, kernel_size1, 
                              stride=(1, stride1), padding=padding1, 
                              dilation=(1, dilation), bias=False, groups=groups)
        self.conv2 = nn.Conv2d(in_channels2, out_channels, kernel_size2, 
                              stride=(1, stride2), padding=padding2, 
                              dilation=(1, dilation), bias=False, groups=groups)
        
        self.activation = neuron
        
        # Normalization layers
        self.normalise1 = nn.LayerNorm(n_features, bias=False) if norm == "layer" else nn.BatchNorm2d(out_channels)
        self.normalise2 = nn.LayerNorm(n_features, bias=False) if norm == "layer" else nn.BatchNorm2d(out_channels)
        
        # Synaptic operation tracking
        self.synapses1 = np.prod(self.conv1.weight.data.shape)
        self.synapses2 = np.prod(self.conv2.weight.data.shape)
        self.input_spike_rate1 = 0
        self.input_spike_rate2 = 0
        
    def forward(self, x1, x2):
        """
        Forward pass through transition block.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Combined output tensor
        """
        # Track input spike rates
        self.input_spike_rate1 = torch.mean((torch.abs(x1) > 0).to(x1.dtype)).item()
        self.input_spike_rate2 = torch.mean((torch.abs(x2) > 0).to(x2.dtype)).item()
        
        # Process first input path
        x1 = self.conv1(x1)
        x1 = self.normalise1(x1)
        out1 = F.relu(x1)
        
        # Process second input path
        x2 = self.conv2(x2)
        x2 = self.normalise2(x2)
        out2 = F.relu(x2)
        
        # Combine and apply final activation
        out = self.activation(out1 + out2) 
        return out
        
    def get_spike_rate(self):
        """Get spike rate from activation."""
        return self.activation.get_spike_rate()
        
    def get_synops(self):
        """Get total synaptic operations count."""
        return self.synapses1 * self.input_spike_rate1 + self.synapses2 * self.input_spike_rate2


class DenoisingNet(nn.Module):
    """
    Main denoising network using neuromorphic architecture.
    
    This network implements a dense, multi-scale architecture for speech denoising
    using spiking neurons and population encoding. It processes audio through
    multiple blocks with skip connections and interframe processing.
    """
    
    def __init__(self, out_features, n_embedding=16, nb_blocks=7, dropRate=0.0,
                 neuron_args={}, pdm_oversampling=128, ks=12, ks1=6, n_hidden=64, plus=8, delay_max=8, n_out=256, 
                 stride=4, dilation=2, add_interframe=True, causal=True, n_out_interframe=None, groups=2, ks_interframe=12, no_sub_block=False):
        """
        Initialize DenoisingNet.
        
        Args:
            out_features: Number of output features
            n_embedding: Number of embedding dimensions
            nb_blocks: Number of processing blocks
            dropRate: Dropout rate
            neuron_args: Neuron configuration dictionary
            pdm_oversampling: PDM oversampling factor
            ks: Kernel size for main convolutions
            ks1: Kernel size for sub-block convolutions
            n_hidden: Number of hidden units
            plus: Additional units per block
            delay_max: Maximum delay for temporal processing
            n_out: Number of output channels
            stride: Stride factor
            dilation: Dilation factor
            add_interframe: Whether to add interframe processing
            causal: Whether to use causal processing
            n_out_interframe: Number of interframe output channels
            groups: Number of groups for grouped convolution
            ks_interframe: Kernel size for interframe processing
            no_sub_block: Whether to skip sub-blocks
        """
        super().__init__()
        
        # Store configuration parameters
        self.dropRate = dropRate
        self.frame_dim_conv = 1
        self.dilation = dilation
        self.ks = ks
        self.ks1 = ks1
        self.n_k = n_hidden
        self.n_out_interframe = n_hidden if n_out_interframe is None else n_out_interframe
        self.sub_block_out = 8
        self.nb_blocks = nb_blocks
        self.pdm_oversampling = pdm_oversampling
        self.out_features = out_features

        # Calculate final number of hidden units
        self.n_k_final = self.n_k + plus * (self.nb_blocks - 1)
        self.n_embedding = n_embedding
        self.n_out = n_out
        self.stride = stride
        self.delay_max = delay_max
        self.add_interframe = add_interframe
        self.causal = causal
        self.pdm_scale = 1
        
        # Input processing with delay and averaging
        if self.pdm_oversampling > 1 and self.delay_max > 0:
            self.net_in = DELAYAvg(self.n_embedding, delay_max=self.delay_max, 
                                  avgpooling=nn.AvgPool2d((1, self.pdm_oversampling), (1, self.pdm_oversampling)))
        else:
            self.net_in = nn.Identity()
            self.n_embedding = 1
            self.pdm_scale = self.pdm_oversampling
            
        # Build main processing blocks
        net = []
        n_in_ = self.n_embedding
        
        for i in range(self.nb_blocks):
            n_out_ = self.n_k + plus * i
            
            # Calculate parameters for second convolution in transblock
            stride_trans_2 = 2**i * self.pdm_scale
            ks_trans_2 = self.ks * stride_trans_2
            pad_trans_2 = int(np.ceil(((ks_trans_2 - 1) * self.dilation + 1 - stride_trans_2) / 2))
            
            if i == 0 and self.pdm_scale > 1:
                # Special handling for first block with PDM oversampling
                net.append(
                    nn.ModuleList([
                        nn.Identity() if no_sub_block else nn.Sequential(
                            SubBlock(n_in_, self.sub_block_out, (1, self.ks1 * self.pdm_scale), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                            SubBlock(n_in_ + self.sub_block_out, self.sub_block_out, (1, self.ks1 * self.pdm_scale), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                            SubBlock(n_in_ + self.sub_block_out * 2, self.sub_block_out, (1, self.ks1 * self.pdm_scale), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                        ),
                        TransBlock(n_in_ + self.sub_block_out * 3 * int(not no_sub_block), self.n_embedding, n_out_, 
                                  out_features, (1, self.ks * self.pdm_scale), (0, pad_trans_2), (1, ks_trans_2), 
                                  stride_trans_2, (0, pad_trans_2), self.dilation, 
                                  NeuronClass(n_out_, neuron_args), groups=groups, stride1=stride_trans_2)
                    ])
                )
            else:
                # Standard block processing
                net.append(
                    nn.ModuleList([
                        nn.Identity() if no_sub_block else nn.Sequential(
                            SubBlock(n_in_, self.sub_block_out, (1, self.ks1), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                            SubBlock(n_in_ + self.sub_block_out, self.sub_block_out, (1, self.ks1), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                            SubBlock(n_in_ + self.sub_block_out * 2, self.sub_block_out, (1, self.ks1), 
                                   NeuronClass(self.sub_block_out, neuron_args)),
                        ),
                        TransBlock(n_in_ + self.sub_block_out * 3 * int(not no_sub_block), self.n_embedding, n_out_, 
                                  out_features // 2**i, (1, self.ks), "same", (1, ks_trans_2), 
                                  stride_trans_2, (0, pad_trans_2), self.dilation, 
                                  NeuronClass(n_out_, neuron_args), groups=groups)
                    ])
                )
            n_in_ = n_out_
            
        self.net = nn.ModuleList(net)

        # Interframe processing (optional)
        down = 2**self.nb_blocks // 2
        if self.add_interframe:
            self.ks_interframe = ks_interframe
            self.pad_interframe = (0, 0, self.ks_interframe - 1, 0) if self.causal else None
            pad_trans = 0 if self.causal else "same"
            self.net_interframe = nn.ModuleList([
                nn.Identity() if no_sub_block else nn.Sequential(
                    SubBlock(self.n_k_final, self.sub_block_out, (self.ks_interframe, 1), 
                            NeuronClass(self.sub_block_out, neuron_args, interframe=True), self.pad_interframe),
                    SubBlock(n_in_ + self.sub_block_out, self.sub_block_out, (self.ks_interframe, 1), 
                            NeuronClass(self.sub_block_out, neuron_args, interframe=True), self.pad_interframe),
                    SubBlock(n_in_ + self.sub_block_out * 2, self.sub_block_out, (self.ks_interframe, 1), 
                            NeuronClass(self.sub_block_out, neuron_args, interframe=True), self.pad_interframe),
                ),
                TransBlock(n_in_ + self.sub_block_out * 3 * int(not no_sub_block), self.n_embedding, 
                          self.n_out_interframe, out_features // down, (self.ks_interframe, 1), pad_trans, 
                          (self.ks_interframe, 1), 1, pad_trans, 1, 
                          NeuronClass(self.n_out_interframe, neuron_args, interframe=True), groups=groups)
            ])
            
        # Output processing layers
        up = down // self.stride
        ks_mask = 4 * up
        ks_codec = 8 * self.stride
        
        # Mask generation and encoding/decoding layers
        self.net_out_mask = nn.ConvTranspose2d(self.n_k_final + self.n_out_interframe * int(self.add_interframe),
                                              self.n_out, (self.frame_dim_conv, ks_mask), 
                                              padding=(0, ks_mask // 2 - 1), stride=(1, up), bias=False)
        self.net_out_enc = nn.Sequential(
            nn.Conv2d(self.n_embedding, self.n_out, (self.frame_dim_conv, ks_codec * self.pdm_scale), 
                     stride=(1, self.stride * self.pdm_scale), padding=(0, ks_codec * self.pdm_scale // 2 - 1), bias=False), 
            nn.ReLU()
        )
        self.net_out_dec = nn.ConvTranspose2d(self.n_out, 1, (self.frame_dim_conv, ks_codec), 
                                             stride=(1, self.stride), padding=(0, ks_codec // 2 - 1), bias=False)
        
        # Synaptic operation tracking
        self.synapses_mask = np.prod(self.net_out_mask.weight.data.shape)
        self.synapses_enc = np.prod(self.net_out_enc[0].weight.data.shape)
        self.synapses_dec = np.prod(self.net_out_dec.weight.data.shape)
        self.input_spike_rate_mask = 0
        self.input_spike_rate_enc = 0
        self.input_spike_rate_dec = 0

    def forward(self, x):
        """
        Forward pass through the denoising network.
        
        Args:
            x: Input tensor of shape (batch, channels, time, features)
            
        Returns:
            Denoised output tensor
        """
        # Calculate output sizes for different stages
        output_size = (x.shape[-2], self.out_features)
        output_size1 = (x.shape[-2], self.out_features // self.stride)
        
        # Input processing with delays and averaging
        x_skipped = self.net_in(x)
        xl = x_skipped
        
        # Process through main blocks with skip connections
        for l in range(self.nb_blocks):
            if l > 0: 
                xl = F.avg_pool2d(xl, (1, 2), (1, 2))  # Downsample for deeper blocks
            xl = self.net[l][0](xl)  # Sub-blocks
            xl = self.net[l][1](xl, x_skipped)  # Transition block with skip connection
            if self.dropRate > 0:
                xl = F.dropout(xl, p=self.dropRate, training=self.training)
        
        # Interframe processing (optional)
        if self.add_interframe:
            xl_pad = F.pad(xl, self.pad_interframe) if self.causal else xl
            x_if = self.net_interframe[0](xl_pad)
            x_skipped_pad = F.avg_pool2d(x_skipped, (1, 2**self.nb_blocks // 2 * self.pdm_scale), 
                                       (1, 2**self.nb_blocks // 2 * self.pdm_scale))
            if self.causal: 
                x_skipped_pad = F.pad(x_skipped_pad, self.pad_interframe)
            x_if = self.net_interframe[1](x_if, x_skipped_pad)
            xl = torch.cat([xl, x_if], dim=1)  # Concatenate main and interframe features

        # Generate mask and apply encoding/decoding
        x_masked = self.net_out_enc(x_skipped) * self.net_out_mask(xl, output_size=output_size1).sigmoid()
        x = self.net_out_dec(x_masked, output_size=output_size)
        
        # Track spike rates for monitoring
        self.input_spike_rate_enc = torch.mean((torch.abs(x_skipped) > 0).to(x_skipped.dtype)).item()
        self.input_spike_rate_mask = torch.mean((torch.abs(xl) > 0).to(xl.dtype)).item()
        self.input_spike_rate_dec = torch.mean((torch.abs(x_masked) > 0).to(x_masked.dtype)).item()
        
        return x

    def log_spike_rate(self):
        """
        Log spike rates from all neuron layers for monitoring.
        
        Returns:
            Dictionary containing spike rates for different network components
        """
        spike_rate = {}
        spike_rate["spike_rate/all"] = 0
        spike_rate["spike_rate/hidden"] = 0
        if self.add_interframe:
            spike_rate["spike_rate/net_interframe"] = 0
            
        # Initialize spike rate tracking for each block
        for i, block in enumerate(self.net):
            spike_rate[f"spike_rate/hidden/{i}"] = 0
            
        # Collect spike rates from all neuron modules
        for name, module in self.named_modules():
            if isinstance(module, NeuronClass):
                spk = module.get_spike_rate().sum().cpu().item()
                name = name.split(".")
                net, net_i = name[0], name[1]
                
                # Categorize spike rates by network component
                if net == "net":
                    spike_rate[f"spike_rate/hidden/{net_i}"] += spk
                    spike_rate["spike_rate/hidden"] += spk
                    spike_rate["spike_rate/all"] += spk
                elif net == "net_interframe":
                    spike_rate["spike_rate/net_interframe"] += spk
                    spike_rate["spike_rate/all"] += spk
                    
        return spike_rate
    
    def get_spike_sum(self):
        """
        Get sum of spike rates from all neuron layers.
        
        Returns:
            Concatenated tensor of spike rates
        """
        spike_sum = torch.cat([module.get_spike_rate() for module in self.modules() if isinstance(module, NeuronClass)])
        return spike_sum
    
    def get_synops(self):
        """
        Get synaptic operations count from all layers.
        
        Returns:
            List of synaptic operation counts
        """
        all_synops = [module.get_synops() for module in self.modules() if isinstance(module, (SubBlock, TransBlock))]
        all_synops += [self.synapses_mask * self.input_spike_rate_mask, 
                      self.synapses_enc * self.input_spike_rate_enc, 
                      self.synapses_dec * self.input_spike_rate_dec]
        return all_synops
        
    def get_synapses(self):
        """
        Get synapse counts from all layers.
        
        Returns:
            List of synapse counts
        """
        all_synapses = [(module.synapses if isinstance(module, SubBlock) else (module.synapses1 + module.synapses2)) 
                        for module in self.modules() if isinstance(module, (SubBlock, TransBlock))]
        all_synapses += [self.synapses_mask, 
                        self.synapses_enc, 
                        self.synapses_dec]
        return all_synapses
        
    def get_neuronops(self):
        """
        Get neuron operation counts from all neuron layers.
        
        Returns:
            List of neuron operation counts
        """
        neuronops = []
        for name, module in self.named_modules():
            if isinstance(module, NeuronClass):
                neuronops.append(module.neuron.n_neuron)
        return neuronops


if __name__ == "__main__":
    # Example usage and testing
    neuron_args = {
        'not_spiking': False,
        'neuron': "ParaLIF-T",
        'recurrent': False,
        'learn_threshold': True,
        'tau_mem': 1e-3,
        'tau_syn': 1e-3,
        'spk_threshold': 1.,
        'learn_tau': False,
        'surrogate_mode': "atan",
        'multi_threshold': True,
        'multi_time_constant': False,
        'population': 5
    }
    
    # Create and test model
    model = DenoisingNet(512, 16, neuron_args=neuron_args, nb_blocks=5, ks=7, ks1=3, 
                        n_hidden=80, plus=10, n_out=128, dilation=4, add_interframe=True)
    
    # Print model statistics
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1000}k")
    print(f"Output shape: {model(torch.rand(2, 1, 32, 512 * 128)).shape}")
    
    # Test spike rate collection
    spike_rate = []
    for name, module in model.named_modules():
        if isinstance(module, NeuronClass):
            spike_rate.append(module.get_spike_rate())
            print(f"Spike rate from {name}: {module.get_spike_rate().shape}")
    
    spike_rate = torch.cat(spike_rate)
    print(f"Total spike rate shape: {spike_rate.shape}")
    
    # Alternative spike rate collection
    spike_rate = torch.cat([module.get_spike_rate() for module in model.modules() if isinstance(module, NeuronClass)])
    print(f"Alternative spike rate shape: {spike_rate.shape}")
    












