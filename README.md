# PDMDNS: End-to-End Neuromorphic Speech Enhancement with PDM Microphones

This repository hosts the code and resources for the paper:

**“End-to-End Neuromorphic Speech Enhancement with PDM Microphones”**  
*Sidi Yaya Arnaud Yarga, Sean U. N. Wood*  
Department of Electrical and Computer Engineering, Université de Sherbrooke  
[**Paper Link**](https://iopscience.iop.org/article/10.1088/2634-4386/adf2d4)

> **PDMDNS** is a real-time speech denoising framework that directly processes raw binary Pulse Density Modulation (PDM) signals using a Spiking Neural Network (SNN), completely bypassing the conventional PDM-to-PCM conversion stage. This allows for energy-efficient, low-latency operation on neuromorphic or edge hardware.

---

## 🔍 Overview

Conventional speech enhancement pipelines rely on Pulse Code Modulation (PCM) and deep neural networks, which are ill-suited for always-on embedded systems due to their computational and preprocessing costs. PDMDNS addresses this by:

- **Eliminating the need for PDM-to-PCM conversion**
- **Leveraging stateless spiking neurons for low-power computation**
- **Simultaneously enhancing speech and converting format**
- **Robustly generalizing to varying PDM sampling rates**

Despite being nearly 3× more efficient in terms of multiply-operations per second (M-Ops/s), PDMDNS achieves a competitive SI-SNR improvement of **+7 dB** and **+3% STOI** across a wide range of input SNRs (20 dB to -5 dB).

---

## 📁 Repository Structure
```
PDMDNS/
├── examples/                  # Sample audio files from the test set: noisy inputs, model outputs (enhanced), and clean references
├── README.md                  # Project overview, installation, usage, and structure (this file)
└── src/                       # Source code and resources
    ├── data/                  # Example of audio files
    ├── intel_code/            # Utilities and scripts related to Intel's neuromorphic DNS challenge (https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge)
    ├── microsoft_dns/         # Utilities and scripts related to the Microsoft Deep Noise Suppression Challenge
    ├── notebooks/             # A demo jupyter notebook for evaluation
    ├── pdmdns_solution/       # Main solution code
    │   ├── models/            # Neural network architectures
    │   ├── scripts/           # Training, evaluation, and inference scripts
    │   └── utils/             # Helper functions, metrics, logging, and configuration tools
    └── saved_checkpoints/     # Pretrained model checkpoints

```


- 🔧 The `src/pdmdns_solution/` directory contain the full implementation of the PDMDNS model, including training, evaluation, and inference scripts.
- 📊 The `src/notebooks/` directory contain a demo jupyter notebook for evaluation.
- 🔊 The `examples/` directory includes audio samples showcasing PDMDNS's performance on real-world noisy signals.

---

## ⚙️ Setup

Before running the code, ensure you are using **Python 3.10 or higher**.

### 📦 Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

Once the environment is ready, you can proceed to training and evaluation as described below.

---


## 🛠️ Usage

This repository provides scripts to **train** and **evaluate** neuromorphic speech denoising models. The training and evaluation pipelines are customizable through command-line arguments.

### 🔄 Example Usage

You can run the scripts using IPython or from the command line. Below are examples of each:

#### ▶️ Train a Model

```python
./src/pdmdns_solution/scripts/train.py --dataset_path ./path --config final --slurm_id 0 --data_length 2 --batch_size 16
        --max_epochs 100 --experiment_name base_experiments Model --pdm_oversampling 128
```

#### 🧪 Evaluate a Model

```python
./src/pdmdns_solution/scripts/eval.py --dataset_path ./path --config final --slurm_id 0 --data_length 30
        --batch_size 2 --experiment_name base_experiments Model --pdm_oversampling 128
```

---

## ⚙️ Configuration Options

You can choose between predefined experiment settings using the `--config final` flag combined with a `--slurm_id`:

| `slurm_id` | Description        | Effect                              |
| ---------- | ------------------ | ----------------------------------- |
| `0`        | Base configuration | Default model setup                 |
| `1`        | Stateful neurons   | Sets `tau_mem = 1e-3`               |
| `2`        | Non-causal model   | Sets `archi_args['causal'] = False` |

If no `slurm_id` is provided, the script falls back to fully **customizable command-line arguments**.

---

## 🧩 Command-Line Arguments

### 🔢 General Training Arguments

| Argument            | Description                             |
| ------------------- | --------------------------------------- |
| `--dataset_path`    | Path to the training or testing dataset |
| `--batch_size`      | Batch size for training/evaluation      |
| `--data_length`     | Length of audio samples in seconds      |
| `--max_epochs`      | Number of training epochs               |
| `--experiment_name` | Logging/Checkpoint sub-directory        |
| `--from_checkpoint` | Resume training from checkpoint         |
| `--multi_gpu`       | Enable multi-GPU training               |

### 🧠 Model Configuration Arguments

| Argument                 | Description                          |
| ------------------------ | ------------------------------------ |
| `--model`                | Select model type (`Model` required) |
| `--n_chan`               | Number of frequency channels         |
| `--dropout`              | Dropout rate                         |
| `--pdm_oversampling`     | Oversampling factor for PDM input    |
| `--not_spiking`          | Use non-spiking (ReLU-based) model   |
| `--neuron`               | Neuron type (`ParaLIF-T`, etc.)      |
| `--tau_mem`, `--tau_syn` | Time constants for neurons           |

---

## 📎 Citation

If you find this work useful, please consider citing our paper:

```
@article{10.1088/2634-4386/adf2d4,
	author={Yarga, Sidi Yaya Arnaud and Wood, Sean U. N.},
	title={End-to-end neuromorphic speech enhancement with PDM microphones},
	journal={Neuromorphic Computing and Engineering},
	url={http://iopscience.iop.org/article/10.1088/2634-4386/adf2d4},
	year={2025}
}
```
---

## 📫 Contact
For questions or collaborations, please contact:
- Sidi Yaya Arnaud Yarga – sidi.yaya.arnaud.yarga@usherbrooke.ca
