# PDMDNS: End-to-End Neuromorphic Speech Enhancement with PDM Microphones

This repository hosts the code and resources for the paper:

**“End-to-End Neuromorphic Speech Enhancement with PDM Microphones”**  
*Sidi Yaya Arnaud Yarga, Sean U. N. Wood*  
Department of Electrical and Computer Engineering, Université de Sherbrooke  
[**Paper Link (TBD)**]

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
├── src/ # Source code (Coming soon)
├── examples/ # Audio examples: noisy, enhanced, and clean references
└── README.md # You are here
```


- 🔧 The `src/` directory will contain the full implementation of the PDMDNS model, including training, evaluation, and inference scripts.
- 🔊 The `examples/` directory includes audio samples showcasing PDMDNS's performance on real-world noisy signals.

---

## 🚀 Coming Soon

- 🧠 Spiking neural network implementation in PyTorch
- 📊 Training and evaluation scripts
- 📦 Pre-trained models
- 🔉 Live inference demo on raw PDM inputs

---

## 📎 Citation

If you find this work useful, please consider citing our paper:

[coming soon]

---

## 📫 Contact
For questions or collaborations, please contact:
- Sidi Yaya Arnaud Yarga – sidi.yaya.arnaud.yarga@usherbrooke.ca
