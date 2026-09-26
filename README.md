# Brain Tumor Segmentation via Federated Learning

> **Enhancing Brain Tumor Segmentation through Federated Learning under Non-IID Constraints**
> Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Bengaluru

---

## Overview

This project implements a **Federated Learning (FL)** pipeline for brain MRI tumor segmentation, tackling the challenge of training across **non-IID distributed datasets** while preserving patient data privacy. It compares two FL aggregation strategies -- FedAvg and FedAdagrad -- under varying non-IID data distributions controlled by a Dirichlet concentration parameter (alpha).

### Key Contributions

- Custom **U-Net with VGG16 encoder** and **CBAM (Convolutional Block Attention Module)** attention blocks
- Custom **Dice + Binary Cross-Entropy** combined loss function
- Comparison of **FedAvg** and **FedAdagrad** aggregation under alpha values of 0.5, 1.0, and 5.0
- 3 simulated clients with Dirichlet-sampled non-IID splits: 981 / 1359 / 724 samples
- 10 FL rounds x 15 local epochs per round
- Evaluation using **Mean IoU** and **Dice Score**

---

## Architecture

```
Client 1 (981 samples)  --+
Client 2 (1359 samples) --+--> FL Server (FedAvg / FedAdagrad) --> Global Model
Client 3 (724 samples)  --+

Per client:
  MRI Input -> VGG16 Encoder -> Skip Connections -> CBAM Attention -> U-Net Decoder -> Segmentation Mask
```

**Model components:**

| Component | Details |
|---|---|
| Encoder | VGG16 (pretrained on ImageNet, fine-tuned) |
| Attention | CBAM -- channel attention + spatial attention |
| Decoder | U-Net transposed convolutions with skip connections |
| Loss | 0.5 x Dice Loss + 0.5 x Binary Cross-Entropy |
| Optimizer | Adam (lr=1e-4) |

---

## Dataset

**Brain Tumor Segmentation Dataset** -- available on Kaggle

- MRI images + binary segmentation masks
- 3064 total samples split across 3 federated clients
- Download: `kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection`

---

## Experiments

| Notebook | FL Strategy | Non-IID Alpha | Description |
|---|---|---|---|
| `Fedavg_0.5&1.ipynb` | FedAvg | 0.5, 1.0 | High and medium non-IID |
| `Fedavg_5.ipynb` | FedAvg | 5.0 | Near-IID distribution |
| `Fedadagrad_0_5.ipynb` | FedAdagrad | 0.5 | Adaptive aggregation, high non-IID |
| `Fedadagrad_1.ipynb` | FedAdagrad | 1.0 | Adaptive aggregation, medium non-IID |
| `Fedadagrad_5.ipynb` | FedAdagrad | 5.0 | Adaptive aggregation, near-IID |

---

## Requirements

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

Python 3.8+, TensorFlow 2.x recommended.

---

## Key Concepts

- **Federated Learning** -- decentralised model training without sharing raw data
- **Non-IID data** -- real-world medical data is non-independently and identically distributed across hospitals
- **Dirichlet distribution** -- controls the degree of data heterogeneity across clients (lower alpha = more non-IID)
- **CBAM** -- attention mechanism that reweights feature maps channel-wise and spatially
- **FedAvg** -- weighted average of client model weights (McMahan et al., 2017)
- **FedAdagrad** -- adaptive gradient-based federated aggregation

---

## Authors

Geda Tejesh Chowdary | Paramkusam Sriharsha | Yelipe Gowtham

Amrita Vishwa Vidyapeetham, Bengaluru
