# Brain Tumor Segmentation via Federated Learning

> **Enhancing Brain Tumor Segmentation through Federated Learning under Non-IID Constraints**  
> Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Bengaluru

---

## Overview

This project implements a **Federated Learning (FL)** pipeline for brain MRI tumor segmentation, tackling the challenge of training across **non-IID distributed datasets** while preserving patient data privacy. It compares two FL aggregation strategies â€” FedAvg and FedAdagrad â€” under varying non-IID data distributions controlled by a Dirichlet concentration parameter (Î±).

### Key Contributions
- Custom **U-Net with VGG16 encoder** and **CBAM (Convolutional Block Attention Module)** attention blocks
- Custom **Dice + Binary Cross-Entropy** combined loss function
- Comparison of **FedAvg** and **FedAdagrad** aggregation under Î± âˆˆ {0.5, 1.0, 5.0}
- 3 simulated clients with Dirichlet-sampled non-IID splits: 981 / 1359 / 724 samples
- 10 FL rounds Ã— 15 local epochs per round
- Evaluation using **Mean IoU** and **Dice Score**

---

## Architecture

```
Client 1 (981 samples)  â”€â”€â”
Client 2 (1359 samples) â”€â”€â”¤â”€â”€â–º FL Server (FedAvg / FedAdagrad) â”€â”€â–º Global Model
Client 3 (724 samples)  â”€â”€â”˜

Per client:
  MRI Input â†’ VGG16 Encoder â†’ Skip Connections â†’ CBAM Attention â†’ U-Net Decoder â†’ Segmentation Mask
```

**Model components:**
| Component | Details |
|---|---|
| Encoder | VGG16 (pretrained on ImageNet, fine-tuned) |
| Attention | CBAM â€” channel attention + spatial attention |
| Decoder | U-Net transposed convolutions with skip connections |
| Loss | 0.5 Ã— Dice Loss + 0.5 Ã— Binary Cross-Entropy |
| Optimizer | Adam (lr=1e-4) |

---

## Dataset

**Brain Tumor Segmentation Dataset** â€” available on Kaggle  
- MRI images + binary segmentation masks
- 3064 total samples split across 3 federated clients
- Download: `kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection`

---

## Experiments

| Notebook | FL Strategy | Non-IID Î± | Description |
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

- **Federated Learning** â€” decentralised model training without sharing raw data
- **Non-IID data** â€” real-world medical data is non-independently and identically distributed across hospitals
- **Dirichlet distribution** â€” controls the degree of data heterogeneity across clients (lower Î± = more non-IID)
- **CBAM** â€” attention mechanism that reweights feature maps channel-wise and spatially
- **FedAvg** â€” weighted average of client model weights (McMahan et al., 2017)
- **FedAdagrad** â€” adaptive gradient-based federated aggregation

---

## Authors

Geda Tejesh Chowdary Â· Paramkusam Sriharsha Â· Yelipe Gowtham  
Amrita Vishwa Vidyapeetham, Bengaluru
