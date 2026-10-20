# Neuro-Symbolic Concept Lattice (NSCL) Framework

This repository contains the source code, configuration files, and reproducibility package for the paper: 
**"Neuro-Symbolic Concept Lattice: Real-Time Multimodal Fusion for Clinical Decision Support"**.

## 1. Environment and Dependencies
The code was developed and tested using the following environment:
* Python 3.9+
* PyTorch 2.1.0
* PyTorch Geometric 2.4.0
* NumPy, Pandas, Scikit-learn

To install the required packages, run:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 2. Datasets (Restricted Access)
Due to data use agreements (DUA), the raw patient data cannot be shared directly in this repository. 
* **MIMIC-IV**: Available from [PhysioNet](https://physionet.org/content/mimiciv/) upon DUA completion.
* **eICU-CRD**: Available from [PhysioNet](https://physionet.org/content/eicu-crd/) upon DUA completion.
* **CheXpert**: Available from [Stanford AIMI](https://stanfordmlgroup.github.io/competitions/chexpert/) upon DUA completion.

## 3. Model Architecture Overview
* **GNN Encoder**: 3 layers, hidden dimension 256, 8 attention heads.
* **Concept Pool**: Embedding dimension 128.
* **Symbolic Layer**: MLP aggregators (2 hidden layers, ReLU activation).
* **Incremental Update**: Implements the Lipschitz stability algorithm (Theorem 4.1).

## 4. Training Configuration
* **Optimizer**: Adam (lr = 1e-4, weight decay = 1e-5)
* **Batch Size**: 64
* **Epochs**: 100 (with early stopping, patience = 10)
* **Hardware**: NVIDIA A100 GPU (40GB)
* **Random Seeds**: [42, 123, 456, 789, 1010]

## 5. Repository Structure
* `models/`: PyTorch implementation of the GNN and Symbolic Layer.
* `configs/`: Hyperparameter configurations.
* `preprocessing/`: Scripts for data extraction and tokenization (to be run after obtaining DUA).
* `utils/`: Helper functions for metric calculation (AUC, F1, CCS).

## 6. Contact
For any questions regarding the code or the paper, please contact: `csqiaolang@163.com`.
