# Deep Generative Neural Networks for Survey Data Augmentation

**Assignment 02 — Reproduction & Extension Study**  
*Applying DGNNs to Data Augmentation for Consumer Survey Data with a Small Sample Size*

| Student | Roll No |
|---------|---------|
| Hamza Ishaq | 25i-7647 |
| Ashifa Ikram | 25i-7609 |
| Mariam Zahid | 25i-7610 |

---

## 📋 Project Overview

This project reproduces and extends the study by Watanuki et al. (2024), which investigates whether synthetically augmented tabular datasets produced by **Deep Generative Neural Networks (DGNNs)** can match or surpass the predictive performance of purely real datasets.

Three state-of-the-art generative models are evaluated:
- **CTGAN** — Conditional Tabular GAN
- **TVAE** — Tabular Variational Autoencoder
- **CopulaGAN** — GAN with copula-based dependency modelling

### Key Research Question
> Can synthetic tabular data from DGNNs serve as a statistically faithful and practically useful substitute or supplement for scarce real medical/survey data in binary classification tasks?

---

## 🗂️ Repository Structure

```
project-root/
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
├── config.yaml                  ← Centralized hyperparameter config
├── train.py                     ← End-to-end training pipeline
├── inference.py                 ← Load synthesizer → generate → evaluate
│
├── data/
│   └── sample_data.csv          ← 10-row sample (Q1–Q26 + country)
│
├── notebooks/
│   └── 01_inference_demo.ipynb  ← Interactive walkthrough demo
│
├── src/
│   ├── model.py                 ← Synthesizer wrappers (CTGAN/TVAE/CopulaGAN)
│   ├── dataset.py               ← Data loading, preprocessing, 11-config builder
│   └── utils.py                 ← Metrics, evaluation, visualization helpers
│
├── results/
│   ├── baseline_metrics.json    ← Real (Original) + Real (Seed) scores
│   ├── improved_metrics.json    ← All 11 configs + novelty extensions
│   └── training_log.csv         ← Per-epoch loss log (CTGAN/TVAE/CopulaGAN)
│
└── checkpoints/
    └── README.md                ← Instructions to download model weights
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dgnn-survey-augmentation.git
cd dgnn-survey-augmentation
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Download `Supplemental Tables.xlsx` from the paper's supplementary material and place it at:
```
data/Supplemental Tables.xlsx
```

---

## 🚀 Quick Start

### Train all three synthesizers
```bash
python train.py --config config.yaml
```

### Run inference & evaluation
```bash
python inference.py --synthesizer copulagan --output results/
```

### Interactive demo
```bash
jupyter notebook notebooks/01_inference_demo.ipynb
```

---

## 📊 Reproduced Results

### Synthetic Data Quality (Table 5)

| Model | Column Shape | Col. Pair Trend | Overall Quality |
|-------|-------------|-----------------|-----------------|
| CTGAN | 0.8937 | 0.9797 | 0.9367 |
| TVAE | 0.8962 | 0.9763 | 0.9363 |
| CopulaGAN | 0.8727 | 0.9716 | 0.9222 |

### ML Classification Results — Averaged across 4 Classifiers (Table 6)

| Dataset | Accuracy | F1 | AUC-ROC |
|---------|----------|----|---------|
| Real (Original) | 0.8319 | 0.8462 | 0.9297 |
| Real (Seed) | 0.8625 | 0.8737 | 0.9691 |
| Synthesized (CTGAN) | 0.8875 | 0.9002 | 0.9477 |
| Synthesized (TVAE) | 0.8562 | 0.8741 | 0.9362 |
| **Synthesized (CopulaGAN)** | **0.9938** | **0.9942** | **0.9987** |
| Seed + CTGAN | 0.8938 | 0.9023 | 0.9612 |
| Seed + TVAE | 0.9000 | 0.9092 | 0.9743 |
| **Seed + CopulaGAN** | **0.9219** | **0.9271** | **0.9854** |

### Novelty Extensions

| Extension | Description | Key Finding |
|-----------|-------------|-------------|
| **Novelty 1** | Augmentation Ratio Sweep (0.5×–5×) | Optimal ratio ≈ 2×–3× |
| **Novelty 2** | Ensemble Synthetic (12th config) | Seed + Ensemble F1 competitive with best single |
| **Novelty 4** | 5-Fold Stratified Cross-Validation | More robust generalization estimates |

---

## 🔬 Experimental Setup

| Component | Specification |
|-----------|---------------|
| Platform | Google Colaboratory |
| Accelerator | GPU (NVIDIA T4 / A100) |
| Python | 3.10+ |
| DGNN Epochs | 30,000 |
| Train/Val/Test Split | 64% / 16% / 20% stratified |
| Classifiers | GradientBoosting, RandomForest, SVM-RBF, SVM-Linear |

---

## 📚 Reference

> Watanuki, S.; Edo, K.; Miura, T. *Applying Deep Generative Neural Networks to Data Augmentation for Consumer Survey Data with a Small Sample Size.* Appl. Sci. 2024, 14, 9030. https://doi.org/10.3390/app14199030

---

## 📝 License

For academic use only. All rights to the original dataset belong to the respective paper authors.
