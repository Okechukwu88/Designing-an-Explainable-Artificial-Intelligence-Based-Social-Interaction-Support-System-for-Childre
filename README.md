Designing an Explainable Artificial Intelligence-Based Social Interaction Support System for Children with Autism Spectrum Disorder:

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX)

> **A Computational Feasibility Study on Explainable AI for Facial Expression Recognition in ASD Support**

---

## 📋 Overview

This repository contains the code and experimental framework for the MRes dissertation:

**"Designing an Explainable Artificial Intelligence-Based Social Interaction Support System for Children with Autism Spectrum Disorder"**

The project investigates the computational feasibility of combining Convolutional Neural Networks (CNNs) with Explainable AI (XAI) methods—specifically Gradient-weighted Class Activation Mapping (Grad-CAM)—to create a facial expression recognition system that can provide child-accessible explanations for emotion predictions.

### Key Contributions

- **Four custom CNN architectures** (110k–1.24M parameters) trained on the ExpW dataset
- **Grad-CAM explainability module** generating anatomically plausible visual heatmaps
- **Rule-based natural language explanation system** producing child-accessible verbal descriptions
- **Honest performance benchmarking** with per-class analysis revealing class imbalance challenges

---

## ⚠️ Important Disclaimer

> **This is a COMPUTATIONAL FEASIBILITY STUDY only.**
>
> - No human participant research of any kind was conducted
> - No direct contact was made with children with ASD, their parents/caregivers, or clinical professionals
> - All findings are computational and require further research for clinical validation
> - The current system achieves zero recall for Neutral expressions and near-zero recall for Happy expressions, and should NOT be used in any form of participant research or deployment

---

## 📊 Dataset

### Expression in-the-Wild (ExpW)

This research uses a subset of the **ExpW dataset** (Zhang et al., 2015):

| Feature | Description |
|---------|-------------|
| Total images (subset) | 59,442 |
| Emotion classes | 4 (Angry, Happy, Sad, Neutral) |
| Source | Google image searches |
| Label method | Query-derived (approx. 75% human-verified accuracy) |
| Split | 60% Train / 20% Validation / 20% Test |

**Dataset Statistics (After Four-Class Reduction):**

| Split | Total | Angry | Happy | Sad | Neutral |
|-------|-------|-------|-------|-----|---------|
| Training | 35,665 | 4,908 (13.8%) | 2,612 (7.3%) | 4,857 (13.6%) | 512 (1.4%) |
| Validation | 11,888 | 1,636 (13.8%) | 871 (7.3%) | 1,619 (13.6%) | 171 (1.4%) |
| Test | 11,889 | 4,858 (40.9%) | 2,453 (20.6%) | 4,565 (38.4%) | 13 (0.1%) |

> ⚠️ **Note:** The Neutral class is severely underrepresented (0.1% of test set), making it unlearnable under standard cross-entropy training.

---

## 🏗️ CNN Architectures

| Model | Conv Blocks | Conv per Block | Dense Head | Trainable Params | Hypothesis |
|-------|-------------|----------------|------------|------------------|------------|
| CNN-v1 | 3 (dual) | 2 | 128 | 304,356 | Baseline reference |
| CNN-v2 | 4 (dual) | 2 | 256 | 1,240,420 | H1: Depth yields higher accuracy |
| CNN-v3 | 3 (single) | 1 | 128 | 110,148 | H2: Compact yields comparable accuracy |
| CNN-v4 | 3 (dual) | 2 | 128 | 314,852 | H3: Batch norm placement affects convergence |

### Common Design Principles

- **Conv2D**: 3×3 kernels, same padding, ReLU activation
- **Batch Normalization**: Regularization
- **MaxPooling**: 2×2 with stride 2
- **Dropout**: 0.25 (convolutional blocks), 0.5 (dense head)
- **Global Average Pooling** (GAP) instead of Flatten
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 64
- **Max Epochs**: 50
- **Early Stopping**: Patience=10
- **Learning Rate Reduction**: ReduceLROnPlateau (factor=0.5, patience=5)

---

## 📈 Results

### Overall Performance

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Test Loss | Total Params |
|-------|---------------|-------------------|-----------------|-----------|--------------|
| **CNN-v1** | **44.40%** | 0.5495 | 0.0840 | 1.0554 | 305,252 |
| CNN-v4 | 41.63% | 0.6486 | 0.0020 | 1.0702 | 315,748 |
| CNN-v2 | 41.13% | 0.4297 | 0.0611 | 1.0732 | 1,242,340 |
| CNN-v3 | 40.23% | 0.4202 | 0.1272 | 1.2086 | 110,596 |
| Chance | 25.00% | 0.2500 | 0.2500 | N/A | N/A |

### Per-Class Performance (CNN-v1)

| Emotion Class | Test Samples | Precision | Recall | F1-Score | Interpretation |
|---------------|--------------|-----------|--------|----------|----------------|
| **Angry** | 4,858 (40.9%) | 0.45 | **0.72** | 0.55 | Best performance; majority class bias |
| **Sad** | 4,565 (38.4%) | 0.44 | **0.39** | 0.41 | Moderate; confusion with Angry |
| **Happy** | 2,453 (20.6%) | 0.37 | **0.01** | 0.02 | Near-zero recall |
| **Neutral** | 13 (0.1%) | 0.00 | **0.00** | 0.00 | Completely unlearnable |

### Key Findings

1. **All models outperform chance** (25%) with CNN-v1 achieving +19.4 percentage points
2. **Depth hypothesis (H1) rejected**: CNN-v2 (1.24M params) did not outperform CNN-v1
3. **Compactness hypothesis (H2) verified**: CNN-v3 achieved comparable accuracy with 64% fewer parameters
4. **Class imbalance is the primary limitation**: Zero Neutral recall and near-zero Happy recall across all models
5. **Grad-CAM provides anatomically plausible explanations** for high-confidence Angry and Sad predictions

---

## 🧠 Explainability Module

### Grad-CAM Implementation

The Grad-CAM module (Selvaraju et al., 2017) generates visual heatmaps showing where the CNN "looks" when making a prediction.

**Region Definitions for Analysis:**

| Region | Row Range | Column Range | Emotion Associations |
|--------|-----------|--------------|---------------------|
| Mouth | 0.60–0.85 | 0.30–0.70 | Happy (upturned), Angry (tight), Sad (downturned) |
| Eyes | 0.25–0.45 | 0.20–0.80 | Sad (drooping), Surprise, Fear |
| Eyebrows | 0.15–0.30 | 0.20–0.80 | Angry (furrowed), Sad (inner brow raised) |
| Nose | 0.40–0.60 | 0.40–0.60 | Secondary cue |
| Forehead | 0.05–0.20 | 0.20–0.80 | Secondary cue |
| Chin | 0.80–0.95 | 0.35–0.65 | Secondary cue |

### Natural Language Explanation Templates

| Predicted Emotion | Salient Region | Explanation Text |
|-------------------|----------------|------------------|
| Happy | Mouth | "their MOUTH is smiling" |
| Happy | Eyes | "their EYES are smiling" |
| Sad | Mouth | "their MOUTH is turned down" |
| Sad | Eyes | "their EYES look droopy" |
| Sad | Eyebrows | "their EYEBROWS are pulled up in the middle" |
| Angry | Eyebrows | "their EYEBROWS are pulled down and together" |
| Angry | Mouth | "their MOUTH looks tight or clenched" |
| Angry | Eyes | "their EYES are staring hard" |
| Neutral | Whole Face | "their face looks calm and relaxed" |
| Any (conf < 0.40) | N/A | "I'm not very sure, but I'm trying my best." |

---

## 📁 Repository Structure

```
xai-siss-asd/
├── data/
│   └── expw/                          # ExpW dataset (not included in repo)
│       ├── raw/                       # Original images
│       └── processed/                 # Preprocessed 48x48 grayscale images

## 🚀 Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.2 (recommended for training)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/okechukwu88/xai-siss-asd.git
cd xai-siss-asd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
tensorflow==2.20.0
opencv-python==4.8.1
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
jupyter==1.0.0
pillow==10.0.0
```

---

## 🏃 Usage

### Data Preparation

```python
# Load and preprocess the ExpW dataset
from src.data.dataset import load_expw
from src.data.preprocessing import preprocess_pipeline

# Load dataset (configure path to your ExpW download)
train_data, val_data, test_data = load_expw(
    path='data/expw/raw',
    target_size=(48, 48),
    split_ratio=(0.6, 0.2, 0.2)
)

# Apply preprocessing pipeline
x_train, y_train = preprocess_pipeline(train_data, augment=True)
x_val, y_val = preprocess_pipeline(val_data, augment=False)
x_test, y_test = preprocess_pipeline(test_data, augment=False)
```

### Model Training

```python
# Train CNN-v1
from src.models.architectures import CNN_v1
from src.models.train import train_model

model = CNN_v1(input_shape=(48, 48, 1), num_classes=4)
history = train_model(
    model=model,
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    epochs=50,
    batch_size=64
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Grad-CAM Explanation

```python
# Generate Grad-CAM heatmap
from src.explainability.gradcam import GradCAM
from src.explainability.region_analysis import analyze_regions
from src.explainability.explanations import generate_explanation

# Load trained model
model = load_model('results/models/cnn_v1_best.h5')

# Generate Grad-CAM for a test image
gradcam = GradCAM(model, last_conv_layer='conv3_block2')
heatmap = gradcam.generate(image, predicted_class)

# Analyze facial regions
region_scores = analyze_regions(heatmap)
explanation = generate_explanation(prediction=angry, region='eyebrows', confidence=0.85)
print(explanation)  # "their EYEBROWS are pulled down and together"
```

---

## 📊 Reproducibility

All experiments use **fixed random seeds** for complete reproducibility:

```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

### Training Conditions

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001, beta_1=0.9, beta_2=0.999) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 64 |
| Max Epochs | 50 |
| Early Stopping | Patience=10 (restore best weights) |
| Learning Rate Reduction | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7) |
| Data Augmentation (Training Only) | Horizontal flip (p=0.5), Rotation (±10°), Zoom (±10%) |

---

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@mastersthesis{nicholas2026xaisiss,
  author  = {Nicholas, Okechukwu Sylvester},
  title   = {Designing an Explainable Artificial Intelligence-Based Social Interaction Support System for Children with Autism Spectrum Disorder},
  school  = {University of Wolverhampton},
  year    = {2026},
  type    = {MRes dissertation}
}
```

## Acknowledgements

- **Supervisor:** Hiran Patel, University of Wolverhampton
- **Funding:** University of Wolverhampton Faculty of Science and Engineering
- **Dataset:** ExpW (Zhang et al., 2015)
- **Libraries:** TensorFlow, OpenCV, NumPy, scikit-learn

---

## ⚠️ Ethical Statement

This research was conducted as a **computational feasibility study only**.

- No human participant research was conducted
- No contact was made with children with ASD, their parents/caregivers, or clinical professionals
- The ExpW dataset was collected from public internet searches and carries inherent label noise
- The current system is **NOT suitable for clinical deployment** or any form of participant research

**Minimum performance threshold** for future participant research (to be achieved before any human evaluation):

- Per-class recall > 0.6 for all target emotions
- No class with recall = 0
- Calibrated confidence with Expected Calibration Error < 0.05

---

## 📞 Contact

**Author:** Okechukwu Sylvester Nicholas

- **Email:** [O.s.nicholas@wlv.ac.uk](mailto:O.s.nicholas@wlv.ac.uk)
- **GitHub:** [@okechukwu88](https://github.com/okechukwu88)

---

## 🔗 Related Links

- [ExpW Dataset on Kaggle](https://www.kaggle.com/datasets/...)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

**Last Updated:** July 2026
