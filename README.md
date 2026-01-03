# 🧠 Multimodal Fusion of MRI and Cognitive Assessments Using CNN-LSTM for Alzheimer's Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/garvita-d/ADNI-Multimodal-CNN-LSTM?style=social)](https://github.com/garvita-d/ADNI-Multimodal-CNN-LSTM/stargazers)

---

## 📘 Overview

This project presents a **multimodal deep learning framework** that integrates 3-Tesla structural MRI data with longitudinal cognitive assessments (MMSE) from the **ADNI dataset** to predict Alzheimer's Disease progression across three stages: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD).

Our CNN-LSTM fusion architecture achieves **64.9% ± 15.7%** balanced accuracy and includes comprehensive **Explainable AI (XAI)** features for clinical interpretability through Grad-CAM, attention mechanisms, and feature importance analysis.

### 🎯 Key Features

- 🧠 **3D CNN** for spatial MRI feature extraction (128-dimensional embeddings)
- 📊 **Bi-LSTM** for temporal cognitive trajectory modeling
- 🔗 **Multimodal Fusion** combining imaging + cognitive data
- 🔍 **Explainable AI** (Grad-CAM, attention visualization, Integrated Gradients)
- ✅ **5-Fold Cross-Validation** with early stopping and class balancing
- 📈 **Comprehensive Evaluation** with confusion matrices and per-class metrics

---

## 🏗️ Architecture

```
┌─────────────────┐
│   MRI Scan      │──► [3D CNN] ──────────────┐
│  (64³ voxels)   │    4 conv blocks          │
└─────────────────┘    ↓ 128-dim              │
                                              ├──► [Fusion Layer] ──► [Classifier]
┌─────────────────┐                           │     (224-dim)         (3 classes)
│  MMSE Series    │──► [Bi-LSTM] ─────────────┤       ↓               CN/MCI/AD
│  (5 timepoints) │    2-layer + attention    │    [Dense]
└─────────────────┘    ↓ 64-dim               │    [BN+ReLU]
                                              │    [Dropout]
┌─────────────────┐                           │       ↓
│Static Cognitive │──► [MLP] ──────────────────┘    [3-class]
│  (11 features)  │    2-layer FC                   softmax
└─────────────────┘    ↓ 32-dim
```

**Architecture Components:**

| Module | Architecture | Output |
|--------|-------------|---------|
| **3D CNN** | Conv3D(32→64→128→256) + BatchNorm + MaxPool | 128-dim |
| **Bi-LSTM** | 2-layer bidirectional + attention mechanism | 64-dim |
| **Static MLP** | FC(64) + ReLU + Dropout + FC(32) | 32-dim |
| **Fusion** | Concat(224) → FC(128) → BN → FC(64) → FC(3) | 3 classes |

---

## 📊 Results Summary

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Mean Balanced Accuracy** | **0.649 ± 0.157** |
| **Best Fold Accuracy** | 0.822 |
| **Overall Accuracy** | 60.0% |
| **CN Precision** | 0.88 |
| **AD Recall** | 1.00 |

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **CN (Cognitively Normal)** | 0.88 | 0.47 | 0.61 | 19 |
| **MCI (Mild Cognitive Impairment)** | 0.35 | 0.39 | 0.37 | 33 |
| **AD (Alzheimer's Disease)** | 0.39 | 1.00 | 0.56 | 7 |

**Key Findings:**
- ✅ High precision for CN detection (reduces false positives)
- ⚠️ MCI classification remains challenging (class imbalance)
- ✅ Perfect recall for AD (no missed AD cases)

---

## 📁 Repository Structure

```
ADNI-Multimodal-CNN-LSTM/
│
├── README.md                      # Project overview and documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/
│   └── README.md                  # Dataset access instructions (ADNI)
│
├── notebooks/
│   ├── 1_preprocessing.ipynb      # Data loading & exploration
│   ├── 2_model_training.ipynb     # Training pipeline with cross-validation
│   └── 3_evaluation_with_xai.ipynb # XAI analysis & visualization
│
├── src/
│   ├── __init__.py                # Package initialization
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_module.py          # 3D CNN architecture (CNN3D)
│   │   ├── lstm_module.py         # Bi-LSTM + Static MLP modules
│   │   └── fusion_model.py        # Multimodal fusion model
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py  # MRI & cognitive data preprocessing
│   │   └── dataset.py             # PyTorch Dataset class
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py               # Training utilities & samplers
│   │   └── validate.py            # Validation functions
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py             # Evaluation metrics & plotting
│       └── xai_utils.py           # XAI visualization (Grad-CAM, Attention)
│
├── scripts/
│   └── extract_zip.py             # ADNI data extraction utility
│
├── reports/
│   ├── Report_on_Multimodal_Fusion.pdf  
│   └── figures/                   # Generated visualizations
│       ├── confusion_matrix.png
│       ├── gradcam_examples.png
│       └── architecture_diagram.png
│
|
```

---

## 🔗 Quick Navigation

| Resource | Path | Description |
|----------|------|-------------|
| 📄 **Research Paper** | [`reports/Report_on_Multimodal_Fusion.pdf`](reports/Report_on_Multimodal_Fusion_of_MRI_and_Cognitive_Assessments_Using_CNN[1].pdf) | Complete methodology & results |
| 📓 **Data Preprocessing** | [`notebooks/1_preprocessing.ipynb`](notebooks/1_preprocessing.ipynb) | Load and explore ADNI data |
| 🏋️ **Model Training** | [`notebooks/2_model_training.ipynb`](notebooks/2_model_training.ipynb) | 5-fold cross-validation pipeline |
| 🔍 **XAI Analysis** | [`notebooks/3_evaluation_with_xai.ipynb`](notebooks/3_evaluation_with_xai.ipynb) | Grad-CAM & attention visualization |
| 🏗️ **Model Code** | [`src/models/fusion_model.py`](src/models/fusion_model.py) | Multimodal fusion architecture |
| 💾 **Dataset Guide** | [`data/README.md`](data/README.md) | How to obtain ADNI data |
| 🛠️ **Preprocessing** | [`src/preprocessing/data_preprocessing.py`](src/preprocessing/data_preprocessing.py) | MRI & cognitive processing |
| 📊 **Evaluation** | [`src/evaluation/metrics.py`](src/evaluation/metrics.py) | Metrics & confusion matrices |

---

## 📊 Dataset

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

**⚠️ IMPORTANT: Data Access Required**

This project uses **restricted data** from ADNI. You must obtain access independently:

1. **Apply for Access:** [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
2. Complete the Data Use Agreement
3. Wait for approval (typically 1-2 weeks)
4. Download required files (see [`data/README.md`](data/README.md))

**Required Files:**
```
data/
├── ADNI1_Annual 2 Yr 3T.zip       # MRI scans (3-Tesla T1-weighted)
├── MMSE_20Nov2025.csv             # Cognitive assessments
├── DXSUM_20Nov2025.csv            # Diagnostic labels
└── ADNI1_Annual_2_Yr_3T_11_20_2025.csv  # Metadata
```

**Dataset Statistics:**
- **Subjects:** 70 (after quality control)
- **MRI Modality:** 3T T1-weighted MPRAGE
- **Classes:** CN (19), MCI (33), AD (7)
- **Cognitive Features:** 11 baseline + longitudinal MMSE
- **Time Points:** Up to 5 visits per subject

**Why ADNI Data is Not Included:**
- Protected by strict privacy agreements
- Cannot be redistributed publicly
- Requires individual researcher approval

**Alternative Options:**
- [OASIS Brain Dataset](https://www.oasis-brains.org/) (publicly available)
- Synthetic demo data for pipeline testing

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 16GB+ RAM

### 1️⃣ Clone Repository

```bash
git clone https://github.com/garvita-d/ADNI-Multimodal-CNN-LSTM.git
cd ADNI-Multimodal-CNN-LSTM
```

### 2️⃣ Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# OR using conda
conda create -n adni-cnn-lstm python=3.8
conda activate adni-cnn-lstm
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=2.0.0` - Deep learning framework
- `torchio>=0.18.0` - Medical image augmentation
- `nibabel>=3.2.0` - NIfTI file handling
- `captum>=0.6.0` - Explainable AI tools
- `scikit-learn>=1.0.0` - Evaluation metrics
- `numpy`, `pandas`, `matplotlib`, `seaborn`

### 4️⃣ Download ADNI Data

1. Access [ADNI website](https://adni.loni.usc.edu/)
2. Download required files (see Dataset section)
3. Place files in `data/` folder

```bash
cd data/
# Extract MRI scans
python ../scripts/extract_zip.py
```

### 5️⃣ Verify Setup

```bash
# Test imports
python -c "from src.models import MultimodalFusionModel; print('✓ Import successful')"

# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🧪 Usage

### Quick Start with Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/1_preprocessing.ipynb      - Data exploration
# 2. notebooks/2_model_training.ipynb     - Training pipeline
# 3. notebooks/3_evaluation_with_xai.ipynb - XAI analysis
```

### Using Source Code Directly

**1. Preprocess Data:**
```python
from src.preprocessing import load_cognitive_data, build_dataset_records, get_cognitive_features

# Load CSV data
merged_df, mmse_df = load_cognitive_data(
    'data/MMSE_20Nov2025.csv',
    'data/DXSUM_20Nov2025.csv'
)

# Get cognitive features
cog_features = get_cognitive_features(merged_df)

# Build dataset records
records = build_dataset_records(
    'data/ADNI',
    merged_df,
    mmse_df,
    cog_features
)
print(f"Created {len(records)} records")
```

**2. Create Dataset:**
```python
from src.preprocessing import ADNIDataset, get_train_transforms
from torch.utils.data import DataLoader

# Create dataset with augmentation
train_transform = get_train_transforms()
dataset = ADNIDataset(records, transform=train_transform, preload=True)

# Create data loader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

**3. Initialize Model:**
```python
from src.models import MultimodalFusionModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalFusionModel(
    static_cog_size=11,  # Number of cognitive features
    num_classes=3         # CN, MCI, AD
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**4. Train Model:**
```python
from src.training import train_epoch, validate, get_class_weights
import torch.optim as optim
import torch.nn as nn

# Setup training
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=get_class_weights(records, device=device))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training loop
for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_bal_acc, preds, labels = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, BalAcc={val_bal_acc:.3f}")
```

**5. XAI Analysis:**
```python
from src.evaluation import compute_gradcam_3d, visualize_gradcam_3d

# Compute Grad-CAM for a sample
mri_sample = next(iter(val_loader))['mri'][:1].to(device)
cam = compute_gradcam_3d(model, mri_sample, target_class=2)  # AD class

# Visualize
visualize_gradcam_3d(mri_sample, cam, save_path='reports/figures/gradcam_ad.png')
```

### Key Hyperparameters

```python
# Training configuration
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
PATIENCE = 15

# Data configuration
TARGET_SHAPE = (64, 64, 64)  # MRI dimensions
MAX_SEQ_LEN = 5               # MMSE time series length
```

---

## 🔬 Model Components

### 1. 3D CNN Module (`src/models/cnn_module.py`)

```python
from src.models import CNN3D

cnn = CNN3D(out_features=128)
# Input: (batch, 1, 64, 64, 64)
# Output: (batch, 128)
```

**Architecture:**
- Conv3D → BatchNorm → ReLU → MaxPool (32 filters)
- Conv3D → BatchNorm → ReLU → MaxPool (64 filters)
- Conv3D → BatchNorm → ReLU → MaxPool (128 filters)
- Conv3D → BatchNorm → ReLU (256 filters)
- AdaptiveAvgPool3D → FC(256→128) → Dropout(0.4)

### 2. Bi-LSTM Module (`src/models/lstm_module.py`)

```python
from src.models import CognitiveLSTM, StaticCogNet

lstm = CognitiveLSTM(input_size=1, hidden_size=64, out_features=64)
static_net = StaticCogNet(input_size=11, out_features=32)
# LSTM Input: (batch, seq_len, 1)
# Static Input: (batch, 11)
```

**LSTM Architecture:**
- 2-layer Bidirectional LSTM (hidden=64)
- Attention mechanism (stored for visualization)
- FC(128→64)

**Static MLP:**
- FC(11→64) → ReLU → Dropout(0.3)
- FC(64→32) → ReLU

### 3. Fusion Model (`src/models/fusion_model.py`)

```python
from src.models import MultimodalFusionModel

model = MultimodalFusionModel(static_cog_size=11, num_classes=3)
# Combines: CNN(128) + LSTM(64) + Static(32) = 224
# Output: (batch, 3) - class logits
```

---

## 🔍 Explainable AI (XAI) Features

### Available XAI Tools

| Tool | Module | Purpose |
|------|--------|---------|
| **Grad-CAM** | `src/evaluation/xai_utils.py` | Visualize CNN attention on MRI |
| **LSTM Attention** | `src/evaluation/xai_utils.py` | Temporal importance weights |
| **Integrated Gradients** | `src/evaluation/xai_utils.py` | Feature importance scores |
| **Confusion Matrix** | `src/evaluation/metrics.py` | Classification performance |

### 1. Grad-CAM Visualization

```python
from src.evaluation import compute_gradcam_3d, visualize_gradcam_3d

# Generate heatmap
cam = compute_gradcam_3d(model, mri_input, target_class=2)  # AD

# Visualize 3 views
visualize_gradcam_3d(
    mri_input, 
    cam, 
    slice_idx=None,  # Auto-select middle slice
    save_path='reports/figures/gradcam_ad.png'
)
```

**Output:** Shows hippocampal atrophy, ventricular enlargement

### 2. LSTM Attention Analysis

```python
from src.evaluation import visualize_lstm_attention

# Get attention from forward pass
attention = model.lstm.attention_weights  # Stored automatically

# Visualize temporal patterns
visualize_lstm_attention(
    attention[0],          # First sample
    mmse_scores=[28, 26, 24, 21, 19],
    save_path='reports/figures/lstm_attention.png'
)
```

**Output:** Attention peaks indicate rapid cognitive decline periods

### 3. Feature Importance

```python
from src.evaluation import compute_feature_importance, plot_feature_importance

# Compute importance scores
importance = compute_feature_importance(model, val_loader, device)

# Plot rankings
plot_feature_importance(
    importance,
    feature_names=['MMSCORE', 'MMDATE', 'MMYEAR', ...],
    save_path='reports/figures/feature_importance.png'
)
```

**Output:** Ranks cognitive features by contribution to predictions

---

## 📈 Evaluation Metrics

### Available Metrics (`src/evaluation/metrics.py`)

```python
from src.evaluation import (
    plot_confusion_matrix,
    print_classification_metrics,
    plot_cv_results
)

# Confusion matrix
plot_confusion_matrix(
    y_true, 
    y_pred, 
    class_names=['CN', 'MCI', 'AD'],
    save_path='reports/confusion_matrix.png'
)

# Detailed metrics
print_classification_metrics(y_true, y_pred, class_names=['CN', 'MCI', 'AD'])

# Cross-validation results
plot_cv_results(
    fold_scores=[0.822, 0.715, 0.612, 0.573, 0.524],
    save_path='reports/figures/cv_results.png'
)
```

### Cross-Validation Results

```
Fold 1: 0.822 ✓ (Best)
Fold 2: 0.715
Fold 3: 0.612
Fold 4: 0.573
Fold 5: 0.524

Mean: 0.649 ± 0.157
```

---

## 🛠️ Scripts & Utilities

### Extract ADNI Data (`scripts/extract_zip.py`)

```bash
# Automatically extract all ADNI zip files
python scripts/extract_zip.py

# Or customize paths
python -c "
from scripts.extract_zip import extract_adni_zips
extract_adni_zips(
    zip_paths=['data/ADNI1_Annual 2 Yr 3T.zip'],
    output_dir='data/'
)
"
```

---

## 👥 Team

**Research Team:**
- **Anya Kalluri** (SE22UCSE033)
- **Niharika Dundigalla** (SE22UCSE087) 
- **Garvita Dalmia** (SE22UCSE099)

**Institution:** Mahindra University, Hyderabad, India  
**Department:** School of Engineering  
**Academic Year:** 2024-2025

---

## 📝 Citation

**ADNI Data Citation:**
```
Data used in preparation of this article were obtained from the 
Alzheimer's Disease Neuroimaging Initiative (ADNI) database 
(adni.loni.usc.edu). As such, the investigators within the ADNI 
contributed to the design and implementation of ADNI and/or provided 
data but did not participate in analysis or writing of this report.
```

---

## 🔮 Future Work

- [ ] Expand dataset size (target: 200+ subjects)
- [ ] Implement attention-based fusion (transformer architecture)
- [ ] Add multi-modal data (PET scans, CSF biomarkers, genetics)
- [ ] Longitudinal progression prediction (time-to-conversion)
- [ ] Transfer learning from pre-trained medical imaging models
- [ ] Federated learning for multi-site collaboration
- [ ] Clinical deployment interface (DICOM integration)

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors in Notebooks:**
```python
# Add to first cell of notebook
import sys
sys.path.append('..')  # Add parent directory to path
```

**Out of Memory Errors:**
```python
# Reduce batch size or MRI resolution
BATCH_SIZE = 2
TARGET_SHAPE = (48, 48, 48)
```

**GPU Not Detected:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Data Files Not Found:**
```bash
# Verify directory structure
ls -R data/
# Should show: ADNI/, *.csv files
```

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

**Key Points:**
- ✅ Free to use, modify, and distribute
- ✅ Commercial use allowed
- ⚠️ ADNI data subject to separate Data Use Agreement
- ⚠️ No warranty or liability

---

## 🙏 Acknowledgments

### Data & Resources
- **ADNI:** Data collection funded by NIH, DOD, pharmaceutical companies
- **PyTorch Team:** Deep learning framework
- **TorchIO:** Medical image augmentation library
- **Captum:** Explainable AI toolkit

### Inspiration
- Korolev et al. (2017) - 3D CNN for AD classification
- Suk et al. (2014) - Multimodal latent feature representation
- Selvaraju et al. (2017) - Grad-CAM visual explanations

---

## 📧 Contact

**Primary Contact:**
- **Garvita Dalmia**
- 📧 Email: se22ucse099@mahindrauniversity.edu.in
- 🔗 GitHub: [@garvita-d](https://github.com/garvita-d)

**Team Members:**
- Anya Kalluri: se22ucse033@mahindrauniversity.edu.in
- Niharika Dundigalla: se22ucse087@mahindrauniversity.edu.in

**Project Links:**
- 🐛 [Report Issues](https://github.com/garvita-d/ADNI-Multimodal-CNN-LSTM/issues)
- 💬 [Discussions](https://github.com/garvita-d/ADNI-Multimodal-CNN-LSTM/discussions)

---

## ⚠️ Disclaimer

**FOR RESEARCH PURPOSES ONLY**

This is an academic research project and is **NOT validated for clinical use**. Always consult qualified healthcare professionals for medical decisions.

---

<div align="center">

**Made with ❤️ by the Mahindra University ML Team**

⭐ Star this repo if you find it useful!

[⬆ Back to Top](#-multimodal-fusion-of-mri-and-cognitive-assessments-using-cnn-lstm-for-alzheimers-disease-prediction)

</div>

---

**Last Updated:** January 2025  
**Version:** 1.0.0
