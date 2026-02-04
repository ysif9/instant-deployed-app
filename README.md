# Brain Tumor Segmentation - INSTANT-ODC AI Hackathon

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Can AI assist radiologists in detecting gliomas?**

Automated segmentation of brain tumor sub-regions from 3D MRI scans using deep learning.

---

## 🎯 Challenge Overview

This project tackles the **Brain Tumor Segmentation Challenge**, focusing on automatically identifying and segmenting three critical sub-regions of gliomas from multi-parametric MRI scans:

- **Necrotic Tumor Core (NCR)** - Dead tissue inside the tumor
- **Peritumoral Edema (ED)** - Swelling surrounding the tumor  
- **GD-enhancing Tumor (ET)** - Active, growing part of the tumor

### Why This Matters

Gliomas are the most common primary brain malignancies with varying degrees of aggressiveness and prognosis. Accurate segmentation of tumor sub-regions is crucial for:
- Treatment planning
- Monitoring disease progression
- Surgical guidance
- Radiotherapy planning

---

## 📊 Dataset

The dataset is derived from the **BraTS (Brain Tumor Segmentation) Challenge** and consists of multi-parametric MRI (mpMRI) scans.

### MRI Modalities

For each patient, **4 co-registered MRI modalities** are provided:

1. **T1** - Native T1-weighted: Structural analysis
2. **T1ce** - Post-contrast T1-weighted: Highlights enhancing tumor (active cells)
3. **T2** - T2-weighted: Shows brain outline and fluids
4. **FLAIR** - T2 Fluid Attenuated Inversion Recovery: Critical for detecting edema

### Label Information

Ground truth masks contain the following labels:

| Label ID | Region | Description |
|----------|--------|-------------|
| **1** | NCR | Necrotic Tumor Core |
| **2** | ED | Peritumoral Edema |
| **4** | ET | GD-enhancing Tumor |

> ⚠️ **Important:** There is **no Label 3**. You may need to remap labels during preprocessing (e.g., 4 → 3) for training, but ensure submission format uses the original label IDs (1, 2, 4).

---

## 📈 Evaluation Metric

Submissions are evaluated using the **Mean Dice Coefficient**, which measures the overlap between predicted and ground truth masks.

### Dice Coefficient Formula

```
Dice = (2 × |X ∩ Y|) / (|X| + |Y|)
```

Where:
- `X` = Predicted set of pixels
- `Y` = Ground truth
- Score range: **0.0** (no overlap) to **1.0** (perfect match)

---

## 📤 Submission Format

Submissions use **Run-Length Encoding (RLE)** to compress 3D masks efficiently.

### CSV Structure

Your submission CSV must contain two columns: `id` and `rle`

Each patient requires **3 rows** (one per class):
- `{PatientID}_1` - Necrotic Core prediction
- `{PatientID}_2` - Edema prediction
- `{PatientID}_4` - Enhancing Tumor prediction

### Example Submission

```csv
id,rle
BraTS2021_00001_1,1 40 55 10 ...
BraTS2021_00001_2,1 1
BraTS2021_00001_4,40 10 100 20 ...
BraTS2021_00002_1,5 30 60 15 ...
BraTS2021_00002_2,2 5
BraTS2021_00002_4,45 12 105 25 ...
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch / TensorFlow
NumPy, Pandas
nibabel (for NIfTI file handling)
scikit-learn
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd instant-deplyed-app

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
instant-deplyed-app/
├── context/
│   └── description.md      # Competition description
├── data/                   # Dataset directory
│   ├── train/             # Training data
│   └── test/              # Test data
├── models/                # Model architectures
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── training/          # Training scripts
│   └── inference/         # Inference and submission generation
├── submissions/           # Generated submission files
└── README.md
```

---

## 🧠 Approach

### Recommended Pipeline

1. **Data Preprocessing**
   - Load NIfTI files using nibabel
   - Normalize MRI intensities
   - Handle label remapping if needed
   - Data augmentation (rotation, flipping, elastic deformation)

2. **Model Architecture**
   - 3D U-Net or V-Net for volumetric segmentation
   - Attention mechanisms for better feature extraction
   - Multi-scale feature fusion

3. **Training Strategy**
   - Loss function: Dice Loss + Cross-Entropy
   - Optimizer: Adam or AdamW
   - Learning rate scheduling
   - Cross-validation for robust evaluation

4. **Post-processing**
   - Connected component analysis
   - Morphological operations
   - Ensemble predictions

5. **Submission Generation**
   - Convert predictions to RLE format
   - Validate submission format

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BraTS Challenge organizers for the dataset
- INSTANT-ODC AI Hackathon organizers
- Medical imaging research community

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Good luck with the challenge! 🎉**

