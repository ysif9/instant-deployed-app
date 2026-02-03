# 🧠 BraTS Brain Tumor Segmentation - Streamlit Application

A comprehensive web application for brain tumor segmentation inference using a 3D U-Net model trained on the BraTS dataset.

## ✨ Features

### 🎯 Core Functionality
- **Multi-Modal MRI Processing**: Upload and process 4 MRI modalities (T1, T1ce, T2, FLAIR)
- **Flexible File Format Support**: Accepts both `.nii` and `.nii.gz` files automatically
- **Advanced Inference Pipeline**: 
  - Sliding window inference with Gaussian weighting
  - Optional Test-Time Augmentation (TTA) for improved accuracy
  - Configurable overlap ratio for boundary consistency
- **Post-Processing**:
  - Small component removal
  - Hole filling
  - Tumor hierarchy enforcement (ET ⊆ TC ⊆ WT)
  - Optional erosion for boundary refinement

### 📊 Visualization
- **2D Slice Viewer**: Navigate through axial, sagittal, and coronal views
- **3D Interactive Visualization**: Rotate, zoom, and explore tumor segmentation in 3D using Plotly
- **Color-Coded Classes**:
  - 🔴 Class 1 (Necrotic Core): Red (#FF6B6B)
  - 🔵 Class 2 (Edema): Cyan (#4ECDC4)
  - 🟡 Class 4 (Enhancing Tumor): Yellow (#FFE66D)

### 💾 Export Options
- **NIfTI Format**: Download segmentation masks as `.nii.gz` files
- **RLE CSV Format**: Competition-ready submission format with run-length encoding

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd instant-deplyed-app

# Install dependencies
uv sync
```

### Running the App

```bash
# Start Streamlit
streamlit run app.py

# Or with uv
uv run streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📖 Usage Guide

### 1. Configure Model Path
- In the sidebar, specify the path to your trained model weights
- Default: `model/best_model.pth`

### 2. Upload MRI Scans
Upload all 4 required NIfTI files (`.nii` or `.nii.gz`):
- **FLAIR**: T2 Fluid Attenuated Inversion Recovery
- **T1**: Native T1-weighted
- **T1ce**: Post-contrast T1-weighted
- **T2**: T2-weighted

### 3. Adjust Settings (Optional)
Expand "🔧 Advanced Settings" in the sidebar:

**Thresholds** (per-class probability thresholds):
- Tumor Core (TC): 0.52
- Whole Tumor (WT): 0.47
- Enhancing Tumor (ET): 0.57

**Inference Options**:
- Test-Time Augmentation (TTA): Enabled by default
- Sliding Window Overlap: 0.6

**Post-processing**:
- Min Component Size: 150 voxels
- Fill Holes: Enabled
- Enforce Hierarchy: Enabled
- Apply Erosion: Disabled

### 4. Run Segmentation
- Click "🚀 Run Segmentation"
- Monitor progress bar
- Wait for completion (2-5 minutes on GPU, longer on CPU)

### 5. Explore Results

**📈 Statistics**: View voxel counts for each class

**🔍 Visualization**:
- **2D Slice Viewer**: Navigate slices in different planes
- **3D Visualization**: Interactive 3D tumor rendering

### 6. Download Results
- **📥 Download Segmentation (NIfTI)**: Combined mask file
- **📥 Download RLE CSV**: Competition submission format

## 🏗️ Model Architecture

**3D U-Net** with:
- Instance Normalization
- LeakyReLU activation (α=0.01)
- 6 encoder/decoder levels
- Feature channels: [64, 96, 128, 192, 256, 384]
- Output: 3 channels (TC, WT, ET)

## 🔧 Pipeline Details

### Preprocessing
1. Load 4 MRI modalities (auto-detects `.nii` or `.nii.gz`)
2. Concatenate into 4-channel input
3. Orient to RAS coordinate system
4. Resample to 1mm³ isotropic spacing
5. Normalize intensity (channel-wise, non-zero voxels)
6. Crop foreground
7. Pad to 128³ patch size

### Inference
1. Sliding window inference (128³ patches, 60% overlap)
2. Gaussian weighting for smooth blending
3. Optional TTA (flip augmentations on 3 axes)
4. Sigmoid activation → probabilities

### Post-processing
1. Apply per-class thresholds
2. Remove padding
3. Enforce hierarchy (ET ⊆ TC ⊆ WT)
4. Remove small components (< 150 voxels)
5. Fill holes
6. Re-enforce hierarchy
7. Convert RAS → LPS orientation
8. Resample to original shape
9. Derive final classes

## 🎯 Tumor Classes

- **Class 1 (NCR - Necrotic Core)**: Dead tissue = TC - ET
- **Class 2 (ED - Edema)**: Swelling = WT - TC  
- **Class 4 (ET - Enhancing Tumor)**: Active tumor

## 🐛 Troubleshooting

### File Format Issues
✅ **Fixed**: App now auto-detects and handles both `.nii` and `.nii.gz`

### Download Permission Error (Windows)
✅ **Fixed**: Improved temporary file handling to avoid file locks

### Model Loading
- Verify model path is correct
- Ensure `.pth` file is valid PyTorch checkpoint
- App uses `strict=False` for compatibility

### Memory Issues
- Reduce batch size (default: 2)
- Disable TTA
- Use CPU if GPU OOM

### Slow Inference
- Enable CUDA if available
- Reduce overlap (faster, slightly lower quality)
- Disable TTA

## 📦 Dependencies

- Python ≥ 3.12
- PyTorch ≥ 2.10.0
- MONAI ≥ 1.5.2
- Streamlit ≥ 1.40.0
- nibabel ≥ 5.3.3
- matplotlib ≥ 3.7.0
- plotly ≥ 5.14.0
- scipy, scikit-image, pandas, numpy, tqdm

## 📝 Recent Updates

### v1.1 (Latest)
- ✅ Fixed file extension handling for both `.nii` and `.nii.gz`
- ✅ Fixed Windows file permission error in downloads
- ✅ Improved temporary file management
- ✅ Enhanced error handling

## 📂 Project Structure

```
instant-deplyed-app/
├── app.py                 # Main Streamlit application
├── verify_app.py          # Dependency verification script
├── pyproject.toml         # Project dependencies
├── README_APP.md          # This file
├── model/
│   └── best_model.pth     # Trained model weights
├── data/                  # Sample MRI data
└── context/               # Reference implementation
```

## 🙏 Credits

Based on the BraTS Challenge with enhancements:
- Correct RLE encoding (C-order, 1-indexed)
- Proper orientation (RAS → LPS)
- Optimized thresholds
- Advanced post-processing
- Test-Time Augmentation

---

**License**: Educational and research use

**Questions?** Check the troubleshooting section or review the inline code comments.

