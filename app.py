"""
BraTS Brain Tumor Segmentation - Streamlit Web Application

This application provides an interactive interface for brain tumor segmentation inference
using a trained 3D U-Net model. It processes 4 MRI modalities (T1, T1ce, T2, FLAIR) and
segments 3 tumor sub-regions: Necrotic Core (label 1), Edema (label 2), and Enhancing Tumor (label 4).

Usage:
    streamlit run app.py

Features:
    - Upload 4 NIfTI files (T1, T1ce, T2, FLAIR)
    - Configure inference parameters (thresholds, TTA, overlap, etc.)
    - Run model inference with progress tracking
    - Interactive 3D visualization and slice-by-slice viewing
    - Download segmentation results (NIfTI and RLE CSV formats)
"""

import os
import tempfile
import glob
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom, label as scipy_label, binary_fill_holes, binary_erosion, center_of_mass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, DeleteItemsd,
    Orientationd, Spacingd, NormalizeIntensityd, CropForegroundd, SpatialPadd,
)
from monai.inferers import SlidingWindowInferer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="BraTS Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]
ORIGINAL_SHAPE = (240, 240, 155)
CLASS_NAMES = {1: "Necrotic Core (NCR)", 2: "Edema (ED)", 4: "Enhancing Tumor (ET)"}
CLASS_COLORS = {1: "#FF6B6B", 2: "#4ECDC4", 4: "#FFE66D"}  # Red, Cyan, Yellow

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    in_channels: int = 4
    out_channels: int = 3
    sw_batch_size: int = 2
    overlap: float = 0.6
    use_amp: bool = True
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Per-class thresholds
    threshold_tc: float = 0.52
    threshold_wt: float = 0.47
    threshold_et: float = 0.57

    # Post-processing
    min_component_size: int = 150
    fill_holes: bool = True
    enforce_hierarchy: bool = True
    apply_erosion: bool = False
    erosion_iterations: int = 0

    # Test-Time Augmentation
    use_tta: bool = True
    tta_flips: List[int] = None

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, True)

    def forward(self, x):
        return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(x))))))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            d = [skip.shape[i+2] - x.shape[i+2] for i in range(3)]
            x = F.pad(x, [d[2]//2, d[2]-d[2]//2, d[1]//2, d[1]-d[1]//2, d[0]//2, d[0]-d[0]//2])
        return self.conv(torch.cat([x, skip], 1))

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        f = [64, 96, 128, 192, 256, 384]
        self.encoders = nn.ModuleList([DownBlock(4 if i==0 else f[i-1], c) for i, c in enumerate(f[:-1])])
        self.bottleneck = ConvBlock(f[-2], f[-1])
        self.decoders = nn.ModuleList([UpBlock(f[-1] if i==0 else f[-2-i+1], f[-2-i], f[-2-i]) for i in range(len(f)-1)])
        self.output_head = nn.Conv3d(f[0], 3, 1)

    def forward(self, x):
        skips = []
        for e in self.encoders:
            x, s = e(x)
            skips.append(s)
        x = self.bottleneck(x)
        for d, s in zip(self.decoders, skips[::-1]):
            x = d(x, s)
        return self.output_head(x)

# ============================================================================
# RLE ENCODING/DECODING
# ============================================================================

def rle_encode_c_order(mask: np.ndarray) -> str:
    """
    RLE encoding using C-order (row-major), 1-indexed.

    Args:
        mask: 3D binary numpy array (240, 240, 155)

    Returns:
        RLE string: "start1 length1 start2 length2 ..."
    """
    flat = mask.flatten(order='C')

    if flat.sum() == 0:
        return ""

    flat = np.concatenate([[0], flat, [0]])
    runs = np.where(flat[1:] != flat[:-1])[0]
    runs = runs.reshape(-1, 2)
    starts = runs[:, 0] + 1  # Convert to 1-indexed
    lengths = runs[:, 1] - runs[:, 0]

    rle_pairs = [f"{s} {l}" for s, l in zip(starts, lengths)]
    return " ".join(rle_pairs)

def rle_decode_c_order(rle_string: str, shape: tuple = (240, 240, 155)) -> np.ndarray:
    """Decode RLE string to 3D mask (C-order, 1-indexed)."""
    if not rle_string or rle_string.strip() == '':
        return np.zeros(shape, dtype=np.uint8)

    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    rle_pairs = rle_string.strip().split()

    for i in range(0, len(rle_pairs), 2):
        start = int(rle_pairs[i]) - 1  # Convert to 0-indexed
        length = int(rle_pairs[i + 1])
        mask[start:start + length] = 1

    return mask.reshape(shape, order='C')

# ============================================================================
# PDF REPORT GENERATION - HELPER FUNCTIONS
# ============================================================================

def calculate_volumes(masks: Dict[int, np.ndarray], voxel_size_mm3: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate tumor volumes in mm³ and cm³.

    Args:
        masks: Dict mapping class labels to 3D mask arrays
        voxel_size_mm3: Volume of a single voxel in mm³ (default: 1.0mm × 1.0mm × 1.0mm = 1.0mm³)

    Returns:
        Dict with volume information for each class and composite regions
    """
    volumes = {}

    # Individual classes
    for class_id, mask in masks.items():
        voxel_count = mask.sum()
        volume_mm3 = voxel_count * voxel_size_mm3
        volume_cm3 = volume_mm3 / 1000.0

        class_name = CLASS_NAMES[class_id]
        volumes[class_name] = {
            'voxels': int(voxel_count),
            'mm3': float(volume_mm3),
            'cm3': float(volume_cm3)
        }

    # Composite regions
    # Tumor Core (TC) = Class 1 + Class 4
    tc_mask = (masks[1] | masks[4]).astype(np.uint8)
    tc_voxels = tc_mask.sum()
    tc_mm3 = tc_voxels * voxel_size_mm3
    tc_cm3 = tc_mm3 / 1000.0

    volumes['Tumor Core (TC)'] = {
        'voxels': int(tc_voxels),
        'mm3': float(tc_mm3),
        'cm3': float(tc_cm3)
    }

    # Whole Tumor (WT) = Class 1 + Class 2 + Class 4
    wt_mask = (masks[1] | masks[2] | masks[4]).astype(np.uint8)
    wt_voxels = wt_mask.sum()
    wt_mm3 = wt_voxels * voxel_size_mm3
    wt_cm3 = wt_mm3 / 1000.0

    volumes['Whole Tumor (WT)'] = {
        'voxels': int(wt_voxels),
        'mm3': float(wt_mm3),
        'cm3': float(wt_cm3)
    }

    return volumes

def find_max_area_slice(masks: Dict[int, np.ndarray], axis: int) -> int:
    """
    Find the slice with the largest total tumor area along a given axis.

    Args:
        masks: Dict mapping class labels to 3D mask arrays
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)

    Returns:
        Index of the slice with maximum tumor area
    """
    # Combine all tumor classes
    combined_mask = np.zeros_like(masks[1], dtype=np.uint8)
    for mask in masks.values():
        combined_mask = combined_mask | mask

    # Calculate area for each slice
    num_slices = combined_mask.shape[axis]
    areas = []

    for i in range(num_slices):
        if axis == 0:  # Sagittal
            slice_mask = combined_mask[i, :, :]
        elif axis == 1:  # Coronal
            slice_mask = combined_mask[:, i, :]
        else:  # Axial
            slice_mask = combined_mask[:, :, i]

        areas.append(slice_mask.sum())

    # Return index of slice with maximum area
    max_idx = int(np.argmax(areas))
    return max_idx

def determine_lateralization(masks: Dict[int, np.ndarray], volume_shape: Tuple[int, int, int] = (240, 240, 155)) -> str:
    """
    Determine tumor lateralization based on center of mass.

    Args:
        masks: Dict mapping class labels to 3D mask arrays
        volume_shape: Expected shape of the volume (default: 240×240×155)

    Returns:
        Lateralization string: "Left hemisphere", "Right hemisphere", or "Bilateral"
    """
    # Combine all tumor classes
    combined_mask = np.zeros_like(masks[1], dtype=np.uint8)
    for mask in masks.values():
        combined_mask = combined_mask | mask

    if combined_mask.sum() == 0:
        return "No tumor detected"

    # Calculate center of mass
    com = center_of_mass(combined_mask)

    # Midline is at x = 120 for a 240×240×155 volume
    midline_x = volume_shape[0] / 2.0

    # Get x-coordinate of center of mass
    com_x = com[0]

    # Define bilateral threshold (within 10% of volume width from midline)
    bilateral_threshold = volume_shape[0] * 0.1

    if abs(com_x - midline_x) < bilateral_threshold:
        return "Bilateral"
    elif com_x < midline_x:
        return "Left hemisphere"
    else:
        return "Right hemisphere"

def generate_clinical_summary(volumes: Dict[str, Dict[str, float]], lateralization: str, masks: Dict[int, np.ndarray]) -> str:
    """
    Generate automated clinical-style text summary.

    Args:
        volumes: Volume information from calculate_volumes()
        lateralization: Lateralization string from determine_lateralization()
        masks: Dict mapping class labels to 3D mask arrays

    Returns:
        Clinical summary text
    """
    summary_parts = []

    # Determine dominant component
    class_volumes = {
        'Necrotic Core (NCR)': volumes['Necrotic Core (NCR)']['cm3'],
        'Edema (ED)': volumes['Edema (ED)']['cm3'],
        'Enhancing Tumor (ET)': volumes['Enhancing Tumor (ET)']['cm3']
    }

    dominant_component = max(class_volumes, key=class_volumes.get)

    # Check presence of each component
    has_necrotic = volumes['Necrotic Core (NCR)']['voxels'] > 0
    has_edema = volumes['Edema (ED)']['voxels'] > 0
    has_enhancing = volumes['Enhancing Tumor (ET)']['voxels'] > 0

    # Build summary
    if volumes['Whole Tumor (WT)']['voxels'] == 0:
        summary_parts.append("No tumor detected in the analyzed MRI sequences.")
    else:
        # Main finding
        wt_cm3 = volumes['Whole Tumor (WT)']['cm3']
        tc_cm3 = volumes['Tumor Core (TC)']['cm3']

        if has_enhancing:
            et_cm3 = volumes['Enhancing Tumor (ET)']['cm3']
            summary_parts.append(
                f"A brain tumor was detected in the {lateralization.lower()} with an enhancing component measuring {et_cm3:.2f} cm³."
            )
        else:
            summary_parts.append(
                f"A brain tumor was detected in the {lateralization.lower()}."
            )

        # Edema
        if has_edema:
            ed_cm3 = volumes['Edema (ED)']['cm3']
            if ed_cm3 > tc_cm3:
                summary_parts.append(f"Significant peritumoral edema is present, measuring {ed_cm3:.2f} cm³.")
            else:
                summary_parts.append(f"Peritumoral edema is present, measuring {ed_cm3:.2f} cm³.")

        # Tumor core and necrosis
        summary_parts.append(f"The tumor core measures {tc_cm3:.2f} cm³.")

        if has_necrotic:
            ncr_cm3 = volumes['Necrotic Core (NCR)']['cm3']
            summary_parts.append(f"A necrotic component is identified, measuring {ncr_cm3:.2f} cm³.")

        # Total volume
        summary_parts.append(f"The whole tumor volume (including all components) is {wt_cm3:.2f} cm³.")

    return " ".join(summary_parts)

# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def post_process_mask(mask: np.ndarray, min_size: int = 100, fill_holes: bool = True,
                      apply_erosion: bool = False, erosion_iterations: int = 1) -> np.ndarray:
    """
    Post-process a binary mask:
    1. Remove small connected components
    2. Fill holes
    3. Optional erosion to shrink boundaries
    """
    if mask.sum() == 0:
        return mask

    cleaned = mask.copy().astype(np.uint8)

    # Remove small components
    labeled, num_features = scipy_label(cleaned)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background
        large_mask = component_sizes >= min_size
        cleaned = large_mask[labeled].astype(np.uint8)

    # Fill holes
    if fill_holes and cleaned.sum() > 0:
        cleaned = binary_fill_holes(cleaned).astype(np.uint8)

    # Apply erosion
    if apply_erosion and cleaned.sum() > 0 and erosion_iterations > 0:
        cleaned = binary_erosion(cleaned, iterations=erosion_iterations).astype(np.uint8)

    return cleaned

def enforce_tumor_hierarchy(tc: np.ndarray, wt: np.ndarray, et: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce anatomical constraints: ET ⊆ TC ⊆ WT
    """
    et_corrected = (et & tc).astype(np.uint8)
    tc_corrected = (tc & wt).astype(np.uint8)
    wt_corrected = wt.astype(np.uint8)
    et_corrected = (et_corrected & tc_corrected).astype(np.uint8)

    return tc_corrected, wt_corrected, et_corrected

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_inference_transforms(config: InferenceConfig) -> Compose:
    """Inference transforms - uses RAS orientation (like training)."""
    return Compose([
        LoadImaged(keys=MODALITY_KEYS, image_only=False),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        DeleteItemsd(keys=MODALITY_KEYS),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=config.target_spacing, mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=config.patch_size),
    ])

def get_pre_crop_transforms() -> Compose:
    """Transforms to get pre-crop shape for coordinate mapping."""
    return Compose([
        LoadImaged(keys=MODALITY_KEYS, image_only=False),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        DeleteItemsd(keys=MODALITY_KEYS),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])

def get_crop_info_transforms() -> Compose:
    """Transforms to get exact crop shape."""
    return Compose([
        LoadImaged(keys=MODALITY_KEYS, image_only=False),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        DeleteItemsd(keys=MODALITY_KEYS),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
    ])


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================

def run_inference_with_tta(model, image_tensor: torch.Tensor, inferer, flip_dims: List[int] = [2, 3, 4]) -> torch.Tensor:
    """
    Run inference with test-time augmentation (flip augmentations).

    Args:
        model: PyTorch model
        image_tensor: Input tensor [B, C, D, H, W]
        inferer: MONAI SlidingWindowInferer
        flip_dims: Dimensions to flip (2=D, 3=H, 4=W for 3D)

    Returns:
        Averaged prediction probabilities
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        with torch.amp.autocast('cuda'):
            out = inferer(image_tensor, model)
            if isinstance(out, tuple):
                out = out[0]
            predictions.append(torch.sigmoid(out))

        # Flipped versions
        for dim in flip_dims:
            img_flipped = torch.flip(image_tensor, dims=[dim])
            with torch.amp.autocast('cuda'):
                out = inferer(img_flipped, model)
                if isinstance(out, tuple):
                    out = out[0]
                pred_flipped = torch.flip(torch.sigmoid(out), dims=[dim])
                predictions.append(pred_flipped)

    # Average all predictions
    avg_pred = torch.stack(predictions, dim=0).mean(dim=0)
    return avg_pred

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def process_single_case(
    model,
    file_paths: Dict[str, str],
    config: InferenceConfig,
    device: torch.device,
    progress_callback=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single patient case and return segmentation masks.

    Args:
        model: Trained UNet3D model
        file_paths: Dict mapping modality names to file paths
        config: InferenceConfig object
        device: torch device
        progress_callback: Optional callback function for progress updates

    Returns:
        Tuple of (original_image, class_1_mask, class_2_mask, class_4_mask)
    """
    if progress_callback:
        progress_callback("Loading and preprocessing images...", 0.1)

    # Setup inferer
    inferer = SlidingWindowInferer(
        roi_size=config.patch_size,
        sw_batch_size=config.sw_batch_size,
        overlap=config.overlap,
        mode="gaussian"
    )

    # Setup transforms
    inference_transforms = get_inference_transforms(config)
    pre_crop_transforms = get_pre_crop_transforms()
    crop_info_transforms = get_crop_info_transforms()

    # Load original file for shape info
    flair_path = file_paths["flair"]
    original_nii = nib.load(flair_path)
    original_shape = original_nii.shape
    original_data = original_nii.get_fdata()

    if progress_callback:
        progress_callback("Preparing data for inference...", 0.2)

    # Get pre-crop info
    pre_data = pre_crop_transforms(dict(file_paths))
    pre_img = pre_data["image"].numpy()
    pre_shape = list(pre_img.shape[1:])

    # Find crop box
    nz = np.where(np.any(pre_img != 0, axis=0))
    crop_start = [int(nz[i].min()) for i in range(3)]

    # Get actual cropped shape
    crop_data = crop_info_transforms(dict(file_paths))
    actual_crop_shape = list(crop_data["image"].shape[1:])
    crop_end = [crop_start[i] + actual_crop_shape[i] for i in range(3)]

    # Full inference transform
    full_data = inference_transforms(dict(file_paths))
    img = full_data["image"]

    # Calculate pad offsets
    pad_offset = [(config.patch_size[i] - actual_crop_shape[i]) // 2
                  if actual_crop_shape[i] < config.patch_size[i] else 0
                  for i in range(3)]

    if progress_callback:
        progress_callback("Running model inference...", 0.4)

    # Run inference
    x = img.unsqueeze(0).to(device)

    if config.use_tta:
        pred_probs = run_inference_with_tta(model, x, inferer, flip_dims=[2, 3, 4])
    else:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = inferer(x, model)
                if isinstance(out, tuple):
                    out = out[0]
                pred_probs = torch.sigmoid(out)

    pred_probs = pred_probs.cpu().numpy()[0]  # [3, D, H, W]

    if progress_callback:
        progress_callback("Applying thresholds and post-processing...", 0.7)

    # Apply per-class thresholds
    tc = (pred_probs[0] > config.threshold_tc).astype(np.uint8)
    wt = (pred_probs[1] > config.threshold_wt).astype(np.uint8)
    et = (pred_probs[2] > config.threshold_et).astype(np.uint8)

    # Unpad
    o = pad_offset
    s = actual_crop_shape
    tc_crop = tc[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]]
    wt_crop = wt[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]]
    et_crop = et[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]]

    # Place back into full RAS volume
    tc_ras = np.zeros(pre_shape, dtype=np.uint8)
    wt_ras = np.zeros(pre_shape, dtype=np.uint8)
    et_ras = np.zeros(pre_shape, dtype=np.uint8)

    c, e = crop_start, crop_end
    tc_ras[c[0]:e[0], c[1]:e[1], c[2]:e[2]] = tc_crop
    wt_ras[c[0]:e[0], c[1]:e[1], c[2]:e[2]] = wt_crop
    et_ras[c[0]:e[0], c[1]:e[1], c[2]:e[2]] = et_crop

    # Enforce hierarchy
    if config.enforce_hierarchy:
        tc_ras, wt_ras, et_ras = enforce_tumor_hierarchy(tc_ras, wt_ras, et_ras)

    # Post-process each mask
    tc_ras = post_process_mask(tc_ras, config.min_component_size, config.fill_holes,
                                config.apply_erosion, config.erosion_iterations)
    wt_ras = post_process_mask(wt_ras, config.min_component_size, config.fill_holes,
                                config.apply_erosion, config.erosion_iterations)
    et_ras = post_process_mask(et_ras, config.min_component_size, config.fill_holes,
                                config.apply_erosion, config.erosion_iterations)

    # Re-enforce hierarchy after post-processing
    if config.enforce_hierarchy:
        tc_ras, wt_ras, et_ras = enforce_tumor_hierarchy(tc_ras, wt_ras, et_ras)

    if progress_callback:
        progress_callback("Converting to original space...", 0.9)

    # RAS → LPS conversion
    tc_lps = np.flip(np.flip(tc_ras, 0), 1).copy()
    wt_lps = np.flip(np.flip(wt_ras, 0), 1).copy()
    et_lps = np.flip(np.flip(et_ras, 0), 1).copy()

    # Resample to original shape if needed
    if tc_lps.shape != original_shape:
        zoom_factors = [o / c for o, c in zip(original_shape, tc_lps.shape)]
        tc_lps = zoom(tc_lps.astype(np.float32), zoom_factors, order=0).astype(np.uint8)
        wt_lps = zoom(wt_lps.astype(np.float32), zoom_factors, order=0).astype(np.uint8)
        et_lps = zoom(et_lps.astype(np.float32), zoom_factors, order=0).astype(np.uint8)

        # Ensure exact shape
        tc_lps = tc_lps[:original_shape[0], :original_shape[1], :original_shape[2]]
        wt_lps = wt_lps[:original_shape[0], :original_shape[1], :original_shape[2]]
        et_lps = et_lps[:original_shape[0], :original_shape[1], :original_shape[2]]

    # Derive BraTS submission classes
    class_1 = ((tc_lps > 0) & (et_lps == 0)).astype(np.uint8)  # NCR/NET = TC - ET
    class_2 = ((wt_lps > 0) & (tc_lps == 0)).astype(np.uint8)  # Edema = WT - TC
    class_4 = et_lps.astype(np.uint8)                          # ET directly

    if progress_callback:
        progress_callback("Complete!", 1.0)

    return original_data, class_1, class_2, class_4



# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(model_path: str, device: torch.device):
    """Load the trained model (cached)."""
    model = UNet3D().to(device)

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None

    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_slice_visualization(image: np.ndarray, masks: Dict[int, np.ndarray],
                               slice_idx: int, axis: int = 2, modality_name: str = "FLAIR"):
    """
    Create a 2D slice visualization with overlaid segmentation masks.

    Args:
        image: 3D image array
        masks: Dict mapping class labels to 3D mask arrays
        slice_idx: Index of the slice to display
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        modality_name: Name of the MRI modality being displayed
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Extract slice
    if axis == 0:  # Sagittal
        img_slice = image[slice_idx, :, :]
        mask_slices = {k: v[slice_idx, :, :] for k, v in masks.items()}
    elif axis == 1:  # Coronal
        img_slice = image[:, slice_idx, :]
        mask_slices = {k: v[:, slice_idx, :] for k, v in masks.items()}
    else:  # Axial
        img_slice = image[:, :, slice_idx]
        mask_slices = {k: v[:, :, slice_idx] for k, v in masks.items()}

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display image
    ax.imshow(img_slice.T, cmap='gray', origin='lower')

    # Overlay masks with transparency
    for class_id, mask_slice in mask_slices.items():
        if mask_slice.sum() > 0:
            # Create colored overlay
            overlay = np.zeros((*mask_slice.shape, 4))
            color = tuple(int(CLASS_COLORS[class_id].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            overlay[mask_slice > 0] = (*color, 0.5)  # 50% transparency
            # Transpose only spatial dimensions (0,1), keep color channel (2) last
            ax.imshow(np.transpose(overlay, (1, 0, 2)), origin='lower')

    # Add legend
    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in sorted(masks.keys())]
    ax.legend(handles=patches, loc='upper right', fontsize=10)

    axis_names = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    ax.set_title(f"{modality_name} - {axis_names[axis]} View (Slice {slice_idx})", fontsize=14)
    ax.axis('off')

    return fig

def create_3d_visualization(image: np.ndarray, masks: Dict[int, np.ndarray],
                           downsample_factor: int = 4, show_brain_surface: bool = True,
                           brain_threshold: float = 0.1, brain_opacity: float = 0.15):
    """
    Create a 3D visualization using plotly with optional brain surface overlay.

    Args:
        image: 3D image array (MRI modality, e.g., FLAIR)
        masks: Dict mapping class labels to 3D mask arrays
        downsample_factor: Factor by which to downsample the volume
        show_brain_surface: Whether to show the brain surface mesh
        brain_threshold: Threshold for brain tissue extraction (relative to max intensity)
        brain_opacity: Opacity of the brain surface mesh (0.0-1.0)
    """
    import plotly.graph_objects as go
    from scipy.ndimage import zoom
    from skimage.measure import marching_cubes
    from skimage.filters import gaussian

    # Downsample for performance
    ds_factor = 1.0 / downsample_factor

    fig = go.Figure()

    # Add brain surface mesh if requested
    if show_brain_surface and image is not None:
        try:
            # Downsample the MRI image
            image_ds = zoom(image, ds_factor, order=1)

            # Normalize image to 0-1 range
            img_min, img_max = image_ds.min(), image_ds.max()
            if img_max > img_min:
                image_norm = (image_ds - img_min) / (img_max - img_min)

                # Apply Gaussian smoothing to reduce noise
                image_smooth = gaussian(image_norm, sigma=1.0)

                # Create binary mask for brain tissue
                # Use adaptive threshold based on image statistics
                threshold_value = brain_threshold
                brain_mask = image_smooth > threshold_value

                # Apply marching cubes to extract brain surface
                # Use a lower level to get outer surface
                try:
                    verts, faces, normals, values = marching_cubes(
                        image_smooth,
                        level=threshold_value,
                        step_size=2  # Further downsample the mesh for performance
                    )

                    # Create mesh trace
                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 2],  # Swap axes to match tumor coordinates
                        y=verts[:, 1],
                        z=verts[:, 0],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color='lightgray',
                        opacity=brain_opacity,
                        name='Brain Surface',
                        hoverinfo='name',
                        lighting=dict(
                            ambient=0.5,
                            diffuse=0.8,
                            specular=0.2,
                            roughness=0.5,
                            fresnel=0.2
                        ),
                        lightposition=dict(
                            x=100,
                            y=200,
                            z=0
                        )
                    ))
                except (ValueError, RuntimeError) as e:
                    # If marching cubes fails, silently skip brain surface
                    pass
        except Exception as e:
            # If brain surface extraction fails, continue without it
            pass

    # Add each tumor class as a separate 3D scatter plot
    for class_id, mask in masks.items():
        if mask.sum() == 0:
            continue

        # Downsample mask
        mask_ds = zoom(mask.astype(float), ds_factor, order=0) > 0.5

        # Get voxel coordinates where mask is True
        z, y, x = np.where(mask_ds)

        if len(x) == 0:
            continue

        # Create 3D scatter plot
        color = CLASS_COLORS[class_id]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=0.6
            ),
            name=CLASS_NAMES[class_id]
        ))

    # Update layout
    fig.update_layout(
        title="3D Tumor Segmentation with Brain Surface",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800,
        showlegend=True
    )

    return fig

# ============================================================================
# PDF CLINICAL REPORT GENERATION
# ============================================================================

def create_slice_image_for_pdf(image: np.ndarray, masks: Dict[int, np.ndarray],
                                slice_idx: int, axis: int, view_name: str) -> BytesIO:
    """
    Create a slice visualization and return as BytesIO for PDF embedding.

    Args:
        image: 3D image array
        masks: Dict mapping class labels to 3D mask arrays
        slice_idx: Index of the slice to display
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        view_name: Name of the view (e.g., "Axial", "Coronal", "Sagittal")

    Returns:
        BytesIO buffer containing the PNG image
    """
    # Extract slice
    if axis == 0:  # Sagittal
        img_slice = image[slice_idx, :, :]
        mask_slices = {k: v[slice_idx, :, :] for k, v in masks.items()}
    elif axis == 1:  # Coronal
        img_slice = image[:, slice_idx, :]
        mask_slices = {k: v[:, slice_idx, :] for k, v in masks.items()}
    else:  # Axial
        img_slice = image[:, :, slice_idx]
        mask_slices = {k: v[:, :, slice_idx] for k, v in masks.items()}

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display image
    ax.imshow(img_slice.T, cmap='gray', origin='lower')

    # Overlay masks with transparency
    for class_id, mask_slice in mask_slices.items():
        if mask_slice.sum() > 0:
            # Create colored overlay
            overlay = np.zeros((*mask_slice.shape, 4))
            color = tuple(int(CLASS_COLORS[class_id].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            overlay[mask_slice > 0] = (*color, 0.5)  # 50% transparency
            ax.imshow(np.transpose(overlay, (1, 0, 2)), origin='lower')

    # Add legend
    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in sorted(masks.keys()) if mask_slices[k].sum() > 0]
    if patches:
        ax.legend(handles=patches, loc='upper right', fontsize=8)

    ax.set_title(f"{view_name} View (Slice {slice_idx})", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf

def generate_clinical_report_pdf(
    original_image: np.ndarray,
    masks: Dict[int, np.ndarray],
    patient_id: str = "patient_001",
    config: InferenceConfig = None
) -> BytesIO:
    """
    Generate a comprehensive clinical report in PDF format.

    Args:
        original_image: 3D MRI image array
        masks: Dict mapping class labels to 3D mask arrays
        patient_id: Patient identifier
        config: InferenceConfig object with model settings

    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14

    # ========== HEADER SECTION ==========
    elements.append(Paragraph("BRAIN TUMOR SEGMENTATION REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))

    # Report metadata
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata_data = [
        ['Patient ID:', patient_id],
        ['Report Date:', report_date],
        ['Model:', '3D U-Net'],
    ]

    if config:
        metadata_data.extend([
            ['Patch Size:', f"{config.patch_size}"],
            ['TTA Enabled:', 'Yes' if config.use_tta else 'No'],
            ['Thresholds:', f"TC={config.threshold_tc:.2f}, WT={config.threshold_wt:.2f}, ET={config.threshold_et:.2f}"],
        ])

    metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4.5*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4788')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    elements.append(metadata_table)
    elements.append(Spacer(1, 0.3*inch))

    # ========== QUANTITATIVE ANALYSIS ==========
    elements.append(Paragraph("QUANTITATIVE ANALYSIS", heading_style))

    # Calculate volumes
    volumes = calculate_volumes(masks, voxel_size_mm3=1.0)

    # Create volume table
    volume_data = [['Region', 'Voxel Count', 'Volume (mm³)', 'Volume (cm³)']]

    # Individual classes
    for class_name in ['Necrotic Core (NCR)', 'Edema (ED)', 'Enhancing Tumor (ET)']:
        vol_info = volumes[class_name]
        volume_data.append([
            class_name,
            f"{vol_info['voxels']:,}",
            f"{vol_info['mm3']:.2f}",
            f"{vol_info['cm3']:.2f}"
        ])

    # Add separator
    volume_data.append(['', '', '', ''])

    # Composite regions
    for region_name in ['Tumor Core (TC)', 'Whole Tumor (WT)']:
        vol_info = volumes[region_name]
        volume_data.append([
            region_name,
            f"{vol_info['voxels']:,}",
            f"{vol_info['mm3']:.2f}",
            f"{vol_info['cm3']:.2f}"
        ])

    volume_table = Table(volume_data, colWidths=[2.2*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    volume_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))

    elements.append(volume_table)
    elements.append(Spacer(1, 0.3*inch))

    # ========== CLINICAL SUMMARY ==========
    elements.append(Paragraph("CLINICAL SUMMARY", heading_style))

    # Determine lateralization
    lateralization = determine_lateralization(masks, volume_shape=original_image.shape)

    # Generate automated summary
    clinical_text = generate_clinical_summary(volumes, lateralization, masks)

    elements.append(Paragraph(f"<b>Lateralization:</b> {lateralization}", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Findings:</b> {clinical_text}", normal_style))
    elements.append(Spacer(1, 0.3*inch))

    # ========== PAGE BREAK ==========
    elements.append(PageBreak())

    # ========== VISUAL DOCUMENTATION ==========
    elements.append(Paragraph("VISUAL DOCUMENTATION", heading_style))
    elements.append(Paragraph("Representative slices showing maximum tumor extent in each anatomical plane:", normal_style))
    elements.append(Spacer(1, 0.2*inch))

    # Find max-area slices for each view
    axial_slice = find_max_area_slice(masks, axis=2)
    coronal_slice = find_max_area_slice(masks, axis=1)
    sagittal_slice = find_max_area_slice(masks, axis=0)

    # Create images for each view
    views = [
        (2, axial_slice, "Axial"),
        (1, coronal_slice, "Coronal"),
        (0, sagittal_slice, "Sagittal")
    ]

    for axis, slice_idx, view_name in views:
        elements.append(Paragraph(f"{view_name} View", subheading_style))

        # Create image
        img_buf = create_slice_image_for_pdf(original_image, masks, slice_idx, axis, view_name)

        # Add to PDF
        img = Image(img_buf, width=5*inch, height=5*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))

    # ========== FOOTER NOTE ==========
    elements.append(Spacer(1, 0.3*inch))
    footer_text = """
    <b>Note:</b> This report was automatically generated using a 3D U-Net deep learning model
    trained on the BraTS dataset. The segmentation identifies three tumor sub-regions:
    Necrotic Core (NCR), Peritumoral Edema (ED), and Enhancing Tumor (ET).
    This analysis is intended for research purposes and should be reviewed by a qualified radiologist
    before clinical use.
    """
    elements.append(Paragraph(footer_text, normal_style))

    # Build PDF
    doc.build(elements)

    buffer.seek(0)
    return buffer

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application."""

    # Header
    st.title("🧠 BraTS Brain Tumor Segmentation")
    st.markdown("""
    This application performs automatic segmentation of brain tumors from MRI scans using a 3D U-Net model.
    Upload 4 MRI modalities (T1, T1ce, T2, FLAIR) to get started.
    """)

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Model path
    model_path = st.sidebar.text_input(
        "Model Weights Path",
        value="model/best_model.pth",
        help="Path to the trained model weights file (.pth)"
    )

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"🖥️ Using device: **{device}**")

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.subheader("Thresholds")
        threshold_tc = st.slider("Tumor Core (TC)", 0.0, 1.0, 0.52, 0.01)
        threshold_wt = st.slider("Whole Tumor (WT)", 0.0, 1.0, 0.47, 0.01)
        threshold_et = st.slider("Enhancing Tumor (ET)", 0.0, 1.0, 0.57, 0.01)

        st.subheader("Inference Options")
        use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True)
        overlap = st.slider("Sliding Window Overlap", 0.0, 0.9, 0.6, 0.05)

        st.subheader("Post-processing")
        min_component_size = st.number_input("Min Component Size (voxels)", 50, 500, 150, 10)
        fill_holes = st.checkbox("Fill Holes", value=True)
        enforce_hierarchy = st.checkbox("Enforce Tumor Hierarchy", value=True)
        apply_erosion = st.checkbox("Apply Erosion", value=False)
        erosion_iterations = st.number_input("Erosion Iterations", 0, 5, 0, 1) if apply_erosion else 0

    # Create config
    config = InferenceConfig(
        threshold_tc=threshold_tc,
        threshold_wt=threshold_wt,
        threshold_et=threshold_et,
        use_tta=use_tta,
        overlap=overlap,
        min_component_size=min_component_size,
        fill_holes=fill_holes,
        enforce_hierarchy=enforce_hierarchy,
        apply_erosion=apply_erosion,
        erosion_iterations=erosion_iterations,
        tta_flips=[0, 1, 2] if use_tta else []
    )

    # Main content area
    st.header("📁 Upload MRI Scans")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        flair_file = st.file_uploader("FLAIR", type=['nii', 'nii.gz'], key='flair')
        t1_file = st.file_uploader("T1", type=['nii', 'nii.gz'], key='t1')
    with col2:
        t1ce_file = st.file_uploader("T1ce", type=['nii', 'nii.gz'], key='t1ce')
        t2_file = st.file_uploader("T2", type=['nii', 'nii.gz'], key='t2')

    # Check if all files are uploaded
    all_files_uploaded = all([flair_file, t1_file, t1ce_file, t2_file])

    if all_files_uploaded:
        st.success("✅ All 4 modalities uploaded successfully!")

        # Run inference button
        if st.button("🚀 Run Segmentation", type="primary"):
            # Load model
            with st.spinner("Loading model..."):
                model = load_model(model_path, device)

            if model is None:
                st.stop()

            # Save uploaded files to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                file_paths = {}

                for modality, uploaded_file in [
                    ('flair', flair_file), ('t1', t1_file),
                    ('t1ce', t1ce_file), ('t2', t2_file)
                ]:
                    # Preserve original file extension
                    original_name = uploaded_file.name
                    if original_name.endswith('.nii.gz'):
                        ext = '.nii.gz'
                    elif original_name.endswith('.nii'):
                        ext = '.nii'
                    else:
                        ext = '.nii'  # Default to .nii

                    temp_path = os.path.join(tmpdir, f"{modality}{ext}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths[modality] = temp_path

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(message, progress):
                    status_text.text(message)
                    progress_bar.progress(progress)

                # Run inference
                try:
                    original_image, class_1, class_2, class_4 = process_single_case(
                        model=model,
                        file_paths=file_paths,
                        config=config,
                        device=device,
                        progress_callback=update_progress
                    )

                    # Store results in session state
                    st.session_state['results'] = {
                        'original_image': original_image,
                        'class_1': class_1,
                        'class_2': class_2,
                        'class_4': class_4,
                        'file_paths': file_paths.copy()
                    }

                    status_text.text("✅ Segmentation complete!")
                    progress_bar.progress(1.0)

                except Exception as e:
                    st.error(f"❌ Error during inference: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

    else:
        st.info("👆 Please upload all 4 MRI modalities to proceed.")

    # Display results if available
    if 'results' in st.session_state:
        st.header("📊 Segmentation Results")

        results = st.session_state['results']
        original_image = results['original_image']
        masks = {
            1: results['class_1'],
            2: results['class_2'],
            4: results['class_4']
        }

        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Necrotic Core (Class 1)", f"{masks[1].sum():,} voxels")
        with col2:
            st.metric("Edema (Class 2)", f"{masks[2].sum():,} voxels")
        with col3:
            st.metric("Enhancing Tumor (Class 4)", f"{masks[4].sum():,} voxels")

        # Visualization tabs
        st.subheader("🔍 Visualization")
        tab1, tab2 = st.tabs(["2D Slice Viewer", "3D Visualization"])

        with tab1:
            st.markdown("**Navigate through different slices and views of the segmentation**")

            # View selection
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                view_axis = st.selectbox(
                    "View Plane",
                    options=[2, 1, 0],
                    format_func=lambda x: {0: "Sagittal", 1: "Coronal", 2: "Axial"}[x],
                    index=0
                )

            # Slice selection
            max_slice = original_image.shape[view_axis] - 1
            with view_col2:
                slice_idx = st.slider(
                    "Slice Index",
                    min_value=0,
                    max_value=max_slice,
                    value=max_slice // 2
                )

            # Create and display visualization
            fig = create_slice_visualization(
                original_image, masks, slice_idx, view_axis, "FLAIR"
            )
            st.pyplot(fig)

        with tab2:
            st.markdown("**Interactive 3D visualization of tumor segmentation**")
            st.info("💡 Tip: Click and drag to rotate, scroll to zoom")

            # Visualization controls
            viz_col1, viz_col2, viz_col3 = st.columns(3)

            with viz_col1:
                downsample = st.slider(
                    "Downsample Factor (higher = faster but less detailed)",
                    min_value=2, max_value=8, value=4, step=1
                )

            with viz_col2:
                show_brain = st.checkbox(
                    "Show Brain Surface",
                    value=True,
                    help="Display semi-transparent brain surface mesh for anatomical context"
                )

            with viz_col3:
                if show_brain:
                    brain_opacity = st.slider(
                        "Brain Opacity",
                        min_value=0.05, max_value=0.5, value=0.15, step=0.05,
                        help="Transparency of the brain surface (lower = more transparent)"
                    )
                    brain_threshold = st.slider(
                        "Brain Threshold",
                        min_value=0.05, max_value=0.3, value=0.1, step=0.01,
                        help="Threshold for brain tissue extraction (adjust if brain surface is incomplete)"
                    )
                else:
                    brain_opacity = 0.15
                    brain_threshold = 0.1

            with st.spinner("Generating 3D visualization..."):
                fig_3d = create_3d_visualization(
                    original_image,
                    masks,
                    downsample,
                    show_brain_surface=show_brain,
                    brain_threshold=brain_threshold,
                    brain_opacity=brain_opacity
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        # Download section
        st.header("📥 Download Results")

        download_col1, download_col2, download_col3 = st.columns(3)

        with download_col1:
            st.subheader("NIfTI Format")

            # Create combined segmentation mask
            combined_mask = np.zeros_like(masks[1], dtype=np.uint8)
            combined_mask[masks[1] > 0] = 1
            combined_mask[masks[2] > 0] = 2
            combined_mask[masks[4] > 0] = 4

            # Save to temporary file and read data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp_path = tmp.name

            # Save NIfTI file (file is now closed)
            nii_img = nib.Nifti1Image(combined_mask, affine=np.eye(4))
            nib.save(nii_img, tmp_path)

            # Read the file data
            with open(tmp_path, 'rb') as f:
                nii_data = f.read()

            # Now we can safely delete the file
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore deletion errors on Windows

            # Provide download button
            st.download_button(
                label="📥 Download Segmentation (NIfTI)",
                data=nii_data,
                file_name="segmentation.nii.gz",
                mime="application/gzip"
            )

        with download_col2:
            st.subheader("RLE CSV Format")

            # Generate RLE encoding
            rle_data = []
            patient_id = "patient_001"

            for class_id, mask in masks.items():
                rle_string = rle_encode_c_order(mask)
                rle_data.append({
                    'id': f"{patient_id}_{class_id}",
                    'rle': rle_string if rle_string else ""
                })

            df = pd.DataFrame(rle_data)
            csv = df.to_csv(index=False)

            st.download_button(
                label="📥 Download RLE CSV",
                data=csv,
                file_name="submission.csv",
                mime="text/csv"
            )

        with download_col3:
            st.subheader("Clinical Report")

            # Generate PDF report
            try:
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = generate_clinical_report_pdf(
                        original_image=original_image,
                        masks=masks,
                        patient_id=patient_id,
                        config=config
                    )

                    pdf_data = pdf_buffer.getvalue()

                st.download_button(
                    label="📄 Export Clinical Report",
                    data=pdf_data,
                    file_name=f"{patient_id}_clinical_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses a 3D U-Net model trained on the BraTS dataset to segment brain tumors into three sub-regions:
    - **Class 1 (Necrotic Core)**: Dead tissue inside the tumor
    - **Class 2 (Edema)**: Swelling surrounding the tumor
    - **Class 4 (Enhancing Tumor)**: Active, growing part of the tumor

    **Model Architecture**: 3D U-Net with Instance Normalization and LeakyReLU activation

    **Pipeline Features**:
    - RAS orientation for inference → LPS for submission
    - Sliding window inference with Gaussian weighting
    - Optional Test-Time Augmentation (TTA)
    - Post-processing: small component removal, hole filling, hierarchy enforcement
    """)

if __name__ == "__main__":
    main()

