import os
import numpy as np
import matplotlib.pyplot as plt

def get_mpr_slices(volume_arr):
    """
    Extracts central slices along 3 anatomical orthogonal planes:
    - Axial: slice along axis 0 (Z, transverse)
    - Coronal: slice along axis 1 (Y, frontal)
    - Sagittal: slice along axis 2 (X, sagittal)
    Expects volume_arr with shape (Z, Y, X).
    """
    z, y, x = volume_arr.shape
    axial_slice = volume_arr[z // 2, :, :]
    coronal_slice = volume_arr[:, y // 2, :]
    sagittal_slice = volume_arr[:, :, x // 2]
    return axial_slice, coronal_slice, sagittal_slice

def plot_mpr_comparison(raw_arr, proc_arr, output_path="data/preprocessing_comparison.png", title="CT Preprocessing & Windowing QA"):
    """
    Generates a 2x3 multi-planar comparison grid:
    Top Row: Raw CT Volume (Full Dynamic HU Range)
    Bottom Row: Preprocessed CT Volume (Standard Liver Window & Resampled)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    raw_ax, raw_cor, raw_sag = get_mpr_slices(raw_arr)
    proc_ax, proc_cor, proc_sag = get_mpr_slices(proc_arr)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Row 1: Raw CT with broad windowing (vmin=-1000, vmax=1000)
    raw_vmin, raw_vmax = -1000, 1000
    axes[0, 0].imshow(raw_ax, cmap="gray", vmin=raw_vmin, vmax=raw_vmax)
    axes[0, 0].set_title(f"Raw Axial (Z={raw_arr.shape[0]//2})", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(raw_cor, cmap="gray", vmin=raw_vmin, vmax=raw_vmax, aspect="auto")
    axes[0, 1].set_title(f"Raw Coronal (Y={raw_arr.shape[1]//2})", fontsize=12)
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(raw_sag, cmap="gray", vmin=raw_vmin, vmax=raw_vmax, aspect="auto")
    axes[0, 2].set_title(f"Raw Sagittal (X={raw_arr.shape[2]//2})", fontsize=12)
    axes[0, 2].axis("off")
    
    # Row 2: Preprocessed CT with normalized / liver windowing ([0.0, 1.0])
    proc_vmin, proc_vmax = 0.0, 1.0 if proc_arr.max() <= 1.0 else (proc_arr.min(), proc_arr.max())
    axes[1, 0].imshow(proc_ax, cmap="gray", vmin=proc_vmin, vmax=proc_vmax)
    axes[1, 0].set_title(f"Preprocessed Axial (Z={proc_arr.shape[0]//2})\nLiver Window Normalized", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(proc_cor, cmap="gray", vmin=proc_vmin, vmax=proc_vmax, aspect="auto")
    axes[1, 1].set_title(f"Preprocessed Coronal (Y={proc_arr.shape[1]//2})\nLiver Window Normalized", fontsize=12)
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(proc_sag, cmap="gray", vmin=proc_vmin, vmax=proc_vmax, aspect="auto")
    axes[1, 2].set_title(f"Preprocessed Sagittal (X={proc_arr.shape[2]//2})\nLiver Window Normalized", fontsize=12)
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] Multi-planar QA comparison saved to: {output_path}")

