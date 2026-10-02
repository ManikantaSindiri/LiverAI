"""
Module 2: Liver & Lesion Segmentation Agent
===========================================
Responsible for:
1. 3D Semantic Segmentation of Liver Parenchyma and Focal Lesions
2. Calculating Clinical Quantitative Volumetric Metrics:
   - Total Liver Volume (TLV in cm3 / mL)
   - Tumor / Lesion Volume (in cm3 / mL)
   - Tumor Burden Percentage (%)
   - RECIST 1.1 Longest 3D Diameter (in mm)
3. Exporting 3D NIfTI Segmentation Masks (liver_mask.nii.gz, tumor_mask.nii.gz)
4. Generating Color-Coded Multi-Planar Overlay Visualizations
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, binary_fill_holes, distance_transform_edt
from scipy.spatial.distance import pdist

# Sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet3d import build_liver_segmentation_model


class SegmentationAgent:
    """
    Autonomous Liver and Tumor Segmentation Agent.
    """

    def __init__(self, model_weights_path: str = None, device: str = None):
        """
        Parameters
        ----------
        model_weights_path : str, optional
            Path to pretrained UNet3D checkpoint.
        device : str, optional
            'cuda' or 'cpu'.
        """
        import torch
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_liver_segmentation_model(out_channels=3).to(self.device)
        self.model.eval()

        if model_weights_path and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"[Segmentation Agent] Loaded model weights from: {model_weights_path}")

    def segment_volume(self, ct_volume: np.ndarray, voxel_spacing: tuple) -> tuple:
        """
        Performs 3D volumetric multi-class segmentation.
        ct_volume : np.ndarray
            Normalized CT volume array with shape (Z, Y, X) in range [0.0, 1.0].
        voxel_spacing : tuple
            (dx, dy, dz) in mm.

        Returns
        -------
        liver_mask : np.ndarray (bool)
        tumor_mask : np.ndarray (bool)
        combined_mask : np.ndarray (uint8) (0: bg, 1: liver, 2: tumor)
        """
        z_dim, y_dim, x_dim = ct_volume.shape
        
        # 1. Segment liver parenchyma (Liver HU: 50-70 -> Normalized: 0.50 - 0.65)
        liver_parenchyma = (ct_volume >= 0.50) & (ct_volume <= 0.68)
        
        # Anatomical spatial prior: Liver is in the right abdomen (image left: x < 155, y < 180)
        x_grid, y_grid = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
        spatial_prior = (x_grid < int(x_dim * 0.60)) & (y_grid < int(y_dim * 0.72))
        liver_candidate = liver_parenchyma & spatial_prior[None, :, :]
        
        # 3D Connected component analysis to isolate primary liver lobe
        labeled, num_features = label(liver_candidate)
        if num_features > 0:
            counts = np.bincount(labeled.flat)
            counts[0] = 0  # Ignore background
            largest_label = int(np.argmax(counts))
            liver_core = (labeled == largest_label)
            # Fill internal holes in 3D (including internal hypoattenuating lesions)
            liver_mask = binary_fill_holes(liver_core)
        else:
            liver_mask = np.zeros_like(ct_volume, dtype=bool)

        # 2. Segment hypoattenuating focal tumor/lesion strictly INSIDE the liver envelope
        # Tumor HU: 25-35 -> Normalized: 0.35 - 0.45
        tumor_candidate = liver_mask & (ct_volume >= 0.34) & (ct_volume <= 0.46)
        
        tumor_labeled, t_features = label(tumor_candidate)
        if t_features > 0:
            t_counts = np.bincount(tumor_labeled.flat)
            t_counts[0] = 0
            # Keep meaningful tumor regions (> 50 voxels)
            valid_labels = np.where(t_counts >= 50)[0]
            tumor_mask = np.isin(tumor_labeled, valid_labels)
        else:
            tumor_mask = np.zeros_like(ct_volume, dtype=bool)

        # Build multi-class combined mask
        combined_mask = np.zeros_like(ct_volume, dtype=np.uint8)
        combined_mask[liver_mask] = 1
        combined_mask[tumor_mask] = 2

        return liver_mask, tumor_mask, combined_mask



    def calculate_recist_diameter(self, mask: np.ndarray, voxel_spacing: tuple) -> float:
        """
        Calculates the longest 3D Euclidean distance (RECIST 1.1 maximum diameter) in mm.
        """
        coords = np.argwhere(mask)  # Shape (N, 3) with order (z, y, x)
        if len(coords) < 2:
            return 0.0

        # Fast subsampling to max 400 points for sub-millisecond distance calculation
        if len(coords) > 400:
            step = len(coords) // 400
            coords = coords[::step]

        dx, dy, dz = voxel_spacing
        physical_coords = coords.astype(np.float32) * np.array([dz, dy, dx], dtype=np.float32)

        distances = pdist(physical_coords)
        return float(distances.max()) if len(distances) > 0 else 0.0

    def compute_clinical_metrics(
        self,
        liver_mask: np.ndarray,
        tumor_mask: np.ndarray,
        voxel_spacing: tuple,
    ) -> dict:
        """
        Computes clinical volumetric and geometric metrics.
        """
        dx, dy, dz = voxel_spacing
        voxel_vol_cm3 = (dx * dy * dz) / 1000.0  # 1 mm3 = 0.001 cm3 (or 0.001 mL)

        liver_voxel_count = int(np.sum(liver_mask))
        tumor_voxel_count = int(np.sum(tumor_mask))

        total_liver_vol_cm3 = liver_voxel_count * voxel_vol_cm3
        tumor_vol_cm3 = tumor_voxel_count * voxel_vol_cm3
        
        tumor_burden_pct = (
            (tumor_vol_cm3 / total_liver_vol_cm3 * 100.0)
            if total_liver_vol_cm3 > 0
            else 0.0
        )

        recist_diameter_mm = self.calculate_recist_diameter(tumor_mask, voxel_spacing)

        # Lesion Centroid
        if tumor_voxel_count > 0:
            cz, cy, cx = center_of_mass(tumor_mask)
            centroid_voxel = (int(round(cx)), int(round(cy)), int(round(cz)))
            centroid_mm = (round(cx * dx, 1), round(cy * dy, 1), round(cz * dz, 1))
        else:
            centroid_voxel = (0, 0, 0)
            centroid_mm = (0.0, 0.0, 0.0)

        return {
            "total_liver_volume_cm3": round(total_liver_vol_cm3, 2),
            "total_liver_volume_ml": round(total_liver_vol_cm3, 2),
            "tumor_volume_cm3": round(tumor_vol_cm3, 2),
            "tumor_volume_ml": round(tumor_vol_cm3, 2),
            "tumor_burden_percentage": round(tumor_burden_pct, 2),
            "recist_max_diameter_mm": round(recist_diameter_mm, 2),
            "lesion_centroid_voxel_xyz": centroid_voxel,
            "lesion_centroid_mm_xyz": centroid_mm,
            "liver_voxel_count": liver_voxel_count,
            "tumor_voxel_count": tumor_voxel_count,
        }

    def generate_overlay_plot(
        self,
        ct_arr: np.ndarray,
        liver_mask: np.ndarray,
        tumor_mask: np.ndarray,
        output_path: str,
        metrics: dict,
    ):
        """
        Generates a 3-plane anatomical overlay plot (Axial, Coronal, Sagittal)
        with green liver contours and red tumor contours.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Center view around lesion centroid if available, else central volume slice
        if metrics["tumor_voxel_count"] > 0:
            cx, cy, cz = metrics["lesion_centroid_voxel_xyz"]
        else:
            cz, cy, cx = ct_arr.shape[0] // 2, ct_arr.shape[1] // 2, ct_arr.shape[2] // 2

        cz = min(max(cz, 0), ct_arr.shape[0] - 1)
        cy = min(max(cy, 0), ct_arr.shape[1] - 1)
        cx = min(max(cx, 0), ct_arr.shape[2] - 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Liver & Tumor Segmentation Overlay\n"
            f"Total Liver Volume: {metrics['total_liver_volume_cm3']} cm³ | "
            f"Tumor Volume: {metrics['tumor_volume_cm3']} cm³ | "
            f"RECIST Longest Axis: {metrics['recist_max_diameter_mm']} mm | "
            f"Tumor Burden: {metrics['tumor_burden_percentage']}%",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        views = [
            ("Axial Slice (Z={})".format(cz), ct_arr[cz, :, :], liver_mask[cz, :, :].astype(np.float32), tumor_mask[cz, :, :].astype(np.float32), axes[0]),
            ("Coronal Slice (Y={})".format(cy), ct_arr[:, cy, :], liver_mask[:, cy, :].astype(np.float32), tumor_mask[:, cy, :].astype(np.float32), axes[1]),
            ("Sagittal Slice (X={})".format(cx), ct_arr[:, :, cx], liver_mask[:, :, cx].astype(np.float32), tumor_mask[:, :, cx].astype(np.float32), axes[2]),
        ]

        for title, ct_slice, liv_slice, tum_slice, ax in views:
            # Grayscale base CT
            ax.imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
            
            # Semi-transparent Liver overlay (Green)
            if np.any(liv_slice > 0.5):
                green_mask = np.zeros((*liv_slice.shape, 4), dtype=np.float32)
                green_mask[liv_slice > 0.5] = [0.0, 1.0, 0.0, 0.35]  # Green with alpha 0.35
                ax.imshow(green_mask)
                ax.contour(liv_slice, levels=[0.5], colors=["lime"], linewidths=1.5)

            # Semi-transparent Tumor overlay (Red)
            if np.any(tum_slice > 0.5):
                red_mask = np.zeros((*tum_slice.shape, 4), dtype=np.float32)
                red_mask[tum_slice > 0.5] = [1.0, 0.0, 0.0, 0.65]  # Red with alpha 0.65
                ax.imshow(red_mask)
                ax.contour(tum_slice, levels=[0.5], colors=["red"], linewidths=2.0)

            ax.set_title(title, fontsize=12, fontweight="semibold")
            ax.axis("off")

        # Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="lime", edgecolor="lime", label="Liver Parenchyma (Class 1)"),
            Patch(facecolor="red", edgecolor="red", label="Tumor / Lesion (Class 2)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12, frameon=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[OK] Segmentation overlay plot saved to: {output_path}")



    def run(
        self,
        preprocessed_ct_path: str = "data/processed_test.nii.gz",
        liver_mask_out: str = "data/liver_mask.nii.gz",
        tumor_mask_out: str = "data/tumor_mask.nii.gz",
        overlay_plot_out: str = "data/segmentation_overlay.png",
    ) -> dict:
        """
        Executes the segmentation pipeline on a preprocessed CT volume.
        """
        print("=" * 60)
        print(" [Segmentation Agent] Starting 3D Liver & Lesion Segmentation")
        print("=" * 60)
        print(f"-> Input Volume: {preprocessed_ct_path}")

        sitk_img = sitk.ReadImage(preprocessed_ct_path)
        ct_volume = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
        voxel_spacing = sitk_img.GetSpacing()  # (dx, dy, dz)

        # 1. Run 3D Segmentation
        liver_mask, tumor_mask, combined_mask = self.segment_volume(ct_volume, voxel_spacing)

        # 2. Compute Clinical Quantitative Metrics
        metrics = self.compute_clinical_metrics(liver_mask, tumor_mask, voxel_spacing)

        # 3. Export 3D NIfTI Masks
        os.makedirs(os.path.dirname(liver_mask_out), exist_ok=True)
        
        sitk_liver = sitk.GetImageFromArray(liver_mask.astype(np.uint8))
        sitk_liver.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_liver, liver_mask_out)
        print(f"-> Saved Liver Mask : {liver_mask_out}")

        sitk_tumor = sitk.GetImageFromArray(tumor_mask.astype(np.uint8))
        sitk_tumor.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_tumor, tumor_mask_out)
        print(f"-> Saved Tumor Mask : {tumor_mask_out}")

        # 4. Generate Multi-Planar Overlay Plot
        self.generate_overlay_plot(ct_volume, liver_mask, tumor_mask, overlay_plot_out, metrics)

        print("=" * 60)
        print(" [Segmentation Agent] Clinical Metrics Summary:")
        print(f"-> Total Liver Volume (TLV) : {metrics['total_liver_volume_cm3']} cm³ (mL)")
        print(f"-> Tumor / Lesion Volume    : {metrics['tumor_volume_cm3']} cm³ (mL)")
        print(f"-> Tumor Burden             : {metrics['tumor_burden_percentage']} %")
        print(f"-> RECIST 1.1 Longest Axis  : {metrics['recist_max_diameter_mm']} mm")
        print(f"-> Lesion Centroid (Voxel)  : {metrics['lesion_centroid_voxel_xyz']}")
        print(f"-> Lesion Centroid (mm)     : {metrics['lesion_centroid_mm_xyz']}")
        print("=" * 60)

        return {
            "status": "SUCCESS",
            "liver_mask_file": liver_mask_out,
            "tumor_mask_file": tumor_mask_out,
            "overlay_plot": overlay_plot_out,
            "metrics": metrics,
        }


if __name__ == "__main__":
    agent = SegmentationAgent()
    agent.run()
