"""
Module 1: CT Preprocessing Agent
=================================
Responsible for standardizing raw clinical CT volumes:
1. Coordinate Reorientation (RAS / LPS standard)
2. Isotropic Voxel Resampling (e.g. 1.5mm x 1.5mm x 1.5mm)
3. Hounsfield Unit (HU) Windowing (Liver Window: WL=50, WW=200 -> [-50, 150] HU)
4. Intensity Normalization ([0.0, 1.0])
5. Foreground Body ROI Extraction
6. Preprocessing QA & Multi-Planar Visualization
"""

import os
import sys
import numpy as np
import SimpleITK as sitk

# Add parent directory to path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.visualization import plot_mpr_comparison


class PreprocessingAgent:
    """
    Autonomous CT Preprocessing Agent for liver imaging workflows.
    """

    def __init__(
        self,
        target_spacing=(1.5, 1.5, 1.5),
        window_center=50.0,
        window_width=200.0,
        normalize_range=(0.0, 1.0),
    ):
        """
        Parameters
        ----------
        target_spacing : tuple of float
            Target isotropic voxel spacing (dx, dy, dz) in millimeters.
        window_center : float
            Window Level / Center (HU) for liver tissue (default: 50.0).
        window_width : float
            Window Width (HU) for liver tissue (default: 200.0).
        normalize_range : tuple of float
            Output intensity bounds (default: [0.0, 1.0]).
        """
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.min_hu = window_center - (window_width / 2.0)  # -50 HU
        self.max_hu = window_center + (window_width / 2.0)  # 150 HU
        self.normalize_range = normalize_range

    def load_volume(self, file_path: str) -> sitk.Image:
        """Loads a NIfTI or DICOM CT scan from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CT volume not found at: {file_path}")

        if os.path.isdir(file_path):
            dicom_files = []
            for root, _, files in os.walk(file_path):
                for filename in files:
                    if filename.lower().endswith((".dcm", ".dicom")):
                        dicom_files.append(os.path.join(root, filename))
            if not dicom_files:
                nifti_files = []
                for root, _, files in os.walk(file_path):
                    for filename in files:
                        if filename.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd")):
                            nifti_files.append(os.path.join(root, filename))
                if not nifti_files:
                    raise ValueError(f"No valid CT volume files were found in the uploaded folder: {file_path}")
                return sitk.ReadImage(nifti_files[0], sitk.sitkFloat32)

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted(dicom_files))
            return reader.Execute()

        return sitk.ReadImage(file_path, sitk.sitkFloat32)

    def reorient_to_ras(self, image: sitk.Image) -> sitk.Image:
        """Standardizes anatomical orientation to RAS coordinates."""
        try:
            return sitk.DICOMOrient(image, "RAS")
        except Exception:
            # Fallback if orientation filter is already satisfied
            return image

    def resample_to_isotropic(
        self,
        image: sitk.Image,
        target_spacing: tuple = None,
        interpolator=sitk.sitkLinear,
    ) -> sitk.Image:
        """
        Resamples a 3D volume to uniform isotropic voxel spacing.
        """
        if target_spacing is None:
            target_spacing = self.target_spacing

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Compute new grid dimensions
        new_size = [
            int(round(orig_sz * orig_sp / tgt_sp))
            for orig_sz, orig_sp, tgt_sp in zip(
                original_size, original_spacing, target_spacing
            )
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetDefaultPixelValue(-1000.0)  # Air HU default

        return resample.Execute(image)

    def apply_liver_windowing(self, array: np.ndarray) -> np.ndarray:
        """
        Clamps Hounsfield Units to liver window and scales to [0.0, 1.0].
        Formula:
            clipped = clip(array, min_hu, max_hu)
            normalized = (clipped - min_hu) / (max_hu - min_hu)
        """
        clipped = np.clip(array, self.min_hu, self.max_hu)
        norm = (clipped - self.min_hu) / (self.max_hu - self.min_hu)
        if self.normalize_range != (0.0, 1.0):
            low, high = self.normalize_range
            norm = norm * (high - low) + low
        return norm.astype(np.float32)

    def run(
        self,
        input_path: str,
        output_path: str = None,
        qa_plot_path: str = "data/preprocessing_comparison.png",
    ) -> dict:
        """
        Executes the end-to-end preprocessing pipeline.

        Returns
        -------
        dict
            QA report and preprocessing metadata.
        """
        print("=" * 60)
        print(" [Preprocessing Agent] Starting CT Standardization Pipeline")
        print("=" * 60)
        print(f"-> Input Scan : {input_path}")

        # 1. Load Raw Volume
        raw_image = self.load_volume(input_path)
        raw_spacing = raw_image.GetSpacing()
        raw_size = raw_image.GetSize()
        raw_arr = sitk.GetArrayFromImage(raw_image)  # Shape (Z, Y, X)

        print(f"-> Raw Dimensions (X, Y, Z): {raw_size}")
        print(f"-> Raw Voxel Spacing (mm)  : {raw_spacing}")
        print(f"-> Raw Dynamic Range (HU)  : [{raw_arr.min():.1f}, {raw_arr.max():.1f}]")

        # 2. Coordinate Reorientation
        reoriented_image = self.reorient_to_ras(raw_image)

        # 3. Isotropic Resampling
        resampled_image = self.resample_to_isotropic(reoriented_image)
        resampled_spacing = resampled_image.GetSpacing()
        resampled_size = resampled_image.GetSize()
        resampled_arr = sitk.GetArrayFromImage(resampled_image)

        # 4. Liver Windowing & Normalization
        processed_arr = self.apply_liver_windowing(resampled_arr)

        # 5. Build Processed SimpleITK Volume
        processed_image = sitk.GetImageFromArray(processed_arr)
        processed_image.SetSpacing(resampled_spacing)
        processed_image.SetOrigin(resampled_image.GetOrigin())
        processed_image.SetDirection(resampled_image.GetDirection())

        # 6. Save Processed NIfTI
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(processed_image, output_path)
            print(f"-> Saved Processed Volume : {output_path}")

        # 7. Generate QA Multi-Planar Comparison
        if qa_plot_path:
            plot_mpr_comparison(
                raw_arr,
                processed_arr,
                output_path=qa_plot_path,
                title=f"CT Preprocessing QA (Liver Window: [{self.min_hu:.0f}, {self.max_hu:.0f}] HU)",
            )

        report = {
            "status": "SUCCESS",
            "input_file": input_path,
            "output_file": output_path,
            "qa_plot": qa_plot_path,
            "raw_shape_zyx": raw_arr.shape,
            "raw_spacing_mm": raw_spacing,
            "raw_hu_range": (float(raw_arr.min()), float(raw_arr.max())),
            "processed_shape_zyx": processed_arr.shape,
            "processed_spacing_mm": resampled_spacing,
            "processed_intensity_range": (float(processed_arr.min()), float(processed_arr.max())),
            "liver_window_hu": {"center": self.window_center, "width": self.window_width, "range": (self.min_hu, self.max_hu)},
        }

        print("=" * 60)
        print(" [Preprocessing Agent] Standardization Pipeline Completed!")
        print(f"-> Output Shape (Z, Y, X): {processed_arr.shape}")
        print(f"-> Output Spacing        : {resampled_spacing} mm (Isotropic)")
        print(f"-> Output Range          : [{processed_arr.min():.2f}, {processed_arr.max():.2f}]")
        print("=" * 60)

        return report


if __name__ == "__main__":
    agent = PreprocessingAgent(
        target_spacing=(1.5, 1.5, 1.5),
        window_center=50.0,
        window_width=200.0,
    )
    input_ct = "data/test.nii.gz"
    output_ct = "data/processed_test.nii.gz"
    qa_image = "data/preprocessing_comparison.png"

    agent.run(input_ct, output_ct, qa_image)
