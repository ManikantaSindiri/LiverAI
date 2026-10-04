"""
Module 3: Lesion Analysis Agent
===============================

Purpose:
1. Extract quantitative lesion radiomics
2. Estimate Couinaud liver segment
3. Provide AI-assisted risk assessment
4. Generate a diagnostic dashboard

IMPORTANT:
This module is intended for research/educational use.
It does NOT provide a confirmed medical diagnosis.
"""

import math
import os
import sys

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass, binary_erosion
from scipy.spatial.distance import pdist


# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


class LesionAnalysisAgent:
    """
    Quantitative liver lesion analysis agent.
    """

    def __init__(self):
        pass

    # ==============================================================
    # 1. RADIOMICS
    # ==============================================================

    def extract_radiomics_profile(
        self,
        raw_ct_arr: np.ndarray,
        liver_mask: np.ndarray,
        tumor_mask: np.ndarray,
        voxel_spacing: tuple,
    ) -> dict:
        """
        Extract quantitative CT intensity and shape features.
        """

        # SimpleITK spacing is:
        # (X, Y, Z)
        dx, dy, dz = voxel_spacing

        voxel_volume_mm3 = dx * dy * dz

        # Approximate surface voxel area
        voxel_area_mm2 = (
            (dx * dy) +
            (dy * dz) +
            (dx * dz)
        ) / 3.0

        tumor_mask = np.asarray(tumor_mask, dtype=bool)
        liver_mask = np.asarray(liver_mask, dtype=bool)

        # ----------------------------------------------------------
        # Check tumor mask
        # ----------------------------------------------------------

        tumor_voxels = raw_ct_arr[tumor_mask]

        if tumor_voxels.size == 0:

            print("[WARNING] Tumor mask contains no positive voxels.")

            return {
                "error": "No lesion voxels detected",

                "lesion_mean_hu": 0.0,
                "lesion_std_hu": 0.0,
                "lesion_median_hu": 0.0,
                "lesion_min_hu": 0.0,
                "lesion_max_hu": 0.0,
                "lesion_p10_hu": 0.0,
                "lesion_p90_hu": 0.0,
                "lesion_iqr_hu": 0.0,

                "liver_mean_hu": 0.0,
                "liver_std_hu": 0.0,

                "hu_attenuation_deficit": 0.0,
                "tumor_to_liver_contrast_ratio": 0.0,

                "sphericity": 0.0,
                "compactness": 0.0,
                "surface_area_mm2": 0.0,

                "necrotic_core_percentage": 0.0,
            }

        # ----------------------------------------------------------
        # Liver voxels excluding tumor
        # ----------------------------------------------------------

        liver_background_mask = (
            liver_mask &
            (~tumor_mask)
        )

        liver_voxels = raw_ct_arr[liver_background_mask]

        # ----------------------------------------------------------
        # Remove invalid values
        # ----------------------------------------------------------

        tumor_voxels = tumor_voxels[
            np.isfinite(tumor_voxels)
        ]

        liver_voxels = liver_voxels[
            np.isfinite(liver_voxels)
        ]

        if tumor_voxels.size == 0:

            return {
                "error": "Tumor mask contains no valid CT voxels",

                "lesion_mean_hu": 0.0,
                "lesion_std_hu": 0.0,
                "lesion_median_hu": 0.0,
                "lesion_min_hu": 0.0,
                "lesion_max_hu": 0.0,
                "lesion_p10_hu": 0.0,
                "lesion_p90_hu": 0.0,
                "lesion_iqr_hu": 0.0,

                "liver_mean_hu": 0.0,
                "liver_std_hu": 0.0,

                "hu_attenuation_deficit": 0.0,
                "tumor_to_liver_contrast_ratio": 0.0,

                "sphericity": 0.0,
                "compactness": 0.0,
                "surface_area_mm2": 0.0,

                "necrotic_core_percentage": 0.0,
            }

        # ==========================================================
        # INTENSITY FEATURES
        # ==========================================================

        lesion_mean_hu = float(
            np.mean(tumor_voxels)
        )

        lesion_std_hu = float(
            np.std(tumor_voxels)
        )

        lesion_median_hu = float(
            np.median(tumor_voxels)
        )

        lesion_min_hu = float(
            np.min(tumor_voxels)
        )

        lesion_max_hu = float(
            np.max(tumor_voxels)
        )

        lesion_p10_hu = float(
            np.percentile(tumor_voxels, 10)
        )

        lesion_p90_hu = float(
            np.percentile(tumor_voxels, 90)
        )

        lesion_iqr_hu = (
            lesion_p90_hu -
            lesion_p10_hu
        )

        # ==========================================================
        # LIVER REFERENCE
        # ==========================================================

        if liver_voxels.size > 0:

            liver_mean_hu = float(
                np.mean(liver_voxels)
            )

            liver_std_hu = float(
                np.std(liver_voxels)
            )

        else:

            liver_mean_hu = 60.0
            liver_std_hu = 6.0

        # ==========================================================
        # CONTRAST
        # ==========================================================

        if abs(liver_mean_hu) > 1e-6:

            contrast_ratio = (
                lesion_mean_hu /
                liver_mean_hu
            )

        else:

            contrast_ratio = 1.0

        hu_deficit = (
            liver_mean_hu -
            lesion_mean_hu
        )

        # ==========================================================
        # VOLUME
        # ==========================================================

        tumor_volume_mm3 = (
            tumor_voxels.size *
            voxel_volume_mm3
        )

        # ==========================================================
        # SURFACE AREA
        # ==========================================================

        try:

            eroded_tumor = binary_erosion(
                tumor_mask
            )

            boundary = (
                tumor_mask &
                (~eroded_tumor)
            )

            surface_voxels_count = int(
                np.sum(boundary)
            )

            surface_area_mm2 = (
                surface_voxels_count *
                voxel_area_mm2
            )

        except Exception:

            surface_area_mm2 = 0.0

        # ==========================================================
        # SPHERICITY
        # ==========================================================

        if surface_area_mm2 > 0:

            sphericity = (
                (
                    np.pi ** (1.0 / 3.0)
                )
                *
                (
                    (6.0 * tumor_volume_mm3)
                    ** (2.0 / 3.0)
                )
                /
                surface_area_mm2
            )

            sphericity = float(
                np.clip(
                    sphericity,
                    0.0,
                    1.0
                )
            )

            compactness = float(
                tumor_volume_mm3 /
                (surface_area_mm2 ** 1.5)
            )

        else:

            sphericity = 0.0
            compactness = 0.0

        # ==========================================================
        # NECROTIC CORE ESTIMATION
        # ==========================================================

        necrotic_fraction = float(
            np.mean(
                tumor_voxels < 20.0
            ) * 100.0
        )

        return {

            "lesion_mean_hu":
                round(lesion_mean_hu, 1),

            "lesion_std_hu":
                round(lesion_std_hu, 1),

            "lesion_median_hu":
                round(lesion_median_hu, 1),

            "lesion_min_hu":
                round(lesion_min_hu, 1),

            "lesion_max_hu":
                round(lesion_max_hu, 1),

            "lesion_p10_hu":
                round(lesion_p10_hu, 1),

            "lesion_p90_hu":
                round(lesion_p90_hu, 1),

            "lesion_iqr_hu":
                round(lesion_iqr_hu, 1),

            "liver_mean_hu":
                round(liver_mean_hu, 1),

            "liver_std_hu":
                round(liver_std_hu, 1),

            "hu_attenuation_deficit":
                round(hu_deficit, 1),

            "tumor_to_liver_contrast_ratio":
                round(contrast_ratio, 2),

            "sphericity":
                round(sphericity, 3),

            "compactness":
                round(compactness, 4),

            "surface_area_mm2":
                round(surface_area_mm2, 1),

            "necrotic_core_percentage":
                round(necrotic_fraction, 1),
        }

    # ==============================================================
    # 2. COUINAUD LOCALIZATION
    # ==============================================================

    def identify_couinaud_segment(
        self,
        tumor_mask: np.ndarray,
        liver_mask: np.ndarray,
        voxel_spacing: tuple,
    ) -> dict:
        """
        Provides an approximate Couinaud segment estimate
        from lesion centroid.

        NOTE:
        True Couinaud segmentation requires anatomical vascular
        landmarks and should not be considered definitive here.
        """

        if (
            not np.any(tumor_mask)
            or
            not np.any(liver_mask)
        ):

            return {
                "segment_number": "Unknown",
                "segment_name": "Unspecified",
                "confidence": "Low",
            }

        # ----------------------------------------------------------
        # Safe centroid calculation
        # ----------------------------------------------------------

        cz, cy, cx = center_of_mass(
            tumor_mask
        )

        if not all(
            math.isfinite(v)
            for v in [cz, cy, cx]
        ):

            return {
                "segment_number": "Unknown",
                "segment_name": "Unspecified",
                "confidence": "Low",
            }

        # ----------------------------------------------------------
        # Liver coordinates
        # ----------------------------------------------------------

        lz, ly, lx = np.where(
            liver_mask
        )

        z_min = np.min(lz)
        z_max = np.max(lz)

        y_min = np.min(ly)
        y_max = np.max(ly)

        x_min = np.min(lx)
        x_max = np.max(lx)

        # ----------------------------------------------------------
        # Normalize coordinates
        # ----------------------------------------------------------

        norm_z = (
            (cz - z_min) /
            max(z_max - z_min, 1)
        )

        norm_y = (
            (cy - y_min) /
            max(y_max - y_min, 1)
        )

        norm_x = (
            (cx - x_min) /
            max(x_max - x_min, 1)
        )

        # ----------------------------------------------------------
        # Approximate segment rules
        # ----------------------------------------------------------

        if (
            norm_y > 0.65
            and
            0.40 <= norm_x <= 0.65
            and
            norm_z > 0.45
        ):

            seg_num = "I"
            seg_name = "Caudate Lobe (Segment I)"

        elif norm_x >= 0.50:

            if norm_x >= 0.70:

                if norm_z >= 0.50:

                    seg_num = "II"
                    seg_name = (
                        "Left Posterosuperior / "
                        "Lateral (Segment II)"
                    )

                else:

                    seg_num = "III"
                    seg_name = (
                        "Left Posteroinferior / "
                        "Lateral (Segment III)"
                    )

            else:

                if norm_z >= 0.50:

                    seg_num = "IVa"
                    seg_name = (
                        "Left Anterosuperior / "
                        "Medial (Segment IVa)"
                    )

                else:

                    seg_num = "IVb"
                    seg_name = (
                        "Left Anteroinferior / "
                        "Medial (Segment IVb)"
                    )

        else:

            if norm_y <= 0.52:

                if norm_z >= 0.50:

                    seg_num = "VIII"
                    seg_name = (
                        "Right Anterosuperior "
                        "(Segment VIII)"
                    )

                else:

                    seg_num = "V"
                    seg_name = (
                        "Right Anteroinferior "
                        "(Segment V)"
                    )

            else:

                if norm_z >= 0.50:

                    seg_num = "VII"
                    seg_name = (
                        "Right Posterosuperior "
                        "(Segment VII)"
                    )

                else:

                    seg_num = "VI"
                    seg_name = (
                        "Right Posteroinferior "
                        "(Segment VI)"
                    )

        return {

            "segment_number": seg_num,

            "segment_name": seg_name,

            "confidence": "Approximate",

            "normalized_coordinates": {

                "right_to_left_x":
                    round(float(norm_x), 2),

                "anterior_to_posterior_y":
                    round(float(norm_y), 2),

                "inferior_to_superior_z":
                    round(float(norm_z), 2),
            },
        }

    # ==============================================================
    # 3. AI-ASSISTED RISK ASSESSMENT
    # ==============================================================

    def evaluate_lirads_score(
        self,
        radiomics: dict,
        recist_diameter_mm: float,
    ) -> dict:
        """
        Provides an educational AI-assisted risk estimate.

        IMPORTANT:
        This is NOT a clinical LI-RADS implementation.
        It should not be interpreted as a confirmed diagnosis.
        """

        contrast_ratio = radiomics.get(
            "tumor_to_liver_contrast_ratio",
            1.0
        )

        hu_deficit = radiomics.get(
            "hu_attenuation_deficit",
            0.0
        )

        mean_hu = radiomics.get(
            "lesion_mean_hu",
            50.0
        )

        lesion_std = radiomics.get(
            "lesion_std_hu",
            10.0
        )

        # ----------------------------------------------------------
        # No lesion
        # ----------------------------------------------------------

        if recist_diameter_mm <= 0:

            return {

                "lirads_category":
                    "Not Assessable",

                "lirads_description":
                    "No measurable lesion detected",

                "risk_category":
                    "Insufficient Data",

                "malignancy_probability_percent":
                    0.0,

                "clinical_recommendation":
                    "Review image quality and segmentation.",

                "assessment_note":
                    "Automated estimate only.",
            }

        # ----------------------------------------------------------
        # Educational rule-based estimate
        # ----------------------------------------------------------

        if (
            mean_hu < 15.0
            and
            lesion_std < 6.0
        ):

            category = "Low Risk"

            description = (
                "Imaging characteristics may be "
                "compatible with a benign low-attenuation lesion."
            )

            risk = 5.0

            recommendation = (
                "Correlation with the complete CT/MRI "
                "study and radiologist review is recommended."
            )

        elif (
            recist_diameter_mm >= 20.0
            and
            (
                hu_deficit >= 20.0
                or
                contrast_ratio < 0.60
            )
        ):

            category = "High Risk"

            description = (
                "The analyzed lesion demonstrates "
                "features associated with a higher-risk "
                "imaging pattern."
            )

            risk = 95.0

            recommendation = (
                "Further evaluation by a qualified radiologist "
                "using appropriate multiphasic contrast-enhanced "
                "CT or MRI is recommended."
            )

        elif recist_diameter_mm >= 20.0:

            category = "Intermediate-High Risk"

            description = (
                "Lesion size and attenuation characteristics "
                "warrant further evaluation."
            )

            risk = 75.0

            recommendation = (
                "Radiologist review and appropriate "
                "contrast-enhanced liver imaging are recommended."
            )

        elif recist_diameter_mm >= 10.0:

            category = "Intermediate Risk"

            description = (
                "Indeterminate lesion requiring "
                "additional imaging assessment."
            )

            risk = 35.0

            recommendation = (
                "Consider follow-up or multiphasic "
                "contrast-enhanced imaging as clinically appropriate."
            )

        else:

            category = "Low-Intermediate Risk"

            description = (
                "Small lesion with indeterminate "
                "automated imaging characteristics."
            )

            risk = 15.0

            recommendation = (
                "Clinical correlation and radiologist "
                "review are recommended."
            )

        return {

            "lirads_category":
                category,

            "lirads_description":
                description,

            "risk_category":
                category,

            "malignancy_probability_percent":
                risk,

            "clinical_recommendation":
                recommendation,

            "assessment_note":
                (
                    "Educational AI-assisted estimate. "
                    "This output is not a confirmed diagnosis "
                    "and is not a substitute for radiologist review."
                ),
        }

    # ==============================================================
    # 4. SAFE CENTROID
    # ==============================================================

    def get_safe_centroid(
        self,
        tumor_mask: np.ndarray,
        raw_ct_arr: np.ndarray,
    ):

        if not np.any(tumor_mask):

            return (
                raw_ct_arr.shape[0] // 2,
                raw_ct_arr.shape[1] // 2,
                raw_ct_arr.shape[2] // 2,
            )

        cz, cy, cx = center_of_mass(
            tumor_mask
        )

        if not all(
            math.isfinite(v)
            for v in [cz, cy, cx]
        ):

            return (
                raw_ct_arr.shape[0] // 2,
                raw_ct_arr.shape[1] // 2,
                raw_ct_arr.shape[2] // 2,
            )

        cz = int(round(cz))
        cy = int(round(cy))
        cx = int(round(cx))

        # Clamp values to valid array ranges

        cz = max(
            0,
            min(cz, raw_ct_arr.shape[0] - 1)
        )

        cy = max(
            0,
            min(cy, raw_ct_arr.shape[1] - 1)
        )

        cx = max(
            0,
            min(cx, raw_ct_arr.shape[2] - 1)
        )

        return cz, cy, cx

    # ==============================================================
    # 5. DASHBOARD
    # ==============================================================

    def generate_diagnostic_dashboard(
        self,
        raw_ct_arr: np.ndarray,
        liver_mask: np.ndarray,
        tumor_mask: np.ndarray,
        analysis_results: dict,
        output_path: str,
    ):
        """
        Generates a 4-panel lesion analysis dashboard.
        """

        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        # ----------------------------------------------------------
        # Safe centroid
        # ----------------------------------------------------------

        cz, cy, cx = self.get_safe_centroid(
            tumor_mask,
            raw_ct_arr
        )

        radiomics = analysis_results[
            "radiomics"
        ]

        couinaud = analysis_results[
            "couinaud_localization"
        ]

        risk = analysis_results[
            "lirads_staging"
        ]

        recist_mm = analysis_results[
            "recist_diameter_mm"
        ]

        # ==========================================================
        # FIGURE
        # ==========================================================

        fig = plt.figure(
            figsize=(16, 12)
        )

        fig.suptitle(
            "LIVER CT — AI-ASSISTED LESION ANALYSIS",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # ==========================================================
        # PANEL 1 — CT + SEGMENTATION
        # ==========================================================

        ax1 = fig.add_subplot(
            2,
            2,
            1
        )

        pad = 45

        ymin = max(
            cy - pad,
            0
        )

        ymax = min(
            cy + pad,
            raw_ct_arr.shape[1]
        )

        xmin = max(
            cx - pad,
            0
        )

        xmax = min(
            cx + pad,
            raw_ct_arr.shape[2]
        )

        crop_ct = raw_ct_arr[
            cz,
            ymin:ymax,
            xmin:xmax
        ]

        crop_liv = liver_mask[
            cz,
            ymin:ymax,
            xmin:xmax
        ]

        crop_tum = tumor_mask[
            cz,
            ymin:ymax,
            xmin:xmax
        ]

        ax1.imshow(
            crop_ct,
            cmap="gray",
            vmin=-50,
            vmax=150
        )

        if np.any(crop_liv):

            ax1.contour(
                crop_liv,
                levels=[0.5],
                colors=["lime"],
                linewidths=1.5
            )

        if np.any(crop_tum):
            crop_tum_bool = crop_tum.astype(bool)

            ax1.contour(
                crop_tum_bool,
                levels=[0.5],
                colors=["red"],
                linewidths=2.5
            )

            overlay = np.zeros(
                (
                    crop_tum.shape[0],
                    crop_tum.shape[1],
                    4
                ),
                dtype=np.float32
            )

            overlay[crop_tum_bool] = [
                1.0,
                0.0,
                0.0,
                0.40
            ]

            ax1.imshow(
                overlay
            )

        ax1.set_title(
            "CT + Liver/Lession Segmentation\n"
            f"Axial Slice: {cz} | "
            f"Longest Axis: {recist_mm:.2f} mm",
            fontsize=12,
            fontweight="bold"
        )

        ax1.axis("off")

        # ==========================================================
        # PANEL 2 — HU HISTOGRAM
        # ==========================================================

        ax2 = fig.add_subplot(
            2,
            2,
            2
        )

        tumor_hu = raw_ct_arr[
            tumor_mask
        ]

        liver_hu = raw_ct_arr[
            liver_mask &
            (~tumor_mask)
        ]

        tumor_hu = tumor_hu[
            np.isfinite(tumor_hu)
        ]

        liver_hu = liver_hu[
            np.isfinite(liver_hu)
        ]

        if liver_hu.size > 0:

            ax2.hist(
                liver_hu,
                bins=35,
                range=(0, 100),
                alpha=0.55,
                label=(
                    f"Liver "
                    f"(Mean {radiomics['liver_mean_hu']} HU)"
                ),
                density=True
            )

        if tumor_hu.size > 0:

            ax2.hist(
                tumor_hu,
                bins=35,
                range=(0, 100),
                alpha=0.65,
                label=(
                    f"Lesion "
                    f"(Mean {radiomics['lesion_mean_hu']} HU)"
                ),
                density=True
            )

        ax2.axvline(
            radiomics["lesion_mean_hu"],
            linestyle="--",
            linewidth=2,
            label="Lesion Mean"
        )

        ax2.axvline(
            radiomics["liver_mean_hu"],
            linestyle="--",
            linewidth=2,
            label="Liver Mean"
        )

        ax2.set_xlabel(
            "Hounsfield Units (HU)"
        )

        ax2.set_ylabel(
            "Probability Density"
        )

        ax2.set_title(
            "CT Attenuation Distribution",
            fontsize=12,
            fontweight="bold"
        )

        ax2.legend(
            fontsize=9
        )

        ax2.grid(
            True,
            linestyle=":",
            alpha=0.6
        )

        # ==========================================================
        # PANEL 3 — QUANTITATIVE FEATURES
        # ==========================================================

        ax3 = fig.add_subplot(
            2,
            2,
            3
        )

        names = [
            "Sphericity",
            "Tumor/Liver\nRatio",
            "Necrosis\n%",
            "IQR/10\nHU",
            "Std/10\nHU",
        ]

        values = [

            radiomics.get(
                "sphericity",
                0
            ),

            radiomics.get(
                "tumor_to_liver_contrast_ratio",
                0
            ),

            radiomics.get(
                "necrotic_core_percentage",
                0
            ) / 100.0,

            radiomics.get(
                "lesion_iqr_hu",
                0
            ) / 10.0,

            radiomics.get(
                "lesion_std_hu",
                0
            ) / 10.0,
        ]

        bars = ax3.bar(
            names,
            values,
            alpha=0.85
        )

        max_value = max(
            values + [1.0]
        )

        for bar, value in zip(
            bars,
            values
        ):

            ax3.text(
                bar.get_x()
                +
                bar.get_width() / 2,

                value
                +
                max_value * 0.02,

                f"{value:.2f}",

                ha="center",

                va="bottom",

                fontsize=10,

                fontweight="bold"
            )

        ax3.set_ylim(
            0,
            max_value * 1.25
        )

        ax3.set_title(
            "Quantitative Lesion Features",
            fontsize=12,
            fontweight="bold"
        )

        ax3.grid(
            axis="y",
            linestyle=":",
            alpha=0.6
        )

        # ==========================================================
        # PANEL 4 — CLINICAL SCORECARD
        # ==========================================================

        ax4 = fig.add_subplot(
            2,
            2,
            4
        )

        ax4.axis("off")

        volume = analysis_results.get(
            "tumor_volume_cm3",
            0.0
        )

        burden = analysis_results.get(
            "tumor_burden_percentage",
            0.0
        )

        if volume is None:
            volume = 0.0

        if burden is None:
            burden = 0.0

        card_text = (
            "AI-ASSISTED LESION ASSESSMENT\n"
            "\n"
            "----------------------------------------\n"
            f"Couinaud Segment : "
            f"{couinaud.get('segment_name', 'Unknown')}\n"
            f"Segment          : "
            f"{couinaud.get('segment_number', 'Unknown')}\n"
            f"Localization      : "
            f"{couinaud.get('confidence', 'Unknown')}\n"
            "\n"
            f"Longest Axis     : "
            f"{recist_mm:.2f} mm\n"
            f"Lesion Volume    : "
            f"{volume:.2f} mL\n"
            f"Tumor Burden     : "
            f"{burden:.2f} %\n"
            "\n"
            "----------------------------------------\n"
            f"Risk Category     : "
            f"{risk.get('risk_category', 'Unknown')}\n"
            f"Estimated Risk    : "
            f"{risk.get('malignancy_probability_percent', 0)} %\n"
            "\n"
            f"Assessment:\n"
            f"{risk.get('lirads_description', '')}\n"
            "\n"
            "----------------------------------------\n"
            "RECOMMENDATION:\n"
            f"{risk.get('clinical_recommendation', '')}\n"
            "\n"
            "----------------------------------------\n"
            "NOTE:\n"
            "Educational AI-assisted estimate only.\n"
            "Not a confirmed medical diagnosis.\n"
            "Radiologist review is recommended."
        )

        bbox_props = dict(
            boxstyle="round,pad=1.0",
            facecolor="#f8f9fa",
            edgecolor="#2b5797",
            linewidth=2.0
        )

        ax4.text(
            0.04,
            0.50,
            card_text,
            transform=ax4.transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            va="center",
            bbox=bbox_props
        )

        # ==========================================================
        # FOOTER
        # ==========================================================

        fig.text(
            0.5,
            0.015,
            "For research and educational use only — "
            "not a substitute for professional medical diagnosis.",
            ha="center",
            fontsize=9
        )

        plt.tight_layout(
            rect=[
                0,
                0.03,
                1,
                0.96
            ]
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150
        )

        plt.close(fig)

        print(
            "[OK] Lesion diagnostic dashboard saved to: "
            f"{output_path}"
        )

    # ==============================================================
    # 6. MAIN PIPELINE
    # ==============================================================

    def run(
        self,
        raw_ct_path="data/test.nii.gz",
        liver_mask_path="data/liver_mask.nii.gz",
        tumor_mask_path="data/tumor_mask.nii.gz",
        dashboard_out="data/lesion_analysis_dashboard.png",
        recist_diameter_mm=None,
        tumor_volume_cm3=None,
        tumor_burden_pct=None,
    ) -> dict:

        print("=" * 60)

        print(
            "[Lesion Analysis Agent] "
            "Starting Quantitative Evaluation"
        )

        print("=" * 60)

        # ==========================================================
        # CHECK FILES
        # ==========================================================

        if not os.path.exists(raw_ct_path):

            raise FileNotFoundError(
                f"CT file not found: {raw_ct_path}"
            )

        if os.path.isdir(raw_ct_path):
            dicom_files = []
            for root, _, files in os.walk(raw_ct_path):
                for filename in files:
                    if filename.lower().endswith((".dcm", ".dicom")):
                        dicom_files.append(os.path.join(root, filename))
            if dicom_files:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(sorted(dicom_files))
                raw_img = reader.Execute()
            else:
                nifti_files = []
                for root, _, files in os.walk(raw_ct_path):
                    for filename in files:
                        if filename.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd")):
                            nifti_files.append(os.path.join(root, filename))
                if not nifti_files:
                    raise ValueError(f"No valid CT volume files were found in the uploaded folder: {raw_ct_path}")
                raw_img = sitk.ReadImage(nifti_files[0])
        else:
            raw_img = sitk.ReadImage(raw_ct_path)

        if not os.path.exists(liver_mask_path):

            raise FileNotFoundError(
                f"Liver mask not found: {liver_mask_path}"
            )

        if not os.path.exists(tumor_mask_path):

            raise FileNotFoundError(
                f"Tumor mask not found: {tumor_mask_path}"
            )

        # ==========================================================
        # LOAD CT
        # ==========================================================

        raw_arr = sitk.GetArrayFromImage(
            raw_img
        )

        # SimpleITK:
        # spacing = (X, Y, Z)

        voxel_spacing = raw_img.GetSpacing()

        print(
            f"[INFO] CT shape: {raw_arr.shape}"
        )

        print(
            f"[INFO] Voxel spacing: {voxel_spacing}"
        )

        # ==========================================================
        # LOAD LIVER MASK
        # ==========================================================

        liv_img = sitk.ReadImage(
            liver_mask_path
        )

        # Resample to CT grid if necessary

        if liv_img.GetSize() != raw_img.GetSize():

            print(
                "[INFO] Resampling liver mask..."
            )

            resample = sitk.ResampleImageFilter()

            resample.SetReferenceImage(
                raw_img
            )

            resample.SetInterpolator(
                sitk.sitkNearestNeighbor
            )

            resample.SetDefaultPixelValue(
                0
            )

            liv_img = resample.Execute(
                liv_img
            )

        liver_mask = (
            sitk.GetArrayFromImage(
                liv_img
            ) > 0
        )

        # ==========================================================
        # LOAD TUMOR MASK
        # ==========================================================

        tum_img = sitk.ReadImage(
            tumor_mask_path
        )

        if tum_img.GetSize() != raw_img.GetSize():

            print(
                "[INFO] Resampling tumor mask..."
            )

            resample = sitk.ResampleImageFilter()

            resample.SetReferenceImage(
                raw_img
            )

            resample.SetInterpolator(
                sitk.sitkNearestNeighbor
            )

            resample.SetDefaultPixelValue(
                0
            )

            tum_img = resample.Execute(
                tum_img
            )

        tumor_mask = (
            sitk.GetArrayFromImage(
                tum_img
            ) > 0
        )

        # ==========================================================
        # MASK CHECK
        # ==========================================================

        liver_voxel_count = int(
            np.sum(liver_mask)
        )

        tumor_voxel_count = int(
            np.sum(tumor_mask)
        )

        print(
            f"[INFO] Liver mask voxels: "
            f"{liver_voxel_count}"
        )

        print(
            f"[INFO] Tumor mask voxels: "
            f"{tumor_voxel_count}"
        )

        if liver_voxel_count == 0:

            print(
                "[WARNING] Liver mask is empty."
            )

        if tumor_voxel_count == 0:

            print(
                "[WARNING] Tumor mask is empty."
            )

        # ==========================================================
        # RADIOMICS
        # ==========================================================

        radiomics = (
            self.extract_radiomics_profile(
                raw_arr,
                liver_mask,
                tumor_mask,
                voxel_spacing,
            )
        )

        # ==========================================================
        # COUINAUD
        # ==========================================================

        couinaud = (
            self.identify_couinaud_segment(
                tumor_mask,
                liver_mask,
                voxel_spacing,
            )
        )

        # ==========================================================
        # RECIST-LIKE LONGEST AXIS
        # ==========================================================

        if recist_diameter_mm is None:

            coords = np.argwhere(
                tumor_mask
            )

            if len(coords) > 400:

                step = max(
                    len(coords) // 400,
                    1
                )

                coords = coords[
                    ::step
                ]

            if len(coords) > 1:

                # Array order = Z,Y,X
                dz = voxel_spacing[2]
                dy = voxel_spacing[1]
                dx = voxel_spacing[0]

                spacing_array = np.array(
                    [dz, dy, dx],
                    dtype=np.float32
                )

                phys = (
                    coords.astype(
                        np.float32
                    )
                    *
                    spacing_array
                )

                try:

                    recist_diameter_mm = float(
                        pdist(
                            phys
                        ).max()
                    )

                except Exception:

                    recist_diameter_mm = 0.0

            else:

                recist_diameter_mm = 0.0

        # ==========================================================
        # RISK ASSESSMENT
        # ==========================================================

        lirads = (
            self.evaluate_lirads_score(
                radiomics,
                recist_diameter_mm,
            )
        )

        # ==========================================================
        # RESULTS
        # ==========================================================

        results = {

            "status": "SUCCESS",

            "recist_diameter_mm":
                round(
                    float(
                        recist_diameter_mm
                    ),
                    2
                ),

            "tumor_volume_cm3":
                tumor_volume_cm3,

            "tumor_burden_percentage":
                tumor_burden_pct,

            "radiomics":
                radiomics,

            "couinaud_localization":
                couinaud,

            "lirads_staging":
                lirads,

            "dashboard_image":
                dashboard_out,
        }

        # ==========================================================
        # DASHBOARD
        # ==========================================================

        self.generate_diagnostic_dashboard(
            raw_arr,
            liver_mask,
            tumor_mask,
            results,
            dashboard_out,
        )

        # ==========================================================
        # CONSOLE SUMMARY
        # ==========================================================

        print("=" * 60)

        print(
            "[Lesion Analysis Agent] "
            "Analysis Summary"
        )

        print("=" * 60)

        print(
            "-> Couinaud Location : "
            f"{couinaud.get('segment_name', 'Unknown')}"
        )

        print(
            "-> Longest Axis      : "
            f"{recist_diameter_mm:.2f} mm"
        )

        print(
            "-> Mean Lesion HU    : "
            f"{radiomics.get('lesion_mean_hu', 0)} HU"
        )

        print(
            "-> Mean Liver HU     : "
            f"{radiomics.get('liver_mean_hu', 0)} HU"
        )

        print(
            "-> Attenuation Diff  : "
            f"{radiomics.get('hu_attenuation_deficit', 0)} HU"
        )

        print(
            "-> Sphericity        : "
            f"{radiomics.get('sphericity', 0)}"
        )

        print(
            "-> AI Risk Category  : "
            f"{lirads.get('risk_category', 'Unknown')}"
        )

        print(
            "-> Estimated Risk    : "
            f"{lirads.get('malignancy_probability_percent', 0)} %"
        )

        print(
            "-> Recommendation    : "
            f"{lirads.get('clinical_recommendation', '')}"
        )

        print("=" * 60)

        return results


# ==============================================================
# DIRECT EXECUTION
# ==============================================================

if __name__ == "__main__":

    agent = LesionAnalysisAgent()

    agent.run()