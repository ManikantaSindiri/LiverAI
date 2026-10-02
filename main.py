import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from agents.preprocessing_agent import PreprocessingAgent
from agents.segmentation_agent import SegmentationAgent
from agents.lesion_analysis_agent import LesionAnalysisAgent


def load_ct_volume(raw_ct_path):
    try:
        image = sitk.ReadImage(raw_ct_path)
    except RuntimeError as exc:
        raise ValueError("The uploaded file could not be read as a CT volume.") from exc

    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            "Upload a 3D, single-channel CT volume (NIfTI or supported 3D DICOM). "
            "A 2D image or individual DICOM slice cannot be segmented."
        )

    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise ValueError("The uploaded CT volume must have axial, coronal, and sagittal dimensions.")
    return image, array


def run_pipeline(raw_ct_path):
    processed_ct_path = "data/processed_test.nii.gz"
    qa_plot_path = "data/preprocessing_comparison.png"
    slice_preview_path = "data/liver_ct_slice.png"
    liver_mask_path = "data/liver_mask.nii.gz"
    tumor_mask_path = "data/tumor_mask.nii.gz"
    overlay_plot_path = "data/segmentation_overlay.png"
    dashboard_plot_path = "data/lesion_analysis_dashboard.png"

    # Step 1: Ensure raw dataset is availa
    # Step 2: Milestone 1 Basic Visualization
    img, arr = load_ct_volume(raw_ct_path)
    mid = arr.shape[0] // 2

    plt.figure(figsize=(7, 7))
    plt.imshow(arr[mid], cmap="gray", vmin=-75, vmax=175)
    plt.title(f"Liver CT - Axial Slice {mid}/{arr.shape[0]}")
    plt.axis("off")
    plt.savefig(slice_preview_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Raw slice preview saved to: {slice_preview_path}")

    # Step 3: Module 1 - CT Preprocessing Agent
    preproc_agent = PreprocessingAgent(
        target_spacing=(1.5, 1.5, 1.5),
        window_center=50.0,
        window_width=200.0,
        normalize_range=(0.0, 1.0),
    )
    preproc_report = preproc_agent.run(
        input_path=raw_ct_path,
        output_path=processed_ct_path,
        qa_plot_path=qa_plot_path,
    )

    # Step 4: Module 2 - Liver & Lesion Segmentation Agent
    seg_agent = SegmentationAgent()
    seg_report = seg_agent.run(
        preprocessed_ct_path=processed_ct_path,
        liver_mask_out=liver_mask_path,
        tumor_mask_out=tumor_mask_path,
        overlay_plot_out=overlay_plot_path,
    )
    seg_metrics = seg_report["metrics"]

    # Step 5: Module 3 - Lesion Analysis Agent
    analysis_agent = LesionAnalysisAgent()
    analysis_report = analysis_agent.run(
        raw_ct_path=raw_ct_path,
        liver_mask_path=liver_mask_path,
        tumor_mask_path=tumor_mask_path,
        dashboard_out=dashboard_plot_path,
        recist_diameter_mm=seg_metrics["recist_max_diameter_mm"],
        tumor_volume_cm3=seg_metrics["tumor_volume_cm3"],
        tumor_burden_pct=seg_metrics["tumor_burden_percentage"],
    )

    # Final Integrated Clinical Decision Support Summary
    radiomics = analysis_report["radiomics"]
    couinaud = analysis_report["couinaud_localization"]
    lirads = analysis_report["lirads_staging"]

    print("\n" + "=" * 70)
    print("        LIVER MULTI-AGENT CLINICAL DECISION SUPPORT REPORT")
    print("=" * 70)
    print(" [1] CT PREPROCESSING AGENT")
    print(f"     • Raw Volume Dimensions  : {preproc_report['raw_shape_zyx']} @ {preproc_report['raw_spacing_mm']} mm")
    print(f"     • Isotropic Standardization: {preproc_report['processed_shape_zyx']} @ {preproc_report['processed_spacing_mm']} mm")
    print(f"     • Multi-Planar QA Chart   : {preproc_report['qa_plot']}")
    print("-" * 70)
    print(" [2] LIVER & LESION SEGMENTATION AGENT")
    print(f"     • Total Liver Volume (TLV): {seg_metrics['total_liver_volume_cm3']} cm³ (mL)")
    print(f"     • Tumor / Lesion Volume  : {seg_metrics['tumor_volume_cm3']} cm³ (mL)")
    print(f"     • Hepatic Tumor Burden   : {seg_metrics['tumor_burden_percentage']} %")
    print(f"     • RECIST 1.1 Longest Axis: {seg_metrics['recist_max_diameter_mm']} mm")
    print(f"     • 3D Segmentation Overlay: {seg_report['overlay_plot']}")
    print("-" * 70)
    print(" [3] LESION ANALYSIS AGENT (RADIOMICS & CLINICAL STAGING)")
    print(f"     • Anatomical Location    : {couinaud['segment_name']}")
    print(f"     • Attenuation (HU)       : Lesion Mean {radiomics['lesion_mean_hu']} HU vs Normal Liver {radiomics['liver_mean_hu']} HU")
    print(f"     • Contrast Deficit       : {radiomics['hu_attenuation_deficit']} HU (Ratio: {radiomics['tumor_to_liver_contrast_ratio']})")
    print(f"     • 3D Sphericity Index    : {radiomics['sphericity']} (Compactness: {radiomics['compactness']})")
    print(f"     • ACR LI-RADS Staging    : {lirads['lirads_category']} - {lirads['lirads_description']}")
    print(f"     • Malignancy Probability : {lirads['malignancy_probability_percent']} %")
    print(f"     • Diagnostic Dashboard   : {analysis_report['dashboard_image']}")
    print("=" * 70)
    print(f" [CLINICAL ACTION PLAN]:\n  >> {lirads['clinical_recommendation']}")
    print("=" * 70)
    return {
        "preprocessing": preproc_report,
        "segmentation": seg_report,
        "analysis": analysis_report,
    }

if __name__ == "__main__":
    run_pipeline("data/test.nii.gz")

