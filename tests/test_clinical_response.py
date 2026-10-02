from app.app import build_clinical_summary


def test_build_clinical_summary_includes_core_sections():
    metrics = {
        "total_liver_volume_cm3": 1054.02,
        "tumor_volume_cm3": 28.49,
        "tumor_burden_percentage": 2.7,
        "recist_max_diameter_mm": 46.74,
    }
    radiomics = {
        "lesion_mean_hu": 28.0,
        "liver_mean_hu": 65.0,
        "hu_attenuation_deficit": 37.0,
        "sphericity": 1.0,
        "compactness": 0.1046,
    }
    location = {"segment_name": "Right Anteroinferior (Segment V)"}
    lirads = {
        "lirads_category": "High Risk",
        "lirads_description": "The analyzed lesion demonstrates features associated with a higher-risk imaging pattern.",
        "malignancy_probability_percent": 95.0,
        "clinical_recommendation": "Further evaluation by a qualified radiologist using appropriate multiphasic contrast-enhanced CT or MRI is recommended.",
    }

    summary = build_clinical_summary(metrics, radiomics, location, lirads)

    assert summary["summary_title"] == "AI Clinical Report"
    assert "High Risk" in summary["findings"]
    assert "46.74 mm" in summary["findings"]
    assert "Further evaluation" in summary["recommendation"]
    assert summary["risk_percent"] == 95.0
