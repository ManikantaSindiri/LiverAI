"""
Module 4: Evidence Retrieval Agent
==================================
Responsible for:
1. Interrogating the Clinical Oncology & Hepatology RAG Knowledge Base
2. Matching patient radiomics, lesion size, and Couinaud location against international guidelines
3. Extracting clinical trial references, evidence levels, and treatment recommendations
"""

import os
import sys
from typing import Dict, Any

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.clinical_guidelines import ClinicalGuidelinesRAG


class EvidenceAgent:
    """
    Autonomous Evidence Retrieval Agent querying international clinical guidelines.
    """

    def __init__(self):
        self.rag = ClinicalGuidelinesRAG()

    def query_evidence(
        self,
        recist_diameter_mm: float,
        lirads_category: str,
        couinaud_segment: str,
        tumor_burden_pct: float,
        total_liver_vol_cm3: float,
    ) -> Dict[str, Any]:
        """
        Retrieves matching guidelines based on quantitative findings.
        """
        # Formulate targeted clinical query
        query = (
            f"solitary liver tumor {recist_diameter_mm}mm {lirads_category} "
            f"segment {couinaud_segment} resection ablation transplantation"
        )
        matched_guidelines = self.rag.retrieve_guidelines(query, top_k=4)

        # Evaluate Milan Criteria Status
        in_milan = recist_diameter_mm <= 50.0  # Single tumor <= 5.0 cm (50mm)
        milan_status = "In-Milan (Eligible for Liver Transplantation)" if in_milan else "Out-of-Milan (> 50 mm)"

        # Primary Curative Recommendation
        if recist_diameter_mm <= 30.0:
            primary_intent = "Thermal Ablation (RFA/MWA) or Anatomical Segmental Resection"
            rationale = "Tumors <= 30 mm achieve comparable long-term local control with thermal ablation or surgical resection."
        elif recist_diameter_mm <= 50.0:
            primary_intent = "Surgical Anatomical Partial Hepatectomy / Segmentectomy"
            rationale = "Solitary lesions between 30mm and 50mm are best managed by anatomical surgical resection when liver remnant is adequate."
        else:
            primary_intent = "Multidisciplinary Surgical Evaluation or Downstaging / TACE"
            rationale = "Large solitary lesions require detailed vascular invasion assessment and future liver remnant volumetric modeling."

        evidence_report = {
            "status": "SUCCESS",
            "search_query": query,
            "milan_criteria_status": milan_status,
            "primary_treatment_pathway": primary_intent,
            "clinical_rationale": rationale,
            "retrieved_guidelines": matched_guidelines,
        }

        return evidence_report

    def run(
        self,
        recist_diameter_mm: float,
        lirads_category: str,
        couinaud_segment: str,
        tumor_burden_pct: float,
        total_liver_vol_cm3: float,
    ) -> Dict[str, Any]:
        """
        Runs the Evidence Retrieval Agent.
        """
        print("=" * 60)
        print(" [Evidence Agent] Searching Clinical Oncology Knowledge Base")
        print("=" * 60)
        print(f"-> Query Context: Size {recist_diameter_mm} mm | {lirads_category} | {couinaud_segment}")

        report = self.query_evidence(
            recist_diameter_mm,
            lirads_category,
            couinaud_segment,
            tumor_burden_pct,
            total_liver_vol_cm3,
        )

        print(f"-> Milan Criteria Status   : {report['milan_criteria_status']}")
        print(f"-> Primary Recommendation  : {report['primary_treatment_pathway']}")
        print(f"-> Retrieved Guidelines ({len(report['retrieved_guidelines'])}):")
        for g in report["retrieved_guidelines"]:
            print(f"   • [{g['id']}] {g['category']} ({g['evidence_level']})")
        print("=" * 60)

        return report


if __name__ == "__main__":
    agent = EvidenceAgent()
    agent.run(
        recist_diameter_mm=46.74,
        lirads_category="LR-5",
        couinaud_segment="Segment V",
        tumor_burden_pct=2.70,
        total_liver_vol_cm3=1054.02,
    )
