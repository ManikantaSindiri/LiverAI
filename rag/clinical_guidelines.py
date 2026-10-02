"""
Clinical Oncology & Hepatology Guidelines Knowledge Base & RAG Engine
======================================================================
Contains clinical evidence, treatment pathways, and staging rules from:
- AASLD (American Association for the Study of Liver Diseases) 2018/2023 Guidelines
- EASL (European Association for the Study of the Liver) 2018 Clinical Practice Guidelines
- NCCN (National Comprehensive Cancer Network) Hepatobiliary Cancers Guidelines v2.2023
- BCLC (Barcelona Clinic Liver Cancer) 2022 Strategy Update
- Milan Criteria for Liver Transplantation (Mazzaferro et al.)
"""

import re
from typing import List, Dict, Any


CLINICAL_KNOWLEDGE_BASE = [
    {
        "id": "AASLD-SURG-01",
        "guideline": "AASLD Guidelines for Treatment of Hepatocellular Carcinoma",
        "category": "Surgical Resection",
        "keywords": ["resection", "surgery", "solitary", "early", "segment", "flr", "lr-5"],
        "eligibility": "Solitary HCC of any size in non-cirrhotic liver or Child-Pugh A cirrhosis without portal hypertension and with adequate Future Liver Remnant (FLR > 25-30% in non-cirrhotic, > 40% in cirrhotic).",
        "evidence_level": "Level 1A (Strong Recommendation, High Quality Evidence)",
        "recommendation": "Surgical partial hepatectomy / anatomical segmentectomy is the primary curative treatment of choice for resectable solitary lesions with adequate functional reserve.",
        "reference": "Heimbach JK, et al. AASLD guidelines for the treatment of hepatocellular carcinoma. Hepatology. 2018;67(1):358-380.",
    },
    {
        "id": "EASL-ABLAT-02",
        "guideline": "EASL Clinical Practice Guidelines: Management of Hepatocellular Carcinoma",
        "category": "Thermal Ablation (RFA / MWA)",
        "keywords": ["ablation", "rfa", "mwa", "radiofrequency", "microwave", "subcentimeter", "small"],
        "eligibility": "Solitary tumors <= 30 mm (or up to 3 nodules <= 30 mm) when surgical resection is contraindicated or as an alternative first-line option.",
        "evidence_level": "Level 1A (Strong Recommendation)",
        "recommendation": "Radiofrequency ablation (RFA) or Microwave ablation (MWA) provides local tumor control comparable to resection for small lesions (< 3 cm) and can be considered for deep parenchymal lesions.",
        "reference": "European Association for the Study of the Liver. EASL Clinical Practice Guidelines: Management of hepatocellular carcinoma. J Hepatol. 2018;69(1):182-236.",
    },
    {
        "id": "MILAN-TRANSPLANT-03",
        "guideline": "Milan Criteria for Orthotopic Liver Transplantation",
        "category": "Liver Transplantation",
        "keywords": ["transplant", "milan", "milan criteria", "cirrhosis", "in-milan"],
        "eligibility": "Single tumor <= 5.0 cm, OR up to 3 tumors each <= 3.0 cm, without macrovascular invasion or extrahepatic spread.",
        "evidence_level": "Level 1A (Universal Standard Criteria)",
        "recommendation": "Patients meeting Milan criteria achieve a 5-year post-transplant survival rate of > 70% with low recurrence (< 15%). Recommended for transplant evaluation if underlying parenchymal disease is present.",
        "reference": "Mazzaferro V, et al. Liver transplantation for the treatment of small hepatocellular carcinomas in patients with cirrhosis. N Engl J Med. 1996;334(11):693-700.",
    },
    {
        "id": "BCLC-STAGE-A-04",
        "guideline": "BCLC 2022 Prognostic and Treatment Algorithm",
        "category": "BCLC Stage A (Early Stage)",
        "keywords": ["bclc", "stage a", "early stage", "solitary", "preserved liver function"],
        "eligibility": "Single nodule of any size (or up to 3 nodules <= 3 cm), preserved liver function (Child-Pugh A), Performance Status 0.",
        "evidence_level": "Level 1A (International Oncology Standard)",
        "recommendation": "Curative therapeutic intent: Resection, Liver Transplantation, or Thermal Ablation. 5-year survival rate expected to exceed 60-70%.",
        "reference": "Reig M, et al. BCLC strategy for prognosis prediction and treatment recommendation: The 2022 update. J Hepatol. 2022;76(3):681-693.",
    },
    {
        "id": "NCCN-TACE-05",
        "guideline": "NCCN Clinical Practice Guidelines in Oncology: Hepatobiliary Cancers",
        "category": "Locoregional Therapy (TACE / TARE)",
        "keywords": ["tace", "chemoembolization", "tare", "intermediate", "unresectable", "bridge"],
        "eligibility": "Multinodular disease (BCLC Stage B), or solitary lesions unresectable due to anatomical location or medical comorbidities, or as bridging therapy to transplant.",
        "evidence_level": "Level 2A (Standard Recommendation)",
        "recommendation": "Conventional or drug-eluting bead TACE (DEB-TACE) or Yttrium-90 radioembolization (TARE) is recommended for non-curative candidates with preserved hepatic function.",
        "reference": "National Comprehensive Cancer Network. Hepatobiliary Cancers (Version 2.2023). NCCN Guidelines.",
    },
    {
        "id": "ACR-LIRADS-06",
        "guideline": "ACR LI-RADS v2018 Core Diagnostic Criteria",
        "category": "Diagnostic Staging & Management",
        "keywords": ["li-rads", "lr-5", "lr-4", "lr-3", "hcc", "washout", "capsule", "diameter"],
        "eligibility": "LR-5: Mass >= 20 mm with non-peripheral washout appearance or enhancing capsule, or mass 10-19 mm with APHE and washout. LR-4: Probably malignant.",
        "evidence_level": "Level 1 (Expert Consensus & Diagnostic Validation)",
        "recommendation": "LR-5 category has > 95% positive predictive value for HCC. Biopsy is not strictly mandatory prior to treatment in high-risk patients with LR-5 lesions.",
        "reference": "American College of Radiology. CT/MRI LI-RADS v2018 Core. ACR.org; 2018.",
    },
]


class ClinicalGuidelinesRAG:
    """
    RAG engine for retrieving clinical oncology and hepatology guidelines.
    """

    def __init__(self, knowledge_base: List[Dict[str, Any]] = None):
        self.kb = knowledge_base if knowledge_base is not None else CLINICAL_KNOWLEDGE_BASE

    def retrieve_guidelines(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves matching guidelines based on keyword matching and relevance scoring.
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored_entries = []

        for entry in self.kb:
            score = 0
            # Match keywords
            for kw in entry["keywords"]:
                if kw in query.lower() or any(token in kw for token in query_tokens):
                    score += 3
            
            # Match text in eligibility and recommendation
            text = f"{entry['eligibility']} {entry['recommendation']} {entry['category']}".lower()
            for token in query_tokens:
                if token in text:
                    score += 1

            if score > 0:
                scored_entries.append((score, entry))

        # Sort by relevance score
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_entries[:top_k]]


if __name__ == "__main__":
    rag = ClinicalGuidelinesRAG()
    results = rag.retrieve_guidelines("solitary liver lesion 46mm LR-5 resection segment V", top_k=3)
    print(f"[OK] Retrieved {len(results)} guidelines for query:")
    for r in results:
        print(f"-> [{r['id']}] {r['category']} ({r['evidence_level']})")
