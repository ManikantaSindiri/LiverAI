"""Text extraction and conservative summaries for uploaded medical reports."""

from io import BytesIO
from pathlib import Path
import re
from typing import Dict, List

from docx import Document
from pypdf import PdfReader


SECTION_HEADINGS = (
    "clinical history",
    "technique",
    "findings",
    "impression",
    "conclusion",
    "recommendation",
    "recommendations",
)
SECTION_PATTERN = re.compile(
    r"(?im)^\s*(clinical history|technique|findings|impression|conclusion|recommendations?)\s*:\s*(.*)$"
)
MEDICAL_TERMS = {
    "lesion": "An area that looks different from the surrounding tissue; the word alone does not say whether it is cancer.",
    "benign": "Non-cancerous.",
    "malignant": "Cancerous; discuss the report wording with your treating clinician.",
    "follow-up": "A later appointment or test to check for change.",
    "stable": "Described as unchanged compared with the prior study referenced in the report.",
    "indeterminate": "The report does not characterize this finding conclusively.",
}
MAX_REPORT_CHARACTERS = 40000


def extract_report_text(filename: str, content: bytes) -> str:
    """Extract selectable text from PDF, DOCX, or plain-text uploads."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise ValueError("This PDF could not be read. Check that the file is valid and try again.") from exc
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension == ".docx":
        document = Document(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    elif extension == ".txt":
        text = content.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Upload a PDF, DOCX, or TXT report.")

    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; upload a text-based report."
        )
    return text[:MAX_REPORT_CHARACTERS]


def analyze_report_text(text: str) -> Dict[str, object]:
    """Return source-grounded report sections and cautious plain-language context."""
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    matches = list(SECTION_PATTERN.finditer(normalized))
    sections: List[Dict[str, str]] = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        body = "\n".join(
            part.strip()
            for part in (match.group(2), normalized[match.end():end])
            if part.strip()
        )
        if body:
            sections.append({"title": match.group(1).title(), "text": body[:6000]})

    if not sections:
        sections.append({"title": "Report text", "text": normalized[:6000]})

    sentences = [
        sentence.strip(" -\t")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if sentence.strip(" -\t")
    ]
    relevant = [
        sentence
        for sentence in sentences
        if re.search(r"liver|lesion|mass|nodule|tumou?r|cyst|impression|follow.?up|\d+(?:\.\d+)?\s*(?:mm|cm)", sentence, re.I)
    ]
    key_points = (relevant or sentences)[:5]

    lower_text = normalized.lower()
    explanations = [
        definition
        for term, definition in MEDICAL_TERMS.items()
        if term in lower_text
    ]
    explanations.insert(
        0,
        "Findings describe observations; Impression or Conclusion is the report author's overall summary.",
    )

    return {
        "sections": sections,
        "key_points": key_points,
        "plain_language": explanations,
        "source_text": normalized[:MAX_REPORT_CHARACTERS],
    }
