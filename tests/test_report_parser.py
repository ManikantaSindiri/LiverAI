from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter

from utils.report_parser import analyze_report_text, extract_report_text


def test_extracts_text_from_docx():
    document = Document()
    document.add_paragraph("Impression: Stable liver lesion.")
    output = BytesIO()
    document.save(output)

    assert extract_report_text("report.docx", output.getvalue()) == "Impression: Stable liver lesion."


def test_rejects_scanned_or_empty_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)

    with pytest.raises(ValueError, match="No selectable text"):
        extract_report_text("scanned.pdf", output.getvalue())


def test_report_summary_keeps_findings_and_impression_source_grounded():
    summary = analyze_report_text(
        "Findings: 2 cm liver lesion.\nImpression: Follow-up recommended."
    )

    assert [section["title"] for section in summary["sections"]] == ["Findings", "Impression"]
    assert any("2 cm liver lesion" in point for point in summary["key_points"])
    assert any("Follow-up recommended" in point for point in summary["key_points"])
    assert any("surrounding tissue" in item for item in summary["plain_language"])
