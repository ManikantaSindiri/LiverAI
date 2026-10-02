from types import SimpleNamespace

import pytest

import app.app as liver_app
from rag.clinical_guidelines import CLINICAL_KNOWLEDGE_BASE


def test_guideline_sources_include_requested_organizations():
    sources = {liver_app.guideline_source(entry) for entry in CLINICAL_KNOWLEDGE_BASE}

    assert {"AASLD", "EASL", "NCCN", "BCLC", "LI-RADS"}.issubset(sources)
    assert len(CLINICAL_KNOWLEDGE_BASE) == 6


def test_chat_sends_retrieved_guidelines_and_returns_citations(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    request_data = {}

    def fake_post(url, *, headers, json, timeout):
        request_data.update(url=url, headers=headers, json=json, timeout=timeout)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "Resection is considered in selected cases [AASLD-SURG-01]."}}]},
        )

    monkeypatch.setattr(liver_app.requests, "post", fake_post)

    answer, sources = liver_app.generate_guideline_chat_response("solitary lesion resection", [])

    assert "[AASLD-SURG-01]" in answer
    assert any(entry["id"] == "AASLD-SURG-01" for entry in sources)
    assert request_data["url"].endswith("/v1/chat/completions")
    assert request_data["headers"]["Authorization"] == "Bearer test-only-key"
    assert "AASLD-SURG-01" in request_data["json"]["messages"][-1]["content"]
    assert request_data["json"]["model"] == "test-model"


def test_chat_requires_local_api_key(monkeypatch):
    monkeypatch.setattr(liver_app, "openai_settings", lambda: ("", "gpt-4o-mini"))

    with pytest.raises(RuntimeError, match="Set OPENAI_API_KEY"):
        liver_app.generate_guideline_chat_response("What is LI-RADS?", [])
