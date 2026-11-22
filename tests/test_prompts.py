"""Tests that each prompt version contains what it should (and nothing it shouldn't)."""

from __future__ import annotations

from summarizer.facts import extract_facts
from summarizer.prompts import PROMPT_BUILDERS, build_v1_baseline, build_v2_structured

RECORD = {
    "registrationNumber": "SYN-000001",
    "profile": {
        "companyName": "Test Sdn. Bhd.",
        "status": "Winding up",
        "companyType": "Private limited company",
        "incorporationDate": "2015-01-01",
        "industry": "Retail",
        "state": "Selangor",
        "natureOfBusiness": "Retail",
        "shareCapital": 100000,
        "formerNames": [],
    },
    "relationships": {
        "directors": [{"name": "Director 1", "designation": "Director", "appointedOn": "2016-01-01"}],
        "shareholders": [{"type": "individual", "name": "Director 1", "percentage": 100.0}],
    },
    "financials": {"years": []},
    "litigation": {"cases": []},
}


def test_every_version_returns_system_and_user_text():
    facts = extract_facts(RECORD)
    for build in PROMPT_BUILDERS.values():
        system, user = build(RECORD, facts)
        assert system
        assert "Test Sdn. Bhd." in user


def test_v1_sends_raw_json_without_computed_signals():
    _, user = build_v1_baseline(RECORD, extract_facts(RECORD))
    assert "registrationNumber" in user
    assert "Risk and context signals" not in user


def test_v2_sends_computed_signals_not_raw_json():
    _, user = build_v2_structured(RECORD, extract_facts(RECORD))
    assert "Winding up" in user
    assert "Risk and context signals" in user
    assert "registrationNumber" not in user
    assert "NONE AVAILABLE" in user
