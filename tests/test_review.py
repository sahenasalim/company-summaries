"""Tests for the blind review tool, using fabricated summaries for every company."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarizer.facts import extract_facts, load_companies
from summarizer.review import make_review, pick_companies, score_review

DATA = Path(__file__).resolve().parents[1] / "data" / "companies.jsonl"
VERSIONS = ("v1_baseline", "v2_structured")


@pytest.fixture
def workspace(tmp_path):
    if not DATA.exists():
        pytest.skip("data/companies.jsonl not found")
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    for version in VERSIONS:
        rows = [{"registrationNumber": r["registrationNumber"], "summary": f"{version} text"} for r in load_companies(DATA)]
        (summaries / f"{version}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def run_make(workspace, seed=42):
    return make_review(VERSIONS, data_path=DATA, summaries_dir=workspace / "summaries", out_dir=workspace / "review", seed=seed)


def test_selection_is_deterministic_and_includes_hard_cases():
    records = load_companies(DATA)
    first, second = pick_companies(records), pick_companies(records)
    assert first == second
    assert 8 <= len(first) <= 12
    by_id = {r["registrationNumber"]: extract_facts(r) for r in records}
    assert any(not by_id[i]["financials"]["years"] for i in first)  # a company with no financials
    assert any(by_id[i]["litigation"]["total"] > 0 and by_id[i]["litigation"]["open_defendant_count"] == 0 for i in first)


def test_make_writes_files_and_key_maps_each_label_to_a_different_version(workspace):
    ids = run_make(workspace)
    key = json.loads((workspace / "review" / "key.json").read_text(encoding="utf-8"))
    assert sorted(key) == sorted(ids)
    for labels in key.values():
        assert {labels["A"], labels["B"]} == set(VERSIONS)
    assert "Do not open `key.json`" in (workspace / "review" / "blind_review.md").read_text(encoding="utf-8")


def test_labels_are_actually_shuffled(workspace):
    run_make(workspace)
    key = json.loads((workspace / "review" / "key.json").read_text(encoding="utf-8"))
    assert {labels["A"] for labels in key.values()} == set(VERSIONS)


def test_score_counts_wins_ties_and_unanswered(workspace):
    ids = run_make(workspace)
    review = workspace / "review" / "blind_review.md"
    key = json.loads((workspace / "review" / "key.json").read_text(encoding="utf-8"))

    text = review.read_text(encoding="utf-8")
    answers = ["A", "B", "tie"] + [""] * (len(ids) - 3)
    for reg, answer in zip(ids, answers, strict=True):
        marker = f"## {ids.index(reg) + 1}. {reg}"
        start = text.index(marker)
        line = "**Preferred (A / B / tie):** "
        pos = text.index(line, start)
        text = text[: pos + len(line)] + answer + text[pos + len(line) :]
    review.write_text(text, encoding="utf-8")

    result = score_review(review, workspace / "review" / "key.json")
    assert result["wins"].get("tie") == 1
    assert result["wins"][key[ids[0]]["A"]] >= 1
    assert result["wins"][key[ids[1]]["B"]] >= 1
    assert sorted(result["unanswered"]) == sorted(ids[3:])
    assert result["total"] == len(ids)
