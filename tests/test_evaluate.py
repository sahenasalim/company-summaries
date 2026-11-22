"""Tests for summarizer.evaluate.

The three summaries below are real v1_baseline outputs that were checked by hand
against the data, so these tests lock in what the manual review found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from summarizer.evaluate import (
    mentions,
    parse_figures,
    score_summary,
    summarise,
    unsupported_figures,
    word_count,
)
from summarizer.facts import extract_facts, load_companies

DATA = Path(__file__).resolve().parents[1] / "data" / "companies.jsonl"

V1_067 = (
    "**Company Overview:**  \n* **Name:** Jaya Prima Travel Sdn. Bhd. (SYN-000067)  \n* **Profile:** Active private "
    "limited company incorporated in April 1995 in Kuala Lumpur. Operates as a travel agency specializing in Umrah and "
    "inbound tourism.  \n* **Leadership:** Led by Managing Director (Director 1, holding 47.38% shares), alongside three "
    "other directors.  \n\n**Financial & Risk Assessment:**  \n* **Revenue & Profitability:** Experiencing a downward "
    "revenue trend (from 49.47M MYR in 2022 to 43.29M MYR in 2024). The company is consistently unprofitable, with net "
    "losses widening from 4.24M MYR (2022) to 7.52M MYR (2024).  \n* **Liquidity & Cash Flow:** Weak and deteriorating "
    "liquidity. Operating cash flow remains negative (5.94M MYR outflow in 2024), and net working capital turned "
    "negative (-4.23M MYR in 2024).  \n* **Audit:** Received clean (unqualified) audit opinions from Audit Firm 7 across "
    "all three years. No recorded litigation.  \n\n*Recommendation:* High financial risk due to consecutive losses and "
    "negative cash flows."
)
V1_148 = (
    "### **Company Overview**\n* **Name:** Indah Prima Medical Sdn. Bhd. (SYN-000148)\n* **Status:** Active "
    "(Incorporated: April 1993, Kelantan)\n* **Business:** Community pharmacy and health supplements\n* **Key "
    "Stakeholders:** Majority shareholder Director 4 (75.61%), managed by Director 1.\n\n### **Financial Health & Risk "
    "Assessment**\n* **Revenue Trend:** Declining year-on-year, dropping from MYR 5.53M (FY2022) to MYR 4.84M "
    "(FY2024).\n* **Profitability:** Highly concerning. The company has posted widening net losses: MYR -516k (FY2022), "
    "MYR -824k (FY2023), and MYR -1.08M (FY2024). \n* **Cash Flow & Liquidity:** Negative operating cash flows "
    "worsening annually (MYR -819k in FY2024). Negative net working capital (MYR -448k) and a weak quick ratio (0.59) "
    "indicate short-term liquidity stress.\n* **Audit:** Received clean (unqualified) audit opinions from Audit Firm 5 "
    "across all three years. No recorded litigation."
)
V1_029 = (
    "### Company Overview: Unity Global Trading Sdn. Bhd.\n* **Status & Profile:** Active private limited company "
    "(Selangor, Malaysia) incorporated in June 2018. Operates in retail (textiles and *baju kurung*). Share capital: "
    "MYR 250,000.\n* **Leadership & Ownership:** Managed by 5 directors. Major shareholders are Director 4 (58.25%) and "
    "Director 5 (21.22%).\n* **Financial Performance (FY2022–FY2024):** \n  * Consistent revenue growth from MYR 5.13M "  # noqa: RUF001 - verbatim model output
    "(FY2022) to MYR 5.35M (FY2024). \n  * Stable profitability with gross margins improving to 35.1% in FY2024, "
    "though net profit margins remain modest (~1.8%). \n  * Clean financial audits with unqualified opinions across all "
    "years.\n* **Risk Assessment:** No recorded litigation. Liquidity is healthy (quick ratio ~1.35), though debt "
    "service coverage is tight (1.07 in FY2024). Suitable for partnership with standard credit monitoring."
)


@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("data/companies.jsonl not found")
    return {r["registrationNumber"]: r for r in load_companies(DATA)}


# ---------------------------------------------------------------- word count
def test_word_count_ignores_markdown_symbols():
    assert word_count("### **Title**\n* one two\n* three") == 4


# ------------------------------------------------------------- number parsing
def test_parse_figures_units_and_scales():
    figs, _ = parse_figures("RM5.35m, MYR 5.13M, 49.47M MYR, -516k, RM564,100, 35.1%")
    values = [f.value for f in figs]
    assert values == pytest.approx([5.35e6, 5.13e6, 49.47e6, 516e3, 564_100, 35.1])
    assert figs[-1].is_percent


def test_parse_figures_skips_identifiers_small_counts_and_years():
    figs, years = parse_figures("SYN-000067 had Director 4 and Audit Firm 7, 5 directors, 12 months, in FY2022 and 2024")
    assert figs == []
    assert years == [2022, 2024]


# ------------------------------------------------------------- faithfulness
def _allowed(real, reg):
    from summarizer.evaluate import allowed_values

    return allowed_values(real[reg], extract_facts(real[reg]))


def test_correct_figures_pass_and_wrong_figure_is_caught(real):
    allowed = _allowed(real, "SYN-000067")
    assert unsupported_figures("Revenue was RM43.29m in FY2024, a loss of RM7.52m.", allowed) == []
    assert unsupported_figures("Revenue was RM9.99m in FY2024.", allowed) == ["9.99m"]


def test_invented_year_is_caught(real):
    assert unsupported_figures("Founded in 1971.", _allowed(real, "SYN-000067")) == ["1971"]


def test_derived_percentage_is_accepted(real):
    # revenue 49,467,702 -> 43,292,441 is a fall of 12.5%
    assert unsupported_figures("Revenue fell 12% over two years.", _allowed(real, "SYN-000067")) == []


# ---------------------------------------------------------------- mentions
def test_unqualified_is_not_an_adverse_audit_mention(real):
    facts = extract_facts(real["SYN-000217"])
    assert not mentions("AUDIT_ADVERSE", "Received unqualified audit opinions.", facts)
    assert mentions("AUDIT_ADVERSE", "The auditor issued a disclaimer of opinion in FY2023.", facts)


def test_status_mention_uses_status_text(real):
    facts = extract_facts(real["SYN-001212"])
    assert mentions("STATUS_NOT_ACTIVE", "The company is in winding up.", facts)
    assert not mentions("STATUS_NOT_ACTIVE", "An active retailer in Selangor.", facts)


# --------------------------------------------- regression on checked outputs
def test_v1_numbers_in_checked_summaries_are_all_supported(real):
    for reg, text in [("SYN-000067", V1_067), ("SYN-000148", V1_148), ("SYN-000029", V1_029)]:
        assert score_summary("v1_baseline", real[reg], text).unsupported == [], reg


def test_v1_length_finding(real):
    assert score_summary("v1", real["SYN-000067"], V1_067).words > 120
    assert score_summary("v1", real["SYN-000148"], V1_148).words <= 120
    assert score_summary("v1", real["SYN-000029"], V1_029).words <= 120


def test_v1_unrequested_recommendations_are_detected(real):
    assert score_summary("v1", real["SYN-000029"], V1_029).gives_recommendation
    assert score_summary("v1", real["SYN-000067"], V1_067).gives_recommendation
    assert not score_summary("v1", real["SYN-000148"], V1_148).gives_recommendation


def test_v1_missed_interest_cover_on_both_loss_makers(real):
    for reg, text in [("SYN-000067", V1_067), ("SYN-000148", V1_148)]:
        score = score_summary("v1", real[reg], text)
        assert score.missed == {"WEAK_INTEREST_COVER"}, reg
        assert score.missed_high == set()


def test_clean_company_requires_no_signals(real):
    score = score_summary("v1", real["SYN-000029"], V1_029)
    assert score.required == set()
    assert score.states_latest_year


# --------------------------------------------------------- litigation alarms
def _closed_cases_only(real):
    for rec in real.values():
        lit = extract_facts(rec)["litigation"]
        if lit["total"] > 0 and lit["open_defendant_count"] == 0:
            return rec
    pytest.skip("no such company in data")


def test_closed_cases_presented_as_threat_are_flagged(real):
    rec = _closed_cases_only(real)
    alarm = score_summary("v", rec, "Faces significant litigation exposure from court cases.")
    calm = score_summary("v", rec, "No recorded litigation of concern.")
    assert alarm.litigation_eligible
    assert alarm.litigation_alarm
    assert not calm.litigation_alarm


# -------------------------------------------------------------- aggregation
def test_summarise_counts_recall_and_handles_empty(real):
    scores = [
        score_summary("v1", real["SYN-000067"], V1_067),
        score_summary("v1", real["SYN-000148"], V1_148),
    ]
    summary = summarise(scores)
    assert summary["summaries"] == 2
    assert summary["per_code"]["WEAK_INTEREST_COVER"] == (0, 2)
    assert summary["recall_high"] == (2, 2)
    assert summarise([])["summaries"] == 0
