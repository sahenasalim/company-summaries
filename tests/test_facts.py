"""Tests for summarizer.facts.

Two layers:
1. Unit tests on tiny synthetic records, so each rule is checked in isolation.
2. Smoke tests on the real dataset, to catch surprises in the actual data.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from summarizer.facts import (
    extract_facts,
    facts_to_text,
    flag_codes,
    fmt_pct,
    fmt_rm,
    load_companies,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "companies.jsonl"


# --------------------------------------------------------------------- builders
def make_year(year: int, **overrides):
    statements = {
        "revenue": 1_000_000,
        "costOfSales": 600_000,
        "grossProfit": 400_000,
        "operatingExpenses": 300_000,
        "operatingProfit": 100_000,
        "depreciation": 10_000,
        "interestExpense": 20_000,
        "profitBeforeTax": 80_000,
        "taxExpense": 20_000,
        "profitAfterTax": 60_000,
        "cash": 100_000,
        "tradeReceivables": 50_000,
        "inventory": 50_000,
        "currentAssets": 400_000,
        "nonCurrentAssets": 600_000,
        "totalAssets": 1_000_000,
        "tradePayables": 50_000,
        "borrowings": 200_000,
        "currentLiabilities": 200_000,
        "totalLiabilities": 400_000,
        "totalEquity": 600_000,
        "operatingCashFlow": 90_000,
        "capitalExpenditure": 20_000,
        "borrowingsRepaid": 10_000,
    }
    opinion = overrides.pop("opinion", "Unqualified")
    statements.update(overrides)
    return {
        "financialYear": year,
        "financialYearEnd": f"{year}-12-31",
        "audit": {"firm": "Audit Firm 1", "opinion": opinion},
        "statements": statements,
        "ratiosAndMetrics": {},
    }


def make_record(years=None, cases=None, **profile_overrides):
    profile = {
        "companyName": "Test Sdn. Bhd.",
        "status": "Active",
        "companyType": "Private limited company",
        "incorporationDate": "2015-01-01",
        "industry": "Retail",
        "state": "Selangor",
        "natureOfBusiness": "Retail",
        "shareCapital": 100_000,
        "formerNames": [],
    }
    profile.update(profile_overrides)
    return {
        "registrationNumber": "SYN-000001",
        "profile": profile,
        "relationships": {
            "directors": [{"name": "Director 1", "designation": "Managing Director", "appointedOn": "2016-01-01"}],
            "shareholders": [{"type": "individual", "name": "Director 1", "percentage": 100.0}],
        },
        "financials": {"years": years if years is not None else [make_year(2022), make_year(2023), make_year(2024)]},
        "litigation": {"cases": cases or []},
    }


def make_case(role="defendant", status="Pending", amount=100_000.0):
    return {
        "caseNumber": "CASE-2024-0001",
        "role": role,
        "caseType": "Debt recovery",
        "court": "Sessions Court (Selangor)",
        "filedOn": "2024-05-01",
        "status": status,
        "claimAmount": amount,
    }


def codes(record):
    return {f["code"] for f in extract_facts(record)["flags"]}


# ------------------------------------------------------------------- formatting
def test_fmt_rm():
    assert fmt_rm(5_352_856) == "RM5.35m"
    assert fmt_rm(564_100) == "RM564,100"
    assert fmt_rm(-275_562) == "-RM275,562"
    assert fmt_rm(-7_515_861) == "-RM7.52m"
    assert fmt_rm(None) == "n/a"


def test_fmt_pct():
    assert fmt_pct(0.1234) == "12%"
    assert fmt_pct(0.1234, 1) == "12.3%"
    assert fmt_pct(None) == "n/a"


# ------------------------------------------------------------------- clean case
def test_healthy_company_has_no_high_or_medium_flags():
    facts = extract_facts(make_record())
    assert flag_codes(facts) == set()


# ------------------------------------------------------------------- status/age
def test_winding_up_is_high():
    assert "STATUS_NOT_ACTIVE" in codes(make_record(status="Winding up"))


def test_recently_incorporated_is_info_only():
    facts = extract_facts(make_record(incorporationDate="2025-06-01"))
    assert "RECENTLY_INCORPORATED" in {f["code"] for f in facts["flags"]}
    assert "RECENTLY_INCORPORATED" not in flag_codes(facts)


def test_recent_name_change_is_medium_old_is_info():
    recent = make_record(formerNames=[{"name": "Old Sdn. Bhd.", "changedOn": "2025-09-01"}])
    old = make_record(formerNames=[{"name": "Old Sdn. Bhd.", "changedOn": "2019-09-01"}])
    assert "NAME_CHANGED_RECENTLY" in codes(recent)
    assert "FORMER_NAMES" in codes(old) and "NAME_CHANGED_RECENTLY" not in codes(old)


# -------------------------------------------------------------------- financials
def test_no_financials_flagged_and_does_not_crash():
    facts = extract_facts(make_record(years=[]))
    assert facts["financials"]["latest"] is None
    assert "NO_FINANCIALS" in codes(make_record(years=[]))
    assert "Financials: NONE AVAILABLE" in facts_to_text(facts)


def test_limited_history_and_gap_year():
    gap = make_record(years=[make_year(2022), make_year(2024)])
    one = make_record(years=[make_year(2024)])
    assert "LIMITED_HISTORY" in codes(gap)
    assert "LIMITED_HISTORY" in codes(one)


def test_adverse_audit_opinion_in_any_year():
    years = [make_year(2022), make_year(2023, opinion="Disclaimer of opinion"), make_year(2024)]
    found = extract_facts(make_record(years=years))["flags"]
    audit = [f for f in found if f["code"] == "AUDIT_ADVERSE"]
    assert audit and "FY2023" in audit[0]["message"]


def test_persistent_losses_vs_single_loss():
    two = make_record(years=[make_year(2022), make_year(2023, profitAfterTax=-5), make_year(2024, profitAfterTax=-9)])
    one = make_record(years=[make_year(2022), make_year(2023), make_year(2024, profitAfterTax=-9)])
    recovered = make_record(years=[make_year(2022, profitAfterTax=-5), make_year(2023, profitAfterTax=-5), make_year(2024)])
    assert "LOSSES_PERSISTENT" in codes(two)
    assert "LOSS_LATEST_YEAR" in codes(one) and "LOSSES_PERSISTENT" not in codes(one)
    assert not ({"LOSSES_PERSISTENT", "LOSS_LATEST_YEAR"} & codes(recovered))


def test_revenue_decline_threshold():
    down = make_record(years=[make_year(2022, revenue=1_000_000), make_year(2024, revenue=800_000)])
    small = make_record(years=[make_year(2022, revenue=1_000_000), make_year(2024, revenue=950_000)])
    assert "REVENUE_DECLINE" in codes(down)
    assert "REVENUE_DECLINE" not in codes(small)


def test_negative_equity_and_liquidity_and_cashflow():
    rec = make_record(years=[make_year(2024, totalEquity=-1, currentAssets=100, currentLiabilities=200, operatingCashFlow=-5)])
    assert {"NEGATIVE_EQUITY", "LIQUIDITY_TIGHT", "NEGATIVE_OPERATING_CASH_FLOW"} <= codes(rec)


def test_interest_cover_handles_zero_interest():
    rec = make_record(years=[make_year(2024, interestExpense=0, operatingProfit=-50)])
    assert "WEAK_INTEREST_COVER" not in codes(rec)  # cannot be calculated, so not flagged
    weak = make_record(years=[make_year(2024, interestExpense=100_000, operatingProfit=50_000)])
    assert "WEAK_INTEREST_COVER" in codes(weak)


# ---------------------------------------------------------------------- litigation
def test_only_open_defendant_cases_count():
    safe = make_record(
        cases=[
            make_case("defendant", "Settled"),
            make_case("defendant", "Dismissed"),
            make_case("defendant", "Withdrawn"),
            make_case("plaintiff", "Pending", 5e6),
        ]
    )
    assert not ({"LITIGATION_OPEN_DEFENDANT", "LITIGATION_EXPOSURE_HIGH"} & codes(safe))
    lit = extract_facts(safe)["litigation"]
    assert lit["open_defendant_count"] == 0 and lit["as_plaintiff"] == 1


def test_litigation_severity_scales_with_equity():
    low = make_record(cases=[make_case(amount=60_000)])  # 10% of 600k equity
    high = make_record(cases=[make_case(amount=400_000)])  # 67% of 600k equity
    assert "LITIGATION_OPEN_DEFENDANT" in codes(low)
    assert "LITIGATION_EXPOSURE_HIGH" in codes(high)


def test_litigation_with_no_financials_does_not_crash():
    rec = make_record(years=[], cases=[make_case()])
    msg = [f for f in extract_facts(rec)["flags"] if f["code"] == "LITIGATION_OPEN_DEFENDANT"]
    assert msg and "no financials" in msg[0]["message"]


def test_judgment_entered_counts_as_open():
    rec = make_record(cases=[make_case(status="Judgment entered", amount=300_000)])
    assert extract_facts(rec)["litigation"]["judgments_against"] == 1


# ---------------------------------------------------------------------- ownership
def test_concentration_and_corporate_holder_are_info():
    rec = make_record()
    rec["relationships"]["shareholders"] = [
        {"type": "company", "name": "Holdco Sdn. Bhd.", "percentage": 80.0},
        {"type": "individual", "name": "Shareholder 1", "percentage": 20.0},
    ]
    facts = extract_facts(rec)
    found = {f["code"] for f in facts["flags"]}
    assert {"OWNERSHIP_CONCENTRATED", "CORPORATE_SHAREHOLDER"} <= found
    assert not ({"OWNERSHIP_CONCENTRATED", "CORPORATE_SHAREHOLDER"} & flag_codes(facts))


# ---------------------------------------------------------------------- invariants
def test_flags_sorted_high_first():
    rec = make_record(status="Winding up", years=[make_year(2024, profitAfterTax=-1)])
    order = [f["severity"] for f in extract_facts(rec)["flags"]]
    assert order == sorted(order, key=["high", "medium", "info"].index)


def test_extract_facts_does_not_mutate_input():
    rec = make_record(cases=[make_case()])
    before = copy.deepcopy(rec)
    extract_facts(rec)
    assert rec == before


# ------------------------------------------------------------- real-data smoke tests
@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("data/companies.jsonl not found")
    return {r["registrationNumber"]: r for r in load_companies(DATA)}


def test_every_real_company_extracts_without_error(real):
    assert len(real) == 50
    for rec in real.values():
        facts = extract_facts(rec)
        assert facts_to_text(facts)


def test_known_real_cases(real):
    assert {"STATUS_NOT_ACTIVE", "LITIGATION_EXPOSURE_HIGH"} <= codes(real["SYN-001212"])
    assert "NO_FINANCIALS" in codes(real["SYN-000302"])
    assert "AUDIT_ADVERSE" in codes(real["SYN-000217"])
    assert "NEGATIVE_EQUITY" in codes(real["SYN-000451"])
    assert flag_codes(extract_facts(real["SYN-000029"])) == set()  # a clean company
