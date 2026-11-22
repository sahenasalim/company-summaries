"""Deterministic fact extraction and risk flagging for a company record.

Why this module exists
----------------------
LLMs are unreliable at arithmetic and at spotting rare signals buried in nested
JSON. So all numbers, trends and risk flags are computed here in plain Python.
The LLM is later asked only to write from these facts, never to calculate.

"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

# The records are "as of" this date.
AS_OF = date(2025, 12, 31)

# Cases a defendant has not yet put behind it. Settled/withdrawn/dismissed
# cases are deliberately NOT treated as risks.
OPEN_STATUSES = {"Pending", "Judgment entered"}

# --- Thresholds (chosen after profiling the 50 companies) ---
REVENUE_DECLINE_THRESHOLD = -0.10  # first-to-latest change
LITIGATION_EXPOSURE_HIGH = 0.50  # open defendant claims / equity
CONCENTRATION_THRESHOLD = 75.0  # single shareholder, percent
YOUNG_COMPANY_MONTHS = 36
RECENT_NAME_CHANGE_MONTHS = 12
INTEREST_COVER_FLOOR = 1.0  # operating profit / interest expense

SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


@dataclass(frozen=True)
class Flag:
    """One risk or context signal, with a ready-to-use plain-English message."""

    code: str
    severity: str  # "high" | "medium" | "info"
    message: str


# ----------------------------------------------------------------- formatting
def fmt_rm(amount: float | None) -> str:
    """Format a ringgit amount compactly and consistently (RM1.95m / RM564,100)."""
    if amount is None:
        return "n/a"
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    if a >= 1_000_000:
        return f"{sign}RM{a / 1_000_000:.2f}m"
    return f"{sign}RM{a:,.0f}"


def fmt_pct(value: float | None, digits: int = 0) -> str:
    """Format a fraction (0.12) as a percentage string ('12%')."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


# -------------------------------------------------------------------- loading
def load_companies(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of company records."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------ section helpers
def _profile_facts(record: dict[str, Any]) -> dict[str, Any]:
    p = record["profile"]
    inc = date.fromisoformat(p["incorporationDate"])
    return {
        "registration_number": record["registrationNumber"],
        "name": p["companyName"],
        "status": p["status"],
        "industry": p["industry"],
        "state": p["state"],
        "nature_of_business": p["natureOfBusiness"],
        "incorporated": p["incorporationDate"],
        "age_months": _months_between(inc, AS_OF),
        "share_capital": p["shareCapital"],
        # Each entry: {"name": ..., "changedOn": "YYYY-MM-DD"}
        "former_names": list(p.get("formerNames") or []),
    }


def _ownership_facts(record: dict[str, Any]) -> dict[str, Any]:
    rel = record["relationships"]
    holders = rel["shareholders"]
    directors = rel["directors"]
    director_names = {d["name"] for d in directors}
    top = max(holders, key=lambda s: s["percentage"]) if holders else None
    recent = [d for d in directors if _months_between(date.fromisoformat(d["appointedOn"]), AS_OF) < 12]
    return {
        "director_count": len(directors),
        "designations": sorted({d["designation"] for d in directors}),
        "recent_appointments_12m": len(recent),
        "shareholder_count": len(holders),
        "top_holder_name": top["name"] if top else None,
        "top_holder_pct": top["percentage"] if top else None,
        "top_holder_is_director": bool(top and top["name"] in director_names),
        "corporate_holders": [{"name": s["name"], "pct": s["percentage"]} for s in holders if s["type"] == "company"],
    }


def _year_summary(y: dict[str, Any]) -> dict[str, Any]:
    s = y["statements"]
    interest = s["interestExpense"]
    return {
        "year": y["financialYear"],
        "year_end": y["financialYearEnd"],
        "opinion": y["audit"]["opinion"],
        "revenue": s["revenue"],
        "profit_after_tax": s["profitAfterTax"],
        "operating_profit": s["operatingProfit"],
        "operating_cash_flow": s["operatingCashFlow"],
        "total_assets": s["totalAssets"],
        "total_liabilities": s["totalLiabilities"],
        "total_equity": s["totalEquity"],
        "cash": s["cash"],
        "borrowings": s["borrowings"],
        "current_assets": s["currentAssets"],
        "current_liabilities": s["currentLiabilities"],
        "net_margin": (s["profitAfterTax"] / s["revenue"]) if s["revenue"] else None,
        "interest_cover": (s["operatingProfit"] / interest) if interest and interest > 0 else None,
    }


def _financial_facts(record: dict[str, Any]) -> dict[str, Any]:
    years = sorted(record["financials"]["years"], key=lambda y: y["financialYear"])
    summaries = [_year_summary(y) for y in years]
    fy = [s["year"] for s in summaries]
    has_gap = any(b - a != 1 for a, b in pairwise(fy))
    return {
        "years": summaries,
        "years_available": fy,
        "has_gap": has_gap,
        "latest": summaries[-1] if summaries else None,
        "first": summaries[0] if summaries else None,
    }


def _litigation_facts(record: dict[str, Any]) -> dict[str, Any]:
    cases = record["litigation"]["cases"]
    defendant = [c for c in cases if c["role"] == "defendant"]
    plaintiff = [c for c in cases if c["role"] == "plaintiff"]
    open_def = [c for c in defendant if c["status"] in OPEN_STATUSES]
    return {
        "total": len(cases),
        "as_defendant": len(defendant),
        "as_plaintiff": len(plaintiff),
        "open_defendant_count": len(open_def),
        "open_defendant_amount": sum(c["claimAmount"] or 0 for c in open_def),
        "open_defendant_types": sorted({c["caseType"] for c in open_def}),
        "judgments_against": sum(1 for c in open_def if c["status"] == "Judgment entered"),
        "closed_defendant_count": len(defendant) - len(open_def),
        "plaintiff_amount": sum(c["claimAmount"] or 0 for c in plaintiff),
    }


# ----------------------------------------------------------------------- flags
def _status_flags(profile: dict[str, Any]) -> list[Flag]:
    flags: list[Flag] = []
    if profile["status"] != "Active":
        flags.append(Flag("STATUS_NOT_ACTIVE", "high", f"Company status is '{profile['status']}', not Active."))
    if profile["age_months"] < YOUNG_COMPANY_MONTHS:
        flags.append(
            Flag(
                "RECENTLY_INCORPORATED",
                "info",
                f"Recently incorporated ({profile['incorporated']}); about {profile['age_months']} months old at 31 Dec 2025.",
            )
        )
    for fn in profile["former_names"]:
        changed = date.fromisoformat(fn["changedOn"])
        recent = _months_between(changed, AS_OF) < RECENT_NAME_CHANGE_MONTHS
        text = f"Formerly '{fn['name']}', renamed on {fn['changedOn']}."
        if recent:
            flags.append(
                Flag("NAME_CHANGED_RECENTLY", "medium", text + f" The change is under {RECENT_NAME_CHANGE_MONTHS} months old.")
            )
        else:
            flags.append(Flag("FORMER_NAMES", "info", text))
    return flags


def _ownership_flags(own: dict[str, Any]) -> list[Flag]:
    flags: list[Flag] = []
    if own["top_holder_pct"] is not None and own["top_holder_pct"] >= CONCENTRATION_THRESHOLD:
        flags.append(
            Flag("OWNERSHIP_CONCENTRATED", "info", f"One shareholder holds {own['top_holder_pct']:.1f}% of the company.")
        )
    if own["corporate_holders"]:
        names = ", ".join(f"{h['name']} ({h['pct']:.1f}%)" for h in own["corporate_holders"])
        flags.append(Flag("CORPORATE_SHAREHOLDER", "info", f"Part-owned by other companies: {names}."))
    return flags


def _audit_flags(fin: dict[str, Any]) -> list[Flag]:
    bad = [(y["year"], y["opinion"]) for y in fin["years"] if y["opinion"] != "Unqualified"]
    if not bad:
        return []
    detail = "; ".join(f"FY{yr}: {op}" for yr, op in bad)
    return [Flag("AUDIT_ADVERSE", "high", f"Non-clean audit opinion ({detail}).")]


def _earnings_flags(fin: dict[str, Any]) -> list[Flag]:
    years, latest, first = fin["years"], fin["latest"], fin["first"]
    flags: list[Flag] = []

    streak = 0
    for y in reversed(years):
        if y["profit_after_tax"] < 0:
            streak += 1
        else:
            break
    if streak >= 2:
        flags.append(
            Flag(
                "LOSSES_PERSISTENT",
                "high",
                f"Loss-making in each of the last {streak} years; latest loss "
                f"{fmt_rm(latest['profit_after_tax'])} (FY{latest['year']}).",
            )
        )
    elif streak == 1:
        flags.append(
            Flag("LOSS_LATEST_YEAR", "medium", f"Made a loss of {fmt_rm(latest['profit_after_tax'])} in FY{latest['year']}.")
        )

    if len(years) >= 2 and first["revenue"] and first["revenue"] > 0:
        change = latest["revenue"] / first["revenue"] - 1
        if change <= REVENUE_DECLINE_THRESHOLD:
            flags.append(
                Flag(
                    "REVENUE_DECLINE",
                    "medium",
                    f"Revenue fell {abs(change) * 100:.0f}% from {fmt_rm(first['revenue'])} "
                    f"(FY{first['year']}) to {fmt_rm(latest['revenue'])} (FY{latest['year']}).",
                )
            )

    if latest["operating_cash_flow"] < 0:
        flags.append(
            Flag(
                "NEGATIVE_OPERATING_CASH_FLOW",
                "medium",
                f"Operating cash flow was negative ({fmt_rm(latest['operating_cash_flow'])}) in FY{latest['year']}.",
            )
        )

    cover = latest["interest_cover"]
    if cover is not None and cover < INTEREST_COVER_FLOOR:
        flags.append(
            Flag(
                "WEAK_INTEREST_COVER",
                "medium",
                f"Operating profit does not cover interest expense in FY{latest['year']} (cover {cover:.1f}x).",
            )
        )
    return flags


def _balance_sheet_flags(latest: dict[str, Any]) -> list[Flag]:
    flags: list[Flag] = []
    if latest["total_equity"] < 0:
        flags.append(
            Flag(
                "NEGATIVE_EQUITY",
                "high",
                f"Liabilities exceed assets: total equity is {fmt_rm(latest['total_equity'])} (FY{latest['year']}).",
            )
        )
    if latest["current_assets"] < latest["current_liabilities"]:
        flags.append(
            Flag(
                "LIQUIDITY_TIGHT",
                "medium",
                f"Current liabilities ({fmt_rm(latest['current_liabilities'])}) exceed "
                f"current assets ({fmt_rm(latest['current_assets'])}) in FY{latest['year']}.",
            )
        )
    return flags


def _financial_flags(fin: dict[str, Any]) -> list[Flag]:
    if not fin["years"]:
        return [Flag("NO_FINANCIALS", "medium", "No financial statements are available.")]
    flags: list[Flag] = []
    if len(fin["years"]) < 3 or fin["has_gap"]:
        yrs = ", ".join(f"FY{y}" for y in fin["years_available"])
        note = " (with a missing year in between)" if fin["has_gap"] else ""
        flags.append(Flag("LIMITED_HISTORY", "info", f"Only {len(fin['years'])} year(s) of financials: {yrs}{note}."))
    flags += _audit_flags(fin)
    flags += _earnings_flags(fin)
    flags += _balance_sheet_flags(fin["latest"])
    return flags


def _litigation_flags(lit: dict[str, Any], fin: dict[str, Any]) -> list[Flag]:
    if not lit["open_defendant_count"]:
        return []
    amount = lit["open_defendant_amount"]
    base = (
        f"{lit['open_defendant_count']} open case(s) against the company "
        f"(pending or judgment entered), claims totalling {fmt_rm(amount)}"
    )
    if lit["judgments_against"]:
        base += f", of which {lit['judgments_against']} already at judgment"
    latest = fin["latest"]
    if latest is None:
        return [
            Flag("LITIGATION_OPEN_DEFENDANT", "medium", base + "; cannot be sized against equity because no financials exist.")
        ]
    equity = latest["total_equity"]
    if equity <= 0:
        return [Flag("LITIGATION_EXPOSURE_HIGH", "high", base + f", while equity is {fmt_rm(equity)}.")]
    ratio = amount / equity
    if ratio >= LITIGATION_EXPOSURE_HIGH:
        return [Flag("LITIGATION_EXPOSURE_HIGH", "high", base + f", equal to {ratio * 100:.0f}% of equity ({fmt_rm(equity)}).")]
    return [Flag("LITIGATION_OPEN_DEFENDANT", "medium", base + f", equal to {ratio * 100:.0f}% of equity ({fmt_rm(equity)}).")]


def _sorted(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: SEVERITY_ORDER[f.severity])


# ------------------------------------------------------------------ public API
def extract_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Turn one raw company record into derived facts plus risk flags."""
    profile = _profile_facts(record)
    ownership = _ownership_facts(record)
    financials = _financial_facts(record)
    litigation = _litigation_facts(record)

    flags = (
        _status_flags(profile)
        + _ownership_flags(ownership)
        + _financial_flags(financials)
        + _litigation_flags(litigation, financials)
    )
    return {
        "profile": profile,
        "ownership": ownership,
        "financials": financials,
        "litigation": litigation,
        "flags": [asdict(f) for f in _sorted(flags)],
    }


def flag_codes(facts: dict[str, Any], severities: tuple[str, ...] = ("high", "medium")) -> set[str]:
    """Codes of flags at the given severities (used as ground truth in evaluation)."""
    return {f["code"] for f in facts["flags"] if f["severity"] in severities}


def facts_to_text(facts: dict[str, Any]) -> str:
    """Render facts as a compact plain-text block to place in an LLM prompt."""
    p, o, fin, lit = facts["profile"], facts["ownership"], facts["financials"], facts["litigation"]
    lines = [
        f"Company: {p['name']} ({p['registration_number']})",
        f"Industry: {p['industry']} | State: {p['state']} | Status: {p['status']}",
        f"Business: {p['nature_of_business']}",
        f"Incorporated: {p['incorporated']} | Share capital: {fmt_rm(p['share_capital'])}",
        f"Board: {o['director_count']} director(s); {o['recent_appointments_12m']} appointed in the 12 months to 31 Dec 2025",
        f"Largest shareholder: {o['top_holder_pct']:.1f}%" + (" (also a director)" if o["top_holder_is_director"] else ""),
    ]
    latest = fin["latest"]
    if latest is None:
        lines.append("Financials: NONE AVAILABLE")
    else:
        lines.append(
            f"Financial years available: {', '.join(f'FY{y}' for y in fin['years_available'])}"
            f" (latest year end {latest['year_end']})"
        )
        lines.extend(
            f"FY{y['year']}: revenue {fmt_rm(y['revenue'])}, profit after tax "
            f"{fmt_rm(y['profit_after_tax'])}, equity {fmt_rm(y['total_equity'])}, "
            f"operating cash flow {fmt_rm(y['operating_cash_flow'])}, audit: {y['opinion']}"
            for y in fin["years"]
        )
    lines.append(
        f"Court cases: {lit['total']} total ({lit['as_defendant']} as defendant, "
        f"{lit['as_plaintiff']} as plaintiff); open cases against company: {lit['open_defendant_count']}"
    )
    lines.append("Risk and context signals (computed, already verified):")
    if facts["flags"]:
        lines += [f"- [{f['severity'].upper()}] {f['message']}" for f in facts["flags"]]
    else:
        lines.append("- none")
    return "\n".join(lines)
