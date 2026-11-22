"""Profile the 50 companies to help choose the risk thresholds in facts.py.

For each threshold, this prints how many companies would be flagged at several
candidate cutoffs.

"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "companies.jsonl"
AS_OF = date(2025, 12, 31)
OPEN = {"Pending", "Judgment entered"}


def load() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def months(start: str) -> int:
    d = date.fromisoformat(start)
    return (AS_OF.year - d.year) * 12 + (AS_OF.month - d.month)


def years_of(rec: dict) -> list[dict]:
    return sorted(rec["financials"]["years"], key=lambda y: y["financialYear"])


def table(title: str, values: dict[str, float], cutoffs: list[float], op: str, fmt: str) -> None:
    """Print how many companies fall past each cutoff (op is '<=' or '>=')."""
    print(f"\n{title}")
    print(f"  companies with a value: {len(values)}")
    for c in cutoffs:
        hit = [k for k, v in values.items() if (v <= c if op == "<=" else v >= c)]
        print(f"  {op} {format(c, fmt):>7}: {len(hit):>2} flagged")


def main() -> None:
    rows = load()
    print(f"Loaded {len(rows)} companies from {DATA.name}")

    # 1. Revenue change, first year to latest year.
    rev = {}
    for r in rows:
        ys = years_of(r)
        if len(ys) >= 2 and ys[0]["statements"]["revenue"] > 0:
            rev[r["registrationNumber"]] = ys[-1]["statements"]["revenue"] / ys[0]["statements"]["revenue"] - 1
    table("REVENUE CHANGE first -> latest year (current rule: <= -10%)", rev, [-0.05, -0.10, -0.15, -0.20, -0.30], "<=", ".0%")

    # 2. Open claims against the company as a share of latest equity.
    lit = {}
    for r in rows:
        ys = years_of(r)
        claims = sum(c["claimAmount"] for c in r["litigation"]["cases"] if c["role"] == "defendant" and c["status"] in OPEN)
        if claims and ys and ys[-1]["statements"]["totalEquity"] > 0:
            lit[r["registrationNumber"]] = claims / ys[-1]["statements"]["totalEquity"]
    table("OPEN CLAIMS / EQUITY (current rule: >= 50% is 'high')", lit, [0.10, 0.25, 0.50, 0.75, 1.00], ">=", ".0%")
    print("  the individual ratios:", ", ".join(f"{v:.0%}" for v in sorted(lit.values())))

    # 3. Largest single shareholder.
    top = {r["registrationNumber"]: max(s["percentage"] for s in r["relationships"]["shareholders"]) for r in rows}
    table("LARGEST SHAREHOLDER % (current rule: >= 75%)", top, [50, 60, 75, 90, 100], ">=", ".0f")

    # 4. Interest cover = operating profit / interest expense, latest year.
    cover = {}
    for r in rows:
        ys = years_of(r)
        if ys and ys[-1]["statements"]["interestExpense"] > 0:
            s = ys[-1]["statements"]
            cover[r["registrationNumber"]] = s["operatingProfit"] / s["interestExpense"]
    table("INTEREST COVER, latest year (current rule: < 1.0, i.e. <= 1.0 here)", cover, [0.0, 1.0, 1.5, 2.0, 3.0], "<=", ".1f")

    # 5. Company age in months at 31 Dec 2025.
    age = {r["registrationNumber"]: months(r["profile"]["incorporationDate"]) for r in rows}
    table("COMPANY AGE in months (current rule: < 24 is 'recent', info only)", age, [12, 24, 36], "<=", ".0f")

    # 6. Months since the most recent name change.
    since = {
        r["registrationNumber"]: min(months(f["changedOn"]) for f in r["profile"]["formerNames"])
        for r in rows
        if r["profile"]["formerNames"]
    }
    table("MONTHS SINCE NAME CHANGE (current rule: < 12 is 'recent')", since, [6, 12, 24, 36], "<=", ".0f")
    print("  the individual values:", sorted(since.values()))

    # 7. Consecutive loss-making years ending at the latest year.
    streaks = {}
    for r in rows:
        n = 0
        for y in reversed(years_of(r)):
            if y["statements"]["profitAfterTax"] < 0:
                n += 1
            else:
                break
        streaks[r["registrationNumber"]] = n
    table("CONSECUTIVE LOSS YEARS (current rule: 2+ is 'high', 1 is 'medium')", streaks, [1, 2, 3], ">=", ".0f")


if __name__ == "__main__":
    main()
