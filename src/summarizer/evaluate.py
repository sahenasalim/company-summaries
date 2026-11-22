"""Score summary versions against facts computed from the source data.

All checks are deterministic and cheap, so they can run on every version. They
are proxies: keyword and number matching can miss nuance, which is why a blind
manual review (review.py) sits alongside them.

Checks
------
length          Word count, ignoring markdown symbols.
faithfulness    Every figure in a summary must trace back to the record or to
                a value derived from it (rounding allowed).
risk recall     Of the HIGH/MEDIUM signals computed in facts.py, how many does
                the summary actually mention?
recommendation  Summaries should not tell the reader what to do.
litigation      Closed cases and cases the company brought must not be
                presented as a threat.
staleness       Does the summary say which financial year the figures are from?

"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from summarizer.facts import extract_facts, flag_codes, load_companies

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "companies.jsonl"
SUMMARIES_DIR = ROOT / "summaries"
REPORTS_DIR = ROOT / "reports"

WORD_LIMIT = 120
SMALL_INT_LIMIT = 100

# --------------------------------------------------------------------- length
_WORD = re.compile(r"[A-Za-z0-9][\w'\u2019.,%/-]*")


def word_count(text: str) -> int:
    """Words in the text, ignoring markdown symbols like '*' and '###'."""
    return len(_WORD.findall(text))


# ------------------------------------------------------------- number parsing
_IDENTIFIERS = re.compile(r"SYN-\d+|CASE-\d{4}-\d+|(?:Audit Firm|Director|Shareholder|Firm|Auditor)\s+\d+", re.I)
_YEAR = re.compile(r"\b(?:FY)?((?:19|20)\d{2})\b")
_NUMBER = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<unit>%|bn\b|billion\b|million\b|m\b|k\b)?",
    re.I,
)
_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


@dataclass(frozen=True)
class Figure:
    """A number as written in text: its value, how precisely, and whether it is a percentage."""

    text: str
    value: float
    tolerance: float
    is_percent: bool


def parse_figures(text: str) -> tuple[list[Figure], list[int]]:
    """Pull money amounts, percentages, ratios and years out of text.

    Signs are ignored (a loss of 7.5m matches -7.5m). Identifiers like SYN-000067 and
    'Director 4' are removed first. Bare whole numbers under 100 are skipped.
    """
    cleaned = _IDENTIFIERS.sub(" ", text)
    years = [int(y) for y in _YEAR.findall(cleaned)]
    cleaned = _YEAR.sub(" ", cleaned)

    figures: list[Figure] = []
    for m in _NUMBER.finditer(cleaned):
        raw, unit = m.group("num"), (m.group("unit") or "").lower()
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        value = float(digits)
        if unit == "%":
            figures.append(Figure(m.group(0).strip(), value, 10**-decimals, True))
            continue
        if not unit and decimals == 0 and value < SMALL_INT_LIMIT:
            continue
        scale = _SCALE[unit]
        figures.append(Figure(m.group(0).strip(), value * scale, 10**-decimals * scale, False))
    return figures, years


# -------------------------------------------------------------- allowed values
_PAIR_METRICS = ("revenue", "profit_after_tax", "operating_cash_flow", "total_equity", "operating_profit")


def _collect(node: Any, out: set[float]) -> None:
    """Add every number found anywhere in a nested structure (including inside strings)."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.add(abs(float(node)))
    elif isinstance(node, str):
        out.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", node))
        figures, years = parse_figures(node)
        out.update(f.value for f in figures)
        out.update(float(y) for y in years)
    elif isinstance(node, dict):
        for value in node.values():
            _collect(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect(value, out)


def allowed_values(record: dict[str, Any], facts: dict[str, Any]) -> set[float]:
    """Every number a faithful summary could legitimately quote for this company."""
    out: set[float] = set()
    _collect(record, out)
    _collect(facts, out)

    years = facts["financials"]["years"]
    for metric in _PAIR_METRICS:
        for a, b in itertools.combinations(years, 2):  # any two years, in either direction
            va, vb = a[metric], b[metric]
            out.add(abs(vb - va))
            if va:
                out.add(abs(vb / va - 1) * 100)
    lit = facts["litigation"]
    out.update([lit["open_defendant_amount"], lit["plaintiff_amount"], float(lit["total"])])
    out.add(facts["profile"]["age_months"] / 12)
    return out


def unsupported_figures(text: str, allowed: set[float]) -> list[str]:
    """Figures in the text that match nothing in the allowed set (within rounding)."""
    figures, years = parse_figures(text)
    bad: list[str] = []
    for f in figures:
        candidates = [a * 100 for a in allowed] + list(allowed) if f.is_percent else allowed
        if not any(abs(f.value - a) <= f.tolerance + 1e-9 for a in candidates):
            bad.append(f.text)
    bad.extend(str(y) for y in years if float(y) not in allowed)
    return bad


# ------------------------------------------------------------- keyword checks
_RECOMMENDATION = re.compile(
    r"recommend|suitab|advis|proceed with|exercise caution|due diligence|should (?:avoid|engage|proceed)"
    r"|avoid doing business|safe to|worth partnering|not worth",
    re.I,
)
_LITIGATION_WORDS = re.compile(r"litigation|lawsuit|court|legal (?:action|proceeding|dispute|case)|sued|\bcases?\b", re.I)
_LITIGATION_NEGATED = re.compile(
    r"no (?:recorded |reported |known |open |active |significant |material )?(?:litigation|lawsuits?|court cases?|legal|cases?)"
    r"|litigation:?\s*(?:none|nil)|none reported|free of litigation|not involved in any|no pending",
    re.I,
)

# Per flag code: wording that shows a summary has mentioned the signal.
MENTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "AUDIT_ADVERSE": re.compile(r"disclaimer|going concern|(?<!un)qualified|adverse|material uncertainty", re.I),
    "NEGATIVE_EQUITY": re.compile(
        r"negative equity|negative net assets|liabilities exceed|deficit|net liabilit|insolven|equity (?:of|is) -", re.I
    ),
    "LOSSES_PERSISTENT": re.compile(r"\bloss|unprofitable|net loss", re.I),
    "LOSS_LATEST_YEAR": re.compile(r"\bloss|unprofitable|net loss", re.I),
    "REVENUE_DECLINE": re.compile(
        r"revenue[^.]{0,40}(?:fell|declin|decreas|drop|down|contract|shr[ua]nk|lower|reduc)"
        r"|(?:declin|fall|fell|drop|decreas|shr[ua]nk|downward)\w*[^.]{0,40}revenue",
        re.I,
    ),
    "NEGATIVE_OPERATING_CASH_FLOW": re.compile(
        r"negative (?:operating )?cash|cash (?:outflow|burn)|operating cash[^.]{0,50}(?:negative|outflow|deficit|burn)"
        r"|(?:negative|outflow)[^.]{0,40}operating cash|cash flow[^.]{0,30}negative",
        re.I,
    ),
    "WEAK_INTEREST_COVER": re.compile(
        r"interest cover|interest coverage|cover(?:s|ing)? (?:its |the )?interest|debt[- ]service|servic\w+ (?:its |the )?debt"
        r"|interest expense|interest burden",
        re.I,
    ),
    "LIQUIDITY_TIGHT": re.compile(
        r"liquidity|current (?:liabilities|ratio|assets)|working capital|quick ratio|short-term", re.I
    ),
    "NAME_CHANGED_RECENTLY": re.compile(
        r"renam|formerly|name change|changed (?:its )?name|previously (?:known|named|traded)", re.I
    ),
    "NO_FINANCIALS": re.compile(
        r"no financial|financials?[^.]{0,25}(?:not available|unavailable|missing|absent|not provided)"
        r"|(?:not available|unavailable|missing)[^.]{0,25}financial|no (?:audited )?accounts|no (?:filed )?statements"
        r"|financial statements[^.]{0,25}(?:not|un)",
        re.I,
    ),
    "LITIGATION_EXPOSURE_HIGH": _LITIGATION_WORDS,
    "LITIGATION_OPEN_DEFENDANT": _LITIGATION_WORDS,
}
_STATUS_WORDS = re.compile(r"winding|wound up|liquidat|struck off|dissolved|not active|inactive|dormant|deregist", re.I)


def mentions(code: str, text: str, facts: dict[str, Any]) -> bool:
    """Does the text mention the signal behind this flag code?"""
    if code == "STATUS_NOT_ACTIVE":
        return bool(_STATUS_WORDS.search(text) or facts["profile"]["status"].lower() in text.lower())
    pattern = MENTION_PATTERNS.get(code)
    return bool(pattern and pattern.search(text))


# ----------------------------------------------------------------- scoring
@dataclass
class SummaryScore:
    version: str
    registration_number: str
    words: int
    unsupported: list[str]
    required: set[str]
    missed: set[str]
    gives_recommendation: bool
    litigation_alarm: bool
    litigation_eligible: bool
    has_financials: bool
    states_latest_year: bool
    states_no_financials: bool = False
    required_high: set[str] = field(default_factory=set)
    missed_high: set[str] = field(default_factory=set)


def score_summary(version: str, record: dict[str, Any], text: str) -> SummaryScore:
    facts = extract_facts(record)
    required = flag_codes(facts)
    required_high = flag_codes(facts, ("high",))
    missed = {c for c in required if not mentions(c, text, facts)}
    latest = facts["financials"]["latest"]
    lit = facts["litigation"]
    eligible = lit["total"] > 0 and lit["open_defendant_count"] == 0
    alarm = eligible and bool(_LITIGATION_WORDS.search(text)) and not _LITIGATION_NEGATED.search(text)
    return SummaryScore(
        version=version,
        registration_number=record["registrationNumber"],
        words=word_count(text),
        unsupported=unsupported_figures(text, allowed_values(record, facts)),
        required=required,
        missed=missed,
        gives_recommendation=bool(_RECOMMENDATION.search(text)),
        litigation_alarm=alarm,
        litigation_eligible=eligible,
        has_financials=latest is not None,
        states_latest_year=latest is not None and str(latest["year"]) in text,
        states_no_financials="NO_FINANCIALS" in required and "NO_FINANCIALS" not in missed,
        required_high=required_high,
        missed_high=missed & required_high,
    )


def load_summaries(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["registrationNumber"]: r["summary"] for r in rows}


# ---------------------------------------------------------------- aggregation
def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def summarise(scores: list[SummaryScore]) -> dict[str, Any]:
    n = len(scores)
    req = sum(len(s.required) for s in scores)
    req_high = sum(len(s.required_high) for s in scores)
    per_code: dict[str, list[int]] = {}
    for s in scores:
        for code in s.required:
            hit = per_code.setdefault(code, [0, 0])
            hit[1] += 1
            hit[0] += code not in s.missed
    with_fin = [s for s in scores if s.has_financials]
    no_fin = [s for s in scores if "NO_FINANCIALS" in s.required]
    eligible = [s for s in scores if s.litigation_eligible]
    return {
        "summaries": n,
        "mean_words": sum(s.words for s in scores) / n if n else 0,
        "over_limit": sum(s.words > WORD_LIMIT for s in scores),
        "with_unsupported_figures": sum(bool(s.unsupported) for s in scores),
        "unsupported_total": sum(len(s.unsupported) for s in scores),
        "recall_all": (req - sum(len(s.missed) for s in scores), req),
        "recall_high": (req_high - sum(len(s.missed_high) for s in scores), req_high),
        "per_code": {c: tuple(v) for c, v in sorted(per_code.items())},
        "recommendations": sum(s.gives_recommendation for s in scores),
        "litigation_alarms": (sum(s.litigation_alarm for s in eligible), len(eligible)),
        "states_latest_year": (sum(s.states_latest_year for s in with_fin), len(with_fin)),
        "no_financials_honest": (sum(s.states_no_financials for s in no_fin), len(no_fin)),
    }


def render_report(results: dict[str, dict[str, Any]]) -> str:
    versions = list(results)
    rows: list[tuple[str, list[str]]] = [
        ("Summaries scored", [str(results[v]["summaries"]) for v in versions]),
        ("Mean words", [f"{results[v]['mean_words']:.0f}" for v in versions]),
        (f"Over {WORD_LIMIT} words", [str(results[v]["over_limit"]) for v in versions]),
        ("Summaries with an unsupported figure", [str(results[v]["with_unsupported_figures"]) for v in versions]),
        ("Unsupported figures (total)", [str(results[v]["unsupported_total"]) for v in versions]),
        ("Recall: high-severity signals", [_pct(*results[v]["recall_high"]) for v in versions]),
        ("Recall: all high+medium signals", [_pct(*results[v]["recall_all"]) for v in versions]),
        ("Unrequested recommendations", [str(results[v]["recommendations"]) for v in versions]),
        ("Litigation mentioned, no open case", [_pct(*results[v]["litigation_alarms"]) for v in versions]),
        ("States latest financial year", [_pct(*results[v]["states_latest_year"]) for v in versions]),
        ("Says plainly 'no financials'", [_pct(*results[v]["no_financials_honest"]) for v in versions]),
    ]
    lines = ["| Metric | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
    lines += [f"| {name} | " + " | ".join(vals) + " |" for name, vals in rows]

    codes = sorted({c for v in versions for c in results[v]["per_code"]})
    lines += [
        "",
        "Recall by signal (mentioned / present):",
        "",
        "| Signal | " + " | ".join(versions) + " |",
        "|---|" + "---|" * len(versions),
    ]
    for code in codes:
        cells = [_pct(*results[v]["per_code"][code]) if code in results[v]["per_code"] else "-" for v in versions]
        lines.append(f"| {code} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_details(path: Path, all_scores: list[SummaryScore]) -> None:
    """One row per summary, for reading the failures behind the averages."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["version", "company", "words", "unsupported_figures", "missed_signals", "recommendation", "litigation_alarm"]
        )
        for s in all_scores:
            w.writerow(
                [s.version, s.registration_number, s.words, "; ".join(s.unsupported), "; ".join(sorted(s.missed)),
                 s.gives_recommendation, s.litigation_alarm]
            )  # fmt: skip


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score summary versions.")
    parser.add_argument("--versions", nargs="*", default=None, help="default: every summaries/*.jsonl")
    args = parser.parse_args(argv)

    versions = args.versions or sorted(p.stem for p in SUMMARIES_DIR.glob("*.jsonl"))
    if not versions:
        print("No summaries found. Run summarizer.generate first.")
        return 1
    records = {r["registrationNumber"]: r for r in load_companies(DATA_PATH)}

    results: dict[str, dict[str, Any]] = {}
    all_scores: list[SummaryScore] = []
    for version in versions:
        summaries = load_summaries(SUMMARIES_DIR / f"{version}.jsonl")
        missing = sorted(set(records) - set(summaries))
        if missing:
            print(f"WARNING {version}: no summary for {len(missing)} companies: {', '.join(missing)}")
        scores = [score_summary(version, records[reg], text) for reg, text in summaries.items() if reg in records]
        results[version] = summarise(scores)
        all_scores += scores

    report = render_report(results)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "evaluation.md").write_text(report + "\n", encoding="utf-8")
    write_details(REPORTS_DIR / "per_company.csv", all_scores)
    print(report)
    print(f"\nSaved reports/evaluation.md and reports/per_company.csv ({len(all_scores)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
