"""Blind, side-by-side manual review of two summary versions."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from summarizer.evaluate import load_summaries
from summarizer.facts import extract_facts, facts_to_text, flag_codes, load_companies

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "companies.jsonl"
SUMMARIES_DIR = ROOT / "summaries"
REVIEW_DIR = ROOT / "review"

DEFAULT_VERSIONS = ("v1_baseline", "v2_structured")
PREFERRED_LINE = re.compile(r"\*\*Preferred \(A / B / tie\):\*\*[ \t]*([A-Za-z]*)")
SECTION_HEAD = re.compile(r"^(\d+)\. (SYN-\d+)")

INSTRUCTIONS = """# Blind review

For each company: read the source facts, then both summaries. Check the numbers against the facts,
and ask: which would help a vetting analyst decide, in 30 seconds, whether to look closer?

Fill in the line `**Preferred (A / B / tie):**` with A, B or tie, and add short notes on anything wrong
(an invented number, a missed risk, a risk presented as bigger or smaller than it is).
Do not open `key.json` until you have finished.
"""


def pick_companies(
    records: list[dict[str, Any]], seed: int = 42, n_high: int = 4, n_medium: int = 3, n_clean: int = 3
) -> list[str]:
    """Choose a mix of high-risk, medium-risk and clean companies, always including the hard cases."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    high, medium, clean = [], [], []
    no_financials, closed_cases_only = [], []
    for rec in sorted(records, key=lambda r: r["registrationNumber"]):
        facts = extract_facts(rec)
        reg = rec["registrationNumber"]
        if flag_codes(facts, ("high",)):
            high.append(reg)
        elif flag_codes(facts, ("medium",)):
            medium.append(reg)
        else:
            clean.append(reg)
        if "NO_FINANCIALS" in flag_codes(facts):
            no_financials.append(reg)
        lit = facts["litigation"]
        if lit["total"] > 0 and lit["open_defendant_count"] == 0:
            closed_cases_only.append(reg)

    # The two traps found while profiling: missing financials, and cases that are all closed.
    chosen = [rng.choice(pool) for pool in (no_financials, closed_cases_only) if pool]
    for pool, n in ((high, n_high), (medium, n_medium), (clean, n_clean)):
        candidates = [c for c in pool if c not in chosen]
        chosen += rng.sample(candidates, min(n, len(candidates)))
    return sorted(dict.fromkeys(chosen))


def make_review(
    versions: tuple[str, str] = DEFAULT_VERSIONS,
    *,
    data_path: Path = DATA_PATH,
    summaries_dir: Path = SUMMARIES_DIR,
    out_dir: Path = REVIEW_DIR,
    seed: int = 42,
) -> list[str]:
    records = {r["registrationNumber"]: r for r in load_companies(data_path)}
    summaries = {v: load_summaries(summaries_dir / f"{v}.jsonl") for v in versions}
    ids = [i for i in pick_companies(list(records.values()), seed) if all(i in s for s in summaries.values())]

    rng = random.Random(seed + 1)  # noqa: S311 - reproducible shuffling, not security
    key: dict[str, dict[str, str]] = {}
    parts = [INSTRUCTIONS]
    for n, reg in enumerate(ids, start=1):
        order = list(versions)
        rng.shuffle(order)
        key[reg] = {"A": order[0], "B": order[1]}
        parts.append(
            f"\n## {n}. {reg}\n\n### Source facts\n\n```text\n{facts_to_text(extract_facts(records[reg]))}\n```\n\n"
            f"### Summary A\n\n{summaries[order[0]][reg]}\n\n### Summary B\n\n{summaries[order[1]][reg]}\n\n"
            "**Preferred (A / B / tie):** \n\n**Notes:** \n"
        )

    out_dir.mkdir(exist_ok=True)
    (out_dir / "blind_review.md").write_text("".join(parts), encoding="utf-8")
    (out_dir / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return ids


def score_review(review_path: Path, key_path: Path) -> dict[str, Any]:
    """Tally preferences per version from a filled-in review file."""
    key = json.loads(key_path.read_text(encoding="utf-8"))
    text = review_path.read_text(encoding="utf-8")
    wins: Counter[str] = Counter()
    unanswered: list[str] = []
    for section in text.split("\n## ")[1:]:
        head = SECTION_HEAD.match(section)
        if not head:
            continue
        reg = head.group(2)
        found = PREFERRED_LINE.search(section)
        choice = found.group(1).strip().lower() if found else ""
        if choice in ("a", "b"):
            wins[key[reg][choice.upper()]] += 1
        elif choice == "tie":
            wins["tie"] += 1
        else:
            unanswered.append(reg)
    return {"wins": dict(wins), "unanswered": unanswered, "total": len(key)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind manual review.")
    parser.add_argument("command", choices=["make", "score"])
    args = parser.parse_args(argv)

    if args.command == "make":
        ids = make_review()
        print(f"Wrote review/blind_review.md for {len(ids)} companies: {', '.join(ids)}")
        return 0

    result = score_review(REVIEW_DIR / "blind_review.md", REVIEW_DIR / "key.json")
    print(f"Preferences out of {result['total']} companies: {result['wins']}")
    if result["unanswered"]:
        print(f"Not yet answered: {', '.join(result['unanswered'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
