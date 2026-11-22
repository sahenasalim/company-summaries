"""Prompt builders, one per summary version."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from summarizer.facts import facts_to_text

Prompt = tuple[str | None, str]

# ----------------------------------------------------------------- v1: baseline
V1_SYSTEM = "You write short summaries of company records for a business team."

V1_USER = """Summarise the company record below in Markdown, in at most 120 words. \
The reader is a team that vets companies before choosing them as suppliers or project partners.

{record_json}"""


def build_v1_baseline(record: dict[str, Any], facts: dict[str, Any]) -> Prompt:  # noqa: ARG001 - baseline ignores facts
    """Baseline: hand the raw JSON to the model, with only the task and a length limit."""
    return V1_SYSTEM, V1_USER.format(record_json=json.dumps(record, ensure_ascii=False, indent=1))


# ------------------------------------------------------------ v2: structured facts
V2_SYSTEM = """You write triage summaries of companies for a team that vets suppliers and project partners. \
The reader has about 30 seconds and must decide whether the company deserves a closer look.

Rules:
1. Use ONLY the facts provided. Never calculate, estimate or add outside knowledge. Copy figures exactly as written.
2. Lead with what matters most: the HIGH signals first, then MEDIUM. Mention INFO signals only if they add useful context.
3. If financial statements are missing, say so plainly. Never imply figures that are not provided.
4. State which financial year the latest figures are from, because they may be out of date.
5. Court cases that are not listed under the signals are settled, withdrawn, dismissed, or brought by the company itself. \
Do not present them as concerns.
6. Be neutral and factual. Do not recommend for or against doing business with the company.
7. 70 to 110 words. Plain Markdown, no headings."""

V2_USER = """Write the summary for this company.

Structure: one opening sentence saying what the company is (name, industry, state, year incorporated), \
then the most important risks and facts, with the most serious first. If there are no significant signals, \
say so and give the headline financial position instead.

{facts_text}"""


def build_v2_structured(record: dict[str, Any], facts: dict[str, Any]) -> Prompt:  # noqa: ARG001 - v2 uses facts only
    """v2: the model sees verified facts and ranked signals computed in Python."""
    return V2_SYSTEM, V2_USER.format(facts_text=facts_to_text(facts))


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Prompt]] = {
    "v1_baseline": build_v1_baseline,
    "v2_structured": build_v2_structured,
}
