# Case study: summarising companies for supplier and partner vetting

**Model:** `gemini-3.5-flash-lite` (Google Gemini API, free tier), temperature 0, one run per company.
**Data used:** all 50 companies in `data/companies.jsonl`.
**Time spent:** **About 7 hours, not counting environment setup. The biggest blocks were profiling the data and choosing thresholds, and generating and checking the summaries for both versions (partly waiting on free-tier rate limits). The rest went on building the fact-extraction layer, the evaluation and review tooling, and writing this document.**

## 1. What is a good summary, and why?

The reader is an analyst on a team that vets companies before choosing them as suppliers or project partners. They have about 30 seconds per company, and the decision they are making is simple: does this company deserve a closer look? The summary is a triage tool, not a report.

I defined a good summary by four qualities, in priority order:

1. **Faithful.** Every figure and claim must trace back to the record. A confident wrong number is worse than a missing one, because the analyst may act on it and has no time to re-check.
2. **Risk-first.** The things that should change the analyst's decision (winding up, a non-clean audit opinion, negative equity, persistent losses, open court cases against the company) come before the profile detail, in order of seriousness.
3. **Honest about gaps.** Three companies have no financial statements at all, and every company's latest financial year is FY2024 while the records are dated 31 December 2025. A good summary says so plainly and does not imply figures it was not given.
4. **Neutral and short.** The team makes the decision, so the summary should inform it and not recommend for or against a company. It should also be short enough to read in the time available (I aimed for about 70 to 120 words).

I did not decide this definition in the abstract. Before designing the improved version, I read three baseline summaries against the source data and noted what went wrong:

| Finding | Evidence |
|---|---|
| Figures were accurate | Every number I checked in the three summaries matched the data |
| One ran over its own length limit | SYN-000067 was 139 words against a 120-word limit (the other two were 114 and 115) |
| Unrequested recommendations | SYN-000029: "Suitable for partnership with standard credit monitoring". SYN-000067: "Recommendation: High financial risk" |
| A real risk went unmentioned | SYN-000067 and SYN-000148 both have operating profit that does not cover interest expense. Neither summary says so |
| Risk was not first | Both loss-making companies opened with name, status and profile bullets, with the risks further down |

The baseline was weaker on prioritisation and restraint than on accuracy, so those are the failures the improved version targets. This was only three summaries, so I did not assume it generalises. That is what the evaluation in section 3 is for.

## 2. What I built

**Core idea: compute the facts in Python, and let the model only write.** Language models are unreliable at arithmetic and at spotting rare signals buried in nested JSON, and the risks that matter here are rare (for example only 2 of 50 companies are winding up, and only 3 audited years have a non-clean opinion). So a deterministic layer works out the numbers and the risk flags, and the model turns verified facts into a short summary.

**`facts.py`** extracts per-company facts (profile, ownership, per-year financials, litigation) and risk flags with a severity of high, medium or info. Decisions worth noting:

- Only **open cases against the company** count as litigation risk (status Pending or Judgment entered, company as defendant). Settled, withdrawn and dismissed cases, and cases the company brought itself, are deliberately ignored.
- Litigation severity depends on size: open claims of 50% or more of equity are high, smaller ones medium. For example SYN-001212 faces RM10.49m of open claims against RM7.91m of equity (133%).
- The provided `leverageRatio` is not used. It is really debt to equity and becomes meaningless when equity is negative (it reaches -106), so the code computes its own measures.
- A name change within the last 12 months is medium severity, because it is a classic vetting signal. Older name changes are context only.
- Missing financials are flagged explicitly, and the code handles them without crashing.

**Thresholds.** I wrote `scripts/profile_data.py`, which counts how many companies each candidate cutoff would flag, and looked for cutoffs that flag a small, meaningful group, ideally at a natural gap in the data. I ran the profiling script, checked its output against each cutoff, and kept them, except for company age, which I raised from 24 to 36 months after the script showed the 24-month rule flagged no companies.

| Rule | Cutoff | Companies flagged | Why this cutoff |
|---|---|---|---|
| Revenue decline | first to latest year, -10% or worse | 12 of 42 | Catches both the mild group (-10% to -14%) and the severe group (-30% or worse) |
| Open claims / equity | 50% or more is high | 3 of 7 | Ratios in the data are 1, 4, 26, 29, 51, 66 and 133, so there is a gap between 29% and 51% |
| Largest shareholder | 75% or more (context only) | 17 of 50 | A 50% cutoff would flag 42 companies, which tells the reader nothing |
| Interest cover | below 1.0 | 11 of 47 | No company sits between 0 and 1.0, so the cutoff is clean |
| Recent name change | under 12 months | 3 of 5 | Values are 1, 3, 6, 15 and 25 months, so there is a gap between 6 and 15 |
| Company age (context only) | under 36 months | 5 of 50 | I changed this from 24 months, because at 24 it flagged nobody |

**Prompts.** Version 1 (`v1_baseline`) is the baseline: it gives the model the raw JSON, the task, the audience and a 120-word limit, and nothing else. I wanted this to be a fair baseline and not a straw man. Version 2 (`v2_structured`) gives the model the verified facts and ranked flags from `facts.py`, plus rules: use only the facts provided, never calculate, lead with high-severity signals, say plainly when financials are missing, name the financial year of the latest figures, do not present closed or plaintiff cases as concerns, do not recommend, and keep to 70 to 110 words.

**Engineering.** `llm.py` saves every model response to `cache/` as soon as it arrives, spaces out requests to respect the free-tier rate limit, and retries with growing delays on rate-limit and server errors. It never caches an empty response. This mattered in practice: during the v2 run the API returned two "503 high demand" errors, both retried automatically and succeeded. The code is covered by 53 tests, including tests built from the real baseline outputs that lock in the manual findings above, and it passes the linter configured in `pyproject.toml`.

## 3. How I know it is good

### Automated checks

`python -m summarizer.evaluate` scores every summary against facts computed from the source data. All checks are deterministic, so they can run on every version:

- **Length:** word count, ignoring markdown symbols.
- **Faithfulness:** every figure in a summary must match a value in the record or one derived from it, allowing for rounding. Unmatched figures are reported.
- **Risk recall:** of the high and medium signals that `facts.py` found for each company, how many does the summary mention?
- **Unrequested recommendations:** wording that tells the reader what to do.
- **Litigation mentioned with no open case:** for companies whose cases are all closed or brought by the company, does the summary mention litigation at all? This is a rough proxy for presenting closed cases as a threat, so I judged the framing in the manual review.
- **Financial year and missing data:** does the summary state which year its figures are from, and does it say plainly when a company has no financials?

Results on all 50 companies:

| Metric | v1_baseline | v2_structured |
|---|---|---|
| Summaries scored | 50 | 50 |
| Mean words | 113 | 92 |
| Over 120 words | 12 | 0 |
| Summaries with an unsupported figure | 2 | 5 |
| Unsupported figures (total) | 2 | 5 |
| Recall: high-severity signals | 15/15 (100%) | 15/15 (100%) |
| Recall: all high+medium signals | 56/71 (79%) | 71/71 (100%) |
| Unrequested recommendations | 13 | 0 |
| Litigation mentioned, no open case | 10/11 (91%) | 4/11 (36%) |
| States latest financial year | 46/47 (98%) | 47/47 (100%) |
| Says plainly 'no financials' | 0/3 (0%) | 3/3 (100%) |

Recall by signal (mentioned / present):

| Signal | v1_baseline | v2_structured |
|---|---|---|
| AUDIT_ADVERSE | 3/3 (100%) | 3/3 (100%) |
| LIQUIDITY_TIGHT | 6/7 (86%) | 7/7 (100%) |
| LITIGATION_EXPOSURE_HIGH | 3/3 (100%) | 3/3 (100%) |
| LITIGATION_OPEN_DEFENDANT | 4/4 (100%) | 4/4 (100%) |
| LOSSES_PERSISTENT | 6/6 (100%) | 6/6 (100%) |
| LOSS_LATEST_YEAR | 5/5 (100%) | 5/5 (100%) |
| NAME_CHANGED_RECENTLY | 3/3 (100%) | 3/3 (100%) |
| NEGATIVE_EQUITY | 1/1 (100%) | 1/1 (100%) |
| NEGATIVE_OPERATING_CASH_FLOW | 11/11 (100%) | 11/11 (100%) |
| NO_FINANCIALS | 0/3 (0%) | 3/3 (100%) |
| REVENUE_DECLINE | 12/12 (100%) | 12/12 (100%) |
| STATUS_NOT_ACTIVE | 2/2 (100%) | 2/2 (100%) |
| WEAK_INTEREST_COVER | 0/11 (0%) | 11/11 (100%) |

v2 improved on every automated check. Its raw count of flagged figures is higher than v1's, but most of those are limits of my checker, as explained below. Its summaries were shorter (mean 92 words against 113, and none over 120 words against 12 for v1), none gave an unrequested recommendation (13 of the v1 summaries did), and it mentioned all 71 high and medium signals against 56 for v1. The difference is concentrated in two signals: weak interest cover, which v1 never mentioned (0 of 11), and missing financials, which v1 never stated (0 of 3). Both versions mentioned all 15 high-severity signals, so the baseline was already reliable on the most serious risks.

Three caveats apply. Recall is measured against the flags computed in `facts.py`, which v2 receives as input, so a score of 100% shows that v2 follows its inputs and does not show that the flags are right. The litigation row only detects whether litigation is mentioned for the 11 companies with no open case against them, not whether it is framed as a threat, so I used the manual review to judge framing. And v2 had more summaries with an unsupported figure than v1 (5 against 2).

I checked each of the seven summaries the figure check flagged (two for v1, five for v2). Four of the five v2 flags are the year 2025 in the phrase "12 months to 31 Dec 2025", which v2 copies from its prompt. It is the reporting date and not a data error, but my checker only allows numbers found in the record, so this is a limitation of the checker. The fifth v2 flag is the year 2023 for SYN-001253. That company has financials for FY2022 and FY2024 only, and the summary correctly names 2023 as the missing year, so this is also a checker limit. After checking, v2 had no real unsupported figures and v1 had one. For v1, the figure 1.90 for SYN-000286 is a correct current ratio (1.895) that the model calculated itself and the record does not store, so that flag is also a checker limit. The other v1 flag, 9.3M for SYN-000397, is a real error: total claims against that company are RM9.54m, and RM7.43m of that is a settled case.

### Blind manual review

Automated checks are proxies. Keyword matching can miss nuance and cannot judge whether a summary is useful. So I also read 12 companies side by side. The tool (`python -m summarizer.review`) picks a mix of high-risk, medium-risk and clean companies, always includes a company with no financials and one whose court cases are all closed (the two traps found while profiling), shuffles which version appears as A or B, and keeps the answer key hidden until scoring. The blinding is imperfect, because the two versions differ visibly in style (v1 uses headings and bullets, v2 uses short prose), so I judged each pair against my four criteria from section 1 and wrote a note for each.

My first-pass preferences split 5 for v1, 5 for v2 and 2 ties. I then rechecked the pairs where the versions differed on a risk against the raw data and revised five of them: four moved from v1 to v2 (SYN-000067, SYN-000397, SYN-001212 and SYN-001294) and one moved from v2 to a tie (SYN-000476). In every revised pair, the v1 summary had missed a risk, overstated an exposure or added an unrequested verdict, so the revisions follow my criteria. The final tally is v2 preferred in 8 of 12, v1 in 1 and 3 ties. Because the revisions came after I saw the evidence, the final tally is less independent than the first pass, so I report both.

Three examples show what the review found, and whether the automated checks saw it:

- **SYN-000397 (v1 overstated the risk).** The v1 summary headlined "over RM9.3M" of claims and "high risk, proceed with caution". RM7.43m of that is one settled case. The open exposure is RM1.56m, 66% of equity, which the v2 summary states correctly. The automated checks flagged only the loose figure "9.3M", not the misleading framing.
- **SYN-001203 (v1 hid a gap).** This company has no financial statements. The v1 summary never says so, while v2 states it as the primary risk. The automated check caught this: v1 stated missing financials in 0 of 3 cases.
- **SYN-001294 (v1 missed a signal).** Operating profit does not cover interest (cover of -10.5x). v1 omitted it and called leverage "manageable", while v2 listed all three medium signals. The automated check caught this pattern: v1 mentioned weak interest cover in 0 of 11 cases.

The one pair where I preferred v1 was SYN-001457, a clean company. v1 showed the losses of 2022 and 2023 and the recovery in 2024, while v2 reported only the latest year, because no flag fired. This shows a limit of the design: v2 only reports what its flags tell it.

### Mistakes I caught along the way

- **A miscounted baseline.** I first reported that all three baseline summaries broke the 120-word limit. That count included markdown symbols such as `*` and `###`. Counted properly the lengths were 114, 139 and 115 words, so only one broke the limit, and the evaluation script now counts words correctly.
- **A threshold that did nothing.** The profiling script showed the company-age rule at 24 months flagged no companies, so I raised it to 36 months, which flags the five companies incorporated in 2023.

### Limits of this evidence

- Risk recall uses keyword patterns, so it can miss a correct paraphrase or count a loose mention. The blind review is the check on that.
- Recall is measured against the flags from `facts.py`, which v2 receives as input. A score of 100% shows that v2 follows its inputs, not that the flags are correct, and the review is the independent check.
- The automated checks cannot detect an overstated exposure. For SYN-000397, one summary headlined over RM9.3m of claims, although RM7.43m of that is a settled case and the open exposure is RM1.56m (66% of equity). Only the manual review caught this.
- The review was not fully blind, because the two versions differ visibly in style.
- v2 copies the wording of the computed flags, so it sometimes leaks internal labels ("medium signals", "an info signal") and awkward phrasing ("100.0% part-owned", "a loss of -RM843,149"). The automated checks cannot see this.
- The thresholds come from the distribution of 50 synthetic companies. They are not validated against real outcomes such as defaults, because no labelled outcomes exist, and some cases sit close to a cutoff (for example a 51% litigation ratio against a 50% rule), so a different team could reasonably draw the line elsewhere.
- Each summary was generated once at temperature 0, so I have not measured how much results vary between runs.
- I read 12 of the 50 summaries myself, and I am a single reviewer.
- The figure check only allows numbers found in the record, so it flags correct derived figures (such as a current ratio the model calculated) and the reporting date copied from the prompt. Of the seven flagged figures, only the one for SYN-000397 is a real error.

## 4. What I would do next

- **Fix the wording.** Rewrite the flag messages in plain English, with no internal labels or double negatives, then regenerate v2 and repeat the review.
- **Measure variation.** Generate each version several times and check whether the conclusions hold.
- **Add a second judge.** Use an LLM judge from a different model family, calibrated against my manual review, to score all 50 on usefulness.
- **Get expert input.** Ask the vetting team where they would draw each threshold and how old financial statements may be before they are flagged as stale (every company's latest year is FY2024), and use any real outcome data to validate both.
- **Make the facts layer safer.** Replace the plain dictionaries in `facts.py` with typed objects, so a mistyped field name is caught by the editor and not at run time.

## How to run

```powershell
uv sync
$env:PYTHONPATH = "src"
uv run python -m summarizer.generate --version v1_baseline
uv run python -m summarizer.generate --version v2_structured
uv run python -m summarizer.evaluate
uv run python -m summarizer.review make
uv run python -m summarizer.review score
uv run pytest
```

Put `GEMINI_API_KEY` in a `.env` file (see `.env.example`). Responses are cached in `cache/`, so re-running costs no API calls.

## AI tools used

I used Claude (Anthropic) as a coding and analysis assistant throughout. It helped me with:

- profiling the dataset and proposing the first risk thresholds;
- drafting `facts.py`, the prompts, the model client, the generation script, the evaluation script and the blind review tool, along with their tests;
- checking specific claims in the reviewed summaries against the raw data, and drafting the wording of my review notes;
- suggesting the structure of this document and a first draft of it.

What I did myself: I set up the project, ran and tested all code on my machine, reviewed and edited the code, made the threshold decisions after running the profiling script, read the baseline summaries against the data, made the final preference decisions in the blind review, and checked the claims in this document against my results. I used the Gemini API (`gemini-3.5-flash-lite`) to generate the summaries.