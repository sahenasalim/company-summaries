"""Generate summaries for a version and save them to summaries/<version>.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from summarizer.facts import extract_facts, load_companies
from summarizer.llm import LLMClient, LLMError
from summarizer.prompts import PROMPT_BUILDERS

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "companies.jsonl"
SUMMARIES_DIR = ROOT / "summaries"
CACHE_DIR = ROOT / "cache"
DEFAULT_MODEL = "gemini-3.5-flash-lite"  # confirm it is on your free tier in AI Studio

logger = logging.getLogger("summarizer.generate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate company summaries.")
    p.add_argument("--version", required=True, choices=sorted(PROMPT_BUILDERS))
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    p.add_argument("--limit", type=int, default=None, help="only the first N companies (for a cheap test run)")
    p.add_argument("--ids", nargs="*", default=None, help="only these registration numbers")
    p.add_argument("--rpm", type=float, default=10.0, help="max requests per minute (match your free-tier limit)")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args(argv)


def select_records(records: list[dict], limit: int | None, ids: list[str] | None) -> list[dict]:
    if ids:
        wanted = set(ids)
        records = [r for r in records if r["registrationNumber"] in wanted]
    return records[:limit] if limit else records


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    records = select_records(load_companies(DATA_PATH), args.limit, args.ids)
    build_prompt = PROMPT_BUILDERS[args.version]
    client = LLMClient(args.model, CACHE_DIR, requests_per_minute=args.rpm)
    logger.info("Version %s | model %s | %d companies", args.version, args.model, len(records))

    results: list[dict[str, str]] = []
    failed: list[str] = []
    for i, record in enumerate(records, start=1):
        reg = record["registrationNumber"]
        system, user = build_prompt(record, extract_facts(record))
        try:
            summary = client.generate(user, system=system, temperature=args.temperature)
        except LLMError:
            logger.exception("%s failed", reg)
            failed.append(reg)
            continue
        results.append({"registrationNumber": reg, "summary": summary})
        logger.info("[%d/%d] %s done", i, len(records), reg)

    SUMMARIES_DIR.mkdir(exist_ok=True)
    out = SUMMARIES_DIR / f"{args.version}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    logger.info(
        "Wrote %d summaries to %s (%d API calls, %d from cache)", len(results), out, client.calls_made, client.cache_hits
    )
    if failed:
        logger.error("%d failed: %s. Re-run the same command to retry only those.", len(failed), ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
