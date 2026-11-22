"""A small, cautious LLM client: disk cache, rate limiting and retries."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# (system prompt or None, user prompt, model, temperature) -> response text
Backend = Callable[[str | None, str, str, float], str]

RETRYABLE_CODES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a call still fails after all retries."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 4.0  # doubles after each failed attempt


def cache_key(model: str, system: str | None, prompt: str, temperature: float) -> str:
    """Stable hash of everything that can change the model's answer."""
    payload = json.dumps([model, system, prompt, temperature], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def gemini_backend(system: str | None, prompt: str, model: str, temperature: float) -> str:
    """Call the Gemini API once. The key is read from GEMINI_API_KEY (never hard-coded)."""
    from google import genai  # imported here so tests and cached re-runs need no SDK or key
    from google.genai import errors, types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Put it in your .env file.")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except errors.APIError as exc:
        # Re-raise with the HTTP code attached so the retry loop can decide.
        raise TransientAPIError(exc.code, str(exc)) from exc
    return response.text or ""


class TransientAPIError(RuntimeError):
    """An API error that carries its HTTP status code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code


class LLMClient:
    """Generate text with caching, a minimum gap between calls, and retries."""

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        *,
        backend: Backend = gemini_backend,
        requests_per_minute: float = 10.0,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._backend = backend
        self._min_gap_s = 60.0 / requests_per_minute
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.calls_made = 0
        self.cache_hits = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        key = cache_key(self.model, system, prompt, temperature)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        text = self._call_with_retries(system, prompt, temperature)
        record = {
            "model": self.model,
            "temperature": temperature,
            "system": system,
            "prompt": prompt,
            "response": text,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return text

    def _wait_for_slot(self) -> None:
        if self._last_call is not None:
            remaining = self._min_gap_s - (self._clock() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()

    def _call_with_retries(self, system: str | None, prompt: str, temperature: float) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            self._wait_for_slot()
            try:
                self.calls_made += 1
                text = self._backend(system, prompt, self.model, temperature)
            except TransientAPIError as exc:
                if exc.code not in RETRYABLE_CODES:
                    raise LLMError(f"Non-retryable API error: {exc}") from exc
                last_error = exc
            else:
                if text.strip():
                    return text.strip()
                last_error = LLMError("Model returned an empty response")  # never cache empties
            delay = self._retry.base_delay_s * 2 ** (attempt - 1)
            logger.warning("Attempt %d/%d failed (%s); retrying in %.0fs", attempt, self._retry.max_attempts, last_error, delay)
            self._sleep(delay)
        raise LLMError(f"Gave up after {self._retry.max_attempts} attempts: {last_error}")
