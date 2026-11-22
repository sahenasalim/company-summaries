"""Tests for the LLM client, using a fake backend (no network, no API key)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarizer.llm import LLMClient, LLMError, RetryPolicy, TransientAPIError, cache_key


class FakeBackend:
    """Replays a scripted list of outcomes: a string is a reply, an exception is raised."""

    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, system: str | None, prompt: str, model: str, temperature: float) -> str:  # noqa: ARG002
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(tmp_path: Path, backend: FakeBackend, **kwargs) -> tuple[LLMClient, list[float]]:
    sleeps: list[float] = []
    client = LLMClient(
        "test-model",
        tmp_path / "cache",
        backend=backend,
        sleep=sleeps.append,
        clock=lambda: 0.0,
        **kwargs,
    )
    return client, sleeps


def test_second_identical_call_is_served_from_cache(tmp_path):
    backend = FakeBackend("hello")
    client, _ = make_client(tmp_path, backend)
    assert client.generate("p", system="s") == "hello"
    assert client.generate("p", system="s") == "hello"
    assert backend.calls == 1
    assert client.cache_hits == 1


def test_cache_survives_a_new_client(tmp_path):
    make_client(tmp_path, FakeBackend("saved"))[0].generate("p")
    second_backend = FakeBackend()  # would raise if called
    assert make_client(tmp_path, second_backend)[0].generate("p") == "saved"
    assert second_backend.calls == 0


def test_cache_key_changes_with_every_input():
    base = cache_key("m", "s", "p", 0.0)
    assert base != cache_key("m2", "s", "p", 0.0)
    assert base != cache_key("m", "s2", "p", 0.0)
    assert base != cache_key("m", "s", "p2", 0.0)
    assert base != cache_key("m", "s", "p", 0.5)


def test_cache_file_records_prompt_and_response(tmp_path):
    client, _ = make_client(tmp_path, FakeBackend("answer"))
    client.generate("my prompt", system="my system")
    saved = json.loads(next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["response"] == "answer"
    assert saved["prompt"] == "my prompt"
    assert saved["model"] == "test-model"


def test_retries_rate_limit_then_succeeds_with_backoff(tmp_path):
    backend = FakeBackend(TransientAPIError(429, "slow down"), TransientAPIError(503, "busy"), "ok")
    client, sleeps = make_client(tmp_path, backend, retry=RetryPolicy(max_attempts=5, base_delay_s=2.0))
    assert client.generate("p") == "ok"
    assert backend.calls == 3
    assert 2.0 in sleeps
    assert 4.0 in sleeps  # delay doubles


def test_gives_up_after_max_attempts_and_caches_nothing(tmp_path):
    backend = FakeBackend(*[TransientAPIError(429, "no")] * 3)
    client, _ = make_client(tmp_path, backend, retry=RetryPolicy(max_attempts=3, base_delay_s=1.0))
    with pytest.raises(LLMError, match="Gave up"):
        client.generate("p")
    assert not list((tmp_path / "cache").glob("*.json"))


def test_client_errors_are_not_retried(tmp_path):
    backend = FakeBackend(TransientAPIError(400, "bad request"), "never reached")
    client, _ = make_client(tmp_path, backend)
    with pytest.raises(LLMError, match="Non-retryable"):
        client.generate("p")
    assert backend.calls == 1


def test_empty_response_is_retried_and_never_cached(tmp_path):
    backend = FakeBackend("   ", "real answer")
    client, _ = make_client(tmp_path, backend)
    assert client.generate("p") == "real answer"
    saved = json.loads(next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["response"] == "real answer"


def test_minimum_gap_between_calls_is_enforced(tmp_path):
    backend = FakeBackend("a", "b")
    client, sleeps = make_client(tmp_path, backend, requests_per_minute=6.0)  # 10 s apart
    client.generate("one")
    client.generate("two")
    assert 10.0 in sleeps
