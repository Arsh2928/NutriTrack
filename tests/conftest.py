"""Pytest configuration ensuring deterministic unit test isolation even when `.env` has live API keys."""

from __future__ import annotations

import os
import pytest

_LIVE_PROVIDER_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "NUTRITRACK_AI_MODE",
    "NUTRITRACK_LLM_PROVIDER",
    "NUTRITRACK_REQUIRE_LLM",
)


@pytest.fixture(autouse=True)
def _isolate_live_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local `.env` API keys from leaking into deterministic unit tests."""
    for key in _LIVE_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
