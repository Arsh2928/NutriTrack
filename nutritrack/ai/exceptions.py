"""Structured exceptions and secret-redaction utilities for NutriTrack Cloud AI Routing."""

from __future__ import annotations

from typing import Optional

from nutritrack.state.run_log import SECRET_VALUE_PATTERNS, sanitize_error_text
from nutritrack.tools.base import LLMUnavailableError

_SECRET_PATTERNS = SECRET_VALUE_PATTERNS


class ProviderError(LLMUnavailableError):
    """Base exception for cloud AI provider errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        category: str = "provider_error",
        status_code: Optional[int] = None,
        retryable: bool = False,
        retry_after_seconds: Optional[float] = None,
        is_quota_exhausted: bool = False,
        is_config_error: bool = False,
    ) -> None:
        clean_msg = sanitize_error_text(message)
        super().__init__(clean_msg)
        self.provider = provider
        self.category = category
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.is_quota_exhausted = is_quota_exhausted
        self.is_config_error = is_config_error


class ProviderConfigError(ProviderError):
    """Invalid API key, missing credentials, permission error, or unsupported model."""


class ProviderRateLimitError(ProviderError):
    """Temporary HTTP 429 rate-limit throttling (RPM/TPM)."""


class ProviderQuotaExhaustedError(ProviderError):
    """Daily or account quota exhausted (HTTP 429 / RESOURCE_EXHAUSTED)."""


class ProviderNetworkError(ProviderError):
    """Internet disconnected, DNS failure, or connection refused."""


class ProviderTimeoutError(ProviderError):
    """Connection or response timeout."""


class ProviderServerError(ProviderError):
    """Upstream HTTP 500/502/503/504 server error."""


class ProviderUnavailableError(LLMUnavailableError):
    """Raised when the selected cloud AI provider (Gemini or Groq) fails or is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        category: str = "provider_error",
        status_code: Optional[int] = None,
        retryable: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(sanitize_error_text(message, 400))
        self.provider = provider
        self.category = category
        self.status_code = status_code
        self.retryable = retryable
        self.reason = sanitize_error_text(reason) if reason else None
