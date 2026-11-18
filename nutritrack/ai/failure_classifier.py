"""Central API & network failure classifier for Gemini and Groq cloud providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional

from nutritrack.ai.exceptions import (
    ProviderError,
    sanitize_error_text,
)

_RETRY_AFTER_MSG_RE = re.compile(
    r"(?:retry in|try again in|retry[_ -]?after[:=\s]+)\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds|m|min|minutes)?",
    re.IGNORECASE,
)
_DURATION_HEADER_RE = re.compile(r"^(?:(\d+)m)?(\d+(?:\.\d+)?)s?$")


@dataclass(frozen=True)
class ClassifiedFailure:
    """Structured result of classifying a provider or network exception."""

    provider: str
    category: str
    sanitized_reason: str
    status_code: Optional[int] = None
    retryable: bool = False
    fallback_eligible: bool = False
    trip_circuit_immediately: bool = False
    is_quota_exhausted: bool = False
    is_rate_limited: bool = False
    is_config_error: bool = False
    is_network_error: bool = False
    retry_after_seconds: Optional[float] = None
    cooldown_override_seconds: Optional[float] = None


def _extract_headers(exc: BaseException) -> dict[str, str]:
    headers: dict[str, str] = {}
    resp = getattr(exc, "response", None)
    raw_headers = getattr(exc, "headers", None) or (getattr(resp, "headers", None) if resp is not None else None)
    if isinstance(raw_headers, Mapping):
        for k, v in raw_headers.items():
            headers[str(k).lower()] = str(v)
    return headers


def _parse_duration_str(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    text = str(raw).strip().lower()
    try:
        val = float(text)
        return max(0.0, val)
    except ValueError:
        pass
    m = _DURATION_HEADER_RE.match(text)
    if m:
        mins = float(m.group(1) or 0.0)
        secs = float(m.group(2) or 0.0)
        return mins * 60.0 + secs
    return None


def _extract_retry_after(exc: BaseException, message: str) -> Optional[float]:
    if isinstance(exc, ProviderError) and exc.retry_after_seconds is not None:
        return max(0.0, float(exc.retry_after_seconds))
    headers = _extract_headers(exc)
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        if key in headers:
            parsed = _parse_duration_str(headers[key])
            if parsed is not None:
                return parsed
    m = _RETRY_AFTER_MSG_RE.search(message or "")
    if m:
        amount = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            return amount / 1000.0
        if unit in {"m", "min", "minutes"}:
            return amount * 60.0
        return amount
    return None


def _extract_status_code(exc: BaseException, message: str) -> Optional[int]:
    if isinstance(exc, ProviderError) and exc.status_code is not None:
        return int(exc.status_code)
    for attr in ("status_code", "status", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val <= 599:
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int) and 100 <= sc <= 599:
            return sc
    for code in (400, 401, 403, 404, 408, 422, 429, 500, 502, 503, 504):
        if re.search(rf"\b(?:http|status|error|code)?\s*{code}\b", message, re.IGNORECASE):
            return code
    return None


def classify_provider_failure(exc: BaseException, *, provider: str = "unknown") -> ClassifiedFailure:
    """Classify an exception from Gemini, Groq, or HTTP transport into actionable routing rules."""
    raw_msg = f"{type(exc).__name__}: {exc}"
    clean_msg = sanitize_error_text(raw_msg)
    lower = clean_msg.lower()
    exc_name = type(exc).__name__.lower()
    status_code = _extract_status_code(exc, clean_msg)
    retry_after = _extract_retry_after(exc, clean_msg)

    # 0. Explicit structured ProviderError subclasses
    if isinstance(exc, ProviderError) and exc.category != "provider_error":
        is_quota = bool(exc.is_quota_exhausted or exc.category == "quota_exhausted_daily")
        is_rl = bool(exc.category == "rate_limit_temporary" or status_code == 429) and not is_quota
        is_cfg = bool(exc.is_config_error or exc.category in {"invalid_api_key", "permission_denied", "unsupported_model"})
        is_net = exc.category in {"internet_disconnected", "dns_failure", "connection_timeout", "response_timeout"}
        return ClassifiedFailure(
            provider=provider,
            category=exc.category,
            status_code=status_code,
            sanitized_reason=clean_msg,
            retryable=bool(exc.retryable and not is_quota and not is_cfg),
            fallback_eligible=exc.category != "invalid_request",
            trip_circuit_immediately=bool(is_quota or is_cfg or exc.category in {"internet_disconnected", "dns_failure"}),
            is_quota_exhausted=is_quota,
            is_rate_limited=is_rl,
            is_config_error=is_cfg,
            is_network_error=is_net,
            retry_after_seconds=retry_after,
            cooldown_override_seconds=retry_after if is_quota else None,
        )

    # 1. DNS resolution failures -> do NOT retry repeatedly, fallback immediately
    if any(
        k in lower
        for k in (
            "gaierror",
            "getaddrinfo failed",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname",
            "dns",
        )
    ):
        return ClassifiedFailure(
            provider=provider,
            category="dns_failure",
            status_code=status_code,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=True,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=True,
        )

    # 2. Internet disconnected / unreachable network -> skip provider immediately and fallback
    if isinstance(exc, ConnectionRefusedError) or any(
        k in lower
        for k in (
            "network is unreachable",
            "enetunreach",
            "ehostunreach",
            "no route to host",
            "internet disconnected",
            "network disconnected",
            "connection refused",
            "connection reset",
            "apiconnectionerror",
            "connecterror",
        )
    ):
        return ClassifiedFailure(
            provider=provider,
            category="internet_disconnected",
            status_code=status_code,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=True,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=True,
        )

    # 3. Connection or response timeout -> bounded retry, then fallback
    if isinstance(exc, TimeoutError) or status_code == 408 or any(
        k in lower or k in exc_name
        for k in ("timeout", "timed out", "apitimeouterror", "readtimeout", "connecttimeout")
    ):
        cat = "response_timeout" if "read" in lower or "response" in lower else "connection_timeout"
        return ClassifiedFailure(
            provider=provider,
            category=cat,
            status_code=status_code,
            sanitized_reason=clean_msg,
            retryable=True,
            fallback_eligible=False,
            trip_circuit_immediately=False,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=True,
        )

    # 4. Invalid API key / HTTP 401 -> do NOT retry; mark configuration error
    if status_code == 401 or any(
        k in lower
        for k in (
            "invalid api key",
            "api_key_invalid",
            "api key not valid",
            "incorrect api key",
            "authenticationerror",
            "unauthenticated",
            "unauthorized",
        )
    ):
        return ClassifiedFailure(
            provider=provider,
            category="invalid_api_key",
            status_code=401,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=True,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=True,
            is_network_error=False,
        )

    # 5. Permission denied / HTTP 403 -> do NOT retry; report access/config failure
    if status_code == 403 or any(
        k in lower for k in ("permissiondenied", "permission_denied", "permission denied", "forbidden", "access denied")
    ):
        return ClassifiedFailure(
            provider=provider,
            category="permission_denied",
            status_code=403,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=True,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=True,
            is_network_error=False,
        )

    # 6. HTTP 429 / Rate Limit vs Daily Quota Exhaustion
    if status_code == 429 or any(
        k in lower for k in ("ratelimit", "rate_limit", "rate limit", "too many requests", "resource_exhausted", "quota")
    ):
        is_daily_quota = any(
            k in lower
            for k in (
                "daily quota",
                "per day",
                "requests per day",
                "rpd",
                "generaterequestsperday",
                "exceeded your current quota",
                "billing",
                "credit balance",
                "quota exhausted",
                "quota_exhausted",
                "daily limit",
            )
        )
        if is_daily_quota:
            return ClassifiedFailure(
                provider=provider,
                category="quota_exhausted_daily",
                status_code=429,
                sanitized_reason=clean_msg,
                retryable=False,
                fallback_eligible=False,
                trip_circuit_immediately=True,
                is_quota_exhausted=True,
                is_rate_limited=False,
                is_config_error=False,
                is_network_error=False,
                retry_after_seconds=retry_after,
                cooldown_override_seconds=retry_after,
            )
        return ClassifiedFailure(
            provider=provider,
            category="rate_limit_temporary",
            status_code=429,
            sanitized_reason=clean_msg,
            retryable=True,
            fallback_eligible=False,
            trip_circuit_immediately=False,
            is_quota_exhausted=False,
            is_rate_limited=True,
            is_config_error=False,
            is_network_error=False,
            retry_after_seconds=retry_after,
        )

    # 7. Unsupported or decommissioned model
    if any(
        k in lower
        for k in (
            "model_not_found",
            "model not found",
            "does not exist",
            "decommissioned",
            "unsupported model",
            "does not support tools",
            "tool calling is not supported",
        )
    ) or (status_code == 404 and "model" in lower):
        return ClassifiedFailure(
            provider=provider,
            category="unsupported_model",
            status_code=status_code or 404,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=True,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=True,
            is_network_error=False,
        )

    # 8a. Malformed tool call / tool schema failure (Groq tool_use_failed / Gemini invalid function call) -> never retry
    if any(
        k in lower
        for k in (
            "tool_use_failed",
            "failed_generation",
            "tool call validation failed",
            "invalid_function_call",
            "malformed_function_call",
            "failed to call a function",
        )
    ):
        return ClassifiedFailure(
            provider=provider,
            category="malformed_tool_call",
            status_code=status_code or 400,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=False,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=False,
        )

    # 8b. Invalid / malformed request (HTTP 400 / 422) -> do NOT retry identical malformed request
    if status_code in {400, 422} or "badrequesterror" in exc_name or "invalid_request" in lower:
        return ClassifiedFailure(
            provider=provider,
            category="invalid_request",
            status_code=status_code or 400,
            sanitized_reason=clean_msg,
            retryable=False,
            fallback_eligible=False,
            trip_circuit_immediately=False,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=False,
        )

    # 9. Upstream HTTP 500/502/503/504 server errors -> bounded retry, then fallback
    if status_code in {500, 502, 503, 504} or any(
        k in lower for k in ("internalservererror", "serviceunavailable", "badgateway", "gatewaytimeout", "overloaded", "server error")
    ):
        return ClassifiedFailure(
            provider=provider,
            category="server_error",
            status_code=status_code or 503,
            sanitized_reason=clean_msg,
            retryable=True,
            fallback_eligible=False,
            trip_circuit_immediately=False,
            is_quota_exhausted=False,
            is_rate_limited=False,
            is_config_error=False,
            is_network_error=False,
        )

    # Default unknown provider error -> fallback eligible, single bounded retry
    return ClassifiedFailure(
        provider=provider,
        category="provider_error",
        status_code=status_code,
        sanitized_reason=clean_msg,
        retryable=True,
        fallback_eligible=False,
        trip_circuit_immediately=False,
        is_quota_exhausted=False,
        is_rate_limited=False,
        is_config_error=False,
        is_network_error=False,
    )
