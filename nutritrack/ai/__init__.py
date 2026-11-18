"""NutriTrack Cloud AI Provider Routing, Failover, Circuit Breaker, and Data-Only Execution Package."""

from nutritrack.ai.circuit_breaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
)
from nutritrack.ai.config import (
    DEFAULT_AI_MODE,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    VALID_AI_MODES,
    AIProviderConfig,
    ai_mode_scope,
    assert_network_allowed,
    clear_registered_ai_mode,
    get_active_ai_mode,
    is_model_supported,
    is_offline_mode,
    load_persisted_mode,
    normalize_ai_mode,
    register_active_ai_mode,
    save_persisted_mode,
)
from nutritrack.ai.exceptions import (
    ProviderConfigError,
    ProviderError,
    ProviderNetworkError,
    ProviderQuotaExhaustedError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    sanitize_error_text,
)
from nutritrack.ai.failure_classifier import ClassifiedFailure, classify_provider_failure
from nutritrack.ai.provider_factory import ProviderFactory
from nutritrack.ai.provider_health import ProviderHealthTracker
from nutritrack.ai.provider_status import (
    DATA_ONLY_BANNER,
    ProviderStatusSnapshot,
    compute_effective_label,
)
from nutritrack.ai.recovery_manager import RecoveryManager
from nutritrack.ai.retry_policy import RetryDecision, RetryPolicy


def __getattr__(name: str):
    if name in ("DataOnlyChatClient", "DataOnlyService"):
        from nutritrack.ai.data_only_service import DataOnlyChatClient, DataOnlyService

        return DataOnlyChatClient if name == "DataOnlyChatClient" else DataOnlyService
    if name in ("AIProviderManager", "RoutedNutriTrackChatClient"):
        from nutritrack.ai.provider_manager import AIProviderManager, RoutedNutriTrackChatClient

        return AIProviderManager if name == "AIProviderManager" else RoutedNutriTrackChatClient
    raise AttributeError(f"module 'nutritrack.ai' has no attribute {name!r}")

__all__ = [
    "AIProviderConfig",
    "AIProviderManager",
    "ClassifiedFailure",
    "CircuitBreaker",
    "DATA_ONLY_BANNER",
    "DEFAULT_AI_MODE",
    "DEFAULT_GEMINI_BASE_URL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_BASE_URL",
    "DEFAULT_GROQ_MODEL",
    "DataOnlyChatClient",
    "DataOnlyService",
    "ProviderConfigError",
    "ProviderError",
    "ProviderFactory",
    "ProviderHealthTracker",
    "ProviderNetworkError",
    "ProviderQuotaExhaustedError",
    "ProviderRateLimitError",
    "ProviderServerError",
    "ProviderStatusSnapshot",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RecoveryManager",
    "RetryDecision",
    "RetryPolicy",
    "RoutedNutriTrackChatClient",
    "STATE_CLOSED",
    "STATE_HALF_OPEN",
    "STATE_OPEN",
    "VALID_AI_MODES",
    "ai_mode_scope",
    "assert_network_allowed",
    "classify_provider_failure",
    "clear_registered_ai_mode",
    "compute_effective_label",
    "get_active_ai_mode",
    "is_model_supported",
    "is_offline_mode",
    "load_persisted_mode",
    "normalize_ai_mode",
    "register_active_ai_mode",
    "sanitize_error_text",
    "save_persisted_mode",
]
