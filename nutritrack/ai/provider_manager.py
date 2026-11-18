"""Central Cloud AI Provider Manager and MAF Routing Chat Client for NutriTrack."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional, Sequence

from agent_framework import BaseChatClient, ChatResponse, FunctionInvocationLayer, Message

from nutritrack.ai.circuit_breaker import STATE_HALF_OPEN, STATE_OPEN, CircuitBreaker
from nutritrack.ai.config import (
    AIProviderConfig,
    ai_mode_scope,
    is_model_supported,
    is_turn_cancelled,
    normalize_ai_mode,
    register_active_ai_mode,
    save_persisted_mode,
)
from nutritrack.ai.data_only_service import DataOnlyService
from nutritrack.ai.exceptions import (
    ProviderUnavailableError,
    ProviderConfigError,
    sanitize_error_text,
)
from nutritrack.ai.failure_classifier import ClassifiedFailure, classify_provider_failure
from nutritrack.ai.provider_factory import ProviderBuilder, ProviderFactory
from nutritrack.ai.provider_health import ProviderHealthTracker
from nutritrack.ai.provider_status import (
    DATA_ONLY_BANNER,
    ProviderStatusSnapshot,
    compute_effective_label,
)
from nutritrack.ai.recovery_manager import RecoveryManager
from nutritrack.ai.retry_policy import RetryPolicy
from nutritrack.state.run_log import RunLog
from nutritrack.tools.base import FUNCTION_INVOCATION_CONFIG


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RoutedNutriTrackChatClient(FunctionInvocationLayer, BaseChatClient):
    """Microsoft Agent Framework `BaseChatClient` + `FunctionInvocationLayer` routed by `AIProviderManager`.

    All five NutriTrack agents share this client so tool invocation loops, shared state, and
    session history remain intact across retries and mid-workflow provider failovers.
    """

    def __init__(self, manager: "AIProviderManager") -> None:
        FunctionInvocationLayer.__init__(
            self,
            function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
        )
        BaseChatClient.__init__(self)
        self.manager = manager

    @property
    def model(self) -> str:
        return self.manager.active_model_name

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        return await self.manager.route_chat_request(
            messages=messages,
            stream=stream,
            options=options,
            **kwargs,
        )


class AIProviderManager:
    """Central coordinator enforcing strict 3-mode AI isolation (`gemini`, `groq`, `offline`)."""

    def __init__(
        self,
        *,
        config: Optional[AIProviderConfig] = None,
        factory: Optional[ProviderFactory] = None,
        run_log: Optional[RunLog] = None,
        clock: Optional[Callable[[], float]] = None,
        async_sleep: Optional[Callable[[float], Any]] = None,
        random_fn: Optional[Callable[[], float]] = None,
        persisted_dir: Optional[Any] = None,
        client_overrides: Optional[dict[str, BaseChatClient | ProviderBuilder]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock or time.time
        self._async_sleep = async_sleep or asyncio.sleep
        self._random_fn = random_fn
        self._persisted_dir = persisted_dir
        self.run_log = run_log
        self._provider_event_log: Optional[RunLog] = (
            RunLog(path=Path(run_log.path).parent / "ai_provider_events.jsonl") if run_log is not None else None
        )

        self.config = config or AIProviderConfig.from_env(persisted_dir=persisted_dir)
        self.data_only_service = DataOnlyService()
        self.factory = factory or ProviderFactory(
            data_only_service=self.data_only_service,
            client_overrides=client_overrides,
        )
        if client_overrides and factory is not None:
            for k, v in client_overrides.items():
                self.factory.set_override(k, v)

        self.retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            base_delay_seconds=self.config.retry_base_delay_seconds,
            max_delay_seconds=self.config.retry_max_delay_seconds,
        )
        self.circuits: dict[str, CircuitBreaker] = {
            "gemini": CircuitBreaker(
                "gemini",
                failure_threshold=self.config.circuit_failure_threshold,
                cooldown_seconds=self.config.circuit_cooldown_seconds,
                clock=self._clock,
            ),
            "groq": CircuitBreaker(
                "groq",
                failure_threshold=self.config.circuit_failure_threshold,
                cooldown_seconds=self.config.circuit_cooldown_seconds,
                clock=self._clock,
            ),
        }
        self.recovery_manager = RecoveryManager(
            health_check_interval_seconds=self.config.health_check_interval_seconds,
            clock=self._clock,
        )
        self.health = ProviderHealthTracker(self.config)
        self.events: list[dict[str, Any]] = []
        self._active_provider: str = self._compute_initial_active_provider()
        register_active_ai_mode(self.config.selected_mode, self._persisted_dir)
        self._routed_client = RoutedNutriTrackChatClient(self)

    @property
    def chat_client(self) -> RoutedNutriTrackChatClient:
        """Return the shared MAF chat client used by all 5 NutriTrack agents."""
        return self._routed_client

    @property
    def selected_mode(self) -> str:
        with self._lock:
            return self.config.selected_mode

    @property
    def active_provider(self) -> str:
        with self._lock:
            return self._active_provider

    @property
    def active_model_name(self) -> str:
        prov = self.active_provider
        if prov == "gemini":
            return self.config.gemini_model
        if prov == "groq":
            return self.config.groq_model
        return self.data_only_service.client.model

    def _get_provider_event_log(self) -> Optional[RunLog]:
        if self.run_log is None:
            return None
        target = Path(self.run_log.path).parent / "ai_provider_events.jsonl"
        if self._provider_event_log is None or self._provider_event_log.path != target:
            self._provider_event_log = RunLog(path=target)
        return self._provider_event_log

    def _log_event(
        self,
        event_type: str,
        *,
        provider: Optional[str] = None,
        detail: Optional[str] = None,
        status: str = "info",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        clean_detail = sanitize_error_text(detail, 280) if detail else None
        with self._lock:
            current_active = self._active_provider
            entry: dict[str, Any] = {
                "timestamp": _now_iso(),
                "event": event_type,
                "mode": self.config.selected_mode,
                "provider": provider,
                "active_provider": current_active,
                "status": status,
                "detail": clean_detail,
            }
            if extra:
                for k, v in extra.items():
                    entry[k] = sanitize_error_text(v) if isinstance(v, str) else v
            self.events.append(entry)
            if len(self.events) > 250:
                self.events = self.events[-250:]
        event_log = self._get_provider_event_log()
        if event_log is not None:
            try:
                event_log.append(
                    run_id="ai-provider-router",
                    user_id="system",
                    agent="AI Provider Manager",
                    task=f"{event_type}:{provider or current_active}",
                    tool=event_type,
                    status="failed" if status in {"failed", "error"} else "success",
                    input_summary=f"mode={self.config.selected_mode}, provider={provider or current_active}",
                    output_summary=clean_detail or event_type,
                    error={"code": event_type, "message": clean_detail} if status in {"failed", "error"} else None,
                    execution_mode="system_routing",
                )
            except Exception:
                pass
        return entry

    def _is_cloud_provider_configured_and_valid(self, provider: str) -> tuple[bool, Optional[ClassifiedFailure]]:
        if provider == "gemini":
            if not self.config.gemini_configured:
                return False, ClassifiedFailure(
                    provider="gemini",
                    category="missing_key",
                    status_code=None,
                    retryable=False,
                    trip_circuit_immediately=True,
                    is_config_error=True,
                    sanitized_reason="GEMINI_API_KEY is not configured.",
                )
            ok_model, model_err = is_model_supported("gemini", self.config.gemini_model)
            if not ok_model:
                return False, ClassifiedFailure(
                    provider="gemini",
                    category="unsupported_model",
                    status_code=None,
                    retryable=False,
                    trip_circuit_immediately=True,
                    is_config_error=True,
                    sanitized_reason=model_err or f"Unsupported Gemini model '{self.config.gemini_model}'.",
                )
            return True, None

        if provider == "groq":
            if not self.config.groq_configured:
                return False, ClassifiedFailure(
                    provider="groq",
                    category="missing_key",
                    status_code=None,
                    retryable=False,
                    trip_circuit_immediately=True,
                    is_config_error=True,
                    sanitized_reason="GROQ_API_KEY is not configured.",
                )
            ok_model, model_err = is_model_supported("groq", self.config.groq_model)
            if not ok_model:
                return False, ClassifiedFailure(
                    provider="groq",
                    category="unsupported_model",
                    status_code=None,
                    retryable=False,
                    trip_circuit_immediately=True,
                    is_config_error=True,
                    sanitized_reason=model_err or f"Unsupported Groq model '{self.config.groq_model}'.",
                )
            return True, None

        return False, None

    def _compute_initial_active_provider(self) -> str:
        mode = self.config.selected_mode
        if mode == "offline":
            return "data_only"

        gem_ok, gem_fail = self._is_cloud_provider_configured_and_valid("gemini")
        if gem_fail and gem_fail.category == "unsupported_model":
            self.circuits["gemini"].record_failure(gem_fail, trip_immediately=True)
            self.health.record_failure("gemini", gem_fail, circuit_opened=True)

        groq_ok, groq_fail = self._is_cloud_provider_configured_and_valid("groq")
        if groq_fail and groq_fail.category == "unsupported_model":
            self.circuits["groq"].record_failure(groq_fail, trip_immediately=True)
            self.health.record_failure("groq", groq_fail, circuit_opened=True)

        if mode == "groq":
            if not groq_ok and groq_fail:
                self.health.set_switch_reason(groq_fail.sanitized_reason)
            else:
                self.health.set_switch_reason(None)
            return "groq"

        if not gem_ok and gem_fail:
            self.health.set_switch_reason(gem_fail.sanitized_reason)
        else:
            self.health.set_switch_reason(None)
        return "gemini"

    def set_mode(self, mode: str, *, persist: bool = True) -> dict[str, Any]:
        """Switch explicitly between `gemini`, `groq`, and `offline` without automatic fallback.

        If `persist=True` and persisting fails, leaves the previous valid mode intact and raises `ProviderConfigError`.
        """
        normalized = normalize_ai_mode(mode)
        with self._lock:
            prev_mode = self.config.selected_mode
            if persist:
                try:
                    save_persisted_mode(normalized, base_dir=self._persisted_dir)
                except Exception as exc:
                    reason = sanitize_error_text(f"Could not save AI mode '{normalized}': {exc}")
                    self._log_event(
                        "mode_persist_failed",
                        provider=self._active_provider,
                        status="failed",
                        detail=reason,
                    )
                    raise ProviderConfigError(
                        reason,
                        provider=normalized,
                        category="config_persist_failed",
                        is_config_error=True,
                    ) from exc

            self.config.selected_mode = normalized
            register_active_ai_mode(normalized, self._persisted_dir)
            self.health.sync_with_config(self.config)
            if normalized == "offline":
                self._active_provider = "data_only"
                self.health.set_switch_reason("Offline Mode selected (deterministic local Python engine).")
            elif normalized == "groq":
                self._active_provider = "groq"
                groq_ok, groq_fail = self._is_cloud_provider_configured_and_valid("groq")
                if not groq_ok and groq_fail:
                    self.health.set_switch_reason(groq_fail.sanitized_reason)
                else:
                    self.health.set_switch_reason(None)
            else:
                self._active_provider = "gemini"
                gem_ok, gem_fail = self._is_cloud_provider_configured_and_valid("gemini")
                if not gem_ok and gem_fail:
                    self.health.set_switch_reason(gem_fail.sanitized_reason)
                else:
                    self.health.set_switch_reason(None)
            self._log_event(
                "mode_changed",
                provider=self._active_provider,
                detail=f"AI mode changed from '{prev_mode}' to '{normalized}'. Active provider: '{self._active_provider}'.",
            )
            return self.get_status()

    def _candidate_chain(self) -> list[str]:
        mode = self.config.selected_mode
        if mode == "offline":
            return ["data_only"]
        if mode == "groq":
            return ["groq"]
        return ["gemini"]

    async def _invoke_cloud_provider(
        self,
        provider: str,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[Optional[ChatResponse], Optional[ClassifiedFailure]]:
        """Attempt to call `provider` (`gemini` or `groq`) with retry policy and circuit breaker protection."""
        ok_cfg, cfg_failure = self._is_cloud_provider_configured_and_valid(provider)
        if not ok_cfg and cfg_failure is not None:
            if cfg_failure.category != "missing_key":
                self.circuits[provider].record_failure(cfg_failure, trip_immediately=True)
                self.health.record_failure(provider, cfg_failure, circuit_opened=True)
                self._log_event(
                    "provider_failure",
                    provider=provider,
                    status="failed",
                    detail=cfg_failure.sanitized_reason,
                    extra={"category": cfg_failure.category},
                )
            return None, cfg_failure

        breaker = self.circuits[provider]
        pre_state = breaker.state
        allowed, allow_reason = breaker.allow_request()
        if not allowed:
            snap = breaker.snapshot()
            skip_failure = ClassifiedFailure(
                provider=provider,
                category=snap.get("last_failure_category") or ("circuit_open" if allow_reason == "circuit_open" else allow_reason),
                status_code=None,
                retryable=False,
                trip_circuit_immediately=False,
                sanitized_reason=(
                    snap.get("last_failure_reason")
                    or f"{provider} circuit breaker is open ({allow_reason})."
                ),
            )
            return None, skip_failure

        is_half_open_probe = allow_reason == "half_open_probe" or pre_state == STATE_HALF_OPEN
        probe_settled = False
        if is_half_open_probe:
            self.recovery_manager.mark_probe_started(provider)
            self.health.set_recovery_attempting(True)
            self._log_event(
                "circuit_half_open",
                provider=provider,
                detail=f"Cooldown expired for {provider}; attempting single HALF_OPEN recovery probe.",
            )

        attempt = 0
        last_failure: Optional[ClassifiedFailure] = None

        try:
            while True:
                if is_turn_cancelled():
                    return None, ClassifiedFailure(
                        provider=provider,
                        category="timeout",
                        sanitized_reason="Turn deadline exceeded or request cancelled.",
                        status_code=504,
                        retryable=False,
                        trip_circuit_immediately=False,
                    )
                attempt += 1
                client = None
                sdk_client = None
                try:
                    client = self.factory.build_client(provider, self.config)
                    sdk_client = getattr(client, "_nutritrack_async_sdk_client", None)
                    self._log_event(
                        "provider_selected",
                        provider=provider,
                        detail=f"Dispatching request to {provider} (attempt {attempt}).",
                    )
                    response = await client._inner_get_response(
                        messages=messages,
                        stream=stream,
                        options=options,
                        **kwargs,
                    )
                    cb_success = breaker.record_success()
                    if is_half_open_probe:
                        probe_settled = True
                        self.recovery_manager.mark_probe_finished()
                        self.health.set_recovery_attempting(False)
                    self.health.record_success(provider)

                    with self._lock:
                        prev_active = self._active_provider
                        self._active_provider = provider

                    if cb_success.get("recovered"):
                        self.health.set_switch_reason(None)
                        self._log_event(
                            "circuit_closed",
                            provider=provider,
                            detail=f"{provider} circuit closed after successful response.",
                        )
                    elif prev_active != provider:
                        self._log_event(
                            "provider_switched",
                            provider=provider,
                            detail=f"Active provider is now '{provider}' (was '{prev_active}').",
                        )
                    return response, None

                except Exception as exc:
                    classified = classify_provider_failure(exc, provider=provider)
                    last_failure = classified
                    self._log_event(
                        "provider_failure",
                        provider=provider,
                        status="failed",
                        detail=f"{provider} attempt {attempt} failed [{classified.category}]: {classified.sanitized_reason}",
                        extra={"category": classified.category, "attempt": attempt},
                    )

                    # In HALF_OPEN probe mode, a single failure immediately reopens the circuit without retrying
                    if is_half_open_probe:
                        cb_res = breaker.record_failure(classified, trip_immediately=True)
                        probe_settled = True
                        self.health.record_failure(provider, classified, circuit_opened=True)
                        self.recovery_manager.mark_probe_finished()
                        self.health.set_recovery_attempting(False)
                        self._log_event(
                            "circuit_opened",
                            provider=provider,
                            status="failed",
                            detail=f"HALF_OPEN recovery probe failed for {provider}; circuit reopened.",
                        )
                        return None, classified

                    decision = self.retry_policy.evaluate(attempt - 1, classified)
                    cb_res = breaker.record_failure(
                        classified,
                        trip_immediately=classified.trip_circuit_immediately
                        or "exceeding max_delay_seconds" in decision.reason,
                    )
                    circuit_is_open = cb_res["current_state"] == STATE_OPEN
                    self.health.record_failure(provider, classified, circuit_opened=circuit_is_open)

                    if cb_res.get("opened_now"):
                        self._log_event(
                            "circuit_opened",
                            provider=provider,
                            status="failed",
                            detail=f"Circuit breaker opened for {provider} after {cb_res['failure_count']} failure(s): {classified.sanitized_reason}",
                        )

                    if decision.should_retry and not circuit_is_open and not is_turn_cancelled():
                        self._log_event(
                            "provider_retry",
                            provider=provider,
                            detail=f"Retrying {provider} in {decision.delay_seconds:.2f}s (attempt {attempt + 1}).",
                            extra={"delay_seconds": decision.delay_seconds},
                        )
                        if decision.delay_seconds > 0:
                            maybe_coro = self._async_sleep(decision.delay_seconds)
                            if asyncio.iscoroutine(maybe_coro):
                                await maybe_coro
                        continue

                    return None, last_failure
                finally:
                    if sdk_client is not None:
                        try:
                            await sdk_client.close()
                        except Exception:
                            pass
        finally:
            if is_half_open_probe and not probe_settled:
                breaker.release_probe()
                self.recovery_manager.mark_probe_finished()
                self.health.set_recovery_attempting(False)

    async def route_chat_request(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute a MAF chat request strictly under an immutable per-request snapshot of `selected_mode`."""
        with self._lock:
            mode = self.config.selected_mode

        with ai_mode_scope(mode):
            # 1. Offline mode: deterministic local Python engine only (zero cloud API calls)
            if mode == "offline":
                with self._lock:
                    self._active_provider = "data_only"
                    self.health.record_success("data_only")
                self._log_event(
                    "provider_selected",
                    provider="data_only",
                    detail="Offline mode active; executing deterministically in local Python engine.",
                )
                return await self.data_only_service.client._inner_get_response(
                    messages=messages,
                    stream=stream,
                    options=options,
                    **kwargs,
                )

            # 2. Strict single-provider execution: never cross-route between Gemini and Groq or auto-switch to Offline
            target_provider = "groq" if mode == "groq" else "gemini"
            response, failure = await self._invoke_cloud_provider(
                target_provider,
                messages=messages,
                stream=stream,
                options=options,
                **kwargs,
            )
            if response is not None:
                return response

            provider_name = "Groq" if target_provider == "groq" else "Google Gemini"
            category = failure.category if failure is not None else "provider_error"
            reason = failure.sanitized_reason if failure is not None else f"{provider_name} is unavailable."
            status_code = failure.status_code if failure is not None else None
            retryable = bool(failure.retryable) if failure is not None else False
            detail_msg = (
                f"{provider_name} request failed ({category}): {reason}. "
                f"Your mode remains '{mode}'. Retry now, check your API key/model settings, "
                "or manually switch to another mode."
            )
            self.health.set_switch_reason(reason)
            with self._lock:
                self._active_provider = target_provider
            self._log_event(
                "provider_failure",
                provider=target_provider,
                status="failed",
                detail=detail_msg,
                extra={"category": category, "status_code": status_code},
            )
            raise ProviderUnavailableError(
                detail_msg,
                provider=target_provider,
                category=category,
                status_code=status_code,
                retryable=retryable,
                reason=reason,
            )

    async def async_probe_providers(self, *, force: bool = True) -> dict[str, Any]:
        """Run a controlled probe strictly against the currently selected cloud provider."""
        with self._lock:
            mode = self.config.selected_mode
        if mode == "offline":
            return self.get_status()

        target_provider = "groq" if mode == "groq" else "gemini"
        probe_messages = [Message("user", ["Reply with OK"])]
        probe_options: dict[str, Any] = {"instructions": "[AGENT:nutrition_manager] Reply with OK."}

        with ai_mode_scope(mode):
            ok_cfg, _ = self._is_cloud_provider_configured_and_valid(target_provider)
            if ok_cfg:
                breaker = self.circuits[target_provider]
                if force:
                    breaker.reset()
                if self.recovery_manager.can_probe(
                    target_provider,
                    selected_mode=mode,
                    circuit_state=breaker.state,
                    force=force,
                ):
                    await self._invoke_cloud_provider(
                        target_provider,
                        messages=probe_messages,
                        stream=False,
                        options=probe_options,
                    )

        with self._lock:
            self._active_provider = target_provider
        return self.get_status()

    def retry_providers(self, *, force: bool = True) -> dict[str, Any]:
        """Synchronous entrypoint for `POST /api/nutritrack/ai/retry` and recovery tests."""
        from nutritrack.agents.runtime import run_async

        return run_async(self.async_probe_providers(force=force))

    def get_status(self) -> dict[str, Any]:
        """Return sanitized `ProviderStatusSnapshot.to_dict()` for Flask APIs and UI."""
        with self._lock:
            self.health.sync_with_config(self.config)
            status_map = self.health.status_map()

            # Reflect live circuit states (e.g., if cooldown expired or circuit opened)
            if self.config.selected_mode != "offline":
                for prov in ("gemini", "groq"):
                    c_snap = self.circuits[prov].snapshot()
                    if c_snap["state"] == STATE_OPEN and status_map.get(prov) not in {
                        "config_error",
                        "quota_exhausted",
                        "rate_limited",
                        "unreachable",
                    }:
                        status_map[prov] = "circuit_open"
                    elif c_snap["state"] == STATE_HALF_OPEN and status_map.get(prov) in {
                        "circuit_open",
                        "rate_limited",
                        "unreachable",
                        "server_error",
                    }:
                        status_map[prov] = "available"

            mode = self.config.selected_mode
            active = "data_only" if mode == "offline" else ("groq" if mode == "groq" else "gemini")
            fallback_active = False

            label = compute_effective_label(
                selected_mode=mode,
                active_provider=active,
                provider_status=status_map,
                recovery_attempting=self.health.recovery_attempting,
            )
            snapshot = ProviderStatusSnapshot(
                selected_mode=mode,
                active_provider=active,
                provider_status=status_map,
                fallback_active=fallback_active,
                last_switch_reason=self.health.last_switch_reason,
                effective_label=label,
                banner=DATA_ONLY_BANNER if mode == "offline" else None,
                last_successful_provider=self.health.last_successful_provider,
                last_successful_request_at=self.health.last_successful_request_at,
                recovery_attempting=self.health.recovery_attempting,
                circuits={k: v.snapshot() for k, v in self.circuits.items()},
                models={
                    "gemini": self.config.gemini_model,
                    "groq": self.config.groq_model,
                    "data_only": self.data_only_service.client.model,
                },
            )
            return snapshot.to_dict()

    def get_providers_diagnostics(self) -> dict[str, Any]:
        """Return detailed diagnostics for `GET /api/nutritrack/ai/providers` (zero secrets)."""
        status = self.get_status()
        details = self.health.details_map()
        return {
            "selected_mode": status["selected_mode"],
            "active_provider": status["active_provider"],
            "effective_label": status["effective_label"],
            "fallback_active": False,
            "last_switch_reason": status["last_switch_reason"],
            "banner": status["banner"],
            "providers": {
                "gemini": {
                    "role": "manual_provider",
                    "name": "Google Gemini API",
                    "configured": self.config.gemini_configured,
                    "model": self.config.gemini_model,
                    "status": status["provider_status"].get("gemini"),
                    "circuit": status["circuits"].get("gemini"),
                    "health": details.get("gemini", {}),
                },
                "groq": {
                    "role": "manual_provider",
                    "name": "Groq API",
                    "configured": self.config.groq_configured,
                    "model": self.config.groq_model,
                    "status": status["provider_status"].get("groq"),
                    "circuit": status["circuits"].get("groq"),
                    "health": details.get("groq", {}),
                },
                "data_only": {
                    "role": "offline_deterministic",
                    "name": "Offline Deterministic Mode",
                    "configured": True,
                    "model": self.data_only_service.client.model,
                    "status": status["provider_status"].get("data_only"),
                    "capabilities": self.data_only_service.describe_capabilities(),
                },
            },
            "recent_events": list(self.events[-20:]),
        }
