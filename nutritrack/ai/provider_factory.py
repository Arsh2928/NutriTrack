"""Factory for creating Microsoft Agent Framework chat clients for Gemini, Groq, and Data-only mode."""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

from nutritrack.ai.config import (
    AIProviderConfig,
    assert_network_allowed,
    is_model_supported,
    is_turn_cancelled,
    remaining_turn_timeout,
)
from nutritrack.ai.data_only_service import DataOnlyService
from nutritrack.ai.exceptions import ProviderConfigError
from nutritrack.tools.base import FUNCTION_INVOCATION_CONFIG

ProviderBuilder = Callable[[AIProviderConfig], BaseChatClient]

# Cache for Gemini 2.5+/3.x tool_call extra_content (e.g., thought_signature) across multi-turn tool loops
_CACHE_LOCK = threading.RLock()
_THOUGHT_SIGNATURES: dict[str, dict[str, Any]] = {}


def _wrap_sdk_chat_completions(async_sdk_client: AsyncOpenAI, provider: str, configured_model: str) -> None:
    """Wrap `async_sdk_client.chat.completions.create` to preserve Gemini thought_signature without changing models."""
    orig_create = async_sdk_client.chat.completions.create

    async def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # Enforce cancellation/deadline before dispatching an outbound request.
        if is_turn_cancelled():
            raise ProviderConfigError(
                "Turn deadline exceeded or request cancelled.",
                provider=provider,
                category="turn_timeout",
                is_config_error=False,
            )
        # Centralized network-access policy guard: block any unpermitted provider or offline outbound AI call.
        assert_network_allowed("ai", provider=provider)
        # Enforce the explicitly configured model; never rotate or auto-discover models.
        if configured_model:
            kwargs["model"] = configured_model
        # Clamp per-call socket timeout to remaining server turn deadline when scoped.
        base_timeout = kwargs.get("timeout")
        if isinstance(base_timeout, (int, float)) and float(base_timeout) > 0:
            kwargs["timeout"] = remaining_turn_timeout(float(base_timeout))
        else:
            clamped = remaining_turn_timeout(300.0)
            if clamped < 300.0:
                kwargs["timeout"] = clamped

        # 1. Inject cached extra_content (thought_signature) into outgoing Gemini assistant tool_calls
        if provider == "gemini":
            msgs = kwargs.get("messages")
            if isinstance(msgs, list):
                with _CACHE_LOCK:
                    sig_snapshot = dict(_THOUGHT_SIGNATURES)
                for msg in msgs:
                    if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
                        for tc_dict in msg["tool_calls"]:
                            if isinstance(tc_dict, dict):
                                tc_id = tc_dict.get("id")
                                if tc_id and tc_id in sig_snapshot and "extra_content" not in tc_dict:
                                    tc_dict["extra_content"] = sig_snapshot[tc_id]

        resp = await orig_create(*args, **kwargs)

        # 2. Record any extra_content (such as Gemini thought_signature) from returned tool_calls
        if provider == "gemini":
            with _CACHE_LOCK:
                for choice in getattr(resp, "choices", None) or []:
                    msg_obj = getattr(choice, "message", None)
                    for tc in getattr(msg_obj, "tool_calls", None) or []:
                        tc_id = getattr(tc, "id", None)
                        extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get(
                            "extra_content"
                        )
                        if tc_id and isinstance(extra, dict) and extra:
                            if len(_THOUGHT_SIGNATURES) > 512:
                                _THOUGHT_SIGNATURES.clear()
                            _THOUGHT_SIGNATURES[tc_id] = extra
        return resp

    setattr(async_sdk_client.chat.completions, "create", _wrapped_create)


def _wrap_live_maf_client(
    client: OpenAIChatCompletionClient,
    provider: str,
    data_only_client: BaseChatClient,
) -> None:
    """Execute intermediate NutriTrack tool steps deterministically and invoke the cloud LLM once for final synthesis."""
    from agent_framework import ChatResponse, Content, Message
    from nutritrack.agents.runtime import _extract_agent_key, _latest_user_text

    orig_inner = client._inner_get_response

    async def _grounded_inner(
        *,
        messages: Any,
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        if is_turn_cancelled():
            raise ProviderConfigError(
                "Turn deadline exceeded or request cancelled.",
                provider=provider,
                category="turn_timeout",
                is_config_error=False,
            )
        assert_network_allowed("ai", provider=provider)

        agent_key = _extract_agent_key(options or {})
        user_text = _latest_user_text(messages or [])
        if not agent_key or user_text.strip() == "Reply with OK":
            return await orig_inner(messages=messages, stream=stream, options=options, **kwargs)

        det_resp = await data_only_client._inner_get_response(
            messages=messages,
            stream=stream,
            options=options,
            **kwargs,
        )
        has_tool_call = any(
            getattr(c, "type", None) == "function_call"
            for m in (getattr(det_resp, "messages", None) or [])
            for c in (getattr(m, "contents", None) or [])
        )
        if has_tool_call or agent_key != "nutrition_manager":
            return det_resp

        det_text = "\n".join(
            getattr(c, "text", "")
            for m in (getattr(det_resp, "messages", None) or [])
            for c in (getattr(m, "contents", None) or [])
            if getattr(c, "type", None) == "text" and getattr(c, "text", "")
        ).strip()

        synth_prompt = (
            f"User request: {user_text}\n\n"
            f"Verified NutriTrack tool results (preserve all numbers, food names, and questions accurately):\n"
            f"{det_text or 'No additional tool output.'}\n\n"
            "Write a concise, helpful response grounded strictly in the verified results above."
        )
        synth_options = {
            k: v
            for k, v in (options or {}).items()
            if k not in ("tools", "tool_choice")
        }
        synth_options["instructions"] = (
            "You are NutriTrack, a nutrition assistant. Respond concisely in 2-4 sentences using ONLY "
            "the verified NutriTrack tool results provided. Never invent calories, macros, or foods."
        )
        cloud_resp = await orig_inner(
            messages=[Message("user", [synth_prompt])],
            stream=stream,
            options=synth_options,
        )
        cloud_text = "\n".join(
            getattr(c, "text", "")
            for m in (getattr(cloud_resp, "messages", None) or [])
            for c in (getattr(m, "contents", None) or [])
            if getattr(c, "type", None) == "text" and getattr(c, "text", "")
        ).strip()
        if not cloud_text:
            return det_resp
        if det_text and ("Week-over-week comparison" in det_text or "?" in det_text) and det_text not in cloud_text:
            return ChatResponse(messages=[Message("assistant", [Content.from_text(f"{cloud_text}\n\n{det_text}")])])
        return cloud_resp

    setattr(client, "_inner_get_response", _grounded_inner)


class ProviderFactory:
    """Creates MAF-compatible chat clients for `gemini`, `groq`, and `data_only`."""

    def __init__(
        self,
        *,
        data_only_service: Optional[DataOnlyService] = None,
        client_overrides: Optional[dict[str, BaseChatClient | ProviderBuilder]] = None,
    ) -> None:
        self.data_only_service = data_only_service or DataOnlyService()
        self._overrides: dict[str, BaseChatClient | ProviderBuilder] = dict(client_overrides or {})

    def set_override(self, provider: str, client_or_builder: Optional[BaseChatClient | ProviderBuilder]) -> None:
        """Register or clear a mock/test client override for `gemini`, `groq`, or `data_only`."""
        if client_or_builder is None:
            self._overrides.pop(provider, None)
        else:
            self._overrides[provider] = client_or_builder

    def build_client(self, provider: str, config: AIProviderConfig) -> BaseChatClient:
        """Return a configured MAF `BaseChatClient` for `provider` (`gemini`, `groq`, or `data_only`).

        Note: Live `AsyncOpenAI` clients are created fresh per invocation on the active asyncio loop
        and closed in `finally:` so `httpx.AsyncClient` never references a closed event loop.
        """
        prov = provider.strip().lower()
        if prov == "data_only":
            if "data_only" in self._overrides:
                ov = self._overrides["data_only"]
                return ov(config) if callable(ov) and not isinstance(ov, BaseChatClient) else ov
            return self.data_only_service.client

        if prov == "gemini":
            if not config.gemini_configured:
                raise ProviderConfigError(
                    "GEMINI_API_KEY is not configured.",
                    provider="gemini",
                    category="missing_key",
                    is_config_error=True,
                )
            ok_model, model_err = is_model_supported("gemini", config.gemini_model)
            if not ok_model:
                raise ProviderConfigError(
                    model_err or f"Unsupported Gemini model '{config.gemini_model}'.",
                    provider="gemini",
                    category="unsupported_model",
                    is_config_error=True,
                )
            if "gemini" in self._overrides:
                ov = self._overrides["gemini"]
                return ov(config) if callable(ov) and not isinstance(ov, BaseChatClient) else ov

            async_sdk_client = AsyncOpenAI(
                api_key=config.gemini_api_key,
                base_url=config.gemini_base_url,
                timeout=config.request_timeout_seconds,
                max_retries=0,  # Disable SDK internal retries; RetryPolicy owns retry loop
            )
            _wrap_sdk_chat_completions(async_sdk_client, "gemini", config.gemini_model)
            client = OpenAIChatCompletionClient(
                model=config.gemini_model,
                async_client=async_sdk_client,
                function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
            )
            _wrap_live_maf_client(client, "gemini", self.data_only_service.client)
            setattr(client, "_nutritrack_async_sdk_client", async_sdk_client)
            return client

        if prov == "groq":
            if not config.groq_configured:
                raise ProviderConfigError(
                    "GROQ_API_KEY is not configured.",
                    provider="groq",
                    category="missing_key",
                    is_config_error=True,
                )
            ok_model, model_err = is_model_supported("groq", config.groq_model)
            if not ok_model:
                raise ProviderConfigError(
                    model_err or f"Unsupported Groq model '{config.groq_model}'.",
                    provider="groq",
                    category="unsupported_model",
                    is_config_error=True,
                )
            if "groq" in self._overrides:
                ov = self._overrides["groq"]
                return ov(config) if callable(ov) and not isinstance(ov, BaseChatClient) else ov

            async_sdk_client = AsyncOpenAI(
                api_key=config.groq_api_key,
                base_url=config.groq_base_url,
                timeout=config.request_timeout_seconds,
                max_retries=0,  # Disable SDK internal retries; RetryPolicy owns retry loop
            )
            _wrap_sdk_chat_completions(async_sdk_client, "groq", config.groq_model)
            client = OpenAIChatCompletionClient(
                model=config.groq_model,
                async_client=async_sdk_client,
                function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
            )
            _wrap_live_maf_client(client, "groq", self.data_only_service.client)
            setattr(client, "_nutritrack_async_sdk_client", async_sdk_client)
            return client

        raise ProviderConfigError(
            f"Unknown provider '{provider}'. Supported providers: gemini, groq, data_only.",
            provider=prov,
            category="invalid_provider",
            is_config_error=True,
        )
