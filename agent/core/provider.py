"""OpenAI-compatible provider configuration and chat client."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agent.core.message import to_openai_message

PROVIDER_CONFIG_PATH = Path.home() / ".config" / "ml-intern" / "provider.json"
DEFAULT_CONTEXT_WINDOW = 200_000


@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str
    context_window: int = DEFAULT_CONTEXT_WINDOW

    def redacted(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "context_window": self.context_window,
        }


class MissingProviderConfig(RuntimeError):
    pass


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _from_mapping(data: dict[str, Any] | None) -> ProviderConfig | None:
    if not data:
        return None
    model = _clean(str(data.get("model") or data.get("model_name") or ""))
    base_url = _clean(str(data.get("base_url") or data.get("openai_base_url") or ""))
    api_key = _clean(str(data.get("api_key") or data.get("openai_api_key") or ""))
    if not (model and base_url and api_key):
        return None
    context_window_raw = data.get("context_window") or data.get("openai_context_window")
    try:
        context_window = int(context_window_raw) if context_window_raw else DEFAULT_CONTEXT_WINDOW
    except (TypeError, ValueError):
        context_window = DEFAULT_CONTEXT_WINDOW
    return ProviderConfig(
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        context_window=max(1_000, context_window),
    )


def env_provider_config() -> ProviderConfig | None:
    return _from_mapping(
        {
            "model": os.environ.get("OPENAI_MODEL"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "context_window": os.environ.get("OPENAI_CONTEXT_WINDOW"),
        }
    )


def load_saved_provider_config(path: Path | None = None) -> ProviderConfig | None:
    path = path or PROVIDER_CONFIG_PATH
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return _from_mapping(data if isinstance(data, dict) else None)


def save_provider_config(config: ProviderConfig, path: Path | None = None) -> None:
    path = path or PROVIDER_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": config.model,
                "base_url": config.base_url,
                "api_key": config.api_key,
                "context_window": config.context_window,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        path.chmod(0o600)
    except OSError:
        pass


def resolve_provider_config(config: Any | None = None) -> ProviderConfig | None:
    env = env_provider_config()
    if env:
        return env
    if config is not None:
        cfg = _from_mapping(
            {
                "model": getattr(config, "model_name", None),
                "base_url": getattr(config, "openai_base_url", None),
                "api_key": getattr(config, "openai_api_key", None),
                "context_window": getattr(config, "openai_context_window", None),
            }
        )
        if cfg:
            return cfg
    return load_saved_provider_config()


def require_provider_config(config: Any | None = None) -> ProviderConfig:
    provider = resolve_provider_config(config)
    if provider is None:
        raise MissingProviderConfig(
            "No OpenAI-compatible provider configured. Run /provider setup "
            "or set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL."
        )
    return provider


def apply_provider_to_config(config: Any, provider: ProviderConfig) -> None:
    config.model_name = provider.model
    config.openai_base_url = provider.base_url
    config.openai_api_key = provider.api_key
    config.openai_context_window = provider.context_window


def provider_from_request(raw: dict[str, Any] | None) -> ProviderConfig | None:
    if not isinstance(raw, dict):
        return None
    provider = raw.get("provider") or raw.get("openai_provider")
    if isinstance(provider, dict):
        return _from_mapping(provider)
    return _from_mapping(raw)


def _usage_total(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0)


def build_client(provider: ProviderConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key)


async def chat_completion(
    *,
    provider: ProviderConfig,
    messages: list[Any],
    tools: list[dict] | None = None,
    stream: bool = False,
    timeout: float | None = None,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> Any:
    client = build_client(provider)
    request: dict[str, Any] = {
        "model": provider.model,
        "messages": [to_openai_message(m) for m in messages],
        "stream": stream,
        **kwargs,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = kwargs.get("tool_choice", "auto")
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort

    # Some OpenAI-compatible servers reject reasoning_effort. Retry once
    # without it so generic local providers work out of the box.
    try:
        return await asyncio.wait_for(
            client.chat.completions.create(**request),
            timeout=timeout,
        )
    except Exception as e:
        if not reasoning_effort or "reasoning" not in str(e).lower():
            raise
        request.pop("reasoning_effort", None)
        return await asyncio.wait_for(
            client.chat.completions.create(**request),
            timeout=timeout,
        )


def completion_usage_total(response: Any) -> int:
    return _usage_total(response)
