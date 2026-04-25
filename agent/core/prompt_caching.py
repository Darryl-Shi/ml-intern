"""Prompt caching hook.

OpenAI-compatible providers do not share a standard cache-control extension,
so this is currently a no-op kept to preserve call-site structure.
"""

from typing import Any


def with_prompt_caching(
    messages: list[Any],
    tools: list[dict] | None,
    model_name: str | None,
) -> tuple[list[Any], list[dict] | None]:
    """Return messages/tools unchanged."""
    _ = model_name
    return messages, tools
