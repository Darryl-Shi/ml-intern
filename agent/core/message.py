"""Small OpenAI-compatible message/tool-call models used internally.

These classes keep the attribute access patterns the agent loop expects while
serializing to plain OpenAI chat-completions dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FunctionCall:
    name: str = ""
    arguments: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: FunctionCall | dict[str, Any] | Any = None

    def __post_init__(self) -> None:
        if self.function is None:
            self.function = FunctionCall()
        elif isinstance(self.function, dict):
            self.function = FunctionCall(
                name=self.function.get("name", ""),
                arguments=self.function.get("arguments", ""),
            )

    def model_dump(self) -> dict[str, Any]:
        fn = (
            self.function.model_dump()
            if hasattr(self.function, "model_dump")
            else {
                "name": getattr(self.function, "name", ""),
                "arguments": getattr(self.function, "arguments", ""),
            }
        )
        return {"id": self.id, "type": self.type, "function": fn}


@dataclass
class Message:
    role: str
    content: Any = None
    tool_calls: list[ToolCall | dict[str, Any] | Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.tool_calls:
            self.tool_calls = [normalize_tool_call(tc) for tc in self.tool_calls]

    def model_dump(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [normalize_tool_call(tc).model_dump() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data


def normalize_tool_call(raw: ToolCall | dict[str, Any] | Any) -> ToolCall:
    if isinstance(raw, ToolCall):
        return raw
    if isinstance(raw, dict):
        return ToolCall(
            id=raw.get("id", ""),
            type=raw.get("type", "function"),
            function=raw.get("function") or {},
        )
    return ToolCall(
        id=getattr(raw, "id", ""),
        type=getattr(raw, "type", "function"),
        function=getattr(raw, "function", None),
    )


def message_from_mapping(raw: dict[str, Any]) -> Message:
    """Load a persisted message while ignoring provider-specific extras."""
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    data = {key: raw[key] for key in allowed if key in raw}
    if "role" not in data:
        raise ValueError("Persisted message is missing role")
    return Message(**data)


def to_openai_message(message: Message | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(message, dict):
        data = dict(message)
    elif hasattr(message, "model_dump"):
        data = message.model_dump()
    else:
        data = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }

    if data.get("tool_calls"):
        data["tool_calls"] = [
            normalize_tool_call(tc).model_dump() for tc in data["tool_calls"]
        ]

    # OpenAI accepts assistant content null when tool_calls are present, but
    # user/system/tool messages should carry a string.
    if data.get("content") is None and not data.get("tool_calls"):
        data["content"] = ""

    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return {k: v for k, v in data.items() if k in allowed and v is not None}


def rough_token_count(messages: list[Message | dict[str, Any] | Any]) -> int:
    chars = 0
    for message in messages:
        data = to_openai_message(message)
        chars += len(str(data.get("content") or ""))
        if data.get("tool_calls"):
            chars += len(str(data["tool_calls"]))
    return max(1, chars // 4)
