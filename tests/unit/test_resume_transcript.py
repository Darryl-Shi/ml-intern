from agent.core.message import Message, ToolCall
from agent.main import _resume_transcript_items


def test_resume_transcript_items_show_visible_chat_and_compact_tools():
    messages = [
        Message(role="system", content="hidden"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="I'll check.", tool_calls=[
            ToolCall(id="call_1", function={"name": "read", "arguments": "{}"})
        ]),
        Message(role="tool", content="line one\nline two", tool_call_id="call_1"),
        Message(role="user", content="[SYSTEM: internal hint]"),
    ]

    assert _resume_transcript_items(messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I'll check."},
        {"role": "tool_call", "content": "read"},
        {"role": "tool", "content": "line one", "name": "call_1"},
    ]
