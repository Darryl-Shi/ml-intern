from agent.core.message import message_from_mapping


def test_message_from_mapping_ignores_provider_specific_fields():
    message = message_from_mapping(
        {
            "role": "assistant",
            "content": "hello",
            "provider_specific_fields": {"ignored": True},
            "function_call": None,
        }
    )

    assert message.role == "assistant"
    assert message.content == "hello"
