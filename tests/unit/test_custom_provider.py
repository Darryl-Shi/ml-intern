"""Custom provider support stays session-scoped and user-billed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent.config import Config
from agent.core.llm_params import _resolve_llm_params
from agent.core.session import Session

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import routes.agent as agent_routes  # noqa: E402
import user_quotas  # noqa: E402


def test_custom_provider_resolves_to_openai_compatible_params():
    params = _resolve_llm_params(
        "ignored-display-model",
        reasoning_effort="high",
        custom_provider={
            "model": "custom-model",
            "base_url": "https://llm.example.test/v1",
            "api_key": "sk-test",
        },
    )

    assert params == {
        "model": "openai/custom-model",
        "api_base": "https://llm.example.test/v1",
        "api_key": "sk-test",
        "reasoning_effort": "high",
    }


def test_unknown_plain_model_is_still_rejected():
    with pytest.raises(HTTPException) as exc:
        agent_routes._parse_model_selection(
            {"model": "not-a-listed/model"},
            require_selection=True,
        )
    assert exc.value.status_code == 400
    assert "Unknown model" in exc.value.detail


def test_valid_custom_provider_selection_is_accepted():
    model, custom = agent_routes._parse_model_selection(
        {
            "custom_provider": {
                "model": "my-model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "secret",
                "label": "Local",
            }
        },
        require_selection=True,
    )

    assert model == "my-model"
    assert custom == {
        "model": "my-model",
        "base_url": "http://localhost:8000/v1",
        "api_key": "secret",
        "label": "Local",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "", "base_url": "http://localhost:8000/v1", "api_key": "secret"},
        {"model": "m", "base_url": "", "api_key": "secret"},
        {"model": "m", "base_url": "localhost:8000/v1", "api_key": "secret"},
        {"model": "m", "base_url": "http://localhost:8000/v1", "api_key": ""},
    ],
)
def test_invalid_custom_provider_selection_is_rejected(payload):
    with pytest.raises(HTTPException) as exc:
        agent_routes._parse_model_selection(
            {"custom_provider": payload},
            require_selection=True,
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_custom_provider_does_not_charge_claude_quota():
    user_quotas._reset_for_tests()
    agent_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            custom_provider={
                "model": "anthropic-compatible-name",
                "base_url": "https://llm.example.test/v1",
                "api_key": "secret",
            },
            config=SimpleNamespace(model_name="anthropic-compatible-name"),
        ),
    )

    await agent_routes._enforce_claude_quota({"user_id": "u1"}, agent_session)

    assert await user_quotas.get_claude_used_today("u1") == 0
    assert agent_session.claude_counted is False


def test_custom_provider_key_is_not_in_session_trajectory():
    api_key = "sk-custom-provider-key-123456789"
    session = Session(
        event_queue=None,
        config=Config(model_name="custom-model"),
        custom_provider={
            "model": "custom-model",
            "base_url": "https://llm.example.test/v1",
            "api_key": api_key,
        },
    )

    trajectory = session.get_trajectory()

    assert trajectory["model_name"] == "custom-model"
    assert "custom_provider" not in trajectory
    assert api_key not in str(trajectory)
