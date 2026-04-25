from types import SimpleNamespace

import pytest

from agent.core import provider as provider_mod
from agent.core.provider import (
    ProviderConfig,
    apply_provider_to_config,
    env_provider_config,
    load_saved_provider_config,
    provider_from_request,
    resolve_provider_config,
    save_provider_config,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_CONTEXT_WINDOW",
    ):
        monkeypatch.delenv(key, raising=False)


def test_env_provider_config_requires_complete_triplet(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "model-a")
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "12345")

    cfg = env_provider_config()

    assert cfg is not None
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "model-a"
    assert cfg.context_window == 12345


def test_save_and_load_provider_config_uses_private_file(tmp_path):
    path = tmp_path / "provider.json"
    cfg = ProviderConfig(
        model="model-a",
        base_url="https://example.test/v1",
        api_key="sk-test",
        context_window=100000,
    )

    save_provider_config(cfg, path)
    loaded = load_saved_provider_config(path)

    assert loaded == cfg
    assert path.stat().st_mode & 0o777 == 0o600


def test_resolve_provider_prefers_env_then_runtime_config(monkeypatch, tmp_path):
    saved = tmp_path / "provider.json"
    save_provider_config(
        ProviderConfig("saved-model", "https://saved.test/v1", "saved-key"),
        saved,
    )
    monkeypatch.setattr(provider_mod, "PROVIDER_CONFIG_PATH", saved)

    runtime = SimpleNamespace(
        model_name="runtime-model",
        openai_base_url="https://runtime.test/v1",
        openai_api_key="runtime-key",
        openai_context_window=64000,
    )
    assert resolve_provider_config(runtime).model == "runtime-model"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert resolve_provider_config(runtime).model == "env-model"


def test_apply_provider_and_request_parsing():
    runtime = SimpleNamespace()
    cfg = provider_from_request(
        {
            "provider": {
                "model": "model-a",
                "base_url": "https://example.test/v1/",
                "api_key": "sk-test",
                "context_window": 32000,
            }
        }
    )

    assert cfg is not None
    assert cfg.base_url == "https://example.test/v1"

    apply_provider_to_config(runtime, cfg)

    assert runtime.model_name == "model-a"
    assert runtime.openai_base_url == "https://example.test/v1"
    assert runtime.openai_api_key == "sk-test"
    assert runtime.openai_context_window == 32000
