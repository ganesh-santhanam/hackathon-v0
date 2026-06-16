import pytest

from industrial_ai.config.settings import DEFAULT_OLLAMA_MODEL, AppSettings, load_settings


def test_settings_defaults_are_local_demo_friendly(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_GENERATE_URL", raising=False)

    settings = load_settings()

    assert settings.environment == "demo"
    assert settings.ollama_model == DEFAULT_OLLAMA_MODEL
    assert settings.ollama_generate_url == "http://localhost:11434/api/generate"
    assert settings.production_mode is False


def test_settings_builds_ollama_generate_url_from_base(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.local:11434/")
    monkeypatch.delenv("OLLAMA_GENERATE_URL", raising=False)

    settings = load_settings()

    assert settings.ollama_base_url == "http://ollama.local:11434"
    assert settings.ollama_generate_url == "http://ollama.local:11434/api/generate"


def test_production_validation_rejects_placeholder_secrets(monkeypatch):
    monkeypatch.setenv("INDUSTRIAL_AI_PRODUCTION", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "changeme")

    settings = AppSettings.from_env()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate_for_runtime()
