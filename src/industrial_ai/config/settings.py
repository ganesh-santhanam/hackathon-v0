from dataclasses import dataclass
import os
from pathlib import Path

from industrial_ai.paths import PROJECT_ROOT
from industrial_ai.security.secrets import find_placeholder_secrets


DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@dataclass(frozen=True)
class AppSettings:
    environment: str
    project_root: Path
    data_dir: Path
    models_dir: Path
    qdrant_url: str
    ollama_model: str
    ollama_base_url: str
    ollama_generate_url: str
    log_level: str
    production_mode: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        environment = os.environ.get("INDUSTRIAL_AI_ENV", "demo").strip().lower() or "demo"
        ollama_base_url = _normalize_base_url(os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        return cls(
            environment=environment,
            project_root=_env_path("INDUSTRIAL_AI_PROJECT_ROOT", PROJECT_ROOT),
            data_dir=_env_path("INDUSTRIAL_AI_DATA_DIR", PROJECT_ROOT / "data"),
            models_dir=_env_path("INDUSTRIAL_AI_MODELS_DIR", PROJECT_ROOT / "models"),
            qdrant_url=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL),
            ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_base_url=ollama_base_url,
            ollama_generate_url=os.environ.get(
                "OLLAMA_GENERATE_URL",
                f"{ollama_base_url}/api/generate",
            ),
            log_level=os.environ.get("INDUSTRIAL_AI_LOG_LEVEL", "INFO").upper(),
            production_mode=_env_bool("INDUSTRIAL_AI_PRODUCTION", default=environment == "production"),
        )

    def validate_for_runtime(self) -> None:
        if not self.production_mode:
            return
        placeholders = find_placeholder_secrets(os.environ)
        if placeholders:
            names = ", ".join(sorted(placeholders))
            raise ValueError(f"Production mode refuses placeholder secret values: {names}")


def load_settings(validate: bool = False) -> AppSettings:
    settings = AppSettings.from_env()
    if validate:
        settings.validate_for_runtime()
    return settings
