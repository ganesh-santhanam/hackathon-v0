import logging

from industrial_ai.config.settings import load_settings
from industrial_ai.security.secrets import redact_text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return redact_text(message)


def configure_logging(level: str | None = None) -> None:
    settings = load_settings()
    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(RedactingFormatter(handler.formatter._fmt if handler.formatter else None))
