import re
from collections.abc import Mapping


REDACTION = "[REDACTED]"
SECRET_FIELD_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization|bearer)\b\s*[:=]\s*['\"]?([^'\"\s,;]+)"
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(hf_[A-Za-z0-9]{12,})\b"),
)
PLACEHOLDER_VALUES = {
    "",
    "changeme",
    "change-me",
    "placeholder",
    "replace-me",
    "replace_me",
    "todo",
    "none",
    "null",
    "your-api-key",
    "your_token_here",
}
SECRET_ENV_NAME_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)")


def redact_text(value: object) -> str:
    text = "" if value is None else str(value)
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTION, text)
    text = SECRET_FIELD_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTION}", text)
    return text


def find_placeholder_secrets(values: Mapping[str, str]) -> list[str]:
    placeholders = []
    for name, value in values.items():
        if not SECRET_ENV_NAME_PATTERN.search(name):
            continue
        if value.strip().lower() in PLACEHOLDER_VALUES:
            placeholders.append(name)
    return placeholders
