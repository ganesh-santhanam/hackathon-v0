from industrial_ai.security.secrets import find_placeholder_secrets, redact_text


def test_redact_text_masks_common_secret_shapes():
    text = (
        "OPENAI_API_KEY=sk-exampleSecretValue "
        "Authorization: Bearer abc123 "
        "HF_TOKEN=hf_abcdefghijklmnop"
    )

    redacted = redact_text(text)

    assert "sk-exampleSecretValue" not in redacted
    assert "abc123" not in redacted
    assert "hf_abcdefghijklmnop" not in redacted
    assert "[REDACTED]" in redacted


def test_find_placeholder_secrets_only_flags_secret_env_names():
    placeholders = find_placeholder_secrets(
        {
            "OPENAI_API_KEY": "changeme",
            "NORMAL_SETTING": "changeme",
            "SERVICE_TOKEN": "real-token-value",
        }
    )

    assert placeholders == ["OPENAI_API_KEY"]
