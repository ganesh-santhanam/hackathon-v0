import pytest

from industrial_ai.security.validation import ensure_path_within_base, safe_upload_path


def test_safe_upload_path_sanitizes_name_and_stays_in_base(tmp_path):
    path = safe_upload_path(
        base_dir=tmp_path,
        machine_id="../../pump-1",
        category="../cable",
        uploaded_filename="../../image.exe",
        allowed_suffixes={".png", ".jpg"},
    )

    assert path.parent == tmp_path.resolve()
    assert path.name == "pump-1-cable.png"


def test_ensure_path_within_base_rejects_traversal(tmp_path):
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(ValueError, match="outside allowed directory"):
        ensure_path_within_base(outside, tmp_path)
