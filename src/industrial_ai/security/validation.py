import re
from pathlib import Path


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename_part(value: str, fallback: str) -> str:
    cleaned = SAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    return cleaned or fallback


def ensure_path_within_base(path: Path, base_dir: Path) -> Path:
    resolved_base = base_dir.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise ValueError(f"Path is outside allowed directory: {path}")
    return resolved_path


def safe_upload_path(
    base_dir: Path,
    machine_id: str,
    category: str,
    uploaded_filename: str,
    allowed_suffixes: set[str],
    default_suffix: str = ".png",
) -> Path:
    suffix = Path(uploaded_filename).suffix.lower()
    if suffix not in allowed_suffixes:
        suffix = default_suffix
    filename = (
        f"{safe_filename_part(machine_id, 'machine')}-"
        f"{safe_filename_part(category, 'category')}{suffix}"
    )
    return ensure_path_within_base(base_dir / filename, base_dir)
