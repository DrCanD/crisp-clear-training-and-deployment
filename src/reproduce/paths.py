"""Resolve project paths independently of the caller's working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    """Resolve a relative configuration path, or an internal in-project Path."""
    path = Path(value)
    if path.is_absolute() and not isinstance(value, Path):
        raise ValueError("Use a path relative to the repository root")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Project paths must remain inside the repository")
    return resolved
