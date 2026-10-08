"""Repository-relative paths and input hashes without numerical dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repository_path(value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Expected a repository-relative path')
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Path escapes the repository')
    return resolved


def check_hashes(directory):
    expected = json.loads((directory / 'sha256.json').read_text())
    for name, digest in expected.items():
        if Path(name).name != name:
            raise ValueError('Hash manifest contains a non-local filename')
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Fixture hash mismatch: {name}')
    return len(expected)
