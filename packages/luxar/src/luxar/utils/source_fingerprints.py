"""Stable fingerprints for Python sources that produce Luxar stores."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable


def fingerprint_source_files(root: Path, paths: Iterable[Path]) -> str:
    """Hash source paths and contents in a stable, boundary-safe order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def production_source_fingerprint(package_root: Path | None = None) -> str:
    """Hash production Luxar Python sources once per package root and process."""
    root = (
        Path(__file__).resolve().parents[1]
        if package_root is None
        else Path(package_root).resolve()
    )
    paths = (
        path
        for path in root.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )
    try:
        return fingerprint_source_files(root, paths)
    except OSError:
        return ""
