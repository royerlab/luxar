"""Stable fingerprints for Python sources that produce Luxar stores."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numcodecs
import zarr

ZARR_FORMAT_ENV_VAR = "LUXAR_ZARR_FORMAT"


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


def fingerprint_production_sources(package_root: Path | None = None) -> str:
    """Hash production Luxar Python sources under one package root.

    This deliberately includes ``demos/``, so editing any demo invalidates every
    fingerprinted demo scene. Shared helpers such as ``_globe_common``,
    ``_lod_policy``, and ``_cinematic_camera`` write scene bytes, and some demos
    import other demo modules, so excluding that tree would leave stale scenes.
    """
    root = (
        Path(__file__).resolve().parents[1]
        if package_root is None
        else Path(package_root).resolve()
    )
    paths = tuple(
        path
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
        and path.name != "conftest.py"
    )
    if not paths:
        return ""
    try:
        return fingerprint_source_files(root, paths)
    except OSError:
        return ""


@lru_cache(maxsize=None)
def production_source_fingerprint(package_root: Path | None = None) -> str:
    """Hash production sources once per package root and process."""
    return fingerprint_production_sources(package_root)


def store_writer_environment() -> dict[str, str | None]:
    """Return installed and configured inputs that affect Zarr output."""
    return {
        ZARR_FORMAT_ENV_VAR: os.environ.get(ZARR_FORMAT_ENV_VAR),
        "numcodecs": numcodecs.__version__,
        "zarr": zarr.__version__,
    }
