"""Path utilities for Luxar dataset generation.

Provides consistent path resolution for all generated datasets,
ensuring they go to the centralized datasets/ folder at project root.

This module enables examples and demos to output to a single location
regardless of where scripts are run from.
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Find the Luxar project root directory.

    Traverses up from this file's location looking for pyproject.toml.
    Result is cached for performance.

    Returns:
        Path to project root containing pyproject.toml

    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def get_datasets_dir() -> Path:
    """Get the datasets directory at project root.

    Creates the directory if it doesn't exist.

    Returns:
        Path to datasets/ directory
    """
    datasets_dir = get_project_root() / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir


def get_examples_output_dir() -> Path:
    """Get output directory for example scripts.

    Creates the directory if it doesn't exist.

    Returns:
        Path to datasets/examples/ directory
    """
    examples_dir = get_datasets_dir() / "examples"
    examples_dir.mkdir(exist_ok=True)
    return examples_dir


def get_demos_output_dir() -> Path:
    """Get output directory for demo scripts.

    Creates the directory if it doesn't exist.

    Returns:
        Path to datasets/demos/ directory
    """
    demos_dir = get_datasets_dir() / "demos"
    demos_dir.mkdir(exist_ok=True)
    return demos_dir


# Recognized Luxar zarr suffixes, longest-first so the more specific compound
# suffixes are matched before the bare ``.zarr``.
_ZARR_SUFFIXES = (".luxar.zarr", ".gsplats.zarr", ".zarr")


def normalize_zarr_path(path: str | Path, canonical_suffix: str) -> Path:
    """Ensure ``path`` ends in ``canonical_suffix``.

    Strips any existing recognized zarr suffix (``.luxar.zarr`` /
    ``.gsplats.zarr`` / ``.zarr``) from the filename and re-appends
    ``canonical_suffix``. Used to enforce the canonical extension for full
    scenes (``.luxar.zarr``) and standalone gsplat files (``.gsplats.zarr``).

    Compression suffixes (``.zip`` / ``.tar.gz`` / ``.gz``) are NOT handled
    here — callers that support compression strip those first and pass the
    inner store name (see ``save_gsplats._resolve_zarr_path``).

    Args:
        path: The output path whose extension should be normalized.
        canonical_suffix: The suffix to enforce, e.g. ``".luxar.zarr"``.

    Returns:
        A new ``Path`` with the canonical suffix. Paths that already end in
        ``canonical_suffix`` are returned unchanged.

    Raises:
        ValueError: If ``path`` has no filename component (e.g. ``"."``).
    """
    p = Path(path)
    name = p.name
    if not name:
        raise ValueError(f"Cannot normalize a path with no filename: {path!r}")
    for suffix in _ZARR_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return p.with_name(name + canonical_suffix)
