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


def _owns_imported_luxar(root: Path) -> bool:
    """True when ``root`` is the dev checkout that owns the imported ``luxar``.

    ``get_project_root`` walks up to the first ``pyproject.toml`` ancestor, which
    for a NON-editable install inside a user's own project venv
    (``<proj>/.venv/.../site-packages/luxar`` — the default uv/poetry layout) is
    the *user's* project, not Luxar's checkout. Trust a root only when it actually
    owns the imported package: a real checkout is ``<root>/packages/luxar/src/luxar``.
    """
    import luxar

    pkg_dir = Path(luxar.__file__).resolve().parent
    return (root / "packages" / "luxar" / "src" / "luxar").resolve() == pkg_dir


def get_datasets_dir(create: bool = True) -> Path:
    """Get the directory for generated example/demo datasets.

    In a Luxar dev checkout this is ``<repo>/datasets``. From an installed wheel —
    or when the only ``pyproject.toml`` ancestor is an UNRELATED project (a
    uv/poetry venv) — there is no repo to write into, so it falls back to the user
    cache dir ``~/.cache/luxar/datasets``. This is what lets demos and examples run
    from a plain ``pip install`` instead of raising "could not find project root"
    (#3097), and keeps a non-editable install from scribbling into the user's own
    project tree.

    Args:
        create: Create the directory (and parents) if it doesn't exist (default
            True). Pass False to resolve the path read-only (e.g. when only
            checking whether an output exists, so listing does not create dirs).

    Returns:
        Path to the datasets/ directory.
    """
    try:
        root: Path | None = get_project_root()
    except RuntimeError:
        root = None
    if root is not None and _owns_imported_luxar(root):
        datasets_dir = root / "datasets"
    else:
        # No owning checkout: write under the user cache dir (resolved at call
        # time so it honours $HOME), matching the ~/.cache/luxar convention the
        # demo downloaders already use.
        datasets_dir = Path.home() / ".cache" / "luxar" / "datasets"
    if create:
        datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_examples_output_dir(create: bool = True) -> Path:
    """Get output directory for example scripts.

    Args:
        create: Create the directory (and its parent) if missing (default True).

    Returns:
        Path to datasets/examples/ directory
    """
    examples_dir = get_datasets_dir(create=create) / "examples"
    if create:
        examples_dir.mkdir(exist_ok=True)
    return examples_dir


def get_demos_output_dir(create: bool = True) -> Path:
    """Get output directory for demo scripts.

    Args:
        create: Create the directory (and its parent) if missing (default True).
            Pass False for read-only path resolution (status/inventory), so a
            read-only command like ``luxar demo list`` never creates ``datasets/``.

    Returns:
        Path to datasets/demos/ directory
    """
    demos_dir = get_datasets_dir(create=create) / "demos"
    if create:
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
