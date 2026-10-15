"""Path utilities for Luxar dataset generation.

Provides consistent path resolution for generated datasets in a Luxar checkout
or an installed package.

This module enables examples and demos to output to a single location
regardless of where scripts are run from.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

#: Environment variable naming a custom-argument demo build (see
#: :func:`get_demos_output_dir`), and the subdirectory such builds live under.
DEMO_VARIANT_ENV = "LUXAR_DEMO_VARIANT"
DEMO_VARIANTS_DIRNAME = "variants"


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


def get_luxar_checkout_root() -> Path | None:
    """The Luxar source-checkout root that owns the imported ``luxar``, or None.

    ``None`` means this ``luxar`` is an installed wheel (or a non-editable install
    inside an unrelated project), so advice that only works from a clone — ``make
    build-launchers``, ``pnpm build`` in ``packages/luxar-viewer`` — must not be
    given as if it could be followed.
    """
    try:
        root = get_project_root()
    except RuntimeError:
        return None
    return root if _owns_imported_luxar(root) else None


def get_datasets_dir(create: bool = True) -> Path:
    """Get the directory for generated example/demo datasets.

    ``LUXAR_DATASETS_DIR`` overrides the location. In a Luxar dev checkout this
    is ``<repo>/datasets``. From an installed wheel —
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
    override = os.environ.get("LUXAR_DATASETS_DIR")
    if override:
        datasets_dir = Path(override).expanduser()
    else:
        try:
            root: Path | None = get_project_root()
        except RuntimeError:
            root = None
        if root is not None and _owns_imported_luxar(root):
            datasets_dir = root / "datasets"
        else:
            # Resolve at call time so a changed $HOME is honoured.
            datasets_dir = Path.home() / ".cache" / "luxar" / "datasets"
    if create:
        datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_examples_output_dir(create: bool = True) -> Path:
    """Get output directory for example scripts.

    Args:
        create: Create the directory (and its parent) if missing (default True).

    Returns:
        Path to the examples/ directory under the resolved datasets directory.
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

    ``LUXAR_DEMO_VARIANT`` (set by ``luxar demo run`` when a demo is given
    non-default arguments) redirects to ``demos/variants/<variant>/``, so a
    custom build never replaces the canonical scene that ``luxar demo`` marks
    built and ``run-all`` skips.

    Returns:
        Path to the demos/ directory under the resolved datasets directory.
    """
    demos_dir = get_datasets_dir(create=create) / "demos"
    variant = os.environ.get(DEMO_VARIANT_ENV, "").strip()
    if variant:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant) or variant in (".", ".."):
            raise ValueError(f"{DEMO_VARIANT_ENV} must be a plain name: {variant!r}")
        demos_dir = demos_dir / DEMO_VARIANTS_DIRNAME / variant
    if create:
        demos_dir.mkdir(parents=True, exist_ok=True)
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
