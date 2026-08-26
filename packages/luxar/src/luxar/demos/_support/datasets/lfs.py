"""Git LFS and local demo-data validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# Directory containing precomputed data shipped with the package (via Git LFS)
# Package-internal: imported by bundles and data_fetch.
_DEMOS_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def is_lfs_pointer(path: Path) -> bool:
    """Check whether *path* is an unpulled Git LFS pointer file.

    LFS pointers are small text files (< 1 KB) whose first line is
    ``version https://git-lfs.github.com/spec/v1``.
    """
    if not path.exists():
        return False
    if path.stat().st_size > 1024:
        return False
    try:
        with open(path, "r") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except (UnicodeDecodeError, OSError):
        return False


# Package-internal: imported by bundles.
def _unshippable_reason(demo_name: str) -> Optional[str]:
    """Why *demo_name*'s data is not shipped, or ``None`` if it should be.

    The manifest is the single source of truth for dataset disposition, so this
    asks it rather than keeping a second list that could drift: a
    ``local-compute`` / ``regenerate`` bucket means the data is deliberately
    absent (we may not redistribute it, or it is cheap to rebuild). Any failure
    to answer is reported as "shippable", which preserves the previous
    behaviour — a missing file then raises the ordinary git-lfs error rather
    than being silently excused.
    """
    try:
        from .data_fetch import dataset_spec, load_manifest

        spec = dataset_spec(demo_name, load_manifest())
    except (ImportError, KeyError, OSError, ValueError):
        # DatasetNotFound subclasses KeyError and a malformed manifest raises
        # ValueError, so the handler needs no imported name — which is what
        # lets it cover a failure of the import above too (naming
        # DatasetNotFound here would raise NameError from the handler instead).
        return None
    if spec.get("bucket") not in ("local-compute", "regenerate"):
        return None
    return str(spec.get("reason") or spec.get("strategy") or "not redistributable")


# Package-internal: imported by bundles.
def _validate_lfs_files(paths: list[Path]) -> None:
    """Raise a helpful error if any *paths* are missing or are LFS pointers."""
    missing = [p for p in paths if not p.exists()]
    pointers = [p for p in paths if p.exists() and is_lfs_pointer(p)]

    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Precomputed data files not found: {names}\n"
            "Run 'git lfs pull' to download the data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )
    if pointers:
        names = ", ".join(p.name for p in pointers)
        raise FileNotFoundError(
            f"Precomputed data files are Git LFS pointers (not pulled): {names}\n"
            "Run 'git lfs pull' to download the actual data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )


def require_local_data(path: Union[str, Path], hint: Optional[str] = None) -> Path:
    """Return ``path`` if it is real local data, else raise a helpful error.

    Guards the local-data demos (LFS-tracked parquet/npz) so an unpulled Git LFS
    pointer raises the clear "run git lfs pull" message instead of a cryptic
    downstream parse error. Wraps :func:`_validate_lfs_files`.
    """
    p = Path(path)
    _validate_lfs_files([p])
    if hint and not p.exists():  # pragma: no cover - _validate_lfs_files raised
        raise FileNotFoundError(hint)
    return p
