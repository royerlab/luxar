"""Scene provenance and staleness helpers for Luxar demos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final, Union

from arbol import aprint

from ...._zarr_compat import is_consolidated, read_node_attrs
from ....utils.source_fingerprints import (
    fingerprint_imported_sources,
    store_writer_environment,
)

#: Scene-root attr holding the fingerprint of the builder that wrote the scene.
BUILDER_FINGERPRINT_ATTR: Final[str] = "builder_fingerprint"
_DEMO_FINGERPRINT_VERSION: Final[int] = 2
_DEMO_IMPORT_CACHE: dict[Path, set[str]] = {}
_DEMO_MODULE_CACHES: dict[Path, dict[str, Path | None]] = {}


def _clear_demo_source_fingerprint_caches() -> None:
    """Clear process-local static import resolution caches."""
    _DEMO_IMPORT_CACHE.clear()
    _DEMO_MODULE_CACHES.clear()


def demo_source_fingerprint(
    module_file: Union[str, Path], *, package_root: Path | None = None
) -> str:
    """Short content hash of a demo and the local code that writes it.

    Call as ``demo_source_fingerprint(__file__)``. The hash covers that demo
    module's reachable imports inside the Luxar package tree and the Zarr
    writer environment, so edits to its producers or encoding-environment
    changes invalidate the cached scene without unrelated modules doing so.

    Args:
        module_file: Path to the demo module (normally ``__file__``).
        package_root: Luxar package root. The installed package is used by default.

    Returns:
        16 hex characters, or ``""`` if any source cannot be read (in which
        case :func:`scene_is_current` degrades to a plain existence check rather
        than rebuilding a large scene on every run).
    """
    root = (
        Path(__file__).resolve().parents[3]
        if package_root is None
        else Path(package_root).resolve()
    )
    module_path = Path(module_file).resolve()
    module_cache = _DEMO_MODULE_CACHES.setdefault(root.parent, {})
    try:
        fingerprint_root = Path(os.path.commonpath((root.parent, module_path)))
        sources = fingerprint_imported_sources(
            fingerprint_root,
            module_path,
            (root.parent,),
            within=root,
            import_cache=_DEMO_IMPORT_CACHE,
            module_cache=module_cache,
        )
    except (OSError, ValueError):
        return ""
    digest = hashlib.sha256()
    digest.update(_DEMO_FINGERPRINT_VERSION.to_bytes(4, "big"))
    digest.update(bytes.fromhex(sources))
    digest.update(json.dumps(store_writer_environment(), sort_keys=True).encode())
    return digest.hexdigest()[:16]


def scene_is_current(
    output_path: Path,
    fingerprint: str,
    *,
    recompute: bool = False,
    keep_stale: bool = False,
) -> bool:
    """True if the scene at ``output_path`` can be reused as-is.

    Demos cache their built scene and, historically, reused it whenever the
    path merely EXISTED. That let a scene built by an older version of the demo
    be served forever: #1957 was reported against an ocean-currents scene whose
    missing streamlines had been fixed three weeks earlier, because the fix
    never rebuilt the stale store on disk.

    So a scene is current only when its save finished AND it was written by
    this exact builder dependency closure and Zarr writer environment. A scene
    from before fingerprinting carries no attr and is treated as stale — one
    rebuild, then it stamps itself.

    This gates SCENE ASSEMBLY only. Downloads, gsplat fits and precomputed
    bundles keep their own caches under ``~/.cache/luxar``, so a source edit
    costs a scene rebuild and never a re-download or a re-fit.

    Args:
        output_path: The ``.luxar.zarr`` the demo would write.
        fingerprint: This build's :func:`demo_source_fingerprint`.
        recompute: The demo's ``--recompute`` flag; forces a rebuild.
        keep_stale: The demo's ``--keep-stale`` flag; reuse whatever is on disk
            even when the builder changed. An escape hatch for an expensive
            scene the caller knows is good enough.

    Returns:
        True to reuse the existing scene, False to rebuild.
    """
    if not output_path.exists():
        return False
    if recompute:
        return False
    if not is_consolidated(output_path):
        return False
    if keep_stale:
        return True
    if not fingerprint:
        # Unreadable source: no basis to call it stale, and rebuilding a large
        # scene on a bad guess is worse than serving the one on disk.
        return True

    attrs = read_node_attrs(output_path) or {}
    stored = attrs.get(BUILDER_FINGERPRINT_ATTR)
    if stored == fingerprint:
        return True

    aprint(
        f"Demo producer changed since this scene was built "
        f"({stored or 'unstamped'} -> {fingerprint}); rebuilding. "
        f"Pass --keep-stale to reuse it instead."
    )
    return False


def print_data_provenance(
    *, title: str, source: str, license: str, url: str, note: str = ""
) -> None:
    """Print a dataset provenance/licence notice before a runtime download.

    Demos that fetch third-party data at runtime print this first, so the user
    sees the source and licence terms of what is about to be downloaded (Luxar
    itself redistributes none of it). See the DATA SOURCE & CITATION docstring
    block each demo also carries.
    """
    aprint("")
    aprint("─" * 70)
    aprint(f"  Dataset: {title}")
    aprint(f"  Source:  {source}")
    aprint(f"  License: {license}")
    aprint(f"  URL:     {url}")
    if note:
        aprint(f"  Note:    {note}")
    aprint("─" * 70)
    aprint("")
