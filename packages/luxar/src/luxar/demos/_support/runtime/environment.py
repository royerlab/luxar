"""Bake a demo's scene environment into its store, best-effort.

A scene whose ``viewer_config.environment.source`` is ``"scene"`` has its
environment captured live by the viewer: six renders of the whole scene into a
cube map, on load and again whenever the geometry, slice or appearance changes.
That is correct but not deterministic — what a bubble reflects depends on how
much data had streamed in when the capture fired, so it differs between
machines and between runs, and every visitor pays the capture.

Baking freezes it: the same six faces ship in the store as
``environment/faces-<digest>``, the viewer prefers them over any live capture,
and the result is identical everywhere. The group is excluded from the scene
``content_hash``, so attaching one does not invalidate warm viewer caches.

NEVER FATAL. The bake needs a development checkout — a built viewer ``dist``
and the viewer's Playwright — which a demo build cannot require. When those are
missing the demo is finished and correct; it simply falls back to the live
capture it would have used anyway.

@module demos/_support/runtime/environment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from arbol import aprint, asection

from ...._zarr_compat import read_node_attrs
from .cli import run_luxar_cli

#: Cube face size the CLI defaults to, mirrored here so a scene naming none
#: still passes an explicit value. A literal rather than an import: see the
#: note at the call site about keeping `luxar.demos` off `luxar.cli`.
_DEFAULT_RESOLUTION = 128

#: Sources whose lighting a bake can freeze. `hdri` is already a fixed file and
#: `room` is procedural and identical everywhere, so neither gains anything.
_BAKEABLE_SOURCES = ("scene",)


def _environment_config(store: Path) -> dict[str, Any] | None:
    """The scene's ``viewer_config.environment`` block, or None if unreadable."""
    try:
        attrs = read_node_attrs(store) or {}
    except Exception:  # noqa: BLE001 - introspection must not fail a demo build
        return None
    env = (attrs.get("viewer_config") or {}).get("environment")
    return env if isinstance(env, dict) else None


def bake_scene_environment(store: Union[str, Path]) -> bool:
    """Capture and attach the environment for ``store``. Returns whether it ran.

    Self-gating: returns False without complaint for a scene that declares no
    environment, one whose source is not a live capture, or one that already
    carries a baked map. Uses the scene's OWN probe and resolution, so the baked
    map is the capture the viewer would have made rather than a different look.
    """
    path = Path(store)
    config = _environment_config(path)
    if not config or config.get("source") not in _BAKEABLE_SOURCES:
        return False
    if (path / "environment").exists():
        return False

    probe = config.get("probe")
    resolution = int(config.get("resolution") or _DEFAULT_RESOLUTION)
    with asection("Baking the scene environment"):
        try:
            # Through the CLI, not by importing the bake: `luxar.demos` must
            # not depend on `luxar.cli` in the import graph, which is the same
            # reason `run_luxar_cli` resolves its app dynamically.
            run_luxar_cli(
                "env",
                "bake",
                str(path),
                "--probe",
                "auto" if probe is None else str(probe),
                "--resolution",
                str(resolution),
            )
        except Exception as error:  # noqa: BLE001 - see the module docstring
            aprint(f"Skipped: {error}")
            aprint("The viewer will capture the environment live instead.")
            return False
        if not (path / "environment").exists():
            aprint("Skipped: the bake produced no environment group.")
            return False
        return True
