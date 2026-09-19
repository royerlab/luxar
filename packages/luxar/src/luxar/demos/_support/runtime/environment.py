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

    with asection("Baking the scene environment"):
        try:
            from ....cli.env_ops.bake import DEFAULT_RESOLUTION, bake_environment

            probe = config.get("probe")
            report = bake_environment(
                path,
                probe="auto" if probe is None else str(probe),
                resolution=int(config.get("resolution") or DEFAULT_RESOLUTION),
            )
        except Exception as error:  # noqa: BLE001 - see the module docstring
            aprint(f"Skipped: {error}")
            aprint("The viewer will capture the environment live instead.")
            return False
        attached = report.attach
        if attached is None:
            aprint("Captured but not attached.")
            return False
        aprint(
            f"✓ attached {attached.array_name} ({report.resolution}px, probe {report.probe})"
        )
        return True
