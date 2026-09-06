"""Baked scene environments (``luxar env bake`` / ``luxar env attach``).

The scene environment lights ``material="physical"`` meshes
(``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.3). A live viewer can
capture it from the scene itself; this package moves that cost off the viewer
for a published scene by storing the six captured cube faces in the store as a
root-level ``environment/`` sidecar group. :mod:`luxar.environment.container` defines the
blob the viewer hands back, :mod:`luxar.environment.attach` writes it into a store, and
:mod:`luxar.environment.bake` drives the headless capture end to end.
"""

from .attach import AttachReport, attach_environment
from .bake import BakeReport, bake_environment
from .container import (
    ENVIRONMENT_FORMAT,
    FACE_ORDER,
    REQUIRED_HEADER_KEYS,
    SAMPLE_FORMAT,
    pack,
    unpack,
)

__all__ = [
    "AttachReport",
    "BakeReport",
    "ENVIRONMENT_FORMAT",
    "FACE_ORDER",
    "REQUIRED_HEADER_KEYS",
    "SAMPLE_FORMAT",
    "attach_environment",
    "bake_environment",
    "pack",
    "unpack",
]
