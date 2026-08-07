"""Per-geometry-type capability flags for the writer side.

``GEOMETRY_TYPES`` (from the format contract) answers *"is this node a geometry
leaf?"* — a **vocabulary**. Several writer-side questions are narrower than that:

* may this type back a ``kind=lod`` group's ``display_type``?
* may it back a ``kind=partition`` group's ``display_type``?

Those are **capabilities**, and a type can be a perfectly valid geometry leaf
without having them: ``mesh`` is writable but has no LOD ladder and no partition
path — see ``docs/specs/MESH_NODE_SPEC.md`` §9. Note the LOD flavours are
excluded for different reasons: the ADDITIVE prefix ladder assumes independent
elements and cannot apply to a surface at all, whereas SUBSTITUTIVE levels make
no such assumption and are missing only a producer (mesh decimation).

Answering a capability question with the vocabulary is how a type gets admitted
to a code path that cannot represent it. Answering it with a hand-written tuple
is how the answer drifts once a fourth type exists — the codebase has hit both,
which is why the viewer grew the same table in
``packages/luxar-viewer/src/types/geometry-capabilities.ts``. Keep the two in
step: a capability that exists on one side and not the other is a bug in
whichever side is behind.

Python has no compile-time exhaustiveness check, so the module-level assertion
below stands in for TypeScript's ``Record<GeometryTypeName, …>``: adding a type
to the contract without giving it a row here fails at **import** time — loudly,
and before any store is written — rather than at the first ``KeyError``.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from ._format_contract import GEOMETRY_TYPES, GeometryTypeName


class GeometryCapabilities(NamedTuple):
    """What a geometry type is allowed to participate in, beyond being a leaf."""

    #: May appear as a ``kind=lod`` group's ``display_type``. Requires a level
    #: ladder whose coarse levels are renderable stand-ins for the fine ones.
    lod: bool

    #: May appear as a ``kind=partition`` group's ``display_type``. Requires a
    #: spatial split that can divide the geometry without corrupting topology.
    partition: bool


GEOMETRY_CAPABILITIES: Final[dict[GeometryTypeName, GeometryCapabilities]] = {
    "points": GeometryCapabilities(lod=True, partition=True),
    "lines": GeometryCapabilities(lod=True, partition=True),
    "gsplats": GeometryCapabilities(lod=True, partition=True),
    # Mesh: writable, but neither LOD nor partition exists yet (spec §9). Flip a
    # flag here when the corresponding path lands — not at the call sites.
    "mesh": GeometryCapabilities(lod=False, partition=False),
}


# The Python stand-in for `Record<GeometryTypeName, GeometryCapabilities>`: a new
# contract entry with no row here is an ImportError at startup, not a KeyError
# mid-write. Checked both ways so a stale row (a type REMOVED from the contract)
# is caught too.
_missing = sorted(set(GEOMETRY_TYPES) - set(GEOMETRY_CAPABILITIES))
_stray = sorted(set(GEOMETRY_CAPABILITIES) - set(GEOMETRY_TYPES))
if _missing or _stray:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "GEOMETRY_CAPABILITIES is out of step with the format contract: "
        f"missing rows for {_missing}, stray rows for {_stray}. Every entry in "
        "contract.yaml::geometry_types needs a row in "
        "luxar/typing_utils/geometry_capabilities.py."
    )
del _missing, _stray


#: ``GEOMETRY_CAPABILITIES`` keyed by plain ``str`` for probing untrusted values.
#: The typed table above stays the source of truth (its key type is what makes a
#: missing row detectable); this view exists because callers ask about a
#: ``display_type`` read from user input or from zarr attrs, which is an arbitrary
#: string and not a ``GeometryTypeName`` until proven otherwise.
#: Built by comprehension, not ``dict(...)``: ``dict`` is invariant in its key
#: type, so a ``dict[GeometryTypeName, V]`` is not a ``dict[str, V]``.
_BY_NAME: Final[dict[str, GeometryCapabilities]] = {
    str(name): row for name, row in GEOMETRY_CAPABILITIES.items()
}


def _row_for(display_type: object) -> GeometryCapabilities | None:
    """Capability row for ``display_type``, or ``None`` if it is not a known type.

    Returning ``None`` — rather than a default-permissive row — is what makes the
    callers below fail closed. That matters because ``display_type`` legitimately
    carries non-geometry marker strings on nested specialized groups, and a
    display type the writer does not recognise must not be granted a geometry
    capability by default.
    """
    if not isinstance(display_type, str):
        return None
    return _BY_NAME.get(display_type)


def supports_lod(display_type: object) -> bool:
    """Whether this geometry type may back a ``kind=lod`` group."""
    row = _row_for(display_type)
    return row is not None and row.lod


def supports_partition(display_type: object) -> bool:
    """Whether this geometry type may back a ``kind=partition`` group."""
    row = _row_for(display_type)
    return row is not None and row.partition


def lod_capable_types() -> tuple[GeometryTypeName, ...]:
    """The LOD-capable geometry types, in contract order (for error messages)."""
    return tuple(t for t in GEOMETRY_TYPES if GEOMETRY_CAPABILITIES[t].lod)


def partition_capable_types() -> tuple[GeometryTypeName, ...]:
    """The partition-capable geometry types, in contract order (error messages)."""
    return tuple(t for t in GEOMETRY_TYPES if GEOMETRY_CAPABILITIES[t].partition)


def require_lod_display_type(display_type: object, context: str) -> None:
    """Raise unless ``display_type`` may back a ``kind=lod`` group.

    A ``kind=lod`` group's display type reaches zarr by three routes — an
    explicit ``add_lod_group(display_type=...)`` kwarg, the finalize-time
    back-fill from the finest child, and the ``compute_lod_display_type`` helper
    — so the rule lives here and each route calls it. Guarding only one leaves
    the others silently open, and the back-fill is the one that actually runs in
    the common path.

    Passes anything that is not a known geometry type: ``display_type`` also
    carries non-geometry marker strings on nested specialized groups, and LOD is
    deliberately heterogeneous-tolerant. Only a geometry type whose row says
    ``lod=False`` is refused.

    Args:
        display_type: The resolved or supplied display type.
        context: What is being rejected, for the message (e.g. a node path).

    Raises:
        ValueError: If ``display_type`` is a geometry type without LOD support.
    """
    if display_type in GEOMETRY_TYPES and not supports_lod(display_type):
        valid = " / ".join(repr(t) for t in lod_capable_types())
        # The per-mechanism explanation is MESH-SPECIFIC and is appended only for
        # mesh. This guard also fires for any future LOD-less geometry type, and
        # for those an explanation about surfaces and QEM decimation would be a
        # confidently wrong diagnostic — worse than a generic one, because it reads
        # as though it were about the type the caller actually named.
        detail = (
            " For mesh the two flavours differ: the ADDITIVE prefix ladder reduces "
            "a set of independent elements, which a connected surface is not, so it "
            "cannot apply; SUBSTITUTIVE levels would work unchanged and are only "
            "missing a producer (the mesh analog is QEM decimation, which does not "
            "exist yet)."
            if display_type == "mesh"
            else ""
        )
        raise ValueError(
            f"{context}: display_type for a kind=lod group must be one of "
            f"{valid}, got {display_type!r}. A geometry type is excluded until it "
            f"has an LOD ladder.{detail} Writing this would produce a kind=lod "
            "group no viewer path can load."
        )


__all__ = [
    "GeometryCapabilities",
    "GEOMETRY_CAPABILITIES",
    "supports_lod",
    "supports_partition",
    "lod_capable_types",
    "partition_capable_types",
    "require_lod_display_type",
]
