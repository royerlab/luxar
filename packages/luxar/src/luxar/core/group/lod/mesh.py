"""Mesh substitutive-LOD axis resolver.

The Mesh sibling of :mod:`luxar.core.group.lod.points` /
:mod:`luxar.core.group.lod.lines` — and, unlike those two, **not** a thin
wrapper over the shared :func:`luxar.core.group.lod.group.resolve_substitutive_axis`.

## Why mesh needs its own resolver

Points and Lines share one implementation because they coarsen the same way:
both LIFT their elements to gsplats and run the gsplat substitutive pipeline. So
they share its whole vocabulary, including four keys that exist only because of
that lift —

* ``truncation_radius`` — feeds ``lift_points_to_gsplats`` / ``lift_lines_to_gsplats``
* ``max_aspect`` — caps per-splat anisotropy on the merged coarse levels
* ``device`` / ``seed`` — the mixture reduction's compute placement and RNG

Mesh does not lift. It coarsens by DECIMATION: merge vertices, reindex the faces,
drop the triangles that collapsed. None of those four keys names anything the
decimator can do, and there is no Gaussian mixture for ``method="kmeans"`` to
reduce. Widening the shared resolver would therefore have meant accepting five
words that quietly do nothing — which is exactly the class of bug the geometry
capability table exists to prevent, one layer down.

What mesh *does* share is the part that is genuinely geometry-agnostic:
``compression_factor`` / ``levels`` / ``coverage_fractions`` / ``coarsen_dims``
mean the same thing here as anywhere, and
:func:`luxar.core.group.lod.group.coverage_fractions` takes plain element counts.
Those are reused verbatim; only the vocabulary around them is mesh's own.

## ``coarsen_dims`` is the shared name for the decimator's ``spatial_dims``

They are the same concept: the dimensions the reduction may merge across, with
the complement acting as hard barriers so a coarse element never blends across a
timepoint or a channel. The decimator's parameter is spelled ``spatial_dims``
because that is what a grid is built over; the *authoring* vocabulary stays
``coarsen_dims`` so a mesh reads like the other three geometries. This module is
where the two names meet.

@module luxar.core.group.lod.mesh
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .group import (
    DEFAULT_MESH_SUBSTITUTIVE_METHOD,
    DEFAULT_SUBSTITUTIVE_K,
    DEFAULT_SUBSTITUTIVE_LEVELS,
    MESH_SUBSTITUTIVE_METHODS,
    _assert_strict_ascending,
    _validate_coarsen_dims_spec,
)

#: Keys the shared (lift-to-gsplats) vocabulary accepts and mesh cannot. Named
#: individually so the error says WHY rather than "unrecognized key" — each of
#: these is a reasonable thing to have tried, having read the Points docs.
_LIFT_ONLY_KEYS: Dict[str, str] = {
    "truncation_radius": (
        "it is the Gaussian truncation radius used when lifting elements to "
        "gsplats, and a mesh is not lifted"
    ),
    "max_aspect": (
        "it caps per-splat anisotropy on merged coarse levels, and a mesh's "
        "coarse levels are triangles, not splats"
    ),
    "device": (
        "the decimator is pure NumPy on the CPU (deliberately — a scipy or torch "
        "dependency would put `luxar mesh lod` behind an optional extra)"
    ),
    "seed": (
        "vertex clustering is deterministic: there is no seeding, no "
        "initialization and nothing to converge, so there is no RNG to fix"
    ),
}


def resolve_substitutive_axis_mesh(spec: Any) -> Optional[Dict[str, Any]]:
    """Normalize a mesh ``substitutive_lod=`` kwarg into a spec dict (or ``None``).

    Vocabulary:

    * ``None`` / ``False`` → no-op (the caller writes a plain mesh leaf).
    * ``True`` / ``dict()`` → defaults (K=4, levels=3, method="auto").
    * ``dict(...)`` → keys ``compression_factor`` (alias ``K``), ``levels``
      (alias ``n_lods``), ``method``, ``coverage_fractions``, ``coarsen_dims``.

    ``method`` accepts :data:`MESH_SUBSTITUTIVE_METHODS`. ``auto`` resolves to
    ``cluster`` today; it stays the default so that gaining a second tier
    (``qem``, issue #1348) is not a behaviour change for anyone who wrote it.

    Raises:
        TypeError: If ``spec`` is neither ``None``, a bool, nor a dict.
        ValueError: On an out-of-range value, an unknown key, or one of the
            lift-only keys in :data:`_LIFT_ONLY_KEYS` — the latter with the
            reason it cannot apply to a surface, since every one of them is
            valid for Points and Lines and so a reasonable thing to have tried.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(
            f"substitutive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)

    # Checked BEFORE the generic unknown-key sweep so these get their specific
    # explanation rather than being lumped into a list of typos.
    for key, why in _LIFT_ONLY_KEYS.items():
        if key in kwargs:
            raise ValueError(
                f"substitutive_lod for Mesh: {key!r} does not apply to a mesh — "
                f"{why}. It is valid for Points/Lines/GSplats, which coarsen by "
                "reducing a Gaussian mixture; a mesh coarsens by decimation."
            )

    compression_factor = int(
        kwargs.pop("compression_factor", kwargs.pop("K", DEFAULT_SUBSTITUTIVE_K))
    )
    if compression_factor < 2:
        raise ValueError(f"compression_factor must be >= 2, got {compression_factor}")

    levels = int(
        kwargs.pop("levels", kwargs.pop("n_lods", DEFAULT_SUBSTITUTIVE_LEVELS))
    )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    method = str(kwargs.pop("method", DEFAULT_MESH_SUBSTITUTIVE_METHOD)).replace(
        "-", "_"
    )
    if method not in MESH_SUBSTITUTIVE_METHODS:
        raise ValueError(
            f"substitutive_lod for Mesh: method must be one of "
            f"{sorted(MESH_SUBSTITUTIVE_METHODS)}; got {method!r}. The "
            "Gaussian-mixture reducers (kmeans, greedy, ...) coarsen a mixture of "
            "independent elements and have no meaning for a connected surface, "
            "which is decimated instead."
        )

    explicit_coverage = kwargs.pop("coverage_fractions", None)
    if explicit_coverage is not None:
        explicit_coverage = [float(m) for m in explicit_coverage]
        if not explicit_coverage:
            raise ValueError(
                "substitutive_lod=dict(coverage_fractions=...) must be non-empty "
                "(one strictly-ascending value in [0, 1] per LOD level)"
            )
        _assert_strict_ascending(
            explicit_coverage, "substitutive_lod=dict(coverage_fractions=...)"
        )
        if explicit_coverage[0] < 0.0 or explicit_coverage[-1] > 1.0:
            raise ValueError(
                "substitutive_lod=dict(coverage_fractions=...): values must lie in "
                f"[0, 1] (coarsest→finest); got {explicit_coverage}"
            )

    # Shape/type only here; names and the "display" default are resolved against
    # the scene in the adder, exactly as the other three geometries do it.
    coarsen_dims = _validate_coarsen_dims_spec(kwargs.pop("coarsen_dims", None))

    if kwargs:
        raise ValueError(
            f"substitutive_lod for Mesh: unrecognized keys {sorted(kwargs)}. "
            "Valid keys: compression_factor (K), levels (n_lods), method, "
            "coverage_fractions, coarsen_dims. (A mesh's vocabulary is SHORTER "
            "than Points/Lines/GSplats — see this module's docstring for the "
            "four lift-only keys it deliberately omits.)"
        )

    return {
        "compression_factor": compression_factor,
        "levels": levels,
        "method": method,
        "coverage_fractions": explicit_coverage,
        "coarsen_dims": coarsen_dims,
    }


__all__ = ["resolve_substitutive_axis_mesh"]
