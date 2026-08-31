"""Node-authoring blocks shared byte-for-byte by the geometry writers.

These are the small, geometry-agnostic steps that :meth:`~luxar.io.compiler.\
LuxarZarrCompiler.write_points`, :meth:`~luxar.io.compiler.LuxarZarrCompiler.\
write_lines` and :meth:`~luxar.io.compiler.LuxarZarrCompiler.write_mesh` perform
identically: normalizing the ``transform`` / ``nd_transform`` attrs and stamping
the default compositing attributes. Both are pure in-place mutators of the
caller's ``attrs`` dict so the on-disk output is unchanged from the inlined
versions.

Also home to the per-geometry ``*_RESERVED_ATTRS`` sets — the writer-authoritative
keys a caller must not supply.
"""

from __future__ import annotations

import difflib
import warnings
from typing import Any, Dict, FrozenSet, Optional

import numpy as np
import zarr
from numpy.typing import NDArray

from ...core.dimensions import Dimensions
from ...core.group.compositing import (
    IDENTITY_COMPOSITING_ATTRS,
    WRITER_STAMPED_APPEARANCE_DEFAULTS,
)
from ...typing_utils.constants import (
    ELEMENT_TEXELS_PER_ELEMENT,
    max_elements_per_node,
)
from ...validation.types import (
    validate_appearance_fraction,
    validate_positive_finite,
    validate_texture_filter,
    validate_texture_wrap,
)


class ElementCapacityWarning(UserWarning):
    """A node may exceed the viewer's per-node element-texture capacity."""


# Writer-authoritative attrs each geometry writer stamps unconditionally.
# User-supplied values for these keys are rejected in the fail-fast gate:
# letting them through would either silently lose the user's value (the stamp
# wins on disk) or blow up post-write with an accidental TypeError when the
# Node object is constructed (``type=`` collides with the Node constructor).
# ``ordering`` IS reserved (all geometry types). The writer stamps it
# authoritatively from the compiler's ``ordering_method`` — the sort order is not
# a per-node request; there is no ``ordering=`` parameter on ``add_points`` /
# ``add_lines`` / ``add_gsplats``, so any ``ordering=`` would arrive through
# ``**attrs``. Leaving it unreserved is not benign: ``Node.__init__`` re-persists
# the caller's ``**attrs`` through ``write_group`` AFTER the geometry writer has
# stamped the group, so an unreserved caller value would be what lands on disk and
# would desync the attr from the actual on-disk sort order (issue #1221). Reserving
# it rejects that value up front and points the caller at the real knob,
# ``LuxarZarrCompiler(ordering_method=...)`` / ``write_gsplats_tree(ordering=...)``.
# Mesh has no spatial index at all (``ordering`` is always ``"none"``), so the same
# reservation just keeps a supplied value from writing a lie the viewer would read.
# The companion ``ordering_min``/``ordering_max``/``ordering_bits_per_dim``/
# ``ordering_dims`` sub-metadata stamps are writer-authoritative too; they are not
# listed here because a caller supplying one is already rejected by the
# unknown-attr gate.
POINTS_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_points",
        "ndim",
        "has_colors",
        "has_radii",
        "has_sharpness",
        "has_scalars",
        "has_labels",
        "has_image_labels",
        "has_keys",
        "position_bounds",
        "max_radius",
        "ordering",
    }
)
LINES_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_vertices",
        "n_segments",
        "ndim",
        "original_line_type",
        "has_colors",
        "has_sharpness",
        "has_scalars",
        "has_labels",
        "has_image_labels",
        "has_keys",
        "max_width",
        "position_bounds",
        "ordering",
    }
)
GSPLATS_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_splats",
        "ndim",
        "has_colors",
        "has_label_ids",
        "label_vocabulary",
        "has_labels",
        "has_image_labels",
        "has_keys",
        "amplitude_range",
        "amplitude_data_range",
        # Writer-derived mass statistics that drive the finalize-time
        # amplitude-window harmonization (see finalize/amplitude_window.py).
        "amplitude_mass",
        "amplitude_mass_weighted_mean",
        "center_bounds",
        "position_bounds",
        "ordering",
    }
)
# ``has_image_labels`` is now reserved by all four sets. It had been missing from
# the three above even though every writer stamps it (the gap MESH_NODE_SPEC.md §9
# recorded). No clobber was possible — ``validate_render_attrs`` rejects a key
# absent from every set as *unknown* — so the cost was only the less accurate
# error message, but the asymmetry made "which flags does this writer own?"
# unanswerable from the sets alone.
MESH_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_vertices",
        "n_faces",
        "ndim",
        "has_normals",
        "normal_dims",
        "has_colors",
        "has_scalars",
        "has_uvs",
        # Texture stamps: all writer-derived, and the three DIMENSION ones are
        # reserved for a sharper reason than tidiness. The viewer's admission gate
        # budgets a node from these numbers before it fetches a chunk, so a
        # caller-supplied value that disagreed with the payload would make the
        # budget mean something other than what it says — which is the whole
        # decompression-bomb surface. They come from the validator, never from
        # `**attrs`.
        "has_texture",
        "texture_encoding",
        "texture_width",
        "texture_height",
        "texture_channels",
        "texture_color_space",
        "texture_data_range",
        "has_labels",
        "has_image_labels",
        "has_keys",
        "shading",
        "double_sided",
        "position_bounds",
        # Reserved like the sibling sets above; for mesh there is additionally
        # no spatial index, so a supplied `ordering` cannot be honoured and would
        # otherwise overwrite the writer's `"none"` on disk (see note at the top).
        "ordering",
    }
)

# The render/appearance attrs whose VALUES are validated below, and the ONLY
# keys advertised in the "Unknown node attribute" hint. A user typo like
# ``blending="max"`` (for ``blending_mode``) used to be persisted silently and
# ignored by the viewer (issue #787); these are the legitimate render keys a
# caller may set on at least one node type. Type-restricted keys remain here so
# the typo hint can advertise the full authoring surface, with per-type refusals
# enforced before writing. Keep in sync with the per-key validators in
# :func:`validate_render_attrs`.
KNOWN_RENDER_ATTRS: FrozenSet[str] = frozenset(
    {
        "absorption",
        "alpha_cutoff",
        "ambient",
        "blending_mode",
        "colormap",
        # Per-element interaction templates (issue #1917). ``link`` builds a
        # URL opened on left-click, ``copy`` a plain string offered by the
        # right-click menu, both substituting the hover vocabulary
        # (``{hover_label}`` / ``{hover_key}`` / ``{hover_node}`` /
        # ``{hover_index}``).
        # ``link_target`` picks the browsing context. Advertised here rather
        # than hidden in ``_ALLOWED_NODE_ATTRS`` for the same reason as
        # lines-only ``join``: they are real knobs a user authors, so a typo
        # deserves to see them in the hint.
        "copy",
        "gamma",
        "intensity",
        "link",
        "link_target",
        # Lines-only join style at degree-2 polyline joints (issue #790).
        # Advertised here rather than hidden in ``_ALLOWED_NODE_ATTRS``
        # because it is a real appearance knob a user authors, so it belongs
        # in the "known render attributes" hint a typo prints.
        "join",
        "layer",
        "offset",
        "opacity",
        # Mesh-only shading controls. Advertised for the same reason as
        # lines-only ``join``; :func:`reject_mesh_only_appearance` refuses
        # them on points, lines, gsplats, and groups before anything is written.
        "shade_exponent",
        "shininess",
        # Mesh-only nD LOADING knob rather than an appearance one, but advertised
        # here for the same reason: a typo must print in the "known render
        # attributes" hint instead of being reported as an unknown key.
        "slab_tolerance",
        "specular",
        # Mesh-only texture sampling. Same reasoning again: authorable knobs, so
        # a typo should see them in the hint.
        "texture_filter",
        "texture_wrap",
        "visible",
    }
)

_MESH_APPEARANCE_VALIDATORS = {
    "ambient": (validate_appearance_fraction, "Ambient"),
    "specular": (validate_appearance_fraction, "Specular"),
    "alpha_cutoff": (validate_appearance_fraction, "Alpha cutoff"),
    "shade_exponent": (validate_positive_finite, "Shade exponent"),
    "shininess": (validate_positive_finite, "Shininess"),
    # Texture sampling. Mesh-only for the same reason the five above are: only a
    # mesh has a texture to sample, so on any other node these are a silent
    # no-op that reads like a working setting.
    "texture_filter": (validate_texture_filter, "Texture filter"),
    "texture_wrap": (validate_texture_wrap, "Texture wrap"),
    # Slab half-width in CELLS, so any positive multiple is meaningful and there
    # is no upper bound to impose. Zero is refused by `validate_positive_finite`
    # and that refusal is load-bearing: a zero slab reduces mesh's whole-triangle
    # membership test to exact float equality with the slice plane, and the node
    # renders nothing (spec §5.2.1 — it is why mesh cannot reuse the Lines arm).
    "slab_tolerance": (validate_positive_finite, "Slab tolerance"),
}

# Non-appearance keys that legitimately reach :func:`validate_render_attrs` and
# must NOT be flagged as unknown. These are user-settable node attrs that are
# processed elsewhere (transforms, LOD selection, gsplat truncation, nD
# visibility broadcast) PLUS structural keys the scene machinery injects into
# the SAME attrs dict before it reaches this gate — node/group type
# discriminators, sibling ordering, specialized-group descriptors, persisted
# bounds, and the geometry writers' internal forwarding flags. Unlike
# ``KNOWN_RENDER_ATTRS`` these are accepted silently (not advertised in the
# error hint). Reserved writer-stamped keys are handled separately via
# ``*_RESERVED_ATTRS`` and are NOT listed here.
_ALLOWED_NODE_ATTRS: FrozenSet[str] = frozenset(
    {
        # Processed by prepare_transform_attrs / apply_gsplat_group_attrs.
        "transform",
        "nd_transform",
        # gsplat Gaussian cutoff, LOD selection, nD broadcast. (``ordering`` is
        # NOT here: it is writer-stamped and reserved via ``*_RESERVED_ATTRS``.)
        "truncation_radius",
        "coverage_fraction",
        "extend_to_all",
        # Viewer-consumed LOD quality stamps injected by additive ladders.
        "level_stats",
        "lod_stats",
        # Provenance for the insertion-time amplitude normalisation (see
        # ``core/group/gsplats_pipeline/amplitude_norm.py``). Records the single
        # factor the whole structure was scaled by, so an authored window, an
        # ``amplitude_mass`` stamp or a later refit can be reconciled with the
        # values actually stored. Not a render attr — the viewer does not read
        # it — hence here rather than in ``KNOWN_RENDER_ATTRS``.
        "amplitude_normalization_factor",
        # Structural keys injected by node construction / specialized-group
        # builders (add_lod_group / add_partition_group) / LOD wrappers.
        "type",
        "child_index",
        "kind",
        "selector",
        "default_level",
        "display_type",
        "max_elements",
        "position_bounds",
        # BSP tree stamped on a kind=partition group by the native leaf adders
        # and gsplat graft path; the viewer reads it for back-to-front ordering.
        "bsp_tree",
        # Geometry-writer internal forwarding flags. NOT an exhaustive list of
        # them: a flag popped BEFORE this gate runs never needs listing here.
        # ``_return_sort_order`` (see ``record_forwarded_sort_order``) is popped
        # as the writers' very first statement and is deliberately absent. Note
        # that "never needs listing" holds only for THIS (writer-internal) call
        # site: since #1529/#1534, ALL FOUR leaf adders (Points, Lines, Mesh,
        # GSplats) also run this same gate at the adder entry, before any
        # writer ever pops such a flag, so a popped-first private flag
        # reaching the adder would be rejected there as unknown instead.
        # ``_return_sort_order`` stays a hypothetical (never caller-supplied),
        # but ``_scalar_data_range`` is a LIVE case on Mesh: it is real,
        # caller-supplied input on the ``luxar mesh lod`` re-authoring path
        # (``cli/mesh_ops/lod_commands.py``), deliberately absent from
        # ``_ALLOWED_NODE_ATTRS`` (never a Node attr), and it only works
        # because ``add_mesh_impl`` pops it (``mesh.py``, near the top of the
        # function) ABOVE its own entry gate — an ordering that gate's own
        # comment now records, and that this comment must not contradict by
        # implying no such case exists.
        "_skip_scene_bounds",
    }
)

# Candidate names used for the "Did you mean 'X'?" hint: the render attrs plus
# the user-facing (non-structural, non-private) allowed keys. Structural /
# private keys are intentionally excluded so a typo isn't matched to ``type`` or
# ``child_index``.
_SUGGESTION_ATTRS: tuple[str, ...] = tuple(
    sorted(
        KNOWN_RENDER_ATTRS
        | {
            "transform",
            "nd_transform",
            "truncation_radius",
            "coverage_fraction",
            "extend_to_all",
        }
    )
)


def validate_node_path(path: str) -> str:
    """Validate a writer node path and return it with the leading ``/`` stripped.

    The compiler-side half of the node-name chokepoint (see
    :func:`~luxar.validation.base.validate_node_name`): every ``/``-separated
    segment of the path must be a valid node name. In particular an EMPTY path
    (or an empty segment, e.g. from ``add_points("", ...)``) resolves to the
    zarr ROOT group via ``require_group("")`` — writing a geometry node there
    stamps ``type='points'`` onto the scene root and makes the store
    unloadable — and dot-prefixed segments collide with zarr's reserved
    metadata keys (``.zgroup``/``.zattrs``/``.zarray``/``.zmetadata``).

    The exact single-segment root path ``overlays`` is likewise rejected: it is
    reserved for screen-space overlay metadata (written internally as
    ``overlays/<name>``), so a raw geometry writer must not claim it. Only the
    bare root name collides; ``overlays/<name>`` and nested paths like
    ``geometry/overlays`` are allowed.

    Raises:
        ValidationError: If the path or any of its segments is invalid.
    """
    from ...validation.base import ValidationError, validate_node_name

    if not isinstance(path, str):
        raise ValidationError(
            f"node path: Expected a string, got {type(path).__name__}"
        )
    stripped = path.lstrip("/")
    if not stripped.strip():
        raise ValidationError(
            f"node path: Path must not be empty (got {path!r}). An empty path "
            "resolves to the zarr ROOT group and would overwrite the scene root.",
            "Provide a non-empty node name/path",
        )
    if stripped == "overlays":
        raise ValidationError(
            "node path: Top-level node path 'overlays' is reserved for "
            "screen-space overlay metadata. Write internal overlays below "
            "'overlays/<name>' or choose a different user node name.",
            "Choose a different top-level node name",
        )
    for segment in stripped.split("/"):
        validate_node_name(segment, context=f"node path {path!r}")
    return stripped


def validate_broadcast_color(colors: Any, context: str = "colors") -> None:
    """Validate a uniform broadcast color (list/tuple) BEFORE any zarr write.

    The encoder broadcasts a color list/tuple to all elements; a wrong-length
    or non-numeric tuple used to die inside the encoder AFTER the positions
    were already written. Shared by the Points and Lines writers' fail-fast
    gates (per the three-geometry symmetry rule).

    Raises:
        ValidationError: If the broadcast color is not a finite numeric
            RGB (3) or RGBA (4) sequence.
    """
    from ...validation.base import ValidationError

    if len(colors) not in (3, 4):
        raise ValidationError(
            f"{context}: Uniform color must have 3 (RGB) or 4 (RGBA) "
            f"components, got {len(colors)}",
            "Pass e.g. colors=(1.0, 0.0, 0.0) for a uniform red",
        )
    import numpy as np

    for i, component in enumerate(colors):
        # np.floating/np.integer included: np.float32 does NOT subclass
        # Python float (np.float64 does), and float32 tuple components — e.g.
        # tuple(color_array[i]) — are a legitimate caller pattern.
        if not isinstance(
            component, (int, float, np.integer, np.floating)
        ) or not np.isfinite(component):
            raise ValidationError(
                f"{context}: Uniform color component {i} must be a finite "
                f"number, got {component!r}",
                "Use finite numeric RGB(A) components",
            )
        # Mirror the ndarray validator's bounds (validate_colors_for_writing):
        # RGB is non-negative (HDR > 1 allowed); the optional 4th component is
        # per-element OPACITY and must stay in [0, 1] — an HDR-RGB tuple would
        # otherwise smuggle an out-of-range alpha past the SDR range check
        # (which scans RGB only), and an SDR tuple would fail only inside the
        # encoder AFTER positions were already written.
        if component < 0:
            raise ValidationError(
                f"{context}: Uniform color component {i} cannot be negative, "
                f"got {component!r}",
                "Use non-negative RGB(A) components",
            )
        if i == 3 and component > 1:
            raise ValidationError(
                f"{context}: Uniform color alpha (component 4) must be in "
                f"[0, 1] — it is per-element opacity, never HDR — got "
                f"{component!r}",
                "Clamp the alpha component to [0, 1]",
            )


def validate_scalars_preflight(
    scalars: Any, n_elements: int, context: str = "scalars"
) -> None:
    """Validate colormap scalars (array or broadcast scalar) BEFORE any write.

    The scalars dataset writer runs late in the pipeline, so a wrong-length or
    NaN scalars input used to leave a partial node behind. Shared by the
    Points and Lines writers' fail-fast gates.

    Raises:
        ValidationError: If scalars are not finite numeric values with one
            entry per element (or a single broadcast value).
    """
    import numpy as np

    from ...validation.base import (
        ValidationError,
        _validate_numeric_finite_values,
    )

    if isinstance(scalars, (int, float)):
        if not np.isfinite(scalars):
            raise ValidationError(
                f"{context}: Scalar value must be finite. Got {scalars}",
                "Provide a finite scalar value",
            )
        # Scalars are stored as float32; a finite value beyond the float32
        # range (|v| > ~3.4e38) overflows to inf on cast, which the encoder
        # rejects AFTER positions are written. Reject here so the fail-fast
        # gate stays sufficient (no partial node).
        if not np.isfinite(np.float32(scalars)):
            raise ValidationError(
                f"{context}: Scalar value {scalars} is not representable as "
                f"float32 (overflows to inf on cast).",
                "Rescale scalars into the float32 range (|value| <= 3.4e38)",
            )
        return

    if not isinstance(scalars, np.ndarray):
        # Same dead-end as the radii/widths/sharpness validators used to
        # have (#752): np.array(np.float32(x)) is 0D and fails the next
        # check, so point at float(...) instead.
        raise ValidationError(
            f"{context}: Expected a 1D numpy array or a Python float, "
            f"got {type(scalars).__name__}",
            "Pass a Python float — e.g. float(scalars) — for a single "
            "broadcast value, or a 1D array with one value per element",
        )

    if scalars.ndim != 1 or (scalars.shape[0] != n_elements and scalars.shape[0] != 1):
        raise ValidationError(
            f"{context}: Expected 1D array with {n_elements} values or 1 "
            f"(broadcast), got shape {scalars.shape}",
            f"Provide exactly {n_elements} scalar values or a single value",
        )

    _validate_numeric_finite_values(scalars, context)

    # Scalars are stored as float32 (see write_scalars). A finite float64
    # value beyond the float32 range overflows to inf on cast — the encoder
    # would reject it AFTER positions are written, leaving a partial node.
    # Validate float32-representability here so the pre-write gate is
    # sufficient. Only float dtypes wider than float32 can overflow (int64
    # tops out at ~9.2e18 << 3.4e38), and the float32 cast is monotone, so
    # casting just the extrema is exact — no full-array copy in preflight.
    if (
        scalars.size > 0
        and np.issubdtype(scalars.dtype, np.floating)
        and scalars.dtype.itemsize > 4
    ):
        extrema = np.array([scalars.min(), scalars.max()], dtype=scalars.dtype)
        if not np.all(np.isfinite(extrema.astype(np.float32))):
            raise ValidationError(
                f"{context}: One or more scalar values are not representable "
                f"as float32 (overflow to inf on cast).",
                "Rescale scalars into the float32 range (|value| <= 3.4e38)",
            )


def prepare_transform_attrs(attrs: Dict[str, Any], store: zarr.Group) -> None:
    """Normalize ``transform`` / ``nd_transform`` attrs in place.

    ``transform`` is round-tripped through :func:`~luxar.core.transforms.\
    prepare_transform_for_zarr` (NumPy→THREE.js column-major); ``nd_transform``
    is validated against the scene dimensions read from ``store.attrs``. No-op
    when neither key is present.
    """
    if "transform" in attrs:
        from ...core.transforms import prepare_transform_for_zarr

        attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

    if "nd_transform" in attrs:
        from ...validation.nd_transforms import validate_nd_transform

        dims = None
        if "scene_dimensions" in store.attrs:
            dims = Dimensions.from_dict(store.attrs["scene_dimensions"])
        attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)


def record_forwarded_sort_order(
    metadata: Dict[str, Any],
    requested: bool,
    sort_order: Optional[NDArray[np.integer]],
) -> None:
    """Record this node's spatial permutation in its metadata, on request.

    Backs the private ``_return_sort_order`` forwarding flag on ``write_points``
    / ``write_lines``: the multi-LOD parent writers need each level's spatial
    permutation to build the ladder's union label CSR, and the permutation is
    never persisted on disk. OPT-IN so the flat write path does not park a big
    index array in the compiler's metadata cache.

    Args:
        metadata: The writer's metadata dict, mutated in place.
        requested: The popped ``_return_sort_order`` flag.
        sort_order: The node's permutation, or ``None`` for no spatial
            reordering (identity).
    """
    if requested:
        metadata["sort_order"] = sort_order


def apply_default_render_attrs(attrs: Dict[str, Any]) -> None:
    """Stamp the default compositing attrs (opacity/absorption/gamma/intensity/
    offset) in place, only where the caller did not supply them.

    ``blending_mode`` is deliberately NOT stamped: unlike these identity-valued
    attrs (multiplicative/additive no-ops under the viewer's hierarchical
    composition), a stamped blending default would OVERRIDE an ancestor-set
    mode under the viewer's nearest-setter-wins rule. An unset leaf inherits
    from the nearest ancestor; the viewer defaults to ``additive`` when no
    ancestor sets it.

    Mirrors the GSplat defaults in :func:`~luxar.io._compiler.gsplat_assembly.\
    apply_gsplat_group_attrs` (which additionally defaults ``truncation_radius``)
    — literally, not by coincidence: both read their values from
    :data:`~luxar.core.group.compositing.WRITER_STAMPED_APPEARANCE_DEFAULTS`,
    which is also what the READER consults to tell a manufactured identity from
    an authored one (the ``gsplat merge`` agreement rule).
    """
    for key in IDENTITY_COMPOSITING_ATTRS:
        if key not in attrs:
            attrs[key] = WRITER_STAMPED_APPEARANCE_DEFAULTS[key]


def warn_if_over_element_cap(
    geometry_type: str,
    count: int,
    node_path: str,
    *,
    enabled: bool = True,
) -> bool:
    """Warn when a node holds more elements than one node can render (#1957).

    The viewer packs per-element render data into an element texture and
    CLAMPS a node that overflows it — ``clampElementCapacity`` drops the tail
    with a single console warning and no other signal. Geometry is stored in
    Hilbert order, so the lost tail is one spatially CONTIGUOUS lobe: the
    symptom is a clean-edged wedge of missing geometry, which reads as a data
    or masking bug rather than a capacity limit. #1957 lost the entire North
    Atlantic to a 2.3% overflow this way.

    The cap depends on the viewer's GPU, so the only bound an author can rely
    on is the 4096-class floor in
    :func:`~luxar.typing_utils.constants.max_elements_per_node`. Warning here
    puts the diagnosis at the point where it is cheap to act on — while the
    data is being authored — instead of leaving it to whoever opens the scene
    on a smaller GPU months later.

    A warning, not an error: a node above the floor still renders whole on a
    16384-class GPU, so refusing to write it would reject data that works.

    Args:
        geometry_type: A key of ``ELEMENT_TEXELS_PER_ELEMENT`` ("points",
            "lines", "gsplats"). Any other type is a no-op.
        count: Elements in this node — segments for lines, points for points,
            splats for gsplats.
        node_path: The node's path, for the message.
        enabled: False to suppress a redundant child warning when its parent
            checks the aggregate count.

    Warns:
        ElementCapacityWarning: If a supported geometry node exceeds the
            conservative viewer capacity floor.

    Returns:
        True if a warning was emitted.
    """
    if not enabled or geometry_type not in ELEMENT_TEXELS_PER_ELEMENT:
        return False
    cap = max_elements_per_node(geometry_type)
    if count <= cap:
        return False
    noun = {"points": "points", "lines": "segments", "gsplats": "splats"}[geometry_type]
    remedy = (
        "Run `luxar gsplat lod --recipe tiles`, or use "
        "partition=dict(max_elements=...) from the scene API"
        if geometry_type == "gsplats"
        else "Split it with partition=dict(max_elements=...)"
    )
    message = (
        f"'{node_path}' holds {count:,} {noun}, above the {cap:,} a single "
        f"{geometry_type} node can render on a 4096-class GPU. If the whole "
        f"node is committed at once, such a GPU can silently drop the tail — "
        f"and because elements are stored in Hilbert "
        f"order, that tail is one contiguous region, so it looks like a "
        f"clean-edged hole in the data (#1957). An nD node sliced on a "
        f"non-displayed dimension commits only its current slice. {remedy} "
        f"to render everywhere."
    )
    warnings.warn(message, ElementCapacityWarning, stacklevel=1)
    return True


def validate_render_attrs(
    attrs: Dict[str, Any],
    reserved_attrs: FrozenSet[str] = frozenset(),
    reject_unknown: bool = True,
) -> None:
    """Validate render attrs that would corrupt a node if written unchecked.

    Called as the FIRST step of every geometry writer — before the zarr group
    is created — so an invalid value fails the write without leaving a partial
    node on disk. Covers every pure attr validator (no store access needed):
    blending_mode / absorption / opacity / gamma / intensity / offset / mesh
    appearance / layer / visible / colormap. The values are validated only (not converted) — the
    writer stores the caller's attrs unchanged.

    When ``reject_unknown`` is set, any attr key that is neither a known render
    attr (``KNOWN_RENDER_ATTRS``), an accepted non-render/structural key
    (``_ALLOWED_NODE_ATTRS``), nor a passed reserved key is rejected up front
    with a "Did you mean ...?" hint. This turns a silently-ignored typo — e.g.
    ``blending="max"`` instead of ``blending_mode="max"`` (issue #787) — into a
    loud fail-fast BEFORE any zarr is written.

    Args:
        attrs: The node attrs dict to validate.
        reserved_attrs: Writer-stamped keys the caller must not supply (see
            ``POINTS_RESERVED_ATTRS`` / ``LINES_RESERVED_ATTRS`` /
            ``GSPLATS_RESERVED_ATTRS``). A collision fails the write up front
            instead of being silently overwritten by the writer's stamps (or
            exploding post-write in the Node constructor).
        reject_unknown: When True (the default), reject unknown attr keys.
            The generic ``write_group`` disables this for the scene root and the
            ``overlays/`` namespace, which carry their own internal attr schemas
            (scene dimensions / viewer config / overlay styling).
    """
    if reserved_attrs:
        collisions = sorted(reserved_attrs & attrs.keys())
        if collisions:
            # ``ordering=`` was a tolerated call pattern before it was
            # reserved (#1221) — the caller's value was re-persisted over the
            # writer's stamp and won on disk — so point migrating callers at
            # the real knob.
            hint = (
                " The sort order is chosen once at compiler construction — "
                "LuxarZarrCompiler(ordering_method=...) — not per node."
                if "ordering" in collisions
                else ""
            )
            raise ValueError(
                f"Attribute(s) {collisions} are reserved: the writer stamps "
                f"them authoritatively (type, element counts, presence flags, "
                f"bounds, ...). Remove them from the node attrs.{hint}"
            )

    if reject_unknown:
        allowed = KNOWN_RENDER_ATTRS | _ALLOWED_NODE_ATTRS | reserved_attrs
        for key in sorted(attrs.keys()):
            if key in allowed:
                continue
            suggestions = difflib.get_close_matches(key, _SUGGESTION_ATTRS, n=1)
            hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
            known = ", ".join(sorted(KNOWN_RENDER_ATTRS))
            raise ValueError(
                f"Unknown node attribute {key!r}.{hint} The viewer would "
                f"silently ignore it. Known render attributes: {known}. Remove "
                f"it or use a supported attribute."
            )

    if "blending_mode" in attrs:
        from ...validation.types import validate_blending_mode

        validate_blending_mode(attrs["blending_mode"])

    if "join" in attrs:
        # The KEY allowlist above catches ``jion=``; this catches ``join="mitre"``.
        # Both matter: an unrecognised style is far more likely a typo than a
        # request for no joins, and the file would otherwise write cleanly and
        # render with the default, giving the author nothing to go on.
        from ...validation.types import validate_line_join

        validate_line_join(attrs["join"])

    if "absorption" in attrs:
        from ...validation.types import validate_absorption

        validate_absorption(attrs["absorption"])

    if "opacity" in attrs:
        from ...validation.types import validate_opacity

        validate_opacity(attrs["opacity"])

    _validate_mesh_appearance_attrs(attrs)

    if "truncation_radius" in attrs:
        from ...validation.types import validate_truncation_radius

        validate_truncation_radius(attrs["truncation_radius"])

    if "gamma" in attrs:
        from ...validation.types import validate_gamma

        validate_gamma(attrs["gamma"])

    if "intensity" in attrs:
        from ...validation.types import validate_intensity

        validate_intensity(attrs["intensity"])

    if "offset" in attrs:
        from ...validation.types import validate_offset

        validate_offset(attrs["offset"])

    if "layer" in attrs:
        from ...validation.types import validate_layer

        validate_layer(attrs["layer"])

    if "visible" in attrs:
        from ...validation.types import validate_visible

        validate_visible(attrs["visible"])

    if "colormap" in attrs and attrs["colormap"] is not None:
        from ...validation.types import validate_colormap

        validate_colormap(attrs["colormap"])

    _validate_interaction_attrs(attrs)


def _validate_mesh_appearance_attrs(attrs: Dict[str, Any]) -> None:
    """Validate seven appearance controls plus slab tolerance in the shared gate."""
    for key, (validator, label) in _MESH_APPEARANCE_VALIDATORS.items():
        if key in attrs:
            validator(attrs[key], label)


def _validate_interaction_attrs(attrs: Dict[str, Any]) -> None:
    """Validate element interaction templates in the shared attr gate."""
    # Checked before the node exists on disk because the failure they prevent
    # is otherwise silent: a bad link writes cleanly and simply does nothing
    # when the user clicks it, with no file or console diagnostic (#1917).
    if "link" in attrs:
        from ...validation.types import validate_link

        validate_link(attrs["link"])

    if "copy" in attrs:
        from ...validation.types import validate_copy_template

        validate_copy_template(attrs["copy"])

    if "link_target" in attrs:
        from ...validation.types import validate_link_target

        validate_link_target(attrs["link_target"])

    # `link_target` alone is inert — it only says WHERE a link would open.
    # Refuse it rather than write a node whose only interaction attr can
    # never be read, which is a typo (`link_taget=`) far more often than a
    # deliberate choice.
    if "link_target" in attrs and "link" not in attrs:
        raise ValueError(
            "link_target was given without link. It only selects the browsing "
            "context for a link, so on its own it has no effect. Add "
            "link='https://...' or drop link_target."
        )
