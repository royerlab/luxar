"""Pre-write gates: the checks a writer and its caller must agree on exactly.

Every function here is called from **two** places that must not drift apart:

* the compiler's geometry writer, as its fail-fast gate before the first array
  reaches disk; and
* :mod:`luxar.core.group.compositing`, against the SOURCE element count before a
  ``partition=`` / ``additive_lod=`` / ``substitutive_lod=`` decomposition splits
  that source into parts.

Sharing the function rather than repeating the checks is the point. A wrong-length
channel used to ride the per-part slicer's pass-through branch into every part and
be ACCEPTED by any part whose own count happened to match (#1437), while the plain
leaf path refused the same input outright. The pre-split gate cannot drift from what
the child write accepts, because it *is* what the child write runs.

Being needed by both is also why they live here rather than in either caller. They
were in :mod:`luxar.io._compiler`, and ``core`` reached back into that private tree
through five deferred, function-local imports — a ``core`` ↔ ``io`` cycle invisible
to mypy and to both import-linter contracts precisely because the imports were
deferred (audit A1-03). ``luxar.validation`` sits below both, so the shared code is
reachable from each without either depending on the other.

These are the *composite* gates. The per-array primitives they call
(``validate_colors_for_writing``, ``validate_radii_for_writing``, …) stay in
:mod:`luxar.validation.base`.
"""

from __future__ import annotations

import difflib
import operator
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import (
    ColorArray,
    PositionArray,
    ScalarArray,
)
from .base import (
    ValidationError,
    _validate_numeric_finite_values,
    validate_cholesky_for_writing,
    validate_colors_for_writing,
    validate_labels_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
    validate_widths_for_writing,
)
from .types import (
    PHYSICAL_MATERIAL_FRACTION_ATTRS,
    validate_appearance_fraction,
    validate_bool_flag,
    validate_hex_color,
    validate_ior,
    validate_mesh_material,
    validate_non_negative_finite,
    validate_positive_finite,
    validate_texture_filter,
    validate_texture_wrap,
)

__all__ = [
    "check_image_label_type",
    "validate_broadcast_color",
    "validate_gsplat_inputs",
    "validate_image_labels_for_writing",
    "validate_line_indices",
    "validate_lines_channels",
    "validate_points_channels",
    "validate_scalars_preflight",
]


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
    from .base import ValidationError

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


def check_image_label_type(item: Any) -> None:
    """Type dispatch, plus the ``ndarray`` SHAPE check: is ``item`` a type (and,
    for an ``ndarray``, a shape) an image label accepts?

    Pulled out of :func:`normalize_image_label` (#1491) so the TYPE question can
    be answered without doing any of the work that question's positive
    answer would then require — no PIL import for a plain ``bytes`` blob, no
    file read for a ``str``/``Path``. That is what lets
    :func:`validate_image_labels_for_writing` run this over every entry in a
    writer's fail-fast PRE-write gate: a single mistyped entry (an ``int`` in
    an otherwise right-length list) used to be refused only from
    :func:`normalize_image_label` itself, deep inside the writer — after the
    caller's other arrays, and on a ``substitutive_lod=`` ladder, after every
    level up to the one carrying it, were already on disk. A wrong-length list
    and a right-length list with one wrong-typed entry are equally plausible
    authoring mistakes and strand the same way; only the length/index check
    moved before this fix, which closed one door and left this one open.

    The ``ndarray`` SHAPE check (``(H, W)`` / ``(H, W, 3)`` / ``(H, W, 4)``)
    is included here too, not deferred to :func:`normalize_image_label`'s PIL
    round-trip: ``item.ndim`` / ``item.shape[2]`` are pure attribute reads,
    same cost class as the ``isinstance`` checks above them, and reading them
    never touches PIL or the store. Measured: the previous version left this
    exact strand open — a ``(4, 5, 2)`` ndarray (an unsupported shape) in an
    otherwise right-length/right-typed ``image_labels`` on a
    ``substitutive_lod=`` Points wrapper reached ``normalize_image_label``
    only from inside the finest child's own write, after every coarser gsplat
    level was already on disk, for the same reason an ``int`` entry did before
    this function existed.

    Deliberately mirrors :func:`normalize_image_label`'s own dispatch order
    exactly, including its one surprising quirk: when Pillow is NOT
    installed, ``from PIL import Image`` raises ``ImportError``
    UNCONDITIONALLY at that point in the dispatch — before ``item``'s actual
    type is even inspected — so ANY item that is not ``None`` / ``bytes`` /
    ``bytearray`` / ``Path`` / ``str`` raises the *same* "Pillow is required to
    encode PIL Image objects…" ``ImportError`` when Pillow is absent, even an
    ``int`` or an ``ndarray``. That is existing behaviour this function
    preserves rather than "fixes" — changing it would make this validator
    disagree with :func:`normalize_image_label` about a type Pillow-absent
    input, defeating the one-implementation point of factoring this out.

    Args:
        item: A single ``image_labels`` entry (or ``None``, meaning "no image
            for this element").

    Raises:
        TypeError: ``item`` is not one of the accepted types.
        ValueError: ``item`` is an ``ndarray`` but not one of the accepted
            shapes.
        ImportError: Pillow is not installed and ``item`` is not one of the
            few types (``None`` / ``bytes`` / ``bytearray`` / ``Path`` /
            ``str``) that never need it.
    """
    if item is None:
        return
    if isinstance(item, (bytes, bytearray)):
        return
    if isinstance(item, Path):
        return
    if isinstance(item, str):
        return

    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "Pillow is required to encode PIL Image objects as image labels. "
            "Install it with: pip install Pillow"
        )

    if isinstance(item, PILImage.Image):
        return
    if isinstance(item, np.ndarray):
        if item.ndim == 2:
            return
        if item.ndim == 3 and item.shape[2] in (3, 4):
            return
        raise ValueError(
            f"Unsupported ndarray shape for image label: {item.shape}. "
            f"Expected (H, W), (H, W, 3), or (H, W, 4)."
        )

    raise TypeError(
        f"Unsupported image label type: {type(item).__name__}. "
        f"Expected bytes, PIL.Image, numpy.ndarray, or file path."
    )


def validate_image_labels_for_writing(image_labels: Any, n_elements: int) -> None:
    """Validate an ``image_labels`` input's shape/indices/types BEFORE any zarr write.

    No store and no file reads, but NOT PIL-free any more (#1491 widened it):
    a Pillow-absent item still routes through :func:`check_image_label_type`'s
    own ``ImportError``, same as :func:`normalize_image_label` would raise for
    it later. So it can still run in a writer's fail-fast pre-write gate, and
    hoisted one level further, in a ``substitutive_lod=`` wrapper's PRE-SPLIT
    gate (see ``core.group.compositing.validate_points_channels_before_split``
    / ``validate_lines_channels_before_split``). :func:`write_image_labels_csr`
    used to run only the length/index checks itself, inline, and leave every
    item's TYPE to be discovered one at a time inside
    :func:`normalize_image_label` — AFTER the caller's other arrays
    (positions/colors/radii/…) were already written, and on a ``kind=lod``
    ladder, after every coarser level was on disk too, since the finest child
    (the one carrying ``image_labels``) writes LAST. A wrong-length list and a
    right-length list with one mistyped entry are equally plausible authoring
    mistakes and stranded identically; extracting BOTH kinds of check lets
    every fail-fast gate that calls this run them before anything is written,
    without duplicating either set of rules.

    Checks run in a fixed order — structural checks on the container first
    (a dense sequence's length, or every sparse key's integrality and bounds),
    THEN a type check of EVERY entry — for both the dense sequence form and
    the sparse ``dict`` form, and this reordering changes behaviour for both,
    not just the ``dict`` one. For the sparse form specifically: ALL key
    checks run before ANY item's type is inspected. Pre-#1491,
    ``write_image_labels_csr`` interleaved the two per key (bound-check
    ``idx``, then immediately :func:`normalize_image_label` ``item``), so a
    dict like ``{0: 12345, 5: b"x"}`` against ``n_elements=3`` raised
    ``TypeError: Unsupported image label type: int`` on key 0's value — key
    5's out-of-range index was never reached. The same call now raises
    ``ValueError: Image label index 5 out of range [0, 3)`` instead: a
    different exception TYPE and message, on a call that goes straight to
    :func:`write_image_labels_csr` (bypassing the higher pre-split gates
    entirely).

    The DENSE form diverges the same way, and not only for the length check:
    the type sweep now precedes every file read and shape check too. Measured:
    ``[b"ok", "/nope/missing.png", 123]`` used to raise ``FileNotFoundError``
    (item 1's missing-file read happened, since :func:`normalize_image_label`
    walked the list in order and item 2's bad type was never reached);
    it now raises ``TypeError: Unsupported image label type: int`` instead,
    since every item's type is checked before any file is opened. Arguably an
    improvement, since ``TypeError`` IS caught by the leaf adders' ``except
    (ValueError, TypeError)`` funnel while ``FileNotFoundError`` is not (see
    the residual paragraph below). Both divergences are deliberate and new,
    not a bug — see :mod:`labels/README.md` and the pinning tests in
    ``io/tests/_compiler/test_labels.py``.

    Deliberately narrow in one remaining way: this does not normalize
    dict->list or encode any blob, and it does not READ a ``str``/``Path``
    file — those still need :func:`normalize_image_label` (which needs an
    actual file read and/or a PIL round-trip) and so remain the writer's job,
    post-write. The ``ndarray`` SHAPE check is NOT part of that residual,
    though: :func:`check_image_label_type` (called for every entry, below)
    validates ``ndim``/``shape[2]`` eagerly, since that costs nothing a type
    check does not already cost — only an unreadable path (``FileNotFoundError``)
    and the actual PIL encode remain genuinely post-write-only. See the
    writers' step-0 comments for that (smaller) residual. Note two of the
    exceptions this function can now raise are NOT caught by the leaf adders'
    ``except (ValueError, TypeError)`` funnel — an ``ImportError`` (Pillow
    absent) escapes unwrapped, exactly as it would from a post-write
    :func:`normalize_image_label` call; that gap is pre-existing, not
    introduced by widening this function to catch types earlier. A
    post-write-only ``FileNotFoundError`` (an unreadable ``str``/``Path``) has
    the same gap, but never reaches this function at all.

    Args:
        image_labels: Per-element images, either the dense sequence form (one
            entry per element, checked by length; must be RE-iterable, see
            above) or the sparse ``Dict[int, Any]`` form (keys checked for
            integer-index semantics and bounds; a missing index just means no
            image, so absence is never an error).
        n_elements: Expected element count.

    Raises:
        ValueError: If a dict key is outside ``[0, n_elements)``, a
            sequence's length does not equal ``n_elements``, or an entry is
            an ``ndarray`` with an unsupported shape (from
            :func:`check_image_label_type`).
        TypeError: If a dict key is not usable as an integer index (a
            ``float``, a ``str``, ...), the dense form is a single-pass
            iterable, or an entry's type is not one
            :func:`check_image_label_type` accepts.
        ImportError: If an entry needs Pillow (anything that is not ``None`` /
            ``bytes`` / ``bytearray`` / ``Path`` / ``str``) and Pillow is not
            installed.
    """
    if isinstance(image_labels, dict):
        for idx in image_labels:
            # An index TYPE check before the bounds one: a key is used as
            # ``normalized[idx]`` in write_image_labels_csr, so anything
            # without integer-index semantics (a float, a np.float64 from an
            # arithmetic slip, a str) raises TypeError from that subscript —
            # POST-write, the exact strand this gate exists to close. `int`,
            # `bool` and any numpy integer satisfy operator.index; `float`
            # deliberately does not, even at an integral value.
            try:
                key = operator.index(idx)
            except TypeError:
                raise TypeError(
                    f"Image label index must be an integer, got "
                    f"{type(idx).__name__}: {idx!r}"
                ) from None
            if key < 0 or key >= n_elements:
                raise ValueError(
                    f"Image label index {idx} out of range [0, {n_elements})"
                )
        for item in image_labels.values():
            check_image_label_type(item)
    else:
        if len(image_labels) != n_elements:
            raise ValueError(
                f"Image labels length ({len(image_labels)}) must match "
                f"element count ({n_elements})"
            )
        # The per-entry sweep below WALKS the sequence, and every caller of
        # this gate walks it again afterwards (the writer materialises and
        # encodes it; a substitutive_lod= wrapper forwards it to the finest
        # child, which re-runs this same gate). A single-pass iterable — one
        # whose ``__iter__`` hands back the same, already-advancing iterator —
        # therefore has nothing left for that second walk, and this gate has
        # no way to hand its own materialised copy back to the caller. Refuse
        # it HERE, before anything is written, rather than let that second walk
        # report "Image labels length (0)" mid-write with the node's other
        # arrays already on disk. Detected WITHOUT consuming an item: a
        # re-iterable sequence (list / tuple / ndarray / pandas.Series / any
        # __getitem__ sequence) hands out a FRESH iterator per iter() call, so
        # only a one-shot one compares identical.
        if iter(image_labels) is iter(image_labels):
            raise TypeError(
                "image_labels must be a re-iterable sequence (list, tuple, "
                "ndarray, ...) or a dict; got a single-pass iterable of type "
                f"{type(image_labels).__name__}, which cannot be validated "
                "before the write without consuming it. Materialise it first: "
                "image_labels=list(...)."
            )
        for item in image_labels:
            check_image_label_type(item)


def validate_points_channels(
    n_points: int,
    *,
    colors: Any = None,
    radii: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> None:
    """Validate every per-point channel against ``n_points``. Pure — no I/O.

    Steps 0d-0f of :func:`write_points`'s fail-fast gate, factored out because a
    SECOND caller needs exactly them and nothing else:
    :func:`luxar.core.group.compositing.validate_points_channels_before_split`
    runs this against the SOURCE point count before a ``partition=`` /
    ``additive_lod=`` / ``substitutive_lod=`` decomposition. Without that, a
    wrong-length channel rode the per-part slicer's pass-through branch into
    every part and was ACCEPTED by any part whose own count happened to match
    (#1437), and the plain-leaf path refuses the same input outright.

    ``image_labels`` (issue #1491) does not fit the per-part-slicer trap
    above at all — it has no slicer in the first place: on
    ``substitutive_lod=`` the value is forwarded ONLY to the finest child
    (Points itself), which the wrapper writes LAST, after every coarse gsplat
    level is already committed. Pre-fix, the length/index check ran INSIDE
    ``write_image_labels_csr``, near the very end of that finest child's own
    write — AFTER its ``type``, ``n_points``, positions, radii, colors,
    sharpness, scalars and ``labels`` were already on disk. So a wrong-length
    ``image_labels`` left a COMPLETE, fully loadable N-level ``kind=lod``
    ladder on disk — every level present, every other array on every level
    present — silently missing only the ``image_label_offsets`` /
    ``image_label_bytes`` the caller actually asked for. That is HARDER to
    notice than a missing level, not milder: nothing about the ladder looks
    incomplete, it simply has no images.

    Sharing the function rather than repeating the checks is what keeps that
    promise true as the rules change: the pre-split gate cannot drift from what
    the child write accepts, because it IS what the child write runs. Same
    contract, and same wording, as the mesh sibling
    :func:`~luxar.io._compiler.geometry_writers.mesh.validate_mesh_arrays`.
    """

    # 0d. Pre-flight length sweep over ALL provided per-point arrays. The
    # spatial-ordering fancy-indexing below silently TRUNCATES a too-long
    # array and raises a raw IndexError on a too-short one, so lengths must
    # be checked before build_points_ordering runs. The per-dataset
    # validators further down remain in place (belt and braces).
    if colors is not None:
        if isinstance(colors, np.ndarray):
            # Points accept RGBA: the alpha column is per-point opacity
            # (consumed by every blending mode; mapped into optical depth in
            # volumetric — see VOLUMETRIC_BLENDING_SPEC.md, phase 3).
            validate_colors_for_writing(colors, n_points, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors, "colors")
    if radii is not None:
        # Validates arrays AND broadcast scalars (same finite/positive rules).
        validate_radii_for_writing(radii, n_points)
    if sharpness is not None:
        # Validates arrays AND broadcast scalars (same [0, 1] bounds).
        validate_sharpness_for_writing(sharpness, n_points)
    if scalars is not None:
        validate_scalars_preflight(scalars, n_points)
    # 0e. Labels: sequence-of-str type + length check (the CSR serializer
    # would otherwise AttributeError on a non-str entry AFTER the arrays
    # were written).
    if labels is not None:
        validate_labels_for_writing(labels, n_points)
    # Keys ride the same pre-flight as labels: the CSR serializer
    # UTF-8-encodes each entry, so a non-str or a length mismatch must be
    # caught BEFORE any array reaches disk (#1917).
    if keys is not None:
        validate_labels_for_writing(keys, n_points, context="keys", noun="Keys")
    # 0f. Image labels: length (dense) / index bounds (sparse dict) — see
    # validate_image_labels_for_writing for why this moved out of the CSR
    # writer itself.
    if image_labels is not None:
        validate_image_labels_for_writing(image_labels, n_points)


def validate_lines_channels(
    n_vertices: int,
    *,
    widths: Any,
    colors: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> None:
    """Validate every per-vertex channel against ``n_vertices``. Pure — no I/O.

    Steps 0e-0h of :func:`write_lines`'s fail-fast gate, factored out for the
    same reason and with the same contract as the Points sibling
    :func:`~luxar.io._compiler.geometry_writers.points.validate_points_channels`
    — read that docstring (including the ``image_labels`` / issue #1491
    paragraph on why the finest-child-written-last ``substitutive_lod=``
    wrapper needs this pre-split, not just at the flat writer). ``widths`` is
    required, so it is validated unconditionally and FIRST; all seven channels
    are per-VERTEX, not per-segment.
    """
    # 0e. Shared validator (the Lines sibling of validate_radii_for_writing)
    validate_widths_for_writing(widths, n_vertices)

    # 0f. Pre-flight length sweep over ALL provided per-vertex arrays. The
    # spatial-ordering fancy-indexing below silently TRUNCATES a too-long
    # array and raises a raw IndexError on a too-short one, so lengths must
    # be checked before build_lines_ordering runs. The per-dataset validators
    # further down remain in place (belt and braces).
    if colors is not None:
        if isinstance(colors, np.ndarray):
            # channels=(3, 4): lines accept RGBA since volumetric phase 4
            # (the alpha column is per-vertex opacity) — mirrors points.
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors, "colors")
    if sharpness is not None:
        # Validates arrays AND broadcast scalars (same [0, 1] bounds).
        validate_sharpness_for_writing(sharpness, n_vertices)
    if scalars is not None:
        validate_scalars_preflight(scalars, n_vertices)
    # 0g. Labels: sequence-of-str type + length check (the CSR serializer
    # would otherwise AttributeError on a non-str entry AFTER the arrays
    # were written).
    if labels is not None:
        validate_labels_for_writing(labels, n_vertices)
    # Keys ride the same pre-flight as labels: the CSR serializer
    # UTF-8-encodes each entry, so a non-str or a length mismatch must be
    # caught BEFORE any array reaches disk (#1917).
    if keys is not None:
        validate_labels_for_writing(keys, n_vertices, context="keys", noun="Keys")
    # 0h. Image labels: length (dense) / index bounds (sparse dict) — see
    # validate_image_labels_for_writing for why this moved out of the CSR
    # writer itself.
    if image_labels is not None:
        validate_image_labels_for_writing(image_labels, n_vertices)


def validate_line_indices(indices: Any, n_vertices: int) -> NDArray[Any]:
    """Validate an ``indexed`` edge list and return it normalized. Pure — no I/O.

    The ``line_type == "indexed"`` half of step 0d of :func:`write_lines`'s
    fail-fast gate (everything after the "requires indices array" check),
    factored out because a SECOND caller needs exactly it:
    :func:`luxar.core.group.compositing.validate_line_indices_before_split` runs
    it in the split paths, where the topology builders
    (``lod.lines.identify_polylines`` / ``make_additive_lod_lines``) check only
    dtype and bounds and then reshape to pairs — so an ``(E, 3)`` array was
    silently reinterpreted as ``3E/2`` edges the author never wound, and an
    odd flat count died on a raw numpy reshape instead of the guided message.
    Shared rather than repeated so the two cannot drift (same contract as
    :func:`~luxar.io._compiler.geometry_writers.points.validate_points_channels`).

    Returns:
        ``indices`` as an ndarray, since the writer uses the normalized array
        downstream (a Python-list ``indices`` is a legitimate adder input).
    """
    # Normalize to ndarray first: a Python-list `indices` is a legitimate
    # adder input (passed through un-arrayed), and `.size` / the reshape
    # in convert_to_indexed both AttributeError on a bare list.
    arr: NDArray[Any] = np.asarray(indices)
    # Accept ONLY the two documented layouts — flat (2E,) or pairs
    # (E, 2). An even-size but wrong-width array (e.g. (E, 3)) would
    # otherwise pass the element-count checks below and then silently
    # reshape into bogus edges inside convert_to_indexed.
    if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[1] != 2):
        raise ValueError(
            "Indices must be a flat (2E,) array or an (E, 2) array of "
            f"pairs, got shape {arr.shape}"
        )
    # Accept BOTH accepted layouts — flat (2E,) and pairs (E, 2) —
    # by counting ELEMENTS, not rows: len() on an (E, 2) array counts
    # edges, which wrongly rejected any odd edge count.
    if arr.size < 2:
        raise ValueError("Indexed requires at least 2 indices")
    if arr.size % 2 != 0:
        raise ValueError("Indices must have an even element count (pairs)")
    # Reject non-integer indices: convert_to_indexed casts with
    # `.astype(np.uint32)`, which silently TRUNCATES a float (1.7 -> 1),
    # so a float array would produce edges the user never authored.
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Indices must be an integer array, got dtype {arr.dtype}")
    # Bounds check BOTH ends before convert_to_indexed casts to uint32:
    # a negative index would silently wrap to ~4 billion and blow up
    # with a raw IndexError deep inside the spatial ordering.
    if np.min(arr) < 0:
        raise ValueError(f"Index {np.min(arr)} < 0 (indices must be >= 0)")
    if np.max(arr) >= n_vertices:
        raise ValueError(f"Index {np.max(arr)} >= n_vertices {n_vertices}")
    return arr


def validate_gsplat_inputs(
    centers: PositionArray,
    amplitudes: Union[ScalarArray, float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[ColorArray, tuple, list]] = None,
    *,
    check_values: bool = True,
) -> Tuple[
    PositionArray,
    Union[ScalarArray, float],
    NDArray[np.float32],
    Optional[Union[ColorArray, tuple, list]],
    int,
    int,
    bool,
]:
    """Validate and normalize gsplat inputs.

    ``check_values=False`` skips the O(N) value scans (finiteness / sign /
    color checks) and keeps only the cheap shape normalization. It is for
    callers that already ran the full check on the SAME arrays — the per-level
    write after ``preflight_validate_leaf`` — so each leaf is value-scanned
    exactly once instead of two or three times.

    Returns:
        (centers, amplitudes, cholesky_factors, colors,
         n_splats, n_dims, cholesky_is_uniform)
    """
    if check_values:
        n_splats, n_dims = validate_positions_for_writing(centers)
    else:
        # Shape/finiteness already checked on these arrays by the caller.
        n_splats, n_dims = centers.shape
    expected_k = n_dims * (n_dims + 1) // 2
    cholesky_is_uniform = False

    if cholesky_factors.ndim == 1:
        if cholesky_factors.shape[0] != expected_k:
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({expected_k},), "
                f"got {cholesky_factors.shape}"
            )
        cholesky_factors = cholesky_factors.reshape(1, expected_k)
        cholesky_is_uniform = True
    elif cholesky_factors.shape != (n_splats, expected_k):
        raise ValueError(
            f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
            f"got {cholesky_factors.shape}"
        )

    # Finiteness + positive-diagonal gate (shape is normalized above). Mirrors
    # the radii/widths validators: a NaN or non-positive diagonal used to pass
    # the shape-only check and either die deep in the encoder after centers were
    # written or be silently clamped to a degenerate covariance.
    if check_values:
        validate_cholesky_for_writing(cholesky_factors, n_dims)

    # Validate amplitudes (finiteness first, mirroring radii/widths — a NaN
    # would silently pass `< 0` since `nan < 0` is False and corrupt the store).
    if isinstance(amplitudes, np.ndarray):
        if amplitudes.shape[0] != n_splats:
            raise ValueError(
                f"Amplitudes shape {amplitudes.shape} doesn't match n_splats {n_splats}"
            )
        if check_values:
            _validate_numeric_finite_values(amplitudes, "amplitudes")
            if np.any(amplitudes < 0):
                min_val = float(np.min(amplitudes))
                raise ValueError(
                    f"Amplitudes must be non-negative (>= 0). "
                    f"Found minimum value: {min_val:.3f}"
                )
    elif isinstance(amplitudes, (int, float)):
        if not np.isfinite(amplitudes):
            raise ValueError(f"Amplitude must be finite. Got {amplitudes}")
        if amplitudes < 0:
            raise ValueError(f"Amplitude must be non-negative (>= 0). Got {amplitudes}")

    # Colors: the same check write_gsplat_arrays historically ran POST-write;
    # running it here puts colors in the pre-group gate on every path (flat
    # write_gsplats and the leaf preflight alike). GSplats accept RGBA — the
    # alpha column is per-splat opacity. Broadcast list/tuple colors get the
    # same pre-write gate Points/Lines use (validate_broadcast_color) — a NaN
    # or wrong-length tuple would otherwise be discovered only in write_colors,
    # after centers/amplitudes/Cholesky were already on disk.
    if check_values:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_splats, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors)

    return (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
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
        # Authored cross-layer draw order (higher = nearer the camera = drawn
        # later). Advertised here for the same reason as lines-only ``join``:
        # a real knob a user authors, so a typo must see it in the hint. See
        # ``docs/guides/specs/LAYER_ORDER_SPEC.md``.
        "layer_order",
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
        # Mesh-only material family (``luxar`` | ``physical``) and the physically
        # based knobs it unlocks (Phase 1 surface knobs, Phase 2 glass family,
        # Phase 3's ``refract_data`` flag).
        # Advertised for the same reason as the shading controls below: real
        # knobs an author types, so a typo must see them in the hint. The adder
        # cross-checks them against each other (a physical knob needs
        # ``material="physical"``; a physical mesh refuses the house-shader
        # knobs; the glass knobs need ``transmission > 0``) — see ``adders/mesh.py``.
        "material",
        "roughness",
        "metalness",
        "clearcoat",
        "clearcoat_roughness",
        "iridescence",
        "sheen",
        "sheen_color",
        "transmission",
        "ior",
        "thickness",
        "attenuation_color",
        "attenuation_distance",
        "dispersion",
        "refract_data",
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


def _validate_compositing_attrs(attrs: Dict[str, Any]) -> None:
    """Validate authored compositing controls in the shared attr gate."""
    if "blending_mode" in attrs:
        from .types import validate_blending_mode

        validate_blending_mode(attrs["blending_mode"])

    if "layer_order" in attrs:
        # Runs UNCONDITIONALLY, which keeps ``layer_order`` out of
        # ``ABSENT_WHEN_NONE_RENDER_ATTRS``: a ``layer_order=None`` refuses
        # loudly here (like ``blending_mode=None``) rather than reaching disk.
        from .types import validate_layer_order

        validate_layer_order(attrs["layer_order"])


def _validate_interaction_attrs(attrs: Dict[str, Any]) -> None:
    """Validate element interaction templates in the shared attr gate."""
    # Checked before the node exists on disk because the failure they prevent
    # is otherwise silent: a bad link writes cleanly and simply does nothing
    # when the user clicks it, with no file or console diagnostic (#1917).
    if "link" in attrs:
        from .types import validate_link

        validate_link(attrs["link"])

    if "copy" in attrs:
        from .types import validate_copy_template

        validate_copy_template(attrs["copy"])

    if "link_target" in attrs:
        from .types import validate_link_target

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


def _validate_mesh_appearance_attrs(attrs: Dict[str, Any]) -> None:
    """Validate the mesh appearance controls plus slab tolerance in the shared gate.

    Value validation only. The CROSS-key rules — a physical knob without
    ``material="physical"``, or a house-shader knob with it — live in the mesh
    adder, which is the only caller that also sees the named ``shading`` and
    ``texture`` parameters those rules must judge.
    """
    for key, (validator, label) in _MESH_APPEARANCE_VALIDATORS.items():
        if key in attrs:
            validator(attrs[key], label)


def _validate_normal_pair(
    normals: Any, normal_dims: Any, n_vertices: int, n_dims: int
) -> None:
    """Normals and their companion attr are a PAIR — each is meaningless alone.

    A normal array with no ``normal_dims`` cannot be oriented (storing normals
    against an implicit "first three dimensions" is the bug the attr exists to
    prevent: for a ``(t, x, y, z)`` mesh those are ``(t, x, y)``), and
    ``normal_dims`` with no normals describes nothing.
    """
    from .base import (
        validate_normal_dims_for_writing,
        validate_normals_for_writing,
    )

    if normals is not None:
        validate_normals_for_writing(normals, n_vertices)
        if normal_dims is None:
            raise ValueError(
                "normal_dims is required when normals are supplied: it names "
                "which three dimension indices the 3-component normals describe. "
                "Pass e.g. normal_dims=(0, 1, 2)."
            )
        validate_normal_dims_for_writing(normal_dims, n_dims)
    elif normal_dims is not None:
        raise ValueError(
            "normal_dims was supplied without normals. It names the dimensions "
            "that a normals array describes, so it has no meaning on its own — "
            "pass normals=..., or drop normal_dims."
        )


def _validate_mesh_metadata(shading: Optional[str], double_sided: Any) -> None:
    """Validate mesh metadata that directly controls viewer rendering."""
    # A typo must not reach zarr: an unrecognised shading value would silently
    # take the stored-normal path.
    if shading is not None and shading not in VALID_SHADING_MODES:
        raise ValueError(
            f"shading must be one of {VALID_SHADING_MODES}, got {shading!r}"
        )
    if not isinstance(double_sided, bool):
        raise ValueError(
            f"double_sided must be a bool, got {type(double_sided).__name__}"
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
    # The material family and its physically based knobs (spec
    # MESH_PHYSICAL_MATERIALS_SPEC §3.1). Fractions in ``[0, 1]`` exactly like
    # ``ambient``: three clamps them anyway, but a clamp is silent and an
    # author who wrote ``roughness=5`` meant something.
    "material": (validate_mesh_material, "Material"),
    **{
        key: (validate_appearance_fraction, key.replace("_", " ").capitalize())
        for key in PHYSICAL_MATERIAL_FRACTION_ATTRS
    },
    "sheen_color": (validate_hex_color, "Sheen color"),
    # The Phase 2 glass family (spec §3.4). ``ior`` has three's own bounds;
    # ``thickness`` is a length that may be zero (a thin-walled bubble);
    # ``attenuation_distance`` is a length that may NOT be zero (it divides);
    # ``dispersion`` is unbounded above in three but a value past 1 is a typo
    # for anything that is not a demonstration — still, it is a positive
    # quantity rather than a fraction, so it is only required to be >= 0.
    "ior": (validate_ior, "Ior"),
    "thickness": (validate_non_negative_finite, "Thickness"),
    "attenuation_color": (validate_hex_color, "Attenuation color"),
    "attenuation_distance": (validate_positive_finite, "Attenuation distance"),
    "dispersion": (validate_non_negative_finite, "Dispersion"),
    # Phase 3 (spec §3.4): draw this glass AFTER the emissive data so it refracts
    # the points, lines and splats behind it. A flag, not a fraction; refused
    # without ``transmission > 0`` by the adder's pairing rule.
    "refract_data": (validate_bool_flag, "Refract data"),
}


#: Accepted shading values, public so the core read type can stay pinned to the writer.
VALID_SHADING_MODES = ("smooth", "flat", "none")


# ---------------------------------------------------------------------------
# Render attributes
#
# `validate_render_attrs` and the four reserved-name sets it checks against are
# one unit: the gate is meaningless without the vocabulary, and every caller
# passes one of these four sets. Both halves were in `io._compiler.node_common`,
# and `core.group.adders.{points,lines,gsplats,mesh}` each reached back into it.
# ---------------------------------------------------------------------------

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


def validate_render_attrs(
    attrs: Dict[str, Any],
    reserved_attrs: FrozenSet[str] = frozenset(),
    reject_unknown: bool = True,
) -> None:
    """Validate render attrs that would corrupt a node if written unchecked.

    Called as the FIRST step of every geometry writer — before the zarr group
    is created — so an invalid value fails the write without leaving a partial
    node on disk. Covers every pure attr validator (no store access needed):
    blending_mode / layer_order / absorption / opacity / gamma / intensity /
    offset / mesh appearance / layer / visible / colormap. The values are
    validated only (not converted) — the writer stores the caller's attrs
    unchanged.

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

    _validate_compositing_attrs(attrs)

    if "join" in attrs:
        # The KEY allowlist above catches ``jion=``; this catches ``join="mitre"``.
        # Both matter: an unrecognised style is far more likely a typo than a
        # request for no joins, and the file would otherwise write cleanly and
        # render with the default, giving the author nothing to go on.
        from .types import validate_line_join

        validate_line_join(attrs["join"])

    if "absorption" in attrs:
        from .types import validate_absorption

        validate_absorption(attrs["absorption"])

    if "opacity" in attrs:
        from .types import validate_opacity

        validate_opacity(attrs["opacity"])

    _validate_mesh_appearance_attrs(attrs)

    if "truncation_radius" in attrs:
        from .types import validate_truncation_radius

        validate_truncation_radius(attrs["truncation_radius"])

    if "gamma" in attrs:
        from .types import validate_gamma

        validate_gamma(attrs["gamma"])

    if "intensity" in attrs:
        from .types import validate_intensity

        validate_intensity(attrs["intensity"])

    if "offset" in attrs:
        from .types import validate_offset

        validate_offset(attrs["offset"])

    if "layer" in attrs:
        from .types import validate_layer

        validate_layer(attrs["layer"])

    if "visible" in attrs:
        from .types import validate_visible

        validate_visible(attrs["visible"])

    if "colormap" in attrs and attrs["colormap"] is not None:
        from .types import validate_colormap

        validate_colormap(attrs["colormap"])

    _validate_interaction_attrs(attrs)


def validate_mesh_arrays(
    vertices: Any,
    faces: Any,
    *,
    normals: Any = None,
    normal_dims: Any = None,
    colors: Any = None,
    scalars: Any = None,
    uvs: Any = None,
    texture: Any = None,
    texture_encoding: str = "raw",
    texture_width: Optional[int] = None,
    texture_height: Optional[int] = None,
    texture_channels: Optional[int] = None,
    texture_color_space: str = "srgb",
    texture_ktx2_mode: str = "uastc",
    texture_ktx2_quality: Optional[int] = None,
    texture_ktx2_rdo_l: Optional[float] = None,
    texture_ktx2_zcmp: Optional[int] = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> Tuple[int, int]:
    """Validate a mesh's arrays and channels. Pure — reads nothing, writes nothing.

    Steps 0c-0h of :func:`write_mesh`'s fail-fast gate, factored out because a
    SECOND caller needs exactly them and nothing else:
    ``add_mesh(substitutive_lod=…)`` runs this before it decimates anything and
    before ``add_lod_group`` creates the zarr group. Without that, a malformed
    optional channel that the decimator ALSO validates per level — colours
    with two components, a wrong-length normals array, a typo'd ``shading`` —
    was refused only from inside whichever child's write hit it first
    (typically ``child_0``, the coarsest, since levels are written
    coarsest-to-finest), with the ``kind=lod`` group already on disk: that
    left a childless ladder if the very first child failed, or a
    finest-child-less one if a later level was the one to fail. The
    plain-leaf path writes nothing in the same situation, and the two must
    agree.

    ``labels`` and a wrong-length ``image_labels`` (#1491) fail a DIFFERENT
    way: both are forwarded ONLY to the FINEST child (the original,
    undecimated surface), written LAST, never to a coarse level. Pre-fix, a
    bad one of these was refused deep inside that finest child's own
    ``write_mesh`` call — AFTER its vertices, faces, normals, colours and
    scalars were already on disk — so the ladder was neither childless nor
    finest-child-less: every level, including the finest, was fully written
    and independently loadable. Only the finest level's own label channel
    (the text-label CSR pair, or ``image_label_offsets`` /
    ``image_label_bytes``) was silently absent. That is harder to notice than
    a missing level, not milder — the ladder looks and loads like a
    correctly-authored object that simply carries no labels.

    Sharing the function rather than repeating the checks is what keeps that
    promise true as the rules change: the ladder gate cannot drift from what the
    child write accepts, because it IS what the child write runs.

    Returns:
        ``(n_vertices, n_dims)``, since the caller needs both and only the
        vertex validator can produce them.
    """
    from .base import (
        _texture_decoded_bytes,
        validate_colors_for_writing,
        validate_faces_for_writing,
        validate_labels_for_writing,
        validate_mesh_decode_budget,
        validate_positions_for_writing,
        validate_texture_for_writing,
        validate_uvs_for_writing,
        validate_vertices_for_writing,
    )

    # Vertices shape/finiteness (shared coordinate path), then the mesh-only
    # vertex-count ceiling. Order matters: the cap reads shape[0], which is only
    # meaningful once the array is known to be 2D.
    n_vertices, n_dims = validate_positions_for_writing(vertices, context="vertices")
    validate_vertices_for_writing(vertices)
    # Faces: layout, integer dtype, and both index bounds. Runs before the
    # writer's uint32 cast, which is what makes the bounds check meaningful.
    validate_faces_for_writing(faces, n_vertices)
    _validate_normal_pair(normals, normal_dims, n_vertices, n_dims)
    if uvs is not None:
        validate_uvs_for_writing(uvs, n_vertices)
    _validate_mesh_metadata(shading, double_sided)
    # Optional per-vertex channels.
    if colors is not None:
        if isinstance(colors, np.ndarray):
            # channels=(3, 4): the optional 4th component is per-vertex opacity,
            # load-bearing in every blending mode — mirrors points/lines.
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors, "colors")
    if scalars is not None:
        validate_scalars_preflight(scalars, n_vertices)
    if labels is not None:
        validate_labels_for_writing(labels, n_vertices)
    # Keys ride the same pre-flight as labels: the CSR serializer
    # UTF-8-encodes each entry, so a non-str or a length mismatch must be
    # caught BEFORE any array reaches disk (#1917).
    if keys is not None:
        validate_labels_for_writing(keys, n_vertices, context="keys", noun="Keys")
    if image_labels is not None:
        validate_image_labels_for_writing(image_labels, n_vertices)
    # Last, because it needs every channel's presence and the validated shapes.
    # A store over the viewer's per-node ceiling does not render at all, so it is
    # refused here rather than in a browser (#2145).
    texture_decoded_bytes = 0
    if texture is not None:
        texture_height, texture_width, texture_channels = validate_texture_for_writing(
            texture,
            texture_encoding,
            texture_width,
            texture_height,
            texture_channels,
            texture_color_space,
            ktx2_mode=texture_ktx2_mode,
            ktx2_quality=texture_ktx2_quality,
            ktx2_rdo_l=texture_ktx2_rdo_l,
            ktx2_zcmp=texture_ktx2_zcmp,
        )
        texture_decoded_bytes = _texture_decoded_bytes(
            texture_encoding, texture_width, texture_height, texture_channels
        )
    validate_mesh_decode_budget(
        n_vertices,
        n_dims,
        int(np.asarray(faces).size // 3),
        normals=normals,
        colors=colors,
        scalars=scalars,
        uvs=uvs,
        texture_decoded_bytes=texture_decoded_bytes,
    )
    return n_vertices, n_dims
