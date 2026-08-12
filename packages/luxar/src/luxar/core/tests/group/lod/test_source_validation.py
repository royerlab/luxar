"""Per-element channels are validated against the SOURCE count, pre-split (#1437).

The LOD-wrapper half of the gate (the partition half lives in
``tests/group/partition/test_source_validation.py``, which also explains why the
parity assertion is exact-message rather than a substring match).
``slice_optional_array`` passes a value whose leading length does not match the
element count through UNCHANGED — that is how a broadcast RGB triple or a scalar
radius reaches every level — so a per-element array of the WRONG length is handed
to every level whole, and a level whose own count happens to equal that array's
length ACCEPTS it. Each additive case below is sized so that coincidence holds
(explicit CUMULATIVE ``counts`` give two equal levels), which is what makes the
per-level check insufficient.

The substitutive wrappers are the shape where a downstream check does eventually
reject (their finest child carries the FULL element set), so those are pinned by
WHERE the write fails: the finest child is written LAST, after ``add_lod_group``
and every coarse gsplat level, so without the gate a wrong-length channel strands
a partial ``kind=lod`` node on disk.

Immediately below the per-channel cases above sits a THIRD section, inserted
between them and the dimension-count one: ``image_labels``, pre-split, on the
Points/Lines ``substitutive_lod=`` wrapper specifically (#1491). It does not
fit the ``slice_optional_array`` shape above at all — it has no per-level
SLICER in the first place, since it is forwarded only to the finest child,
written LAST — so a wrong-length or out-of-range-sparse value used to be
refused only after that finest child's OTHER arrays (and every coarser gsplat
level) were already committed, leaving a complete, loadable ladder silently
missing only its images. It earns its own section (message, non-double-
prefixing, sparse form, and a decoding control) rather than joining the
parametrized lists above.

The next section covers the SCENE-DIMENSION COUNT gate (#1446), which is the
same bug class one validator over: the count check sat below the LOD
branches, so an additive ladder wrote ``additive_<i>`` nodes whose column count
contradicted the scene's dimensions, and the substitutive / ``lod_group=``
wrappers refused only from inside ``child_0`` — after the ``kind=lod`` group was
already on disk. Each case asserts message parity with the flat call AND an empty
store; the controls assert the hoist did not multiply the per-dimension range
``UserWarning`` across levels.

The fourth and last section is the same bug class one step further out (#1471),
and it is where the pattern breaks: ``labels`` / ``image_labels`` on a
multi-CHILD gsplats wrapper cannot be hoisted at all, only REFUSED. Every
substitutive level is its own set of merged representative splats with its own
count, so no single list has a per-element correspondence to slice — which is
why those cases assert a new message rather than parity with a flat one, and why
the precedence tests matter more here: the refusal has no flat counterpart, so it
must sit BELOW every check that does.

A note on ``assert_same_refusal`` / the ``"child_" not in message``
idiom used throughout this file (#1491): now that :func:`funnel_add_error`
un-nests unconditionally for a SAME-geometry recursive child, the absence of
``"child_"`` in a message is no longer, by itself, evidence that the gate fired
EARLY (pre-split, nothing written). A late refusal from inside a same-kind
child (e.g. the Mesh ``substitutive_lod=`` ladder's own ``child_3``) is un-nested
just the same, so it ALSO reads without ``"child_"`` in it — only the
accompanying store assertion (nothing / only the expected node persisted)
still discriminates early from late.

The fifth and last section is the NODE-ATTRS gate. It started (#1529) as a
Points/Lines-only variant of the #1437 bug and covered all THREE Points/Lines
split paths, not only substitutive_lod=: substitutive_lod= and partition= both
forward the non-compositing remainder of the caller's ``**attrs`` to a
synthesised child (a gsplats ``child_0``, or a ``part_i``), and additive_lod=
goes straight to the multi-LOD writer, which calls this same validator with NO
reserved-attrs set at all. So an attrs key the flat writer would reject up
front — either a key ``GSPLATS_RESERVED_ATTRS`` reserves but
``POINTS_RESERVED_ATTRS`` / ``LINES_RESERVED_ATTRS`` do not
(``amplitude_range=``), a plain unknown-key typo (``blending=``), or
(additive_lod= only) a genuinely points/lines-reserved key misreported as
*unknown* instead of *reserved* (``ordering=``, ``max_radius=``/
``max_width=``) — used to be refused only from inside the first child, by
which point the wrapper group itself (childless: zero coarse levels / parts)
was already on disk; a ``position_bounds=`` collision under additive_lod= did
not even raise at all — real ladder data was written and the writer's own
stamp silently clobbered, breaking ``finalize()`` later with an unrelated
``ValueError``.

#1534 then extended the same gate to Mesh and GSplats, which had NOT gained it
under #1529: their own ``partition=`` paths still stranded a childless
``kind=partition`` node on a bad attr, exactly the #1529 shape one geometry
type over — see ``tests/group/partition/test_source_validation.py`` for those
cases. Mesh's ``substitutive_lod=`` wrapper was the one path that already ran
this validator pre-#1534 (with the right reserved set, ``MESH_RESERVED_ATTRS``)
— but it ran the check AFTER resolving its own ``extend_to_all``, the opposite
of the Points/Lines order, and its ``additive_lod=`` reveal ladder ran no
attrs check of its own at all (falling through to ``write_mesh_multi_lod``'s
unreserved one, same shape as the Points/Lines additive bug — and, like that
bug, with BOTH of its symptoms: a mesh-``RESERVED`` key like ``ordering=`` came
back *unknown* rather than *reserved* (still refused, nothing written), while
``position_bounds=`` — reserved for mesh but ALSO listed in
``_ALLOWED_NODE_ATTRS`` — did not raise at all: a real ladder was written and
the writer's own stamp was silently clobbered, breaking ``finalize()`` later
with an unrelated ``ValueError``. See the comment above
``TestMeshAdditiveNodeAttrsGate`` below for the measured repro). #1534 moved
the check to the top of ``add_mesh_impl``, above every one of its three
structural branches, which fixes all three: partition= no longer strands,
neither additive_lod= symptom survives, and every branch (including the flat
fall-through) now agrees with Points/Lines on ordering against
``extend_to_all`` too — see the ``TestMeshFlatPathNodeAttrsGateOutranksExtendToAll``
and ``TestMeshSubstitutiveNodeAttrsGateOutranksExtendToAll`` classes below.
GSplats' only structural door on ``add_gsplats`` is ``partition=`` (its
``lod_group=``/``additive_lod=`` doors live on the separate
``add_gsplats_from_data`` adder — out of scope for THIS section, but see the
dedicated ``TestGSplatsFromDataNodeAttrsGate`` section further down this same
file, which closes the ``lod_group=`` half of the same stranding class);
#1534 hoists the same check above it.

A round-2 review finding on that same ``lod_group=`` gate: its ``labels``/
``image_labels`` exclusion list was incomplete. ``partition`` sits in the
identical position — a named parameter of the LEAF ``Group.add_gsplats`` that
is NOT a parameter of ``add_gsplats_from_data_impl``, so it arrives inside
``**attrs`` here and must ride through, unexcluded, to drive each child's own
BSP split — and a previously-working ``add_gsplats_from_data(..., lod_group=
<...>, partition={...})`` call started answering ``Unknown node attribute
'partition'. Did you mean 'absorption'?`` with nothing written. Fixed by
adding ``partition`` to the exclusion set; see
``TestGSplatsFromDataNodeAttrsGateStillForwardsPartition`` further down this
file. One consequence of this gate now covering the ``lod_group=`` door at
all: a bad ``colormap``/``opacity`` VALUE (not just an unknown/reserved KEY)
now outranks the dedicated labels refusal below, matching the flat path's own
precedence between its attrs gate and its labels handling — desirable parity,
documented at the ``_reject_before_wrapper`` call site in ``from_data.py``,
not an accident. This does not close every gap in this door. A
present-but-conflicting ``colors=`` / ``truncation_radius=None`` reaching
``**attrs`` here is tracked separately in #1496 and deliberately left alone.
Nor does the exclusion of ``partition`` extend to its VALUE: an invalid
partition spec (``partition="nonsense"``, ``partition={"max_elements": 0}``)
is still refused only from inside ``child_0``, after the ``kind=lod`` wrapper
is on disk — the spec check lives in the leaf adder, one level below this
gate, so it is the same stranding shape one door over and needs its own
change, not a wider exclusion here.

Unlike the #1437/#1446 sections this is not a per-element SLICE hoisted
upward; it is the SAME pure attrs validator (``validate_render_attrs``) the
flat writer already calls, run once more, earlier, on the un-split caller
attrs — so every split path refuses byte-identically to the flat path, and the
flat path simply validates twice (idempotent), which also means a multi-fault
flat call now reports the attrs fault where it used to report an
``extend_to_all`` fault, on all four geometry types this file covers. Two
precedence controls follow the #1437/#1446 pattern: the colours/colormap gate
still outranks this one, on every path (unchanged); and this gate now itself
outranks the #1437 channel gate at the top of the partition/substitutive/
multi-LOD wrappers — the flat writer's own order (attrs before channels) is
what the split paths now match, where before #1529/#1534 they did not. A THIRD
set of placement tests below pins that this gate lives at the ADDER ENTRY, not
duplicated at the top of each wrapper: an implementation that instead put the
same check at the top of the partition/substitutive wrappers would pass every
message-parity test above but would still let that wrapper's own kwarg-spec
check (a malformed ``partition=`` rule, an invalid ``substitutive_lod=``
``compression_factor``) run first, and on the flat path would still let
``extend_to_all`` resolution run first — only a call that trips both faults at
once, and asserts the ADDER's ordering wins, can tell the two implementations
apart.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pytest
import zarr

from luxar.io.reader import LuxarScene

from ..conftest import (
    IMAGE_LABELS,
    LABEL_KWARGS,
    LABELS,
    N_LABELLED,
    assert_same_refusal,
    assert_uniform,
    bad_ndim_positions,
    cholesky_rows_nd,
    count_range_warnings,
    finalized_group_keys,
    grid_mesh,
    int64_rgb,
    open_ranged_scene,
    open_scene,
    random_positions,
    refusal,
)

# 100 elements with cumulative counts [50, 100] gives two levels of 50, so a
# 50-long channel matches every level and only a check against the full 100 can
# reject it (the same construction the #1422 labels tests use).
_N = 100
_HALF = 50

# Polyline units for Lines: with line_type="segments" each segment is a 2-vertex
# polyline, so cumulative [25, 50] gives two levels of 50 VERTICES each.
_LINES_LADDER = {"counts": [_HALF // 2, _N // 2]}
_POINTS_LADDER = {"counts": [_HALF, _N]}

_POINTS_CASES = [
    ("colors", np.zeros((_HALF, 3), dtype=np.float32)),
    ("radii", np.full(_HALF, 0.5, dtype=np.float32)),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32)),
]

_LINES_CASES = [
    ("widths", np.full(_HALF, 0.2, dtype=np.float32)),
    ("colors", np.zeros((_HALF, 3), dtype=np.float32)),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32)),
]

# The substitutive wrappers need enough elements for the gsplat reduce to
# synthesise coarse levels (a degenerate input falls back to a flat node).
_SUB_N = 400
_SUB_HALF = 200

_POINTS_SUB_CASES = [
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32)),
    ("radii", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32)),
]

_LINES_SUB_CASES = [
    ("widths", np.full(_SUB_HALF, 0.2, dtype=np.float32)),
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32)),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32)),
]


def _points_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    # Assigned last and explicitly, never as a duplicate dict-literal key.
    kwargs[channel] = value
    return kwargs


def _lines_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"widths": 0.2}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    kwargs[channel] = value
    return kwargs


#: The uniform colour forms the flat path accepts, all four of which the
#: substitutive wrappers used to refuse downstream in the gsplat lift (#1444),
#: each paired with the row EVERY level must end up carrying — alpha included,
#: because gsplats carry per-splat alpha and all three shaders scale intensity
#: by it (dropping it would make the coarse levels 1/alpha too bright the
#: instant the ladder switches off the finest child).
_BROADCAST_RGB = (0.25, 0.5, 1.0)
_BROADCAST_ALPHA = 0.5
_BROADCAST_COLORS = [
    (_BROADCAST_RGB, _BROADCAST_RGB),  # RGB tuple
    (np.array([_BROADCAST_RGB], dtype=np.float32), _BROADCAST_RGB),  # (1, 3) row
    (  # RGBA tuple
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
    ),
    (  # (1, 4) row
        np.array([(*_BROADCAST_RGB, _BROADCAST_ALPHA)], dtype=np.float32),
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
    ),
    (  # opaque RGBA — above the clamp, see _assert_coarse_levels_carry_color
        (*_BROADCAST_RGB, 1.0),
        (*_BROADCAST_RGB, 1.0),
    ),
    (  # just BELOW the clamp: still bit-exact, which the tolerance split pins
        (*_BROADCAST_RGB, 0.99),
        (*_BROADCAST_RGB, 0.99),
    ),
]

#: The merge round-trips a per-splat alpha through optical depth, which caps it
#: at ``ALPHA_CLAMP = 511/512``: an authored alpha ABOVE that comes back clamped
#: on every coarse level (measured: 1.0 → 0.998046875, 0.999 → 0.998046875),
#: a step the finest child does not have. Anything at or below the clamp — RGB,
#: alpha 0.5, alpha 0.99 — is bit-exact and is asserted as such, so this
#: tolerance is reserved for the clamped case and cannot absorb a future drift
#: elsewhere.
_ALPHA_CLAMP = 511.0 / 512.0
_ALPHA_CLAMP_ATOL = 2.5e-3
_EXACT_ATOL = 1e-6

#: Colours whose dtype the leaf write refuses (COLOR arrays are floating, uint8
#: or uint16), in both the uniform-row and per-element shapes. Normalising
#: either would bake a near-black coarse level that the encoder then rejects at
#: the finest child — after the coarse levels are on disk. Nothing may be
#: written for them.
_BAD_DTYPE_COLORS = [
    ("uniform row", np.array([[255, 0, 0]], dtype=np.int64)),
    ("per-element", np.tile([255, 0, 0], (_SUB_N, 1)).astype(np.int64)),
]


def _assert_coarse_levels_carry_color(
    reader: LuxarScene, node: str, coarse: "list[str]", want: Any
) -> None:
    """Every coarse gsplat level of a substitutive group carries the authored colour.

    A uniform colour is exactly the case a coarse level can honour trivially
    (every merged representative is that same colour), so this is an equality
    check, not a "some colour was written" one — and it covers the alpha column,
    whose loss would be a brightness jump at the LOD seam rather than a refusal.
    Equality is EXACT unless the authored alpha exceeds the merge's
    optical-depth :data:`_ALPHA_CLAMP`, the only value the round-trip changes.
    """
    alpha = want[3] if len(want) == 4 else None
    atol = (
        _ALPHA_CLAMP_ATOL if alpha is not None and alpha > _ALPHA_CLAMP else _EXACT_ATOL
    )
    assert coarse, "no coarse gsplat levels were written"
    for child in coarse:
        data = reader.get_gsplats(f"{node}/{child}")
        assert data.colors is not None, f"{child} lost its colours"
        n = int(np.asarray(data.centers).shape[0])
        assert np.asarray(data.colors).shape[1] == len(want), (
            f"{child} carries {np.asarray(data.colors).shape[1]} channels, "
            f"expected {len(want)} (a dropped alpha renders 1/alpha too bright)"
        )
        assert_uniform(data.colors, want, n, atol=atol)


def _n_levels(path: str, node: str) -> int:
    store = zarr.open_group(path, mode="r")
    n_levels = int(store[node].attrs["n_additive_sublods"])
    assert n_levels > 1, "the ladder did not fire, so this tests the flat path"
    return n_levels


class TestPointsAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _POINTS_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_add_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"points_add_flat_{channel}.luxar.zarr")
        positions = random_positions(_N, seed=31)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, **_points_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                additive_lod=_POINTS_LADDER,
                **_points_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "p" not in compiler.store


class TestLinesAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _LINES_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_add_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lines_add_flat_{channel}.luxar.zarr")
        vertices = random_positions(_N, seed=32)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                **_lines_kwargs(channel, value),
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                additive_lod=_LINES_LADDER,
                **_lines_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "line" not in compiler.store


class TestPointsSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _POINTS_SUB_CASES)
    def test_wrong_length_channel_is_refused_before_anything_is_written(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_sub_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"points_sub_flat_{channel}.luxar.zarr")
        positions = random_positions(_SUB_N, seed=41)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, **_points_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                substitutive_lod=True,
                **_points_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        # The gate runs above add_lod_group, so no partial kind=lod node is left.
        assert "p" not in compiler.store


class TestLinesSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _LINES_SUB_CASES)
    def test_wrong_length_channel_is_refused_before_anything_is_written(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_sub_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lines_sub_flat_{channel}.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=42)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                **_lines_kwargs(channel, value),
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                substitutive_lod=True,
                **_lines_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "line" not in compiler.store


# ---------------------------------------------------------------------------
# image_labels, pre-split, on the Points/Lines substitutive wrapper (#1491)
# ---------------------------------------------------------------------------
#
# image_labels does not fit the _POINTS_SUB_CASES / _LINES_SUB_CASES shape
# above: it is forwarded ONLY to the finest child (never sliced/broadcast),
# and its own error wording ("Image labels length …") does not contain the
# parameter's name the way the other channels' does — so it gets its own
# section rather than joining those parametrized lists.
#
# Raw bytes blobs (not PIL images / ndarrays), matching the CSR unit tests in
# io/tests/_compiler/test_labels.py, so these tests exercise the length/index
# gate without depending on Pillow being installed.

_IMG_SUB_SHORT = [f"blob-{i}".encode() for i in range(_SUB_HALF)]  # 200, wrong
_IMG_SUB_FULL = [f"blob-{i}".encode() for i in range(_SUB_N)]  # 400, matches


class TestPointsSubstitutiveImageLabelsSourceValidation:
    def test_wrong_length_is_refused_before_anything_is_written(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, "points_sub_img_short.luxar.zarr")
        flat_compiler, flat_scene, _ = open_scene(
            tmp_path, "points_sub_img_short_flat.luxar.zarr"
        )
        positions = random_positions(_SUB_N, seed=91)

        flat = refusal(
            lambda: flat_scene.add_points("p", positions, image_labels=_IMG_SUB_SHORT)
        )
        split = refusal(
            lambda: scene.add_points(
                "p", positions, image_labels=_IMG_SUB_SHORT, substitutive_lod=True
            )
        )

        assert_same_refusal(flat, split)
        assert "Image labels length (200) must match element count (400)" in str(split)
        # The gate runs above the lift/add_lod_group, so no coarse gsplat level
        # (and no childless wrapper) is left — the pre-fix strand this closes.
        assert "p" not in compiler.store
        # The FLAT writer's own new step-0f gate must refuse just as cleanly —
        # pre-fix this call still wrote positions/radii/chunk_bounds before
        # write_image_labels_csr's inline check caught it. Bound and checked
        # here rather than discarded, or this half of the gate is unpinned.
        assert "p" not in flat_compiler.store

    def test_message_is_not_double_prefixed(self, tmp_path: Any) -> None:
        """Pre-fix: ``Could not add points 'p': Could not add points 'child_3': …``.

        ``child_3`` is the auto-generated finest-child name (three coarse gsplat
        levels precede it here) — an internal node the caller never typed. The
        outer funnel must un-nest that before re-raising, so the message names
        only the node the caller passed.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_sub_img_prefix.luxar.zarr")
        positions = random_positions(_SUB_N, seed=92)

        split = refusal(
            lambda: scene.add_points(
                "p", positions, image_labels=_IMG_SUB_SHORT, substitutive_lod=True
            )
        )

        message = str(split)
        assert message.startswith("Could not add points 'p': ")
        assert "child_" not in message
        assert message.count("Could not add") == 1
        assert "p" not in compiler.store

    def test_out_of_range_sparse_index_is_refused_before_anything_is_written(
        self, tmp_path: Any
    ) -> None:
        """The dict (sparse) form is checked pre-split too, not only the length."""
        compiler, scene, _ = open_scene(tmp_path, "points_sub_img_sparse.luxar.zarr")
        positions = random_positions(_SUB_N, seed=93)

        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                image_labels={_SUB_N: b"blob"},
                substitutive_lod=True,
            )
        )

        assert f"Image label index {_SUB_N} out of range [0, {_SUB_N})" in str(split)
        assert "child_" not in str(split)
        assert "p" not in compiler.store

    def test_valid_image_labels_still_reach_the_finest_child_intact(
        self, tmp_path: Any
    ) -> None:
        """The control: a correctly-sized ``image_labels`` still writes end to end.

        The new gate must not reject legal input. The coarse gsplat levels carry
        no ``image_labels`` channel at all (only the finest — original — Points
        child does), so the ladder is complete once that child's flag is set.
        "Intact" is checked by DECODING the finest child's CSR pair and comparing
        it against the authored blobs byte-for-byte, not merely by the presence
        of the ``has_image_labels`` flag.
        """
        from luxar.io.tests.test_image_labels import _decode_image_labels_from_zarr

        compiler, scene, path = open_scene(tmp_path, "points_sub_img_ok.luxar.zarr")
        positions = random_positions(_SUB_N, seed=94)

        scene.add_points(
            "p", positions, image_labels=_IMG_SUB_FULL, substitutive_lod=True
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["p"].attrs["kind"] == "lod"
        children = sorted(store["p"].group_keys())
        assert len(children) > 1
        finest = children[-1]
        assert store["p"][finest].attrs["type"] == "points"
        assert store["p"][finest].attrs["has_image_labels"] is True
        assert store["p"][finest].attrs["n_points"] == _SUB_N
        for coarse in children[:-1]:
            assert store["p"][coarse].attrs.get("has_image_labels") is not True
        # Sorted rather than positional: the finest child's own spatial ordering
        # (Morton/Hilbert, on by default) permutes both positions and their
        # image labels together, so the ON-DISK index of a given blob need not
        # match its index in the authored list — only the SET of blobs must
        # survive intact.
        decoded = _decode_image_labels_from_zarr(path, f"p/{finest}")
        assert sorted(decoded) == sorted(_IMG_SUB_FULL)


class TestLinesSubstitutiveImageLabelsSourceValidation:
    def test_wrong_length_is_refused_before_anything_is_written(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_img_short.luxar.zarr")
        flat_compiler, flat_scene, _ = open_scene(
            tmp_path, "lines_sub_img_short_flat.luxar.zarr"
        )
        vertices = random_positions(_SUB_N, seed=95)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                widths=0.2,
                image_labels=_IMG_SUB_SHORT,
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                widths=0.2,
                image_labels=_IMG_SUB_SHORT,
                substitutive_lod=True,
            )
        )

        assert_same_refusal(flat, split)
        assert "Image labels length (200) must match element count (400)" in str(split)
        assert "line" not in compiler.store
        # The FLAT writer's own new step-0h gate must refuse just as cleanly —
        # see the Points twin above for the pre-fix strand this pins.
        assert "line" not in flat_compiler.store

    def test_message_is_not_double_prefixed(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_img_prefix.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=96)

        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                widths=0.2,
                image_labels=_IMG_SUB_SHORT,
                substitutive_lod=True,
            )
        )

        message = str(split)
        assert message.startswith("Could not add lines 'line': ")
        assert "child_" not in message
        assert message.count("Could not add") == 1
        assert "line" not in compiler.store

    def test_valid_image_labels_still_reach_the_finest_child_intact(
        self, tmp_path: Any
    ) -> None:
        """The control: a correctly-sized ``image_labels`` still writes end to end.

        "Intact" is checked by DECODING the finest child's CSR pair and
        comparing it against the authored blobs, not merely the presence of
        the ``has_image_labels`` flag — see the Points twin for why the
        comparison is a SET (``sorted(...)``) rather than positional.
        """
        from luxar.io.tests.test_image_labels import _decode_image_labels_from_zarr

        compiler, scene, path = open_scene(tmp_path, "lines_sub_img_ok.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=97)

        scene.add_lines(
            "line",
            vertices,
            line_type="segments",
            widths=0.2,
            image_labels=_IMG_SUB_FULL,
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["line"].attrs["kind"] == "lod"
        children = sorted(store["line"].group_keys())
        assert len(children) > 1
        finest = children[-1]
        assert store["line"][finest].attrs["type"] == "lines"
        assert store["line"][finest].attrs["has_image_labels"] is True
        assert store["line"][finest].attrs["n_vertices"] == _SUB_N
        for coarse in children[:-1]:
            assert store["line"][coarse].attrs.get("has_image_labels") is not True
        decoded = _decode_image_labels_from_zarr(path, f"line/{finest}")
        assert sorted(decoded) == sorted(_IMG_SUB_FULL)

    def test_out_of_range_sparse_index_is_refused_before_anything_is_written(
        self, tmp_path: Any
    ) -> None:
        """The Lines twin of the Points sparse-dict case above.

        The dict (sparse) form is checked pre-split too, not only the length —
        this was the only geometry missing this case (#1491 review).
        """
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_img_sparse.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=99)

        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                widths=0.2,
                image_labels={_SUB_N: b"blob"},
                substitutive_lod=True,
            )
        )

        assert f"Image label index {_SUB_N} out of range [0, {_SUB_N})" in str(split)
        assert "child_" not in str(split)
        assert "line" not in compiler.store


# ---------------------------------------------------------------------------
# image_labels on the FLAT GSplats writer (#1491) — GSplats has no
# substitutive_lod= wrapper of its own (a gsplat leaf IS the coarse-level
# representation the other three geometry types lift into), so its
# image_labels check lives only in write_gsplats' own step-0e gate, never in a
# pre-split gate. That inline gate had ZERO coverage anywhere before this: no
# test exercised a wrong-length or an out-of-range sparse image_labels on a
# direct add_gsplats call.
# ---------------------------------------------------------------------------


class TestGSplatsImageLabelsFlatGate:
    def test_wrong_length_is_refused_with_nothing_written(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "gsplats_img_short.luxar.zarr")
        centers = random_positions(_SUB_N, seed=101)
        cholesky = cholesky_rows_nd(_SUB_N, 3)
        amplitudes = np.ones(_SUB_N, dtype=np.float32)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                image_labels=_IMG_SUB_SHORT,
            )
        )

        assert "Image labels length (200) must match element count (400)" in str(exc)
        assert "g" not in compiler.store

    def test_out_of_range_sparse_index_is_refused_with_nothing_written(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, "gsplats_img_sparse.luxar.zarr")
        centers = random_positions(_SUB_N, seed=102)
        cholesky = cholesky_rows_nd(_SUB_N, 3)
        amplitudes = np.ones(_SUB_N, dtype=np.float32)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                image_labels={_SUB_N: b"blob"},
            )
        )

        assert f"Image label index {_SUB_N} out of range [0, {_SUB_N})" in str(exc)
        assert "g" not in compiler.store

    def test_valid_image_labels_write_normally(self, tmp_path: Any) -> None:
        """The control: a correctly-sized ``image_labels`` still writes end to end."""
        compiler, scene, path = open_scene(tmp_path, "gsplats_img_ok.luxar.zarr")
        centers = random_positions(_SUB_N, seed=103)
        cholesky = cholesky_rows_nd(_SUB_N, 3)
        amplitudes = np.ones(_SUB_N, dtype=np.float32)

        scene.add_gsplats(
            "g",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            image_labels=_IMG_SUB_FULL,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["has_image_labels"] is True
        assert store["g"].attrs["n_splats"] == _SUB_N


# 24 vertices as 12 edges (see the partition sibling for why this shape).
_N_IDX = 24
_N_EDGES = 12
_INDEX_CASES = [
    ("bad_layout", np.arange(_N_IDX, dtype=np.uint32).reshape(8, 3)),
    ("odd_flat", np.arange(_N_IDX - 1, dtype=np.uint32)),
]
_INDEX_SPLITS = [
    ("additive", {"additive_lod": {"n_lods": 3}}),
    ("substitutive", {"substitutive_lod": True}),
]


class TestLinesIndicesLodValidation:
    """The LOD half of the ``indices`` gate (the partition half is in partition/).

    ``make_additive_lod_lines`` checks dtype and bounds and then reshapes to
    pairs, so a malformed edge list was reinterpreted rather than refused; the
    substitutive path refused it only from the finest child, with the coarse
    levels already on disk.
    """

    @pytest.mark.parametrize("case,indices", _INDEX_CASES)
    @pytest.mark.parametrize("split,split_kwargs", _INDEX_SPLITS)
    def test_malformed_indices_refused_exactly_as_the_flat_path(
        self,
        tmp_path: Any,
        case: str,
        indices: Any,
        split: str,
        split_kwargs: Dict[str, Any],
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"idx_{split}_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"idx_flat_{split}_{case}.luxar.zarr")
        vertices = random_positions(_N_IDX, seed=15)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, indices=indices, line_type="indexed"
            )
        )
        split_exc = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                indices=indices,
                line_type="indexed",
                **split_kwargs,
            )
        )

        assert_same_refusal(flat, split_exc)
        assert "line" not in compiler.store

    @pytest.mark.parametrize(
        "layout,indices",
        [
            ("pairs", np.arange(_N_IDX, dtype=np.uint32).reshape(-1, 2)),
            ("flat", np.arange(_N_IDX, dtype=np.uint32)),
        ],
    )
    def test_legal_layouts_still_ladder_with_every_edge_intact(
        self, tmp_path: Any, layout: str, indices: Any
    ) -> None:
        """The control: both documented layouts ladder, and keep all 12 edges.

        Edge count is the assertion that matters — an ``(E, 3)`` array
        reinterpreted as ``3E/2`` edges is the bug this gate exists for, which a
        mere "the node exists" check cannot see.
        """
        compiler, scene, path = open_scene(tmp_path, f"idx_ok_add_{layout}.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N_IDX, seed=16),
            widths=0.2,
            indices=indices,
            line_type="indexed",
            additive_lod={"n_lods": 3},
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        store = zarr.open_group(path, mode="r")
        total_segments = sum(
            int(store["line"][f"additive_{i}"].attrs["n_segments"])
            for i in range(n_levels)
        )
        assert total_segments == _N_EDGES


class TestLegalBroadcastFormsStillReachEveryLevel:
    """The negative controls: a legal broadcast still reaches every LOD level.

    A case the plain leaf accepts must not be refused by the ladder, and the
    values must land on the right elements — so each control reads every level
    back through the decoder.
    """

    def test_points_ladder_broadcast_channels(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_add_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_N, seed=51),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),  # (1, 3) row
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            additive_lod=_POINTS_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "p")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_points(f"p/additive_{i}")
            n_level = data.positions.shape[0]
            total += n_level
            assert_uniform(data.colors, [1.0, 0.0, 0.0], n_level)
            assert_uniform(data.radii, 0.5, n_level)
        assert total == _N

    def test_points_ladder_rgb_triple_on_a_three_point_node(
        self, tmp_path: Any
    ) -> None:
        """The count collision, in the ladder: 3 points, a 3-component RGB.

        Without the broadcast classification the triple satisfies
        ``slice_optional_array``'s length test and each level receives a slice of
        its COMPONENTS instead of the authored colour.
        """
        compiler, scene, path = open_scene(tmp_path, "points_add_rgb3.luxar.zarr")
        positions = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_points(
            "p", positions, colors=[0.25, 0.5, 1.0], additive_lod={"counts": [1, 3]}
        )
        compiler.finalize()

        n_levels = _n_levels(path, "p")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_points(f"p/additive_{i}")
            n_level = data.positions.shape[0]
            total += n_level
            assert_uniform(data.colors, [0.25, 0.5, 1.0], n_level)
        assert total == 3

    def test_lines_ladder_rgba_quadruple_on_a_four_vertex_node(
        self, tmp_path: Any
    ) -> None:
        """The Lines ladder's count collision — the last uncovered wrapper cell.

        4 vertices as two 2-vertex segment polylines, laddered one polyline per
        level, with a 4-component uniform RGBA whose own length equals the vertex
        count. Without the classifier in ``add_lines_multi_lod_wrapper_impl`` the
        tuple is gathered and each level gets a 2-element slice of the
        COMPONENTS: measured, that raises
        ``colors: Expected shape (2, 3) or (1, 3), got (2,)``.
        """
        compiler, scene, path = open_scene(tmp_path, "lines_add_rgba4.luxar.zarr")
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        scene.add_lines(
            "line",
            vertices,
            widths=0.2,
            colors=(0.25, 0.5, 1.0, 0.5),
            line_type="segments",
            additive_lod={"counts": [1, 2]},
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert data.colors.shape[-1] == 4
            assert_uniform(data.colors, [0.25, 0.5, 1.0, 0.5], n_level)
        assert total == 4

    def test_lines_ladder_broadcast_channels(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_add_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=52),
            widths=0.3,  # scalar
            colors=(0.25, 0.5, 1.0),  # uniform RGB
            sharpness=0.7,  # scalar
            line_type="segments",
            additive_lod=_LINES_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert_uniform(data.colors, [0.25, 0.5, 1.0], n_level)
            assert_uniform(data.widths, 0.3, n_level)
        assert total == _N

    def test_lines_ladder_broadcast_width_array(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_add_width1.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=53),
            widths=np.array([0.25], dtype=np.float32),  # (1,) broadcast
            line_type="segments",
            additive_lod=_LINES_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert_uniform(data.widths, 0.25, n_level)
        assert total == _N

    @pytest.mark.parametrize("colors,want", _BROADCAST_COLORS)
    def test_points_substitutive_broadcast_channels(
        self, tmp_path: Any, colors: Any, want: Any
    ) -> None:
        # Uniform ``colors`` INCLUDED: every broadcast form the flat path accepts
        # now reaches disk under ``substitutive_lod=`` too — the lift broadcasts
        # it to the coarse gsplat levels instead of refusing it (#1444), alpha
        # and all, so every level renders at the authored opacity.
        compiler, scene, path = open_scene(tmp_path, "points_sub_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_SUB_N, seed=54),
            colors=colors,  # uniform RGB(A)
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["p"].attrs["kind"] == "lod"
        children = sorted(store["p"].group_keys())
        assert len(children) > 1
        # The finest child is the original Points node, written LAST.
        reader = LuxarScene.load(path)
        finest = children[-1]
        data = reader.get_points(f"p/{finest}")
        assert data.positions.shape[0] == _SUB_N
        assert_uniform(data.radii, 0.5, _SUB_N)
        assert_uniform(data.colors, want, _SUB_N)
        _assert_coarse_levels_carry_color(reader, "p", children[:-1], want)

    @pytest.mark.parametrize("colors,want", _BROADCAST_COLORS)
    def test_lines_substitutive_broadcast_channels(
        self, tmp_path: Any, colors: Any, want: Any
    ) -> None:
        # Colours included for the same reason as the points twin above: the
        # lift broadcasts a uniform colour onto the beads instead of gathering
        # its components as vertex rows (which raised a bare IndexError) (#1444).
        compiler, scene, path = open_scene(tmp_path, "lines_sub_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_SUB_N, seed=55),
            widths=0.3,  # scalar
            colors=colors,  # uniform RGB(A)
            sharpness=0.7,  # scalar
            line_type="segments",
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["line"].attrs["kind"] == "lod"
        children = sorted(store["line"].group_keys())
        assert len(children) > 1
        reader = LuxarScene.load(path)
        finest = children[-1]
        data = reader.get_lines(f"line/{finest}")
        assert data.vertices.shape[0] == _SUB_N
        assert_uniform(data.widths, 0.3, _SUB_N)
        assert_uniform(data.colors, want, _SUB_N)
        _assert_coarse_levels_carry_color(reader, "line", children[:-1], want)

    def test_substitutive_one_bead_line_refuses_per_element_rgba(
        self, tmp_path: Any
    ) -> None:
        # A segment far shorter than its width lifts to exactly ONE bead, so the
        # per-bead colour array is (1, 4). Re-classifying it as the uniform form
        # would admit a per-element RGBA and bake the MEAN of two different
        # alphas — a value neither vertex has — into the coarse level. The
        # vertex-level verdict is final, so this is refused, and nothing lands.
        compiler, scene, path = open_scene(tmp_path, "one_bead_rgba.luxar.zarr")
        verts = np.array([[0, 0, 0], [0.001, 0, 0]], dtype=np.float32)
        varying = np.array([[0.2, 0.4, 0.6, 0.3], [0.2, 0.4, 0.6, 0.9]], np.float32)
        exc = refusal(
            lambda: scene.add_lines(
                "l",
                verts,
                widths=10.0,
                colors=varying,
                line_type="segments",
                substitutive_lod=True,
            )
        )
        assert "per-element" in str(exc)
        assert "l" not in set(zarr.open_group(path, mode="r").group_keys())

        # ... and the uniform twin on the very same geometry still writes, alpha
        # intact on the coarse gsplat child.
        scene.add_lines(
            "ok",
            verts,
            widths=10.0,
            colors=(*_BROADCAST_RGB, _BROADCAST_ALPHA),
            line_type="segments",
            substitutive_lod=True,
        )
        compiler.finalize()
        store = zarr.open_group(path, mode="r")
        assert store["ok"].attrs["kind"] == "lod"
        children = sorted(store["ok"].group_keys())
        _assert_coarse_levels_carry_color(
            LuxarScene.load(path),
            "ok",
            children[:-1],
            (*_BROADCAST_RGB, _BROADCAST_ALPHA),
        )

    @pytest.mark.parametrize("shape,colors", _BAD_DTYPE_COLORS)
    @pytest.mark.parametrize("geometry", ["points", "lines"])
    def test_substitutive_bad_dtype_colors_write_nothing(
        self, tmp_path: Any, geometry: str, shape: str, colors: Any
    ) -> None:
        # A colour of a dtype the leaf refuses must be refused BEFORE the lift
        # builds anything — uniform row and per-element array alike: the coarse
        # gsplat children are written first and the finest child LAST, so
        # discovering the dtype at the encoder would strand a partial kind=lod
        # node — the #1437 stranding class.
        compiler, scene, path = open_scene(
            tmp_path, f"{geometry}_sub_dtype_{shape.replace(' ', '_')}.luxar.zarr"
        )
        add = scene.add_points if geometry == "points" else scene.add_lines
        kwargs: Dict[str, Any] = (
            {"radii": 0.5} if geometry == "points" else {"widths": 0.3}
        )
        exc = refusal(
            lambda: add(
                "n",
                random_positions(_SUB_N, seed=56),
                colors=colors,
                substitutive_lod=True,
                **kwargs,
            )
        )
        assert "dtype" in str(exc)
        store = zarr.open_group(path, mode="r")
        assert "n" not in set(store.group_keys()), (
            f"a partial node was stranded on disk: {sorted(store.group_keys())}"
        )
        compiler.finalize()


# ---------------------------------------------------------------------------
# Scene-dimension COUNT, pre-split (#1446)
# ---------------------------------------------------------------------------

# 400 elements is enough for the substitutive reduce to synthesise coarse levels
# (a degenerate input falls back to a flat node, which would test nothing).
_DIM_N = 400
_DIM_HALF = 200

# Cumulative ladder counts. For Lines these are POLYLINE counts: with
# line_type="segments" the 400 vertices are 200 two-vertex polylines, and the
# ladder MUST end up with more than one level — a single-polyline input
# degenerates to one level and falls through to the flat write, which refuses on
# its own and would make the case vacuous.
_DIM_POINTS_LADDER = {"counts": [_DIM_HALF, _DIM_N]}
_DIM_LINES_LADDER = {"counts": [_DIM_HALF // 2, _DIM_N // 2]}


class TestPointsAdditiveLodDimensionCount:
    def test_mismatched_ndim_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """A ladder must not write levels whose column count contradicts the scene.

        Pre-fix this call was ACCEPTED — ``refusal()`` is what catches that — and
        left ``p/additive_0``, ``p/additive_1`` on disk at ndim=4 in a
        3-dimension scene, while the same call without ``additive_lod=`` refuses.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_add_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_add_ndim_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=61)

        flat = refusal(lambda: flat_scene.add_points("p", positions))
        split = refusal(
            lambda: scene.add_points("p", positions, additive_lod=_DIM_POINTS_LADDER)
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "p" not in compiler.store


class TestLinesAdditiveLodDimensionCount:
    def test_mismatched_ndim_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """The Lines twin — 200 polylines, so the ladder really does fire.

        ``refusal()`` is the bug-catching assertion: pre-fix the ladder wrote
        every level at ndim=4.
        """
        compiler, scene, _ = open_scene(tmp_path, "lines_add_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_add_ndim_flat.luxar.zarr")
        vertices = bad_ndim_positions(_DIM_N, seed=62)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                additive_lod=_DIM_LINES_LADDER,
            )
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "line" not in compiler.store


class TestPointsSubstitutiveLodDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """Pre-fix this refused, but blamed ``child_0`` and stranded ``kind=lod``.

        ``assert_same_refusal`` fails first pre-fix — the message named
        ``child_0`` and a synthesised level's own row count — and the store
        assertion after it is an independent check on the other half of the same
        bug: pre-fix ``p`` is in the store, as a childless wrapper.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_sub_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_sub_ndim_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=63)

        flat = refusal(lambda: flat_scene.add_points("p", positions))
        split = refusal(lambda: scene.add_points("p", positions, substitutive_lod=True))

        assert_same_refusal(flat, split)
        assert "p" not in compiler.store


class TestLinesSubstitutiveLodDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """The Lines twin of the Points substitutive case; same two assertions."""
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_sub_ndim_flat.luxar.zarr")
        vertices = bad_ndim_positions(_DIM_N, seed=64)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                substitutive_lod=True,
            )
        )

        assert_same_refusal(flat, split)
        assert "line" not in compiler.store


def _multi_substitutive_4d_data(n_fine: int = 8, n_coarse: int = 2) -> Any:
    """A 2-level substitutive ``GSplatData`` whose centers have FOUR columns.

    Stored levels (not a compute spec) on purpose: ``lod_group=`` then needs no
    reduce, so the pre-fix path really did reach ``add_lod_group`` and strand a
    childless ``kind=lod`` group.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    def level(n: int, seed: int, compression_factor: int, level_index: int) -> Any:
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=bad_ndim_positions(n, seed=seed),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=cholesky_rows_nd(n, 4),
                )
            ],
            compression_factor=compression_factor,
            level_index=level_index,
        )

    return GSplatData.from_substitutive_levels(
        [level(n_fine, 65, 1, 0), level(n_coarse, 66, 4, 1)]
    )


class TestGSplatsLodGroupDimensionCount:
    """``lod_group=`` is the third wrapper with the same leak (#1446).

    Both cases compare against a DIRECT ``add_gsplats`` call rather than against
    ``add_gsplats_from_data(..., lod_group=False)``: the pre-dispatch gate in
    ``from_data`` formats its ``Could not add gsplats '<name>': `` prefix by hand,
    and ``lod_group=False`` would route through that same hand-written string —
    a parity assertion comparing a string with itself, which could never catch the
    prefix drifting from what ``add_gsplats_impl``'s own funnel produces. The
    direct call is the independent witness.
    """

    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """Pre-fix: refused from inside ``child_0``, ``g`` already a ``kind=lod``.

        ``assert_same_refusal`` is what fails first pre-fix (the message named
        ``child_0`` and the coarsest level's 2-row shape). The store assertion is
        an independent check on the same bug — pre-fix ``g`` really is in the
        store, as a childless wrapper.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "gsplats_lg_ndim_flat.luxar.zarr")
        data = _multi_substitutive_4d_data()

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "g" not in compiler.store

    def test_mismatched_dim_order_length_leaves_no_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        """The ``dim_order=`` half of the same leak — the shape the docstring shows.

        With a ``dim_order`` the post-transform width is the scene's by
        construction, so the column-count check can never fire downstream; what
        raises from inside ``child_0`` is ``dim_order``'s own length-vs-columns
        check. Measured pre-fix: ``Could not add gsplats 'child_0': dim_order has
        3 names but data has 4 columns``, with ``g`` on disk as a childless
        ``kind=lod`` group. ``assert_same_refusal`` fails first pre-fix (the
        sentence is the same, the node it blames is not); the store assertion is
        an independent check on the wrapper that got written.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_dimorder.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "gsplats_lg_dimorder_flat.luxar.zarr")
        data = _multi_substitutive_4d_data()
        dim_order = ["X", "Y", "Z"]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                dim_order=dim_order,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, dim_order=dim_order
            )
        )

        assert_same_refusal(flat, split)
        assert "dim_order has 3 names but data has 4 columns" in str(split)
        assert "g" not in compiler.store

    def test_a_structural_kwarg_fault_still_outranks_the_width_fault(
        self, tmp_path: Any
    ) -> None:
        """Precedence control: ``coverage_fraction`` is refused before the width.

        The gate sits BELOW the multi-substitutive ``coverage_fraction`` refusal
        on purpose, mirroring how the three leaf adders keep their
        colours/colormap gate above their count check. A call that trips both must
        still hear about the kwarg.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_cov.luxar.zarr")
        data = _multi_substitutive_4d_data()

        with pytest.raises(ValueError, match="coverage_fraction must not be passed"):
            scene.add_gsplats_from_data(
                "g", data, lod_group=True, coverage_fraction=0.5
            )

        assert "g" not in compiler.store


# ---------------------------------------------------------------------------
# Node-attrs gate on ``add_gsplats_from_data``'s own doors (issue #1534 review
# finding 2 — a DIFFERENT adder than the four covered by the fifth section's
# docstring above, which explicitly scopes itself to ``add_points`` /
# ``add_lines`` / ``add_mesh`` / ``add_gsplats``)
# ---------------------------------------------------------------------------
#
# ``lod_group=`` dispatches through ``_reject_before_wrapper``
# (``gsplats_pipeline/from_data.py``), which judges everything checkable
# BEFORE ``add_gsplats_as_lod_group_impl`` calls ``add_lod_group`` — but, unlike
# the #1529/#1534 leaf-adder hoist, it never ran ``validate_render_attrs`` at
# all. Measured pre-fix: ``child_0``'s own write already answers a
# ``GSPLATS_RESERVED_ATTRS`` key (``ordering=``, ``position_bounds=``, ...)
# with the correct *reserved* verdict — write_gsplats has always used the
# right reserved set, so there is no wrong-verdict/silent-accept-and-clobber
# half to this one, unlike the Points/Lines/Mesh ADDITIVE bug — but that
# refusal came from inside ``child_0``, by which point the ``kind=lod``
# wrapper was already on disk: the same #1529/#1534 partition/substitutive
# shape, one adder layer up.
#
# ``additive_lod=`` dispatches through ``add_gsplats_multi_lod_impl`` ->
# ``write_gsplat_leaf_subtree``, whose own ``validate_render_attrs`` call
# (with the correct ``GSPLATS_RESERVED_ATTRS``) already runs BEFORE that
# writer creates any group — so that door never stranded and needed no
# change; the control test below just pins that it stays that way.

_GSPLATS_FROM_DATA_RESERVED_CASES = [("ordering", "morton"), ("center_bounds", None)]


class TestGSplatsFromDataNodeAttrsGate:
    @pytest.mark.parametrize("key,value", _GSPLATS_FROM_DATA_RESERVED_CASES)
    def test_lod_group_reserved_attr_no_longer_strands_a_childless_wrapper(
        self, tmp_path: Any, key: str, value: Any
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"gsplats_fromdata_lg_{key}.luxar.zarr"
        )
        _, flat_scene, _ = open_scene(
            tmp_path, f"gsplats_fromdata_lg_{key}_flat.luxar.zarr"
        )
        data = _multi_substitutive_3d_data()

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                **{key: value},
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, **{key: value}
            )
        )

        assert_same_refusal(flat, split)
        assert f"'{key}'" in str(split)
        assert "are reserved" in str(split)
        assert "g" not in compiler.store
        # Pre-fix: the flat-write verdict was already correct (write_gsplats has
        # always reserved this key), but it fired from inside ``child_0`` — by
        # then ``add_lod_group`` had already created ``g`` as a childless
        # ``kind=lod`` group, which survived finalize().
        assert "g" not in finalized_group_keys(compiler, path)

    def test_lod_group_position_bounds_no_longer_strands_a_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "gsplats_fromdata_lg_posbounds.luxar.zarr"
        )
        data = _multi_substitutive_3d_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                data,
                lod_group=True,
                position_bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            )
        )

        assert "position_bounds" in str(split)
        assert "are reserved" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    def test_additive_lod_reserved_attr_control_already_refused_cleanly(
        self, tmp_path: Any
    ) -> None:
        """Control: this door needed no change here — pin that it stays refused.

        ``at_substitutive(0)`` takes the finest level of a 2-level substitutive
        ``GSplatData`` and reduces it to a single-substitutive one; combined with
        ``additive_lod={"n_lods": 2}`` this reaches ``add_gsplats_multi_lod_impl``,
        the same shape ``TestTheGsplatsAdditiveLadderStillHasNoLabelsChannel``
        uses above.
        """
        compiler, scene, path = open_scene(
            tmp_path, "gsplats_fromdata_add_attrs.luxar.zarr"
        )
        data = _multi_substitutive_3d_data().at_substitutive(0)

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, additive_lod={"n_lods": 2}, ordering="morton"
            )
        )

        assert "'ordering'" in str(split)
        assert "are reserved" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)


class TestGSplatsFromDataNodeAttrsGateStillForwardsPartition:
    """``partition=`` is a leaf ``add_gsplats`` named param, not an unknown attr.

    Regression for a round-2 review finding on this same gate: ``partition`` sits
    in EXACTLY the position ``labels``/``image_labels`` do — a named parameter of
    the leaf ``Group.add_gsplats`` that is NOT a named parameter of
    ``add_gsplats_from_data_impl``, so it arrives inside ``**attrs`` here and must
    ride, unexcluded, into ``child_attrs`` for each substitutive child to apply
    its own BSP split. Excluding only ``labels``/``image_labels`` (as this gate
    did immediately after #1534) made a previously-working
    ``partition={"max_elements": N}`` call answer ``Unknown node attribute
    'partition'. Did you mean 'absorption'?`` with nothing written, on a call
    whose SINGLE-level twin (no ``lod_group=``) kept working — measured before
    this fix.
    """

    def test_partition_still_builds_a_partitioned_child_under_a_ladder(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "gsplats_fromdata_lg_partition.luxar.zarr"
        )
        data = _multi_substitutive_3d_data()

        # max_elements=2 splits the 8-splat finest level into 4 parts of 2 but
        # leaves the 2-splat coarsest level whole (1 part -> falls through to a
        # plain leaf), matching the ``child_1 kind=partition -> part_0..3``
        # shape measured pre-fix on main.
        node = scene.add_gsplats_from_data(
            "g", data, lod_group=True, partition={"max_elements": 2}
        )
        compiler.finalize()

        assert node is not None
        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "lod"
        children = set(store["g"].group_keys())
        assert children == {"child_0", "child_1"}
        partitioned = [
            c for c in children if store["g"][c].attrs.get("kind") == "partition"
        ]
        assert len(partitioned) == 1, (
            "expected exactly one child (the 8-splat finest level) to have been "
            f"partitioned; got {partitioned} among {children}"
        )
        assert set(store["g"][partitioned[0]].group_keys()) == {
            "part_0",
            "part_1",
            "part_2",
            "part_3",
        }

    def test_a_bad_attr_alongside_partition_is_still_refused_up_front(
        self, tmp_path: Any
    ) -> None:
        """``partition=`` riding through must not blunt the gate for real typos."""
        compiler, scene, path = open_scene(
            tmp_path, "gsplats_fromdata_lg_partition_badattr.luxar.zarr"
        )
        data = _multi_substitutive_3d_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                data,
                lod_group=True,
                partition={"max_elements": 2},
                blending="max",
            )
        )

        assert "Unknown node attribute 'blending'" in str(split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)


def _coloured_multi_substitutive_data(ndim: int) -> Any:
    """A 2-level substitutive ``GSplatData`` carrying per-splat colours."""
    return _multi_substitutive_data(
        lambda n, seed: (
            bad_ndim_positions(n, seed=seed)
            if ndim == 4
            else random_positions(n, seed=seed)
        ),
        lambda n: cholesky_rows_nd(n, ndim),
        colors_for=lambda n, _level: np.zeros((n, 3), dtype=np.float32),
    )


class TestGSplatsLodGroupColoursGate:
    """The colours/colormap exclusion is judged before the width, on both paths.

    The three leaf adders put that gate above their count check; the ``from_data``
    gate has to answer the same way or ``lod_group=`` disagrees with its own flat
    path about which fault a call that trips both is told about.
    """

    def test_a_colours_fault_outranks_a_width_fault(self, tmp_path: Any) -> None:
        """Both faults at once: the colours one wins, flat and split alike.

        Measured with the width check first (the shape this landed in briefly):
        the split path answered ``Dimension mismatch …`` where the flat path
        answered ``Cannot specify both …``. ``assert_same_refusal`` is what
        catches that.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_width.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_width_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(4)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert_same_refusal(flat, split)
        assert "both 'colors' and 'colormap'" in str(split)
        assert "g" not in compiler.store

    def test_a_colours_fault_alone_leaves_no_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        """Width perfectly fine — the same stranding class, a different fault.

        Pre-fix: ``Could not add gsplats 'child_0': Cannot specify both …`` with
        ``g`` on disk as a childless ``kind=lod`` group that survives
        ``finalize()``. The store assertion is the one that catches it here (the
        sentence is the same either way, only the node it blames differs, which
        ``assert_same_refusal`` also sees).
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_only.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_only_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(3)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert_same_refusal(flat, split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store

    def test_a_dim_order_fault_outranks_the_colours_fault(self, tmp_path: Any) -> None:
        """Three faults deep: the ``dim_order`` spec wins, flat and split alike.

        The flat path applies ``dim_order`` (spec + ``fill``, then ``fill_sigma``)
        while transforming the arrays, which is ABOVE its colours/colormap gate —
        so a call carrying colours, a ``colormap`` and a bad ``dim_order`` hears
        about the ``dim_order``. Measured with the colours gate first (the shape
        this landed in briefly): the split path answered ``Cannot specify both
        'colors' and 'colormap'`` where the flat path answered ``dim_order has 3
        names but data has 4 columns``. Together with
        :meth:`test_a_colours_fault_outranks_a_width_fault` this pins the whole
        order — dim_order spec, then colours, then width.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_dimorder.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_dimorder_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(4)
        dim_order = ["X", "Y", "Z"]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
                dim_order=dim_order,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis", dim_order=dim_order
            )
        )

        assert_same_refusal(flat, split)
        assert "dim_order has 3 names but data has 4 columns" in str(split)
        assert "g" not in compiler.store

    def test_a_colour_on_a_coarse_level_alone_is_seen_by_the_gate(
        self, tmp_path: Any
    ) -> None:
        """Only the COARSE level carries colours — the finest carries none.

        ``GSplatData.colors`` is the FINEST level's ladder merged, so a gate that
        asked only that question passed this call, and the coarsest child (written
        first) then raised ``Could not add gsplats 'child_0': Cannot specify both
        …`` with ``g`` already on disk as a childless ``kind=lod`` group. Measured
        in exactly that shape before the gate was widened to every level. No flat
        twin: a single leaf cannot express per-level colours, so the assertions
        are the blamed node and the empty store.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_coarse.luxar.zarr")
        data = _multi_substitutive_data(
            lambda n, seed: random_positions(n, seed=seed),
            lambda n: cholesky_rows_nd(n, 3),
            colors_for=lambda n, level_index: (
                None if level_index == 0 else np.zeros((n, 3), dtype=np.float32)
            ),
        )
        assert data.colors is None, "the finest level must be the uncoloured one"

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert "both 'colors' and 'colormap'" in str(split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store


class TestRangeWarningsAreNotMultipliedByTheHoist:
    """The control: only the COUNT half was hoisted above the branches.

    ``validate_data_dimensions`` also warns once per dimension whose values fall
    outside its declared ``range``. Hoisting the WHOLE validator above the
    branches would have added one such warning per dimension for the SOURCE array
    on top of whatever each path already emits — three extra here. Both counts
    below are the measured pre-hoist numbers, so either kind of drift fails.

    ``warnings.catch_warnings`` rather than ``pytest.warns``, because the additive
    count is ZERO and ``pytest.warns`` cannot express "no warnings". That zero is
    a KNOWN GAP, not the desired end state: the multi-LOD writer never runs the
    range half at all, so a laddered node silently loses the three warnings the
    same data gets on the flat path. Closing it means hoisting the range half too
    and suppressing it per child — cross-cutting, and out of scope for #1446,
    which changed only the count half. Pinned at 0 here so the gap is visible and
    cannot widen unnoticed.
    """

    def test_points_ladder_does_not_gain_a_source_range_warning(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_ranged_scene(tmp_path, "points_add_warn.luxar.zarr")
        positions = random_positions(_DIM_N, seed=67)  # spans [0, 100), range (0, 10)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_points("p", positions, additive_lod=_DIM_POINTS_LADDER)
        compiler.finalize()

        assert count_range_warnings(records) == 0

    def test_points_substitutive_warns_once_per_dimension_per_leaf(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_ranged_scene(
            tmp_path, "points_sub_warn.luxar.zarr"
        )
        positions = random_positions(_DIM_N, seed=68)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_points("p", positions, substitutive_lod=True)
        compiler.finalize()

        # Every child is written through a leaf adder (the finest is the original
        # Points node, the coarse ones are synthesised gsplats), so each
        # contributes one warning per dimension. Asserted against the children
        # actually written rather than a bare literal, so a change in the number
        # of synthesised levels cannot silently weaken the count.
        store = zarr.open_group(path, mode="r")
        n_leaves = len(list(store["p"].group_keys()))
        assert n_leaves > 1
        assert count_range_warnings(records) == 3 * n_leaves


# The documented precedence (CHANGELOG #1446, core/group/README.md): the
# scene-dimension count is checked ABOVE the #1437 per-element channel gate, so a
# call that trips both hears about the column count — on every path. Parametrized
# over both wrapper families rather than split across the two files, because the
# claim is precisely that they all answer the same way.
_PRECEDENCE_SPLITS = [
    ("partition", {"partition": {"max_elements": _DIM_HALF}}),
    ("additive", {"additive_lod": _DIM_POINTS_LADDER}),
    ("substitutive", {"substitutive_lod": True}),
]


class TestDimensionCountPrecedesTheChannelGate:
    @pytest.mark.parametrize("case,split_kwargs", _PRECEDENCE_SPLITS)
    def test_a_wrong_length_colour_does_not_mask_the_column_count(
        self, tmp_path: Any, case: str, split_kwargs: Dict[str, Any]
    ) -> None:
        """Both faults at once: the column count is reported, flat and split alike.

        The #1437 channel gate sits at the top of each WRAPPER impl, the count
        check at the top of the leaf ADDER — so the count is reached first on the
        split paths too, and every path gives the caller the same sentence. The
        first assertion pins which fault wins on the flat path (the reference the
        other three are compared against), the second that the split paths agree.
        """
        compiler, scene, _ = open_scene(tmp_path, f"prec_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"prec_{case}_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=69)
        # Half-length per-point colours: on their own these are what the #1437
        # gate refuses.
        colors = np.zeros((_DIM_HALF, 3), dtype=np.float32)

        flat = refusal(lambda: flat_scene.add_points("p", positions, colors=colors))
        split = refusal(
            lambda: scene.add_points("p", positions, colors=colors, **split_kwargs)
        )

        assert "Dimension mismatch" in str(flat)
        assert_same_refusal(flat, split)
        assert "p" not in compiler.store


def _multi_substitutive_data(
    centers_for: Any,
    chol_for: Any,
    n_fine: int = 8,
    n_coarse: int = 2,
    colors_for: Any = None,
) -> Any:
    """A 2-level substitutive ``GSplatData`` built from two per-level factories.

    Stored levels rather than a compute spec, for the same reason as the 4-column
    twin above: ``lod_group=`` then needs no reduce, so the pre-fix path really
    did reach ``add_lod_group`` and strand a childless ``kind=lod`` group.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    def level(n: int, seed: int, compression_factor: int, level_index: int) -> Any:
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=centers_for(n, seed),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol_for(n),
                    # ``colors_for`` takes the level index as well as the count so
                    # a case can colour ONE level (see the coarse-only case).
                    colors=None if colors_for is None else colors_for(n, level_index),
                )
            ],
            compression_factor=compression_factor,
            level_index=level_index,
        )

    return GSplatData.from_substitutive_levels(
        [level(n_fine, 81, 1, 0), level(n_coarse, 82, 4, 1)]
    )


def _multi_substitutive_3d_data() -> Any:
    """The well-formed 3-column twin: only the ``dim_order`` spec is at fault."""
    return _multi_substitutive_data(
        lambda n, seed: random_positions(n, seed=seed),
        lambda n: cholesky_rows_nd(n, 3),
    )


# Every refusal ``apply_dim_order`` / ``apply_dim_order_cholesky`` can reach from
# the SPEC alone — no transformed array needed — so every one of them is
# checkable before the wrapper group is created. Measured pre-fix: each named
# ``child_0`` with ``g`` already on disk as a childless ``kind=lod`` group.
_DIM_ORDER_SPEC_CASES = [
    ("duplicate_names", {"dim_order": ["X", "Y", "X"]}),
    ("name_not_in_scene", {"dim_order": ["X", "Y", "W"]}),
    ("fill_key_not_in_scene", {"dim_order": ["X", "Y", "Z"], "fill": {"Q": 0.0}}),
    ("fill_key_in_dim_order", {"dim_order": ["X", "Y", "Z"], "fill": {"X": 0.0}}),
    (
        "fill_sigma_key_not_in_scene",
        {"dim_order": ["X", "Y", "Z"], "fill_sigma": {"Q": 1.0}},
    ),
    (
        "fill_sigma_key_in_dim_order",
        {"dim_order": ["X", "Y", "Z"], "fill_sigma": {"X": 1.0}},
    ),
    # Not a sequence at all: the validator raises TypeError, which the leaf
    # adders' funnel converts to a ValueError. The pre-dispatch gate must catch
    # both or the two paths differ in exception TYPE, not just wording.
    ("dim_order_not_a_sequence", {"dim_order": 3}),
]


class TestGSplatsLodGroupDimOrderSpec:
    @pytest.mark.parametrize("case,kwargs", _DIM_ORDER_SPEC_CASES)
    def test_a_bad_spec_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, case: str, kwargs: Dict[str, Any]
    ) -> None:
        """Data is well-formed; only the ``dim_order``/``fill``/``fill_sigma`` spec is not.

        The flat reference is a direct ``add_gsplats`` with the same kwargs — the
        independent witness for the hand-written prefix (see the sibling class).
        ``assert_same_refusal`` is what fails pre-fix, on the blamed node name;
        the ``child_0`` and store assertions after it pin the two halves of the
        symptom separately.
        """
        compiler, scene, _ = open_scene(tmp_path, f"lg_spec_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lg_spec_{case}_flat.luxar.zarr")
        data = _multi_substitutive_3d_data()

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                **kwargs,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data("g", data, lod_group=True, **kwargs)
        )

        assert_same_refusal(flat, split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store


class TestGSplatsLodGroupRankGuard:
    """1-D centers: ``AdditiveSubLOD`` accepts them, so they reach the gate.

    Without the rank guard in ``validate_dimension_count`` this was a bare
    ``IndexError`` from ``shape[1]`` — which escapes the adders'
    ``except (ValueError, TypeError)`` funnels entirely.
    """

    def test_lod_group_refuses_1d_centers_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, "lg_rank.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_rank_flat.luxar.zarr")
        data = _multi_substitutive_data(
            lambda n, seed: np.zeros((n,), dtype=np.float32),
            lambda n: cholesky_rows_nd(n, 3),
        )

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "Centers must have shape (N, D)" in str(split)
        assert "g" not in compiler.store

    def test_additive_path_refuses_1d_centers_with_a_value_error(
        self, tmp_path: Any
    ) -> None:
        """The other ``GSplatData`` path through the guard, reached per sub-LOD.

        No exact-message parity here on purpose: ``add_gsplats_multi_lod_impl``
        validates each sub-LOD in its own right, so the shape it reports is that
        sub-LOD's ``(4,)``, not the concatenated ``(8,)`` a flat call would see.
        What matters is the exception TYPE — a ``ValueError`` through the funnel,
        not the raw ``IndexError`` this used to be.
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        compiler, scene, _ = open_scene(tmp_path, "add_rank.luxar.zarr")
        sublods = [
            AdditiveSubLOD(
                centers=np.zeros((4,), dtype=np.float32),
                amplitudes=np.zeros((4,), dtype=np.float32),
                cholesky_factors=cholesky_rows_nd(4, 3),
            )
            for _ in range(2)
        ]
        data = GSplatData(additive_sublods=sublods)

        with pytest.raises(ValueError, match=r"Centers must have shape \(N, D\)"):
            scene.add_gsplats_from_data("g", data)

        assert "g" not in compiler.store


# ---------------------------------------------------------------------------
# labels / image_labels on a multi-child gsplats wrapper (#1471)
# ---------------------------------------------------------------------------

# 8 finest / 2 coarsest splats — the issue's own shape. Neither channel is a
# named kwarg of ``add_gsplats_from_data``: both are named params of the LEAF
# adder only, so they arrive inside ``**attrs`` and rode into every child through
# ``child_attrs``, unsliced. The fixtures themselves live in ``../conftest`` —
# the graft half of this gate is in the partition/ sibling and states them once.
_N_FINE = N_LABELLED

# All three doors into the multi-substitutive branch. ``default`` (the kwarg
# omitted entirely, auto-lowering a stored pyramid) is the one
# ``add_gsplats_from_file`` uses for every matrix-shaped file, so it is the most
# likely real-world door and must not be the untested one. ``compute`` needs a
# FLAT input by construction: compute kwargs on data that already has stored
# levels are refused outright ("Pass recompute=True to override the stored
# pyramid"), which would make the case test that refusal instead.
_LOD_GROUP_ROUTES = [
    ("explicit", lambda: _multi_substitutive_3d_data(), {"lod_group": True}),
    ("default", lambda: _multi_substitutive_3d_data(), {}),
    (
        "compute",
        lambda: _multi_substitutive_3d_data().at_substitutive(0),
        {"lod_group": {"levels": 2, "compression_factor": 4}},
    ),
]


class TestGSplatsLodGroupRefusesLabels:
    """A substitutive ladder cannot carry per-element labels at all (#1471).

    Unlike every other case in this file the fix is a REFUSAL, not a hoist: each
    level is its own set of merged representative splats with its own count, so
    no single list has a per-element correspondence to carry.

    The two channels stranded slightly different wreckage, and the worse one is
    ``image_labels``. Pre-fix with ``labels``: ``Could not add gsplats 'child_0':
    labels: Labels length (8) must match element count (2)``, leaving ``g`` as a
    CHILDLESS ``kind=lod`` group that survives ``finalize()``. Pre-fix with
    ``image_labels``: the same shape of message, but the labels are written after
    the geometry, so ``child_0`` is left HALF-WRITTEN — ``amplitudes``,
    ``centers``, ``cholesky_factors_diag``/``_offdiag`` and ``chunk_bounds`` all
    on disk under the stranded wrapper. Both halves are asserted separately: the
    message (which named the wrong node and the wrong fault) and the store.
    """

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    @pytest.mark.parametrize("route,make_data,route_kwargs", _LOD_GROUP_ROUTES)
    def test_refused_up_front_with_nothing_written(
        self,
        tmp_path: Any,
        channel: str,
        kwargs: Dict[str, Any],
        _attr: str,
        route: str,
        make_data: Any,
        route_kwargs: Dict[str, Any],
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"lg_{channel}_{route}.luxar.zarr")
        data = make_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data("g", data, **route_kwargs, **kwargs)
        )

        assert isinstance(split, ValueError)
        # The hand-written prefix names the CALLER's node, not an internal child.
        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "child_0" not in str(split)
        assert (
            f"{channel} is not supported on a multi-level substitutive pyramid "
            "(auto-lowered to a kind=lod group)" in str(split)
        )
        # The three halves the message must state: why, this door's own remedy,
        # and the general one. The REMEDY assertion matters as much as the
        # structure one — the two doors' constants exist so their wording cannot
        # drift, and without this a swap between them passes silently.
        assert "no single list has a per-element correspondence" in str(split)
        assert "lod_group=False" in str(split)
        assert "add_lod_group()" in str(split)
        assert "g" not in compiler.store
        # And no stranded wrapper — childless or half-written — survives finalize.
        assert "g" not in finalized_group_keys(compiler, path)

    def test_labels_outranks_image_labels_when_both_are_passed(
        self, tmp_path: Any
    ) -> None:
        """Deterministic tie-break: ``labels`` is asked first, so it is reported.

        Both are refused for the identical reason, so which one is named is
        arbitrary on the merits — but it must not be arbitrary in practice. The
        order is the leaf adder's signature order (``labels`` then
        ``image_labels``).
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_both_labels.luxar.zarr")
        data = _multi_substitutive_3d_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                data,
                lod_group=True,
                labels=LABELS,
                image_labels=IMAGE_LABELS,
            )
        )

        assert "labels is not supported" in str(split)
        assert "image_labels is not supported" not in str(split)
        assert "g" not in compiler.store

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    @pytest.mark.parametrize("fault", ["dim_order", "colours", "column_count"])
    def test_a_flat_parity_fault_still_outranks_the_labels_refusal(
        self,
        tmp_path: Any,
        channel: str,
        kwargs: Dict[str, Any],
        _attr: str,
        fault: str,
    ) -> None:
        """Precedence control: the refusal is LAST in the gate, and must stay there.

        Every other check in the gate mirrors a fault the FLAT path reports at
        that position; this one has no flat counterpart at all (the flat path
        ACCEPTS ``labels`` and validates it last, in the writer sweep). Ranked any
        higher it would answer ``labels is not supported …`` where the flat path
        answers something else, silently changing what the #1446 parity assertions
        elsewhere in this file mean.

        Parametrized over all THREE preceding checks — the ``dim_order`` spec, the
        colours/colormap exclusion and the column count — because pinning only one
        of them leaves a mutant that hops the labels raise above the other two
        alive. (The rank guard is the fourth and cannot be combined: 1-D centers
        make every other fault unreachable.)
        """
        compiler, scene, _ = open_scene(tmp_path, f"lg_{channel}_{fault}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lg_{channel}_{fault}_flat.luxar.zarr")
        if fault == "dim_order":
            data = _multi_substitutive_3d_data()
            extra: Dict[str, Any] = {"dim_order": ["X", "Y", "X"]}
            expected = "dim_order has duplicate names"
        elif fault == "colours":
            data = _coloured_multi_substitutive_data(3)
            extra = {"colormap": "viridis"}
            expected = "Cannot specify both 'colors' and 'colormap'"
        else:
            data = _multi_substitutive_4d_data()
            extra = {}
            expected = "centers array has 4 columns"

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                **extra,
                **kwargs,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, **extra, **kwargs
            )
        )

        assert_same_refusal(flat, split)
        assert expected in str(split)
        assert "is not supported on a multi-level substitutive" not in str(split)
        assert "g" not in compiler.store

    def test_a_structural_kwarg_fault_outranks_the_labels_refusal(
        self, tmp_path: Any
    ) -> None:
        """``coverage_fraction`` is refused ABOVE the whole gate, labels included.

        The documented precedence in ``add_gsplats_from_data``'s docstring, with
        no test until now.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_cov_labels.luxar.zarr")
        data = _multi_substitutive_3d_data()

        with pytest.raises(ValueError, match="coverage_fraction must not be passed"):
            scene.add_gsplats_from_data(
                "g", data, lod_group=True, coverage_fraction=0.5, labels=LABELS
            )

        assert "g" not in compiler.store

    def test_an_empty_labels_list_is_refused_rather_than_read_as_absent(
        self, tmp_path: Any
    ) -> None:
        """``[]`` is not None, so it is a labels REQUEST — and an impossible one."""
        compiler, scene, _ = open_scene(tmp_path, "lg_empty_labels.luxar.zarr")

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", _multi_substitutive_3d_data(), lod_group=True, labels=[]
            )
        )

        assert "labels is not supported" in str(split)
        assert "g" not in compiler.store

    def test_a_numpy_image_labels_array_is_refused_like_a_list(
        self, tmp_path: Any
    ) -> None:
        """``image_labels`` accepts several container types; the gate is truthy-free.

        An ``is not None`` test rather than a truth test, so a numpy array — whose
        ``__bool__`` raises on more than one element — cannot slip past.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_np_image_labels.luxar.zarr")

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                _multi_substitutive_3d_data(),
                lod_group=True,
                image_labels=np.zeros((_N_FINE, 2, 2, 3), dtype=np.uint8),
            )
        )

        assert "image_labels is not supported" in str(split)
        assert "g" not in compiler.store


class TestAnExplicitNoneMeansNoLabels:
    """``labels=None`` must be indistinguishable from omitting it (#1471).

    ``labels=maybe_labels`` is an idiomatic call form, and the gate correctly
    reads None as "absent" — but the KEY survived in ``**attrs`` and rode into
    ``child_attrs``. ``validate_render_attrs`` rejects an unknown key by NAME and
    never looks at its value, so any child that took the additive-ladder writer
    (``write_gsplat_leaf_subtree``) raised ``Unknown node attribute 'labels'``
    from inside ``child_0`` with the ``kind=lod`` wrapper already on disk —
    exactly the strand #1471 is about, reached by a call that asked for no labels
    at all. Stock ``gsplat lod --recipe levels`` output hits this without any
    ``additive_lod=`` of its own, since its per-level stream ladders are on by
    default.

    Only the ``with_ladder`` params are true pre-fix failures. The ``plain`` two
    are CONTROLS: a single-sublod child is written by ``Group.add_gsplats``,
    which binds ``labels`` as a named param, so the stray key never reaches an
    attr validator and they pass with the strip removed. They are here so the
    normalisation cannot be "fixed" by making the plain path refuse instead.
    """

    @pytest.mark.parametrize("channel", ["labels", "image_labels"])
    @pytest.mark.parametrize(
        "case,extra",
        [("with_ladder", {"additive_lod": {"n_lods": 2}}), ("plain", {})],
    )
    def test_none_writes_the_whole_ladder_normally(
        self, tmp_path: Any, channel: str, case: str, extra: Dict[str, Any]
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"none_{channel}_{case}.luxar.zarr"
        )

        scene.add_gsplats_from_data(
            "g",
            _multi_substitutive_3d_data(),
            lod_group=True,
            **extra,
            **{channel: None},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "lod"
        # The bug's signature was a wrapper with NO children at all.
        assert sorted(store["g"].group_keys()) == ["child_0", "child_1"]


class TestLabelsStillWorkWhereTheyAlwaysDid:
    """The negatives: only MULTI-child gsplats wrappers are closed (#1471)."""

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_lod_group_false_collapses_to_a_labelled_finest_leaf(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        """``lod_group=False`` writes ONE leaf carrying all 8 splats — still labelled.

        This is the escape hatch the refusal names, so it has to keep working.
        """
        compiler, scene, path = open_scene(tmp_path, f"lg_false_{channel}.luxar.zarr")
        data = _multi_substitutive_3d_data()

        scene.add_gsplats_from_data("g", data, lod_group=False, **kwargs)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") != "lod"
        assert store["g"].attrs[attr] is True

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_a_single_level_gsplatdata_is_untouched(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        """A one-level ``GSplatData`` never enters the multi-substitutive branch."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        compiler, scene, path = open_scene(tmp_path, f"single_{channel}.luxar.zarr")
        data = GSplatData(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=random_positions(_N_FINE, seed=71),
                    amplitudes=np.ones(_N_FINE, dtype=np.float32),
                    cholesky_factors=cholesky_rows_nd(_N_FINE, 3),
                )
            ]
        )

        scene.add_gsplats_from_data("g", data, **kwargs)
        compiler.finalize()

        assert zarr.open_group(path, mode="r")["g"].attrs[attr] is True


class TestTheGsplatsAdditiveLadderStillHasNoLabelsChannel:
    """Not a #1471 door: this path already refused, and for a different reason."""

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_the_additive_ladder_keeps_its_own_pre_existing_answer(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """``additive_lod=`` on gsplats has NO labels channel — and still says so.

        ``write_gsplat_leaf_subtree`` documents labels as a leaf-only scene
        feature that stays on ``write_gsplats``, so the ladder path answers with
        the unknown-attr refusal (the ladder-union label support of
        ``validate_ladder_labels`` is Points/Lines only). #1471 did not touch that
        path; this pins that it did not drift into the new wording either.
        """
        compiler, scene, _ = open_scene(tmp_path, f"add_{channel}.luxar.zarr")
        data = _multi_substitutive_3d_data().at_substitutive(0)

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, additive_lod={"n_lods": 2}, **kwargs
            )
        )

        assert f"Unknown node attribute '{channel}'" in str(split)
        assert "is not supported on a multi-level substitutive" not in str(split)
        assert "g" not in compiler.store


# ---------------------------------------------------------------------------
# Unwritable colour DTYPE, pre-split (#1489)
# ---------------------------------------------------------------------------

#: Vertices per side of the mesh substitutive case. 15x15 = 225 vertices, enough
#: for the decimator to synthesise coarse levels (a surface it cannot reduce
#: falls back to a plain leaf and the ladder branch is never entered).
_DTYPE_MESH_SIDE = 15
_DTYPE_MESH_N = _DTYPE_MESH_SIDE * _DTYPE_MESH_SIDE


class TestLodWrappersRefuseAnUnwritableColorDtype:
    """The LOD half of #1489 — the same hole the partition/ sibling documents.

    An integer COLOR array must be ``uint8``/``uint16``; the encoder always said
    so, but from inside ``write_colors``, which on a laddered node is reached
    only after a whole CHILD is on disk. Measured before the fix:
    ``additive_lod=True`` left ``n/additive_0`` behind, and a mesh
    ``substitutive_lod=True`` left ``n/child_0``. Points and Lines survived the
    substitutive case only because ``luxar.gsplats.lift`` carries its own dtype
    mirror (#1444/#1485) — one path, one geometry pair, which is precisely why
    the rule belongs in the shared validator instead.

    Parity with the flat path is byte-exact for the same reason as everywhere
    else in this module: both paths now run the ONE validator.
    """

    @pytest.mark.parametrize("geometry", ["points", "lines"])
    def test_additive_int64_colors_are_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, geometry: str
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"{geometry}_add_dt.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"{geometry}_add_dt_flat.luxar.zarr")
        coords = random_positions(_N, seed=91)
        colors = int64_rgb(_N)

        def call(target: Any, **extra: Any) -> Any:
            if geometry == "points":
                return target.add_points("n", coords, colors=colors, **extra)
            return target.add_lines(
                "n", coords, widths=0.3, line_type="segments", colors=colors, **extra
            )

        ladder = _POINTS_LADDER if geometry == "points" else _LINES_LADDER
        flat = refusal(lambda: call(flat_scene))
        split = refusal(lambda: call(scene, additive_lod=ladder))

        assert_same_refusal(flat, split)
        assert "Integer COLOR arrays must use dtype uint8 or uint16" in str(split)
        assert "n" not in compiler.store
        assert "n" not in finalized_group_keys(compiler, path)

    def test_mesh_substitutive_int64_colors_are_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """The row the gsplat-lift mirror never covered: a mesh coarsens by decimation.

        Points and Lines lift to gsplats and are caught by ``lift``'s own dtype
        check; a mesh takes a different producer entirely, so before the shared
        rule this call wrote ``n/child_0`` and then refused from inside it.
        """
        compiler, scene, path = open_scene(tmp_path, "mesh_sub_dt.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "mesh_sub_dt_flat.luxar.zarr")
        vertices, faces = grid_mesh(_DTYPE_MESH_SIDE)
        colors = int64_rgb(_DTYPE_MESH_N)

        flat = refusal(lambda: flat_scene.add_mesh("n", vertices, faces, colors=colors))
        split = refusal(
            lambda: scene.add_mesh(
                "n", vertices, faces, colors=colors, substitutive_lod=True
            )
        )

        assert_same_refusal(flat, split)
        assert "Integer COLOR arrays must use dtype uint8 or uint16" in str(split)
        assert "n" not in compiler.store
        assert "n" not in finalized_group_keys(compiler, path)

    def test_the_same_mesh_colours_as_uint8_still_ladder(self, tmp_path: Any) -> None:
        """The negative control: only the DTYPE was ever wrong.

        A surface too coarse to reduce writes a plain leaf instead of a ladder,
        which would make the refusal above vacuous — so pin that this very mesh
        with ``.astype(np.uint8)`` colours really does produce a ``kind=lod``
        group with more than one child.
        """
        compiler, scene, path = open_scene(tmp_path, "mesh_sub_dt_ok.luxar.zarr")
        vertices, faces = grid_mesh(_DTYPE_MESH_SIDE)

        scene.add_mesh(
            "n",
            vertices,
            faces,
            colors=int64_rgb(_DTYPE_MESH_N).astype(np.uint8),
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["n"].attrs["kind"] == "lod"
        assert len(sorted(store["n"].group_keys())) > 1


def _dtype_multi_substitutive_data(colors_for: Any) -> Any:
    """A 2-level substitutive ``GSplatData`` whose colours vary BY LEVEL.

    ``colors_for(n, level_index)`` — level 0 is the FINEST (what a flat call
    forwards as ``colors=``), level 1 the coarse one, and the writer emits them
    coarsest-first as ``child_0`` / ``child_1``. Stored levels, not a compute
    spec, so ``lod_group=`` really does reach ``add_lod_group``.
    """
    return _multi_substitutive_data(
        lambda n, seed: random_positions(n, seed=seed),
        lambda n: cholesky_rows_nd(n, 3),
        colors_for=colors_for,
    )


class TestGSplatsLodGroupColourDtype:
    """The gsplats substitutive door: ``lod_group=``, not ``substitutive_lod=`` (#1489).

    GSplats have no ``substitutive_lod=`` kwarg — their substitutive door is
    ``add_gsplats_from_data(..., lod_group=…)`` with multi-level data — and
    ``add_gsplats_as_lod_group_impl`` creates the ``kind=lod`` group BEFORE it
    writes any child. So the shared validator alone did not save this path: it
    only ran from inside a child. Measured with the dtype rule neutered, and each
    residue SURVIVED ``finalize()``:

    * both levels ``int64``   → ``g`` on disk as ``kind=lod`` holding a partial
      ``child_0`` (positions written, colours not);
    * finest bad, coarse fine → ``kind=lod`` holding a COMPLETE ``child_0`` and a
      partial ``child_1``, i.e. a half-written ladder the viewer would load;
    * coarse bad, finest fine → ``kind=lod`` holding a partial ``child_0``.

    The fix is one call inside ``_reject_before_wrapper``'s existing colours
    sweep — the one that already walks every level for the colours/colormap rule
    and whose own comment names this stranding shape. Levels are walked
    finest-first, so an all-bad ladder reports the level a flat call would.
    """

    #: ``(case, colors_for)``. Colours are (n, 3) red; only the dtype varies.
    _CASES = [
        ("both_levels", lambda n, _lvl: int64_rgb(n)),
        (
            "finest_only",
            lambda n, lvl: int64_rgb(n) if lvl == 0 else int64_rgb(n).astype(np.uint8),
        ),
        (
            "coarse_only",
            lambda n, lvl: int64_rgb(n).astype(np.uint8) if lvl == 0 else int64_rgb(n),
        ),
    ]

    @pytest.mark.parametrize("case,colors_for", _CASES)
    def test_int64_colours_leave_no_partial_ladder(
        self, tmp_path: Any, case: str, colors_for: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"lg_dtype_{case}.luxar.zarr")
        data = _dtype_multi_substitutive_data(colors_for)

        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert "Integer COLOR arrays must use dtype uint8 or uint16" in str(split)
        # The refusal must name the NODE the caller asked for, never an internal
        # child — the tell that it came from the pre-wrapper gate and not from
        # inside a write that had already happened.
        assert "child_0" not in str(split) and "child_1" not in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    def test_an_all_bad_ladder_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """Parity where a flat counterpart exists: the finest level's colours.

        ``lod_group=False`` on the same data forwards ``result.colors`` — the
        FINEST level's array — so the two paths must agree byte-for-byte. This is
        also what pins the finest-first walk order: reporting the coarse level
        first would name a different dtype the moment the two levels differ.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_dtype_parity.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_dtype_parity_flat.luxar.zarr")
        data = _dtype_multi_substitutive_data(lambda n, _lvl: int64_rgb(n))

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "g" not in compiler.store

    def test_the_width_fault_still_outranks_the_dtype_one(self, tmp_path: Any) -> None:
        """Precedence: the dtype check sits BELOW the dimension count, as flat does.

        The leaf writer runs its channel sweep after ``_validate_dimension_count``,
        so a 4-column ladder in a 3-dimension scene must hear about its width even
        though its colours are also unwritable — otherwise ``lod_group=`` disagrees
        with its own flat path about which of two faults it reports.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_dtype_width.luxar.zarr")
        data = _multi_substitutive_data(
            lambda n, seed: bad_ndim_positions(n, seed=seed),
            lambda n: cholesky_rows_nd(n, 4),
            colors_for=lambda n, _lvl: int64_rgb(n),
        )

        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert "4 columns" in str(split)
        assert "uint8 or uint16" not in str(split)
        assert "g" not in compiler.store

    def test_the_same_ladder_with_uint8_colours_still_writes(
        self, tmp_path: Any
    ) -> None:
        """The negative control: only the DTYPE was ever wrong.

        Without this the refusals above would also pass if the ladder never
        materialised — so pin that this very data with ``.astype(np.uint8)``
        colours really does produce a ``kind=lod`` group with both children.
        """
        compiler, scene, path = open_scene(tmp_path, "lg_dtype_ok.luxar.zarr")
        data = _dtype_multi_substitutive_data(
            lambda n, _lvl: int64_rgb(n).astype(np.uint8)
        )

        scene.add_gsplats_from_data("g", data, lod_group=True)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "lod"
        assert sorted(store["g"].group_keys()) == ["child_0", "child_1"]


def _two_rung_finest_level_data(second_rung_colors: Any) -> Any:
    """A ladder whose FINEST level has two additive rungs, the second one bad.

    Every other gsplats fixture in this module builds levels with exactly one
    rung, so ``for sub in level.additive_sublods[:1]`` used to be an invisible
    mutation — it killed 0 of 105 tests. A multi-rung level is the STOCK shape,
    not an exotic one: ``gsplat lod --recipe levels`` carries stream ladders by
    default, so every level of an ordinary file has several rungs.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    def rung(n: int, seed: int, colors: Any) -> Any:
        return AdditiveSubLOD(
            centers=random_positions(n, seed=seed),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=cholesky_rows_nd(n, 3),
            colors=colors,
        )

    clean = int64_rgb(4).astype(np.uint8)
    return GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    rung(4, 71, clean),
                    rung(4, 72, second_rung_colors),
                ],
                compression_factor=1,
                level_index=0,
            ),
            SubstitutiveLevel(
                additive_sublods=[rung(2, 73, int64_rgb(2).astype(np.uint8))],
                compression_factor=4,
                level_index=1,
            ),
        ]
    )


class TestGSplatsLodGroupChecksEveryAdditiveRung:
    """The gate walks the whole ladder, not just each level's first rung (#1489).

    A substitutive level is itself an additive ladder, and the sweep loops over
    BOTH axes. Nothing pinned the inner one until this class: truncating it to
    ``additive_sublods[:1]`` left every other test in the module green.
    """

    def test_a_bad_second_rung_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "lg_rung_bad.luxar.zarr")
        data = _two_rung_finest_level_data(int64_rgb(4))

        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert "Integer COLOR arrays must use dtype uint8 or uint16" in str(split)
        assert "child_0" not in str(split) and "child_1" not in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    def test_the_same_ladder_with_a_clean_second_rung_still_writes(
        self, tmp_path: Any
    ) -> None:
        """The control: two rungs are a legal shape, so the refusal is about dtype."""
        compiler, scene, path = open_scene(tmp_path, "lg_rung_ok.luxar.zarr")
        data = _two_rung_finest_level_data(int64_rgb(4).astype(np.uint8))

        scene.add_gsplats_from_data("g", data, lod_group=True)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "lod"
        assert sorted(store["g"].group_keys()) == ["child_0", "child_1"]


#: ``(id, colours factory, expected message fragment)`` — the colours faults the
#: leaf refuses for a reason OTHER than dtype. Each one used to blame ``child_0``
#: and leave ``g`` on disk as a childless ``kind=lod`` group surviving
#: ``finalize()``; they close together with the dtype one because the gate runs
#: the WHOLE shared validator, not the dtype rule alone.
_NON_DTYPE_COLOUR_FAULTS = [
    (
        "negative_float",
        lambda n: np.full((n, 3), -0.5, dtype=np.float32),
        "Colors cannot be negative",
    ),
    (
        "nan_float",
        lambda n: np.full((n, 3), np.nan, dtype=np.float32),
        "NaN or Inf value(s)",
    ),
    (
        "alpha_above_one",
        lambda n: np.concatenate(
            [
                np.full((n, 3), 0.5, dtype=np.float32),
                np.full((n, 1), 1.5, dtype=np.float32),
            ],
            axis=1,
        ),
        "alpha channel must be within [0, 1]",
    ),
]


class TestGSplatsLodGroupRunsTheWholeColoursValidator:
    """Hoisting the dtype rule ALONE would have broken parity and left three doors.

    ``validate_color_dtype`` is the fifth check inside
    ``validate_colors_for_writing``; lifting just it into the pre-wrapper gate
    jumps it over the four above. Measured with only the dtype helper hoisted: an
    all-``-1`` ``int32`` ladder answered with the DTYPE message where the flat
    path answers ``Colors cannot be negative`` — and then advised
    ``colors.astype(np.uint8)``, which turns -1 into 255. The gate calls the whole
    validator instead, so the internal order is the leaf's by construction.

    NOT fixed here, and deliberately not claimed anywhere: a call that ALSO trips
    a cholesky fault reports the COLOURS fault under the wrapper where the flat
    path reports cholesky. That is the pre-existing pre-split-gate asymmetry
    documented in ``core.group.compositing`` — verified identical to the one the
    ``labels`` gate already has (measured: with a bad Cholesky diagonal, the
    wrapper answers ``labels is not supported …`` where flat answers
    ``cholesky_factors: Cholesky diagonal must be positive``).
    """

    @pytest.mark.parametrize("case,colors_for,expected", _NON_DTYPE_COLOUR_FAULTS)
    def test_a_non_dtype_colours_fault_leaves_no_wrapper(
        self, tmp_path: Any, case: str, colors_for: Any, expected: str
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"lg_colfault_{case}.luxar.zarr")
        data = _dtype_multi_substitutive_data(lambda n, _lvl: colors_for(n))

        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert expected in str(split)
        assert "child_0" not in str(split) and "child_1" not in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    def test_a_negative_integer_ladder_reports_negativity_not_dtype(
        self, tmp_path: Any
    ) -> None:
        """The parity case that a dtype-only hoist got wrong, both ways round.

        ``int32`` is unwritable AND negative. The flat path reports negativity
        (dtype is checked last inside the validator, matching the encoder); the
        wrapped path must say the same thing, byte for byte, or the caller is
        told to cast -1 to uint8.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_negint.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_negint_flat.luxar.zarr")
        data = _dtype_multi_substitutive_data(
            lambda n, _lvl: np.full((n, 3), -1, dtype=np.int32)
        )

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "Colors cannot be negative" in str(split)
        assert "uint8 or uint16" not in str(split)
        assert "g" not in compiler.store


class TestTheColoursHoistDoesNotMultiplyHdrWarnings:
    """The control on the hoist's other half: it adds refusals, not noise.

    ``validate_colors_for_writing`` does not only refuse — it also WARNS, once,
    on float colours above 10.0. Every rung the pre-wrapper gate inspects is
    validated again by the child that writes it, so a gate that let the warning
    through would emit each one twice (measured before the suppression: 4 for the
    2-level ladder below, where the flat path emits one per leaf). The same
    concern :class:`TestRangeWarningsAreNotMultipliedByTheHoist` pins for the
    #1446 count hoist, one validator over.

    Counted against the children actually written rather than a literal, so a
    change in how many levels reach disk cannot silently weaken it.
    """

    def test_an_hdr_ladder_warns_once_per_child(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lg_hdr_warn.luxar.zarr")
        data = _dtype_multi_substitutive_data(
            lambda n, _lvl: np.full((n, 3), 50.0, dtype=np.float32)
        )

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_gsplats_from_data("g", data, lod_group=True)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        n_children = len(sorted(store["g"].group_keys()))
        assert n_children > 1
        hdr = [r for r in records if "HDR colors with maximum value" in str(r.message)]
        assert len(hdr) == n_children


# ---------------------------------------------------------------------------
# Node-attrs gate, pre-split, on the Points/Lines substitutive wrapper (#1529)
# ---------------------------------------------------------------------------
#
# The substitutive wrapper forwards the non-compositing remainder of **attrs
# to a synthesised gsplats child_0, so an attrs key the flat writer refuses up
# front used to be caught
# only from inside that child — by which point the kind=lod wrapper itself was
# already on disk, childless (child_0 is the FIRST coarse level, so no level
# had actually been written yet). `amplitude_range` is reserved for GSPLATS
# but not for POINTS/LINES, so it is the case that shows the wrapper refusing
# for a DIFFERENT reason than the flat leaf does pre-fix (reserved vs.
# unknown); `blending` (a typo for `blending_mode`) is the plain unknown-key
# case, which also exercises the "Did you mean ...?" hint.


class TestPointsSubstitutiveNodeAttrsGate:
    def test_reserved_but_unknown_attr_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_sub_attrs_amp.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_sub_attrs_amp_flat.luxar.zarr")
        positions = random_positions(_SUB_N, seed=111)

        flat = refusal(
            lambda: flat_scene.add_points("p", positions, amplitude_range=[0.0, 1.0])
        )
        split = refusal(
            lambda: scene.add_points(
                "p", positions, substitutive_lod=True, amplitude_range=[0.0, 1.0]
            )
        )

        assert_same_refusal(flat, split)
        assert "amplitude_range" in str(split)
        assert "p" not in compiler.store
        # The gate runs above add_lod_group, so no partial kind=lod node
        # survives finalize() either — the #1529 stranding this closes.
        assert "p" not in finalized_group_keys(compiler, path)

    def test_unknown_attr_typo_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_sub_attrs_typo.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_sub_attrs_typo_flat.luxar.zarr")
        positions = random_positions(_SUB_N, seed=112)

        flat = refusal(lambda: flat_scene.add_points("p", positions, blending="max"))
        split = refusal(
            lambda: scene.add_points(
                "p", positions, substitutive_lod=True, blending="max"
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "p" not in compiler.store
        assert "p" not in finalized_group_keys(compiler, path)

    def test_truncation_radius_is_accepted_on_both_paths(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "points_sub_attrs_trunc.luxar.zarr"
        )
        flat_compiler, flat_scene, flat_path = open_scene(
            tmp_path, "points_sub_attrs_trunc_flat.luxar.zarr"
        )
        positions = random_positions(_SUB_N, seed=113)

        # Control on the flat path: truncation_radius is a known render attr.
        flat_scene.add_points("p", positions, truncation_radius=3.0)
        flat_compiler.finalize()

        scene.add_points("p", positions, substitutive_lod=True, truncation_radius=3.0)
        compiler.finalize()

        flat_store = zarr.open_group(flat_path, mode="r")
        assert flat_store["p"].attrs["type"] == "points"
        assert flat_store["p"].attrs["truncation_radius"] == 3.0
        assert "kind" not in flat_store["p"].attrs

        store = zarr.open_group(path, mode="r")
        assert store["p"].attrs["kind"] == "lod"
        children = sorted(store["p"].group_keys())
        assert len(children) > 1
        # #1529 asked the harder half of the question explicitly: an attr the
        # flat path ACCEPTS must not be silently dropped on the way to the
        # synthesised children (which would write cleanly and be worse than a
        # refusal). Every level — the lifted gsplat coarse ones and the finest
        # points child alike — carries the caller's value, as the flat leaf does.
        for child in children:
            assert store["p"][child].attrs["truncation_radius"] == 3.0

    def test_a_colours_fault_still_outranks_the_attrs_gate(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "points_sub_attrs_prec.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_sub_attrs_prec_flat.luxar.zarr")
        positions = random_positions(_SUB_N, seed=114)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p",
                positions,
                colors=[1.0, 0.0, 0.0],
                colormap="viridis",
                bogus_attr=1,
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                substitutive_lod=True,
                colors=[1.0, 0.0, 0.0],
                colormap="viridis",
                bogus_attr=1,
            )
        )

        assert_same_refusal(flat, split)
        assert "both 'colors' and 'colormap'" in str(split)
        assert "p" not in compiler.store


class TestLinesSubstitutiveNodeAttrsGate:
    def test_reserved_but_unknown_attr_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_sub_attrs_amp.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_sub_attrs_amp_flat.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=115)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                amplitude_range=[0.0, 1.0],
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                substitutive_lod=True,
                amplitude_range=[0.0, 1.0],
            )
        )

        assert_same_refusal(flat, split)
        assert "amplitude_range" in str(split)
        assert "line" not in compiler.store
        # The gate runs above add_lod_group, so no partial kind=lod node
        # survives finalize() either — the #1529 stranding this closes.
        assert "line" not in finalized_group_keys(compiler, path)

    def test_unknown_attr_typo_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_sub_attrs_typo.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_sub_attrs_typo_flat.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=116)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments", blending="max"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                substitutive_lod=True,
                blending="max",
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "line" not in compiler.store
        assert "line" not in finalized_group_keys(compiler, path)

    def test_truncation_radius_is_accepted_on_both_paths(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_sub_attrs_trunc.luxar.zarr")
        flat_compiler, flat_scene, flat_path = open_scene(
            tmp_path, "lines_sub_attrs_trunc_flat.luxar.zarr"
        )
        vertices = random_positions(_SUB_N, seed=117)

        flat_scene.add_lines(
            "line", vertices, widths=0.2, line_type="segments", truncation_radius=3.0
        )
        flat_compiler.finalize()

        scene.add_lines(
            "line",
            vertices,
            widths=0.2,
            line_type="segments",
            substitutive_lod=True,
            truncation_radius=3.0,
        )
        compiler.finalize()

        flat_store = zarr.open_group(flat_path, mode="r")
        assert flat_store["line"].attrs["type"] == "lines"
        assert flat_store["line"].attrs["truncation_radius"] == 3.0
        assert "kind" not in flat_store["line"].attrs

        store = zarr.open_group(path, mode="r")
        assert store["line"].attrs["kind"] == "lod"
        children = sorted(store["line"].group_keys())
        assert len(children) > 1
        # Same as the Points case above: #1529 asked whether an accepted
        # gsplats-relevant attr reaches the synthesised levels or is silently
        # dropped. It reaches every one of them, matching the flat leaf.
        for child in children:
            assert store["line"][child].attrs["truncation_radius"] == 3.0

    def test_a_colours_fault_still_outranks_the_attrs_gate(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_attrs_prec.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_sub_attrs_prec_flat.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=118)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                colors=[1.0, 0.0, 0.0],
                colormap="viridis",
                bogus_attr=1,
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                substitutive_lod=True,
                colors=[1.0, 0.0, 0.0],
                colormap="viridis",
                bogus_attr=1,
            )
        )

        assert_same_refusal(flat, split)
        assert "both 'colors' and 'colormap'" in str(split)
        assert "line" not in compiler.store


# ---------------------------------------------------------------------------
# Node-attrs gate, pre-split, on the Points/Lines PARTITION path (#1529)
# ---------------------------------------------------------------------------
#
# The gate at the top of add_points_impl / add_lines_impl runs above the
# partition= branch too, not just substitutive_lod=. Pre-fix, a bad attr
# raised the right message but still left a childless kind=partition node for
# the caller's name on disk: validate_render_attrs was reached only from
# add_points_impl (points.py:283) delegating to
# add_points_partition_wrapper_impl, whose per-part
# wrapper.add_points(name=f"part_{i}") call (points.py:521) reaches the
# child's own write_points → node_common.py:528 — add_partition_group only
# creates the wrapper group, it does not itself recurse. 200 elements at
# max_elements=100 gives exactly two
# parts (same construction as tests/group/partition/test_source_validation.py).

_PART_N = 200
_PART_HALF = 100


class TestPointsPartitionNodeAttrsGate:
    def test_unknown_attr_typo_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "points_part_attrs_typo.luxar.zarr"
        )
        _, flat_scene, _ = open_scene(
            tmp_path, "points_part_attrs_typo_flat.luxar.zarr"
        )
        positions = random_positions(_PART_N, seed=123)

        flat = refusal(lambda: flat_scene.add_points("p", positions, blending="max"))
        split = refusal(
            lambda: scene.add_points(
                "p", positions, partition={"max_elements": _PART_HALF}, blending="max"
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "p" not in compiler.store
        # Pre-fix this raised the SAME message but still left a childless
        # kind=partition "p" on disk, surviving finalize() — the #1529
        # stranding this closes.
        assert "p" not in finalized_group_keys(compiler, path)


class TestLinesPartitionNodeAttrsGate:
    def test_unknown_attr_typo_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_part_attrs_typo.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_part_attrs_typo_flat.luxar.zarr")
        vertices = random_positions(_PART_N, seed=124)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments", blending="max"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                partition={"max_elements": _PART_HALF},
                blending="max",
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "line" not in compiler.store
        assert "line" not in finalized_group_keys(compiler, path)


class TestPointsPartitionNodeAttrsGateOutranksTheChannelGate:
    def test_attrs_fault_reported_before_the_channel_fault(self, tmp_path: Any) -> None:
        """Both faults at once: the attrs fault wins, flat and split alike.

        Pre-fix, the #1437 channel gate at the top of the partition wrapper
        ran first (nothing hoisted the attrs check above it), so a call with
        both a bad attr AND a wrong-length channel reported the channel
        fault — while the flat call on the SAME input reported the attrs
        fault, since the flat writer validates attrs (step 0a) before
        channels (steps 0d-0f). The split path now agrees with the flat one.
        """
        compiler, scene, _ = open_scene(
            tmp_path, "points_part_attrs_outranks_channel.luxar.zarr"
        )
        _, flat_scene, _ = open_scene(
            tmp_path, "points_part_attrs_outranks_channel_flat.luxar.zarr"
        )
        positions = random_positions(_PART_N, seed=125)
        # Half-length colours: on their own, this is what the #1437 channel
        # gate refuses.
        wrong_length_colors = np.zeros((_PART_HALF, 3), dtype=np.float32)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, colors=wrong_length_colors, blending="max"
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                partition={"max_elements": _PART_HALF},
                colors=wrong_length_colors,
                blending="max",
            )
        )

        assert "Did you mean 'blending_mode'?" in str(flat)
        assert_same_refusal(flat, split)
        assert "p" not in compiler.store


class TestLinesPartitionNodeAttrsGateOutranksTheChannelGate:
    """Lines counterpart of the Points class above — same construction."""

    def test_attrs_fault_reported_before_the_channel_fault(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "lines_part_attrs_outranks_channel.luxar.zarr"
        )
        _, flat_scene, _ = open_scene(
            tmp_path, "lines_part_attrs_outranks_channel_flat.luxar.zarr"
        )
        vertices = random_positions(_PART_N, seed=131)
        # Half-length colours: on their own, this is what the #1437 channel
        # gate refuses.
        wrong_length_colors = np.zeros((_PART_HALF, 3), dtype=np.float32)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                colors=wrong_length_colors,
                blending="max",
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                partition={"max_elements": _PART_HALF},
                colors=wrong_length_colors,
                blending="max",
            )
        )

        assert "Did you mean 'blending_mode'?" in str(flat)
        assert_same_refusal(flat, split)
        assert "line" not in compiler.store

    def test_wrong_length_colors_alone_trips_the_channel_gate(
        self, tmp_path: Any
    ) -> None:
        """Control: without the attrs fault, the half-length colours alone are
        what the #1437 channel gate refuses — otherwise the test above could
        silently collapse into a duplicate of TestLinesPartitionNodeAttrsGate
        if this fixture's colours ever became valid.
        """
        _, scene, _ = open_scene(
            tmp_path, "lines_part_attrs_outranks_channel_control.luxar.zarr"
        )
        vertices = random_positions(_PART_N, seed=131)
        wrong_length_colors = np.zeros((_PART_HALF, 3), dtype=np.float32)

        alone = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                partition={"max_elements": _PART_HALF},
                colors=wrong_length_colors,
            )
        )

        assert "doesn't match" in str(alone)


# ---------------------------------------------------------------------------
# Node-attrs gate PLACEMENT: entry vs. each wrapper's own kwarg-spec check,
# vs. extend_to_all resolution (#1529)
# ---------------------------------------------------------------------------
#
# The message-parity tests above hold just as well for a WRONG implementation
# that put the same validate_render_attrs() call at the top of each of the
# three wrappers instead of at the adder entry (add_points_impl /
# add_lines_impl, before any branch runs) — every one of them would still
# refuse byte-identically to the flat path, with nothing written. What
# distinguishes the two implementations is PRECEDENCE over each wrapper's own
# kwarg-spec check (a malformed partition= rule, an invalid substitutive_lod=
# compression_factor) and over extend_to_all resolution: only the adder-entry
# placement outranks those too. Each case below fires the OTHER fault alone
# first (to confirm it exists and is what a wrapper-level placement would
# report), then adds a bad node attr and checks the attrs fault wins instead.


class TestPointsPartitionNodeAttrsGateOutranksThePartitionRuleCheck:
    def test_partition_rule_alone_raises_the_rule_error(self, tmp_path: Any) -> None:
        _, scene, _ = open_scene(tmp_path, "points_part_rule_alone.luxar.zarr")
        positions = random_positions(_PART_N, seed=132)

        exc = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                partition={"max_elements": _PART_HALF, "rule": "bogus"},
            )
        )
        assert "partition rule must be" in str(exc)

    def test_attrs_fault_outranks_the_partition_rule_check(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "points_part_rule_outranked.luxar.zarr"
        )
        positions = random_positions(_PART_N, seed=133)

        exc = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                partition={"max_elements": _PART_HALF, "rule": "bogus"},
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "partition rule" not in str(exc)
        assert "p" not in compiler.store


class TestPointsSubstitutiveNodeAttrsGateOutranksTheSpecCheck:
    def test_compression_factor_alone_raises_the_spec_error(
        self, tmp_path: Any
    ) -> None:
        _, scene, _ = open_scene(tmp_path, "points_sub_spec_alone.luxar.zarr")
        positions = random_positions(_SUB_N, seed=134)

        exc = refusal(
            lambda: scene.add_points(
                "p", positions, substitutive_lod={"compression_factor": -1}
            )
        )
        assert "compression_factor must be" in str(exc)

    def test_attrs_fault_outranks_the_spec_check(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "points_sub_spec_outranked.luxar.zarr"
        )
        positions = random_positions(_SUB_N, seed=135)

        exc = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                substitutive_lod={"compression_factor": -1},
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "compression_factor" not in str(exc)
        assert "p" not in compiler.store


class TestFlatPathNodeAttrsGateOutranksExtendToAll:
    """Pins the #1529 precedence change: attrs now beats extend_to_all too.

    Pre-#1529, ``_resolve_extend_to_all`` ran (in the adder, on the flat
    fall-through) before the writer ever reached its own attrs gate, so a
    call tripping both faults reported the ``extend_to_all`` one. The
    adder-entry gate now runs first even on the flat path, so the attrs
    fault wins instead — a deliberate side effect of validating attrs first,
    not a change to any single-fault call's outcome.
    """

    def test_extend_to_all_alone_raises_the_extend_to_all_error(
        self, tmp_path: Any
    ) -> None:
        _, scene, _ = open_scene(tmp_path, "flat_extend_alone.luxar.zarr")
        positions = random_positions(_N, seed=136)

        exc = refusal(lambda: scene.add_points("p", positions, extend_to_all=["NOPE"]))
        assert "Unknown dimension(s) in extend_to_all" in str(exc)

    def test_attrs_fault_outranks_extend_to_all(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "flat_extend_outranked.luxar.zarr")
        positions = random_positions(_N, seed=137)

        exc = refusal(
            lambda: scene.add_points(
                "p", positions, extend_to_all=["NOPE"], blending="max"
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "extend_to_all" not in str(exc)
        assert "p" not in compiler.store


# ---------------------------------------------------------------------------
# Node-attrs gate, pre-split, on the Points/Lines ADDITIVE path (#1529)
# ---------------------------------------------------------------------------
#
# additive_lod= goes straight to the multi-LOD writer, which calls
# validate_render_attrs with NO reserved-attrs set at all — so a genuinely
# RESERVED key (``ordering=``, ``max_radius=`` / ``max_width=``) was reported
# as *unknown* instead of the correct #1221 *reserved* verdict, and a
# ``position_bounds=`` collision didn't raise at all: the multi-LOD writer's
# unreserved call let it straight onto disk, silently clobbering the writer's
# own stamp, and ``finalize()`` then failed later with an unrelated
# ``ValueError: Could not finalize Zarr store: list indices must be integers
# or slices, not str`` (measured: the compiler funnels the underlying
# ``TypeError`` into a ``ValueError``).

_ADD_RESERVED_CASES_POINTS = [("ordering", "morton"), ("max_radius", 5.0)]
_ADD_RESERVED_CASES_LINES = [("ordering", "morton"), ("max_width", 5.0)]


class TestPointsAdditiveNodeAttrsGate:
    @pytest.mark.parametrize("key,value", _ADD_RESERVED_CASES_POINTS)
    def test_reserved_attr_gets_the_reserved_verdict_not_unknown(
        self, tmp_path: Any, key: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_add_attrs_{key}.luxar.zarr")
        _, flat_scene, _ = open_scene(
            tmp_path, f"points_add_attrs_{key}_flat.luxar.zarr"
        )
        positions = random_positions(_N, seed=126)

        flat = refusal(lambda: flat_scene.add_points("p", positions, **{key: value}))
        split = refusal(
            lambda: scene.add_points(
                "p", positions, additive_lod=_POINTS_LADDER, **{key: value}
            )
        )

        assert_same_refusal(flat, split)
        assert f"'{key}'" in str(split)
        assert "are reserved" in str(split)
        assert "p" not in compiler.store

    def test_position_bounds_no_longer_clobbers_the_writers_own_stamp(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "points_add_attrs_posbounds.luxar.zarr"
        )
        positions = random_positions(_N, seed=127)

        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                additive_lod=_POINTS_LADDER,
                position_bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            )
        )

        assert "position_bounds" in str(split)
        assert "are reserved" in str(split)
        assert "p" not in compiler.store
        # Nothing was written, so finalize() must succeed cleanly. Pre-fix,
        # this call did not raise at all and finalize() later failed with an
        # unrelated ValueError once the clobbered position_bounds was read back.
        assert "p" not in finalized_group_keys(compiler, path)


class TestLinesAdditiveNodeAttrsGate:
    @pytest.mark.parametrize("key,value", _ADD_RESERVED_CASES_LINES)
    def test_reserved_attr_gets_the_reserved_verdict_not_unknown(
        self, tmp_path: Any, key: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_add_attrs_{key}.luxar.zarr")
        _, flat_scene, _ = open_scene(
            tmp_path, f"lines_add_attrs_{key}_flat.luxar.zarr"
        )
        vertices = random_positions(_N, seed=128)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments", **{key: value}
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                additive_lod=_LINES_LADDER,
                **{key: value},
            )
        )

        assert_same_refusal(flat, split)
        assert f"'{key}'" in str(split)
        assert "are reserved" in str(split)
        assert "line" not in compiler.store

    def test_position_bounds_no_longer_clobbers_the_writers_own_stamp(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "lines_add_attrs_posbounds.luxar.zarr"
        )
        vertices = random_positions(_N, seed=129)

        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                additive_lod=_LINES_LADDER,
                position_bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            )
        )

        assert "position_bounds" in str(split)
        assert "are reserved" in str(split)
        assert "line" not in compiler.store
        assert "line" not in finalized_group_keys(compiler, path)


# ---------------------------------------------------------------------------
# Node-attrs gate, pre-split, on the Mesh ADDITIVE path (#1534)
# ---------------------------------------------------------------------------
#
# additive_lod= dispatches straight to add_mesh_multi_lod_wrapper_impl's
# writer.write_mesh_multi_lod, whose OWN fail-fast calls validate_render_attrs
# with NO reserved-attrs set at all — the same shape as the Points/Lines
# additive bug (#1529) one geometry type over, and it has BOTH of that bug's
# two symptoms, not only the milder one. For a genuinely mesh-RESERVED key that
# is NOT also in ``_ALLOWED_NODE_ATTRS`` (``ordering=``, ``has_labels=``) the
# unreserved call reports *unknown* instead of the correct *reserved* verdict —
# still refused, nothing written, only the WRONG VERDICT is wrong. For
# ``position_bounds=`` — reserved for mesh, but ALSO listed in
# ``_ALLOWED_NODE_ATTRS`` (it is legitimately unreserved elsewhere, e.g. the
# generic ``write_group`` path) — the unreserved call does not raise at ALL:
# it is accepted, written into every level's parent group, and then correct
# (``global_bounds``) — but the RETURNED ``Mesh`` node's own ``Node.__init__``
# re-persists the caller's stale, un-mutated copy of ``attrs`` right back over
# it (``write_group`` merges and overwrites), silently clobbering the writer's
# own stamp. Measured pre-fix:
# ``add_mesh("m", V, F, additive_lod=True, position_bounds=[[0,1],[0,1],[0,1]])``
# writes a complete 4-level reveal ladder, with ``m.position_bounds`` on disk
# left as the caller's raw list-of-pairs instead of the writer's ``{"min":
# ..., "max": ...}`` dict — then ``compiler.finalize()`` dies with an unrelated
# ``ValueError: Could not finalize Zarr store: list indices must be integers
# or slices, not str`` from ``expand_bounds_with_transforms`` indexing that
# list with ``"min"``/``"max"``. Byte-for-byte the symptom
# ``changelog.d/1529.md`` documents for the Points/Lines additive door, and
# exactly what the dedicated ``test_position_bounds_no_longer_clobbers_the_
# writers_own_stamp`` tests pin for Points and Lines above. Both symptoms are
# fixed the same way: #1534 moved the check to the top of ``add_mesh_impl``,
# above every structural branch, so neither reaches ``write_mesh_multi_lod``
# unreserved any more (parity against the flat leaf, whose ``write_mesh`` call
# already used ``MESH_RESERVED_ATTRS`` and is therefore unchanged by #1534).

_MESH_ADD_RESERVED_CASES = [("ordering", "morton"), ("has_labels", True)]


class TestMeshAdditiveNodeAttrsGate:
    @pytest.mark.parametrize("key,value", _MESH_ADD_RESERVED_CASES)
    def test_reserved_attr_gets_the_reserved_verdict_not_unknown(
        self, tmp_path: Any, key: str, value: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"mesh_add_attrs_{key}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"mesh_add_attrs_{key}_flat.luxar.zarr")
        vertices, faces = grid_mesh(13)

        flat = refusal(
            lambda: flat_scene.add_mesh("m", vertices, faces, **{key: value})
        )
        split = refusal(
            lambda: scene.add_mesh(
                "m", vertices, faces, additive_lod=True, **{key: value}
            )
        )

        assert_same_refusal(flat, split)
        assert f"'{key}'" in str(split)
        assert "are reserved" in str(split)
        assert "m" not in compiler.store
        assert "m" not in finalized_group_keys(compiler, path)

    def test_position_bounds_no_longer_clobbers_the_writers_own_stamp(
        self, tmp_path: Any
    ) -> None:
        """The worse-than-wrong-verdict half of the bug (see the module comment
        above the class): pre-fix this call did not raise at all, wrote a real
        ladder, and then broke ``finalize()`` once the clobbered
        ``position_bounds`` was read back — the Mesh twin of the Points/Lines
        tests of the same name above.
        """
        compiler, scene, path = open_scene(
            tmp_path, "mesh_add_attrs_posbounds.luxar.zarr"
        )
        vertices, faces = grid_mesh(13)

        split = refusal(
            lambda: scene.add_mesh(
                "m",
                vertices,
                faces,
                additive_lod=True,
                position_bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            )
        )

        assert "position_bounds" in str(split)
        assert "are reserved" in str(split)
        assert "m" not in compiler.store
        # Nothing was written, so finalize() must succeed cleanly. Pre-fix, this
        # call did not raise at all — a real ladder was written and finalize()
        # later failed with an unrelated ValueError once the clobbered
        # position_bounds was read back by expand_bounds_with_transforms.
        assert "m" not in finalized_group_keys(compiler, path)


# ---------------------------------------------------------------------------
# Node-attrs gate PLACEMENT, continued: Mesh/GSplats partition-rule and
# extend_to_all precedence (#1534)
# ---------------------------------------------------------------------------
#
# Same construction as TestPointsPartitionNodeAttrsGateOutranksThePartitionRuleCheck
# / TestFlatPathNodeAttrsGateOutranksExtendToAll above, extended to the two
# geometry types #1529 did not cover. Each class fires the OTHER fault alone
# first (proving it exists and is what a wrapper-level — rather than
# adder-entry — placement would report), then adds a bad node attr and checks
# the attrs fault wins instead.


class TestMeshPartitionNodeAttrsGateOutranksThePartitionRuleCheck:
    def test_partition_rule_alone_raises_the_rule_error(self, tmp_path: Any) -> None:
        _, scene, _ = open_scene(tmp_path, "mesh_part_rule_alone.luxar.zarr")
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh(
                "m", vertices, faces, partition={"max_elements": 8, "rule": "bogus"}
            )
        )
        assert "partition rule must be" in str(exc)

    def test_attrs_fault_outranks_the_partition_rule_check(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(tmp_path, "mesh_part_rule_outranked.luxar.zarr")
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh(
                "m",
                vertices,
                faces,
                partition={"max_elements": 8, "rule": "bogus"},
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "partition rule" not in str(exc)
        assert "m" not in compiler.store


class TestGSplatsPartitionNodeAttrsGateOutranksThePartitionRuleCheck:
    def test_partition_rule_alone_raises_the_rule_error(self, tmp_path: Any) -> None:
        _, scene, _ = open_scene(tmp_path, "gsplats_part_rule_alone.luxar.zarr")
        centers = random_positions(_PART_N, seed=138)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=cholesky_rows_nd(_PART_N, 3),
                partition={"max_elements": _PART_HALF, "rule": "bogus"},
            )
        )
        assert "partition rule must be" in str(exc)

    def test_attrs_fault_outranks_the_partition_rule_check(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "gsplats_part_rule_outranked.luxar.zarr"
        )
        centers = random_positions(_PART_N, seed=139)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=cholesky_rows_nd(_PART_N, 3),
                partition={"max_elements": _PART_HALF, "rule": "bogus"},
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "partition rule" not in str(exc)
        assert "g" not in compiler.store


class TestMeshFlatPathNodeAttrsGateOutranksExtendToAll:
    """Pins the #1534 precedence change on Mesh's plain-leaf path.

    Pre-fix, ``_resolve_extend_to_all`` ran in the adder (mesh's flat
    fall-through) before ``write_mesh`` ever reached its own attrs gate, so a
    call tripping both faults reported the ``extend_to_all`` one. The
    adder-entry gate (mirroring #1529 on Points/Lines) now runs first even on
    the flat mesh path, so the attrs fault wins instead.
    """

    def test_extend_to_all_alone_raises_the_extend_to_all_error(
        self, tmp_path: Any
    ) -> None:
        _, scene, _ = open_scene(tmp_path, "mesh_flat_extend_alone.luxar.zarr")
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh("m", vertices, faces, extend_to_all=["NOPE"])
        )
        assert "Unknown dimension(s) in extend_to_all" in str(exc)

    def test_attrs_fault_outranks_extend_to_all(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "mesh_flat_extend_outranked.luxar.zarr"
        )
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh(
                "m", vertices, faces, extend_to_all=["NOPE"], blending="max"
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "extend_to_all" not in str(exc)
        assert "m" not in compiler.store


class TestMeshSubstitutiveNodeAttrsGateOutranksExtendToAll:
    """Pins the #1534 reconciliation named in this module's docstring.

    Mesh's substitutive wrapper used to resolve ``extend_to_all`` (its own
    preflight, inside ``_maybe_add_mesh_substitutive_lod``) BEFORE its attrs
    gate — the opposite of the Points/Lines #1529 order. The entry-level gate
    now sits above that dispatch, so a call tripping both reports the attrs
    fault here too, closing the divergence.
    """

    def test_extend_to_all_alone_raises_the_extend_to_all_error(
        self, tmp_path: Any
    ) -> None:
        _, scene, _ = open_scene(tmp_path, "mesh_sub_extend_alone.luxar.zarr")
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh(
                "m", vertices, faces, substitutive_lod=True, extend_to_all=["NOPE"]
            )
        )
        assert "Unknown dimension(s) in extend_to_all" in str(exc)

    def test_attrs_fault_outranks_extend_to_all(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "mesh_sub_extend_outranked.luxar.zarr"
        )
        vertices, faces = grid_mesh(4)

        exc = refusal(
            lambda: scene.add_mesh(
                "m",
                vertices,
                faces,
                substitutive_lod=True,
                extend_to_all=["NOPE"],
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "extend_to_all" not in str(exc)
        assert "m" not in compiler.store


class TestGSplatsFlatPathNodeAttrsGateOutranksExtendToAll:
    """Pins the #1534 precedence change on GSplats' plain-leaf path.

    Pre-fix, GSplats had no adder-entry attrs gate at all, so
    ``_resolve_extend_to_all`` (run in the adder before the flat write) beat
    the writer's own attrs check on a call tripping both. The new entry gate
    mirrors Points/Lines' #1529 fix and now wins here too.
    """

    def test_extend_to_all_alone_raises_the_extend_to_all_error(
        self, tmp_path: Any
    ) -> None:
        _, scene, _ = open_scene(tmp_path, "gsplats_flat_extend_alone.luxar.zarr")
        centers = random_positions(_N, seed=140)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=cholesky_rows_nd(_N, 3),
                extend_to_all=["NOPE"],
            )
        )
        assert "Unknown dimension(s) in extend_to_all" in str(exc)

    def test_attrs_fault_outranks_extend_to_all(self, tmp_path: Any) -> None:
        compiler, scene, _ = open_scene(
            tmp_path, "gsplats_flat_extend_outranked.luxar.zarr"
        )
        centers = random_positions(_N, seed=141)

        exc = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=cholesky_rows_nd(_N, 3),
                extend_to_all=["NOPE"],
                blending="max",
            )
        )
        assert "Did you mean 'blending_mode'?" in str(exc)
        assert "extend_to_all" not in str(exc)
        assert "g" not in compiler.store
