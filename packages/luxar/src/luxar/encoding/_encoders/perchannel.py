"""Per-channel and scalar dtype encoders: coordinate, color, bounded/positive
scalar, geolog scalar, and the per-channel linear/log/signed-log/geolog family."""

import warnings
from typing import Any, Literal, Optional

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from luxar._zarr_compat import create_array

from ...typing_utils.constants import COORDINATE_U16_MAX_EXTENT
from ..compression import resolve_compressor
from ..modes import EncodingMode
from ..semantic_types import SemanticType
from .base import BaseEncoderMixin
from .delta_codec import probe_delta_filter
from .structural import LUT_SCALAR_MAX_DISTINCT

#: Coordinates are always uint16 -- never uint8 (256 levels is far too coarse for
#: positions). Named once so the grid snap and the encode call cannot drift apart
#: on the width they assume.
_COORD_BITS = 16

#: Number of quantization intervals of the COORDINATE fixed-point grid.
COORDINATE_LEVELS = float(2**_COORD_BITS - 1)

_SCALAR_LUT_PROBE_VALUES = 1024


def _float32_cast_slack(
    arr: np.ndarray, *, check_values: bool = False
) -> Optional[float]:
    """Return an upward pad for the viewer's float32 cast.

    ``check_values`` measures exact stored broadcast/LUT values, falling back
    to the conservative dtype-level bound above the float32 range; otherwise
    that bound covers the precision-mode float32 store.
    """
    if np.can_cast(arr.dtype, np.float32, casting="safe"):
        return None
    if check_values:
        values = arr.astype(np.float64, copy=False)
        with np.errstate(over="ignore"):
            upward = np.asarray(arr, dtype=np.float32).astype(np.float64) - values
        max_upward = max(0.0, float(np.max(upward)))
        if np.isfinite(max_upward):
            return max_upward or None
    max_val = float(np.max(arr))
    return max(
        max_val * float(np.finfo(np.float32).eps),
        float(np.finfo(np.float32).smallest_subnormal),
    )


def gridded_axis_step(
    col: np.ndarray, lo: float, extent: float, levels: float
) -> Optional[tuple[float, int]]:
    """The spacing of the regular grid ``col`` lies on, or ``None``.

    Returns ``(step, n_distinct)``. The test IS the guarantee: a candidate
    spacing is accepted only after replaying the encoder's own quantization and
    the decoder's own dequantization over the distinct values and getting all of
    them back bit-exactly at the COORDINATE decode dtype (float32). That is both
    stricter and more permissive than testing the gaps for equality, and both
    directions matter:

    * A grid with MISSING rungs still qualifies. Frames ``0,1,2,7,8,9`` are what
      a spatial tile of a stacked dataset sees, or what a filter that empties one
      timepoint in one region leaves behind; an "all gaps equal" test rejects
      that axis and leaves it broken by exactly the defect the snap exists to
      prevent.
    * A grid the float32 input only approximates still qualifies, because the
      spacing is least-squares refit against every rung rather than taken from
      the smallest gap. A 0.1 s frame interval jitters by more than a relative
      1e-6 once rounded to float32, so a gap-equality test with any usable
      tolerance rejects it.
    * Conversely, continuous values whose smallest gap merely happens to be
      coarse do NOT qualify, because they do not survive the replay.

    The only cap on distinct values is ``levels`` itself: an axis with more
    distinct values than the encoding has levels cannot be represented on any
    grid, so there is nothing to snap to. Capping lower would reject stacks that
    fit perfectly -- a 10 000-frame timelapse quantizes exactly at u16 -- and
    would save no work, since ``np.unique`` has already run by then.

    Module-level rather than a private encoder method because it is a SHARED
    predicate: the gsplat writer's sigma rail
    (:func:`~luxar.io._compiler.gsplat_assembly._center_quantization_offender`)
    must know whether this encoder will store an axis exactly before deciding to
    escalate it to float32, and the two must never answer differently. Call it
    with exactly the ``lo``/``extent``/``levels`` the encoder will use.

    A caller that needs the distinct values for something ELSE as well —
    :meth:`PerChannelEncoderMixin.coordinate_round_trip_slack` needs the COUNT
    to short-circuit its LUT probe — goes through
    :func:`_gridded_step_from_uniques` with the ``np.unique`` it already ran,
    rather than paying for a second pass. The split is deliberately a PRIVATE
    sibling sharing one body: this public entry point keeps its exact
    signature, so the shared contract cannot acquire a "and pass the right
    uniques" footgun that an out-of-module caller could get wrong.
    """
    return _gridded_step_from_uniques(np.unique(col), lo, extent, levels)


def _gridded_step_from_uniques(
    uniq: np.ndarray, lo: float, extent: float, levels: float
) -> Optional[tuple[float, int]]:
    """:func:`gridded_axis_step`'s body, over an ALREADY-computed ``np.unique``.

    ``uniq`` must be exactly ``np.unique(col)`` — sorted ascending, deduplicated
    — for the column the ``lo``/``extent`` were derived from.
    """
    uniq = np.asarray(uniq, dtype=np.float64)
    if uniq.size < 2 or uniq.size > levels + 1:
        return None
    offsets = uniq - lo
    coarsest = float(np.diff(uniq).min())
    if coarsest <= 0.0 or extent / coarsest > levels:
        return None
    # Rung index of each distinct value on the candidate grid, then a
    # least-squares refit of the spacing against all of them.
    rung = np.round(offsets / coarsest)
    if rung[0] != 0.0 or np.any(np.diff(rung) <= 0.0) or rung[-1] > levels:
        return None
    # Twelve significant digits pin #2855's reduction result before the same
    # value decides snap eligibility and becomes stored coordinate metadata.
    step = float(f"{rung @ offsets / (rung @ rung):.12g}")
    # `span` must be derived exactly as the quantizer will derive it from the
    # stored rails (`hi - lo`), not as `step * levels`: for a large `lo` the two
    # differ in the last bits, and this replay is only a guarantee if it is the
    # same arithmetic.
    candidate = lo + step * levels
    span = candidate - lo
    if span <= 0.0:
        return None
    codes = np.round((np.clip(uniq, lo, candidate) - lo) / span * levels)
    back = lo + codes / levels * span
    if not np.array_equal(back.astype(np.float32), uniq.astype(np.float32)):
        return None
    return step, int(uniq.size)


def _coordinate_u16_slack(
    arr: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[NDArray[np.float64], int]:
    """Per-axis uint16 half-quantum slack and largest axis cardinality."""
    slack = np.zeros(arr.shape[1], dtype=np.float64)
    max_axis_distinct = 1
    for axis in range(arr.shape[1]):
        extent = float(hi[axis] - lo[axis])
        if extent <= 0.0:
            # Constant axis: every value maps to level 0 and decodes to `lo`.
            # One distinct value, so it cannot raise the maximum.
            continue
        uniq = np.unique(arr[:, axis])
        max_axis_distinct = max(max_axis_distinct, int(uniq.size))
        if (
            _gridded_step_from_uniques(uniq, float(lo[axis]), extent, COORDINATE_LEVELS)
            is not None
        ):
            # `_snap_gridded_axes` will snap this axis onto the data's own
            # spacing, and `gridded_axis_step` proved it round-trips exactly by
            # replaying the encode and the decode.
            continue
        slack[axis] = extent / (2.0 * COORDINATE_LEVELS)
    return slack, max_axis_distinct


class PerChannelEncoderMixin(BaseEncoderMixin):
    """Per-channel / scalar dtype encoding strategies for :class:`ArrayEncoder`."""

    def _snap_gridded_axes(
        self, name: str, arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, bits: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Widen `hi` on gridded axes so their values quantize exactly.

        Returns possibly-adjusted ``(lo, hi)``. An axis qualifies when every one of
        its distinct values sits on one regular grid that still fits in the target
        integer width (see :func:`gridded_axis_step`). Anything else is left
        exactly as it was.
        """
        if arr.ndim != 2:
            # No per-axis columns to test; the generic per-column quantizer
            # handles this shape on its own. Leave the scales untouched.
            return lo, hi
        levels = float(2**bits - 1)
        lo = np.array(lo, dtype=np.float64, copy=True)
        hi = np.array(hi, dtype=np.float64, copy=True)
        snapped = []
        for axis in range(arr.shape[1]):
            extent = float(hi[axis] - lo[axis])
            if extent <= 0.0:
                continue  # constant axis: already exact
            found = gridded_axis_step(arr[:, axis], float(lo[axis]), extent, levels)
            if found is None:
                continue
            step, n_distinct = found
            hi[axis] = lo[axis] + step * levels
            snapped.append((axis, n_distinct, step))

        if snapped:
            detail = ", ".join(
                "axis %d (%d values, step %.6g)" % (a, n, st) for a, n, st in snapped
            )
            # Reported, not warned. `warnings.warn` means "the caller should act";
            # this is a transparent correctness fix that costs nothing and needs no
            # action, and warning here would fire on every stacked scene (and break
            # tests that legitimately assert their own code stays silent).
            aprint(f"  ✓ COORDINATE '{name}': grid-snapped {detail} — exact")
        return lo, hi

    def coordinate_round_trip_slack(
        self,
        data: np.ndarray,
        mode: EncodingMode,
        *,
        allow_lut: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        """How far can encoding ``data`` as a COORDINATE move a value, per axis?

        Returns ``None`` when the write is EXACT on every axis, otherwise a
        ``(d,)`` float64 vector of per-axis slack (0.0 on axes that are exact).

        Why this exists: the chunk-bounds writers
        (:func:`~luxar.io._ordering.points.compute_chunk_bounds_points`,
        :func:`~luxar.io._ordering.lines.compute_vertex_chunk_bounds`,
        :func:`~luxar.io._ordering.lines.compute_segment_chunk_bounds`,
        :func:`~luxar.io._ordering.gsplats.compute_chunk_bounds_gsplats`) compute
        a bound the READER trusts for containment, but they see the AUTHORED
        coordinates while the store holds :meth:`_encode_coordinate`'s per-axis
        uint16 fixed point. A decoded coordinate that lands outside its own
        chunk's stored bound is a chunk the viewer never fetches — geometry
        disappears with nothing to notice. So the bound writer has to know how
        far the store will move a coordinate BEFORE it writes the bound, and
        only the encoder knows that. This is that question, asked without
        writing anything.

        It is the CONSERVATIVE half-quantum bound (``extent / 2·levels``), not a
        measured maximum: measuring would mean quantizing the array a second
        time, and the bound writer wants a pad, not a statistic. It is
        deliberately PER-AXIS — the scales are per-axis, so one wide continuous
        axis must not inflate the bounds of a snapped, exactly-stored time axis
        beside it.

        ``allow_lut`` MIRRORS :meth:`~luxar.encoding.encoder.ArrayEncoder.encode`'s
        own parameter and must be passed the same value the write will use. A
        LUT stores the values verbatim, so a LUT-eligible array is exact — but
        only if the write is actually allowed to reach for a LUT. The lines
        writer encodes ``vertices`` with ``allow_lut=False`` (the spatial-index
        loader reads that array as raw chunked zarr), so a LUT-eligible lines
        vertices array is quantized like any other and its bounds need the
        pad; asking with the default ``True`` there would report "exact" and
        write bounds the decoded vertices escape.

        Cost, honestly: the LUT probe
        (:meth:`~luxar.encoding.encoder.ArrayEncoder.encodes_as_lut`) is a
        whole-ARRAY ``np.unique`` and is the DOMINANT term whenever it runs —
        more than the grid loop and the reductions combined — and
        :meth:`encode` recomputes the plan from scratch afterwards, so a caller
        that asks and then encodes pays it twice. Two things keep it off the
        hot path:

        * It is asked LAST, after the cheap exits and the per-axis grid loop,
          and only when some axis came out with nonzero slack: if every axis is
          already exact the answer is ``None`` regardless of the LUT, so the
          probe would be pure waste.
        * It is asked only when no single AXIS already has more distinct values
          than a scalar-mode LUT can hold
          (:data:`~luxar.encoding._encoders.structural.LUT_SCALAR_MAX_DISTINCT`,
          256 — a COORDINATE array is never the ≤4-channel 2-D COLOR shape that
          reaches the uint16 ROW-mode tier). Distinct values in one column are a
          subset of the whole array's, so one column above the cap PROVES the
          array cannot LUT-encode, and the probe can be skipped with the same
          answer. The count is free: the grid loop's own ``np.unique`` supplies
          it, via :func:`_gridded_step_from_uniques`.

        Measured on float32 continuous coordinates, best of 3 — the case that
        used to pay in full, because a continuous array has neither an exact
        axis nor any chance of a LUT:

        ==========  ==========  =========  ==============
        array       before      after      of which probe
        ==========  ==========  =========  ==============
        1M × 3       467 ms      120 ms     352 ms
        5M × 3      3408 ms      793 ms    2804 ms
        ==========  ==========  =========  ==============

        i.e. the predicate now costs what the ``allow_lut=False`` (lines) path
        always cost (112 ms / 713 ms), and the second, redundant ``np.unique``
        inside :meth:`encode` no longer has a first one to be redundant WITH.
        In a whole 1M-point compile the predicate falls from ~350 ms to ~160 ms
        of a ~2 s total. The
        remaining pathological case is genuinely irreducible: an array with
        ≤256 distinct values per axis but more than 256 overall (e.g. three
        disjoint 200-value palettes) still pays one full probe, because only
        the whole-array pass can settle it. Memoizing ``_lut_plan`` would help
        THERE, and nowhere the profiles actually showed.

        Cheap exits first is also what the gsplat sigma rail
        (:func:`~luxar.io._compiler.gsplat_assembly._axis_center_offender`)
        does, and for the same reason.

        The exits below replay :meth:`_encode_coordinate`'s own — reordered as
        described, which is safe because they are independent tests of the same
        array — and deliberately WITHOUT its warnings: this is a query, and a
        duplicate warning at query time would be noise.

        Args:
            data: The coordinate array (N, d) exactly as it will be encoded
            mode: The encoding mode it will be encoded under
            allow_lut: Whether the write will permit a LUT encoding, i.e. the
                ``allow_lut`` the matching :meth:`encode` call passes.

        Returns:
            ``None`` if every axis round-trips exactly, else the per-axis slack.
        """
        if mode not in (EncodingMode.AUTO, EncodingMode.MEMORY):
            # PRECISION is float32 (exact). The only other mode is CUSTOM,
            # which never reaches `_encode_coordinate` at all: `encode` either
            # raises at its `custom_encoder is None` check or routes to
            # `_encode_custom`. Either way no COORDINATE fixed-point store
            # happens here, so there is no displacement for this answer to
            # describe.
            return None

        arr = np.asarray(data).astype(np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            # Not the (N, d) shape this predicate is defined for — NOT a claim
            # of exactness. A 1-D COORDINATE array really is quantized (a
            # `linspace(0, 1000, 5000)` moves by the full half-quantum), but the
            # per-axis scales this answer is expressed in do not exist for it,
            # and the bound builders all require (N, d) and never see anything
            # else. An empty array has nothing to move.
            return None

        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        if not (bool(np.all(np.isfinite(lo))) and bool(np.all(np.isfinite(hi)))):
            # A NaN/inf coordinate has no meaningful displacement, and returning
            # a NaN entry would trip the bound builders' own finiteness check
            # with a misleading message. The compiler's fail-fast position gate
            # rejects such data long before here; this is for a direct caller.
            return None
        if float((hi - lo).max()) >= COORDINATE_U16_MAX_EXTENT:
            return None  # the extent rail falls back to float32 (exact)

        slack, max_axis_distinct = _coordinate_u16_slack(arr, lo, hi)

        if not slack.any():
            return None

        if (
            allow_lut
            and max_axis_distinct <= LUT_SCALAR_MAX_DISTINCT
            and self.encodes_as_lut(data, SemanticType.COORDINATE)
        ):
            # A coordinate LUT's float32 viewer cast is the same rounding map
            # applied by the outward-f32 bound store, so it cannot cross that bound.
            return None
        return slack

    def positive_scalar_round_trip_slack(
        self,
        data: np.ndarray,
        mode: EncodingMode,
        *,
        positive_scalar_encoding: Literal["linear", "log"] = "linear",
        positive_scalar_bits: Optional[Literal[8, 16]] = None,
        allow_lut: bool = True,
    ) -> Optional[float]:
        """How far can encoding ``data`` as POSITIVE_SCALAR enlarge a value?

        Returns ``None`` when the write and viewer decode are exact, otherwise
        one conservative float64 pad for the whole array. The chunk-bounds
        writers add it to a point radius or line width on spatial dimensions
        only, so a decoded footprint cannot escape a bound built from the
        authored scalar.

        The exits mirror :meth:`_encode_positive_scalar` plus the broadcast/LUT
        paths that precede it in :meth:`ArrayEncoder.encode`.
        The answer is valid only when the matching write has deduplication
        disabled, so it cannot resolve to an ``array_ref`` with another
        array's encoding parameters.
        Unlike :meth:`coordinate_round_trip_slack`, this query raises for
        ``CUSTOM``: a POSITIVE_SCALAR write can reach :meth:`_encode_custom`,
        whose arbitrary transform has no displacement model. The coordinate
        sibling returns ``None`` because ``CUSTOM`` does not reach
        :meth:`_encode_coordinate`, so no coordinate fixed-point displacement
        applies.
        Linear quantization uses half a grid quantum; geometric-log encoding
        uses the corresponding half-step at the array maximum, capped at the
        maximum because its grid is anchored there and cannot decode above it.
        Broadcast and LUT values are checked directly for upward float32 cast
        displacement. PRECISION uses a conservative authored-dtype cast term.
        The Python reader's final cast is covered by ``max_val`` times the
        wider epsilon of float32 and the authored dtype, floored at one
        subnormal quantum of either dtype. The viewer additionally needs
        ``1.5 * span * eps32`` for its staged-float32 linear affine chain.
        Geometric-log anchors are already stored at float32 precision; their
        remaining term is the upward displacement of either decoded endpoint
        from the authored minimum or maximum.

        Args:
            data: The positive-scalar array exactly as it will be encoded.
            mode: The encoding mode it will be encoded under.
            positive_scalar_encoding: The matching write's linear/log choice.
            positive_scalar_bits: The matching write's AUTO quantization tier.
                An 8-bit tier selects geometric-log uint8 regardless of the
                linear/log choice.
            allow_lut: Whether the matching write permits exact LUT storage.

        Returns:
            ``None`` when the matching write and viewer decode are exact,
            otherwise one conservative array-wide outward pad.
        """
        arr = np.asarray(data)
        if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            return None
        if mode not in (
            EncodingMode.AUTO,
            EncodingMode.MEMORY,
            EncodingMode.PRECISION,
        ):
            raise ValueError(
                "Cannot bound a CUSTOM POSITIVE_SCALAR array: an arbitrary "
                "custom_encoder has no round-trip displacement model"
            )

        if self._is_uniform(arr):
            first = float(arr.flat[0])
            displacement = max(0.0, first - float(np.min(arr)))
            viewer_cast_slack = _float32_cast_slack(
                np.asarray([first]), check_values=True
            )
            return displacement + (viewer_cast_slack or 0.0) or None

        if mode == EncodingMode.PRECISION:
            return _float32_cast_slack(arr)

        # A scalar LUT has at most 256 values. A small prefix with more
        # distinct values proves the full array cannot take that exit and
        # avoids a second full-array ``np.unique`` on the common continuous
        # radii/widths path (``encode`` performs its own LUT plan later).
        prefix = arr.ravel()[:_SCALAR_LUT_PROBE_VALUES]
        if (
            allow_lut
            and np.unique(prefix).size <= LUT_SCALAR_MAX_DISTINCT
            and self.encodes_as_lut(arr, SemanticType.POSITIVE_SCALAR)
        ):
            return _float32_cast_slack(np.unique(arr), check_values=True)

        return self._positive_scalar_quantization_slack(
            arr, mode, positive_scalar_encoding, positive_scalar_bits
        )

    def _positive_scalar_quantization_slack(
        self,
        arr: np.ndarray,
        mode: EncodingMode,
        positive_scalar_encoding: Literal["linear", "log"],
        positive_scalar_bits: Optional[Literal[8, 16]],
    ) -> Optional[float]:
        """Return a pad for quantization and the Python/viewer decode paths."""

        max_val = float(np.max(arr))
        if max_val == 0.0:
            return None

        bits = self._compute_quantization_bits(arr)
        use_geolog = (
            positive_scalar_encoding == "log"
            or bits == 0
            or (mode == EncodingMode.AUTO and positive_scalar_bits == 8)
        )
        rounding_slack = 0.0
        if use_geolog:
            nonzero = arr[arr > 0].astype(np.float64, copy=False)
            min_val = float(nonzero.min())
            min_log = float(np.float32(np.log(min_val)))
            max_log = float(np.float32(np.log(nonzero.max())))
            quant_bits = (
                8 if mode == EncodingMode.MEMORY or positive_scalar_bits == 8 else 16
            )
            intervals = (1 << quant_bits) - 2
            # The grid is anchored at max_log, so no code decodes above max_val.
            half_step = min(
                float(np.expm1((max_log - min_log) / (2.0 * intervals))), 1.0
            )
            slack = max_val * half_step
            if max_val <= float(np.finfo(np.float32).max):
                with np.errstate(over="ignore"):
                    decoded_min = float(np.float32(np.exp(min_log)))
                    decoded_max = float(np.float32(np.exp(max_log)))
                if np.isfinite(decoded_min) and np.isfinite(decoded_max):
                    rounding_slack = max(
                        0.0,
                        decoded_min - min_val,
                        decoded_max - max_val,
                    )
        else:
            min_val = float(np.min(arr))
            span = max_val - min_val
            if span == 0.0:
                return None
            # With u = eps32 / 2, the viewer's six staged f32 roundings are
            # bounded by u * (min + max + 4 * span). The decode ULP below pays
            # 2u * max, leaving 3u * span = 1.5 * eps32 * span here.
            rounding_slack = 1.5 * span * float(np.finfo(np.float32).eps)
            levels = (1 << bits) - 1
            slack = span / (2.0 * levels)

        decode_eps = float(np.finfo(np.float32).eps)
        decode_floor = float(np.finfo(np.float32).smallest_subnormal)
        if np.issubdtype(arr.dtype, np.floating):
            decode_eps = max(decode_eps, float(np.finfo(arr.dtype).eps))
            decode_floor = max(
                decode_floor, float(np.finfo(arr.dtype).smallest_subnormal)
            )
        decode_ulp = max(max_val * decode_eps, decode_floor)
        return float(
            min(
                slack + decode_ulp + rounding_slack,
                float(np.finfo(np.float64).max),
            )
        )

    def _encode_coordinate(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode COORDINATE semantic type (positions / centers / vertices).

        ``PRECISION`` → float32 (exact). ``AUTO`` / ``MEMORY`` → **uint16 per-axis
        fixed-point** (the generic ``linear_perchannel_u16`` encoding): each axis is
        quantized over its own ``[min, max]`` to 65536 uniform levels and decoded
        back to float32 on read — visually lossless (sub-unit) and ~2× smaller than
        float32. Coordinates never use uint8 (256 levels is far too coarse for
        positions), so MEMORY uses u16 like AUTO.

        float16 is deliberately NOT used: its *relative* precision degrades with
        magnitude (ULP ≈ 2 at coordinate 2048), a footgun for absolute positions.
        uint16 fixed-point is uniform absolute precision. The rail is **array-local**
        (from the data's own per-axis extent): an extent ≥ 2¹⁶ can't resolve a unit
        step at uint16, so AUTO/MEMORY fall back to float32; an extent > 2¹² warns
        that sub-unit headroom is shrinking.

        A GRIDDED axis — one whose distinct values all sit on a single regular
        grid — additionally gets its quantization grid snapped onto the data's own
        spacing (see :func:`gridded_axis_step` / :meth:`_snap_gridded_axes`), so
        it round-trips bit-exactly at uint16. That is what keeps a
        stacked/categorical axis (built with ``sigma=0``) usable, and it costs
        nothing: ``lo``/``hi`` are stored per axis already, so it is a scale
        choice rather than a dtype change.

        Neither rail knows anything but the coordinates. A *second*,
        geometry-aware rail lives at the gsplat write choke point
        (:func:`~luxar.io._compiler.gsplat_assembly.write_gsplat_arrays`), which
        also has the Cholesky factors in hand: it escalates centers to
        ``PRECISION`` when HALF an axis's grid step — the worst-case round-trip
        displacement — exceeds the per-splat marginal σ for more than 0.1% of the
        splats on that axis, i.e. when quantization can move those centers clear
        of their own cores (a population test, so the odd needle splat does not
        cost the array its uint16 win). Tripping that gate is necessary but not
        sufficient: the rail fires only where this encoder has NO exact path of
        its own. It runs :func:`gridded_axis_step` on any axis that trips the
        population gate and skips the axis when the snap will store it exactly,
        and it asks
        :meth:`~luxar.encoding.encoder.ArrayEncoder.encodes_as_lut` before
        escalating, since a LUT-eligible centers array is already stored verbatim
        (exactly, at ~1 B/value). What is left for it is a degenerate
        sub-population on a NON-gridded, non-LUT axis — a ``sigma=0`` track stack
        merged into a fit whose time axis is continuous, say — where the splats
        really are destroyed. Callers encoding COORDINATE data that carries its
        own notion of extent should route through that choke point rather than
        here.

        A caller that instead needs to know HOW FAR this method can move a value
        — the points/lines/gsplats chunk-bounds writers, which must pad a bound
        the reader trusts — asks :meth:`coordinate_round_trip_slack`, directly
        above. It replays these same exits without writing (and takes the same
        ``allow_lut`` the write will use); keep the two in step.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data (N, d)
            mode: Encoding mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        if mode not in (
            EncodingMode.PRECISION,
            EncodingMode.AUTO,
            EncodingMode.MEMORY,
        ):
            raise ValueError(f"Unexpected mode for COORDINATE: {mode}")

        if mode == EncodingMode.PRECISION:
            self._write_float(
                zarr_group, name, data, np.dtype("float32"), chunks, compressor
            )
            return

        # AUTO / MEMORY: uint16 per-axis fixed-point, with an array-local extent rail.
        # Per-axis min/max is computed ONCE here (rail + quantization scales share it).
        arr = np.asarray(data).astype(np.float64)
        lo = hi = None
        if arr.shape[0] > 0:
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            max_extent = float((hi - lo).max())
            if max_extent >= COORDINATE_U16_MAX_EXTENT:
                warnings.warn(
                    f"COORDINATE '{name}': per-axis extent {max_extent:.0f} ≥ 2¹⁶; "
                    "uint16 fixed-point cannot resolve a unit step — storing float32.",
                    UserWarning,
                    stacklevel=2,
                )
                self._write_float(
                    zarr_group, name, data, np.dtype("float32"), chunks, compressor
                )
                return
            if max_extent > 4096.0:
                warnings.warn(
                    f"COORDINATE '{name}': per-axis extent {max_extent:.0f} > 2¹²; "
                    "uint16 fixed-point sub-unit headroom is shrinking "
                    f"(step ≈ {max_extent / 65535.0:.4g}).",
                    UserWarning,
                    stacklevel=2,
                )

        # A GRIDDED axis (few distinct values, regularly spaced) is snapped so the
        # quantization grid coincides with the data's own. This is what keeps a
        # stacked/categorical axis usable: `combine_as_new_dimension(sigma=0)`
        # gives such an axis an effective sigma of 1e-7, so ANY rounding puts a
        # frame thousands of sigma from where it belongs and it stops matching a
        # slice query at all — every frame disappears except the endpoints and
        # the few that coincidentally land on the quantization grid (on a
        # 100-frame stack that is four of them, #1748). Widening `hi` costs
        # nothing: `lo`/`hi` are already stored per
        # axis, so this is a scale choice, not a format or dtype change.
        if lo is not None and hi is not None:
            lo, hi = self._snap_gridded_axes(name, arr, lo, hi, _COORD_BITS)

        # Coordinates always u16 (never u8) for both AUTO and MEMORY. The decode
        # contract for COORDINATE is float32 (GPU/viewer target) regardless of the
        # input dtype — matching PRECISION's float32 cast — so original_dtype is
        # pinned to float32 (an integer original_dtype would truncate on decode).
        self._encode_linear_perchannel(
            zarr_group,
            name,
            arr,
            _COORD_BITS,
            chunks,
            compressor,
            lo=lo,
            hi=hi,
            original_dtype="float32",
        )

    def _encode_color(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        color_mode: Optional[str],
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode COLOR semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            color_mode: "sdr" or "hdr" for COLOR
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)
        # Determine target dtype based on mode and color_mode
        if np.issubdtype(data.dtype, np.integer):
            # Integer input: already quantized, keep as-is
            encoded_data = data
            encoder_name = str(data.dtype)
        elif color_mode == "hdr":
            # HDR colors: wide-range positive per-channel intensities. AUTO →
            # geolog_perchannel_u16, MEMORY → u8 (2026-07 HDR-color spike:
            # per-channel TRUE-log dominates linear and log1p at EVERY
            # measured dynamic range — 2..12.6 decades, 6 datasets; u16 keeps
            # faint-exposure renders ≥147 dB where log1p drops to 88 dB —
            # and float16 was already refuted 4x worse for wide-range
            # positives). PRECISION stays float32.
            if mode == EncodingMode.PRECISION or data.ndim != 2:
                encoded_data = data.astype(np.float32, copy=False)
                encoder_name = "float32"
            elif mode in (EncodingMode.AUTO, EncodingMode.MEMORY):
                self._encode_geolog_perchannel(
                    zarr_group,
                    name,
                    data,
                    16 if mode == EncodingMode.AUTO else 8,
                    chunks=chunks,
                    compressor=compressor,
                )
                return
            else:
                raise ValueError(f"Unexpected mode for HDR COLOR: {mode}")
        elif color_mode == "sdr":
            # SDR colors: can quantize
            if mode == EncodingMode.PRECISION:
                encoded_data = data.astype(np.float32, copy=False)
                encoder_name = "float32"
            elif mode == EncodingMode.MEMORY or mode == EncodingMode.AUTO:
                # Quantize to uint8: [0, 1] → [0, 255]
                encoded_data = self._quantize_normalized_clip(
                    data, 0.0, 1.0, 255, np.dtype(np.uint8)
                )
                encoder_name = "rgb_uint8"
            else:
                raise ValueError(f"Unexpected mode for SDR COLOR: {mode}")
        else:
            raise ValueError("color_mode required for float COLOR arrays")

        # Probe-gated delta: SDR rgb_uint8 and integer-passthrough color codes
        # are Hilbert-ordered like every other per-element array, so spatially
        # coherent colors compress better as columnar residuals. float32
        # encodings decline automatically (probe accepts u8/u16 only).
        comp = resolve_compressor(compressor, encoded_data.dtype)
        create_array(
            zarr_group,
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(encoded_data, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": encoder_name,
            "original_dtype": original_dtype,
        }

    def _encode_bounded_scalar(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        bounds: Optional[tuple[float, float]],
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode BOUNDED_SCALAR semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            bounds: Min/max bounds (None = auto-detect)
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)

        # Determine bounds
        if bounds is None:
            min_val = float(np.min(data))
            max_val = float(np.max(data))
        else:
            min_val, max_val = bounds
            # Validate data is within bounds
            if np.any(data < min_val) or np.any(data > max_val):
                raise ValueError(
                    f"Data outside specified bounds [{min_val}, {max_val}]: "
                    f"found [{np.min(data)}, {np.max(data)}]"
                )

        metadata: dict[str, Any]
        # Select encoding based on mode
        if mode == EncodingMode.PRECISION:
            # No quantization
            encoded_data = data.astype(np.float32, copy=False)
            encoder_name = "float32"
            metadata = {"name": encoder_name, "original_dtype": original_dtype}
        elif mode == EncodingMode.MEMORY or mode == EncodingMode.AUTO:
            span = max_val - min_val
            if span == 0:
                # Degenerate case: all values equal - use uint8
                encoded_data = np.zeros_like(data, dtype=np.uint8)
                encoder_name = "bounded_scalar_uint8"
                metadata = {
                    "name": encoder_name,
                    "min": min_val,
                    "max": max_val,
                    "bits": 8,
                    "original_dtype": original_dtype,
                }
            else:
                # Choose quantization based on dynamic range. Measure on
                # MAGNITUDES: _compute_quantization_bits sizes the range from
                # positive values only (its documented "non-negative" contract),
                # so signed data (newly reachable via colormap scalars) would
                # otherwise ignore its negative span — an all-negative array
                # would collapse to uint8. Signed integers are upcast first:
                # np.abs wraps at the type minimum (|int64 min| stays
                # negative), which would corrupt both the bit selection and
                # the float16 guard below. For non-negative data np.abs is a
                # no-op, so existing callers stay byte-identical.
                if np.issubdtype(data.dtype, np.signedinteger):
                    magnitudes = np.abs(data.astype(np.float64))
                else:
                    magnitudes = np.abs(data)
                bits = self._compute_quantization_bits(magnitudes)

                if bits == 8:
                    # Dynamic range <= 256, uint8 is sufficient
                    encoded_data = self._quantize_normalized_clip(
                        data, min_val, span, 255, np.dtype(np.uint8)
                    )
                    encoder_name = "bounded_scalar_uint8"
                    metadata = {
                        "name": encoder_name,
                        "min": min_val,
                        "max": max_val,
                        "bits": 8,
                        "original_dtype": original_dtype,
                    }
                elif bits == 16:
                    # Dynamic range <= 65536, uint16 is sufficient
                    encoded_data = self._quantize_normalized_clip(
                        data, min_val, span, 65535, np.dtype(np.uint16)
                    )
                    encoder_name = "bounded_scalar_uint16"
                    metadata = {
                        "name": encoder_name,
                        "min": min_val,
                        "max": max_val,
                        "bits": 16,
                        "original_dtype": original_dtype,
                    }
                else:
                    # Dynamic range too wide for integer quantization → float.
                    # float16 tops out at 65504; fall back to float32 when any
                    # value would overflow to inf (newly reachable for
                    # signed/wide colormap scalars).
                    if self._float16_allowed and float(np.max(magnitudes)) <= 65504.0:
                        encoded_data = data.astype(np.float16)
                        encoder_name = "float16"
                    else:
                        encoded_data = data.astype(np.float32)
                        encoder_name = "float32"
                    metadata = {"name": encoder_name, "original_dtype": original_dtype}
        else:
            raise ValueError(f"Unexpected mode for BOUNDED_SCALAR: {mode}")

        comp = resolve_compressor(compressor, encoded_data.dtype)
        create_array(
            zarr_group,
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(encoded_data, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = metadata

    def _encode_positive_scalar(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        encoding_type: str,
        bits: Optional[int] = None,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode POSITIVE_SCALAR semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            encoding_type: "linear" or "log" encoding
            bits: Optional AUTO quantization tier. An 8-bit tier selects
                geometric-log uint8 regardless of ``encoding_type``.
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)
        metadata: dict[str, Any]
        auto_bits = bits if bits is not None else 16

        if mode == EncodingMode.PRECISION:
            # No quantization
            encoded_data = data.astype(np.float32)
            encoder_name = "float32"
            metadata = {"name": encoder_name, "original_dtype": original_dtype}
        elif mode == EncodingMode.MEMORY or mode == EncodingMode.AUTO:
            # Analyze range
            max_val = float(np.max(data))

            if mode == EncodingMode.AUTO and bits == 8 and max_val > 0:
                self._encode_geolog_scalar(
                    zarr_group,
                    name,
                    data,
                    8,
                    chunks,
                    compressor,
                )
                return
            if encoding_type == "log" and max_val > 0:
                # Geometric-log encoding (min/max-anchored, reserved zero
                # level): uniform RELATIVE precision across the whole range,
                # and by construction no nonzero value can decode to zero.
                self._encode_geolog_scalar(
                    zarr_group,
                    name,
                    data,
                    auto_bits if mode == EncodingMode.AUTO else 8,
                    chunks,
                    compressor,
                )
                return
            elif encoding_type == "log":
                # All zeros with explicit log request: degenerate constant.
                encoded_data = np.zeros_like(data, dtype=np.uint8)
                encoder_name = "bounded_scalar_uint8"
                metadata = {
                    "name": encoder_name,
                    "min": 0.0,
                    "max": 0.0,
                    "bits": 8,
                    "original_dtype": original_dtype,
                }
            else:
                # Linear encoding - choose dtype based on dynamic range
                quantization_bits = self._compute_quantization_bits(data)

                if max_val == 0:
                    # All zeros - use uint8
                    encoded_data = np.zeros_like(data, dtype=np.uint8)
                    encoder_name = "bounded_scalar_uint8"
                    metadata = {
                        "name": encoder_name,
                        "min": 0.0,
                        "max": 0.0,
                        "bits": 8,
                        "original_dtype": original_dtype,
                    }
                elif quantization_bits in (8, 16):
                    # Rescale-first: anchor the grid at the array's OWN
                    # [min, max] rather than [0, max] — data far from zero
                    # (e.g. radii in [10, 11]) no longer wastes code space
                    # on the empty [0, min) span. min == 0 whenever the data
                    # contains zeros, so zero handling is unchanged.
                    min_val = float(np.min(data))
                    span = max(max_val - min_val, 0.0)
                    levels = 255 if quantization_bits == 8 else 65535
                    if span == 0.0:
                        encoded_data = np.zeros_like(
                            data,
                            dtype=np.uint8 if quantization_bits == 8 else np.uint16,
                        )
                    else:
                        encoded_data = self._quantize_normalized_clip(
                            data,
                            min_val,
                            span,
                            levels,
                            np.dtype(np.uint8 if quantization_bits == 8 else np.uint16),
                        )
                    encoder_name = f"bounded_scalar_uint{quantization_bits}"
                    metadata = {
                        "name": encoder_name,
                        "min": min_val,
                        "max": max_val,
                        "bits": quantization_bits,
                        "original_dtype": original_dtype,
                    }
                else:
                    # Dynamic range > 65536: linear quantization cannot hold
                    # both ends, and float32 wastes 4 B on a relative-precision
                    # quantity. Rescale-first geometric-log quantization keeps
                    # uniform relative precision across the whole range
                    # (AUTO -> u16, ~0.01% for 7 decades; MEMORY -> u8).
                    self._encode_geolog_scalar(
                        zarr_group,
                        name,
                        data,
                        auto_bits if mode == EncodingMode.AUTO else 8,
                        chunks,
                        compressor,
                    )
                    return
        else:
            raise ValueError(f"Unexpected mode for POSITIVE_SCALAR: {mode}")

        comp = resolve_compressor(compressor, encoded_data.dtype)
        create_array(
            zarr_group,
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(encoded_data, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = metadata

    def _encode_geolog_scalar(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        bits: int,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Write a wide-dynamic-range positive scalar as ``geolog_scalar_uint{bits}``.

        Rescale-first log quantization: the grid is anchored to the array's OWN
        nonzero ``[min, max]`` (stored as ``min_log``/``max_log``), the same
        array-local principle as the COORDINATE fixed-point grid and the
        per-column Cholesky scales. Level 0 is reserved for exact zeros.
        """
        original_dtype = str(data.dtype)
        x = np.asarray(data, dtype=np.float64)
        nz = x > 0
        min_log = float(np.float32(np.log(x[nz].min())))
        max_log = float(np.float32(np.log(x[nz].max())))
        codes = self._geolog_forward(x, bits, min_log, max_log)
        # Invariant, not a data property: the reserved zero level makes
        # nonzero -> 0 impossible; a violation would be an encoder bug.
        if bool(((codes == 0) & nz).any()):
            raise RuntimeError(
                f"geolog encoding bug: nonzero values of '{name}' mapped to "
                "the reserved zero level"
            )
        comp = resolve_compressor(compressor, codes.dtype)
        create_array(
            zarr_group,
            name,
            data=codes,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(codes, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"geolog_scalar_uint{bits}",
            "min_log": min_log,
            "max_log": max_log,
            "bits": bits,
            "original_dtype": original_dtype,
        }

    def _encode_linear_perchannel(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        bits: int,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        *,
        lo: Optional[np.ndarray] = None,
        hi: Optional[np.ndarray] = None,
        original_dtype: Optional[str] = None,
    ) -> None:
        """Generic per-channel LINEAR (fixed-point) quantization of an (N, C) array.

        The identity-transform sibling of ``_encode_log_perchannel`` /
        ``_encode_signed_log_perchannel``: each column is quantized over its own
        ``[lo, hi]`` to ``2**bits`` uniform levels — no companding, so it handles
        negative values (the correct transform for coordinates). Unlike the log
        siblings this takes ``bits`` directly rather than a mode: its consumers
        (currently COORDINATE) own the mode policy. ``lo``/``hi`` accept
        precomputed per-column scales; ``original_dtype`` overrides the stored
        decode dtype (COORDINATE pins it to float32, the decode contract).
        """
        y = np.asarray(data).astype(np.float64)
        u, lo_list, hi_list = self._quantize_per_column(y, bits, lo=lo, hi=hi)
        comp = resolve_compressor(compressor, u.dtype)
        create_array(
            zarr_group,
            name,
            data=u,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(u, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"linear_perchannel_u{bits}",
            "col_lo": lo_list,
            "col_hi": hi_list,
            "bits": bits,
            "original_dtype": original_dtype or str(data.dtype),
        }

    def _encode_log_perchannel(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        bits: Optional[int] = None,
    ) -> None:
        """Generic per-channel LOG quantization of a non-negative (N, C) array.

        Each channel (column) gets its own ``[lo, hi]`` in log space, so a wide
        per-channel dynamic range keeps relative precision. Scales are anchored
        at each column's NONZERO min/max and code 0 is a RESERVED ZERO LEVEL
        (``zero_level: true``): entries ``<= 0`` round-trip to exactly 0 and
        never consume code range (rescale-first, as ``geolog_scalar``).
        PRECISION → float32; AUTO/MEMORY → ``log_perchannel_u8``. AUTO
        escalates to ``log_perchannel_u16`` only through
        :meth:`encode_cholesky_split`, whose encode-time certificate measures
        the actual Σ reconstruction error — pair callers should use that entry
        point. (First consumer: the Cholesky diagonal; reusable for any
        positive per-channel field.)
        """
        original_dtype = str(data.dtype)
        if mode == EncodingMode.PRECISION:
            self._write_float(
                zarr_group, name, data, np.dtype("float32"), chunks, compressor
            )
            return
        bits = bits or 8
        x = np.asarray(data)
        lo, hi = self._perchannel_log_scales(x, signed=False)
        y = self._perchannel_log_forward(x, signed=False)
        nonzero = self._perchannel_nonzero_mask(x, signed=False)
        u = self._quantize_perchannel_zero_level(y, nonzero, bits, lo, hi)
        comp = resolve_compressor(compressor, u.dtype)
        create_array(
            zarr_group,
            name,
            data=u,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(u, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"log_perchannel_u{bits}",
            "col_lo": lo.tolist(),
            "col_hi": hi.tolist(),
            "bits": bits,
            "zero_level": True,
            "original_dtype": original_dtype,
        }

    def _encode_signed_log_perchannel(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        bits: Optional[int] = None,
    ) -> None:
        """Generic per-channel SIGNED-LOG quantization of a signed (N, C) array.

        ``y = sign(x)·log1p(|x|)`` companding gives fine resolution near zero
        (where most mass of a zero-centred signal lives) and coarse in the tails,
        with per-channel ``[lo, hi]`` anchored at each column's NONZERO min/max
        and code 0 a RESERVED ZERO LEVEL (``zero_level: true``) — exact zeros
        (e.g. the off-diagonal of an axis-aligned splat) round-trip to exactly
        0 instead of a tiny spurious correlation. PRECISION → float32;
        AUTO/MEMORY → ``signed_log_perchannel_u8``, with AUTO escalation to u16
        owned by :meth:`encode_cholesky_split` (see ``_encode_log_perchannel``).
        (First consumer: the Cholesky off-diagonal.)
        """
        original_dtype = str(data.dtype)
        if mode == EncodingMode.PRECISION:
            self._write_float(
                zarr_group, name, data, np.dtype("float32"), chunks, compressor
            )
            return
        bits = bits or 8
        x = np.asarray(data)
        lo, hi = self._perchannel_log_scales(x, signed=True)
        y = self._perchannel_log_forward(x, signed=True)
        nonzero = self._perchannel_nonzero_mask(x, signed=True)
        u = self._quantize_perchannel_zero_level(y, nonzero, bits, lo, hi)
        comp = resolve_compressor(compressor, u.dtype)
        create_array(
            zarr_group,
            name,
            data=u,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(u, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"signed_log_perchannel_u{bits}",
            "col_lo": lo.tolist(),
            "col_hi": hi.tolist(),
            "bits": bits,
            "zero_level": True,
            "original_dtype": original_dtype,
        }

    def _encode_geolog_perchannel(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        bits: int,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Per-channel TRUE-log quantization of a positive (N, C) array.

        The per-channel member of the geolog family (scalar sibling:
        ``geolog_scalar``; identity sibling: ``linear_perchannel``; log1p
        siblings: ``log_perchannel`` / ``signed_log_perchannel``): each
        column is quantized on a min/max-anchored geometric grid in log
        space — ``y = ln(x)`` over the column's own positive ``[min, max]``
        — so relative precision is uniform across the column's entire
        dynamic range. Code 0 is the RESERVED ZERO LEVEL (exact zeros — and
        policy-clamped negatives — round-trip to exactly 0; no positive
        input can quantize to zero by construction); nonzero codes span
        ``1..2**bits - 1`` with denominator ``2**bits - 2``. First consumer:
        HDR colors (AUTO → u16, MEMORY → u8).
        """
        original_dtype = str(data.dtype)
        x = np.asarray(data, dtype=np.float64)
        lo, hi = self._perchannel_geolog_scales(x)
        nonzero = self._perchannel_nonzero_mask(x, signed=False)
        # log of positives only; zero-level entries never read y (masked to
        # code 0 by the quantizer), the 1.0 placeholder just avoids log(0).
        y = np.log(np.where(nonzero, x, 1.0))
        u = self._quantize_perchannel_zero_level(y, nonzero, bits, lo, hi)
        comp = resolve_compressor(compressor, u.dtype)
        create_array(
            zarr_group,
            name,
            data=u,
            chunks=chunks,
            compressor=comp,
            filters=probe_delta_filter(u, chunks, comp),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"geolog_perchannel_u{bits}",
            "col_lo": lo.tolist(),
            "col_hi": hi.tolist(),
            "bits": bits,
            "zero_level": True,
            "original_dtype": original_dtype,
        }
