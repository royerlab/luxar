"""Shared foundation mixin: instance-attribute contract, cross-mixin stubs,
and the low-level quantization/validation primitives reused across domains."""

from typing import Any, Literal, Optional, Union

import numpy as np
import zarr

from luxar._zarr_compat import create_array

from ...validation.base import _validate_numeric_finite_values
from ..compression import resolve_compressor
from ..modes import EncodingMode
from ..registry import ArrayRefRegistry
from ..semantic_types import SemanticType


class BaseEncoderMixin:
    """Foundation the domain mixins inherit so cross-mixin ``self.`` calls
    resolve (and type-check) through a single base.

    ``ArrayEncoder.__init__`` assigns the instance attributes annotated below;
    the three method stubs (``encode`` / ``_encode_dtype`` / ``encodes_as_lut``)
    stay implemented on ``ArrayEncoder`` but are declared here so mixins that
    call them type-check.
    """

    # Instance attributes assigned by ArrayEncoder.__init__.
    _registry: ArrayRefRegistry
    _broadcast_rtol: float
    _broadcast_atol: float
    _float16_allowed: bool
    _lut_json_max_bytes: int

    def encode(
        self,
        data: Union[np.ndarray, float, int, np.number, tuple, list],
        zarr_group: zarr.Group,
        name: str,
        semantic_type: SemanticType,
        mode: EncodingMode = EncodingMode.AUTO,
        n_elements: Optional[int] = None,
        bounds: Optional[tuple[float, float]] = None,
        positive_scalar_encoding: Literal["linear", "log"] = "linear",
        custom_encoder: Optional[str] = None,
        color_mode: Optional[Literal["sdr", "hdr"]] = None,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        deduplicate: bool = True,
        allow_lut: bool = True,
        _perchannel_bits: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def _encode_dtype(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        semantic_type: SemanticType,
        mode: EncodingMode,
        bounds: Optional[tuple[float, float]],
        positive_scalar_encoding: str,
        custom_encoder: Optional[str],
        color_mode: Optional[str],
        chunks: Optional[tuple],
        compressor: Optional[Any],
        perchannel_bits: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def encodes_as_lut(self, data: np.ndarray, semantic_type: SemanticType) -> bool:
        raise NotImplementedError

    def _is_uniform(self, data: np.ndarray) -> bool:
        """Check if all values are identical (within tolerance).

        For 1D arrays: checks if all elements equal the first element.
        For 2D arrays: checks if all rows equal the first row.

        Args:
            data: Array to check

        Returns:
            True if array is uniform (all values or rows identical)
        """
        if self._broadcast_rtol == 0.0 and self._broadcast_atol == 0.0:
            # Exact equality (default, safe for all types)
            if data.ndim == 1:
                return bool(np.all(data == data[0]))
            else:
                # For 2D arrays, check if all rows equal first row
                return bool(np.all(data == data[0]))
        else:
            # Approximate equality (use with caution)
            if data.ndim == 1:
                reference = data[0]
            else:
                reference = data[0]

            return bool(
                np.allclose(
                    data,
                    reference,
                    rtol=self._broadcast_rtol,
                    atol=self._broadcast_atol,
                )
            )

    def _compute_quantization_bits(self, data: np.ndarray) -> int:
        """Compute optimal quantization bits based on dynamic range.

        Analyzes the ratio between maximum and minimum non-zero values
        to determine if 8-bit, 16-bit, or floating-point encoding is needed.

        Args:
            data: Input array (must be non-negative for meaningful results)

        Returns:
            8: if dynamic range <= 256 (uint8 sufficient)
            16: if dynamic range <= 65536 (uint16 sufficient)
            0: if float should be used (very wide dynamic range)

        Note:
            For data with all zeros, returns 8 (any precision is fine).
            For data with only one unique non-zero value, returns 8.
        """
        max_val = float(np.max(data))
        if max_val == 0:
            return 8  # All zeros, any precision is fine

        # Find minimum non-zero value
        nonzero_mask = data > 0
        if not np.any(nonzero_mask):
            return 8  # All zeros

        min_nonzero = float(np.min(data[nonzero_mask]))
        if min_nonzero == 0:
            return 8  # Should not happen, but be safe

        dynamic_range = max_val / min_nonzero

        if dynamic_range <= 256:
            return 8  # uint8 sufficient
        elif dynamic_range <= 65536:
            return 16  # uint16 sufficient
        else:
            return 0  # Use float (dynamic range too wide for integer quantization)

    def _validate_input(
        self,
        data: np.ndarray,
        semantic_type: SemanticType,
        color_mode: Optional[str],
    ) -> None:
        """Validate input data meets semantic type constraints.

        Args:
            data: Input array
            semantic_type: Semantic type to validate against
            color_mode: Color mode for COLOR type

        Raises:
            ValueError: If constraints are violated
        """
        # EN-1: route NaN/Inf detection through the canonical validator so
        # error messages, suggestions, and fast-path semantics stay in lock-
        # step with `validation.base._validate_numeric_finite_values`. The
        # helper handles both NaN and Inf in a single np.isfinite scan and
        # raises ValidationError (a ValueError subclass).
        if np.issubdtype(data.dtype, np.number):
            _validate_numeric_finite_values(data, "input data")

        # Semantic type specific validation
        if semantic_type == SemanticType.COLOR:
            # Colors must be non-negative
            if np.any(data < 0):
                raise ValueError("COLOR semantic type requires non-negative values")

            if np.issubdtype(data.dtype, np.integer):
                if data.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                    raise ValueError(
                        "Integer COLOR arrays must use dtype uint8 or uint16 "
                        f"(got {data.dtype})"
                    )
                if color_mode not in (None, "sdr"):
                    raise ValueError(
                        "Integer COLOR arrays are SDR; color_mode must be None or 'sdr'"
                    )

            # Float colors require explicit color_mode
            if np.issubdtype(data.dtype, np.floating):
                if color_mode is None:
                    raise ValueError(
                        "Float COLOR arrays require explicit color_mode "
                        "parameter ('sdr' or 'hdr')"
                    )
                if color_mode not in ("sdr", "hdr"):
                    raise ValueError(
                        f"color_mode must be 'sdr' or 'hdr', got {color_mode!r}"
                    )
                if color_mode == "sdr":
                    # SDR mode: values must be in [0, 1]
                    if np.any(data > 1.0):
                        raise ValueError(
                            "SDR color_mode requires values in [0, 1] "
                            f"(found max={np.max(data)})"
                        )

        elif semantic_type == SemanticType.POSITIVE_SCALAR:
            # Positive scalars must be non-negative
            if np.any(data < 0):
                raise ValueError(
                    "POSITIVE_SCALAR semantic type requires non-negative values"
                )

        elif semantic_type == SemanticType.BOUNDED_SCALAR:
            # No specific constraint (bounds will be validated in encoding)
            pass

        elif semantic_type == SemanticType.INDEX:
            # Indices must be non-negative integers
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError("INDEX semantic type requires integer dtype")
            if np.any(data < 0):
                raise ValueError("INDEX semantic type requires non-negative values")

    @staticmethod
    def _quantize_normalized_clip(
        data: np.ndarray,
        min_val: float,
        span: float,
        levels: int,
        dtype: np.dtype,
        *,
        round_values: bool = True,
    ) -> np.ndarray:
        """Affine-normalize to ``[0, levels]`` then clip-and-cast to ``dtype``.

        The shared quantization core of the bounded / positive / custom scalar
        encoders: map ``[min_val, min_val + span] → [0, levels]``, optionally
        round (vs truncate toward zero), clip to ``[0, levels]``, and cast.
        Callers own the degenerate ``span == 0`` (all-equal) branch — this
        assumes ``span != 0``. ``round_values=False`` reproduces the custom
        encoder's historical truncating behaviour exactly. The affine map uses
        float64 so narrow input dtypes cannot overflow or change its precision.
        """
        working = data.astype(np.float64, copy=False)
        normalized = (working - min_val) / span
        scaled = normalized * levels
        if round_values:
            scaled = np.round(scaled)
        return np.clip(scaled, 0, levels).astype(dtype)

    @staticmethod
    def _quantize_per_column(
        y: np.ndarray,
        bits: int,
        lo: Optional[np.ndarray] = None,
        hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list, list]:
        """Quantize a 2-D (N, C) array to uint with PER-COLUMN min/max.

        Returns ``(uint_array, lo, hi)`` where ``lo``/``hi`` are length-C lists
        (one min/max per column). A constant column (lo==hi) maps every value to
        level 0; decode then returns ``lo`` for that column. Callers that already
        computed the per-column min/max (e.g. the COORDINATE extent rail) can pass
        ``lo``/``hi`` to skip the redundant reduction pass.

        Idempotency: quantization error does NOT compound across save/load
        cycles. After the first encode→decode→re-encode, the decoded values lie
        within ``[lo, hi]`` per column, so re-deriving ``lo``/``hi`` here yields
        the same scales and the same codes — subsequent cycles add zero error.

        A 1-D input is treated as ONE column and its rails are still emitted as
        length-1 LISTS. The reductions below give 0-d scalars there, and a bare
        scalar ``col_lo``/``col_hi`` is not decodable — the decoder requires
        equal-length 1-D rails and raises on shapes ``()``, so the array would
        be written successfully and be unreadable. ``np.atleast_1d`` is a no-op
        on the (N, C) path every production writer uses.
        """
        levels = (1 << bits) - 1
        udtype = np.uint8 if bits == 8 else np.uint16
        if y.shape[0] == 0:  # defensive; empty arrays are handled before dispatch
            lo = np.zeros(y.shape[1], dtype=np.float64)
            hi = np.zeros(y.shape[1], dtype=np.float64)
            return y.astype(udtype), lo.tolist(), hi.tolist()
        if lo is None:
            lo = y.min(axis=0)
        if hi is None:
            hi = y.max(axis=0)
        rng = np.maximum(hi - lo, 1e-30)
        u = np.round((np.clip(y, lo, hi) - lo) / rng * levels).astype(udtype)
        return (
            u,
            np.atleast_1d(lo).astype(np.float64).tolist(),
            np.atleast_1d(hi).astype(np.float64).tolist(),
        )

    @staticmethod
    def _geolog_forward(
        data: np.ndarray, bits: int, min_log: float, max_log: float
    ) -> np.ndarray:
        """Compand + quantize a non-negative array to geolog codes.

        Level 0 is RESERVED for exact zeros; nonzero values map to levels
        ``[1, 2**bits - 1]`` over ``y = ln(x)`` rescaled to
        ``[min_log, max_log]`` — so no nonzero input can quantize to zero,
        and relative precision is uniform across the whole range.
        """
        top = (1 << bits) - 1
        codes = np.zeros(data.shape, dtype=np.uint16 if bits > 8 else np.uint8)
        nz = data > 0
        if nz.any():
            y = np.log(data[nz].astype(np.float64))
            rng = max(max_log - min_log, 0.0)
            if rng == 0.0 or top == 1:
                u = np.ones(y.shape, dtype=np.float64)
            else:
                u = 1 + np.round(np.clip((y - min_log) / rng, 0.0, 1.0) * (top - 1))
            codes[nz] = u.astype(codes.dtype)
        return codes

    @staticmethod
    def _geolog_inverse(
        codes: np.ndarray, bits: int, min_log: float, max_log: float
    ) -> np.ndarray:
        """Exact inverse of :meth:`_geolog_forward` (mirrors the decoders)."""
        top = (1 << bits) - 1
        out = np.zeros(codes.shape, dtype=np.float64)
        nz = codes > 0
        if nz.any():
            rng = max(max_log - min_log, 0.0)
            denom = max(top - 1, 1)
            out[nz] = np.exp(min_log + (codes[nz].astype(np.float64) - 1) / denom * rng)
        return out.astype(np.float32)

    @staticmethod
    def _perchannel_log_forward(data: np.ndarray, *, signed: bool) -> np.ndarray:
        """Compand an (N, C) array into log space (the exact encode transform)."""
        x = data.astype(np.float64)
        if signed:
            return np.asarray(np.sign(x) * np.log1p(np.abs(x)))
        return np.asarray(np.log1p(np.maximum(x, 0.0)))

    @staticmethod
    def _perchannel_nonzero_mask(data: np.ndarray, *, signed: bool) -> np.ndarray:
        """Entries that get a real code (the rest take the reserved zero level).

        Unsigned columns clamp negatives to 0 by policy (see the forward
        compand), so anything ``<= 0`` decodes to exactly 0 via the reserved
        level rather than to ``expm1(col_lo)``.
        """
        return np.asarray(data != 0 if signed else data > 0)

    @staticmethod
    def _perchannel_log_scales(
        data: np.ndarray, *, signed: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-column COMPANDED-domain scales over the NONZERO entries only.

        Zero entries take the reserved zero level, so anchoring the scales at
        zero would waste code range on values that never use it — the exact
        anti-pattern ``geolog_scalar`` removed for scalars (rescale first).
        The forward compand is monotonic, so companding each column's raw
        nonzero min/max equals the companded nonzero min/max; the column loop
        keeps peak memory at one column (the covariance certificate feeds
        full multi-million-row arrays through this). A column with no nonzero
        entries gets ``lo == hi == 0``.
        """
        x = np.asarray(data)
        n_cols = x.shape[1]
        raw_lo = np.zeros(n_cols, dtype=np.float64)
        raw_hi = np.zeros(n_cols, dtype=np.float64)
        for c in range(n_cols):
            col = x[:, c]
            nz = col[BaseEncoderMixin._perchannel_nonzero_mask(col, signed=signed)]
            if nz.size:
                raw_lo[c] = float(nz.min())
                raw_hi[c] = float(nz.max())
        lo = BaseEncoderMixin._perchannel_log_forward(raw_lo[None, :], signed=signed)
        hi = BaseEncoderMixin._perchannel_log_forward(raw_hi[None, :], signed=signed)
        return lo[0], hi[0]

    @staticmethod
    def _quantize_perchannel_zero_level(
        y: np.ndarray,
        nonzero: np.ndarray,
        bits: int,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Per-column quantization with a RESERVED ZERO LEVEL.

        Code 0 is reserved for exact zeros (decode returns exactly 0); nonzero
        entries map to codes ``1..2**bits-1`` spanning each column's companded
        ``[lo, hi]``. The reserved level makes exact zeros round-trip exactly
        (an axis-aligned splat keeps zero correlations) and lets the scales
        come from the nonzero entries only. Same layout as ``geolog_scalar``.
        """
        top = (1 << bits) - 1
        udtype = np.uint8 if bits == 8 else np.uint16
        rng = np.maximum(hi - lo, 1e-30)
        codes = 1.0 + np.round((np.clip(y, lo, hi) - lo) / rng * (top - 1))
        return np.asarray(np.where(nonzero, codes, 0.0).astype(udtype))

    @staticmethod
    def _perchannel_log_roundtrip(
        data: np.ndarray,
        bits: int,
        *,
        signed: bool,
        lo: Optional[np.ndarray] = None,
        hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Quantized encode→decode round-trip of the per-channel log encodings.

        Uses the SAME forward compand, nonzero-anchored scales, and
        reserved-zero quantizer as the real encoders and the same inverse as
        the decoder, so a certificate computed from it can never diverge from
        what actually lands on disk. ``lo``/``hi`` accept precomputed
        COMPANDED-domain column scales — the certificate passes full-array
        scales while round-tripping only a row sample.
        """
        x = np.asarray(data)
        if lo is None or hi is None:
            lo, hi = BaseEncoderMixin._perchannel_log_scales(x, signed=signed)
        y = BaseEncoderMixin._perchannel_log_forward(x, signed=signed)
        nonzero = BaseEncoderMixin._perchannel_nonzero_mask(x, signed=signed)
        u = BaseEncoderMixin._quantize_perchannel_zero_level(y, nonzero, bits, lo, hi)
        denom = max((1 << bits) - 2, 1)
        yq = lo + (u.astype(np.float64) - 1.0) / denom * np.maximum(hi - lo, 1e-30)
        xq = np.sign(yq) * np.expm1(np.abs(yq)) if signed else np.expm1(yq)
        return np.asarray(np.where(u == 0, 0.0, xq))

    @staticmethod
    def _perchannel_geolog_scales(
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-column TRUE-log scales over the POSITIVE entries: (ln min, ln max).

        The per-channel sibling of the scalar geolog anchors (``min_log`` /
        ``max_log``): each column's grid is anchored to its own positive
        min/max in log space, giving uniform relative precision across the
        column's whole dynamic range (log1p companding degenerates to linear
        below 1 — the 2026-07 HDR-color spike measured it failing at wide
        range where true log stays uniform). A column with no positive
        entries gets ``lo == hi == 0`` (harmless: all its codes are the
        reserved zero level).
        """
        x = np.asarray(data)
        n_cols = x.shape[1]
        lo = np.zeros(n_cols, dtype=np.float64)
        hi = np.zeros(n_cols, dtype=np.float64)
        for c in range(n_cols):
            col = x[:, c]
            pos = col[col > 0]
            if pos.size:
                lo[c] = float(np.log(float(pos.min())))
                hi[c] = float(np.log(float(pos.max())))
        return lo, hi

    def _write_float(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        target_dtype: "np.dtype[Any]",
        chunks: Optional[tuple],
        compressor: Optional[Any],
    ) -> None:
        """Write ``data`` cast to ``target_dtype`` with a plain dtype encoding."""
        original_dtype = str(data.dtype)
        create_array(
            zarr_group,
            name,
            data=data.astype(target_dtype, copy=False),
            chunks=chunks,
            compressor=resolve_compressor(compressor, target_dtype),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }
