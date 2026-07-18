"""Per-channel and scalar dtype encoders: coordinate, color, bounded/positive
scalar, geolog scalar, and the per-channel linear/log/signed-log/geolog family."""

import warnings
from typing import Any, Optional

import numpy as np
import zarr

from ..compression import resolve_compressor
from ..modes import EncodingMode
from .base import BaseEncoderMixin
from .delta_codec import probe_delta_filter


class PerChannelEncoderMixin(BaseEncoderMixin):
    """Per-channel / scalar dtype encoding strategies for :class:`ArrayEncoder`."""

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
            if max_extent >= 65536.0:
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

        # Coordinates always u16 (never u8) for both AUTO and MEMORY. The decode
        # contract for COORDINATE is float32 (GPU/viewer target) regardless of the
        # input dtype — matching PRECISION's float32 cast — so original_dtype is
        # pinned to float32 (an integer original_dtype would truncate on decode).
        self._encode_linear_perchannel(
            zarr_group,
            name,
            arr,
            16,
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
                encoded_data = data.astype(np.float32)
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
                encoded_data = data.astype(np.float32)
                encoder_name = "float32"
            elif mode == EncodingMode.MEMORY or mode == EncodingMode.AUTO:
                # Quantize to uint8: [0, 1] → [0, 255]
                encoded_data = np.clip(data * 255.0, 0, 255).astype(np.uint8)
                encoder_name = "rgb_uint8"
            else:
                raise ValueError(f"Unexpected mode for SDR COLOR: {mode}")
        else:
            raise ValueError("color_mode required for float COLOR arrays")

        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=resolve_compressor(compressor, encoded_data.dtype),
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
            encoded_data = data.astype(np.float32)
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
                # Choose quantization based on dynamic range
                bits = self._compute_quantization_bits(data)

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
                    # Dynamic range > 65536, use float
                    if self._float16_allowed:
                        encoded_data = data.astype(np.float16)
                        encoder_name = "float16"
                    else:
                        encoded_data = data.astype(np.float32)
                        encoder_name = "float32"
                    metadata = {"name": encoder_name, "original_dtype": original_dtype}
        else:
            raise ValueError(f"Unexpected mode for BOUNDED_SCALAR: {mode}")

        comp = resolve_compressor(compressor, encoded_data.dtype)
        zarr_group.create_dataset(
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
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)
        metadata: dict[str, Any]

        if mode == EncodingMode.PRECISION:
            # No quantization
            encoded_data = data.astype(np.float32)
            encoder_name = "float32"
            metadata = {"name": encoder_name, "original_dtype": original_dtype}
        elif mode == EncodingMode.MEMORY or mode == EncodingMode.AUTO:
            # Analyze range
            max_val = float(np.max(data))

            if encoding_type == "log" and max_val > 0:
                # Geometric-log encoding (min/max-anchored, reserved zero
                # level): uniform RELATIVE precision across the whole range,
                # and by construction no nonzero value can decode to zero.
                self._encode_geolog_scalar(
                    zarr_group,
                    name,
                    data,
                    16 if mode == EncodingMode.AUTO else 8,
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
                bits = self._compute_quantization_bits(data)

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
                elif bits in (8, 16):
                    # Rescale-first: anchor the grid at the array's OWN
                    # [min, max] rather than [0, max] — data far from zero
                    # (e.g. radii in [10, 11]) no longer wastes code space
                    # on the empty [0, min) span. min == 0 whenever the data
                    # contains zeros, so zero handling is unchanged.
                    min_val = float(np.min(data))
                    span = max(max_val - min_val, 0.0)
                    levels = 255 if bits == 8 else 65535
                    if span == 0.0:
                        encoded_data = np.zeros_like(
                            data, dtype=np.uint8 if bits == 8 else np.uint16
                        )
                    else:
                        encoded_data = self._quantize_normalized_clip(
                            data,
                            min_val,
                            span,
                            levels,
                            np.dtype(np.uint8 if bits == 8 else np.uint16),
                        )
                    encoder_name = f"bounded_scalar_uint{bits}"
                    metadata = {
                        "name": encoder_name,
                        "min": min_val,
                        "max": max_val,
                        "bits": bits,
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
                        16 if mode == EncodingMode.AUTO else 8,
                        chunks,
                        compressor,
                    )
                    return
        else:
            raise ValueError(f"Unexpected mode for POSITIVE_SCALAR: {mode}")

        comp = resolve_compressor(compressor, encoded_data.dtype)
        zarr_group.create_dataset(
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
        """Write a wide-dynamic-range positive scalar as ``geolog_scalar_u{bits}``.

        Rescale-first log quantization: the grid is anchored to the array's OWN
        nonzero ``[min, max]`` (stored as ``min_log``/``max_log``), the same
        array-local principle as the COORDINATE fixed-point grid and the
        per-column Cholesky scales. Level 0 is reserved for exact zeros.
        """
        original_dtype = str(data.dtype)
        x = np.asarray(data, dtype=np.float64)
        nz = x > 0
        min_log = float(np.log(x[nz].min()))
        max_log = float(np.log(x[nz].max()))
        codes = self._geolog_forward(x, bits, min_log, max_log)
        # Invariant, not a data property: the reserved zero level makes
        # nonzero -> 0 impossible; a violation would be an encoder bug.
        if bool(((codes == 0) & nz).any()):
            raise RuntimeError(
                f"geolog encoding bug: nonzero values of '{name}' mapped to "
                "the reserved zero level"
            )
        comp = resolve_compressor(compressor, codes.dtype)
        zarr_group.create_dataset(
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
        zarr_group.create_dataset(
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
        zarr_group.create_dataset(
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
        zarr_group.create_dataset(
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
        zarr_group.create_dataset(
            name,
            data=u,
            chunks=chunks,
            compressor=resolve_compressor(compressor, u.dtype),
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
