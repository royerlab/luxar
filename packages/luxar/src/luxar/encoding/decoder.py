"""Array decoder for encoded zarr arrays.

The decoder reads encoding metadata and applies appropriate decoding
transformations to recover original data.
"""

from typing import Any, Optional, Sequence

import numpy as np
import zarr


class ArrayDecoder:
    """Decode any encoded array from zarr.

    The decoder handles all encoding types by reading metadata and applying
    the appropriate inverse transformation. It supports recursive decoding
    for array references.
    """

    DIRECT_ENCODINGS = {
        "none",
        "float16",
        "float32",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    QUANTIZED_ENCODINGS = {
        "bounded_scalar_uint8",
        "bounded_scalar_uint16",
        "log_scalar_uint8",
        "log_scalar_uint16",
        "geolog_scalar_uint8",
        "geolog_scalar_uint16",
        "rgb_uint8",
        "rgb_uint16",
        # Generic per-channel quantization (per-column min/max). log / signed-log
        # first used for the split Cholesky diagonal / off-diagonal; linear
        # (identity) used for COORDINATE positions/centers/vertices.
        "log_perchannel_u8",
        "log_perchannel_u16",
        "signed_log_perchannel_u8",
        "signed_log_perchannel_u16",
        "linear_perchannel_u8",
        "linear_perchannel_u16",
        # per-channel TRUE-log (min/max-anchored geometric grid, reserved
        # zero level always); first used for HDR colors.
        "geolog_perchannel_u8",
        "geolog_perchannel_u16",
    }
    SPECIAL_ENCODINGS = {"broadcasted", "array_ref", "lut_uint8", "lut_uint16"}
    KNOWN_ENCODINGS = DIRECT_ENCODINGS | QUANTIZED_ENCODINGS | SPECIAL_ENCODINGS

    def decode(
        self,
        zarr_array: zarr.Array,
        zarr_root: Optional[zarr.Group] = None,
    ) -> np.ndarray:
        """Decode array based on encoding metadata.

        Returns numpy array with dtype matching the original input:
        - Quantized encodings use 'original_dtype' from metadata
        - LUT encoding uses 'original_dtype' from metadata
        - Passthrough returns stored dtype
        - Broadcasted returns stored dtype (expanded to full size)

        Args:
            zarr_array: Zarr array to decode
            zarr_root: Zarr root group (required for array_ref resolution)

        Returns:
            Decoded numpy array

        Raises:
            ValueError: If metadata is missing or invalid
            ValueError: If array_ref target not found
        """
        enc = self._encoding_metadata(zarr_array)
        name = enc["name"]

        # Special encodings (must handle first)
        if name == "broadcasted":
            return self._expand_broadcasted(zarr_array, enc)
        elif name == "array_ref":
            return self._follow_ref(zarr_array, enc, zarr_root)
        elif name in {"lut_uint8", "lut_uint16"}:
            return self._decode_lut(zarr_array, enc)

        # Quantized encodings (require inverse transformation)
        elif name == "bounded_scalar_uint8":
            return self._decode_bounded_scalar(zarr_array, enc)
        elif name == "bounded_scalar_uint16":
            return self._decode_bounded_scalar(zarr_array, enc)
        elif name == "log_scalar_uint8":
            return self._decode_log_scalar(zarr_array, enc)
        elif name == "log_scalar_uint16":
            return self._decode_log_scalar(zarr_array, enc)
        elif name in {"geolog_scalar_uint8", "geolog_scalar_uint16"}:
            return self._decode_geolog_scalar(zarr_array, enc)
        elif name == "rgb_uint8":
            return self._decode_color(zarr_array, enc)
        elif name == "rgb_uint16":
            return self._decode_color(zarr_array, enc)
        elif name in {"log_perchannel_u8", "log_perchannel_u16"}:
            return self._decode_log_perchannel(zarr_array, enc)
        elif name in {"signed_log_perchannel_u8", "signed_log_perchannel_u16"}:
            return self._decode_signed_log_perchannel(zarr_array, enc)
        elif name in {"linear_perchannel_u8", "linear_perchannel_u16"}:
            return self._decode_linear_perchannel(zarr_array, enc)
        elif name in {"geolog_perchannel_u8", "geolog_perchannel_u16"}:
            return self._decode_geolog_perchannel(zarr_array, enc)

        # Direct dtype encodings mean the stored values are already decoded.
        elif name in self.DIRECT_ENCODINGS:
            return np.asarray(zarr_array[:])

        # ``_encoding_metadata`` validates this path, so this is defensive only.
        else:  # pragma: no cover
            raise ValueError(f"Unknown encoding name: {name}")

    def _encoding_metadata(self, zarr_array: zarr.Array) -> dict[str, Any]:
        """Return validated encoding metadata.

        Missing ``encoding`` metadata means direct storage. If the ``encoding``
        object exists, it must be explicit and name a known encoder.
        """
        raw = zarr_array.attrs.get("encoding", None)
        if raw is None:
            return {"name": "none"}
        if not isinstance(raw, dict):
            raise ValueError("encoding metadata must be an object")

        enc = dict(raw)
        name = enc.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "encoding.name is required when encoding metadata is present"
            )
        if name not in self.KNOWN_ENCODINGS:
            raise ValueError(f"Unknown encoding name: {name}")
        if name == "array_ref":
            self._require_fields(enc, "array_ref", ("target",))
        elif "target" in enc:
            raise ValueError("encoding.target is only valid for array_ref")

        if name == "broadcasted":
            self._require_fields(enc, name, ("n_elements",))
        elif name in {"lut_uint8", "lut_uint16"}:
            self._require_fields(enc, name, ("lut", "original_dtype"))
        elif name in {"bounded_scalar_uint8", "bounded_scalar_uint16"}:
            self._require_fields(enc, name, ("min", "max", "bits", "original_dtype"))
        elif name in {"log_scalar_uint8", "log_scalar_uint16"}:
            self._require_fields(enc, name, ("max_log", "bits", "original_dtype"))
        elif name in {"geolog_scalar_uint8", "geolog_scalar_uint16"}:
            self._require_fields(
                enc, name, ("min_log", "max_log", "bits", "original_dtype")
            )
        elif name in {"rgb_uint8", "rgb_uint16"}:
            self._require_fields(enc, name, ("original_dtype",))
        elif name in {
            "log_perchannel_u8",
            "log_perchannel_u16",
            "signed_log_perchannel_u8",
            "signed_log_perchannel_u16",
            "linear_perchannel_u8",
            "linear_perchannel_u16",
            "geolog_perchannel_u8",
            "geolog_perchannel_u16",
        }:
            self._require_fields(
                enc, name, ("col_lo", "col_hi", "bits", "original_dtype")
            )

        return enc

    def _require_fields(
        self, enc: dict[str, Any], encoding_name: str, fields: tuple[str, ...]
    ) -> None:
        """Validate required metadata fields for an encoder."""
        missing = [field for field in fields if field not in enc]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"{encoding_name} encoding requires metadata field(s): {joined}"
            )

    def _decode_bounded_scalar(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode bounded scalar: uint → original dtype using min/max.

        Quantization formula (mirrored in the encoder):

            normalized = uint_value / (2 ** bits - 1)
            original   = normalized * (max - min) + min

        The denominator is ``2**bits - 1`` (not ``2**bits``) so the
        maximum representable code (e.g. 255 for bits=8) maps exactly to
        ``max_val`` and the minimum (0) maps exactly to ``min_val``. The
        quantization step is ``(max - min) / (2**bits - 1)`` — for the
        sharpness encoding (normalized [0, 1] knob) this is ``1 / 255 ≈ 0.0039``.

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype

        Raises:
            ValueError: If metadata is malformed (non-finite bounds,
                ``max <= min``, or ``bits <= 0``).
        """
        data = np.asarray(arr[:])
        min_val = float(enc["min"])
        max_val = float(enc["max"])
        bits = int(enc["bits"])
        if not (np.isfinite(min_val) and np.isfinite(max_val)):
            raise ValueError(
                f"bounded_scalar requires finite min/max, got min={min_val} max={max_val}"
            )
        if max_val <= min_val:
            raise ValueError(
                f"bounded_scalar requires max > min, got min={min_val} max={max_val}"
            )
        if bits <= 0:
            raise ValueError(f"bounded_scalar requires bits > 0, got {bits}")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))

        # Use float64 intermediate for precision, then cast to original dtype
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = normalized * (max_val - min_val) + min_val
        return np.asarray(result, dtype=original_dtype)

    def _decode_log_scalar(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode log scalar: uint → original dtype using expm1.

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype

        Raises:
            ValueError: If metadata is malformed (non-finite or
                non-positive ``max_log``, or ``bits <= 0``).
        """
        data = np.asarray(arr[:])
        max_log = float(enc["max_log"])
        bits = int(enc["bits"])
        if not np.isfinite(max_log) or max_log <= 0.0:
            raise ValueError(
                f"log_scalar requires finite, positive max_log, got {max_log}"
            )
        if bits <= 0:
            raise ValueError(f"log_scalar requires bits > 0, got {bits}")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))

        # The viewer's shared decode_log_scalar_* f32 kernel is the display contract;
        # small differences from this float64 metadata/reference helper are expected.
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = np.expm1(normalized * max_log)
        return np.asarray(result, dtype=original_dtype)

    def _decode_geolog_scalar(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode geometric-log scalar (min/max-anchored, reserved zero level).

        Level 0 decodes to exactly 0; levels ``[1, 2**bits - 1]`` decode to
        ``exp(min_log + (u - 1)/(2**bits - 2) * (max_log - min_log))`` —
        uniform relative precision across the array's own nonzero range.

        Raises:
            ValueError: If metadata is malformed (non-finite ``min_log``/
                ``max_log``, ``max_log < min_log``, or ``bits <= 0``).
        """
        data = np.asarray(arr[:])
        min_log = float(enc["min_log"])
        max_log = float(enc["max_log"])
        bits = int(enc["bits"])
        if not (np.isfinite(min_log) and np.isfinite(max_log)):
            raise ValueError(
                f"geolog_scalar requires finite min_log/max_log, got "
                f"{min_log}/{max_log}"
            )
        if max_log < min_log:
            raise ValueError(
                f"geolog_scalar requires max_log >= min_log, got {min_log}/{max_log}"
            )
        if bits <= 0:
            raise ValueError(f"geolog_scalar requires bits > 0, got {bits}")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))

        top = (1 << bits) - 1
        out = np.zeros(data.shape, dtype=np.float64)
        nz = data > 0
        if nz.any():
            denom = max(top - 1, 1)
            out[nz] = np.exp(
                min_log
                + (data[nz].astype(np.float64) - 1.0) / denom * (max_log - min_log)
            )
        return np.asarray(out, dtype=original_dtype)

    @staticmethod
    def _perchannel_scales(data: np.ndarray, enc: dict, name: str) -> tuple:
        """Validate + return ``(lo, hi, bits)`` for a per-channel dequant.

        Mirrors the finiteness/length rigor of the scalar decoders
        (``_decode_bounded_scalar`` / ``_decode_log_scalar``) and the viewer's
        ``ArrayDecoder.makePerChannelDequant`` (which throws rather than
        silently zero-filling): a malformed file with non-finite or
        wrong-length per-column scales must raise, not corrupt silently.

        Raises:
            ValueError: If ``bits <= 0``, ``col_lo``/``col_hi`` are non-finite,
                differ in length, mismatch the array's column count, or any
                ``col_hi < col_lo`` (a constant column with ``hi == lo`` is
                valid — it decodes every code to ``lo``).
        """
        lo = np.asarray(enc["col_lo"], dtype=np.float64)
        hi = np.asarray(enc["col_hi"], dtype=np.float64)
        bits = int(enc["bits"])
        if bits <= 0:
            raise ValueError(f"{name} requires bits > 0, got {bits}")
        if lo.shape != hi.shape or lo.ndim != 1:
            raise ValueError(
                f"{name} requires equal-length 1-D col_lo/col_hi, "
                f"got shapes {lo.shape} and {hi.shape}"
            )
        # A 1-D array is ONE column of N values, not one row of N channels —
        # that is how the encoder reduces it (``_quantize_per_column`` /
        # ``_encode_coordinate`` both take ``min/max`` over axis 0), so reading
        # its channel count off ``shape[-1]`` rejected every 1-D array the
        # writer can produce.
        cols = data.shape[-1] if data.ndim >= 2 else 1
        if lo.shape[0] != cols:
            raise ValueError(
                f"{name} expects {cols} per-column scales, got {lo.shape[0]}"
            )
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError(f"{name} requires finite col_lo/col_hi")
        if np.any(hi < lo):
            raise ValueError(f"{name} requires col_hi >= col_lo for every column")
        return lo, hi, bits

    @staticmethod
    def _perchannel_companded(
        data: np.ndarray,
        enc: dict,
        lo: np.ndarray,
        hi: np.ndarray,
        bits: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Map stored codes back to companded values, honoring ``zero_level``.

        Current arrays (``zero_level: true``) reserve code 0 for exact zeros
        and span nonzero codes ``1..2**bits-1`` over the nonzero-anchored
        ``[lo, hi]``; legacy arrays span all codes ``0..2**bits-1``. Returns
        ``(y, zero_mask)`` where ``zero_mask`` is ``None`` for legacy arrays.
        """
        rng = np.maximum(hi - lo, 1e-30)
        if enc.get("zero_level"):
            denom = max((1 << bits) - 2, 1)
            return lo + (data - 1.0) / denom * rng, data == 0
        return lo + data / ((1 << bits) - 1) * rng, None

    def _decode_log_perchannel(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode generic per-channel log quantization of an (N, C) array.

        Inverse of ``_encode_log_perchannel``. With ``zero_level: true``
        (current writer): ``x = 0`` for code 0, else
        ``expm1(col_lo[c] + (u-1)/(levels-1)·(col_hi[c]-col_lo[c]))``. Legacy
        arrays (no flag): ``x = expm1(col_lo[c] + u/levels·(col_hi[c]-col_lo[c]))``.
        """
        data = np.asarray(arr[:]).astype(np.float64)
        lo, hi, bits = self._perchannel_scales(data, enc, "log_perchannel")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))
        y, zero_mask = self._perchannel_companded(data, enc, lo, hi, bits)
        x = np.expm1(y)
        if zero_mask is not None:
            x = np.where(zero_mask, 0.0, x)
        return np.asarray(x, dtype=original_dtype)

    def _decode_signed_log_perchannel(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode generic per-channel signed-log quantization of an (N, C) array.

        Inverse of ``_encode_signed_log_perchannel``. With ``zero_level: true``
        (current writer): ``x = 0`` for code 0, else
        ``y = col_lo[c] + (u-1)/(levels-1)·rng; x = sign(y)·expm1(|y|)``. Legacy
        arrays (no flag): ``y = col_lo[c] + u/levels·rng; x = sign(y)·expm1(|y|)``.
        """
        data = np.asarray(arr[:]).astype(np.float64)
        lo, hi, bits = self._perchannel_scales(data, enc, "signed_log_perchannel")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))
        y, zero_mask = self._perchannel_companded(data, enc, lo, hi, bits)
        x = np.sign(y) * np.expm1(np.abs(y))
        if zero_mask is not None:
            x = np.where(zero_mask, 0.0, x)
        return np.asarray(x, dtype=original_dtype)

    def _decode_linear_perchannel(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode generic per-channel LINEAR (fixed-point) quantization of an
        (N, C) array — identity transform, per-column min/max.

        Inverse of the ``linear_perchannel_*`` encoding (COORDINATE positions):
        ``x = col_lo[c] + u/levels·(col_hi[c]-col_lo[c])`` per column. No log —
        handles negative values, so it is the correct transform for coordinates.
        """
        data = np.asarray(arr[:]).astype(np.float64)
        lo, hi, bits = self._perchannel_scales(data, enc, "linear_perchannel")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))
        rng = np.maximum(hi - lo, 1e-30)
        return np.asarray(lo + data / ((1 << bits) - 1) * rng, dtype=original_dtype)

    def _decode_geolog_perchannel(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode per-channel TRUE-log quantization of a positive (N, C) array.

        Inverse of ``_encode_geolog_perchannel``. The reserved zero level is
        part of this encoding's NAME contract (there is no legacy all-levels
        variant): code 0 decodes to exactly 0; codes ``1..2**bits - 1`` decode
        to ``exp(col_lo[c] + (u-1)/(2**bits - 2)·(col_hi[c]-col_lo[c]))`` —
        ``col_lo``/``col_hi`` are the column's positive min/max in LOG space,
        so relative precision is uniform across the column's dynamic range.
        """
        data = np.asarray(arr[:]).astype(np.float64)
        lo, hi, bits = self._perchannel_scales(data, enc, "geolog_perchannel")
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))
        rng = np.maximum(hi - lo, 1e-30)
        denom = max((1 << bits) - 2, 1)
        y = lo + (data - 1.0) / denom * rng
        return np.asarray(np.where(data == 0, 0.0, np.exp(y)), dtype=original_dtype)

    def _decode_color(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode color: uint8/uint16 [0,max] → original dtype [0,1].

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype in [0, 1] range
        """
        data = np.asarray(arr[:])
        original_dtype = np.dtype(enc.get("original_dtype", "float32"))

        # Determine max value based on stored dtype
        max_val = 255.0 if data.dtype == np.uint8 else 65535.0

        # Convert to [0, 1] range
        result = data.astype(np.float64) / max_val
        return result.astype(original_dtype)

    def _decode_lut(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode LUT-encoded array.

        Args:
            arr: Zarr array with indices
            enc: Encoding metadata with LUT

        Returns:
            Decoded array by looking up values in LUT
        """
        indices = np.asarray(arr[:], dtype=np.int64)
        lut = np.array(enc["lut"], dtype=enc["original_dtype"])
        lut_mode = enc.get("lut_mode", "scalar")  # Default for 1D

        if lut_mode == "row":
            # Row mode: indices are (N,), lut is (K, d)
            # Result is (N, d)
            return np.asarray(lut[indices])
        else:
            # Scalar mode: indices match original shape
            # lut is (K,), indices may be (N,) or (N, d)
            return np.asarray(lut[indices])

    def _expand_broadcasted(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Expand broadcasted array to full size.

        Args:
            arr: Zarr array with single value
            enc: Encoding metadata with n_elements

        Returns:
            Expanded array with repeated value
        """
        data = np.asarray(arr[:])  # Shape (1,) or (1, d)
        n_elements = enc["n_elements"]

        # Repeat the single value n_elements times along axis 0
        return np.repeat(data, n_elements, axis=0)

    def _follow_ref(
        self,
        arr: zarr.Array,
        enc: dict,
        zarr_root: Optional[zarr.Group],
    ) -> np.ndarray:
        """Follow array reference and decode target (recursive).

        Args:
            arr: Zarr array with reference (empty)
            enc: Encoding metadata with target path
            zarr_root: Zarr root group for path resolution

        Returns:
            Decoded array from target

        Raises:
            ValueError: If zarr_root is None
            ValueError: If target path not found
        """
        if zarr_root is None:
            raise ValueError("zarr_root required for array_ref decoding")

        target_path = enc["target"]

        # Resolve relative path
        # target_path is like "../points_1/colors" or "other_group/data"
        try:
            target_array = zarr_root[target_path]
        except KeyError:
            raise ValueError(f"Array reference target not found: {target_path}")

        # IMPORTANT: Recursively decode the target array
        # This handles cases where target is itself encoded (e.g., LUT)
        return self.decode(target_array, zarr_root)


def decode_coordinate_columns(
    zarr_array: zarr.Array, columns: Sequence[int]
) -> np.ndarray:
    """Decode selected coordinate columns without reading the whole array."""
    selected = np.asarray(zarr_array[:, list(columns)])
    encoding = dict(zarr_array.attrs.get("encoding", {}))
    name = encoding.get("name", "none")
    if name in ArrayDecoder.DIRECT_ENCODINGS:
        return selected
    if name not in {"linear_perchannel_u8", "linear_perchannel_u16"}:
        raise ValueError(f"unsupported coordinate encoding {name!r}")

    low = np.asarray(encoding["col_lo"], dtype=np.float64)
    high = np.asarray(encoding["col_hi"], dtype=np.float64)
    bits = int(encoding["bits"])
    column_count = zarr_array.shape[-1] if zarr_array.ndim >= 2 else 1
    if (
        bits <= 0
        or low.shape != high.shape
        or low.ndim != 1
        or low.shape[0] != column_count
    ):
        raise ValueError(f"malformed {name} coordinate encoding")
    if not (
        np.all(np.isfinite(low))
        and np.all(np.isfinite(high))
        and np.all(high >= low)
    ):
        raise ValueError(f"malformed {name} coordinate encoding")
    selected_low = low[list(columns)]
    selected_high = high[list(columns)]
    decoded = selected_low + selected.astype(np.float64) / ((1 << bits) - 1) * (
        selected_high - selected_low
    )
    return np.asarray(decoded, dtype=np.dtype(encoding.get("original_dtype", "float32")))
