"""Structural encoders: passthrough, broadcast, array-ref, LUT, index,
unit-vector, and custom encodings (structure/dispatch, not per-channel math)."""

import json
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import zarr

from ..compression import resolve_compressor
from ..modes import EncodingMode
from ..semantic_types import SemanticType
from .base import BaseEncoderMixin


@dataclass(frozen=True)
class _LutPlan:
    """A fully-resolved LUT encoding decision (eligibility + payload).

    Produced by :meth:`ArrayEncoder._lut_plan` from a SINGLE ``np.unique``
    pass, so the eligibility check and the written encoding can never
    disagree (the previous split ``_should_use_lut`` / ``_encode_lut``
    design ran ``np.unique`` twice — once to decide, once to build).
    """

    encoding_name: str  # "lut_uint8" | "lut_uint16"
    lut_list: list  # JSON-ready values (nested rows for row mode)
    indices: np.ndarray  # uint8/uint16; 1-D for row mode, data.shape for scalar
    lut_mode: str  # "row" | "scalar"


class StructuralEncoderMixin(BaseEncoderMixin):
    """Structural encoding strategies for :class:`ArrayEncoder`."""

    def _lut_plan(
        self, data: np.ndarray, semantic_type: SemanticType
    ) -> Optional[_LutPlan]:
        """Decide LUT eligibility and build the payload in ONE unique pass.

        Tiers:
        - **uint8** (K ≤ 256): the legacy rules, preserved verbatim so
          existing stores re-encode byte-identically — row mode for 2D COLOR
          (≤4 channels) requires N ≥ 2K when the input is uint8, everything
          else requires ``data.size ≥ 4K``.
        - **uint16** (257 ≤ K ≤ 65,536): ROW MODE (colors) ONLY, gated by a
          byte-modeled benefit rule. Scalar mode is structurally excluded:
          uint16 indices cost exactly what the quantized scalar alternatives
          cost (≤2 B/element), so the LUT JSON would be pure overhead — a
          measured ~2× store regression. Row mode genuinely wins because ONE
          index covers all channels (2 B/row vs ≥3 B/row for the cheapest
          quantized color). The LUT values live as JSON in ``.zattrs`` and
          are DUPLICATED by consolidated ``.zmetadata`` (parsed at
          scene-open for every node), so element-count heuristics lie here.
          Accept only when the doubled JSON costs at most half the raw byte
          savings over the cheapest quantized color (1 B/channel — the
          conservative floor) AND stays under ``lut_json_max_bytes``.
          Accepted uint16 LUTs are therefore always strictly smaller than
          even the most aggressive lossy alternative — while being EXACT
          (the LUT stores the original values).

        INDEX arrays never LUT-encode (either tier): the viewer's line
        segments loader reads them RAW with no encoding dispatch, so a LUT
        would silently corrupt connectivity — and the smallest-uint INDEX
        encoding is already within one byte of what LUT indices would cost.
        (This also closes a latent pre-existing hazard: small graphs with
        ≤256 unique vertex ids could historically lut_uint8-encode.)

        Returns ``None`` when LUT encoding should not be used.
        """
        if semantic_type == SemanticType.INDEX:
            return None

        is_color_2d = (
            data.ndim == 2
            and semantic_type == SemanticType.COLOR
            and data.shape[1] <= 4
        )

        # Cheap short-circuit BEFORE the unique pass: 1-D uint8 is already
        # optimal (1 B/element; a LUT could not beat it).
        if not is_color_2d and data.dtype == np.uint8 and data.ndim == 1:
            return None

        if is_color_2d:
            unique, inverse = np.unique(data, axis=0, return_inverse=True)
            # numpy 2.0 briefly returned a non-1-D inverse for axis!=None;
            # normalize (harmless on all versions).
            indices_flat = inverse.reshape(-1)
            lut_mode = "row"
            n_indices = data.shape[0]
        else:
            unique, inverse = np.unique(data.ravel(), return_inverse=True)
            indices_flat = inverse.reshape(-1)
            lut_mode = "scalar"
            n_indices = data.size

        k = len(unique)
        if k > 65_536:
            return None

        # JSON fidelity guard (applies to BOTH tiers): 64-bit integers with
        # |v| > 2^53 do not survive the JSON round-trip (viewers parse the
        # LUT through binary64). Compare in the INTEGER domain — casting to
        # float64 first would round ±(2^53 + 1) down to exactly 2^53 and let
        # the one non-round-trippable boundary value slip through.
        # Pre-existing latent bug in the uint8 tier; cheap on ≤65,536 uniques.
        if data.dtype in (np.int64, np.uint64) and bool(
            np.any(unique > 2**53) or np.any(unique < -(2**53))
        ):
            return None

        lut_list = unique.tolist()
        if k <= 256:
            # uint8 tier: legacy eligibility, verbatim.
            if lut_mode == "row" and data.dtype == np.uint8:
                if data.shape[0] < 2 * k:
                    return None
            elif data.size < 4 * k:
                # "At least 4x savings" heuristic (row-mode non-uint8 colors
                # historically fell through to this same check with
                # data.size = N*d, preserved here).
                return None
            idx_dtype: Any = np.uint8
            encoding_name = "lut_uint8"
        else:
            # uint16 tier: ROW MODE ONLY. Scalar-mode u16 indices (2 B/elem)
            # cost exactly what quantized scalar encodings cost, so the LUT
            # JSON would be pure overhead — measured ~2× store regression.
            if lut_mode != "row":
                return None
            # ×2 models .zattrs + consolidated .zmetadata duplication.
            lut_json_bytes = 2 * len(json.dumps(lut_list))
            if lut_json_bytes > self._lut_json_max_bytes:
                return None
            indices_bytes = 2 * n_indices
            # Cheapest realistic color alternative the dtype ladder would
            # produce: 1 B/channel (rgb_uint8 SDR / geolog_perchannel_u8
            # under MEMORY — the conservative floor; HDR AUTO's geolog u16
            # is 2 B/channel, so real savings are usually larger).
            alt_bytes = data.size * 1
            if lut_json_bytes > (alt_bytes - indices_bytes) / 2:
                return None
            idx_dtype = np.uint16
            encoding_name = "lut_uint16"

        indices = (
            indices_flat.astype(idx_dtype)
            if lut_mode == "row"
            else indices_flat.astype(idx_dtype).reshape(data.shape)
        )
        return _LutPlan(
            encoding_name=encoding_name,
            lut_list=lut_list,
            indices=indices,
            lut_mode=lut_mode,
        )

    def _write_passthrough(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        chunks: Optional[tuple],
        compressor: Optional[Any],
    ) -> None:
        """Write array without encoding (passthrough).

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        zarr_group.create_dataset(
            name,
            data=data,
            chunks=chunks,
            compressor=resolve_compressor(compressor, data.dtype),
            overwrite=True,
        )
        # Set encoding metadata to "none"
        zarr_group[name].attrs["encoding"] = {"name": "none"}

    def _encode_broadcasted(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        n_elements: int,
        chunks: Optional[tuple],
        compressor: Optional[Any],
        semantic_type: SemanticType,
    ) -> None:
        """Encode uniform array as broadcasted.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Uniform array data
            n_elements: Number of elements this represents
            chunks: Optional chunk shape (ignored for broadcast, uses minimal)
            compressor: Optional compressor
            semantic_type: Semantic type of the array (only COLOR stamps
                original_dtype so integer colors are restored/normalized)
        """
        # Store single value with shape (1,) or (1, d)
        if data.ndim == 1:
            broadcast_data = data[:1]
        else:
            broadcast_data = data[:1, :]

        zarr_group.create_dataset(
            name,
            data=broadcast_data,
            compressor=resolve_compressor(compressor, broadcast_data.dtype),
            overwrite=True,
        )

        # Set encoding metadata
        encoding: dict[str, Any] = {
            "name": "broadcasted",
            "n_elements": n_elements,
        }
        # Only COLOR stamps original_dtype: it lets the viewer restore native
        # integer color dtypes (uint8/uint16) so they normalize instead of
        # rendering as raw 0-255 floats. Non-color arrays (radii/sharpness/
        # amplitudes) must decode as Float32, so stamping their integer dtype
        # would wrongly widen (e.g. uint16 radii ÷65535 → points vanish).
        if semantic_type == SemanticType.COLOR:
            encoding["original_dtype"] = str(broadcast_data.dtype)
        zarr_group[name].attrs["encoding"] = encoding

    def _scalar_to_array(
        self,
        data: Union[float, int, np.number, tuple, list],
        semantic_type: SemanticType,
    ) -> np.ndarray:
        """Convert scalar input to (1,) or (1, d) array.

        Args:
            data: Scalar value (float, int, tuple, or list)
            semantic_type: Semantic type to determine array format

        Returns:
            NumPy array with shape (1,) or (1, d)

        Raises:
            ValueError: If tuple/list length is invalid for semantic type
        """
        if isinstance(data, (tuple, list)):
            # Color tuple/list: (R, G, B) or (R, G, B, A)
            if semantic_type != SemanticType.COLOR:
                raise ValueError(
                    f"Tuple/list input only supported for COLOR semantic type, "
                    f"got {semantic_type}"
                )
            if len(data) not in (3, 4):
                raise ValueError(
                    f"Color tuple/list must have 3 or 4 elements, got {len(data)}"
                )
            # Convert to (1, d) array
            return np.array([data], dtype=np.float32)
        else:
            # Python or NumPy scalar
            if isinstance(data, (int, float, np.number)):
                # Convert to (1,) array
                return np.array([float(data)], dtype=np.float32)
            raise ValueError(
                f"Unsupported scalar type: {type(data)}. "
                f"Expected float, int, tuple, list, or numpy scalar."
            )

    def _encode_broadcasted_scalar(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        n_elements: int,
        chunks: Optional[tuple],
        compressor: Optional[Any],
        semantic_type: SemanticType,
    ) -> None:
        """Encode scalar as broadcasted array.

        This is called when scalar input is detected. The data is already
        converted to (1,) or (1, d) format by _scalar_to_array().

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data with shape (1,) or (1, d)
            n_elements: Number of elements this scalar represents
            chunks: Optional chunk shape (ignored for broadcast)
            compressor: Optional compressor
            semantic_type: Semantic type of the array (only COLOR stamps
                original_dtype)
        """
        # Write the single value
        zarr_group.create_dataset(
            name,
            data=data,
            compressor=resolve_compressor(compressor, data.dtype),
            overwrite=True,
        )

        # Set encoding metadata. Only COLOR stamps original_dtype (see
        # _encode_broadcasted); the scalar path is always float32 so this is a
        # no-op either way, but the guard keeps the behaviour color-scoped.
        encoding: dict[str, Any] = {
            "name": "broadcasted",
            "n_elements": n_elements,
        }
        if semantic_type == SemanticType.COLOR:
            encoding["original_dtype"] = str(data.dtype)
        zarr_group[name].attrs["encoding"] = encoding

    def _encode_array_ref(
        self,
        zarr_group: zarr.Group,
        name: str,
        target_path: str,
        hash_str: str,
        shape: tuple,
        dtype: np.dtype,
        chunks: Optional[tuple],
        compressor: Optional[Any],
    ) -> None:
        """Encode array as reference to existing array.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            target_path: Path to original array
            hash_str: Full content hash
            shape: Original array shape
            dtype: Original array dtype
            chunks: Optional chunk shape (minimal for empty array)
            compressor: Optional compressor
        """
        # Create empty array with preserved dimensionality
        if len(shape) == 1:
            empty_shape = (0,)
        else:
            empty_shape = (0, *shape[1:])

        empty_data = np.array([], dtype=dtype).reshape(empty_shape)
        zarr_group.create_dataset(
            name,
            data=empty_data,
            compressor=resolve_compressor(compressor, empty_data.dtype),
            overwrite=True,
        )

        # Set encoding metadata
        zarr_group[name].attrs["encoding"] = {
            "name": "array_ref",
            "target": target_path,
            "hash": hash_str,
            "original_shape": list(shape),
            "original_dtype": str(dtype),
        }

    def _encode_lut(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        plan: _LutPlan,
        chunks: Optional[tuple],
        compressor: Optional[Any],
    ) -> None:
        """Write a LUT encoding decided by :meth:`_lut_plan`.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Original array (for original_dtype/original_shape metadata)
            plan: The resolved LUT decision (indices + values + mode)
            chunks: Optional chunk shape (of the ORIGINAL array)
            compressor: Optional compressor
        """
        indices_chunks: Optional[tuple]
        if plan.lut_mode == "row":
            # Indices are 1-D; keep only the first chunk dimension.
            if chunks is not None and len(chunks) > 1:
                indices_chunks = (chunks[0],)
            else:
                indices_chunks = chunks
        else:
            # Scalar mode: indices keep the original shape.
            indices_chunks = chunks

        zarr_group.create_dataset(
            name,
            data=plan.indices,
            chunks=indices_chunks,
            compressor=resolve_compressor(compressor, plan.indices.dtype),
            overwrite=True,
        )

        metadata = {
            "name": plan.encoding_name,
            "lut": plan.lut_list,
            "original_dtype": str(data.dtype),
        }
        # Add lut_mode and original_shape for 2D arrays (1-D arrays omit
        # lut_mode by contract; decoders default it).
        if data.ndim > 1:
            metadata["lut_mode"] = plan.lut_mode
            metadata["original_shape"] = list(data.shape)

        zarr_group[name].attrs["encoding"] = metadata

    def _encode_index(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode INDEX semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)

        # Select smallest uint dtype that fits max value
        target_dtype: np.dtype[Any]
        max_val = int(np.max(data))

        if max_val <= 255:
            target_dtype = np.dtype("uint8")
        elif max_val <= 65535:
            target_dtype = np.dtype("uint16")
        elif max_val <= 4294967295:
            target_dtype = np.dtype("uint32")
        else:
            target_dtype = np.dtype("uint64")

        encoded_data = data.astype(target_dtype)
        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=resolve_compressor(compressor, encoded_data.dtype),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }

    #: CUSTOM-mode dispatch: encoding name -> the handler method that produces
    #: ``(encoded_data, metadata)``. Replaces a string-keyed if/elif ladder;
    #: every key must be a valid format-contract encoding name (see
    #: ``test_custom_dispatch_contract``).
    _CUSTOM_DISPATCH: dict[str, str] = {
        "float32": "_custom_passthrough",
        "float16": "_custom_passthrough",
        "uint8": "_custom_passthrough",
        "uint16": "_custom_passthrough",
        "uint32": "_custom_passthrough",
        "uint64": "_custom_passthrough",
        "bounded_scalar_uint8": "_custom_bounded_scalar",
        "bounded_scalar_uint16": "_custom_bounded_scalar",
        "log_scalar_uint8": "_custom_log_scalar",
        "log_scalar_uint16": "_custom_log_scalar",
        "rgb_uint8": "_custom_rgb",
        "rgb_uint16": "_custom_rgb",
    }

    def _encode_custom(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        encoder_name: str,
        bounds: Optional[tuple[float, float]],
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode using an explicitly specified encoder (CUSTOM mode).

        Dispatches ``encoder_name`` through :data:`_CUSTOM_DISPATCH` to a
        handler that returns ``(encoded_data, metadata)``; the shared tail
        writes the array and its ``encoding`` attrs.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            encoder_name: Explicit encoder name
            bounds: Bounds for bounded_scalar encoders
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        handler_name = self._CUSTOM_DISPATCH.get(encoder_name)
        if handler_name is None:
            raise ValueError(f"Unknown custom encoder: {encoder_name}")
        encoded_data, metadata = getattr(self, handler_name)(
            data, encoder_name, bounds, str(data.dtype)
        )
        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=resolve_compressor(compressor, encoded_data.dtype),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = metadata

    def _custom_passthrough(
        self,
        data: np.ndarray,
        encoder_name: str,
        bounds: Optional[tuple[float, float]],
        original_dtype: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """float32/float16 + direct uint8/16/32/64: dtype cast, no quantization."""
        encoded_data = data.astype(np.dtype(encoder_name))
        return encoded_data, {"name": encoder_name, "original_dtype": original_dtype}

    def _custom_bounded_scalar(
        self,
        data: np.ndarray,
        encoder_name: str,
        bounds: Optional[tuple[float, float]],
        original_dtype: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """bounded_scalar_uint{8,16}: linear quantize over explicit bounds."""
        if bounds is None:
            raise ValueError(f"{encoder_name} requires bounds parameter")
        min_val, max_val = bounds
        bits = 8 if "uint8" in encoder_name else 16
        max_int = (2**bits) - 1
        span = max_val - min_val
        if span == 0:
            encoded_data = np.zeros_like(
                data, dtype=np.uint8 if bits == 8 else np.uint16
            )
        else:
            # round_values=False preserves the custom encoder's historical
            # truncating quantization (the semantic-type encoders round).
            encoded_data = self._quantize_normalized_clip(
                data,
                min_val,
                span,
                max_int,
                np.dtype(np.uint8 if bits == 8 else np.uint16),
                round_values=False,
            )
        return encoded_data, {
            "name": encoder_name,
            "min": min_val,
            "max": max_val,
            "bits": bits,
            "original_dtype": original_dtype,
        }

    def _custom_log_scalar(
        self,
        data: np.ndarray,
        encoder_name: str,
        bounds: Optional[tuple[float, float]],
        original_dtype: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """log_scalar_uint{8,16}: log1p compand then linear quantize."""
        bits = 8 if "uint8" in encoder_name else 16
        max_int = (2**bits) - 1
        max_val = float(np.max(data))
        max_log = float(np.log1p(max_val))
        log_vals = np.log1p(data)
        normalized = log_vals / max_log
        encoded_data = np.clip(normalized * max_int, 0, max_int).astype(
            np.uint8 if bits == 8 else np.uint16
        )
        return encoded_data, {
            "name": encoder_name,
            "max_log": max_log,
            "bits": bits,
            "original_dtype": original_dtype,
        }

    def _custom_rgb(
        self,
        data: np.ndarray,
        encoder_name: str,
        bounds: Optional[tuple[float, float]],
        original_dtype: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """rgb_uint{8,16}: SDR color quantize (truncating)."""
        max_int = 255 if "uint8" in encoder_name else 65535
        encoded_data = np.clip(data * max_int, 0, max_int).astype(
            np.uint8 if "uint8" in encoder_name else np.uint16
        )
        return encoded_data, {"name": encoder_name, "original_dtype": original_dtype}
