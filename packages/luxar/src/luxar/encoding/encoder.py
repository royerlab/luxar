"""Array encoder with priority-based encoding selection.

The encoder follows a strict priority order:
1. Broadcasting (if all values identical)
2. Array Reference (if duplicate exists)
3. LUT Encoding (if ≤256 unique values and mode != PRECISION)
4. Dtype Encoding (based on semantic type and mode)
"""

import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
import zarr

from ..validation.base import _validate_numeric_finite_values
from .modes import EncodingMode
from .registry import ArrayRefRegistry
from .semantic_types import SemanticType

# Escalation threshold for the AUTO covariance certificate: p95 over splats of
# the relative Frobenius error of Σ = L·Lᵀ after a quantized round-trip. AUTO
# tries u8 first and only escalates (u16, then float32) when the measured error
# exceeds this bound — so it is a hard invariant of AUTO output, not a heuristic.
# Calibration (2026-07 covariance spike, real light-sheet fit): u8 measured
# relF p95 ≈ 0.02 while rendering ~46 dB below the fit-error floor (invisible);
# 0.05 keeps a wide safety margin yet is reachable by genuinely hard data
# (e.g. merged stores whose σ columns span many decades).
COV_CERT_RELF_P95_MAX = 0.05

#: Row cap for the certificate measurement. The p95 statistic needs a bounded
#: sample, not every splat — without a cap, a 10M-splat flat fit would build
#: ~2.7 GB of transient float64 Σ scratch just to certify. Evenly-spaced rows
#: keep the sample deterministic and (over Hilbert-ordered splats) spatially
#: uniform; the quantization scales always come from the full columns.
COV_CERT_SAMPLE_MAX = 262_144


class ArrayEncoder:
    """Unified encoder with internal registry for deduplication.

    The encoder writes directly to zarr groups and maintains an internal
    registry for detecting duplicate arrays. It follows the priority order
    specified in the encoding specification.
    """

    def __init__(
        self,
        broadcast_rtol: float = 0.0,
        broadcast_atol: float = 0.0,
        float16_allowed: bool = False,
    ) -> None:
        """Initialize encoder with optional broadcasting tolerance and float16 control.

        Args:
            broadcast_rtol: Relative tolerance for broadcasting check
                (default: 0.0 = exact equality)
            broadcast_atol: Absolute tolerance for broadcasting check
                (default: 0.0 = exact equality)
            float16_allowed: Allow float16 encoding in MEMORY mode (default: False for TypeScript compatibility)

        Note: Using non-zero tolerance is experimental and should be used
        with caution. Exact equality (default) is safe for all semantic types
        including indices and colors.
        """
        self._registry = ArrayRefRegistry()
        self._broadcast_rtol = broadcast_rtol
        self._broadcast_atol = broadcast_atol
        self._float16_allowed = float16_allowed
        # Internal tier override consumed by the per-channel log encoders.
        # Set (via try/finally) only by encode_cholesky_split, which owns the
        # certified u8→u16 escalation for the diag/offdiag PAIR. Deliberately
        # not a public encode() parameter: callers never choose bit widths.
        self._perchannel_bits_override: Optional[int] = None

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
    ) -> None:
        """Encode array or scalar and write to zarr group.

        Follows priority order:
        1. Broadcasting (if scalar input OR all values identical within tolerance)
        2. Array reference (if duplicate exists)
        3. LUT encoding (if ≤256 unique values and mode != PRECISION)
        4. Dtype encoding (based on semantic type and mode)

        Args:
            data: Input data - can be:
                  - NumPy array: standard path
                  - Python scalar (float, int): requires n_elements
                  - Tuple/list (for colors): e.g., (1.0, 0.0, 0.0)
            zarr_group: Zarr group to write to
            name: Array name within the group
            semantic_type: Semantic type (REQUIRED - must be explicit)
            mode: Encoding mode (AUTO, PRECISION, MEMORY, CUSTOM)
            n_elements: Broadcast target element count.
                        - Required if data is scalar/tuple/list because the
                          scalar value is stored once and represents this many
                          elements.
                        - Optional for array input, but when provided it opts
                          into broadcast/uniform validation: arrays must have
                          shape (1, ...) or be a full-length uniform array.
                          Omit n_elements for non-uniform full arrays.
            bounds: Min/max bounds for BOUNDED_SCALAR (None = auto-detect)
            positive_scalar_encoding: "linear" or "log" for POSITIVE_SCALAR
            custom_encoder: Explicit encoder name for CUSTOM mode
            color_mode: "sdr" or "hdr" for COLOR semantic type (required for float)
            chunks: Optional chunk shape for zarr dataset
            compressor: Optional compressor for zarr dataset
            deduplicate: When True (default), a byte-identical array already
                         written elsewhere is stored as a lightweight
                         ``array_ref``. Pass False for arrays whose reader
                         cannot resolve refs (e.g. line vertices/segments,
                         read as raw chunked zarr) so they are always
                         materialised.

        Raises:
            ValueError: If scalar input lacks n_elements
            ValueError: If n_elements provided but data has different length
            ValueError: If semantic type constraints are violated
            ValueError: If NaN or Inf values are present
            ValueError: If CUSTOM mode lacks custom_encoder
            ValueError: If COLOR with float dtype lacks color_mode
        """
        # Validate semantic type is provided
        if not isinstance(semantic_type, SemanticType):
            raise ValueError("semantic_type must be explicitly specified")

        # Validate CUSTOM mode
        if mode == EncodingMode.CUSTOM and custom_encoder is None:
            raise ValueError("CUSTOM mode requires custom_encoder parameter")

        # Handle scalar input - convert to array for broadcasting path
        if not isinstance(data, np.ndarray):
            # Scalar input detected
            if n_elements is None:
                raise ValueError(
                    "Scalar input requires n_elements parameter "
                    "(how many elements this scalar represents)"
                )

            # COORDINATE type blocks broadcasting
            if semantic_type == SemanticType.COORDINATE:
                raise ValueError(
                    "COORDINATE semantic type does not support scalar/broadcasting. "
                    "Positions must always be provided as full arrays."
                )

            # Convert scalar to appropriate array format
            data = self._scalar_to_array(data, semantic_type)

            # Validate the converted scalar array
            self._validate_input(data, semantic_type, color_mode)

            # Directly encode as broadcasted (skip uniformity check)
            self._encode_broadcasted_scalar(
                zarr_group, name, data, n_elements, chunks, compressor
            )
            return

        # Empty array handling: pass through without encoding. This must run
        # before the n_elements uniformity check below — `_is_uniform` indexes
        # `data[0]` and would raise IndexError on an empty array.
        if data.size == 0:
            self._write_passthrough(zarr_group, name, data, chunks, compressor)
            return

        # Array input path - validate n_elements if provided
        if n_elements is not None:
            if data.shape[0] != n_elements and data.shape[0] != 1:
                raise ValueError(
                    f"n_elements={n_elements} but data has shape {data.shape}. "
                    f"Expected either ({n_elements}, ...) or (1, ...) for broadcasting"
                )

            # If full array provided with n_elements, it MUST be uniform
            # (otherwise user should omit n_elements and let encoder decide)
            if data.shape[0] == n_elements and not self._is_uniform(data):
                raise ValueError(
                    f"n_elements={n_elements} provided with full array, but array has "
                    f"varying values. For non-uniform data, omit n_elements parameter."
                )

        # Validate input data
        self._validate_input(data, semantic_type, color_mode)

        # Get path for registry (relative to zarr root)
        array_path = f"{zarr_group.path}/{name}" if zarr_group.path else name

        # Priority 1: Broadcasting (skip for COORDINATE - positions must not be broadcasted)
        if self._is_uniform(data) and semantic_type != SemanticType.COORDINATE:
            # Use n_elements if provided, otherwise infer from data shape
            broadcast_n_elements = (
                n_elements if n_elements is not None else data.shape[0]
            )
            self._encode_broadcasted(
                zarr_group, name, data, broadcast_n_elements, chunks, compressor
            )
            return

        # Priority 2: Array Reference (content-dedup across the store).
        #
        # Callers pass deduplicate=False for arrays whose consumer cannot
        # resolve an `array_ref` — notably the line vertices/segments arrays,
        # which the lines spatial-index loader reads as raw chunked zarr and
        # never ref-resolves (and whose target would carry a different node's
        # chunk ordering anyway). Deduping a byte-identical such array (e.g.
        # two identical-shape components in a partitioned indexed line graph)
        # would otherwise make the referrer load as empty, dropping geometry.
        if deduplicate:
            match = self._registry.check(data, array_path)
            if match.is_duplicate:
                if match.target_path is None:
                    raise ValueError(
                        "Duplicate array match missing target path in registry."
                    )
                self._encode_array_ref(
                    zarr_group,
                    name,
                    match.target_path,
                    match.hash,
                    data.shape,
                    data.dtype,
                    chunks,
                    compressor,
                )
                return

        # Priority 3: LUT Encoding (skip in PRECISION mode)
        if mode != EncodingMode.PRECISION:
            if self._should_use_lut(data, semantic_type):
                self._encode_lut(
                    zarr_group, name, data, semantic_type, chunks, compressor
                )
                return

        # Priority 4: Dtype Encoding
        self._encode_dtype(
            zarr_group,
            name,
            data,
            semantic_type,
            mode,
            bounds,
            positive_scalar_encoding,
            custom_encoder,
            color_mode,
            chunks,
            compressor,
        )

    def reset(self) -> None:
        """Clear registry (call between independent scenes)."""
        self._registry.clear()

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

    def _should_use_lut(self, data: np.ndarray, semantic_type: SemanticType) -> bool:
        """Determine if LUT encoding should be used.

        Args:
            data: Input array
            semantic_type: Semantic type

        Returns:
            True if LUT encoding is beneficial
        """
        # LUT encoding criteria:
        # 1. ≤256 unique values (fits in uint8 indices)
        # 2. Array length >> unique count (meaningful savings)
        # 3. For 1D uint8: skip (already optimal)
        # 4. For 2D uint8 colors: check unique rows (LUT can still help)

        # For 2D arrays with COLOR semantic type, check unique rows
        if data.ndim == 2 and semantic_type == SemanticType.COLOR:
            if data.shape[1] <= 4:  # RGB or RGBA
                # Row mode: check unique rows
                # Use a view trick to compare entire rows
                unique_rows = np.unique(data, axis=0)
                unique_count = len(unique_rows)

                # For uint8 colors, LUT is beneficial if few unique rows
                # Original: N × d × 1 byte
                # LUT: N × 1 byte + K × d × 1 byte
                # Beneficial if N > 2*K (at least 2x savings)
                if data.dtype == np.uint8:
                    if unique_count > 256:
                        return False
                    if data.shape[0] < 2 * unique_count:
                        return False
                    return True
            else:
                # Fallback to scalar mode
                unique_count = len(np.unique(data))
        else:
            # Scalar mode: check unique values
            if data.dtype == np.uint8 and data.ndim == 1:
                return False  # 1D uint8 already optimal

            unique_count = len(np.unique(data))

        if unique_count > 256:
            return False

        # Require at least 4x savings for non-color arrays
        if data.size < 4 * unique_count:
            return False

        return True

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
            name, data=data, chunks=chunks, compressor=compressor, overwrite=True
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
    ) -> None:
        """Encode uniform array as broadcasted.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Uniform array data
            n_elements: Number of elements this represents
            chunks: Optional chunk shape (ignored for broadcast, uses minimal)
            compressor: Optional compressor
        """
        # Store single value with shape (1,) or (1, d)
        if data.ndim == 1:
            broadcast_data = data[:1]
        else:
            broadcast_data = data[:1, :]

        zarr_group.create_dataset(
            name, data=broadcast_data, compressor=compressor, overwrite=True
        )

        # Set encoding metadata
        zarr_group[name].attrs["encoding"] = {
            "name": "broadcasted",
            "n_elements": n_elements,
        }

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
        """
        # Write the single value
        zarr_group.create_dataset(
            name, data=data, compressor=compressor, overwrite=True
        )

        # Set encoding metadata
        zarr_group[name].attrs["encoding"] = {
            "name": "broadcasted",
            "n_elements": n_elements,
        }

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
            name, data=empty_data, compressor=compressor, overwrite=True
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
        semantic_type: SemanticType,
        chunks: Optional[tuple],
        compressor: Optional[Any],
    ) -> None:
        """Encode array using lookup table.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            semantic_type: Semantic type for row vs scalar mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        # Determine LUT mode
        is_color_2d = (
            data.ndim == 2
            and semantic_type == SemanticType.COLOR
            and data.shape[1] <= 4
        )
        indices_chunks: Optional[tuple] = None
        if is_color_2d:
            # Row mode: treat each row as a value
            unique_rows, indices = np.unique(data, axis=0, return_inverse=True)
            lut = unique_rows.tolist()
            lut_mode = "row"
            indices_array = indices.astype(np.uint8)
            # Adjust chunks for 1D indices array
            if chunks is not None and len(chunks) > 1:
                indices_chunks = (chunks[0],)  # Use only first dimension
            else:
                indices_chunks = chunks
        else:
            # Scalar mode: treat each element individually
            unique_vals, indices = np.unique(data.ravel(), return_inverse=True)
            lut = unique_vals.tolist()
            lut_mode = "scalar"
            indices_array = indices.reshape(data.shape).astype(np.uint8)
            # Chunks match data shape
            indices_chunks = chunks

        # Write indices
        zarr_group.create_dataset(
            name,
            data=indices_array,
            chunks=indices_chunks,
            compressor=compressor,
            overwrite=True,
        )

        # Set encoding metadata
        metadata = {
            "name": "lut_uint8",
            "lut": lut,
            "original_dtype": str(data.dtype),
        }

        # Add lut_mode and original_shape for 2D arrays
        if data.ndim > 1:
            metadata["lut_mode"] = lut_mode
            metadata["original_shape"] = list(data.shape)

        zarr_group[name].attrs["encoding"] = metadata

    def encode_cholesky_split(
        self,
        zarr_group: zarr.Group,
        diag: np.ndarray,
        offdiag: np.ndarray,
        ndim: int,
        mode: EncodingMode = EncodingMode.AUTO,
        *,
        diag_name: str = "cholesky_factors_diag",
        offdiag_name: str = "cholesky_factors_offdiag",
        n_elements: Optional[int] = None,
        chunks_diag: Optional[tuple] = None,
        chunks_offdiag: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        certificate_threshold: float = COV_CERT_RELF_P95_MAX,
    ) -> None:
        """Encode a split Cholesky pair (diag + offdiag) under one owned policy.

        The joint entry point for CHOLESKY_DIAG/CHOLESKY_OFFDIAG pairs: both
        halves always land on the SAME precision tier, and the tier choice is
        made here — callers never pass bit widths or run quality checks.

        Policy: PRECISION → float32; MEMORY → u8 unconditionally; AUTO → u8
        with an encode-time **certificate** — the pair is round-tripped through
        the exact quantization transform, Σ = L·Lᵀ is rebuilt from both halves,
        and the p95 per-splat relative Frobenius error is measured. If it
        exceeds ``certificate_threshold`` (default
        :data:`COV_CERT_RELF_P95_MAX`), AUTO escalates to u16, and — should
        even u16 fail (practically unreachable) — to float32. The measured
        certificate is recorded in each array's own ``encoding`` attrs as pure
        provenance (never needed for decode; arrays stay self-describing).

        1D gsplats have no off-diagonal terms: an ``offdiag`` with zero columns
        is accepted and simply not written (readers reconstruct the empty
        block). Both arrays are written with ``deduplicate=False`` — the
        per-channel scales live in each array's own attrs, so an ``array_ref``
        would strand the viewer without them.
        """
        diag = np.asarray(diag)
        offdiag = np.asarray(offdiag)
        n_off = ndim * (ndim - 1) // 2
        if diag.ndim != 2 or diag.shape[1] != ndim:
            raise ValueError(
                f"diag must have shape (N, {ndim}) for ndim={ndim}; got {diag.shape}"
            )
        if offdiag.ndim != 2 or offdiag.shape[1] != n_off:
            raise ValueError(
                f"offdiag must have shape (N, {n_off}) for ndim={ndim}; "
                f"got {offdiag.shape}"
            )
        write_offdiag = offdiag.shape[1] > 0

        # AUTO: certify u8, escalate on measured Σ error. Skipped for a uniform
        # pair — the broadcast priority path stores the exact float row, so
        # there is no quantization to certify. The n_rows > 0 short-circuit
        # also protects _is_uniform (which indexes data[0]) from empty arrays:
        # zero-splat pairs fall straight through to encode()'s size==0
        # passthrough, matching the pre-joint-call behavior.
        n_rows = diag.shape[0]
        pair_uniform = n_rows > 0 and (
            self._is_uniform(diag) and (not write_offdiag or self._is_uniform(offdiag))
        )
        certificate: Optional[dict] = None
        chosen_bits: Optional[int] = 8
        eff_mode = mode
        if mode == EncodingMode.AUTO and n_rows > 0 and not pair_uniform:
            # The percentile only needs a bounded SAMPLE of rows, so the f64 Σ
            # scratch stays capped (a 10M-splat flat fit would otherwise build
            # ~2.7 GB of transient Σ arrays). Evenly-spaced rows = deterministic
            # and, over Hilbert-ordered splats, spatially uniform. The
            # quantization SCALES, however, must come from the FULL columns —
            # exactly what the real encode uses — or the certificate lies; the
            # forward compand is monotonic, so companding the raw column
            # min/max equals the full-array companded min/max.
            capped = n_rows > COV_CERT_SAMPLE_MAX
            sel: Any = (
                np.linspace(0, n_rows - 1, COV_CERT_SAMPLE_MAX).astype(np.intp)
                if capped
                else slice(None)
            )
            diag_s = diag[sel]
            off_s = offdiag[sel]
            n_s = diag_s.shape[0]

            def col_scales(x: np.ndarray, *, signed: bool) -> tuple:
                lo = self._perchannel_log_forward(x.min(axis=0)[None, :], signed=signed)
                hi = self._perchannel_log_forward(x.max(axis=0)[None, :], signed=signed)
                return lo[0], hi[0]

            lo_d, hi_d = col_scales(diag, signed=False)
            lo_o, hi_o = (
                col_scales(offdiag, signed=True) if write_offdiag else (None, None)
            )

            # Reference for the error measurement is the FORWARD-VALID input:
            # the log encoder clamps invalid negative diagonal entries by
            # policy, and that validation loss must not read as quantization
            # error (it is identical at every tier, so escalating cannot
            # recover it). The reference Σ is tier-independent — build it once,
            # outside the escalation loop.
            diag_ref = np.maximum(diag_s.astype(np.float64), 0.0)
            s_ref = self._sigma_from_split(diag_ref, off_s, ndim).reshape(n_s, -1)
            den = np.maximum(np.linalg.norm(s_ref, axis=1), 1e-30)
            tried: list[tuple[int, float]] = []
            chosen_bits = None
            for bits in (8, 16):
                diag_q = self._perchannel_log_roundtrip(
                    diag_s, bits, signed=False, lo=lo_d, hi=hi_d
                )
                off_q = (
                    self._perchannel_log_roundtrip(
                        off_s, bits, signed=True, lo=lo_o, hi=hi_o
                    )
                    if write_offdiag
                    else off_s
                )
                s_q = self._sigma_from_split(diag_q, off_q, ndim).reshape(n_s, -1)
                relf = self._relf_p95(s_ref, den, s_q)
                tried.append((bits, relf))
                if relf <= certificate_threshold:
                    chosen_bits = bits
                    break
            if chosen_bits is not None:
                tier = f"u{chosen_bits}"
                value = tried[-1][1]
            else:
                tier, value = "float32", 0.0
                eff_mode = EncodingMode.PRECISION
            certificate = {
                "metric": "cov_relf_p95",
                "value": float(value),
                "threshold": float(certificate_threshold),
                "tier": tier,
            }
            if capped:
                certificate["sample"] = int(n_s)
            if tier == "u16":
                warnings.warn(
                    f"CHOLESKY '{diag_name}': covariance certificate "
                    f"cov_relf_p95 = {tried[0][1]:.4g} > {certificate_threshold} "
                    f"at uint8 — escalating to uint16 "
                    f"(measured {tried[1][1]:.4g}).",
                    UserWarning,
                    stacklevel=2,
                )
            elif tier == "float32":
                warnings.warn(
                    f"CHOLESKY '{diag_name}': covariance certificate "
                    f"cov_relf_p95 = {tried[1][1]:.4g} > {certificate_threshold} "
                    "even at uint16 — storing float32.",
                    UserWarning,
                    stacklevel=2,
                )

        # Delegate each half to the full encode() priority ladder (broadcast /
        # LUT / dtype) with the certified tier forced for the quantized path.
        self._perchannel_bits_override = chosen_bits
        try:
            self.encode(
                data=diag,
                zarr_group=zarr_group,
                name=diag_name,
                semantic_type=SemanticType.CHOLESKY_DIAG,
                mode=eff_mode,
                n_elements=n_elements,
                chunks=chunks_diag,
                compressor=compressor,
                deduplicate=False,
            )
            if write_offdiag:
                self.encode(
                    data=offdiag,
                    zarr_group=zarr_group,
                    name=offdiag_name,
                    semantic_type=SemanticType.CHOLESKY_OFFDIAG,
                    mode=eff_mode,
                    n_elements=n_elements,
                    chunks=chunks_offdiag,
                    compressor=compressor,
                    deduplicate=False,
                )
        finally:
            self._perchannel_bits_override = None

        # Record the certificate as provenance inside each array's own encoding
        # attrs — only where the tier decision actually applied (the broadcast /
        # LUT priority paths store exact values, so no certificate there).
        if certificate is not None:
            targets = [diag_name] + ([offdiag_name] if write_offdiag else [])
            for arr_name in targets:
                if arr_name not in zarr_group:
                    continue
                enc = dict(zarr_group[arr_name].attrs.get("encoding", {}))
                enc_name = str(enc.get("name", ""))
                quantized = enc_name.startswith(
                    ("log_perchannel", "signed_log_perchannel")
                )
                if quantized or (
                    certificate["tier"] == "float32" and enc_name == "float32"
                ):
                    enc["certificate"] = certificate
                    zarr_group[arr_name].attrs["encoding"] = enc

    @staticmethod
    def _cov_relf_p95(
        diag: np.ndarray,
        diag_q: np.ndarray,
        offdiag: np.ndarray,
        offdiag_q: np.ndarray,
        ndim: int,
    ) -> float:
        """p95 over splats of the relative Frobenius error of Σ = L·Lᵀ.

        Composed from :meth:`_sigma_from_split` + :meth:`_relf_p95`; the
        escalation loop in :meth:`encode_cholesky_split` uses the pieces
        directly so the tier-independent reference Σ is built only once.
        """
        n = diag.shape[0]
        s0 = ArrayEncoder._sigma_from_split(diag, offdiag, ndim).reshape(n, -1)
        sq = ArrayEncoder._sigma_from_split(diag_q, offdiag_q, ndim).reshape(n, -1)
        den = np.maximum(np.linalg.norm(s0, axis=1), 1e-30)
        return ArrayEncoder._relf_p95(s0, den, sq)

    @staticmethod
    def _sigma_from_split(
        diag: np.ndarray, offdiag: np.ndarray, ndim: int
    ) -> np.ndarray:
        """(N, d, d) Σ = L·Lᵀ from split Cholesky halves.

        Rebuilds lower-triangular L with a local row-major
        ``np.tril_indices`` layout — the same packing convention as
        ``gsplats.utils.trils`` (locked by a parity test), kept local so the
        encoding package takes no gsplats dependency.
        """
        n = diag.shape[0]
        rows, cols = np.tril_indices(ndim)
        off = rows != cols
        tri = np.zeros((n, ndim, ndim), dtype=np.float64)
        tri[:, np.arange(ndim), np.arange(ndim)] = diag.astype(np.float64)
        if offdiag.shape[1] > 0:
            tri[:, rows[off], cols[off]] = offdiag.astype(np.float64)
        return np.asarray(np.einsum("nij,nkj->nik", tri, tri))

    @staticmethod
    def _relf_p95(s_ref: np.ndarray, den: np.ndarray, s_q: np.ndarray) -> float:
        """p95 of per-row relative Frobenius error given flattened Σ matrices."""
        num = np.linalg.norm(s_q - s_ref, axis=1)
        return float(np.percentile(num / den, 95))

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
    ) -> None:
        """Encode array using dtype-based encoding.

        This is the fallback encoding that handles quantization and dtype
        conversion based on semantic type and mode.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            semantic_type: Semantic type
            mode: Encoding mode
            bounds: Bounds for BOUNDED_SCALAR
            positive_scalar_encoding: "linear" or "log" for POSITIVE_SCALAR
            custom_encoder: Explicit encoder for CUSTOM mode
            color_mode: "sdr" or "hdr" for COLOR
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        if mode == EncodingMode.CUSTOM:
            if custom_encoder is None:
                raise ValueError("CUSTOM mode requires custom_encoder parameter")
            # Use explicitly specified encoder
            self._encode_custom(
                zarr_group, name, data, custom_encoder, bounds, chunks, compressor
            )
        elif semantic_type == SemanticType.COORDINATE:
            self._encode_coordinate(zarr_group, name, data, mode, chunks, compressor)
        elif semantic_type == SemanticType.COLOR:
            self._encode_color(
                zarr_group, name, data, mode, color_mode, chunks, compressor
            )
        elif semantic_type == SemanticType.BOUNDED_SCALAR:
            self._encode_bounded_scalar(
                zarr_group, name, data, mode, bounds, chunks, compressor
            )
        elif semantic_type == SemanticType.POSITIVE_SCALAR:
            self._encode_positive_scalar(
                zarr_group,
                name,
                data,
                mode,
                positive_scalar_encoding,
                chunks,
                compressor,
            )
        elif semantic_type == SemanticType.CHOLESKY:
            self._encode_cholesky(zarr_group, name, data, mode, chunks, compressor)
        elif semantic_type == SemanticType.CHOLESKY_DIAG:
            # Cholesky diagonal is positive → generic per-channel log encoding.
            self._encode_log_perchannel(
                zarr_group, name, data, mode, chunks, compressor
            )
        elif semantic_type == SemanticType.CHOLESKY_OFFDIAG:
            # Cholesky off-diagonal is signed → generic per-channel signed-log.
            self._encode_signed_log_perchannel(
                zarr_group, name, data, mode, chunks, compressor
            )
        elif semantic_type == SemanticType.INDEX:
            self._encode_index(zarr_group, name, data, mode, chunks, compressor)
        elif semantic_type == SemanticType.UNIT_VECTOR:
            self._encode_unit_vector(zarr_group, name, data, mode, chunks, compressor)
        else:
            raise ValueError(f"Unknown semantic type: {semantic_type}")

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
            # HDR colors: use float
            if mode == EncodingMode.PRECISION or mode == EncodingMode.AUTO:
                encoded_data = data.astype(np.float32)
                encoder_name = "float32"
            elif mode == EncodingMode.MEMORY:
                # Check if float16 is allowed, fallback to float32 if not
                if self._float16_allowed:
                    encoded_data = data.astype(np.float16)
                    encoder_name = "float16"
                else:
                    encoded_data = data.astype(np.float32)
                    encoder_name = "float32"
            else:
                raise ValueError("HDR colors require float dtype")
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
            compressor=compressor,
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
                    normalized = (data - min_val) / span
                    # Use rounding for better accuracy (not truncation)
                    encoded_data = np.clip(np.round(normalized * 255), 0, 255).astype(
                        np.uint8
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
                    normalized = (data - min_val) / span
                    # Use rounding for better accuracy (not truncation)
                    encoded_data = np.clip(
                        np.round(normalized * 65535), 0, 65535
                    ).astype(np.uint16)
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

        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=compressor,
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

            if encoding_type == "log":
                # Logarithmic encoding
                bits = 8
                if max_val == 0:
                    encoded_data = np.zeros_like(data, dtype=np.uint8)
                    max_log = 0.0
                else:
                    max_log = float(np.log1p(max_val))
                    log_vals = np.log1p(data)
                    normalized = log_vals / max_log
                    encoded_data = np.clip(normalized * 255, 0, 255).astype(np.uint8)

                encoder_name = "log_scalar_uint8"
                metadata = {
                    "name": encoder_name,
                    "max_log": max_log,
                    "bits": bits,
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
                elif bits == 8:
                    # Dynamic range <= 256, uint8 is sufficient
                    normalized = data / max_val
                    # Use rounding for better accuracy (not truncation)
                    encoded_data = np.clip(np.round(normalized * 255), 0, 255).astype(
                        np.uint8
                    )
                    encoder_name = "bounded_scalar_uint8"
                    metadata = {
                        "name": encoder_name,
                        "min": 0.0,
                        "max": max_val,
                        "bits": 8,
                        "original_dtype": original_dtype,
                    }
                elif bits == 16:
                    # Dynamic range <= 65536, uint16 is sufficient
                    normalized = data / max_val
                    # Use rounding for better accuracy (not truncation)
                    encoded_data = np.clip(
                        np.round(normalized * 65535), 0, 65535
                    ).astype(np.uint16)
                    encoder_name = "bounded_scalar_uint16"
                    metadata = {
                        "name": encoder_name,
                        "min": 0.0,
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
            raise ValueError(f"Unexpected mode for POSITIVE_SCALAR: {mode}")

        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = metadata

    def _encode_cholesky(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode CHOLESKY semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)

        target_dtype: np.dtype[Any]
        if mode == EncodingMode.PRECISION or mode == EncodingMode.AUTO:
            target_dtype = np.dtype("float32")
        elif mode == EncodingMode.MEMORY:
            # Check if float16 is allowed, fallback to float32 if not
            if self._float16_allowed:
                target_dtype = np.dtype("float16")
            else:
                target_dtype = np.dtype("float32")
        else:
            raise ValueError(f"Unexpected mode for CHOLESKY: {mode}")

        encoded_data = data.astype(target_dtype)
        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }

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
        return u, lo.astype(np.float64).tolist(), hi.astype(np.float64).tolist()

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
        zarr_group.create_dataset(
            name, data=u, chunks=chunks, compressor=compressor, overwrite=True
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"linear_perchannel_u{bits}",
            "col_lo": lo_list,
            "col_hi": hi_list,
            "bits": bits,
            "original_dtype": original_dtype or str(data.dtype),
        }

    @staticmethod
    def _perchannel_log_forward(data: np.ndarray, *, signed: bool) -> np.ndarray:
        """Compand an (N, C) array into log space (the exact encode transform)."""
        x = data.astype(np.float64)
        if signed:
            return np.asarray(np.sign(x) * np.log1p(np.abs(x)))
        return np.asarray(np.log1p(np.maximum(x, 0.0)))

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

        Uses the SAME forward compand and per-column quantizer as the real
        encoders and the same inverse as the decoder, so a certificate computed
        from it can never diverge from what actually lands on disk. ``lo``/``hi``
        accept precomputed COMPANDED-domain column scales — the certificate
        passes full-array scales while round-tripping only a row sample.
        """
        y = ArrayEncoder._perchannel_log_forward(data, signed=signed)
        u, lo_list, hi_list = ArrayEncoder._quantize_per_column(y, bits, lo=lo, hi=hi)
        levels = (1 << bits) - 1
        lo_a = np.asarray(lo_list, dtype=np.float64)
        hi_a = np.asarray(hi_list, dtype=np.float64)
        yq = lo_a + u.astype(np.float64) / levels * (hi_a - lo_a)
        if signed:
            return np.asarray(np.sign(yq) * np.expm1(np.abs(yq)))
        return np.asarray(np.expm1(yq))

    def _encode_log_perchannel(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Generic per-channel LOG quantization of a non-negative (N, C) array.

        Each channel (column) gets its own ``[lo, hi]`` in log space, so a wide
        per-channel dynamic range keeps relative precision. PRECISION → float32;
        AUTO/MEMORY → ``log_perchannel_u8``. AUTO escalates to
        ``log_perchannel_u16`` only through :meth:`encode_cholesky_split`, whose
        encode-time certificate measures the actual Σ reconstruction error —
        pair callers should use that entry point. (First consumer: the Cholesky
        diagonal; reusable for any positive per-channel field.) Negative inputs
        are clamped to 0 before the log.
        """
        original_dtype = str(data.dtype)
        if mode == EncodingMode.PRECISION:
            self._write_float(
                zarr_group, name, data, np.dtype("float32"), chunks, compressor
            )
            return
        bits = self._perchannel_bits_override or 8
        y = self._perchannel_log_forward(data, signed=False)
        u, lo, hi = self._quantize_per_column(y, bits)
        zarr_group.create_dataset(
            name, data=u, chunks=chunks, compressor=compressor, overwrite=True
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"log_perchannel_u{bits}",
            "col_lo": lo,
            "col_hi": hi,
            "bits": bits,
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
    ) -> None:
        """Generic per-channel SIGNED-LOG quantization of a signed (N, C) array.

        ``y = sign(x)·log1p(|x|)`` companding gives fine resolution near zero
        (where most mass of a zero-centred signal lives) and coarse in the tails,
        with per-channel ``[lo, hi]``. PRECISION → float32; AUTO/MEMORY →
        ``signed_log_perchannel_u8``, with AUTO escalation to u16 owned by
        :meth:`encode_cholesky_split` (see ``_encode_log_perchannel``). (First
        consumer: the Cholesky off-diagonal.)
        """
        original_dtype = str(data.dtype)
        if mode == EncodingMode.PRECISION:
            self._write_float(
                zarr_group, name, data, np.dtype("float32"), chunks, compressor
            )
            return
        bits = self._perchannel_bits_override or 8
        y = self._perchannel_log_forward(data, signed=True)
        u, lo, hi = self._quantize_per_column(y, bits)
        zarr_group.create_dataset(
            name, data=u, chunks=chunks, compressor=compressor, overwrite=True
        )
        zarr_group[name].attrs["encoding"] = {
            "name": f"signed_log_perchannel_u{bits}",
            "col_lo": lo,
            "col_hi": hi,
            "bits": bits,
            "original_dtype": original_dtype,
        }

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
        zarr_group.create_dataset(
            name,
            data=data.astype(target_dtype),
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }

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
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }

    def _encode_unit_vector(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode UNIT_VECTOR semantic type.

        Currently uses standard float encoding. Specialized encodings
        (octahedral) are not yet implemented.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)

        target_dtype: np.dtype[Any]
        if mode == EncodingMode.PRECISION or mode == EncodingMode.AUTO:
            target_dtype = np.dtype("float32")
        elif mode == EncodingMode.MEMORY:
            # Check if float16 is allowed, fallback to float32 if not
            if self._float16_allowed:
                target_dtype = np.dtype("float16")
            else:
                target_dtype = np.dtype("float32")
        else:
            raise ValueError(f"Unexpected mode for UNIT_VECTOR: {mode}")

        encoded_data = data.astype(target_dtype)
        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
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
        """Encode using explicitly specified encoder.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            encoder_name: Explicit encoder name
            bounds: Bounds for bounded_scalar encoders
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)
        metadata: dict[str, Any]

        if encoder_name in ("float32", "float16"):
            # Passthrough with dtype conversion
            target_dtype = np.dtype(encoder_name)
            encoded_data = data.astype(target_dtype)
            metadata = {"name": encoder_name, "original_dtype": original_dtype}

        elif encoder_name in ("uint8", "uint16", "uint32", "uint64"):
            # Direct uint conversion (for INDEX)
            target_dtype = np.dtype(encoder_name)
            encoded_data = data.astype(target_dtype)
            metadata = {"name": encoder_name, "original_dtype": original_dtype}

        elif encoder_name in ("bounded_scalar_uint8", "bounded_scalar_uint16"):
            # Bounded scalar encoding
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
                normalized = (data - min_val) / span
                encoded_data = np.clip(normalized * max_int, 0, max_int).astype(
                    np.uint8 if bits == 8 else np.uint16
                )

            metadata = {
                "name": encoder_name,
                "min": min_val,
                "max": max_val,
                "bits": bits,
                "original_dtype": original_dtype,
            }

        elif encoder_name in ("log_scalar_uint8", "log_scalar_uint16"):
            # Log scalar encoding
            bits = 8 if "uint8" in encoder_name else 16
            max_int = (2**bits) - 1

            max_val = float(np.max(data))
            max_log = float(np.log1p(max_val))
            log_vals = np.log1p(data)
            normalized = log_vals / max_log
            encoded_data = np.clip(normalized * max_int, 0, max_int).astype(
                np.uint8 if bits == 8 else np.uint16
            )

            metadata = {
                "name": encoder_name,
                "max_log": max_log,
                "bits": bits,
                "original_dtype": original_dtype,
            }

        elif encoder_name in ("rgb_uint8", "rgb_uint16"):
            # Color encoding (SDR)
            max_int = 255 if "uint8" in encoder_name else 65535
            encoded_data = np.clip(data * max_int, 0, max_int).astype(
                np.uint8 if "uint8" in encoder_name else np.uint16
            )
            metadata = {"name": encoder_name, "original_dtype": original_dtype}

        else:
            raise ValueError(f"Unknown custom encoder: {encoder_name}")

        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = metadata
