"""Array encoder with priority-based encoding selection.

The encoder follows a strict priority order:
1. Broadcasting (if all values identical)
2. Array Reference (if duplicate exists)
3. LUT Encoding (if ≤65536 unique values, tiered uint8/uint16 indices,
   and mode != PRECISION)
4. Dtype Encoding (based on semantic type and mode)
"""

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import zarr

from ._encoders.cholesky import CholeskyEncoderMixin
from ._encoders.perchannel import PerChannelEncoderMixin
from ._encoders.structural import StructuralEncoderMixin
from .modes import EncodingMode
from .registry import ArrayRefRegistry
from .semantic_types import SemanticType

__all__ = ["ArrayEncoder"]


class ArrayEncoder(
    StructuralEncoderMixin, PerChannelEncoderMixin, CholeskyEncoderMixin
):
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
        lut_json_max_bytes: int = 512 * 1024,
    ) -> None:
        """Initialize encoder with optional broadcasting tolerance and float16 control.

        Args:
            broadcast_rtol: Relative tolerance for broadcasting check
                (default: 0.0 = exact equality)
            broadcast_atol: Absolute tolerance for broadcasting check
                (default: 0.0 = exact equality)
            float16_allowed: Allow float16 encoding in MEMORY mode (default: False for TypeScript compatibility)
            lut_json_max_bytes: Metadata-health cap for the uint16 LUT tier:
                the LUT values live as JSON in ``.zattrs`` AND are duplicated
                by consolidated ``.zmetadata``, which the viewer parses at
                scene-open for ALL nodes — one greedy LUT would tax every
                load. The estimated doubled JSON size must stay under this
                cap (default 512 KiB ≈ 4-4.5K unique float RGB rows, since the
                estimate doubles the raw JSON for .zmetadata). The
                uint8 tier is unaffected (≤256 values is at most ~22 KiB).

        Note: Using non-zero tolerance is experimental and should be used
        with caution. Exact equality (default) is safe for all semantic types
        including indices and colors.
        """
        self._registry = ArrayRefRegistry()
        self._broadcast_rtol = broadcast_rtol
        self._broadcast_atol = broadcast_atol
        self._float16_allowed = float16_allowed
        self._lut_json_max_bytes = lut_json_max_bytes

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
        """Encode array or scalar and write to zarr group.

        Follows priority order:
        1. Broadcasting (if scalar input OR all values identical within tolerance)
        2. Array reference (if duplicate exists)
        3. LUT encoding (≤65536 unique values, tiered uint8/uint16 indices, mode != PRECISION)
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
            allow_lut: When True (default), arrays with few unique values may
                         store as an exact ``lut_uint8/16``. Pass False for
                         arrays whose reader has no encoding dispatch (line
                         vertices — same raw-read rationale as
                         ``deduplicate``); grid-snapped coordinates would
                         otherwise LUT-encode and decode as indices.
            _perchannel_bits: Internal-only. Forces the quantization tier (8 or
                         16) of the per-channel log encoders for the
                         CHOLESKY_DIAG / CHOLESKY_OFFDIAG semantic types. Set
                         only by :meth:`encode_cholesky_split`, which owns the
                         certified u8→u16 escalation for the diag/offdiag pair;
                         ``None`` (all other callers) defaults to 8.

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
                zarr_group, name, data, n_elements, chunks, compressor, semantic_type
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
                zarr_group,
                name,
                data,
                broadcast_n_elements,
                chunks,
                compressor,
                semantic_type,
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

        # Priority 3: LUT Encoding (skip in PRECISION mode; callers pass
        # allow_lut=False for arrays whose reader has no encoding dispatch —
        # same rationale as deduplicate=False above)
        if allow_lut and mode != EncodingMode.PRECISION:
            plan = self._lut_plan(data, semantic_type)
            if plan is not None:
                self._encode_lut(zarr_group, name, data, plan, chunks, compressor)
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
            _perchannel_bits,
        )

    def reset(self) -> None:
        """Clear registry (call between independent scenes)."""
        self._registry.clear()

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
            perchannel_bits: Internal tier override (8/16) forwarded to the
                Cholesky per-channel log encoders; ``None`` defaults to 8.
        """
        # CUSTOM mode overrides semantic-type selection with an explicit encoder.
        if mode == EncodingMode.CUSTOM:
            if custom_encoder is None:
                raise ValueError("CUSTOM mode requires custom_encoder parameter")
            self._encode_custom(
                zarr_group, name, data, custom_encoder, bounds, chunks, compressor
            )
            return

        # Registry: semantic type -> the writer that owns it. Each entry closes
        # over this call's arguments (the writers keep their heterogeneous
        # signatures) — this table replaces a 10-branch if/elif dispatch.
        dispatch: dict[SemanticType, Callable[[], None]] = {
            SemanticType.COORDINATE: lambda: self._encode_coordinate(
                zarr_group, name, data, mode, chunks, compressor
            ),
            SemanticType.COLOR: lambda: self._encode_color(
                zarr_group, name, data, mode, color_mode, chunks, compressor
            ),
            SemanticType.BOUNDED_SCALAR: lambda: self._encode_bounded_scalar(
                zarr_group, name, data, mode, bounds, chunks, compressor
            ),
            SemanticType.POSITIVE_SCALAR: lambda: self._encode_positive_scalar(
                zarr_group,
                name,
                data,
                mode,
                positive_scalar_encoding,
                chunks,
                compressor,
            ),
            SemanticType.CHOLESKY: lambda: self._encode_cholesky(
                zarr_group, name, data, mode, chunks, compressor
            ),
            # Cholesky diagonal is positive → generic per-channel log encoding.
            SemanticType.CHOLESKY_DIAG: lambda: self._encode_log_perchannel(
                zarr_group, name, data, mode, chunks, compressor, perchannel_bits
            ),
            # Cholesky off-diagonal is signed → generic per-channel signed-log.
            SemanticType.CHOLESKY_OFFDIAG: lambda: self._encode_signed_log_perchannel(
                zarr_group, name, data, mode, chunks, compressor, perchannel_bits
            ),
            SemanticType.INDEX: lambda: self._encode_index(
                zarr_group, name, data, mode, chunks, compressor
            ),
        }
        handler = dispatch.get(semantic_type)
        if handler is None:
            raise ValueError(f"Unknown semantic type: {semantic_type}")
        handler()
