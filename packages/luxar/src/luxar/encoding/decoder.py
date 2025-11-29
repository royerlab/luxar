"""Array decoder for encoded zarr arrays.

The decoder reads encoding metadata and applies appropriate decoding
transformations to recover original data.
"""

from typing import Optional

import numpy as np
import zarr


class ArrayDecoder:
    """Decode any encoded array from zarr.

    The decoder handles all encoding types by reading metadata and applying
    the appropriate inverse transformation. It supports recursive decoding
    for array references.
    """

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
        enc = zarr_array.attrs.get("encoding", {})
        name = enc.get("name", "none")

        # Special encodings (must handle first)
        if name == "broadcasted":
            return self._expand_broadcasted(zarr_array, enc)
        elif name == "array_ref":
            return self._follow_ref(zarr_array, enc, zarr_root)
        elif name == "lut_uint8":
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
        elif name == "rgb_uint8":
            return self._decode_color(zarr_array, enc)
        elif name == "rgb_uint16":
            return self._decode_color(zarr_array, enc)

        # Passthrough (none, float16, float32, uint8, etc.)
        else:
            return zarr_array[:]

    def _decode_bounded_scalar(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode bounded scalar: uint → original dtype using min/max.

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype
        """
        data = arr[:]
        min_val = enc["min"]
        max_val = enc["max"]
        bits = enc["bits"]
        original_dtype = enc.get("original_dtype", "float32")

        # Use float64 intermediate for precision, then cast to original dtype
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = normalized * (max_val - min_val) + min_val
        return result.astype(original_dtype)

    def _decode_log_scalar(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode log scalar: uint → original dtype using expm1.

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype
        """
        data = arr[:]
        max_log = enc["max_log"]
        bits = enc["bits"]
        original_dtype = enc.get("original_dtype", "float32")

        # Use float64 intermediate for precision, then cast to original dtype
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = np.expm1(normalized * max_log)
        return result.astype(original_dtype)

    def _decode_color(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Decode color: uint8/uint16 [0,max] → original dtype [0,1].

        Args:
            arr: Zarr array with encoded data
            enc: Encoding metadata

        Returns:
            Decoded array with original dtype in [0, 1] range
        """
        data = arr[:]
        original_dtype = enc.get("original_dtype", "float32")

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
        indices = arr[:]
        lut = np.array(enc["lut"], dtype=enc["original_dtype"])
        lut_mode = enc.get("lut_mode", "scalar")  # Default for 1D

        if lut_mode == "row":
            # Row mode: indices are (N,), lut is (K, d)
            # Result is (N, d)
            return lut[indices]
        else:
            # Scalar mode: indices match original shape
            # lut is (K,), indices may be (N,) or (N, d)
            return lut[indices]

    def _expand_broadcasted(self, arr: zarr.Array, enc: dict) -> np.ndarray:
        """Expand broadcasted array to full size.

        Args:
            arr: Zarr array with single value
            enc: Encoding metadata with n_elements

        Returns:
            Expanded array with repeated value
        """
        data = arr[:]  # Shape (1,) or (1, d)
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
