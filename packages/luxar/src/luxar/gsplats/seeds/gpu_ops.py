# gpu_ops.py
"""
GPU-accelerated operations for seed generation using PyTorch.

This module provides GPU implementations of core operations used in seeding:
- Sobel gradient computation
- Amplitude interpolation via grid sampling

All operations use pure PyTorch (no kornia/faiss dependencies) for minimal
dependencies and easier maintenance.

Expected speedups: substantial for large 3D/4D volumes on CUDA GPUs —
often orders of magnitude depending on GPU and problem size.
"""

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _get_device(device: Optional[str] = None) -> str:
    """
    Resolve device string to actual device.

    Parameters
    ----------
    device : str, optional
        Device specification. Options:
        - None: Returns 'cpu' (backward compatible default)
        - 'auto': Auto-detect (cuda > mps > cpu)
        - 'cpu': Force CPU
        - 'cuda': NVIDIA GPU (if available, else fallback to cpu with warning)
        - 'mps': Apple Metal (if available, else fallback to cpu with warning)
        - 'cuda:0', 'cuda:1': Specific GPU device

    Returns
    -------
    str
        Resolved device string ('cpu', 'cuda', 'mps', or 'cuda:N').

    Examples
    --------
    >>> _get_device(None)
    'cpu'
    >>> _get_device('auto')  # Returns 'cuda' if available, else 'mps', else 'cpu'
    'cuda'
    >>> _get_device('cuda')  # Returns 'cuda' if available, else 'cpu' with warning
    'cuda'
    """
    if device is None:
        return "cpu"

    from luxar.gsplats.utils.device import is_mps_available, resolve_torch_device

    if device == "auto":
        return str(resolve_torch_device())

    if device == "cpu":
        return "cpu"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return device

    if device == "mps":
        if not is_mps_available():
            warnings.warn(
                "MPS requested but not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return device

    # Unknown device - fallback to CPU
    warnings.warn(
        f"Unknown device '{device}', falling back to CPU.", RuntimeWarning, stacklevel=2
    )
    return "cpu"


def should_use_gpu(V: np.ndarray, device: str) -> bool:
    """
    Determine if GPU should be used based on volume size and device.

    For small volumes (<50³), CPU overhead is acceptable and GPU transfer
    overhead may dominate. For large volumes, GPU provides significant speedup.

    Parameters
    ----------
    V : np.ndarray
        Input volume to process.
    device : str
        Target device ('cpu', 'cuda', 'mps', etc.).

    Returns
    -------
    bool
        True if GPU should be used, False otherwise.

    Notes
    -----
    - Always returns False for 'cpu' device
    - Returns False for volumes smaller than 50³ voxels
    - Returns True for large volumes on GPU devices
    """
    if device == "cpu":
        return False

    # Skip GPU for small volumes (overhead not worth it)
    volume_size = np.prod(V.shape)
    if volume_size < 50**3:
        return False

    return True


def _conv1d_along_axis(
    x: torch.Tensor, kernel: torch.Tensor, axis: int, padding: str = "same"
) -> torch.Tensor:
    """
    Apply 1D convolution along a specific axis of nD tensor.

    This enables separable convolutions for efficient nD filtering.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (D0, D1, ..., Dn-1) (no batch dimension).
    kernel : torch.Tensor
        1D convolution kernel of shape (K,).
    axis : int
        Axis along which to apply convolution (0 to n-1).
    padding : str, default='same'
        Padding mode: 'same' or 'valid'.

    Returns
    -------
    torch.Tensor
        Convolved tensor with same shape as input (if padding='same').

    Notes
    -----
    Implementation strategy:
    1. Permute to move target axis to last position
    2. Reshape to (batch, channels, length) for conv1d
    3. Apply conv1d
    4. Reshape and permute back to original layout

    This is more efficient than using conv2d/conv3d for separable kernels.
    """
    # Permute to move axis to last position
    # Example for 3D (axis=1): (D0, D1, D2) → (D0, D2, D1)
    ndim = x.ndim
    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    x_perm = x.permute(*perm)

    # Reshape to (batch * prod(other_dims), 1, length)
    orig_shape = x_perm.shape
    batch_size = int(np.prod(orig_shape[:-1]))
    length = orig_shape[-1]
    x_flat = x_perm.reshape(batch_size, 1, length)

    # Prepare kernel for conv1d: (out_channels=1, in_channels=1, kernel_size)
    # PyTorch conv1d performs cross-correlation, so flip kernel for true convolution
    kernel_1d = kernel.flip(0).view(1, 1, -1)

    # Compute padding for 'same' mode
    if padding == "same":
        pad_total = len(kernel) - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_flat = F.pad(x_flat, (pad_left, pad_right), mode="replicate")

    # Apply conv1d
    y_flat = F.conv1d(x_flat, kernel_1d)

    # Reshape back to original layout
    y_perm = y_flat.reshape(*orig_shape)

    # Permute back to original axis order
    y = y_perm.permute(*perm)

    return y


def _compute_nd_sobel_magnitude_gpu(V_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute nD Sobel gradient magnitude on GPU.

    Uses the full separable Sobel kernel matching ``scipy.ndimage.sobel``:
    for each axis, apply the differentiation kernel ``[-1, 0, 1]`` along
    that axis and the smoothing kernel ``[1, 2, 1]`` (unnormalized) along
    all perpendicular axes. This is substantially faster than scipy on
    large volumes (often orders of magnitude, GPU-dependent).

    Parameters
    ----------
    V_tensor : torch.Tensor
        Input n-dimensional image/volume on GPU.

    Returns
    -------
    torch.Tensor
        Gradient magnitude (same shape as V_tensor).

    Notes
    -----
    Sobel gradient for each axis is computed as a separable convolution:

    - Differentiation kernel ``[-1, 0, 1]`` along the target axis
    - Smoothing kernel ``[1, 2, 1]`` (unnormalized) along every other axis

    The overall magnitude is ``sqrt(sum(sobel_i^2))``.

    This matches ``scipy.ndimage.sobel`` which applies the same separable
    decomposition.  The ``[1, 2, 1]`` smoothing suppresses high-frequency
    noise, producing more robust edge detection than a bare central
    difference.
    """
    ndim = V_tensor.ndim
    diff_kernel = torch.tensor([-1.0, 0.0, 1.0], device=V_tensor.device)
    # Unnormalized smoothing kernel, matching scipy.ndimage.sobel
    smooth_kernel = torch.tensor([1.0, 2.0, 1.0], device=V_tensor.device)

    grad_sq_sum = torch.zeros_like(V_tensor)

    for axis in range(ndim):
        result = V_tensor
        for other_axis in range(ndim):
            if other_axis == axis:
                result = _conv1d_along_axis(
                    result, diff_kernel, other_axis, padding="same"
                )
            else:
                result = _conv1d_along_axis(
                    result, smooth_kernel, other_axis, padding="same"
                )
        grad_sq_sum += result**2

    return torch.sqrt(grad_sq_sum)


def sample_amplitudes_gpu(
    V: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """
    Sample amplitudes from volume at given coordinates using GPU interpolation.

    This is substantially faster than scipy.ndimage.map_coordinates for large
    volumes (often orders of magnitude on GPU; varies by hardware).

    Parameters
    ----------
    V : torch.Tensor
        Input n-dimensional volume on GPU.
    coords : torch.Tensor
        Coordinates to sample at, shape (N, ndim). Values are in voxel coordinates.
    mode : str, default='bilinear'
        Interpolation mode: 'bilinear' (linear) or 'nearest'.

    Returns
    -------
    torch.Tensor
        Sampled values at each coordinate, shape (N,).

    Notes
    -----
    Implementation uses F.grid_sample:
    - Normalize coordinates from [0, shape-1] to [-1, 1]
    - Add batch and channel dimensions
    - Reshape coords for grid_sample format
    - Sample using grid_sample
    - Extract results

    This replaces scipy.ndimage.map_coordinates which runs on CPU.

    Grid sample coordinate system:
    - -1: Left/top edge
    - 0: Center
    - +1: Right/bottom edge
    - Mapping: normalized = 2 * (voxel / (shape - 1)) - 1
    """
    ndim = V.ndim
    device = V.device

    # Check dimensionality support upfront
    if ndim not in [2, 3]:
        raise NotImplementedError(
            f"GPU interpolation only supports 2D and 3D volumes. "
            f"Got {ndim}D volume. Use device='cpu' for {ndim}D volumes."
        )

    # Normalize coordinates to [-1, 1] for grid_sample
    # Input coords are in voxel space [0, shape-1]
    # grid_sample expects [-1, 1] where -1 is left edge, +1 is right edge

    # IMPORTANT: grid_sample uses (x, y, z) order while scipy uses (row, col, depth) order
    # For 2D: scipy (row, col) = (y, x) → grid_sample needs (x, y)
    # For 3D: scipy (depth, row, col) = (z, y, x) → grid_sample needs (x, y, z)
    # Solution: reverse the coordinate order
    coords_reversed = coords.flip(dims=[1])

    shape_tensor = torch.tensor(V.shape, device=device, dtype=coords.dtype)
    shape_reversed = shape_tensor.flip(dims=[0])
    coords_norm = 2.0 * coords_reversed / (shape_reversed - 1) - 1.0

    # Add batch and channel dimensions: (1, 1, D0, D1, ...)
    V_batch = V.unsqueeze(0).unsqueeze(0)

    # Reshape coords for grid_sample
    # grid_sample expects: (N, D_out0, D_out1, ..., ndim)
    # We want to sample N points, so output shape is (N, 1, 1, ..., 1)
    if ndim == 2:
        # For 2D: (N, 2) → (1, N, 1, 2)
        grid = coords_norm.view(1, -1, 1, 2)
    elif ndim == 3:
        # For 3D: (N, 3) → (1, N, 1, 1, 3)
        grid = coords_norm.view(1, -1, 1, 1, 3)

    # Sample using grid_sample
    sampled = F.grid_sample(
        V_batch,
        grid,
        mode=mode,
        padding_mode="border",  # Replicate edge values (matches scipy 'nearest')
        align_corners=True,  # Match scipy interpolation at edges
    )

    # Extract results: (1, 1, N, 1, ...) → (N,)
    # Use reshape(-1) instead of squeeze() to handle N=1 correctly
    return sampled.reshape(-1)


def estimate_gpu_memory_needed(V: np.ndarray, operation: str = "sobel") -> int:
    """
    Estimate GPU memory needed for operation.

    Parameters
    ----------
    V : np.ndarray
        Input volume.
    operation : str
        Operation type: 'sobel', 'interpolation'.

    Returns
    -------
    int
        Estimated memory needed in bytes.

    Notes
    -----
    Rough estimates:
    - Sobel: 5x volume size (input + ndim gradients + output)
    - Interpolation: 2x volume size (input + output)
    """
    base_size = V.nbytes

    if operation == "sobel":
        return base_size * 5
    elif operation == "interpolation":
        return base_size * 2
    else:
        return base_size * 5  # Conservative estimate


def check_gpu_memory(V: np.ndarray, device: str, operation: str = "sobel") -> bool:
    """
    Check if GPU has enough memory for operation.

    Parameters
    ----------
    V : np.ndarray
        Input volume.
    device : str
        Target device ('cuda', 'cuda:0', etc.).
    operation : str
        Operation type.

    Returns
    -------
    bool
        True if enough memory, False otherwise.

    Notes
    -----
    If memory check fails, operation should fallback to CPU automatically.
    This is a conservative estimate (uses 80% of available memory as threshold).
    """
    if not device.startswith("cuda"):
        return True  # MPS doesn't expose memory info, assume OK

    needed = estimate_gpu_memory_needed(V, operation)

    try:
        if ":" in device:
            device_id = int(device.split(":")[1])
        else:
            device_id = 0

        props = torch.cuda.get_device_properties(device_id)
        available = props.total_memory * 0.8  # Use 80% as safety margin

        if needed > available:
            warnings.warn(
                f"GPU memory insufficient ({needed / 1e9:.2f}GB needed, "
                f"{available / 1e9:.2f}GB available). Falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

        return True

    except Exception:
        # If we can't check memory, assume it's OK (will fail gracefully if OOM)
        return True
