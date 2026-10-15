"""Quality metrics for Gaussian splat reconstructions.

All functions operate on PyTorch tensors and stay on the input device,
avoiding unnecessary GPU-CPU transfers.  Only scalar results are moved
to CPU (via ``.item()``).
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Estimated peak live full-volume float32 tensors in _ssim_nd (used by the
# auto-tiling heuristic). Measured: ~7 (27.5 B/voxel RSS on a 65 Mvox CPU run);
# 8 leaves margin.
_SSIM_PEAK_TENSOR_COUNT = 8

# Default tile size (per spatial axis) for tiled SSIM.
_SSIM_DEFAULT_TILE_SIZE = 128

# Peak-temporary budget for untiled SSIM on devices whose free memory cannot be
# queried (CPU, MPS). 256 MiB = 8 Mvox at the 8-tensor estimate. Tiling costs
# little (65 Mvox on CPU: 2.3 s tiled vs 1.9 s untiled, 0.37 vs 1.8 GB).
_SSIM_HOST_PEAK_BUDGET_BYTES = 256 * 1024**2


def _gaussian_kernel_1d(
    window_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Create a 1-D Gaussian kernel (normalised to sum=1)."""
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    )
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _gpu_free_memory(device: torch.device) -> int | None:
    """Return free GPU memory in bytes, or *None* for non-CUDA devices."""
    if device.type != "cuda":
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return free
    except Exception:
        return None


def _should_tile_ssim(
    shape: Tuple[int, ...],
    device: torch.device,
    memory_fraction: float = 0.5,
) -> bool:
    """Decide whether SSIM should use the tiled path.

    On CUDA, returns *True* if the estimated peak memory of the non-tiled
    path exceeds *memory_fraction* of free GPU memory. On every other device
    (CPU, MPS) free memory is not queryable, so the untiled path is used only
    while its estimated peak (``_SSIM_PEAK_TENSOR_COUNT`` float32 volumes)
    stays under a fixed :data:`_SSIM_HOST_PEAK_BUDGET_BYTES`. Past that the
    tiled path runs; its result equals the untiled one up to float summation
    order. Untiled, a 65 Mvox volume needs ~1.8 GB of temporaries on top of
    its inputs; tiled, ~0.37 GB.
    """
    numel = math.prod(shape)
    peak_bytes = _SSIM_PEAK_TENSOR_COUNT * numel * 4  # float32
    free = _gpu_free_memory(device)
    if free is None:
        return peak_bytes > _SSIM_HOST_PEAK_BUDGET_BYTES
    return peak_bytes > memory_fraction * free


# ---------------------------------------------------------------------------
# SSIM (core — memory-optimised)
# ---------------------------------------------------------------------------


def _gaussian_filter_valid(
    x: torch.Tensor, weights: List[float], out_dtype: torch.dtype
) -> torch.Tensor:
    """Separable Gaussian filter with VALID borders along every axis.

    One 1-D pass per axis, each a sum of ``len(weights)`` shifted slices
    accumulated in place: exactly the n-D outer-product kernel up to float
    rounding, at ``window_size * ndim`` multiply-adds per voxel instead of
    ``window_size ** ndim``. No convolution op is used on purpose: CPU
    ``conv3d`` lowers an 11^3 kernel to im2col, whose column buffer costs
    ~1331 floats (5.3 KB) per output voxel — 37 GB for a 200^3 volume.
    Here each pass holds only its input and its output.
    """
    out = x
    for axis in range(x.ndim):
        length = out.shape[axis] - len(weights) + 1
        acc = out.narrow(axis, 0, length).to(out_dtype, copy=True)
        acc.mul_(weights[0])
        for k in range(1, len(weights)):
            acc.add_(out.narrow(axis, k, length), alpha=weights[k])
        out = acc
    return out


def _ssim_nd(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
) -> Tuple[float, int]:
    """Compute SSIM for a 2-D or 3-D tensor pair with a separable Gaussian.

    Uses valid (no-padding) filtering so the border region — where the
    kernel would overlap with implicit zeros — is excluded from the mean.
    This matches the standard scikit-image implementation.

    Returns ``(ssim_sum, num_voxels)`` so callers can do voxel-weighted
    averaging across tiles.  Mean SSIM = ``ssim_sum / num_voxels``.

    Memory: at most ~6 live full-volume float32 tensors (the five filtered
    moments, one being built, plus one product temporary) — independent of
    the device, see :func:`_gaussian_filter_valid`.
    """
    ndim = pred.ndim
    if ndim not in (2, 3):
        raise ValueError(f"_ssim_nd expects 2D or 3D tensors, got {ndim}D")

    sigma = 1.5
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    weights = _gaussian_kernel_1d(window_size, sigma, torch.device("cpu")).tolist()
    dtype = torch.promote_types(
        torch.promote_types(pred.dtype, target.dtype), torch.float32
    )

    def blur(x: torch.Tensor) -> torch.Tensor:
        return _gaussian_filter_valid(x, weights, dtype)

    # --- Step 1: means ---
    mu_p = blur(pred)
    mu_t = blur(target)

    # --- Step 2: squared / cross means ---
    mu_p_sq = mu_p.square()
    mu_t_sq = mu_t.square()
    mu_pt = mu_p.mul_(mu_t)  # mu_p -> mu_pt
    del mu_t

    # --- Step 3: sigmas (the product temporary is freed after each blur) ---
    sigma_p_sq = blur(pred.to(dtype).square()).sub_(mu_p_sq)
    sigma_t_sq = blur(target.to(dtype).square()).sub_(mu_t_sq)
    sigma_pt = blur(pred.to(dtype) * target).sub_(mu_pt)

    # --- Step 4: numerator in-place (reuse mu_pt, sigma_pt) ---
    # numerator = (2*mu_pt + C1) * (2*sigma_pt + C2)
    mu_pt.mul_(2.0).add_(C1)
    sigma_pt.mul_(2.0).add_(C2)
    mu_pt.mul_(sigma_pt)
    del sigma_pt

    # --- Step 5: denominator in-place (reuse mu_p_sq, sigma_p_sq) ---
    # denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    mu_p_sq.add_(mu_t_sq).add_(C1)
    del mu_t_sq
    sigma_p_sq.add_(sigma_t_sq).add_(C2)
    del sigma_t_sq
    mu_p_sq.mul_(sigma_p_sq)
    del sigma_p_sq

    # --- Step 6: SSIM map ---
    mu_pt.div_(mu_p_sq)
    del mu_p_sq

    ssim_sum = float(mu_pt.sum().item())
    num_voxels = mu_pt.numel()
    del mu_pt

    return ssim_sum, num_voxels


# ---------------------------------------------------------------------------
# SSIM (tiled)
# ---------------------------------------------------------------------------


def _compute_tile_slices(
    volume_shape: Tuple[int, ...],
    tile_size: int,
    overlap: int,
) -> List[Tuple[slice, ...]]:
    """Generate overlapping tile slices covering a volume.

    Each tile is *tile_size* voxels per axis.  Adjacent tiles overlap by
    *overlap* voxels so that valid-convolution outputs abut seamlessly.
    The last tile on each axis is extended to reach the volume boundary.
    """
    ndim = len(volume_shape)
    stride = tile_size - overlap

    axis_slices: List[List[slice]] = []
    for d in range(ndim):
        size = volume_shape[d]
        slices_d: List[slice] = []
        pos = 0
        while pos < size:
            end = min(pos + tile_size, size)
            # If the remaining strip is too small for a valid conv, merge
            # it into the previous tile.
            if end - pos <= overlap and slices_d:
                prev = slices_d[-1]
                slices_d[-1] = slice(prev.start, end)
            else:
                slices_d.append(slice(pos, end))
            pos += stride
        axis_slices.append(slices_d)

    return list(itertools.product(*axis_slices))


def _ssim_nd_tiled(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
    tile_size: int = _SSIM_DEFAULT_TILE_SIZE,
) -> Tuple[float, int]:
    """Compute SSIM by splitting the volume into overlapping tiles.

    Each tile overlaps its neighbours by ``window_size - 1`` voxels.
    Because ``_ssim_nd`` uses valid convolution (``padding=0``), the
    overlap region is exactly the border that gets trimmed — so each
    tile's output covers a non-overlapping region of the full SSIM map.
    The tiled result is therefore numerically identical to the non-tiled
    result (modulo float summation order).

    Returns ``(ssim_sum, num_voxels)`` — same contract as ``_ssim_nd``.
    """
    ndim = pred.ndim
    overlap = window_size - 1

    # If volume fits in a single tile, skip tiling overhead.
    if all(s <= tile_size for s in pred.shape):
        return _ssim_nd(pred, target, window_size, data_range)

    tile_slices = _compute_tile_slices(pred.shape, tile_size, overlap)

    total_sum = 0.0
    total_voxels = 0

    for slices in tile_slices:
        tile_p = pred[slices]
        tile_t = target[slices]

        # Skip tiles too small for the convolution kernel.
        if any(tile_p.shape[d] < window_size for d in range(ndim)):
            continue

        s, n = _ssim_nd(tile_p, tile_t, window_size, data_range)
        del tile_p, tile_t
        total_sum += s
        total_voxels += n

    return total_sum, total_voxels


# ---------------------------------------------------------------------------
# SSIM (public API)
# ---------------------------------------------------------------------------


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) on the input device.

    For 2-D and 3-D tensors a true n-D SSIM is computed via ``F.conv{2,3}d``.
    For higher-dimensional tensors the SSIM is averaged over all 3-D
    sub-volumes along the leading dimensions.

    Large volumes are automatically split into overlapping tiles to avoid
    GPU out-of-memory errors.  The tiling threshold is based on estimated
    peak memory vs. available GPU memory.

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and reference tensors (same shape, >= 2-D).
    window_size : int
        Side length of the Gaussian weighting window (must be odd).
    data_range : float, optional
        Dynamic range of the data.  If *None*, computed as
        ``target.max() - target.min()``.

    Returns
    -------
    float
        Mean SSIM in [−1, 1] (typically [0, 1] for non-negative data).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        return 1.0  # constant images are identical

    # Clamp window_size to the smallest spatial dimension (must be odd)
    min_dim = min(pred.shape[-min(pred.ndim, 3) :])
    if window_size > min_dim:
        window_size = min_dim if min_dim % 2 == 1 else max(min_dim - 1, 1)

    ndim = pred.ndim
    if ndim in (2, 3):
        use_tiled = _should_tile_ssim(pred.shape, pred.device)
        if use_tiled:
            s_sum, n_vox = _ssim_nd_tiled(pred, target, window_size, data_range)
        else:
            s_sum, n_vox = _ssim_nd(pred, target, window_size, data_range)
        return s_sum / n_vox if n_vox > 0 else 0.0

    # >3D: voxel-weighted average over all 3D sub-volumes along leading dims
    leading = pred.shape[:-3]
    total_sum = 0.0
    total_voxels = 0
    for idx in torch.cartesian_prod(*[torch.arange(s) for s in leading]):
        idx_tuple = tuple(idx.tolist()) if idx.ndim > 0 else (idx.item(),)
        sub_p = pred[idx_tuple]
        sub_t = target[idx_tuple]
        use_tiled = _should_tile_ssim(sub_p.shape, sub_p.device)
        if use_tiled:
            s, n = _ssim_nd_tiled(sub_p, sub_t, window_size, data_range)
        else:
            s, n = _ssim_nd(sub_p, sub_t, window_size, data_range)
        total_sum += s
        total_voxels += n
    return total_sum / total_voxels if total_voxels > 0 else 0.0


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 * log10(data_range² / MSE).

    Parameters
    ----------
    pred, target : torch.Tensor
        Same shape.
    data_range : float, optional
        If *None*, uses ``target.max() - target.min()``.

    Returns
    -------
    float
        PSNR in dB.  Returns ``float('inf')`` when MSE is zero.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0.0:
        return float("inf")

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        return float("inf")

    return 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()


# ---------------------------------------------------------------------------
# Foreground PSNR
# ---------------------------------------------------------------------------

#: Histogram resolution for :func:`otsu_threshold`. 256 is skimage's default, so
#: the two pick the SAME bin, not merely a neighbouring one — ``test_metrics.py``
#: pins that exactly, because a one-bin tolerance survives the index swap that is
#: the porting error worth guarding against.
_OTSU_BINS = 256


def otsu_threshold(target: torch.Tensor, bins: int = _OTSU_BINS) -> float:
    """Otsu's between-class-variance threshold, on the input device.

    Reimplemented here rather than delegating to ``skimage.filters`` because
    scikit-image lives in the ``demos`` extra. Calibration now shares this
    dependency-free implementation, so the *definition* of foreground does not
    depend on which extras happened to be installed.

    Follows scikit-image's formulation exactly (cumulative class weights and
    means over histogram bin *centres*, threshold taken at the argmax of the
    between-class variance) so the two are numerically interchangeable.

    Returns ``target.min()`` for a constant volume, which selects nothing under
    the strict ``>`` that :func:`compute_foreground_psnr` applies.
    """
    flat = target.reshape(-1).to(torch.float32)
    lo = float(flat.min().item())
    hi = float(flat.max().item())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return lo

    counts = torch.histc(flat, bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins + 1, device=counts.device)
    centres = (edges[:-1] + edges[1:]) / 2.0

    weight1 = torch.cumsum(counts, 0)
    weight2 = torch.flip(torch.cumsum(torch.flip(counts, [0]), 0), [0])
    # Bins are non-empty by construction only in aggregate; guard the divisions
    # so an empty leading/trailing class yields 0 variance rather than a NaN
    # that would poison the argmax.
    cw = counts * centres
    mean1 = torch.cumsum(cw, 0) / weight1.clamp(min=1e-12)
    mean2 = torch.flip(torch.cumsum(torch.flip(cw, [0]), 0), [0]) / weight2.clamp(
        min=1e-12
    )
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    variance = torch.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)
    return float(centres[int(torch.argmax(variance).item())].item())


def compute_foreground_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    threshold: float | None = None,
) -> Tuple[float, float, float]:
    """PSNR restricted to foreground voxels of *target*.

    Global PSNR is dominated by background on sparse volumes -- a light-sheet
    stack that is 97.8% empty scores well for reconstructing the emptiness --
    so the foreground number is the one that says whether the *signal* survived
    the fit. Reported alongside, never instead of, the global figure.

    The error is averaged over foreground voxels only, but ``data_range`` is
    taken from the **whole** target, matching
    :func:`luxar.gsplats.calibration.metrics.held_out_psnr_foreground` so the
    two are comparable. Using the foreground's own range instead would shrink
    the reference and silently inflate the result.

    Parameters
    ----------
    pred, target : torch.Tensor
        Same shape. Foreground is defined on *target*, never on *pred*: a fit
        that hallucinates signal must be scored against where the signal
        actually is.
    data_range : float, optional
        Defaults to ``target.max() - target.min()`` over the whole volume.
    threshold : float, optional
        Foreground is ``target > threshold``. Defaults to Otsu.

    Returns
    -------
    (psnr_db, threshold, fraction)
        ``fraction`` is the share of voxels counted as foreground -- report it,
        because a PSNR over 0.01% of the volume means something very different
        from one over 40%. ``psnr_db`` is ``nan`` when the foreground is empty.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if threshold is None:
        threshold = otsu_threshold(target)

    mask = target > threshold
    n_fg = int(mask.sum().item())
    fraction = n_fg / max(target.numel(), 1)
    if n_fg == 0:
        return float("nan"), float(threshold), fraction

    mse = torch.mean((pred[mask] - target[mask]) ** 2).item()
    if mse == 0.0:
        return float("inf"), float(threshold), fraction

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        return float("inf"), float(threshold), fraction

    psnr = 10.0 * math.log10(data_range**2 / mse)
    return psnr, float(threshold), fraction


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    ssim_window_size: int = 11,
) -> Dict[str, float]:
    """Compute a suite of quality metrics between predicted and target volumes.

    All heavy computation stays on the input device; only scalar results are
    returned.

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and reference tensors (same shape, >= 2-D).
    data_range : float, optional
        Dynamic range.  If *None*, computed from *target*.
    ssim_window_size : int
        SSIM window size.

    Returns
    -------
    dict
        Keys: ``mse``, ``psnr_db``, ``ssim``, ``rel_l2``, ``max_abs_error``,
        ``foreground_psnr_db``, ``foreground_threshold``,
        ``foreground_fraction``.  The foreground trio is the honest score on
        sparse data -- see :func:`compute_foreground_psnr`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    diff = pred - target

    mse = torch.mean(diff**2).item()
    max_abs_error = torch.max(torch.abs(diff)).item()
    target_norm = torch.linalg.norm(target.reshape(-1)).item()
    rel_l2 = float(torch.linalg.norm(diff.reshape(-1)).item() / (target_norm + 1e-12))

    if data_range is None:
        data_range = float((target.max() - target.min()).item())

    psnr_db = compute_psnr(pred, target, data_range=data_range)
    ssim = compute_ssim(
        pred, target, window_size=ssim_window_size, data_range=data_range
    )
    fg_psnr_db, fg_threshold, fg_fraction = compute_foreground_psnr(
        pred, target, data_range=data_range
    )

    return {
        "mse": mse,
        "psnr_db": psnr_db,
        "ssim": ssim,
        "rel_l2": rel_l2,
        "max_abs_error": max_abs_error,
        "foreground_psnr_db": fg_psnr_db,
        "foreground_threshold": fg_threshold,
        "foreground_fraction": fg_fraction,
    }
