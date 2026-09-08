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
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Estimated peak live full-volume tensors in the optimised _ssim_nd (used by
# the auto-tiling heuristic).  The actual peak is 6, we use 8 for safety.
_SSIM_PEAK_TENSOR_COUNT = 8

# Default tile size (per spatial axis) for tiled SSIM.
_SSIM_DEFAULT_TILE_SIZE = 128


def _gaussian_kernel_1d(
    window_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Create a 1-D Gaussian kernel (normalised to sum=1)."""
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    )
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _gaussian_kernel_nd(
    window_size: int, sigma: float, ndim: int, device: torch.device
) -> torch.Tensor:
    """Create an n-D Gaussian kernel via iterated outer products.

    Returns a tensor of shape ``(1, 1, *([window_size] * ndim))`` suitable
    for use as a convolution weight.
    """
    k = _gaussian_kernel_1d(window_size, sigma, device)
    kernel = k
    for _ in range(ndim - 1):
        kernel = kernel.unsqueeze(-1) * k
    # (1, 1, W, W, ...) for conv weight
    return kernel.unsqueeze(0).unsqueeze(0)


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

    Returns *True* if the estimated peak memory of the non-tiled path
    exceeds *memory_fraction* of free GPU memory.  Always *False* for
    CPU tensors.
    """
    free = _gpu_free_memory(device)
    if free is None:
        return False
    numel = math.prod(shape)
    peak_bytes = _SSIM_PEAK_TENSOR_COUNT * numel * 4  # float32
    return peak_bytes > memory_fraction * free


# ---------------------------------------------------------------------------
# SSIM (core — memory-optimised)
# ---------------------------------------------------------------------------


def _ssim_nd(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
) -> Tuple[float, int]:
    """Compute SSIM for a 2-D or 3-D tensor pair using convolution.

    Uses valid (no-padding) convolution so the border region — where the
    kernel would overlap with implicit zeros — is excluded from the mean.
    This matches the standard scikit-image implementation.

    Returns ``(ssim_sum, num_voxels)`` so callers can do voxel-weighted
    averaging across tiles.  Mean SSIM = ``ssim_sum / num_voxels``.

    Memory-optimised: peak ~6 live full-volume tensors (down from ~15 in
    the naive implementation) via in-place ops and explicit ``del``.
    """
    ndim = pred.ndim
    if ndim not in (2, 3):
        raise ValueError(f"_ssim_nd expects 2D or 3D tensors, got {ndim}D")

    device = pred.device
    sigma = 1.5
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_nd(window_size, sigma, ndim, device)

    # Add batch + channel dims: (B=1, C=1, *spatial)
    p = pred.unsqueeze(0).unsqueeze(0)
    t = target.unsqueeze(0).unsqueeze(0)

    conv_fn = F.conv2d if ndim == 2 else F.conv3d

    # --- Step 1: means (live: mu_p, mu_t = 2 tensors) ---
    mu_p = conv_fn(p, kernel, padding=0)
    mu_t = conv_fn(t, kernel, padding=0)

    # --- Step 2: squared / cross means (live: +3 = 5 tensors) ---
    mu_p_sq = mu_p.square()
    mu_t_sq = mu_t.square()
    mu_pt = mu_p * mu_t
    del mu_p, mu_t  # live: mu_p_sq, mu_t_sq, mu_pt = 3

    # --- Step 3: sigmas (live: peak 6 tensors) ---
    sigma_p_sq = conv_fn(p * p, kernel, padding=0)
    sigma_p_sq.sub_(mu_p_sq)

    sigma_t_sq = conv_fn(t * t, kernel, padding=0)
    sigma_t_sq.sub_(mu_t_sq)

    sigma_pt = conv_fn(p * t, kernel, padding=0)
    sigma_pt.sub_(mu_pt)

    del p, t  # free unsqueezed views

    # --- Step 4: numerator in-place (reuse mu_pt, sigma_pt) ---
    # numerator = (2*mu_pt + C1) * (2*sigma_pt + C2)
    mu_pt.mul_(2.0).add_(C1)  # mu_pt -> numerator_a
    sigma_pt.mul_(2.0).add_(C2)  # sigma_pt -> numerator_b
    mu_pt.mul_(sigma_pt)  # mu_pt -> full numerator
    del sigma_pt  # live: mu_p_sq, mu_t_sq, sigma_p_sq, sigma_t_sq, mu_pt = 5

    # --- Step 5: denominator in-place (reuse mu_p_sq, sigma_p_sq) ---
    # denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    mu_p_sq.add_(mu_t_sq).add_(C1)  # mu_p_sq -> denom_a
    del mu_t_sq  # live: 4
    sigma_p_sq.add_(sigma_t_sq).add_(C2)  # sigma_p_sq -> denom_b
    del sigma_t_sq  # live: 3
    mu_p_sq.mul_(sigma_p_sq)  # mu_p_sq -> full denominator
    del sigma_p_sq  # live: mu_pt(=num), mu_p_sq(=den) = 2

    # --- Step 6: SSIM map ---
    mu_pt.div_(mu_p_sq)  # mu_pt -> ssim_map
    del mu_p_sq  # live: 1

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
