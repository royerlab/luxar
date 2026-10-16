"""Quality metrics for Gaussian splat reconstructions.

All functions operate on PyTorch tensors and stay on the input device,
avoiding unnecessary GPU-CPU transfers.  Only scalar results are moved
to CPU (via ``.item()``).
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterator, List, Tuple

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


def _ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
) -> torch.Tensor:
    """The VALID-region SSIM map of a 2-D or 3-D tensor pair (separable Gaussian).

    Valid (no-padding) filtering excludes the border where the kernel would
    overlap implicit zeros, matching scikit-image. The map is therefore smaller
    than the inputs by ``window_size - 1`` along every axis: map voxel ``i``
    describes input voxel ``i + window_size // 2``.

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
    return mu_pt.div_(mu_p_sq)


#: ``(ssim_sum, num_voxels, foreground_ssim_sum, foreground_voxels)``.
_SSIMSums = Tuple[float, int, float, int]


def _ssim_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
    fg_field: torch.Tensor | None = None,
    fg_threshold: float = 0.0,
) -> _SSIMSums:
    """SSIM-map sums over every valid voxel and over the foreground ones.

    Foreground is ``fg_field > fg_threshold``, evaluated on the input voxel each
    valid map voxel describes (the field is cropped by ``window_size // 2``).
    The mask is built per call, so under :func:`_ssim_sums_tiled` it is only
    ever tile-sized. Without a field the foreground sums are ``(0.0, 0)``.
    """
    ssim_map = _ssim_map(pred, target, window_size, data_range)
    ssim_sum = float(ssim_map.sum().item())
    num_voxels = ssim_map.numel()
    fg_sum, fg_voxels = 0.0, 0
    if fg_field is not None and num_voxels > 0:
        half = window_size // 2
        crop = tuple(slice(half, half + n) for n in ssim_map.shape)
        mask = fg_field[crop] > fg_threshold
        fg_voxels = int(mask.sum().item())
        if fg_voxels:
            fg_sum = float(ssim_map.mul_(mask).sum().item())
        del mask
    del ssim_map
    return ssim_sum, num_voxels, fg_sum, fg_voxels


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
    """
    ssim_sum, num_voxels, _, _ = _ssim_sums(pred, target, window_size, data_range)
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


def _ssim_sums_tiled(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
    tile_size: int = _SSIM_DEFAULT_TILE_SIZE,
    fg_field: torch.Tensor | None = None,
    fg_threshold: float = 0.0,
) -> _SSIMSums:
    """:func:`_ssim_sums` over overlapping tiles — same result, tile-sized memory.

    Each tile overlaps its neighbours by ``window_size - 1`` voxels. Because
    the map uses valid convolution, the overlap is exactly the border each tile
    trims, so every tile's map covers a non-overlapping region of the full map
    and the sums equal the untiled ones up to float summation order.
    """
    ndim = pred.ndim
    overlap = window_size - 1

    # If volume fits in a single tile, skip tiling overhead.
    if all(s <= tile_size for s in pred.shape):
        return _ssim_sums(pred, target, window_size, data_range, fg_field, fg_threshold)

    totals = [0.0, 0, 0.0, 0]
    for slices in _compute_tile_slices(pred.shape, tile_size, overlap):
        tile_p = pred[slices]
        # Skip tiles too small for the convolution kernel.
        if any(tile_p.shape[d] < window_size for d in range(ndim)):
            continue
        tile_field = fg_field[slices] if fg_field is not None else None
        sums = _ssim_sums(
            tile_p, target[slices], window_size, data_range, tile_field, fg_threshold
        )
        for index, value in enumerate(sums):
            totals[index] += value
    return float(totals[0]), int(totals[1]), float(totals[2]), int(totals[3])


def _ssim_nd_tiled(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
    tile_size: int = _SSIM_DEFAULT_TILE_SIZE,
) -> Tuple[float, int]:
    """Compute SSIM by splitting the volume into overlapping tiles.

    The tiled result is numerically identical to the non-tiled result (modulo
    float summation order); see :func:`_ssim_sums_tiled`.

    Returns ``(ssim_sum, num_voxels)`` — same contract as ``_ssim_nd``.
    """
    ssim_sum, num_voxels, _, _ = _ssim_sums_tiled(
        pred, target, window_size, data_range, tile_size
    )
    return ssim_sum, num_voxels


# ---------------------------------------------------------------------------
# SSIM (public API)
# ---------------------------------------------------------------------------


def _ssim_volume_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
    fg_field: torch.Tensor | None,
    fg_threshold: float,
) -> _SSIMSums:
    """Tiled-or-not SSIM sums for one 2-D/3-D volume (the auto-tiling decision)."""
    if _should_tile_ssim(pred.shape, pred.device):
        return _ssim_sums_tiled(
            pred,
            target,
            window_size,
            data_range,
            fg_field=fg_field,
            fg_threshold=fg_threshold,
        )
    return _ssim_sums(pred, target, window_size, data_range, fg_field, fg_threshold)


def compute_ssim_with_foreground(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
    fg_field: torch.Tensor | None = None,
    fg_threshold: float = 0.0,
) -> Tuple[float, float]:
    """``(ssim, foreground_ssim)``: mean SSIM over all valid voxels, and over
    the foreground ones (``fg_field > fg_threshold``).

    Same SSIM as :func:`compute_ssim` — one SSIM map, two averages — so on a
    sparse volume the foreground figure is not diluted by thousands of trivially
    matching empty voxels. ``foreground_ssim`` is ``nan`` without a field or
    when the foreground has no valid voxel.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if fg_field is not None and fg_field.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: foreground field {fg_field.shape} vs target {target.shape}"
        )

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        has_fg = fg_field is not None and any(
            bool((chunk > fg_threshold).any()) for chunk in _flat_chunks(fg_field)
        )
        return 1.0, 1.0 if has_fg else float("nan")

    # Clamp window_size to the smallest spatial dimension (must be odd)
    min_dim = min(pred.shape[-min(pred.ndim, 3) :])
    if window_size > min_dim:
        window_size = min_dim if min_dim % 2 == 1 else max(min_dim - 1, 1)

    if pred.ndim in (2, 3):
        totals = _ssim_volume_sums(
            pred, target, window_size, data_range, fg_field, fg_threshold
        )
    else:
        # >3D: voxel-weighted average over all 3D sub-volumes along leading dims
        acc = [0.0, 0, 0.0, 0]
        leading = pred.shape[:-3]
        for idx in torch.cartesian_prod(*[torch.arange(s) for s in leading]):
            idx_tuple = tuple(idx.tolist()) if idx.ndim > 0 else (idx.item(),)
            sub_field = fg_field[idx_tuple] if fg_field is not None else None
            sums = _ssim_volume_sums(
                pred[idx_tuple],
                target[idx_tuple],
                window_size,
                data_range,
                sub_field,
                fg_threshold,
            )
            for index, value in enumerate(sums):
                acc[index] += value
        totals = (float(acc[0]), int(acc[1]), float(acc[2]), int(acc[3]))

    ssim_sum, n_vox, fg_sum, fg_vox = totals
    ssim = ssim_sum / n_vox if n_vox > 0 else 0.0
    fg_ssim = fg_sum / fg_vox if fg_vox > 0 else float("nan")
    return ssim, fg_ssim


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) on the input device.

    For 2-D and 3-D tensors a true n-D SSIM uses separable Gaussian filtering.
    For higher-dimensional tensors the SSIM is averaged over all 3-D
    sub-volumes along the leading dimensions.

    Large volumes are automatically split into overlapping tiles. The tiling
    threshold compares estimated peak memory with a fraction of free CUDA
    memory when available, or with a fixed budget on CPU and MPS (and when
    CUDA free memory cannot be queried).

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
    ssim, _ = compute_ssim_with_foreground(
        pred, target, window_size=window_size, data_range=data_range
    )
    return ssim


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


#: Sample cap for :func:`otsu_threshold`'s histogram. Above this many voxels a
#: fixed-seed uniform random subsample feeds the histogram, so the cost is
#: bounded on gigavoxel stacks. Random rather than strided: a stride sharing a
#: large factor with the fastest axis aliases onto a few columns of every row
#: (a structured volume with stride == X read 0.0023 for an exact 0.199). The
#: bin EDGES still come from the full volume's min/max, so a subsampled
#: threshold stays on the exact histogram's bin grid.
_OTSU_MAX_SAMPLES = 5_000_000


def otsu_sample_indices(n: int, cap: int = _OTSU_MAX_SAMPLES) -> torch.Tensor:
    """Deterministic uniform random sample of ``cap`` flat indices in ``[0, n)``.

    Fixed seed, so a threshold is reproducible run to run; drawn with
    replacement (no ``O(n)`` permutation). Shared with
    ``calibration.content._otsu_threshold`` so both subsample alike.
    """
    gen = torch.Generator().manual_seed(0x0750)
    return torch.randint(0, n, (min(n, cap),), generator=gen, dtype=torch.int64)


def _otsu_bin_counts(
    flat: torch.Tensor, lo: float, hi: float, bins: int
) -> torch.Tensor:
    """``torch.histc(flat, bins, lo, hi)`` without its per-element temporaries.

    ``torch.histc`` on MPS materialises roughly ``numel x bins x 4`` bytes (an
    8 Mvox volume asked for 9.4 GB; above ~16 Mvox it fails outright with a
    "64-bit indexing" error), so the counting is done with ``bincount`` on the
    CPU instead. The bin index uses histc's own formula (``(x - lo) * bins /
    (hi - lo)``, with ``x == hi`` folded into the last bin) so the counts match
    up to float rounding at bin edges.
    """
    host = flat.detach().to("cpu", torch.float32)
    idx = ((host - lo) * (bins / (hi - lo))).to(torch.int64).clamp_(0, bins - 1)
    return torch.bincount(idx, minlength=bins).to(torch.float64)


def otsu_threshold(
    target: torch.Tensor, bins: int = _OTSU_BINS, upper: float | None = None
) -> float:
    """Otsu's between-class-variance threshold.

    Reimplemented here rather than delegating to ``skimage.filters`` because
    scikit-image lives in the ``demos`` extra. Calibration now shares this
    dependency-free implementation, so the *definition* of foreground does not
    depend on which extras happened to be installed.

    Follows scikit-image's formulation exactly (cumulative class weights and
    means over histogram bin *centres*, threshold taken at the argmax of the
    between-class variance) so the two are numerically interchangeable.

    The min/max reductions run on the input device; the histogram is counted
    on the CPU over at most :data:`_OTSU_MAX_SAMPLES` random samples, so the
    memory cost is bounded whatever the device or volume size (see
    :func:`_otsu_bin_counts`).

    Returns ``target.min()`` for a constant volume, which selects nothing under
    the strict ``>`` that :func:`compute_foreground_psnr` applies.

    ``upper`` caps the histogram range (values above it are left out of the
    histogram — they are foreground under any threshold below it). Pass a robust
    peak (:func:`robust_peak`) so one hot pixel cannot stretch the 256 bins over
    a range where all real signal falls into the first few.
    """
    flat = target.reshape(-1)
    lo = float(flat.min().item())
    hi = float(flat.max().item())
    capped = False
    if upper is not None and lo < upper < hi:
        hi, capped = float(upper), True
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return lo

    if flat.numel() > _OTSU_MAX_SAMPLES:
        flat = flat[otsu_sample_indices(flat.numel()).to(flat.device)]
    if capped:
        # Leave the values above the cap OUT of the histogram (the counter would
        # otherwise fold them into the last bin). After the subsample, so the
        # mask is at most sample-sized.
        flat = flat[flat <= hi]
    counts = _otsu_bin_counts(flat, lo, hi, bins)
    edges = torch.linspace(lo, hi, bins + 1, dtype=torch.float32).to(torch.float64)
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
# Robust peak + streamed error sums
# ---------------------------------------------------------------------------

#: Quantile of the reference that sets the ROBUST PSNR peak. One hot pixel (a
#: cosmic ray, a saturated bead, a stitching artefact) sets ``max - min`` and so
#: the classical full-range PSNR, by itself; at 99.99% it takes more than one
#: voxel in ten thousand to move the peak.
ROBUST_PEAK_QUANTILE = 0.9999

#: Voxels per streamed chunk (4 Mvox = 16 MiB of float32): every full-volume
#: reduction below walks the volume in views of this size, so its temporaries
#: are chunk-sized whatever the volume's.
_STREAM_CHUNK_VOXELS = 1 << 22


def _flat_chunks(
    t: torch.Tensor, chunk: int = _STREAM_CHUNK_VOXELS
) -> Iterator[torch.Tensor]:
    """Consecutive views of ``t`` flattened (a copy only if ``t`` is non-contiguous)."""
    flat = t.reshape(-1)
    for start in range(0, flat.numel(), chunk):
        yield flat[start : start + chunk]


def robust_peak(target: torch.Tensor, quantile: float = ROBUST_PEAK_QUANTILE) -> float:
    """The ``quantile`` of ``target`` (nearest rank), streamed and exact.

    Keeps a running top-``k`` over chunk-sized views, where ``k`` is the number of
    voxels at or above the requested rank (``N - ceil(q N) + 1``; 6.7 k for a
    256x512x512 volume at 99.99%), so nothing full-volume is allocated and the
    answer is the exact order statistic — not a histogram bin edge. ``torch.quantile``
    is not an option: it refuses inputs above 16 M elements. MPS chunks are
    reduced on the CPU (its ``topk`` caps ``k``).
    """
    n = target.numel()
    if n == 0:
        return float("nan")
    k_top = max(1, min(n, n - math.ceil(quantile * n) + 1))
    best: torch.Tensor | None = None
    for chunk in _flat_chunks(target):
        values = chunk.detach()
        if values.device.type == "mps":
            values = values.cpu()
        values = values.to(torch.float32)
        top = torch.topk(values, min(k_top, values.numel())).values
        merged = top if best is None else torch.cat((best, top))
        best = torch.topk(merged, min(k_top, merged.numel())).values
    assert best is not None
    return float(best.min().item())


def _streamed_extrema(target: torch.Tensor) -> Tuple[float, float]:
    """``(min, max)`` of ``target`` without a full-volume temporary."""
    lo, hi = math.inf, -math.inf
    for chunk in _flat_chunks(target):
        lo = min(lo, float(chunk.min().item()))
        hi = max(hi, float(chunk.max().item()))
    return lo, hi


def _error_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    fg_field: torch.Tensor,
    fg_threshold: float,
) -> Dict[str, float]:
    """Squared-error / norm / foreground sums in one chunked pass.

    Replaces the whole-volume ``pred - target``, ``diff ** 2``, ``abs(diff)`` and
    boolean-mask temporaries (each as large as the volume) with chunk-sized ones.
    Partial sums are float32 per chunk, accumulated in Python floats.
    """
    sse = max_abs = target_sq = fg_sse = 0.0
    fg_voxels = 0
    for p, t, f in zip(
        _flat_chunks(pred), _flat_chunks(target), _flat_chunks(fg_field), strict=True
    ):
        if not t.is_floating_point():
            t = t.to(torch.float32)
        if not p.is_floating_point():
            p = p.to(torch.float32)
        diff = p - t
        max_abs = max(max_abs, float(diff.abs().max().item()))
        diff.square_()
        sse += float(diff.sum().item())
        mask = f > fg_threshold
        fg_voxels += int(mask.sum().item())
        fg_sse += float(diff.mul_(mask).sum().item())
        target_sq += float(t.square().sum().item())
        del diff, mask
    return {
        "sse": sse,
        "max_abs": max_abs,
        "target_sq": target_sq,
        "fg_sse": fg_sse,
        "fg_voxels": float(fg_voxels),
    }


def _psnr_from(data_range: float, mse: float) -> float:
    """``10 log10(range^2 / mse)``; ``inf`` for a zero error or a zero range."""
    if mse == 0.0 or data_range == 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range**2 / mse)


# ---------------------------------------------------------------------------
# Foreground definition
# ---------------------------------------------------------------------------


def _tent_blur(t: torch.Tensor) -> torch.Tensor:
    """Separable ``[0.25, 0.5, 0.25]`` blur with replicated edges, as float32.

    The torch twin of ``seeds.utils.soft_blur_nd`` (scipy ``convolve1d``,
    ``mode="nearest"``), so the foreground can be defined on the scoring device.
    Never modifies ``t``; peaks at two volumes besides it.
    """
    out = t.to(torch.float32)
    for axis in range(t.ndim):
        src = out
        out = src * 0.5
        n = src.shape[axis]
        out.narrow(axis, 1, n - 1).add_(src.narrow(axis, 0, n - 1), alpha=0.25)
        out.narrow(axis, 0, n - 1).add_(src.narrow(axis, 1, n - 1), alpha=0.25)
        out.narrow(axis, 0, 1).add_(src.narrow(axis, 0, 1), alpha=0.25)
        out.narrow(axis, n - 1, 1).add_(src.narrow(axis, n - 1, 1), alpha=0.25)
        del src
    return out


def default_foreground(
    target: torch.Tensor, peak: float | None = None
) -> Tuple[torch.Tensor, float]:
    """The shared foreground definition: ``(field, threshold)``, fg = field > thr.

    One light tent blur (the smoothed-Otsu field calibration scores with), then
    Otsu with the histogram capped at the target's robust peak (``peak``,
    default :func:`robust_peak`). The cap is taken on the UNBLURRED target: the
    blur spreads a hot pixel over its neighbours, enough to own the field's own
    top quantile. Uncapped, one hot pixel stretches the 256 bins until every real
    structure sits in the first few and below the threshold — a 0.00% foreground.
    """
    if peak is None:
        peak = robust_peak(target)
    field = _tent_blur(target)
    return field, otsu_threshold(field, upper=peak)


def _resolve_foreground(
    target: torch.Tensor,
    field: torch.Tensor | None,
    threshold: float | None,
    peak: float,
) -> Tuple[torch.Tensor, float]:
    if field is None:
        if threshold is not None:
            return target, float(threshold)
        return default_foreground(target, peak)
    if field.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: foreground field {field.shape} vs target {target.shape}"
        )
    if threshold is None:
        threshold = otsu_threshold(field)
    return field, float(threshold)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    ssim_window_size: int = 11,
    *,
    foreground_field: torch.Tensor | None = None,
    foreground_threshold: float | None = None,
    robust_quantile: float = ROBUST_PEAK_QUANTILE,
) -> Dict[str, float]:
    """Compute a suite of quality metrics between predicted and target volumes.

    All heavy computation stays on the input device; only scalar results are
    returned. Full-volume reductions are streamed (chunk-sized temporaries) and
    SSIM tiles itself, so the peak beyond the two inputs is bounded.

    PSNR comes in two peaks. ``psnr_robust_db`` — the HEADLINE — uses the
    robust range ``p_q(target) - min(target)`` (``q`` = ``robust_quantile``,
    99.99% by default), which one hot pixel cannot move. ``psnr_db`` keeps the
    classical full range ``max - min`` (or ``data_range`` when given): it is the
    figure stamped into every store so far, so it is kept for comparability and
    reported beside the headline, labelled. A robust range of zero (more than
    99.99% of voxels at the minimum) falls back to the full range. The same pair
    exists for the foreground PSNR. On a fit's basis the target minimum is the
    fit's floor (the background-relative reference is clipped at 0), so the
    robust peak is "reference p99.99 minus the floor".

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and reference tensors (same shape, >= 2-D).
    data_range : float, optional
        Full dynamic range for ``psnr_db`` and SSIM. If *None*, ``max - min`` of
        *target*.
    ssim_window_size : int
        SSIM window size.
    foreground_field, foreground_threshold : optional
        Foreground is ``foreground_field > foreground_threshold``. With neither,
        the ONE shared definition applies (:func:`default_foreground`): a tent-
        blurred target thresholded by Otsu over a histogram capped at the robust
        peak — the fit summary and ``gsplat compare`` both use it, so a stored
        foreground score and a recomputed one agree. A threshold alone applies
        to the target itself (``compare --fg-threshold``); a field is a mask
        volume or any other caller-defined field.
    robust_quantile : float
        Quantile setting the robust peak.

    Returns
    -------
    dict
        ``mse``, ``psnr_db`` (full range), ``psnr_robust_db`` (headline),
        ``ssim``, ``foreground_ssim``, ``rel_l2``, ``max_abs_error``,
        ``foreground_psnr_db`` (full range), ``foreground_psnr_robust_db``,
        ``foreground_threshold``, ``foreground_fraction``, ``data_range`` and
        ``robust_data_range`` (the two peaks). SSIM uses the full range.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    peak = robust_peak(target, robust_quantile)
    field, threshold = _resolve_foreground(
        target, foreground_field, foreground_threshold, peak
    )

    sums = _error_sums(pred, target, field, threshold)
    n = max(target.numel(), 1)
    mse = sums["sse"] / n
    t_min, t_max = _streamed_extrema(target)
    full_range = float(data_range) if data_range is not None else t_max - t_min
    robust_range = peak - t_min
    if not robust_range > 0.0:
        robust_range = full_range

    fg_voxels = int(sums["fg_voxels"])
    if fg_voxels:
        fg_mse = sums["fg_sse"] / fg_voxels
        fg_psnr = _psnr_from(full_range, fg_mse)
        fg_psnr_robust = _psnr_from(robust_range, fg_mse)
    else:
        fg_psnr = fg_psnr_robust = float("nan")

    ssim, fg_ssim = compute_ssim_with_foreground(
        pred,
        target,
        window_size=ssim_window_size,
        data_range=full_range,
        fg_field=field,
        fg_threshold=threshold,
    )
    if not fg_voxels:
        fg_ssim = float("nan")
    rel_l2 = float(math.sqrt(sums["sse"]) / (math.sqrt(sums["target_sq"]) + 1e-12))

    return {
        "mse": mse,
        "psnr_db": _psnr_from(full_range, mse),
        "psnr_robust_db": _psnr_from(robust_range, mse),
        "ssim": ssim,
        "foreground_ssim": fg_ssim,
        "rel_l2": rel_l2,
        "max_abs_error": sums["max_abs"],
        "foreground_psnr_db": fg_psnr,
        "foreground_psnr_robust_db": fg_psnr_robust,
        "foreground_threshold": threshold,
        "foreground_fraction": fg_voxels / n,
        "data_range": full_range,
        "robust_data_range": robust_range,
    }
