"""Export Luxar Gaussian splats to the classical INRIA 3DGS PLY format.

The inverse of :mod:`.classical_splats`: a :class:`GSplatData` becomes a
``point_cloud.ply`` that classical viewers (SuperSplat, PlayCanvas, gsplat.js,
antimatter15, …) load directly — packed Cholesky factors are eigendecomposed
back to log-scales + rotation quaternions, amplitudes map to opacity logits,
and per-splat (or colormap-baked) RGB becomes the SH DC band.

Semantics notes
---------------
- Luxar ``amplitudes`` are unbounded emission weights while classical opacity
  lives in (0, 1): the default ``normalized`` policy rescales robustly (99.5th
  percentile → 1). A per-splat color ALPHA channel (RGBA colors) is classical
  opacity itself and multiplies into both data-driven policies verbatim — data
  that came from :func:`~.classical_splats.import_gsplats` (amplitudes = 1,
  opacity in alpha) round-trips losslessly under the default policy.
- If the data was imported, the orientation applied at import time (recorded
  in ``stats["interop"]``) is inverted by default so import → export is an
  identity in the source frame.
- INRIA PLY is strictly 3D: nD data must be sliced to 3D first (``timepoint``
  / ``slice_dim`` + ``slice_index``); 1D/2D data is embedded with a tiny
  isotropic sigma.

NumPy + stdlib only — no torch, no external writers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

from luxar.gsplats.interop._color import linear_to_srgb
from luxar.gsplats.interop.classical_splats import SH_C0, rotmat_to_quat

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.gsplat_data import GSplatData

__all__ = ["gsplat_data_to_inria_ply", "export_inria_ply"]

#: Opacity logits are clamped so sigmoid stays strictly inside (0, 1)
#: (the SuperSplat/PlayCanvas decoders use the same ±40 guard).
_LOGIT_CLAMP = 40.0

OpacityPolicy = Literal["normalized", "amplitude", "constant"]
ColorSource = Literal["auto", "colors", "colormap", "white"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.clip(-np.log(1.0 / p - 1.0), -_LOGIT_CLAMP, _LOGIT_CLAMP)


def _normalized_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Robustly rescale unbounded amplitudes into [0, 1] (99.5th pct → 1)."""
    hi = float(np.percentile(amplitudes, 99.5))
    if hi <= 0:
        hi = float(amplitudes.max()) or 1.0
    return np.clip(amplitudes / hi, 0.0, 1.0)


def _opacity_logits(
    amplitudes: np.ndarray,
    alpha: Optional[np.ndarray],
    policy: OpacityPolicy,
    constant_opacity: float,
) -> np.ndarray:
    """Opacity logits from amplitudes and the optional color alpha channel.

    A per-splat alpha (RGBA colors) IS classical opacity, so it multiplies
    into both data-driven policies verbatim — never rescaled. Data imported
    from a classical file (amplitudes = 1, opacity in alpha) round-trips
    bit-faithfully under both ``normalized`` (ones normalize to ones) and
    ``amplitude``.
    """
    if alpha is None:
        alpha = np.ones_like(np.asarray(amplitudes, dtype=np.float64))
    if policy == "normalized":
        return _logit(_normalized_amplitudes(amplitudes) * alpha)
    if policy == "amplitude":
        return _logit(amplitudes * alpha)
    if policy == "constant":
        return np.full(
            amplitudes.shape, _logit(np.asarray(constant_opacity)), dtype=np.float64
        )
    raise ValueError(
        f"Unknown opacity policy {policy!r}; expected normalized|amplitude|constant"
    )


def _resolve_colors(
    amplitudes: np.ndarray,
    colors: Optional[np.ndarray],
    color_source: ColorSource,
    colormap: Optional[str],
) -> np.ndarray:
    """Per-splat RGB in [0, 1] following the auto > colors > colormap > white chain.

    Per-splat ``colors`` are Luxar's *linear-light* store, so they are converted
    back to display-referred sRGB here — the inverse of the import boundary's
    sRGB → linear (see :mod:`._color`) — so the exported DC round-trips and
    reference viewers show the original colors. Colormap / white sources already
    produce display values and are passed through unchanged.
    """
    if color_source == "colors" and colors is None:
        raise ValueError("color_source='colors' but the dataset has no colors array")
    if color_source == "colors" or (color_source == "auto" and colors is not None):
        return linear_to_srgb(np.asarray(colors, dtype=np.float64)).astype(np.float64)

    use_colormap = color_source == "colormap" or (
        color_source == "auto" and colormap is not None
    )
    if use_colormap:
        if colormap is None:
            raise ValueError("color_source='colormap' requires a colormap name")
        from luxar.colormaps import resolve_colormap

        lut = resolve_colormap(colormap)  # (256, 3) uint8
        idx = np.clip(
            np.round(_normalized_amplitudes(amplitudes) * 255), 0, 255
        ).astype(np.intp)
        return lut[idx].astype(np.float64) / 255.0

    if color_source in ("auto", "white"):
        return np.ones((amplitudes.shape[0], 3), dtype=np.float64)
    raise ValueError(
        f"Unknown color source {color_source!r}; expected auto|colors|colormap|white"
    )


def _undo_import_orientation(
    centers: np.ndarray, sigma: np.ndarray, data: "GSplatData"
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the orientation matrix recorded at import time (if any).

    Import applied ``p' = p·Mᵀ`` and ``Σ' = M·Σ·Mᵀ`` with an orthogonal ``M``
    (rotation and/or axis flips), so the inverse is ``p = p'·M``, ``Σ = Mᵀ·Σ'·M``.
    """
    interop = data.stats.get("interop") if isinstance(data.stats, dict) else None
    if not interop or "orientation_matrix" not in interop:
        return centers, sigma
    M = np.asarray(interop["orientation_matrix"], dtype=np.float64)
    if M.shape != (3, 3):
        return centers, sigma
    return centers @ M, M.T @ sigma @ M


def _scales_and_quats(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose covariances into (log-scales, w-first quaternions)."""
    eigvals, eigvecs = np.linalg.eigh(sigma)  # ascending, orthonormal columns
    floor = np.maximum(eigvals[:, -1:] * 1e-12, 1e-24)
    scales = np.sqrt(np.maximum(eigvals, floor))
    # eigh may return an improper basis (det = -1): flip one column to make a
    # proper rotation, which leaves Σ = R·diag(s²)·Rᵀ unchanged.
    det = np.linalg.det(eigvecs)
    eigvecs[det < 0, :, 0] *= -1.0
    quats = rotmat_to_quat(eigvecs)
    return np.log(scales), quats


def _select_3d(
    data: "GSplatData",
    timepoint: Optional[int],
    slice_dim: Optional[int],
    slice_index: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Reduce nD splat arrays to 3D (centers, Σ, amplitudes, colors).

    - d == 3: pass through.
    - d < 3: embed with a tiny isotropic sigma on the missing axes.
    - d > 3: select the splats whose coordinate along one dimension rounds to
      the requested index (``timepoint`` is shorthand for the LAST dimension —
      where ``combine_as_new_dimension`` / ``gsplat merge --as-dimension``
      stack time), then drop that dimension. Only a single sliced dimension is
      supported — pre-slice with ``gsplat slice`` for higher-dimensional data.
    """
    from luxar.gsplats.utils.trils import unpack_tril

    d = data.ndim
    centers = np.asarray(data.centers, dtype=np.float64)
    L = unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float64), d)
    sigma = L @ L.transpose(0, 2, 1)
    amplitudes = np.asarray(data.amplitudes, dtype=np.float64)
    colors = None if data.colors is None else np.asarray(data.colors, dtype=np.float64)

    if timepoint is not None:
        if slice_dim is not None or slice_index is not None:
            raise ValueError("Pass either timepoint or slice_dim/slice_index, not both")
        slice_dim, slice_index = d - 1, timepoint

    if d > 3:
        if slice_dim is None or slice_index is None:
            raise ValueError(
                f"Dataset is {d}D but INRIA PLY is strictly 3D — select a 3D "
                "subset with timepoint=N (slices the last, stacked dimension) "
                "or slice_dim=D, slice_index=I"
            )
        if d != 4:
            raise ValueError(
                f"Dataset is {d}D; only one dimension can be sliced away here. "
                "Reduce it to 4D/3D first (e.g. `luxar gsplat slice`)."
            )
        if not 0 <= slice_dim < d:
            raise ValueError(f"slice_dim {slice_dim} out of range for {d}D data")
        mask = np.round(centers[:, slice_dim]) == slice_index
        if not np.any(mask):
            raise ValueError(
                f"No splats at slice_dim={slice_dim}, slice_index={slice_index}"
            )
        keep = [i for i in range(d) if i != slice_dim]
        centers = centers[mask][:, keep]
        sigma = sigma[mask][:, keep, :][:, :, keep]
        amplitudes = amplitudes[mask]
        colors = colors[mask] if colors is not None else None
    elif slice_dim is not None or slice_index is not None:
        raise ValueError(f"slice_dim/slice_index only apply to >3D data (got {d}D)")

    if centers.shape[1] < 3:
        pad = 3 - centers.shape[1]
        n = centers.shape[0]
        centers = np.concatenate([centers, np.zeros((n, pad))], axis=1)
        embedded = np.full((n, 3, 3), 0.0)
        embedded[:, : 3 - pad, : 3 - pad] = sigma
        for k in range(3 - pad, 3):
            embedded[:, k, k] = 1e-14  # tiny isotropic sigma on padded axes
        sigma = embedded

    # Guarantee symmetric input for eigh after all the slicing/embedding.
    sigma = (sigma + sigma.transpose(0, 2, 1)) / 2.0
    return centers, sigma, amplitudes, colors


def gsplat_data_to_inria_ply(
    data: "GSplatData",
    *,
    opacity_policy: OpacityPolicy = "normalized",
    constant_opacity: float = 1.0,
    color_source: ColorSource = "auto",
    colormap: Optional[str] = None,
    sh_degree: int = 0,
    undo_orientation: bool = True,
    timepoint: Optional[int] = None,
    slice_dim: Optional[int] = None,
    slice_index: Optional[int] = None,
) -> bytes:
    """Serialize a :class:`GSplatData` as an INRIA 3DGS ``point_cloud.ply``.

    Args:
        data: Source splats (any matrix shape; the finest content is exported).
        opacity_policy: ``normalized`` (robust rescale of amplitudes into
            (0, 1), the honest default for unbounded emission weights),
            ``amplitude`` (clip raw values), or ``constant`` (fixed
            ``constant_opacity``). A color alpha channel multiplies into both
            data-driven policies verbatim, so imported data (amplitudes = 1,
            opacity in alpha) round-trips losslessly under the default.
        color_source: ``auto`` = per-splat colors if present, else colormap if
            given, else white; or force ``colors`` / ``colormap`` / ``white``.
        colormap: Colormap name for baking scalar amplitudes to RGB.
        sh_degree: 0 (default) writes only the DC band; higher degrees emit
            zero-filled ``f_rest`` bands for viewers that insist on them.
        undo_orientation: Invert the import-time orientation recorded in
            ``stats["interop"]`` so import → export round-trips exactly.
        timepoint / slice_dim / slice_index: 3D selection for nD data;
            ``timepoint`` slices the last (stacked) dimension (see
            :func:`_select_3d`).

    Returns:
        The complete PLY file contents.
    """
    if not 0 <= sh_degree <= 3:
        raise ValueError(f"sh_degree must be in [0, 3]; got {sh_degree}")
    if data.n_splats == 0:
        raise ValueError("Cannot export an empty splat set")

    centers, sigma, amplitudes, per_splat_colors = _select_3d(
        data, timepoint, slice_dim, slice_index
    )
    if undo_orientation:
        centers, sigma = _undo_import_orientation(centers, sigma, data)

    # RGBA colors: the alpha channel is per-splat opacity — it feeds the PLY
    # opacity field (via _opacity_logits), never the DC color bands.
    alpha: Optional[np.ndarray] = None
    if per_splat_colors is not None and per_splat_colors.shape[1] == 4:
        alpha = per_splat_colors[:, 3]
        per_splat_colors = per_splat_colors[:, :3]

    log_scales, quats = _scales_and_quats(sigma)
    opacity = _opacity_logits(amplitudes, alpha, opacity_policy, constant_opacity)

    rgb = _resolve_colors(amplitudes, per_splat_colors, color_source, colormap)
    f_dc = (rgb - 0.5) / SH_C0

    n = centers.shape[0]
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    props = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(n_rest)]
        + [
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Generated by luxar (https://github.com/royerlab/luxar)\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    body = np.zeros((n, len(props)), dtype="<f4")
    body[:, 0:3] = centers
    body[:, 6:9] = f_dc  # nx, ny, nz stay zero
    off = 9 + n_rest  # f_rest bands stay zero
    body[:, off] = opacity
    body[:, off + 1 : off + 4] = log_scales
    body[:, off + 4 : off + 8] = quats
    return header.encode("ascii") + body.tobytes()


def export_inria_ply(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs: object,
) -> int:
    """Export a ``.gsplats.zarr`` to an INRIA PLY file; returns the splat count.

    Keyword arguments are forwarded to :func:`gsplat_data_to_inria_ply`.
    Partition / nested trees have no flat equivalent — flatten first
    (``luxar gsplat flatten``).
    """
    from luxar.gsplats.gsplat_data import GSplatData

    try:
        # include_stats: the import-time orientation matrix lives in
        # stats["interop"] — without it undo_orientation silently no-ops.
        data = GSplatData.load(Path(input_path), include_stats=True)
    except ValueError as exc:
        raise ValueError(
            f"{Path(input_path).name}: not a flat/matrix-shaped gsplat store "
            f"({exc}). Collapse it first with `luxar gsplat flatten`."
        ) from exc

    payload = gsplat_data_to_inria_ply(data, **kwargs)  # type: ignore[arg-type]
    Path(output_path).write_bytes(payload)
    # Splat count after any nD slicing = rows in the vertex element.
    header_end = payload.index(b"end_header\n")
    for line in payload[:header_end].decode("ascii").splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    return data.n_splats  # pragma: no cover - header always has the element
