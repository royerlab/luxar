"""Convert decoded classical splats (:class:`ClassicalSplats`) to a
:class:`~luxar.gsplats.gsplat_data.GSplatData`: orientation fix, covariance
rebuild + robust Cholesky, opacity→amplitude, DC→RGB. Extracted from
``classical_splats.py`` in the per-concern split; re-exported there so
``classical_to_gsplat_data`` keeps its public import path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from luxar.gsplats.interop._color import srgb_to_linear
from luxar.gsplats.interop._quat import quat_to_rotmat

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.interop.classical_splats import ClassicalSplats


def _orientation_matrix(rotate_x180: bool, flip: str) -> np.ndarray:
    """Build the (3, 3) orientation matrix applied to imported world coordinates.

    ``rotate_x180`` is the canonical COLMAP fix (captures store +Y down / +Z
    forward, so they appear upside-down in Y-up viewers); ``flip`` mirrors the
    named axes on top of that (a reflection — allowed here because the
    covariance is rebuilt from scratch rather than routed through
    ``GSplatData.transform``, whose diagonal path rejects negative scales).
    """
    M = np.eye(3, dtype=np.float64)
    if rotate_x180:
        M = np.diag([1.0, -1.0, -1.0]) @ M
    for axis in flip:
        try:
            idx = "xyz".index(axis.lower())
        except ValueError:
            raise ValueError(f"flip axes must be drawn from 'xyz'; got {flip!r}")
        M[idx] *= -1.0
    return M


def _robust_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Batch Cholesky with a per-splat eigenvalue-clamp fallback for non-PD input.

    Mirrors the regularization strategy of
    :func:`luxar.gsplats.utils.trils.embed_cholesky_packed`: quantized or
    degenerate source files can yield covariance matrices that are only
    positive *semi*-definite; those get their eigenvalues floored and are
    re-factorized individually.
    """
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        pass

    # Fully vectorized detect-and-repair (no per-splat Python loop, so a single
    # degenerate splat among millions doesn't drop the whole import to O(N)
    # scalar LAPACK calls). Batch eigvalsh finds the non-PD subset; only those
    # get their eigenvalues floored and recomposed via batch eigh.
    eigvals_all = np.linalg.eigvalsh(sigma)  # ascending per splat
    scale = np.maximum(np.abs(eigvals_all[:, -1]), 1e-14)
    floor = scale * 1e-9  # per-splat relative floor
    bad = eigvals_all[:, 0] < floor
    if not np.any(bad):
        # PD everywhere but the batch call still failed (rare numerical noise) —
        # symmetrize and retry once.
        sym = (sigma + np.swapaxes(sigma, -2, -1)) / 2.0
        return np.linalg.cholesky(sym)

    fixed = sigma.copy()
    eigvals, eigvecs = np.linalg.eigh(sigma[bad])
    eigvals = np.maximum(eigvals, floor[bad, None])
    fixed[bad] = eigvecs @ (eigvals[..., None] * np.swapaxes(eigvecs, -2, -1))
    return np.linalg.cholesky(fixed)


def classical_to_gsplat_data(
    cs: ClassicalSplats,
    *,
    rotate_x180: Optional[bool] = None,
    flip: str = "",
) -> "GSplatData":
    """Convert decoded classical splats to a :class:`GSplatData`.

    The covariance is rebuilt as ``Σ = (M·R) · diag(scales²) · (M·R)ᵀ`` where
    ``R`` comes from the quaternion and ``M`` is the orientation matrix
    (:func:`_orientation_matrix`), then factorized to Luxar's packed
    lower-triangular Cholesky form. Opacities ride in the RGBA color alpha
    channel (amplitudes are constant 1 — see the inline note); the DC
    color (display-referred sRGB) is converted to Luxar's linear-light store via
    :func:`~luxar.gsplats.interop._color.srgb_to_linear`. Columns stay in world
    (x, y, z) order — that is what downstream dimension inference labels x/y/z.

    ``rotate_x180=None`` (default) applies the 180°-about-X COLMAP → Y-up fix
    exactly when the source dialect needs it (``cs.y_up`` False); SPZ declares
    RUB/Y-up data and is left untouched. Pass an explicit bool to override.

    The applied orientation and source dialect are recorded under
    ``stats["interop"]`` so an eventual export can invert them.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import pack_tril

    if cs.n_splats == 0:
        raise ValueError("Cannot convert an empty splat set")

    if rotate_x180 is None:
        rotate_x180 = not cs.y_up
    M = _orientation_matrix(rotate_x180, flip)

    positions = (cs.positions.astype(np.float64) @ M.T).astype(np.float32)
    R = M @ quat_to_rotmat(cs.quaternions)  # (N, 3, 3), orientation folded in
    s2 = (cs.scales.astype(np.float64) ** 2)[:, None, :]  # (N, 1, 3)
    sigma = (R * s2) @ np.swapaxes(R, -2, -1)  # R · diag(s²) · Rᵀ
    L = _robust_cholesky(sigma)
    cholesky_factors = pack_tril(L).astype(np.float32)

    # Learned 3DGS opacity is a genuine PER-SPLAT property — it rides in the
    # color alpha channel (per-splat opacity: every blending mode multiplies a
    # splat's contribution by it, and volumetric maps it into optical depth,
    # so dark solid surfaces really occlude — see VOLUMETRIC_BLENDING_SPEC.md).
    # Amplitude is therefore constant 1: the alpha factor already carries the
    # per-splat weight, and folding opacity into amplitude too would render
    # o² in every mode. Mass-ranked ops (LOD ladders, culling) stay meaningful
    # through the alpha-aware `effective_amplitudes` helper.
    amplitudes = np.ones(cs.n_splats, dtype=np.float32)
    # Classical DC color is display-referred (sRGB); Luxar's viewer treats
    # per-splat color as linear light and applies the sRGB OETF once at output.
    # Convert sRGB → linear here so imports render with the same colors a
    # reference viewer (SuperSplat/PlayCanvas) shows instead of washing white.
    # Alpha is coverage, not light — it stays linear (no sRGB transfer).
    colors = np.concatenate(
        [
            srgb_to_linear(cs.colors),
            np.clip(cs.opacities, 0.0, 1.0)[:, None].astype(np.float32),
        ],
        axis=1,
    )

    stats = {
        "interop": {
            "source_format": cs.source_format,
            "source_sh_degree": int(cs.sh_degree),
            "orientation_matrix": M.tolist(),
            # Provenance marker: opacity lives in the color alpha channel
            # (amplitudes are constant 1). Export reads this back verbatim.
            "opacity_in_alpha": True,
        }
    }
    return GSplatData(
        centers=positions,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        colors=colors,
        stats=stats,
    )
