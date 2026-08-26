"""Pin the packed-Cholesky convention: the diagonal carries σ, not 1/σ.

``cholesky_factors`` is the packed lower-triangular factor **L of the
covariance** (Σ = L·Lᵀ), so an isotropic Gaussian of standard deviation σ is
``[σ, 0, σ, 0, 0, σ]``. The inverted reading — L as the factor of the
*precision* matrix Σ⁻¹, giving ``[1/σ, 0, 1/σ, 0, 0, 1/σ]`` — is a plausible
mistake that nothing else in the suite catches: the diagonal stays positive, so
validation passes and a perfectly valid store is written whose splats are wrong
by a factor of 1/σ² in linear extent.

Existing coverage does not close this gap. Round-trip tests assert that stored
bytes decode back to the input array, which is convention-blind (it round-trips
``1/σ`` just as faithfully). The renderer parity tests feed a synthetic L and
check that both backends agree on Σ = L·Lᵀ, which says nothing about what the
*author* meant by σ. These tests bridge the two: author σ, recover σ.

Every σ here is deliberately ≠ 1, because σ == 1/σ at 1.0 — a unit-σ fixture
passes under either convention and would make this file vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding.decoder import ArrayDecoder
from luxar.gsplats.gsplat_data import GSplatData

#: Deliberately ≠ 1 (see module docstring); 1/SIGMA == 1.25 is 56% away, far
#: outside any quantization tolerance below.
SIGMA = 0.8

#: Per-axis standard deviations for the anisotropic case, all ≠ 1 and all
#: distinct so a transposed or reversed packing fails too.
SIGMAS_XYZ = (0.4, 1.6, 0.5)


def _packed_isotropic_3d(sigma: float) -> np.ndarray:
    """``[L00, L10, L11, L20, L21, L22]`` for an isotropic 3D Gaussian."""
    return np.array([sigma, 0.0, sigma, 0.0, 0.0, sigma], dtype=np.float32)


def test_marginal_sigma_equals_authored_diagonal() -> None:
    """A diagonal of σ must read back as a marginal σ, not 1/σ."""
    data = GSplatData(
        centers=np.zeros((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=_packed_isotropic_3d(SIGMA)[None, :],
    )

    sigmas = data.marginal_sigmas()[0]

    np.testing.assert_allclose(sigmas, SIGMA, rtol=1e-6)
    assert not np.allclose(sigmas, 1.0 / SIGMA), (
        "marginal sigma came back as 1/σ — the packed diagonal is being read "
        "as a factor of the precision matrix Σ⁻¹ instead of the covariance Σ"
    )


def test_anisotropic_diagonal_maps_axis_for_axis() -> None:
    """Each diagonal entry is that axis' σ, in packing order."""
    sx, sy, sz = SIGMAS_XYZ
    data = GSplatData(
        centers=np.zeros((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=np.array([[sx, 0.0, sy, 0.0, 0.0, sz]], dtype=np.float32),
    )

    np.testing.assert_allclose(data.marginal_sigmas()[0], SIGMAS_XYZ, rtol=1e-6)


def test_offdiagonal_widens_its_row() -> None:
    """Σ[1,1] = L10² + L11², so an off-diagonal term GROWS the Y extent.

    Under the inverted (precision) reading the coupling would shrink it, so the
    direction of this change is itself a convention check.
    """
    sigma, off = 0.6, 0.5
    packed = np.array([sigma, off, sigma, 0.0, 0.0, sigma], dtype=np.float32)
    data = GSplatData(
        centers=np.zeros((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=packed[None, :],
    )

    sigmas = data.marginal_sigmas()[0]

    np.testing.assert_allclose(sigmas[0], sigma, rtol=1e-6)
    np.testing.assert_allclose(sigmas[1], np.hypot(off, sigma), rtol=1e-6)
    assert sigmas[1] > sigmas[0]


def test_authored_sigma_survives_a_scene_roundtrip(tmp_path) -> None:
    """Compile hand-authored splats and recover σ from the written store.

    Read through ``ArrayDecoder`` rather than off the raw zarr array: the store
    splits the packed factor into ``cholesky_factors_diag`` /
    ``cholesky_factors_offdiag`` and quantizes each (log-per-channel by
    default), so the stored codes are not the σ values. Decoding is also what
    the viewer does, which is the path that matters.

    Per-axis σ here so a broadcast/uniform shortcut in the writer cannot make
    the assertion vacuous, and the rows are Hilbert-reordered on write, so
    compare as a set of rows rather than element-wise.
    """
    path = tmp_path / "convention.luxar.zarr"
    n = 64
    sx, sy, sz = SIGMAS_XYZ
    rng = np.random.default_rng(0)

    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats(
            "blobs",
            centers=rng.normal(scale=0.3, size=(n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.repeat(
                np.array([[sx, 0.0, sy, 0.0, 0.0, sz]], dtype=np.float32), n, axis=0
            ),
        )

    node = zarr.open_group(path, mode="r")["blobs"]
    decoder = ArrayDecoder()
    diag = decoder.decode(node["cholesky_factors_diag"])
    offdiag = decoder.decode(node["cholesky_factors_offdiag"])

    # Σ_ii = Σ_j L_ij², i.e. diagonal² plus that row's off-diagonal terms.
    sigmas = np.sqrt(
        np.stack(
            [
                diag[:, 0] ** 2,
                diag[:, 1] ** 2 + offdiag[:, 0] ** 2,
                diag[:, 2] ** 2 + offdiag[:, 1] ** 2 + offdiag[:, 2] ** 2,
            ],
            axis=1,
        )
    )

    assert sigmas.shape == (n, 3)
    np.testing.assert_allclose(
        sigmas, np.broadcast_to(np.array(SIGMAS_XYZ), sigmas.shape), rtol=1e-2
    )
    assert sigmas[:, 1].min() > sigmas[:, 0].max(), (
        "the sy=1.6 axis did not come back as the widest — an inverted "
        "(precision) packing would make it the narrowest"
    )


@pytest.mark.parametrize("sigma", [0.02, 0.5, 2.0, 50.0])
def test_extent_grows_with_sigma(sigma: float) -> None:
    """Extent must be monotonically increasing in σ (scale-like, not precision-like)."""
    data = GSplatData(
        centers=np.zeros((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=_packed_isotropic_3d(sigma)[None, :],
    )

    np.testing.assert_allclose(data.marginal_sigmas()[0], sigma, rtol=1e-5)
