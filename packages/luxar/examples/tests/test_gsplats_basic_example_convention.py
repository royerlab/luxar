"""Pin ``gsplats_basic_example``'s hand-authoring helpers to the real convention.

The example is the canonical "author a splat without a fitting pipeline"
template, so its ``isotropic_cholesky`` / ``axis_aligned_cholesky`` /
``tilted_cholesky`` helpers are read as documentation. They once packed
``1/σ`` — the Cholesky factor of the *precision* matrix — which renders every
splat at ``1/σ`` instead of σ while writing a perfectly valid store.

``test_examples_smoke`` cannot catch that: it asserts the script runs and emits
a zarr, and a wrongly-shaped splat does both. These tests assert the numbers,
using σ ≠ 1 throughout (σ == 1/σ at 1.0 hides the inversion entirely).

See ``luxar.gsplats.tests.test_cholesky_convention`` for the same invariant
pinned on the library side.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def example():
    """Import ``gsplats_basic_example`` by file path (the dir is not a package)."""
    path = EXAMPLES_DIR / "gsplats_basic_example.py"
    if not path.exists():
        pytest.skip(f"Example missing on disk: {path}")
    # The example does `from _overlay_style import add_explainer`, which only
    # resolves when the examples dir is importable.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location("_convention_gsplats_basic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sigmas(packed: np.ndarray) -> np.ndarray:
    """Recover per-axis σ from a packed 3D factor via Σ = L·Lᵀ."""
    L = np.array(
        [
            [packed[0], 0.0, 0.0],
            [packed[1], packed[2], 0.0],
            [packed[3], packed[4], packed[5]],
        ]
    )
    return np.sqrt(np.diag(L @ L.T))


def test_isotropic_helper_packs_sigma_not_its_inverse(example) -> None:
    sigma = 0.8  # != 1, so σ and 1/σ are distinguishable
    np.testing.assert_allclose(
        _sigmas(example.isotropic_cholesky(sigma=sigma)), sigma, rtol=1e-6
    )


def test_axis_aligned_helper_maps_each_axis_to_its_own_sigma(example) -> None:
    sx, sy, sz = 0.4, 1.6, 0.5
    np.testing.assert_allclose(
        _sigmas(example.axis_aligned_cholesky(sx=sx, sy=sy, sz=sz)),
        (sx, sy, sz),
        rtol=1e-6,
    )


def test_tilted_helper_keeps_x_and_widens_y(example) -> None:
    """The off-diagonal adds to its own row: σ_y = hypot(off_xy, sy)."""
    sx, sy, sz, off = 0.6, 0.6, 1.2, 0.5
    sigmas = _sigmas(example.tilted_cholesky(sx=sx, sy=sy, sz=sz, off_xy=off))

    np.testing.assert_allclose(sigmas[0], sx, rtol=1e-6)
    np.testing.assert_allclose(sigmas[1], np.hypot(off, sy), rtol=1e-6)
    np.testing.assert_allclose(sigmas[2], sz, rtol=1e-6)


def test_scene_description_matches_the_geometry(example) -> None:
    """The green splat is described as elongated along Y — assert it truly is.

    Under the old inverted packing the same call produced a splat *squashed*
    along Y, so the on-screen explainer text contradicted the data.
    """
    sigmas = _sigmas(example.axis_aligned_cholesky(sx=0.4, sy=1.6, sz=0.4))
    assert sigmas[1] > sigmas[0] and sigmas[1] > sigmas[2]
