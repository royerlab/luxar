"""`gsplat compare` must score on the fit's basis, and validate an override (#1173).

A render is background-relative; a raw reference is not. These pin that the
command reconciles them from the store, that an explicit override is honoured, and
that a nonsensical override is refused rather than silently scoring the wrong
thing.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli.main import app
from luxar.gsplats.gsplat_data import GSplatData

PEDESTAL = 600.0


@pytest.fixture
def fitted_store_and_reference(tmp_path):
    """A store advertising a known basis, plus the raw volume it was fitted from."""
    volume = np.full((10, 10, 10), PEDESTAL, dtype=np.float32)
    volume[3:7, 3:7, 3:7] += 1200.0
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)

    rng = np.random.default_rng(0)
    n = 10
    data = GSplatData(
        centers=rng.uniform(3.0, 7.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 0.9, n).astype(np.float32),
        cholesky_factors=np.tile([1.2, 0, 1.2, 0, 0, 1.2], (n, 1)).astype(np.float32),
        stats={"image_min": PEDESTAL, "floor": PEDESTAL},
    )
    store = tmp_path / "fit.gsplats.zarr"
    data.save(store, include_fitting_info=True)
    return store, ref


def _run(*args: str):
    return CliRunner().invoke(app, ["gsplat", "compare", *args])


def test_the_stored_basis_is_read_and_reported(fitted_store_and_reference) -> None:
    """Silence here would leave a reader unable to tell which basis a score is on."""
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref))
    assert result.exit_code == 0, result.output
    assert "image_min=600" in result.output
    assert "from dataset" in result.output


def test_an_explicit_override_wins_and_says_so(fitted_store_and_reference) -> None:
    """For a store fitted before the level was persisted."""
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref), "--image-min", "100")
    assert result.exit_code == 0, result.output
    assert "image_min=100" in result.output
    assert "from --image-min" in result.output


@pytest.mark.parametrize("bad", ["-50", "nan", "inf"])
def test_a_nonsensical_override_is_refused(fitted_store_and_reference, bad) -> None:
    """A level is what the fit SUBTRACTED, so it cannot be negative or
    non-finite. A negative one shifts the reference the wrong way, which would
    score against the wrong thing silently — and typing `-50` for `50` is the
    obvious way to get there. Validated to the same standard as a stored level,
    which `fit_image_min` already refuses.
    """
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref), "--image-min", bad)
    assert result.exit_code == 1, result.output
    assert "--image-min must be" in result.output


def test_a_store_with_no_basis_warns_rather_than_pretending(tmp_path) -> None:
    """Absence is ordinary — a merge drops keys its inputs disagree on — so the
    command must say the score is on the raw volume instead of implying otherwise.
    """
    volume = np.full((8, 8, 8), 300.0, dtype=np.float32)
    volume[2:5, 2:5, 2:5] += 900.0
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)

    rng = np.random.default_rng(1)
    data = GSplatData(
        centers=rng.uniform(2.0, 5.0, (8, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 0.9, 8).astype(np.float32),
        cholesky_factors=np.tile([1.0, 0, 1.0, 0, 0, 1.0], (8, 1)).astype(np.float32),
    )
    store = tmp_path / "nobasis.gsplats.zarr"
    data.save(store)

    result = _run(str(store), str(ref))
    assert result.exit_code == 0, result.output
    assert "image_min" in result.output  # the hint names the missing stat
