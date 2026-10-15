"""A warm start's iteration-0 model must reproduce the seed it was handed.

``fit_gaussian_splats(V, seeds=<previous fit>)`` converts the seed's centers,
Cholesky factors and amplitudes back into model parameters. These tests stop the
fit right before its first optimizer step and compare that initial model, both
element-wise and as a rendered volume, against the seed. Three conversions used
to change the parameters before any optimization:

* a ``voxel_size`` fit writes PHYSICAL units by default, and the warm start read
  them as voxel indices (24^3 fixture: 47 dB seed -> 18 dB iteration 0);
* supplied Cholesky diagonals in ``[sigma_min, sigma_min + 0.1)`` were lifted by
  the fresh-seed gradient-safety margin, widening near-floor splats;
* background-relative amplitudes were capped at 1.0 although a fit's own
  ``amp_max`` follows the normalized peak and can exceed it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import luxar.gsplats.fit_gsplats as fit_module  # noqa: E402
from luxar.gsplats.fit_basis import reference_on_fit_basis  # noqa: E402
from luxar.gsplats.fit_gsplats import fit_gaussian_splats  # noqa: E402
from luxar.gsplats.fitting.initialization import (  # noqa: E402
    apply_sigma_min_diag_floor,
    apply_supplied_sigma_min_diag_floor,
)
from luxar.gsplats.fitting.preprocessing import (  # noqa: E402
    _InitContext,
    _rescale_init_amps,
)
from luxar.gsplats.gsplat_data import GSplatData  # noqa: E402
from luxar.gsplats.lod.volume_refit import (  # noqa: E402
    VolumeRefitConfig,
    volume_refine_splats,
)
from luxar.gsplats.rendering.volume_rendering import (  # noqa: E402
    render_to_volume_tensor,
)
from luxar.gsplats.utils.trils import pack_tril, unpack_tril  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_VOXEL_SIZE = (2.0, 0.5, 0.75)


def _blob_volume() -> np.ndarray:
    """Small 3D volume: a pedestal of 10 plus a few Gaussian blobs."""
    rng = np.random.default_rng(0)
    size = 14
    z, y, x = np.mgrid[:size, :size, :size].astype(np.float32)
    volume = np.full((size, size, size), 10.0, dtype=np.float32)
    for _ in range(5):
        c = rng.uniform(3, size - 3, 3)
        s = rng.uniform(0.8, 2.0)
        r2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        volume += 100.0 * rng.uniform(0.4, 1.0) * np.exp(-r2 / (2 * s * s))
    return volume


_FIT_KWARGS: dict[str, Any] = dict(
    seeds=40,
    n_iters=40,
    device="cpu",
    verbose=False,
    enable_dynamic_ops=False,
    sort_splats_enabled=False,
    cull_retention=None,
)


class _StopBeforeFirstStep(Exception):
    pass


def _initial_model(
    monkeypatch: pytest.MonkeyPatch, volume: np.ndarray, **kwargs: Any
) -> tuple[Any, Any]:
    """Run a warm start up to (not into) its optimization loop."""
    captured: dict[str, Any] = {}

    def _capture(components: Any, _loss: Any, _config: Any, pre: Any) -> Any:
        captured["model"] = components.model
        captured["pre"] = pre
        raise _StopBeforeFirstStep

    with monkeypatch.context() as patch:
        patch.setattr(fit_module, "run_optimization_loop", _capture)
        with pytest.raises(_StopBeforeFirstStep):
            fit_gaussian_splats(volume, **{**_FIT_KWARGS, **kwargs})
    return captured["model"], captured["pre"]


def _to_voxel(seed: GSplatData, voxel_size: tuple[float, ...] | None) -> GSplatData:
    """The seed's mixture on the volume's voxel grid (what the model holds)."""
    if voxel_size is None:
        return seed
    vs = np.asarray(voxel_size, dtype=np.float64)
    d = seed.centers.shape[1]
    tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
    return GSplatData(
        centers=(seed.centers / vs).astype(np.float32),
        amplitudes=seed.amplitudes,
        cholesky_factors=(seed.cholesky_factors / tril_scales).astype(np.float32),
        truncation_radius=seed.truncation_radius,
    )


def _assert_initial_model_reproduces(
    model: Any,
    pre: Any,
    seed_voxel: GSplatData,
    shape: tuple[int, ...],
) -> None:
    with torch.no_grad():
        centers, L, amps = model.current_params()
        rendered_model = model.forward().cpu().numpy() * pre.intensity_range
    seed_L = unpack_tril(seed_voxel.cholesky_factors, seed_voxel.centers.shape[1])
    np.testing.assert_allclose(centers.cpu().numpy(), seed_voxel.centers, atol=1e-4)
    np.testing.assert_allclose(L.cpu().numpy(), seed_L, atol=1e-3)
    np.testing.assert_allclose(
        amps.cpu().numpy() * pre.intensity_range,
        seed_voxel.amplitudes,
        rtol=1e-4,
        atol=1e-4 * float(seed_voxel.amplitudes.max()),
    )
    rendered_seed = render_to_volume_tensor(
        seed_voxel, shape=shape, device="cpu", truncate=seed_voxel.truncation_radius
    ).numpy()
    peak = float(np.abs(rendered_seed).max())
    assert float(np.abs(rendered_model - rendered_seed).max()) < 1e-4 * peak


class TestWarmStartRoundTrip:
    def test_voxel_fit_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        model, pre = _initial_model(
            monkeypatch, volume, seeds=seed, seed_amps_background_relative=True
        )
        assert pre.intensity_range == pytest.approx(seed.stats["intensity_range"])
        _assert_initial_model_reproduces(model, pre, seed, volume.shape)

    def test_physical_unit_fit_round_trips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A voxel_size fit writes physical units; feeding it back with the same
        settings must divide them back out rather than read them as voxels."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, voxel_size=_VOXEL_SIZE, **_FIT_KWARGS)
        # Non-vacuous: the stored seed really is in physical units.
        assert not np.allclose(_to_voxel(seed, _VOXEL_SIZE).centers, seed.centers)
        model, pre = _initial_model(
            monkeypatch,
            volume,
            seeds=seed,
            voxel_size=_VOXEL_SIZE,
            seed_amps_background_relative=True,
        )
        _assert_initial_model_reproduces(
            model, pre, _to_voxel(seed, _VOXEL_SIZE), volume.shape
        )

    def test_explicit_seed_space_overrides_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A voxel-space seed (``output_space="voxel"`` fit) fed to a physical
        fit is declared with ``seed_space="voxel"`` and is not divided again."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(
            volume, voxel_size=_VOXEL_SIZE, output_space="voxel", **_FIT_KWARGS
        )
        model, pre = _initial_model(
            monkeypatch,
            volume,
            seeds=seed,
            voxel_size=_VOXEL_SIZE,
            seed_amps_background_relative=True,
            seed_space="voxel",
        )
        _assert_initial_model_reproduces(model, pre, seed, volume.shape)

    def test_undeclared_seed_stays_in_voxel_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``generate_seeds()`` output is voxel indices and is NOT declared a
        fit's output, so the default must leave its frame alone."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        model, _pre = _initial_model(
            monkeypatch, volume, seeds=seed, voxel_size=_VOXEL_SIZE
        )
        with torch.no_grad():
            centers, _L, _amps = model.current_params()
        np.testing.assert_allclose(centers.cpu().numpy(), seed.centers, atol=1e-4)

    def test_refit_on_the_fit_basis_round_trips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``lod --refine volume`` wiring: the target is shifted onto the
        seed's basis and refitted with ``floor="none"`` — a different but
        compatible normalization that resolves the same intensity range."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        target = reference_on_fit_basis(volume, seed.stats["image_min"])
        model, pre = _initial_model(
            monkeypatch,
            target,
            seeds=seed,
            floor="none",
            seed_amps_background_relative=True,
            truncate=seed.truncation_radius,
        )
        assert pre.image_min == pytest.approx(0.0)
        assert pre.intensity_range == pytest.approx(seed.stats["intensity_range"])
        _assert_initial_model_reproduces(model, pre, seed, volume.shape)

    def test_large_off_diagonal_cholesky_round_trips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strongly sheared splats keep their off-diagonal Cholesky terms (no
        eccentricity limit, so the model represents the seed exactly)."""
        volume = _blob_volume()
        rng = np.random.default_rng(1)
        n = 12
        L = np.zeros((n, 3, 3), dtype=np.float32)
        L[:, [0, 1, 2], [0, 1, 2]] = rng.uniform(0.8, 2.0, (n, 3))
        L[:, 1, 0] = rng.uniform(-2.5, 2.5, n)
        L[:, 2, 0] = rng.uniform(-2.5, 2.5, n)
        L[:, 2, 1] = rng.uniform(-2.5, 2.5, n)
        seed = GSplatData(
            centers=rng.uniform(3, 11, (n, 3)).astype(np.float32),
            amplitudes=rng.uniform(20, 80, n).astype(np.float32),
            cholesky_factors=pack_tril(L).astype(np.float32),
            truncation_radius=3.0,
        )
        model, pre = _initial_model(
            monkeypatch,
            volume,
            seeds=seed,
            seed_amps_background_relative=True,
            max_eccentricity=None,
            truncate=3.0,
        )
        _assert_initial_model_reproduces(model, pre, seed, volume.shape)

    def test_volume_refine_splats_starts_on_its_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``volume_refine_splats`` itself (raw volume + recorded image_min)
        hands the inner fit a model that reproduces the seed."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        config = VolumeRefitConfig(iters=30, image_min=seed.stats["image_min"])
        captured: dict[str, Any] = {}

        def _capture(components: Any, _loss: Any, _config: Any, pre: Any) -> Any:
            captured["model"] = components.model
            captured["pre"] = pre
            raise _StopBeforeFirstStep

        with monkeypatch.context() as patch:
            patch.setattr(fit_module, "run_optimization_loop", _capture)
            with pytest.raises(_StopBeforeFirstStep):
                volume_refine_splats(seed, volume, config=config, device="cpu")
        _assert_initial_model_reproduces(
            captured["model"], captured["pre"], seed, volume.shape
        )
        # And end to end, the never-worse guard keeps a result no worse.
        out, stats = volume_refine_splats(seed, volume, config=config, device="cpu")
        assert out.n_splats == seed.n_splats
        assert stats["improved"] is not stats["seed_won"]
        if stats["improved"]:
            assert stats["mse_refit"] < stats["mse_seed"]
        else:
            assert out is seed

    def test_amplitudes_above_amp_max_start_trainable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A seed amplitude above this fit's ``amp_max`` is clipped just under
        the cap, so the forward clamp still passes it a gradient."""
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        amp_max = 0.3
        above = seed.amplitudes / seed.stats["intensity_range"] > amp_max
        assert int(above.sum()) > 0
        model, _pre = _initial_model(
            monkeypatch,
            volume,
            seeds=seed,
            seed_amps_background_relative=True,
            amp_max=amp_max,
        )
        model.forward().sum().backward()
        grad = model.raw_a.grad.detach().cpu().numpy()
        assert np.all(grad[above] != 0.0)
        with torch.no_grad():
            _c, _L, amps = model.current_params()
        np.testing.assert_allclose(amps.cpu().numpy()[above], amp_max, rtol=1e-5)

    def test_seed_space_warns_when_it_cannot_apply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        volume = _blob_volume()
        seed = fit_gaussian_splats(volume, **_FIT_KWARGS)
        with pytest.warns(UserWarning, match="without voxel_size"):
            _initial_model(
                monkeypatch,
                volume,
                seeds=seed,
                seed_amps_background_relative=True,
                seed_space="real",
            )
        with pytest.warns(UserWarning, match="only applies to a GSplatData"):
            _initial_model(
                monkeypatch, volume, seeds=seed.centers.copy(), seed_space="voxel"
            )

    def test_seed_space_is_validated(self) -> None:
        with pytest.raises(ValueError, match="seed_space"):
            fit_gaussian_splats(_blob_volume(), seed_space="pixels", **_FIT_KWARGS)


class TestSuppliedSigmaFloor:
    def test_representable_diagonals_are_kept(self) -> None:
        floor = np.array([0.3, 0.3], dtype=np.float32)
        diag = np.array([[0.32, 1.5], [0.35, 0.39]], dtype=np.float32)
        np.testing.assert_array_equal(
            apply_supplied_sigma_min_diag_floor(diag, floor), diag
        )
        # The fresh-seed floor would have widened every near-floor value.
        assert np.all(apply_sigma_min_diag_floor(diag, floor)[diag < 0.4] >= 0.4)

    def test_exact_floor_is_nudged_and_below_floor_gets_the_margin(self) -> None:
        floor = np.array([0.3], dtype=np.float32)
        out = apply_supplied_sigma_min_diag_floor(
            np.array([[0.3], [0.1]], dtype=np.float32), floor
        )
        assert 0.3 < out[0, 0] < 0.3 + 1e-3
        assert out[1, 0] == pytest.approx(0.4)

    def test_one_ulp_below_the_floor_is_nudged_not_widened(self) -> None:
        """A float32 round trip can land an at-floor diagonal an ulp below it;
        that must not reintroduce the full 0.1 widening."""
        floor = np.array([0.3], dtype=np.float32)
        below = np.nextafter(np.float32(0.3), np.float32(0.0))
        out = apply_supplied_sigma_min_diag_floor(
            np.array([[below]], dtype=np.float32), floor
        )
        assert 0.3 < out[0, 0] < 0.3 + 1e-3


def test_background_relative_amplitudes_are_not_capped_at_one() -> None:
    """A fit under a supplied ``norm_range`` can hold normalized amplitudes
    above 1.0; the warm start must not cap them (the model clamps at its own
    ``amp_max``)."""
    ctx = _InitContext(
        init_amps=np.array([-1.0, 50.0, 150.0]), init_amps_background_relative=True
    )
    _rescale_init_amps(ctx, image_min=10.0, intensity_range=100.0, verbose=False)
    np.testing.assert_allclose(ctx.init_amps, [0.0, 0.5, 1.5])
