"""A/B toggles for the relocation dynamic-ops path (issue #2734).

Each lever defaults to CURRENT behavior, so an unset config is a byte-for-byte
no-op. These tests pin each lever's non-default behavior and are written to FAIL
before the lever existed (a default-config control is asserted beside each).
"""

import numpy as np
import torch

from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.operations import _relocate_splats_batch
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def _single_splat_model() -> GaussianSplatModel:
    centers = np.array([[1.0, 1.0]], dtype=np.float32)
    L0 = np.eye(2, dtype=np.float32)[None, :, :] * 0.2
    amps0 = np.array([0.1], dtype=np.float32)
    return GaussianSplatModel(
        shape=(8, 8),
        centers0=centers,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.1, 0.1],
        device=torch.device("cpu"),
    )


def _relocate(model, cfg, optimizer=None):
    residual = torch.zeros((8, 8), dtype=torch.float32)
    residual[3, 4] = 1.0
    splat_indices = torch.tensor([0], dtype=torch.long)
    peak_coords = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    _relocate_splats_batch(
        model, splat_indices, peak_coords, residual, cfg, optimizer=optimizer
    )


class TestDefaultsAreNoOp:
    def test_new_fields_default_to_current_behavior(self) -> None:
        cfg = DynamicOpsConfig()
        assert cfg.relocation_stop_fraction == 1.0
        assert cfg.reset_adam_on_relocation is True
        assert cfg.relocation_reseed_sigma_vox is None


class TestReseedSigma:
    def test_default_reseeds_at_init_sigma(self) -> None:
        """None (default) reseeds at init_sigma_vox (current behavior, 0.5)."""
        model = _single_splat_model()
        cfg = DynamicOpsConfig()  # relocation_reseed_sigma_vox is None
        cfg.init_sigma_vox = 0.5
        _relocate(model, cfg)
        _, Ls, _ = model.current_params()
        diag = torch.diagonal(Ls[0], dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.full((2,), 0.5), atol=1e-6)

    def test_override_reseeds_at_the_given_sigma(self) -> None:
        """A float override reseeds relocated splats at that sigma, not 0.5."""
        model = _single_splat_model()
        cfg = DynamicOpsConfig(relocation_reseed_sigma_vox=2.0)
        cfg.init_sigma_vox = 0.5  # would be used if the override were ignored
        _relocate(model, cfg)
        _, Ls, _ = model.current_params()
        diag = torch.diagonal(Ls[0], dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.full((2,), 2.0), atol=1e-6)
        # And emphatically NOT the 0.5 default (the pre-change behavior).
        assert not torch.allclose(diag, torch.full((2,), 0.5), atol=1e-2)


class TestAdamReset:
    @staticmethod
    def _adam_with_state(model) -> torch.optim.Adam:
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        pred = model()
        pred.sum().backward()
        opt.step()  # lazily creates exp_avg / exp_avg_sq for every param
        # Plant a sentinel so a reset is observable at the relocated index.
        for param in (model.raw_mu, model.raw_L_diag, model.L_off, model.raw_a):
            opt.state[param]["exp_avg"][0] = 7.0
            opt.state[param]["exp_avg_sq"][0] = 7.0
        return opt

    def test_default_resets_adam_moments(self) -> None:
        """True (default) zeros the Adam moments of a relocated splat."""
        model = _single_splat_model()
        opt = self._adam_with_state(model)
        cfg = DynamicOpsConfig()  # reset_adam_on_relocation True
        _relocate(model, cfg, optimizer=opt)
        for param in (model.raw_mu, model.raw_L_diag, model.L_off, model.raw_a):
            assert torch.all(opt.state[param]["exp_avg"][0] == 0.0)
            assert torch.all(opt.state[param]["exp_avg_sq"][0] == 0.0)

    def test_disabled_keeps_adam_moments(self) -> None:
        """False keeps the Adam moments (the sentinel survives relocation)."""
        model = _single_splat_model()
        opt = self._adam_with_state(model)
        cfg = DynamicOpsConfig(reset_adam_on_relocation=False)
        _relocate(model, cfg, optimizer=opt)
        for param in (model.raw_mu, model.raw_L_diag, model.L_off, model.raw_a):
            assert torch.all(opt.state[param]["exp_avg"][0] == 7.0)
            assert torch.all(opt.state[param]["exp_avg_sq"][0] == 7.0)


class TestStopFraction:
    """relocation_stop_fraction freezes relocation past a fraction of the run."""

    def _count_dynamic_calls(self, monkeypatch, stop_fraction: float) -> int:
        from luxar.gsplats import fit_gsplats  # noqa: F401
        from luxar.gsplats.fitting import optimization

        calls = {"n": 0}

        def fake_apply(*_args, **_kwargs):
            calls["n"] += 1
            return False

        monkeypatch.setattr(optimization, "apply_dynamic_operations", fake_apply)

        from luxar.gsplats.fit_gsplats import fit_gaussian_splats
        from luxar.gsplats.seeds import seed_from_grid

        rng = np.random.default_rng(0)
        V = rng.random((24, 24)).astype(np.float32)
        seeds = seed_from_grid(V, spacing=4.0)
        cfg = DynamicOpsConfig(step_every=5, relocation_stop_fraction=stop_fraction)
        fit_gaussian_splats(
            V,
            seeds=seeds,
            n_iters=20,
            lr=0.1,
            max_abs_error=1e-9,  # effectively never converges -> loop runs all 20 iters
            early_stop_patience=None,
            enable_dynamic_ops=True,
            dynamic_config=cfg,
            verbose=False,
            napari_movie=False,
            sort_splats_enabled=False,
        )
        return calls["n"]

    def test_full_schedule_relocates_in_the_second_half(self, monkeypatch) -> None:
        """Default 1.0: dynamic ops fire at it=5,10,15,20 -> 4 calls."""
        assert self._count_dynamic_calls(monkeypatch, 1.0) == 4

    def test_half_schedule_freezes_the_second_half(self, monkeypatch) -> None:
        """0.5 on n_iters=20: only it<=10 fires -> it=5,10 -> 2 calls."""
        assert self._count_dynamic_calls(monkeypatch, 0.5) == 2
