"""The public fitters share the CLI's near-lossless post-fit trim."""

import numpy as np
import pytest


@pytest.mark.parametrize("progressive", [False, True])
def test_default_cull_matches_explicit_near_lossless_retention(
    progressive: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    rng = np.random.default_rng(3)
    volume = rng.random((32, 32), dtype=np.float32) * 0.12
    volume[4:11, 5:13] += 0.8
    volume[19:27, 17:26] += 0.5

    def fit(cull_retention: float | None, *, use_default: bool = False):
        np.random.seed(0)
        torch.manual_seed(0)
        cull_kwarg = {} if use_default else {"cull_retention": cull_retention}
        if progressive:
            return fit_progressive_gaussian_splats(
                volume,
                max_splats=120,
                max_splats_per_pass=60,
                iters_per_pass=50,
                max_passes=2,
                psnr_patience=0.0,
                residual_pass_min_iters=50,
                verbose=use_default,
                device="cpu",
                floor="none",
                **cull_kwarg,
            )
        return fit_gaussian_splats(
            volume,
            seeds=120,
            n_iters=50,
            verbose=use_default,
            device="cpu",
            enable_dynamic_ops=False,
            seed_method="peaks",
            floor="none",
            **cull_kwarg,
        )

    # The explicit unculled result is the same fit before the last-step trim.
    unculled = fit(None)
    default = fit(None, use_default=True)
    assert "retention=99.9%" in capsys.readouterr().out
    expected = unculled.cull(method="cumulative", retention=0.999)
    old = unculled.cull(method="cumulative", retention=0.98 if progressive else 0.95)
    assert expected.n_splats > old.n_splats
    assert default.n_splats == expected.n_splats
    assert "psnr_db" in default.stats
    np.testing.assert_allclose(default.centers, expected.centers)
    np.testing.assert_allclose(default.amplitudes, expected.amplitudes)
