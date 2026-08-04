"""Guards for cross-language numeric constants.

These pin values that must stay literally equal across the Python package and
the TypeScript viewer. The two languages cannot share a symbol, so each side
asserts its own value and names the other file — the same convention used for
``ALPHA_CLAMP`` (``luxar.gsplats.utils.alpha`` <->
``rendering/materials/_shared/volumetric.ts``).
"""

import math

from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


class TestDefaultTruncationRadius:
    """The canonical GSplat truncation radius ``T``."""

    def test_value(self) -> None:
        # MIRROR: GSPLAT_DEFAULT_TRUNCATION_RADIUS in
        # packages/luxar-viewer/src/config/constants.ts must hold this value.
        # If you change one, change the other — a viewer test pins that side.
        assert DEFAULT_TRUNCATION_RADIUS == 2.75

    def test_is_a_usable_truncation(self) -> None:
        # T sets the shift C = exp(-T^2/2) and the renormalization 1/(1-C).
        # A T that made C reach 1 would divide by zero.
        shift_c = math.exp(-0.5 * DEFAULT_TRUNCATION_RADIUS**2)
        assert 0.0 < shift_c < 1.0
        assert math.isfinite(1.0 / (1.0 - shift_c))

    def test_fit_default_uses_the_constant(self) -> None:
        # The fitter is the anchor: whatever it uses is stamped onto every
        # fitted dataset, so a drift here silently mismatches every read-side
        # fallback. ConstraintConfig carries the user-facing default;
        # fit_gaussian_splats' own signature must agree with it.
        import inspect

        from luxar.gsplats.fit_gsplats import fit_gaussian_splats
        from luxar.gsplats.fitting.config import ConstraintConfig

        assert ConstraintConfig().truncate == DEFAULT_TRUNCATION_RADIUS

        sig_default = (
            inspect.signature(fit_gaussian_splats).parameters["truncate"].default
        )
        assert sig_default == DEFAULT_TRUNCATION_RADIUS

    def test_gsplat_data_stamps_the_constant(self) -> None:
        # A dataset built without an explicit radius must carry the canonical
        # value — this is what the viewer reads back.
        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.ones((2, 6), dtype=np.float32),
        )
        assert data.truncation_radius == DEFAULT_TRUNCATION_RADIUS

    def test_lift_deliberately_diverges(self) -> None:
        # luxar.gsplats.lift is the ONE documented exemption: its T is a
        # profile-matching parameter (the point super-Gaussian and the gsplat
        # kernel coincide at T* = sqrt(2 ln 100) = 3.0349), not a render
        # default. Moving it to 2.75 degrades the lift ~8.5x.
        from luxar.gsplats.lift import LIFT_TRUNCATION_RADIUS

        assert LIFT_TRUNCATION_RADIUS == 3.0
        assert LIFT_TRUNCATION_RADIUS != DEFAULT_TRUNCATION_RADIUS
        t_star = math.sqrt(2.0 * math.log(100.0))
        # The lift's value is the round number NEAR the exact coincidence.
        assert abs(LIFT_TRUNCATION_RADIUS - t_star) < 0.05
