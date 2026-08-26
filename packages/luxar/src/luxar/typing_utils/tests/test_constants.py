"""Guards for cross-language constants.

These pin values that must stay literally equal across the Python package and
the TypeScript viewer. Numeric constants use same-value assertions on each side;
shared vocabularies parse the viewer source here so changing only one language
fails directly.
"""

import math
from pathlib import Path

from luxar.conftest import find_repo_relative_file, read_ts_string_union
from luxar.typing_utils.constants import (
    DEFAULT_POINT_RADIUS,
    DEFAULT_TRUNCATION_RADIUS,
    LINE_JOIN_STYLES,
    LOD_SELECTORS,
)


def _viewer_type_source(filename: str) -> str:
    rel = Path("packages/luxar-viewer/src/types") / filename
    start = Path(__file__).resolve()
    source_path = find_repo_relative_file(rel, start)
    assert source_path is not None, (
        f"cannot locate {rel} in any ancestor of {start}. If the viewer file moved, "
        "update this test — do NOT delete the cross-language lock."
    )
    return source_path.read_text(encoding="utf-8")


def test_line_join_styles_match_the_viewer_union() -> None:
    """Writer validation and viewer parsing accept exactly the same spellings."""
    viewer_styles = read_ts_string_union(
        _viewer_type_source("line-join.ts"), "LineJoinStyle"
    )
    assert viewer_styles == LINE_JOIN_STYLES


def test_lod_selectors_match_the_viewer_metadata_union() -> None:
    """Authored selector units stay valid on both sides of the file format."""
    viewer_selectors = read_ts_string_union(
        _viewer_type_source("lod-group.ts"), "selector"
    )
    assert viewer_selectors == LOD_SELECTORS


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


class TestDefaultPointRadius:
    """The radius a point is drawn with when a node stores no ``radii``."""

    def test_value(self) -> None:
        # MIRROR: DEFAULT_POINT_RADIUS in
        # packages/luxar-viewer/src/config/constants.ts must hold this value.
        # If you change one, change the other — a viewer test pins that side.
        assert DEFAULT_POINT_RADIUS == 0.5

    def test_authoring_default_is_the_same_constant(self) -> None:
        # The authoring default (what add_points materializes when the caller
        # supplies no radii) and the bounds default (what the spatial index
        # expands a no-radii chunk by) are the same number BY CONSTRUCTION —
        # the adder imports the constant rather than restating 0.5.
        from luxar.core.group.adders.points import (
            DEFAULT_POINT_RADIUS as ADDER_DEFAULT,
        )

        assert ADDER_DEFAULT is DEFAULT_POINT_RADIUS

    def test_no_radii_chunk_bounds_use_it(self) -> None:
        # The write-side guarantee: a chunk bound is never tighter than the
        # footprint the renderer draws.
        import numpy as np

        from luxar.io.ordering import compute_chunk_bounds_points

        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        bounds = compute_chunk_bounds_points(positions, radii=None, chunk_size=1)
        assert bounds[0, 0, 0] == -DEFAULT_POINT_RADIUS
        assert bounds[0, 0, 1] == DEFAULT_POINT_RADIUS
