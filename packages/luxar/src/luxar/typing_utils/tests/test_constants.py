"""Guards for cross-language constants.

These pin values that must stay literally equal across the Python package and
the TypeScript viewer. Numeric constants use same-value assertions on each side;
shared vocabularies parse the viewer source here so changing only one language
fails directly. The numeric convention is the same one used for ``ALPHA_CLAMP``
(``luxar.gsplats.utils.alpha`` <->
``rendering/materials/_shared/volumetric.ts``).
"""

import math

import pytest

from luxar.conftest import read_ts_string_literals, viewer_source
from luxar.typing_utils.constants import (
    DEFAULT_POINT_RADIUS,
    DEFAULT_TRUNCATION_RADIUS,
)
from luxar.typing_utils.geometry_capabilities import (
    lod_capable_types,
    partition_capable_types,
)


def test_lod_display_types_match_the_capability_table() -> None:
    """The viewer's metadata union names exactly the LOD-capable geometries."""
    viewer_display_types = read_ts_string_literals(
        viewer_source("src/types/lod-group.ts").read_text(encoding="utf-8"),
        "display_type",
    )
    assert viewer_display_types == set(lod_capable_types())


def test_partition_display_types_match_the_capability_table() -> None:
    """The viewer's metadata union names exactly partition-capable geometries."""
    viewer_display_types = read_ts_string_literals(
        viewer_source("src/types/partition-group.ts").read_text(encoding="utf-8"),
        "display_type",
    )
    assert viewer_display_types == set(partition_capable_types())


def test_monitor_display_types_match_specialized_group_metadata() -> None:
    """Monitor display types cover the union of specialized group geometries."""
    viewer_display_types = read_ts_string_literals(
        viewer_source("src/types/data-monitor-types.ts").read_text(encoding="utf-8"),
        "displayType",
    )
    assert viewer_display_types == set(lod_capable_types()) | set(
        partition_capable_types()
    )


def test_string_literal_parser_ignores_block_comments() -> None:
    """A commented declaration cannot shadow the live TypeScript property."""
    source = """/*
selector: 'wrong' | 'commented';
*/
selector: 'coverage' | 'screen-area';
"""
    assert read_ts_string_literals(source, "selector") == {
        "coverage",
        "screen-area",
    }


def test_string_literal_parser_rejects_ambiguous_declarations() -> None:
    """Two live declarations fail loudly instead of selecting the first one."""
    source = """selector: 'first' | 'value';
selector: 'coverage' | 'screen-area';
"""
    with pytest.raises(AssertionError, match="found 2"):
        read_ts_string_literals(source, "selector")


def test_string_literal_parser_rejects_repeated_members() -> None:
    """A duplicated member cannot collapse silently through set conversion."""
    with pytest.raises(AssertionError, match="repeats a member"):
        read_ts_string_literals("type X = 'a' | 'a';", "X")


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
