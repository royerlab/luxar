"""Tests for the h2afva single-stack demo's recompute path.

The interesting logic is the calibration check. Five h2afva calibrations exist on
the acquisition box, they differ only in the splat-density block that sizes every
content box, and handing the fit the wrong one yields a plausible archive built
to the wrong budget -- a failure visible only as a splat count nobody checks.
Two of the five each match the recorded fit on ONE of the two density attributes,
so a check on either attribute alone accepts a wrong cal. These tests pin that
both are required.
"""

import json

import pytest

from luxar.demos import demo_gsplats_3d_h2afva_stack as demo


def _cal(tmp_path, exponent, cap, name="cal.json"):
    path = tmp_path / name
    payload = {"splat_density": {}}
    if exponent is not None:
        payload["splat_density"]["saturation_exponent"] = exponent
    if cap is not None:
        payload["splat_density"]["saturation_cap"] = cap
    path.write_text(json.dumps(payload))
    return path


class TestTheRecordedCalIsAccepted:
    def test_the_exact_recorded_density_passes(self, tmp_path):
        path = _cal(tmp_path, demo.CAL_SATURATION_EXPONENT, demo.CAL_SATURATION_CAP)
        demo._validate_cal(path)  # must not raise

    def test_extra_precision_in_the_exponent_still_passes(self, tmp_path):
        # The real file carries 0.5956732209523588; the constant is rounded to
        # 4 dp because the fit log rounds for display. Full precision must pass.
        path = _cal(tmp_path, 0.59570000123, demo.CAL_SATURATION_CAP)
        demo._validate_cal(path)


class TestASiblingCalMatchingOneAttributeIsRefused:
    """The two near-misses that actually exist on the box."""

    def test_right_exponent_wrong_cap_is_refused(self, tmp_path):
        # cal_h2afva_t126_gain.json: exponent 0.5957, cap 218435.
        path = _cal(tmp_path, demo.CAL_SATURATION_EXPONENT, 218435)
        with pytest.raises(SystemExit) as exc:
            demo._validate_cal(path)
        assert "218435" in str(exc.value)

    def test_right_cap_wrong_exponent_is_refused(self, tmp_path):
        # cal_h2afva_t252_gain.json: exponent 1.4190592760855256, cap 233937.
        path = _cal(tmp_path, 1.4190592760855256, demo.CAL_SATURATION_CAP)
        with pytest.raises(SystemExit) as exc:
            demo._validate_cal(path)
        assert "1.4191" in str(exc.value)


class TestACalWithoutADensityIsRefusedAsUnusable:
    """Two of the five candidates carry no saturation figures at all.

    Absent is not the same as wrong, and it must not be read as a zero that
    happens to mismatch -- the message has to say the cal cannot have planned
    this fit, so the operator looks for a different file.
    """

    def test_a_missing_exponent_is_refused(self, tmp_path):
        path = _cal(tmp_path, None, demo.CAL_SATURATION_CAP)
        with pytest.raises(SystemExit, match="no splat_density"):
            demo._validate_cal(path)

    def test_a_missing_cap_is_refused(self, tmp_path):
        path = _cal(tmp_path, demo.CAL_SATURATION_EXPONENT, None)
        with pytest.raises(SystemExit, match="no splat_density"):
            demo._validate_cal(path)

    def test_a_cal_with_no_density_block_at_all_is_refused(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"k_star": 118715}))
        with pytest.raises(SystemExit, match="no splat_density"):
            demo._validate_cal(path)


class TestTheRecipeConstantsMatchTheRecordedRun:
    """The recipe was recovered from run logs that are the only copy.

    These are not tautologies against the source: each value is transcribed from
    a specific logged line, and a future edit that "tidies" one of them silently
    rebuilds a different archive. The splat count is the load-bearing one --
    the decimation study cuts its four levels from this archive, so a drift here
    relabels all four.
    """

    def test_the_source_array_key_is_the_qualified_one(self):
        # The store holds sibling arrays; an unqualified open picks the wrong one.
        assert demo.SOURCE_ARRAY_KEY == "h2afva/fused"

    def test_the_recording_extent_is_pinned(self):
        assert demo.SOURCE_TIMEPOINTS == 253
        assert demo.SOURCE_SPATIAL == (407, 2048, 2048)

    def test_the_demo_timepoint_is_the_scanned_one(self):
        assert demo.TIMEPOINT == 234

    def test_the_fit_is_content_tiled(self):
        # Uniform tiling would plan a different box set entirely.
        assert demo.TILING == "content"

    def test_the_anisotropy_scale_is_z_four(self):
        assert demo.VOXEL_SCALE == (4.0, 1.0, 1.0)

    def test_the_expected_count_matches_the_decimation_parent(self):
        assert demo.EXPECTED_SPLATS == 1_653_405
