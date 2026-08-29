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

import numpy as np
import pytest

from luxar.demos import demo_gsplats_3d_h2afva_stack as demo
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.tree import GSplatLeaf, GSplatPartition


def _leaf_with_rungs(rungs=demo.EXPECTED_RUNGS, *, splats_per_rung=1):
    sublod = AdditiveSubLOD(
        centers=np.zeros((splats_per_rung, 3), dtype=np.float32),
        amplitudes=np.ones(splats_per_rung, dtype=np.float32),
        cholesky_factors=np.ones((splats_per_rung, 6), dtype=np.float32),
    )
    return GSplatLeaf(additive_sublods=[sublod] * rungs, meta={})


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


class TestTheRecomputeRecipeMatchesTheShippedPartition:
    def test_the_fit_is_laddered_and_the_partition_is_never_flattened(
        self, monkeypatch, tmp_path
    ):
        source = tmp_path / "source.zarr"
        source.mkdir()
        cal = tmp_path / "cal.json"
        cal.write_text("{}")
        calls = []
        monkeypatch.setattr(demo, "SOURCE_ARG", source)
        monkeypatch.setattr(demo, "CAL_ARG", cal)
        monkeypatch.setattr(demo, "_validate_cal", lambda _path: None)
        monkeypatch.setattr(demo, "_validate_source", lambda _path: None)
        monkeypatch.setattr(demo, "run_luxar_cli", lambda *args: calls.append(args))
        partition = GSplatPartition(
            children=[_leaf_with_rungs()] * demo.EXPECTED_PARTS,
            meta={},
        )
        monkeypatch.setattr(demo, "load_gsplat_node", lambda _path: (partition, {}))
        monkeypatch.setattr(
            demo,
            "_validate_rebuilt_archive",
            lambda _node: demo.EXPECTED_SPLATS,
        )

        result = demo.recompute_archive(tmp_path / "work")

        assert result.name == "h2afva_stack.gsplats.zarr"
        assert calls[0][-4:] == ("--recipe", "stream", "--n-lods", "6")
        assert calls[1][1] == "transform"
        assert calls[1][calls[1].index("--scale") + 1] == "4.0,1.0,1.0"
        assert calls[2][1] == "transform"
        assert calls[2][calls[2].index("--scale") + 1] == "0.40625,0.40625,0.40625"
        assert all(call[1] != "flatten" for call in calls)

    def test_a_flat_rebuild_is_rejected_before_scene_construction(self):
        with pytest.raises(RuntimeError, match="not a spatial partition"):
            demo._validate_rebuilt_archive(_leaf_with_rungs())

    def test_the_recorded_partition_shape_is_accepted(self):
        splats_per_rung = 2
        partition = GSplatPartition(
            children=[
                _leaf_with_rungs(splats_per_rung=splats_per_rung)
            ]
            * demo.EXPECTED_PARTS,
            meta={},
        )

        assert demo._validate_rebuilt_archive(partition) == (
            demo.EXPECTED_PARTS * demo.EXPECTED_RUNGS * splats_per_rung
        )
