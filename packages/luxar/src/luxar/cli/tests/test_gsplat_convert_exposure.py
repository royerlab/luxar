"""``gsplat convert`` auto exposure: the opening view is not saturated.

Additive blending sums every splat along the view ray, so per-splat
normalisation alone opens a dense fit as a white blob. Convert predicts that sum
down the opening view axis and dims the amplitudes — never brightens — so its
p99.9 is ``--target-brightness``, stamping the exact raw→stored factor so the
stored amplitudes divide back to the fit's own values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar import LuxarZarrCompiler
from luxar._zarr_compat import open_group as zc_open_group
from luxar._zarr_compat import read_node_attrs
from luxar.cli import app
from luxar.cli.gsplat_ops.scene_exposure import _exposure_line
from luxar.cli.tests._testing import normalized_cli_output
from luxar.core.dimension_inference import build_dimensions_from_data
from luxar.gsplats.exposure import ExposurePlan, HiddenAxis, plan_projected_exposure
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.tree import iter_leaves
from luxar.io._compiler.gsplat_tree import read_gsplat_node

_EXPOSURE_ATTRS = (
    "amplitude_normalization_mode",
    "projected_sum_p999",
    "projected_sum_target",
    "projected_sum_axis",
)


def _dense(path: Path, *, provenance: bool = True, scale: float = 500.0) -> Path:
    """A thick column of overlapping splats: saturates when summed down z."""
    rng = np.random.default_rng(7)
    n = 600
    centers = np.column_stack(
        [rng.uniform(0, 40, n), rng.normal(10, 2, n), rng.normal(10, 2, n)]
    ).astype(np.float32)
    data = GSplatData(
        centers=centers,
        amplitudes=(rng.uniform(0.5, 1.0, n) * scale).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.5, 0, 1.5, 0, 0, 1.5], dtype=np.float32), (n, 1)
        ),
        stats={"source_shape": [40, 20, 20]} if provenance else {},
    )
    data.save(path)
    return path


def _convert(runner: CliRunner, src: Path, out: Path, *extra: str) -> str:
    result = runner.invoke(
        app, ["gsplat", "convert", str(src), str(out), "--no-center", *extra]
    )
    output = normalized_cli_output(result)
    assert result.exit_code == 0, output
    return output


def _stored_amplitudes(out: Path, name: str = "gsplats") -> np.ndarray:
    root = zc_open_group(str(out), mode="r")
    node = read_gsplat_node(root[name], root)
    return np.concatenate(
        [
            np.asarray(s.amplitudes)
            for leaf in iter_leaves(node)
            for s in leaf.additive_sublods
        ]
    )


def _partition(runner: CliRunner, src: Path, dst: Path) -> Path:
    result = runner.invoke(
        app, ["gsplat", "partition", str(src), str(dst), "--parts", "2"]
    )
    assert result.exit_code == 0, normalized_cli_output(result)
    return dst


def test_default_dims_a_saturating_fit_and_stamps_the_prediction(
    tmp_path: Path,
) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out)

    assert "Exposure: scaled splat brightness" in output
    assert "--no-auto-exposure to disable" in output
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["projected_sum_target"] == 2.0
    assert attrs["projected_sum_p999"] == pytest.approx(2.0)
    # The fit's columns are (z, y, x); the image-convention camera looks down z.
    assert attrs["projected_sum_axis"] == "z"
    raw = GSplatData.load(src).amplitudes
    per_splat = 1.0 / float(np.percentile(raw, 99.9))
    assert attrs["amplitude_normalization_factor"] < per_splat


def test_centered_hidden_axis_uses_the_written_coordinates(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 2000
    centers = np.column_stack(
        [
            rng.normal(0, 6, n),
            rng.normal(0, 6, n),
            rng.normal(0, 6, n),
            rng.normal(150, 10, n),
        ]
    ).astype(np.float32)
    src = tmp_path / "four_dimensional.gsplats.zarr"
    GSplatData(
        centers=centers,
        amplitudes=np.full(n, 500, dtype=np.float32),
        cholesky_factors=np.tile(
            np.array([1.5, 0, 1.5, 0, 0, 1.5, 0, 0, 0, 1.5], dtype=np.float32),
            (n, 1),
        ),
    ).save(src)
    runner = CliRunner()
    centered = tmp_path / "centered.luxar.zarr"
    uncentered = tmp_path / "uncentered.luxar.zarr"
    result = runner.invoke(app, ["gsplat", "convert", str(src), str(centered)])
    assert result.exit_code == 0, normalized_cli_output(result)
    _convert(runner, src, uncentered)

    got = read_node_attrs(centered / "gsplats")
    reference = read_node_attrs(uncentered / "gsplats")
    assert got["projected_sum_p999"] > 0
    assert got["amplitude_normalization_factor"] < 1 / 500
    assert got["projected_sum_p999"] == pytest.approx(
        reference["projected_sum_p999"], rel=1e-4
    )
    assert got["amplitude_normalization_factor"] == pytest.approx(
        reference["amplitude_normalization_factor"], rel=1e-4
    )


def test_empty_opening_prediction_is_reported_as_empty() -> None:
    centers = np.array([[0, 0, 0, 100]], dtype=np.float32)
    chol = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float32)
    plan = plan_projected_exposure(
        centers,
        chol,
        np.array([1], dtype=np.float32),
        view_axis=0,
        display_dims=[0, 1, 2],
        hidden=[HiddenAxis(3, 0, False, 0, (0, 100))],
    )
    assert plan.predicted_before == 0
    assert _exposure_line(plan) == (
        "Exposure: nothing visible at the opening position; brightness unchanged"
    )


def test_empty_opening_prediction_can_still_be_capped_elsewhere() -> None:
    plan = ExposurePlan(
        factor=0.25,
        predicted_before=0.0,
        predicted_after=0.0,
        target=2.0,
        side_p999=32.0,
        side_capped=True,
        cap_source="other timepoints",
    )
    assert _exposure_line(plan) == (
        "Exposure: nothing visible at the opening position; "
        "scaled splat brightness ×0.25, capped by other timepoints"
    )


def test_stored_amplitudes_divide_back_to_the_fit(tmp_path: Path) -> None:
    """Round trip: stored / amplitude_normalization_factor == raw amplitudes."""
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out)
    factor = read_node_attrs(out / "gsplats")["amplitude_normalization_factor"]
    raw = np.sort(GSplatData.load(src).amplitudes.astype(np.float64))
    restored = np.sort(_stored_amplitudes(out).astype(np.float64) / factor)
    np.testing.assert_allclose(restored, raw, rtol=1e-4)


def test_never_brightens_a_dim_fit(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--target-brightness", "1e9")
    assert "brightness unchanged" in output
    attrs = read_node_attrs(out / "gsplats")
    raw = GSplatData.load(src).amplitudes
    assert attrs["amplitude_normalization_factor"] == pytest.approx(
        1.0 / float(np.percentile(raw, 99.9)), rel=1e-6
    )


def _legacy_scene_attrs(src: Path, out: Path) -> Dict[str, Any]:
    """What convert wrote before auto exposure, for a positional store."""
    data = GSplatData.load(src)
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(
            dimensions=build_dimensions_from_data(data.centers, None)
        )
        scene.add_gsplats_from_data(
            name="gsplats",
            result=data,
            normalize_amplitudes=True,
            opacity=1.0,
            absorption=1.0,
            blending_mode="additive",
            layer=True,
        )
    return read_node_attrs(out / "gsplats")


def test_no_auto_exposure_reproduces_the_legacy_attrs(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr", provenance=False)
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--no-auto-exposure")
    assert "Exposure:" not in output
    legacy = _legacy_scene_attrs(src, tmp_path / "legacy.luxar.zarr")
    assert read_node_attrs(out / "gsplats") == legacy
    assert not set(_EXPOSURE_ATTRS) & set(legacy)


def test_partition_is_auto_exposed_with_one_factor(tmp_path: Path) -> None:
    runner = CliRunner()
    part = _partition(
        runner, _dense(tmp_path / "dense.gsplats.zarr"), tmp_path / "p.gsplats.zarr"
    )
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(runner, part, out)
    assert "Exposure: scaled splat brightness" in output
    factors = set()
    for i in range(2):
        attrs = read_node_attrs(out / "gsplats" / f"part_{i}")
        assert attrs["amplitude_normalization_mode"] == "projected"
        assert attrs["projected_sum_p999"] == pytest.approx(2.0)
        factors.add(attrs["amplitude_normalization_factor"])
    assert len(factors) == 1


def test_scale_intensity_now_scales_a_partition(tmp_path: Path) -> None:
    runner = CliRunner()
    src = _dense(tmp_path / "dense.gsplats.zarr")
    part = _partition(runner, src, tmp_path / "p.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(runner, part, out, "--scale-intensity", "0.25")
    assert "ignored" not in output
    assert "manual ×0.25" in output
    raw = np.sort(GSplatData.load(src).amplitudes.astype(np.float64))
    # The partition parts re-encode their amplitudes (uint8 within each part's
    # own bounds), so equality holds to that quantization, not to float32.
    np.testing.assert_allclose(
        np.sort(_stored_amplitudes(out).astype(np.float64)), raw * 0.25, rtol=5e-3
    )
    attrs = read_node_attrs(out / "gsplats" / "part_0")
    assert attrs["amplitude_normalization_mode"] == "manual"
    assert attrs["amplitude_normalization_factor"] == 0.25


def test_non_summing_blending_keeps_per_splat_normalization(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--blending-mode", "volumetric")
    assert "keeps per-splat normalization" in output
    attrs = read_node_attrs(out / "gsplats")
    assert "amplitude_normalization_mode" not in attrs
    raw = GSplatData.load(src).amplitudes
    assert attrs["amplitude_normalization_factor"] == pytest.approx(
        1.0 / float(np.percentile(raw, 99.9)), rel=1e-6
    )


def test_side_cap_is_named_when_it_binds(tmp_path: Path) -> None:
    """A slab seen face-on is dim but blazes edge-on: the side views decide."""
    rng = np.random.default_rng(3)
    n = 3000
    centers = np.column_stack(
        [rng.normal(5, 0.5, n), rng.uniform(0, 60, n), rng.uniform(0, 60, n)]
    ).astype(np.float32)
    src = tmp_path / "slab.gsplats.zarr"
    GSplatData(
        centers=centers,
        amplitudes=(rng.uniform(0.5, 1.0, n) * 300).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
        ),
        stats={"source_shape": [10, 60, 60]},
    ).save(src)
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out)
    assert "capped by side views" in output
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["projected_sum_p999"] < 2.0


@pytest.mark.parametrize(
    ("tone", "target"), [("None", 0.9), ("Linear", 0.9), ("AgX", 2.0)]
)
def test_default_target_follows_the_tone_mapping(
    tmp_path: Path, tone: str, target: float
) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out, "--tone-mapping", tone)
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["projected_sum_target"] == target
    assert attrs["projected_sum_p999"] == pytest.approx(target)


def test_explicit_target_beats_the_tone_mapping_default(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    _convert(
        CliRunner(), src, out, "--tone-mapping", "None", "--target-brightness", "1.5"
    )
    assert read_node_attrs(out / "gsplats")["projected_sum_target"] == 1.5


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_scale_intensity_is_a_usage_error(
    tmp_path: Path, value: str
) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "convert",
            str(src),
            str(tmp_path / "s.luxar.zarr"),
            "--scale-intensity",
            value,
        ],
    )
    assert result.exit_code == 2
    assert "positive" in normalized_cli_output(result)
    assert not (tmp_path / "s.luxar.zarr").exists()
