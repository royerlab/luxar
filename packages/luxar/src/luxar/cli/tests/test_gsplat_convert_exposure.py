"""``gsplat convert`` auto exposure: the opening view is not saturated.

Additive blending sums every splat along the view ray, so per-splat
normalisation alone opens a dense fit as a white blob. Convert predicts that sum
down the opening view axis and dims the node's ``display_gain`` — never
brightens — so its p99.9 is ``--target-brightness``. The amplitudes themselves
stay in the fit's data units (WP16), labelled by ``amplitude_unit``.
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


def _gain(out: Path) -> float:
    """The ``display_gain`` convert stamped (on the leaf, or a wrapper group)."""
    return float(read_node_attrs(out / "gsplats")["display_gain"])


def _per_splat(src: Path) -> float:
    """The legacy per-splat normalisation factor of the store's amplitudes."""
    return 1.0 / float(np.percentile(GSplatData.load(src).amplitudes, 99.9))


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
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["display_gain"] < _per_splat(src)
    # The data are not rescaled, so there is no raw→stored factor to record.
    assert "amplitude_normalization_factor" not in attrs


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
    assert got["display_gain"] < 1 / 500
    assert got["projected_sum_p999"] == pytest.approx(
        reference["projected_sum_p999"], rel=1e-4
    )
    assert got["display_gain"] == pytest.approx(reference["display_gain"], rel=1e-4)


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


def test_scene_keeps_the_fit_amplitudes_in_data_units(tmp_path: Path) -> None:
    """The scene stores the fit's own amplitudes; brightness is the gain."""
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out)
    raw = np.sort(GSplatData.load(src).amplitudes.astype(np.float64))
    stored = np.sort(_stored_amplitudes(out).astype(np.float64))
    np.testing.assert_allclose(stored, raw, rtol=1e-4)
    attrs = read_node_attrs(out / "gsplats")
    # The window the Layers panel opens on is in data units too.
    lo, hi = attrs["amplitude_data_range"]
    assert hi == pytest.approx(float(np.percentile(raw, 99.9)), rel=1e-3)
    assert 0.0 < attrs["display_gain"] < 1.0


def test_never_brightens_a_dim_fit(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--target-brightness", "1e9")
    assert "brightness unchanged" in output
    assert _gain(out) == pytest.approx(_per_splat(src), rel=1e-6)


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


def test_no_auto_exposure_gain_reproduces_the_legacy_render(tmp_path: Path) -> None:
    """``--no-auto-exposure``: data units, gain = the legacy per-splat factor.

    The viewer multiplies the gain into every amplitude-linear term, so
    ``stored × gain`` is what the legacy (normalised) scene stored, and its
    colormap window is the legacy one in data units.
    """
    src = _dense(tmp_path / "dense.gsplats.zarr", provenance=False)
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--no-auto-exposure")
    assert "Exposure:" not in output
    legacy = _legacy_scene_attrs(src, tmp_path / "legacy.luxar.zarr")
    attrs = read_node_attrs(out / "gsplats")
    factor = legacy["amplitude_normalization_factor"]
    assert attrs["display_gain"] == pytest.approx(factor, rel=1e-6)
    assert attrs["amplitude_normalization_mode"] == "per_splat"
    assert attrs["amplitude_unit"] == "a.u."
    np.testing.assert_allclose(
        np.asarray(attrs["amplitude_data_range"]) * attrs["display_gain"],
        legacy["amplitude_data_range"],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.sort(_stored_amplitudes(out)) * attrs["display_gain"],
        np.sort(_stored_amplitudes(tmp_path / "legacy.luxar.zarr")),
        rtol=1e-3,
    )
    assert not {"projected_sum_p999", "projected_sum_target"} & set(attrs)


def test_partition_is_auto_exposed_with_one_factor(tmp_path: Path) -> None:
    runner = CliRunner()
    part = _partition(
        runner, _dense(tmp_path / "dense.gsplats.zarr"), tmp_path / "p.gsplats.zarr"
    )
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(runner, part, out)
    assert "Exposure: scaled splat brightness" in output
    # ONE gain, on the kind=partition wrapper (``display_gain`` composes), so
    # every part renders at the same exposure; no part carries its own.
    assert read_node_attrs(out / "gsplats")["kind"] == "partition"
    assert 0.0 < _gain(out) < 1.0
    for i in range(2):
        attrs = read_node_attrs(out / "gsplats" / f"part_{i}")
        assert attrs["amplitude_normalization_mode"] == "projected"
        assert attrs["projected_sum_p999"] == pytest.approx(2.0)
        assert "display_gain" not in attrs


def test_scale_intensity_now_scales_a_partition(tmp_path: Path) -> None:
    runner = CliRunner()
    src = _dense(tmp_path / "dense.gsplats.zarr")
    part = _partition(runner, src, tmp_path / "p.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(runner, part, out, "--scale-intensity", "0.25")
    assert "ignored" not in output
    assert "manual gain ×0.25" in output
    raw = np.sort(GSplatData.load(src).amplitudes.astype(np.float64))
    # The partition parts re-encode their amplitudes (uint8 within each part's
    # own bounds), so equality holds to that quantization, not to float32.
    np.testing.assert_allclose(
        np.sort(_stored_amplitudes(out).astype(np.float64)), raw, rtol=5e-3
    )
    assert _gain(out) == 0.25
    attrs = read_node_attrs(out / "gsplats" / "part_0")
    assert attrs["amplitude_normalization_mode"] == "manual"


def test_non_summing_blending_keeps_per_splat_normalization(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = _convert(CliRunner(), src, out, "--blending-mode", "volumetric")
    assert "keeps per-splat normalization" in output
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["amplitude_normalization_mode"] == "per_splat"
    assert attrs["display_gain"] == pytest.approx(_per_splat(src), rel=1e-6)


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


def _fit_like(path: Path, stats: Dict[str, Any], *, colors: bool = False) -> Path:
    """A small fit-shaped store whose root stats carry ``stats``."""
    rng = np.random.default_rng(5)
    n = 200
    GSplatData(
        centers=rng.uniform(0, 20, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(100.0, 4000.0, n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
        ),
        colors=rng.uniform(0, 1, (n, 3)).astype(np.float32) if colors else None,
        stats={"source_shape": [20, 20, 20], **stats},
    ).save(path)
    return path


@pytest.mark.parametrize(
    ("stats", "unit", "offset"),
    [
        (
            {"source_dtype": "uint16", "intensity_range": 3900.0, "image_min": 112.0},
            "counts above background",
            112.0,
        ),
        ({"source_dtype": "uint16", "intensity_range": 3900.0}, "counts", None),
        (
            {"source_dtype": "float32", "intensity_range": 2.5, "image_min": 0.25},
            "data units above background",
            0.25,
        ),
        ({"source_dtype": "uint16"}, "a.u.", None),
        ({}, "a.u.", None),
    ],
)
def test_unit_labels_follow_the_fit_basis(
    tmp_path: Path, stats: Dict[str, Any], unit: str, offset: Any
) -> None:
    src = _fit_like(tmp_path / "fit.gsplats.zarr", stats)
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out)
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["amplitude_unit"] == unit
    if offset is None:
        assert "amplitude_offset" not in attrs
    else:
        assert attrs["amplitude_offset"] == pytest.approx(offset)


def test_colormap_intensity_window_is_restated_in_data_units(tmp_path: Path) -> None:
    """``--intensity I`` windowed the gained amplitudes ``[0, 1/I]``; now ``I·g``."""
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {"intensity_range": 3900.0})
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out, "--colormap", "viridis", "--intensity", "2")
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["intensity"] == pytest.approx(2.0 * attrs["display_gain"])


def test_direct_colour_intensity_is_a_unitless_gain(tmp_path: Path) -> None:
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {}, colors=True)
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out, "--intensity", "2")
    assert read_node_attrs(out / "gsplats")["intensity"] == 2.0


def test_unreachable_intensity_window_is_a_usage_error(tmp_path: Path) -> None:
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {})
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "convert",
            str(src),
            str(tmp_path / "s.luxar.zarr"),
            "--scale-intensity",
            "1000",
            "--intensity",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "Layers panel" in normalized_cli_output(result)


def test_source_data_range_is_the_datas_span_not_the_splat_window(
    tmp_path: Path,
) -> None:
    """``source_data_range`` = [0, image_max - image_min]: the DATA's span above
    the floor, which overlapping Gaussians' peak amplitudes do not reach."""
    stats = {
        "source_dtype": "uint16",
        "image_min": 100.0,
        "image_max": 12100.0,
        "intensity_range": 12000.0,
    }
    src = _fit_like(tmp_path / "fit.gsplats.zarr", stats)
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out)
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["source_data_range"] == [0.0, 12000.0]
    assert attrs["amplitude_offset"] == 100.0
    assert attrs["amplitude_data_range"][1] < attrs["source_data_range"][1]


def test_no_basis_no_source_data_range(tmp_path: Path) -> None:
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {})
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out)
    assert "source_data_range" not in read_node_attrs(out / "gsplats")


@pytest.mark.parametrize(
    ("flag", "value", "expect"),
    [
        ("--scale-intensity", "0.5", {"intensity_range": 1950.0, "image_max": 4012.0}),
        ("--normalize-intensity", "1.0", None),
    ],
)
def test_transform_intensity_keeps_the_basis_honest(
    tmp_path: Path, flag: str, value: str, expect: Any
) -> None:
    """``--scale-intensity`` scales the amplitude-unit basis with the
    amplitudes; ``--normalize-intensity`` (an arbitrary target) drops it, so a
    converted scene says ``a.u.`` instead of claiming counts."""
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    stats = {
        "source_dtype": "uint16",
        "image_min": 112.0,
        "image_max": 4012.0,
        "intensity_range": 3900.0,
    }
    src = _fit_like(tmp_path / "fit.gsplats.zarr", stats)
    dst = tmp_path / "t.gsplats.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "transform", str(src), str(dst), flag, value]
    )
    assert result.exit_code == 0, normalized_cli_output(result)
    _, after = load_gsplat_node(dst, include_stats=True)
    assert after["image_min"] == pytest.approx(112.0)
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), dst, out)
    attrs = read_node_attrs(out / "gsplats")
    if expect is None:
        assert "intensity_range" not in after and "image_max" not in after
        assert attrs["amplitude_unit"] == "a.u."
    else:
        for key, want in expect.items():
            assert after[key] == pytest.approx(want)
        assert after["intensity_scale"] == pytest.approx(0.5)
        assert attrs["amplitude_unit"] == "a.u."
        assert "amplitude_offset" not in attrs
        assert attrs["source_data_range"] == pytest.approx([0.0, 1950.0])
