"""``gsplat convert`` auto exposure: the opening view is not saturated.

Additive blending sums every splat along the view ray, so per-splat
normalisation alone opens a dense fit as a white blob. Convert predicts that sum
down the opening view axis and dims the node's ``display_gain`` — never
brightens — so its p99.9 is ``--target-brightness``. The amplitudes themselves
stay in the fit's data units (WP16), labelled by ``amplitude_unit``.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar import LuxarZarrCompiler
from luxar._zarr_compat import open_group as zc_open_group
from luxar._zarr_compat import read_node_attrs
from luxar.cli import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.core.dimension_inference import build_dimensions_from_data
from luxar.core.scene_exposure import _exposure_line
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

    assert "Exposure: display gain" in output
    assert "below per-splat normalisation" in output
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
    assert _exposure_line(plan, 0.5) == (
        "Exposure: nothing visible at the opening position; "
        "display gain 0.5 (per-splat normalisation)"
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
    assert _exposure_line(plan, 4e-4) == (
        "Exposure: nothing visible at the opening position; "
        "display gain 0.0001 (×0.25 below per-splat normalisation), "
        "capped by other timepoints"
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
    assert "(per-splat normalisation)" in output
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
    assert "Exposure: display gain" in output
    assert "below per-splat normalisation" in output
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


def test_volumetric_convert_reports_the_optical_depth_its_absorption_gives(
    tmp_path: Path,
) -> None:
    """No auto exposure runs for volumetric, so kappa needs the ray density.

    The narrated density is the predictor's opening-view p99.9 with every
    colour gain at 1 (tau = kappa * opacity * gained ray mass, colour-free), and
    the suggested kappa is its reciprocal.
    """
    import re

    src = _dense(tmp_path / "dense.gsplats.zarr")
    out = tmp_path / "scene.luxar.zarr"
    output = " ".join(
        _convert(
            CliRunner(),
            src,
            out,
            "--blending-mode",
            "volumetric",
            "--absorption",
            "0.5",
        ).split()
    )
    match = re.search(
        r"carry density ([0-9.e+]+), so --absorption 0.5 gives optical depth "
        r"~([0-9.e+]+) there .*--absorption ([0-9.e+-]+) would give ~1",
        output,
    )
    assert match, output
    density, tau, suggested = (float(g) for g in match.groups())
    assert tau == pytest.approx(0.5 * density, rel=0.01)
    assert suggested == pytest.approx(1.0 / density, rel=0.05)
    # A thick column far exceeds the additive target of 2: the old README
    # anchor ("brightest rays ~2") would have been off by this factor.
    assert density > 2.0


def test_volumetric_convert_suggests_the_additive_display_gain(
    tmp_path: Path,
) -> None:
    """Volumetric has no auto exposure, so convert names the gain additive uses.

    The suggested ``--display-gain`` is exactly what additive auto exposure
    stamps, and its kappa is the reciprocal of the density at that gain.
    """
    import re

    src = _dense(tmp_path / "dense.gsplats.zarr")
    _convert(CliRunner(), src, tmp_path / "additive.luxar.zarr")
    additive_gain = _gain(tmp_path / "additive.luxar.zarr")
    output = " ".join(
        _convert(
            CliRunner(),
            src,
            tmp_path / "vol.luxar.zarr",
            "--blending-mode",
            "volumetric",
        ).split()
    )
    density = float(re.search(r"carry density ([0-9.e+]+), so", output).group(1))
    match = re.search(
        r"--display-gain ([0-9.e+-]+) \(an absolute gain, not a multiplier\); "
        r"with it the brightest rays carry density ([0-9.e+-]+) and "
        r"--absorption ([0-9.e+-]+) gives ~1",
        output,
    )
    assert match, output
    gain, gained_density, kappa = (float(g) for g in match.groups())
    assert gain == pytest.approx(additive_gain, rel=0.05)
    volumetric_gain = _gain(tmp_path / "vol.luxar.zarr")
    assert gained_density == pytest.approx(
        density * additive_gain / volumetric_gain, rel=0.01
    )
    assert kappa == pytest.approx(1.0 / gained_density, rel=0.05)


def test_volumetric_convert_with_a_manual_gain_suggests_none(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    output = _convert(
        CliRunner(),
        src,
        tmp_path / "vol.luxar.zarr",
        "--blending-mode",
        "volumetric",
        "--display-gain",
        "1e-4",
    )
    assert "carry density" in output
    assert "Additive auto exposure would use" not in output


def test_additive_convert_prints_no_volumetric_line(tmp_path: Path) -> None:
    src = _dense(tmp_path / "dense.gsplats.zarr")
    output = _convert(CliRunner(), src, tmp_path / "scene.luxar.zarr")
    assert "Volumetric:" not in output


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


def test_unknown_colormap_reports_missing_install(tmp_path: Path, monkeypatch) -> None:
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {})
    out = tmp_path / "scene.luxar.zarr"
    real_import = builtins.__import__

    def without_optional_colormaps(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"matplotlib", "colorcet"}:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional_colormaps)
    result = CliRunner().invoke(
        app, ["gsplat", "convert", str(src), str(out), "--colormap", "notacolormap"]
    )
    output = normalized_cli_output(result)
    assert result.exit_code == 2, output
    assert "Unknown colormap 'notacolormap'" in output
    assert "pip install matplotlib colorcet" in output
    assert not out.exists()


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
            {
                "source_dtype": "uint16",
                "intensity_range": 3900.0,
                "image_min": 112.0,
                "floor": 112.0,
            },
            "counts above background",
            112.0,
        ),
        # `--floor none`: the subtracted level is the data minimum.
        (
            {"source_dtype": "uint16", "intensity_range": 3900.0, "image_min": 112.0},
            "counts above minimum",
            112.0,
        ),
        # A float source below zero still had its minimum subtracted.
        (
            {"source_dtype": "float32", "intensity_range": 2.5, "image_min": -0.75},
            "data units above minimum",
            -0.75,
        ),
        # A level negligible against the range is no offset.
        (
            {"source_dtype": "float32", "intensity_range": 2.5, "image_min": 1e-9},
            "data units",
            None,
        ),
        ({"source_dtype": "uint16", "intensity_range": 3900.0}, "counts", None),
        (
            {
                "source_dtype": "float32",
                "intensity_range": 2.5,
                "image_min": 0.25,
                "floor": 0.25,
            },
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


def test_merge_with_unbased_splats_has_no_count_unit(tmp_path: Path) -> None:
    based = _fit_like(
        tmp_path / "based.gsplats.zarr",
        {
            "source_dtype": "uint16",
            "image_min": 112.0,
            "image_max": 4012.0,
            "intensity_range": 3900.0,
        },
    )
    unbased = _fit_like(tmp_path / "unbased.gsplats.zarr", {})
    merged = tmp_path / "merged.gsplats.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "merge", str(based), str(unbased), "-o", str(merged)]
    )
    assert result.exit_code == 0, normalized_cli_output(result)
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), merged, out)
    attrs = read_node_attrs(out / "gsplats")
    assert attrs["amplitude_unit"] == "a.u."
    assert "source_data_range" not in attrs


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


def test_identity_intensity_is_not_restated(tmp_path: Path) -> None:
    """``--intensity 1`` authors no window, even under a large manual gain."""
    src = _fit_like(tmp_path / "fit.gsplats.zarr", {"intensity_range": 3900.0})
    out = tmp_path / "scene.luxar.zarr"
    _convert(CliRunner(), src, out, "--scale-intensity", "1000", "--intensity", "1")
    assert read_node_attrs(out / "gsplats")["intensity"] == 1.0


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
            "2",
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
        "floor": 100.0,
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
        ("--normalize-intensity", "1.0", "normalize"),
    ],
)
def test_transform_intensity_keeps_the_basis_honest(
    tmp_path: Path, flag: str, value: str, expect: Any
) -> None:
    """``--scale-intensity`` and ``--normalize-intensity`` (each one global
    factor) scale the amplitude-unit basis with the amplitudes and record the
    factor, so a converted scene says ``a.u.`` instead of claiming counts."""
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    stats = {
        "source_dtype": "uint16",
        "image_min": 112.0,
        "image_max": 4012.0,
        "intensity_range": 3900.0,
        "floor": 112.0,
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
    if expect == "normalize":
        scale = after["intensity_scale"]
        assert scale != pytest.approx(1.0)
        assert after["intensity_range"] == pytest.approx(3900.0 * scale)
        assert after["image_max"] == pytest.approx(4012.0)
        assert attrs["amplitude_unit"] == "a.u."
        assert "amplitude_offset" not in attrs
    else:
        for key, want in expect.items():
            assert after[key] == pytest.approx(want)
        assert after["intensity_scale"] == pytest.approx(0.5)
        assert attrs["amplitude_unit"] == "a.u."
        assert "amplitude_offset" not in attrs
        assert attrs["source_data_range"] == pytest.approx([0.0, 1950.0])


@pytest.mark.slow
def test_two_tile_batch_merge_keeps_the_intensity_basis(tmp_path: Path) -> None:
    """A 2-tile uint16 OME-Zarr batch merge records the shared basis, so
    ``gsplat convert`` labels its amplitudes ``counts`` rather than ``a.u.``."""
    import zarr
    from typer.testing import CliRunner

    from luxar._zarr_compat import read_node_attrs
    from luxar.cli import app
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )
    from luxar.gsplats.batch.local_runner import run_batch_local
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    rng = np.random.default_rng(3)
    V = np.full((16, 16, 32), 100.0, np.float32)
    zz, yy, xx = np.mgrid[0:16, 0:16, 0:32]
    for _ in range(8):
        cz, cy, cx = rng.integers(3, 13), rng.integers(3, 13), rng.integers(3, 29)
        V += 3000.0 * np.exp(-((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
    src = tmp_path / "vol.ome.zarr"
    root = zarr.open_group(str(src), mode="w")
    arr = root.create_array("0", shape=V.shape, chunks=(16, 16, 16), dtype="u2")
    arr[:] = V.astype(np.uint16)
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ],
        }
    ]
    out = tmp_path / "out"
    plan = plan_batch(
        input_path=src,
        output_dir=out,
        tiling="uniform",
        tile_size=16,
        tile_overlap=0,
        axes_list=["z", "y", "x"],
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(preset="draft", seeds="150", iters=25),
        denoise=DenoiseConfig(),
        content=ContentKnobs(),
        merge=MergeConfig(),
        merge_recipe_args={},
    )
    assert plan.manifest.total_tasks == 2
    final = run_batch_local(
        plan.manifest, out, gpus="cpu", jobs_per_gpu=2, verbose=False
    )
    node, stats = load_gsplat_node(final, include_stats=True)
    assert type(node).__name__ == "GSplatPartition"
    assert stats["source_dtype"] == "uint16"
    assert stats["intensity_range"] > 0
    scene = tmp_path / "scene.luxar.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "convert", str(final), str(scene), "--no-center"]
    )
    assert result.exit_code == 0, result.output
    leaf = read_node_attrs(scene / "gsplats" / "part_0")
    assert leaf["amplitude_unit"].startswith("counts")


def test_a_lod_convert_emits_the_aces_lut_notice_once(tmp_path: Path) -> None:
    """Each LOD level's leaf writer and the Node's group write both warned.

    The notice also names the CLI flag, since `gsplat convert` users have no
    `viewer_config` to edit.
    """
    import warnings

    src = _dense(tmp_path / "dense.gsplats.zarr")
    lod = tmp_path / "lod.gsplats.zarr"
    runner = CliRunner()
    result = runner.invoke(
        app, ["gsplat", "lod", str(src), str(lod), "--recipe", "levels"]
    )
    assert result.exit_code == 0, normalized_cli_output(result)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _convert(runner, lod, tmp_path / "scene.luxar.zarr", "--colormap", "plasma")
    notices = [w for w in record if "sets no tone_mapping" in str(w.message)]
    assert len(notices) == 1
    assert "--tone-mapping None" in str(notices[0].message)


def test_dense_view_hint_fires_only_above_the_dense_threshold() -> None:
    """A dense opening view points at volumetric; a sparse one says nothing."""
    from luxar.core.scene_exposure import (
        DENSE_PROJECTED_SUM,
        dense_view_hint,
    )

    assert dense_view_hint(DENSE_PROJECTED_SUM / 4) is None
    hint = dense_view_hint(DENSE_PROJECTED_SUM * 1.5)
    assert hint is not None
    assert "--blending-mode volumetric" in hint
    assert f"~{DENSE_PROJECTED_SUM * 1.5:.0f}" in hint


def test_python_recipe_gets_the_gain_convert_stamps(tmp_path: Path) -> None:
    """``auto_exposure_attrs`` is convert's exposure for the Python adders.

    The adders' own default (per-splat normalisation) opens this dense column as
    a white blob; the helper hands them convert's dimmed ``display_gain``, so a
    scene built from Python opens the way ``gsplat convert`` does.
    """
    from luxar import CameraConfig, Dimensions, ViewerConfig
    from luxar.core.scene_exposure import auto_exposure_attrs

    src = _dense(tmp_path / "dense.gsplats.zarr")
    converted = tmp_path / "convert.luxar.zarr"
    _convert(CliRunner(), src, converted)
    camera = CameraConfig(up=(0, -1, 0), view_axis="xy")
    order = ["z", "y", "x"]

    attrs = auto_exposure_attrs(src, dim_order=order, camera=camera)
    assert attrs == auto_exposure_attrs(
        src, dim_order=order, dimensions=Dimensions.default_3d(), camera=camera
    )
    side_camera = CameraConfig(view_axis="yz")
    assert auto_exposure_attrs(src, dim_order=order, camera=side_camera) == (
        auto_exposure_attrs(
            src, dim_order=order, dimensions=Dimensions.default_3d(), camera=side_camera
        )
    )
    assert attrs["normalize_amplitudes"] is False
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["projected_sum_axis"] == "z"
    assert attrs["display_gain"] == pytest.approx(_gain(converted), rel=1e-6)
    assert attrs["display_gain"] < _per_splat(src)

    out = tmp_path / "python.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(camera=camera),
        )
        scene.add_gsplats_from_file(
            "gsplats",
            src,
            dim_order=order,
            blending_mode="additive",
            fitted_defaults=False,
            **attrs,
        )
    stored = read_node_attrs(out / "gsplats")
    assert stored["display_gain"] == pytest.approx(attrs["display_gain"])
    # Data units kept: the stored amplitudes are the fit's, not rescaled.
    np.testing.assert_allclose(
        np.sort(_stored_amplitudes(out)),
        np.sort(GSplatData.load(src).amplitudes),
        rtol=1e-2,
    )


def test_python_recipe_keeps_per_splat_gain_off_summing_modes(tmp_path: Path) -> None:
    from luxar.core.scene_exposure import auto_exposure_attrs

    src = _dense(tmp_path / "dense.gsplats.zarr")
    attrs = auto_exposure_attrs(src, dim_order=["z", "y", "x"], blending_mode="max")
    assert attrs["amplitude_normalization_mode"] == "per_splat"
    assert attrs["display_gain"] == pytest.approx(_per_splat(src), rel=1e-3)


def test_python_recipe_takes_any_column_count_without_dimensions(
    tmp_path: Path,
) -> None:
    """With no ``dimensions=`` the layout is convert's own, so a 4-column
    (t, z, y, x) stack needs nothing more and gets convert's gain."""
    from luxar.core.scene_exposure import auto_exposure_attrs

    rng = np.random.default_rng(3)
    n = 800
    centers = np.column_stack(
        [
            rng.integers(0, 3, n).astype(np.float32),
            rng.uniform(0, 40, n),
            rng.normal(10, 2, n),
            rng.normal(10, 2, n),
        ]
    ).astype(np.float32)
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, [0, 2, 5, 9]] = [0.01, 1.5, 1.5, 1.5]
    src = tmp_path / "tl.gsplats.zarr"
    GSplatData(
        centers=centers,
        amplitudes=(rng.uniform(0.5, 1.0, n) * 500).astype(np.float32),
        cholesky_factors=chol,
        stats={"source_shape": [3, 40, 20, 20]},
    ).save(src)
    converted = tmp_path / "convert.luxar.zarr"
    _convert(CliRunner(), src, converted)
    attrs = auto_exposure_attrs(src)
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["display_gain"] == pytest.approx(_gain(converted), rel=1e-6)


def _labelled(path: Path) -> Path:
    """A uint16-sourced store with a fit basis: labelled ``counts above background``."""
    rng = np.random.default_rng(5)
    n = 400
    GSplatData(
        centers=np.column_stack(
            [rng.uniform(0, 40, n), rng.normal(10, 2, n), rng.normal(10, 2, n)]
        ).astype(np.float32),
        amplitudes=rng.uniform(200, 900, n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.5, 0, 1.5, 0, 0, 1.5], dtype=np.float32), (n, 1)
        ),
        stats={
            "source_dtype": "uint16",
            "intensity_range": 1000.0,
            "image_min": 100.0,
            "floor": 100.0,
            "source_shape": [40, 20, 20],
        },
    ).save(path, include_fitting_info=True)
    return path


def test_python_recipe_is_source_consistent(tmp_path: Path) -> None:
    """Path, node + stats and GSplatData + stats give the same gain AND labels;
    without stats an in-memory source keeps the gain, the labels degrade."""
    from luxar.core.scene_exposure import auto_exposure_attrs
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    src = _labelled(tmp_path / "labelled.gsplats.zarr")
    order = ["z", "y", "x"]
    node, stats = load_gsplat_node(src, include_stats=True)
    by_path = auto_exposure_attrs(src, dim_order=order)
    assert by_path["amplitude_unit"] == "counts above background"
    assert by_path["amplitude_offset"] == pytest.approx(100.0)
    for source in (node, GSplatData.load(src)):
        with_stats = auto_exposure_attrs(source, dim_order=order, stats=stats)
        assert with_stats == pytest.approx(by_path)
        bare = auto_exposure_attrs(source, dim_order=order)
        assert bare["display_gain"] == pytest.approx(by_path["display_gain"])
    assert auto_exposure_attrs(node, dim_order=order)["amplitude_unit"] == "a.u."
    assert (
        auto_exposure_attrs(GSplatData.load(src), dim_order=order)["amplitude_unit"]
        == "data units above background"
    )


def test_python_recipe_takes_a_partition_by_path(tmp_path: Path) -> None:
    from luxar.core.scene_exposure import auto_exposure_attrs

    src = _labelled(tmp_path / "flat.gsplats.zarr")
    part = _partition(CliRunner(), src, tmp_path / "part.gsplats.zarr")
    attrs = auto_exposure_attrs(part, dim_order=["z", "y", "x"])
    flat = auto_exposure_attrs(src, dim_order=["z", "y", "x"])
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["display_gain"] == pytest.approx(flat["display_gain"], rel=0.05)
    # Exactly the gain convert stamps on the grafted partition.
    converted = tmp_path / "convert.luxar.zarr"
    _convert(CliRunner(), part, converted)
    assert auto_exposure_attrs(part)["display_gain"] == pytest.approx(
        _gain(converted), rel=1e-6
    )


def test_python_recipe_falls_back_when_a_hidden_dim_has_no_column(
    tmp_path: Path,
) -> None:
    from luxar import Dimension, Dimensions
    from luxar.core.scene_exposure import auto_exposure_attrs

    src = _dense(tmp_path / "dense.gsplats.zarr")
    dims = Dimensions(
        [
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
            Dimension("t", display=False, discrete=True, range=(0, 3), step=1),
        ]
    )
    with pytest.warns(UserWarning, match="no center column"):
        attrs = auto_exposure_attrs(src, dim_order=["z", "y", "x"], dimensions=dims)
    assert attrs["amplitude_normalization_mode"] == "per_splat"
    assert attrs["display_gain"] == pytest.approx(_per_splat(src), rel=1e-3)


def test_max_blending_warns_on_a_tiled_fit(tmp_path: Path) -> None:
    """`max` over Hann-blended tiles shows seams; convert says so (only then)."""
    from luxar.cli.gsplat_ops.scene_build import max_blending_seam_warning

    src = _dense(tmp_path / "in.gsplats.zarr")
    plain = _convert(
        CliRunner(), src, tmp_path / "a.luxar.zarr", "--blending-mode", "max"
    )
    assert "dark seams" not in plain
    data = GSplatData.load(src)
    data.stats["tiled_fitting"] = True
    tiled = tmp_path / "tiled.gsplats.zarr"
    data.save(tiled)
    output = _convert(
        CliRunner(), tiled, tmp_path / "b.luxar.zarr", "--blending-mode", "max"
    )
    assert "dark seams" in output and "--tiling none" in output
    additive = _convert(CliRunner(), tiled, tmp_path / "c.luxar.zarr")
    assert "dark seams" not in additive
    # content tiling keeps whole splats per core box; a partition cut from ONE
    # fit records that fit's iterations; an unmarked partition is conditional.
    assert (
        max_blending_seam_warning("max", {"planned_fit": True}, partition=True) is None
    )
    assert max_blending_seam_warning("max", {"iterations": 10}, partition=True) is None
    hedged = max_blending_seam_warning("max", {}, partition=True)
    assert hedged is not None and hedged.startswith("⚠ --blending-mode max: if this")
