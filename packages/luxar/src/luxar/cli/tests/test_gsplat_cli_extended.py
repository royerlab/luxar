"""Tests for the new gsplat CLI commands: fit, convert, render, merge.

Also tests the config system (presets, YAML loading, dump) and volume loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_config import (
    PRESETS,
    build_dimensions_from_data,
    dump_default_config,
    load_fit_config,
    load_volume,
    parse_hex_color,
    parse_seeds,
    parse_shape,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_gsplats(tmp_path: Path) -> Path:
    """Create a small .gsplats.zarr for testing (no fitting required)."""
    from luxar.gsplats.gsplat_data import GSplatData

    n, d = 5, 3
    data = GSplatData(
        centers=np.random.rand(n, d).astype(np.float32) * 10,
        amplitudes=np.random.rand(n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
        ),
    )
    out = tmp_path / "test.gsplats.zarr"
    data.save(out)
    return out


@pytest.fixture
def small_volume_npy(tmp_path: Path) -> Path:
    """Create a small 16^3 .npy volume for testing."""
    volume = np.random.rand(16, 16, 16).astype(np.float32) * 0.5
    # Add a bright blob so fitting finds something
    volume[6:10, 6:10, 6:10] = 1.0
    path = tmp_path / "volume.npy"
    np.save(str(path), volume)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Config system tests
# ═══════════════════════════════════════════════════════════════════════


class TestPresets:
    def test_all_presets_exist(self) -> None:
        assert "draft" in PRESETS
        assert "standard" in PRESETS
        assert "hifi" in PRESETS

    def test_presets_have_n_iters(self) -> None:
        for name, preset in PRESETS.items():
            assert "n_iters" in preset, f"Preset '{name}' missing n_iters"

    def test_draft_is_fastest(self) -> None:
        assert PRESETS["draft"]["n_iters"] < PRESETS["standard"]["n_iters"]
        assert PRESETS["standard"]["n_iters"] < PRESETS["hifi"]["n_iters"]


class TestLoadFitConfig:
    def test_defaults_only(self) -> None:
        config = load_fit_config()
        assert "n_iters" in config
        assert "lr" in config

    def test_preset_overrides_defaults(self) -> None:
        config = load_fit_config(preset="draft")
        assert config["n_iters"] == 2000

    def test_yaml_overrides_preset(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("n_iters: 42\n")
        config = load_fit_config(preset="standard", config_path=yaml_path)
        assert config["n_iters"] == 42

    def test_cli_overrides_everything(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("n_iters: 42\n")
        config = load_fit_config(
            preset="hifi",
            config_path=yaml_path,
            cli_overrides={"n_iters": 7},
        )
        assert config["n_iters"] == 7

    def test_none_cli_values_ignored(self) -> None:
        """Audit W3 fix: pin n_iters > 0 instead of just `is not None`.
        A mutation that defaulted to 0 would pass the truthiness check
        but be a real bug.
        """
        config = load_fit_config(cli_overrides={"n_iters": None, "lr": 0.05})
        assert config["lr"] == 0.05
        # n_iters should be the function default, not None — and the
        # default must be a positive iteration count.
        assert isinstance(config["n_iters"], int)
        assert config["n_iters"] > 0

    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            load_fit_config(preset="nonexistent")

    def test_passes_through_seed_kwargs(self, tmp_path: Path) -> None:
        """Seed generation kwargs (e.g., num_scales) must pass through,
        not be silently dropped by a whitelist filter."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("n_iters: 100\nnum_scales: 4\npercentile_thresh: 80\n")
        config = load_fit_config(config_path=yaml_path)
        assert config["n_iters"] == 100
        assert config["num_scales"] == 4
        assert config["percentile_thresh"] == 80

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        config = load_fit_config(config_path=yaml_path)
        assert "n_iters" in config

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_fit_config(config_path=yaml_path)

    def test_seeds_excluded_from_config(self) -> None:
        """seeds is handled separately, must not appear in config."""
        config = load_fit_config()
        assert "seeds" not in config


class TestDumpDefaultConfig:
    def test_valid_yaml(self) -> None:
        text = dump_default_config("standard")
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)

    def test_preset_reflected(self) -> None:
        text = dump_default_config("draft")
        parsed = yaml.safe_load(text)
        assert parsed["n_iters"] == 2000

    def test_all_presets_produce_valid_yaml(self) -> None:
        for preset_name in PRESETS:
            text = dump_default_config(preset_name)
            parsed = yaml.safe_load(text)
            assert isinstance(parsed, dict), f"Preset '{preset_name}' dump failed"


class TestParsers:
    def test_parse_seeds_none(self) -> None:
        assert parse_seeds(None) is None

    def test_parse_seeds_auto(self) -> None:
        assert parse_seeds("auto") is None

    def test_parse_seeds_int(self) -> None:
        assert parse_seeds("8000") == 8000

    def test_parse_seeds_float(self) -> None:
        assert parse_seeds("0.1") == 0.1

    def test_parse_seeds_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="Compression ratio"):
            parse_seeds("1.5")

    def test_parse_hex_color_valid(self) -> None:
        assert parse_hex_color("#ff0000") == (1.0, 0.0, 0.0)
        assert parse_hex_color("00ff00") == (0.0, 1.0, 0.0)

    def test_parse_hex_color_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid hex color"):
            parse_hex_color("xyz")

    def test_parse_shape(self) -> None:
        assert parse_shape("128,128,128") == (128, 128, 128)
        assert parse_shape("64, 32") == (64, 32)


# ═══════════════════════════════════════════════════════════════════════
# Volume loader tests
# ═══════════════════════════════════════════════════════════════════════


class TestLoadVolume:
    def test_load_npy(self, tmp_path: Path) -> None:
        vol = np.random.rand(8, 8, 8).astype(np.float32)
        path = tmp_path / "test.npy"
        np.save(str(path), vol)
        loaded = load_volume(path)
        assert loaded.shape == (8, 8, 8)
        assert loaded.dtype == np.float32

    def test_load_npz(self, tmp_path: Path) -> None:
        vol = np.random.rand(8, 8).astype(np.float32)
        path = tmp_path / "test.npz"
        np.savez(str(path), mydata=vol)
        loaded = load_volume(path, array_key="mydata")
        assert loaded.shape == (8, 8)

    def test_load_npz_first_key(self, tmp_path: Path) -> None:
        vol = np.random.rand(8, 8).astype(np.float32)
        path = tmp_path / "test.npz"
        np.savez(str(path), data=vol)
        loaded = load_volume(path)
        assert loaded.shape == (8, 8)

    def test_load_zarr_array(self, tmp_path: Path) -> None:
        import zarr

        vol = np.random.rand(8, 8, 8).astype(np.float32)
        path = tmp_path / "test.zarr"
        z = zarr.open(str(path), mode="w", shape=vol.shape, dtype=vol.dtype)
        z[:] = vol
        loaded = load_volume(path)
        assert loaded.shape == (8, 8, 8)

    def test_load_zarr_5d(self, tmp_path: Path) -> None:
        import zarr

        # Simulate 5D OME-ZARR (T=2, C=3, Z=4, Y=4, X=4)
        vol = np.random.rand(2, 3, 4, 4, 4).astype(np.float32)
        path = tmp_path / "test.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.create_dataset("0", data=vol)
        loaded = load_volume(path, channel=1, timepoint=0)
        assert loaded.shape == (4, 4, 4)

    def test_load_validates_minimum_dims(self, tmp_path: Path) -> None:
        # 1D array should fail
        path = tmp_path / "test.npy"
        np.save(str(path), np.array([1, 2, 3], dtype=np.float32))
        with pytest.raises(ValueError, match="at least 2D"):
            load_volume(path)

    def test_load_npz_invalid_key(self, tmp_path: Path) -> None:
        vol = np.random.rand(8, 8).astype(np.float32)
        path = tmp_path / "test.npz"
        np.savez(str(path), data=vol)
        with pytest.raises(ValueError, match="not found"):
            load_volume(path, array_key="nonexistent")

    def test_load_converts_dtype_to_float32(self, tmp_path: Path) -> None:
        """uint8 and int16 volumes should be converted to float32."""
        vol_uint8 = np.random.randint(0, 255, (8, 8, 8), dtype=np.uint8)
        path = tmp_path / "uint8.npy"
        np.save(str(path), vol_uint8)
        loaded = load_volume(path)
        assert loaded.dtype == np.float32

    def test_load_zarr_group_with_array_key(self, tmp_path: Path) -> None:
        import zarr

        vol = np.random.rand(4, 4, 4).astype(np.float32)
        path = tmp_path / "test.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.create_dataset("my_volume", data=vol)
        root.create_dataset("other_data", data=np.zeros(10))
        loaded = load_volume(path, array_key="my_volume")
        assert loaded.shape == (4, 4, 4)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


class TestBuildDimensions:
    def test_3d(self) -> None:
        centers = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
        dims = build_dimensions_from_data(centers)
        assert dims.ndim == 3

    def test_2d(self) -> None:
        centers = np.array([[0, 0], [5, 5]], dtype=np.float32)
        dims = build_dimensions_from_data(centers)
        assert dims.ndim == 2

    def test_degenerate_single_point(self) -> None:
        """Should not crash when all centers are identical (min==max)."""
        centers = np.array([[5, 5, 5]], dtype=np.float32)
        dims = build_dimensions_from_data(centers)
        assert dims.ndim == 3
        # Range should be valid (min < max)
        for dim in dims.dimensions:
            assert dim.range is not None
            assert dim.range[0] < dim.range[1]

    def test_degenerate_flat_dimension(self) -> None:
        """One dimension has zero range (e.g., 2D data embedded in 3D)."""
        centers = np.array([[0, 0, 0], [10, 10, 0]], dtype=np.float32)
        dims = build_dimensions_from_data(centers)
        assert dims.ndim == 3
        # Z dimension should have been expanded
        z_dim = dims.dimensions[2]
        assert z_dim.range is not None
        assert z_dim.range[0] < z_dim.range[1]


# ═══════════════════════════════════════════════════════════════════════
# Command integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestFitCommand:
    def test_dump_config(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["gsplat", "fit", "--dump-config"])
        assert result.exit_code == 0
        assert "n_iters" in result.stdout

    def test_dump_config_with_preset(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["gsplat", "fit", "--dump-config", "--preset", "hifi"]
        )
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.stdout)
        assert parsed["n_iters"] == 10000

    def test_fit_small_volume(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "fitted.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(out),
                "--preset",
                "draft",
                "--iters",
                "5",
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert result.exit_code == 0, f"fit failed: {result.stdout}"
        assert out.exists()

    def test_fit_output_is_loadable(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """Verify fit output can be loaded back as GSplatData."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "fitted.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(out),
                "--preset",
                "draft",
                "--iters",
                "5",
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert result.exit_code == 0

        # Load and verify structure
        data = GSplatData.load(out)
        assert data.n_splats > 0
        assert data.ndim == 3
        assert data.amplitudes.shape == (data.n_splats,)

    def test_fit_with_yaml_config(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """End-to-end: fit using a YAML config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("n_iters: 3\ncull_retention: 0.95\n")

        out = tmp_path / "fitted.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(out),
                "--config",
                str(config_path),
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert result.exit_code == 0, f"fit with config failed: {result.stdout}"
        assert out.exists()

    def test_fit_missing_input(self, runner: CliRunner, tmp_path: Path) -> None:
        """Fit with nonexistent input file should fail."""
        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "fit", "/nonexistent/volume.npy", str(out), "--quiet"],
        )
        assert result.exit_code != 0

    def test_fit_missing_output_arg(
        self, runner: CliRunner, small_volume_npy: Path
    ) -> None:
        """Fit without output path should fail."""
        result = runner.invoke(
            app,
            ["gsplat", "fit", str(small_volume_npy)],
        )
        assert result.exit_code != 0


class TestConvertCommand:
    def test_convert_basic(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "scene.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "convert", str(sample_gsplats), str(out)],
        )
        assert result.exit_code == 0, f"convert failed: {result.stdout}"
        assert out.exists()

        import zarr

        store = zarr.open_group(str(out), mode="r")
        assert "gsplats" in store

    def test_convert_no_center(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "scene.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "convert", str(sample_gsplats), str(out), "--no-center"],
        )
        assert result.exit_code == 0

    def test_convert_with_scale_intensity(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "scene.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "convert",
                str(sample_gsplats),
                str(out),
                "--scale-intensity",
                "0.5",
            ],
        )
        assert result.exit_code == 0, (
            f"convert --scale-intensity failed: {result.stdout}"
        )
        assert out.exists()

    def test_convert_scene_has_valid_structure(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Verify the converted scene has proper zarr structure."""
        import zarr

        out = tmp_path / "scene.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "convert", str(sample_gsplats), str(out)],
        )
        assert result.exit_code == 0

        store = zarr.open_group(str(out), mode="r")
        # Scene root should have scene_dimensions
        assert "scene_dimensions" in store.attrs
        # Should have a gsplats node
        assert "gsplats" in store
        gsplats_group = store["gsplats"]
        assert gsplats_group.attrs.get("type") == "gsplats"
        # Should have data arrays
        assert "centers" in gsplats_group
        assert "amplitudes" in gsplats_group
        assert "cholesky_factors" in gsplats_group


class TestRenderCommand:
    def test_render_to_npy(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "rendered.npy"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "render",
                str(sample_gsplats),
                str(out),
                "--shape",
                "8,8,8",
            ],
        )
        assert result.exit_code == 0, f"render failed: {result.stdout}"
        assert out.exists()
        vol = np.load(str(out))
        assert vol.shape == (8, 8, 8)

    def test_render_auto_shape(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "rendered.npy"
        result = runner.invoke(
            app,
            ["gsplat", "render", str(sample_gsplats), str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()

    def test_render_output_is_valid_array(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Rendered volume should be a valid float array with non-negative values."""
        out = tmp_path / "rendered.npy"
        result = runner.invoke(
            app,
            ["gsplat", "render", str(sample_gsplats), str(out), "--shape", "8,8,8"],
        )
        assert result.exit_code == 0
        vol = np.load(str(out))
        assert vol.dtype == np.float32 or vol.dtype == np.float64
        assert np.all(np.isfinite(vol))
        assert np.min(vol) >= 0.0  # Gaussian sums are non-negative


class TestMergeCommand:
    def test_merge_concatenate(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        # Create second dataset
        from luxar.gsplats.gsplat_data import GSplatData

        data2 = GSplatData(
            centers=np.random.rand(3, 3).astype(np.float32) * 10,
            amplitudes=np.random.rand(3).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (3, 1)
            ),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "merged.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(path2),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"merge failed: {result.stdout}"
        assert out.exists()

        merged = GSplatData.load(out)
        assert merged.n_splats == 8  # 5 + 3

    def test_merge_as_dimension(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        # Create second dataset with same shape
        from luxar.gsplats.gsplat_data import GSplatData

        data2 = GSplatData(
            centers=np.random.rand(4, 3).astype(np.float32) * 10,
            amplitudes=np.random.rand(4).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (4, 1)
            ),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "stacked.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(path2),
                "-o",
                str(out),
                "--as-dimension",
                "--values",
                "0,1",
            ],
        )
        assert result.exit_code == 0, f"merge --as-dimension failed: {result.stdout}"

        merged = GSplatData.load(out)
        assert merged.ndim == 4  # 3D -> 4D
        assert merged.n_splats == 9  # 5 + 4

    def test_merge_channel_colors(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        data2 = GSplatData(
            centers=np.random.rand(3, 3).astype(np.float32) * 10,
            amplitudes=np.random.rand(3).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (3, 1)
            ),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "colored.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(path2),
                "-o",
                str(out),
                "--channel-colors",
                "#ff0000,#00ff00",
            ],
        )
        assert result.exit_code == 0, f"merge --channel-colors failed: {result.stdout}"

        merged = GSplatData.load(out)
        assert merged.colors is not None
        assert merged.n_splats == 8

    def test_merge_mutual_exclusivity(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "err.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(sample_gsplats),
                "-o",
                str(out),
                "--as-dimension",
                "--channel-colors",
                "#ff0000,#00ff00",
            ],
        )
        assert result.exit_code != 0

    def test_merge_too_few_inputs(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "err.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code != 0

    def test_merge_wrong_color_count(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """3 colors but 2 datasets should fail."""
        out = tmp_path / "err.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(sample_gsplats),
                "-o",
                str(out),
                "--channel-colors",
                "#ff0000,#00ff00,#0000ff",
            ],
        )
        assert result.exit_code != 0

    def test_merge_wrong_values_count(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """3 values but 2 datasets should fail."""
        out = tmp_path / "err.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(sample_gsplats),
                "-o",
                str(out),
                "--as-dimension",
                "--values",
                "0,1,2",
            ],
        )
        assert result.exit_code != 0

    def test_merge_three_datasets(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Merge 3 datasets via concatenation."""
        from luxar.gsplats.gsplat_data import GSplatData

        # Create two more datasets
        for name in ["b.gsplats.zarr", "c.gsplats.zarr"]:
            d = GSplatData(
                centers=np.random.rand(2, 3).astype(np.float32) * 10,
                amplitudes=np.random.rand(2).astype(np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (2, 1)
                ),
            )
            d.save(tmp_path / name)

        out = tmp_path / "merged3.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(sample_gsplats),
                str(tmp_path / "b.gsplats.zarr"),
                str(tmp_path / "c.gsplats.zarr"),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"3-way merge failed: {result.stdout}"

        merged = GSplatData.load(out)
        assert merged.n_splats == 9  # 5 + 2 + 2


# ═══════════════════════════════════════════════════════════════════════
# End-to-end workflow tests
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndWorkflows:
    """Tests that chain multiple CLI commands into full workflows."""

    def test_fit_then_convert(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: fit volume → convert to scene → verify structure."""
        import zarr

        gsplats_path = tmp_path / "fitted.gsplats.zarr"
        scene_path = tmp_path / "scene.zarr"

        # Step 1: Fit
        r1 = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(gsplats_path),
                "--preset",
                "draft",
                "--iters",
                "5",
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert r1.exit_code == 0, f"fit step failed: {r1.stdout}"

        # Step 2: Convert
        r2 = runner.invoke(
            app,
            ["gsplat", "convert", str(gsplats_path), str(scene_path), "--center"],
        )
        assert r2.exit_code == 0, f"convert step failed: {r2.stdout}"

        # Step 3: Verify scene structure
        store = zarr.open_group(str(scene_path), mode="r")
        assert "scene_dimensions" in store.attrs
        assert "gsplats" in store
        assert "centers" in store["gsplats"]

    def test_fit_then_render(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: fit volume → render back → verify shape matches."""
        gsplats_path = tmp_path / "fitted.gsplats.zarr"
        rendered_path = tmp_path / "rendered.npy"

        # Step 1: Fit
        r1 = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(gsplats_path),
                "--preset",
                "draft",
                "--iters",
                "5",
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert r1.exit_code == 0

        # Step 2: Render to same shape as input
        r2 = runner.invoke(
            app,
            [
                "gsplat",
                "render",
                str(gsplats_path),
                str(rendered_path),
                "--shape",
                "16,16,16",
            ],
        )
        assert r2.exit_code == 0, f"render step failed: {r2.stdout}"

        # Step 3: Verify shape
        vol = np.load(str(rendered_path))
        assert vol.shape == (16, 16, 16)

    def test_fit_then_info(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: fit volume → inspect with gsplat info."""
        gsplats_path = tmp_path / "fitted.gsplats.zarr"

        # Step 1: Fit
        r1 = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(gsplats_path),
                "--preset",
                "draft",
                "--iters",
                "5",
                "--seeds",
                "10",
                "--quiet",
            ],
        )
        assert r1.exit_code == 0

        # Step 2: Info
        r2 = runner.invoke(app, ["gsplat", "info", str(gsplats_path)])
        assert r2.exit_code == 0
        assert "DATASET INFORMATION" in r2.stdout
        assert "Splats:" in r2.stdout


# ═══════════════════════════════════════════════════════════════════════
# Zip compression tests
# ═══════════════════════════════════════════════════════════════════════


class TestZipCompression:
    """Tests that zip archives use ZIP_STORED (no double compression)."""

    def test_zip_uses_stored_not_deflated(self, tmp_path: Path) -> None:
        """Zarr data is already compressed internally, so zip should use
        ZIP_STORED to avoid wasteful double-compression."""
        import zipfile

        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData(
            centers=np.random.rand(10, 3).astype(np.float32) * 10,
            amplitudes=np.random.rand(10).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (10, 1)
            ),
        )
        out = tmp_path / "test.gsplats.zarr.zip"
        data.save(out, compress="zip")

        assert out.exists()

        # Verify all entries use ZIP_STORED (no compression)
        with zipfile.ZipFile(out, "r") as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, (
                    f"File '{info.filename}' uses compression type "
                    f"{info.compress_type}, expected ZIP_STORED (0)"
                )

    def test_zip_roundtrip(self, tmp_path: Path) -> None:
        """Save with compress=zip, then load back and verify data integrity."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData(
            centers=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            amplitudes=np.array([0.5, 0.8], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (2, 1)
            ),
        )
        out = tmp_path / "roundtrip.gsplats.zarr.zip"
        original.save(out, compress="zip")

        loaded = GSplatData.load(out)
        assert loaded.n_splats == 2
        np.testing.assert_allclose(loaded.centers, original.centers, atol=0.1)
        np.testing.assert_allclose(loaded.amplitudes, original.amplitudes, atol=0.1)


# ═══════════════════════════════════════════════════════════════════════
# Filter command tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_gsplats_for_filter(tmp_path: Path) -> Path:
    """Create a .gsplats.zarr with controlled data for filter tests."""
    from luxar.gsplats.gsplat_data import GSplatData

    n = 10
    rng = np.random.RandomState(42)
    data = GSplatData(
        centers=rng.rand(n, 3).astype(np.float32) * 100,
        amplitudes=np.linspace(0.1, 1.0, n, dtype=np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
        ),
    )
    out = tmp_path / "filter_test.gsplats.zarr"
    data.save(out)
    return out


class TestFilterCommand:
    """Tests for luxar gsplat filter CLI command."""

    def test_filter_by_amplitude(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "0.5",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        filtered = GSplatData.load(out)
        assert filtered.n_splats < 10
        assert filtered.n_splats > 0
        assert np.all(filtered.amplitudes >= 0.5)

    def test_filter_by_bbox(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--bbox",
                "0,50,0,50,0,50",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        filtered = GSplatData.load(out)
        assert filtered.n_splats <= 10
        assert np.all(filtered.centers <= 50)

    def test_filter_combined(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "0.3",
                "--eccentricity-max",
                "10.0",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

    def test_filter_no_criteria(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        """No filter flags → all splats pass through."""
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"

        from luxar.gsplats.gsplat_data import GSplatData

        filtered = GSplatData.load(out)
        assert filtered.n_splats == 10

    def test_filter_with_compression(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "filtered.gsplats.zarr.zip"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "0.5",
                "--compress",
                "zip",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()
        assert out.is_file()  # zip is a file, not directory

    def test_filter_removes_most(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        """Restrictive criteria remove most splats."""
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "0.9",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        filtered = GSplatData.load(out)
        assert filtered.n_splats < 5
        assert filtered.n_splats > 0

    def test_filter_normalized_amplitude(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "0.5",
                "--amplitude-normalized",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        filtered = GSplatData.load(out)
        assert filtered.n_splats < 10
        assert filtered.n_splats > 0


# ═══════════════════════════════════════════════════════════════════════
# Split command tests
# ═══════════════════════════════════════════════════════════════════════


class TestSplitCommand:
    """Tests for luxar gsplat split CLI command."""

    def test_split_by_parts(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "split_output"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "split",
                str(sample_gsplats),
                str(out_dir),
                "--parts",
                "3",
            ],
        )
        assert result.exit_code == 0, f"split failed: {result.stdout}"
        assert out_dir.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        parts = []
        for i in range(3):
            p = out_dir / f"part_{i:03d}.gsplats.zarr"
            assert p.exists(), f"Missing {p.name}"
            parts.append(GSplatData.load(p))

        total = sum(p.n_splats for p in parts)
        assert total == 5  # sample_gsplats has 5 splats

    def test_split_by_indices(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "split_output"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "split",
                str(sample_gsplats),
                str(out_dir),
                "--indices",
                "2,4",
            ],
        )
        assert result.exit_code == 0, f"split failed: {result.stdout}"

        from luxar.gsplats.gsplat_data import GSplatData

        parts = [
            GSplatData.load(out_dir / f"part_{i:03d}.gsplats.zarr") for i in range(3)
        ]
        assert parts[0].n_splats == 2
        assert parts[1].n_splats == 2
        assert parts[2].n_splats == 1

    def test_split_with_compression(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "split_output"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "split",
                str(sample_gsplats),
                str(out_dir),
                "--parts",
                "2",
                "--compress",
                "zip",
            ],
        )
        assert result.exit_code == 0, f"split failed: {result.stdout}"
        assert out_dir.exists()

    def test_split_missing_mode(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Neither --parts nor --indices -> error."""
        out_dir = tmp_path / "split_output"
        result = runner.invoke(
            app,
            ["gsplat", "split", str(sample_gsplats), str(out_dir)],
        )
        assert result.exit_code == 1

    def test_split_roundtrip_with_merge(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Split then merge should preserve total splat count."""
        split_dir = tmp_path / "split_output"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "split",
                str(sample_gsplats),
                str(split_dir),
                "--parts",
                "2",
            ],
        )
        assert result.exit_code == 0, f"split failed: {result.stdout}"

        # Merge back
        merged = tmp_path / "merged.gsplats.zarr"
        part_paths = [str(split_dir / f"part_{i:03d}.gsplats.zarr") for i in range(2)]
        result = runner.invoke(
            app,
            ["gsplat", "merge"] + part_paths + ["-o", str(merged)],
        )
        assert result.exit_code == 0, f"merge failed: {result.stdout}"

        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats)
        recombined = GSplatData.load(merged)
        assert recombined.n_splats == original.n_splats


# ═══════════════════════════════════════════════════════════════════════
# Slice command tests
# ═══════════════════════════════════════════════════════════════════════


class TestSliceCommand:
    """Tests for luxar gsplat slice CLI command."""

    def test_slice_basic(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        """Slice with bounded ranges on all dimensions."""
        out = tmp_path / "sliced.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "slice",
                str(sample_gsplats_for_filter),
                str(out),
                "0:50, 0:50, 0:50",
            ],
        )
        assert result.exit_code == 0, f"slice failed: {result.stdout}"
        assert out.exists()

        from luxar.gsplats.gsplat_data import GSplatData

        sliced = GSplatData.load(out)
        assert sliced.n_splats <= 10
        assert np.all(sliced.centers <= 50)

    def test_slice_open_ranges(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        """Open-ended ranges: ':50, :, 10:'."""
        out = tmp_path / "sliced.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "slice",
                str(sample_gsplats_for_filter),
                str(out),
                ":50, :, 10:",
            ],
        )
        assert result.exit_code == 0, f"slice failed: {result.stdout}"
        assert out.exists()

    def test_slice_all(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        """':, :, :' keeps everything."""
        out = tmp_path / "sliced.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "slice",
                str(sample_gsplats_for_filter),
                str(out),
                ":, :, :",
            ],
        )
        assert result.exit_code == 0, f"slice failed: {result.stdout}"

        from luxar.gsplats.gsplat_data import GSplatData

        sliced = GSplatData.load(out)
        assert sliced.n_splats == 10

    def test_slice_with_compression(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "sliced.gsplats.zarr.zip"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "slice",
                str(sample_gsplats_for_filter),
                str(out),
                ":, :, :",
                "--compress",
                "zip",
            ],
        )
        assert result.exit_code == 0, f"slice failed: {result.stdout}"
        assert out.exists()
        assert out.is_file()


# ═══════════════════════════════════════════════════════════════════════
# Compare command tests
# ═══════════════════════════════════════════════════════════════════════


class TestCompareCommand:
    def test_compare_basic(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
    ) -> None:
        """Compare gsplats against a reference volume."""
        result = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(sample_gsplats),
                str(small_volume_npy),
                "--shape",
                "16,16,16",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"compare failed: {result.stdout}"
        assert "PSNR" in result.stdout
        assert "SSIM" in result.stdout
        assert "MSE" in result.stdout

    def test_compare_json_output(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        """Verify JSON output contains expected keys."""
        import json

        json_path = tmp_path / "metrics.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(sample_gsplats),
                str(small_volume_npy),
                "--shape",
                "16,16,16",
                "--device",
                "cpu",
                "--output-json",
                str(json_path),
            ],
        )
        assert result.exit_code == 0, f"compare failed: {result.stdout}"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        for key in [
            "mse",
            "psnr_db",
            "ssim",
            "rel_l2",
            "max_abs_error",
            "n_splats",
            "ndim",
            "shape",
        ]:
            assert key in data, f"Missing key: {key}"

    def test_compare_quiet(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        """Quiet mode should suppress terminal table."""
        json_path = tmp_path / "metrics.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(sample_gsplats),
                str(small_volume_npy),
                "--shape",
                "16,16,16",
                "--device",
                "cpu",
                "--quiet",
                "--output-json",
                str(json_path),
            ],
        )
        assert result.exit_code == 0, f"compare failed: {result.stdout}"
        assert "QUALITY COMPARISON" not in result.stdout
        assert json_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# Transform command tests
# ═══════════════════════════════════════════════════════════════════════


class TestTransformCommand:
    """Tests for luxar gsplat transform CLI command."""

    def test_transform_scale(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Scale factors are applied correctly to centers."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats)
        out = tmp_path / "scaled.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out), "--scale", "2,3,4"],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        assert out.exists()

        scaled = GSplatData.load(out)
        assert scaled.n_splats == original.n_splats
        # Save/load reorders splats (spatial ordering), so compare sorted values
        np.testing.assert_allclose(
            np.sort(scaled.centers[:, 0]),
            np.sort(original.centers[:, 0] * 2),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.sort(scaled.centers[:, 1]),
            np.sort(original.centers[:, 1] * 3),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.sort(scaled.centers[:, 2]),
            np.sort(original.centers[:, 2] * 4),
            atol=1e-3,
        )

    def test_transform_center(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Centering moves the centroid to near origin."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "centered.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out), "--center"],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        centered = GSplatData.load(out)
        # Amplitude-weighted centroid should be near zero
        total_amp = centered.amplitudes.sum()
        centroid = (centered.centers.T @ centered.amplitudes) / total_amp
        np.testing.assert_allclose(centroid, 0.0, atol=1e-3)

    def test_transform_scale_intensity(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Amplitude scaling works correctly."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats)
        out = tmp_path / "dimmed.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats),
                str(out),
                "--scale-intensity",
                "0.5",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        dimmed = GSplatData.load(out)
        # Save/load reorders splats (spatial ordering), so compare sorted values
        np.testing.assert_allclose(
            np.sort(dimmed.amplitudes), np.sort(original.amplitudes * 0.5), atol=1e-5
        )

    def test_transform_rotate_z(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """90-degree Z rotation swaps X and Y (with sign flip)."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats)
        out = tmp_path / "rotated.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out), "--rotate-z", "90"],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        # Save/load reorders splats (spatial ordering), so compare sorted values
        # After 90° Z rotation: new_x = -old_y, new_y = old_x, z unchanged
        expected_x = np.sort(-original.centers[:, 1])
        expected_y = np.sort(original.centers[:, 0])
        expected_z = np.sort(original.centers[:, 2])
        np.testing.assert_allclose(
            np.sort(rotated.centers[:, 0]), expected_x, atol=1e-3
        )
        np.testing.assert_allclose(
            np.sort(rotated.centers[:, 1]), expected_y, atol=1e-3
        )
        np.testing.assert_allclose(
            np.sort(rotated.centers[:, 2]), expected_z, atol=1e-3
        )

    def test_transform_combined(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Multiple transforms can be combined."""
        out = tmp_path / "combined.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats),
                str(out),
                "--scale",
                "1,1,2",
                "--scale-intensity",
                "0.1",
                "--center",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        assert out.exists()

    def test_transform_no_flags_errors(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Error when no transforms are specified."""
        out = tmp_path / "noop.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out)],
        )
        assert result.exit_code != 0

    def test_transform_wrong_scale_dims(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Error when scale factor count doesn't match ndim."""
        out = tmp_path / "bad.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out), "--scale", "1,2"],
        )
        assert result.exit_code != 0

    def test_transform_translate(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Translation shifts centers correctly."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats)
        out = tmp_path / "translated.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats),
                str(out),
                "--translate",
                "10,20,30",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        translated = GSplatData.load(out)
        # Save/load reorders splats (spatial ordering), so compare sorted values
        np.testing.assert_allclose(
            np.sort(translated.centers[:, 0]),
            np.sort(original.centers[:, 0] + 10),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.sort(translated.centers[:, 1]),
            np.sort(original.centers[:, 1] + 20),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.sort(translated.centers[:, 2]),
            np.sort(original.centers[:, 2] + 30),
            atol=1e-3,
        )


# ═══════════════════════════════════════════════════════════════════════
# Calibrate (cal) command tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def smooth_blob_volume(tmp_path: Path) -> Path:
    """A small smooth-blob volume that splats can fit quickly on CPU.

    16^3 keeps the fit time per K under a few seconds even with patience-based
    stopping; the smooth Gaussian blob gives the auto-seeder something to find.
    """
    rng = np.random.default_rng(0)
    Y, X, Z = np.meshgrid(
        np.linspace(0, 1, 16),
        np.linspace(0, 1, 16),
        np.linspace(0, 1, 16),
        indexing="ij",
    )
    signal = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) * 8)
    V = (signal + 0.02 * rng.standard_normal(signal.shape)).astype(np.float32)
    V = np.clip(V, 0, 1)
    path = tmp_path / "smooth_blob.npy"
    np.save(str(path), V)
    return path


@pytest.fixture
def fast_fit_config(tmp_path: Path) -> Path:
    """YAML config that caps iters low and disables GPU paths for fast CPU tests."""
    cfg = {
        "n_iters": 30,
        "early_stop_patience": 30,
        "use_cuda": False,
        "use_metal": False,
    }
    path = tmp_path / "fast.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestCalibrateCommand:
    def test_cal_basic(
        self,
        runner: CliRunner,
        smooth_blob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: cal sweeps a 2-K grid and writes a valid JSON result."""
        import json

        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(smooth_blob_volume),
                str(out_json),
                "--n-grid",
                "2",
                "--k-min",
                "20",
                "--k-max",
                "100",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"
        assert out_json.exists()

        with open(out_json) as f:
            data = json.load(f)

        # Required top-level keys
        for key in [
            "k_values_requested",
            "k_values_effective",
            "held_out_psnr_db",
            "train_psnr_db",
            "full_psnr_db",
            "full_ssim",
            "held_out_peak",
            "noise_floor",
            "fit_times_seconds",
            "mask_seed",
            "mask_fraction",
            "donut_radius",
            "volume_shape",
            "timestamp",
        ]:
            assert key in data, f"Missing key in JSON: {key}"

        # Sweep was 2 K values
        assert len(data["k_values_requested"]) == 2
        assert len(data["held_out_psnr_db"]) == 2

        # K* must be one of the swept values
        assert data["held_out_peak"]["k_star"] in data["k_values_requested"]
        assert data["held_out_peak"]["type"] in ("peak", "plateau", "signal_limited")

        # Noise floor present and finite (or sentinel)
        nf = data["noise_floor"]
        for sig_key in ("sigma_hat", "sigma_laplacian", "sigma_haar"):
            assert sig_key in nf

        # Stdout shows the recommended K* line
        assert "Recommended K" in result.stdout

    def test_cal_explicit_grid_overrides(
        self,
        runner: CliRunner,
        smooth_blob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """--k-grid takes precedence over --n-grid/--k-min/--k-max."""
        import json

        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(smooth_blob_volume),
                str(out_json),
                "--k-grid",
                "30,90",
                # These should be ignored when --k-grid is present
                "--n-grid",
                "10",
                "--k-min",
                "1",
                "--k-max",
                "999",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"
        with open(out_json) as f:
            data = json.load(f)
        assert data["k_values_requested"] == [30, 90]

    def test_cal_invalid_progression(
        self,
        runner: CliRunner,
        smooth_blob_volume: Path,
        tmp_path: Path,
    ) -> None:
        """Bogus --progression name surfaces as a non-zero exit code."""
        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(smooth_blob_volume),
                str(out_json),
                "--progression",
                "cubic",  # not a valid name
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code != 0

    def test_cal_pdf_and_keep_fits(
        self,
        runner: CliRunner,
        smooth_blob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """--pdf and --keep-fits both produce their expected outputs."""
        # Skip if matplotlib isn't installed (the PDF path degrades but the
        # rest of cal must still complete; this test specifically checks PDF).
        pytest.importorskip("matplotlib")

        out_json = tmp_path / "cal.json"
        out_pdf = tmp_path / "cal.pdf"
        out_fits = tmp_path / "fits"

        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(smooth_blob_volume),
                str(out_json),
                "--k-grid",
                "30,90",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
                "--pdf",
                str(out_pdf),
                "--keep-fits",
                str(out_fits),
            ],
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 1000  # not an empty PDF
        # Per-K fits persisted
        assert out_fits.is_dir()
        persisted = sorted(out_fits.iterdir())
        assert len(persisted) == 2


# ═══════════════════════════════════════════════════════════════════════
# LOD command tests (additive + substitutive)
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def medium_gsplats(tmp_path: Path) -> Path:
    """A 32-splat .gsplats.zarr large enough for both LOD subcommands.

    Additive needs ``data.n_splats >= n_lods`` (default 4). Substitutive
    needs enough splats that ``ceil(N / K^L)`` stays positive — 32 splats
    with default ``K=4, L=3`` yields levels 32 / 8 / 2 / 1.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    rng = np.random.default_rng(0)
    n, d = 32, 3
    data = GSplatData(
        centers=(rng.random((n, d)) * 10).astype(np.float32),
        amplitudes=rng.random(n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
        ),
    )
    out = tmp_path / "medium.gsplats.zarr"
    data.save(out)
    return out


class TestLODCommand:
    def test_lod_additive_basic(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`lod additive` produces a valid multi-LOD .gsplats.zarr."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "ladder.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "additive",
                str(medium_gsplats),
                str(out),
                "--n-lods",
                "2",
                "--method",
                "self_energy",  # cheap, avoids dense Gram on tiny inputs
            ],
        )
        assert result.exit_code == 0, f"lod additive failed:\n{result.stdout}"
        assert out.exists()
        ladder = GSplatData.load(out)
        # Multi-LOD with 2 levels; sum of per-level counts equals total
        assert ladder.n_additive_sublods == 2
        per_level = [
            ladder.additive_sublod(i).n_splats for i in range(ladder.n_additive_sublods)
        ]
        assert sum(per_level) == ladder.n_splats == 32

    def test_lod_additive_quiet_suppresses_per_lod_lines(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`--quiet` suppresses 'LOD 0: ... splats' progress lines."""
        out_loud = tmp_path / "loud.gsplats.zarr"
        out_quiet = tmp_path / "quiet.gsplats.zarr"

        loud = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "additive",
                str(medium_gsplats),
                str(out_loud),
                "--n-lods",
                "2",
                "--method",
                "self_energy",
            ],
        )
        assert loud.exit_code == 0, f"loud invocation failed:\n{loud.stdout}"

        quiet = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "additive",
                str(medium_gsplats),
                str(out_quiet),
                "--n-lods",
                "2",
                "--method",
                "self_energy",
                "--quiet",
            ],
        )
        assert quiet.exit_code == 0, f"quiet invocation failed:\n{quiet.stdout}"

        # The per-LOD enumeration "LOD 0:" / "LOD 1:" appears in loud only
        assert "LOD 0" in loud.stdout or "LOD 1" in loud.stdout
        assert "LOD 0" not in quiet.stdout
        assert "LOD 1" not in quiet.stdout

    def test_lod_substitutive_basic(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`lod substitutive` produces a single v2.0 .gsplats.zarr."""
        from luxar.gsplats import GSplatData

        out_path = tmp_path / "hierarchy.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "substitutive",
                str(medium_gsplats),
                str(out_path),
                "--K",
                "2",
                "--L",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"lod substitutive failed:\n{result.stdout}"
        assert out_path.is_dir()
        loaded = GSplatData.load(out_path)
        # n_substitutive = levels + 1 (= 3 for L=2)
        assert loaded.n_substitutive == 3
        # Each substitutive level is a single additive sub-LOD
        for lev in loaded.substitutive_levels:
            assert lev.n_additive_lods == 1
        # compression_factor = K^level_index = 2^s
        assert [lev.compression_factor for lev in loaded.substitutive_levels] == [
            1,
            2,
            4,
        ]

    def test_lod_substitutive_quiet_suppresses_per_level_lines(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`--quiet` suppresses per-level enumeration."""
        out_path = tmp_path / "hierarchy_quiet.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "substitutive",
                str(medium_gsplats),
                str(out_path),
                "--K",
                "2",
                "--L",
                "1",
                "--device",
                "cpu",
                "--quiet",
            ],
        )
        assert result.exit_code == 0, f"quiet substitutive failed:\n{result.stdout}"
        assert out_path.is_dir()
        # The per-level "level 0: ... splats" / "Wrote ..." lines are gated
        assert "level 0:" not in result.stdout
        assert "Wrote" not in result.stdout

    def test_lod_pyramid_full_matrix(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`lod pyramid` builds the full [N, M] LOD matrix in one shot."""
        from luxar.gsplats import GSplatData

        out_path = tmp_path / "pyramid.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "pyramid",
                str(medium_gsplats),
                str(out_path),
                "--substitutive",
                "K=2,L=2",
                "--additive",
                "2",
                "--additive-method",
                "self_energy",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"lod pyramid failed:\n{result.stdout}"
        loaded = GSplatData.load(out_path)
        assert loaded.n_substitutive == 3
        # Each substitutive level has its own additive ladder (<= 2; may clamp on tiny levels)
        for lev in loaded.substitutive_levels:
            assert 1 <= lev.n_additive_lods <= 2

    def test_lod_additive_substitutive_level_kwarg(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """`lod additive --substitutive-level` targets a specific level of a pyramid."""
        from luxar.gsplats import GSplatData

        # First build a 3-level substitutive pyramid (M_i = 1 each).
        pyramid_path = tmp_path / "pyramid.gsplats.zarr"
        r1 = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "substitutive",
                str(medium_gsplats),
                str(pyramid_path),
                "--K",
                "2",
                "--L",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert r1.exit_code == 0, f"substitutive prep failed:\n{r1.stdout}"

        # Now add an additive ladder on substitutive level 1 only.
        ladder_path = tmp_path / "ladder.gsplats.zarr"
        r2 = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "additive",
                str(pyramid_path),
                str(ladder_path),
                "--n-lods",
                "2",
                "--method",
                "self_energy",
                "--substitutive-level",
                "1",
            ],
        )
        assert r2.exit_code == 0, f"additive on level 1 failed:\n{r2.stdout}"

        loaded = GSplatData.load(ladder_path)
        assert loaded.n_substitutive == 3
        # Level 0 and 2 carried over unchanged: still M=1
        assert loaded.substitutive_levels[0].n_additive_lods == 1
        assert loaded.substitutive_levels[2].n_additive_lods == 1
        # Level 1 now has additive ladder
        assert loaded.substitutive_levels[1].n_additive_lods >= 1

    def test_lod_additive_substitutive_level_out_of_bounds(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
    ) -> None:
        """Out-of-bounds --substitutive-level should error cleanly."""
        out_path = tmp_path / "fail.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                "additive",
                str(medium_gsplats),
                str(out_path),
                "--n-lods",
                "2",
                "--method",
                "self_energy",
                "--substitutive-level",
                "5",
            ],
        )
        assert result.exit_code != 0
        assert not out_path.exists()


class TestMigrateFormatCommand:
    """`luxar gsplat migrate-format` end-to-end CLI tests.

    Builds legacy fixtures (v1.0, v1.1, substitutive directory) and
    verifies the CLI handler invokes `migrate_format` and produces a
    valid v2.0 file. Mirrors the unit tests in
    `gsplats/io/tests/test_migrate_format.py`.
    """

    @staticmethod
    def _identity_chol(n: int):
        import numpy as np

        row = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype="float32")
        return np.tile(row, (n, 1))

    def _make_v1_0(self, path: Path, n: int = 5) -> None:
        import numpy as np
        import zarr

        store = zarr.DirectoryStore(str(path))
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update(
            {
                "format_version": "1.0",
                "format_type": "gsplats_zarr",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "luxar_gsplats_version": "test",
            }
        )
        splats = root.create_group("splats")
        splats.attrs.update(
            {
                "type": "gsplats",
                "n_splats": n,
                "ndim": 3,
                "has_colors": False,
                "ordering": "none",
                "truncation_radius": 3.0,
            }
        )
        rng = np.random.default_rng(0)
        splats.create_dataset(
            "centers", data=(rng.random((n, 3)) * 10).astype("float32")
        )
        splats.create_dataset("amplitudes", data=rng.random(n).astype("float32"))
        splats.create_dataset("cholesky_factors", data=self._identity_chol(n))
        splats.create_dataset("chunk_bounds", data=np.zeros((1, 3, 2), dtype="float32"))
        zarr.consolidate_metadata(store)

    def _make_v1_1(self, path: Path, lod_sizes=(6, 3)) -> None:
        import numpy as np
        import zarr

        store = zarr.DirectoryStore(str(path))
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update(
            {
                "format_version": "1.1",
                "format_type": "gsplats_zarr",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "luxar_gsplats_version": "test",
                "n_lods": len(lod_sizes),
            }
        )
        splats = root.create_group("splats")
        splats.attrs.update(
            {
                "type": "gsplats",
                "n_lods": len(lod_sizes),
                "truncation_radius": 3.0,
            }
        )
        rng = np.random.default_rng(0)
        for i, n in enumerate(lod_sizes):
            lod = splats.create_group(f"lod_{i}")
            lod.attrs.update({"n_splats": n, "ndim": 3, "ordering": "none"})
            lod.create_dataset(
                "centers", data=(rng.random((n, 3)) * 10).astype("float32")
            )
            lod.create_dataset("amplitudes", data=rng.random(n).astype("float32"))
            lod.create_dataset("cholesky_factors", data=self._identity_chol(n))
            lod.create_dataset(
                "chunk_bounds", data=np.zeros((1, 3, 2), dtype="float32")
            )
        zarr.consolidate_metadata(store)

    def _make_sub_dir(self, dir_path: Path, level_sizes=(16, 4, 1)) -> None:
        import json

        dir_path.mkdir(parents=True, exist_ok=True)
        levels_data = []
        for i, n in enumerate(level_sizes):
            file_name = f"level_{i}.gsplats.zarr"
            self._make_v1_0(dir_path / file_name, n=n)
            levels_data.append({"level": i, "file": file_name, "n_splats": n})
        manifest = {
            "lod_kind": "substitutive",
            "compression_factor": 4,
            "levels": len(level_sizes) - 1,
            "method": "kmeans_lloyd",
            "lloyd_iterations": 5,
            "candidate_bins_k": 12,
            "seed": None,
            "input_n_splats": level_sizes[0],
            "input_ndim": 3,
            "levels_data": levels_data,
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest))

    def test_migrate_v1_0(self, runner: CliRunner, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "v2.gsplats.zarr"
        self._make_v1_0(legacy, n=7)
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out)],
        )
        assert result.exit_code == 0, f"migrate-format v1.0 failed:\n{result.stdout}"
        assert "Detected legacy format: v1.0" in result.stdout
        data = GSplatData.load(out)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        assert data.n_splats == 7

    def test_migrate_v1_1(self, runner: CliRunner, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "v2.gsplats.zarr"
        self._make_v1_1(legacy, lod_sizes=(8, 4, 2))
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out)],
        )
        assert result.exit_code == 0, f"migrate-format v1.1 failed:\n{result.stdout}"
        assert "Detected legacy format: v1.1" in result.stdout
        data = GSplatData.load(out)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 3
        assert [s.n_splats for s in data.additive_sublods] == [8, 4, 2]

    def test_migrate_substitutive_directory(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from luxar.gsplats import GSplatData

        legacy = tmp_path / "pyr"
        out = tmp_path / "v2.gsplats.zarr"
        self._make_sub_dir(legacy, level_sizes=(16, 4, 1))
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out)],
        )
        assert result.exit_code == 0, (
            f"migrate-format substitutive directory failed:\n{result.stdout}"
        )
        assert "Detected legacy format: substitutive_dir" in result.stdout
        data = GSplatData.load(out)
        assert data.n_substitutive == 3
        for level in data.substitutive_levels:
            assert level.n_additive_lods == 1
        assert [s.n_splats_total for s in data.substitutive_levels] == [16, 4, 1]
        assert [s.compression_factor for s in data.substitutive_levels] == [1, 4, 16]

    def test_migrate_refuses_existing_output(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "v2.gsplats.zarr"
        out.mkdir()
        self._make_v1_0(legacy, n=3)
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out)],
        )
        assert result.exit_code != 0
        assert "exists" in result.stdout.lower()

    def test_migrate_overwrite(self, runner: CliRunner, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "v2.gsplats.zarr"
        out.mkdir()
        self._make_v1_0(legacy, n=4)
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out), "--overwrite"],
        )
        assert result.exit_code == 0, f"--overwrite failed:\n{result.stdout}"
        assert GSplatData.load(out).n_splats == 4

    def test_migrate_quiet_suppresses_trailing_summary(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "v2.gsplats.zarr"
        self._make_v1_0(legacy, n=3)
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out), "--quiet"],
        )
        assert result.exit_code == 0, f"--quiet failed:\n{result.stdout}"
        assert "Detected legacy format: v1.0" in result.stdout
        assert "Wrote v2.0 file" not in result.stdout
