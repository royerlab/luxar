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
        assert config["n_iters"] == 500

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
        config = load_fit_config(cli_overrides={"n_iters": None, "lr": 0.05})
        assert config["lr"] == 0.05
        # n_iters should be the function default, not None
        assert config["n_iters"] is not None

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
        assert parsed["n_iters"] == 500

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
        assert parsed["n_iters"] == 6000

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
        np.testing.assert_allclose(np.sort(scaled.centers[:, 0]), np.sort(original.centers[:, 0] * 2), atol=1e-3)
        np.testing.assert_allclose(np.sort(scaled.centers[:, 1]), np.sort(original.centers[:, 1] * 3), atol=1e-3)
        np.testing.assert_allclose(np.sort(scaled.centers[:, 2]), np.sort(original.centers[:, 2] * 4), atol=1e-3)

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
                "gsplat", "transform", str(sample_gsplats), str(out),
                "--scale-intensity", "0.5",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        dimmed = GSplatData.load(out)
        # Save/load reorders splats (spatial ordering), so compare sorted values
        np.testing.assert_allclose(np.sort(dimmed.amplitudes), np.sort(original.amplitudes * 0.5), atol=1e-5)

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
        np.testing.assert_allclose(np.sort(rotated.centers[:, 0]), expected_x, atol=1e-3)
        np.testing.assert_allclose(np.sort(rotated.centers[:, 1]), expected_y, atol=1e-3)
        np.testing.assert_allclose(np.sort(rotated.centers[:, 2]), expected_z, atol=1e-3)

    def test_transform_combined(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Multiple transforms can be combined."""
        out = tmp_path / "combined.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "transform", str(sample_gsplats), str(out),
                "--scale", "1,1,2",
                "--scale-intensity", "0.1",
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
                "gsplat", "transform", str(sample_gsplats), str(out),
                "--translate", "10,20,30",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        translated = GSplatData.load(out)
        # Save/load reorders splats (spatial ordering), so compare sorted values
        np.testing.assert_allclose(np.sort(translated.centers[:, 0]), np.sort(original.centers[:, 0] + 10), atol=1e-3)
        np.testing.assert_allclose(np.sort(translated.centers[:, 1]), np.sort(original.centers[:, 1] + 20), atol=1e-3)
        np.testing.assert_allclose(np.sort(translated.centers[:, 2]), np.sort(original.centers[:, 2] + 30), atol=1e-3)
