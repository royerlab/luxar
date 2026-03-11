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
        sharpnesses=np.full(n, 2.0, dtype=np.float32),
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
            load_fit_config(preset="ultra")

    def test_sharpness_range_tuple_conversion(self) -> None:
        config = load_fit_config(preset="standard")
        sr = config["sharpness_range"]
        assert isinstance(sr, tuple)
        assert len(sr) == 2

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
                "gsplat", "fit",
                str(small_volume_npy), str(out),
                "--preset", "draft",
                "--iters", "5",
                "--seeds", "10",
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
                "gsplat", "fit",
                str(small_volume_npy), str(out),
                "--preset", "draft",
                "--iters", "5",
                "--seeds", "10",
                "--quiet",
            ],
        )
        assert result.exit_code == 0

        # Load and verify structure
        data = GSplatData.load(out)
        assert data.n_splats > 0
        assert data.ndim == 3
        assert data.amplitudes.shape == (data.n_splats,)
        assert data.sharpnesses.shape == (data.n_splats,)

    def test_fit_with_yaml_config(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """End-to-end: fit using a YAML config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("n_iters: 3\ncull_ratio: 0.0\n")

        out = tmp_path / "fitted.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "fit",
                str(small_volume_npy), str(out),
                "--config", str(config_path),
                "--seeds", "10",
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
                "gsplat", "convert",
                str(sample_gsplats), str(out),
                "--scale-intensity", "0.5",
            ],
        )
        assert result.exit_code == 0, f"convert --scale-intensity failed: {result.stdout}"
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
                "gsplat", "render",
                str(sample_gsplats), str(out),
                "--shape", "8,8,8",
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
            sharpnesses=np.full(3, 2.0, dtype=np.float32),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "merged.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "merge",
                str(sample_gsplats), str(path2),
                "-o", str(out),
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
            sharpnesses=np.full(4, 2.0, dtype=np.float32),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "stacked.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "merge",
                str(sample_gsplats), str(path2),
                "-o", str(out),
                "--as-dimension",
                "--values", "0,1",
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
            sharpnesses=np.full(3, 2.0, dtype=np.float32),
        )
        path2 = tmp_path / "test2.gsplats.zarr"
        data2.save(path2)

        out = tmp_path / "colored.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "merge",
                str(sample_gsplats), str(path2),
                "-o", str(out),
                "--channel-colors", "#ff0000,#00ff00",
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
                "gsplat", "merge",
                str(sample_gsplats), str(sample_gsplats),
                "-o", str(out),
                "--as-dimension",
                "--channel-colors", "#ff0000,#00ff00",
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
                "gsplat", "merge",
                str(sample_gsplats),
                "-o", str(out),
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
                "gsplat", "merge",
                str(sample_gsplats), str(sample_gsplats),
                "-o", str(out),
                "--channel-colors", "#ff0000,#00ff00,#0000ff",
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
                "gsplat", "merge",
                str(sample_gsplats), str(sample_gsplats),
                "-o", str(out),
                "--as-dimension",
                "--values", "0,1,2",
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
                sharpnesses=np.full(2, 2.0, dtype=np.float32),
            )
            d.save(tmp_path / name)

        out = tmp_path / "merged3.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat", "merge",
                str(sample_gsplats),
                str(tmp_path / "b.gsplats.zarr"),
                str(tmp_path / "c.gsplats.zarr"),
                "-o", str(out),
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
                "gsplat", "fit",
                str(small_volume_npy), str(gsplats_path),
                "--preset", "draft", "--iters", "5", "--seeds", "10", "--quiet",
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
                "gsplat", "fit",
                str(small_volume_npy), str(gsplats_path),
                "--preset", "draft", "--iters", "5", "--seeds", "10", "--quiet",
            ],
        )
        assert r1.exit_code == 0

        # Step 2: Render to same shape as input
        r2 = runner.invoke(
            app,
            [
                "gsplat", "render",
                str(gsplats_path), str(rendered_path),
                "--shape", "16,16,16",
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
                "gsplat", "fit",
                str(small_volume_npy), str(gsplats_path),
                "--preset", "draft", "--iters", "5", "--seeds", "10", "--quiet",
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
            sharpnesses=np.full(10, 2.0, dtype=np.float32),
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
            sharpnesses=np.full(2, 2.0, dtype=np.float32),
        )
        out = tmp_path / "roundtrip.gsplats.zarr.zip"
        original.save(out, compress="zip")

        loaded = GSplatData.load(out)
        assert loaded.n_splats == 2
        np.testing.assert_allclose(loaded.centers, original.centers, atol=0.1)
        np.testing.assert_allclose(loaded.amplitudes, original.amplitudes, atol=0.1)
