"""Tests for the gsplat CLI commands: fit, convert, render, merge.

Also tests the config system (presets, YAML loading, dump) and volume loader.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import typer
import yaml
from typer.testing import CliRunner

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

import zarr

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import create_array, read_array_meta, read_node_attrs
from luxar._zarr_compat import open_group as zc_open_group
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
from luxar.cli.tests._testing import normalized_cli_output
from luxar.conftest import confine_temp_dirs
from luxar.core.dimension_inference import infer_discrete_step

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI SGR codes so option-name substring checks are color-robust.

    Rich renders an option like ``--floor`` as ``-`` + SGR + ``-floor`` when
    color is on (CI forces a TTY), which breaks a naive ``"--floor" in stdout``.
    Stripping the SGR codes rejoins the name.
    """
    return _ANSI_RE.sub("", text)


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
def sample_gsplats_4d(tmp_path: Path) -> Path:
    """A 4D stacked .gsplats.zarr: spatial x,y,z + a trailing time column.

    Built with ``combine_as_new_dimension`` (the repo's stacking convention:
    the new axis is appended LAST, with sigma=0 so the time axis is
    degenerate). Two distinct timepoints (t=0 and t=7); each splat carries a
    unique amplitude so rows can be re-aligned after the spatial reordering
    that save/load applies (amplitudes are rotation-invariant).
    """
    from luxar.gsplats.gsplat_data import GSplatData

    centers = np.array(
        [[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    identity_chol = np.tile(
        np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (3, 1)
    )

    def _make_timepoint(amplitudes: list[float]) -> GSplatData:
        return GSplatData(
            centers=centers.copy(),
            amplitudes=np.array(amplitudes, dtype=np.float32),
            cholesky_factors=identity_chol.copy(),
        )

    combined = GSplatData.combine_as_new_dimension(
        [_make_timepoint([0.1, 0.2, 0.3]), _make_timepoint([0.4, 0.5, 0.6])],
        values=[0.0, 7.0],
        sigma=0.0,
    )
    out = tmp_path / "test4d.gsplats.zarr"
    combined.save(out)
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

    def test_command_defaults_displace_function_defaults(self) -> None:
        """A per-command default beats the harvested function default (#1729).

        This is the only layer that can: `setdefault` on the resolved config is a
        no-op, because every signature default is already a key by then.
        """
        assert load_fit_config()["cull_retention"] == 0.95  # function default
        config = load_fit_config(command_defaults={"cull_retention": 0.999})
        assert config["cull_retention"] == 0.999

    def test_preset_beats_command_defaults(self) -> None:
        config = load_fit_config(
            preset="draft", command_defaults={"n_iters": 123, "cull_retention": 0.999}
        )
        assert config["n_iters"] == PRESETS["draft"]["n_iters"]
        # The command default is still visible where the preset says nothing.
        config = load_fit_config(preset="draft", command_defaults={"lr": 0.077})
        assert config["lr"] == 0.077

    def test_yaml_beats_command_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("cull_retention: 0.5\n")
        config = load_fit_config(
            config_path=yaml_path, command_defaults={"cull_retention": 0.999}
        )
        assert config["cull_retention"] == 0.5

    def test_cli_beats_command_defaults(self) -> None:
        config = load_fit_config(
            cli_overrides={"cull_retention": 0.0},
            command_defaults={"cull_retention": 0.999},
        )
        # 0.0 is not None, so the "keep every splat" value must survive.
        assert config["cull_retention"] == 0.0

    def test_command_defaults_skip_none_values(self) -> None:
        """A sentinel-free dict: a None entry leaves the layer below alone."""
        config = load_fit_config(command_defaults={"cull_retention": None})
        assert config["cull_retention"] == 0.95

    def test_omitting_command_defaults_changes_nothing(self) -> None:
        assert load_fit_config() == load_fit_config(command_defaults=None)
        assert load_fit_config(preset="hifi") == load_fit_config(
            preset="hifi", command_defaults={}
        )


class TestSourceDtypePrecedence:
    """A `source_dtype:` a user put in a --config must beat the loader's guess.

    `load_fit_config` passes arbitrary YAML keys through, so the key reaches the
    fit config — and there is no CLI flag to override it afterwards. On a float32
    .npy exported from a 16-bit acquisition the loader can only report float32,
    which is exactly the 2x-overstated `source_bytes` the stamp exists to prevent.
    """

    def test_a_user_supplied_dtype_wins(self) -> None:
        from luxar.cli.gsplat_ops.fitting.fit import _stamp_source_dtype

        config = {"source_dtype": "uint16"}
        _stamp_source_dtype(config, {"source_dtype": "float32"})
        assert config["source_dtype"] == "uint16"

    def test_the_signature_derived_none_is_still_filled_in(self) -> None:
        """`get_fit_defaults()` injects `source_dtype: None` from the signature,
        so mere PRESENCE of the key cannot mean "the user chose this"."""
        from luxar.cli.gsplat_ops.fitting.fit import _stamp_source_dtype

        assert load_fit_config()["source_dtype"] is None  # precondition
        config = load_fit_config()
        _stamp_source_dtype(config, {"source_dtype": "uint16"})
        assert config["source_dtype"] == "uint16"

    def test_yaml_source_dtype_reaches_the_config(self, tmp_path: Path) -> None:
        """The precondition for the above being a real path, not a hypothetical."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("source_dtype: uint16\n")
        assert load_fit_config(config_path=yaml_path)["source_dtype"] == "uint16"


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
        path = tmp_path / "test.luxar.zarr"
        z = zarr.open(str(path), mode="w", shape=vol.shape, dtype=vol.dtype)
        z[:] = vol
        loaded = load_volume(path)
        assert loaded.shape == (8, 8, 8)

    def test_load_zarr_5d(self, tmp_path: Path) -> None:
        import zarr

        # Simulate 5D OME-ZARR (T=2, C=3, Z=4, Y=4, X=4)
        vol = np.random.rand(2, 3, 4, 4, 4).astype(np.float32)
        path = tmp_path / "test.luxar.zarr"
        root = zarr.open_group(str(path), mode="w")
        create_array(root, "0", data=vol)
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
        path = tmp_path / "test.luxar.zarr"
        root = zarr.open_group(str(path), mode="w")
        create_array(root, "my_volume", data=vol)
        create_array(root, "other_data", data=np.zeros(10))
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

    @pytest.mark.parametrize(
        ("coordinates", "expected_step"),
        [([1, 6, 11], 5.0), ([-9, -4, 1], 5.0), ([1, 7, 11], 2.0)],
    )
    def test_nd_hidden_dimension_infers_offset_integer_stride(
        self, coordinates: list[int], expected_step: float
    ) -> None:
        centers = np.column_stack(
            (np.arange(3), np.arange(3), np.arange(3), coordinates)
        ).astype(np.float32)
        dims = build_dimensions_from_data(centers)

        hidden = dims.dimensions[3]
        assert hidden.range == (float(min(coordinates)), float(max(coordinates)))
        assert hidden.step == expected_step
        assert hidden.discrete is True

    def test_nd_hidden_dimension_falls_back_for_inexact_float32_integer(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        centers = np.array([[0, 0, 0, 0], [1, 1, 1, (1 << 24) + 2]], dtype=np.float64)

        dims = build_dimensions_from_data(centers)

        assert dims.dimensions[3].step == 1.0
        assert "within ±2^24" in capsys.readouterr().out

    @pytest.mark.parametrize("coordinates", [[0.0, np.inf], [np.nan]])
    def test_discrete_step_falls_back_for_non_finite_coordinates(
        self, coordinates: list[float], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert infer_discrete_step(np.asarray(coordinates)) == 1.0
        assert "must be finite" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════
# Command integration tests
# ═══════════════════════════════════════════════════════════════════════


# Full expected output of `gsplat fit --dump-config --preset hifi`, snapshotted
# from the command before the run_fit_volume decomposition (2026-07) — a
# byte-exact safety net for the fit command's --dump-config early-exit path.
_DUMP_CONFIG_HIFI_GOLDEN = """\
# ============================================================
# Luxar Gaussian Splat Fitting Configuration
# ============================================================
# Base preset: hifi
# Priority: CLI flags > YAML config > preset > command defaults > function defaults
#
# Usage:
#   luxar gsplat fit volume.npy output.gsplats.zarr --config this_file.yaml
#   luxar gsplat fit volume.npy output.gsplats.zarr --preset hifi --config overrides.yaml
#
# Seed generation kwargs (num_scales, percentile_thresh, spacing, etc.)
# can also be set here and will be forwarded to the seed generator.

# --- Basic Parameters ---
n_iters: 10000            # Max optimization iterations
lr: 0.01                      # Adam learning rate
loss_type: "l1"        # Loss function: l1, mse, or poisson
seed_method: "auto"  # Seed method: auto, edges, grid, decomposition

# --- Preprocessing ---
norm_percentile: 0.0  # Percentile clipping (0=full range, >0=robust)
floor: "auto"                # Background/DC suppression: auto | pN (e.g. p10) | <float> | none
downscale: null            # Downsample by integer factor (null=disabled, e.g. 4 or [1,4,4])

# --- Regularization ---
asymmetric_penalty: 1.0  # Over-prediction penalty (null=disabled)
l1_amp: null              # L1 on amplitudes (null=auto: 0.1*lr)
l1_diag: null            # L1 on Cholesky diagonals (null=auto: 0.01*lr)

# --- Shape Constraints ---
sigma_min_diag: 0.28867513459481287  # Min Cholesky diagonal
sigma_max_diag: null  # Max Cholesky diagonal (null=unbounded)
amp_max: null            # Max amplitude (null=auto: 1.0)
max_eccentricity: 15.0  # Max axis ratio (null=no constraint)
truncate: 2.75          # Truncation radius in sigma

# --- Convergence ---
max_abs_error: null  # Absolute error threshold (null=auto: 0.01)
rel_l2_target: null  # Relative L2 threshold (null=disabled)
gradient_clip: null   # Gradient norm clipping (null=disabled)
scheduler_type: "plateau"  # LR scheduler: plateau or exponential
patience: 15          # Iterations before LR reduction
lr_reduction_factor: 0.9  # LR multiplier on plateau
early_stop_patience: 400  # Stop after N iters without improvement

# --- Dynamic Operations ---
enable_dynamic_ops: true  # Enable splat relocation during optimization
dynamic_ops_verbose: false  # Verbose logging for dynamic ops

# --- Post-Processing ---
cull_retention: 0.999  # Post-fit cumulative culling (0-1, null=disabled)
voxel_footprint_correction: false  # Inflate covariances by voxel footprint

# --- Boundary Containment ---
boundary_penalty: null  # Boundary penalty weight (null=disabled)
clip_to_bounds: false  # Hard clip splats to volume bounds

# --- Anisotropic Voxels ---
voxel_size: null      # Physical spacing (null=isotropic)
output_space: "real"  # Output coords: real or voxel

# --- Performance ---
sort_splats_enabled: true  # Periodic Morton-code sorting for GPU cache locality
sort_splats_interval: 1000  # Sort every N iterations (also sorts at iteration 0)

# --- Hardware ---
use_metal: true        # Metal acceleration (macOS Apple Silicon)
use_cuda: true          # Custom CUDA kernels (NVIDIA GPUs)

"""


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

    def test_dump_config_hifi_golden(self, runner: CliRunner) -> None:
        """`fit --dump-config --preset hifi` output is byte-identical to the
        pre-refactor snapshot (safety net for the run_fit_volume decomposition)."""
        result = runner.invoke(
            app, ["gsplat", "fit", "--dump-config", "--preset", "hifi"]
        )
        assert result.exit_code == 0
        assert result.stdout == _DUMP_CONFIG_HIFI_GOLDEN

    def test_fit_help_exposes_floor_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["gsplat", "fit", "--help"])
        assert result.exit_code == 0
        assert "--floor" in _plain(result.stdout)

    def test_dump_config_includes_floor_default(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["gsplat", "fit", "--dump-config"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.stdout)
        assert parsed["floor"] == "auto"

    def test_cal_help_exposes_floor_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["gsplat", "cal", "--help"])
        assert result.exit_code == 0
        assert "--floor" in _plain(result.stdout)

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

    def test_fit_accepts_auto_device(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "fitted-auto.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(small_volume_npy),
                str(out),
                "--iters",
                "1",
                "--seeds",
                "2",
                "--tiling",
                "none",
                "--device",
                "auto",
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


class TestCullCommand:
    def test_cull_accepts_auto_device(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "culled-auto.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cull",
                str(sample_gsplats),
                str(out),
                "--method",
                "redundancy",
                "--shape",
                "16,16,16",
                "--device",
                "auto",
            ],
        )

        assert result.exit_code == 0, f"cull failed: {result.stdout}"
        assert out.exists()

    def test_cull_target_is_shifted_to_the_stored_fit_basis(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData.load(sample_gsplats)
        data.stats["image_min"] = 2.0
        based = tmp_path / "based.gsplats.zarr"
        data.save(based)
        target = tmp_path / "target.npy"
        np.save(target, np.full((16, 16, 16), 2.0, np.float32))
        captured = {}

        def fake_cull(self, *, target=None, **kwargs):
            captured["target"] = np.asarray(target).copy()
            return self

        monkeypatch.setattr(GSplatData, "cull", fake_cull)
        out = tmp_path / "culled.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cull",
                str(based),
                str(out),
                "--method",
                "error_budget",
                "--target",
                str(target),
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert np.count_nonzero(captured["target"]) == 0


class TestConvertCommand:
    def test_convert_basic(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "scene.luxar.zarr"
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
        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "convert", str(sample_gsplats), str(out), "--no-center"],
        )
        assert result.exit_code == 0

    def test_convert_partition_file_grafts(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """`gsplat convert` on a kind=partition file grafts the node tree into
        the scene. Regression: it used to crash with a raw traceback because
        GSplatData.load can't represent a partition root."""
        part = tmp_path / "part.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
            ).exit_code
            == 0
        )
        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(app, ["gsplat", "convert", str(part), str(out)])
        assert result.exit_code == 0, f"convert on partition failed: {result.stdout}"

        import zarr

        store = zarr.open_group(str(out), mode="r")
        assert store["gsplats"].attrs["kind"] == "partition"

    def test_convert_with_scale_intensity(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        raw = tmp_path / "raw.gsplats.zarr"
        raw_data = GSplatData.load(sample_gsplats).scale_intensity(1000.0)
        raw_data.save(raw)
        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "convert",
                str(raw),
                str(out),
                "--scale-intensity",
                "0.5",
            ],
        )
        assert result.exit_code == 0, (
            f"convert --scale-intensity failed: {result.stdout}"
        )
        attrs = read_node_attrs(out / "gsplats")
        expected_hi = float(np.percentile(raw_data.amplitudes * 0.5, 99.9))
        assert attrs["amplitude_data_range"][1] == pytest.approx(expected_hi, rel=1e-3)
        assert "amplitude_normalization_factor" not in attrs

    def test_convert_scene_has_valid_structure(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Verify the converted scene has proper zarr structure."""
        import zarr

        out = tmp_path / "scene.luxar.zarr"
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
        # Should have data arrays (v3.1 splits Cholesky into diag + offdiag)
        assert "centers" in gsplats_group
        assert "amplitudes" in gsplats_group
        assert "cholesky_factors_diag" in gsplats_group
        assert "cholesky_factors_offdiag" in gsplats_group

    def test_convert_colormap_and_tone_mapping(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """--colormap writes the node attr; --tone-mapping writes the scene
        viewer_config."""
        import zarr

        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "convert",
                str(sample_gsplats),
                str(out),
                "--colormap",
                "plasma",
                "--tone-mapping",
                "Neutral",
                "--gamma",
                "1.0",
                "--intensity",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.stdout
        store = zarr.open_group(str(out), mode="r")
        assert store["gsplats"].attrs.get("colormap") == "plasma"
        assert store["gsplats"].attrs.get("layer") is True
        assert store.attrs.get("viewer_config", {}).get("tone_mapping") == "Neutral"

    def test_convert_invalid_colormap_errors(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "convert",
                str(sample_gsplats),
                str(tmp_path / "s.luxar.zarr"),
                "--colormap",
                "notacolormap",
            ],
        )
        assert result.exit_code != 0

    def test_convert_invalid_tone_mapping_errors(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "convert",
                str(sample_gsplats),
                str(tmp_path / "s.luxar.zarr"),
                "--tone-mapping",
                "Fancy",
            ],
        )
        assert result.exit_code != 0

    def test_convert_no_layer(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        import zarr

        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "convert", str(sample_gsplats), str(out), "--no-layer"],
        )
        assert result.exit_code == 0, result.stdout
        store = zarr.open_group(str(out), mode="r")
        assert store["gsplats"].attrs.get("layer") is False


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
        scene_path = tmp_path / "scene.luxar.zarr"

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

    def test_filter_percentile_syntax(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        # 'pNN' percentile value syntax: --amplitude-max p90 drops the brightest
        # (the fixture amplitudes are linspace(0.1, 1.0, 10)).
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-max",
                "p90",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        from luxar.gsplats.gsplat_data import GSplatData

        assert GSplatData.load(out).n_splats == 9

    def test_filter_volume_and_mass(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        # Previously-uncovered --volume/--mass criteria at the CLI level.
        out = tmp_path / "filtered.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--volume-max",
                "1000",
                "--mass-min",
                "0.0",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

    def test_filter_isolation(
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
                "--isolation-max",
                "p50",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert out.exists()

    def test_filter_dry_run_writes_nothing(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "should_not_exist.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--scale-max",
                "p90",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        assert not out.exists()
        assert "dry-run" in result.stdout.lower()

    def test_filter_soft_highpass(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "soft.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--soft-highpass",
                "p50",
            ],
        )
        assert result.exit_code == 0, f"filter failed: {result.stdout}"
        from luxar.gsplats.gsplat_data import GSplatData

        # Soft filter attenuates, never removes: count is preserved.
        assert GSplatData.load(out).n_splats == 10

    def test_filter_bad_percentile_errors(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--scale-max",
                "p150",
            ],  # percentile out of [0,100]
        )
        assert result.exit_code != 0

    def test_filter_mixed_mode_errors(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        # Mixing percentile + absolute on one attribute's min/max is rejected
        # (would otherwise silently read the absolute value as a percentile).
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--amplitude-min",
                "p10",
                "--amplitude-max",
                "0.9",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_filter_duplicate_spatial_dims_errors(
        self, runner: CliRunner, sample_gsplats_for_filter: Path, tmp_path: Path
    ) -> None:
        # A duplicate axis in --spatial-dims used to be silently accepted and
        # double-counted in the spatial-metric geometric mean (issue #765);
        # it must now be rejected instead of producing plausible-but-wrong output.
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "filter",
                str(sample_gsplats_for_filter),
                str(out),
                "--scale-max",
                "p90",
                "--spatial-dims",
                "0,0,1",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()


# ═══════════════════════════════════════════════════════════════════════
# Partition command tests
# ═══════════════════════════════════════════════════════════════════════


class TestPartitionCommand:
    """Tests for luxar gsplat partition CLI command."""

    def _read_partition(self, path: Path):
        """Read a kind=partition .gsplats.zarr → its tree node."""
        import zarr

        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        root = zarr.open_group(str(path), mode="r")
        return root, read_gsplat_node(root, root)

    def test_partition_by_parts_single_file(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(out), "--parts", "3"],
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        # One self-contained kind=partition file (NOT a directory of N files).
        assert out.is_dir()  # a zarr dir
        from luxar.gsplats.tree import iter_leaves

        root, node = self._read_partition(out)
        assert root.attrs["kind"] == "partition"
        # all splats preserved across the spatial parts
        assert sum(leaf.n_splats for leaf in iter_leaves(node)) == 5

    def test_partition_by_max_elements(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "partition",
                str(sample_gsplats),
                str(out),
                "--max-elements",
                "2",
            ],
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        from luxar.gsplats.tree import iter_leaves

        _, node = self._read_partition(out)
        leaves = list(iter_leaves(node))
        assert all(leaf.n_splats <= 2 for leaf in leaves)
        assert sum(leaf.n_splats for leaf in leaves) == 5

    def test_partition_with_compression(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "part.gsplats.zarr.zip"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "partition",
                str(sample_gsplats),
                str(out),
                "--parts",
                "2",
                "--compress",
                "zip",
            ],
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        assert out.is_file()  # single compressed archive

    def test_partition_missing_mode(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Neither --max-elements nor --parts -> error."""
        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "partition", str(sample_gsplats), str(out)]
        )
        assert result.exit_code == 1

    def test_partition_indices_flag_removed(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """--indices was removed (BSP is spatial, not index-based)."""
        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(out), "--indices", "2,4"],
        )
        assert result.exit_code != 0  # unknown option

    def test_partition_total_preserved(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Spatial partition preserves the total splat count in one file."""
        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "partition", str(sample_gsplats), str(out), "--parts", "2"]
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.tree import iter_leaves

        original = GSplatData.load(sample_gsplats)
        _, node = self._read_partition(out)
        assert sum(leaf.n_splats for leaf in iter_leaves(node)) == original.n_splats


# ═══════════════════════════════════════════════════════════════════════
# View command tests
# ═══════════════════════════════════════════════════════════════════════


class TestViewCommand:
    """`gsplat view` serves the .gsplats.zarr node tree directly (no scene
    round-trip), so it works for every tree shape — including partition files
    that `GSplatData.load` refuses (decision 5 gap)."""

    def _invoke_view(self, runner: CliRunner, path: Path):
        """Invoke `gsplat view` with all blocking/IO side effects mocked out."""
        from unittest.mock import patch

        captured: dict = {}

        def _fake_serve_data(target, *args, **kwargs):
            captured["serve_target"] = target
            # Recorded from INSIDE the server thread, while the target is still
            # on disk: `view` removes any temp directory it extracted as soon as
            # it returns, so an archive case cannot be inspected afterwards.
            captured["target_is_dir"] = target.is_dir()
            try:
                captured["root_format_type"] = zc_open_group(
                    str(target), mode="r"
                ).attrs.get("format_type")
            except Exception as exc:  # not a store root — a finding, not a crash
                captured["root_format_type"] = repr(exc)

        def _fake_wait(_host, _port, thread=None, **_kwargs):
            # Join the data thread instead of racing it: `_serve_data` is the
            # only thing that records what the command served, and every
            # assertion here is about exactly that.
            if thread is not None:
                thread.join(timeout=30)
            return True

        with (
            patch("luxar.cli.utils.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.utils.find_available_port",
                side_effect=lambda p, **_kw: p,
            ),
            patch("luxar.cli.utils.wait_for_server", side_effect=_fake_wait),
            patch("luxar.cli.serving._serve_data", side_effect=_fake_serve_data),
            patch("luxar.cli.serving._serve_viewer"),
        ):
            result = runner.invoke(app, ["gsplat", "view", str(path), "--no-open"])
        return result, captured

    @staticmethod
    def _flat_zip(store: Path, archive: Path) -> Path:
        """Zip a store with its root AT THE ARCHIVE ROOT (the #1628 shape)."""
        import zipfile

        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for f in sorted(store.rglob("*")):
                if f.is_file():
                    zip_ref.write(f, arcname=str(f.relative_to(store)))
        return archive

    def test_view_leaf_serves_directly(
        self, runner: CliRunner, sample_gsplats: Path
    ) -> None:
        result, captured = self._invoke_view(runner, sample_gsplats)
        assert result.exit_code == 0, f"view failed: {result.stdout}"
        # Served the file itself — no temp scene compile.
        assert captured["serve_target"] == sample_gsplats

    def test_view_partition_does_not_crash(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Previously `view` round-tripped via GSplatData.load → ValueError on a
        partition tree. Serving directly must succeed."""
        part = tmp_path / "part.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
            ).exit_code
            == 0
        )
        result, captured = self._invoke_view(runner, part)
        assert result.exit_code == 0, f"view on partition failed: {result.stdout}"
        assert captured["serve_target"] == part

    def test_view_refuses_a_plain_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """A regular file that is not an archive gets a one-line refusal."""
        path = tmp_path / "notes.txt"
        path.write_text("not a zarr store")
        result, captured = self._invoke_view(runner, path)
        assert result.exit_code == 1, result.stdout
        assert "Not a .gsplats.zarr directory or archive" in _plain(result.stdout)
        assert "serve_target" not in captured

    @pytest.mark.skipif(
        not hasattr(os, "mkfifo"), reason="no os.mkfifo on this platform"
    )
    def test_view_refuses_a_non_regular_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A FIFO is refused too, not served as a store directory.

        The refusal cannot be delegated to ``resolve_store_path``: that raises for
        ``path.is_file()``, i.e. for REGULAR files only, so a FIFO or a device node
        (both of which satisfy typer's ``exists=True``) fell through and became the
        serve target — the data server thread then died on "Data mount root must be
        a directory" while the command sat blocked on a viewer serving nothing.
        """
        fifo = tmp_path / "f.fifo"
        os.mkfifo(fifo)
        result, captured = self._invoke_view(runner, fifo)
        assert result.exit_code == 1, result.stdout
        assert "Not a .gsplats.zarr directory or archive" in _plain(result.stdout)
        assert "serve_target" not in captured

    def test_view_serves_a_flat_archive_as_a_directory(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """A FLAT archive is EXTRACTED and the extraction is what gets served.

        Nothing else in this class invokes `view` on an archive at all, so both
        the "Extracting compressed dataset..." branch and this change's own claim
        that a flat archive works here were unexercised. A store is mounted over
        HTTP as a DIRECTORY, so resolving with ``flat_zip_in_place=True`` — which
        hands back the zip itself — would give the data server a file as its
        mount root and fail inside the server thread while the command sat
        blocked on a viewer serving nothing.
        """
        archive = self._flat_zip(sample_gsplats, tmp_path / "flat.gsplats.zarr.zip")

        result, captured = self._invoke_view(runner, archive)
        assert result.exit_code == 0, f"view on a flat archive failed: {result.stdout}"
        assert "Extracting compressed dataset" in _plain(result.stdout)
        assert captured["serve_target"] != archive
        assert captured["target_is_dir"] is True, captured["serve_target"]
        # And it is the store ROOT, not one of its array sub-directories: that
        # arbitrary-child resolution is the bug #1628 is about.
        assert captured["root_format_type"] == "gsplats_zarr"

    def test_view_removes_its_extraction_on_a_normal_return(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """The extracted copy is cleaned up when `view` simply finishes.

        Cleanup used to live only in the `KeyboardInterrupt` and generic
        `except` handlers, so a normal return from the blocking viewer — the
        ordinary way this command ends — left a full uncompressed copy of the
        dataset in ``/tmp``. The shared confinement helper keeps the probe from
        seeing (or being confused by) a concurrent process's extraction.
        """
        confine_temp_dirs(tmp_path, monkeypatch)
        archive = self._flat_zip(sample_gsplats, tmp_path / "flat.gsplats.zarr.zip")

        result, captured = self._invoke_view(runner, archive)
        assert result.exit_code == 0, f"view on a flat archive failed: {result.stdout}"
        # The premise: something WAS extracted under the redirected temp root.
        served = captured["serve_target"]
        assert str(served).startswith(str(tmp_path)), served
        assert list(tmp_path.glob("luxar_gsplat_*")) == []


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
    @pytest.mark.parametrize("value", ["-5", "nan", "inf"])
    def test_compare_rejects_invalid_image_min(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
        value: str,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(sample_gsplats),
                str(small_volume_npy),
                "--image-min",
                value,
            ],
        )
        assert result.exit_code != 0
        assert "finite, non-negative" in _plain(result.stderr)

    def test_compare_stored_basis_matches_a_pre_shifted_reference(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        import json

        from luxar.gsplats.gsplat_data import GSplatData

        level = 0.25
        data = GSplatData.load(sample_gsplats)
        data.stats["image_min"] = level
        based = tmp_path / "based.gsplats.zarr"
        data.save(based)

        raw_json = tmp_path / "raw.json"
        raw = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(based),
                str(small_volume_npy),
                "--device",
                "cpu",
                "--output-json",
                str(raw_json),
            ],
        )
        assert raw.exit_code == 0, raw.stdout
        assert "from dataset" in raw.stdout

        shifted_path = tmp_path / "shifted.npy"
        shifted = np.clip(np.load(small_volume_npy) - level, 0.0, None)
        np.save(shifted_path, shifted)
        shifted_json = tmp_path / "shifted.json"
        control = runner.invoke(
            app,
            [
                "gsplat",
                "compare",
                str(based),
                str(shifted_path),
                "--image-min",
                "0",
                "--device",
                "cpu",
                "--output-json",
                str(shifted_json),
            ],
        )
        assert control.exit_code == 0, control.stdout

        raw_metrics = json.loads(raw_json.read_text())
        shifted_metrics = json.loads(shifted_json.read_text())
        for key in ("mse", "psnr_db", "ssim", "rel_l2", "max_abs_error"):
            assert raw_metrics[key] == pytest.approx(shifted_metrics[key])

    def test_compare_accepts_auto_device(
        self,
        runner: CliRunner,
        sample_gsplats: Path,
        small_volume_npy: Path,
    ) -> None:
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
                "auto",
            ],
        )

        assert result.exit_code == 0, f"compare failed: {result.stdout}"
        assert "Rendering on " in result.stdout
        assert "Rendering on auto" not in result.stdout

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

    @staticmethod
    def _aligned_by_amplitude(data: "GSplatData") -> np.ndarray:
        """Centers sorted by the per-splat unique amplitude key.

        Save/load reorders splats spatially; amplitudes are unique in the 4D
        fixture and invariant under rotation, so sorting by them re-aligns
        rows between the original and transformed datasets.
        """
        order = np.argsort(data.amplitudes)
        return np.asarray(data.centers[order])

    def test_transform_rotate_x_4d_preserves_time(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """--rotate-x on 4D stacked data rotates dims 1,2 and leaves time alone.

        Regression test for the rotation being embedded in the LAST 3 center
        dims: on (x, y, z, t) data that mixed z with t, collapsing all
        timepoints and corrupting the time column.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats_4d)
        out = tmp_path / "rotated4d.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-x",
                "90",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        orig_c = self._aligned_by_amplitude(original)
        rot_c = self._aligned_by_amplitude(rotated)

        # Time column (trailing stacked dim) is exactly preserved per splat.
        np.testing.assert_array_equal(rot_c[:, 3], orig_c[:, 3])
        # 90° X rotation acts on dims 1,2: new_y = -z, new_z = y; x unchanged.
        np.testing.assert_allclose(rot_c[:, 0], orig_c[:, 0], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 1], -orig_c[:, 2], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 2], orig_c[:, 1], atol=1e-3)

    def test_transform_rotate_4d_time_axis_stays_degenerate(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """Rotation must not leak spatial covariance into the time axis.

        Covariance transforms as A·Σ·Aᵀ, so a rotation placed on the wrong
        dims gives the zero-variance time axis a spatial sigma — breaking the
        degenerate-axis auto-detection every scale/eccentricity/isolation
        filter depends on. Uses the same max-marginal-sigma notion as
        ``luxar.gsplats.utils.spatial_axes``.
        """
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.utils.spatial_axes import (
            SPATIAL_SIGMA_EPS,
            spatial_axes_from_max_sigma,
        )

        out = tmp_path / "rotated4d.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-x",
                "90",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        max_sigma = rotated.marginal_sigmas().max(axis=0)
        assert max_sigma[3] <= SPATIAL_SIGMA_EPS, (
            f"time axis gained spatial sigma {max_sigma[3]} after rotation"
        )
        np.testing.assert_array_equal(
            spatial_axes_from_max_sigma(max_sigma), np.array([0, 1, 2])
        )

    def test_transform_rotate_z_4d_matches_3d_plane(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """--rotate-z on 4D rotates the same (x, y) plane it does on 3D."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats_4d)
        out = tmp_path / "rotated4d_z.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-z",
                "90",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        orig_c = self._aligned_by_amplitude(original)
        rot_c = self._aligned_by_amplitude(rotated)

        # Same plane as the 3D test: new_x = -y, new_y = x; z and t unchanged.
        np.testing.assert_allclose(rot_c[:, 0], -orig_c[:, 1], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 1], orig_c[:, 0], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 2], orig_c[:, 2], atol=1e-3)
        np.testing.assert_array_equal(rot_c[:, 3], orig_c[:, 3])

    def test_transform_spatial_dims_escape_hatch(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """--spatial-dims 0,2,3 rotates dims 0,2,3 and leaves dim 1 untouched.

        The escape hatch for a direct nD fit: the user picks which three
        center dims the 3x3 rotation acts on. The non-contiguous selection
        with a Z rotation (which moves the X role) discriminates against BOTH
        regressions: the legacy last-3 embedding (would rotate dims 1,2,3)
        and a parsed-but-ignored flag (would rotate dims 0,1,2).
        """
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats_4d)
        out = tmp_path / "rotated4d_dims023.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-z",
                "90",
                "--spatial-dims",
                "0,2,3",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        orig_c = self._aligned_by_amplitude(original)
        rot_c = self._aligned_by_amplitude(rotated)

        # Dim 1 is exactly untouched (not selected).
        np.testing.assert_array_equal(rot_c[:, 1], orig_c[:, 1])
        # 90° Z rotation with (X,Y,Z) roles = dims (0,2,3): new X = -Y,
        # new Y = X, Z unchanged → new dim0 = -dim2, new dim2 = dim0,
        # dim3 (the Z role, here time) exactly unchanged.
        np.testing.assert_allclose(rot_c[:, 0], -orig_c[:, 2], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 2], orig_c[:, 0], atol=1e-3)
        np.testing.assert_array_equal(rot_c[:, 3], orig_c[:, 3])

    def test_transform_spatial_dims_order_assigns_rotation_roles(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """A non-sorted axis list must retain its X/Y/Z role assignment."""
        from luxar.gsplats.gsplat_data import GSplatData

        original = GSplatData.load(sample_gsplats_4d)
        out = tmp_path / "rotated4d_dims203.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-z",
                "90",
                "--spatial-dims",
                "2,0,3",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        rotated = GSplatData.load(out)
        orig_c = self._aligned_by_amplitude(original)
        rot_c = self._aligned_by_amplitude(rotated)

        # (X,Y,Z) roles = dims (2,0,3): new X = -Y, new Y = X.
        np.testing.assert_allclose(rot_c[:, 2], -orig_c[:, 0], atol=1e-3)
        np.testing.assert_allclose(rot_c[:, 0], orig_c[:, 2], atol=1e-3)
        np.testing.assert_array_equal(rot_c[:, 1], orig_c[:, 1])
        np.testing.assert_array_equal(rot_c[:, 3], orig_c[:, 3])

    # ── a partition's split planes across transform / additive ──────────
    #
    # `bsp_tree` records how the parts stack up, in the CENTERS' coordinate
    # space. Two ways to get it wrong, both silent: drop it (ordering quietly
    # degrades to the centroid heuristic) or keep it stale after moving the
    # centers (the traversal still returns a valid-looking permutation, so the
    # viewer draws confidently in the wrong order).

    @staticmethod
    def _partition_store(path: Path, n: int = 400) -> None:
        """Write a real BSP partition, so it genuinely carries split planes."""
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        rng = np.random.default_rng(17)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        node = GSplatData(
            centers=(rng.random((n, 3)) * 100).astype(np.float32),
            amplitudes=rng.uniform(0.2, 1.0, size=(n,)).astype(np.float32),
            cholesky_factors=chol,
        ).to_spatial_partition(max_elements=n // 4)
        assert node.bsp_tree is not None
        write_gsplats_tree(path, node)

    @staticmethod
    def _assert_planes_match_centers(path: Path) -> None:
        """Every stored plane must still separate the parts it claims to."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node

        node, _ = load_gsplat_node(path)
        tree = node.bsp_tree
        assert tree is not None, "partition lost its split planes"

        def labels(nd: dict) -> list[int]:
            if "part" in nd:
                return [nd["part"]]
            return labels(nd["left"]) + labels(nd["right"])

        def centers(i: int) -> np.ndarray:
            return np.concatenate(
                [np.asarray(s.centers) for s in node.children[i].additive_sublods]
            )

        def check(nd: dict) -> None:
            if "part" in nd:
                return
            axis, split = nd["axis"], nd["split"]
            for label in labels(nd["left"]):
                assert centers(label)[:, axis].max() < split
            for label in labels(nd["right"]):
                assert centers(label)[:, axis].min() >= split
            check(nd["left"])
            check(nd["right"])

        check(tree)

    @pytest.mark.parametrize(
        "args",
        [
            ["--scale", "4,1,1"],
            ["--translate", "10,-5,3"],
            ["--center"],
            ["--rotate-z", "90"],
            ["--scale-intensity", "0.5"],
            ["--scale", "2,2,2", "--translate", "5,5,5", "--center"],
        ],
    )
    def test_transform_carries_partition_split_planes(
        self, runner: CliRunner, tmp_path: Path, args: list
    ) -> None:
        """Axis-preserving transforms keep the planes AND move them correctly."""
        src = tmp_path / "part.gsplats.zarr"
        self._partition_store(src)
        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "transform", str(src), str(out), *args])
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        self._assert_planes_match_centers(out)

    def test_transform_drops_split_planes_on_an_arbitrary_rotation(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A non-quarter-turn rotation shears the cells out of axis-alignment,
        which the format cannot express — drop the tree, and say so."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node

        src = tmp_path / "part.gsplats.zarr"
        self._partition_store(src)
        out = tmp_path / "rot37.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "transform", str(src), str(out), "--rotate-z", "37"]
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        node, _ = load_gsplat_node(out)
        assert node.bsp_tree is None
        assert "Dropped the split planes" in result.stdout

    def test_additive_preserves_partition_split_planes(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Re-laddering leaves moves no centers, so the planes stay valid."""
        src = tmp_path / "part.gsplats.zarr"
        self._partition_store(src)
        out = tmp_path / "laddered.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "additive", str(src), str(out), "--n-lods", "2"]
        )
        assert result.exit_code == 0, f"additive failed: {result.stdout}"
        self._assert_planes_match_centers(out)

    @pytest.mark.parametrize(
        "bad_value",
        [
            "-1,1,2",
            "1_0,1,2",
            "+0,+1,+2",
            "０,１,２",
            "٠,١,٢",
            # all-digit token above CPython's int-str conversion limit
            # (sys.get_int_max_str_digits, 4300): int() raises ValueError
            # even though the syntax gate passes — same clean error, no
            # traceback, still before the load.
            pytest.param("9" * 5000 + ",1,2", id="digit-limit-overflow"),
        ],
    )
    def test_transform_spatial_dims_invalid_syntax_rejected_before_load(
        self,
        runner: CliRunner,
        sample_gsplats_4d: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        bad_value: str,
    ) -> None:
        """Universally invalid axis syntax must fail before reading the store."""
        import importlib

        load_module = importlib.import_module("luxar.gsplats.io.load_gsplats")
        original_load = load_module.load_gsplat_node
        load_calls: list[Path] = []

        def _recording_load(
            path: str | Path, include_stats: bool = False
        ) -> tuple[Any, dict[str, Any]]:
            load_calls.append(Path(path))
            return original_load(path, include_stats=include_stats)

        monkeypatch.setattr(load_module, "load_gsplat_node", _recording_load)

        out = tmp_path / "invalid-syntax.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-x",
                "90",
                "--spatial-dims",
                bad_value,
            ],
        )

        assert result.exit_code != 0
        assert "ASCII digits 0-9" in _plain(result.stdout)
        assert load_calls == [], "invalid syntax must be rejected before dataset load"
        assert not out.exists(), "no output must be written on a failed run"

    @pytest.mark.parametrize(
        ("bad_value", "expected_msg"),
        [
            ("0,1", "exactly 3 axis indices"),
            ("0,1,9", "out of range"),
            ("0,1,1", "must be distinct"),
            ("0,1,x", "valid axis index"),
            ("0,,1,2", "valid axis index"),  # empty tokens are rejected,
            ("0,1,2,", "valid axis index"),  # not silently filtered out
        ],
    )
    def test_transform_spatial_dims_validation_errors(
        self,
        runner: CliRunner,
        sample_gsplats_4d: Path,
        tmp_path: Path,
        bad_value: str,
        expected_msg: str,
    ) -> None:
        """Malformed --spatial-dims exits non-zero, naming the problem."""
        out = tmp_path / "bad.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-x",
                "90",
                "--spatial-dims",
                bad_value,
            ],
        )
        assert result.exit_code != 0
        assert expected_msg in _plain(result.stdout)
        assert not out.exists(), "no output must be written on a failed run"

    def test_transform_spatial_dims_without_rotation_errors(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """--spatial-dims without any --rotate-* is a clear error."""
        out = tmp_path / "noop.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--translate",
                "1,2,3,0",
                "--spatial-dims",
                "0,1,2",
            ],
        )
        assert result.exit_code != 0
        plain = _plain(result.stdout)
        assert "--spatial-dims" in plain
        assert "rotate" in plain
        assert not out.exists(), "no output must be written on a failed run"

    def test_transform_spatial_dims_alone_hits_no_transforms_error(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """--spatial-dims as the ONLY flag keeps the 'No transforms' precedence."""
        out = tmp_path / "noop2.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--spatial-dims",
                "0,1,2",
            ],
        )
        assert result.exit_code != 0
        assert "No transforms specified" in _plain(result.stdout)
        assert not out.exists()

    def test_transform_rotate_4d_default_warns(
        self, runner: CliRunner, sample_gsplats_4d: Path, tmp_path: Path
    ) -> None:
        """Rotating >3D data without --spatial-dims warns about the guess."""
        out = tmp_path / "rotated4d_warn.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(sample_gsplats_4d),
                str(out),
                "--rotate-x",
                "90",
            ],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        plain = _plain(result.stdout)
        assert "rotating dims 0, 1, 2" in plain
        assert "--spatial-dims" in plain

    def test_transform_rotate_4d_all_extent_warns_direct_nd_fit(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The warning sharpens when >3 axes carry real extent (direct nD fit)."""
        from luxar.gsplats.gsplat_data import GSplatData

        n = 4
        identity_chol_4d = np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0], dtype=np.float32),
            (n, 1),
        )
        data = GSplatData(
            centers=np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            amplitudes=np.linspace(0.1, 0.4, n).astype(np.float32),
            cholesky_factors=identity_chol_4d,
        )
        src = tmp_path / "direct4d.gsplats.zarr"
        data.save(src)

        out = tmp_path / "direct4d_rot.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(src), str(out), "--rotate-x", "90"],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        plain = _plain(result.stdout)
        assert "direct nD fit" in plain
        assert "--spatial-dims" in plain

    def test_transform_rotate_3d_default_no_warning(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """No spatial-dims warning on plain 3D data."""
        out = tmp_path / "rotated3d.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(sample_gsplats), str(out), "--rotate-z", "90"],
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        plain = _plain(result.stdout)
        assert "rotating dims 0, 1, 2" not in plain
        assert "--spatial-dims" not in plain

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

    def test_transform_preserves_partition(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Tree-aware transform keeps a kind=partition a partition (PR-4).

        Pre-fix, `transform` did GSplatData.load → ValueError on a partition
        root → exit 1 with no output. Now it walks the tree leaf-by-leaf,
        preserving the part structure and total splat count.
        """
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import (
            GSplatPartition,
            global_amplitude_max,
            total_splats,
        )

        part = tmp_path / "part.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
            ).exit_code
            == 0
        )
        src_node, _ = load_gsplat_node(part)
        assert isinstance(src_node, GSplatPartition)
        n_parts, n_splats = src_node.n_children, total_splats(src_node)

        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "transform",
                str(part),
                str(out),
                "--scale",
                "2,2,2",
                "--center",
                "--normalize-intensity",
                "1.0",
            ],
        )
        assert result.exit_code == 0, f"transform on partition failed: {result.stdout}"

        dst_node, _ = load_gsplat_node(out)
        assert isinstance(dst_node, GSplatPartition)
        assert dst_node.n_children == n_parts
        assert total_splats(dst_node) == n_splats
        # --normalize-intensity 1.0 → the GLOBAL max amplitude is exactly 1.0
        assert global_amplitude_max(dst_node) == pytest.approx(1.0, abs=1e-5)

    def test_transform_partition_rotation_updates_geometry(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """Rotation traverses partition leaves without flattening the tree."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition, center_bounds, total_splats

        part = tmp_path / "rot-part.gsplats.zarr"
        partition_result = runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
        )
        assert partition_result.exit_code == 0, partition_result.stdout

        src_node, _ = load_gsplat_node(part)
        assert isinstance(src_node, GSplatPartition)
        src_lo, src_hi = center_bounds(src_node)

        out = tmp_path / "rotated-part.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "transform", str(part), str(out), "--rotate-z", "90"],
        )
        assert result.exit_code == 0, f"partition rotation failed: {result.stdout}"

        dst_node, _ = load_gsplat_node(out)
        assert isinstance(dst_node, GSplatPartition)
        assert dst_node.n_children == src_node.n_children
        assert total_splats(dst_node) == total_splats(src_node)

        dst_lo, dst_hi = center_bounds(dst_node)
        expected_lo = np.array([-src_hi[1], src_lo[0], src_lo[2]])
        expected_hi = np.array([-src_lo[1], src_hi[0], src_hi[2]])
        np.testing.assert_allclose(dst_lo, expected_lo, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(dst_hi, expected_hi, rtol=1e-4, atol=1e-3)

    def test_transform_partition_center_uses_global_centroid(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """--center on a partition moves the GLOBAL amplitude-weighted centroid
        to the origin (a single global shift, not per-part centering)."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import amplitude_weighted_centroid

        part = tmp_path / "part.gsplats.zarr"
        runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
        )
        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "transform", str(part), str(out), "--center"]
        )
        assert result.exit_code == 0, f"--center on partition failed: {result.stdout}"
        dst_node, _ = load_gsplat_node(out)
        np.testing.assert_allclose(
            amplitude_weighted_centroid(dst_node), 0.0, atol=1e-4
        )

    def test_transform_partition_scales_geometry(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """A scale-only transform on a partition actually scales the geometry
        (its center-bounds extent doubles). Without an assertion tied to the
        scaled centers, a regression dropping the tree-path scale ships green."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition, center_bounds

        part = tmp_path / "part.gsplats.zarr"
        runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
        )
        src_node, _ = load_gsplat_node(part)
        lo, hi = center_bounds(src_node)
        src_extent = hi - lo

        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "transform", str(part), str(out), "--scale", "2,2,2"]
        )
        assert result.exit_code == 0, f"--scale on partition failed: {result.stdout}"
        dst_node, _ = load_gsplat_node(out)
        assert isinstance(dst_node, GSplatPartition)
        lo2, hi2 = center_bounds(dst_node)
        # extent scaled ~2x on every spatial axis
        np.testing.assert_allclose(hi2 - lo2, src_extent * 2.0, rtol=1e-4, atol=1e-4)

    def test_transform_nested_group_rederives_coverage_fraction(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A spatial scale on a multiscale-style tree (a lod group whose finest
        child is a partition carrying its OWN coverage_fraction) must NOT leave the
        stale group-node threshold on disk — the writer re-derives it from the
        transformed subtree splat counts.

        Pre-fix, the tree path scrubbed coverage_fraction only from leaves while
        ``map_leaves`` copied the partition GROUP meta verbatim, so the stale
        value survived (and the partition writer branch re-applied it).
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

        def _leaf(scale: float, seed: int) -> GSplatLeaf:
            rng = np.random.default_rng(seed)
            n = 8
            centers = (rng.uniform(0, 10, size=(n, 3)) * scale).astype(np.float32)
            chol = np.zeros((n, 6), dtype=np.float32)
            chol[:, [0, 2, 5]] = 1.0  # positive diagonal
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        # multiscale-like: lod( coarse_leaf, partition[ leaf, leaf ] ); stamp a
        # deliberately-wrong coverage_fraction on the partition GROUP node.
        # The ladder must be FULLY authored (coarse leaf too): the writer's
        # selector/threshold consistency gate re-derives partially-authored
        # ladders at the first write, which would erase the stale value before
        # the transform ever saw it — the very premise this test needs.
        STALE = 0.5
        fine = GSplatPartition(
            children=[_leaf(1.0, 0), _leaf(1.0, 1)], meta={"coverage_fraction": STALE}
        )
        coarse = _leaf(0.3, 2)
        coarse.meta["coverage_fraction"] = 0.0
        root = GSplatLodGroup(children=[coarse, fine])

        src = tmp_path / "multiscale.gsplats.zarr"
        write_gsplats_tree(src, root)
        # confirm the stale value round-trips on load (the precondition for the bug)
        loaded, _ = load_gsplat_node(src)
        assert loaded.children[1].meta.get("coverage_fraction") == pytest.approx(STALE)

        out = tmp_path / "scaled.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "transform", str(src), str(out), "--scale", "4,4,4"]
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"
        dst, _ = load_gsplat_node(out)
        assert isinstance(dst, GSplatLodGroup)
        # the partition group's coverage_fraction was re-derived, not the stale value
        new_cov = dst.children[1].meta.get("coverage_fraction")
        assert new_cov is not None
        assert new_cov != pytest.approx(STALE), (
            f"stale group-node coverage_fraction survived the scale: {new_cov}"
        )

    def test_transform_tree_preserves_fitting_and_pipeline_metadata(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        """The tree-walking path (partition / nested input) must round-trip the
        fitting/ and pipeline/ provenance groups exactly like the flat path
        does — pre-fix it called write_gsplats_tree without the
        split_fitting_info 4-tuple, silently stripping them."""
        import zarr

        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        # Build a kind=partition store carrying fitting stats + a pipeline/
        # group (the reduction provenance this branch preserves on round-trip).
        part = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "partition", str(sample_gsplats), str(part), "--parts", "2"],
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        node, _ = load_gsplat_node(part)
        src = tmp_path / "src.gsplats.zarr"
        write_gsplats_tree(
            src,
            node,
            fitting_info={"fitter_name": "luxar", "psnr_db": 30.5},
            pipeline_info={"lod_kind": "substitutive", "compression_factor": 4},
        )

        out = tmp_path / "out.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "transform", str(src), str(out), "--scale", "2,2,2"]
        )
        assert result.exit_code == 0, f"transform failed: {result.stdout}"

        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["kind"] == "partition"  # structure preserved
        assert "fitting" in root, "fitting/ group stripped by the tree path"
        assert root["fitting"].attrs["fitter_name"] == "luxar"
        assert root["fitting"].attrs["psnr_db"] == 30.5
        assert "pipeline" in root, "pipeline/ group stripped by the tree path"
        assert root["pipeline"].attrs["lod_kind"] == "substitutive"
        assert root["pipeline"].attrs["compression_factor"] == 4


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


@pytest.fixture
def multiblob_volume(tmp_path: Path) -> Path:
    """A 40^3 volume with many blobs so feature count grows with crop size —
    needed by --fit-exponent (distinct n_features across region scales)."""
    rng = np.random.default_rng(0)
    V = np.zeros((40, 40, 40), np.float32)
    zz, yy, xx = np.mgrid[0:40, 0:40, 0:40]
    for _ in range(24):
        cz, cy, cx = rng.integers(4, 36, 3)
        V += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)).astype(
            np.float32
        )
    V = np.clip(V, 0, 1)
    path = tmp_path / "multiblob.npy"
    np.save(str(path), V)
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

    def test_cal_fit_exponent_writes_fitted_alpha(
        self,
        runner: CliRunner,
        multiblob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """--fit-exponent calibrates K* at several scales and writes a fitted
        alpha into both exponent_fit and splat_density (overriding the 0.44
        default), which the planner / `fit --tiling content` then read."""
        import json

        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(multiblob_volume),
                str(out_json),
                "--n-grid",
                "2",
                "--k-min",
                "50",
                "--k-max",
                "300",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
                "--fit-exponent",
                "--exponent-scales",
                "16,28",
            ],
        )
        assert result.exit_code == 0, f"cal --fit-exponent failed:\n{result.stdout}"
        with open(out_json) as f:
            data = json.load(f)
        ef = data["exponent_fit"]
        assert ef is not None, "exponent_fit not written"
        assert ef["n_points"] == 2
        assert ef["scales"] == [16, 28]
        # the fitted alpha is mirrored into splat_density (what the planner reads)
        assert data["splat_density"]["saturation_exponent"] == pytest.approx(
            ef["alpha"]
        )
        # stdout reports the fitted exponent
        assert "Fitted" in result.stdout

    def test_cal_fit_exponent_degenerate_keeps_default_alpha(
        self,
        runner: CliRunner,
        smooth_blob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """When the scales can't yield ≥2 distinct feature counts (a single-blob
        16³ volume), --fit-exponent must NOT clobber the default α=0.44: it keeps
        the default and reports the failure (the invariant the None branch holds)."""
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
                "--fit-exponent",
                "--exponent-scales",
                "8,12",
            ],
        )
        assert result.exit_code == 0, f"cal --fit-exponent failed:\n{result.stdout}"
        with open(out_json) as f:
            data = json.load(f)
        # the default α survives (either the fit failed → None, or it was degenerate)
        assert data["splat_density"]["saturation_exponent"] == pytest.approx(0.44)
        # and the run told the user it kept the default
        assert "keeping" in result.stdout.lower()

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

    def test_cal_auto_region_marks_k_star_region_scoped(
        self,
        runner: CliRunner,
        multiblob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """``cal --auto-region`` qualifies its K* as region-scoped (#1556).

        ``fit --seeds`` is a WHOLE-VOLUME budget that a tiled fit divides across
        its tiles, so a region-scoped K* handed to it under-seeds the volume by
        roughly the tile count. The headline must therefore say which kind of
        number it is printing, and point at the density-transfer route instead.
        ``--region-size 16`` on the 40^3 fixture forces a real crop (a region as
        large as the volume short-circuits to ``strategy="whole"``, which IS
        whole-volume and must NOT be flagged — see the sibling test).
        """
        import json

        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(multiblob_volume),
                str(out_json),
                "--k-grid",
                "30,90",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
                "--auto-region",
                "--region-size",
                "16",
            ],
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"

        with open(out_json) as f:
            region = json.load(f)["calibration_region"]
        assert region is not None and region["strategy"] != "whole"

        assert "region-scoped" in result.stdout
        assert "--seeds" in result.stdout
        assert "--tiling content" in result.stdout

    def test_cal_default_k_star_is_not_flagged(
        self,
        runner: CliRunner,
        multiblob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """Guard: a default (whole-volume) ``cal`` must NOT print the caveat.

        Its K* is exactly the number ``fit --seeds`` wants, so flagging it would
        steer users away from the documented ``cal`` -> ``fit --seeds`` pipeline.
        """
        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(multiblob_volume),
                str(out_json),
                "--k-grid",
                "30,90",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"
        assert "Recommended K" in result.stdout
        assert "region-scoped" not in result.stdout

    def test_cal_auto_region_whole_volume_shortcircuit_is_not_flagged(
        self,
        runner: CliRunner,
        multiblob_volume: Path,
        fast_fit_config: Path,
        tmp_path: Path,
    ) -> None:
        """``--auto-region`` on a volume smaller than ``--region-size`` is NOT flagged.

        ``select_calibration_region`` short-circuits to ``strategy="whole"`` and
        returns the volume untouched, so K* really is whole-volume even though
        ``calibration_region`` is populated. Flagging on the mere presence of
        that provenance would misdirect every user of a sub-256^3 volume.
        """
        import json

        out_json = tmp_path / "cal.json"
        result = runner.invoke(
            app,
            # fmt: off
            [
                "gsplat",
                "cal",
                str(multiblob_volume),
                str(out_json),
                "--k-grid",
                "30,90",
                "--preset",
                "draft",
                "--config",
                str(fast_fit_config),
                "--device",
                "cpu",
                "--auto-region",  # default --region-size 256 > 40 -> "whole"
            ],
            # fmt: on
        )
        assert result.exit_code == 0, f"cal failed:\n{result.stdout}"

        with open(out_json) as f:
            region = json.load(f)["calibration_region"]
        assert region is not None and region["strategy"] == "whole"
        assert "region-scoped" not in result.stdout


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
    """`luxar gsplat lod --recipe ...` — the unified recipe command."""

    def test_recipe_required(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Omitting --recipe errors cleanly and writes nothing."""
        out = tmp_path / "none.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "lod", str(medium_gsplats), str(out)])
        assert result.exit_code != 0
        assert not out.exists()

    def test_unknown_recipe_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "bad.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "bogus"]
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_irrelevant_option_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A partition option is rejected for the `additive` recipe."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--max-elements",
                "100",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_radial_method_is_accepted_and_reveals_outward(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`-m radial` builds a real ladder whose shells grow outward.

        `radial` was invisible to this command until ``VALID_ADDITIVE_METHODS``
        stopped being a hand-copied tuple, so this pins the CLI wiring — and it
        checks the OUTPUT, not just the exit code: a method that were silently
        ignored would also exit 0.
        """
        from luxar.gsplats.io import load_gsplats

        out = tmp_path / "reveal.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-m",
                "radial",
                "--n-lods",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

        subs = load_gsplats(out).additive_sublods
        assert len(subs) > 1, "expected a real multi-level ladder"
        arrs = [np.asarray(s.centers, dtype=np.float64) for s in subs]
        allc = np.concatenate(arrs)
        centre = (allc.min(axis=0) + allc.max(axis=0)) / 2.0
        # PER-SHELL bounds, not a cumulative maximum. This assertion used to be
        # `cum_max == sorted(cum_max)` over cumulative prefixes, which is true of
        # EVERY ordering by construction — it had no teeth, and the trailing
        # `cum_max[0] < cum_max[-1]` only showed the globally farthest splat was
        # not in the first chunk. Requiring each shell to start no closer than the
        # previous one ended is the property only a radial ordering has; verified
        # by mutation (a "radial silently ignored" mutant passes the old form and
        # fails this one).
        bounds = [
            (
                float(np.linalg.norm(a - centre, axis=1).min()),
                float(np.linalg.norm(a - centre, axis=1).max()),
            )
            for a in arrs
        ]
        for i in range(1, len(bounds)):
            assert bounds[i][0] >= bounds[i - 1][1] - 1e-6, (
                f"shell {i} starts at r={bounds[i][0]:.3f} but shell {i - 1} "
                f"reached r={bounds[i - 1][1]:.3f} — shells are not nested: {bounds}"
            )
        # And the ladder must span a real range, so a degenerate single-radius
        # fixture cannot satisfy the above vacuously.
        assert bounds[-1][1] > bounds[0][1], bounds

    def test_radial_ladder_carries_no_energy_stamps(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A reveal must not be `1/e(k)`-brightened, so it is authored unstamped.

        Paired with a sensitivity control below on the same fixture: without it,
        this would still pass if stamping broke everywhere.
        """
        import zarr

        out = tmp_path / "reveal.gsplats.zarr"
        result = runner.invoke(
            app,
            # fmt: off
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-m",
                "radial",
                "--n-lods",
                "4",
            ],
            # fmt: on
        )
        assert result.exit_code == 0, result.output

        root = zarr.open(str(out), mode="r")
        n = int(root.attrs["n_additive_sublods"])
        for i in range(n):
            stats = dict(root[f"additive_{i}"].attrs.get("lod_stats", {}))
            assert "energy_fraction_cum" not in stats, f"additive_{i} was stamped"
            # Provenance still recorded — only the brightness keys are dropped.
            assert stats.get("lod_method") == "radial"
        assert "reference_energy" not in dict(root.attrs.get("level_stats", {}))

    def test_non_reveal_ladder_is_stamped_control(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """SENSITIVITY CONTROL for the test above — same fixture, same ladder
        shape, an energy-ordered method, which MUST carry the stamps."""
        import zarr

        out = tmp_path / "energy.gsplats.zarr"
        result = runner.invoke(
            app,
            # fmt: off
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-m",
                "self_energy",
                "--n-lods",
                "4",
            ],
            # fmt: on
        )
        assert result.exit_code == 0, result.output

        root = zarr.open(str(out), mode="r")
        n = int(root.attrs["n_additive_sublods"])
        for i in range(n):
            stats = dict(root[f"additive_{i}"].attrs.get("lod_stats", {}))
            assert "energy_fraction_cum" in stats, f"additive_{i} is unstamped"
        assert "reference_energy" in dict(root.attrs["level_stats"])

    def test_reveal_centre_relocates_the_first_shell(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`--reveal-centre` must change the OUTPUT, not merely be accepted.

        Pinned to a corner, the first shell must sit closer to that corner than
        the default (bbox-centred) ladder's first shell does.
        """
        from luxar.gsplats.io import load_gsplats

        corner = np.array([0.0, 0.0, 0.0])

        def first_shell_dist_to_corner(path: Path) -> float:
            arrs = [
                np.asarray(s.centers, dtype=np.float64)
                for s in load_gsplats(path).additive_sublods
            ]
            return float(np.linalg.norm(arrs[0] - corner, axis=1).mean())

        default_out = tmp_path / "default.gsplats.zarr"
        pinned_out = tmp_path / "pinned.gsplats.zarr"
        base = ["gsplat", "lod", str(medium_gsplats)]
        tail = ["--recipe", "stream", "-m", "radial", "--n-lods", "4"]

        assert runner.invoke(app, base + [str(default_out)] + tail).exit_code == 0
        pinned = runner.invoke(
            app, base + [str(pinned_out)] + tail + ["--reveal-centre", "0,0,0"]
        )
        assert pinned.exit_code == 0, pinned.output

        assert first_shell_dist_to_corner(pinned_out) < first_shell_dist_to_corner(
            default_out
        )

    def test_reveal_knobs_rejected_without_radial(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Reveal knobs under a non-reveal ordering are an error, not a no-op.

        Silently ignoring them would hand back an energy-ordered ladder while the
        user believed they had asked for a repositioned reveal.
        """
        for flag, value in (("--reveal-centre", "0,0,0"), ("--spatial-dims", "0,1")):
            out = tmp_path / f"x{flag}.gsplats.zarr"
            result = runner.invoke(
                app,
                # fmt: off
                [
                    "gsplat",
                    "lod",
                    str(medium_gsplats),
                    str(out),
                    "--recipe",
                    "stream",
                    flag,
                    value,
                ],
                # fmt: on
            )
            assert result.exit_code != 0, f"{flag} was silently accepted"
            assert not out.exists()

    def test_spatial_dims_out_of_range_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            # fmt: off
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-m",
                "radial",
                "--spatial-dims",
                "0,1,5",
            ],
            # fmt: on
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_reveal_centre_length_must_match_spatial_dims(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The centre carries one coordinate per measured axis."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            # fmt: off
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-m",
                "radial",
                "--spatial-dims",
                "0,1",
                "--reveal-centre",
                "1,2,3",
            ],
            # fmt: on
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_spatial_dims_preserves_the_listed_order(self) -> None:
        """`--spatial-dims` must NOT sort — the order pairs with `--reveal-centre`.

        It went through `sorted(set(...))`, so `--spatial-dims 2,0 --reveal-centre
        10,20` silently meant "axis 0 centred at 10, axis 2 at 20" rather than the
        pairing the user typed. Deliberately unlike `--coarsen-dims`, where a
        barrier SET is order-free.
        """
        from luxar.cli.reveal_options import parse_reveal_spatial_dims

        assert parse_reveal_spatial_dims("2,0", 3) == [2, 0]
        assert parse_reveal_spatial_dims("0,2", 3) == [0, 2]

    @pytest.mark.parametrize(
        "spec", ["nan,0,0", "inf,0,0", "0,-inf,0"], ids=["nan", "inf", "-inf"]
    )
    def test_reveal_centre_rejects_non_finite_coordinates(self, spec: str) -> None:
        """`float("nan")` parses happily, so this needed an explicit check.

        With a non-finite centre every distance is non-finite; they all compare
        equal under the stable argsort, so the ladder comes out in input order and
        the user gets no reveal and no error.
        """
        from luxar.cli.reveal_options import parse_reveal_centre

        with pytest.raises(typer.BadParameter, match="finite"):
            parse_reveal_centre(spec)

    def test_spatial_dims_rejects_duplicates(self) -> None:
        """A duplicate was silently collapsed by `set()`; it now errors, because a
        repeat would count that axis twice in the distance."""
        from luxar.cli.reveal_options import parse_reveal_spatial_dims

        with pytest.raises(typer.BadParameter, match="must not repeat"):
            parse_reveal_spatial_dims("0,0", 3)

    def test_coarsen_dims_rejected_for_additive(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--coarsen-dims is a substitutive-only knob; additive must reject it."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--coarsen-dims",
                "0,1",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_coarsen_dims_out_of_range_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """An index >= ndim (3D data) is a clean BadParameter, writes nothing."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--coarsen-dims",
                "0,1,5",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_coarsen_dims_non_integer_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--coarsen-dims",
                "a,b",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_coarsen_dims_substitutive_runs(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Valid indices on a substitutive build succeed (barrier = dim 2)."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "sub.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--coarsen-dims",
                "0,1",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        assert GSplatData.load(out).n_substitutive >= 2

    def test_quality_stamps_rejected_for_stream(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--quality-stamps is a substitutive-only knob; stream rejects it."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--quality-stamps",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_quality_stamps_persist_through_levels(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Default-on Q·e stamps round-trip to disk: every substitutive level
        carries quality + reference_energy and every sub-LOD carries a
        monotone energy_fraction_cum ending at 1.0; --no-quality-stamps drops
        the Q keys (the free e(k)/w stamps remain)."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "q.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "levels"],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        levels = loaded.substitutive_levels
        assert levels[0].stats["quality"] == 1.0  # the finest IS the reference
        for lev in levels:
            assert 0.0 <= lev.stats["quality"] <= 1.0
            assert lev.stats["reference_energy"] > 0
            e = [sub.stats["energy_fraction_cum"] for sub in lev.additive_sublods]
            assert e == sorted(e)
            assert e[-1] == pytest.approx(1.0)

        out_off = tmp_path / "noq.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out_off),
                "--recipe",
                "levels",
                "--no-quality-stamps",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        for lev in GSplatData.load(out_off).substitutive_levels:
            assert "quality" not in lev.stats

    def test_legacy_recipe_names_error_with_pointer(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Pre-rename spellings are rejected with a pointer to the new name
        (no silent aliasing on the CLI; stored manifests translate silently)."""
        out = tmp_path / "x.gsplats.zarr"
        for legacy, current in (
            ("substitutive", "levels"),
            ("pyramid", "levels"),
            ("additive", "stream"),
            ("partitioned", "tiles"),
            ("multiscale", "overview"),
            ("mosaic", "adaptive"),
        ):
            result = runner.invoke(
                app,
                ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", legacy],
            )
            assert result.exit_code != 0, legacy
            assert current in normalized_cli_output(result), (
                legacy,
                normalized_cli_output(result),
            )
            assert not out.exists()

    def test_no_additive_rejected_for_additive_recipe(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--no-additive contradicts recipes whose ladder is definitional."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--no-additive",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_recipe_provenance_records_recipe_not_as_lod_kind(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A standalone lod build records the RECIPE that built it (build
        provenance) WITHOUT overwriting the mechanism ``lod_kind``. A recipe
        name must never masquerade as lod_kind (viewer/migrator read lod_kind
        as a mechanism)."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "prov.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-L",
                "1",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        stats = GSplatData.load(out, include_stats=True).stats
        assert stats.get("recipe") == "levels"
        assert stats.get("lod_kind") == "substitutive"  # mechanism, not "levels"

    def test_substitutive_default_has_additive_ladders(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Additive LODs by default: a plain substitutive build ladders every
        level; --no-additive restores bare leaves."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "lad.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-L",
                "1",
                "--n-lods",
                "2",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        assert all(lev.n_additive_lods == 2 for lev in loaded.substitutive_levels)
        out2 = tmp_path / "bare.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out2),
                "--recipe",
                "levels",
                "-L",
                "1",
                "--no-additive",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        loaded2 = GSplatData.load(out2)
        assert all(lev.n_additive_lods == 1 for lev in loaded2.substitutive_levels)

    def test_refine_rejected_for_additive_recipe(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--refine is a substitutive-only knob; additive must reject it."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--refine",
                "l2",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_refine_iters_requires_refine_l2(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--refine-iters",
                "50",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_refine_invalid_value_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--refine",
                "banana",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_recipe_substitutive_with_refine_smoke(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--refine l2 end-to-end: output loads and levels are reduced."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "refined.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-L",
                "1",
                "--refine",
                "l2",
                "--refine-iters",
                "4",
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0, f"refine smoke failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        assert loaded.n_substitutive == 2
        assert loaded.at_substitutive(1).n_splats < loaded.at_substitutive(0).n_splats

    def test_refine_volume_requires_target(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--refine",
                "volume",
            ],
        )
        assert result.exit_code != 0
        assert "--target" in normalized_cli_output(result)
        assert not out.exists()

    def test_channel_without_target_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A sub-volume selector without --target is a silent no-op unless
        guarded — the command must reject it, not build a levels LOD that
        ignored --channel."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--channel",
                "1",
            ],
        )
        assert result.exit_code != 0
        assert "--target" in normalized_cli_output(result)
        assert not out.exists()

    def test_target_requires_refine_volume(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--target",
                str(small_volume_npy),
            ],
        )
        assert result.exit_code != 0
        assert "--refine volume" in normalized_cli_output(result)
        assert not out.exists()

    def test_refine_volume_rejected_for_stream_recipe(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        """--target/--refine are substitutive knobs; the token machinery must
        reject them for the stream recipe."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--refine",
                "volume",
                "--target",
                str(small_volume_npy),
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_refine_volume_accepted_for_adaptive_recipe(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        """Per-part levels re-fit against each tile's own CROP of the volume, so
        the adaptive recipe now accepts ``--refine volume``. It used to be
        rejected because a part fitted against the FULL volume is pulled out of
        its tile to explain a neighbour's signal."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "adaptive",
                "--refine",
                "volume",
                "--refine-iters",
                "3",
                "--target",
                str(small_volume_npy),
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        assert out.exists()

    def test_target_axes_requires_a_target(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """``--target-axes`` describes a ``--target``; alone it is a typo, not a
        silently ignored option."""
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(tmp_path / "x.gsplats.zarr"),
                "--recipe",
                "levels",
                "--target-axes",
                "t,z,y,x",
            ],
        )
        assert result.exit_code != 0
        assert "--target-axes" in normalized_cli_output(result)

    def test_target_axes_excludes_the_slicing_selectors(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        small_volume_npy: Path,
        tmp_path: Path,
    ) -> None:
        """``--target-axes`` KEEPS the stacked axis; ``--timepoint`` drops it.

        Combined, the labels no longer describe the array the re-fit is handed —
        and since the selectors are applied by the positional heuristic, the
        mismatch would be silent rather than loud.
        """
        for flag, value in (("--timepoint", "0"), ("--channel", "0")):
            result = runner.invoke(
                app,
                [
                    "gsplat",
                    "lod",
                    str(medium_gsplats),
                    str(tmp_path / "x.gsplats.zarr"),
                    "--recipe",
                    "levels",
                    "--refine",
                    "volume",
                    "--target",
                    str(small_volume_npy),
                    "--target-axes",
                    "z,y,x",
                    flag,
                    value,
                ],
            )
            assert result.exit_code != 0, flag
            assert "--target-axes" in normalized_cli_output(result)

    def test_target_axes_opens_the_target_lazily(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """A ``--target-axes`` target must not be materialised by the front door.

        The re-fit only ever SLICES the target (one barrier group / one tile crop
        at a time), which is the whole reason a stacked timelapse is workable — one
        timepoint of a 253-timepoint 407x2048x2048 uint16 stack is 3.4 GB against
        431 GB whole. Reading it eagerly at the CLI boundary throws that away, so
        assert the eager loader is not on this path at all.
        """
        import zarr

        import luxar.cli.gsplat_config as gsplat_config

        volume = np.random.rand(16, 16, 16).astype(np.float32) * 0.5
        volume[6:10, 6:10, 6:10] = 1.0
        target = tmp_path / "target.zarr"
        z = zarr.open(str(target), mode="w", shape=volume.shape, dtype=volume.dtype)
        z[:] = volume

        def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the target was loaded eagerly")

        monkeypatch.setattr(gsplat_config, "load_volume", _boom)

        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "--levels",
                "1",
                "--refine",
                "volume",
                "--refine-iters",
                "3",
                "--target",
                str(target),
                "--target-axes",
                "z,y,x",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        assert out.exists()

    def test_recipe_levels_with_refine_volume_smoke(
        self, runner: CliRunner, small_volume_npy: Path, tmp_path: Path
    ) -> None:
        """--refine volume --target end-to-end: fit the volume, build levels
        with a volume re-fit, and confirm the provenance round-trips. The
        never-worse guard makes the outcome deterministic (seed or better)."""
        import numpy as np

        from luxar.gsplats.fit_gsplats import fit_gaussian_splats
        from luxar.gsplats.gsplat_data import GSplatData

        volume = np.load(small_volume_npy)
        fine = fit_gaussian_splats(
            volume, seeds=40, n_iters=80, device="cpu", verbose=False
        )
        src = tmp_path / "fit.gsplats.zarr"
        fine.save(src)
        out = tmp_path / "vr.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(src),
                str(out),
                "--recipe",
                "levels",
                "-L",
                "1",
                "--refine",
                "volume",
                "--refine-iters",
                "10",
                "--target",
                str(small_volume_npy),
            ],
        )
        assert result.exit_code == 0, f"volume-refit smoke failed:\n{result.stdout}"
        loaded = GSplatData.load(out, include_stats=True)
        assert loaded.n_substitutive == 2
        assert loaded.stats["refine"] == "volume"
        assert loaded.stats["refine_iters"] == 10
        lev = loaded.substitutive_levels[1]
        assert lev.stats["refine"] == "volume"
        assert {"mse_seed", "mse_refit", "improved"} <= set(lev.stats["refine_stats"])

    def test_recipe_flat(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "flat"]
        )
        assert result.exit_code == 0, f"flat failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        assert loaded.n_substitutive == 1
        assert loaded.n_additive_sublods == 1
        assert loaded.n_splats == 32

    def test_recipe_additive(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "ladder.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--n-lods",
                "2",
                "--add-method",
                "self_energy",
            ],
        )
        assert result.exit_code == 0, f"additive failed:\n{result.stdout}"
        ladder = GSplatData.load(out)
        assert ladder.n_additive_sublods == 2
        per_level = [
            ladder.additive_sublod(i).n_splats for i in range(ladder.n_additive_sublods)
        ]
        assert sum(per_level) == ladder.n_splats == 32

    def test_recipe_substitutive(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats import GSplatData

        out = tmp_path / "hierarchy.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-K",
                "2",
                "-L",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"substitutive failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        assert loaded.n_substitutive == 3  # L + 1
        assert [lev.compression_factor for lev in loaded.substitutive_levels] == [
            1,
            2,
            4,
        ]

    def test_recipe_substitutive_stamps_coverage_fractions(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """#5: `gsplat lod --recipe levels` stamps SCREEN-AREA
        ``coverage_fraction`` (occupancy halving; group ``selector`` =
        ``"screen-area"``) on the on-disk lod children: coarsest = 0.0, finest
        = the half-screen anchor (0.5), one area-halving per level, strictly
        ascending — independent of per-level counts."""
        import zarr

        from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR

        out = tmp_path / "sub.gsplats.zarr"
        r = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-K",
                "2",
                "-L",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert r.exit_code == 0, f"substitutive failed:\n{r.stdout}"
        g = zarr.open_group(str(out), mode="r")
        ch = sorted(
            (k for k in g.group_keys() if k.startswith("child_")),
            key=lambda s: int(s.split("_")[1]),
        )
        cov = [float(g[k].attrs["coverage_fraction"]) for k in ch]
        assert g.attrs["selector"] == "screen-area"
        assert cov[0] == 0.0
        assert cov[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
        # coverage_i halves per level below the finest (count-independent).
        for i in range(1, len(cov)):
            assert cov[i] == pytest.approx(
                WHOLE_OBJECT_FINEST_ANCHOR / 2 ** (len(cov) - 1 - i)
            )
        assert all(cov[i] > cov[i - 1] for i in range(1, len(cov)))

    def test_recipe_pyramid(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats import GSplatData

        out = tmp_path / "pyramid.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-K",
                "2",
                "-L",
                "2",
                "--n-lods",
                "2",
                "--add-method",
                "self_energy",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"pyramid failed:\n{result.stdout}"
        loaded = GSplatData.load(out)
        assert loaded.n_substitutive == 3
        for lev in loaded.substitutive_levels:
            assert 1 <= lev.n_additive_lods <= 2

    def test_recipe_partitioned(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`partitioned` writes a kind=partition tree of per-part additive ladders."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import (
            GSplatLeaf,
            GSplatPartition,
            iter_leaves,
            total_splats,
        )

        out = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "tiles",
                "--max-elements",
                "12",
                "--n-lods",
                "2",
            ],
        )
        assert result.exit_code == 0, f"partitioned failed:\n{result.stdout}"
        node, _ = load_gsplat_node(out)
        assert isinstance(node, GSplatPartition)
        assert node.n_children >= 2
        for leaf in iter_leaves(node):
            assert isinstance(leaf, GSplatLeaf)
        assert total_splats(node) == 32

    def test_recipe_multiscale(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`multiscale` writes an unbalanced lod(coarse leaf + partition fine)."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import (
            GSplatLeaf,
            GSplatLodGroup,
            GSplatPartition,
            total_splats,
        )

        out = tmp_path / "ms.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "overview",
                "--max-elements",
                "12",
                "--n-lods",
                "2",
                "-K",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"multiscale failed:\n{result.stdout}"
        node, _ = load_gsplat_node(out)
        assert isinstance(node, GSplatLodGroup)
        assert node.n_children == 2
        coarse, fine = node.children  # coarsest→finest in memory
        assert isinstance(fine, GSplatPartition)
        assert isinstance(coarse, GSplatLeaf)
        assert total_splats(fine) == 32
        assert coarse.n_splats < 32

    def test_recipe_mosaic(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`mosaic` writes a kind=partition whose every part is its own
        substitutive lod group (per-part coarse↔fine swap)."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition, total_splats

        out = tmp_path / "mosaic.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "adaptive",
                "--max-elements",
                "12",
                "-K",
                "2",
                "--levels",
                "1",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"mosaic failed:\n{result.stdout}"
        node, _ = load_gsplat_node(out)
        assert isinstance(node, GSplatPartition)
        assert node.n_children >= 2
        # every part is its own substitutive lod group
        assert all(isinstance(p, GSplatLodGroup) for p in node.children)
        # conservation at the finest level (parts tile the original 32 splats);
        # per-part lod children are coarsest→finest, so the finest is the last.
        finest_total = sum(total_splats(p.children[-1]) for p in node.children)
        assert finest_total == 32
        assert total_splats(node) > 32  # synthesized coarse levels add storage

    def test_recipe_mosaic_parts_drives_partition(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--parts must drive mosaic's per-part cap too (regression: the CLI's
        partition-recipe branch listed only partitioned/multiscale, so --parts was
        silently ignored for mosaic and it collapsed to a single default-capped
        part)."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition

        out = tmp_path / "mp.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "adaptive",
                "--parts",
                "4",
                "-K",
                "2",
                "--levels",
                "1",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"mosaic --parts failed:\n{result.stdout}"
        node, _ = load_gsplat_node(out)
        assert isinstance(node, GSplatPartition)
        # 32 splats / 4 parts -> cap 8 -> 4 BSP parts (pre-fix: --parts ignored -> 1).
        assert node.n_children == 4

    def test_recipe_mosaic_accepts_additive_option(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Additive LODs by default: mosaic's per-part substitutive levels are
        laddered, so additive knobs (--n-lods) are legal for it now."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "adaptive",
                "--n-lods",
                "2",
                "--max-elements",
                "16",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        assert out.exists()

    def _multiscale_fine_threshold(
        self, runner: CliRunner, src: Path, out: Path, *flags: str
    ) -> float:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(src),
                str(out),
                "--recipe",
                "overview",
                "--max-elements",
                "12",
                "--n-lods",
                "2",
                "-K",
                "2",
                "--device",
                "cpu",
                *flags,
            ],
        )
        assert result.exit_code == 0, f"multiscale failed:\n{result.stdout}"
        import zarr

        g = zarr.open_group(str(out), mode="r")
        assert g["child_0"].attrs.get("coverage_fraction") == 0.0  # coarsest cap
        return float(g["child_1"].attrs.get("coverage_fraction"))

    def test_recipe_multiscale_stamps_coverage_fractions(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """multiscale stamps screen-area ``coverage_fraction`` on the coarse
        cap (0.0, always-eligible) and the fine partition wrapper (the finest rung)
        — the on-disk attrs the viewer's selector reads.

        The finest rung is ``PARTITION_FINEST_AREA`` (screen-area 1.0 —
        fills-screen), not the whole-object 0.5: overview's fine child is the
        whole dataset as a ``kind=partition`` and is by contract a zoom-in
        branch, so the pair keeps the fills-screen anchor rather than the
        whole-object half-screen one (``partitioned_coverage_fractions``)."""
        from luxar.core.group.lod.group import PARTITION_FINEST_AREA

        out = tmp_path / "ms.gsplats.zarr"
        fine_cov = self._multiscale_fine_threshold(runner, medium_gsplats, out)
        # partitioned_coverage_fractions([N_coarse, N_fine]) = [0.0, 1.0].
        assert fine_cov == pytest.approx(PARTITION_FINEST_AREA)

    def test_quiet_suppresses_saved_line(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "quiet.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--n-lods",
                "2",
                "--add-method",
                "self_energy",
                "--quiet",
            ],
        )
        assert result.exit_code == 0, f"quiet failed:\n{result.stdout}"
        assert f"Saved to {out}" not in result.stdout

    def test_overwrite_required_for_existing_output(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "exists.gsplats.zarr"
        out.mkdir()
        result = runner.invoke(
            app,
            ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "flat"],
        )
        assert result.exit_code != 0
        ok = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "flat",
                "--overwrite",
            ],
        )
        assert ok.exit_code == 0, f"overwrite failed:\n{ok.stdout}"

    @staticmethod
    def _make_2d_gsplats(path: Path) -> Path:
        """A 2D (ndim=2) fitted .gsplats.zarr — BSP partitioning needs >=2 dims."""
        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData

        n = 30
        rng = np.random.default_rng(0)
        chol = np.zeros((n, 3), dtype=np.float32)
        chol[:, [0, 2]] = 1.0  # packed lower-tri diagonal for 2D
        GSplatData(
            centers=rng.uniform(0, 50, (n, 2)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
            cholesky_factors=chol,
        ).save(path)
        return path

    @pytest.mark.parametrize("recipe", ["tiles", "overview", "adaptive"])
    def test_2d_input_partition_recipes_accepted(
        self, runner: CliRunner, tmp_path: Path, recipe: str
    ) -> None:
        """2D input to a partition-based recipe is accepted (BSP needs >=2 dims)
        and writes a real partition-bearing tree."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import (
            GSplatLodGroup,
            GSplatPartition,
            total_splats,
        )

        src = self._make_2d_gsplats(tmp_path / "in2d.gsplats.zarr")
        out = tmp_path / "out2d.gsplats.zarr"
        args = [
            "gsplat",
            "lod",
            str(src),
            str(out),
            "--recipe",
            recipe,
            "--max-elements",
            "12",
        ]
        if recipe in ("overview", "adaptive"):
            args += ["-K", "2", "--device", "cpu"]
        if recipe == "adaptive":
            args += ["--levels", "1"]

        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{recipe} on 2D failed:\n{result.stdout}"

        node, _ = load_gsplat_node(out)
        if recipe == "tiles":
            # a partition at the root tiling all 30 splats
            assert isinstance(node, GSplatPartition)
            assert node.n_children >= 2
            assert total_splats(node) == 30
        elif recipe == "overview":
            # unbalanced lod(coarse leaf + partition fine)
            assert isinstance(node, GSplatLodGroup)
            _, fine = node.children  # coarsest→finest in memory
            assert isinstance(fine, GSplatPartition)
            assert total_splats(fine) == 30
        else:  # adaptive
            # a partition whose every part is its own substitutive lod group
            assert isinstance(node, GSplatPartition)
            assert node.n_children >= 2
            assert all(isinstance(p, GSplatLodGroup) for p in node.children)
            finest_total = sum(total_splats(p.children[-1]) for p in node.children)
            assert finest_total == 30

    @pytest.mark.parametrize("recipe", ["tiles", "overview", "adaptive"])
    def test_1d_input_partition_recipes_clean_error(
        self, runner: CliRunner, tmp_path: Path, recipe: str
    ) -> None:
        """1D input to a partition-based recipe is rejected cleanly (BSP needs
        >=2 dims) with a typer BadParameter, not a raw traceback."""
        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData

        n = 30
        rng = np.random.default_rng(0)
        src = tmp_path / "in1d.gsplats.zarr"
        GSplatData(
            centers=rng.uniform(0, 50, (n, 1)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
            cholesky_factors=np.ones((n, 1), dtype=np.float32),
        ).save(src)
        out = tmp_path / "out1d.gsplats.zarr"

        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(src),
                str(out),
                "--recipe",
                recipe,
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()
        io = normalized_cli_output(result)
        assert "Traceback" not in io
        assert "spatial dimensions" in io
        assert "Use --recipe stream or levels" in io

    def test_2d_input_matrix_recipe_still_works(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The matrix recipes have no >=3D requirement — 2D additive succeeds."""
        src = self._make_2d_gsplats(tmp_path / "in2d.gsplats.zarr")
        out = tmp_path / "out2d.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(src),
                str(out),
                "--recipe",
                "stream",
                "--n-lods",
                "2",
            ],
        )
        assert result.exit_code == 0, f"2D additive failed:\n{result.stdout}"

    def test_invalid_ordering_clean_error(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A bad --ordering errors cleanly up front, not via a deep traceback."""
        out = tmp_path / "o.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "flat",
                "--ordering",
                "bogus",
            ],
        )
        assert result.exit_code != 0
        io = normalized_cli_output(result)
        assert "Traceback" not in io
        assert "ordering" in io.lower()

    def test_substitutive_method_short_flag_hint(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """`--recipe levels -m kmeans` (a substitutive algorithm name fed
        to the ADDITIVE --add-method) still points the user to --subst-method
        via the method-value validation."""
        out = tmp_path / "o.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-m",
                "kmeans",
            ],
        )
        assert result.exit_code != 0
        assert "--subst-method" in normalized_cli_output(result)

    def test_count_growing_recipe_fitting_count_matches_leaves(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """`fitting/n_splats` must equal the file's true leaf total, not the
        (smaller) source-fit count — multiscale synthesises an extra coarse cap.
        """
        import numpy as np
        import zarr

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import total_splats

        n = 600
        rng = np.random.default_rng(0)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, (n, 3))
        src = tmp_path / "fitted.gsplats.zarr"
        GSplatData(
            centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
            cholesky_factors=chol,
            stats={"n_splats": n, "psnr_db": 31.5},  # source-fit provenance
        ).save(src)

        out = tmp_path / "ms.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(src),
                str(out),
                "--recipe",
                "overview",
                "--max-elements",
                "150",
                "-K",
                "4",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"multiscale failed:\n{result.stdout}"
        node, _ = load_gsplat_node(out)
        leaf_total = total_splats(node)
        assert leaf_total > n, "multiscale should grow the splat count (coarse cap)"
        root = zarr.open_group(str(out), mode="r")
        persisted = root["fitting"].attrs.get("n_splats")
        assert persisted == leaf_total, (
            f"stale fitting/n_splats {persisted} != true leaf total {leaf_total}"
        )

    @pytest.mark.parametrize("breakpoints", ["energy:1.5", "counts:999999"])
    def test_breakpoints_out_of_range_clean_error(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path, breakpoints: str
    ) -> None:
        """Out-of-range --breakpoints error cleanly, not via a raw traceback."""
        out = tmp_path / "o.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--breakpoints",
                breakpoints,
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()
        assert "Traceback" not in normalized_cli_output(result)

    def test_multiscale_default_compression_factor_is_n_aware(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """multiscale's default K scales with N (was a fixed 4 → a 2.9M cap on
        large fits). Patch the cap target so K = round(N/target) is unambiguous
        and distinct from the old fixed default, then assert it's logged + used.
        """
        import luxar.cli.lod as lod_mod

        # 32 splats / target 4 → K = 8 (the old fixed default was 4).
        monkeypatch.setattr(lod_mod, "_MULTISCALE_CAP_TARGET", 4)
        out = tmp_path / "ms.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "overview",
            ],
        )
        assert result.exit_code == 0, (
            f"multiscale failed: {normalized_cli_output(result)}"
        )
        io = normalized_cli_output(result)
        assert "compression-factor defaulting to 8" in io, io
        assert out.exists()

    def test_multiscale_explicit_compression_factor_wins(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """An explicit -K is never overridden by the N-aware default."""
        out = tmp_path / "msk.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "overview",
                "-K",
                "3",
            ],
        )
        assert result.exit_code == 0, (
            f"multiscale -K failed: {normalized_cli_output(result)}"
        )
        # The auto-default message must NOT fire when -K is given.
        assert "compression-factor defaulting" not in normalized_cli_output(result)
        # Strong check: the explicit K=3 must actually SHAPE the coarse cap, not
        # merely suppress the default message. medium_gsplats has N=32, so the
        # single-level cap holds ceil(32/3)=11 representatives (vs ceil(32/2)=16
        # under the default K=2). A regression that parses but drops the explicit
        # K would leave 16 here and pass the message check above — this catches it.
        import math

        import zarr

        from luxar.gsplats.tree import GSplatLodGroup
        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        root = zarr.open_group(str(out), mode="r")
        node = read_gsplat_node(root, root)
        assert isinstance(node, GSplatLodGroup)  # multiscale → kind=lod root
        coarse_cap = node.children[0]  # coarsest→finest in memory
        assert coarse_cap.n_splats == math.ceil(32 / 3) == 11, (
            f"coarse cap has {coarse_cap.n_splats} splats; expected ceil(32/3)=11 "
            f"(K=3 not applied — got the default-K=2 value 16?)"
        )

    def test_multiscale_levels_rejected_with_helpful_message(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--levels has no meaning for multiscale (single-level cap); the error
        must steer the user to -K rather than just 'not used by'."""
        out = tmp_path / "msl.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "overview",
                "--levels",
                "3",
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()
        assert "single-level" in normalized_cli_output(result)

    def test_lod_stream_breakpoints_grammar(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """'stream:C' is a valid --breakpoints form; each leaf gets a geometric
        ladder sized against its own N."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "stream.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-b",
                "stream:10",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        loaded = GSplatData.load(out, include_stats=True)
        incs = [s.n_splats for s in loaded.additive_sublods]
        assert incs[0] == 10 and sum(incs) == 32  # medium_gsplats N=32
        assert loaded.additive_sublods[0].stats["lod_breakpoints_kind"] == "stream"

    @pytest.mark.parametrize("bad", ["stream:", "stream:0", "stream:abc"])
    def test_lod_stream_breakpoints_invalid(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path, bad: str
    ) -> None:
        out = tmp_path / "bad.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "-b",
                bad,
            ],
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_lod_target_ms_derives_stream_breakpoints(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--target-ms logs the bytes/splat derivation and builds a ladder."""
        out = tmp_path / "tms.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--target-ms",
                "200",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        io = normalized_cli_output(result)
        assert "--target-ms 200" in io and "stream:" in io
        assert "measured from input store" in io  # input is a real store
        assert out.exists()

    def test_lod_target_ms_encoding_change_uses_analytic(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """An explicit --encoding that differs from the input's stored encoding
        re-encodes the output, so --target-ms must size against the analytic
        estimate for the TARGET encoding — not the measured input bytes (e.g.
        u16 input + --encoding precision would be ~2x off budget)."""
        out = tmp_path / "tms_prec.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--target-ms",
                "200",
                "--encoding",
                "precision",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        io = normalized_cli_output(result)
        assert "re-encodes the output" in io
        assert "analytic estimate" in io
        assert "measured from input store" not in io
        # 3D precision analytic: 1.5·4·(3+1+6) = 60 B → 625000/60 = 10417.
        assert "stream:10417" in io
        assert out.exists()

    def test_detect_store_encoding_classifies_modes(self, tmp_path: Path) -> None:
        """detect_store_encoding reads the on-disk encoding attrs of the split
        Cholesky arrays: AUTO quantizes to u8 WITH a covariance certificate
        (u16 when escalated / legacy), MEMORY to u8 without one, PRECISION
        stores float32. Needs NON-uniform cholesky (uniform stores broadcast
        them)."""
        from luxar.cli.gsplat_ops.recipe_shared import detect_store_encoding
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(1)
        n = 16
        data = GSplatData(
            centers=(rng.random((n, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            # Varied, positive-diagonal factors → really encoded, not broadcast.
            cholesky_factors=(rng.random((n, 6)) * 0.5 + 0.5).astype(np.float32),
        )
        for mode, expected in (
            (EncodingMode.AUTO, "auto"),
            (EncodingMode.PRECISION, "precision"),
            (EncodingMode.MEMORY, "memory"),
        ):
            out = tmp_path / f"{expected}.gsplats.zarr"
            data.save(out, encoding_mode=mode)
            assert detect_store_encoding(out) == expected, mode
        assert detect_store_encoding(tmp_path / "nope.gsplats.zarr") is None

    @pytest.mark.parametrize("fmt", [2, 3])
    def test_detect_store_encoding_reads_both_on_disk_formats(
        self, tmp_path: Path, fmt: int
    ) -> None:
        """The classifier must read a store written in EITHER on-disk format.

        Its first implementation globbed for the format-2 ``.zattrs`` document by
        name, which matches nothing in a format-3 store — and the miss was
        invisible, because "found no encoding attrs" is spelled ``None``, the
        same perfectly ordinary value a zip archive returns. Callers read that as
        "unclassifiable, don't assume a mode" and carried on. Pinning BOTH
        formats is what makes the failure loud; the default-format test above
        only ever exercises whichever format is current.
        """
        import luxar._zarr_compat as zc
        from luxar.cli.gsplat_ops.recipe_shared import detect_store_encoding
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(1)
        n = 16
        data = GSplatData(
            centers=(rng.random((n, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            cholesky_factors=(rng.random((n, 6)) * 0.5 + 0.5).astype(np.float32),
        )
        original = zc.ZARR_FORMAT
        zc.set_zarr_format(fmt)
        try:
            out = tmp_path / "store.gsplats.zarr"
            data.save(out, encoding_mode=EncodingMode.MEMORY)
            # The store really is in the format under test — otherwise this
            # would pass for the wrong reason.
            assert (out / "zarr.json").exists() == (fmt == 3)
            assert detect_store_encoding(out) == "memory"
        finally:
            zc.set_zarr_format(original)

    def test_detect_store_encoding_escalated_legacy_and_certified_f32(
        self, tmp_path: Path
    ) -> None:
        """The certificate-based branches: an ESCALATED AUTO store (u16 with a
        certificate) and a LEGACY pre-certificate AUTO store (bare u16) both
        classify as "auto"; a certified-float32 store (the f32 rung) is "auto"
        while bare float32 stays "precision"."""

        from luxar.cli.gsplat_ops.recipe_shared import detect_store_encoding
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(2)
        n = 64
        chol = (rng.random((n, 6)) * 0.5 + 0.5).astype(np.float32)
        chol[:4] = 1e8  # outliers stretch the log range → AUTO escalates to u16
        data = GSplatData(
            centers=(rng.random((n, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            cholesky_factors=chol,
        )
        out = tmp_path / "escalated.gsplats.zarr"
        with pytest.warns(UserWarning, match="escalating to uint16"):
            data.save(out)
        # Locate the array by DIRECTORY and mutate through zarr's own attrs
        # writer: the two formats keep attributes in different places
        # (`.zattrs` vs inside `zarr.json`), so hand-editing one of them only
        # edits the store that happens to be in that format.
        chol_dir = next(
            d
            for d in out.rglob("cholesky_factors_diag")
            if read_array_meta(d) is not None
        )
        chol = zarr.open_array(str(chol_dir), mode="r+")
        enc = dict(chol.attrs["encoding"])
        assert enc["name"] == "log_perchannel_u16"  # really escalated
        assert detect_store_encoding(out) == "auto"  # u16 (certified) → auto

        # Legacy pre-certificate AUTO store: bare u16, no certificate key.
        chol.attrs["encoding"] = {k: v for k, v in enc.items() if k != "certificate"}
        assert detect_store_encoding(out) == "auto"  # bare u16 (legacy) → auto

        # Certified float32 (the practically-unreachable f32 rung): auto, not
        # precision — the certificate key is the discriminator.
        chol.attrs["encoding"] = {
            "name": "float32",
            "original_dtype": "float32",
            "certificate": {
                "metric": "cov_relf_p95",
                "value": 0.0,
                "threshold": 0.05,
                "tier": "float32",
            },
        }
        assert detect_store_encoding(out) == "auto"

    def test_lod_target_ms_and_breakpoints_mutually_exclusive(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--target-ms",
                "200",
                "-b",
                "equal-count",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in normalized_cli_output(result)
        assert not out.exists()

    def test_lod_bandwidth_knob_requires_target_ms(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--bandwidth-mbps/--bytes-per-splat without --target-ms are loudly
        rejected (never silently ignored)."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "stream",
                "--bandwidth-mbps",
                "50",
            ],
        )
        assert result.exit_code != 0
        # Rich hard-wraps the error box; collapse box glyphs + whitespace so
        # the multi-word phrase survives wrapping.
        flat = normalized_cli_output(result)
        assert "only apply with --target-ms" in flat

    def test_lod_target_ms_accepted_for_substitutive(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Additive ladders are on by default for substitutive, so the
        streaming-sizing knob applies there now (sizes each level's ladder)."""
        out = tmp_path / "x.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(out),
                "--recipe",
                "levels",
                "-L",
                "1",
                "--target-ms",
                "200",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        assert out.exists()

    def test_truncation_sigmas_passes_through_as_none_by_default(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """#1180: an absent ``--truncation-sigmas`` must reach ``RecipeParams``
        as ``None`` — the sentinel the additive builders resolve to the dataset's
        own ``truncation_radius``. Substituting a float here (the historical
        ``else 3.0``) re-hardcodes the 3σ pruning bug, invisibly: the build still
        succeeds, just at a support the data never had. An explicit value is
        forwarded verbatim.
        """
        import luxar.gsplats.lod.recipes as recipes_mod

        seen: list[object] = []
        real = recipes_mod.build_recipe

        def _spy(data, recipe, params):
            seen.append(params.truncation_sigmas)
            return real(data, recipe, params)

        monkeypatch.setattr(recipes_mod, "build_recipe", _spy)

        default_out = tmp_path / "default.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(default_out),
                "--recipe",
                "stream",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{normalized_cli_output(result)}"
        assert seen == [None], seen

        explicit_out = tmp_path / "explicit.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(explicit_out),
                "--recipe",
                "stream",
                "--truncation-sigmas",
                "2.0",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{normalized_cli_output(result)}"
        assert seen[-1] == pytest.approx(2.0), seen


class TestLODCarriesAuthoredAppearance:
    """`gsplat lod` rebuilds the STRUCTURE and must not touch the APPEARANCE.

    Regression for #1600: the recipe builders construct fresh nodes that know
    nothing about the input, so the writer's own defaults used to take over —
    ``blending_mode`` disappeared entirely and opacity/gamma/intensity/absorption
    snapped back to their identity. Anyone who tuned a dataset in the Layers
    panel and then re-laddered it silently lost all of it, with a result that
    still looked structurally perfect.
    """

    #: Non-default value per carried attr. Identity values (``opacity=1.0``,
    #: ``absorption=1.0``) would make the bug INVISIBLE — the stamped default
    #: coincides with the input — so every value here differs from the default.
    AUTHORED: dict[str, Any] = {
        "blending_mode": "volumetric",
        "opacity": 0.75,
        "absorption": 0.37,
        "gamma": 1.3,
        "intensity": 2.5,
        "offset": 0.125,
        "layer": False,
        "visible": False,
        "nd_transform": {"time": {"scale": 2.0, "offset": 1.0}},
        # A builtin name that is NOT the writer's manufactured default
        # ("gray"): the default would coincide with what the writer stamps on a
        # colorless leaf anyway, so a dropped carry would still look right.
        # Not "custom" either — that sentinel names a sibling colormap_lut
        # ARRAY the attrs-only carry deliberately refuses to fake (see
        # ``read_authored_appearance``).
        "colormap": "inferno",
    }

    #: Carried by the registry but not exercised here, each for a stated reason.
    #: Asserted against the registry below so a NEW key cannot slip through
    #: unnoticed — it lands in neither dict and the coverage test fails.
    NOT_EXERCISED = {
        "join": "lines-only; a gsplats node rejects it",
    }

    def test_authored_table_covers_the_registry(self) -> None:
        """Guard the guard: every carried attr is either exercised or excused.

        Without this, adding an attr to ``AUTHORED_APPEARANCE_ATTRS`` would be
        silently uncovered by the round-trip test below.
        """
        from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS

        accounted = set(self.AUTHORED) | set(self.NOT_EXERCISED)
        assert accounted == set(AUTHORED_APPEARANCE_ATTRS), (
            "AUTHORED_APPEARANCE_ATTRS changed: "
            f"unaccounted={sorted(set(AUTHORED_APPEARANCE_ATTRS) - accounted)}, "
            f"stale={sorted(accounted - set(AUTHORED_APPEARANCE_ATTRS))}"
        )

    def test_transform_is_not_carried(self) -> None:
        """``transform`` is compositing but deliberately NOT carried.

        The stored value is column-major; the leaf writer hands whatever it gets
        to ``prepare_transform_for_zarr``, which reads row-major and transposes.
        See :func:`test_an_authored_transform_does_not_break_the_rebuild` for
        what carrying it actually did.
        """
        from luxar.core.group.compositing import (
            AUTHORED_APPEARANCE_ATTRS,
            COMPOSITING_ATTRS,
        )

        assert "transform" in COMPOSITING_ATTRS
        assert "transform" not in AUTHORED_APPEARANCE_ATTRS

    def test_an_authored_transform_does_not_break_the_rebuild(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A source root transform must not be fed back through the writer.

        The negative control for the exclusion above, on the LEAF-rooted path
        (``stream`` → ``GSplatData.save``), which is where a carried transform is
        transposed a second time: a translation lands in the bottom row and the
        command dies on ``validate_transform`` ("bottom row must be [0, 0, 0,
        1]"), and a rotation is silently inverted. Re-adding ``transform`` to the
        carried set turns this exit 0 into exit 1.
        """
        from luxar.core.transforms import prepare_transform_for_zarr, translate

        stored = prepare_transform_for_zarr(translate(5.0, 0.0, 0.0))
        self._authored_input(medium_gsplats, {**self.AUTHORED, "transform": stored})
        out = tmp_path / "with_transform.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "stream"]
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        got = self._root_attrs(out)
        # Not carried at all — the rebuild leaves the attr alone rather than
        # writing a re-transposed (wrong) one.
        assert "transform" not in got
        # ...and the rest of the appearance still travels.
        assert got["blending_mode"] == "volumetric"

    @staticmethod
    def _root_attrs(store: Path) -> dict[str, Any]:
        """A store root's attributes, whichever format wrote it.

        Format 2 keeps them in ``.zattrs``, format 3 nests them inside
        ``zarr.json`` — so naming either document reads nothing at all on the
        other format. ``None`` means "no readable node here", which is a test
        failure rather than "no attributes".
        """
        attrs = read_node_attrs(store)
        assert attrs is not None, f"no readable zarr node at {store}"
        return attrs

    @staticmethod
    def _authored_input(src: Path, authored: dict[str, Any]) -> None:
        """Stamp ``authored`` onto an existing store's root.

        Edited through the facade rather than by writing the attributes
        document directly. That is not merely a portability nicety: the
        consolidated index SHADOWS per-node attributes, so the edit has to
        reach the index too. Hand-editing dealt with that by deleting
        ``.zmetadata``, which only exists at format 2 — at format 3 the index
        is embedded in the root document, so the unlink was a silent no-op and
        the reader kept serving pre-edit attrs.

        Re-consolidating through the facade is also what keeps exactly ONE
        index, at the root; re-opening with plain ``zarr.open_group`` would
        write the stale in-memory tree back out as a nested one.

        Re-consolidating rather than deleting is what leaves the fixture store
        self-CONSISTENT: the index now agrees with the edit, so a consumer that
        opens it consolidated sees the authored value too, instead of just
        losing the index the writers all assume is there.
        """
        group = zc_open_group(src, mode="r+")
        group.attrs.update(authored)
        zc_consolidate(group)

    #: Rewriting commands that must pass appearance through, as
    #: ``label -> full argv template``. ``{in}`` / ``{out}`` are substituted with
    #: the input and output stores; ``{in2}`` asks for a SECOND authored input
    #: (see :meth:`_resolve`), which is what a multi-input command needs.
    #:
    #: Keyed by COMMAND rather than by recipe because #1600 is a property of every
    #: in->out command, not of `lod` alone: `additive` was found dropping the same
    #: eight attrs by the same missing propagation. Auditing another command
    #: (`flatten`, `decimate`, `reencode`, ...) should be one row here.
    #:
    #: A TEMPLATE rather than "extra argv after (input, output)": that older shape
    #: hardwired ``[cmd0, cmd1, IN, OUT, *rest]`` and so could not express
    #: ``gsplat merge IN1 IN2 -o OUT`` at all — which is precisely why `merge` sat
    #: unaudited through four passes of #1600 while writing no ``root_attrs``
    #: whatsoever. The sibling table in ``test_gsplat_content_scoped_metrics.py``
    #: already used templates; this one now matches it.
    #:
    #: Both write paths are represented on purpose. `lod --recipe stream|levels`
    #: goes through ``GSplatData.save``; `--recipe adaptive` and `additive` go
    #: through ``write_gsplats_tree``. A fix applied to only one would pass a
    #: single-row test.
    REWRITERS: dict[str, list[str]] = {
        "lod:stream": ["gsplat", "lod", "{in}", "{out}", "--recipe", "stream"],
        "lod:levels": ["gsplat", "lod", "{in}", "{out}", "--recipe", "levels"],
        "lod:adaptive": ["gsplat", "lod", "{in}", "{out}", "--recipe", "adaptive"],
        "additive": ["gsplat", "additive", "{in}", "{out}", "--n-lods", "4"],
        "flatten": ["gsplat", "flatten", "{in}", "{out}"],
        "partition": ["gsplat", "partition", "{in}", "{out}", "--parts", "2"],
        "decimate": ["gsplat", "decimate", "{in}", "{out}", "-f", "0.5"],
        "reencode": ["gsplat", "reencode", "{in}", "{out}", "-e", "memory"],
        "cull": ["gsplat", "cull", "{in}", "{out}", "-m", "cumulative", "-r", "0.9"],
        "filter": ["gsplat", "filter", "{in}", "{out}", "--amplitude-min", "0.05"],
        "slice": ["gsplat", "slice", "{in}", "{out}", "0:80, :, :"],
        "transform": ["gsplat", "transform", "{in}", "{out}", "--scale", "2,1,1"],
        # N inputs: carried only where they AGREE (see the merge tests below).
        "merge": ["gsplat", "merge", "{in}", "{in2}", "-o", "{out}"],
    }

    #: ``gsplat`` commands that are NOT an in->out ``.gsplats.zarr`` rewriter, each
    #: with the reason. Union'd with :data:`REWRITERS` by
    #: :meth:`test_every_gsplat_command_is_classified`, so a NEW command has to
    #: opt in to the appearance carry or explicitly excuse itself. Without that
    #: guard a command simply never appears here and inherits a silent pass —
    #: which is exactly how `merge` shipped with no carry at all.
    NOT_A_REWRITER: dict[str, str] = {
        "fit": "produces a fit from a volume; there is no source root to carry",
        "cal": "writes a calibration JSON",
        "benchmark": "GPU profile, no dataset output",
        "compare": "measures, writes no store",
        "render": "writes a volume",
        "info": "read-only",
        "view": "read-only (serves a viewer)",
        "napari": "read-only",
        "doctor": "read-only diagnostics",
        "convert": "writes a .luxar.zarr scene whose appearance it AUTHORS from "
        "its own flags (--colormap/--tone-mapping/--gamma/...)",
        "export": "writes a classical PLY, which has no appearance attrs",
        "import": "reads a foreign file that carries no Luxar appearance",
        "denoise": "operates on a volume, not on splats",
        "annotate-quality": "rewrites the input IN PLACE, stamping attrs through "
        "an r+ open; the root's own attrs are never re-written, so there is "
        "nothing to carry across",
        # ── batch-fit group ──
        "batch-fit run": "produces the fit (local multi-GPU)",
        "batch-fit submit": "produces the fit (Slurm array)",
        "batch-fit merge": "assembles the fit's OWN tiles — written by that same "
        "run, so nothing has been authored on them to carry. It is still the "
        "one other N-input assembler, and `agreed_authored_appearance` is "
        "shaped for it should tiles ever become authorable (#1600 follow-up)",
        "batch-fit status": "read-only",
        "batch-fit validate": "read-only (or deletes corrupt tiles with --fix)",
        "batch-fit cancel": "cancels Slurm jobs",
        "batch-fit denoise-calibrate": "operates on a volume, writes JSON",
        "batch-fit denoise-preprocess": "operates on a volume",
        "batch-fit resolve-floor": "resolves a floor level, writes JSON/manifest",
    }

    #: Genuine in->out rewriters that DO carry the appearance but cannot be
    #: driven from this table's fixture, with the reason and where they are
    #: covered instead. A third category rather than a convenient exemption:
    #: calling one of these "not a rewriter" would be false, and the point of
    #: the guard is that the classification stays honest.
    REWRITERS_NOT_TABLE_DRIVEN: dict[str, str] = {
        "migrate-format": "needs a LEGACY input; it refuses a current-format "
        "store, which is all this fixture can build. Carries via "
        "`root_attrs=source_appearance` in `gsplats/io/migrate.py`",
    }

    def test_every_gsplat_command_is_classified(self) -> None:
        """A new command cannot skip the carry by not being in the table.

        ``merge`` slipped four passes of the #1600 audit for exactly this
        reason: the table listed the commands someone thought of, and nothing
        compared it against the commands that actually exist.
        """
        # The sibling metrics table's enumerator, imported rather than copied so
        # the two guards cannot drift apart on what "a gsplat command" is.
        from luxar.cli.tests.test_gsplat_content_scoped_metrics import (
            _registered_gsplat_commands,
        )

        registered = _registered_gsplat_commands()
        assert "merge" in registered and "batch-fit merge" in registered, (
            f"the enumeration missed something obvious: {sorted(registered)}"
        )

        # Read the command off the argv template, not the label: `lod` appears
        # under three recipe-suffixed labels. Resolved against the REGISTERED
        # names rather than taken as `argv[1]`, because a command can be two
        # tokens (`batch-fit merge`): a nested row would otherwise classify as
        # `batch-fit`, which is not a command at all, and trip the "names
        # commands that no longer exist" assertion with a misleading message
        # instead of the "unclassified command" one it belongs to.
        def command_of(argv: list[str]) -> str:
            two = " ".join(argv[1:3])
            return two if two in registered else argv[1]

        classified = (
            {command_of(argv) for argv in self.REWRITERS.values()}
            | set(self.NOT_A_REWRITER)
            | set(self.REWRITERS_NOT_TABLE_DRIVEN)
        )
        assert not registered - classified, (
            "unclassified gsplat command(s) — add a REWRITERS row (and make it "
            "carry the appearance), or a NOT_A_REWRITER / "
            "REWRITERS_NOT_TABLE_DRIVEN reason: "
            f"{sorted(registered - classified)}"
        )
        assert not classified - registered, (
            f"the tables name commands that no longer exist: "
            f"{sorted(classified - registered)}"
        )
        assert all(self.NOT_A_REWRITER.values()), "every exemption needs a reason"
        assert all(self.REWRITERS_NOT_TABLE_DRIVEN.values()), "...and so does this one"
        assert not set(self.NOT_A_REWRITER) & set(self.REWRITERS_NOT_TABLE_DRIVEN), (
            "a command cannot be both exempt and a rewriter"
        )

    @classmethod
    def _resolve(
        cls, argv: list[str], src: Path, out: Path, tmp_path: Path
    ) -> list[str]:
        """Substitute ``{in}`` / ``{in2}`` / ``{out}`` into an argv template.

        ``{in2}`` materializes a SECOND input carrying the same authored
        appearance — a copy of ``src``, stamped through the same facade edit —
        so a multi-input row exercises the agreement path with two real stores
        rather than the same path passed twice.
        """
        second = src
        if any("{in2}" in arg for arg in argv):
            second = cls._copy_authored(src, tmp_path / "second.gsplats.zarr")
        subs = {"in": str(src), "in2": str(second), "out": str(out)}
        return [arg.format(**subs) for arg in argv]

    @staticmethod
    def _copy_authored(
        src: Path, dst: Path, authored: dict[str, Any] | None = None
    ) -> Path:
        """A byte copy of ``src``, optionally re-stamped with ``authored``."""
        import shutil

        shutil.copytree(src, dst)
        if authored is not None:
            TestLODCarriesAuthoredAppearance._authored_input(dst, authored)
        return dst

    @pytest.mark.parametrize("label", sorted(REWRITERS))
    def test_authored_appearance_survives_the_rebuild(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        label: str,
    ) -> None:
        """Every carried attr comes back off the output ROOT, per command."""
        self._authored_input(medium_gsplats, self.AUTHORED)
        out = tmp_path / f"carried_{label.replace(':', '_')}.gsplats.zarr"
        argv = self._resolve(self.REWRITERS[label], medium_gsplats, out, tmp_path)
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{label} failed:\n{result.stdout}"
        got = self._root_attrs(out)
        for key, want in self.AUTHORED.items():
            assert key in got, f"{label}: dropped {key!r} (had {want!r})"
            assert got[key] == want, f"{label}: {key} = {got[key]!r}, want {want!r}"

    # ── `gsplat merge`: N inputs, so the appearance has to be AGREED ──────────
    #
    # The table row above covers the unanimous case. These cover what only a
    # multi-input command can get wrong: a disagreement, an input that authored
    # nothing, the key whose ABSENCE is itself a value (`visible`), and the
    # cases where the merge itself invalidates a key (`colormap`).

    #: Keys the writer stamps an identity for on EVERY save, so a store nobody
    #: ever tuned still carries them on disk. Presence therefore proves nothing
    #: about authorship for these, which is why the vote treats a value EQUAL to
    #: the stamp as silence — asserted against the writer's own single-sourced
    #: mapping below so the two cannot drift.
    IDENTITY_STAMPED = (
        "opacity",
        "absorption",
        "gamma",
        "intensity",
        "offset",
        "layer",
        "colormap",
    )

    def test_identity_stamped_matches_the_writers_own_mapping(self) -> None:
        """This table IS the writer's stamp set — not a hand-kept copy of it.

        The vote's "a manufactured value is silence" rule reads
        ``WRITER_STAMPED_APPEARANCE_DEFAULTS``; if a writer ever stamps a new
        key without adding it there, the rule goes stale and the merge starts
        dropping that key on the commonest merge of all. Pin them together.
        """
        from luxar.core.group.compositing import WRITER_STAMPED_APPEARANCE_DEFAULTS

        assert set(self.IDENTITY_STAMPED) == set(WRITER_STAMPED_APPEARANCE_DEFAULTS)

    def test_every_identity_stamped_attr_has_a_default_to_stamp(self) -> None:
        """``IDENTITY_COMPOSITING_ATTRS`` ⊆ ``WRITER_STAMPED_APPEARANCE_DEFAULTS``.

        The tuple names the attrs the writers loop over; the dict supplies the
        value each one is stamped with (``apply_default_render_attrs`` and
        ``apply_gsplat_group_attrs`` both index the dict BY the tuple). So a key
        added to the tuple alone ``KeyError``s on every node write — a total
        failure to compile any scene, from a one-line edit that reads as
        harmless. The assertion above pins the merge table to the dict, which
        leaves the tuple's side of the same pairing unguarded.

        Containment, not equality: ``colormap`` is legitimately in the dict and
        not in the tuple, because its stamp is CONDITIONAL (a colorless leaf
        only) and so cannot ride the unconditional loop.
        """
        from luxar.core.group.compositing import (
            IDENTITY_COMPOSITING_ATTRS,
            WRITER_STAMPED_APPEARANCE_DEFAULTS,
        )

        missing = set(IDENTITY_COMPOSITING_ATTRS) - set(
            WRITER_STAMPED_APPEARANCE_DEFAULTS
        )
        assert not missing, (
            "IDENTITY_COMPOSITING_ATTRS names attrs with no stamped value; the "
            f"writers KeyError on every node write: {sorted(missing)}"
        )

    def _merge(
        self,
        runner: CliRunner,
        inputs: list[Path],
        out: Path,
        *extra: str,
    ) -> tuple[dict[str, Any], str]:
        """Run `gsplat merge`, returning ``(output root attrs, plain stdout)``."""
        result = runner.invoke(
            app,
            ["gsplat", "merge", *[str(p) for p in inputs], "-o", str(out), *extra],
        )
        assert result.exit_code == 0, f"merge failed:\n{result.stdout}"
        return self._root_attrs(out), _plain(result.stdout)

    def test_merge_drops_a_disagreed_key_and_says_so(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A key the inputs disagree on is dropped — the rest still travel.

        Two datasets tuned differently have no single ``blending_mode``, and
        promoting the first input's would silently relabel the second's splats.
        ``blending_mode`` is the key that shows this cleanly: it has no stamped
        identity, so "dropped" is literally absent from the output root.
        ``opacity`` is checked as well, where "dropped" means the writer's own
        1.0 rather than either input's authored value.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        other = self._copy_authored(
            medium_gsplats,
            tmp_path / "disagrees.gsplats.zarr",
            {**self.AUTHORED, "blending_mode": "additive", "opacity": 0.4},
        )
        out = tmp_path / "disagree.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, other], out)

        assert "blending_mode" not in got, (
            f"picked a side on a disagreement: {got['blending_mode']!r}"
        )
        assert got["opacity"] == 1.0, (
            f"picked a side on a disagreement: opacity={got['opacity']!r}"
        )
        # Loud, not silent: appearance is hand-authored, so the user has to be
        # told which of their choices did not survive.
        assert "blending_mode" in stdout and "opacity" in stdout
        assert "disagree" in stdout.lower()
        # ...and loud about what DID survive: the one-line announcement is the
        # only positive confirmation the user gets that a carry happened at
        # all, so pin the whole key list, not just its presence. (Deleting the
        # `aprint` otherwise leaves every test in this class green — the output
        # store is identical either way.)
        assert (
            "Carrying authored appearance: "
            + ", ".join(sorted(set(self.AUTHORED) - {"blending_mode", "opacity"}))
        ) in stdout, f"the carried keys are not announced:\n{stdout}"
        # ...and every key they DO agree on is still carried, which is the
        # assertion that fails outright if the carry is removed.
        for key, want in self.AUTHORED.items():
            if key in ("blending_mode", "opacity"):
                continue
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"

    @pytest.mark.parametrize("flavour", ["fresh-save", "stripped"])
    def test_merge_carries_a_key_only_one_input_authored(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        flavour: str,
    ) -> None:
        """An input that authored nothing casts NO vote (it must not veto).

        This is the clause a naive ``all(v == first)`` over all N inputs gets
        wrong: it would compare an authored ``"volumetric"`` against a missing
        value and drop the key, so merging a tuned dataset with an untouched
        fit would erase the tuning entirely.

        Two flavours of "silent", because on-disk silence is narrower than it
        sounds. ``fresh-save`` is the realistic one and the whole reason the
        vote treats a MANUFACTURED value as silence too: a plain
        ``GSplatData.save`` STAMPS :data:`IDENTITY_STAMPED`, so on disk that
        input opines on seven of the keys — it just opines with exactly what
        the writer would put back anyway. ``stripped`` removes them, exercising
        the absent-key half of the clause on every key at once. Either way the
        tuned input's whole look must survive.

        ``visible`` is the documented exception and is asserted separately (see
        :meth:`test_merge_visible_false_needs_unanimity`): absence there is a
        positive "shown" vote, not silence, so a unilateral ``visible=false``
        must NOT ride along.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        silent = self._copy_authored(medium_gsplats, tmp_path / "silent.gsplats.zarr")
        self._strip_authored(silent)
        if flavour == "fresh-save":
            # Put back exactly what a bare save writes, and nothing else.
            self._authored_input(silent, self._writer_identity_stamps(tmp_path))

        before = self._root_attrs(silent)
        no_opinion = [k for k in self.AUTHORED if k not in before]
        expected_silent = (
            set(self.AUTHORED)
            if flavour == "stripped"
            else set(self.AUTHORED) - set(self.IDENTITY_STAMPED)
        )
        assert set(no_opinion) == expected_silent, (
            f"fixture assumption broken ({flavour}): silent on {sorted(no_opinion)}, "
            f"expected {sorted(expected_silent)}"
        )
        assert no_opinion, "a test with nothing silent proves nothing"

        out = tmp_path / f"novote_{flavour}.gsplats.zarr"
        got, _ = self._merge(runner, [medium_gsplats, silent], out)
        # Every authored key, not just the absent ones: in `fresh-save` the
        # sibling's seven stamps are silence too, which is exactly the
        # regression this covers (they used to read as seven disagreements and
        # revert the merged root to the untouched look).
        for key in self.AUTHORED:
            if key == "visible":
                continue
            assert got.get(key) == self.AUTHORED[key], (
                f"a silent input vetoed {key!r}: got {got.get(key)!r}, "
                f"want {self.AUTHORED[key]!r}"
            )

    def test_merge_of_a_tuned_and_an_untouched_input_is_quiet(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The commonest merge of all must carry the tuning and say nothing.

        A tuned dataset plus a freshly fitted one is what people actually merge,
        and it used to be the rule's worst case: the writer's identity stamps
        (``opacity=1.0``, ``colormap="gray"``, …) are indistinguishable on disk
        from an authored identity, so SEVEN keys read as disagreements, all
        seven were dropped, and the writer then stamped its defaults back —
        which is the untouched input's value. The user got the untouched look
        either way; the warnings only announced a loss that had already
        happened. Now a manufactured value casts no vote.

        The second input is a REAL bare ``GSplatData.save``, not a store with
        the appearance keys deleted: the point is precisely that a fresh save is
        not attr-free. ``visible`` is left unauthored here because it is the one
        key where the untouched input DOES legitimately disagree (absence votes
        "shown"); it has its own test.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        authored = {k: v for k, v in self.AUTHORED.items() if k != "visible"}
        self._authored_input(medium_gsplats, authored)
        fresh = tmp_path / "fresh.gsplats.zarr"
        GSplatData.load(medium_gsplats).save(fresh)
        fresh_attrs = self._root_attrs(fresh)
        assert all(k in fresh_attrs for k in self.IDENTITY_STAMPED), (
            "fixture assumption broken: a bare save is supposed to STAMP the "
            f"identity values, got {sorted(fresh_attrs)}"
        )

        out = tmp_path / "tuned_plus_fresh.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, fresh], out)
        for key, want in authored.items():
            assert got.get(key) == want, (
                f"{key} = {got.get(key)!r}, want {want!r} — a writer stamp "
                "beat an authored value"
            )
        assert "disagree" not in stdout.lower(), (
            f"warned about a choice nobody made:\n{stdout}"
        )

    @classmethod
    def _writer_identity_stamps(cls, tmp_path: Path) -> dict[str, Any]:
        """The appearance attrs a bare ``GSplatData.save`` writes, measured.

        Measured off a throwaway store rather than hardcoded: the point of the
        ``fresh-save`` flavour is to reproduce what the writer actually does,
        and a stale literal would quietly turn it into the ``stripped`` one.
        """
        from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS
        from luxar.gsplats.gsplat_data import GSplatData

        probe = tmp_path / "probe.gsplats.zarr"
        if not probe.exists():
            GSplatData(
                centers=np.zeros((4, 3), dtype=np.float32),
                amplitudes=np.ones(4, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (4, 1)
                ),
            ).save(probe)
        attrs = cls._root_attrs(probe)
        return {k: v for k, v in attrs.items() if k in AUTHORED_APPEARANCE_ATTRS}

    @staticmethod
    def _strip_authored(store: Path) -> None:
        """Remove every authored appearance key from a store root.

        Through the facade, and re-consolidated, for the reasons spelled out on
        :meth:`_authored_input` — a deletion that misses the consolidated index
        is a no-op the reader never sees.
        """
        from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS

        group = zc_open_group(store, mode="r+")
        attrs = dict(group.attrs)
        group.attrs.put(
            {k: v for k, v in attrs.items() if k not in AUTHORED_APPEARANCE_ATTRS}
        )
        zc_consolidate(group)

    def test_merge_as_dimension_carries_nd_transform(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """``--as-dimension`` does NOT invalidate ``nd_transform``.

        It was excluded on the theory that the mode changes the dimension set
        an ``nd_transform`` is keyed against. Measured, it does not:
        ``combine_as_new_dimension`` APPENDS the new axis LAST, so every
        existing dimension keeps both its name and its index and the new axis
        simply has no entry — the identity, which is the right default. An
        entry that validated against the inputs' dimensions validates unchanged
        against the stacked set, and the positional names the viewer
        synthesizes for a detached root (``X``/``Y``/``Z``/``dim3``/…) are
        stable too, the new axis becoming ``dim4``.

        So the stacked output carries the appearance exactly like a plain
        concatenating merge does, and this test asserts the whole set rather
        than carving out an exception nobody could justify.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        other = self._copy_authored(medium_gsplats, tmp_path / "second.gsplats.zarr")
        out = tmp_path / "asdim.gsplats.zarr"
        got, stdout = self._merge(
            runner, [medium_gsplats, other], out, "--as-dimension"
        )
        assert got["ndim"] == 4, "fixture: the mode should have stacked a 4th axis"
        for key, want in self.AUTHORED.items():
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"
        assert "Not carrying authored 'nd_transform'" not in stdout, (
            f"still warning about an exclusion that no longer exists:\n{stdout}"
        )

    def test_merge_help_matches_nd_transform_and_colormap_behavior(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app,
            ["gsplat", "merge", "--help"],
            env={"FORCE_COLOR": "1", "TERM": "xterm-256color"},
        )
        assert result.exit_code == 0, result.stdout
        help_text = " ".join(_plain(result.stdout).split())
        assert "nd_transform remains valid under --as-dimension" in help_text
        assert "when a colored input authored no palette" in help_text
        assert "when any input uses a custom LUT" in help_text
        assert "nd_transform under --as-dimension" not in help_text

    @staticmethod
    def _colored_copy(src: Path, dst: Path) -> Path:
        """``src`` re-saved WITH per-splat RGB (the fixture is colorless)."""
        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData.load(src)
        rng = np.random.default_rng(7)
        GSplatData(
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
            colors=rng.random((data.n_splats, 3)).astype(np.float32),
        ).save(dst)
        return dst

    def test_merge_drops_colormap_when_it_manufactures_colors(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A palette must not ride onto an output the merge gave COLORS.

        ``GSplatData.concatenate`` white-fills the inputs that have no colors
        when the set is MIXED, so a plain merge (and ``--as-dimension``) can
        produce a colored output from a colorless input — no ``--channel-colors``
        needed. That matters because the viewer makes an ancestor palette
        override per-splat RGB unconditionally (``data/attrs-composer.ts`` →
        ``materials/gsplat/shader-tsl.ts``), so a carried ``colormap`` renders
        the COLORED input through a scalar ramp. Keying the exclusion on the
        ``--channel-colors`` flag missed this entirely; the fixture being
        colorless is why the unanimous table row could not catch it either.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        colored = self._colored_copy(medium_gsplats, tmp_path / "colored.gsplats.zarr")
        self._authored_input(colored, self.AUTHORED)
        assert self._root_attrs(colored)["has_colors"] is True, "fixture: needs RGB"
        assert self._root_attrs(medium_gsplats).get("has_colors") is False

        out = tmp_path / "mixed_colors.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, colored], out)
        assert got["has_colors"] is True, "fixture: the merge should white-fill"
        assert "colormap" not in got, (
            f"carried a palette onto a colored output: {got.get('colormap')!r}"
        )
        assert "colormap" in stdout, "the loss has to be announced"
        # ...as ONE readable line. This reason already ends in an em-dash
        # clause, and chaining the "what lands instead" clause onto it with a
        # second dash made a 430-character run-on sentence; it starts a new
        # sentence now.
        line = next(
            ln for ln in stdout.splitlines() if "Not carrying authored 'colormap'" in ln
        )
        assert line.count("—") == 1, f"two em-dash clauses in one sentence:\n{line}"
        for key, want in self.AUTHORED.items():
            if key == "colormap":
                continue
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"

    def test_merge_keeps_colormap_when_both_inputs_are_colorless(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """...and the exclusion is not simply always-on.

        The control for the test above: two colorless inputs produce a colorless
        output, nothing is manufactured, and the agreed palette rides along.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        other = self._copy_authored(
            medium_gsplats, tmp_path / "second_gray.gsplats.zarr", self.AUTHORED
        )
        out = tmp_path / "both_gray.gsplats.zarr"
        got, _ = self._merge(runner, [medium_gsplats, other], out)
        assert got["has_colors"] is False
        assert got.get("colormap") == self.AUTHORED["colormap"]

    def test_colorless_colormap_disagreement_names_the_gray_output(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        self._authored_input(medium_gsplats, {"colormap": "inferno"})
        other = self._copy_authored(
            medium_gsplats,
            tmp_path / "second_colorless.gsplats.zarr",
            {"colormap": "viridis"},
        )

        out = tmp_path / "colorless_palette_disagreement.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, other], out)

        assert got["has_colors"] is False
        assert got["colormap"] == "gray"
        line = next(
            ln for ln in stdout.splitlines() if "disagree on authored 'colormap'" in ln
        )
        assert "writer then stamps its own 'gray'" in line
        assert "gets no palette of its own" not in line

    def test_merge_drops_colormap_when_a_colored_input_has_no_palette(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A colored input with no palette is an explicit per-splat-RGB vote."""
        first = self._colored_copy(medium_gsplats, tmp_path / "rgb_a.gsplats.zarr")
        second = self._colored_copy(medium_gsplats, tmp_path / "rgb_b.gsplats.zarr")
        self._authored_input(first, {"colormap": "inferno"})
        assert self._root_attrs(second)["has_colors"] is True
        assert "colormap" not in self._root_attrs(second), (
            "fixture: the second colored input must rely on its per-splat RGB"
        )

        out = tmp_path / "colored_palette_disagreement.gsplats.zarr"
        got, stdout = self._merge(runner, [first, second], out)

        assert got["has_colors"] is True
        assert "colormap" not in got, (
            f"repainted the palette-free RGB input with {got.get('colormap')!r}"
        )
        assert "colormap" in stdout, "the dropped palette has to be announced"

    def test_merge_counts_gray_as_authored_on_a_colored_input(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The writer manufactures gray only for colorless stores."""
        first = self._colored_copy(medium_gsplats, tmp_path / "rgb_a.gsplats.zarr")
        second = self._colored_copy(medium_gsplats, tmp_path / "rgb_b.gsplats.zarr")
        self._authored_input(first, {"colormap": "inferno"})
        self._authored_input(second, {"colormap": "gray"})

        out = tmp_path / "colored_gray_disagreement.gsplats.zarr"
        got, stdout = self._merge(runner, [first, second], out)

        assert "colormap" not in got, (
            "treated an authored gray palette on a colored input as a writer default"
        )
        assert "disagree on authored 'colormap'" in stdout

    def test_merge_refuses_a_palette_when_an_input_has_a_custom_lut(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A ``colormap: "custom"`` input must not inherit a sibling's palette.

        The strip that turns ``"custom"`` into "no colormap" (its LUT is a
        sibling ARRAY the attrs-only carry cannot reach) is right for ONE input
        — the writer's default takes over. With siblings in play it silently
        demoted the input to "no opinion", so the merged root adopted the OTHER
        input's palette and applied it to the custom-LUT splats too. Refuse the
        key outright instead.
        """
        custom = self._copy_authored(
            medium_gsplats,
            tmp_path / "custom.gsplats.zarr",
            {**self.AUTHORED, "colormap": "custom"},
        )
        other = self._copy_authored(
            medium_gsplats, tmp_path / "inferno.gsplats.zarr", self.AUTHORED
        )
        out = tmp_path / "custom_merge.gsplats.zarr"
        got, stdout = self._merge(runner, [custom, other], out)
        assert got.get("colormap") != self.AUTHORED["colormap"], (
            "adopted a sibling's palette over a custom LUT"
        )
        assert "custom colormap LUT" in stdout
        assert "not extracted — only the root's metadata is read" in stdout
        assert "never even opened" not in stdout
        # The rest of the agreed look is unaffected.
        for key, want in self.AUTHORED.items():
            if key == "colormap":
                continue
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"

    def test_merge_warns_once_for_n_custom_lut_inputs(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """N custom-LUT inputs produce ONE warning, not N identical lines."""
        customs = [
            self._copy_authored(
                medium_gsplats,
                tmp_path / f"custom{i}.gsplats.zarr",
                {**self.AUTHORED, "colormap": "custom"},
            )
            for i in range(3)
        ]
        out = tmp_path / "three_custom.gsplats.zarr"
        _, stdout = self._merge(runner, customs, out)
        assert stdout.count("custom colormap LUT") == 1, (
            f"expected one line for three inputs:\n{stdout}"
        )

    def test_merge_visible_false_needs_unanimity(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """``visible=false`` rides along only when EVERY input hides.

        ``visible`` is one of the keys with no writer stamp, so the no-vote
        clause used to "rescue" it: one hidden input put ``visible=false`` on
        the merged root and the whole merged dataset opened hidden
        (``ui/layers/layer-state.ts`` reads ``node.attrs.visible !== false``).
        Absence is not silence for this key — the format gives it a meaning, so
        an input without it votes ``true``.
        """
        hidden = self._copy_authored(
            medium_gsplats, tmp_path / "hidden.gsplats.zarr", {"visible": False}
        )
        shown = self._copy_authored(medium_gsplats, tmp_path / "shown.gsplats.zarr")
        assert "visible" not in self._root_attrs(shown), (
            "fixture assumption broken: a bare save must not stamp `visible`"
        )

        # One hidden, one silent: NOT unanimous, so nothing is carried and the
        # merged dataset opens visible.
        out = tmp_path / "one_hidden.gsplats.zarr"
        got, stdout = self._merge(runner, [hidden, shown], out)
        assert got.get("visible") is not False, (
            "one input's visible=false hid the whole merged dataset"
        )
        assert "visible" in stdout, "the dropped hide has to be announced"

        # Both hidden: unanimous, so the hide is preserved.
        hidden2 = self._copy_authored(
            medium_gsplats, tmp_path / "hidden2.gsplats.zarr", {"visible": False}
        )
        out2 = tmp_path / "both_hidden.gsplats.zarr"
        got2, _ = self._merge(runner, [hidden, hidden2], out2)
        assert got2.get("visible") is False, (
            f"a unanimous hide was lost: visible={got2.get('visible')!r}"
        )

    def test_disagreement_warning_says_what_actually_lands(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """ "Dropped" is never a hole — the warning must name the real outcome.

        For an identity-stamped key the writer stamps its default right after,
        which IS the untouched input's value, so "dropping it rather than
        picking one" described a neutrality that does not exist.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        other = self._copy_authored(
            medium_gsplats,
            tmp_path / "other_opacity.gsplats.zarr",
            {**self.AUTHORED, "opacity": 0.4, "blending_mode": "additive"},
        )
        out = tmp_path / "outcome.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, other], out)
        assert got["opacity"] == 1.0
        assert "the writer then stamps its own 1.0" in stdout, (
            f"the opacity warning does not say what lands:\n{stdout}"
        )
        assert "blending_mode" not in got
        assert "leaves it unset" in stdout, (
            f"the blending_mode warning does not say what lands:\n{stdout}"
        )

    def test_disagreement_names_the_dissenting_input(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """...and WHICH input said what.

        Values alone stop being actionable past two inputs: with three stores
        where only the third dissents, a values-only message is byte-identical
        to the two-input one, so the user cannot tell which store to re-tune or
        drop. One name per DISTINCT value (the first that voted it), not one
        per input — the majority's other members add nothing.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        agrees = self._copy_authored(
            medium_gsplats, tmp_path / "agrees.gsplats.zarr", self.AUTHORED
        )
        dissents = self._copy_authored(
            medium_gsplats,
            tmp_path / "dissents.gsplats.zarr",
            {**self.AUTHORED, "opacity": 0.4},
        )
        out = tmp_path / "named.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, agrees, dissents], out)

        assert got["opacity"] == 1.0, "fixture: the disagreement should drop it"
        # Scoped to the warning LINE: every input's name also appears in the
        # command's own "Loading …" sections.
        line = next(
            (
                ln
                for ln in stdout.splitlines()
                if "disagree on authored 'opacity'" in ln
            ),
            "",
        )
        assert line, f"no opacity disagreement warning at all:\n{stdout}"
        assert "0.4 from dissents.gsplats.zarr" in line, (
            f"the warning does not name the dissenting input:\n{line}"
        )
        assert "0.75 from medium.gsplats.zarr" in line, (
            f"...nor the input the majority value came from:\n{line}"
        )
        assert "agrees.gsplats.zarr" not in line, (
            f"named every voter rather than one per distinct value:\n{line}"
        )

    def test_disagreement_keeps_paths_when_long_basenames_collide(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Shortening must not make distinct dissenting inputs indistinguishable."""
        basename = "scene_with_a_reasonably_long_name.gsplats.zarr"
        first_dir = tmp_path / "run_a"
        second_dir = tmp_path / "run_b"
        first_dir.mkdir()
        second_dir.mkdir()
        first = self._copy_authored(
            medium_gsplats,
            first_dir / basename,
            {**self.AUTHORED, "opacity": 0.75},
        )
        second = self._copy_authored(
            medium_gsplats,
            second_dir / basename,
            {**self.AUTHORED, "opacity": 0.4},
        )

        _, stdout = self._merge(
            runner, [first, second], tmp_path / "colliding_names.gsplats.zarr"
        )
        line = next(
            ln for ln in stdout.splitlines() if "disagree on authored 'opacity'" in ln
        )

        assert str(first) in line, f"first input became ambiguous:\n{line}"
        assert str(second) in line, f"second input became ambiguous:\n{line}"

    def test_merge_channel_colors_drops_colormap(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """``--channel-colors`` invalidates ``colormap`` even under agreement.

        The mode BAKES per-splat RGB, so a palette that mapped scalar amplitudes
        no longer describes what is rendered. It also has to be dropped here
        rather than left to the writer: ``apply_gsplat_group_attrs`` gates its
        manufactured ``colormap="gray"`` on ``not has_colors``, so on an RGB
        store a carried palette would be the ONLY colormap present.
        """
        self._authored_input(medium_gsplats, self.AUTHORED)
        other = self._copy_authored(medium_gsplats, tmp_path / "second.gsplats.zarr")
        out = tmp_path / "chancolors.gsplats.zarr"
        got, stdout = self._merge(
            runner,
            [medium_gsplats, other],
            out,
            "--channel-colors",
            "#ff0080,#00ff00",
        )
        assert got.get("has_colors") is True, "fixture: the mode should bake RGB"
        assert "colormap" not in got, f"carried a stale palette: {got.get('colormap')}"
        assert "colormap" in stdout
        for key, want in self.AUTHORED.items():
            if key == "colormap":
                continue
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"

    def test_merge_channel_colors_drops_colormap_from_colored_inputs(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """...and the flag-keyed branch is what does it, with nothing manufactured.

        The test above drives two COLORLESS inputs, so the merge manufactures
        RGB and the white-fill branch of ``_mode_invalidated_appearance``
        already excludes ``colormap`` on its own — replacing the
        ``--channel-colors`` branch with a no-op leaves it green. This is the
        case only that branch covers: both inputs already have per-splat RGB,
        nothing is manufactured, and yet the baked channel color makes an
        agreed palette untrue of the output (the viewer's ancestor palette
        would override the RGB the flag just wrote).

        The two branches emit DIFFERENT reasons, which is the only thing in the
        output that distinguishes them — hence the assertion on the reason text
        rather than on the absent key.
        """
        first = self._colored_copy(medium_gsplats, tmp_path / "rgb_a.gsplats.zarr")
        second = self._colored_copy(medium_gsplats, tmp_path / "rgb_b.gsplats.zarr")
        for store in (first, second):
            self._authored_input(store, self.AUTHORED)
            assert self._root_attrs(store)["has_colors"] is True, (
                "fixture: BOTH inputs must already be colored, or the "
                "white-fill branch fires instead"
            )

        out = tmp_path / "chancolors_rgb.gsplats.zarr"
        got, stdout = self._merge(
            runner,
            [first, second],
            out,
            "--channel-colors",
            "#ff0080,#00ff00",
        )
        assert got.get("has_colors") is True
        assert "colormap" not in got, (
            f"carried a palette over baked channel RGB: {got.get('colormap')!r}"
        )
        assert "bakes a per-channel RGB" in stdout, (
            f"the --channel-colors exclusion did not fire:\n{stdout}"
        )
        assert "white fill" not in stdout, (
            f"the manufactured-colors branch fired instead:\n{stdout}"
        )
        for key, want in self.AUTHORED.items():
            if key == "colormap":
                continue
            assert got.get(key) == want, f"{key} = {got.get(key)!r}, want {want!r}"

    def test_merge_channel_colors_is_quiet_on_untouched_inputs(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The canonical `merge ch0 ch1 -o out --channel-colors …` warns nothing.

        ``colormap`` IS excluded by the mode here, but neither input ever
        authored one — a bare save stamps the manufactured ``"gray"`` — so
        there is nothing to lose and announcing the exclusion would be noise
        about a choice nobody made. This is the multi-channel merge people
        actually run, and it is what the "no votes → skip the key entirely"
        short-circuit exists for: without it every stamped-default key warns on
        every plain merge as well.
        """
        other = self._copy_authored(medium_gsplats, tmp_path / "ch1.gsplats.zarr")
        assert self._root_attrs(medium_gsplats).get("colormap") == "gray", (
            "fixture assumption broken: a bare save is supposed to stamp the "
            "manufactured palette, which is what must read as silence"
        )
        out = tmp_path / "quiet_channels.gsplats.zarr"
        got, stdout = self._merge(
            runner,
            [medium_gsplats, other],
            out,
            "--channel-colors",
            "#ff0080,#00ff00",
        )
        assert got.get("has_colors") is True, "fixture: the mode should bake RGB"
        assert "colormap" not in got
        assert "⚠️" not in stdout, f"warned about a choice nobody made:\n{stdout}"

    def test_merge_invents_nothing(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The N-input carry ECHOES too — it never manufactures a look.

        The multi-input twin of :meth:`test_carry_invents_nothing`: two
        untouched inputs agree on exactly the writer's own identity values, so
        the merged root must equal them, and the keys with no identity must stay
        absent rather than being invented by the agreement pass.

        It must also be SILENT. Every key on those roots is a writer stamp, so
        none of them casts a vote; drop the "no votes → skip the key" clause and
        each one falls through to the disagreement branch with an EMPTY value
        list, printing ``Inputs disagree on authored 'opacity' ()`` on the most
        ordinary merge there is. The store is byte-identical either way, so the
        assertions above cannot see it — only the stdout one can.
        """
        other = self._copy_authored(medium_gsplats, tmp_path / "second.gsplats.zarr")
        before = self._root_attrs(medium_gsplats)
        out = tmp_path / "bare_merge.gsplats.zarr"
        got, stdout = self._merge(runner, [medium_gsplats, other], out)
        assert "⚠️" not in stdout, (
            f"warned about untouched inputs that authored nothing:\n{stdout}"
        )
        for key in self.AUTHORED:
            assert (key in got) == (key in before), (
                f"{key}: presence changed (input={key in before}, output={key in got})"
            )
            if key in before:
                assert got[key] == before[key], (
                    f"{key}: {got[key]!r} != input {before[key]!r}"
                )
        assert "blending_mode" not in got

    @staticmethod
    def _authored_archive(src: Path, out: Path, authored: dict[str, Any]) -> None:
        """Re-save ``src`` as a compressed archive whose ROOT carries ``authored``."""
        from luxar.gsplats.gsplat_data import GSplatData

        compress = "zip" if out.name.endswith(".zip") else "tar.gz"
        GSplatData.load(src).save(out, compress=compress, root_attrs=authored)

    @staticmethod
    def _archive_root_attrs(archive: Path) -> dict[str, Any]:
        """Root attrs of an archive, obtained by actually EXTRACTING it.

        Deliberately not the peek helper the fix added: the "is this test even
        exercising anything" guard below has to be able to DISAGREE with the code
        under test, so it takes the long way round. Reading the extracted root
        through ``read_node_attrs`` keeps that independence — it is the facade's
        document reader, not the appearance path under test — while staying
        correct for whichever format wrote the archive.
        """
        import shutil

        from luxar.gsplats.io._archive import extract_compressed_zarr

        extracted = extract_compressed_zarr(archive)
        try:
            attrs = read_node_attrs(extracted)
            assert attrs is not None, f"no readable zarr node in {archive}"
            return attrs
        finally:
            shutil.rmtree(extracted.parent, ignore_errors=True)

    @pytest.mark.parametrize("recipe", ["levels", "stream"])
    @pytest.mark.parametrize("suffix", ["zip", "tar.gz"])
    def test_authored_appearance_survives_an_archive_input(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        suffix: str,
        recipe: str,
    ) -> None:
        """An ARCHIVE input carries its appearance across too (#1604).

        ``.gsplats.zarr.zip`` / ``.tar.gz`` are first-class recipe inputs — the
        loader extracts them transparently — but the appearance read bailed on
        anything that was not a directory, so an archived dataset lost every
        authored value on a rebuild while the identical directory kept them. Both
        suffixes run because they take separate extraction paths, and both a
        GROUP-rooted recipe (``levels``, whose caller attrs are copied onto the
        root verbatim) and a LEAF-rooted one (``stream``, whose attrs go through
        ``apply_gsplat_group_attrs``) because those are two different write paths.
        """
        archive = tmp_path / f"authored.gsplats.zarr.{suffix}"
        self._authored_archive(medium_gsplats, archive, self.AUTHORED)
        # Not vacuous: the INPUT archive really does carry every authored value.
        in_attrs = self._archive_root_attrs(archive)
        for key, want in self.AUTHORED.items():
            assert in_attrs.get(key) == want, (
                f"input archive lacks {key!r}: got {in_attrs.get(key)!r}, "
                f"want {want!r} — the test would pass vacuously"
            )

        tag = f"{suffix.replace('.', '_')}_{recipe}"
        out = tmp_path / f"carried_{tag}.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "lod", str(archive), str(out), "--recipe", recipe],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        got = self._root_attrs(out)
        for key, want in self.AUTHORED.items():
            assert key in got, f"{tag}: dropped {key!r} (had {want!r})"
            assert got[key] == want, f"{tag}: {key} = {got[key]!r}, want {want!r}"

    def test_carry_invents_nothing(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The carry only ever ECHOES the input — it never invents a value.

        This is what makes the round-trip test above meaningful: a writer that
        unconditionally stamped ``blending_mode="volumetric"`` would also pass
        that one. Here nothing is authored, so every carried key must match the
        input's own root exactly (present with the same value, or absent).

        Note a bare ``GSplatData.save()`` root is NOT attr-free: it already
        stamps the identity values (``opacity=1.0``, ``absorption=1.0``, ...).
        Echoing those is correct and composes to a no-op, so the assertion is
        input-vs-output equality rather than plain absence.
        """
        before = self._root_attrs(medium_gsplats)
        out = tmp_path / "bare.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "lod", str(medium_gsplats), str(out), "--recipe", "adaptive"],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        got = self._root_attrs(out)
        for key in self.AUTHORED:
            assert (key in got) == (key in before), (
                f"{key}: presence changed (input={key in before}, output={key in got})"
            )
            if key in before:
                assert got[key] == before[key], (
                    f"{key}: {got[key]!r} != input {before[key]!r}"
                )
        # blending_mode has no identity value, so nothing stamps one unasked.
        assert "blending_mode" not in before
        assert "blending_mode" not in got

    def test_structural_attrs_win_over_a_carried_collision(
        self, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A carried attr must never clobber a structural one.

        ``root_attrs`` rides the writer's LOWEST-precedence channel, so a
        colliding ``kind``/``type`` loses to the tree's own structure.
        Regression against re-introducing the meta-clobbers-structural ordering
        bug from the other direction.

        Driven at the writer rather than through the CLI on purpose:
        ``read_authored_appearance`` filters to the appearance keys, so a
        colliding ``kind`` can never reach ``root_attrs`` from a source root —
        a CLI-level version of this test would pass with the precedence
        reversed, which is exactly what it is meant to catch.
        """
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLodGroup

        leaf, _ = load_gsplat_node(medium_gsplats)
        out = tmp_path / "collide.gsplats.zarr"
        write_gsplats_tree(
            out,
            GSplatLodGroup(children=[leaf, leaf]),
            root_attrs={
                "kind": "bogus",
                "type": "bogus",
                "blending_mode": "volumetric",
            },
        )
        got = self._root_attrs(out)
        assert got["kind"] == "lod"
        assert got["type"] == "group"
        assert got["blending_mode"] == "volumetric"


class TestAdditiveCommand:
    """`gsplat additive` gives every leaf of an existing tree an additive
    ladder, structure-preservingly (the per-leaf counterpart of `lod --recipe
    additive`, which needs a flat input)."""

    def _make_substitutive(self, src: Path, out: Path) -> None:
        """Build a small 3-level substitutive kind=lod tree from src."""
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

        data = GSplatData.load(src)
        sub = build_recipe(
            data,
            "levels",
            RecipeParams(
                compression_factor=4,
                levels=2,
                device="cpu",
                additive_ladders=False,  # bare levels: the command under test
                # retrofits the ladders itself
            ),
        )
        sub.save(out)

    def test_additive_on_substitutive_tree_preserves_levels(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A kind=lod input keeps its substitutive levels; EVERY leaf gains a
        stream ladder sized to its own N."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

        sub = tmp_path / "sub.gsplats.zarr"
        self._make_substitutive(medium_gsplats, sub)
        out = tmp_path / "pyr.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "additive", str(sub), str(out), "-b", "stream:5"]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        node, _ = load_gsplat_node(out, include_stats=True)
        assert isinstance(node, GSplatLodGroup)
        assert len(node.children) == 3  # levels preserved
        for child in node.children:
            assert isinstance(child, GSplatLeaf)
            incs = [s.n_splats for s in child.additive_sublods]
            assert sum(incs) == child.n_splats
            assert incs[0] <= 5 or len(incs) == 1
            assert child.additive_sublods[0].stats["lod_breakpoints_kind"] == "stream"

    def test_additive_on_partition_preserves_parts(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A kind=partition input keeps its parts; each part gains a ladder."""
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition, iter_leaves

        part = tmp_path / "part.gsplats.zarr"
        r = runner.invoke(
            app,
            ["gsplat", "partition", str(medium_gsplats), str(part), "--parts", "3"],
        )
        assert r.exit_code == 0, normalized_cli_output(r)
        out = tmp_path / "part_add.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "additive", str(part), str(out), "--n-lods", "2"]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        node, _ = load_gsplat_node(out, include_stats=True)
        assert isinstance(node, GSplatPartition)
        total = sum(leaf.n_splats for leaf in iter_leaves(node))
        assert total == 32  # count conserved

    def test_additive_target_ms_logs_derivation(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        sub = tmp_path / "sub.gsplats.zarr"
        self._make_substitutive(medium_gsplats, sub)
        out = tmp_path / "tms.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "additive", str(sub), str(out), "--target-ms", "200"]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        io = normalized_cli_output(result)
        assert "stream:" in io and "B/splat" in io

    def test_additive_target_ms_encoding_change_uses_analytic(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Mirror of `gsplat lod`: an explicit --encoding that re-encodes the
        output sizes --target-ms from the analytic target-encoding estimate."""
        out = tmp_path / "tms_prec.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "additive",
                str(medium_gsplats),
                str(out),
                "--target-ms",
                "200",
                "--encoding",
                "precision",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        io = normalized_cli_output(result)
        assert "re-encodes the output" in io
        assert "analytic estimate" in io
        assert "measured from input store" not in io

    def test_additive_counts_exceeding_union_rejected(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Explicit counts: are clamped PER LEAF, but a largest count exceeding
        the whole dataset's N is a typo and must abort loudly — not be silently
        clamped down (e.g. counts:1000000 on a 32-splat dataset)."""
        out = tmp_path / "typo.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "additive",
                str(medium_gsplats),
                str(out),
                "-b",
                "counts:1000000",
            ],
        )
        assert result.exit_code != 0
        assert "exceeds N=32" in normalized_cli_output(result)
        assert not out.exists()

    def test_additive_overwrite_guard_and_exclusions(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "o.gsplats.zarr"
        assert (
            runner.invoke(
                app, ["gsplat", "additive", str(medium_gsplats), str(out)]
            ).exit_code
            == 0
        )
        # exists → refused without --overwrite
        again = runner.invoke(
            app, ["gsplat", "additive", str(medium_gsplats), str(out)]
        )
        assert again.exit_code != 0
        # --overwrite succeeds
        ok = runner.invoke(
            app,
            ["gsplat", "additive", str(medium_gsplats), str(out), "--overwrite"],
        )
        assert ok.exit_code == 0
        # exclusions mirror `gsplat lod`
        bad = runner.invoke(
            app,
            [
                "gsplat",
                "additive",
                str(medium_gsplats),
                str(out),
                "--overwrite",
                "--target-ms",
                "200",
                "-b",
                "equal-count",
            ],
        )
        assert bad.exit_code != 0
        assert "mutually exclusive" in normalized_cli_output(bad)

    def test_additive_reladders_existing_ladder(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """An already-laddered leaf is rebuilt from the flattened union."""
        from luxar.gsplats.gsplat_data import GSplatData

        first = tmp_path / "l1.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                [
                    "gsplat",
                    "additive",
                    str(medium_gsplats),
                    str(first),
                    "--n-lods",
                    "4",
                ],
            ).exit_code
            == 0
        )
        second = tmp_path / "l2.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "additive", str(first), str(second), "--n-lods", "2"]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        loaded = GSplatData.load(second, include_stats=True)
        assert loaded.n_additive_sublods == 2
        assert loaded.n_splats == 32  # union conserved
        # Ladder PROVENANCE must reflect the NEW ladder (n_lods=2), not the
        # source's stale stats (n_lods=4) — a blind meta copy regressed this.
        lvl_stats = loaded.substitutive_levels[0].stats
        assert lvl_stats.get("lod_n_lods") == 2, lvl_stats

    def test_re_laddering_re_derives_the_reveal_stamp_rule(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Re-laddering an already-radial ladder re-decides the stamps from the
        NEW method — it never inherits the old one's suppression or lack of it.

        This is what makes the no-energy-stamps rule safe to state
        unconditionally: there is no route by which a suppressed reveal ladder
        quietly acquires stamps, or by which an energy-ordered rebuild quietly
        loses them. Both directions are asserted, because the second is the
        sensitivity control for the first — without it, "no stamps" could just
        mean this command never stamps anything.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        radial = tmp_path / "radial.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                [
                    "gsplat",
                    "additive",
                    str(medium_gsplats),
                    str(radial),
                    "-m",
                    "radial",
                ],
            ).exit_code
            == 0
        )

        def stamps(path: Path) -> tuple[str | None, list[float | None], object]:
            data = GSplatData.load(path, include_stats=True)
            subs = data.additive_sublods
            return (
                subs[0].stats.get("lod_method"),
                [s.stats.get("energy_fraction_cum") for s in subs],
                data.substitutive_levels[0].stats.get("reference_energy"),
            )

        method, e_cum, ref = stamps(radial)
        assert method == "radial"
        assert e_cum == [None] * len(e_cum), e_cum
        assert ref is None

        # Re-ladder the REVEAL as a reveal again: still unstamped.
        again = tmp_path / "again.gsplats.zarr"
        assert (
            runner.invoke(
                app, ["gsplat", "additive", str(radial), str(again), "-m", "radial"]
            ).exit_code
            == 0
        )
        method, e_cum, ref = stamps(again)
        assert method == "radial"
        assert e_cum == [None] * len(e_cum), e_cum
        assert ref is None

        # Sensitivity control: re-ladder the SAME reveal with an energy-ordered
        # method. The stamps must come back, and the method must no longer be a
        # reveal — proving the assertions above track the method, not the command.
        energy = tmp_path / "energy.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "additive", str(radial), str(energy), "-m", "self_energy"],
            ).exit_code
            == 0
        )
        method, e_cum, ref = stamps(energy)
        assert method == "self_energy"
        assert all(v is not None for v in e_cum), e_cum
        assert ref is not None

        # The direction that actually exercises the source-stats merge: the
        # input already CARRIES a weight, so suppressing the fresh one is not
        # enough — the merge that preserves source-only stat entries must not
        # carry the stale `reference_energy` back over the reveal's omission.
        # (The arms above all start from a source with no weight, so they pass
        # either way; this one fails without the drop in `_ladder_leaf`.)
        from_stamped = tmp_path / "from_stamped.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "additive", str(energy), str(from_stamped), "-m", "radial"],
            ).exit_code
            == 0
        )
        method, e_cum, ref = stamps(from_stamped)
        assert method == "radial"
        assert e_cum == [None] * len(e_cum), e_cum
        assert ref is None, ref


class TestFlattenCommand:
    """`gsplat flatten` collapses any tree (esp. a kind=partition) into a single
    flat leaf — the bridge from a tiled `batch-fit merge` output to `gsplat lod`."""

    def _make_partition(self, runner: CliRunner, src: Path, out: Path) -> int:
        """Build a kind=partition file from `src`; return the original count."""
        from luxar.gsplats.gsplat_data import GSplatData

        result = runner.invoke(
            app, ["gsplat", "partition", str(src), str(out), "--parts", "3"]
        )
        assert result.exit_code == 0, f"partition failed: {result.stdout}"
        return GSplatData.load(src).n_splats

    def test_flatten_partition_roundtrip(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """A partition (which GSplatData.load refuses) flattens to a matrix-shaped
        leaf with the splat count conserved and loadable by GSplatData.load."""
        from luxar.gsplats.gsplat_data import GSplatData

        part = tmp_path / "part.gsplats.zarr"
        n0 = self._make_partition(runner, medium_gsplats, part)

        # Pre-fix sanity: the partition genuinely can't be loaded flat.
        with pytest.raises(ValueError):
            GSplatData.load(part)

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(part), str(flat)])
        assert result.exit_code == 0, f"flatten failed: {result.stdout}"
        assert flat.exists()

        loaded = GSplatData.load(flat)
        assert loaded.n_splats == n0  # count conserved
        assert loaded.n_substitutive == 1  # a single flat leaf (no LOD/partition)

    def test_flatten_streams_one_default_leaf_at_a_time(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The partition root is never materialized; each selected leaf is read alone."""
        import importlib

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.io._compiler import gsplat_tree

        load_module = importlib.import_module("luxar.gsplats.io.load_gsplats")

        source = GSplatData.load(medium_gsplats)
        palette = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [128, 128, 128],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        colored = GSplatData(
            centers=source.centers,
            amplitudes=source.amplitudes,
            cholesky_factors=source.cholesky_factors,
            colors=palette[np.arange(source.n_splats) % len(palette)],
        )
        colored_path = tmp_path / "colored.gsplats.zarr"
        colored.save(colored_path)
        part = tmp_path / "part.gsplats.zarr"
        total = self._make_partition(runner, colored_path, part)
        expected_node, _ = load_module.load_gsplat_node(part)
        expected = GSplatData.from_default_selection(expected_node).flattened()

        real_load_node = load_module.load_gsplat_node
        monkeypatch.setattr(
            load_module,
            "load_gsplat_node",
            lambda *args, **kwargs: pytest.fail("flatten materialized the whole tree"),
        )
        real_read = gsplat_tree.read_gsplat_node
        largest_read = 0
        total_read = 0

        def tracked_read(*args: Any, **kwargs: Any) -> Any:
            nonlocal largest_read, total_read
            node = real_read(*args, **kwargs)
            largest_read = max(largest_read, node.n_splats)
            total_read += node.n_splats
            return node

        monkeypatch.setattr(gsplat_tree, "read_gsplat_node", tracked_read)

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(part), str(flat)])
        assert result.exit_code == 0, result.stdout
        assert largest_read < total
        assert total_read == total
        monkeypatch.setattr(load_module, "load_gsplat_node", real_load_node)

        actual = GSplatData.load(flat)
        assert actual.n_splats == total
        expected_order = np.lexsort(expected.centers.T[::-1])
        actual_order = np.lexsort(actual.centers.T[::-1])
        np.testing.assert_allclose(
            actual.centers[actual_order], expected.centers[expected_order]
        )
        np.testing.assert_allclose(
            actual.amplitudes[actual_order], expected.amplitudes[expected_order]
        )
        np.testing.assert_allclose(
            actual.cholesky_factors[actual_order],
            expected.cholesky_factors[expected_order],
        )
        np.testing.assert_array_equal(
            actual.colors[actual_order], expected.colors[expected_order]
        )
        root = zc_open_group(str(flat), mode="r")
        assert root.attrs["ordering"] == "hilbert"
        assert root["chunk_bounds"].shape[0] >= 1

    def test_flatten_roundtrips_lut_encoded_colors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Palette colors use their encoded logical shape during metadata sizing."""
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(31)
        n_splats = 3000
        palette = rng.integers(0, 256, size=(8, 3), dtype=np.uint8)
        source_data = GSplatData(
            centers=rng.normal(size=(n_splats, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n_splats).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                (n_splats, 1),
            ),
            colors=palette[np.arange(n_splats) % len(palette)],
        )
        source = tmp_path / "palette.gsplats.zarr"
        source_data.save(source)
        source_root = zc_open_group(str(source), mode="r")
        assert source_root["colors"].ndim == 1
        assert source_root["colors"].attrs["encoding"]["name"].startswith("lut_")

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(source), str(flat)])
        assert result.exit_code == 0, result.stdout
        expected = GSplatData.load(source)
        actual = GSplatData.load(flat)
        expected_order = np.lexsort(expected.centers.T[::-1])
        actual_order = np.lexsort(actual.centers.T[::-1])
        np.testing.assert_array_equal(
            actual.colors[actual_order], expected.colors[expected_order]
        )

    @pytest.mark.skipif(
        not Path("/proc/self/smaps_rollup").exists(),
        reason="anonymous-memory accounting requires Linux procfs",
    )
    def test_flatten_peak_anonymous_memory_is_bounded(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Flatten stays well below the resident cost of a whole-tree concat."""
        import subprocess
        import sys
        import textwrap

        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(23)
        n_splats = 1_000_000
        source = tmp_path / "source.gsplats.zarr"
        GSplatData(
            centers=rng.normal(size=(n_splats, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n_splats).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                (n_splats, 1),
            ),
        ).save(source)
        partition = tmp_path / "partition.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "partition", str(source), str(partition), "--parts", "8"],
        )
        assert result.exit_code == 0, result.stdout

        flat = tmp_path / "flat.gsplats.zarr"
        script = textwrap.dedent(
            """
            import gc
            import threading
            from pathlib import Path

            from typer.testing import CliRunner
            from luxar._zarr_compat import open_group
            from luxar.cli import app
            from luxar.cli.gsplat_ops.transforms.partition_flatten import (
                _default_leaf_paths,
                _iter_flatten_splat_sets,
            )
            from luxar.io.ordering import sort_splats_spatial

            def anonymous_bytes():
                for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
                    if line.startswith('Anonymous:'):
                        return int(line.split()[1]) * 1024
                raise RuntimeError('Anonymous total missing from smaps_rollup')

            root = open_group(__import__('sys').argv[1], mode='r')
            first = next(_iter_flatten_splat_sets(root, _default_leaf_paths(root)[:1]))
            sort_splats_spatial(first.centers, method='hilbert')
            del first, root
            gc.collect()
            baseline = anonymous_bytes()
            samples = [baseline]
            stop = threading.Event()

            def sample_memory():
                while not stop.wait(0.01):
                    samples.append(anonymous_bytes())

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            try:
                result = CliRunner().invoke(
                    app, ['gsplat', 'flatten', __import__('sys').argv[1], __import__('sys').argv[2]]
                )
            finally:
                stop.set()
                sampler.join()
            if result.exit_code != 0:
                raise RuntimeError(result.stdout) from result.exception
            print(f'PEAK_DELTA={max(samples) - baseline}')
            """
        )
        measured = subprocess.run(
            [sys.executable, "-c", script, str(partition), str(flat)],
            check=True,
            capture_output=True,
            text=True,
        )

        flat_payload_bytes = n_splats * (3 * 4 + 4 + 6 * 4)
        peak_delta = int(measured.stdout.rsplit("PEAK_DELTA=", 1)[1].splitlines()[0])
        assert peak_delta < flat_payload_bytes * 2.5

    def test_flatten_does_not_invent_large_offset_barrier(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Streaming fallback matches canonical detection at large offsets."""
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(31)
        n_splats = 4_000
        centers = np.empty((n_splats, 4), dtype=np.float32)
        centers[:, :3] = rng.uniform(0, 100, (n_splats, 3))
        centers[:, 3] = rng.uniform(100_000, 100_600, n_splats)
        cholesky = np.zeros((n_splats, 10), dtype=np.float32)
        cholesky[:, [0, 2, 5, 9]] = 3.0
        source = tmp_path / "source.gsplats.zarr"
        GSplatData(
            centers=centers,
            amplitudes=rng.uniform(0.1, 1.0, n_splats).astype(np.float32),
            cholesky_factors=cholesky,
        ).save(source)

        partition = tmp_path / "partition.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "partition", str(source), str(partition), "--parts", "4"],
        )
        assert result.exit_code == 0, result.stdout

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(partition), str(flat)])
        assert result.exit_code == 0, result.stdout

        expected = zc_open_group(str(source), mode="r")
        actual = zc_open_group(str(flat), mode="r")
        assert actual.attrs["slice_dims"] == expected.attrs["slice_dims"] == []
        assert (
            actual.attrs["ordering_dims"]
            == expected.attrs["ordering_dims"]
            == [
                0,
                1,
                2,
                3,
            ]
        )

    def test_flatten_rejects_high_cardinality_coarsen_barrier(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A continuous axis stamped as a barrier fails before run expansion."""
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(32)
        n_splats = 1_100
        centers = rng.uniform(0, 100, (n_splats, 3)).astype(np.float32)
        centers[:, 2] = np.arange(n_splats, dtype=np.float32)
        source = tmp_path / "source.gsplats.zarr"
        GSplatData(
            centers=centers,
            amplitudes=rng.uniform(0.1, 1.0, n_splats).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                (n_splats, 1),
            ),
            stats={"coarsen_dims": [0, 1]},
        ).save(source)

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(source), str(flat)])
        assert result.exit_code != 0
        assert "barrier axes [2] derived from coarsen_dims [0, 1]" in result.stdout
        assert "at most 1024 values" in result.stdout
        assert not flat.exists()

    def test_flatten_preserves_barrier_chunk_layout(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A partition of additive ladders keeps the stacked-time chunk barrier."""
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node

        rng = np.random.default_rng(17)
        n_spatial = 600
        spatial = GSplatData(
            centers=rng.normal(size=(n_spatial, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n_spatial).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                (n_spatial, 1),
            ),
        )
        stacked = GSplatData.combine_as_new_dimension(
            [spatial] * 8,
            values=list(range(8)),
            sigma=0.0,
        )
        source = tmp_path / "stacked.gsplats.zarr"
        stacked.save(source)

        tiled = tmp_path / "tiled.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(source),
                str(tiled),
                "--recipe",
                "tiles",
                "--max-elements",
                "1200",
            ],
        )
        assert result.exit_code == 0, result.stdout

        node, _ = load_gsplat_node(tiled)
        expected = GSplatData.from_default_selection(node).flattened()
        expected_path = tmp_path / "expected.gsplats.zarr"
        expected.save(
            expected_path,
            encoding_mode=EncodingMode.PRECISION,
            barrier_dims=[3],
        )

        actual_path = tmp_path / "actual.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(tiled), str(actual_path)])
        assert result.exit_code == 0, result.stdout

        expected_root = zc_open_group(str(expected_path), mode="r")
        actual_root = zc_open_group(str(actual_path), mode="r")
        for key in ("slice_dims", "ordering_dims", "chunk_size"):
            assert actual_root.attrs[key] == expected_root.attrs[key]
        expected_bounds = np.asarray(expected_root["chunk_bounds"])
        actual_bounds = np.asarray(actual_root["chunk_bounds"])
        assert len(expected_bounds) >= 4
        assert actual_bounds.shape == expected_bounds.shape
        np.testing.assert_allclose(
            actual_bounds[:, 3, 1] - actual_bounds[:, 3, 0],
            expected_bounds[:, 3, 1] - expected_bounds[:, 3, 0],
        )

    def test_flatten_detects_barrier_across_stale_empty_stamps(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Global detection recovers a barrier from stale per-leaf empty stamps."""
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node

        rng = np.random.default_rng(29)
        n_spatial = 160
        spatial = GSplatData(
            centers=rng.normal(size=(n_spatial, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, n_spatial).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                (n_spatial, 1),
            ),
        )
        stacked = GSplatData.combine_as_new_dimension(
            [spatial] * 8,
            values=list(range(8)),
            sigma=0.0,
        )
        source = tmp_path / "stacked.gsplats.zarr"
        stacked.save(source)

        partition = tmp_path / "partition.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "partition",
                str(source),
                str(partition),
                "--parts",
                "4",
            ],
        )
        assert result.exit_code == 0, result.stdout
        partition_root = zc_open_group(str(partition), mode="a")
        part_names = [name for name in partition_root if str(name).startswith("part_")]
        assert len(part_names) == 4
        for name in part_names:
            partition_root[name].attrs["slice_dims"] = []

        node, _ = load_gsplat_node(partition)
        expected = GSplatData.from_default_selection(node).flattened()
        expected_path = tmp_path / "expected.gsplats.zarr"
        expected.save(
            expected_path,
            encoding_mode=EncodingMode.PRECISION,
            barrier_dims=[3],
        )

        actual_path = tmp_path / "actual.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "flatten", str(partition), str(actual_path)]
        )
        assert result.exit_code == 0, result.stdout

        expected_root = zc_open_group(str(expected_path), mode="r")
        actual_root = zc_open_group(str(actual_path), mode="r")
        assert actual_root.attrs["slice_dims"] == [3]
        expected_bounds = np.asarray(expected_root["chunk_bounds"])
        actual_bounds = np.asarray(actual_root["chunk_bounds"])
        np.testing.assert_allclose(
            actual_bounds[:, 3, 1] - actual_bounds[:, 3, 0],
            expected_bounds[:, 3, 1] - expected_bounds[:, 3, 0],
        )

    def test_flatten_rejects_unsupported_format(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        import shutil

        shutil.copytree(medium_gsplats, legacy)
        zc_open_group(str(legacy), mode="a").attrs["format_version"] = "2.0"

        output = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(legacy), str(output)])
        assert result.exit_code == 1
        assert "Unsupported format_version: '2.0'" in _plain(result.stdout)
        assert "migrate-format" in _plain(result.stdout)
        assert not output.exists()

    def test_flatten_subtree_error_names_store_root(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        partition = tmp_path / "partition.gsplats.zarr"
        self._make_partition(runner, medium_gsplats, partition)

        output = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "flatten", str(partition / "part_0"), str(output)]
        )
        assert result.exit_code == 1
        message = _plain(result.stdout)
        assert "node-tree subtree" in message
        assert ".gsplats.zarr store root" in message
        assert not output.exists()

    def test_flatten_creates_output_parent_and_stores_zip(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        import zipfile

        partition = tmp_path / "partition.gsplats.zarr"
        self._make_partition(runner, medium_gsplats, partition)
        output = tmp_path / "nested" / "flat.gsplats.zarr.zip"

        result = runner.invoke(
            app,
            [
                "gsplat",
                "flatten",
                str(partition),
                str(output),
                "--compress",
                "zip",
            ],
        )
        assert result.exit_code == 0, result.stdout
        with zipfile.ZipFile(output) as archive:
            assert archive.infolist()
            assert {entry.compress_type for entry in archive.infolist()} == {
                zipfile.ZIP_STORED
            }

    def test_flatten_corrects_inherited_fitting_count(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        source_root = zc_open_group(str(medium_gsplats), mode="a")
        source_root.require_group("fitting").attrs["n_splats"] = 32
        ladder = tmp_path / "levels.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(ladder),
                "--recipe",
                "levels",
                "-K",
                "4",
                "-L",
                "2",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.stdout

        flat = tmp_path / "flat.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "flatten", str(ladder), str(flat)])
        assert result.exit_code == 0, result.stdout
        root = zc_open_group(str(flat), mode="r")
        assert root["fitting"].attrs["n_splats"] == root.attrs["n_splats"]

    def test_flatten_rejects_whole_array_encoding_modes(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "flat.gsplats.zarr"
        for mode in ("auto", "memory"):
            result = runner.invoke(
                app,
                [
                    "gsplat",
                    "flatten",
                    str(medium_gsplats),
                    str(output),
                    "--encoding",
                    mode,
                ],
            )
            assert result.exit_code == 1
            assert "requires --encoding precision" in _plain(result.stdout)
            assert not output.exists()

    @pytest.mark.parametrize(
        "lod_args",
        [
            ["--recipe", "stream", "--n-lods", "3"],
            ["--recipe", "levels", "-K", "2", "-L", "1", "--device", "cpu"],
        ],
    )
    def test_flatten_ladder_collapses_before_partitioning(
        self,
        runner: CliRunner,
        medium_gsplats: Path,
        tmp_path: Path,
        lod_args: list[str],
    ) -> None:
        """Flattening a matrix ladder restores the documented partition remedy."""
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.gsplats.gsplat_data import GSplatData

        ladder = tmp_path / "ladder.gsplats.zarr"
        ladder_result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(medium_gsplats),
                str(ladder),
                *lod_args,
            ],
        )
        assert ladder_result.exit_code == 0, ladder_result.stdout
        loaded_ladder = GSplatData.load(ladder)
        assert loaded_ladder.n_additive_sublods > 1 or loaded_ladder.n_substitutive > 1

        flat = tmp_path / "flat.gsplats.zarr"
        flatten_result = runner.invoke(
            app, ["gsplat", "flatten", str(ladder), str(flat)]
        )
        assert flatten_result.exit_code == 0, flatten_result.stdout
        loaded = GSplatData.load(flat)
        assert loaded.n_splats == GSplatData.load(medium_gsplats).n_splats
        assert loaded.n_additive_sublods == 1
        assert loaded.n_substitutive == 1

        scene_path = tmp_path / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_file("g", flat, partition={"max_elements": 8})

        store = zarr.open_group(scene_path, mode="r")
        assert store["g"].attrs["kind"] == "partition"

    def test_flatten_then_multiscale_lod(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """The end-to-end unblock: partition → flatten → `lod --recipe overview`
        (which rejects a partition directly)."""
        part = tmp_path / "part.gsplats.zarr"
        self._make_partition(runner, medium_gsplats, part)

        # `lod` on the partition directly must fail and point at `flatten`.
        bad = tmp_path / "bad.gsplats.zarr"
        rej = runner.invoke(
            app, ["gsplat", "lod", str(part), str(bad), "--recipe", "overview"]
        )
        assert rej.exit_code != 0
        assert "flatten" in (rej.stdout + (rej.stderr or ""))

        flat = tmp_path / "flat.gsplats.zarr"
        assert (
            runner.invoke(app, ["gsplat", "flatten", str(part), str(flat)]).exit_code
            == 0
        )
        out = tmp_path / "ms.gsplats.zarr"
        ok = runner.invoke(
            app, ["gsplat", "lod", str(flat), str(out), "--recipe", "overview"]
        )
        assert ok.exit_code == 0, f"lod after flatten failed: {ok.stdout}"
        assert out.exists()

    def test_flatten_overwrite_guard(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """Without --overwrite an existing output is refused."""
        part = tmp_path / "part.gsplats.zarr"
        self._make_partition(runner, medium_gsplats, part)
        flat = tmp_path / "flat.gsplats.zarr"
        assert (
            runner.invoke(app, ["gsplat", "flatten", str(part), str(flat)]).exit_code
            == 0
        )
        again = runner.invoke(app, ["gsplat", "flatten", str(part), str(flat)])
        assert again.exit_code != 0
        # With --overwrite it succeeds.
        ok = runner.invoke(
            app, ["gsplat", "flatten", str(part), str(flat), "--overwrite"]
        )
        assert ok.exit_code == 0


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

        store = zarr.storage.LocalStore(str(path))
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
        create_array(
            splats, "centers", data=(rng.random((n, 3)) * 10).astype("float32")
        )
        create_array(splats, "amplitudes", data=rng.random(n).astype("float32"))
        create_array(splats, "cholesky_factors", data=self._identity_chol(n))
        create_array(splats, "chunk_bounds", data=np.zeros((1, 3, 2), dtype="float32"))
        zc_consolidate(store)

    def _make_v1_1(self, path: Path, lod_sizes=(6, 3)) -> None:
        import numpy as np
        import zarr

        store = zarr.storage.LocalStore(str(path))
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
            create_array(
                lod, "centers", data=(rng.random((n, 3)) * 10).astype("float32")
            )
            create_array(lod, "amplitudes", data=rng.random(n).astype("float32"))
            create_array(lod, "cholesky_factors", data=self._identity_chol(n))
            create_array(lod, "chunk_bounds", data=np.zeros((1, 3, 2), dtype="float32"))
        zc_consolidate(store)

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

    def test_migrate_v1_0_lossless_flag(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """`--lossless` preserves legacy float32 Cholesky factors exactly."""
        import numpy as np
        import zarr

        from luxar.gsplats import GSplatData

        legacy = tmp_path / "legacy.gsplats.zarr"
        out = tmp_path / "lossless.gsplats.zarr"
        self._make_v1_0(legacy, n=7)
        src_chol = np.asarray(
            zarr.open_group(str(legacy), mode="r")["splats"]["cholesky_factors"]
        )
        result = runner.invoke(
            app,
            ["gsplat", "migrate-format", str(legacy), str(out), "--lossless"],
        )
        assert result.exit_code == 0, f"--lossless migrate failed:\n{result.stdout}"
        data = GSplatData.load(out)
        np.testing.assert_array_equal(
            data.additive_sublods[0].cholesky_factors, src_chol
        )

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


class TestAxesSpec:
    """--axes lets single-volume fit/cal consume non-canonically-ordered nD data
    (the single-volume counterpart of batch-fit submit --axes)."""

    def test_apply_axes_spec_slices_channel_axis(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        # ZCYX volume (channel is axis 1, not the canonical CZYX axis 0)
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        out = _apply_axes_spec(arr, "z,c,y,x", channel=1, timepoint=None)
        assert out.shape == (2, 4, 5)  # c dropped, z/y/x kept in order
        np.testing.assert_array_equal(out, arr[:, 1, :, :])

    def test_apply_axes_spec_time_and_channel(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        # TZCYX → pick t=2, c=1
        arr = np.random.rand(3, 4, 2, 5, 6).astype(np.float32)
        out = _apply_axes_spec(arr, "t,z,c,y,x", channel=1, timepoint=2)
        assert out.shape == (4, 5, 6)
        np.testing.assert_array_equal(out, arr[2, :, 1, :, :])

    def test_apply_axes_spec_decodes_folded_channel_axes(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.arange(2 * 3 * 2 * 4 * 5 * 6, dtype=np.float32).reshape(
            2, 3, 2, 4, 5, 6
        )
        out = _apply_axes_spec(
            arr,
            "camera,time,channel,z,y,x",
            channel=2,
            timepoint=1,
        )
        np.testing.assert_array_equal(out, arr[1, 1, 0])

    def test_apply_axes_spec_reports_folded_channel_shape(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.zeros((2, 3, 2, 4, 5, 6), dtype=np.float32)
        with pytest.raises(
            ValueError,
            match=r"--channel.*camera=2.*channel=2",
        ):
            _apply_axes_spec(
                arr,
                "camera,time,channel,z,y,x",
                channel=4,
                timepoint=1,
            )

    def test_apply_axes_spec_rejects_multiple_time_axes(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.zeros((4, 3, 2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="more than one time axis"):
            _apply_axes_spec(
                arr,
                "time,t,z,y,x",
                channel=None,
                timepoint=2,
            )

    def test_apply_axes_spec_defaults_to_zero(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.random.rand(2, 3, 4, 5).astype(np.float32)
        out = _apply_axes_spec(arr, "c,z,y,x", channel=None, timepoint=None)
        np.testing.assert_array_equal(out, arr[0])  # channel defaults to 0

    def test_apply_axes_spec_validates(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="labels but the array"):
            _apply_axes_spec(arr, "z,y", channel=None, timepoint=None)  # too few
        with pytest.raises(ValueError, match="not recognised"):
            _apply_axes_spec(arr, "z,bogus,x", channel=None, timepoint=None)

    def test_load_volume_axes_override_npy(self, tmp_path: Path) -> None:
        """load_volume(axes=...) reorders/slices a non-canonical .npy stack."""
        from luxar.cli.gsplat_config import load_volume

        # a 4D ZCYX stack (would be mis-read as CZYX by the positional heuristic)
        arr = np.random.rand(6, 2, 8, 9).astype(np.float32)
        p = tmp_path / "zcyx.npy"
        np.save(p, arr)
        vol = load_volume(p, channel=1, axes="z,c,y,x")
        assert vol.shape == (6, 8, 9)
        np.testing.assert_array_equal(vol, arr[:, 1, :, :])


class TestAxesThreadingAndSqueeze:
    """Regressions for review #4: --axes must reach the uniform parallel tile
    workers, and an explicitly-kept size-1 spatial axis must not be squeezed."""

    def test_uniform_worker_cmd_forwards_axes(self) -> None:
        """build_worker_cmd emits --axes so parallel `fit --tile` workers load
        with the same axis spec the parent used (else grids disagree / corrupt)."""
        from luxar.gsplats.fit_tiled_parallel import build_worker_cmd

        cmd = build_worker_cmd(
            ["luxar"],
            "in.zarr",
            "out.zarr",
            0,
            4,
            256,
            32,
            channel=1,
            axes="z,c,y,x",
        )
        assert "--axes" in cmd
        assert cmd[cmd.index("--axes") + 1] == "z,c,y,x"
        # without axes, no --axes flag (back-compat)
        cmd2 = build_worker_cmd(["luxar"], "in.zarr", "out.zarr", 0, 4, 256, 32)
        assert "--axes" not in cmd2

    def test_load_volume_axes_keeps_size_one_spatial_axis(self, tmp_path: Path) -> None:
        """A single-z-plane stack kept via --axes z,y,x must stay 3D (1,Y,X) —
        np.squeeze must NOT drop the declared z axis."""
        from luxar.cli.gsplat_config import load_volume

        arr = np.random.rand(1, 8, 9).astype(np.float32)  # (z=1, y, x)
        p = tmp_path / "thin.npy"
        np.save(p, arr)
        vol = load_volume(p, axes="z,y,x")
        assert vol.shape == (1, 8, 9)  # z axis preserved (no squeeze)
        # contrast: WITHOUT --axes, squeeze drops the size-1 leading dim
        assert load_volume(p).shape == (8, 9)

    def test_apply_axes_spec_rejects_out_of_range_index(self) -> None:
        from luxar.io.volume import _apply_axes_spec

        arr = np.zeros((2, 3, 4, 5), dtype=np.float32)  # c=2 on axis 0
        with pytest.raises(ValueError, match="out of range"):
            _apply_axes_spec(arr, "c,z,y,x", channel=5, timepoint=None)

    def test_load_zarr_raw_returns_lazy_array_not_materialized(
        self, tmp_path: Path
    ) -> None:
        """--axes path must hand _apply_axes_spec a LAZY zarr array, so a huge nD
        movie is sliced to one 3D volume WITHOUT loading the whole thing into RAM.
        Regression: raw=True used to `return np.array(arr)` (materialized all of it)
        → OOM on a real 329-timepoint stack."""
        import zarr

        from luxar.io.volume import _load_zarr_volume

        p = tmp_path / "movie.zarr"
        z = zarr.open_array(
            str(p), mode="w", shape=(4, 8, 8, 8), chunks=(1, 8, 8, 8), dtype="f4"
        )
        z[:] = np.arange(4 * 8 * 8 * 8, dtype=np.float32).reshape(4, 8, 8, 8)
        arr = _load_zarr_volume(p, None, 2, None, raw=True)
        assert isinstance(arr, zarr.Array)  # lazy handle, NOT a materialized ndarray

    def test_load_volume_axes_slices_zarr_timepoint(self, tmp_path: Path) -> None:
        """load_volume(--axes t,z,y,x, timepoint=k) returns the correct 3D slice
        of a 4D zarr movie (the lazily-sliced path)."""
        import zarr

        from luxar.cli.gsplat_config import load_volume

        p = tmp_path / "movie.zarr"
        data = np.arange(4 * 8 * 8 * 8, dtype=np.float32).reshape(4, 8, 8, 8)
        z = zarr.open_array(
            str(p), mode="w", shape=data.shape, chunks=(1, 8, 8, 8), dtype="f4"
        )
        z[:] = data
        vol = load_volume(p, timepoint=2, axes="t,z,y,x")
        assert vol.shape == (8, 8, 8)
        np.testing.assert_array_equal(vol, data[2])


def _chol_base(path: Path) -> Path:
    """Return the group holding the Cholesky arrays (leaf root or child_0)."""
    for base in (path, path / "child_0"):
        if read_array_meta(base / "cholesky_factors_diag") is not None:
            return base
    raise AssertionError(f"no split-Cholesky arrays under {path}")


def _diag_dtype(path: Path) -> str:
    """The diag array's dtype as a NUMPY name (``uint8``, ``float32``).

    Read through zarr rather than off the metadata document: the two formats
    spell the field differently (`dtype: "|u1"` vs `data_type: "uint8"`), and
    the numpy name is what the assertions actually mean.
    """
    base = _chol_base(path)
    return str(zarr.open_array(str(base / "cholesky_factors_diag"), mode="r").dtype)


def _varying_gsplats(path: Path, n: int = 300, d: int = 3) -> Path:
    """Save a small dataset with VARYING Cholesky columns so the per-column
    quantizer engages (a constant column falls back to float32 in any mode).
    Saved with PRECISION so the source dtype is deterministically float32 —
    AUTO is an adaptive ladder and may pick uint8 or uint16 by certificate."""
    from luxar.encoding import EncodingMode
    from luxar.gsplats.gsplat_data import GSplatData

    rng = np.random.default_rng(7)
    tril = d * (d + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.2).astype(np.float32)
    di = np.cumsum(np.arange(1, d + 1)) - 1
    chol[:, di] = np.abs(chol[:, di]) + 0.5
    GSplatData(
        centers=(rng.standard_normal((n, d)) * 5).astype(np.float32),
        amplitudes=(np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32),
        cholesky_factors=chol,
    ).save(path, encoding_mode=EncodingMode.PRECISION)
    return path


class TestReencode:
    """`luxar gsplat reencode` — re-quantize Cholesky factors to a new file."""

    def test_memory_encoding_yields_uint8(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # memory encoding must store the Cholesky diag as uint8. The source is
        # saved PRECISION (float32), so the encoding demonstrably changes
        # regardless of what the adaptive AUTO ladder would pick.
        src = _varying_gsplats(tmp_path / "src.gsplats.zarr")
        src_dtype = _diag_dtype(src)
        assert src_dtype == "float32"
        out = tmp_path / "u8.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "reencode", str(src), str(out), "-e", "memory"]
        )
        assert result.exit_code == 0, result.output
        assert _diag_dtype(out) == "uint8"
        assert _diag_dtype(out) != src_dtype  # the encoding actually changed

    def test_precision_encoding_yields_float32(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "f32.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "reencode", str(sample_gsplats), str(out), "-e", "precision"],
        )
        assert result.exit_code == 0, result.output
        assert _diag_dtype(out) == "float32"

    def test_reencode_preserves_splat_count_and_geometry(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        src = GSplatData.load(sample_gsplats)
        out = tmp_path / "u8.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "reencode", str(sample_gsplats), str(out), "-e", "memory"]
        )
        assert result.exit_code == 0, result.output
        got = GSplatData.load(out)
        assert got.n_splats == src.n_splats
        assert got.ndim == src.ndim
        # uint8 Cholesky is lossy, and `-e memory` also re-quantizes centers to
        # per-axis uint16 — sub-1e-4 here only because this fixture's per-axis
        # extent is <= 10, so the quantization step is <= 10/65535 ≈ 1.5e-4.
        np.testing.assert_allclose(got.centers, src.centers, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "encoding,exact", [("precision", True), ("auto", False), ("memory", False)]
    )
    def test_ordinary_centers_are_bit_exact_only_under_precision(
        self, runner: CliRunner, tmp_path: Path, encoding: str, exact: bool
    ) -> None:
        """`reencode` re-encodes the CENTERS too, which its name does not suggest.

        On ordinary spatial splats `auto` and `memory` both quantize a coordinate
        column to uint16 over its own [min, max], so the endpoints land on exact
        codes and every INTERIOR value rounds. Only `precision` round-trips the
        column exactly. (A gridded column is the *exception* — see
        `test_stacked_axis_centers_stay_exact_in_every_mode` below.)
        """
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        # Ordinary 3-D splats with a well-resolved sigma of 1 per axis, so the
        # centers sigma rail leaves these on uint16 and the quantization is
        # actually exercised. The coordinates are IRREGULAR on purpose: an evenly
        # spaced column (0, 1, 2) is a grid, which the encoder snaps to and
        # stores exactly — that would measure the snap, not the quantization.
        src = tmp_path / "plain.gsplats.zarr"
        GSplatData(
            centers=np.array(
                [[-7.0, 2.0, 3.0], [1.0, -5.0, 6.0], [4.0, 8.0, 9.0]], dtype=np.float32
            ),
            amplitudes=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (3, 1)
            ),
        ).save(src, encoding_mode=EncodingMode.PRECISION)

        out = tmp_path / f"{encoding}.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "reencode", str(src), str(out), "-e", encoding]
        )
        assert result.exit_code == 0, result.output
        assert zarr.open_array(str(out / "centers"), mode="r").dtype == (
            np.float32 if exact else np.uint16
        )

        # Compare the SET of coordinates on one column: save/load reorders rows.
        col = np.unique(GSplatData.load(out).centers[:, 0])
        assert col.shape == (3,)
        np.testing.assert_array_equal(col[[0, 2]], [-7.0, 4.0])  # endpoints exact
        if exact:
            assert col[1] == 1.0
        else:
            assert col[1] != 1.0
            # One quantization step over the column's [-7, 4] extent.
            assert abs(col[1] - 1.0) < 11.0 / 65535.0

    @pytest.mark.parametrize("encoding", ["precision", "auto", "memory"])
    def test_stacked_axis_centers_stay_exact_in_every_mode(
        self, runner: CliRunner, tmp_path: Path, encoding: str
    ) -> None:
        """A stacked (sigma=0) axis is exempt from the quantization drift.

        `combine_as_new_dimension(..., sigma=0)` floors that column's sigma to
        1e-7, so a uint16 grid step of 3e-5 would displace a center by ~150 sigma
        and every interior frame would stop matching a slice query. The column is
        GRIDDED, though, so the encoder widens its upper rail until the
        quantization grid coincides with the data's own spacing and every frame
        round-trips bit-exactly (#1748) — under `auto` and `memory` as much as
        under `precision`, and without changing the dtype.

        The centers stay `linear_perchannel_u16` here, which is the load-bearing
        half: the geometry-aware sigma rail also sees 100% of these splats as
        unrepresentable, and if it did not defer to the snap it would store the
        whole centers array as float32 — twice the bytes and a `UserWarning`, for
        an axis that was never at risk. This test previously asserted the drift
        itself and described the consequence in its own docstring ("sits ~150
        sigma from its own slice and the slice renders as nothing"); that was the
        bug, pinned as expected behaviour. `test_gridded_axis_snap.py` covers the
        snap directly.
        """
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        def _timepoint() -> GSplatData:
            return GSplatData(
                centers=np.array(
                    [[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, 9.0]],
                    dtype=np.float32,
                ),
                amplitudes=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (3, 1)
                ),
            )

        # Three timepoints, so the stacked column has an interior value at all.
        src = tmp_path / "stacked.gsplats.zarr"
        GSplatData.combine_as_new_dimension(
            [_timepoint(), _timepoint(), _timepoint()],
            values=[0.0, 1.0, 2.0],
            sigma=0.0,
        ).save(src, encoding_mode=EncodingMode.PRECISION)

        out = tmp_path / f"{encoding}.gsplats.zarr"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.invoke(
                app, ["gsplat", "reencode", str(src), str(out), "-e", encoding]
            )
        assert result.exit_code == 0, result.output

        # The rail did NOT fire: the snap already stores this axis exactly, so
        # the centers keep the uint16 size win under the lossy modes, silently.
        assert not [w for w in caught if "fixed-point step" in str(w.message)]
        assert zarr.open_array(str(out / "centers"), mode="r").dtype == (
            np.float32 if encoding == "precision" else np.uint16
        )

        # Compare the SET of stacked coordinates: save/load reorders rows.
        stacked = np.unique(GSplatData.load(out).centers[:, 3])
        assert stacked.shape == (3,)
        np.testing.assert_array_equal(stacked, [0.0, 1.0, 2.0])

    def test_reencode_preserves_pipeline_provenance(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A kind=lod tree's pipeline/ provenance group survives the round-trip
        (write_gsplats_tree drops it unless re-supplied — regression guard)."""
        import zarr

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.lod import make_substitutive_lod

        n, d = 400, 3
        rng = np.random.default_rng(3)
        tril = d * (d + 1) // 2
        chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
        di = np.cumsum(np.arange(1, d + 1)) - 1
        chol[:, di] = np.abs(chol[:, di]) + 0.5
        base = GSplatData(
            centers=rng.standard_normal((n, d)).astype(np.float32),
            amplitudes=(np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32),
            cholesky_factors=chol,
        )
        pyr = make_substitutive_lod(base, compression_factor=4, levels=2, device="cpu")
        src = tmp_path / "pyr.gsplats.zarr"
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import tree_from_substitutive_levels

        node = tree_from_substitutive_levels(list(pyr.substitutive_levels))
        write_gsplats_tree(
            src, node, pipeline_info={"lod_kind": "substitutive", "method": "test_tag"}
        )
        pre = dict(zarr.open_group(str(src), mode="r")["pipeline"].attrs)
        assert pre.get("method") == "test_tag"

        out = tmp_path / "pyr_u8.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "reencode", str(src), str(out), "-e", "memory"]
        )
        assert result.exit_code == 0, result.output
        post_root = zarr.open_group(str(out), mode="r")
        assert "pipeline" in post_root
        assert dict(post_root["pipeline"].attrs).get("method") == "test_tag"


class TestAnnotateQualityCommand:
    """`luxar gsplat annotate-quality` — in-place Q·e stamp retrofit."""

    def _build_unstamped_levels(
        self, runner: CliRunner, source: Path, out: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "lod",
                str(source),
                str(out),
                "--recipe",
                "levels",
                "--no-quality-stamps",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"

    def test_annotate_stamps_in_place(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        """--with-quality retrofits Q + w onto a store built without them."""
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "lvl.gsplats.zarr"
        self._build_unstamped_levels(runner, medium_gsplats, out)
        for lev in GSplatData.load(out).substitutive_levels:
            assert "quality" not in lev.stats

        result = runner.invoke(
            app,
            [
                "gsplat",
                "annotate-quality",
                str(out),
                "--with-quality",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        levels = GSplatData.load(out).substitutive_levels
        assert levels[0].stats["quality"] == 1.0
        for lev in levels:
            assert 0.0 <= lev.stats["quality"] <= 1.0
            assert lev.stats["reference_energy"] > 0
            for sub in lev.additive_sublods:
                assert 0.0 < sub.stats["energy_fraction_cum"] <= 1.0

    def test_annotate_dry_run_writes_nothing(
        self, runner: CliRunner, medium_gsplats: Path, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / "lvl.gsplats.zarr"
        self._build_unstamped_levels(runner, medium_gsplats, out)
        before_hash = zarr.open_group(str(out), mode="r").attrs["content_hash"]
        result = runner.invoke(
            app, ["gsplat", "annotate-quality", str(out), "--dry-run"]
        )
        assert result.exit_code == 0, f"failed:\n{result.stdout}"
        for lev in GSplatData.load(out).substitutive_levels:
            assert "quality" not in lev.stats
        assert zarr.open_group(str(out), mode="r").attrs["content_hash"] == before_hash

    def test_annotate_rejects_compressed_store(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        fake = tmp_path / "x.gsplats.zarr.zip"
        fake.write_bytes(b"not a zip")
        result = runner.invoke(app, ["gsplat", "annotate-quality", str(fake)])
        assert result.exit_code != 0


# ═══════════════════════════════════════════════════════════════════════
# `gsplat info` measures volumes at the DATASET's truncation radius (#1180)
# ═══════════════════════════════════════════════════════════════════════


class TestInfoVolumeTruncation:
    """`gsplat info` used to compute (and label) volumes at a hardcoded 3σ, so a
    dataset fitted at the canonical 2.75 was reported (3/2.75)^ndim too large."""

    def _gsplats_at_radius(self, tmp_path: Path, radius: float) -> Path:
        from luxar.gsplats.gsplat_data import GSplatData

        n, d = 6, 3
        data = GSplatData(
            centers=(np.arange(n * d, dtype=np.float32).reshape(n, d)),
            amplitudes=np.linspace(0.2, 1.0, n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
            ),
            truncation_radius=radius,
        )
        out = tmp_path / f"r{radius}.gsplats.zarr"
        data.save(out)
        return out

    @staticmethod
    def _volume_stat(stdout: str, row: str) -> float:
        """The number printed on one row of the VOLUME statistics table.

        Scoped to the volume table: ``_print_statistics_table`` renders the same
        row names for the amplitude distribution just above it. Not line-anchored
        — arbol prefixes each line with its own tree glyph.
        """
        table = stdout.split("Volume Statistics:", 1)[1]
        match = re.search(rf"\b{row}\s*:\s*(\S+)", table)
        assert match is not None, f"no {row!r} row in:\n{table}"
        return float(match.group(1))

    def test_printed_volumes_are_computed_at_the_dataset_radius(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The volume NUMBERS — not only the labels — come from the dataset's σ.

        Labels and computation are two separate reads of ``truncation_radius``,
        so the headers can correctly say "1-Sigma" while ``_compute_splat_volumes``
        is still called with the old hardcoded 3.0. That is exactly the #1180
        mis-measurement, and a label-only assertion cannot see it. These splats
        are unit-sigma (identity Cholesky), so their volume is ``T**ndim``: at
        radius 1.0 the table must print 1.0, where a 3σ computation prints 27.
        """
        from luxar.cli.gsplat_ops.inspect_commands import _compute_splat_volumes
        from luxar.gsplats.gsplat_data import GSplatData

        path = self._gsplats_at_radius(tmp_path, 1.0)
        data = GSplatData.load(path)
        expected = _compute_splat_volumes(
            data.cholesky_factors, data.ndim, data.truncation_radius
        )
        at_3 = _compute_splat_volumes(data.cholesky_factors, data.ndim, 3.0)
        # Precondition: the two candidate computations are 3**ndim apart, so the
        # printed 6-decimal / 4-significant-digit forms cannot coincide.
        assert float(np.mean(at_3)) == pytest.approx(27.0 * float(np.mean(expected)))

        r = runner.invoke(app, ["gsplat", "info", str(path), "--no-histograms"])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        out = _plain(r.stdout)

        for row, reduce in (("Min", np.min), ("Mean", np.mean), ("Max", np.max)):
            printed = self._volume_stat(out, row)
            assert printed == pytest.approx(float(reduce(expected)), rel=1e-6), row
            assert printed != pytest.approx(float(reduce(at_3)), rel=1e-3), row
        # The SUMMARY line reports the same array through a different formatter.
        assert f"Mean splat volume (1σ): {float(np.mean(expected)):.4e}" in out
        assert f"{float(np.mean(at_3)):.4e}" not in out

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_volumes_scale_as_truncate_to_the_ndim(self, ndim: int) -> None:
        """The formula is unchanged; only the σ it is evaluated at is now the
        dataset's. Volumes at 2.75 must be (2.75/3)^ndim of those at 3.0."""
        from luxar.cli.gsplat_ops.inspect_commands import _compute_splat_volumes

        tril = ndim * (ndim + 1) // 2
        chol = np.zeros((4, tril), dtype=np.float32)
        diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
        chol[:, diag_idx] = np.linspace(0.5, 2.0, 4)[:, None]

        at_275 = _compute_splat_volumes(chol, ndim, 2.75)
        at_3 = _compute_splat_volumes(chol, ndim, 3.0)

        assert at_275 == pytest.approx(at_3 * (2.75 / 3.0) ** ndim, rel=1e-6)
        # Not accidentally equal — the hardcoded-3 bug would make them identical.
        assert not np.allclose(at_275, at_3)

    def test_header_reports_the_dataset_radius(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The section header carries the dataset's own radius, and an integral
        radius prints without a trailing `.0`."""
        canonical = self._gsplats_at_radius(tmp_path, 2.75)
        r = runner.invoke(app, ["gsplat", "info", str(canonical), "--no-histograms"])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        assert "VOLUME ANALYSIS (2.75-Sigma)" in _plain(r.stdout)
        assert "3-Sigma" not in _plain(r.stdout)
        # The SUMMARY line reports the same volumes array; its label must agree.
        assert "Mean splat volume (2.75σ)" in _plain(r.stdout)
        assert "(3σ)" not in _plain(r.stdout)

        integral = self._gsplats_at_radius(tmp_path, 4.0)
        r2 = runner.invoke(app, ["gsplat", "info", str(integral), "--no-histograms"])
        assert r2.exit_code == 0, f"failed:\n{r2.stdout}"
        assert "VOLUME ANALYSIS (4-Sigma)" in _plain(r2.stdout)

    def test_histogram_title_reports_the_dataset_radius(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = self._gsplats_at_radius(tmp_path, 2.75)
        r = runner.invoke(app, ["gsplat", "info", str(path)])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        assert "Volume Distribution (2.75σ)" in _plain(r.stdout)


# ═══════════════════════════════════════════════════════════════════════
# `gsplat info` quotes the source grid ONCE
# ═══════════════════════════════════════════════════════════════════════


class TestInfoSourceGridReportedOnce:
    """One report must not give two numbers for one quantity.

    The source-volume block RECOMPUTES `voxels/splat` from the splats actually
    stored (post-fit culling is on by default, so the fit-time stamp is stale on
    nearly every dataset), while the catch-all "Additional Metadata" dump printed
    the stamped `voxels_per_splat` underneath it.
    """

    #: What a 100³ uint16 fit stamps, plus a `voxels_per_splat` assembled at 50
    #: splats — before the default cull left the 6 that are actually stored.
    _STAMP = {
        "source_shape": [100, 100, 100],
        "source_dtype": "uint16",
        "source_voxels": 1_000_000,
        "source_bytes": 2_000_000,
        "fitted_shape": [100, 100, 100],
        "fitted_voxels": 1_000_000,
        "occupancy": 0.01,
        "voxels_per_splat": 20_000.0,
    }

    def _stamped(self, tmp_path: Path, **stats: Any) -> tuple[Path, int]:
        from luxar.gsplats.gsplat_data import GSplatData

        n, d = 6, 3
        data = GSplatData(
            centers=np.arange(n * d, dtype=np.float32).reshape(n, d),
            amplitudes=np.linspace(0.2, 1.0, n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
            ),
            stats=dict(stats),
        )
        out = tmp_path / "stamped.gsplats.zarr"
        data.save(out, ordering="none")
        return out, n

    def test_one_voxels_per_splat_figure_not_two(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path, n = self._stamped(tmp_path, **self._STAMP)
        r = runner.invoke(app, ["gsplat", "info", str(path), "--no-histograms"])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        out = _plain(r.stdout)

        recomputed = self._STAMP["fitted_voxels"] / n  # 166,666.67
        assert f"voxels/splat: {recomputed:,.0f}" in out
        # The stamped twin (20,000) must not also be printed, under its own key
        # or any other spelling of the same number.
        assert "voxels_per_splat" not in out, (
            "the stamped voxels_per_splat is still dumped in Additional Metadata:\n"
            + out
        )
        assert f"{self._STAMP['voxels_per_splat']:,.0f}" not in out
        assert out.count("voxels/splat") == 1

    def test_no_source_grid_key_is_printed_twice(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Every key the source block owns belongs to the source block alone."""
        from luxar.cli.gsplat_ops.inspect_commands import _SOURCE_GRID_STATS_KEYS

        path, _ = self._stamped(tmp_path, **self._STAMP)
        r = runner.invoke(app, ["gsplat", "info", str(path), "--no-histograms"])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        out = _plain(r.stdout)

        assert "Source volume: 100 x 100 x 100 uint16" in out  # the block DID run
        # Scoped to the catch-all dump: the source block prints its own labels
        # ("occupancy:", "voxels/splat:"), which is precisely where they belong.
        # With every stamp accounted for above, the section may not render at all.
        marker = "Additional Metadata:"
        dump = out.split(marker, 1)[1] if marker in out else ""
        for key in _SOURCE_GRID_STATS_KEYS:
            assert f"{key}:" not in dump, f"{key!r} was dumped as raw metadata too"
            assert f"{key}:" not in out.split("METADATA", 1)[-1], (
                f"{key!r} appears in the METADATA section as well as the source block"
            )

    def test_stamps_still_reported_when_the_source_block_bails(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Suppression must follow what was REPORTED, not what could be.

        Without `source_shape` the source block prints nothing at all, so the
        metadata dump is the only place those keys can appear — suppressing them
        unconditionally would silently drop them from the report.
        """
        stamp = {k: v for k, v in self._STAMP.items() if k != "source_shape"}
        path, _ = self._stamped(tmp_path, **stamp)
        r = runner.invoke(app, ["gsplat", "info", str(path), "--no-histograms"])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        out = _plain(r.stdout)

        assert "Source volume:" not in out
        assert "voxels_per_splat: 20000" in out.replace(",", "")
        assert "occupancy: 0.01" in out


class TestInfoPartitionSize:
    """The node-tree report must measure a directory store, like the flat one.

    `Path.stat().st_size` on a `.gsplats.zarr` DIRECTORY is the ~4 KB directory
    entry, not the chunks in it — and a partition is the shape most likely to be
    a directory, so `info` reported a plausible-looking 4.0 KB for every one.
    """

    def test_size_is_the_summed_chunk_size(
        self, runner: CliRunner, sample_gsplats: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "part.gsplats.zarr"
        assert (
            runner.invoke(
                app,
                ["gsplat", "partition", str(sample_gsplats), str(out), "--parts", "2"],
            ).exit_code
            == 0
        )
        real = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        assert real > out.stat().st_size  # the premise: the entry understates it

        r = runner.invoke(app, ["gsplat", "info", str(out)])
        assert r.exit_code == 0, f"failed:\n{r.stdout}"
        text = _plain(r.stdout)
        assert "Root kind: partition" in text  # the tree-summary path, not the flat one
        from luxar.cli.gsplat_ops.inspect_commands import _store_size
        from luxar.cli.utils import format_memory_size

        assert f"Size: {format_memory_size(_store_size(out))}" in text, text


class TestInfoFlatArchivedPartition:
    """`gsplat info` on a FLAT-archived partition reports the WHOLE tree (#1628).

    A "flat" archive holds the store at its ROOT (``zarr.json``/``.zgroup`` and
    ``part_0/…`` at depth 0), which is what ``zip -r x.gsplats.zarr.zip .`` from
    inside a store produces. The store-root resolution used to define the store
    as a top-level DIRECTORY, so its fallback landed on an arbitrary child picked
    by ``iterdir()`` order — for a partition, ``fitting`` or one of the ``part_N``
    groups. Measured on this code that is exit 1 with "Invalid format_type: None",
    and it can be nothing else: only the store ROOT carries that key, and the
    ``Root kind:`` line is printed only after the node load has SUCCEEDED. The
    issue was filed with `Root kind: leaf`, one part's splat count and exit 0
    instead, which means a store shaped differently from a `gsplat partition`
    output. Same root cause, and either answer is wrong.
    """

    N_SPLATS = 40
    PARTS = 4

    @staticmethod
    def _flat_zip(store: Path, archive: Path) -> Path:
        import zipfile

        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for f in sorted(store.rglob("*")):
                if f.is_file():
                    zip_ref.write(f, arcname=str(f.relative_to(store)))
        return archive

    def test_info_reports_the_partition_not_one_part(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        n = self.N_SPLATS
        src = tmp_path / "src.gsplats.zarr"
        GSplatData(
            centers=np.random.default_rng(0).random((n, 3)).astype(np.float32) * 10,
            amplitudes=np.random.default_rng(1).random(n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
            ),
        ).save(src)

        part = tmp_path / "part.gsplats.zarr"
        r = runner.invoke(
            app,
            [
                "gsplat",
                "partition",
                str(src),
                str(part),
                "--parts",
                str(self.PARTS),
            ],
        )
        assert r.exit_code == 0, f"partition failed:\n{r.stdout}"
        archive = self._flat_zip(part, tmp_path / "flat.gsplats.zarr.zip")

        r = runner.invoke(app, ["gsplat", "info", str(archive)])
        assert r.exit_code == 0, f"info failed:\n{r.stdout}"
        text = _plain(r.stdout)
        assert "Root kind: partition" in text, text
        assert f"Parts: {self.PARTS}" in text, text
        assert f"Total splats (all leaves): {n:,}" in text, text


class TestParallelTiledDownscaleFactorsThreading:
    """The grid→splat scale must reach the merge (issue #1587).

    ``dispatch_parallel_tiled`` deliberately computes its tile grid on the
    POST-downscale shape while every worker rescales its splats back to full
    resolution — and, with a ``voxel_size``, emits physical coordinates on top
    of that.  The merge builds the partition's split planes from that grid, so
    it needs the PRODUCT of both factors to state them in the splats' own
    frame — otherwise every plane is a factor too small and the viewer's
    back-to-front part ordering (#1555) is computed against planes that
    separate nothing.

    Patched at the ``fit_tiled_parallel`` boundary: the real typer command,
    volume load, tiling resolution and grid math all run.
    """

    @staticmethod
    def _volume(path: Path) -> None:
        v = np.zeros((48, 48), np.float32)
        yy, xx = np.ogrid[:48, :48]
        for cy, cx in [(12, 12), (36, 36), (12, 36), (36, 12)]:
            v += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0).astype(np.float32)
        np.save(path, v)

    @staticmethod
    def _one_splat_result() -> "GSplatData":
        from luxar.gsplats.gsplat_data import GSplatData

        chol = np.zeros((1, 3), dtype=np.float32)  # packed 2D: [l00, l10, l11]
        chol[0, 0] = chol[0, 2] = 1.0
        return GSplatData(
            centers=np.full((1, 2), 1.0, dtype=np.float32),
            amplitudes=np.ones((1,), dtype=np.float32),
            cholesky_factors=chol,
            stats={"time_seconds": 0.0},
        )

    def _run(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        extra: "list[str]",
        case: str,
    ) -> "dict[str, Any]":
        pytest.importorskip("torch", reason="the fit CLI imports the torch fitter")
        captured: dict[str, Any] = {}

        def _fake_parallel(**kwargs: Any) -> "GSplatData":
            captured.update(kwargs)
            return self._one_splat_result()

        monkeypatch.setattr(
            "luxar.gsplats.fit_tiled_parallel.fit_tiled_parallel", _fake_parallel
        )
        vol = tmp_path / "vol.npy"
        self._volume(vol)
        # An explicit per-case output name: deriving it from the arguments
        # would silently collide the moment two cases pass the same count.
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(vol),
                str(tmp_path / f"out_{case}.gsplats.zarr"),
                "--tiling",
                "uniform",
                "--tile-size",
                "24",
                "--overlap",
                "4",
                "-j",
                "2",
                "--seeds",
                "10",
                "--device",
                "cpu",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured, "fit_tiled_parallel was never reached"
        return captured

    def test_downscale_factors_reach_the_merge(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ANISOTROPIC on purpose: an axis transposition anywhere in the chain
        would survive the isotropic spelling."""
        captured = self._run(
            runner, tmp_path, monkeypatch, ["--downscale", "1,2"], "aniso"
        )
        from luxar.gsplats.fitting.downscale import downscale_volume

        source = np.load(tmp_path / "vol.npy")
        expected = downscale_volume(source, (1, 2))
        # The grid IS in downscaled voxels, per axis (48 -> 48, 48 -> 24)...
        assert captured["volume_shape"] == (48, 24)
        assert np.allclose(captured["volume"], expected)
        assert captured["volume"].flags.c_contiguous
        assert captured["volume"].flags.owndata
        # ...so the factors that lift it back must travel with it, in that
        # same axis order.
        assert captured["grid_scale"] == (1.0, 2.0)

    def test_no_downscale_threads_no_factors(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``--downscale`` the two frames already agree — no factors."""
        captured = self._run(runner, tmp_path, monkeypatch, [], "plain")
        assert captured["volume_shape"] == (48, 48)
        assert captured["volume"].shape == (48, 48)
        assert captured["grid_scale"] is None

    def test_voxel_size_composes_with_the_downscale_factors(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``voxel_size`` from ``--config`` moves the workers' splats too, and
        the two terms MULTIPLY (issue #1587)."""
        config = tmp_path / "fit.yaml"
        config.write_text("voxel_size: [4.0, 1.0]\noutput_space: real\n")
        captured = self._run(
            runner,
            tmp_path,
            monkeypatch,
            ["--downscale", "1,2", "--config", str(config)],
            "voxel_size",
        )
        assert captured["volume_shape"] == (48, 24)
        assert captured["grid_scale"] == (4.0, 2.0)

    def test_voxel_output_space_drops_the_voxel_size_term(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``output_space: voxel`` the centers stay in voxels, so only the
        downscale factors apply."""
        config = tmp_path / "fit_voxel.yaml"
        config.write_text("voxel_size: [4.0, 1.0]\noutput_space: voxel\n")
        captured = self._run(
            runner,
            tmp_path,
            monkeypatch,
            ["--downscale", "1,2", "--config", str(config)],
            "voxel_space",
        )
        assert captured["grid_scale"] == (1.0, 2.0)


class TestRefineVolumeRejectsARescaledFrame:
    """``--refine volume`` needs the tile grid and the splats in ONE frame.

    A per-part volume re-fit crops the source to the part's own ``bsp_tree``
    cell and uses that cell as VOXEL INDICES. A real-space ``voxel_size``
    breaks that exactly as ``--downscale`` does (and was already refused for):
    the grid is in voxels while the splats are in physical units, so every crop
    lands a factor off. Refused up front, in the same voice — the sequential
    partition path reaches this with no ``--downscale`` anywhere in sight
    (#1587).
    """

    @staticmethod
    def _volume(path: Path) -> None:
        v = np.zeros((48, 48), np.float32)
        yy, xx = np.ogrid[:48, :48]
        for cy, cx in [(12, 12), (36, 36), (12, 36), (36, 12)]:
            v += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0).astype(np.float32)
        np.save(path, v)

    def _invoke(
        self,
        runner: CliRunner,
        tmp_path: Path,
        config_text: str,
        case: str,
        extra: "list[str]" = [],
    ) -> "Any":
        pytest.importorskip("torch", reason="the fit CLI imports the torch fitter")
        vol = tmp_path / "vol.npy"
        self._volume(vol)
        config = tmp_path / f"{case}.yaml"
        config.write_text(config_text)
        out = tmp_path / f"out_{case}.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(vol),
                str(out),
                "--tiling",
                "uniform",
                "--tile-size",
                "24",
                "--overlap",
                "4",
                "--seeds",
                "10",
                "--iters",
                "3",
                "--recipe",
                "levels",
                "--refine",
                "volume",
                "--config",
                str(config),
                "--device",
                "cpu",
                *extra,
            ],
        )
        return result, out

    def test_a_real_space_voxel_size_is_refused_on_the_sequential_path(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result, out = self._invoke(runner, tmp_path, "voxel_size: [4.0, 1.0]\n", "real")
        output = normalized_cli_output(result)
        assert result.exit_code != 0, output
        plain = output
        assert "--refine volume" in plain and "voxel_size" in plain
        # Refused BEFORE any fitting, like its --downscale sibling.
        assert not out.exists()

    def _run_past_the_guard(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        config_text: str,
        case: str,
    ) -> None:
        """Assert the guard does NOT fire for ``config_text``.

        Runs the parallel branch with the merge stubbed out, so the assertion is
        "the fit started", not a real (slow) fit.
        """
        reached: dict[str, Any] = {}

        def _fake_parallel(**kwargs: Any) -> Any:
            reached.update(kwargs)
            return TestParallelTiledDownscaleFactorsThreading._one_splat_result()

        monkeypatch.setattr(
            "luxar.gsplats.fit_tiled_parallel.fit_tiled_parallel", _fake_parallel
        )
        result, _ = self._invoke(runner, tmp_path, config_text, case, extra=["-j", "2"])
        assert result.exit_code == 0, result.output
        assert reached, "the fit never started — the guard fired anyway"

    def test_output_space_voxel_is_not_refused(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity control: the guard is value-scoped, not a blanket ban on
        ``voxel_size``. With ``output_space: voxel`` the centers stay in voxels,
        so the grid and the splats already agree and the fit must proceed."""
        self._run_past_the_guard(
            runner,
            tmp_path,
            monkeypatch,
            "voxel_size: [4.0, 1.0]\noutput_space: voxel\n",
            "voxelspace",
        )

    def test_a_unit_voxel_size_is_not_refused(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spacing of 1 is an identity, not a frame change."""
        self._run_past_the_guard(
            runner, tmp_path, monkeypatch, "voxel_size: 1.0\n", "unit"
        )
