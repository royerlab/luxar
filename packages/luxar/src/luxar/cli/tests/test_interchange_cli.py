"""Tests for `luxar gsplat import` (classical splat formats → .gsplats.zarr)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.interop.tests._synthetic import (
    SUFFIXES,
    WRITERS,
    make_ground_truth,
)

runner = CliRunner()


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("classical")
    gt = make_ground_truth(n=16, seed=7)
    out = {}
    for fmt, writer in WRITERS.items():
        path = tmp / f"fixture_{fmt}{SUFFIXES[fmt]}"
        writer(path, gt)
        out[fmt] = path
    return out


class TestGsplatImport:
    @pytest.mark.parametrize("fmt", ["inria", "splat", "spz", "supersplat"])
    def test_import_each_dialect(
        self, fmt: str, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        out = tmp_path / f"{fmt}.gsplats.zarr"
        result = runner.invoke(app, ["gsplat", "import", str(fixtures[fmt]), str(out)])
        assert result.exit_code == 0, result.stdout
        assert "✓ Verified v3.2 output" in result.stdout

        loaded = GSplatData.load(out)
        assert loaded.n_splats == 16
        assert loaded.ndim == 3
        assert loaded.colors is not None
        assert np.isfinite(loaded.cholesky_factors).all()

    def test_explicit_format_overrides_sniffing(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "explicit.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "import", str(fixtures["splat"]), str(out), "-f", "splat"],
        )
        assert result.exit_code == 0, result.stdout

    def test_existing_output_requires_overwrite(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "twice.gsplats.zarr"
        first = runner.invoke(
            app, ["gsplat", "import", str(fixtures["splat"]), str(out)]
        )
        assert first.exit_code == 0, first.stdout
        second = runner.invoke(
            app, ["gsplat", "import", str(fixtures["splat"]), str(out)]
        )
        assert second.exit_code != 0
        third = runner.invoke(
            app,
            ["gsplat", "import", str(fixtures["splat"]), str(out), "--overwrite"],
        )
        assert third.exit_code == 0, third.stdout

    def test_no_reorient_changes_geometry(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        out_default = tmp_path / "default.gsplats.zarr"
        out_raw = tmp_path / "raw.gsplats.zarr"
        for out, extra in ((out_default, []), (out_raw, ["--no-reorient"])):
            result = runner.invoke(
                app,
                ["gsplat", "import", str(fixtures["inria"]), str(out), *extra],
            )
            assert result.exit_code == 0, result.stdout
        rotated = GSplatData.load(out_default).centers
        raw = GSplatData.load(out_raw).centers
        # 180° about X: y and z negate. Hilbert ordering permutes rows, so
        # compare order-insensitively via sorted columns.
        assert np.allclose(np.sort(rotated[:, 1]), np.sort(-raw[:, 1]), atol=1e-2)
        assert np.allclose(np.sort(rotated[:, 0]), np.sort(raw[:, 0]), atol=1e-2)

    def test_bad_file_fails_cleanly(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.splat"
        bad.write_bytes(b"x" * 33)  # not a multiple of 32
        result = runner.invoke(
            app, ["gsplat", "import", str(bad), str(tmp_path / "o.gsplats.zarr")]
        )
        assert result.exit_code != 0
        assert "Error" in result.stdout

    def test_help_lists_import(self) -> None:
        result = runner.invoke(app, ["gsplat", "--help"])
        assert result.exit_code == 0
        assert "import" in result.stdout


class TestGsplatExport:
    @pytest.fixture()
    def imported_store(self, fixtures: dict[str, Path], tmp_path: Path) -> Path:
        out = tmp_path / "imported.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "import", str(fixtures["splat"]), str(out)]
        )
        assert result.exit_code == 0, result.stdout
        return out

    def test_export_round_trip_via_cli(
        self, imported_store: Path, tmp_path: Path
    ) -> None:
        from luxar.gsplats.interop.classical_splats import read_inria_ply

        out = tmp_path / "back.ply"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "export",
                str(imported_store),
                str(out),
                "--opacity",
                "amplitude",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "✓ Verified INRIA PLY" in result.stdout
        cs = read_inria_ply(out)
        assert cs.n_splats == 16
        assert np.isfinite(cs.scales).all()

    def test_export_requires_overwrite(
        self, imported_store: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "twice.ply"
        first = runner.invoke(app, ["gsplat", "export", str(imported_store), str(out)])
        assert first.exit_code == 0, first.stdout
        second = runner.invoke(app, ["gsplat", "export", str(imported_store), str(out)])
        assert second.exit_code != 0
        third = runner.invoke(
            app, ["gsplat", "export", str(imported_store), str(out), "--overwrite"]
        )
        assert third.exit_code == 0, third.stdout

    def test_export_rejects_unknown_colormap(
        self, imported_store: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "export",
                str(imported_store),
                str(tmp_path / "o.ply"),
                "--color",
                "colormap",
                "--colormap",
                "definitely-not-a-colormap",
            ],
        )
        assert result.exit_code != 0

    def test_help_lists_export(self) -> None:
        result = runner.invoke(app, ["gsplat", "--help"])
        assert result.exit_code == 0
        assert "export" in result.stdout
