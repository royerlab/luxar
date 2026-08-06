"""Tests for `luxar mesh import` (classical mesh formats → a .luxar.zarr scene).

Mirrors `test_interchange_cli.py`, the gsplat equivalent: parametrize over every
dialect from the shared synthetic writers, then pin the option surface and the error
paths. No binary fixtures are committed — every input is written byte-exact at test time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.io.reader import LuxarScene
from luxar.mesh.interop import MESH_FORMATS
from luxar.mesh.interop.tests._synthetic import (
    SUFFIXES,
    WRITERS,
    make_ground_truth,
    write_gsplat_ply,
    write_ply_binary,
    write_stl_binary,
)

runner = CliRunner()
GT = make_ground_truth()


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("mesh_dialects")
    out = {}
    for fmt, writer in WRITERS.items():
        path = tmp / f"fixture_{fmt}{SUFFIXES[fmt]}"
        writer(path, GT)
        out[fmt] = path
    return out


class TestMeshImport:
    @pytest.mark.parametrize("fmt", MESH_FORMATS)
    def test_import_each_dialect(
        self, fmt: str, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / f"{fmt}.luxar.zarr"
        result = runner.invoke(app, ["mesh", "import", str(fixtures[fmt]), str(out)])
        assert result.exit_code == 0, result.stdout

        node = LuxarScene.load(out).get_mesh("mesh")
        assert node.vertices.shape == (4, 3)
        assert node.faces.shape == (4, 3)
        # Welding is what makes this 4 rather than 12 for the soup formats; asserting
        # the count here is what would catch a regression that skipped it.
        assert int(node.faces.max()) < 4

    def test_centering_is_on_by_default_and_can_be_turned_off(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        centred, raw = tmp_path / "c.luxar.zarr", tmp_path / "r.luxar.zarr"
        assert (
            runner.invoke(
                app, ["mesh", "import", str(fixtures["ply"]), str(centred)]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                app, ["mesh", "import", str(fixtures["ply"]), str(raw), "--no-center"]
            ).exit_code
            == 0
        )
        cv = LuxarScene.load(centred).get_mesh("mesh").vertices
        rv = LuxarScene.load(raw).get_mesh("mesh").vertices
        # Bounding-box midpoint, not the vertex mean: for the tetrahedron those differ
        # (0.5 vs 0.25), so this also pins WHICH centre was used.
        mid = (cv.min(axis=0) + cv.max(axis=0)) / 2
        assert abs(float(mid.max())) < 1e-5
        assert float(rv.min()) == pytest.approx(0.0, abs=1e-6)

    def test_scale_multiplies_the_extent(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "s.luxar.zarr"
        result = runner.invoke(
            app, ["mesh", "import", str(fixtures["ply"]), str(out), "--scale", "10"]
        )
        assert result.exit_code == 0, result.stdout
        v = LuxarScene.load(out).get_mesh("mesh").vertices
        assert float(v.max() - v.min()) == pytest.approx(10.0, rel=1e-4)

    def test_node_name_is_configurable(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "n.luxar.zarr"
        result = runner.invoke(
            app, ["mesh", "import", str(fixtures["obj"]), str(out), "--name", "Skull"]
        )
        assert result.exit_code == 0, result.stdout
        assert LuxarScene.load(out).get_mesh("Skull").faces.shape == (4, 3)

    def test_no_keep_normals_drops_them(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        """Both the on-disk result AND the exit code.

        Without the exit-code assertions this test was vacuous. `_verify` runs AFTER the
        scene is written, so a verification failure still leaves a correct store on disk
        — the two disk assertions passed while the command exited 1. Which is exactly
        what it was doing: `_verify` compared the read-back against the FILE's normals
        rather than the ones actually written, so every `--no-keep-normals` import
        reported "normals lost" on a completely successful import.
        """
        kept, dropped = tmp_path / "k.luxar.zarr", tmp_path / "d.luxar.zarr"
        r1 = runner.invoke(app, ["mesh", "import", str(fixtures["ply"]), str(kept)])
        assert r1.exit_code == 0, r1.stdout
        r2 = runner.invoke(
            app,
            ["mesh", "import", str(fixtures["ply"]), str(dropped), "--no-keep-normals"],
        )
        assert r2.exit_code == 0, r2.stdout
        assert LuxarScene.load(kept).get_mesh("mesh").normals is not None
        assert LuxarScene.load(dropped).get_mesh("mesh").normals is None

    def test_explicit_format_overrides_sniffing(self, tmp_path: Path) -> None:
        mislabelled = tmp_path / "actually_stl.dat"
        write_stl_binary(mislabelled, GT)
        out = tmp_path / "f.luxar.zarr"
        assert (
            runner.invoke(app, ["mesh", "import", str(mislabelled), str(out)]).exit_code
            == 1
        )
        result = runner.invoke(
            app, ["mesh", "import", str(mislabelled), str(out), "-f", "stl"]
        )
        assert result.exit_code == 0, result.stdout

    def test_existing_output_requires_overwrite(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "o.luxar.zarr"
        assert (
            runner.invoke(
                app, ["mesh", "import", str(fixtures["ply"]), str(out)]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                app, ["mesh", "import", str(fixtures["ply"]), str(out)]
            ).exit_code
            == 1
        )
        result = runner.invoke(
            app, ["mesh", "import", str(fixtures["ply"]), str(out), "--overwrite"]
        )
        assert result.exit_code == 0, result.stdout

    @pytest.mark.parametrize("as_parent", [False, True])
    def test_an_output_that_would_destroy_the_input_is_refused(
        self, as_parent: bool, tmp_path: Path
    ) -> None:
        """`--overwrite` deletes the output before the input is ever read.

        So an output that IS the input — or a directory containing it, where `rmtree`
        takes the whole tree — would remove the source and only then discover there is
        nothing left to import. Both are refused up front, and the source is still there
        afterwards.
        """
        source = tmp_path / "model" / "mesh.ply"
        source.parent.mkdir()
        write_ply_binary(source, GT)
        target = source.parent if as_parent else source

        result = runner.invoke(
            app, ["mesh", "import", str(source), str(target), "--overwrite"]
        )
        assert result.exit_code == 1
        assert "destroy the source" in result.stdout
        assert source.exists(), "the input must survive a refused import"

    def test_a_gsplat_ply_names_the_other_command(self, tmp_path: Path) -> None:
        splats = tmp_path / "splats.ply"
        write_gsplat_ply(splats)
        result = runner.invoke(
            app, ["mesh", "import", str(splats), str(tmp_path / "x.luxar.zarr")]
        )
        assert result.exit_code == 1
        assert "luxar gsplat import" in result.stdout

    def test_bad_file_fails_cleanly(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.ply"
        bad.write_bytes(b"\x00" * 200)
        result = runner.invoke(
            app, ["mesh", "import", str(bad), str(tmp_path / "y.luxar.zarr")]
        )
        assert result.exit_code == 1
        assert "Error" in result.stdout

    def test_help_lists_import(self) -> None:
        result = runner.invoke(app, ["mesh", "--help"])
        assert result.exit_code == 0
        assert "import" in result.stdout
