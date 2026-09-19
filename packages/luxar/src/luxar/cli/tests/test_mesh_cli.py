"""Tests for the `luxar mesh` commands — `import` and `lod`.

`import` (classical mesh formats → a .luxar.zarr scene) mirrors
`test_interchange_cli.py`, the gsplat equivalent: parametrize over every dialect from the
shared synthetic writers, then pin the option surface and the error paths. No binary
fixtures are committed — every input is written byte-exact at test time.

`lod` (a scene's mesh → a `kind=lod` ladder) is a read/write shell around
`add_mesh(substitutive_lod=…)`, so what is tested here is the shell: what the rewrite
carries across, and what it refuses before it deletes anything.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import typer
import zarr
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.io.reader import LuxarScene
from luxar.mesh.interop import MESH_FORMATS
from luxar.mesh.interop.tests._synthetic import (
    SUFFIXES,
    WRITERS,
    GroundTruth,
    make_ground_truth,
    write_gsplat_ply,
    write_ply_binary,
    write_ply_orphan_vertices,
    write_stl_binary,
)

runner = CliRunner()
GT = make_ground_truth()

_ANSI = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
#: Rich panel borders — the frame around an error, not part of its text.
_BOX = re.compile(r"[\u2500-\u257f]")


def _plain(text: str) -> str:
    """CLI output as flat text: no ANSI codes, no wrap-induced line breaks.

    Rich renders an error panel with colour and hard-wraps it to the console
    width, and it decides whether to colourise from the *environment* — off
    under a plain pytest run, on when something sets `FORCE_COLOR` (as CI
    does). So `"--subst-method cluster" in result.output` is environment-
    dependent: locally the flag arrives intact, in CI it arrives as
    ``\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-subst\\x1b[0m…``.

    A POSITIVE assertion against raw output therefore fails only in CI — and,
    worse, a NEGATIVE one ("this flag is not mentioned") passes everywhere for
    the wrong reason, because the escape codes guarantee no match. Normalise
    before asserting either way.
    """
    # Box-drawing characters go too, not just ANSI. Rich frames an error panel and
    # hard-wraps inside it, so a message that wraps between two tokens arrives as
    # `--subst-method │ │ cluster` — the substring is broken by the BORDER rather
    # than by an escape code, and an assertion on the phrase fails for a reason
    # that has nothing to do with the message. Dropping the frame and collapsing
    # whitespace leaves the sentence the user actually reads.
    return re.sub(r"\s+", " ", _BOX.sub(" ", _ANSI.sub("", text))).strip()


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

    def test_imports_a_time_and_channel_indexed_vtp_directory_as_one_5d_mesh(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "meshes" / "cells"
        source.mkdir(parents=True)
        WRITERS["vtp"](source / "P12_Ch0-registered-T0001.vtp", GT)
        WRITERS["vtp"](source / "P12_Ch2-registered-T0006.vtp", GT)
        out = tmp_path / "cells.luxar.zarr"

        result = runner.invoke(
            app,
            [
                "mesh",
                "import",
                str(source),
                str(out),
                "--scale",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Reading 1/2: P12_Ch0-registered-T0001.vtp" in result.output
        assert "Reading 2/2: P12_Ch2-registered-T0006.vtp" in result.output

        node = LuxarScene.load(out).get_mesh("mesh")
        assert node.vertices.shape == (8, 5)
        assert node.faces.shape == (8, 3)
        assert np.array_equal(np.unique(node.vertices[:, 3]), [1, 6])
        assert np.array_equal(np.unique(node.vertices[:, 4]), [0, 2])
        assert np.array_equal(node.vertices[:, :3].min(axis=0), [-1.5, -1.5, -1.5])
        assert np.array_equal(node.vertices[:, :3].max(axis=0), [1.5, 1.5, 1.5])
        assert np.array_equal(node.faces[4:], node.faces[:4] + 4)
        dimensions = zarr.open_group(out, mode="r").attrs["scene_dimensions"]
        assert [item["name"] for item in dimensions["dimensions"]] == [
            "x",
            "y",
            "z",
            "t",
            "c",
        ]
        assert dimensions["dimensions"][3]["discrete"] is True
        assert dimensions["dimensions"][3]["display"] is False
        assert dimensions["dimensions"][3]["unit"] == "frame"
        assert dimensions["dimensions"][3]["range"] == [1.0, 6.0]
        assert dimensions["dimensions"][3]["step"] == 5.0
        assert dimensions["dimensions"][4]["discrete"] is True
        assert dimensions["dimensions"][4]["display"] is False
        assert dimensions["dimensions"][4]["unit"] == "index"
        assert dimensions["dimensions"][4]["range"] == [0.0, 2.0]
        assert dimensions["dimensions"][4]["step"] == 2.0

    @pytest.mark.parametrize("axis", ["t", "c"])
    @pytest.mark.parametrize(
        ("coordinates", "expected_step"),
        [
            ([0, 1, 2], 1.0),
            ([0, 5, 10], 5.0),
            ([0, 6, 10], 2.0),
            ([1, 6, 11], 5.0),
            ([0], 1.0),
            ([7], 1.0),
        ],
        ids=[
            "consecutive",
            "strided",
            "irregular",
            "offset-stride",
            "singleton-zero",
            "singleton-nonzero",
        ],
    )
    def test_directory_dimension_step_follows_coordinate_stride(
        self,
        tmp_path: Path,
        axis: str,
        coordinates: list[int],
        expected_step: float,
    ) -> None:
        source = tmp_path / "frames"
        source.mkdir()
        for coordinate in coordinates:
            filename = (
                f"surface-T{coordinate:04d}.vtp"
                if axis == "t"
                else f"surface-Ch{coordinate}-T0001.vtp"
            )
            WRITERS["vtp"](source / filename, GT)
        out = tmp_path / "frames.luxar.zarr"

        result = runner.invoke(
            app, ["mesh", "import", str(source), str(out), "--no-center"]
        )

        assert result.exit_code == 0, result.output
        node = LuxarScene.load(out).get_mesh("mesh")
        dimensions = zarr.open_group(out, mode="r").attrs["scene_dimensions"]
        dimension_names = [item["name"] for item in dimensions["dimensions"]]
        dimension_index = dimension_names.index(axis)
        assert np.array_equal(np.unique(node.vertices[:, dimension_index]), coordinates)
        dimension = next(
            item for item in dimensions["dimensions"] if item["name"] == axis
        )
        assert dimension["range"] == [float(coordinates[0]), float(coordinates[-1])]
        assert dimension["step"] == expected_step

    def test_directory_pattern_rebases_heterogeneous_ply_meshes(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "frames"
        source.mkdir()
        larger = GroundTruth(
            vertices=np.vstack((GT.vertices, [[2.0, 0.0, 0.0]])).astype(np.float32),
            faces=np.vstack((GT.faces, [[0, 1, 4]])).astype(np.uint32),
            normals=np.vstack((GT.normals, [[1.0, 0.0, 0.0]])).astype(np.float32),
            colors=np.vstack((GT.colors, [[255, 0, 255]])).astype(np.uint8),
        )
        write_ply_binary(source / "surface-T0001.ply", GT)
        write_ply_binary(source / "surface-T0002.ply", larger)
        out = tmp_path / "frames.luxar.zarr"

        result = runner.invoke(
            app,
            [
                "mesh",
                "import",
                str(source),
                str(out),
                "--pattern",
                "*.ply",
                "--no-center",
            ],
        )
        assert result.exit_code == 0, result.output

        node = LuxarScene.load(out).get_mesh("mesh")
        assert node.vertices.shape == (9, 4)
        first_faces = node.faces[: GT.faces.shape[0]]
        second_faces = node.faces[GT.faces.shape[0] :]
        assert int(first_faces.min()) == 0
        assert int(first_faces.max()) == 3
        assert int(second_faces.min()) == 4
        assert int(second_faces.max()) == 8

    def test_directory_import_accepts_a_named_group_index_regex(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "frames"
        source.mkdir()
        WRITERS["vtp"](source / "surface.0042.vtp", GT)
        out = tmp_path / "surface.luxar.zarr"

        result = runner.invoke(
            app,
            [
                "mesh",
                "import",
                str(source),
                str(out),
                "--index-regex",
                r"surface\.(?P<t>\d+)",
            ],
        )

        assert result.exit_code == 0, result.output
        node = LuxarScene.load(out).get_mesh("mesh")
        assert node.vertices.shape == (4, 4)
        assert np.array_equal(node.vertices[:, 3], [42] * 4)

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

    def test_centering_ignores_vertices_outside_the_surface(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "orphan.ply"
        output = tmp_path / "orphan.luxar.zarr"
        write_ply_orphan_vertices(source)

        result = runner.invoke(app, ["mesh", "import", str(source), str(output)])

        assert result.exit_code == 0, result.stdout
        vertices = LuxarScene.load(output).get_mesh("mesh").vertices
        assert vertices.shape == (3, 3)
        np.testing.assert_allclose(
            (vertices.min(axis=0) + vertices.max(axis=0)) / 2,
            [0.0, 0.0, 0.0],
            atol=1e-6,
        )
        assert float(vertices.max() - vertices.min()) == pytest.approx(1.0)

    def test_no_weld_keeps_vertices_outside_the_surface(self, tmp_path: Path) -> None:
        source = tmp_path / "orphan.ply"
        output = tmp_path / "orphan.luxar.zarr"
        write_ply_orphan_vertices(source)

        result = runner.invoke(
            app,
            [
                "mesh",
                "import",
                str(source),
                str(output),
                "--no-weld",
                "--no-center",
            ],
        )

        assert result.exit_code == 0, result.stdout
        vertices = LuxarScene.load(output).get_mesh("mesh").vertices
        assert vertices.shape == (5, 3)
        assert float(vertices.max()) == 200.0

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

    @pytest.mark.parametrize("bad", ["0", "-1", "nan"])
    def test_a_non_positive_scale_is_refused(
        self, bad: str, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        """Each of these fails SILENTLY rather than loudly if it is let through.

        Zero collapses every triangle to a point (welding and the degenerate-face drop
        both ran before the scale, so nothing downstream notices); a negative value
        mirrors the mesh while leaving its stored normals and its triangle winding
        untouched, so it lights and culls from the wrong side; NaN poisons the bounding
        box the centring step reads.
        """
        out = tmp_path / f"bad{bad}.luxar.zarr"
        result = runner.invoke(
            app, ["mesh", "import", str(fixtures["ply"]), str(out), "--scale", bad]
        )
        assert result.exit_code == 1, result.stdout
        assert "--scale" in result.stdout
        assert not out.exists(), "a rejected scale must not leave a partial scene"

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

    @pytest.mark.parametrize(
        ("index_regex", "message"),
        [
            (r"frame_(?P<t>\d+", "Invalid index regex"),
            (r"frame_(?P<c>\d+)", "named 't' capture"),
        ],
    )
    def test_a_rejected_index_regex_leaves_existing_output_intact(
        self, tmp_path: Path, index_regex: str, message: str
    ) -> None:
        source = tmp_path / "frames"
        source.mkdir()
        out = tmp_path / "existing.luxar.zarr"
        out.mkdir()
        marker = out / "keep.me"
        marker.write_text("existing output")

        result = runner.invoke(
            app,
            [
                "mesh",
                "import",
                str(source),
                str(out),
                "--index-regex",
                index_regex,
                "--overwrite",
            ],
        )

        assert result.exit_code == 1
        assert message in result.stdout
        assert marker.read_text() == "existing output"

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

    def test_the_group_help_names_every_dialect_it_reads(self) -> None:
        """The `luxar mesh` blurb enumerates the formats, so it goes stale silently.

        No gate catches it: `test_docs_command_coverage` checks command paths and option
        spellings, not help PROSE. A dialect the importer supports but the blurb omits
        reads as unsupported to anyone who looks at `--help` first.
        """
        result = runner.invoke(app, ["mesh", "--help"])
        assert result.exit_code == 0
        blurb = _plain(result.stdout).lower()
        missing = [fmt for fmt in MESH_FORMATS if fmt not in blurb]
        assert not missing, (
            f"`luxar mesh --help` does not mention {missing}, which "
            "`luxar mesh import` reads"
        )


def _grid_mesh(n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A welded n×n triangulated plane — big enough to decimate twice.

    A plane rather than the sphere the decimator's own tests use: this file is
    about the COMMAND, so the surface only has to be large, welded and cheap.
    """
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    vertices = np.stack(
        [gx.ravel(), gy.ravel(), np.zeros(n * n, dtype=np.float32)], axis=1
    )
    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    ).astype(np.uint32)
    return vertices, faces


def _write_source(
    path: Path, viewer_config: object = None, **mesh_kwargs: object
) -> tuple[np.ndarray, np.ndarray]:
    from luxar import Dimensions, LuxarZarrCompiler

    vertices, faces = _grid_mesh()
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), viewer_config=viewer_config
        )
        scene.add_mesh("surf", vertices, faces, **mesh_kwargs)
    return vertices, faces


def _run(
    run_lod: Callable[..., list[int]],
    source: Path,
    out: Path,
    overwrite: bool = False,
) -> list[int]:
    """`run_lod` with the knobs these tests do not vary."""
    return run_lod(
        input_path=source,
        output_path=out,
        node_name=None,
        levels=2,
        compression_factor=4,
        method="cluster",
        overwrite=overwrite,
    )


class TestMeshLod:
    """`luxar mesh lod` — a scene's mesh node rewritten as a `kind=lod` ladder."""

    def test_explicit_qem_builds_a_ladder(self, tmp_path: Path) -> None:
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name=None,
            levels=2,
            compression_factor=4,
            method="qem",
            overwrite=False,
        )
        assert len(counts) == 3
        assert LuxarScene.load(out).get_node_metadata("surf")["kind"] == "lod"

    def test_the_source_transform_and_appearance_survive_the_rewrite(
        self, tmp_path: Path
    ) -> None:
        """The command forwarded only `shading` and `double_sided`.

        Everything else the user authored — the transform, the colormap, the
        scalars it maps, the compositing keys — was silently dropped, so the
        ladder came back at the wrong place in the scene and in the wrong
        colours. The transform is the sharp end: the surface simply lands
        somewhere else, with nothing in the output saying so.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod
        from luxar.core.transforms import read_transform_from_zarr, translate

        source = tmp_path / "src.luxar.zarr"
        transform = translate(3.0, -2.0, 1.0)
        scalars = _grid_mesh()[0][:, 0].copy()
        _write_source(
            source,
            transform=transform,
            colormap="viridis",
            scalars=scalars,
            opacity=0.4,
            blending_mode="additive",
        )

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name=None,
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2, "the grid must produce a real ladder"

        written = LuxarScene.load(out)
        wrapper = written.get_node_metadata("surf")
        assert wrapper["kind"] == "lod"
        # The compositing keys ride on the wrapper group, which is where the
        # adder puts them and where the viewer inherits them from.
        assert wrapper["opacity"] == pytest.approx(0.4)
        assert wrapper["blending_mode"] == "additive"
        # Read back through the column-major decoder: a transform round-tripped
        # through the raw attr list instead of `get_mesh` would come back
        # TRANSPOSED, which for a translation moves it into the bottom row.
        np.testing.assert_allclose(
            read_transform_from_zarr(wrapper["transform"]), transform, atol=1e-6
        )

        children = sorted(
            (m for m in written.list_meshes() if m.startswith("surf/child_")),
            key=lambda p: int(p.rsplit("_", 1)[1]),
        )
        assert len(children) >= 2
        for child in children:
            assert written.get_node_metadata(child)["colormap"] == "viridis"
            assert written.get_mesh(child).scalars is not None

    def test_a_rejected_method_leaves_an_existing_output_intact(
        self, tmp_path: Path
    ) -> None:
        """`--overwrite` used to delete the destination before validating anything.

        So an invalid `--subst-method` removed the output and only then exited 1,
        having written nothing in its place.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        out.mkdir()
        (out / "keepme.txt").write_text("previous output")

        with pytest.raises(ValueError, match="method"):
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=2,
                compression_factor=4,
                method="bogus",
                overwrite=True,
            )
        assert (out / "keepme.txt").read_text() == "previous output"

    def test_two_dimensional_qem_refusal_leaves_existing_output_intact(
        self, tmp_path: Path
    ) -> None:
        from luxar import Dimension, Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        dimensions = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("t", unit="s", display=False, discrete=True),
            ]
        )
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=dimensions)
            scene.add_mesh("surf", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        out.mkdir()
        (out / "keepme.txt").write_text("previous output")

        with pytest.raises(ValueError, match="requires at least 3 coarsening"):
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=2,
                compression_factor=4,
                method="qem",
                overwrite=True,
            )
        assert (out / "keepme.txt").read_text() == "previous output"

    def test_the_renamed_method_flag_works_and_the_old_one_points_at_it(
        self, tmp_path: Path
    ) -> None:
        """`--method` → `--subst-method` (2026-08), with a pointer for the old one.

        Both halves are asserted because either alone is satisfiable by a mistake:
        the pointer without the new flag means nothing works, and the new flag
        without the pointer leaves `-m qem` scripts to fail later against the
        additive flag `-m` would name should mesh gain an additive ordering knob.

        BOTH former spellings are exercised. `-m` was a real short form here (unlike
        on `gsplat lod`, where it survived the rename), so declaring only `--method`
        on the hidden legacy option sends `-m cluster` to typer's bare "No such
        option: -m" — no replacement, no value carried, which is exactly what the
        pointer exists to avoid.
        """
        source = tmp_path / "src.luxar.zarr"
        _write_source(source)

        ok = runner.invoke(
            app,
            [
                "mesh",
                "lod",
                str(source),
                str(tmp_path / "new.luxar.zarr"),
                "--subst-method",
                "cluster",
            ],
        )
        assert ok.exit_code == 0, ok.output
        assert (
            LuxarScene.load(tmp_path / "new.luxar.zarr").get_node_metadata("surf")[
                "kind"
            ]
            == "lod"
        )

        for spelling, stem in (("--method", "old"), ("-m", "old_short")):
            old = runner.invoke(
                app,
                [
                    "mesh",
                    "lod",
                    str(source),
                    str(tmp_path / f"{stem}.luxar.zarr"),
                    spelling,
                    "cluster",
                ],
            )
            assert old.exit_code != 0
            pointer = normalized_cli_output(old)
            assert "--subst-method cluster" in pointer, pointer
            # A bare "No such option" would also be a non-zero exit, so assert the
            # replacement AND that typer never got to reject the flag itself.
            assert "No such option" not in pointer, pointer
            # The LONG form must not send a mesh user to `--add-method`: that is
            # the gsplat replacement for the bare flag, and mesh's bare `--method`
            # was the substitutive one. Asserted on the normalised text, or Rich's
            # escape codes make the absence unfalsifiable — see `_plain`.
            #
            # NARROWED for `-m` when the reveal recipe claimed that short form
            # (#1498 reserved it for exactly that). `-m` is now a live flag naming
            # the additive ordering, so a message about it necessarily says
            # `--add-method`; what still must hold — and is asserted above for both
            # spellings — is that the user is pointed at `--subst-method` WITH
            # their value, so the pointer stays paste-able.
            if spelling == "--method":
                assert "--add-method" not in pointer, pointer
            assert not (tmp_path / f"{stem}.luxar.zarr").exists()

    def test_overwrite_DOES_replace_an_existing_output(self, tmp_path: Path) -> None:
        # The twin of the test above, and the one that keeps the deletion honest:
        # with only the refusal path covered, `--overwrite` could stop
        # overwriting altogether and every test would still pass.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        out.mkdir()
        (out / "stale.txt").write_text("the previous output")

        counts = _run(run_lod, source, out, overwrite=True)
        assert len(counts) >= 2
        assert not (out / "stale.txt").exists(), "the old store must be gone"
        assert LuxarScene.load(out).get_node_metadata("surf")["kind"] == "lod"

    def test_a_non_builtin_colormap_survives_as_its_LUT(self, tmp_path: Path) -> None:
        """`colormap='custom'` on disk is a SENTINEL, not a name.

        The writer resolves anything outside the ~20 builtin names — an ndarray
        LUT, but also a plain matplotlib/colorcet name like 'magma' — to a
        `colormap_lut` dataset plus that word. Forwarding the word reached the
        resolver as a name and raised "Unknown colormap 'custom'", so a magma
        mesh could not be laddered at all; accepting it would have been worse,
        silently swapping the author's palette for the default. The existing
        appearance test uses 'viridis', a builtin, which never takes this path.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(
            source,
            colormap="magma",
            scalars=_grid_mesh()[0][:, 0].copy(),
        )
        loaded_source = LuxarScene.load(source)
        assert loaded_source.get_node_metadata("surf")["colormap"] == "custom"
        source_lut = loaded_source.get_colormap_lut("surf")
        assert source_lut is not None and source_lut.shape == (256, 3)

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        written = LuxarScene.load(out)
        children = [m for m in written.list_meshes() if m.startswith("surf/child_")]
        assert len(children) >= 2
        for child in children:
            assert written.get_node_metadata(child)["colormap"] == "custom"
            child_lut = written.get_colormap_lut(child)
            assert child_lut is not None
            np.testing.assert_array_equal(child_lut, source_lut)

    def test_a_widened_scalar_window_is_forwarded_not_recomputed(
        self, tmp_path: Path
    ) -> None:
        """Re-laddering a node whose stamp is wider than its values recoloured it.

        `scalar_data_range` is the window the viewer normalizes the colormap
        against, and it is not always the min/max of the values stored under it:
        a ladder stamps ONE shared window on every child, so a coarse child's
        stamp spans the whole field while its own cluster-averaged values are
        contracted. Recomputing from the decoded values there rewrote the window
        onto the narrower span and the surface came back in different colours.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        # Authored deliberately wider than the values, which is exactly the shape
        # a ladder child has on disk — and reachable without building one first.
        scalars = _grid_mesh()[0][:, 0].copy()
        window = (-10.0, 10.0)
        _write_source(
            source,
            colormap="viridis",
            scalars=scalars,
            _scalar_data_range=window,
        )
        loaded_source = LuxarScene.load(source)
        assert loaded_source.get_node_metadata("surf")["scalar_data_range"] == list(
            window
        )

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        written = LuxarScene.load(out)
        children = [m for m in written.list_meshes() if m.startswith("surf/child_")]
        assert len(children) >= 2
        for child in children:
            stamped = written.get_node_metadata(child)["scalar_data_range"]
            assert stamped == pytest.approx(list(window)), (
                f"{child} re-windowed the colormap onto its own values "
                f"({stamped}) instead of keeping the authored window {window}"
            )

    def test_the_scene_viewer_config_comes_across(self, tmp_path: Path) -> None:
        """Dropping it undoes the LUT fidelity above on the scenes that have one.

        With no `tone_mapping` the viewer applies its ACES default, which shifts
        hues by design — and the compiler re-emits the "can distort LUT colors"
        notice as it writes. `luxar mesh import` states ACES explicitly, so the
        documented import → lod pipeline lost it too.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod
        from luxar.core.viewer_config import ViewerConfig

        source = tmp_path / "src.luxar.zarr"
        _write_source(source, viewer_config=ViewerConfig(tone_mapping="Neutral"))
        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2
        carried = LuxarScene.load(out).viewer_config
        assert carried is not None and carried.tone_mapping == "Neutral"

    def test_a_source_with_no_viewer_config_gets_the_explicit_ACES_default(
        self, tmp_path: Path
    ) -> None:
        # The same fallback `luxar mesh import` writes, for the same reason: ACES
        # is the house default, and saying so keeps the compiler's
        # "nothing was chosen" tone-mapping notice quiet.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2
        carried = LuxarScene.load(out).viewer_config
        assert carried is not None and carried.tone_mapping == "ACES"

    def test_an_ordinary_single_mesh_scene_warns_about_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The baseline: one unlabelled mesh, no siblings, nothing to warn about.

        Without this, the drop warnings below could be firing on every run
        (false alarm) and the other two tests would still pass.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "Not carried into the new scene" not in stdout
        assert "has no field for them" not in stdout

    def test_a_sibling_points_node_is_named_as_dropped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Issue #1357: every OTHER node in the source scene is left out.

        A `dots` points node next to the mesh `surf` used to simply disappear
        from the output with nothing in the console saying so.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces)
            scene.add_points("dots", np.zeros((3, 3), dtype=np.float32))

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "Not carried into the new scene" in stdout
        assert "'dots' (points)" in stdout
        # The picked mesh survived — only its sibling should be reported.
        assert "'surf' (mesh)" not in stdout

    def test_keyed_mesh_names_the_dropped_keys_channel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`keys` (#1917) is dropped by this round trip for exactly the same
        reason as `labels` — `MeshData` has no field for it — and a keyed mesh
        losing every link target in silence is the worse failure, because
        nothing downstream looks wrong: the surface renders, it just no longer
        goes anywhere when clicked.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces, keys=["k"] * vertices.shape[0])

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'surf' has per-vertex keys" in stdout
        assert "has no field for them" in stdout

    def test_keys_alone_do_not_claim_a_dropped_hover_overlay(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Keys must not be folded into the LABEL channel list.

        That list also decides whether the source had an auto-injected hover
        overlay, and finalize injects one from labels alone — never from keys.
        A keys-only mesh therefore has no overlay, and treating it as labelled
        would suppress a genuine dropped-group report on some future scene.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces, keys=["k"] * vertices.shape[0])

        out = tmp_path / "out.luxar.zarr"
        _run(run_lod, source, out)

        stdout = capsys.readouterr().out
        assert "'surf' has per-vertex keys" in stdout
        # No labels were written, so no `__hover_text` overlay exists to lose.
        assert "__hover_text" not in stdout
        assert "per-vertex labels" not in stdout

    def test_labelled_mesh_names_the_dropped_label_channel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The other silent drop from #1357: per-vertex `labels`.

        `MeshData` has no field for them, so the round trip cannot carry what
        it cannot read — but nothing used to tell the user that happened.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces, labels=["v"] * vertices.shape[0])

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'surf' has per-vertex labels" in stdout
        assert "has no field for them" in stdout
        # Labels auto-inject a hover overlay at finalize (`__hover_text`),
        # whose container `overlays` has no `type` attr and lands in
        # `list_groups()`. That container is not genuine dropped content —
        # the labels warning above already covers the loss — so it must not
        # ALSO show up as a bogus dropped group. (The compiler's own
        # "Created group: overlays/__hover_text" diagnostic, from writing the
        # SOURCE scene, is unrelated and legitimately present.)
        assert "'overlays' (group)" not in stdout
        assert "(overlay)" not in stdout

    def test_textured_mesh_names_and_safely_drops_texture_data(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Texture sampling attrs must not strand an otherwise valid rewrite."""
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, _faces = _grid_mesh()
        uvs = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        texture = np.zeros((2, 2, 3), dtype=np.uint8)
        _write_source(
            source,
            uvs=uvs,
            texture=texture,
            texture_wrap="clamp",
            texture_filter="nearest",
        )

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'surf' has a texture and per-vertex UV coordinates" in stdout
        assert "will NOT be carried into the new scene" in stdout
        written = LuxarScene.load(out)
        for mesh_path in written.list_meshes():
            mesh = written.get_mesh(mesh_path)
            assert mesh.uvs is None
            assert mesh.texture is None

    def test_a_mesh_nested_under_a_transformed_group_names_that_group(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 2/3: an ANCESTOR group's own attrs are lost too —
        but the message must name EXACTLY what is lost, no more.

        `run_lod` forwards only the picked mesh LEAF's `data.metadata` — an
        ancestor group's `transform`/`opacity`/… never reach it, exactly the
        way `COMPOSITING_ATTRS` land on the `kind=lod` WRAPPER group rather
        than its children (see the comment a few lines above in
        `lod_commands.py`). A mesh authored at `surfaces/skull` under a group
        translated off-origin at reduced opacity used to come back at the
        origin, fully opaque, with nothing saying so. Round 3's fix: `surfaces`
        here carries ONLY `opacity`/`transform` (a plain `add_group` does not
        auto-stamp the neutral defaults a mesh LEAF does), so the message must
        name exactly those two keys — no `absorption`/`gamma`/`intensity`/
        `offset` noise.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod
        from luxar.core.transforms import translate

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group(
                "surfaces", transform=translate(100.0, 0.0, 0.0), opacity=0.25
            )
            group.add_mesh("skull", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        # Narrowed to the ANCESTOR WARNING LINE itself (not the whole
        # compile's stdout): a bare substring check over everything printed
        # would also pass on an unrelated future line that happens to
        # mention "gamma" or "offset", and — since a plain `add_group` never
        # stores the four neutral-default keys either way — could never
        # actually catch a regression here; the neutral-default LOGIC itself
        # is pinned by `test_run_lod_chained_on_its_own_output_prints_no_ancestor_warning`
        # below, which does exercise a wrapper that carries them.
        ancestor_lines = [
            line for line in stdout.splitlines() if "is nested under group" in line
        ]
        assert len(ancestor_lines) == 1
        assert "'surfaces'" in ancestor_lines[0]
        assert "which sets opacity, transform;" in ancestor_lines[0]
        for noise in ("absorption", "gamma", "intensity", "offset"):
            assert noise not in ancestor_lines[0]

    def test_run_lod_chained_on_its_own_output_prints_no_ancestor_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 3: presence alone over-reports on THIS command's
        own output.

        `run_lod` forwards the picked mesh leaf's stored `opacity`/
        `absorption`/`gamma`/`intensity`/`offset` — neutral defaults though
        they are (1.0/1.0/1.0/1.0/0.0) — and the adder splits
        `COMPOSITING_ATTRS` onto the `kind=lod` wrapper it builds. So a
        wrapper THIS COMMAND itself just wrote carries all five, at values
        bit-identical to the leaf's — a key-presence-only gate would print
        "which sets absorption, gamma, intensity, offset, opacity" for a
        chained re-ladder that loses nothing at all.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        ladder = tmp_path / "ladder.luxar.zarr"
        assert len(_run(run_lod, source, ladder)) >= 2
        capsys.readouterr()  # discard the first run's own output

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=ladder,
            output_path=out,
            node_name="surf/child_0",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 1

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_a_plain_ancestor_group_prints_no_compositing_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 2: gate the ancestor warning on an ACTUAL loss.

        A bare namespace group — no `transform`, no `opacity`, nothing in
        `COMPOSITING_ATTRS` — has nothing for the picked mesh to inherit, so
        nesting under one must not print a "placement is lost" warning that
        would be false.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces")
            group.add_mesh("skull", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_re_laddering_an_existing_ladder_level_prints_no_false_alarm(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--node surf/child_0` — this file's own supported re-lod workflow.

        The source here is built the way a fitting pipeline (or a direct
        ``add_mesh(substitutive_lod=…)`` call) actually produces a `kind=lod`
        wrapper: no compositing kwarg passed at all, so `surf` carries only
        structural bookkeeping (`kind`, `child_index`, `content_hash`,
        `selector`, `default_level`, `position_bounds`, …) — none of it a
        `COMPOSITING_ATTRS` member — and re-laddering one of its levels must
        not print a false "placement/compositing is lost" warning.

        (Deliberately NOT built by running `run_lod` twice: `run_lod` itself
        forwards the picked mesh's stored `opacity`/`absorption`/`gamma`/
        `intensity`/`offset` — defaults though they are — as explicit
        `add_mesh` kwargs, which the wrapper-splitting logic then DOES stamp
        onto its own `kind=lod` wrapper. That is a real, separate property of
        this round trip, not the false alarm this test is pinning.)
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "surf",
                vertices,
                faces,
                substitutive_lod={
                    "levels": 2,
                    "compression_factor": 4,
                    "method": "auto",
                },
            )

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surf/child_0",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 1

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_an_ancestor_identity_transform_prints_no_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 4: the identity-`transform` SKIP branch, pinned.

        Every other ancestor test either has no `transform` key at all, or a
        genuinely non-identity one — neither exercises the branch that
        decides an authored-but-no-op `transform` is not a real loss. Making
        `_is_identity_transform` always return `False` (or deleting the
        branch) would leave every other test green while this one catches
        it.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod
        from luxar.core.transforms import identity

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces", transform=identity())
            group.add_mesh("skull", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def _four_d_dimensions(self):
        from luxar import Dimension, Dimensions

        return Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
            ]
        )

    def test_an_ancestor_real_nd_transform_names_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 4: the `nd_transform` REAL-loss branch, pinned.

        No other test authors an `nd_transform` at all, so
        `_is_identity_nd_transform` is never even called — making it always
        return `True` (or deleting the whole `nd_transform` branch) leaves
        every other test green while an ancestor's real nD placement is
        dropped in complete silence. A 4-D scene is needed: `nd_transform`
        names a real non-displayed dimension ("Time" here).
        """
        from luxar import LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        time_col = np.full((vertices.shape[0], 1), 5.0, dtype=np.float32)
        vertices_4d = np.hstack([vertices, time_col]).astype(np.float32)
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=self._four_d_dimensions())
            group = scene.add_group(
                "surfaces", nd_transform={"Time": {"scale": 2.0, "offset": 10.0}}
            )
            group.add_mesh("skull", vertices_4d, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        ancestor_lines = [
            line for line in stdout.splitlines() if "is nested under group" in line
        ]
        assert len(ancestor_lines) == 1
        assert "nd_transform" in ancestor_lines[0]

    def test_an_ancestor_identity_nd_transform_prints_no_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The no-op twin of the test above: a no-op affine entry is not a
        real loss either."""
        from luxar import LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        time_col = np.full((vertices.shape[0], 1), 5.0, dtype=np.float32)
        vertices_4d = np.hstack([vertices, time_col]).astype(np.float32)
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=self._four_d_dimensions())
            group = scene.add_group(
                "surfaces", nd_transform={"Time": {"scale": 1.0, "offset": 0.0}}
            )
            group.add_mesh("skull", vertices_4d, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_a_leaf_overriding_blending_mode_suppresses_the_ancestor_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 4: the `_LEAF_OVERRIDES_ANCESTOR` skip, pinned.

        No other test sets `blending_mode` on BOTH the ancestor and the
        picked leaf, so this branch (nearest-setter-wins: the leaf's own
        value travels regardless, so the ancestor's is not actually lost)
        is never reached by the existing suite.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces", blending_mode="additive")
            group.add_mesh("skull", vertices, faces, blending_mode="normal")

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_an_ancestor_join_is_always_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 4: `join` is lines-only, so it can never be a mesh's
        loss — probed with `add_group('surfaces', join='miter')`.

        `add_mesh` refuses `join` outright (`reject_lines_only_join`), so a
        mesh leaf can never carry it, and the old
        `key in _LEAF_OVERRIDES_ANCESTOR` skip (which only fires when the
        LEAF also sets the key) could never apply to it — it used to print
        a false "which sets join" warning for an attribute that changes
        nothing about a mesh ladder.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces", join="miter")
            group.add_mesh("skull", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="surfaces/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "is nested under group" not in stdout

    def test_a_user_authored_overlay_is_named_as_dropped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """All four overlay types are reported before the rewrite drops them.

        `overlays` itself is not a user node (`Scene` refuses to let anyone
        create one by that name) and must not be reported; a genuine overlay
        living under it — text, HTML, image, or video — is real content and
        must be named.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces)
            scene.add_text("hello", (0.05, 0.05), name="caption")
            scene.add_html("<b>hi</b>", (0.1, 0.1), name="note")
            scene.add_image(
                np.zeros((2, 2, 3), dtype=np.uint8), (0.9, 0.05), name="logo"
            )
            scene.add_video(
                b"\x1a\x45\xdf\xa3" + b"\x00" * 60,
                (0.15, 0.15),
                name="turntable",
            )

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "Not carried into the new scene" in stdout
        assert "'overlays/caption' (overlay)" in stdout
        assert "'overlays/note' (overlay)" in stdout
        assert "'overlays/logo' (overlay)" in stdout
        assert "'overlays/turntable' (overlay)" in stdout
        # The reserved container itself is not a dropped node.
        assert "'overlays' (group)" not in stdout

    def test_a_user_overlay_named_like_the_hover_overlay_is_still_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 3: the hover-overlay skip must be gated on
        PROVENANCE, not name alone.

        `next_overlay_name` does not reserve `__hover_text` /
        `__hover_image` — a user is free to name their own overlay that,
        and on an UNLABELLED scene (so no hover overlay is even
        auto-injected) a name-only check drops it with no warning at all.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces)  # unlabelled: no auto-inject
            scene.add_text("hi", (0.1, 0.1), name="__hover_text")

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'overlays/__hover_text' (overlay)" in stdout

    def test_a_hover_true_overlay_on_an_unlabelled_mesh_is_still_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 4: the round-3 fix still had the hole open.

        `hover` is a PUBLIC kwarg of `Scene.add_text`, so a user overlay
        named `__hover_text` WITH `hover=True` satisfies both the name and
        the provenance check round 3 added — on an UNLABELLED mesh (so no
        hover overlay is auto-injected, and the per-vertex-label warning
        that is the skip's whole justification never fires either), that
        combination used to vanish with no warning of any kind.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces)  # unlabelled: no auto-inject
            scene.add_text("hi", (0.1, 0.1), name="__hover_text", hover=True)

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'overlays/__hover_text' (overlay)" in stdout

    def test_a_hover_true_overlay_on_a_LABELLED_mesh_is_still_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 5: the labelled scene is the case the name+`hover`
        pair cannot decide either.

        `auto_inject_hover_overlay` bails the moment ANY overlay already sets
        `hover` — so on a labelled scene a user's own
        `add_text(..., name='__hover_text', hover=True)` is the ONLY hover
        overlay in the store, no injection ever happened, and the label
        warning does not cover it. Provenance is the injected PLACEHOLDER
        payload (`{hover_label}`), which this overlay does not carry.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces, labels=["v"] * vertices.shape[0])
            scene.add_text("mine", (0.1, 0.1), name="__hover_text", hover=True)

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "'overlays/__hover_text' (overlay)" in stdout
        # …and the label channel is still reported in its own right.
        assert "'surf' has per-vertex labels" in stdout

    def test_only_the_NEAREST_ancestor_blending_mode_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 5: nearest-setter-wins applies between ancestors too.

        Under `outer(additive) / inner(normal) / mesh` the viewer's composer
        gives the mesh `inner`'s mode — `outer`'s was ALREADY shadowed in the
        SOURCE scene, so the rewrite does not lose it. Evaluating each
        ancestor in isolation named both groups.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            outer = scene.add_group("outer", blending_mode="additive")
            inner = outer.add_group("inner", blending_mode="normal")
            inner.add_mesh("skull", vertices, faces)

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="outer/inner/skull",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        ancestor_lines = [
            line for line in stdout.splitlines() if "is nested under group" in line
        ]
        assert len(ancestor_lines) == 1
        assert "'outer/inner'" in ancestor_lines[0]
        assert "which sets blending_mode;" in ancestor_lines[0]

    def test_ancestor_layer_and_visible_are_judged_against_their_defaults(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 5: `layer`/`visible` have defaults too.

        `Node.layer` reads False when absent and `Node.visible` reads True, so
        a group authoring those exact values is a no-op for the picked mesh —
        the same false alarm the neutral-scalar gate exists to prevent. The
        other value IS a real loss and must still be named.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod
        from luxar.io.reader import LuxarScene

        neutral = tmp_path / "neutral.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(neutral) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces", layer=False, visible=True)
            group.add_mesh("skull", vertices, faces)
        # Non-vacuous: both keys really are on the stored group.
        stored = LuxarScene.load(neutral).get_node_metadata("surfaces")
        assert stored["layer"] is False and stored["visible"] is True

        assert (
            len(
                run_lod(
                    input_path=neutral,
                    output_path=tmp_path / "a.luxar.zarr",
                    node_name="surfaces/skull",
                    levels=2,
                    compression_factor=4,
                    method="auto",
                    overwrite=False,
                )
            )
            >= 2
        )
        assert "is nested under group" not in capsys.readouterr().out

        real = tmp_path / "real.luxar.zarr"
        with LuxarZarrCompiler(real) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surfaces", layer=True, visible=False)
            group.add_mesh("skull", vertices, faces)

        assert (
            len(
                run_lod(
                    input_path=real,
                    output_path=tmp_path / "b.luxar.zarr",
                    node_name="surfaces/skull",
                    levels=2,
                    compression_factor=4,
                    method="auto",
                    overwrite=False,
                )
            )
            >= 2
        )
        ancestor_lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if "is nested under group" in line
        ]
        assert len(ancestor_lines) == 1
        assert "which sets layer, visible;" in ancestor_lines[0]

    def test_an_aborted_run_prints_no_drop_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 2: the drop report must sit AFTER the validators.

        An invalid method aborts before `add_mesh` is ever called and before
        anything is written; the drop warnings, printed earlier in the
        function, used to fire anyway and announce drops that never
        happened.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("surf", vertices, faces)
            scene.add_points("dots", np.zeros((3, 3), dtype=np.float32))

        out = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match="method"):
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=2,
                compression_factor=4,
                method="bogus",
                overwrite=False,
            )

        stdout = capsys.readouterr().out
        assert "Not carried into the new scene" not in stdout

    def test_many_siblings_collapse_into_one_counted_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 2/3: a wall of one-line-per-sibling doesn't scale —
        and the collapsed wording must not overclaim "parts" of anything.

        A large spatial partition can have hundreds of parts, and `_pick_mesh`
        forces `--node` once a scene has more than one mesh — so re-laddering
        one part used to print one `aprint` per OTHER sibling. Four points
        clouds under a 'stuff' group stand in for hundreds here; the point is
        the collapse threshold (more than three sharing a parent + kind), not
        the specific count — and "parts" is Luxar's term of art for
        `kind=partition` children specifically, which these are not, so
        round 3 reworded the collapsed line to something accurate in every
        case.
        """
        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        vertices, faces = _grid_mesh()
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("cover", vertices, faces)
            group = scene.add_group("stuff")
            for i in range(4):
                group.add_points(f"cloud_{i}", np.zeros((3, 3), dtype=np.float32))

        out = tmp_path / "out.luxar.zarr"
        counts = run_lod(
            input_path=source,
            output_path=out,
            node_name="cover",
            levels=2,
            compression_factor=4,
            method="auto",
            overwrite=False,
        )
        assert len(counts) >= 2

        stdout = capsys.readouterr().out
        assert "4 other points nodes in 'stuff'" in stdout
        assert "parts" not in stdout
        # Collapsed, not itemized: none of the four clouds gets its own line.
        for i in range(4):
            assert f"'stuff/cloud_{i}'" not in stdout


class TestMeshLodOutputPaths:
    """The guards must test the path the COMPILER writes to, not the raw argument.

    `LuxarZarrCompiler` normalizes its store path to `<stem>.luxar.zarr`, so every
    guard that compared the raw `--output` guarded a path nothing writes to. The
    first test below is the benign shape of that (the ladder was written but the
    read-back looked for it under the un-normalized name and raised
    `FileNotFoundError: Scene not found`); the rest are demonstrated data loss —
    the source rewritten, an existing store replaced with no `--overwrite`, or a
    scene written into the input's own tree.
    """

    def test_an_unsuffixed_output_lands_in_the_normalized_store(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        counts = _run(run_lod, source, tmp_path / "bare")
        assert len(counts) >= 2
        assert (tmp_path / "bare.luxar.zarr").is_dir()
        assert LuxarScene.load(source).get_mesh("surf").vertices.shape[0] > 0

    def test_an_output_that_normalizes_ONTO_the_input_is_refused(
        self, tmp_path: Path
    ) -> None:
        # `--output scene` next to `scene.luxar.zarr` normalizes to the input
        # store: the raw comparison passed and the compiler rewrote the source.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "scene.luxar.zarr"
        _write_source(source)
        with pytest.raises(ValueError, match="input scene itself"):
            _run(run_lod, source, tmp_path / "scene", overwrite=True)
        assert LuxarScene.load(source).get_mesh("surf").faces.shape[0] > 0

    def test_an_existing_normalized_output_needs_overwrite(
        self, tmp_path: Path
    ) -> None:
        # `--output out` with an existing `out.luxar.zarr`: the exists() check
        # looked at `out`, so the store was replaced with no `--overwrite`.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        existing = tmp_path / "out.luxar.zarr"
        existing.mkdir()
        (existing / "keepme.txt").write_text("previous output")

        with pytest.raises(FileExistsError):
            _run(run_lod, source, tmp_path / "out")
        assert (existing / "keepme.txt").read_text() == "previous output"

    def test_an_output_directory_CONTAINING_the_input_is_refused(
        self, tmp_path: Path
    ) -> None:
        # `rmtree` on a directory the input lives in takes the source with it.
        # The other half of the guard `luxar mesh import` already carries, and
        # reachable here BECAUSE of the normalization: `--output holder` resolves
        # to `holder.luxar.zarr`, which is exactly the directory holding the
        # input scene.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        holder = tmp_path / "holder.luxar.zarr"
        holder.mkdir()
        source = holder / "in.luxar.zarr"
        _write_source(source)
        with pytest.raises(ValueError, match="containing it"):
            _run(run_lod, source, tmp_path / "holder", overwrite=True)
        assert LuxarScene.load(source).get_mesh("surf").faces.shape[0] > 0

    def test_an_output_INSIDE_the_input_store_is_refused(self, tmp_path: Path) -> None:
        # The other direction, and the one the message already promised to
        # cover ("outside the input's tree"): a destination under the source
        # store passed the one-way check and wrote a whole nested scene into it,
        # leaving the source with a stray `nested.luxar.zarr` group inside.
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        with pytest.raises(ValueError, match="inside it"):
            _run(run_lod, source, source / "nested", overwrite=True)
        assert not (source / "nested.luxar.zarr").exists()
        assert LuxarScene.load(source).get_mesh("surf").faces.shape[0] > 0

    def test_an_unrecognized_recipe_is_refused_before_the_output_is_deleted(
        self, tmp_path: Path
    ) -> None:
        """`--recipe` is validated at the CLI, but `run_lod` is directly callable.

        A misspelled recipe used to fall through to the substitutive arm, so
        `recipe="reveaal"` silently wrote a DECIMATED ladder where a reveal was
        asked for — a different product, not a near miss.

        The output-preserved half is a REGRESSION pin, not a bug being fixed:
        specs are already built ~100 lines before the `--overwrite` `rmtree`, so
        no store was ever lost to this. It is asserted because that ordering is
        what keeps the raise cheap, and nothing else pins it.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        before = _run(run_lod, source, out)

        with pytest.raises(ValueError, match="recipe must be one of"):
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=2,
                compression_factor=4,
                method="auto",
                overwrite=True,
                recipe="reveaal",
            )
        # Still the SUBSTITUTIVE ladder the first run wrote — not deleted, and not
        # replaced by the reveal's shape. Asserted on the store, since a wrong
        # product here is precisely a store with the other topology in it.
        assert out.is_dir()
        group = zarr.open_group(str(out), mode="r")["surf"]
        assert group.attrs["kind"] == "lod"
        assert "n_additive_sublods" not in group.attrs
        assert len(before) == len([k for k in group.keys() if k.startswith("child_")])


def test_every_method_named_in_the_help_EXAMPLES_is_a_real_method() -> None:
    """A copy-pasteable example must not name a method the command rejects.

    This guards every copy-pasteable `--subst-method` example against drifting
    away from the methods the command actually accepts.

    Derived from the docstring rather than pinning the current text, so the
    example set can grow freely and only an INVALID method fails. The help text is
    the one place a wrong method name costs a user a round trip instead of a type
    error, which is why it gets a test and the prose does not.
    """
    from luxar.cli.mesh_ops.lod_commands import lod_command
    from luxar.core.group.lod.group import MESH_SUBSTITUTIVE_METHODS

    # `lod_command`, not `run_lod`: typer renders help from the COMMAND callback's
    # docstring, and that is where the examples live. Reading the wrong one made
    # the regex match nothing — which the emptiness guard below caught rather
    # than letting `set() - valid` pass as "no invalid methods".
    #
    # `[\s=]+`, not `\s+`: `--subst-method=qem` is the same command line, and with
    # a space-only pattern it slips past a growing examples block unseen (the
    # emptiness guard only catches it while it is the ONLY example).
    doc = lod_command.__doc__ or ""
    named = set(re.findall(r"--subst-method[\s=]+(\S+)", doc))
    assert named, "no --subst-method example found — did the examples block move?"
    invalid = named - set(MESH_SUBSTITUTIVE_METHODS)
    assert not invalid, (
        f"`luxar mesh lod --help` shows --subst-method {sorted(invalid)}, which the "
        f"command rejects (valid: {sorted(MESH_SUBSTITUTIVE_METHODS)})"
    )


class TestMeshLodRevealRecipe:
    """`luxar mesh lod --recipe reveal` — the additive ladder on the CLI.

    The reveal was authorable only from Python until this landed: `-m/--add-method`
    was RESERVED by the #1498 rename and never filled in. These tests are about the
    command surface — that the knobs reach the written store, and that a knob aimed
    at the wrong recipe is refused rather than dropped. What a reveal IS (contiguous
    prefixes, interleaved stacked timepoints) is pinned deterministically in
    `core/tests/test_mesh.py`, and is not re-litigated here.
    """

    @staticmethod
    def _reveal(
        run_lod: Callable[..., list[int]],
        source: Path,
        out: Path,
        **knobs: object,
    ) -> list[int]:
        """`run_lod` on the reveal recipe. Returns the per-level FACE counts."""
        return run_lod(
            input_path=source,
            output_path=out,
            node_name=None,
            levels=3,
            compression_factor=4,
            method="auto",
            overwrite=False,
            recipe="reveal",
            **knobs,
        )

    def test_the_default_recipe_still_writes_a_substitutive_ladder(
        self, tmp_path: Path
    ) -> None:
        """No `--recipe` must behave exactly as before this command grew one.

        The whole design rests on `levels` being the default, so every existing
        script and every doc example keeps working untouched. Asserted on the
        STORE — a `kind=lod` group and no `n_additive_sublods` anywhere — rather
        than on the return value, because the return value is the one thing a
        recipe branch could get right while writing the wrong shape.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        _run(run_lod, source, out)

        group = zarr.open_group(str(out), mode="r")["surf"]
        assert group.attrs["kind"] == "lod"
        assert "n_additive_sublods" not in group.attrs

    def test_reveal_writes_one_node_whose_levels_PARTITION_the_faces(
        self, tmp_path: Path
    ) -> None:
        """A reveal is one node with `additive_<i>` subgroups, not N sibling nodes.

        The face SUM is the load-bearing half: the levels are disjoint groups the
        viewer concatenates, so they must sum to the source's face count exactly.
        A ladder that duplicated faces across levels — or dropped some — would
        still load and still look plausible at the finest level.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _, faces = _write_source(source)
        out = tmp_path / "out.luxar.zarr"

        level_faces = self._reveal(run_lod, source, out, n_lods=4)

        group = zarr.open_group(str(out), mode="r")["surf"]
        assert group.attrs["n_additive_sublods"] == 4
        assert group.attrs.get("kind") is None, "a reveal lives INSIDE the leaf"
        assert len(level_faces) == 4
        assert sum(level_faces) == int(faces.shape[0])
        assert all(count > 0 for count in level_faces)

    def test_n_lods_reaches_the_written_ladder(self, tmp_path: Path) -> None:
        """The knob must change the STORE, not merely be accepted.

        Threading is what breaks — a flag parsed, validated and then dropped
        before the adder is the failure this whole command surface risks — so the
        assertion is on the level count on disk.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)

        for wanted in (2, 6):
            out = tmp_path / f"out{wanted}.luxar.zarr"
            level_faces = self._reveal(run_lod, source, out, n_lods=wanted)
            group = zarr.open_group(str(out), mode="r")["surf"]
            assert group.attrs["n_additive_sublods"] == wanted
            assert len(level_faces) == wanted

    def test_counts_are_read_as_CUMULATIVE_boundaries(self, tmp_path: Path) -> None:
        """`--counts a,b,c` gives four levels sized a, b-a, c-b, rest.

        The alias for the resolver's `counts` key, which takes cumulative CUT
        positions. Read as per-level increments instead, the same string yields
        different level sizes and a short final level — the exact defect #1506
        fixed one layer down. Pinned here too because the CLI is a second entry
        point into that vocabulary.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _, faces = _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        total = int(faces.shape[0])

        level_faces = self._reveal(run_lod, source, out, counts="100,300,600")

        assert level_faces == [100, 200, 300, total - 600]
        assert sum(level_faces) == total

    def test_reveal_center_changes_which_faces_land_in_the_first_level(
        self, tmp_path: Path
    ) -> None:
        """The centre must reach the ordering, not just the argument parser.

        SENSITIVITY IS THE HARD PART HERE. The reveal grows outward from a centre,
        so two centres only produce different ladders when the surface is
        ASYMMETRIC about them — on a mesh symmetric between the two, the distance
        field is a relabelling and the first level can come out identical, which
        would make this pass against a `--reveal-center` that was parsed and
        thrown away. The grid spans [-1, 1]², so the two corners below are maximally
        far apart and the first level around each is a different corner's faces.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)

        def first_level_centroid(tag: str, centre: str) -> tuple[float, float]:
            out = tmp_path / f"out_{tag}.luxar.zarr"
            self._reveal(
                run_lod,
                source,
                out,
                n_lods=4,
                reveal_center=centre,
                spatial_dims="0,1",
            )
            level = zarr.open_group(str(out), mode="r")["surf/additive_0"]
            # The level's own vertex coordinates identify it; face indices are
            # local to each level's gathered table and so cannot be compared.
            #
            # Its MEAN position, not its extremes: both shells reach inward to
            # roughly the middle of the grid, so `min`/`max` over a level are
            # nearly equal and say nothing about which corner it grew from. The
            # centroid does.
            verts = np.asarray(level["vertices"][:])
            return float(verts[:, 0].mean()), float(verts[:, 1].mean())

        near_min = first_level_centroid("min", "-1,-1")
        near_max = first_level_centroid("max", "1,1")

        # SENSITIVITY: the two must differ at all. If they do not, the flag is
        # being parsed and dropped — or the fixture is symmetric about the two
        # centres, in which case this test proves nothing and the fixture is the
        # thing to fix.
        assert near_min != near_max, (
            "the two centres produced the same first level — --reveal-center is "
            "being parsed and dropped, or this fixture is symmetric about them"
        )
        # And the DIRECTION is right, not merely different: the shell grown from
        # the (-1,-1) corner must sit further toward it on both axes than the one
        # grown from (1,1). Difference alone would also pass for a centre that
        # perturbed the order arbitrarily.
        assert near_min[0] < near_max[0]
        assert near_min[1] < near_max[1]

    @pytest.mark.parametrize(
        ("knobs", "expect_typed", "expect_instead"),
        [
            ({"n_lods": 4}, "--n-lods", "-L/--levels"),
            ({"add_method": "radial"}, "-m/--add-method", "--subst-method"),
            ({"reveal_center": "0,0,0"}, "--reveal-center", "no equivalent"),
            ({"spatial_dims": "0,1,2"}, "--spatial-dims", "no equivalent"),
            ({"counts": "10,20"}, "--counts/--breakpoints", "no equivalent"),
        ],
    )
    def test_a_reveal_knob_under_recipe_levels_is_REFUSED_naming_both_flags(
        self,
        tmp_path: Path,
        knobs: dict,
        expect_typed: str,
        expect_instead: str,
    ) -> None:
        """Refused, not silently dropped — and the message names both flags.

        A user who passes `--n-lods` has said what they want. Dropping it and
        writing a 3-level decimation answers a different question without saying
        so, which is the failure mode the #1498 rename existed to remove from this
        flag surface. Naming the equivalent flag is what makes the error
        actionable rather than merely correct.
        """
        from luxar.cli.mesh_ops.lod_commands import _reject_cross_recipe_flags

        with pytest.raises(typer.BadParameter) as excinfo:
            _reject_cross_recipe_flags(
                recipe="levels",
                add_method=knobs.get("add_method"),
                n_lods=knobs.get("n_lods"),
                counts=knobs.get("counts"),
                reveal_center=knobs.get("reveal_center"),
                spatial_dims=knobs.get("spatial_dims"),
                levels_given=False,
                compression_given=False,
                subst_method_given=False,
            )
        message = _plain(str(excinfo.value))
        assert expect_typed in message
        assert expect_instead in message

    @pytest.mark.parametrize(
        ("given", "expect_typed", "expect_instead"),
        [
            ({"levels_given": True}, "-L/--levels", "--n-lods"),
            ({"subst_method_given": True}, "--subst-method", "-m/--add-method"),
            ({"compression_given": True}, "-K/--compression-factor", "no equivalent"),
        ],
    )
    def test_a_substitutive_knob_under_recipe_reveal_is_REFUSED(
        self, given: dict, expect_typed: str, expect_instead: str
    ) -> None:
        """The mirror direction, which a one-sided gate would leave open.

        `-K` is the interesting row: a reveal has NO equivalent, because its
        levels are a face partition rather than a reduction, so there is no
        per-level factor to give. The message says that instead of naming a flag
        that does not exist.
        """
        from luxar.cli.mesh_ops.lod_commands import _reject_cross_recipe_flags

        with pytest.raises(typer.BadParameter) as excinfo:
            _reject_cross_recipe_flags(
                recipe="reveal",
                add_method=None,
                n_lods=None,
                counts=None,
                reveal_center=None,
                spatial_dims=None,
                levels_given=given.get("levels_given", False),
                compression_given=given.get("compression_given", False),
                subst_method_given=given.get("subst_method_given", False),
            )
        message = _plain(str(excinfo.value))
        assert expect_typed in message
        assert expect_instead in message

    def test_an_unknown_recipe_is_refused_by_name(self) -> None:
        from luxar.cli.mesh_ops.lod_commands import _reject_cross_recipe_flags

        with pytest.raises(typer.BadParameter, match="--recipe must be one of"):
            _reject_cross_recipe_flags(
                recipe="bogus",
                add_method=None,
                n_lods=None,
                counts=None,
                reveal_center=None,
                spatial_dims=None,
                levels_given=False,
                compression_given=False,
                subst_method_given=False,
            )

    def test_a_decimation_method_passed_to_add_method_points_at_subst_method(
        self, tmp_path: Path
    ) -> None:
        """`-m cluster` — the pre-rename spelling — must still migrate the user.

        `-m` was held by the hidden legacy option while it was RESERVED; claiming
        it for `--add-method` is what the reservation was for. The pointer did not
        disappear with it: `cluster` is a decimation method, so the recipe gate
        catches the combination and names `--subst-method`.
        """
        from luxar.cli.mesh_ops.lod_commands import _reject_cross_recipe_flags

        with pytest.raises(typer.BadParameter) as excinfo:
            _reject_cross_recipe_flags(
                recipe="levels",
                add_method="cluster",
                n_lods=None,
                counts=None,
                reveal_center=None,
                spatial_dims=None,
                levels_given=False,
                compression_given=False,
                subst_method_given=False,
            )
        assert "--subst-method" in _plain(str(excinfo.value))

    def test_n_lods_and_counts_together_are_refused(self) -> None:
        """Both size the ladder, so passing both is ambiguous rather than additive."""
        from luxar.cli.mesh_ops.lod_commands import _reject_cross_recipe_flags

        with pytest.raises(typer.BadParameter, match="both size the ladder"):
            _reject_cross_recipe_flags(
                recipe="reveal",
                add_method=None,
                n_lods=4,
                counts="10,20",
                reveal_center=None,
                spatial_dims=None,
                levels_given=False,
                compression_given=False,
                subst_method_given=False,
            )

    def test_an_invalid_add_method_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """Pre-deletion, like `--subst-method`: a bad method must not cost a store.

        `run_lod` deletes the output under `--overwrite` before it writes, so a
        method rejected only by the adder would be diagnosed with the previous
        store already gone. The existing output below must survive.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        out.mkdir()
        (out / "keep.txt").write_text("the previous output")

        with pytest.raises(typer.BadParameter, match="--add-method must be one of"):
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=3,
                compression_factor=4,
                method="auto",
                overwrite=True,
                recipe="reveal",
                add_method="salience",
            )
        assert (out / "keep.txt").exists(), "the output was deleted before validation"

    @pytest.mark.parametrize(
        ("knob", "value"),
        [
            ("n_lods", 6),
            ("add_method", "radial"),
            ("counts", "100,300"),
            ("reveal_center", "0,0"),
            ("spatial_dims", "0,1"),
        ],
    )
    def test_a_reveal_ARGUMENT_under_recipe_levels_is_refused_by_run_lod_itself(
        self, tmp_path: Path, knob: str, value: object
    ) -> None:
        """The gate one layer below the flag surface.

        `run_lod` is directly callable, and the substitutive arm reads none of the
        five reveal-only arguments — so `run_lod(recipe="levels", n_lods=6)` built
        a 3-level decimation and said nothing, which is precisely the silent drop
        `--recipe` exists to prevent. The CLI cannot reach this (its own gate
        catches the combination first), so nothing else covers it.

        The existing output must survive too: `--overwrite` deletes it before the
        write, so a refusal that arrived later would cost a store.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"
        out.mkdir()
        (out / "keep.txt").write_text("the previous output")

        with pytest.raises(typer.BadParameter) as excinfo:
            run_lod(
                input_path=source,
                output_path=out,
                node_name=None,
                levels=3,
                compression_factor=4,
                method="auto",
                overwrite=True,
                recipe="levels",
                **{knob: value},
            )
        assert "--recipe reveal" in _plain(str(excinfo.value))
        assert (out / "keep.txt").exists(), "the output was deleted before validation"

    def test_SENSITIVITY_recipe_levels_still_runs_with_no_reveal_arguments(
        self, tmp_path: Path
    ) -> None:
        """The control for the gate above: it must refuse the arguments, not the arm.

        A check that raised unconditionally would pass every row above while
        breaking the command's whole default path.
        """
        from luxar.cli.mesh_ops.lod_commands import run_lod

        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"

        assert len(_run(run_lod, source, out)) >= 2


class TestMeshLodRecipeGateThroughTheRealCLI:
    """The cross-recipe gate driven by `CliRunner`, not by synthetic booleans.

    The unit tests above hand `_reject_cross_recipe_flags` its `*_given` flags
    directly, which pins the gate's LOGIC and nothing about whether the command can
    compute them. It could not: they were derived by comparing each value against
    its default, so `--recipe reveal --levels 3` — a levels-only flag whose value
    happens to BE the default — read as "not given" and was silently ignored. The
    gate was correct and unreachable for exactly the user most likely to be
    surprised, the one who spells out a default.

    These go through the real parser so the supply signal is the real one.
    """

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--levels", "3"),
            ("-L", "3"),
            ("--compression-factor", "4"),
            ("-K", "4"),
            ("--subst-method", "auto"),
        ],
    )
    def test_a_levels_flag_at_its_DEFAULT_value_is_still_refused_under_reveal(
        self, tmp_path: Path, flag: str, value: str
    ) -> None:
        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"

        result = runner.invoke(
            app,
            ["mesh", "lod", str(source), str(out), "--recipe", "reveal", flag, value],
        )

        assert result.exit_code != 0, (
            f"{flag} {value} was accepted under --recipe reveal; the gate is "
            "inferring 'given' from the value again"
        )
        message = normalized_cli_output(result)
        assert "--recipe levels" in message, message
        assert not out.exists(), "a refused invocation must write nothing"

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--add-method", "radial"), ("-m", "radial"), ("--n-lods", "4")],
    )
    def test_a_reveal_flag_at_its_DEFAULT_value_is_still_refused_under_levels(
        self, tmp_path: Path, flag: str, value: str
    ) -> None:
        """The mirror. `--n-lods 4` and `-m radial` are the reveal defaults."""
        source = tmp_path / "src.luxar.zarr"
        _write_source(source)
        out = tmp_path / "out.luxar.zarr"

        result = runner.invoke(app, ["mesh", "lod", str(source), str(out), flag, value])

        assert result.exit_code != 0, (
            f"{flag} {value} was accepted under the default recipe"
        )
        message = normalized_cli_output(result)
        assert "--recipe reveal" in message or "--subst-method" in message
        assert not out.exists()

    def test_SENSITIVITY_neither_recipe_refuses_its_OWN_flags_at_defaults(
        self, tmp_path: Path
    ) -> None:
        """The control the two above need.

        A gate that treated every parameter as supplied would pass both of them
        while refusing every legitimate invocation — including one that spells out
        its own recipe's defaults. Both of these must SUCCEED.
        """
        source = tmp_path / "src.luxar.zarr"
        _write_source(source)

        levels_out = tmp_path / "levels.luxar.zarr"
        levels = runner.invoke(
            app,
            ["mesh", "lod", str(source), str(levels_out), "--levels", "3", "-K", "4"],
        )
        assert levels.exit_code == 0, _plain(levels.output)

        reveal_out = tmp_path / "reveal.luxar.zarr"
        reveal = runner.invoke(
            app,
            [
                "mesh",
                "lod",
                str(source),
                str(reveal_out),
                "--recipe",
                "reveal",
                "-m",
                "radial",
                "--n-lods",
                "4",
            ],
        )
        assert reveal.exit_code == 0, _plain(reveal.output)
        parent = zarr.open_group(str(reveal_out), mode="r")["surf"]
        assert parent.attrs["n_additive_sublods"] == 4

    def test_the_help_summary_names_both_ladders(self) -> None:
        """`mesh lod --help`'s one-liner is the command's docstring summary.

        It said "Build a substitutive LOD ladder", which stopped being the whole
        truth the moment `--recipe reveal` existed — and it is the first line a
        user reads.
        """
        result = runner.invoke(app, ["mesh", "lod", "--help"])
        assert result.exit_code == 0
        summary = _plain(result.output)
        assert "reveal" in summary, summary
