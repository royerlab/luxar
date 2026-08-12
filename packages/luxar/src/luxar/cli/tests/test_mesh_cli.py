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

from pathlib import Path
from typing import Callable

import numpy as np
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
        method="auto",
        overwrite=overwrite,
    )


class TestMeshLod:
    """`luxar mesh lod` — a scene's mesh node rewritten as a `kind=lod` ladder."""

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

        So `--method qem` (a real name, not yet a real tier) removed the output
        and only then exited 1, having written nothing in its place.
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
                method="qem",
                overwrite=True,
            )
        assert (out / "keepme.txt").read_text() == "previous output"

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
        """Reviewer round 2: all THREE overlay types, not just `add_text`.

        `overlays` itself is not a user node (`Scene` refuses to let anyone
        create one by that name) and must not be reported; a genuine overlay
        living under it — text, HTML, AND image — is real content and must be
        named. `add_image` was the specific gap: `overlay_image` was missing
        from the type filter, so an image overlay was still silently dropped.
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

        out = tmp_path / "out.luxar.zarr"
        assert len(_run(run_lod, source, out)) >= 2

        stdout = capsys.readouterr().out
        assert "Not carried into the new scene" in stdout
        assert "'overlays/caption' (overlay)" in stdout
        assert "'overlays/note' (overlay)" in stdout
        assert "'overlays/logo' (overlay)" in stdout
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

    def test_an_aborted_run_prints_no_drop_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reviewer round 2: the drop report must sit AFTER the validators.

        `--method qem` aborts before `add_mesh` is ever called and before
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
                method="qem",
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
