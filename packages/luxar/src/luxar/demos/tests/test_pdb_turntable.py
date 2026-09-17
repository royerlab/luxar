"""Unit tests for the PDB turntable helper's pure pieces.

PyMOL is a special-case (conda/Homebrew-only) dependency and the renderer needs
a GPU context, so nothing here runs either. The tests pin what the demo relies
on: the surface-export script, cache keying (colour included), the pastel
palette, the soft gate that lets the demo build without turntables when a tool
is absent, and the per-structure skip.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from luxar.demos import _pdb_turntable as tt


def test_cache_key_changes_with_every_render_parameter() -> None:
    base = tt.cache_key("1omg", 900, 768)
    assert base.startswith("1OMG_v")  # id is upper-cased into the stem
    assert tt.cache_key("1OMG", 900, 768) == base  # case-insensitive
    assert tt.cache_key("1OMG", 720, 768) != base
    assert tt.cache_key("1OMG", 900, 512) != base
    assert tt.cache_key("1OMG", 900, 768, style_version=tt.STYLE_VERSION + 1) != base
    # The colour is part of the key: two stories sharing a PDB id rendered in
    # different colours never collide.
    assert tt.cache_key("1OMG", 900, 768, color=(1, 0, 0)) != tt.cache_key(
        "1OMG", 900, 768, color=(0, 0, 1)
    )
    assert tt.color_hex((1.0, 0.5, 0.0)) == "#ff8000"
    # Meshes are keyed separately so a style change reuses the surfaces.
    assert tt.mesh_dir_name("1omg", 1) == f"1OMG_mesh_v{tt.MESH_VERSION}_q1"
    assert tt.mesh_dir_name("1OMG", 0) != tt.mesh_dir_name("1OMG", 1)


def test_defaults_give_a_slow_30_second_turn() -> None:
    assert tt.DEFAULT_FRAMES / tt.DEFAULT_FPS == 30
    assert tt.DEFAULT_FRAMES == 900  # 0.4 deg per frame
    assert tt.TURN_DIRECTION in (1.0, -1.0)


def test_surface_quality_follows_structure_size() -> None:
    assert tt.surface_quality_for(4_779) == 1  # hemoglobin
    assert tt.surface_quality_for(tt.LARGE_STRUCTURE_ATOMS) == 1
    assert tt.surface_quality_for(54_036) == 0  # photosystem II


def test_pymol_surface_script_exports_one_obj_per_chain() -> None:
    script = tt.pymol_surface_script(Path("/x/1OMG.pdb"), Path("/y/mesh"), quality=1)
    assert "cmd.load('/x/1OMG.pdb', 'mol')" in script
    assert "cmd.remove('solvent')" in script and "cmd.remove('hydro')" in script
    assert "cmd.set('surface_quality', 1)" in script
    # `not solvent`, not `polymer`: see
    # test_surface_script_draws_the_whole_molecule_not_just_the_polymer.
    assert "cmd.get_chains('mol and not solvent')" in script
    # Per-chain surfaces: hide everything, show this chain, save its OBJ.
    assert "cmd.show('surface', \"not solvent and chain '%s'\" % chain)" in script
    assert "cmd.save('/y/mesh' + '/chain_%d.obj' % i)" in script
    assert "'/y/mesh' + '/chains.json'" in script
    # Nothing else is drawn — the surface is what gets exported.
    for absent in ("cartoon", "spheres", "png(", "ray"):
        assert absent not in script, absent


def test_chain_palette_keeps_the_story_hue_but_pins_lightness_and_saturation() -> None:
    import colorsys

    neon_green = (0.4, 0.98, 0.4)  # the PSII story highlight, far too hot for clay
    shades = tt.chain_palette(neon_green, 4)
    assert len(shades) == 4 and len(set(shades)) == 4  # distinct per chain
    h0 = colorsys.rgb_to_hls(*neon_green)[0]
    for r, g, b in shades:
        h, lightness, sat = colorsys.rgb_to_hls(r, g, b)
        assert abs(((h - h0 + 0.5) % 1.0) - 0.5) < 0.04  # within ~15 deg of hue
        assert 0.5 <= lightness <= 0.7 and 0.4 <= sat <= 0.6
    assert tt.chain_palette(neon_green, 1) == [tt.chain_palette(neon_green, 1)[0]]


def test_render_turntables_soft_gates_on_missing_tools(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tt, "find_pymol", lambda: None)
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    assert tt.render_turntables(["1OMG", "2HHB"], tmp_path) == {}
    out = capsys.readouterr().out
    assert "brew install pymol" in out and "conda install" in out

    monkeypatch.setattr(tt, "find_pymol", lambda: ["/usr/bin/pymol", "-cq"])
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: None)
    assert tt.render_turntables(["1OMG"], tmp_path) == {}
    assert "ffmpeg" in capsys.readouterr().out

    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tt, "has_moderngl", lambda: False)
    assert tt.render_turntables(["1OMG"], tmp_path) == {}
    assert "luxar[demos]" in capsys.readouterr().out  # bounded, per INSTALL_SPECS


def test_render_turntables_returns_cached_assets_before_tool_gates(
    monkeypatch, tmp_path
) -> None:
    stem = tt.cache_key("1OMG", tt.DEFAULT_FRAMES, tt.DEFAULT_SIZE)
    webm = tmp_path / f"{stem}.webm"
    poster = tmp_path / f"{stem}.png"
    webm.write_bytes(b"cached video")
    poster.write_bytes(b"cached poster")
    (tmp_path / "1OMG.json").write_text('{"title": "Title 1OMG"}')

    monkeypatch.setattr(tt, "find_pymol", lambda: None)
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: None)

    assets = tt.render_turntables(["1OMG", "2HHB"], tmp_path)

    assert set(assets) == {"1OMG"}
    assert assets["1OMG"] == tt.TurntableAssets(
        "1OMG", webm, poster, "Title 1OMG", tt.DEFAULT_FRAMES, tt.DEFAULT_FPS
    )


def test_render_turntables_skips_when_no_gpu_context(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tt, "find_pymol", lambda: ["/usr/bin/pymol", "-cq"])
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tt, "has_moderngl", lambda: True)

    def no_context(*_a: object, **_k: object) -> None:
        raise RuntimeError("cannot create an OpenGL context")

    monkeypatch.setattr(tt, "ClayRenderer", no_context)
    assert tt.render_turntables(["1OMG"], tmp_path) == {}
    assert "no GPU context" in capsys.readouterr().out


def test_render_turntables_releases_renderer_when_environment_setup_fails(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tt, "find_pymol", lambda: ["/usr/bin/pymol", "-cq"])
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tt, "has_moderngl", lambda: True)
    released: list[bool] = []

    class FakeRenderer:
        def __init__(self, _size: int) -> None:
            pass

        def set_environment(self, _faces: object) -> None:
            raise ValueError("cube faces must be square")

        def release(self) -> None:
            released.append(True)

    monkeypatch.setattr(tt, "ClayRenderer", FakeRenderer)

    assert tt.render_turntables(["1OMG"], tmp_path) == {}
    output = capsys.readouterr().out
    assert "environment setup failed" in output
    assert "no GPU context" not in output
    assert released == [True]


def test_render_turntable_releases_owned_renderer_when_environment_setup_fails(
    monkeypatch, tmp_path
) -> None:
    released: list[bool] = []

    class FakeRenderer:
        def __init__(self, _size: int) -> None:
            pass

        def set_environment(self, _faces: object) -> None:
            raise ValueError("cube faces must be square")

        def release(self) -> None:
            released.append(True)

    monkeypatch.setattr(tt, "ClayRenderer", FakeRenderer)
    monkeypatch.setattr(tt, "has_moderngl", lambda: True)
    monkeypatch.setattr(
        tt, "fetch_pdb", lambda _pdb_id, cache: (cache / "1OMG.pdb", "Conotoxin")
    )
    monkeypatch.setattr(tt, "structure_atoms", lambda _pdb_id, _cache: 100)
    monkeypatch.setattr(
        tt, "export_surface_meshes", lambda *_args, **_kwargs: [tmp_path / "mesh.npz"]
    )
    monkeypatch.setattr(
        tt,
        "meshes_from_files",
        lambda *_args, **_kwargs: [SimpleNamespace(positions=np.zeros((3, 3)))],
    )

    with pytest.raises(ValueError, match="cube faces must be square"):
        tt.render_turntable(
            "1OMG",
            tmp_path,
            pymol=["/usr/bin/pymol", "-cq"],
            ffmpeg="/usr/bin/ffmpeg",
        )

    assert released == [True]


def test_render_turntables_reports_and_skips_a_failing_structure(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tt, "find_pymol", lambda: ["/usr/bin/pymol", "-cq"])
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tt, "has_moderngl", lambda: True)
    released: list[bool] = []

    class FakeRenderer:
        def __init__(self, size: int) -> None:
            self.size = size
            self.environment: object = "unset"

        def set_environment(self, faces: object) -> None:
            self.environment = faces

        def release(self) -> None:
            released.append(True)

    monkeypatch.setattr(tt, "ClayRenderer", FakeRenderer)
    calls: list[tuple[str, object, object]] = []

    def fake_render(
        pdb_id: str, cache_dir: Path, **kwargs: object
    ) -> tt.TurntableAssets:
        calls.append((pdb_id, kwargs["color"], kwargs["renderer"]))
        if pdb_id == "BAD1":
            raise RuntimeError("PyMOL exported no surface for BAD1")
        return tt.TurntableAssets(
            pdb_id, cache_dir / "a.webm", cache_dir / "a.png", "t", 900, 30
        )

    monkeypatch.setattr(tt, "render_turntable", fake_render)
    assets = tt.render_turntables(
        ["3WU2", "1OMG", "BAD1"], tmp_path, colors={"3wu2": (0.4, 0.98, 0.4)}
    )
    assert set(assets) == {"1OMG", "3WU2"}
    assert "BAD1 failed" in capsys.readouterr().out
    # Story colours reach the renderer (case-insensitively); others default;
    # ONE renderer (GL context) is shared by every structure and released once.
    by_id = {c[0]: c for c in calls}
    assert by_id["3WU2"][1] == (0.4, 0.98, 0.4)
    assert by_id["1OMG"][1] == tt.DEFAULT_COLOR
    assert len({id(c[2]) for c in calls}) == 1 and isinstance(calls[0][2], FakeRenderer)
    # No environment was given: the shared renderer is told so (studio lights).
    assert calls[0][2].environment is None
    assert released == [True]


def test_cache_key_and_cached_lookup_include_the_environment_digest(tmp_path) -> None:
    import numpy as np

    faces = np.zeros((6, 4, 4, 4), dtype=np.float32)
    faces[0, 0, 0] = (1.0, 0.5, 0.25, 1.0)
    digest = tt.environment_digest(faces)
    assert len(digest) == 8 and tt.environment_digest(None) == ""
    base = tt.cache_key("1OMG", 900, 768)
    lit = tt.cache_key("1OMG", 900, 768, env_digest=digest)
    assert lit != base and lit.startswith("1OMG_v")
    # A different map is a different render; the same map is the same file.
    other = faces.copy()
    other[1, 1, 1] = (0.0, 0.0, 1.0, 1.0)
    assert (
        tt.cache_key("1OMG", 900, 768, env_digest=tt.environment_digest(other)) != lit
    )
    assert (
        tt.cache_key("1OMG", 900, 768, env_digest=tt.environment_digest(faces.copy()))
        == lit
    )
    # The cache lookup honours it: an unlit render does not satisfy a lit request.
    stem = tt.cache_key("1OMG", tt.DEFAULT_FRAMES, tt.DEFAULT_SIZE)
    (tmp_path / f"{stem}.webm").write_bytes(b"v")
    (tmp_path / f"{stem}.png").write_bytes(b"p")
    kw = dict(
        frames=tt.DEFAULT_FRAMES,
        fps=tt.DEFAULT_FPS,
        size=tt.DEFAULT_SIZE,
        color=tt.DEFAULT_COLOR,
    )
    assert tt._cached_turntable_assets("1OMG", tmp_path, **kw) is not None
    assert (
        tt._cached_turntable_assets("1OMG", tmp_path, env_digest=digest, **kw) is None
    )


def test_load_environment_faces_reads_a_baked_store_and_is_none_otherwise(
    tmp_path,
) -> None:
    import numpy as np

    from luxar import Dimensions, LuxarZarrCompiler
    from luxar._zarr_compat import create_array, open_group
    from luxar.environment import (
        ENVIRONMENT_FORMAT,
        FACE_ORDER,
        attach_environment,
        pack,
    )

    assert tt.load_environment_faces(tmp_path / "missing.luxar.zarr") is None
    store = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud", np.random.default_rng(1).random((50, 3)).astype(np.float32)
        )
    assert tt.load_environment_faces(store) is None  # built, not baked

    res = 8
    faces = np.zeros((6, res, res, 4), dtype=np.float32)
    faces[2] = (0.1, 0.8, 0.3, 1.0)  # a green sky on +Y
    header = {
        "format": ENVIRONMENT_FORMAT,
        "face_order": list(FACE_ORDER),
        "coordinate_system": "webgl",
        "probe": {"spec": "auto", "position": [0.5, 0.5, 0.5]},
        "resolution": res,
        "scene_content_hash": str(
            dict(open_group(store, mode="r").attrs)["content_hash"]
        ),
        "appearance": {},
        "baked_at": "2026-09-08T00:00:00Z",
        "viewer_version": "test",
    }
    attach_environment(store, pack(header, faces.astype(np.float16).view(np.uint16)))

    loaded = tt.load_environment_faces(store)
    assert loaded is not None and loaded.shape == (6, res, res, 4)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded[2, 0, 0, :3], (0.1, 0.8, 0.3), atol=1e-3)
    assert not loaded[3].any()

    root = open_group(store, mode="a")
    env = root["environment"]
    malformed = np.ones((6, res, res // 2, 4), dtype=np.float16).view(np.uint16)
    create_array(env, "malformed", data=malformed, overwrite=True)
    env.attrs["faces"] = "malformed"
    assert tt.load_environment_faces(store) is None


# ---------------------------------------------------------------------------
# Completeness (2026-09-17: render the biological assembly, ligands included)
# ---------------------------------------------------------------------------


def test_surface_script_draws_the_whole_molecule_not_just_the_polymer(
    tmp_path: Path,
) -> None:
    """Cofactors and ligands are part of the molecule and must be drawn.

    The first version enumerated `mol and polymer` and drew `polymer and
    chain X`, which dropped hemoglobin's four hemes, photosystem II's Mn4CaO5
    cluster and chlorophylls, the PLP that does the decarboxylation, the
    catalytic zinc of the nisin cyclase, the spike's glycan shield, and the
    propionate one panel called the odorant the receptor was holding.
    """
    script = tt.pymol_surface_script(
        tmp_path / "2HHB.cif", tmp_path / "mesh", quality=2
    )
    assert "not solvent and chain" in script
    assert "polymer and chain" not in script, "polymer-only drops every ligand"
    # Chains are enumerated over the same selection, so a ligand deposited on
    # its own chain id still gets a mesh instead of vanishing.
    assert "cmd.get_chains('mol and not solvent')" in script
    # Crystallisation additives go first: completeness means the molecule, not
    # the drop it was grown in.
    assert "cmd.remove('resn " in script
    for additive in ("GOL", "SO4", "PEG"):
        assert additive in script, additive


def test_the_additive_list_keeps_metals_and_cofactors() -> None:
    """A longer denylist would start deleting chemistry."""
    additives = set(tt.CRYSTALLISATION_ADDITIVES)
    for keep in (
        "HEM",
        "CLA",
        "OEX",
        "PLP",
        "ZN",
        "MG",
        "NA",
        "CA",
        "FE2",
        "ATP",
        "ADP",
        "NAG",
        "PPI",
        "PO4",
    ):
        assert keep not in additives, f"{keep} is chemistry, not crystallisation"
    for drop in ("GOL", "EDO", "MPD", "SO4", "DMS", "TRS"):
        assert drop in additives, drop


def test_the_assembly_is_preferred_over_the_asymmetric_unit() -> None:
    """A deposit's asymmetric unit is bookkeeping, not a molecule.

    Audited over the protein-universe tour's twenty entries (2026-09-16):
    4OO8 packs TWO whole Cas9-RNA-DNA complexes, 8RUC is half a RuBisCO
    (8 chains of the L8S8 hexadecamer) and 1U94 is one protomer of a
    six-subunit RecA filament — each a case the story's own text described
    correctly while the picture showed something else.
    """
    assert "assembly1" in tt.RCSB_ASSEMBLY_URL
    # Tried FIRST, ahead of the legacy PDB and the entry mmCIF.
    src = inspect.getsource(tt.fetch_pdb)
    order = [
        src.index("RCSB_ASSEMBLY_URL"),
        src.index("RCSB_FILE_URL"),
        src.index("RCSB_CIF_URL"),
    ]
    assert order == sorted(order), "the assembly must be tried first"


def test_the_render_cache_key_covers_the_geometry() -> None:
    """Bumping MESH_VERSION must invalidate the VIDEO, not just the surfaces.

    It used not to: the key held STYLE_VERSION only, so a corrected structure
    recomputed every surface and then served the WebM built from the old ones.
    """
    base = tt.cache_key("2HHB", 900, 512)
    bumped = tt.cache_key("2HHB", 900, 512, style_version=tt.STYLE_VERSION + 1)
    assert base != bumped
    assert f"m{tt.MESH_VERSION}" in base
    src = inspect.getsource(tt.cache_key)
    assert "MESH_VERSION" in src


def test_the_surface_script_clears_pymols_ignore_flag(tmp_path: Path) -> None:
    """Selecting a ligand is not the same as surfacing it.

    PyMOL sets an `ignore` flag on non-polymer atoms, and a flagged atom takes
    no part in the SURFACE even when selected and shown. Without clearing it,
    `not solvent and chain X` merely selects the ligands while the surface
    still traces the protein alone — so an atom-count check passes and the
    picture does not change. Verified by geometry instead: with the flag
    cleared, hemoglobin's chain A loses 2,574 vertices, because the heme fills
    the pocket whose concave lining used to be surfaced.
    """
    script = tt.pymol_surface_script(
        tmp_path / "2HHB.cif", tmp_path / "mesh", quality=2
    )
    assert "cmd.flag('ignore', 'not solvent', 'clear')" in script
    # Before the chains are enumerated and drawn, or it has no effect.
    assert script.index("cmd.flag('ignore'") < script.index("cmd.get_chains")


def test_a_surfaceless_chain_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One bad chain must not cost the other structures their turntables.

    It did: a glycan chain of 6VXX raised out of the mesh export, and the
    ELEVEN structures queued behind it got no turntable at all — while the
    build still exited 0, because turntables are optional and the failure was
    swallowed upstream.
    """
    good = tmp_path / "chain_0.obj"
    good.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//1 2//2 3//3\n")
    atoms_no_surface = tmp_path / "chain_1.obj"
    atoms_no_surface.write_text("# a chain PyMOL declined to surface\n")
    absent = tmp_path / "chain_2.obj"

    assert tt.obj_has_geometry(good)
    assert not tt.obj_has_geometry(atoms_no_surface)
    assert not tt.obj_has_geometry(absent)
    # Vertices without faces are not geometry either.
    verts_only = tmp_path / "chain_3.obj"
    verts_only.write_text("v 0 0 0\nv 1 0 0\n")
    assert not tt.obj_has_geometry(verts_only)
