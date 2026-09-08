"""Unit tests for the PDB turntable helper's pure pieces.

PyMOL is a special-case (conda/Homebrew-only) dependency and the renderer needs
a GPU context, so nothing here runs either. The tests pin what the demo relies
on: the surface-export script, cache keying (colour included), the pastel
palette, the soft gate that lets the demo build without turntables when a tool
is absent, and the per-structure skip.
"""

from __future__ import annotations

from pathlib import Path

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
    assert "cmd.get_chains('mol and polymer')" in script
    # Per-chain surfaces: hide everything, show this chain, save its OBJ.
    assert "cmd.show('surface', \"polymer and chain '%s'\" % chain)" in script
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
    from luxar._zarr_compat import open_group
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
