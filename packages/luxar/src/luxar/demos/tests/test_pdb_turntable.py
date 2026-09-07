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
    assert released == [True]
