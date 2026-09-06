"""Unit tests for the PyMOL turntable helper's pure pieces.

PyMOL is a special-case (conda/Homebrew-only) dependency, so nothing here runs
it. The tests pin what the demo relies on: the render script's style
invariants (transparent background, one full turn, ray tracing), the alpha
WebM encode flags, cache keying, and the soft gate that lets the demo build
without turntables when the tools are absent.
"""

from __future__ import annotations

from pathlib import Path

from luxar.demos import _pdb_turntable as tt


def test_cache_key_changes_with_every_render_parameter() -> None:
    base = tt.cache_key("1omg", 360, 1024)
    assert base.startswith("1OMG_v")  # id is upper-cased into the stem
    assert tt.cache_key("1OMG", 360, 1024) == base  # case-insensitive
    assert tt.cache_key("1OMG", 720, 1024) != base
    assert tt.cache_key("1OMG", 360, 512) != base
    assert tt.cache_key("1OMG", 360, 1024, style_version=tt.STYLE_VERSION + 1) != base


def test_pymol_script_turns_exactly_once_with_a_transparent_ray_traced_background() -> (
    None
):
    script = tt.pymol_script(
        Path("/x/1OMG.pdb"), Path("/y/frames"), frames=360, size=1024
    )
    assert "cmd.load('/x/1OMG.pdb', 'mol')" in script
    assert "cmd.set('ray_opaque_background', 0)" in script  # alpha channel
    assert "for i in range(360):" in script
    assert "cmd.turn('y', -1.0)" in script  # 360 frames → exactly 1° per frame, spun
    # in the negative sense to match the scene's auto-rotation; standing on the
    # longest axis (orient lays it along x, the quarter turn makes it vertical).
    assert "cmd.turn('z', 90)" in script
    assert "ray=1" in script and "width=1024, height=1024" in script
    # The look the demo promises: an opaque clay surface of the polymer alone,
    # per-chain shades of the story colour, soft shadows + ambient occlusion.
    for needle in (
        "cmd.show('surface', 'polymer')",
        "cmd.set('transparency', 0.0)",
        "cmd.set('ambient_occlusion_mode', 1)",
        "cmd.set('ray_shadows', 1)",
        "cmd.set_color('story_%d' % i",
        "cmd.color('story_%d' % i",
    ):
        assert needle in script, needle
    for absent in ("cartoon", "spheres", "hetatm", "util."):
        assert absent not in script, absent
    # A long complex must stay inside the square frame at every angle.
    assert "complete=1" in script and "cmd.clip('slab', 10000)" in script
    # Quality follows structure size; the ray-thread budget is explicit so
    # parallel jobs share the cores.
    assert f"large = n_atoms > {tt.LARGE_STRUCTURE_ATOMS}" in script
    assert "cmd.set('surface_quality', -1 if large else 0)" in script
    assert "cmd.set('antialias', 1)" in script
    # The defaults give a slow 30 s turn: 900 frames of 0.4 deg at 30 fps.
    assert tt.DEFAULT_FRAMES / tt.DEFAULT_FPS == 30
    assert "cmd.turn('y', -0.4)" in tt.pymol_script(
        Path("a.pdb"), Path("f"), frames=tt.DEFAULT_FRAMES, size=768
    )
    assert "cmd.set('max_threads', 4)" in script
    assert "cmd.set('max_threads', 6)" in tt.pymol_script(
        Path("a.pdb"), Path("f"), frames=360, size=768, threads=6
    )
    # Non-integer steps still sum to one turn.
    assert "cmd.turn('y', -2.5)" in tt.pymol_script(
        Path("a.pdb"), Path("f"), frames=144, size=256
    )


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
    # The colour is part of the cache key, so two stories sharing a PDB id
    # rendered in different colours never collide.
    assert tt.cache_key("1OMG", 360, 768, color=(1, 0, 0)) != tt.cache_key(
        "1OMG", 360, 768, color=(0, 0, 1)
    )
    assert tt.color_hex((1.0, 0.5, 0.0)) == "#ff8000"
    assert "shades = [" in tt.pymol_script(
        Path("a.pdb"), Path("f"), frames=360, size=768, color=neon_green
    )


def test_ffmpeg_command_encodes_vp9_with_alpha_at_the_requested_rate() -> None:
    cmd = tt.ffmpeg_command("/usr/bin/ffmpeg", Path("/f"), Path("/out.webm"), fps=60)
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert cmd[cmd.index("-framerate") + 1] == "60"
    assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuva420p"  # the alpha plane
    # VP9 alpha is silently dropped when alt-ref frames are on.
    assert cmd[cmd.index("-auto-alt-ref") + 1] == "0"
    assert cmd[-1] == "/out.webm"
    assert "-an" in cmd


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


def test_render_turntables_reports_and_skips_a_failing_structure(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(tt, "find_pymol", lambda: ["/usr/bin/pymol", "-cq"])
    monkeypatch.setattr(tt, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tt, "render_threads", lambda: 12)
    sizes = {"1OMG": 300, "BAD1": 500, "3WU2": tt.LARGE_STRUCTURE_ATOMS + 1}
    monkeypatch.setattr(tt, "structure_atoms", lambda pdb_id, cache_dir: sizes[pdb_id])
    calls: list[tuple[str, object]] = []
    colors: dict[str, object] = {}

    def fake_render(
        pdb_id: str, cache_dir: Path, **kwargs: object
    ) -> tt.TurntableAssets:
        calls.append((pdb_id, kwargs["threads"]))
        colors[pdb_id] = kwargs["color"]
        if pdb_id == "BAD1":
            raise RuntimeError("PyMOL produced 0 of 360 frames for BAD1")
        return tt.TurntableAssets(
            pdb_id, cache_dir / "a.webm", cache_dir / "a.png", "t", 360, 60
        )

    monkeypatch.setattr(tt, "render_turntable", fake_render)
    assets = tt.render_turntables(
        ["3WU2", "1OMG", "BAD1"], tmp_path, jobs=2, colors={"3wu2": (0.4, 0.98, 0.4)}
    )
    assert set(assets) == {"1OMG", "3WU2"}
    # Story colours reach the renderer (case-insensitively); others default.
    assert colors["3WU2"] == (0.4, 0.98, 0.4)
    assert colors["1OMG"] == tt.DEFAULT_COLOR
    assert "BAD1 failed" in capsys.readouterr().out
    # Small structures share the cores across the pool; the large one runs
    # LAST, alone, with every core.
    assert sorted(calls[:2]) == [("1OMG", 6), ("BAD1", 6)]
    assert calls[2] == ("3WU2", 12)
