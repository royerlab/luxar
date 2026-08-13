"""Scene-attribute regression tests for the Gaia Milky Way demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos._dependencies import SUBSTITUTIVE_LOD_MODULES, is_installed
from luxar.demos.demo_gaia_milky_way_3m import (
    compute_radii,
    load_and_convert_gaia_data,
)


def _write_tiny_gaia_table(path: Path, n_stars: int = 16) -> None:
    """Write the five raw columns consumed by the demo converter."""
    root = zarr.open(str(path), mode="w")
    values = {
        "x_kpc": np.linspace(-2.0, 2.0, n_stars, dtype=np.float32),
        "y_kpc": np.linspace(-1.0, 1.0, n_stars, dtype=np.float32),
        "z_kpc": np.linspace(-0.2, 0.2, n_stars, dtype=np.float32),
        "phot_g_mean_mag": np.linspace(2.0, 20.0, n_stars, dtype=np.float32),
        "bp_rp": np.linspace(-0.5, 3.0, n_stars, dtype=np.float32),
    }
    for name, data in values.items():
        root.create_dataset(name, data=data, shape=data.shape, dtype=data.dtype)


def _marker_label(node: zarr.Group) -> str:
    """Decode a single-point node's one hover label from its UTF-8 CSR pair."""
    offsets = node["label_offsets"][:]
    return bytes(node["label_bytes"][offsets[0] : offsets[1]]).decode("utf-8")


def test_authored_nodes_keep_gaia_volumetric_appearance(tmp_path: Path) -> None:
    """The built scene pins the demo's volumetric compositing knobs, and where.

    Two claims: the four "Stars" knobs hold their tuned values, and they are set
    on the kind=lod WRAPPER with every level inert under it — which is what makes
    every level of the mixed ladder composite identically. The markers' kappa is
    pinned too, recomputed from the demo's own radius law.
    """
    raw = tmp_path / "gaia.zarr"
    scene_path = tmp_path / "gaia.luxar.zarr"
    _write_tiny_gaia_table(raw)

    assert load_and_convert_gaia_data(raw, scene_path) == 16

    scene = zarr.open(str(scene_path), mode="r")

    # "Stars" carries three knobs that are tuned as ONE set, because they land
    # on the same two shader terms: ray mass = falloff * opacity, optical depth
    # tau = kappa * ray mass, radiance = colour * intensity * ray mass * S(tau).
    # opacity 0.5 and kappa 0.12 keep tau low enough that the disc stays
    # translucent front to back, and intensity 0.175 buys back the emission the
    # lower ray mass gives up (a 0-5.7 display range, i.e. 1/intensity, over the
    # 0-32.3 data range). Changing one alone re-lights the scene, so all three
    # are pinned together — and they are pinned on the kind=lod WRAPPER, which
    # is what makes every level of this mixed ladder (lifted gsplats coarse,
    # Points finest) composite with the same tau and the same gain.
    stars_node = scene["Stars"]
    stars = dict(stars_node.attrs)
    assert stars["blending_mode"] == "volumetric"
    assert stars["opacity"] == pytest.approx(0.5)
    assert stars["absorption"] == pytest.approx(0.12)
    assert stars["intensity"] == pytest.approx(0.175)

    # The markers keep their ORIGINAL authored look, so their kappa is not a
    # free knob: it is the historical 1.3 rescaled through the 2026-08-02
    # ray-mass unification, which dropped tau's world-radius factor (a bare 1.3
    # would now absorb ~3.5x harder). Preserving the authored look is exactly
    # kappa * radius * chord — the value the demo computes as MARKER_ABSORPTION,
    # recomputed here from first principles rather than copied, so a change to
    # either side has to be deliberate.
    #
    # The demo sizes a marker off its OWN radius law at "mid-brightness": the
    # magnitude whose normalized brightness is exactly 0.5, which compute_radii's
    # (21 - mag) / 18 puts at G = 12.0. Derived here instead of restating the
    # expression, so retuning the radius law has to be deliberate too.
    mid_star_radius = float(compute_radii(np.array([12.0], dtype=np.float32))[0])
    marker_radius = mid_star_radius * 10.0 * 10  # x SCALE=10, marker = 10x a star
    expected_marker_kappa = 1.3 * marker_radius * float(np.sqrt(np.pi / np.log(100.0)))
    for name in ("Sun", "Betelgeuse", "Rigel"):
        marker = dict(scene[name].attrs)
        assert marker["blending_mode"] == "volumetric"
        assert marker["opacity"] == pytest.approx(1.0)
        assert marker["absorption"] == pytest.approx(expected_marker_kappa)

    # The "on the WRAPPER" half of the "Stars" claim needs the wrapper to exist:
    # without torch+scipy the demo's substitutive_lod_or_flat() degrades "Stars"
    # to a flat leaf, and the value assertions above pass while saying nothing
    # about levels. Skip rather than fail on such a machine — but only after the
    # marker checks, which do not depend on the ladder.
    missing = [m for m in SUBSTITUTIVE_LOD_MODULES if not is_installed(m)]
    if missing:
        pytest.skip(
            f"substitutive Points LOD needs {list(SUBSTITUTIVE_LOD_MODULES)}; "
            f"{missing} missing, so 'Stars' is a flat leaf here and the "
            "wrapper-vs-level claim is not testable"
        )
    assert stars["kind"] == "lod", (
        "'Stars' must be the kind=lod wrapper — a flat leaf would satisfy the "
        "value assertions above without pinning anything about the ladder"
    )
    # ...and every level must be INERT under it — which is NOT the same as
    # carrying no attrs. The viewer composes opacity/absorption/intensity (and
    # gamma) MULTIPLICATIVELY root->leaf, and offset additively (viewer
    # src/data/attrs-composer.ts), so a level is inert exactly when its value is
    # the identity 1.0. The writers stamp exactly those 1.0s onto every node
    # unconditionally (`apply_default_render_attrs`,
    # io/_compiler/node_common.py:450-472, mirrored for gsplats in
    # io/_compiler/gsplat_assembly.py), so absence is not available to assert and
    # would not mean anything if it were.
    #
    # `blending_mode` is the exception, and the reason it gets its own check: it
    # has no identity value, so it is nearest-setter-wins, and the writers
    # deliberately refuse to stamp a default for it. It is therefore the one knob
    # a level could use to override the wrapper for itself alone.
    #
    # Every child is checked, not just the first: the coarse gsplat levels and
    # the finest Points level are written by different writers.
    levels = sorted(stars_node.group_keys())
    assert levels, "a kind=lod wrapper must have level children"
    for level in levels:
        level_attrs = dict(stars_node[level].attrs)
        assert "blending_mode" not in level_attrs, (
            f"level {level!r} sets its own blending_mode; under nearest-setter-wins "
            "that overrides the wrapper's 'volumetric' for this level alone"
        )
        for key in ("opacity", "absorption", "intensity"):
            child_value = level_attrs.get(key, 1.0)
            assert child_value == pytest.approx(1.0), (
                f"level {level!r} has {key}={child_value}, not the multiplicative "
                f"identity — it would rescale the wrapper's {stars[key]}"
            )
            # The invariant that actually matters: what the renderer composes for
            # this level is the wrapper's value, unchanged.
            assert stars[key] * child_value == pytest.approx(stars[key])


def test_named_star_legend_is_derived_from_the_marker_nodes(tmp_path: Path) -> None:
    """Legend rows and swatches restate the markers, they do not re-invent them.

    The legend duplicates information that also lives on the marker nodes (the
    label text) or is computed from them (the swatch colour), so both are
    asserted against the nodes rather than against literals — a marker recoloured
    or relabelled without touching the legend fails here.
    """
    raw = tmp_path / "gaia.zarr"
    scene_path = tmp_path / "gaia.luxar.zarr"
    _write_tiny_gaia_table(raw)
    load_and_convert_gaia_data(raw, scene_path)

    scene = zarr.open(str(scene_path), mode="r")
    overlays = scene["overlays"]
    html_overlays = [
        dict(overlays[g].attrs)
        for g in overlays.group_keys()
        if dict(overlays[g].attrs).get("type") == "overlay_html"
    ]
    assert len(html_overlays) == 1, "expected exactly one HTML overlay (the legend)"
    legend = html_overlays[0]
    assert legend["anchor"] == "bottom-left"

    # The marker set comes from the STORE, not a literal: a marker node is a
    # scene-level node carrying hover labels ("Stars" carries none).
    marker_names = sorted(
        g for g in scene.group_keys() if "label_bytes" in set(scene[g].array_keys())
    )
    for name in marker_names:
        node = scene[name]
        # Same string in the tooltip and in the legend row.
        assert _marker_label(node) in legend["html"]
        # `float(c) * 255` below only holds while colours stay float 0-1 on disk;
        # a uint8 (0-255) encoding would multiply by 255 twice and mismatch by a
        # confusing factor rather than naming the cause.
        assert np.issubdtype(node["colors"].dtype, np.floating), (
            f"{name!r} colours are {node['colors'].dtype} on disk, not float — the "
            "0-1 to 0-255 conversion here no longer applies (encoding changed)"
        )
        r, g, b = (int(round(float(c) * 255)) for c in node["colors"][0])
        assert f"background:rgb({r},{g},{b})" in legend["html"]

    # One swatch per marker, no more and no fewer: a fourth marker left out of
    # the legend, or a legend row for a marker that is gone, both fail here where
    # the per-marker loop above would not notice.
    swatches = legend["html"].count("background:rgb(")
    assert swatches == len(marker_names) > 0, (
        f"legend has {swatches} swatches for {len(marker_names)} marker nodes "
        f"({marker_names})"
    )

    # The labels are only *visible* because the compiler auto-injects a hover
    # overlay when any node carries labels; a suppressed or pre-empted injection
    # would leave them as dead bytes on disk.
    assert "__hover_text" in set(overlays.group_keys())


class TestDataFileResolution:
    """The Gaia catalog is CC BY-NC, so it is not shipped with the repository.

    What the demo owes a user without it is a message that names the file, the
    place to put it, why it is absent, the command that rebuilds it, the issue
    that will do that automatically, and the mandatory ESA/DPAC acknowledgement —
    not a ``git lfs pull`` for a file that is no longer in the tree.
    """

    def test_cache_wins_over_the_legacy_in_repo_copy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        repo = tmp_path / "repo" / "milky_way_gaia_3m.zarr.zip"
        for p in (cache, repo):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stand-in")
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", repo)

        assert demo.resolve_data_file() == cache

    def test_legacy_in_repo_copy_still_works(self, tmp_path: Path, monkeypatch) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        repo = tmp_path / "repo" / "milky_way_gaia_3m.zarr.zip"
        repo.parent.mkdir(parents=True)
        repo.write_bytes(b"stand-in")
        monkeypatch.setattr(demo, "CACHE_FILE", tmp_path / "absent.zip")
        monkeypatch.setattr(demo, "REPO_FILE", repo)

        assert demo.resolve_data_file() == repo

    def test_absent_everywhere_explains_why(self, tmp_path: Path, monkeypatch) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")

        with pytest.raises(FileNotFoundError) as excinfo:
            demo.resolve_data_file()
        message = str(excinfo.value)
        assert str(cache) in message
        # Why it is absent, and the rebuild that fills the gap until #1575
        # automates it — both are what the message owes the reader.
        assert "NonCommercial" in message
        assert "scripts/generate_galaxy_simple.py" in message
        assert "1575" in message
        # Using Gaia data at all obliges the acknowledgement, so it travels with
        # the instructions rather than only living in the module docstring.
        assert "Gaia Data Processing and Analysis Consortium (DPAC)" in message
        assert "git lfs" not in message.lower()
