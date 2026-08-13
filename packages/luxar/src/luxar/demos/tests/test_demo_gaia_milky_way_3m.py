"""Scene-attribute regression tests for the Gaia Milky Way demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos.demo_gaia_milky_way_3m import load_and_convert_gaia_data


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


def test_authored_nodes_keep_gaia_volumetric_appearance(tmp_path: Path) -> None:
    """The built scene pins the appearance settings introduced with containment."""
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
    stars = dict(scene["Stars"].attrs)
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
    marker_radius = (0.001 + 0.01 * 0.5**2) * 10.0 * 10  # typical star radius x10, SCALE=10
    expected_marker_kappa = 1.3 * marker_radius * float(np.sqrt(np.pi / np.log(100.0)))
    for name in ("Sun", "Betelgeuse", "Rigel"):
        marker = dict(scene[name].attrs)
        assert marker["blending_mode"] == "volumetric"
        assert marker["opacity"] == pytest.approx(1.0)
        assert marker["absorption"] == pytest.approx(expected_marker_kappa)


def _marker_label(node: zarr.Group) -> str:
    """Decode a single-point node's one hover label from its UTF-8 CSR pair."""
    offsets = node["label_offsets"][:]
    return bytes(node["label_bytes"][offsets[0] : offsets[1]]).decode("utf-8")


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

    for name in ("Sun", "Betelgeuse", "Rigel"):
        node = scene[name]
        # Same string in the tooltip and in the legend row.
        assert _marker_label(node) in legend["html"]
        r, g, b = (int(round(float(c) * 255)) for c in node["colors"][0])
        assert f"background:rgb({r},{g},{b})" in legend["html"]

    # The labels are only *visible* because the compiler auto-injects a hover
    # overlay when any node carries labels; a suppressed or pre-empted injection
    # would leave them as dead bytes on disk.
    assert "__hover_text" in set(overlays.group_keys())
