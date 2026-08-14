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

    # "Stars" is a MIXED substitutive ladder: the coarse levels are lifted
    # gsplats, whose tau = kappa*rayMass never carried the point shader's
    # world-radius factor. So the 2026-08-02 ray-mass unification leaves their
    # render — and therefore this kappa — untouched. What changed is the finest
    # (Points) level, which used to be ~35x more transparent than the levels it
    # replaces and now matches them. Rescaling kappa here would break the
    # coarse levels instead.
    stars = dict(scene["Stars"].attrs)
    assert stars["blending_mode"] == "volumetric"
    assert stars["opacity"] == pytest.approx(1.0)
    assert stars["absorption"] == pytest.approx(1.3)
    assert stars["intensity"] == pytest.approx(0.075)

    # The markers are PLAIN points (no LOD wrapper), so they DID need kappa
    # rescaled to survive that change: tau dropped its world-radius factor, so
    # the old 1.3 would now absorb ~3.5x harder. Preserving the authored look is
    # exactly kappa * radius * chord — the value the demo computes as
    # MARKER_ABSORPTION, recomputed here from first principles rather than
    # copied, so a change to either side has to be deliberate.
    marker_radius = (0.001 + 0.01 * 0.5**2) * 10.0 * 10  # typical star radius x10, SCALE=10
    expected_marker_kappa = 1.3 * marker_radius * float(np.sqrt(np.pi / np.log(100.0)))
    for name in ("Sun", "Betelgeuse", "Rigel"):
        marker = dict(scene[name].attrs)
        assert marker["blending_mode"] == "volumetric"
        assert marker["opacity"] == pytest.approx(1.0)
        assert marker["absorption"] == pytest.approx(expected_marker_kappa)


class TestDataFileResolution:
    """The Gaia catalog is CC BY-NC, so it is not shipped with the repository.

    What the demo owes a user without it is a message that names the file, the
    place to put it, and the issue that will build it — not a ``git lfs pull``
    for a file that is no longer in the tree.
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
        assert "1461" in message
        assert "git lfs" not in message.lower()
