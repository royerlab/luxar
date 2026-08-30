"""Regression guard for the CELLxGENE Census UMAP demo's baked appearance.

Issue #1375: the demo opened far too dark, mostly through the baked display
window, which the Layers panel stores as ``intensity = 1/(max - min)``: the
authored window was ``[0, 2.361]``, i.e. ``intensity = 0.4235``, and a window
max ABOVE 1 attenuates the authored direct colours (here to 0.42x) instead of
brightening them. Undoing that is only part of it — the cloud was still ~4.5x
under at the identity window. This test builds the scene on a tiny synthetic cache — no
download, no torch required — and checks the persisted appearance attrs of the
``cells`` node, including the invariant that the window must brighten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos import demo_cellxgene_census_umap as _demo
from luxar.demos.demo_cellxgene_census_umap import build_scene

#: The appearance baked in `scene.add_points("cells", ...)` after #1375.
EXPECTED_OPACITY = 0.39
EXPECTED_INTENSITY = 4.52
# Retuned from 6.5: at that kappa a cell absorbed 0.92 of what was behind
# it and its own self-screening S(tau)=0.36 ate most of its emission, so
# the cloud read as a screened shell rather than depth-ordered structure.
EXPECTED_ABSORPTION = 2.12


def test_default_cache_uses_the_manifest_resolved_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = tmp_path / "resolved.npz"
    calls: list[tuple[str, object]] = []

    monkeypatch.delenv("CENSUS_UMAP_CACHE", raising=False)
    monkeypatch.delenv("CENSUS_UMAP_MAX_CELLS", raising=False)
    monkeypatch.setattr(_demo, "parse_demo_flags", lambda: {"no_serve": True})
    monkeypatch.setattr(
        _demo,
        "ensure_dataset",
        lambda name: calls.append(("ensure", name)) or [resolved],
    )
    monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        _demo,
        "build_scene",
        lambda cache, output, *, max_cells: (
            calls.append(("build", (cache, output, max_cells))) or 7
        ),
    )

    _demo.main()

    assert calls == [
        ("ensure", "census_umap_1m"),
        (
            "build",
            (
                resolved,
                tmp_path / "cellxgene_census_umap.luxar.zarr",
                1_000_000,
            ),
        ),
    ]


def test_explicit_cache_override_bypasses_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "custom.npz"
    override.write_bytes(b"custom")
    seen: list[Path] = []

    monkeypatch.setenv("CENSUS_UMAP_CACHE", str(override))
    monkeypatch.delenv("CENSUS_UMAP_MAX_CELLS", raising=False)
    monkeypatch.setattr(_demo, "parse_demo_flags", lambda: {"no_serve": True})
    monkeypatch.setattr(
        _demo,
        "ensure_dataset",
        lambda name: pytest.fail(f"unexpected manifest lookup for {name}"),
    )
    monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        _demo,
        "build_scene",
        lambda cache, output, *, max_cells: seen.append(cache) or 7,
    )

    _demo.main()

    assert seen == [override]


def test_missing_explicit_cache_keeps_generation_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.npz"

    monkeypatch.setenv("CENSUS_UMAP_CACHE", str(missing))
    monkeypatch.setattr(_demo, "parse_demo_flags", lambda: {"no_serve": True})
    monkeypatch.setattr(
        _demo,
        "ensure_dataset",
        lambda name: pytest.fail(f"unexpected manifest lookup for {name}"),
    )
    monkeypatch.setattr(
        _demo,
        "build_scene",
        lambda *args, **kwargs: pytest.fail("missing override must not build"),
    )

    _demo.main()

    output = capsys.readouterr().out
    assert str(missing) in output
    assert "scripts/gen_census_umap.py" in output
    assert "CENSUS_UMAP_CACHE=<cache>.npz" in output


def test_manifest_payload_fault_does_not_recommend_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CENSUS_UMAP_CACHE", raising=False)
    monkeypatch.setattr(_demo, "parse_demo_flags", lambda: {"no_serve": True})
    monkeypatch.setattr(
        _demo,
        "ensure_dataset",
        lambda name: (_ for _ in ()).throw(
            FileNotFoundError(f"bad in-repo payload for {name}")
        ),
    )
    monkeypatch.setattr(
        _demo,
        "build_scene",
        lambda *args, **kwargs: pytest.fail("payload faults must not build"),
    )

    with pytest.raises(FileNotFoundError, match="bad in-repo payload"):
        _demo.main()


def _write_synthetic_cache(path: Path, n: int = 200) -> Path:
    """Write an npz with exactly the keys ``load_cache`` reads.

    ``coords`` (N,3) f32, one ``<field>_code`` int array per coloring, and a
    ``labels_json`` map field -> category labels. A few hundred cells keeps the
    scene build itself around a second; the LOD path's ``torch`` import
    dominates the rest.
    """
    rng = np.random.default_rng(0)
    labels = {
        "cell_type": [f"cell type {i}" for i in range(5)],
        "tissue_general": ["blood", "brain", "liver"],
        "disease": ["normal", "carcinoma"],
    }
    arrays = {
        f"{field}_code": rng.integers(0, len(names), n).astype(np.int32)
        for field, names in labels.items()
    }
    np.savez(
        path,
        coords=rng.normal(size=(n, 3)).astype(np.float32),
        labels_json=json.dumps(labels),
        **arrays,
    )
    return path


def _find_node(group, name: str):
    """Depth-first search for a child group called ``name``.

    The demo's ``cells`` node is a ``kind=lod`` group when ``luxar[gsplats]``
    is installed and a flat Points leaf when it is not (``substitutive_lod_or_
    flat`` degrades), so the layout is not fixed — search by name instead.
    """
    for child_name, child in group.groups():
        if child_name == name:
            return child
        found = _find_node(child, name)
        if found is not None:
            return found
    return None


@pytest.fixture(scope="module")
def cells_attrs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Attrs of the persisted ``cells`` node, built from a synthetic cache."""
    tmp_path = tmp_path_factory.mktemp("census_umap")
    cache = _write_synthetic_cache(tmp_path / "census_umap_test.npz")
    output = tmp_path / "cellxgene_census_umap.luxar.zarr"
    build_scene(cache, output, max_cells=None)
    cells = _find_node(zarr.open_group(output, mode="r"), "cells")
    assert cells is not None, "no `cells` node in the built scene"
    return dict(cells.attrs)


class TestBakedAppearance:
    def test_display_window_brightens_rather_than_dims(self, cells_attrs: dict) -> None:
        # THE #1375 invariant: the panel's display range is stored as
        # `intensity = 1/(max - min)`, `offset = -min/(max - min)`, so a window
        # max > 1 attenuates the authored direct colours and the scene opens
        # too dark. The demo bakes a `[0, max]` window (no offset), which is
        # what reduces the invariant to `intensity > 1`.
        offset = float(cells_attrs.get("offset", 0.0))
        assert offset == 0.0, (
            f"baked window has a non-zero offset ({offset}); the min is no "
            "longer 0 so `intensity > 1` alone no longer means brightening"
        )
        intensity = float(cells_attrs["intensity"])
        assert intensity > 1.0, (
            f"baked display window [0, {1.0 / intensity:.3f}] dims the authored "
            "colours instead of brightening them (#1375)"
        )

    def test_tuned_values_are_baked(self, cells_attrs: dict) -> None:
        assert cells_attrs["intensity"] == pytest.approx(EXPECTED_INTENSITY)
        assert cells_attrs["absorption"] == pytest.approx(EXPECTED_ABSORPTION)
        assert cells_attrs["opacity"] == pytest.approx(EXPECTED_OPACITY)

    def test_blending_is_volumetric(self, cells_attrs: dict) -> None:
        # Absorption only means anything under emission-absorption compositing.
        assert cells_attrs["blending_mode"] == "volumetric"
