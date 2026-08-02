"""Regression tests for the Mouse Multiome demo's Points-LOD dependency gate.

The scene build used to request substitutive Points LOD unconditionally. That
path imports ``luxar.gsplats.lod``, which needs torch (coarsening kernels) and
scipy (``additive.py`` imports ``scipy.sparse`` at module level) — neither is a
core dependency — so a dependency-free run died with a ``ModuleNotFoundError``
instead of producing a viewable scene. The fix gates LOD on both and falls back
to flat Points, so the demo runs with either one blocked.

``create_mouse_scene`` takes ``(output_path, coordinates, attributes,
category_maps)`` directly, so these tests fabricate in-memory inputs and need no
local parquet file, network, or zarr store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The demo module imports pandas at module level, and pandas lives in the
# `demos` extra (not `test`) — skip rather than error out at collection.
pytest.importorskip("pandas")

from luxar.demos.demo_mouse_multiome_peak_umap import create_mouse_scene  # noqa: E402

N_POINTS = 40


def _fabricate() -> tuple[np.ndarray, dict, dict]:
    """Minimal valid inputs: N points and two contiguous categorical attributes."""
    rng = np.random.default_rng(0)
    coordinates = rng.standard_normal((N_POINTS, 3)).astype(np.float32)
    # celltype + chromosome are the first two entries of the demo's attr_types,
    # so available_attrs / category labels line up without holes.
    attributes = {
        "celltype": rng.integers(0, 3, size=N_POINTS).astype(np.int32),
        "chromosome": rng.integers(0, 3, size=N_POINTS).astype(np.int32),
    }
    category_maps = {
        "celltype": ["Neuron", "Muscle", "Skin"],
        "chromosome": ["chr1", "chr2", "chr3"],
    }
    return coordinates, attributes, category_maps


@pytest.mark.parametrize("blocked", ["torch", "scipy"])
def test_scene_builds_flat_without_lod_dep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    blocked: str,
) -> None:
    """With torch OR scipy blocked, the scene builds as a flat Points leaf."""
    coordinates, attributes, category_maps = _fabricate()

    # A None entry makes is_installed() return False AND makes any fresh
    # `import <blocked>` raise — the exact reproduction, even where it IS
    # installed.
    monkeypatch.setitem(sys.modules, blocked, None)

    out = tmp_path / "mouse.luxar.zarr"
    got = create_mouse_scene(out, coordinates, attributes, category_maps)

    assert got == N_POINTS
    assert out.exists(), f"scene was not written with {blocked} blocked"

    stdout = capsys.readouterr().out
    assert "skipping Points LOD" in stdout, "degradation notice not printed"
    assert blocked in stdout, f"degradation notice did not name {blocked}"

    # Structural proof of the fallback, independent of import ordering: a flat
    # Points leaf writes ``Cells/positions`` and has no ``kind: lod``.
    cells = out / "Cells"
    assert (cells / "positions").exists(), (
        "flat Points leaf missing positions — LOD group written instead"
    )
    attrs = json.loads((cells / ".zattrs").read_text())
    assert attrs.get("kind") != "lod", (
        f"expected a flat Points leaf, got a substitutive-LOD group: {attrs.get('kind')!r}"
    )


def test_lod_group_built_when_deps_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive branch: with torch and scipy installed, the LOD group IS built.

    At N=40 with compression_factor=8, levels=3 the node is a real substitutive
    LOD group (``kind == "lod"``), so this fails if the gate is inverted or the
    LOD dropped entirely, and the degradation notice must stay quiet.
    """
    pytest.importorskip("torch")
    pytest.importorskip("scipy")

    coordinates, attributes, category_maps = _fabricate()

    out = tmp_path / "mouse.luxar.zarr"
    got = create_mouse_scene(out, coordinates, attributes, category_maps)

    assert got == N_POINTS
    assert out.exists()
    assert "skipping Points LOD" not in capsys.readouterr().out, (
        "degradation notice printed even though torch and scipy are installed"
    )
    attrs = json.loads((out / "Cells" / ".zattrs").read_text())
    assert attrs.get("kind") == "lod", (
        f"expected a substitutive-LOD group with the deps present: {attrs.get('kind')!r}"
    )
