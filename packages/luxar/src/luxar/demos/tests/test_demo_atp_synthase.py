"""Guard the appearance attrs ``demo_atp_synthase`` bakes into its scene.

The look of this demo is one tuned setting — ``volumetric`` compositing, the
absorption coefficient kappa, and the display window the gain encodes — arrived
at interactively in the Layers panel and then baked. Nothing else pins it, so a
silent revert to ``normal``, a retuned kappa, or an attribute quietly dropped on
the write path would ship unnoticed. This builds the scene and reads the attrs
back off the zarr store, the way the other per-demo appearance guards do.

No network and no PDB download: the demo is loaded by file path and its
``download_pdb`` is monkeypatched to write a tiny synthetic structure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import zarr

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_atp_synthase.py"


def _load_demo_module() -> ModuleType:
    name = "_luxar_demo_atp_synthase_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}", allow_module_level=True)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()

#: (atom name, chain, residue, xyz, element) — two chains so ``chain_to_color``
#: produces more than one hue, and four elements so the vdW radius table is hit.
_SYNTHETIC_ATOMS: list[tuple[str, str, int, tuple[float, float, float], str]] = [
    ("N", "A", 1, (0.0, 0.0, 0.0), "N"),
    ("CA", "A", 1, (1.5, 0.0, 0.0), "C"),
    ("O", "A", 1, (2.0, 1.2, 0.0), "O"),
    ("SG", "A", 2, (3.1, -0.8, 0.4), "S"),
    ("N", "B", 1, (0.0, 4.0, 0.0), "N"),
    ("CA", "B", 1, (1.5, 4.0, 0.6), "C"),
    ("O", "B", 1, (2.0, 5.2, -0.3), "O"),
    ("SG", "B", 2, (3.1, 3.2, 0.9), "S"),
]


def _atom_line(
    serial: int,
    name: str,
    chain: str,
    resseq: int,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    """One fixed-column PDB ``ATOM`` record, as ``parse_pdb_atoms`` slices it."""
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5d} {name:<4s} ALA {chain}{resseq:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}"
    )


def _write_synthetic_pdb(path: Path) -> Path:
    lines = [
        _atom_line(i + 1, name, chain, resseq, xyz, element)
        for i, (name, chain, resseq, xyz, element) in enumerate(_SYNTHETIC_ATOMS)
    ]
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def scene_attrs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build the demo scene offline; hand back the ``atp_synthase`` node attrs."""

    def fake_download_pdb(pdb_id: str, output_path: Path) -> Path:
        return _write_synthetic_pdb(output_path)

    monkeypatch.setattr(_demo, "download_pdb", fake_download_pdb)

    output = tmp_path / "atp.luxar.zarr"
    _demo.generate_atp_synthase(output)
    return dict(zarr.open_group(str(output), mode="r")["atp_synthase"].attrs)


def test_bakes_the_tuned_volumetric_look(scene_attrs: dict[str, Any]) -> None:
    """The whole tuned setting reaches the store — blending, kappa, exposure."""
    assert scene_attrs.get("blending_mode") == "volumetric", (
        "the atoms must composite as an emission-absorption medium; `normal` "
        "renders the complex as an opaque surface and hides its interior"
    )
    assert scene_attrs.get("absorption") == 2.5, (
        "absorption kappa sets how strongly the medium self-screens, and the "
        "exposure below was tuned against this value"
    )
    assert scene_attrs.get("intensity") == 1.62, (
        "display gain 1.62 ~= 1/0.616 — the Layers-panel display window the "
        "look was tuned at, which self-screening at this kappa needs"
    )
    assert scene_attrs.get("opacity") == 1.0
    assert scene_attrs["offset"] == 0.0, (
        "the tuned display window starts at 0; a non-zero offset moves it"
    )
    assert scene_attrs.get("layer") is True, (
        "`layer=True` is what puts the node in the Layers panel, where the "
        "absorption slider lives"
    )


def test_burial_shading_varies_without_changing_hue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(4)
    positions = rng.normal(size=(800, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= rng.random((800, 1)) ** (1.0 / 3.0)
    radii = np.full(len(positions), 0.12, dtype=np.float32)
    base = np.array([0.8, 0.4, 0.2], dtype=np.float32)
    colors = np.tile(base, (len(positions), 1))
    monkeypatch.setattr(_demo, "AO_GRID_CELLS", 32)
    monkeypatch.setattr(_demo, "AO_RADIUS_NM", 0.5)

    shaded = _demo._apply_burial_shading(colors, positions, radii)
    scale = shaded / base[None, :]

    assert float(scale[:, 0].max()) == pytest.approx(1.0, abs=1e-6)
    assert float(scale[:, 0].min()) < 0.9
    np.testing.assert_allclose(scale[:, 1], scale[:, 0], atol=1e-6)
    np.testing.assert_allclose(scale[:, 2], scale[:, 0], atol=1e-6)


def test_standing_camera_puts_the_wide_end_up_and_the_stalk_right() -> None:
    """A synthetic F1F0: a long cylinder, a fat head at +z, a thin stalk at +x."""
    import numpy as np

    rng = np.random.default_rng(0)
    stalk_axis = rng.normal(size=(2000, 3)) * [0.6, 0.6, 3.0]  # long along z
    head = rng.normal(size=(3000, 3)) * [2.5, 2.5, 1.0] + [0.0, 0.0, 4.0]  # wide, at +z
    peripheral = rng.normal(size=(600, 3)) * [0.3, 0.3, 3.0] + [
        4.0,
        0.0,
        1.0,
    ]  # off-axis +x
    pts = np.concatenate([stalk_axis, head, peripheral]).astype(np.float32)

    cam = _load_demo_module().standing_camera(pts)
    up = np.asarray(cam.up)
    assert up[2] > 0.95, f"up should be the +z head direction, got {up}"
    view = np.asarray(cam.target) - np.asarray(cam.position)
    view /= np.linalg.norm(view)
    right = np.cross(view, up)
    assert right[0] > 0.9, f"screen-right should point at the +x stalk, got {right}"
    # Framed: the camera stands off far enough to see the whole cloud.
    radius = np.linalg.norm(pts - pts.mean(0), axis=1).max()
    assert np.linalg.norm(np.asarray(cam.position) - np.asarray(cam.target)) > radius
