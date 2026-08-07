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
