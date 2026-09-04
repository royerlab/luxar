"""`builtins.py` says it is auto-generated. That has to stay true.

The file's header credits ``scripts/generate_builtin_colormaps.py``, which is an
instruction to the next author: regenerate rather than hand-edit. Running it
used to DELETE four shipped colormaps -- ``orange``, ``bop_blue``,
``bop_orange``, ``bop_purple`` -- because two hardcoded name lists had to agree
and did not. One decided what was *generated* (``LINEAR_RAMPS`` +
``matplotlib_maps``); a second, inside each writer, decided what was *written*.
The four had been hand-added to both output files and folded back into neither.

The failure mode is the bad kind: the header invites exactly the action that
destroys data, and the loss is silent -- the script prints "Done!" and the
colormaps are simply gone from both the Python and the TypeScript output.

So this asserts the round trip the header promises. It compares LUT BYTES, not
just names, so a generator change that quietly alters a shipped colormap fails
too -- those bytes are baked into published scenes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from luxar.colormaps.builtins import BUILTIN_COLORMAP_NAMES, get_builtin_lut

REPO = Path(__file__).resolve().parents[6]
GENERATOR = REPO / "scripts/generate_builtin_colormaps.py"


def _generator() -> ModuleType:
    """Import the generator script by path; it is not an installed module."""
    spec = importlib.util.spec_from_file_location("_cmap_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cmap_generator"] = module
    spec.loader.exec_module(module)
    return module


generated = pytest.mark.skipif(
    not GENERATOR.exists(), reason=f"generator not present at {GENERATOR}"
)


@generated
def test_the_generator_is_importable_and_produces_something() -> None:
    """Fail closed: a generator that produces nothing would pass every check."""
    produced = _generator().generate_all()
    assert len(produced) >= 16, f"generator produced only {sorted(produced)}"
    assert len(BUILTIN_COLORMAP_NAMES) >= 16, "builtins.py exports almost nothing"


@generated
def test_regenerating_would_not_drop_a_shipped_colormap() -> None:
    """Every colormap `builtins.py` ships must be one the generator produces."""
    produced = set(_generator().generate_all())
    shipped = set(BUILTIN_COLORMAP_NAMES)
    lost = sorted(shipped - produced)
    assert not lost, (
        f"running scripts/generate_builtin_colormaps.py would DELETE {lost} from "
        f"builtins.py and colormap-data.ts. Add them to the generator (a plain "
        f"black-to-colour ramp goes in LINEAR_RAMPS) rather than hand-editing "
        f"the generated files, which is how this drift started."
    )


@generated
def test_the_generator_reproduces_every_shipped_lut_exactly() -> None:
    """Byte equality, not just name presence.

    These LUTs are baked into published scenes, so a generator change that
    shifts a colormap by one level is a data change, not a refactor.
    """
    produced = _generator().generate_all()
    differing = [
        name
        for name in sorted(set(BUILTIN_COLORMAP_NAMES) & set(produced))
        if not np.array_equal(get_builtin_lut(name), produced[name])
    ]
    assert not differing, (
        f"the generator no longer reproduces the shipped bytes for {differing}. "
        f"These LUTs are baked into published scenes — confirm the change is "
        f"intended before regenerating."
    )


@generated
def test_generator_categories_preserve_ui_groups() -> None:
    """Named UI groups must not silently fall into the remainder bucket."""
    generator = _generator()
    grouped = generator._categorise(generator.generate_all())

    assert grouped["BOP (Blue-Orange-Purple)"] == [
        "bop_blue",
        "bop_orange",
        "bop_purple",
    ]
    assert set(grouped["Microscopy linear ramps"]) == set(generator.LINEAR_RAMPS) - {
        "bop_blue",
        "bop_orange",
        "bop_purple",
    }
