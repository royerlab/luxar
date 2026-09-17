#!/usr/bin/env python3
"""Generate built-in colormap LUT data for Luxar.

Produces Python and TypeScript source files with pre-computed 256x3 uint8 LUT arrays.
Requires matplotlib for generation (not at runtime).

Usage:
    python scripts/generate_builtin_colormaps.py
"""

from pathlib import Path

import numpy as np

# ============================================================================
# Colormap definitions
# ============================================================================

# Linear ramps: simple (0,0,0) -> (R,G,B) gradients
LINEAR_RAMPS = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (255, 255, 255),
    # Added to builtins.py by hand after an earlier run and never folded back
    # here, so regenerating silently DELETED four shipped colormaps. Endpoints
    # recovered from the shipped LUTs, which are exact make_linear_ramp output
    # (verified bit-for-bit, and NOT copies of napari's similarly-named maps --
    # those differ by up to 135 per channel).
    "orange": (255, 165, 0),
    "bop_blue": (0, 38, 255),
    "bop_orange": (255, 128, 0),
    "bop_purple": (153, 0, 255),
}


def make_linear_ramp(r: int, g: int, b: int) -> np.ndarray:
    """Create a 256x3 linear ramp from black to (r, g, b)."""
    t = np.linspace(0, 1, 256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.round(t * r).astype(np.uint8)
    lut[:, 1] = np.round(t * g).astype(np.uint8)
    lut[:, 2] = np.round(t * b).astype(np.uint8)
    return lut


def make_fire() -> np.ndarray:
    """Fire colormap: black -> red -> yellow -> white."""
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    # Red channel: ramps up quickly
    lut[:, 0] = np.round(np.clip(t * 3, 0, 1) * 255).astype(np.uint8)
    # Green channel: ramps up in middle
    lut[:, 1] = np.round(np.clip((t - 0.33) * 3, 0, 1) * 255).astype(np.uint8)
    # Blue channel: ramps up last
    lut[:, 2] = np.round(np.clip((t - 0.67) * 3, 0, 1) * 255).astype(np.uint8)
    return lut


def make_ice() -> np.ndarray:
    """Ice colormap: black -> blue -> cyan -> white."""
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 2] = np.round(np.clip(t * 3, 0, 1) * 255).astype(np.uint8)
    lut[:, 1] = np.round(np.clip((t - 0.33) * 3, 0, 1) * 255).astype(np.uint8)
    lut[:, 0] = np.round(np.clip((t - 0.67) * 3, 0, 1) * 255).astype(np.uint8)
    return lut


def make_phase() -> np.ndarray:
    """Phase/cyclic colormap using HSV hue rotation."""
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    # HSV hue rotation with full saturation and value
    for i, h in enumerate(t):
        h6 = h * 6.0
        sector = int(h6) % 6
        f = h6 - int(h6)
        if sector == 0:
            r, g, b = 1.0, f, 0.0
        elif sector == 1:
            r, g, b = 1.0 - f, 1.0, 0.0
        elif sector == 2:
            r, g, b = 0.0, 1.0, f
        elif sector == 3:
            r, g, b = 0.0, 1.0 - f, 1.0
        elif sector == 4:
            r, g, b = f, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, 1.0 - f
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


def get_matplotlib_colormap(name: str) -> np.ndarray:
    """Sample a matplotlib colormap to 256x3 uint8."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name)
    t = np.linspace(0, 1, 256)
    rgba = cmap(t)
    return (rgba[:, :3] * 255).round().astype(np.uint8)


# ============================================================================
# Generation
# ============================================================================


def generate_all() -> dict[str, np.ndarray]:
    """Generate all built-in colormaps."""
    colormaps: dict[str, np.ndarray] = {}

    # Linear ramps
    for name, (r, g, b) in LINEAR_RAMPS.items():
        colormaps[name] = make_linear_ramp(r, g, b)

    # Domain-specific
    colormaps["fire"] = make_fire()
    colormaps["ice"] = make_ice()
    colormaps["phase"] = make_phase()

    # Matplotlib-based (perceptual + diverging)
    matplotlib_maps = [
        "viridis",
        "inferno",
        "plasma",
        "turbo",
        "RdBu",
        "coolwarm",
    ]
    for name in matplotlib_maps:
        colormaps[name] = get_matplotlib_colormap(name)

    return colormaps


#: Presentation grouping for the generated files. Only affects the comment
#: headers and the order of emission -- `_categorise` guarantees that every
#: generated colormap lands in exactly one group, so this cannot drop one.
_CATEGORY_ORDER: dict[str, list[str]] = {
    "BOP (Blue-Orange-Purple)": ["bop_blue", "bop_orange", "bop_purple"],
    "Perceptually uniform": ["viridis", "inferno", "plasma", "turbo"],
    "Domain-specific": ["fire", "ice", "phase"],
    "Diverging": ["RdBu", "coolwarm"],
}


def _categorise(colormaps: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Group every generated colormap, with the linear ramps as the remainder.

    The writers used to iterate a hardcoded name list parallel to the one that
    decides what gets GENERATED. The two drifted: `orange`, `bop_blue`,
    `bop_orange` and `bop_purple` were hand-added to the generated files and
    folded back into neither, so re-running this script silently deleted four
    shipped colormaps from both the Python and the TypeScript output.

    Deriving the grouping from `colormaps` removes the second list. The
    assertion below removes the possibility of a third.
    """
    grouped = {
        name: [n for n in names if n in colormaps]
        for name, names in _CATEGORY_ORDER.items()
    }
    claimed = {n for names in grouped.values() for n in names}
    grouped["Microscopy linear ramps"] = [n for n in colormaps if n not in claimed]

    written = {n for names in grouped.values() for n in names}
    missing = set(colormaps) - written
    assert not missing, f"generated but never written: {sorted(missing)}"
    assert set(grouped["Microscopy linear ramps"]) == set(LINEAR_RAMPS) - set(
        _CATEGORY_ORDER["BOP (Blue-Orange-Purple)"]
    ), (
        "a new non-ramp colormap needs a _CATEGORY_ORDER entry; otherwise it "
        "is incorrectly filed under Microscopy linear ramps"
    )
    # Order the sections so the ramps come first, as they did before.
    return {
        "Microscopy linear ramps": grouped["Microscopy linear ramps"],
        **{k: v for k, v in grouped.items() if k != "Microscopy linear ramps"},
    }


def format_lut_python(name: str, lut: np.ndarray) -> str:
    """Format a LUT as a Python bytes literal."""
    data = lut.tobytes()
    return f'    "{name}": {data!r},'


def format_lut_typescript(name: str, lut: np.ndarray) -> str:
    """Format a LUT as a TypeScript Uint8Array."""
    values = lut.ravel().tolist()
    # Format as array of numbers, ~20 values per line
    lines = []
    for i in range(0, len(values), 30):
        chunk = values[i : i + 30]
        lines.append("    " + ",".join(str(v) for v in chunk) + ",")
    array_body = "\n".join(lines)
    return f'  "{name}": new Uint8Array([\n{array_body}\n  ]),'


def write_python_builtins(colormaps: dict[str, np.ndarray], path: str) -> None:
    """Write Python builtins.py file."""
    # Categories for documentation
    categories = _categorise(colormaps)

    lines = [
        '"""Built-in colormap lookup tables.',
        "",
        "Auto-generated by scripts/generate_builtin_colormaps.py",
        "Each colormap is stored as raw bytes: 256 entries x 3 channels (RGB) = 768 bytes.",
        '"""',
        "",
        "from typing import Dict",
        "",
        "import numpy as np",
        "",
        "# fmt: off",
        "",
    ]

    # Write categorized colormaps
    for category, names in categories.items():
        lines.append(f"# {category}")
        lines.append(f"# {'=' * len(category)}")
        lines.append("")
        for name in names:
            lut = colormaps[name]
            data = lut.tobytes()
            lines.append(f"_{name.upper()}_BYTES = {data!r}")
            lines.append("")

    # Write the lookup dict
    lines.append("# fmt: on")
    lines.append("")
    lines.append("")
    lines.append("def _bytes_to_lut(data: bytes) -> np.ndarray:")
    lines.append('    """Convert raw bytes to (256, 3) uint8 numpy array."""')
    lines.append(
        "    return np.frombuffer(data, dtype=np.uint8).reshape(256, 3).copy()"
    )
    lines.append("")
    lines.append("")

    # Build BUILTIN_COLORMAPS dict
    lines.append("# Lazy cache for numpy arrays")
    lines.append("_cache: Dict[str, np.ndarray] = {}")
    lines.append("")
    lines.append("")
    lines.append("# Map of colormap name -> raw bytes")
    lines.append("_COLORMAP_BYTES: Dict[str, bytes] = {")
    for category, names in categories.items():
        lines.append(f"    # {category}")
        for name in names:
            lines.append(f'    "{name}": _{name.upper()}_BYTES,')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def get_builtin_lut(name: str) -> np.ndarray:")
    lines.append('    """Get a built-in colormap as a (256, 3) uint8 numpy array.')
    lines.append("")
    lines.append("    Returns a copy to prevent mutation of the cached array.")
    lines.append('    """')
    lines.append("    if name not in _COLORMAP_BYTES:")
    lines.append('        raise KeyError(f"Unknown built-in colormap: {name!r}")')
    lines.append("    if name not in _cache:")
    lines.append("        _cache[name] = _bytes_to_lut(_COLORMAP_BYTES[name])")
    lines.append("    return _cache[name].copy()")
    lines.append("")
    lines.append("")
    lines.append("BUILTIN_COLORMAP_NAMES: list[str] = list(_COLORMAP_BYTES.keys())")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    total_bytes = sum(len(colormaps[n].tobytes()) for n in colormaps)
    print(f"Wrote {path} ({len(colormaps)} colormaps, ~{total_bytes // 1024}KB data)")


def write_typescript_data(colormaps: dict[str, np.ndarray], path: str) -> None:
    """Write TypeScript colormap-data.ts file."""
    categories = _categorise(colormaps)

    lines = [
        "/**",
        " * Built-in colormap lookup tables.",
        " *",
        " * Auto-generated by scripts/generate_builtin_colormaps.py",
        " * Each colormap is 256 entries x 3 channels (RGB) = 768 bytes.",
        " *",
        " * @module rendering/colormap-data",
        " */",
        "",
        "/* eslint-disable */",
        "// prettier-ignore",
        "export const BUILTIN_COLORMAPS: Record<string, Uint8Array> = {",
    ]

    for category, names in categories.items():
        lines.append(f"  // {category}")
        for name in names:
            lines.append(format_lut_typescript(name, colormaps[name]))

    lines.append("};")
    lines.append("")
    lines.append("/** All available built-in colormap names */")
    lines.append(
        "export const BUILTIN_COLORMAP_NAMES: string[] = Object.keys(BUILTIN_COLORMAPS);"
    )
    lines.append("")
    lines.append("/** Colormap categories for UI organization */")
    lines.append("// prettier-ignore")
    lines.append("export const COLORMAP_CATEGORIES: Record<string, string[]> = {")
    for category, names in categories.items():
        names_str = ", ".join(f"'{n}'" for n in names)
        lines.append(f"  '{category}': [{names_str}],")
    lines.append("};")
    # No trailing "" here: the join below already terminates the file with a
    # single newline. Appending one produced a blank final line that the
    # end-of-file-fixer pre-commit hook then stripped, so every regeneration
    # left a one-line diff and the output was never idempotent.

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {path}")


def _assert_names_match_contract(colormaps: dict[str, np.ndarray]) -> None:
    """Refuse to write a name set that differs from the format contract.

    ``format-contract/contract.yaml::builtin_colormaps`` is what the Python
    writer validates a ``colormap`` attr against and what the viewer's union
    type is generated from. Both generated LUT files are projections of the SAME
    name list, so a colormap added here but not there (or vice versa) would ship
    a name one side cannot resolve.
    """
    import yaml

    contract = Path(__file__).resolve().parent.parent / "format-contract/contract.yaml"
    declared = list(yaml.safe_load(contract.read_text())["builtin_colormaps"])
    generated = list(colormaps)
    if generated != declared:
        raise SystemExit(
            "generate_builtin_colormaps: generated names differ from "
            f"format-contract/contract.yaml::builtin_colormaps.\n  generated: "
            f"{generated}\n  contract:  {declared}\nEdit the contract first "
            "(then `make gen-contract`), so both languages learn the name."
        )


if __name__ == "__main__":
    print("Generating built-in colormaps...")
    colormaps = generate_all()
    _assert_names_match_contract(colormaps)

    # Verify all are (256, 3) uint8
    for name, lut in colormaps.items():
        assert lut.shape == (256, 3), f"{name}: shape {lut.shape}"
        assert lut.dtype == np.uint8, f"{name}: dtype {lut.dtype}"
        print(f"  {name}: range [{lut.min()}, {lut.max()}]")

    write_python_builtins(
        colormaps,
        "packages/luxar/src/luxar/colormaps/builtins.py",
    )
    write_typescript_data(
        colormaps,
        "packages/luxar-viewer/src/rendering/colormap-data.ts",
    )
    print("Done!")
