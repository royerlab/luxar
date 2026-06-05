# Colormaps

Colormap (CLUT) support for mapping scalar data to RGB colors in Luxar scenes.

## Key Functions

- `resolve_colormap(name_or_array)` — Resolve a colormap name or custom array to a `(256, 3)` uint8 LUT
- `BUILTIN_COLORMAP_NAMES` — List of all built-in colormap names

## Built-in Colormaps

| Category | Names |
|----------|-------|
| Microscopy linear ramps | green, magenta, cyan, red, blue, yellow, gray, orange |
| BOP (Blue-Orange-Purple) | bop_blue, bop_orange, bop_purple |
| Perceptually uniform | viridis, inferno, plasma, turbo |
| Domain-specific | fire, ice, phase (cyclic) |
| Diverging | RdBu, coolwarm |

## Usage

```python
from luxar.colormaps import resolve_colormap, BUILTIN_COLORMAP_NAMES

# Built-in by name
lut = resolve_colormap("viridis")  # (256, 3) uint8

# Matplotlib name (if installed)
lut = resolve_colormap("cividis")

# Colorcet name (if installed)
lut = resolve_colormap("bgy")

# Custom numpy array
import numpy as np
custom = np.random.rand(128, 3).astype(np.float32)  # [0, 1] range
lut = resolve_colormap(custom)  # Resampled to (256, 3) uint8
```

## Integration with Scene Nodes

Colormaps are set as node attributes via `colormap=` on `add_points()`, `add_lines()`, `add_gsplats()`:

```python
# GSplats: amplitudes are the scalar source (no extra data needed)
scene.add_gsplats("channel_gfp", centers, amplitudes, cholesky, colormap="green")

# Points: explicit scalars array for colormap lookup
scene.add_points("pts", positions, scalars=values, colormap="viridis")
```

When `colormap` is set, `colors` must not be provided (mutually exclusive).

## Resolution Order

1. Built-in colormaps (instant, no dependencies)
2. Matplotlib (`matplotlib.colormaps[name]`, falling back to `pyplot.get_cmap`) — if installed
3. Colorcet (`colorcet.palette`) — if installed
4. `ValueError` with helpful message

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports (`resolve_colormap`, `BUILTIN_COLORMAP_NAMES`) |
| `registry.py` | `resolve_colormap()` and resolution logic |
| `builtins.py` | Pre-generated 256x3 uint8 LUT data, `get_builtin_lut()`, `BUILTIN_COLORMAP_NAMES` |
| `tests/test_registry.py` | Tests covering all resolution paths |
