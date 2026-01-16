# luxar.demos - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-01-16

## Purpose

This package provides self-contained demonstration scripts showcasing Luxar's capabilities for data generation, visualization, and rendering features. Each demo is a standalone Python script that can be run independently.

---

## Core Concepts

### Self-Contained Principle

All demos follow the **self-contained principle**: each demo file must be understandable in isolation.

**Requirements**:
- All data generation logic implemented inline in the demo file
- No imports from other demos (except `launch_viewer` helper)
- No custom utility functions from within the demos package
- Single-file understanding: readers should understand entire workflow from one file
- Use only Luxar core APIs (Scene, Points, Lines, GSplats, etc.)

**Rationale**: Demos serve as learning examples. Users should be able to copy a single file and understand the complete workflow without chasing dependencies.

### Demo Architecture

Each demo follows a consistent pattern:

```
1. Generation    -> Algorithmic data generation (with optional seed for reproducibility)
2. Scene Setup   -> Create Scene, configure dimensions
3. Writing       -> Progressive writing to Zarr using LuxarZarrCompiler
4. Serving       -> Launch viewer via launch_viewer() helper
5. Cleanup       -> Automatic temporary directory cleanup (context manager)
```

---

## Data Structures

### Demo Output Structure

Demos generate standard Luxar Zarr archives:

```
demo_output.zarr/
  .zattrs           # Scene metadata, dimensions, content_hash
  .zmetadata        # Consolidated zarr metadata
  points/           # Point cloud data (if present)
    positions/
    colors/
    radii/
  lines/            # Line data (if present)
    vertices/
    segments/
  gsplats/          # Gaussian splats (if present)
    centers/
    cholesky/
    colors/
```

### Temporary File Management

Demos use context managers for automatic cleanup:

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "demo.zarr"
    # ... generate and write data ...
    launch_viewer(output_path)
    # Cleanup happens automatically after viewer closes
```

---

## Algorithms

### Common Generation Patterns

**1. Parametric Surfaces**
- Sphere, torus, Klein bottle
- UV parameterization with configurable resolution
- Color mapping from position or parameter

**2. Attractors**
- Lorenz attractor (chaotic system)
- Euler integration with configurable dt
- Time-based coloring

**3. Fractals**
- Mandelbulb (3D Mandelbrot analog)
- Iteration-based escape coloring
- Ray marching for surface detection

**4. Procedural Systems**
- L-systems for tree/plant generation
- Recursive subdivision
- Random seeding for reproducibility

**5. Physical Simulations**
- Particle systems with forces
- N-body gravitational simulation
- Fluid-like behaviors

---

## launch_viewer() Helper Function

**Location**: `luxar.utils.demos.launch_viewer()`
**Alias**: `from luxar.demos import launch_viewer`

### Purpose

Start a web server and open the viewer for a Zarr dataset.

### Signature

```python
def launch_viewer(
    zarr_path: PathLike,
    port: Optional[int] = None,
    open_browser: bool = True
) -> None:
    """
    Launch the Luxar viewer for a zarr dataset.

    Args:
        zarr_path: Path to the zarr archive to serve
        port: Port number (auto-discovers available port if None)
        open_browser: Whether to open browser automatically
    """
```

### Behavior

1. Uses `sys.executable -m luxar serve` to start server
2. Auto-discovers available port if not specified
3. Handles CORS and directory listing
4. Opens browser automatically (if enabled)
5. Graceful shutdown on Ctrl+C
6. Reuses the tested CLI server implementation

---

## Validation Rules

### Demo File Requirements

1. **Docstring**: Must have module-level docstring explaining:
   - What the demo visualizes
   - Key Luxar features demonstrated
   - Expected output description

2. **Reproducibility**: Must accept optional `seed` parameter for reproducible output

3. **Size Configuration**: Must have configurable data size (n_points, resolution, etc.)

4. **Main Guard**: Must use `if __name__ == "__main__":` pattern

### Example Template

```python
"""
Demo: <Name>

Demonstrates <feature> with <technique>.

Features showcased:
- <Feature 1>
- <Feature 2>

Output: <Description of visualization>
"""

from luxar import Scene, Points
from luxar.io import LuxarZarrCompiler
from luxar.demos import launch_viewer
import numpy as np
import tempfile
from pathlib import Path


def generate_data(n_points: int = 10000, seed: int | None = 42):
    """Generate demo data."""
    rng = np.random.default_rng(seed)
    # ... generation logic ...
    return positions, colors


def main():
    n_points = 10000
    seed = 42

    positions, colors = generate_data(n_points, seed)

    scene = Scene(name="Demo Scene")
    scene.add_points("demo_points", positions=positions, colors=colors)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "demo.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            compiler.write_scene(scene)

        launch_viewer(output_path)


if __name__ == "__main__":
    main()
```

---

## Cross-Language Compatibility

Demos generate standard Luxar zarr format, compatible with:
- TypeScript viewer (luxar-viewer)
- Python reading via `LuxarScene`
- Any zarr-compatible reader

See `luxar.encoding/SPECIFICATIONS.md` for format details.

---

## Related Specifications

- `luxar.core/SPECIFICATIONS.md` - Scene graph and data models
- `luxar.io/SPECIFICATIONS.md` - Writing and reading zarr archives
- `luxar.encoding/SPECIFICATIONS.md` - Array encoding formats
- `docs/guides/user/LUXAR_ZARR_FORMAT.md` - Complete format specification

---

## Changelog

- **v1.1.0** (2026-01-16): Enhanced specification
  - Added demo architecture documentation
  - Added launch_viewer() helper specification
  - Added self-contained principle documentation
  - Added common generation patterns
  - Added validation rules and template

- **v1.0.0** (2026-01-02): Initial specification
  - Basic package purpose
