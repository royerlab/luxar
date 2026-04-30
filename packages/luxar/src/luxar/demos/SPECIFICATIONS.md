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
    output_path: Union[str, Path],
    open_browser: bool = True,
) -> None:
    """
    Launch the Luxar viewer for a zarr dataset.

    Args:
        output_path: Path to the .zarr dataset to view
        open_browser: Whether to open browser automatically
    """
```

### Behavior

1. Uses `sys.executable -m luxar serve <path> --viewer` to start server
2. Appends `--open` flag if `open_browser=True`
3. Port discovery is handled by the CLI serve command internally
4. Handles `KeyboardInterrupt` for graceful Ctrl+C shutdown
5. Provides helpful error messages if `luxar` is not installed or viewer not built
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

from luxar import LuxarZarrCompiler, Dimensions
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

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "demo.zarr"
        dims = Dimensions.default_3d()
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("demo_points", positions=positions, colors=colors)

        launch_viewer(output_path)


if __name__ == "__main__":
    main()
```

---

## Demo Inventory

### Basic Visualization

| File | Description |
|------|-------------|
| `demo_lorenz.py` | Lorenz attractor trajectory with time-based color gradient |
| `demo_rainbow_sphere.py` | Fibonacci spiral sphere with rainbow color gradient |
| `demo_cubic_array.py` | Dense 100-cubed cubic array for depth-of-field testing |
| `demo_sharpness_showcase.py` | Comprehensive showcase of point sharpness feature variations |
| `demo_volumetric_cloud.py` | Volumetric cloud structures using fractal noise density |

### Mathematical & Procedural

| File | Description |
|------|-------------|
| `demo_mandelbulb.py` | Mandelbulb 3D fractal with distance estimation and iteration coloring |
| `demo_spiral_galaxy.py` | Multi-armed logarithmic spiral galaxy with realistic star populations |
| `demo_lsystem_forest.py` | Procedural L-system trees showcasing the Lines node type |
| `demo_quasicrystal_3d.py` | 3D aperiodic quasicrystal via cut-and-project from 6D |
| `demo_turing_patterns.py` | Gray-Scott reaction-diffusion Turing patterns with temporal evolution |
| `demo_quantum_orbitals.py` | Hydrogen atom electron probability density (quantum orbitals) |

### Physics Simulations

| File | Description |
|------|-------------|
| `demo_particle_collision.py` | Particle physics collision detector visualization (static) |
| `demo_particle_collision_animated.py` | Time-animated particle collision with growing tracks |

### Molecular & Structural Biology

| File | Description |
|------|-------------|
| `demo_atp_synthase.py` | ATP Synthase molecular structure with color-coded subunits |
| `demo_nuclear_pore_complex.py` | Nuclear pore complex with 8-fold rotational symmetry from PDB |
| `demo_bioluminescent_ocean.py` | Bioluminescent ocean scene with jellyfish and plankton |

### Scientific Data (Real Datasets)

| File | Description |
|------|-------------|
| `demo_gaia_milky_way_3m.py` | 3 million real stars from Gaia DR3 catalog |
| `demo_gaia_milky_way_8m.py` | 8 million real stars from pre-computed CSV dataset |
| `demo_earthquakes_3d.py` | Global earthquake visualization from USGS on 3D Earth sphere |
| `demo_cosmicflows_laniakea.py` | Cosmicflows-4 galaxies and RK4 velocity-field streamlines in supergalactic coordinates |
| `demo_storm_3d_microtubules.py` | 3D STORM super-resolution microscopy of microtubules |

### Embeddings & High-Dimensional Data

| File | Description |
|------|-------------|
| `demo_arxiv_paper_embeddings.py` | ArXiv paper embeddings in 3D semantic space |
| `demo_arxiv_embeddings_kaggle.py` | ArXiv embeddings from Kaggle OpenAI dataset |
| `demo_protein_embeddings_cafa5.py` | 142k protein ProtT5 embeddings in 3D function landscape |
| `demo_zebrahub_integrated_cells.py` | 95k zebrafish single cells 3D UMAP with categorical navigation |
| `demo_mouse_multiome_peak_umap.py` | ~192k mouse single-cell ATAC-seq peaks 3D UMAP |
| `demo_human_multiome_peak_umap.py` | ~1M human single-cell ATAC-seq peaks 3D UMAP |
| `demo_zebrahub_multiome_peak_umap.py` | 640k zebrafish ATAC-seq peaks 3D UMAP |

### Multi-Dimensional (nD)

| File | Description |
|------|-------------|
| `demo_5d_spiral_galaxy.py` | 5D spiral galaxy with time evolution and channel dimensions |
| `demo_4d_fractals.py` | 4D geometric fractal explorer with categorical dimension |
| `demo_network_performance.py` | Large 4D dataset for network performance testing |

### Gaussian Splatting (gsplats)

| File | Description |
|------|-------------|
| `demo_gsplats_3d_organoid_dapi_nuclei.py` | 3D organoid DAPI nuclei from IDR microscopy data |
| `demo_gsplats_3d_organoid_multichannel.py` | Multi-channel 3D organoid from IDR microscopy data |
| `demo_gsplats_3d_tribolium_embryo.py` | 3D Tribolium beetle embryo light-sheet volume |
| `demo_gsplats_4d_zebrafish_timelapse.py` | 4D zebrafish embryo time-lapse confocal with time slider |
| `demo_gsplats_3d_kidney_multichannel_toggles.py` | 3D multi-channel kidney with boolean toggles (legacy pattern) |
| `demo_gsplats_4d_celegans_tracking.py` | 4D C. elegans nuclei tracking with gsplats and polylines |

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
