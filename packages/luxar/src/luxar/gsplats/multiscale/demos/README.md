# luxar.gsplats.multiscale.demos

Runnable demo scripts for the multi-scale image decomposition package
(`luxar.gsplats.multiscale`). Each script loads (or synthesises) some data,
runs `decompose_image`, prints an energy-distribution / convergence report to
the console, and — where applicable — opens an interactive napari viewer for
side-by-side inspection of the scale components.

## Scope

- Demonstrate `decompose_image` on 2D images, 3D volumes, and real microscopy
  data, across the four built-in initialization methods.
- Provide console-only (headless) code paths so the demos double as smoke
  tests in CI: pass `--no-napari` (or `--no-viz`) to skip the GUI.

All demos import the public API from the parent package:

```python
from luxar.gsplats.multiscale import (
    decompose_image,
    show_optimization_movie,
    upsample_for_visualization,
)
```

## Files

```
demos/
├── demo_compare_initialization.py    # Compare pyramid/uniform/coarse/finest init (ASCII plots)
├── demo_decompose_2d.py              # 2D astronaut / synthetic image → napari
├── demo_decompose_2d_mitosis.py      # 2D human-mitosis histology (scikit-image) → napari
├── demo_decompose_3d.py              # 3D synthetic volume → napari (ndisplay=3)
└── demo_decompose_3d_dapi_nuclei.py  # 3D DAPI nuclei from IDR (full resolution) → napari
```

| Script | Data | Default scales | Iters | Headless flag |
|--------|------|----------------|-------|---------------|
| `demo_compare_initialization.py` | astronaut (cropped 256²) | `[1, 2, 4, 8]` | 500 | `--no-viz` |
| `demo_decompose_2d.py` | astronaut, synthetic fallback | `[1, 2, 4, 8, 16, 32]` | 5000 | `--no-napari` |
| `demo_decompose_2d_mitosis.py` | human mitosis (cropped 256²) | `[1, 2, 4, 8, 16]` | 5000 | `--no-napari` |
| `demo_decompose_3d.py` | synthetic 64³ volume | `[1, 2, 4, 8, 16]` | 5000 | `--no-napari` |
| `demo_decompose_3d_dapi_nuclei.py` | IDR DAPI nuclei (full res) | `[1, 2, 4, 8]` | 3000 | `--no-napari` |

Each script exposes its knobs as module-level constants near the top
(`SCALES`, `N_ITERS`, `DEVICE`, and per-demo extras such as `VOLUME_SIZE` or
`DAPI_CHANNEL`) — edit these to retarget a demo. `DEVICE = None` auto-selects
the backend; set `"cuda"`, `"cpu"`, or `"mps"` to force one.

## Running

```bash
# Compare the four initialization methods (ASCII convergence plot + summary table)
hatch run python -m luxar.gsplats.multiscale.demos.demo_compare_initialization

# 2D decomposition with interactive napari viewer
hatch run python -m luxar.gsplats.multiscale.demos.demo_decompose_2d

# Headless (no GUI) — useful for smoke testing
hatch run python -m luxar.gsplats.multiscale.demos.demo_decompose_2d --no-napari
hatch run python -m luxar.gsplats.multiscale.demos.demo_compare_initialization --no-viz
```

## What each demo shows

### `demo_compare_initialization.py`
Runs `decompose_image` four times — once per `init_method`
(`"pyramid"`, `"uniform"`, `"coarse"`, `"finest"`) — on a cropped astronaut
image and renders pure-ASCII output: a colour-coded convergence plot of
reconstruction loss over iterations, a summary table (init loss, final loss,
time, coarse-scale energy %), and per-method energy-distribution bar charts.
Use it to compare convergence speed and quality of the initialization
strategies. Headless via `--no-viz`.

### `demo_decompose_2d.py` / `demo_decompose_2d_mitosis.py`
Decompose a 2D image into the configured scales with movie recording enabled
(`napari_movie`, `movie_every`). After printing the energy distribution, each
scale is upsampled back to full resolution via `upsample_for_visualization`
(using the interpolation mode reported in `stats`), then displayed in a napari
grid alongside the original, the reconstruction (sum of scales), and the
absolute/signed residuals. The convergence movie is replayed with
`show_optimization_movie` once the viewer is closed.

### `demo_decompose_3d.py`
Builds a synthetic `VOLUME_SIZE³` test volume (mixed low/medium/high frequency
components), decomposes it, and opens napari in 3D (`ndisplay=3`) with MIP
rendering. Also prints a compression-potential table (voxels and storage saved
per scale) and a console summary that is shown even in `--no-napari` mode.

### `demo_decompose_3d_dapi_nuclei.py`
Loads real DAPI-stained nuclei from the Image Data Resource (IDR) OME-Zarr
store at `https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr` via
`fsspec`, at **full resolution** (no downscaling). Handles 3D/4D/5D OME-Zarr
layouts, extracts `TIME_POINT`/`DAPI_CHANNEL`, and falls back to a synthetic
nucleus phantom if the remote store is unreachable. Reports overall
multi-scale compression vs. the original volume.

## Dependencies

Beyond `luxar[gsplats]`, the demos require:

- `napari` — interactive visualization (all demos except the ASCII-only
  `demo_compare_initialization.py`)
- `scikit-image` — sample images (`astronaut`, `human_mitosis`); the 2D demo
  degrades to a synthetic image if unavailable
- `zarr` + `fsspec` — remote OME-Zarr loading (`demo_decompose_3d_dapi_nuclei.py`)

See the parent package README at `../README.md` for the decomposition API and
algorithm details.
