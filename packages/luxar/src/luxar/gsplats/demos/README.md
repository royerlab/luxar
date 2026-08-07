# Gaussian Splatting Demos

Comprehensive demonstration suite for Luxar's Gaussian splatting implementation, showcasing 2D/3D/4D fitting, compression analysis, and advanced features.

## Quick Start

**New to Gaussian splatting?** Start here:
```bash
# 1. Simple API introduction
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_basic_fitting.py

# 2. See detailed metrics
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_performance_metrics.py

# 3. Explore compression interactively
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_2d_synthetic_blobs.py
```

**Headless mode** (for testing/CI): Add `--no-napari` flag to any demo

---

## Demo Categories

### **Learning & Tutorial (Start Here)**

| Demo | Purpose | Best For |
|------|---------|----------|
| **demo_basic_fitting.py** | Simple API introduction | Learning the `fit_gaussian_splats()` API |
| **demo_performance_metrics.py** | Convergence & quality metrics | Understanding optimization behavior |

### **Dimensional Progression (2D -> 3D -> 4D)**

| Demo | Dimensions | Data | Key Feature |
|------|------------|------|-------------|
| **demo_2d_synthetic_blobs.py** | 2D | Synthetic blobs | Oriented ellipse visualization |
| **demo_3d_synthetic_phantom.py** | 3D | Controlled phantom | Wireframe ellipsoid rendering |
| **demo_4d_hypercube.py** | 4D | Hypercube | nD algorithm validation |

**Learning path**: Work through 2D → 3D → 4D to understand dimensional scaling

### **Real Data Applications**

| Demo | Data Source | Domain | Unique Feature |
|------|-------------|--------|----------------|
| **demo_3d_dapi_microscopy.py** | IDR (remote zarr) | Microscopy | Remote data loading, OME-ZARR |
| **demo_splats_mitosis.py** | scikit-image | Biology | Histology imaging |
| **demo_splats_astronaut.py** | scikit-image | Photography | Facial features, textures |
| **demo_splats_coins.py** | scikit-image | Photography | Metallic surfaces |

### **Progressive Fitting**

| Demo | Base Demo | Focus |
|------|-----------|-------|
| **demo_progressive_fitting.py** | Synthetic | Core progressive fitting API |
| **demo_splats_mitosis_progressive.py** | Mitosis | Progressive fitting on biological data |
| **demo_splats_astronaut_progressive.py** | Astronaut | Progressive fitting on photo data |
| **demo_splats_coins_progressive.py** | Coins | Progressive fitting on coins image |
| **demo_3d_celegans_confocal_progressive.py** | C. elegans | Progressive fitting on 3D confocal |
| **demo_3d_dapi_progressive.py** | DAPI | Progressive fitting on 3D microscopy |

### **C. elegans Confocal**

| Demo | Focus | Demonstrates |
|------|-------|--------------|
| **demo_3d_celegans_confocal.py** | 3D confocal | Real 3D confocal microscopy fitting |
| **demo_3d_celegans_culling.py** | Culling | Splat culling on confocal data |

### **Advanced & Specialized**

| Demo | Focus | Demonstrates |
|------|-------|--------------|
| **demo_splats_mitosis_intgrad.py** | Algorithm | CLAHE seeding with intensity gradients |
| **demo_splats_mitosis_explicit_seeding.py** | Seeding | Explicit seed generation for mitosis |
| **demo_boundary_containment.py** | Boundaries | Boundary containment during fitting |
| **demo_tiled_fitting.py** | Tiling | Tiled fitting for large volumes |
| **demo_substitutive_lod_dapi.py** | Substitutive LOD | Cost-aware Lloyd vs amplitude-culling baseline |

### **Shared Helpers (not a demo)**

`_demo_common.py` holds the overlay/metric helpers the demos share —
`ellipse_polygon_from_L` (2D t-sigma contour), `ellipsoid_wireframe_from_L` (3D
t-sigma wireframe) and `psnr` (peak = target's dynamic range). Import them
(`from luxar.gsplats.demos._demo_common import ...`) rather than pasting another
copy into a new demo; they are unit-tested in `tests/test_demo_common.py`.

---

## Detailed Demo Descriptions

### **demo_basic_fitting.py**
**What it does**: Minimal working example of Gaussian splat fitting
- Uses high-level `fit_gaussian_splats()` API
- Automatic seed generation
- Dynamic operations (seeding, pruning) enabled by default
- Simple napari visualization

**Key concepts**: Standard Adam optimizer, early stopping, fixed-pool splat relocation

**Usage**: `python demo_basic_fitting.py [--no-napari] [--n-iters N] [--disable-dynamic]`

---

### **demo_performance_metrics.py**
**What it does**: Comprehensive performance benchmarking and quality analysis
- Detailed timing metrics (total time, iterations per second)
- Quality metrics (MSE, relative L2 error, PSNR)
- Early stopping analysis (iterations saved)
- Active splat counting (pruning effectiveness)

**Key concepts**: Convergence monitoring, quality assessment, benchmarking

**Usage**: `python demo_performance_metrics.py [--no-napari] [--n-iters N]`

---

### **demo_2d_synthetic_blobs.py**
**What it does**: Interactive 2D compression analysis with visual splat exploration
- Energy-based splat ranking (L2 importance)
- Interactive compression slider (all splats → minimal)
- Oriented 2σ ellipse overlays showing splat shapes
- Bits-per-pixel compression accounting

**Key concepts**: Compression by importance, full-covariance ellipses, interactive analysis

**Usage**: `python demo_2d_synthetic_blobs.py [--no-napari]`

**Controls**:
- Top slider: Compression level
- Toggle layers: Compare input vs reconstruction
- Cyan ellipses: Splat shapes and orientations

---

### **demo_3d_synthetic_phantom.py**
**What it does**: 3D volumetric compression with wireframe ellipsoid visualization
- Systematic 3D phantom with 15 controlled Gaussian blobs
- Full-covariance 3D ellipsoids (6 parameters each)
- Wireframe rendering (3 principal plane circles)
- MIP (Maximum Intensity Projection) rendering

**Key concepts**: 3D Gaussians, volumetric compression, ellipsoid wireframes

**Usage**: `python demo_3d_synthetic_phantom.py [--no-napari]`

**Controls**:
- Top slider: Compression level
- Mouse + Shift: Rotate 3D view
- Mouse wheel: Zoom

**Performance**: 1000 iterations (reduced for 3D computational cost)

---

### **demo_3d_dapi_microscopy.py**
**What it does**: Real DAPI-stained nuclear microscopy from Image Data Resource
- Remote zarr loading from IDR (https://idr.openmicroscopy.org/)
- OME-ZARR format handling (5D: T×C×Z×Y×X)
- Automatic downscaling to 128³ voxels
- Biological structure fitting

**Key concepts**: Real data challenges, remote loading, OME-ZARR, nuclear morphology

**Data**: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr
**Fallback**: Creates synthetic nucleus-like blobs if remote load fails

**Usage**: `python demo_3d_dapi_microscopy.py [--no-napari]`

---

### **demo_4d_hypercube.py**
**What it does**: 4D hypercube validation demonstrating nD scalability
- 4D data: (8, 64, 64, 64) = 2M hypervoxels
- 10-parameter covariance matrices (symmetric 4×4)
- 100 random 4D Gaussian blobs
- Complete nD pipeline validation

**Key concepts**: nD generalization, hypercube navigation, dimensional scalability

**Usage**: `python demo_4d_hypercube.py [--no-napari]`

**Performance**: 400 iterations (reduced for 4D computational cost)
**Technical**: Validates nD rendering, energy ranking, compression algorithms

---

### **demo_splats_astronaut.py**
**What it does**: Compression analysis on classic astronaut photograph
- Complex color image (converted to grayscale)
- Rich textures: facial features, helmet, fabric
- Interactive compression slider
- 2000 iterations for high quality

**Key concepts**: Photography compression, texture preservation, facial features

**Usage**: `python demo_splats_astronaut.py [--no-napari]`

---

### **demo_splats_coins.py**
**What it does**: Coins image with metallic textures and circular objects
- Metallic surface gradients
- Circular coin shapes
- Illumination variations
- Interactive compression analysis

**Key concepts**: Metallic textures, circular features, illumination

**Usage**: `python demo_splats_coins.py [--no-napari]`

---

### **demo_splats_mitosis.py**
**What it does**: Biological histology data compression analysis
- Human mitosis dataset from scikit-image
- Cell structures and chromatin patterns
- Interactive compression slider
- Biological feature preservation

**Key concepts**: Biological imaging, histology, cellular structures

**Usage**: `python demo_splats_mitosis.py [--no-napari]`

---

### **demo_splats_mitosis_intgrad.py**
**What it does**: Tests CLAHE-based seeding with artificial intensity gradient
- Creates challenging scenario: top dim (10%), bottom bright (100%)
- Validates CLAHE can discover structures in dim regions
- Before/after CLAHE visualization
- Residual balance analysis

**Key concepts**: CLAHE seeding, intensity gradients, adaptive histogram equalization

**Usage**: `python demo_splats_mitosis_intgrad.py [--no-napari]`

**Technical**: Validates that CLAHE-enhanced peak detection finds dim structures

---

### **demo_splats_mitosis_explicit_seeding.py**
**What it does**: Demonstrates explicit seed initialization using the seeding API
- Uses `seed_from_decomposition()`, `seed_from_grid()`, or `seed_from_edges()` to generate seeds with scale-informed Gaussian shapes before fitting

**Usage**: `python demo_splats_mitosis_explicit_seeding.py [--no-napari]`

---

### **demo_progressive_fitting.py**
**What it does**: Progressive (multi-pass) Gaussian splatting on synthetic data
- Each pass fits splats to the residual of the previous approximation, building a multi-LOD GSplatData representation from coarse to fine detail

**Usage**: `python demo_progressive_fitting.py [--no-napari]`

---

### **demo_splats_mitosis_progressive.py**
**What it does**: Progressive multi-pass fitting on the scikit-image human mitosis dataset
- Builds a multi-LOD representation where each pass captures progressively finer biological detail from the histology image

**Usage**: `python demo_splats_mitosis_progressive.py [--no-napari]`

---

### **demo_splats_astronaut_progressive.py**
**What it does**: Progressive fitting on the astronaut photograph
- Handles challenging textures (sharp edges, facial detail, helmet patterns) by starting with coarse structure and progressively adding finer detail across passes

**Usage**: `python demo_splats_astronaut_progressive.py [--no-napari]`

---

### **demo_splats_coins_progressive.py**
**What it does**: Progressive multi-pass fitting on the scikit-image coins dataset
- Each pass fits splats to the residual, building a coarse-to-fine representation of circular metallic surfaces

**Usage**: `python demo_splats_coins_progressive.py [--no-napari]`

---

### **demo_3d_celegans_confocal.py**
**What it does**: 3D Gaussian splatting on a single timepoint from a C. elegans embryo confocal dataset
- Handles anisotropic voxel spacing (5:1 Z-anisotropy: 0.75 um Z vs 0.15 um XY)

**Usage**: `python demo_3d_celegans_confocal.py [--no-napari]`

---

### **demo_3d_celegans_confocal_progressive.py**
**What it does**: Progressive multi-pass fitting on 3D C. elegans confocal data
- Combines progressive LOD fitting with real 3D confocal microscopy, building coarse-to-fine detail across passes

**Usage**: `python demo_3d_celegans_confocal_progressive.py [--no-napari]`

---

### **demo_3d_celegans_culling.py**
**What it does**: Contribution-based culling on pre-computed C. elegans Gaussian splats
- Compares full (unculled) reconstruction against increasingly aggressive culling levels to show quality-vs-size trade-offs

**Usage**: `python demo_3d_celegans_culling.py [--no-napari]`

---

### **demo_3d_dapi_progressive.py**
**What it does**: Progressive fitting on real 3D DAPI-stained nuclear microscopy from the Image Data Resource
- Multi-pass fitting on remote OME-ZARR data, building a multi-LOD representation of 3D nuclear structures

**Usage**: `python demo_3d_dapi_progressive.py [--no-napari]`

---

### **demo_boundary_containment.py**
**What it does**: Demonstrates the `boundary_penalty` parameter during optimization
- Adds a differentiable loss term that discourages splats from extending beyond the volume boundaries

**Usage**: `python demo_boundary_containment.py [--no-napari]`

---

### **demo_tiled_fitting.py**
**What it does**: Tiled fitting on a 3x3 grid using the cells3d max-projection
- Splits a large image into overlapping tiles with Hann cosine apodization, fits each tile independently, then concatenates results seamlessly

**Usage**: `python demo_tiled_fitting.py [--no-napari]`

---

### **demo_substitutive_lod_dapi.py**
**What it does**: Builds a substitutive LOD hierarchy from progressively-fitted gsplats of a DAPI nuclear volume
- Fits progressive gsplats to a moderate splat budget on a downscaled (64³) DAPI volume
- Builds a substitutive LOD hierarchy with `make_substitutive_lod()` at K=4, L=3 (cost-increment Lloyd refinement)
- Compares against an amplitude-culling baseline at matched per-level counts
- Reports relative L² error and per-level PSNR, mirroring supp-doc `substitutive_lod` Experiment C

**Usage**: `python demo_substitutive_lod_dapi.py [--no-napari]`

---

## Running Demos

### **Standard Execution (with napari visualization)**
```bash
cd packages/luxar/src/luxar/gsplats/demos
python demo_basic_fitting.py
```

### **From Project Root with Hatch**
```bash
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_basic_fitting.py
```

### **Headless Mode (no GUI)**
```bash
python demo_basic_fitting.py --no-napari
python demo_2d_synthetic_blobs.py --no-napari
python demo_4d_hypercube.py --no-napari
```

### **Custom Parameters**
```bash
# More iterations
python demo_basic_fitting.py --n-iters 2000

# Disable dynamic operations
python demo_basic_fitting.py --disable-dynamic

# Headless with custom iterations
python demo_performance_metrics.py --no-napari --n-iters 500
```

---

## Quick Reference Table

| Demo Name | Dim | Data Type | Iterations | Key Feature | Run Time |
|-----------|-----|-----------|------------|-------------|----------|
| demo_basic_fitting | 2D | Synthetic | 1000 | Simple API | ~5s |
| demo_performance_metrics | 2D | Synthetic | 300 | Metrics | ~3s |
| demo_2d_synthetic_blobs | 2D | Synthetic | 4000 | Compression | ~8s |
| demo_3d_synthetic_phantom | 3D | Synthetic | 1000 | 3D Ellipsoids | ~45s |
| demo_3d_dapi_microscopy | 3D | Real (IDR) | 6000 | Remote Zarr | ~90s |
| demo_4d_hypercube | 4D | Synthetic | 400 | nD Validation | ~60s |
| demo_splats_astronaut | 2D | Real (photo) | 2000 | Photography | ~12s |
| demo_splats_coins | 2D | Real (photo) | 4000 | Metallic | ~10s |
| demo_splats_mitosis | 2D | Real (bio) | 5000 | Histology | ~10s |
| demo_splats_mitosis_intgrad | 2D | Real (bio) | 2000 | CLAHE Test | ~15s |
| demo_splats_mitosis_explicit_seeding | 2D | Real (bio) | - | Explicit Seeding | ~10s |
| demo_progressive_fitting | 2D | Synthetic | - | Progressive API | ~10s |
| demo_splats_mitosis_progressive | 2D | Real (bio) | - | Progressive | ~15s |
| demo_splats_astronaut_progressive | 2D | Real (photo) | - | Progressive | ~15s |
| demo_splats_coins_progressive | 2D | Real (photo) | - | Progressive | ~12s |
| demo_3d_celegans_confocal | 3D | Real (confocal) | - | 3D Confocal | ~90s |
| demo_3d_celegans_confocal_progressive | 3D | Real (confocal) | - | Progressive 3D | ~120s |
| demo_3d_celegans_culling | 3D | Real (confocal) | - | Culling | ~30s |
| demo_3d_dapi_progressive | 3D | Real (IDR) | - | Progressive 3D | ~120s |
| demo_boundary_containment | 2D | Synthetic | - | Boundaries | ~10s |
| demo_tiled_fitting | 3D | Synthetic | - | Tiled Fitting | ~60s |
| demo_substitutive_lod_dapi | 3D | Real (IDR) | - | Substitutive LOD | ~60s |

*Run times are approximate on modern CPU (M1/M2 or recent Intel/AMD)*

---

## Recommended Learning Path

### **Beginner** (New to Gaussian Splatting)
1. **demo_basic_fitting.py** - Understand the API
2. **demo_performance_metrics.py** - See convergence behavior
3. **demo_2d_synthetic_blobs.py** - Explore compression interactively

### **Intermediate** (Understanding Dimensions)
4. **demo_splats_mitosis.py** - Real biological data
5. **demo_3d_synthetic_phantom.py** - Extend to 3D
6. **demo_3d_dapi_microscopy.py** - Real 3D microscopy

### **Advanced** (Specialized Features)
7. **demo_4d_hypercube.py** - nD algorithm validation
8. **demo_splats_mitosis_intgrad.py** - CLAHE seeding validation

---

## What Each Demo Demonstrates

### **Core Features (All Demos)**
- Standard PyTorch Adam optimizer with gradient dilution compensation
- Dynamic operations (fixed-pool splat relocation)
- Automatic seed generation with intelligent defaults
- Early stopping based on convergence criteria
- Structured logging with arbol
- Headless operation (`--no-napari` flag)

### **Unique Features by Demo**

**Compression Analysis** (with interactive slider):
- demo_2d_synthetic_blobs.py
- demo_3d_synthetic_phantom.py
- demo_3d_dapi_microscopy.py
- demo_4d_hypercube.py
- demo_splats_astronaut.py
- demo_splats_coins.py
- demo_splats_mitosis.py

**Oriented Ellipse Visualization**:
- demo_2d_synthetic_blobs.py (2D ellipses)
- demo_splats_astronaut.py (2D ellipses)
- demo_splats_coins.py (2D ellipses)
- demo_splats_mitosis.py (2D ellipses)

**3D Ellipsoid Wireframes**:
- demo_3d_synthetic_phantom.py (wireframe circles)
- demo_3d_dapi_microscopy.py (wireframe circles)

**Remote Data Loading**:
- demo_3d_dapi_microscopy.py (IDR via fsspec)

**Algorithm Validation**:
- demo_4d_hypercube.py (nD algorithms)
- demo_splats_mitosis_intgrad.py (CLAHE seeding)

---

## Visualization Features

### **Napari Layers (varies by demo)**
- **Input**: Original image/volume
- **Reconstruction**: Fitted Gaussian splat rendering
- **Residual**: Absolute error (input - reconstruction)
- **Ellipses/Wireframes**: Oriented splat shape visualization
- **Centers**: Splat center positions

### **Interactive Controls**
- **Compression slider** (axis 0): Explore quality vs compression trade-offs
- **Dimension sliders**: Navigate through 3D/4D data
- **Layer toggles**: Show/hide individual layers
- **3D rotation**: Mouse + Shift to rotate view
- **Zoom**: Mouse wheel

### **Text Overlays**
Most demos display real-time metrics:
- Number of splats (kept vs total)
- Model size (bits, bits-per-pixel/voxel)
- Compression percentage
- Reconstruction error (relative L2)

---

## Data Sources

### **Synthetic Data** (Generated)
- **binary_blobs**: scikit-image blob generation with Gaussian smoothing
- **3D phantom**: Controlled Gaussian blobs with known properties
- **4D hypercube**: Random 4D Gaussians for nD validation

### **Real Data** (scikit-image)
- **astronaut**: Classic 512×512 RGB photograph (cropped to 320×320)
- **coins**: Grayscale 303×384 with metallic textures
- **human_mitosis**: RGB histology (cropped to 256×256)

### **Real Data** (Remote)
- **DAPI microscopy**: IDR zarr (OME-ZARR 5D format, downscaled to 128³)

---

## Common Parameters

### **Optimization**
- `n_iters`: Maximum iterations (default varies: 300-2000)
- `lr`: Learning rate (default: 0.01 or auto)
- `max_abs_error`: Convergence threshold (default: 0.01 = 1% of range)
- `loss_type`: "l1" (default; robust to outliers, preserves sharp features), "mse", or "poisson"

### **Regularization**
- `l1_amp`: Amplitude sparsity (default: 0.1 × lr)
- `l1_diag`: Diagonal regularization (default: 0.01 × lr)

### **Dynamic Operations**
- `enable_dynamic_ops`: Enable/disable (default: True)
- `dynamic_config`: DynamicOpsConfig() for advanced control

### **Visualization**
- `napari_movie`: Record optimization movie (default: True when napari enabled)
- `movie_every`: Frame recording interval (default: 1-50 depending on demo)

---

## Testing & Validation

### **Compilation Check**
```bash
# Verify all demos compile
python -m py_compile demo_*.py
```

### **Quick Validation Run**
```bash
# Run all demos in headless mode with reduced iterations
for demo in demo_*.py; do
    echo "Testing: $demo"
    timeout 60 python "$demo" --no-napari --n-iters 50 2>&1 | grep -E "(✓|✅|Error)" || true
done
```

### **Full Suite**
```bash
# Run all demos with full visualization (interactive)
for demo in demo_basic_fitting.py demo_2d_synthetic_blobs.py demo_3d_synthetic_phantom.py; do
    python "$demo"
done
```

---

## Performance Expectations

### **Timing Guidelines (M1/M2/M3 MacBook Pro)**
- **2D demos** (256×256): 3-15 seconds for 1000 iterations
- **3D demos** (64³): 30-60 seconds for 1000 iterations
- **4D demos** (8×64³): 60-120 seconds for 400 iterations

### **Memory Usage**
- **2D**: ~200 MB
- **3D**: ~500 MB - 1 GB
- **4D**: ~1-2 GB

### **Device Recommendations**
- **CUDA GPU**: Best performance (if available)
- **CPU**: Recommended for MPS users (10× faster than MPS for this workload)
- **MPS**: Supported but slower due to PyTorch MPS limitations
  - See main README "Apple Silicon Performance Notes"
  - Use `device="cpu"` for better performance on M1/M2/M3/M4

---

## What to Look For

### **In Compression Demos**
- How reconstruction quality degrades with fewer splats
- Ellipse orientations align with image features
- Energy-ranked splats capture most important structures first
- Bits-per-pixel decreases with compression

### **In Performance Demos**
- Early stopping saves iterations when convergence criteria met
- Dynamic operations improve quality (compare with `--disable-dynamic`)
- Standard Adam + fixed-pool relocation maintains smooth convergence

### **In 3D/4D Demos**
- Wireframe ellipsoids align with 3D structures
- nD algorithms scale correctly (4D uses 16 floats/splat vs 2D's 7 floats/splat)
- MIP rendering reveals volumetric structure

### **In Real Data Demos**
- Algorithm handles noise and irregular shapes
- Biological/photographic features preserved
- Compression trade-offs on real-world textures

---

## Troubleshooting

### **"No splats were fitted"**
- Try lowering thresholds or increasing iterations
- Check data range (should be normalized)
- Enable `verbose=True` to see seeding details

### **Napari window doesn't open**
- Check napari installation: `pip install napari[all]`
- Try `--no-napari` flag to verify fitting works headless

### **Slow performance on Mac**
- Use `device="cpu"` instead of MPS for 10× speedup
- MPS has significant overhead for triangular solve operations
- See main gsplats README for Apple Silicon notes

### **Remote data loading fails** (demo_3d_dapi_microscopy.py)
- Requires internet connection
- Falls back to synthetic data automatically
- Check firewall/proxy settings

---

## Related Documentation

- **Main gsplats README**: `../README.md` - Full package overview
- **GLOSSARY**: `../GLOSSARY.md` - Terminology reference

---

## Contributing New Demos

When adding new demos, follow these standards:

1. **Naming**: `demo_<descriptive_name>.py` (avoid generic suffixes like "_napari")
2. **Header**: Use comprehensive docstring template (see existing demos)
3. **Arbol**: Use `aprint()` and `asection()` for all console output
4. **Headless**: Support `--no-napari` flag via `sys.argv`
5. **Documentation**: Update this README with new demo entry
6. **Cross-refs**: Add "Related demos" section in docstring

---

**Last updated**: 2026-06 (added demo_substitutive_lod_dapi; refreshed iteration counts)
