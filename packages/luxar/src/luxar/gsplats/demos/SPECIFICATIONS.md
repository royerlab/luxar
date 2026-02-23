# Gaussian Splatting Demos

## Demo Best Practices

### File Structure & Naming
- Use descriptive filenames following pattern: `demo_[purpose]_[details].py`
- Include shebang line for executable scripts: `#!/usr/bin/env python3`
- Add module-level docstring explaining demo purpose and key features

### Import Organization
- Group imports logically: standard library, third-party (napari, numpy, sklearn), luxar components
- Use specific imports to make dependencies clear
- Always import `arbol` components for console output

### Configuration Management
- Place all configurable parameters at the top in a clearly marked section
- Use descriptive UPPER_CASE names for constants (e.g., `USE_POISSON`, `N_ITERS`, `DEVICE`)
- Include inline comments explaining parameter purpose and reasonable ranges
- Example configuration block:
```python
# ======= Demo knobs =======
USE_POISSON = True  # True: Poisson deviance; False: MSE
L1_AMP = 0.001     # L1 regularization strength
N_ITERS = 500      # Number of optimization iterations
DEVICE = None      # None -> auto; or "cuda"/"cpu"/"mps:0"
# ==========================
```

### Console Output Standards
- **MANDATORY**: Use `arbol` for ALL console output instead of `print()`
- Use `aprint()` for individual messages
- Use `asection()` context manager for hierarchical organization of multi-step processes
- Set reasonable nesting depth: `Arbol.max_depth = 3`
- Structure major operations with clear section titles:
  - "Data Generation" / "Creating test data"
  - "Candidate Generation" / "Finding candidates"
  - "Fitting Gaussian splats"
  - "Visualization Preparation"

### Argument Parsing
- Include `argparse` for runtime flexibility with standard options:
  - `--no-napari`: Disable visualization for headless testing
  - `--n-iters`: Number of optimization iterations
  - `--disable-dynamic`: Disable dynamic operations (enabled by default)
- Provide sensible defaults and descriptive help text
- Structure as `main()` function when using argument parsing

### Error Handling & Validation
- Check for empty results and provide meaningful error messages:
```python
if len(seeds.centers) == 0:
    raise RuntimeError("No seeds found; try lowering thresholds.")
if len(amps) == 0:
    raise RuntimeError("No splats were fitted; try lowering thresholds or increasing iterations.")
```
- Use `RuntimeError` with descriptive messages when operations fail
- Validate critical assumptions before proceeding

### Napari Integration Patterns
- Use consistent viewer setup and layer management
- Standard layer names: "input"/"Target", "reconstruction", "absolute residual"/"Absolute Error"
- Consistent colormap choices:
  - Data layers: "magma", "viridis", "plasma"
  - Residuals/errors: "turbo", "hot"
- Include interactive text overlays with performance metrics
- For 3D demos, use `viewer = napari.Viewer(ndisplay=3)` and set good camera angles
- Connect slider events for interactive exploration of results

### Performance & Visualization
- Use appropriate data types (`np.float32`) for memory efficiency
- Include timing information and convergence metrics in output
- For 3D visualizations, limit wireframes/shapes to reasonable numbers (e.g., max 50)
- Provide navigation tips and usage instructions in console output

### Code Organization
- Helper functions should have clear docstrings with parameter and return descriptions
- Use descriptive variable names that make the mathematical operations clear
- Include inline comments for complex mathematical computations (e.g., energy scoring, ellipse generation)
- Group related functionality together (e.g., all compression analysis code in one section)

### Data Processing Standards
- Use robust data normalization techniques (e.g., percentile-based rescaling)
- Apply appropriate filtering and preprocessing (e.g., `filters.gaussian()` for synthetic data)
- Include data range and shape information in console output

### Mathematical Computations
- Document mathematical formulas in comments, especially for energy scoring and shape generation
- Use stable numerical methods (avoid direct matrix inversion, use triangular solves)
- Include validation of computed results where appropriate

These practices ensure consistency, maintainability, and usability across all demo files in the gsplats package.

## List of Demos

### Available Demo Files

1. **demo_basic_fitting.py**
   - **Description**: Clean demonstration of the high-level `fit_gaussian_splats()` API with optional dynamic operations (fixed-pool splat relocation). Uses synthetic blob data to showcase the recommended interface for Gaussian splat fitting with comprehensive structured logging.
   - **Usage**: `python demo_basic_fitting.py [--no-napari] [--n-iters N] [--disable-dynamic]`
   - **Best for**: Learning the recommended API and understanding dynamic splat relocation during optimization

2. **demo_performance_metrics.py**
   - **Description**: Comprehensive performance benchmarking demo with detailed timing metrics, convergence analysis, and quality assessments on synthetic 2D blob data. Shows convergence speed, final reconstruction quality (MSE, PSNR, relative L2 error), and active splat counting with early stopping capabilities.
   - **Usage**: `python demo_performance_metrics.py [--no-napari] [--n-iters N]`
   - **Best for**: Understanding optimizer performance, convergence behavior, and quality metrics

3. **demo_2d_synthetic_blobs.py**
   - **Description**: Interactive 2D compression analysis featuring a napari viewer with animated compression slider that shows reconstruction quality vs model complexity trade-offs. Displays oriented ellipse overlays representing 2σ contours of fitted Gaussians, with real-time bit-per-pixel analysis and energy-based splat ranking for progressive compression visualization.
   - **Usage**: `python demo_2d_synthetic_blobs.py [--no-napari]`
   - **Best for**: Understanding compression trade-offs, energy ranking, and visualizing oriented 2D Gaussian representations

4. **demo_3d_synthetic_phantom.py**
   - **Description**: 3D volumetric Gaussian splatting demo with interactive 3D napari viewer featuring wireframe ellipsoid visualization. Fits full-covariance 3D Gaussians to synthetic volumetric blob data and provides compression analysis with wireframe representations of 3D ellipsoids, demonstrating volumetric data reconstruction and progressive compression in three dimensions.
   - **Usage**: `python demo_3d_synthetic_phantom.py [--no-napari]`
   - **Best for**: Exploring 3D Gaussian splatting, volumetric data compression, and understanding full-covariance 3D ellipsoid fitting

5. **demo_3d_dapi_microscopy.py**
   - **Description**: Real DAPI-stained nuclear microscopy data from Image Data Resource (IDR). Demonstrates remote zarr loading, OME-ZARR format handling, and 3D Gaussian fitting to biological structures with automatic downscaling.
   - **Usage**: `python demo_3d_dapi_microscopy.py [--no-napari]`
   - **Best for**: Working with real microscopy data, remote data loading, and biological structure fitting

6. **demo_4d_hypercube.py**
   - **Description**: 4D hypercube validation demonstrating complete nD pipeline with 4-dimensional data. Validates auto-candidate generation, dynamic operations, and compression in 4D.
   - **Usage**: `python demo_4d_hypercube.py [--no-napari]`
   - **Best for**: Understanding nD scalability and 4D data handling

7. **demo_splats_mitosis.py**
   - **Description**: Real-world application demo using the scikit-image human mitosis histology dataset. Demonstrates Poisson deviance loss (appropriate for biological imaging), contrast normalization, and tissue-optimized parameters with interactive compression analysis. Shows practical application to biological microscopy data with specialized preprocessing for brightfield histology images.
   - **Usage**: `python demo_splats_mitosis.py [--no-napari]`
   - **Best for**: Seeing practical application to biological imaging, understanding Poisson loss, and histology-specific preprocessing

8. **demo_splats_astronaut.py**
   - **Description**: Astronaut photo compression analysis with full-covariance fitting and interactive visualization
   - **Usage**: `python demo_splats_astronaut.py [--no-napari]`
   - **Best for**: Complex photograph with rich textures and facial features

9. **demo_splats_coins.py**
   - **Description**: Coins image with metallic textures and circular objects for compression analysis
   - **Usage**: `python demo_splats_coins.py [--no-napari]`
   - **Best for**: Metallic surfaces and illumination gradients

10. **demo_splats_mitosis_intgrad.py**
    - **Description**: Tests CLAHE-based seeding with artificial intensity gradient on mitosis data
    - **Usage**: `python demo_splats_mitosis_intgrad.py [--no-napari]`
    - **Best for**: Validating CLAHE seeding in challenging intensity conditions

11. **demo_multiscale_fitting.py**
    - **Description**: Compares single-scale vs multi-scale fitting showing speedup and quality trade-offs
    - **Usage**: `python demo_multiscale_fitting.py [--no-napari]`
    - **Best for**: Understanding multi-scale optimization benefits

12. **demo_splats_mitosis_explicit_seeding.py**
    - **Description**: Demonstrates the new explicit seeding API where seeds are generated using `seed_from_decomposition()`, `seed_from_gaussian()`, or `seed_from_moments()`. Shows how GSplatData with scale-informed shapes flows from seeding to fitting. Includes compression analysis with interactive napari visualization.
    - **Usage**: `python demo_splats_mitosis_explicit_seeding.py [--no-napari]`
    - **Best for**: Learning the explicit seeding API and understanding how scale information is preserved from detection to initialization

### Running Demos

**Standard execution (with napari visualization):**
```bash
python demo_basic_fitting.py
python demo_performance_metrics.py
python demo_2d_synthetic_blobs.py
```

**Headless execution (for testing/CI):**
```bash
python demo_basic_fitting.py --no-napari --n-iters 50
python demo_2d_synthetic_blobs.py --no-napari
python demo_performance_metrics.py --no-napari
```

**From project root with hatch:**
```bash
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_basic_fitting.py --no-napari
```

### Demo Categories

**Learning & Tutorial demos:**
- `demo_basic_fitting.py` - Start here for API overview
- `demo_performance_metrics.py` - Understanding optimization behavior and metrics
- `demo_splats_mitosis_explicit_seeding.py` - Explicit seeding API with GSplatData

**Dimensional Progression (2D → 3D → 4D):**
- `demo_2d_synthetic_blobs.py` - 2D compression with oriented ellipses
- `demo_3d_synthetic_phantom.py` - 3D volumetric compression with ellipsoid wireframes
- `demo_4d_hypercube.py` - 4D hypercube nD validation

**Real Data Applications:**
- `demo_3d_dapi_microscopy.py` - Real microscopy from IDR
- `demo_splats_mitosis.py` - Biological histology
- `demo_splats_astronaut.py` - Photography
- `demo_splats_coins.py` - Metallic textures

**Advanced/Specialized:**
- `demo_multiscale_fitting.py` - Multi-scale vs single-scale comparison
- `demo_splats_mitosis_intgrad.py` - CLAHE seeding validation

All demos follow the standardized best practices outlined in this specification, ensuring consistent code quality, documentation, and user experience across the entire demo collection.
