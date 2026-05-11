# luxar.gsplats - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2025-01

## Documentation Structure

This specification serves as the **hub** for the entire gsplats package. For detailed information on specific components, see:

- **Multi-Scale Decomposition**: [multiscale/SPECIFICATIONS.md](./multiscale/SPECIFICATIONS.md) - Image decomposition for efficient multi-scale fitting
- **Fitting Pipeline**: [fitting/SPECIFICATIONS.md](./fitting/SPECIFICATIONS.md) - Modular 6-stage fitting pipeline architecture
- **Optimizers**: [optim/SPECIFICATIONS.md](./optim/SPECIFICATIONS.md) - Standard Adam with gradient dilution compensation
- **Models**: [models/SPECIFICATIONS.md](./models/SPECIFICATIONS.md) - PyTorch model and rendering engine
- **Utilities**: [utils/SPECIFICATIONS.md](./utils/SPECIFICATIONS.md) - Matrix operations and gradient dilution utilities
- **Seeds**: [seeds/SPECIFICATIONS.md](./seeds/SPECIFICATIONS.md) - Seed generation methods for Gaussian splatting
- **I/O**: [io/SPECIFICATIONS.md](./io/SPECIFICATIONS.md) - GSplats serialization and inspection
- **CLAHE**: [clahe/SPECIFICATIONS.md](./clahe/SPECIFICATIONS.md) - CLAHE-based perceptual sampling
- **Terminology Glossary**: [GLOSSARY.md](./GLOSSARY.md) - Standard terminology and naming conventions

**Reading Order**:
1. **New to Gaussian Splatting?** Start with [Overview](#overview) below, then [Section 2: Gaussian Splat Model](#2-gaussian-splat-model-modelsgspatsgsplat_modelpy)
2. **Implementing features?** Jump to the relevant package specification above
3. **Debugging?** See [Testing Requirements](#8-integration-requirements) and individual package test sections

**Quick Start**:
```python
from luxar.gsplats import fit_gaussian_splats

# Fit Gaussian splats to your data
result = fit_gaussian_splats(
    V,                    # Your nD image/volume
    n_iters=1000,        # Max iterations
    max_abs_error=0.01,  # Convergence threshold
    verbose=True         # Show progress
)
# Access via result.centers, result.amplitudes, result.cholesky_factors, result.stats

# Transform the result
centered = result.center_at_centroid()  # Center at origin
dimmed = centered.scale_intensity(0.1)  # Reduce brightness 10x
shifted = dimmed.translate([10, 20, 30])  # Translate in space
```

## 0. GSplat Data Container (`gsplat_data.py`)

### Class: `GSplatData`

Container for Gaussian splat fitting results with transformation methods.

**Attributes**:
- `centers`: (N, d) array of splat center positions
- `amplitudes`: (N,) array of non-negative splat amplitudes
- `cholesky_factors`: (N, d*(d+1)//2) packed Cholesky factors L where Σ = L @ L.T
- `colors`: Optional (N, 3) RGB colors (uint8 or float32)
- `stats`: Dictionary of optimization statistics

**Transformation Methods**:

**`translate(offset: np.ndarray) -> GSplatData`**
- Translates all splat centers by the given offset vector
- Returns new GSplatData with shifted centers
- All other attributes (amplitudes, covariances, etc.) unchanged
- Example: `shifted = data.translate(np.array([10, 20, 30]))`

**`center_at_centroid() -> GSplatData`**
- Centers the splats at their amplitude-weighted centroid (center of mass)
- Centroid computed as: `Σ(amplitude_i * center_i) / Σ(amplitude_i)`
- Returns new GSplatData with centroid at origin [0, 0, ...]
- Useful for easier camera framing in viewers
- Example: `centered = data.center_at_centroid()`

**`scale_intensity(factor: float) -> GSplatData`**
- Multiplies all amplitudes by the given factor
- factor < 1: dims the representation
- factor > 1: brightens the representation
- Useful for visualization adjustment without re-fitting
- Example: `dimmed = data.scale_intensity(0.1)  # Reduce by 10x`

**Save/Load Methods**:
- `save(path, ...)`: Save to .gsplats.zarr format with encoding options
- `load(path)`: Load from .gsplats.zarr format

## Overview

Implement an n-dimensional Gaussian splatting system for image/volume reconstruction using collections of oriented Gaussian functions. Each "splat" uses a shifted Gaussian truncation formula for C⁰ continuity at the truncation boundary (no discontinuity when hard-truncated at T sigma):

```
D² = (x - μ_k)^T * Σ_k^(-1) * (x - μ_k)   (Mahalanobis distance squared)
C     = exp(-0.5 * T²)                       (boundary value, T=3 → 0.01111)
scale = 1 / (1 - C)                          (peak-preserving rescale, T=3 → 1.01123)
f_k(x) = a_k * scale * max(0, exp(-0.5 * D²) - C)
```

where μ_k is the center, Σ_k is the covariance matrix, a_k is the amplitude, and T is the truncation radius.

Key requirements:
- Support arbitrary dimensions with optimized 2D/3D fast paths
- Standard PyTorch Adam with gradient dilution compensation (50x+ faster)
- Fixed-pool splat relocation during optimization (no topology changes)
- Mathematically stable parameterizations avoiding singularities
- GPU-accelerated rendering with memory management

## 1. Seed Generation Sub-Package (`seeds/`)

> Full specification: [seeds/SPECIFICATIONS.md](./seeds/SPECIFICATIONS.md)


Seed generation provides initial Gaussian splat positions and shapes for the fitting pipeline.

### Unified Entry Point: `generate_seeds(V, method="auto", **kwargs)`

The `generate_seeds()` function in `seeds/generate.py` is the recommended entry point. It dispatches to one or more seeding methods and combines results with deduplication.

- **method="auto"** (default): Fast combination of edges (60%) + grid (40%)
- **method="edges"**: Edge-based seeding via nD Sobel gradients with Poisson disk sampling
- **method="grid"**: Uniform grid with optional jitter and intensity filtering
- **method="decomposition"**: Multi-scale decomposition peak detection (slower, best for blobs)
- Comma-separated combinations (e.g., `"decomposition,edges,grid"`) are supported

### Three Seeding Methods

1. **`seed_from_edges(V, ...)`** (`edges.py`): Detects edges using nD Sobel gradients, samples via weighted Poisson disk sampling. Returns isotropic Gaussians (sigma=1.0).

2. **`seed_from_grid(V, ...)`** (`grid.py`): Places seeds on a regular grid with aspect-ratio-aware spacing. Supports jitter, intensity filtering. Returns isotropic Gaussians (sigma=spacing/2).

3. **`seed_from_decomposition(V, ...)`** (`multiscale_decomposition.py`): Decomposes image into multiple scales via `decompose_image()`, finds local maxima per scale. Returns isotropic Gaussians where sigma equals the detection scale factor.

### Shared Utilities (`utils.py`)

- **`local_maxima(img, radius, thresh, top_k)`**: L-infinity neighborhood peak detection using `scipy.ndimage.maximum_filter`
- **`dedupe_farthest_first(coords, min_distance, intensities)`**: Greedy spatial deduplication with SpatialHashGrid acceleration. Returns `(deduped_coords, kept_indices)` for O(1) attribute lookup.
- **`sigmas_to_cholesky_isotropic(sigmas, ndim)`**: Converts per-seed isotropic sigmas to packed lower-triangular Cholesky factors
- **`combine_seeds(*arrays, min_distance)`**: Merges seed coordinate arrays with optional deduplication

### GPU Acceleration (`gpu_ops.py`)

All seeding methods accept an optional `device` parameter (`'cuda'`, `'mps'`, `'auto'`) for GPU acceleration via PyTorch. Provides substantial speedup for large volumes (>100 cubed) — often orders of magnitude depending on GPU and problem size. Operations: Sobel gradients (all dimensions), peak detection (2D/3D), amplitude interpolation (2D/3D).


## 2. Gaussian Splat Model (`models/gsplats/gsplat_model.py`)

### Core Class: `GaussianSplatModel(nn.Module)`

**Parameters:**
- `raw_mu`: Unconstrained center parameters, shape (N, d)
- `raw_L_diag`: Unconstrained diagonal parameters, shape (N, d)
- `L_off`: Off-diagonal elements, shape (N, tril_size(d)-d)
- `raw_a`: Unconstrained amplitude parameters, shape (N,)
**Parameterization:**
- Centers: `mu = sigmoid(raw_mu) * clamp(shape - 1, min=1)`
- Diagonal: `L_diag = min_diag + softplus(raw_L_diag)`
- Off-diagonal: `L_off = raw_L_off` (unconstrained)
- Amplitudes: `a = softplus(raw_a)`

**Initialization:**
- Transform initial centers to logit space: `raw_mu = log(u) - log(1-u)` where `u = centers/max(shape-1, 1)`
- Use `stable_inverse_softplus` for diagonal and amplitude initialization
- Clamp normalized coordinates to [1e-6, 1-1e-6] to avoid sigmoid saturation

**Key Methods:**
- `current_params()`: Return transformed (centers, L_matrices, amplitudes) as 3-tuple
- `_build_L()`: Reconstruct lower-triangular matrices from parameters
- `forward()`: Render all splats using main rendering function
- `n_splats()`: Return current number of splats
- `prune_(keep_mask)`: Remove splats by boolean mask
- `append_(centers, Ls, amps)`: Add new splats
- `replace_with(centers, Ls, amps)`: Replace all parameters
- `_to_internal_params(centers, Ls, amps)`: Convert external to internal parameterization

### Rendering Function: `render_gaussians(shape, centers, Ls, amps, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`

**Algorithm:**
1. **Compute AABB per splat**:
   - Compute radii: `radii = ceil(truncate * sqrt(diag(Sigma)))`
2. **Optional amplitude-aware shrinking**: if `a * scale * (exp(-0.5 * t^2) - C) < intensity_floor`, reduce radius (where `C = exp(-0.5 * T²)`, `scale = 1/(1-C)`)
3. Group splats by box dimensions for grid reuse: `{(h1,h2,...): [indices]}`
4. For each group:
   - Generate coordinate grid using `meshgrid`
   - Process in memory chunks to prevent OOM
   - Solve `L * y = (x - mu)` for all points (avoid matrix inversion)
   - **Apply shifted Gaussian**: Compute `amplitude * scale * max(0, exp(-0.5 * ||y||^2) - C)` (C⁰ continuous at truncation boundary)
   - Accumulate into output using `index_add_`

**Fast paths for 2D/3D:**
- `_render_gaussians_2d`: Explicit 2�2 forward substitution
- `_render_gaussians_3d`: Explicit 3�3 forward substitution
- Use manual solution instead of `solve_triangular` for performance

**Memory management:**
- Calculate optimal chunk size based on available GPU memory
- CUDA: use 60% of `torch.cuda.mem_get_info()`, fallback 2GB
- Account for tensor memory: `K * (2*d + 2) * bytes_per_element * 1.5`
- Clamp chunk sizes to [1024, 1048576]

**Note:** GSplats use a shifted Gaussian truncation formula `I(x) = a * scale * max(0, exp(-0.5 * D²) - C)` where `C = exp(-0.5 * T²)` and `scale = 1/(1-C)`. This ensures C⁰ continuity at the truncation boundary. Per-splat sharpness has been removed as of March 2026. All splats use s=2 (standard Gaussian).

## 3. Standard Optimizer Integration (`optim/`)

### Overview

The optimizer module provides a factory function for creating standard PyTorch Adam optimizers with automatic gradient dilution compensation. This enables 50x+ faster optimization compared to per-splat alternatives.

### create_optimizer_and_scheduler Function

```python
def create_optimizer_and_scheduler(
    model,
    lr: float = 1e-3,
    scheduler_type: Optional[str] = "plateau",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    ...
) -> Tuple[torch.optim.Optimizer, Optional[LRScheduler]]:
```

**Key Features:**
- Automatic gradient dilution compensation based on dimensionality
- Standard `torch.optim.Adam` for fast vectorized optimization
- Flexible scheduler support: ReduceLROnPlateau, ExponentialLR, or None

**Gradient Dilution Compensation:**
- 2D: 1.0× (baseline)
- 3D: 1.8×
- 4D: 8.5×

**Why Standard Adam Works with Fixed-Pool Architecture:**
1. **No tensor shape changes**: Splat pool size is fixed throughout optimization
2. **Relocation = parameter update**: Just modifies values, not tensor shapes
3. **Momentum adaptation**: Stale momentum at relocated splat quickly overwritten by new gradients
4. **Full adaptation**: Within 1-3 iterations after relocation

## 4. Dynamic Operations (`fitting/dynamic_ops/`)

### Philosophy
Dynamic operations use **fixed-pool splat relocation** to address reconstruction deficiencies. Instead of adding/removing splats (which changes tensor shapes), weak splats are relocated to high-residual regions. This enables use of standard PyTorch Adam (50x+ faster).

### Configuration: `DynamicOpsConfig`
- `step_every=50`: Run operations every N iterations during optimization
- `k_max_residuals=40`: Number of strongest residual peaks to analyze per iteration
- `nms_radius_vox=2.0`: Minimum distance between detected residual peaks (non-maximum suppression)
- `min_contribution_threshold=0.01`: Fixed threshold for influence detection

**Relocation Parameters**:
- `relocation_percentile=1.0`: Percentage of least important splats eligible for relocation
- `max_relocations_per_step=64`: Maximum relocations per dynamic ops step
- `init_sigma_vox=0.5`: Initial sigma for relocated splats (isotropic)

**Safety Parameters**:
- `min_splats_to_keep=10`: Minimum number of splats to retain

### Fixed-Pool Relocation Strategy

Dynamic operations use **fixed-pool splat relocation** instead of add/remove:

- **Identify weak splats**: Bottom N% by importance (amplitude × volume)
- **Find residual peaks**: High-error regions via non-maximum suppression
- **Relocate**: Move weak splats to peaks (parameter updates only, no shape changes)
- **Standard Adam works**: Tensor shapes never change

### Core Function: `apply_dynamic_operations(model, V_target, V_pred, cfg, max_abs_error_threshold, verbose=False)`

The dynamic operations algorithm runs every `step_every` iterations:

### **Step 1: Residual Peak Analysis**

1. **Compute residual image**: `residual = V_target - V_pred`
2. **Find k strongest peaks**: Identify the `k_max_residuals` locations with highest absolute residual values
3. **Apply spatial exclusion**: Use non-maximum suppression with radius `nms_radius_vox` to ensure peaks are spatially separated
4. **Rank by magnitude**: Process peaks in descending order of residual magnitude
5. **Convergence guard**: If strongest residual peak is below threshold, skip all dynamic operations

### **Step 2: Weak Splat Identification**

Identify splats that are candidates for relocation:

1. **Calculate splat importance**: For each splat k, compute `importance_k = amplitude_k × volume_k`
   - `amplitude_k = a_k` (splat amplitude)
   - `volume_k = prod(diag(L_k))` (approximates volume)
   - Low importance = small AND dim → not useful where it is

2. **Select relocation candidates**: Identify the `p%` least important splats (default `p=5%`)
   - Rank all splats by importance in ascending order
   - Select bottom `p% × N_splats` splats as relocation candidates

### **Step 3: Peak-Splat Matching and Relocation**

Match weak splats to residual peaks and relocate:

**Matching Algorithm**:
1. Process peaks in order of residual magnitude (strongest first)
2. For each peak:
   - Check if any non-weak splat already has significant influence there
   - If yes, skip this peak (existing coverage)
   - If no, assign closest available weak splat to this peak
3. Cap relocations at `max_relocations_per_step`

**Relocation**:
For each matched (splat_idx, peak_coords) pair:
1. **CENTER**: Convert coords to raw (logit) space and update `model.raw_mu[splat_idx]`
2. **AMPLITUDE**: Set to residual value at new location via `model.raw_a[splat_idx]`
3. **COVARIANCE**: Reset to isotropic (`init_sigma_vox`) via `model.raw_L_diag[splat_idx]`
4. **SHARPNESS**: Keep unchanged (optimizer will adjust)

### **Mathematical Foundations**

**Residual Peak Detection**:
- Use efficient peak detection with configurable exclusion zones
- Rank peaks by `|residual[i,j]|` in descending order
- Apply 2D/3D non-maximum suppression to avoid clustering

**Splat Importance Calculation**:
- For splat k: `importance_k = amplitude_k × volume_k`
- `volume_k = prod(diag(L_k))` (approximates Gaussian volume)
- Low importance = small AND dim → candidate for relocation

**Influence Detection**:
- For point (i,j), compute each splat's contribution: `contribution_k = a_k * scale * max(0, exp(-0.5 * (x_ij - μ_k)^T * Σ_k^(-1) * (x_ij - μ_k)) - C)` where `C = exp(-0.5 * T²)`, `scale = 1/(1-C)`
- Use early filtering with max reach: `candidates = splats where ||p - center|| ≤ 6 × max_sigma`

### **Operational Flow**
1. **Every `step_every` iterations**:
   - Compute current residual image
   - Execute Step 1: Peak detection with convergence guard
   - Execute Step 2: Weak splat identification
   - Execute Step 3: Peak-splat matching and relocation
   - Model parameters modified in-place (no shape changes)

This approach ensures dynamic operations are driven by reconstruction quality while maintaining fixed tensor shapes for standard PyTorch Adam optimization (50x+ faster).

## 5. Main Fitting Interface (`fit_gsplats.py`)

### Architecture Overview

The main fitting interface provides both functional and object-oriented APIs:
- **Functional API**: `fit_gaussian_splats()` - Convenience function for one-shot fitting
- **Object-Oriented API**: `GaussianSplatFitter` class - Reusable fitter with stateful configuration

**Implementation Architecture**:
The implementation uses a **modular 6-stage pipeline** (see [fitting/SPECIFICATIONS.md](./fitting/SPECIFICATIONS.md)):
1. **`prepare_fit_config()`**: Validate inputs and prepare configuration object
2. **`preprocess_data()`**: Normalize data, generate/validate candidates
3. **`initialize_optimization()`**: Create model, optimizer, and scheduler
4. **`create_loss_function()`**: Build loss function with regularization
5. **`run_optimization_loop()`**: Execute iterative optimization with dynamic operations
6. **`finalize_results()`**: Extract parameters, rescale intensities, compute statistics

This modular design separates concerns, improves testability, and makes the codebase maintainable.
The root-level functions delegate to the `fitting/` sub-package for actual implementation.

### Primary Function: `fit_gaussian_splats(V, seeds=None, norm_percentile=0.0, init_sigma_vox=0.5, n_iters=1000, lr=0.01, loss_type="l1", asymmetric_penalty=1.0, l1_amp=None, l1_diag=None, max_abs_error=None, seed_method="auto", ...)`

**Functional Signature**:
```python
def fit_gaussian_splats(
    V: np.ndarray,
    seeds: Optional[np.ndarray | float] = None,
    norm_percentile: float = 0.0,
    init_sigma_vox: float = 0.5,  # Default changed to 0.5 for single-voxel splats
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 1.0,
    l1_amp: Optional[float] = None,
    seed_method: str = "auto",  # DEFAULT: edges (60%) + grid (40%)
    l1_diag: Optional[float] = None,
    # ... additional parameters
) -> GSplatData
```

**Implementation Flow**:
1. Create `GaussianSplatFitter` instance with device and dynamic ops config
2. Call `fitter.fit()` which executes the 6-stage pipeline
3. Return GSplatData from finalization stage


**Input Validation and Error Handling**:
- **V validation**: Non-empty, finite values, valid NumPy array
- **seeds validation**: If array, shape must match V.ndim; if float, must be in (0, 1]
- **Parameter validation**: All learning rates, sigmas, and thresholds must be positive
- **Scale validation** (multi-scale): Ensures downsampled dimensions ≥ _MIN_SCALE_DIM (8 pixels)
- **Exception types**: `TypeError` for type errors, `ValueError` for invalid values, `RuntimeError` for optimization failures

**Edge Case Handling**:
- **Empty candidates**: Returns zero-sized arrays with correct shape (0, d + d*(d+1)//2 + 1)
- **Invalid inputs**: Raises `TypeError` or `ValueError` with descriptive messages
- **Scale validation**: Multi-scale fitting validates minimum dimension size (_MIN_SCALE_DIM = 8)

**Input validation:**
- Ensure V is non-empty with valid dimensions
- Validate seeds shape matches V.ndim (if provided)
- Check all hyperparameters are positive/valid
- Validate `max_abs_error` is positive if specified

**Auto-Seed Generation:**
- **Default behavior**: If `seeds=None`, automatically generate seeds using `seed_method="auto"` (edges + grid)
- **Auto method (DEFAULT)**: Combines edges (60%) + grid (40%) for fast, comprehensive seed coverage
- **Universal scale series**: `(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)` works optimally for all dimensions from fine details to large structures
- **Volume-proportional density**: `peaks_per_scale = max(50, int(V.size * 0.002))` scales seed count with image size (~0.2% of pixels)
- **Inclusive detection**: `percentile_thresh=70` for comprehensive feature coverage
- **Standard parameters**: `min_distance=2.0, add_intensity_grid=False` for robust detection
- **Logging**: Auto-generation usage is logged for transparency
- **Alternative methods**: Can specify `seed_method="edges"` (edges only), `seed_method="grid"` (grid only), `seed_method="decomposition"` (decomposition only), or comma-separated combinations

**Auto-Convergence Threshold:**
- **Default behavior**: If `max_abs_error=None`, automatically set threshold to 1% of normalized image dynamic range
- **Calculation**: `auto_threshold = 0.01` (since images are normalized to [0,1] using robust 1%-99% percentile range)
- **Rationale**: Provides sensible convergence criteria for all datasets without user configuration
- **Logging**: Auto-threshold usage is logged for transparency

**Initialization:**
- **Configurable normalization** using `norm_percentile` parameter:
  - **Full range** (`norm_percentile=0`): `V_min = min(V)`, `V_max = max(V)` - uses complete dynamic range
  - **Percentile clipping** (`norm_percentile>0`): `V_min = percentile(V, p)`, `V_max = percentile(V, 100-p)` - robust to outliers
  - **Default**: `norm_percentile=0.0` for full range normalization (maximum dynamic range utilization)
- Normalize V to [0,1]: `V_norm = (V - V_min) / (V_max - V_min)`
- Store normalization parameters for intensity rescaling
- Create `GaussianSplatModel` with candidate centers
- Initialize with isotropic covariances: `L0[i,i] = init_sigma_vox`
- Sample initial amplitudes from normalized image at center locations

**Intensity Rescaling:**
- **After optimization**: Rescale amplitudes to original intensity range
- **Formula**: `amps_original = amps_normalized × (V_max - V_min)`
- **Rationale**: Ensures Gaussian splat representation directly reconstructs original image intensities
- **Benefit**: Users can render splats directly without manual intensity scaling

**Important Limitation:**
- **DC component removal**: Gaussian splatting cannot represent constant (uniform) background intensities
- **Rationale**: Gaussians have finite support and integrate to finite values, unlike infinite uniform fields
- **Implication**: The method approximates variations and structures, not absolute baseline intensities
- **Reconstruction**: Resulting image may have different baseline than original, but preserves relative structure

**Optimization loop:**
1. Setup standard optimizer and scheduler via `create_optimizer_and_scheduler()`
   - **Gradient dilution compensation**: Automatically applied at optimizer creation
     - **Problem**: Higher dimensions have more parameters per splat, diluting gradients
     - **Parameter count**: 2D (5 params: 2 pos + 3 cov), 3D (9 params: 3 pos + 6 cov), 4D (14 params: 4 pos + 10 cov)
     - **Result**: Learning rate scaling - 2D (×1.0), 3D (×1.8), 4D (×8.5)
2. **Log convergence criteria**: Explicitly state convergence threshold (given or auto-calculated)
3. **Initialize best state tracking**: Track best loss and corresponding splat configuration
4. For each iteration:
   - Forward pass: `pred = model()`
   - Loss computation: MSE or Poisson + optional L1 regularization
   - Backward pass: compute gradients
   - Apply gradient clipping if specified
   - Optimizer step (standard PyTorch Adam)
   - Scheduler step for learning rate adaptation
   - Dynamic operations (fixed-pool relocation) if enabled and scheduled
   - Convergence check: compute `max_abs_error = max(|pred - target|)` and stop if below threshold
   - **Best state tracking**: Save current state if loss is lowest seen so far
   - Display max abs error during training in addition to losses
5. **Restore best state**: Return splat configuration that achieved lowest loss during optimization
6. **Log termination reason**: Explicitly state why optimization ended and which iteration's state was restored

**Best State Tracking:**
- **Quality guarantee**: Always return the splat configuration that achieved the lowest loss
- **Non-monotonic optimization**: Max absolute error fluctuates during optimization due to dynamic operations
- **Best state preservation**: Save splat parameters whenever loss improves
- **State restoration**: Return best configuration instead of potentially suboptimal final state
- **Statistics alignment**: Report statistics from best iteration, not final iteration


**Return Statistics Structure**:
The `stats` dictionary returned by `fit_gaussian_splats()` includes:
- **Core metrics**: `time_seconds`, `iterations`, `converged`, `final_error`, `best_iteration`
- **Optimization trajectory**: `loss_history`, `error_history` (if tracking enabled)
- **Movie data**: `movie_frames`, `movie_shape` (if `napari_movie=True`)
- **Best state info**: All statistics reflect the best state encountered during optimization


**Convergence detection:**
- Primary criterion: Maximum absolute error `max_abs_error = max(|prediction - target|)`
- Stop when `max_abs_error < threshold` with explicit logging of convergence achievement
- Always stop when reaching `n_iters` maximum iterations with explicit iteration limit notification
- Default `n_iters=1000` provides generous limit when using `max_abs_error` convergence
- **Logging requirements**:
  - State convergence threshold at optimization start
  - Report current max absolute error during training
  - Log new best states when encountered (based on loss)
  - Explicitly state termination reason and which iteration's state was restored


**Verbose Output and Visualization**:
When `verbose=True`, the function displays:
- **Optimization progress**: Iteration-by-iteration metrics via arbol sections
- **Compression analysis**: Comparing splat representation size to original data
- **Termination summary**: Final timing, iteration count, and convergence status
- **Optimization movie**: Interactive napari visualization (if `napari_movie=True`)

The verbose output uses the `arbol` library for hierarchical console logging.


**Loss functions:**
- MSE: `mean((pred - target)²)`
- **Asymmetric MSE**: `mean(where(pred > target, F * (pred - target)², (pred - target)²))` where F is over-prediction penalty factor
- Poisson: `mean(2 * (pred - target + target * log(target/pred)))`
- **Asymmetric Poisson**: Apply same over-prediction penalty to Poisson deviance
- **L1 (Mean Absolute Error)**: `mean(|pred - target|)`
- **Asymmetric L1**: `mean(where(pred > target, F * |pred - target|, |pred - target|))` where F is over-prediction penalty factor
- **Proportional L1 Regularization**:
  - **Amplitude regularization**: `+ l1_amp * mean(|softplus(raw_a)|)` where `l1_amp = 0.1 * lr` by default (10% of base LR = 5% of amplitude LR 2.0×)
  - **Diagonal regularization**: `+ l1_diag * mean(|softplus(raw_L_diag)|)` where `l1_diag = 0.01 * lr` by default (1% of base LR)
  - **Rationale**: L1 regularization should scale with optimization strength for consistent sparsity pressure
  - **Dimensional scaling**: Works correctly with gradient dilution compensation (higher LR → higher L1)
  - **Auto-tuning**: Eliminates need for manual L1 adjustment when changing learning rates

**Loss Function Selection Guide:**
- **MSE**: Best for smooth data with Gaussian noise, fast convergence, well-behaved gradients
- **Poisson**: Optimal for count/photon data, fluorescence microscopy, low-light imaging
- **L1**: Robust to outliers, preserves sharp features, encourages sparse residuals, excellent with asymmetric penalty

**Asymmetric Loss Rationale:**
- **Over-prediction** (`pred > target`, negative residual): Penalized by factor F (**default 1.0**) since non-negative Gaussian sums cannot easily reduce intensity
- **Under-prediction** (`pred < target`, positive residual): Normal penalty since additional Gaussians can easily add intensity
- **Model alignment**: Reflects the additive constraint of Gaussian splatting where reducing intensity is harder than adding it
- **Default enabled**: `asymmetric_penalty=1.0` by default
- **L1 synergy**: L1 + asymmetric penalty provides exceptional robustness and stability for challenging datasets

### GaussianSplatFitter Class

**Purpose**: Object-oriented interface providing reusable fitter with stateful configuration.

**Constructor**:
```python
GaussianSplatFitter(
    device: Optional[str] = None,
    enable_dynamic_ops: bool = False,
    dynamic_config: Optional[DynamicOpsConfig] = None
)
```

**Key Method**:
```python
def fit(
    self,
    V: np.ndarray,
    seeds: Optional[np.ndarray | float] = None,
    # ... all parameters from fit_gaussian_splats()
) -> GSplatData
```

**Relationship to Functional API**:
- `fit_gaussian_splats()` creates a `GaussianSplatFitter` instance internally
- Both APIs share the same implementation via the 6-stage pipeline
- The class allows configuration reuse across multiple fitting operations
- Identical parameter signatures and return values

**Device Selection Logic**:
- Auto-detection order: CUDA → MPS → CPU
- MPS (Apple Silicon) supported
- Explicit device specification overrides auto-detection

## 6. Utilities

### Matrix Utilities (`utils/trils.py`)
- `tril_size(d)`: Return `d*(d+1)//2`
- `pack_tril(L)`: Extract lower triangle in row-major order
- `unpack_tril(v, d)`: Reconstruct matrices from packed format

### Numerical Utilities (`models/utils/inverse_softplus.py`)
- `stable_inverse_softplus(y, beta=1.0)`: Handle large values with asymptotic approximation
- For `beta*y >= 50`: return `y` (asymptotic case)
- Otherwise: return `log(expm1(beta*y)) / beta`

### Rendering Wrappers
- `render_gaussians_numpy()`: NumPy wrapper with no gradients
- `render_gaussians_pytorch()`: PyTorch wrapper accepting packed parameters

## 8. Calibration (`calibration.py`)

### Purpose
Implement the manuscript's blind-spot cross-validation protocol (Supp. Doc. 2, `splat_count_vs_quality`) to recommend a principled splat budget `K*` for a given dataset, and report the dataset's noise-floor PSNR ceiling. Purely additive — `fit_gaussian_splats` is called unchanged at each K in the sweep.

### Why a separate module, not a `fit --cv` flag
Held-out PSNR is a *capacity*-selection criterion (across K), not an *iteration*-selection one. At fixed K, bounded splat parameters (clamped Cholesky diagonals, L1 amplitude regularisation, conservative culling) prevent the held-out trajectory from peak-and-declining over iterations — within-fit held-out monitoring would add no value beyond the existing patience-based early stop. Calibration therefore sits outside the fitter and treats `fit_gaussian_splats` as a black box.

### Pipeline
1. Generate a deterministic Bernoulli mask `M` over voxel positions (`fraction = 0.05`, `seed = 42` by default; matches Batson & Royer 2019).
2. Replace masked voxels with the median of their `(2r+1)^D` donut neighbourhood (centre excluded) to produce `V_filled`. Default `r = 1` → 3^D donut, 26 neighbours in 3D. Edge handling: `numpy reflect` padding.
3. For each `K` in the sweep grid:
   - Fit a Gaussian-splat model with `fit_gaussian_splats(V_filled, seeds=K, **fit_kwargs)`.
   - Render the result back to volume via `render_to_volume_tensor`.
   - Compute held-out PSNR at masked positions against the *original* (pre-fill) `V`; train PSNR at unmasked positions; full-volume PSNR/SSIM against the original.
4. Estimate the noise floor on `V` (after `[0, 1]` normalisation) via a three-estimator ensemble: discrete-Laplacian MAD (Immerkaer 1996, kernel-norm `K = 2D(2D+1)`), Haar HH-subband MAD (Donoho & Johnstone 1994), and background-region MAD on voxels below the 10th percentile. Take the median across finite estimators (robust to one outlier on the low side, typical when the dark tail is quantised).
5. Detect the held-out peak via the manuscript's hybrid rule:
   - **peak**: argmax is strictly interior AND `mean(pre-argmax)` and `mean(post-argmax)` are both ≥ 0.1 dB below the peak. Return the argmax.
   - **signal_limited**: the argmax is the last K and the curve rose by ≥ 0.3 dB over the sweep — the volume is signal-limited under the current model class (typical for clean light-sheet data). Return the last K.
   - **plateau**: otherwise. Return the smallest K within 0.3 dB of the maximum (onset of diminishing returns).

### Public API

```python
def cv_mask(shape, fraction=0.05, seed=42) -> np.ndarray  # bool
def donut_median_fill(V, mask, radius=1) -> np.ndarray
def held_out_psnr(V_hat, V_original, mask, data_range=None) -> float

@dataclass
class NoiseFloor:
    sigma_hat: float
    sigma_laplacian: float
    sigma_haar: float
    sigma_background: float
    psnr_max_db: float    # = -20 * log10(sigma_hat) for [0,1]-normalised data

def estimate_noise_floor(V) -> NoiseFloor

def build_k_grid(
    explicit=None, n_points=10, k_min=1_000, k_max=512_000,
    progression="exp", power=2,
) -> List[int]

@dataclass
class HeldOutPeak:
    k_star: int
    type: Literal["peak", "plateau", "signal_limited"]
    confidence_db: float

def find_k_star(k_values, held_out_psnr_values) -> HeldOutPeak

@dataclass
class CalibrationResult:
    k_values_requested: List[int]
    k_values_effective: List[int]   # post-cull splat counts
    held_out_psnr_db: List[float]
    train_psnr_db: List[float]
    held_out_mse: List[float]
    full_psnr_db: List[float]
    full_ssim: List[float]
    held_out_peak: HeldOutPeak
    noise_floor: NoiseFloor
    fit_times_seconds: List[float]
    splat_paths: Optional[List[str]]
    mask_seed: int
    mask_fraction: float
    donut_radius: int
    fit_config: Dict[str, Any]
    volume_shape: List[int]
    volume_dtype: str
    timestamp: str
    def to_json(self, path) -> None
    @classmethod
    def from_json(cls, path) -> CalibrationResult

def calibrate(
    V, k_grid, *,
    fit_kwargs=None, mask_seed=42, mask_fraction=0.05, donut_radius=1,
    keep_fits=None, progress_callback=None,
) -> CalibrationResult
```

### Determinism and reproducibility
- `cv_mask` uses `np.random.RandomState(seed).rand(*shape) < fraction` — identical mask across runs at the same `(shape, fraction, seed)`.
- `fit_kwargs.seeds` is overridden per-K and pop-ed before forwarding; all other `fit_kwargs` are passed verbatim.
- `verbose=False` is set as the default for the inner fits; override via `fit_kwargs={"verbose": True}` when debugging.
- `keep_fits=Path(...)` persists each fit as `<dir>/k<8-digit>.gsplats.zarr`. The fitter's standard stats (`psnr_db`, `ssim`, `time_seconds`, ...) ride along via `GSplatData.stats`; calibration-specific fields are kept in the global `cal.json` (referenced from `splat_paths`) since `GSplatData.save()` whitelists fitting-info keys.

### CLI integration (`luxar gsplat cal`)
The CLI delegates to `calibrate` with parsed K-grid args, mask args, and `load_fit_config` for the standard preset/YAML/CLI override chain. Output: a formatted Arbol summary table + JSON sidecar via `CalibrationResult.to_json`. Optional `--pdf` triggers `calibration_report.render_calibration_report`. Optional `--keep-fits <dir>` enables slice-montage panels in the PDF.

### Failure modes the test suite covers
- All-zero volume → ensemble σ̂ = 0, PSNR ceiling = +inf (or sentinel ">60 dB" in CLI output).
- Constant volume passed to `donut_median_fill` → identity (median of constant is the constant).
- Very small (16³) volumes → calibration runs end-to-end on CPU in seconds.
- NaN / +inf in held-out curves → preserved through JSON round-trip via `null` sentinels.
- Synthetic Gaussian noise of known σ → `estimate_noise_floor` recovers σ within 25%.

## 7. Integration Requirements

### Device Support
- Auto-detect best device: CUDA > MPS > CPU
- Handle device-specific limitations (e.g., MPS doesn't support torch.unique with dim)
- Provide fallbacks for missing functionality

### Memory Management
- Implement adaptive chunking based on available memory
- Use grid caching with process-wide cache keyed by (device, dtype, strides, shape)
- Monitor and report memory usage for debugging

### Error Handling
- Comprehensive input validation with descriptive messages
- Graceful degradation for edge cases (empty arrays, uniform images, etc.)
- Fallback paths for numerical instabilities

### Optional Features
- Napari integration for real-time visualization and optimization movies
- Debug visualizations showing operation effects
- Comprehensive logging with configurable verbosity
- Support for headless operation (disable napari with flags)


### 4D Validation Demo (`demos/demo_4d_hypercube.py`)

**Purpose**: Comprehensive validation of nD capabilities with 4-dimensional hypercube data

**4D Data Structure**:
- **Dimensions**: (time/spectral, z, y, x) with moderate size (32×32×32×8) for efficiency
- **Synthetic content**: Multiple 4D Gaussian blobs varying across all dimensions
- **Validation target**: Test auto-candidate generation and nD dynamic operations

**4D Visualization**:
- **napari 4D support**: Native 4D visualization with dimension sliders
- **Axis labels**: ["time/spectral", "z", "y", "x"] for intuitive navigation
- **Compression analysis**: 4D-specific storage efficiency calculations
- **Interactive exploration**: Full 4D navigation and quality assessment

**nD Algorithm Validation**:
- **Universal scales**: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) tested in 4D space
- **Volume-proportional scaling**: Candidate density adapts to 4D volume size
- **Dynamic operations**: Convergence-based seeding, LR boosting, pruning in 4D
- **Parameter efficiency**: 4×4 covariance matrices (15 parameters per splat)

### Main Exports
Export primary user-facing functions and classes:
- `fit_gaussian_splats`
- `GaussianSplatFitter`
- `DynamicOpsConfig`

## Terminology Glossary

**Note**: For the complete, detailed glossary with usage guidelines, see [GLOSSARY.md](./GLOSSARY.md)

This section provides a quick reference for the most important terms. For comprehensive definitions, naming conventions, and cross-references, consult the dedicated glossary.

### Core Concepts

**Splat**
- A single oriented Gaussian function with shifted truncation: `f(x) = a * scale * max(0, exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ)) - C)` where `C = exp(-0.5 * T²)`, `scale = 1/(1-C)` for C⁰ continuity at truncation boundary
- Also called: Gaussian splat, oriented Gaussian
- NOT: blob, particle, or kernel (avoid these terms for consistency)

**Centers (μ)**
- Position of splat center in voxel coordinates
- Parameter name: `raw_mu` (logit-space), transformed via sigmoid to stay in image bounds
- NOT: positions, means, or locations

**Amplitudes (a)**
- Non-negative scalar controlling splat brightness/intensity
- Parameter name: `raw_a` (transformed via softplus)
- NOT: weights, intensities, or coefficients

**Cholesky Factors (L)**
- Lower-triangular matrix where `Σ = L @ L^T` (covariance parameterization)
- Parameter names: `raw_L_diag` (diagonal), `L_off` (off-diagonal)
- NOT: covariance matrix directly (we use Cholesky for stability)

### Operations

**Seeding**
- **Initial seeding**: Generating candidate splat locations from image features (startup)
- **Dynamic seeding**: Adding new splats in high-residual regions during optimization
- Use "seeding" for initial generation, "adding" for dynamic operations

**Culling**
- **Quality-based removal**: Removing low-importance splats based on reconstruction quality
- Distinct from generic "removing" which includes any deletion operation
- Use "culling" when referring to quality-based removal (see ``GSplatData.cull()``)

**Dynamic Operations**
- Umbrella term for adaptive topology changes during optimization
- Includes: seeding new splats, culling low-quality splats, learning rate boosting
- Also called: adaptive operations, topology management
- NOT: splat management (too vague)

**Adding (Dynamic)**
- Adding new splats during optimization (after initial seeding)
- Method names: `append_()`, `add_splats()`
- Context: Dynamic operations

**Removing (Generic)**
- Any deletion of splats from the model
- Method names: `prune_()`, `remove_splats()`
- Context: Generic or implementation-specific

### Optimization Terms

**Gradient Dilution**
- Phenomenon where higher dimensions have more parameters per splat, diluting gradients
- **Gradient dilution compensation**: Scaling learning rate to counteract dilution
- Factor calculation: See [utils/SPECIFICATIONS.md](./utils/SPECIFICATIONS.md)
- NOT: gradient scaling, LR adjustment (less specific)

**Per-Splat State**
- Individual optimizer state (momentum, learning rate) maintained for each splat
- Enables momentum preservation during topology changes
- NOT: splat-wise state, individual state

**Best State Tracking**
- Saving the parameter configuration that achieved the lowest loss
- Provides quality guarantee even with non-monotonic optimization
- Also called: quality guarantee
- NOT: optimal state, peak performance (less precise)

**Parameter-Type-Specific Learning Rates**
- Different learning rate multipliers for different parameter types:
  - Position: ×0.1 (slow, prevents migration)
  - Variance: ×1.0 (normal)
  - Amplitude: ×2.0 (fast convergence)
  - Sharpness: ×0.5 (conservative)

### Dimensionality

**d**
- Number of dimensions (mathematical notation)
- Examples: d=2 (2D image), d=3 (3D volume), d=4 (4D hypercube)

**nD**
- n-dimensional (prose and code)
- Follows NumPy convention
- NOT: N-dimensional, n-D

**Dimensionality**
- Full word form when writing prose
- Example: "The algorithm supports arbitrary dimensionality"

### Technical Abbreviations

**AABB**
- Axis-Aligned Bounding Box
- Used for efficient rendering (only compute Gaussian contribution within AABB)

**LR**
- Learning Rate
- NOT: lr (use lowercase in code, uppercase in prose)

**DoG**
- Difference of Gaussians
- Used in candidate generation: `DoG = gaussian_filter(V, σ) - gaussian_filter(V, 1.6*σ)`

**L1 Regularization**
- L1 norm penalty: `loss += λ * |parameter|`
- Encourages sparsity (drives values toward zero)

**MSE**
- Mean Squared Error: `mean((pred - target)²)`

**MAE / L1 Loss**
- Mean Absolute Error: `mean(|pred - target|)`

### Parameter Names (Use These Consistently)

**Raw Parameters** (learnable, unconstrained or constrained):
- `raw_mu`: Center parameters in logit space
- `raw_L_diag`: Diagonal Cholesky parameters (via softplus)
- `L_off`: Off-diagonal Cholesky parameters (unconstrained)
- `raw_a`: Amplitude parameters (via softplus)

**Transformed Parameters** (after activation functions):
- `centers` or `μ`: Actual center positions in voxel coordinates
- `L` or `Ls`: Cholesky factors (lower-triangular matrices)
- `amps` or `a`: Actual amplitudes (non-negative)

**Configuration Parameters**:
- `sigma_min_diag`: Minimum diagonal values (per-dimension sequence or float, broadcast across dimensions)
- `sigma_max_diag`: Maximum diagonal values (per-dimension sequence)
- `truncate`: Gaussian truncation radius T in standard deviations (shifted formula uses C=exp(-0.5*T²) for C⁰ continuity)
- `init_sigma_vox`: Initial sigma for isotropic covariances

### Mathematical Notation

**Σ (Sigma)**
- Covariance matrix (positive definite)
- Relationship: `Σ = L @ L^T`

**μ (mu)**
- Center position vector, shape (d,)

**L (L-matrix)**
- Lower-triangular Cholesky factor
- Ensures Σ is always positive definite

### Cross-Package References

When referencing other specifications, use this format:

**Format**: `[Package Name](./package/SPECIFICATIONS.md)` → **Section Name**

**Examples**:
- Gradient dilution details: [utils/SPECIFICATIONS.md](./utils/SPECIFICATIONS.md) → Section 2
- Fitting pipeline: [fitting/SPECIFICATIONS.md](./fitting/SPECIFICATIONS.md) → Pipeline Architecture
- Optimizer factory: [optim/SPECIFICATIONS.md](./optim/SPECIFICATIONS.md) → create_optimizer_and_scheduler

## Conclusion

This specification provides complete implementation details for a mathematically rigorous, computationally efficient, and feature-rich Gaussian splatting system with dynamic optimization capabilities validated from 2D to 4D and beyond.

**For detailed information on specific components**, see the [Documentation Structure](#documentation-structure) section at the top of this document.

---

## Changelog

- **v2.0.0** (January 2025): Standard PyTorch Adam optimizer
  - Replaced per-splat optimizer with standard `torch.optim.Adam` (50x+ faster)
  - Fixed-pool splat relocation instead of add/remove topology changes
  - Simplified optimizer setup via `create_optimizer_and_scheduler()`
  - Gradient dilution compensation applied at optimizer creation time

- **v1.0.1** (2025-11-28): Sharpness bounds clarification
  - Added explicit clarification of official [0, 31] bounds vs fitting implementation [0.164, 24.47]
  - Documented that narrower fitting range is implementation detail, not format constraint

- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented hub structure linking to sub-specifications
  - Specified core mathematical formulations
  - Defined multi-scale decomposition approach
  - Established cross-package reference format
