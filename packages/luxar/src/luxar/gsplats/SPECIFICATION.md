# Gaussian Splatting Implementation Specification

## Overview

Implement an n-dimensional Gaussian splatting system for image/volume reconstruction using collections of oriented Gaussian functions. Each "splat" represents: `f_k(x) = a_k * exp(-0.5 * (x - �_k)^T * �_k^(-1) * (x - �_k))` where �_k is the center, �_k is the covariance matrix, and a_k is the amplitude.

Key requirements:
- Support arbitrary dimensions with optimized 2D/3D fast paths
- Per-splat optimization with individual learning rates and momentum
- Dynamic topology changes (add/remove splats) during optimization
- Mathematically stable parameterizations avoiding singularities
- GPU-accelerated rendering with memory management

## 1. Candidate Generation (`candidates.py`)

### Core Function: `find_candidates_overcomplete_nd(V, spacing=None, scales=(0.7,1.0,1.4,2.0,2.8,4.0), peaks_per_scale=1000, percentile_thresh=70.0, min_dist=2.0, add_intensity_grid=True, grid_step=None, grid_percentile=60.0)`

Generate overcomplete candidate locations using three complementary methods:

**Method 1: Multi-scale Gaussian peaks**
- For each scale � in `scales`: apply `gaussian_filter(V, sigma=�)`
- Find local maxima using `maximum_filter` with kernel size `2*ceil(1.5*�)+1`
- Keep peaks where `image[x] == max_filtered[x] AND image[x] >= percentile(image, percentile_thresh)`
- Return top `peaks_per_scale` strongest peaks per scale

**Method 2: Difference of Gaussians (DoG)**
- Compute `DoG = gaussian_filter(V, �) - gaussian_filter(V, 1.6*�)` for each scale
- Find local maxima using same approach as Method 1
- Return top `peaks_per_scale//2` peaks per scale

**Method 3: Intensity-weighted grid sampling** (if `add_intensity_grid=True`)
- Create regular grid with step `grid_step` (default: `4*min(scales)`)
- Offset grid by half-step: start at `step//2` to avoid boundaries
- Apply `uniform_filter` with size `max(1, step//3)` for local averaging
- Keep points above `percentile(local_intensities, grid_percentile)`

**Spatial deduplication:**
- Use `scipy.spatial.cKDTree` for datasets >100 points, else greedy O(N�)
- Remove points within `min_dist` Euclidean distance
- Process in detection order, keeping first occurrence

**Sub-pixel refinement:**
- For each candidate, extract 3�3�...�3 neighborhood (clamped to image bounds)
- Compute intensity-weighted centroid: `� = �(w_i * x_i) / �(w_i)` where `w_i = intensity_i - min_intensity`

### Helper Functions:
- `_local_maxima(img, radius, thresh, top_k)`: L neighborhood maxima detection
- `_dog_response(vol, sigma, k=1.6)`: Difference of Gaussians computation
- `_dedupe(coords, min_dist)`: Spatial deduplication with KDTree/greedy fallback

## 2. Gaussian Splat Model (`models/gsplats/gsplat_model.py`)

### Core Class: `GaussianSplatModel(nn.Module)`

**Parameters:**
- `raw_mu`: Unconstrained center parameters, shape (N, d)
- `raw_L_diag`: Unconstrained diagonal parameters, shape (N, d)  
- `L_off`: Off-diagonal elements, shape (N, tril_size(d)-d)
- `raw_a`: Unconstrained amplitude parameters, shape (N,)

**Parameterization:**
- Centers: `� = sigmoid(raw_�) * clamp(shape - 1, min=1)`
- Diagonal: `L_diag = �_min_diag + softplus(raw_L_diag)`
- Off-diagonal: `L_off = raw_L_off` (unconstrained)
- Amplitudes: `a = softplus(raw_a)`

**Initialization:**
- Transform initial centers to logit space: `raw_� = log(u) - log(1-u)` where `u = centers/max(shape-1, 1)`
- Use `stable_inverse_softplus` for diagonal and amplitude initialization
- Clamp normalized coordinates to [1e-6, 1-1e-6] to avoid sigmoid saturation

**Key Methods:**
- `current_params()`: Return transformed (centers, L_matrices, amplitudes)
- `_build_L()`: Reconstruct lower-triangular matrices from parameters
- `forward()`: Render all splats using main rendering function
- `n_splats()`: Return current number of splats
- `prune_(keep_mask)`: Remove splats by boolean mask
- `append_(centers, Ls, amps)`: Add new splats
- `replace_with(centers, Ls, amps)`: Replace all parameters

### Rendering Function: `render_gaussians(shape, centers, Ls, amps, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`

**Algorithm:**
1. Compute AABB per splat: `radii = ceil(truncate * sqrt(diag(�)))`
2. Optional amplitude-aware shrinking: if `a * exp(-0.5 * t�) < intensity_floor`, reduce radius
3. Group splats by box dimensions for grid reuse: `{(h1,h2,...): [indices]}`
4. For each group:
   - Generate coordinate grid using `meshgrid`
   - Process in memory chunks to prevent OOM
   - Solve `L * y = (x - �)` for all points (avoid matrix inversion)
   - Compute `exp(-0.5 * ||y||�) * amplitude`
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

## 3. Per-Splat Optimization (`optim/`)

### PerSplatAdam Class

**State structure:** `{splat_idx: {"lr": float, "step": int, "exp_avg_mu": tensor, "exp_avg_sq_mu": tensor, ...}}`

Maintain separate momentum buffers for each parameter type (�, L_diag, L_off, a) per splat.

**Key methods:**
- `step()`: Update all splats using individual learning rates and momentum
- `add_splats(n_new, lr_new=None)`: Extend state for new splats
- `remove_splats(keep_mask)`: Remove and reindex splat states
- `set_learning_rate(splat_idx, lr)`: Individual learning rate control
- `get_effective_learning_rates()`: Return per-splat learning rates

**Adam update per splat:**
1. Extract gradients for all parameter types with bounds checking
2. Update momentum: `exp_avg = �� * exp_avg + (1-��) * grad`
3. Update second moment: `exp_avg_sq = �� * exp_avg_sq + (1-��) * grad�`
4. Apply bias correction and compute step
5. Update parameters in-place

### Per-Splat Schedulers

**PerSplatReduceLROnPlateau:**
- Track loss per splat individually or use global metric
- Reduce learning rate when plateau detected: `new_lr = max(lr * factor, min_lr)`
- Support multiple input types: scalar (all splats), tensor (per-splat), dict (explicit)

**PerSplatExponentialLR:**
- Apply decay: `lr = lr * �`
- Optional age-based decay: newer splats decay slower

### ModelOptimizerCoordinator

**Purpose:** Synchronize model topology changes with optimizer/scheduler state

**Methods:**
- `prune_splats(keep_mask)`: Update model, optimizer, and scheduler atomically
- `add_splats(centers, Ls, amps, lr_new=None)`: Coordinated addition
- Ensure state consistency across all components

## 4. Dynamic Operations (`dynamic_ops.py`)

### Philosophy
Dynamic operations address reconstruction deficiencies by analyzing the residual image (target - prediction) to identify where the current Gaussian splat representation fails. The approach focuses on three core operations: **seeding** new splats where coverage is missing, **splitting** problematic splats that are too large or elongated, and **pruning** splats that contribute minimally to reconstruction quality.

### Configuration: `DynamicOpsConfig`
- `step_every=50`: Run operations every N iterations during optimization
- `k_max_residuals=10`: Number of strongest residual peaks to analyze per iteration
- `nms_radius_vox=2.0`: Minimum distance between detected residual peaks (non-maximum suppression)
- `min_contribution_threshold=0.05`: Fixed threshold for influence detection in splitting decisions
- `relative_contribution_factor=0.1`: Adaptive threshold factor for amplitude validation (threshold = local_residual × factor)
- `learning_rate_threshold=1e-6`: Learning rate below which splats are considered "stagnant"

### Core Function: `apply_dynamic_operations(model, optimizer, scheduler, V_target, V_pred, cfg, current_lr, max_abs_error_threshold, device, verbose=False)`

The dynamic operations algorithm runs every `step_every` iterations and performs three sequential operations:

### **Step 1: Residual Peak Analysis**
1. **Compute residual image**: `residual = V_target - V_pred`
2. **Find k strongest peaks**: Identify the `k_max_residuals` locations with highest absolute residual values
3. **Apply spatial exclusion**: Use non-maximum suppression with radius `nms_radius_vox` to ensure peaks are spatially separated
4. **Rank by magnitude**: Process peaks in descending order of residual magnitude
5. **Convergence guard**: If a finite convergence threshold is set (`max_abs_error_threshold != inf`) and the strongest residual peak is below this threshold, skip all dynamic operations to avoid adding splats just before convergence

### **Step 2: Adaptive Splat Operations**
For each detected residual peak location, determine if coverage is sufficient using convergence criteria:

**Case A: Missing Coverage (Seeding)**
- **Detection criterion**: Coverage at the peak location is insufficient based on convergence criteria:
  - **If convergence threshold is set** (`max_abs_error_threshold != inf`): Residual magnitude > `max_abs_error_threshold`
  - **If no convergence threshold** (`max_abs_error_threshold == inf`): Always seed (minimize error as much as possible)
- **Action**: Create new Gaussian splat fitted to local residual
- **Rationale**: Uses convergence-based detection to directly align with optimization goals, not arbitrary influence thresholds
- **Initialization**:
  - Center: Peak location coordinates
  - Covariance: **Adaptive sizing** - estimated from local residual structure analysis (weighted covariance of residual distribution)
  - Amplitude: Estimated via least-squares fitting `a = <local_residual, gaussian> / <gaussian, gaussian>`
- **Validation**: **Adaptive amplitude threshold** - only add if `estimated_amplitude ≥ local_residual × relative_contribution_factor`
  - **Rationale**: Threshold scales with problem magnitude, preventing plateau issues from fixed thresholds

**Case B: Problematic Existing Splat (Splitting)**
- **Detection criterion**: Existing splat coverage is present but inadequate, requiring refinement:
  - **Coverage exists**: Find splat with highest influence at the peak location (using existing influence calculation)
  - **Splitting criteria**: The dominant splat meets geometric criteria for splitting:
    - Principal axis length: `sqrt(λ_max) > split_size_threshold` (where `λ_max` is largest eigenvalue of covariance)
    - Elongation ratio: `λ_max / λ_min > split_elongation_threshold`
  - **Convergence criteria**: Same as Case A - residual magnitude indicates insufficient coverage
- **Action**: Split the problematic splat along its principal eigenvector
- **Split procedure**:
  - Compute principal eigenvector `v1` corresponding to `λ_max`
  - Create two child splats: `μ_children = μ_parent ± 0.3 * sqrt(λ_max) * v1`
  - Scale down covariances: `Σ_children = 0.6 * Σ_parent`
  - Distribute amplitude: `a_children = 0.6 * a_parent` each
- **Replace**: Remove parent splat and add two children

### **Step 3: Global Pruning Analysis**
Independent of residual peaks, analyze all splats for minimal contribution:

**Stagnant Splat Detection**:
- **Learning rate criterion**: Current per-splat learning rate < `learning_rate_threshold`
- **Impact assessment**: Calculate maximum possible change in reconstruction:
  - `max_change = current_lr * max_gradient * amplitude * max_gaussian_value`
  - Compare against local residual: `local_residual = mean(|residual|)` in splat's 2σ region
- **Removal criterion**: `max_change < min_contribution_threshold AND max_change < 0.1 * local_residual`
- **Additional check**: Ensure splat contributes less than 1% to overall reconstruction quality

**Pruning Logic**:
```
for each splat:
    if (learning_rate < lr_threshold AND
        max_possible_change < min_contribution AND
        max_possible_change < 0.1 * local_residual AND
        splat_contribution < 0.01 * total_reconstruction_energy):
        mark_for_removal(splat)
```

### **Mathematical Foundations**

**Residual Peak Detection**:
- Use efficient peak detection with configurable exclusion zones
- Rank peaks by `|residual[i,j]|` in descending order
- Apply 2D/3D non-maximum suppression to avoid clustering

**Splat Contribution Analysis**:
- For point (i,j), compute each splat's contribution: `contribution_k = a_k * exp(-0.5 * (x_ij - μ_k)^T * Σ_k^(-1) * (x_ij - μ_k))`
- Identify dominant contributor: `argmax_k(contribution_k)`

**Impact Estimation**:
- Maximum gradient magnitude from optimizer state
- Gaussian maximum value: `a_k` (at center μ_k)
- Learning rate from per-splat optimizer state

### **Operational Flow**
1. **Every `step_every` iterations**:
   - Compute current residual image
   - Execute Step 1: Peak detection
   - Execute Step 2: Seeding/Splitting decisions
   - Execute Step 3: Global pruning
   - Update model, optimizer, and scheduler states atomically

This approach ensures that dynamic operations are directly driven by reconstruction quality, focusing computational resources on the most problematic regions while removing splats that no longer contribute meaningfully to the optimization process.

## 5. Main Fitting Interface (`fit_gsplats.py`)

### Primary Function: `fit_gaussian_splats(V, centers_overcomplete, init_sigma_vox=1.5, n_iters=1000, lr=0.2, loss_type="mse", asymmetric_penalty=10.0, max_abs_error=None, ...)`

**Input validation:**
- Ensure V is non-empty with valid dimensions
- Validate centers_overcomplete shape matches V.ndim
- Check all hyperparameters are positive/valid
- Validate `max_abs_error` is positive if specified (None means no convergence threshold)

**Initialization:**
- Normalize V to [0,1] using robust percentiles (1st, 99th)
- Create `GaussianSplatModel` with candidate centers
- Initialize with isotropic covariances: `L0[i,i] = init_sigma_vox`
- Sample initial amplitudes from normalized image at center locations

**Optimization loop:**
1. Setup per-splat optimizer and scheduler
2. For each iteration:
   - Forward pass: `pred = model()`
   - Loss computation: MSE or Poisson + optional L1 regularization
   - Backward pass: compute gradients
   - Apply gradient clipping if specified
   - Optimizer step with per-splat learning rates
   - Scheduler step for learning rate adaptation
   - Dynamic operations if enabled and scheduled
   - Convergence check: compute `max_abs_error = max(|pred - target|)` and stop if below threshold.
   - Display max abs error during training in addition to losses.

**Convergence detection:**
- Primary criterion: Maximum absolute error `max_abs_error = max(|prediction - target|)` 
- Stop when `max_abs_error < threshold` if `max_abs_error` parameter is specified
- Always stop when reaching `n_iters` maximum iterations
- Default `n_iters=1000` provides generous limit when using `max_abs_error` convergence
- Remove complex loss window comparison and early stopping patience logic

**Loss functions:**
- MSE: `mean((pred - target)²)`
- **Asymmetric MSE**: `mean(where(pred > target, F * (pred - target)², (pred - target)²))` where F is over-prediction penalty factor
- Poisson: `mean(2 * (pred - target + target * log(target/pred)))`
- **Asymmetric Poisson**: Apply same over-prediction penalty to Poisson deviance
- Optional L1: `+ l1_amp * mean(|softplus(raw_a)|)`

**Asymmetric Loss Rationale:**
- **Over-prediction** (`pred > target`, negative residual): Heavily penalized by factor F (**default 10x**) since non-negative Gaussian sums cannot easily reduce intensity
- **Under-prediction** (`pred < target`, positive residual): Normal penalty since additional Gaussians can easily add intensity
- **Model alignment**: Reflects the additive constraint of Gaussian splatting where reducing intensity is harder than adding it
- **Default enabled**: `asymmetric_penalty=10.0` by default for optimal results with additive Gaussian models

### GaussianSplatFitter Class
Object-oriented interface with identical functionality to functional API.

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

## 7. Integration Requirements

### Device Support
- Auto-detect best device: CUDA � CPU (MPS supported but may be slower)
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


### Main Exports
Export primary user-facing functions and classes:
- `fit_gaussian_splats`
- `GaussianSplatFitter` 
- `DynamicOpsConfig`

This specification provides complete implementation details for a mathematically rigorous, computationally efficient, and feature-rich Gaussian splatting system with dynamic optimization capabilities.

