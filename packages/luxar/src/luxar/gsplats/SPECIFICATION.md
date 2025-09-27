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
5. **Apply parameter-type-specific learning rates**: `effective_lr = base_lr × parameter_multiplier`
6. Update parameters in-place with type-specific rates

**Parameter-Type-Specific Learning Rate Multipliers (Hard-coded):**
- **Position parameters (μ)**: `×0.1` - Prevents splat migration and proliferation
- **Variance parameters (L_diag, L_off)**: `×1.0` - Normal covariance adaptation
- **Amplitude parameters (a)**: `×2.0` - Fast intensity convergence

**Anti-Proliferation Rationale:**
- **Root cause**: Splats migrating away from seeded locations causes runaway seeding cycles
- **Solution**: Slow position updates (×0.1) keep splats spatially stable while accelerating intensity adaptation (×2.0)
- **Result**: Eliminates splat proliferation while improving convergence quality

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
Dynamic operations address reconstruction deficiencies by analyzing the residual image (target - prediction) to identify where the current Gaussian splat representation fails. The approach focuses on two core operations: **seeding** new splats where coverage is missing and **pruning** splats that contribute minimally to reconstruction quality.

### Configuration: `DynamicOpsConfig`
- `step_every=50`: Run operations every N iterations during optimization
- `k_max_residuals=10`: Number of strongest residual peaks to analyze per iteration
- `nms_radius_vox=2.0`: Minimum distance between detected residual peaks (non-maximum suppression)
- `min_contribution_threshold=0.05`: Fixed threshold for influence detection in splitting decisions
- `relative_contribution_factor=0.1`: Adaptive threshold factor for amplitude validation (threshold = local_residual × factor)

**Adaptive Learning Rate Parameters**:
- `lr_boost_factor=1.5`: Multiplication factor for problematic region learning rates
- `boost_influence_threshold=0.05`: Minimum influence to consider splat as "covering" problematic region

**Principled Pruning Parameters**:
- `pruning_percentile=5.0`: Percentage of least important splats to consider for removal
- `min_splats_to_keep=10`: Minimum number of splats to retain regardless of importance

### Core Function: `apply_dynamic_operations(model, optimizer, scheduler, V_target, V_pred, cfg, current_lr, max_abs_error_threshold, device, verbose=False)`

The dynamic operations algorithm runs every `step_every` iterations and performs two core operations:

### **Step 1: Residual Peak Analysis**
1. **Compute residual image**: `residual = V_target - V_pred`
2. **Find k strongest peaks**: Identify the `k_max_residuals` locations with highest absolute residual values
3. **Apply spatial exclusion**: Use non-maximum suppression with radius `nms_radius_vox` to ensure peaks are spatially separated
4. **Rank by magnitude**: Process peaks in descending order of residual magnitude
5. **Convergence guard**: If a finite convergence threshold is set (`max_abs_error_threshold != inf`) and the strongest residual peak is below this threshold, skip all dynamic operations to avoid adding splats just before convergence

### **Step 2: Adaptive Splat Operations**
For each detected residual peak location, determine if coverage is sufficient using convergence criteria, and adaptively boost learning rates for problematic regions:

**Case A: Missing Coverage (Seeding)**
- **Detection criterion**: Coverage at the peak location is insufficient based on convergence criteria:
  - **If convergence threshold is set** (`max_abs_error_threshold != inf`): Residual magnitude > `max_abs_error_threshold`
  - **If no convergence threshold** (`max_abs_error_threshold == inf`): Always seed (minimize error as much as possible)
- **Action**: Create new Gaussian splat fitted to local residual
- **Rationale**: Uses convergence-based detection to directly align with optimization goals, not arbitrary influence thresholds
- **Initialization** (Ultra-Simple Approach):
  - Center: Peak location coordinates
  - Covariance: **Isotropic** - `L = eye(d) × init_sigma_vox` (simple spherical/circular splats)
  - Amplitude: **Direct residual value** - `amplitude = |residual[center_coordinates]|`
  - **Rationale**: Simple, fast, robust approach that relies on optimization to evolve optimal shapes
  - **Benefits**: Eliminates complex rendering and covariance analysis, always numerically stable
- **Validation**: **Adaptive amplitude threshold** - only add if `estimated_amplitude ≥ local_residual × relative_contribution_factor`
  - **Rationale**: Threshold scales with problem magnitude, preventing plateau issues from fixed thresholds

**Case B: Existing Coverage with Inadequate Quality**
- **Detection criterion**: Existing splat coverage is present but insufficient based on convergence criteria
- **Action**: Adaptive Learning Rate Boosting only (splitting removed for simplicity)
  - **Purpose**: "Unfreeze" splats covering problematic regions to help them adapt
  - **Target identification**: Splat with highest influence at the residual peak location
  - **Boost calculation**: `new_lr = min(current_lr × boost_factor, base_lr)`
  - **Default boost factor**: 1.5 (50% increase)
  - **Safety cap**: Never exceed original starting learning rate (`base_lr`)
  - **Rationale**: Give existing splats chance to fix problems; if ineffective, seeding will occur next iteration

### **Step 3: Principled Splat Pruning Analysis**
Independent of residual peaks, identify and remove splats that do not meaningfully contribute to reconstruction quality:

**Importance-Based Pre-filtering**:
1. **Calculate splat importance**: For each splat k, compute `importance_k = amplitude_k × volume_k`
   - `amplitude_k = a_k` (splat amplitude)
   - `volume_k = prod(diag(L_k))` (approximates `sqrt(det(Σ_k))` for computational efficiency)
   - This approximates splat "mass": `∫ f_k(x) dx ≈ a_k × (2π)^(d/2) × sqrt(det(Σ_k))`

2. **Select pruning candidates**: Identify the `p%` least important splats (default `p=5%`)
   - Rank all splats by importance in ascending order
   - Select bottom `p% × N_splats` splats as removal candidates
   - Pre-filtering reduces computational cost by ~20x (only test 5% of splats)

**Local Convergence-Based Removal Validation**:
For each candidate splat k in the low-importance set:

1. **Define influence region**:
   - Compute 3σ elliptical region around splat center: `region_k = {x : (x - μ_k)^T Σ_k^{-1} (x - μ_k) ≤ 9}`
   - This defines the spatial area where splat k has significant contribution

2. **Local removal test**:
   - Render prediction without splat k in influence region only: `pred_local_without_k`
   - Extract local target values: `target_local = target[region_k]`
   - Compute local residual: `local_residual = |pred_local_without_k - target_local|`

3. **Local convergence validation**:
   - Find maximum local residual: `max_local_residual = max(local_residual)`
   - **If convergence threshold set**: Compare with threshold directly
   - **If no convergence threshold**: Use adaptive tolerance based on current local error

4. **Local convergence decision**:
   - **Remove** if: `max_local_residual ≤ max_abs_error_threshold`
   - **Keep** if: `max_local_residual > max_abs_error_threshold`
   - **Conservative**: Only remove if local convergence is guaranteed to be maintained

**Pruning Configuration**:
- `pruning_percentile = 5.0`: Percentage of least important splats to consider for removal
- `min_splats_to_keep = 10`: Minimum number of splats to retain regardless of importance

**Local Convergence-Based Pruning Algorithm**:
```
importance = [amplitude_k * prod(diag(L_k)) for all splats k]
candidates = bottom_percentile(importance, pruning_percentile)
removal_list = []

for splat_k in candidates:
    # Define local influence region (3σ ellipse)
    influence_region = compute_elliptical_region(splat_k.center, splat_k.covariance, radius=3.0)

    # Render locally without splat k
    pred_local_without_k = render_region_without_splat(influence_region, model, exclude=splat_k)
    target_local = target[influence_region]

    # Check local convergence impact
    local_residual = abs(pred_local_without_k - target_local)
    max_local_residual = max(local_residual)

    # Local convergence-based decision
    if max_abs_error_threshold != inf:
        can_remove = max_local_residual <= max_abs_error_threshold
    else:
        current_local_max = max(abs(pred_current[influence_region] - target_local))
        tolerance = current_local_max * quality_tolerance_factor
        can_remove = max_local_residual <= current_local_max + tolerance

    if can_remove:
        removal_list.append(splat_k)

if len(removal_list) > 0 and (total_splats - len(removal_list)) >= min_splats_to_keep:
    remove_splats(removal_list)
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

### Primary Function: `fit_gaussian_splats(V, centers_overcomplete=None, norm_percentile=0.0, init_sigma_vox=1.5, n_iters=1000, lr=0.01, loss_type="l1", asymmetric_penalty=10.0, l1_amp=None, max_abs_error=None, ...)`

**Input validation:**
- Ensure V is non-empty with valid dimensions
- Validate centers_overcomplete shape matches V.ndim (if provided)
- Check all hyperparameters are positive/valid
- Validate `max_abs_error` is positive if specified

**Auto-Candidate Generation:**
- **Default behavior**: If `centers_overcomplete=None`, automatically generate candidates using dimension-aware defaults
- **Universal scale series**: `(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)` works optimally for all dimensions from fine details to large structures
- **Volume-proportional density**: `peaks_per_scale = max(50, int(V.size * 0.002))` scales candidate count with image size (~0.2% of pixels)
- **Inclusive detection**: `percentile_thresh=70` for comprehensive feature coverage
- **Standard parameters**: `min_dist=2.0, add_intensity_grid=False` for robust detection
- **Logging**: Auto-generation usage is logged for transparency

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
1. Setup per-splat optimizer and scheduler
2. **Enhanced gradient dilution compensation**: Scale learning rate by combined parameter complexity and dimensional spatial complexity
   - **Problem**: Higher dimensions suffer from both parameter dilution and spatial complexity challenges
   - **Parameter dilution**: 2D (5 params), 3D (10 params), 4D (15 params), nD (d + d(d+1)/2 + 1 params)
   - **Spatial complexity**: 4D space is geometrically more complex than 2D/3D for optimization
   - **Enhanced compensation formula**: `effective_lr = base_lr × dimensional_complexity × parameter_complexity`
     - `dimensional_complexity = d^0.8` (accounts for spatial optimization difficulty)
     - `parameter_complexity = params_current / params_2d` (accounts for gradient dilution)
   - **Result**: More aggressive scaling: 2D (×1.0), 3D (×5.2), 4D (×12.0) for effective nD optimization
3. **Log convergence criteria**: Explicitly state convergence threshold (given or auto-calculated)
4. **Initialize best state tracking**: Track best max absolute error and corresponding splat configuration
4. For each iteration:
   - Forward pass: `pred = model()`
   - Loss computation: MSE or Poisson + optional L1 regularization
   - Backward pass: compute gradients
   - Apply gradient clipping if specified
   - Optimizer step with per-splat learning rates and parameter-type multipliers
   - Scheduler step for learning rate adaptation
   - Dynamic operations if enabled and scheduled
   - Convergence check: compute `max_abs_error = max(|pred - target|)` and stop if below threshold
   - **Best state tracking**: Save current state if `max_abs_error` is lowest seen so far
   - Display max abs error during training in addition to losses
5. **Restore best state**: Return splat configuration that achieved lowest max absolute error during optimization
6. **Log termination reason**: Explicitly state why optimization ended and which iteration's state was restored

**Best State Tracking:**
- **Quality guarantee**: Always return the splat configuration that achieved the lowest max absolute error
- **Non-monotonic optimization**: Max absolute error fluctuates during optimization due to dynamic operations
- **Best state preservation**: Save splat parameters whenever max absolute error improves
- **State restoration**: Return best configuration instead of potentially suboptimal final state
- **Statistics alignment**: Report statistics from best iteration, not final iteration

**Convergence detection:**
- Primary criterion: Maximum absolute error `max_abs_error = max(|prediction - target|)`
- Stop when `max_abs_error < threshold` with explicit logging of convergence achievement
- Always stop when reaching `n_iters` maximum iterations with explicit iteration limit notification
- Default `n_iters=1000` provides generous limit when using `max_abs_error` convergence
- **Logging requirements**:
  - State convergence threshold at optimization start
  - Report current max absolute error during training
  - Log new best states when encountered
  - Explicitly state termination reason and which iteration's state was restored

**Loss functions:**
- MSE: `mean((pred - target)²)`
- **Asymmetric MSE**: `mean(where(pred > target, F * (pred - target)², (pred - target)²))` where F is over-prediction penalty factor
- Poisson: `mean(2 * (pred - target + target * log(target/pred)))`
- **Asymmetric Poisson**: Apply same over-prediction penalty to Poisson deviance
- **L1 (Mean Absolute Error)**: `mean(|pred - target|)`
- **Asymmetric L1**: `mean(where(pred > target, F * |pred - target|, |pred - target|))` where F is over-prediction penalty factor
- **Proportional L1 Regularization**: `+ l1_amp * mean(|softplus(raw_a)|)` where `l1_amp = 0.1 * lr` by default
  - **Rationale**: L1 regularization should scale with optimization strength for consistent sparsity pressure
  - **Dimensional scaling**: Works correctly with gradient dilution compensation (higher LR → higher L1)
  - **Auto-tuning**: Eliminates need for manual L1 adjustment when changing learning rates

**Loss Function Selection Guide:**
- **MSE**: Best for smooth data with Gaussian noise, fast convergence, well-behaved gradients
- **Poisson**: Optimal for count/photon data, fluorescence microscopy, low-light imaging
- **L1**: Robust to outliers, preserves sharp features, encourages sparse residuals, excellent with asymmetric penalty

**Asymmetric Loss Rationale:**
- **Over-prediction** (`pred > target`, negative residual): Heavily penalized by factor F (**default 10x**) since non-negative Gaussian sums cannot easily reduce intensity
- **Under-prediction** (`pred < target`, positive residual): Normal penalty since additional Gaussians can easily add intensity
- **Model alignment**: Reflects the additive constraint of Gaussian splatting where reducing intensity is harder than adding it
- **Default enabled**: `asymmetric_penalty=10.0` by default for optimal results with additive Gaussian models
- **L1 synergy**: L1 + asymmetric penalty provides exceptional robustness and stability for challenging datasets

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


### 4D Validation Demo (`demos/demo_splats_4d_napari.py`)

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
- **Dynamic operations**: Convergence-based seeding, splitting, pruning in 4D
- **Parameter efficiency**: 4×4 covariance matrices (15 parameters per splat)

### Main Exports
Export primary user-facing functions and classes:
- `fit_gaussian_splats`
- `GaussianSplatFitter`
- `DynamicOpsConfig`

This specification provides complete implementation details for a mathematically rigorous, computationally efficient, and feature-rich Gaussian splatting system with dynamic optimization capabilities validated from 2D to 4D and beyond.

