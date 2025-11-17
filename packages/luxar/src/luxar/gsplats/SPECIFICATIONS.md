# Gaussian Splatting Implementation Specification

## Documentation Structure

This specification serves as the **hub** for the entire gsplats package. For detailed information on specific components, see:

- **Multi-Scale Decomposition**: [multiscale/SPECIFICATIONS.md](./multiscale/SPECIFICATIONS.md) - Image decomposition for efficient multi-scale fitting
- **Fitting Pipeline**: [fitting/SPECIFICATIONS.md](./fitting/SPECIFICATIONS.md) - Modular 6-stage fitting pipeline architecture
- **Optimizers**: [optim/SPECIFICATIONS.md](./optim/SPECIFICATIONS.md) - Per-splat Adam optimizer and schedulers
- **Models**: [models/SPECIFICATIONS.md](./models/SPECIFICATIONS.md) - PyTorch model and rendering engine
- **Utilities**: [utils/SPECIFICATIONS.md](./utils/SPECIFICATIONS.md) - Matrix operations and gradient dilution utilities
- **Terminology Glossary**: [GLOSSARY.md](./GLOSSARY.md) - Standard terminology and naming conventions

**Reading Order**:
1. **New to Gaussian Splatting?** Start with [Overview](#overview) below, then [Section 2: Gaussian Splat Model](#2-gaussian-splat-model-modelsgspatsgsplat_modelpy)
2. **Implementing features?** Jump to the relevant package specification above
3. **Debugging?** See [Testing Requirements](#8-integration-requirements) and individual package test sections

**Quick Start**:
```python
from luxar.gsplats import fit_gaussian_splats

# Fit Gaussian splats to your data
params, amps, stats = fit_gaussian_splats(
    V,                    # Your nD image/volume
    n_iters=1000,        # Max iterations
    max_abs_error=0.01,  # Convergence threshold
    verbose=True         # Show progress
)
```

## Overview

Implement an n-dimensional Gaussian splatting system for image/volume reconstruction using collections of oriented Gaussian functions. Each "splat" represents: `f_k(x) = a_k * exp(-0.5 * (x - μ_k)^T * Σ_k^(-1) * (x - μ_k))` where μ_k is the center, Σ_k is the covariance matrix, and a_k is the amplitude.

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

**Method 4: CLAHE-based perceptual sampling** (if `add_clahe_sampling=True`)
- Apply CLAHE to enhance local contrast (tile_size=16, clip_limit=2.0)
- Sample from CLAHE-equalized intensities as probability distribution
- Generates perceptually-balanced candidates (dim structures get fair representation)
- Number of samples: `clahe_samples_per_scale` (default: `peaks_per_scale`)

**Spatial deduplication (Farthest-First Selection):**
- Sort all candidates by detection strength (intensity or CLAHE value)
- Initialize with strongest candidate
- Iteratively select candidate furthest from all previously selected
- Continue until budget exhausted or all candidates processed
- **Result**: Maximum spatial diversity with quality priority
- **Complexity**: O(k² × n) where k is output size, n is input candidates

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
- `sharpness_offsets_raw`: Sharpness offset parameters, shape (N,) - controls edge falloff

**Parameterization:**
- Centers: `� = sigmoid(raw_�) * clamp(shape - 1, min=1)`
- Diagonal: `L_diag = �_min_diag + softplus(raw_L_diag)`
- Off-diagonal: `L_off = raw_L_off` (unconstrained)
- Amplitudes: `a = softplus(raw_a)`
- Sharpness: `s = 2 * exp(sharpness_offsets_raw)` (exponential mapping, default s=2)

**Initialization:**
- Transform initial centers to logit space: `raw_� = log(u) - log(1-u)` where `u = centers/max(shape-1, 1)`
- Use `stable_inverse_softplus` for diagonal and amplitude initialization
- Clamp normalized coordinates to [1e-6, 1-1e-6] to avoid sigmoid saturation
- Initialize sharpness offsets to 0: `sharpness_offsets_raw = 0` (gives s = 2, standard Gaussian)

**Key Methods:**
- `current_params()`: Return transformed (centers, L_matrices, amplitudes, sharpness) as 4-tuple
- `_build_L()`: Reconstruct lower-triangular matrices from parameters
- `forward()`: Render all splats using main rendering function with sharpness
- `n_splats()`: Return current number of splats
- `prune_(keep_mask)`: Remove splats by boolean mask (including sharpness)
- `append_(centers, Ls, amps, sharpness=None)`: Add new splats (defaults to s=2 if None)
- `replace_with(centers, Ls, amps, sharpness=None)`: Replace all parameters
- `_to_internal_params(centers, Ls, amps, sharpness=None)`: Convert external to internal parameterization

### Rendering Function: `render_gaussians(shape, centers, Ls, amps, sharpness, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`

**Algorithm:**
1. **Compute sharpness-adjusted AABB per splat**:
   - For generalized Gaussian `exp(-0.5 * ||y||^s)`, the effective radius is adjusted: `effective_truncate = truncate^(2/s)`
   - **Rationale**: For same threshold as standard Gaussian (s=2), solve `r^s = truncate²` → `r = truncate^(2/s)`
   - Compute radii: `radii = ceil(effective_truncate * sqrt(diag(Σ)))`
   - **Examples**: s=2.0 → 3^1 = 3 (unchanged), s=1.5 → 3^1.33 ≈ 4.73 (larger for soft splats), s=3.0 → 3^0.67 ≈ 2.08 (smaller for sharp splats)
2. **Optional amplitude-aware shrinking**: if `a * exp(-0.5 * t^s) < intensity_floor`, reduce radius using sharpness-adjusted threshold
3. Group splats by box dimensions for grid reuse: `{(h1,h2,...): [indices]}`
4. For each group:
   - Generate coordinate grid using `meshgrid`
   - Process in memory chunks to prevent OOM
   - Solve `L * y = (x - �)` for all points (avoid matrix inversion)
   - **Apply generalized Gaussian**: Compute `exp(-0.5 * ||y||�^(s/2)) * amplitude`
     - Standard case (s=2): `exp(-0.5 * ||y||�)`
     - Sharper case (s>2): `exp(-0.5 * ||y||�^(s/2))` - faster decay, sharper edges
     - Softer case (s<2): `exp(-0.5 * ||y||�^(s/2))` - slower decay, heavier tails
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

### 2.1 Per-Splat Sharpness Feature

**Overview:**
Per-splat sharpness extends standard Gaussian splatting to **generalized Gaussian distributions**, enabling adaptive edge control for better sparse approximations. Each splat can learn its optimal sharpness parameter independently.

**Mathematical Formulation:**
- **Standard Gaussian**: `I(x) = a * exp(-0.5 * ||y||²)`
- **Generalized Gaussian**: `I(x) = a * exp(-0.5 * ||y||^s)` where s is sharpness
- **Computational form**: `exp(-0.5 * expo^(s/2))` where `expo = ||y||²` (Mahalanobis distance squared)

**Sharpness Parameter s:**
- `s = 2`: Standard Gaussian (smooth exponential falloff, infinite support)
- `s > 2`: Sharper edges, more compact support → **better sparse representations**
- `s < 2`: Softer edges, heavier tails (Laplacian-like for s=1)
- `s → ∞`: Approaches box function (hard edges)
- `s → 0`: Approaches uniform (very soft)

**Exponential Parameterization:**
```
s = 2 * exp(s')
```
where `s'` is the learned **sharpness offset** parameter.

**Benefits of exponential parameterization:**
1. **Zero-centered learning**: `s' = 0` → `s = 2` (standard Gaussian is natural default)
2. **Symmetric exploration**: Can increase/decrease sharpness from sensible baseline
3. **Always positive**: `s > 0` guaranteed for all `s' ∈ ℝ`
4. **L1 regularization friendly**: Pushing `s' → 0` encourages standard Gaussians
5. **Smooth gradients**: Exponential provides stable optimization dynamics

**Inverse transformation:**
```
s' = log(s / 2)
```
Used when initializing from existing sharpness values.

**Optimization Strategy:**
1. **Initialization**: All splats start at `s' = 0` (standard Gaussian, s=2)
2. **L1 regularization**: `loss += l1_sharpness * mean(|s'|)`
   - Encourages splats to remain at standard Gaussian unless beneficial
   - Promotes sparsity in sharpness parameter space
   - Default: `l1_sharpness = 0.05 * lr` (5% of base LR)
3. **Differential learning rate**: Sharpness parameters use fixed slower learning rate
   - Fixed: `sharpness_lr = 0.5 * base_lr` (hardcoded, not gradient-dilution-compensated)
   - Rationale: Sharpness is dimensionality-independent (always 1 scalar), so no gradient dilution applied
   - Moderate learning speed (0.5×) for conservative shape parameter updates
   - Prevents instability from rapid sharpness changes
4. **Adaptive learning**: Splats learn optimal sharpness based on local structure
   - Sharp features → learn s > 2
   - Smooth regions → stay near s = 2
   - Noisy areas → may learn s < 2 for robustness

**Implementation Details:**

**Model parameter:**
```python
self.sharpness_offsets_raw = nn.Parameter(
    torch.zeros(N, dtype=torch.float32, device=device)
)
```

**Forward transformation:**
```python
sharpness = 2.0 * torch.exp(self.sharpness_offsets_raw)  # s = 2 * exp(s')
```

**Rendering integration:**
```python
# Compute Mahalanobis distance squared
expo = torch.sum(y * y, dim=1)  # ||y||² where y = L^(-1) * (x - μ)

# Apply generalized Gaussian falloff
vals = torch.exp(-0.5 * torch.pow(expo, sharpness[:, None] / 2.0)) * amplitude[:, None]
```

**Dynamic operations:**
- `prune_()`: Remove sharpness parameters for pruned splats
- `append_()`: Initialize new splats with `s' = 0` (or inherit from parent)
- `replace_with()`: Accept optional sharpness parameter

**Benefits for Gaussian Splatting:**
1. **Better sparse approximations**: Sharper splats reduce overlap, fewer splats needed
2. **Sharp feature preservation**: Edges and boundaries represented more accurately
3. **Adaptive support**: Each splat learns optimal spatial extent
4. **Improved compression**: More efficient tiling of spatial domain
5. **Backward compatible**: Default `s=2` recovers standard Gaussian behavior

**Configuration Parameters:**
- `l1_sharpness`: L1 regularization strength on `s'` (default 0.05 * lr, 5% of base learning rate)

**Example sharpness values:**
- Smooth blobs: `s ≈ 1.5-2.0` (soft Gaussian-like)
- Medium features: `s ≈ 2.0-3.0` (standard to slightly sharp)
- Sharp edges: `s ≈ 3.0-6.0` (compact, efficient)
- Very sharp boundaries: `s ≈ 6.0+` (nearly box-like)

**Visualization of falloff:**
For 1D profile at distance r from center:
- `s=1`: Linear-like decay, heavy tails
- `s=2`: Classic Gaussian bell curve
- `s=4`: Flatter center, rapid edge decay
- `s=8`: Nearly flat center, very sharp drop

**Trade-offs:**
- Higher sharpness → sharper reconstruction, but harder optimization
- Lower sharpness → smoother reconstruction, more robust to noise
- L1 regularization balances these by encouraging standard Gaussian unless needed

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

**Parameter-Type-Specific Learning Rate Multipliers:**
- **Position parameters (μ)**: `×0.1` (hardcoded) - Prevents splat migration and proliferation
- **Variance parameters (L_diag, L_off)**: `×1.0` (hardcoded, with gradient dilution compensation) - Normal covariance adaptation
- **Amplitude parameters (a)**: `×2.0` (hardcoded) - Fast intensity convergence
- **Sharpness parameters (s')**: `×0.5` (hardcoded, no gradient dilution) - Conservative shape parameter updates

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

### Residual-Based Seeding Strategy

Dynamic operations use **residual-based seeding** to target reconstruction deficiencies:

- **Method**: Find peaks in residual image using non-maximum suppression
- **Target**: Locations with highest reconstruction error
- **Strength**: Directly addresses reconstruction deficiencies

### Core Function: `apply_dynamic_operations(model, optimizer, scheduler, V_target, V_pred, cfg, current_lr, max_abs_error_threshold, device, verbose=False)`

The dynamic operations algorithm runs every `step_every` iterations:

### **Step 1: Residual Peak Analysis**

1. **Compute residual image**: `residual = V_target - V_pred`
2. **Find k strongest peaks**: Identify the `k_max_residuals` locations with highest absolute residual values
3. **Apply spatial exclusion**: Use non-maximum suppression with radius `nms_radius_vox` to ensure peaks are spatially separated
4. **Rank by magnitude**: Process peaks in descending order of residual magnitude
5. **Convergence guard**: If strongest residual peak is below threshold, skip all dynamic operations

### **Step 2: Adaptive Splat Operations**

For each detected residual peak location, determine if coverage is sufficient using convergence criteria, and adaptively boost learning rates for problematic regions:

**Case A: Missing Coverage (Seeding)**
- **Detection criterion**: Residual magnitude > `max_abs_error_threshold` at peak location
- **Action**: Create new Gaussian splat fitted to local residual
- **Initialization**:
  - Center: Peak location coordinates
  - Covariance: Isotropic - `L = eye(d) × init_sigma_vox` (simple spherical/circular splats)
  - Amplitude: Direct residual value - `amplitude = |residual[center_coordinates]|`
  - Sharpness: Standard Gaussian - `s' = 0` (gives `s = 2`, standard Gaussian falloff)
- **Validation**: Only add if `estimated_amplitude ≥ local_residual × relative_contribution_factor`

**Case B: Existing Coverage with Inadequate Quality**
- **Detection criterion**: Existing splat coverage present but residual still exceeds threshold
- **Action**: Adaptive Learning Rate Boosting
  - **Purpose**: "Unfreeze" splats covering problematic regions to help them adapt
  - **Target identification**: Splat with highest influence at the residual peak location
  - **Boost calculation**: `new_lr = min(current_lr × boost_factor, base_lr)`
  - **Default boost factor**: 1.5 (50% increase)
  - **Safety cap**: Never exceed original starting learning rate (`base_lr`)

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

### Primary Function: `fit_gaussian_splats(V, seeds=None, norm_percentile=0.0, init_sigma_vox=0.5, n_iters=1000, lr=0.01, loss_type="l1", asymmetric_penalty=10.0, l1_amp=None, l1_diag=None, l1_sharpness=None, max_abs_error=None, ...)`

**Input validation:**
- Ensure V is non-empty with valid dimensions
- Validate seeds shape matches V.ndim (if provided)
- Check all hyperparameters are positive/valid
- Validate `max_abs_error` is positive if specified

**Auto-Candidate Generation:**
- **Default behavior**: If `seeds=None`, automatically generate candidates using dimension-aware defaults
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
   - **Gradient dilution compensation**: Optimizer internally applies gradient dilution compensation to learning rate
     - **Problem**: Higher dimensions have more parameters per splat, diluting gradients
     - **Parameter count**: 2D (5 params: 2 pos + 3 cov), 3D (10 params: 3 pos + 6 cov), 4D (15 params: 4 pos + 10 cov), nD (d + d(d+1)/2 params)
     - **Compensation formula** (for d ≤ 3): `effective_lr = base_lr × (params_current / params_2d)`
     - **Enhanced formula** (for d > 3): `effective_lr = base_lr × d^0.8 × (params_current / params_2d)`
     - **Result**: Learning rate scaling - 2D (×1.0), 3D (×2.0), 4D (×7.1)
     - **Sharpness exception**: Sharpness always uses base_lr (no gradient dilution) since it's a single scalar regardless of dimension
2. **Log convergence criteria**: Explicitly state convergence threshold (given or auto-calculated)
3. **Initialize best state tracking**: Track best max absolute error and corresponding splat configuration
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
- **Proportional L1 Regularization**:
  - **Amplitude regularization**: `+ l1_amp * mean(|softplus(raw_a)|)` where `l1_amp = 0.1 * lr` by default (10% of base LR = 5% of amplitude LR 2.0×)
  - **Diagonal regularization**: `+ l1_diag * mean(|softplus(raw_L_diag)|)` where `l1_diag = 0.01 * lr` by default (1% of base LR)
  - **Sharpness regularization**: `+ l1_sharpness * mean(|sharpness_offsets_raw|)` where `l1_sharpness = 0.05 * lr` by default (5% of base LR = 10% of sharpness LR 0.5×)
  - **Rationale**: L1 regularization should scale with optimization strength for consistent sparsity pressure
  - **Dimensional scaling**: Works correctly with gradient dilution compensation (higher LR → higher L1)
  - **Auto-tuning**: Eliminates need for manual L1 adjustment when changing learning rates
  - **Sharpness sparsity**: Optional L1 on sharpness offsets encourages standard Gaussians (`s' = 0`, `s = 2`) unless beneficial to deviate

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

## 6. Multi-Scale Gaussian Splat Fitting

### Overview and Motivation

Multi-scale Gaussian splat fitting leverages multi-scale image decomposition to significantly accelerate the fitting of large Gaussian splats. The key insight is that coarse-scale decomposed images contain fewer voxels, enabling much faster optimization while still capturing large-scale structures.

**Core Benefits:**
- **Computational efficiency**: Scale factor r reduces voxel count by (1/r)^d
  - 2D: 8× scale = 64× fewer pixels
  - 3D: 8× scale = 512× fewer voxels
  - 4D: 8× scale = 4096× fewer hypervoxels
- **Hierarchical representation**: Natural separation of features by spatial scale
- **Faster convergence**: Fewer voxels per scale means faster iteration times
- **Automatic scale separation**: Decomposition naturally separates coarse and fine features

**When to Use:**
- Large images/volumes where single-scale fitting is slow
- Data with clear hierarchical structure (coarse + fine features)
- When you want explicit separation of features by scale
- To achieve 10-100× speedup for 3D/4D datasets

### Mathematical Formulation

**Multi-scale decomposition** produces a set of images at different resolutions:

```
V = Σₖ upsample(Vₖ, scale_factor=sₖ)
```

where:
- V is the original full-resolution image
- Vₖ is the image at scale k (downsampled by factor sₖ)
- sₖ ∈ {1, 2, 4, 8, ...} are the scale factors (increasing → coarser)

**Independent fitting** is performed on each scale:

```
For each scale k:
    Fit Gaussians on Vₖ independently
    → Get parameters: (centers_k, L_matrices_k, amplitudes_k, sharpness_k)
```

**Parameter scaling** transforms parameters from scale space back to full resolution:

```
centers_full = centers_scale × scale_factor
L_matrices_full = L_matrices_scale × scale_factor
amplitudes_full = amplitudes_scale  (no scaling)
sharpness_full = sharpness_scale    (no scaling)
```

**Combination** produces the final multi-scale splat representation:

```
params_combined = vstack([params_1, params_2, ..., params_K])
amps_combined = concatenate([amps_1, amps_2, ..., amps_K])
sharpness_combined = concatenate([sharpness_1, sharpness_2, ..., sharpness_K])
```

### Parameter Scaling Rules

**Centers (μ):**
- **Scaling**: `μ_full = μ_scale × scale_factor`
- **Rationale**: Voxel coordinates scale linearly with resolution
- **Example**: Center at (10, 20) in 2× downsampled image → (20, 40) in full image

**Cholesky Factors (L):**
- **Scaling**: `L_full = L_scale × scale_factor`
- **Rationale**: Covariance matrix Σ = L @ L^T has units of squared distance
- **Mathematical**: If distances scale by r, then Σ scales by r², so L scales by r
- **Example**: `L_scale = [[2, 0], [1, 3]]` at 2× scale → `L_full = [[4, 0], [2, 6]]`

**Amplitudes (a):**
- **Scaling**: No scaling required (`a_full = a_scale`)
- **Rationale**: Multi-scale decomposition ensures each scale has correct intensity distribution
- **Mathematical**: Decomposition V = Σ Vₖ already accounts for energy distribution
- **Note**: Each scale captures appropriate portion of total intensity

**Sharpness (s):**
- **Scaling**: No scaling required (`s_full = s_scale`)
- **Rationale**: Sharpness is a dimensionless per-standard-deviation quantity
- **Mathematical Proof**:
  - Mahalanobis distance: `y = Σ^(-1/2) @ (x - μ)` is dimensionless
  - When scaling: `x' = x/r, μ' = μ/r, Σ' = Σ/r²`
  - Result: `y' = (Σ/r²)^(-1/2) @ ((x/r) - (μ/r)) = (r² / Σ)^(1/2) @ (x-μ)/r = y`
  - Therefore: Mahalanobis distance is scale-invariant
- **Interpretation**: Sharpness controls falloff per sigma, independent of absolute spatial scale

### Algorithm

**Thin Wrapper Architecture**: Use existing `fit_gaussian_splats()` as building block

**Step 1: Multi-Scale Decomposition**
```python
from luxar.gsplats.multiscale import decompose_image

scales_list, stats = decompose_image(
    V,
    scales=scales,  # e.g., [1, 2, 4, 8]
    n_iters=n_iters_decomp,
    verbose=verbose
)
```

**Step 2: Independent Fitting Per Scale**
```python
all_params = []
all_amps = []
all_sharpness = []

for scale_idx, (scale_factor, V_scale) in enumerate(zip(scales, scales_list)):
    # Adjust init_sigma for scale
    init_sigma_scaled = base_init_sigma * scale_factor

    # Fit Gaussians on this scale
    params, amps, stats_scale = fit_gaussian_splats(
        V_scale,
        init_sigma_vox=init_sigma_scaled,
        n_iters=n_iters_per_scale,
        lr=lr,
        **kwargs
    )

    # Extract sharpness from params (always present in last column)
    d = V.ndim
    sharpness_scale = params[:, -1]
    params_geom = params[:, :-1]  # Everything except sharpness

    # Scale parameters back to full resolution
    if scale_factor > 1:
        # Scale centers (first d columns)
        params_geom[:, :d] *= scale_factor
        # Scale Cholesky factors (remaining columns)
        params_geom[:, d:] *= scale_factor

    all_params.append(params_geom)
    all_amps.append(amps)
    all_sharpness.append(sharpness_scale)
```

**Step 3: Combination**
```python
# Combine all scales
params_combined = np.vstack(all_params)
amps_combined = np.concatenate(all_amps)
sharpness_combined = np.concatenate(all_sharpness)

# Add sharpness column back to create final params
params_final = np.column_stack([params_combined, sharpness_combined])

return params_final, amps_combined, combined_stats
```

### API Design

**Primary Function Signature:**

```python
def fit_multiscale_gaussian_splats(
    V: np.ndarray,
    scales: List[int] = None,
    base_init_sigma: float = 1.5,
    n_iters_decomp: int = 1000,
    n_iters_per_scale: int = 500,
    lr: float = 0.01,
    verbose: bool = False,
    **fit_kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit Gaussian splats using multi-scale decomposition for computational efficiency.

    This function decomposes the input image into multiple scales, fits Gaussian splats
    independently on each scale (coarse scales have fewer voxels → faster), scales
    parameters back to full resolution, and combines all splats together.

    Parameters
    ----------
    V : np.ndarray
        Input image of shape (d1, d2, ..., dn). Must be non-empty and finite.
    scales : List[int], optional
        Scale factors for decomposition, e.g., [1, 2, 4, 8].
        Larger values = coarser scales with fewer voxels.
        Default: [1, 2, 4, 8] for most applications.
    base_init_sigma : float, default=1.5
        Base initial sigma in voxels. Will be multiplied by scale_factor
        for each scale to create scale-appropriate Gaussians.
    n_iters_decomp : int, default=1000
        Number of iterations for multi-scale decomposition.
    n_iters_per_scale : int, default=500
        Number of iterations for fitting Gaussians on each scale.
        Fewer iterations needed per scale due to fewer voxels.
    lr : float, default=0.01
        Learning rate passed to fit_gaussian_splats().
    verbose : bool, default=False
        Enable verbose logging of decomposition and fitting progress.
    **fit_kwargs
        Additional arguments passed to fit_gaussian_splats() for each scale.
        Supports all parameters: loss_type, asymmetric_penalty, l1_amp,
        dynamic_ops, max_abs_error, etc.

    Returns
    -------
    params : np.ndarray
        Combined parameters from all scales, shape (N_total, d + d*(d+1)//2 + 1).
        Includes: centers, Cholesky factors, and sharpness values.
        All parameters scaled to full resolution.
    amplitudes : np.ndarray
        Combined amplitudes from all scales, shape (N_total,).
    stats : Dict[str, Any]
        Statistics including:
        - 'decomposition_stats': Stats from decompose_image()
        - 'per_scale_stats': List of stats from each scale's fitting
        - 'n_splats_per_scale': Number of splats fitted per scale
        - 'total_splats': Total number of splats across all scales
        - 'computational_speedup': Estimated speedup vs single-scale fitting

    Notes
    -----
    - Computational speedup: Scale factor r reduces voxel count by (1/r)^d
      * 2D with scale 8: 64× fewer pixels
      * 3D with scale 8: 512× fewer voxels
    - Coarse scales capture large structures efficiently
    - Fine scales capture details at full resolution
    - Amplitudes and sharpness values do not scale (see Parameter Scaling Rules)
    - Uses existing fit_gaussian_splats() as building block (thin wrapper)

    Examples
    --------
    >>> # Simple 2D example
    >>> params, amps, stats = fit_multiscale_gaussian_splats(
    ...     image_2d,
    ...     scales=[1, 2, 4],
    ...     n_iters_per_scale=300
    ... )

    >>> # 3D volume with custom parameters
    >>> params, amps, stats = fit_multiscale_gaussian_splats(
    ...     volume_3d,
    ...     scales=[1, 2, 4, 8, 16],
    ...     base_init_sigma=2.0,
    ...     n_iters_decomp=2000,
    ...     n_iters_per_scale=500,
    ...     lr=0.02,
    ...     loss_type='l1',
    ...     max_abs_error=0.01,
    ...     verbose=True
    ... )
    """
```

**Return Statistics:**

```python
stats = {
    'decomposition_stats': {
        'final_error': float,
        'energy_distribution': List[float],  # Per-scale energy fractions
        'time_seconds': float
    },
    'per_scale_stats': [
        {
            'scale_factor': int,
            'scale_shape': Tuple[int, ...],
            'n_voxels': int,
            'n_splats': int,
            'final_error': float,
            'time_seconds': float
        },
        ...
    ],
    'n_splats_per_scale': List[int],
    'total_splats': int,
    'computational_speedup': float,  # Estimated vs single-scale
    'total_time_seconds': float
}
```

### Thin Wrapper Architecture

**Design Philosophy:**
- **Reuse existing implementation**: Call `fit_gaussian_splats()` as black-box building block
- **Minimal new code**: ~150-200 lines total for orchestration
- **No reimplementation**: Leverage all existing features (dynamic ops, convergence, etc.)
- **Simple coordination**: Decompose → Loop → Fit → Scale → Combine

**Implementation Structure:**

```python
def fit_multiscale_gaussian_splats(V, scales, ...):
    # 1. Decompose (uses luxar.gsplats.multiscale.decompose_image)
    scales_list, decomp_stats = decompose_image(V, scales=scales, ...)

    # 2. Loop through scales
    all_results = []
    for scale_factor, V_scale in zip(scales, scales_list):
        # 3. Fit using existing function (black box)
        params, amps, stats = fit_gaussian_splats(
            V_scale,
            init_sigma_vox=base_init_sigma * scale_factor,
            ...
        )

        # 4. Scale parameters back
        scaled_params = _scale_parameters(params, scale_factor, V.ndim)
        all_results.append((scaled_params, amps, stats))

    # 5. Combine
    return _combine_scales(all_results)
```

**Benefits:**
- Inherits all features from `fit_gaussian_splats()` automatically
- Easy to maintain (single implementation to update)
- Consistent behavior across single-scale and multi-scale fitting
- Can pass any parameter through to underlying fitter

### Computational Complexity

**Voxel Count Reduction:**

For scale factor r in d-dimensional space:
- **Original voxels**: N = ∏ᵢ shape[i]
- **Downsampled voxels**: N' = N / r^d
- **Speedup factor**: r^d

**Examples:**

| Dimension | Scale | Voxel Reduction | Example (512³) |
|-----------|-------|-----------------|----------------|
| 2D        | 2×    | 4×              | 512² → 128K    |
| 2D        | 8×    | 64×             | 512² → 4K      |
| 3D        | 2×    | 8×              | 512³ → 16M     |
| 3D        | 4×    | 64×             | 512³ → 2M      |
| 3D        | 8×    | 512×            | 512³ → 256K    |
| 4D        | 4×    | 256×            | 64⁴ → 64K      |
| 4D        | 8×    | 4096×           | 64⁴ → 4K       |

**Total Computational Cost:**

For scales [1, 2, 4, 8] in 3D:
- Scale 1×: 100% of voxels (baseline)
- Scale 2×: 12.5% of voxels (8× fewer)
- Scale 4×: 1.56% of voxels (64× fewer)
- Scale 8×: 0.195% of voxels (512× fewer)
- **Total**: ~114% of baseline (14% overhead for 3 additional scales!)

**Expected Speedup:**

With proper iteration distribution (more iterations on coarse scales):
- 2D: 10-20× speedup
- 3D: 50-100× speedup
- 4D: 200-500× speedup

### Expected Results

**Splat Distribution:**
- Coarse scales: Fewer, larger Gaussians capturing global structure
- Fine scales: More, smaller Gaussians capturing details
- Natural hierarchy: Energy concentrated in coarse scales (as per decomposition)

**Reconstruction Quality:**
- Similar or better than single-scale fitting
- Better capture of multi-scale features
- Potential for hierarchical LOD representation

**Performance:**
- 10-100× faster for 3D datasets
- Memory usage proportional to number of splats, not voxels
- Scalable to very large volumes

**Use Cases:**
- Large microscopy volumes (1024³ and beyond)
- Time-lapse 3D imaging (xyzt)
- Multi-channel 3D volumes (xyzc)
- Any high-dimensional data with hierarchical structure

## 7. Utilities

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

## 8. Integration Requirements

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

## Terminology Glossary

**Note**: For the complete, detailed glossary with usage guidelines, see [GLOSSARY.md](./GLOSSARY.md)

This section provides a quick reference for the most important terms. For comprehensive definitions, naming conventions, and cross-references, consult the dedicated glossary.

### Core Concepts

**Splat**
- A single oriented Gaussian function: `f(x) = a * exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))`
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

**Sharpness (s)**
- Controls edge falloff in generalized Gaussian: `exp(-0.5 * ||y||^s)`
- Parameter name: `sharpness_offsets_raw` (where `s = 2 * exp(s')`)
- `s = 2`: standard Gaussian, `s > 2`: sharper edges, `s < 2`: softer edges
- NOT: shape parameter, falloff rate

### Operations

**Seeding**
- **Initial seeding**: Generating candidate splat locations from image features (startup)
- **Dynamic seeding**: Adding new splats in high-residual regions during optimization
- Use "seeding" for initial generation, "adding" for dynamic operations

**Pruning**
- **Quality-based removal**: Removing low-importance splats based on reconstruction quality
- Distinct from generic "removing" which includes any deletion operation
- Use "pruning" when referring to quality-based removal

**Dynamic Operations**
- Umbrella term for adaptive topology changes during optimization
- Includes: seeding new splats, pruning low-quality splats, learning rate boosting
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
- Saving the parameter configuration that achieved the lowest max absolute error
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
- `sharpness_offsets_raw`: Sharpness offset parameters (s')

**Transformed Parameters** (after activation functions):
- `centers` or `μ`: Actual center positions in voxel coordinates
- `L` or `Ls`: Cholesky factors (lower-triangular matrices)
- `amps` or `a`: Actual amplitudes (non-negative)
- `sharpness` or `s`: Actual sharpness values (s = 2 * exp(s'))

**Configuration Parameters**:
- `sigma_min_diag`: Minimum diagonal values (per-dimension sequence)
- `sigma_max_diag`: Maximum diagonal values (per-dimension sequence)
- `truncate`: Gaussian truncation radius in standard deviations
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

**s (sharpness)**
- Generalized Gaussian exponent
- Computational form: `exp(-0.5 * ||y||^s)` where s=sharpness

**s' (sharpness offset)**
- Learnable parameter: `s = 2 * exp(s')`
- Zero-centered: s'=0 gives s=2 (standard Gaussian)

### Cross-Package References

When referencing other specifications, use this format:

**Format**: `[Package Name](./package/SPECIFICATIONS.md)` → **Section Name**

**Examples**:
- Gradient dilution details: [utils/SPECIFICATIONS.md](./utils/SPECIFICATIONS.md) → Section 2
- Fitting pipeline: [fitting/SPECIFICATIONS.md](./fitting/SPECIFICATIONS.md) → Pipeline Architecture
- Per-splat optimizer: [optim/SPECIFICATIONS.md](./optim/SPECIFICATIONS.md) → PerSplatAdam

## Conclusion

This specification provides complete implementation details for a mathematically rigorous, computationally efficient, and feature-rich Gaussian splatting system with dynamic optimization capabilities validated from 2D to 4D and beyond.

**For detailed information on specific components**, see the [Documentation Structure](#documentation-structure) section at the top of this document.

