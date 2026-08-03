# Gaussian Splatting Terminology Glossary

This glossary defines standard terminology used throughout the gsplats package specifications and codebase. Use these terms consistently when writing documentation, code comments, and specifications.

## Core Concepts

### Splat
**Definition**: A single oriented Gaussian function in the reconstruction.

**Mathematical form** (shifted Gaussian for C⁰ continuity at truncation boundary): `f_k(x) = a_k * scale * max(0, exp(-0.5 * (x-μ_k)^T * Σ_k^(-1) * (x-μ_k)) - C)` where `C = exp(-0.5 * T²)`, `scale = 1/(1-C)`, T = truncation radius

**Standard terms**:
- ✅ **Splat** (preferred)
- ✅ **Gaussian splat**
- ✅ **Oriented Gaussian**

**Avoid**:
- ❌ Blob
- ❌ Particle
- ❌ Kernel
- ❌ Basis function

**Rationale**: "Splat" is the established term in the graphics/splatting literature.

---

### Centers (μ)
**Definition**: Position of splat center in voxel coordinates.

**Shape**: `(d,)` for single splat, `(N, d)` for batch of N splats

**Parameter names**:
- `raw_mu`: Learnable parameter in logit space
- `centers` or `μ`: Transformed parameter in voxel coordinates

**Standard terms**:
- ✅ **Centers** (preferred)
- ✅ **μ** (mathematical notation)

**Avoid**:
- ❌ Positions
- ❌ Means
- ❌ Locations

**Rationale**: "Centers" is unambiguous and matches mathematical convention (μ).

---

### Amplitudes (a)
**Definition**: Non-negative scalar controlling splat brightness/intensity.

**Shape**: `(,)` for single splat (scalar), `(N,)` for batch

**Parameter names**:
- `raw_a`: Learnable parameter (via softplus activation)
- `amps` or `a`: Transformed parameter (non-negative)

**Standard terms**:
- ✅ **Amplitudes** (preferred)
- ✅ **a** (mathematical notation)

**Avoid**:
- ❌ Weights
- ❌ Intensities
- ❌ Coefficients
- ❌ Strengths

**Rationale**: "Amplitudes" clearly indicates the multiplicative scaling factor.

---

### Cholesky Factors (L)
**Definition**: Lower-triangular matrix parameterizing covariance: `Σ = L @ L^T`

**Shape**: `(d, d)` for single splat, `(N, d, d)` for batch

**Parameter names**:
- `raw_L_diag`: Learnable diagonal elements (via softplus for positivity)
- `L_off`: Learnable off-diagonal elements (unconstrained)
- `L` or `Ls`: Reconstructed lower-triangular matrices

**Standard terms**:
- ✅ **Cholesky factors** (preferred)
- ✅ **L-matrices**
- ✅ **Lower-triangular factors**

**Avoid**:
- ❌ Covariance matrices (Σ ≠ L, they're related by Σ = L @ L^T)
- ❌ Covariance roots
- ❌ Variance matrices

**Rationale**: Cholesky decomposition is the standard technique; using precise terminology avoids confusion.

---

## Operations

### Seeding
**Definition**: Generating initial splat center locations from image features.

**Two contexts**:
1. **Initial seeding**: Startup phase, generating candidate locations
2. **Dynamic relocation**: During optimization, moving weak splats to high-residual regions (fixed pool, constant count)

**Standard terms**:
- ✅ **Seeding** (for initial candidate generation)
- ✅ **Relocation** (for dynamic operations during optimization — fixed pool, no add/remove)
- ✅ **Dynamic relocation** (to clarify context)

**Functions**:
- `seed_from_edges()`, `seed_from_grid()`, `seed_from_decomposition()`: Individual seeding methods
- `generate_seeds()`: Unified seeding entry point
- `append_()`: model method that appends splats (the default dynamic ops use fixed-pool relocation with a constant splat count, not add/remove)

**Rationale**: "Seeding" emphasizes the initialization aspect; "relocation" makes the constant-count behavior explicit — the pool is fixed, splats move rather than being added or removed.

---

### Pruning
**Definition**: Removing splats based on quality/importance criteria.

**Criteria**:
- Low importance (amplitude × volume)
- Minimal contribution to reconstruction
- Local convergence maintained after removal

**Standard terms**:
- ✅ **Pruning** (quality-based removal)
- ✅ **Removing** (generic deletion)

**Functions**:
- `prune_()`: model method that drops splats (not used by the default fixed-pool relocation, which keeps a constant count)

**Avoid**:
- ❌ Deleting (too generic)
- ❌ Culling (unclear criteria)

**Rationale**: "Pruning" is standard in neural network literature for quality-based removal.

---

### Dynamic Operations
**Definition**: Fixed-pool splat relocation during optimization (constant splat count, no topology changes).

**Includes**:
- Detecting weak splats (low importance = amplitude × volume)
- Detecting high-residual peaks
- Relocating weak splats to those peaks (parameter update only, no add/remove)

**Standard terms**:
- ✅ **Dynamic operations** (umbrella term)
- ✅ **Adaptive operations**
- ✅ **Fixed-pool relocation**

**Avoid**:
- ❌ Splat management (too vague)
- ❌ Population control

**Configuration**: `DynamicOpsConfig` class

**Rationale**: "Dynamic operations" clearly indicates runtime adaptivity.

---

## Optimization Terminology

### Gradient Dilution
**Definition**: Phenomenon where higher dimensions have more parameters per splat, spreading gradients thinner.

**Cause**: Parameter count scales as `d + d*(d+1)//2`
- 2D: 5 parameters → baseline
- 3D: 9 parameters → 1.8× dilution
- 4D: 14 parameters → ~8.5× dilution (enhanced with dimensional complexity)

**Standard terms**:
- ✅ **Gradient dilution** (the phenomenon)
- ✅ **Gradient dilution compensation** (the solution)
- ✅ **Gradient dilution factor** (the multiplier)

**Avoid**:
- ❌ Gradient scaling (too generic)
- ❌ LR adjustment (doesn't explain why)
- ❌ Dimensional compensation (unclear)

**Related**: See [utils/README.md](./utils/README.md) → `calculate_gradient_dilution_factor()`

---

### Standard Adam with Fixed-Pool Architecture
**Definition**: Standard PyTorch Adam optimizer working with fixed tensor shapes.

**Key Insight**:
- Fixed-pool architecture keeps tensor shapes constant
- Splat relocation = parameter updates (no shape changes)
- Adam's momentum naturally adapts to relocated splats
- 50x+ faster than per-splat alternatives

**Standard terms**:
- ✅ **Standard Adam** (preferred)
- ✅ **Fixed-pool architecture**

**Avoid**:
- ❌ Per-splat optimizer (removed for performance)
- ❌ Individual state management

**Benefit**: 50x+ faster optimization with simpler architecture

**Related**: See [optim/README.md](./optim/README.md) → create_optimizer_and_scheduler

---

### Best State Tracking
**Definition**: Saving the parameter configuration that achieved the lowest loss during optimization.

**Purpose**: Quality guarantee even with non-monotonic optimization (e.g., due to dynamic operations)

**Standard terms**:
- ✅ **Best state tracking** (process)
- ✅ **Quality guarantee** (benefit)
- ✅ **Best state** (the saved configuration)

**Avoid**:
- ❌ Optimal state (implies global optimum)
- ❌ Peak performance (vague)
- ❌ Checkpoint (implies persistence)

**Related**: See [fitting/README.md](./fitting/README.md) → Best State Tracking

---

### Gradient Dilution Compensation
**Definition**: Automatic learning rate scaling based on dimensionality to counteract gradient dilution.

**Values** (applied by `create_optimizer_and_scheduler()`):
- 2D: ×1.0 (baseline)
- 3D: ×1.8
- 4D: ×8.5

**Standard terms**:
- ✅ **Gradient dilution compensation** (full)
- ✅ **Dilution factor** (abbreviated)

**Avoid**:
- ❌ Per-parameter-type LR (old architecture)
- ❌ Manual LR scaling (automatic now)

**Rationale**: Higher dimensions have more parameters per splat, diluting gradients.

---

## Dimensionality

### d
**Usage**: Mathematical notation for dimension count

**Examples**:
- d=2 → 2D image
- d=3 → 3D volume
- d=4 → 4D hypercube

**Context**: Equations, formulas, variable names

---

### nD
**Usage**: Prose and code for "n-dimensional"

**Convention**: Follows NumPy/SciPy standard

**Examples**:
- "nD Gaussian splatting"
- "Works for arbitrary nD data"

**Standard terms**:
- ✅ **nD** (preferred, NumPy convention)

**Avoid**:
- ❌ N-dimensional (verbose)
- ❌ n-D (hyphenated)
- ❌ ND (ambiguous)

---

### Dimensionality
**Usage**: Full word form in prose

**Examples**:
- "The algorithm supports arbitrary dimensionality"
- "Higher dimensionality requires gradient dilution compensation"

**Context**: Explanatory text, not mathematical notation

---

## Technical Abbreviations

### AABB
**Full name**: Axis-Aligned Bounding Box

**Definition**: Rectangular region enclosing splat's significant contribution

**Purpose**: Computational efficiency - only render within AABB

**Calculation**: `lo = center - radii`, `hi = center + radii + 1`

---

### LR
**Full name**: Learning Rate

**Usage**:
- ✅ **LR** in prose and headings
- ✅ **lr** in code and variable names

**Examples**:
- "The LR is reduced when plateaus are detected" (prose)
- `lr = 0.01` (code)

---

### DoG
**Full name**: Difference of Gaussians

**Formula**: `DoG(x, σ) = G(x, σ) - G(x, k*σ)` where typically k=1.6

**Usage**: Candidate generation for detecting features at different scales

---

### MSE
**Full name**: Mean Squared Error

**Formula**: `MSE = mean((pred - target)²)`

**Loss type**: Penalizes large errors heavily, sensitive to outliers

---

### MAE
**Full name**: Mean Absolute Error

**Formula**: `MAE = mean(|pred - target|)`

**Also called**: L1 Loss

**Loss type**: Robust to outliers, preserves sharp features

---

## Parameter Naming Conventions

### Raw Parameters (Learnable)
These are the actual `nn.Parameter` objects that PyTorch optimizes:

- **`raw_mu`**: Centers in logit space → `sigmoid(raw_mu)` gives normalized coordinates
- **`raw_L_diag`**: Diagonal elements → `σ_min + softplus(raw_L_diag)` gives positive diagonals
- **`L_off`**: Off-diagonal elements (unconstrained, no activation needed)
- **`raw_a`**: Amplitudes → `softplus(raw_a)` gives non-negative amplitudes

**Convention**: Use `raw_` prefix for parameters that need activation functions

---

### Transformed Parameters (Computed)
These are the actual values used in rendering and returned by `current_params()`:

- **`centers`** or **`μ`**: Center positions in voxel coordinates, shape `(N, d)`
- **`L`** or **`Ls`**: Cholesky factors (lower-triangular), shape `(N, d, d)`
- **`amps`** or **`a`**: Amplitudes (non-negative), shape `(N,)`

**Convention**: Use descriptive names without `raw_` prefix

---

### Configuration Parameters
These are user-facing configuration options:

- **`sigma_min_diag`**: Minimum allowed diagonal values (per-dimension), prevents degeneracy
- **`sigma_max_diag`**: Maximum allowed diagonal values (per-dimension), prevents over-smoothing
- **`truncate`**: Truncation radius in standard deviations (fitting default 2.75; model/rendering default 3.0)
- **`init_sigma_vox`**: Initial sigma for isotropic covariance initialization
- **`norm_percentile`**: Percentile for robust normalization (0 = full range)
- **`asymmetric_penalty`**: Over-prediction penalty factor (default: 1.0)

**Convention**: Use descriptive names with units when applicable (`_vox` = voxels)

---

## Mathematical Notation

### Σ (Sigma)
**Meaning**: Covariance matrix (positive definite)

**Relationship**: `Σ = L @ L^T` where L is Cholesky factor

**Usage**: Mathematical formulas only (not in code)

---

### μ (mu)
**Meaning**: Center position vector

**Shape**: `(d,)` for single splat

**Usage**: Mathematical formulas and brief code comments

---

### L (L-matrix)
**Meaning**: Lower-triangular Cholesky factor

**Property**: Ensures Σ is positive definite

**Usage**: Both math and code (`L` or `Ls` for batch)

---

## Optimization Concepts

### Gradient Dilution Compensation
**Full term**: Gradient dilution compensation

**Short forms**:
- ✅ Gradient dilution (the phenomenon)
- ✅ Dilution compensation (the solution)
- ✅ GD compensation (very brief)

**Definition**: Scaling learning rate to counteract gradient dilution from higher parameter counts

**Formula**: `effective_lr = base_lr × gradient_dilution_factor`

**See**: [utils/README.md](./utils/README.md) → `calculate_gradient_dilution_factor()`

---

### Convergence Threshold
**Standard terms**:
- ✅ **`max_abs_error`** (parameter name)
- ✅ **Maximum absolute error threshold** (full description)
- ✅ **Convergence threshold** (brief)

**Definition**: Optimization stops when `max(|pred - target|) < threshold`

**Default**: 0.01 (1% of normalized [0,1] range)

---

### L1 Regularization
**Standard forms**:
- ✅ **L1 regularization** (preferred)
- ✅ **L1 penalty**
- ✅ **Lasso penalty** (when emphasizing sparsity)

**Formula**: `loss += λ * Σ|θ_i|`

**Effect**: Encourages sparsity (drives parameters toward zero)

**Parameter names**:
- `l1_amp`: L1 on amplitudes
- `l1_diag`: L1 on diagonal elements

**Proportional defaults**: All default to proportion of base learning rate
- `l1_amp = 0.1 * lr` (10% of base LR)
- `l1_diag = 0.01 * lr` (1% of base LR)

---

## Device and Performance

### Device
**Standard terms**:
- ✅ **CPU**: Central Processing Unit (universal fallback)
- ✅ **CUDA**: NVIDIA GPU acceleration (preferred when available)
- ✅ **MPS**: Apple Silicon GPU (experimental)

**Auto-detection order**: CUDA → MPS → CPU

---

### Chunk Size
**Definition**: Number of spatial points processed in a single batch during rendering

**Purpose**: Memory management (avoid OOM on GPU)

**Calculation**: Based on available device memory

**Typical values**: 1024 to 1,048,576 points

---

### Truncation Radius
**Parameter name**: `truncate`

**Definition**: Distance in standard deviations beyond which Gaussian is considered negligible

**Default**: 3.0σ for the model/renderer (captures ~99.7% of Gaussian mass); the fitting API (`fit_gaussian_splats`) defaults to 2.75σ

**Effect**: Limits evaluation to the radius where the Gaussian contribution is negligible.

---

## Loss Functions

### Asymmetric Loss
**Definition**: Different penalties for over-prediction vs under-prediction

**Formula**: `penalty = F if pred > target else 1.0`

**Default factor**: F = 1.0

**Rationale**: Additive Gaussian models struggle to reduce intensity (can't have negative splats)

**Applies to**: MSE, Poisson, and L1 loss types

---

### Loss Types

**MSE (Mean Squared Error)**:
- Formula: `mean((pred - target)²)`
- Best for: Smooth data, Gaussian noise
- Properties: Fast convergence, sensitive to outliers

**Poisson (Poisson Deviance)**:
- Formula: `2 * mean(pred - target + target * log(target/pred))`
- Best for: Count data, photon counts, fluorescence microscopy
- Properties: Appropriate noise model for Poisson statistics

**L1 (Mean Absolute Error)**:
- Formula: `mean(|pred - target|)`
- Best for: Robust fitting, sharp features, outliers
- Properties: Preserves edges, encourages sparse residuals

---

## File and Module Naming

### Module References
When referencing modules in documentation:

**Format**: `` `module_name.py` `` or `package.submodule`

**Examples**:
- `candidates.py`: Candidate generation module
- `luxar.gsplats.fitting`: Fitting pipeline package
- `gsplat_model.py`: Model definition

---

### Package Paths
When referencing other packages:

**Format**: `[Package Name](relative/path/README.md)`

**Examples**:
- [Main README](README.md)
- [fitting/README.md](fitting/README.md)
- [optim/README.md](optim/README.md)

**Convention**: Always use relative paths from the current document.

---

## Usage Guidelines

### When Writing Documentation

1. **Use standard terms** from this glossary
2. **Avoid synonyms** (pick one term and stick with it)
3. **Define on first use** if not in glossary
4. **Cross-reference** other docs using standard format
5. **Update glossary** when introducing new concepts

### When Writing Code

1. **Use parameter naming conventions** from this glossary
2. **Comment with standard terms** for clarity
3. **Raise errors with glossary terms** for consistency
4. **Log messages** should use glossary terminology

### When Reviewing

1. **Check terminology consistency** against glossary
2. **Flag synonyms** and suggest standard terms
3. **Update glossary** if new standard terms emerge
4. **Verify cross-references** are accurate

---

## Quick Reference

### Common Term Pairs

| Correct ✅ | Avoid ❌ |
|-----------|---------|
| Splat | Blob, particle, kernel |
| Centers (μ) | Positions, means, locations |
| Amplitudes (a) | Weights, intensities |
| Cholesky factors (L) | Covariance matrices |
| Seeding | Initialization, spawning |
| Pruning | Deleting, culling |
| Dynamic operations | Splat management |
| Gradient dilution | Gradient scaling |
| Per-splat state | Splat-wise state |
| Best state tracking | Checkpointing |
| nD | N-dimensional, n-D |

### Parameter Name Quick Reference

| Concept | Raw Parameter | Transformed | Config |
|---------|--------------|-------------|--------|
| Centers | `raw_mu` | `centers`, `μ` | - |
| Diagonal | `raw_L_diag` | `L` (diagonal) | `sigma_min_diag`, `sigma_max_diag` |
| Off-diagonal | `L_off` | `L` (off-diag) | - |
| Amplitude | `raw_a` | `amps`, `a` | - |

---

## See Also

- **Main package**: [README.md](./README.md)
- **Fitting Pipeline**: [fitting/README.md](./fitting/README.md)
- **Optimizers**: [optim/README.md](./optim/README.md) - Standard Adam with gradient dilution
- **Dynamic Operations**: [fitting/dynamic_ops/README.md](./fitting/dynamic_ops/README.md) - Fixed-pool relocation
- **Models**: [models/README.md](./models/README.md) - PyTorch models
- **Utilities**: [utils/README.md](./utils/README.md) - Mathematical utilities
- **Multi-Scale**: [multiscale/README.md](./multiscale/README.md) - Image decomposition
