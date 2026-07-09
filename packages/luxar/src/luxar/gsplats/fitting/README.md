# Fitting Pipeline Architecture

This directory contains the modular fitting pipeline for Gaussian splat optimization, organized into focused, maintainable modules following clean architecture principles.

## Overview

The fitting pipeline orchestrates the entire process of fitting n-dimensional Gaussian splats to input data. It follows a clear, unidirectional data flow through six well-defined stages.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        fit_gsplats.py                                │
│                    (Orchestration Layer)                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Validation & Configuration                                 │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ validation.py: prepare_fit_config()                             │ │
│ │ • Validates all input parameters                                │ │
│ │ • Checks array dimensions and types                             │ │
│ │ • Validates numeric ranges and constraints                      │ │
│ │ • Creates FitConfig dataclass                                   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2: Data Preprocessing                                         │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ preprocessing.py: preprocess_data()                             │ │
│ │ • Normalizes input data to [0, 1] range                         │ │
│ │ • Generates seed centers (auto or from user input)              │ │
│ │ • Sets L1 regularization defaults (proportional to base lr)    │ │
│ │ • Sets convergence thresholds                                   │ │
│ │ • Creates PreprocessedData dataclass                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3: Model & Optimizer Initialization                           │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ initialization.py: initialize_optimization()                    │ │
│ │ • Creates GaussianSplatModel with initial parameters            │ │
│ │ • Initializes standard PyTorch Adam optimizer                   │ │
│ │ • Sets up learning rate scheduler (plateau/exponential)         │ │
│ │ • Returns ModelComponents dataclass                             │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4: Loss Function Creation                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ losses.py: create_loss_function()                               │ │
│ │ • Creates loss function based on config (MSE/Poisson/L1)        │ │
│ │ • Applies asymmetric penalty for over-prediction (default 1.0)  │ │
│ │ • Adds L1 regularization on amplitudes (sparsity)               │ │
│ │ • Adds L1 regularization on diagonals (shape control)           │ │
│ │ • Adds boundary penalty for splats extending beyond bounds       │ │
│ │ • Returns closure that computes total loss                      │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 5: Optimization Loop                                          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ optimization.py: run_optimization_loop()                        │ │
│ │ • Runs main training loop (forward/backward/step)               │ │
│ │ • Tracks best state based on loss                               │ │
│ │ • Checks convergence criteria                                   │ │
│ │ • Applies dynamic operations (splat relocation)                 │ │
│ │ • Periodic Z-order sorting for cache locality (sorting.py)      │ │
│ │ • Records movie frames (optional)                               │ │
│ │ • Returns OptimizationResults dataclass                         │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 6: Result Finalization                                        │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ results.py: finalize_results()                                  │ │
│ │ • Extracts final parameters from best state                     │ │
│ │ • Rescales amplitudes to original intensity range               │ │
│ │ • Applies post-processing (clip-to-bounds, voxel footprint)     │ │
│ │ • Rescales from downscaled to original coordinates              │ │
│ │ • Computes quality metrics (PSNR, SSIM, MSE) in voxel space    │ │
│ │ • Compiles optimization statistics                              │ │
│ │ • Stores movie frames for visualization                         │ │
│ │ • Returns GSplatData dataclass                                  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Optional: Visualization                                              │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ visualization.py                                                 │ │
│ │ • display_compression_analysis() - Shows compression stats      │ │
│ │ • show_optimization_movie() - Displays napari convergence movie │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `config.py` - Type-Safe Configuration
**Purpose:** Defines dataclasses for type-safe configuration and data passing.

**Key Classes:**
- `FitConfig`: All parameters needed for fitting (mutable dataclass)
- `PreprocessedData`: Normalized data and metadata
- `OptimizationResults`: Results from optimization
- `ModelComponents`: Model, optimizer, scheduler
- `OptimConfig`: Frozen dataclass for optimization hyperparameters (n_iters, lr, etc.)
- `LossConfig`: Frozen dataclass for loss function configuration (loss_type defaults to `"l1"`, asymmetric_penalty, etc.)
- `ConstraintConfig`: Frozen dataclass for constraint configuration (amp_max, max_eccentricity, etc.)

**Key Type Alias:**
- `IterCallback = Callable[[int, torch.Tensor, Dict[str, Any]], None]`: Optional per-iteration
  callback signature wired through `FitConfig.iter_callback`. Invoked inside `torch.no_grad()`
  alongside the periodic eval with the 1-based iteration index, the detached prediction tensor,
  and an info dict (`loss`, `best_loss`, `max_abs_error`, `rel_l2`, `n_splats`). Used for
  validation-set scoring, custom snapshot logging, or stop-on-external-criterion.

**Design Pattern:** Configuration objects with validation at boundaries.

### `validation.py` - Input Validation
**Purpose:** Validates all user inputs at the API boundary before processing.

**Key Function:** `prepare_fit_config(fitter, V, **kwargs) -> FitConfig`

**Validations:**
- Array dimensions and shapes
- Numeric ranges (lr > 0, sigma constraints)
- Type compatibility (loss_type in allowed values)
- Parameter consistency (sigma_max > sigma_min)

**Philosophy:** Fail fast with clear error messages at the API boundary.

### `preprocessing.py` - Data Preparation
**Purpose:** Prepares input data for optimization.

**Key Function:** `preprocess_data(config: FitConfig) -> PreprocessedData`

**Operations:**
1. **Background floor suppression** (`config.floor`, default `"auto"`): subtracts a
   constant background pedestal / DC offset before normalization by raising the
   effective `image_min` (`_resolve_floor` → `estimate_floor`). `auto` = capped
   histogram mode (a no-op on clean data), `pN` = Nth percentile, `<float>` =
   fixed, `none` = disabled. The subtracted level is recorded on
   `PreprocessedData.floor`; it is NOT added back (output amplitudes are
   background-relative).
2. **Normalization**: Converts input to [0, 1] range (floor/percentile-based or full range)
3. **Seed Generation**: Creates initial splat positions (auto or user-provided)
4. **L1 Regularization Defaults**: Sets proportional defaults based on base learning rate and parameter type multipliers
5. **Convergence Threshold**: Sets sensible default if not specified

**Key Insight:** L1 regularization is proportional to learning rate for consistent sparsity pressure across different learning rate choices.

### `initialization.py` - Model Setup
**Purpose:** Initializes the optimization components.

**Key Function:** `initialize_optimization(config, preprocessed_data) -> ModelComponents`

**Creates:**
1. **GaussianSplatModel**: PyTorch model with initial parameters
2. **Standard Adam**: PyTorch Adam optimizer with gradient dilution compensation
3. **Scheduler**: ReduceLROnPlateau or ExponentialLR

**Design Choice:** Standard PyTorch Adam with fixed-pool architecture enables 50x+ faster optimization compared to per-splat alternatives.

### `losses.py` - Loss Functions
**Purpose:** Creates the loss function based on configuration.

**Key Function:** `create_loss_function(config, preprocessed_data, model) -> Callable`

**Supported Losses:**
- **MSE**: Mean Squared Error (general purpose)
- **Poisson**: Poisson deviance (count/photon data)
- **L1**: Mean Absolute Error (robust to outliers)

**Asymmetric Penalty:** Configurable penalty for over-prediction (pred > target), default 1.0
- **Rationale**: Additive Gaussian models can easily add splats (fix under-prediction) but struggle to reduce intensity (fix over-prediction)

**Regularization:**
- **L1 on amplitudes**: Encourages sparsity (fewer active splats)
- **L1 on diagonal**: Encourages smaller, more isotropic splats
- **Boundary penalty**: Penalizes splats extending beyond volume bounds

### `optimization.py` - Training Loop
**Purpose:** Runs the main optimization loop.

**Key Function:** `run_optimization_loop(components, loss_fn, config, preprocessed_data) -> OptimizationResults`

**Loop Structure:**
1. Forward pass → Compute loss
2. Backward pass → Compute gradients
3. Gradient clipping → Stability
4. Optimizer step → Update parameters
5. Scheduler step → Adapt learning rate
6. Convergence check → Early stopping
7. Dynamic operations → Adaptive topology
8. Best state tracking → Quality guarantee

**Key Innovation:** Tracks best state throughout optimization (not just final state), ensuring quality even with non-monotonic convergence.

### `results.py` - Result Finalization
**Purpose:** Processes and packages optimization results.

**Key Function:** `finalize_results(optimization_results, config, preprocessed_data) -> GSplatData`

**Operations:**
1. Extracts parameters from best state (not final state)
2. Converts tensors to numpy arrays with separate fields:
   - `centers`: Shape `(N, d)` - splat center positions
   - `cholesky_factors`: Shape `(N, d*(d+1)//2)` - packed lower-triangular Cholesky factors
   - `amplitudes`: Shape `(N,)` - rescaled to original intensity range
3. Applies optional post-processing:
   - Clip-to-bounds (ensures splats stay within volume)
   - Voxel footprint correction (inflates covariances)
   - Downscale rescaling (restores original resolution coordinates)
   - Physical coordinate conversion (voxel to real space via voxel_size)
4. Computes quality metrics (PSNR, SSIM, MSE) by rendering splats back to a volume and comparing against the original input. Only computed in voxel space (skipped when `output_space="real"`)
5. Compiles comprehensive statistics and quality metrics
6. Stores movie frames for visualization
7. Returns `GSplatData` dataclass containing all results

**Critical Details:**
- Returns a structured `GSplatData` dataclass with named fields (centers, amplitudes, cholesky_factors, stats)
- Amplitudes are rescaled using the original intensity range, allowing direct comparison with input data
- All arrays are separate fields, not concatenated (easier to work with)

### `visualization.py` - Display Helpers
**Purpose:** Optional visualization of results.

**Key Functions:**
- `display_compression_analysis()`: Shows compression statistics
- `show_optimization_movie()`: Displays napari convergence movie

**Design Choice:** Visualization is separate from core fitting pipeline, allowing headless operation.

### `sorting.py` - Z-Order Sorting
**Purpose:** Reorders splats by Morton (Z-order) code for GPU cache locality.

**Key Function:** `sort_splats_by_morton_order(model, optimizer, relocation_tracker)`

Permutes all model parameters, optimizer state, and relocation tracker state to match the Morton code ordering of splat centers. Applied at initialization and periodically during optimization.

### `downscale.py` - Volume Downscaling
**Purpose:** Anti-aliased integer downscaling of volumes before fitting, with coordinate rescaling back to original resolution after fitting.

**Key Functions:**
- `normalize_downscale(downscale, ndim)` - Normalize downscale parameter to per-axis tuple
- `downscale_volume(V, factors)` - Gaussian blur + decimation
- `rescale_centers(centers, factors)` - Scale centers back to original coordinates
- `rescale_cholesky_packed(cholesky_packed, factors)` - Scale packed Cholesky factors back

## Data Flow

### Configuration Flow
```
User Parameters
    ↓
prepare_fit_config() → FitConfig (validated)
    ↓
preprocess_data() → PreprocessedData (normalized, with metadata)
    ↓
initialize_optimization() → ModelComponents (model + optimizer + scheduler)
```

### Optimization Flow
```
ModelComponents + PreprocessedData + LossFunction
    ↓
run_optimization_loop()
    │
    ├─> Forward pass (model prediction)
    ├─> Loss computation
    ├─> Backward pass (gradients)
    ├─> Optimizer step (parameter update)
    ├─> Convergence check
    ├─> Dynamic operations (topology changes)
    └─> Best state tracking
    ↓
OptimizationResults (best parameters + statistics)
```

### Result Flow
```
OptimizationResults
    ↓
finalize_results()
    │
    ├─> Extract best state parameters
    ├─> Rescale amplitudes to original range
    ├─> Apply post-processing (clip-to-bounds, voxel footprint, downscale rescaling)
    ├─> Compute quality metrics (PSNR, SSIM, MSE)
    ├─> Compile statistics
    └─> Package movie frames
    ↓
GSplatData (centers, amplitudes, cholesky_factors, stats)
```

## Design Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility. Changes to one stage don't affect others.

### 2. Type Safety
All data passed between stages uses typed dataclasses, catching errors at configuration time rather than runtime.

### 3. Unidirectional Flow
Data flows in one direction through the pipeline. No circular dependencies or backflow.

### 4. Fail Fast
Input validation happens early at API boundaries with clear error messages.

### 5. Testability
Each module can be tested independently with mock inputs.

### 6. Best State Guarantee
Optimization always returns the best state encountered, not the final state. This protects against non-monotonic convergence and dynamic operation fluctuations.

## Key Algorithms

### Gradient Dilution Compensation
**Problem:** Higher dimensions have more parameters per splat, diluting gradients.

**Where Applied:** Optimizer factory function (`create_optimizer_and_scheduler`) automatically applies gradient dilution compensation.

**Solution:** Scale learning rate based on dimensional and parameter complexity:
```python
# For d ≤ 3: Conservative scaling
gradient_dilution_factor = params_current / params_2d

# For d > 3: Enhanced scaling
dimensional_complexity = d ** 0.8  # Spatial complexity
parameter_complexity = params_current / params_2d  # Parameter dilution
gradient_dilution_factor = dimensional_complexity * parameter_complexity

effective_lr = base_lr * gradient_dilution_factor
```

**Impact:** 2D: 1.0x, 3D: 1.8x, 4D: 8.5x learning rate multiplier.

### Best State Tracking
**Problem:** Optimization may not be monotonic, especially with dynamic operations.

**Solution:** Track best state based on loss:
```python
if current_loss < best_loss:
    best_loss = current_loss
    best_state = save_current_state()

# At end, restore best state
return best_state  # Not final state!
```

**Impact:** Guarantees quality even with convergence fluctuations.

### Convergence Criteria
**Metric:** Maximum absolute error (not mean error)
```python
max_abs_error = max(|prediction - target|)
converged = max_abs_error < threshold
```

**Rationale:** Ensures all regions meet quality standards, not just on average.

## Extension Points

### Adding New Loss Functions
1. Add implementation to `losses.py`:
   ```python
   def _compute_my_loss(pred, target, asymmetric_penalty):
       # Implementation
       return loss
   ```

2. Add to `create_loss_function()`:
   ```python
   elif loss_type.lower() == "my_loss":
       data = _compute_my_loss(pred, V_t, asymmetric_penalty)
   ```

3. Update validation in `validation.py`:
   ```python
   if loss_type not in ["mse", "poisson", "l1", "my_loss"]:
       raise ValueError(...)
   ```

### Adding New Preprocessing Steps
Add functions to `preprocessing.py` and call from `preprocess_data()`.

### Modifying Optimization Loop
Key extension points in `optimization.py`:
- Before loop: Add initialization
- In loop: Add custom operations
- After loop: Add post-processing

## Testing Strategy

The fitting pipeline has comprehensive test coverage in `fitting/tests/`.

### Unit Tests (`fitting/tests/`)

**Configuration & Validation:**
- `test_fitting_config.py` - Dataclass creation and validation
- `test_fitting_preprocessing.py` - Data normalization and seed generation
- `test_fitting_validation.py` - Input validation and error handling

**Pipeline Components:**
- `test_initialization.py` - Model, optimizer, and scheduler initialization
- `test_losses.py` - Loss functions, asymmetric penalties, L1 regularization
- `test_optimization.py` - Training loop, convergence, dynamic operations
- `test_results.py` - Result finalization, amplitude rescaling, statistics
- `test_visualization.py` - Compression analysis, napari movie (mocked)
- `test_sorting.py` - Z-order Morton code sorting
- `test_downscale.py` - Volume downscaling and coordinate rescaling

**Test Coverage:**
- All modules tested with normal cases, edge cases, and error conditions
- 2D, 3D, and nD scenarios covered
- Device compatibility (CPU, CUDA, MPS when available)
- Gradient flow and numerical stability verified
- No interactive windows opened during tests (napari mocked)

**Running Tests:**
```bash
# Run all fitting tests
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/ -v

# Run specific test file
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/test_losses.py -v
```

### Integration Tests (`../tests/`)
See `../tests/test_fit_gsplats.py` and `../tests/test_gsplats_integration.py` for:
- Complete pipeline flow with known ground truth
- Edge cases (empty input, single splat, uniform images)
- Convergence behavior across loss types
- Dynamic operations integration

### Property Tests
Verify mathematical invariants:
- Normalized data is in [0, 1]
- Best state has lowest error
- Gradient dilution formula is correct
- Amplitude rescaling preserves relative magnitudes

## Performance Considerations

### Memory
- Preprocessing creates a copy of input data
- Movie recording can consume significant memory (use `movie_max_frames`)
- Best state tracking stores one snapshot (minimal overhead)

### Computation
- Most time spent in optimization loop (Stage 5)
- Preprocessing and finalization are fast (<1% of total time)
- GPU acceleration primarily benefits optimization loop

## References

**Related Modules:**
- `../models/gsplats/gsplat_model.py` - PyTorch model definition
- `../optim/integration.py` - Optimizer factory with gradient dilution
- `./dynamic_ops/` - Fixed-pool splat relocation operations
- `./sorting.py` - Z-order (Morton code) sorting for cache locality
- `./downscale.py` - Volume downscaling and coordinate rescaling
- `../seeds/` - Seed generation

**Documentation:**
- `../README.md` - Main package documentation
