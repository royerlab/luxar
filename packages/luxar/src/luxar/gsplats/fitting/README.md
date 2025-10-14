# Fitting Pipeline Architecture

This directory contains the refactored modular fitting pipeline for Gaussian splat optimization. The pipeline was refactored from a monolithic 480+ line method into focused, maintainable modules following clean architecture principles.

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
│ │ • Initializes per-splat Adam optimizer                          │ │
│ │ • Sets up learning rate scheduler (plateau/exponential)         │ │
│ │ • Creates ModelOptimizerCoordinator for dynamic ops             │ │
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
│ │ • Applies asymmetric penalty (10× for over-prediction)          │ │
│ │ • Adds L1 regularization on amplitudes (sparsity)               │ │
│ │ • Adds L1 regularization on diagonals (shape control)           │ │
│ │ • Adds L1 regularization on sharpness (standard Gaussian bias)  │ │
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
│ │ • Tracks best state based on max absolute error                 │ │
│ │ • Checks convergence criteria                                   │ │
│ │ • Applies dynamic operations (seeding/pruning)                  │ │
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
│ │ • Compiles optimization statistics                              │ │
│ │ • Stores movie frames for visualization                         │ │
│ │ • Returns (params, amps, stats) tuple                           │ │
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
- `FitConfig`: All parameters needed for fitting
- `PreprocessedData`: Normalized data and metadata
- `OptimizationResults`: Results from optimization
- `ModelComponents`: Model, optimizer, scheduler, coordinator

**Design Pattern:** Immutable configuration objects with validation at boundaries.

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
1. **Normalization**: Converts input to [0, 1] range (percentile-based or full range)
2. **Seed Generation**: Creates initial splat positions (auto or user-provided)
3. **L1 Regularization Defaults**: Sets proportional defaults based on base learning rate and parameter type multipliers
4. **Convergence Threshold**: Sets sensible default if not specified

**Key Insight:** L1 regularization is proportional to learning rate for consistent sparsity pressure across different learning rate choices.

### `initialization.py` - Model Setup
**Purpose:** Initializes the optimization components.

**Key Function:** `initialize_optimization(config, preprocessed_data) -> ModelComponents`

**Creates:**
1. **GaussianSplatModel**: PyTorch model with initial parameters
2. **PerSplatAdam**: Optimizer with per-splat state management
3. **Scheduler**: ReduceLROnPlateau or ExponentialLR
4. **Coordinator**: Manages model-optimizer synchronization for dynamic ops

**Design Choice:** Per-splat optimizer enables momentum preservation during topology changes.

### `losses.py` - Loss Functions
**Purpose:** Creates the loss function based on configuration.

**Key Function:** `create_loss_function(config, preprocessed_data, model) -> Callable`

**Supported Losses:**
- **MSE**: Mean Squared Error (general purpose)
- **Poisson**: Poisson deviance (count/photon data)
- **L1**: Mean Absolute Error (robust to outliers)

**Asymmetric Penalty:** 10× penalty for over-prediction (pred > target)
- **Rationale**: Additive Gaussian models can easily add splats (fix under-prediction) but struggle to reduce intensity (fix over-prediction)

**Regularization:**
- **L1 on amplitudes**: Encourages sparsity (fewer active splats)
- **L1 on diagonal**: Encourages smaller, more isotropic splats
- **L1 on sharpness**: Encourages standard Gaussian (s=2) unless sharper edges improve fit

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

**Key Function:** `finalize_results(optimization_results, config, preprocessed_data) -> (params, amps, stats)`

**Operations:**
1. Extracts parameters from best state (not final state)
2. Rescales amplitudes to original intensity range
3. Compiles comprehensive statistics
4. Stores movie frames for visualization

**Critical Detail:** Amplitudes are rescaled using the original intensity range, allowing direct comparison with input data.

### `visualization.py` - Display Helpers
**Purpose:** Optional visualization of results.

**Key Functions:**
- `display_compression_analysis()`: Shows compression statistics
- `show_optimization_movie()`: Displays napari convergence movie

**Design Choice:** Visualization is separate from core fitting pipeline, allowing headless operation.

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
    ├─> Compile statistics
    └─> Package movie frames
    ↓
(params, amps, stats) - Ready for use
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

**Where Applied:** Optimizer (`per_splat_adam.py`) automatically applies gradient dilution compensation internally.

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

**Impact:** 2D: 1.0×, 3D: 2.0×, 4D: 7.1× learning rate multiplier.

**Sharpness Exception:** Sharpness parameters always use `base_lr` without gradient dilution, since sharpness is a single scalar value regardless of dimension (no parameter dilution occurs).

### Best State Tracking
**Problem:** Optimization may not be monotonic, especially with dynamic operations.

**Solution:** Track best state based on max absolute error:
```python
if current_max_abs_error < best_max_abs_error:
    best_max_abs_error = current_max_abs_error
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

The fitting pipeline has comprehensive test coverage in `fitting/tests/` (84 tests, 100% module coverage).

### Unit Tests (`fitting/tests/`)

**Configuration & Validation:**
- `test_fitting_config.py` (4 tests) - Dataclass creation and validation
- `test_fitting_preprocessing.py` (8 tests) - Data normalization and seed generation
- `test_fitting_validation.py` (16 tests) - Input validation and error handling

**Pipeline Components:**
- `test_initialization.py` (9 tests) - Model, optimizer, and scheduler initialization
- `test_losses.py` (12 tests) - Loss functions, asymmetric penalties, L1 regularization
- `test_optimization.py` (15 tests) - Training loop, convergence, dynamic operations
- `test_results.py` (12 tests) - Result finalization, amplitude rescaling, statistics
- `test_visualization.py` (9 tests) - Compression analysis, napari movie (mocked)

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

## Migration Notes

### From Monolithic to Modular (January 2025)
The refactoring preserved all functionality while improving:
- **Maintainability**: Each module is ~100-200 lines vs 480+ line method
- **Testability**: Individual components can be tested in isolation
- **Clarity**: Pipeline flow is explicit and documented
- **Type Safety**: Configuration passed via typed dataclasses

**Breaking Changes:** None - the public API (`fit_gaussian_splats()`) remains identical.

## References

**Related Modules:**
- `../models/gsplats/gsplat_model.py` - PyTorch model definition
- `../optim/per_splat_adam.py` - Per-splat optimizer
- `../dynamic_ops.py` - Adaptive topology operations
- `../candidates.py` - Seed generation

**Documentation:**
- `../README.md` - Main package documentation
- `../docs/dynamic_gsplats_management.md` - Dynamic operations details
