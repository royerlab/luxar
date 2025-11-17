# Gaussian Splat Fitting Pipeline Specification

## Overview

The fitting package implements a modular, type-safe pipeline for fitting n-dimensional Gaussian splats to image/volume data through gradient-based optimization. The pipeline is decomposed into six well-defined stages with clear data flow and separation of concerns.

**Prerequisite Reading**: For core Gaussian splatting concepts, see [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2: Gaussian Splat Model

**Related Specifications**:
- **Model & Rendering**: [models/SPECIFICATIONS.md](../models/SPECIFICATIONS.md)
- **Optimizers**: [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md)
- **Utilities**: [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md)
- **Dynamic Operations**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 4

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Type Safety**: All data passed between stages uses typed dataclasses
3. **Unidirectional Flow**: Data flows in one direction through the pipeline
4. **Fail Fast**: Input validation happens early at API boundaries
5. **Best State Guarantee**: Always returns the best state encountered, not the final state
6. **Testability**: Each module can be tested independently

## Pipeline Architecture

```
User Input → Validation → Preprocessing → Initialization → Loss Creation → Optimization → Finalization → Output
```

## Core Data Structures (`config.py`)

### FitConfig

**Purpose**: Immutable configuration object containing all validated parameters for the fitting process.

**Fields**:
```python
@dataclass
class FitConfig:
    """Configuration for Gaussian splat fitting."""

    # Input data
    V: np.ndarray                            # Input image/volume to fit (ndim >= 1)
    seeds: Optional[np.ndarray | float]      # Centers array (N, d) OR float proportion (e.g., 0.01 = 1%)

    # Normalization
    norm_percentile: float                   # Percentile for normalization (0 = full range, >0 = percentile clipping)

    # Model parameters
    init_sigma_vox: float                    # Initial sigma in voxels for isotropic covariances
    sigma_min_diag: Optional[Sequence[float]]  # Minimum diagonal values per dimension
    sigma_max_diag: Optional[Sequence[float]]  # Maximum diagonal values per dimension
    truncate: float                          # Gaussian truncation radius in standard deviations

    # Optimization parameters
    n_iters: int                             # Maximum number of optimization iterations
    lr: float                                # Base learning rate (before gradient dilution compensation)
    max_abs_error: Optional[float]           # Convergence threshold (None = auto: 1% of normalized range)
    gradient_clip: Optional[float]           # Gradient clipping value (None = no clipping)

    # Loss function
    loss_type: str                           # Loss function: "mse", "poisson", or "l1"
    asymmetric_penalty: Optional[float]      # Over-prediction penalty factor (default 10.0)
    l1_amp: Optional[float]                  # L1 regularization on amplitudes (default: 0.1 * lr)
    l1_diag: Optional[float]                 # L1 regularization on diagonal elements (default: 0.01 * lr)
    l1_sharpness: Optional[float]            # L1 regularization on sharpness offsets (default: 0.05 * lr)

    # Scheduler parameters
    scheduler_type: str                      # "plateau" or "exponential"
    patience: int                            # Iterations to wait before LR reduction
    factor: float                            # LR reduction factor

    # Dynamic operations
    enable_dynamic_ops: bool                 # Enable adaptive seeding/pruning
    dynamic_config: DynamicOpsConfig         # Configuration for dynamic operations
    dynamic_ops_verbose: bool                # Verbose logging for dynamic operations

    # Movie recording
    napari_movie: bool                       # Record optimization movie for napari
    movie_every: int                         # Record frame every N iterations
    movie_max_frames: Optional[int]          # Maximum frames to store (default 10000)

    # Device and logging
    device: torch.device                     # PyTorch device: cpu, cuda, or mps
    verbose: bool                            # Enable progress logging
```

**Validation Rules**:
- `V.ndim >= 1` (at least 1D data)
- `V.size > 0` (non-empty)
- `seeds` shape matches `(N, V.ndim)` if provided
- `norm_percentile >= 0.0`
- `init_sigma_vox > 0`, `sigma_min_vox > 0`, `sigma_max_vox > sigma_min_vox`
- `n_iters >= 0`, `lr > 0`
- `loss_type in ["mse", "poisson", "l1"]`
- `asymmetric_penalty >= 1.0`
- All L1 regularization values `>= 0` if specified
- `grad_clip > 0` if specified
- `plateau_patience >= 1`, `plateau_factor in (0, 1)`, `plateau_min_lr > 0`
- `exp_gamma in (0, 1]`
- `movie_every >= 1`

### PreprocessedData

**Purpose**: Contains normalized data and derived metadata after preprocessing stage.

**Fields**:
```python
@dataclass
class PreprocessedData:
    """Data that has been preprocessed and is ready for optimization."""

    # Preprocessed input data
    V_normalized: np.ndarray    # Input normalized to [0, 1], shape V.shape (NumPy array)
    V_tensor: torch.Tensor      # Same as V_normalized but as PyTorch tensor

    # Seed centers for initialization
    seed_centers: np.ndarray    # Initial splat centers, shape (N, d)

    # Normalization metadata
    image_min: float            # Original minimum value (for rescaling)
    image_max: float            # Original maximum value (for rescaling)
    intensity_range: float      # image_max - image_min

    # Dimensions
    d: int                      # Number of dimensions
    N: int                      # Number of candidate splats

    # Convergence threshold
    max_abs_error: float        # Max absolute error threshold for convergence
```

**Derivation Logic**:
- **Normalization**: `V_normalized = (V - image_min) / (image_max - image_min)` where min/max are computed using `norm_percentile`
- **Tensor Conversion**: `V_tensor = torch.tensor(V_normalized, device=device)`
- **Dimensions**: `d = V.ndim`, `N = len(seed_centers)`
- **Convergence Threshold**: If `max_abs_error` is None, set to `0.01` (1% of normalized [0,1] range)

**Note**: L1 regularization defaults are set in the preprocessing step by modifying `FitConfig` in-place (lines 60-72 in preprocessing.py), not stored in PreprocessedData.

### ModelComponents

**Purpose**: Contains all optimization components (model, optimizer, scheduler, coordinator).

**Fields**:
```python
@dataclass
class ModelComponents:
    model: GaussianSplatModel              # PyTorch model with splat parameters
    optimizer: PerSplatAdam                # Per-splat Adam optimizer
    scheduler: Union[PerSplatReduceLROnPlateau, PerSplatExponentialLR]  # Learning rate scheduler
    coordinator: ModelOptimizerCoordinator  # Coordinates dynamic operations
```

### OptimizationResults

**Purpose**: Contains raw results from optimization loop before finalization.

**Fields**:
```python
@dataclass
class OptimizationResults:
    """Results from the optimization process."""

    # Model parameters (from best state) - stored as tensors directly
    centers: torch.Tensor                  # Best centers, shape (N, d)
    Ls: torch.Tensor                       # Best Cholesky factors, shape (N, d, d)
    amps: torch.Tensor                     # Best amplitudes, shape (N,)
    sharpness: torch.Tensor                # Best sharpness values, shape (N,)

    # Optimization metadata
    converged_early: bool                  # True if convergence criterion met before n_iters
    actual_iters: int                      # Actual iterations run (may be < n_iters)
    best_iteration: int                    # Iteration where best state occurred
    best_loss: float                       # Reconstruction loss of best state
    best_max_abs_error: float              # Max absolute error of best state

    # Movie frames (if enabled)
    movie_frames: Optional[Dict[str, Any]]  # Movie data for visualization

    # Timing
    start_time: float                      # Optimization start time (seconds since epoch)
    end_time: float                        # Optimization end time (seconds since epoch)
```

**Note**: The actual implementation stores best parameters as tensors directly rather than as a state_dict, which is more efficient and allows direct access without reconstruction.

## Pipeline Stages

### Stage 1: Validation (`validation.py`)

**Purpose**: Validate all user inputs at API boundary before processing.

**Main Function**:
```python
def prepare_fit_config(
    fitter: "GaussianSplatFitter",
    V: np.ndarray,
    seeds: Optional[np.ndarray | float] = None,  # Can be array OR proportion
    norm_percentile: float = 0.0,
    init_sigma_vox: float = 1.5,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: Optional[float] = None,
    l1_diag: Optional[float] = None,
    l1_sharpness: Optional[float] = None,
    sigma_min_diag: Optional[Sequence[float]] = None,  # Per-dimension
    sigma_max_diag: Optional[Sequence[float]] = None,  # Per-dimension
    truncate: float = 3.0,
    verbose: bool = True,
    max_abs_error: Optional[float] = None,
    gradient_clip: Optional[float] = 1.0,  # Note: default is 1.0, not None
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    scheduler_type: str = "plateau",
    patience: int = 10,  # Generic parameter (not plateau_patience)
    factor: float = 0.5,  # Generic parameter (not plateau_factor)
    dynamic_ops_verbose: bool = False,
) -> FitConfig:
    """
    Validate all inputs and create immutable FitConfig.

    Validation Steps:
    1. Check V is non-empty numpy array with ndim >= 1
    2. Verify V contains finite values only
    3. Validate seeds shape matches (N, V.ndim) if provided
    4. Check all numeric parameters are in valid ranges
    5. Verify loss_type is supported
    6. Validate scheduler parameters
    7. Check device compatibility
    8. Create and return FitConfig

    Raises:
        ValueError: If any validation fails with descriptive message
        TypeError: If types are incorrect
    """
```

**Validation Logic**:
- **Array Validation**: Non-empty, finite values, correct dimensionality
- **Seed Validation**: If provided, shape `(N, d)` where `d = V.ndim`
- **Numeric Ranges**: All hyperparameters within valid ranges (lr > 0, etc.)
- **Type Validation**: loss_type, scheduler_type in allowed values
- **Consistency Checks**: sigma_max > sigma_min, etc.
- **Device Validation**: Check device availability, auto-detect if None

### Stage 2: Preprocessing (`preprocessing.py`)

**Purpose**: Prepare input data for optimization.

**Main Function**:
```python
def preprocess_data(config: FitConfig) -> PreprocessedData:
    """
    Normalize data and generate/validate seeds.

    Steps:
    1. Normalize V to [0, 1] using percentile-based or full-range normalization
    2. Generate seed centers if not provided (auto-candidate generation)
    3. Set L1 regularization defaults proportional to learning rate
    4. Set convergence threshold default (1% of normalized range)
    5. Create and return PreprocessedData
    """
```

**Normalization Algorithm**:
```python
if norm_percentile == 0.0:
    # Full range normalization
    V_min = V.min()
    V_max = V.max()
else:
    # Percentile-based normalization (robust to outliers)
    V_min = np.percentile(V, norm_percentile)
    V_max = np.percentile(V, 100 - norm_percentile)

# Normalize to [0, 1]
V_range = V_max - V_min
if V_range < 1e-10:  # Uniform image
    V_normalized = torch.zeros_like(V)
else:
    V_normalized = torch.tensor((V - V_min) / V_range, dtype=torch.float32, device=device)
    V_normalized = torch.clamp(V_normalized, 0.0, 1.0)
```

**Seed Generation**:
- **If seeds provided**: Validate shape and convert to numpy
- **If seeds is None**: Auto-generate using `find_candidates_overcomplete_nd()`:
  ```python
  seeds = find_candidates_overcomplete_nd(
      V,
      spacing=None,
      scales=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),  # Universal scale series
      peaks_per_scale=max(50, int(V.size * 0.002)),  # Volume-proportional density (~0.2%)
      percentile_thresh=70.0,  # Inclusive detection
      min_dist=2.0,
      add_intensity_grid=False,  # Grid sampling disabled by default
      grid_step=None,
      grid_percentile=60.0
  )
  ```

**L1 Regularization Defaults**:
```python
# Default: proportional to base learning rate for consistent sparsity pressure
l1_amp_final = l1_amp if l1_amp is not None else (0.1 * lr)  # 10% of base LR
l1_diag_final = l1_diag if l1_diag is not None else (0.01 * lr)  # 1% of base LR
l1_sharpness_final = l1_sharpness if l1_sharpness is not None else (0.05 * lr)  # 5% of base LR
```

**Rationale**: L1 regularization proportional to LR ensures consistent sparsity pressure across different learning rate choices and works correctly with gradient dilution compensation.

**Convergence Threshold Default**:
```python
if max_abs_error is None:
    convergence_threshold = 0.01  # 1% of normalized [0,1] range
else:
    convergence_threshold = max_abs_error
```

### Stage 3: Initialization (`initialization.py`)

**Purpose**: Create and initialize all optimization components.

**Main Function**:
```python
def initialize_optimization(
    config: FitConfig,
    preprocessed_data: PreprocessedData
) -> ModelComponents:
    """
    Initialize model, optimizer, scheduler, and coordinator.

    Steps:
    1. Create GaussianSplatModel with initial parameters
    2. Initialize PerSplatAdam optimizer with parameter-type-specific learning rates
    3. Create learning rate scheduler (plateau or exponential)
    4. Create ModelOptimizerCoordinator for dynamic operations
    5. Return ModelComponents
    """
```

**Model Initialization**:
```python
# Create model
model = GaussianSplatModel(
    centers=preprocessed_data.seed_centers,
    shape=config.V.shape,
    device=config.device,
    sigma_min=config.sigma_min_vox,
    sigma_max=config.sigma_max_vox,
    init_sigma=config.init_sigma_vox
)
```

**GaussianSplatModel Initialization Logic**:
1. **Centers**: Transform to logit space
   ```python
   # Normalize coordinates to [0, 1]
   u = centers / np.maximum(shape - 1, 1)
   u = np.clip(u, 1e-6, 1 - 1e-6)  # Avoid saturation

   # Transform to logit space
   raw_mu = torch.tensor(np.log(u) - np.log(1 - u), dtype=torch.float32, device=device)
   ```

2. **Diagonal Elements**: Inverse softplus of initial sigma
   ```python
   raw_L_diag = stable_inverse_softplus(
       torch.full((N, d), init_sigma, device=device)
   )
   ```

3. **Off-Diagonal Elements**: Initialize to zero (isotropic covariances)
   ```python
   L_off = torch.zeros((N, tril_size(d) - d), dtype=torch.float32, device=device)
   ```

4. **Amplitudes**: Sample from normalized image at center locations
   ```python
   # Sample intensities at seed centers
   intensities = sample_intensities_at_centers(V_normalized, centers)

   # Transform to raw space
   raw_a = stable_inverse_softplus(torch.tensor(intensities, dtype=torch.float32, device=device))
   ```

5. **Sharpness Offsets**: Initialize to zero (standard Gaussian, s=2)
   ```python
   sharpness_offsets_raw = torch.zeros(N, dtype=torch.float32, device=device)
   ```

**Optimizer Initialization**:
```python
optimizer = PerSplatAdam(
    model.parameters(),
    n_splats=preprocessed_data.n_seeds,
    lr=config.lr,  # Base learning rate (gradient dilution applied internally)
    betas=(0.9, 0.999),
    eps=1e-8,
    shape=config.V.shape  # For gradient dilution compensation
)
```

**PerSplatAdam Internal Behavior**:
- Applies gradient dilution compensation automatically based on dimensionality
- Uses parameter-type-specific learning rate multipliers:
  - Position (μ): `×0.1` (slow position updates, prevent proliferation)
  - Variance (L_diag, L_off): `×1.0` with gradient dilution compensation
  - Amplitude (a): `×2.0` (fast intensity convergence)
  - Sharpness (s'): `×0.5` (conservative, no gradient dilution since always 1 scalar)

**Scheduler Initialization**:
```python
if config.scheduler_type == "plateau":
    scheduler = PerSplatReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.plateau_factor,
        patience=config.plateau_patience,
        min_lr=config.plateau_min_lr,
        verbose=config.verbose
    )
else:  # "exponential"
    scheduler = PerSplatExponentialLR(
        optimizer,
        gamma=config.exp_gamma,
        verbose=config.verbose
    )
```

**Coordinator Initialization**:
```python
coordinator = ModelOptimizerCoordinator(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### Stage 4: Loss Function Creation (`losses.py`)

**Purpose**: Create loss function based on configuration.

**Main Function**:
```python
def create_loss_function(
    config: FitConfig,
    preprocessed_data: PreprocessedData,
    model: GaussianSplatModel
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create loss function closure that computes total loss.

    Returns:
        loss_fn: Callable that takes prediction tensor and returns total loss
    """
```

**Loss Function Structure**:
```python
def loss_fn(pred: torch.Tensor) -> torch.Tensor:
    """
    Compute loss between prediction and target.

    Note: Forward pass (model()) is done outside this function in the optimization loop.
    This allows better control over when gradients are computed.

    Parameters:
        pred: Model prediction from model.forward()

    Returns:
        total_loss: Sum of reconstruction loss and all regularization terms
    """
    # 1. Compute reconstruction loss (with asymmetric penalty)
    if loss_type == "mse":
        data = _compute_mse_loss(pred, V_t, asymmetric_penalty)
    elif loss_type == "poisson":
        data = _compute_poisson_loss(pred, V_t, asymmetric_penalty)
    elif loss_type == "l1":
        data = _compute_l1_loss(pred, V_t, asymmetric_penalty)

    # 2. Add L1 regularization on amplitudes (if enabled)
    if l1_amp is not None and l1_amp > 0:
        data = data + l1_amp * torch.mean(torch.abs(F.softplus(model.raw_a)))

    # 3. Add L1 regularization on diagonal elements (if enabled)
    if l1_diag is not None and l1_diag > 0:
        data = data + l1_diag * torch.mean(torch.abs(F.softplus(model.raw_L_diag)))

    # 4. Add L1 regularization on sharpness offsets (if enabled)
    if l1_sharpness is not None and l1_sharpness > 0:
        data = data + l1_sharpness * torch.mean(torch.abs(model.sharpness_offsets_raw))

    return data  # Returns loss only, not statistics
```

**Rationale for Architectural Change**:
- Forward pass (`model()`) is done in the optimization loop, not inside loss function
- Allows optimizer to control when gradients are computed
- Better separation of concerns: loss function only computes loss, not predictions
- Statistics tracking is done in the optimization loop if needed

**Reconstruction Loss Implementations**:

1. **MSE Loss**:
   ```python
   def _compute_mse_loss(pred, target, asymmetric_penalty):
       squared_errors = (pred - target) ** 2

       # Apply asymmetric penalty
       if asymmetric_penalty > 1.0:
           over_prediction_mask = pred > target
           penalties = torch.where(
               over_prediction_mask,
               torch.full_like(squared_errors, asymmetric_penalty),
               torch.ones_like(squared_errors)
           )
           squared_errors = squared_errors * penalties

       return torch.mean(squared_errors)
   ```

2. **Poisson Loss**:
   ```python
   def _compute_poisson_loss(pred, target, asymmetric_penalty):
       # Add epsilon to avoid log(0)
       eps = 1e-8
       pred_safe = pred + eps
       target_safe = target + eps

       # Poisson deviance: 2 * (pred - target + target * log(target/pred))
       poisson_deviance = 2.0 * (pred_safe - target_safe + target_safe * torch.log(target_safe / pred_safe))

       # Apply asymmetric penalty
       if asymmetric_penalty > 1.0:
           over_prediction_mask = pred > target
           penalties = torch.where(
               over_prediction_mask,
               torch.full_like(poisson_deviance, asymmetric_penalty),
               torch.ones_like(poisson_deviance)
           )
           poisson_deviance = poisson_deviance * penalties

       return torch.mean(poisson_deviance)
   ```

3. **L1 Loss**:
   ```python
   def _compute_l1_loss(pred, target, asymmetric_penalty):
       abs_errors = torch.abs(pred - target)

       # Apply asymmetric penalty
       if asymmetric_penalty > 1.0:
           over_prediction_mask = pred > target
           penalties = torch.where(
               over_prediction_mask,
               torch.full_like(abs_errors, asymmetric_penalty),
               torch.ones_like(abs_errors)
           )
           abs_errors = abs_errors * penalties

       return torch.mean(abs_errors)
   ```

**Asymmetric Penalty Rationale**:
- **Over-prediction** (pred > target): Heavily penalized (default 10×) because additive Gaussian models struggle to reduce intensity
- **Under-prediction** (pred < target): Normal penalty since adding more splats easily increases intensity
- **Default**: 10.0× penalty for optimal performance with additive models

### Stage 5: Optimization Loop (`optimization.py`)

**Purpose**: Run main training loop with convergence tracking and dynamic operations.

**Main Function**:
```python
def run_optimization_loop(
    components: ModelComponents,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],  # Takes pred, returns loss
    config: FitConfig,
    preprocessed_data: PreprocessedData
) -> OptimizationResults:
    """
    Run optimization loop with best state tracking.

    Steps:
    1. Initialize tracking variables (best state, history)
    2. Log convergence criteria
    3. For each iteration:
       a. Forward pass and loss computation
       b. Backward pass and gradient computation
       c. Gradient clipping (if enabled)
       d. Optimizer step
       e. Scheduler step
       f. Convergence check (max absolute error)
       g. Best state tracking
       h. Dynamic operations (if enabled and scheduled)
       i. Movie frame recording (if enabled)
       j. Progress logging
    4. Restore best state
    5. Return OptimizationResults
    """
```

**Loop Structure**:
```python
# Initialize tracking
best_max_abs_error = float('inf')
best_centers = None
best_Ls = None
best_amps = None
best_sharpness = None
best_iteration = 0
best_loss = float('inf')
converged_early = False

# Log convergence threshold
if verbose:
    aprint(f"Convergence threshold: max absolute error < {max_abs_error:.6f}")

# Main loop
for iteration in range(n_iters):
    # 1. Forward pass (model prediction)
    pred = components.model()

    # 2. Compute loss (pass prediction to loss function)
    loss = loss_fn(pred)

    # 2. Compute max absolute error
    with torch.no_grad():
        max_abs_error = torch.max(torch.abs(pred - V_normalized)).item()

    # 3. Backward pass
    components.optimizer.zero_grad()
    loss.backward()

    # 4. Gradient clipping
    if config.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(components.model.parameters(), config.gradient_clip)

    # 5. Optimizer step
    components.optimizer.step()

    # 6. Scheduler step
    components.scheduler.step(loss.detach())  # Detached to avoid graph retention

    # 7. Best state tracking (save tensors directly, not state_dict)
    if max_abs_error < best_max_abs_error:
        improvement_ratio = (best_max_abs_error - max_abs_error) / best_max_abs_error

        # Only log significant improvements (> 5% reduction)
        if improvement_ratio > 0.05 or iteration == 0:
            if verbose:
                aprint(f"New best: max_abs_error = {max_abs_error:.6f} (iteration {iteration})")

        best_max_abs_error = max_abs_error
        best_loss = loss.item()
        best_iteration = iteration

        # Save current parameters (deep copy of tensors directly)
        centers, Ls, amps, sharpness = components.model.current_params()
        best_centers = centers.detach().clone()
        best_Ls = Ls.detach().clone()
        best_amps = amps.detach().clone()
        best_sharpness = sharpness.detach().clone()

    # 8. Convergence check
    if max_abs_error < convergence_threshold:
        converged = True
        if verbose:
            aprint(f"Converged at iteration {iteration}: max_abs_error = {max_abs_error:.6f} < {convergence_threshold:.6f}")
        break

    # 9. Dynamic operations
    if config.enable_dynamic_ops and iteration % config.dynamic_ops_config.step_every == 0:
        apply_dynamic_operations(
            model=components.model,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            V_target=preprocessed_data.V_normalized,
            V_pred=pred,
            cfg=config.dynamic_ops_config,
            current_lr=config.lr,
            max_abs_error_threshold=preprocessed_data.convergence_threshold,
            device=config.device,
            verbose=config.verbose
        )

    # 10. Movie frame recording
    if config.napari_movie and iteration % config.movie_every == 0:
        record_movie_frame(...)

    # 11. Progress logging
    if config.verbose and iteration % 50 == 0:
        aprint(f"Iter {iteration}: loss={total_loss:.5f}, max_abs_err={max_abs_error:.6f}, n_splats={components.model.n_splats()}")

    # 12. Record history
    history.append({
        "iteration": iteration,
        **loss_stats,
        "max_abs_error": max_abs_error,
        "n_splats": components.model.n_splats(),
        "lr_mean": components.optimizer.get_mean_lr(),
        "lr_min": components.optimizer.get_min_lr(),
        "lr_max": components.optimizer.get_max_lr()
    })

# Create results (best tensors already saved, no need to restore state_dict)
return OptimizationResults(
    centers=best_centers,
    Ls=best_Ls,
    amps=best_amps,
    sharpness=best_sharpness,
    converged_early=converged_early,
    actual_iters=iteration + 1,
    best_iteration=best_iteration,
    best_loss=best_loss,
    best_max_abs_error=best_max_abs_error,
    movie_frames=movie_data if config.napari_movie else None,
    start_time=start_time,
    end_time=time.time()
)
```

**Best State Tracking Logic**:
- **Metric**: Maximum absolute error (not mean error) ensures all regions meet quality
- **Trigger**: Save state whenever current error is better than best seen so far
- **Smart Logging**: Only log improvements > 5% to reduce noise
- **Deep Copy**: Use `.detach().clone()` to avoid interfering with gradients
- **Restoration**: Always restore best state at end, not final state

**Convergence Criterion**:
```python
converged = max_abs_error < convergence_threshold
```

**Rationale**: Max absolute error ensures worst-case quality, not just average quality.

### Stage 6: Result Finalization (`results.py`)

**Purpose**: Extract parameters, rescale amplitudes, compile statistics.

**Main Function**:
```python
def finalize_results(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Finalize optimization results for return to user.

    Steps:
    1. Extract parameters from OptimizationResults (already contains best tensors)
    2. Convert tensors to numpy
    3. Pack parameters: [centers | packed_Cholesky | sharpness]
    4. Rescale amplitudes to original intensity range
    5. Compile comprehensive statistics
    6. Return (params, amplitudes, stats) tuple
    """
```

**Parameter Extraction**:
```python
# Extract parameters from optimization results (already contains best tensors)
centers_np = optimization_results.centers.cpu().numpy()      # Shape: (N, d)
Ls_np = optimization_results.Ls.cpu().numpy()                # Shape: (N, d, d)
amps_np = optimization_results.amps.cpu().numpy()            # Shape: (N,)
sharpness_np = optimization_results.sharpness.cpu().numpy()  # Shape: (N,)
```

**Parameter Packing**:
```python
# Pack Cholesky factors (lower triangular)
from luxar.gsplats.utils import pack_tril

N, d = centers_np.shape
packed_L = pack_tril(Ls_np)  # Shape: (N, d*(d+1)//2)

# Combine: [centers | packed_L | sharpness]
params = np.column_stack([
    centers_np,       # (N, d)
    packed_L,         # (N, d*(d+1)//2)
    sharpness_np      # (N, 1)
])
# Final shape: (N, d + d*(d+1)//2 + 1)
```

**Amplitude Rescaling**:
```python
# Rescale amplitudes to original intensity range
image_min = preprocessed_data.image_min
image_max = preprocessed_data.image_max
intensity_range = preprocessed_data.intensity_range

if intensity_range < 1e-10:  # Uniform image
    amplitudes_rescaled = amps_np  # No rescaling needed
else:
    amplitudes_rescaled = amps_np * intensity_range
```

**Rationale**: Optimization happens in normalized [0,1] space, but users expect amplitudes in original intensity units.

**Statistics Compilation**:
```python
stats = {
    # Optimization results
    "converged_early": optimization_results.converged_early,
    "best_iteration": optimization_results.best_iteration,
    "actual_iters": optimization_results.actual_iters,
    "best_max_abs_error": optimization_results.best_max_abs_error,
    "best_loss": optimization_results.best_loss,

    # Final state
    "n_splats": len(amps_np),
    "n_seeds": preprocessed_data.N,  # Number of initial candidates

    # Sharpness statistics
    "sharpness_mean": float(np.mean(sharpness_np)),
    "sharpness_std": float(np.std(sharpness_np)),
    "sharpness_min": float(np.min(sharpness_np)),
    "sharpness_max": float(np.max(sharpness_np)),
    "sharpness_median": float(np.median(sharpness_np)),

    # Configuration
    "loss_type": config.loss_type,
    "asymmetric_penalty": config.asymmetric_penalty,
    "l1_amp": config.l1_amp,
    "l1_diag": config.l1_diag,
    "l1_sharpness": config.l1_sharpness,
    "convergence_threshold": preprocessed_data.max_abs_error,

    # Movie frames
    "movie_frames": optimization_results.movie_frames,

    # Timing
    "optimization_time": optimization_results.end_time - optimization_results.start_time,

    # Device
    "device": str(config.device)
}
```

**Return Value**:
```python
return params, amplitudes_rescaled, stats
```

## Visualization Module (`visualization.py`)

### display_compression_analysis()

**Purpose**: Display compression statistics comparing original data size to splat representation.

**Function Signature**:
```python
def display_compression_analysis(
    V_shape: Tuple[int, ...],
    n_splats: int,
    V_dtype: np.dtype = np.float32,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate and display compression statistics.

    Returns:
        stats: Dictionary with compression metrics
    """
```

**Calculation**:
```python
# Original data size
original_bytes = np.prod(V_shape) * np.dtype(V_dtype).itemsize

# Splat representation size
d = len(V_shape)
params_per_splat = d + d*(d+1)//2 + 1  # centers + Cholesky + sharpness
splat_bytes = n_splats * (params_per_splat + 1) * 4  # +1 for amplitude, 4 bytes per float32

# Compression ratio
compression_ratio = original_bytes / splat_bytes

stats = {
    "original_bytes": original_bytes,
    "splat_bytes": splat_bytes,
    "compression_ratio": compression_ratio,
    "savings_percent": (1 - 1/compression_ratio) * 100
}

if verbose:
    aprint(f"Original: {original_bytes / 1e6:.2f} MB")
    aprint(f"Splats: {splat_bytes / 1e6:.2f} MB")
    aprint(f"Compression: {compression_ratio:.2f}×")
    aprint(f"Savings: {stats['savings_percent']:.1f}%")

return stats
```

### show_optimization_movie()

**Purpose**: Display napari movie of optimization progress (target, reconstruction, residual).

**Function Signature**:
```python
def show_optimization_movie(
    movie_frames: Dict[str, List],
    original_shape: Tuple[int, ...],
    title: str = "Optimization Progress"
) -> None:
    """
    Display napari viewer with optimization movie.

    Requires:
        - napari installed
        - movie_frames from optimization results

    Layers:
        - Target (constant)
        - Reconstruction (changing)
        - Residual (abs difference, changing)

    Time Slider:
        - Scrub through optimization iterations
    """
```

**Implementation**:
```python
import napari

# Extract frames
target_frames = np.array(movie_frames["target"])
recon_frames = np.array(movie_frames["reconstruction"])
residual_frames = np.array(movie_frames["residual"])
iterations = movie_frames["iterations"]

# Create viewer
viewer = napari.Viewer(title=title)

# Add layers
viewer.add_image(target_frames, name="Target", colormap="gray")
viewer.add_image(recon_frames, name="Reconstruction", colormap="gray")
viewer.add_image(residual_frames, name="Residual", colormap="inferno")

# Set time points
viewer.dims.set_point(0, iterations)

napari.run()
```

## Mathematical Foundations

### Gradient Dilution Compensation

**Problem**: Higher dimensions have more parameters per splat, diluting gradients across parameters.

**Parameter Counts**:
- 2D: 5 parameters (2 pos + 3 cov)
- 3D: 10 parameters (3 pos + 6 cov)
- 4D: 15 parameters (4 pos + 10 cov)
- nD: `d + d*(d+1)//2` parameters

**Solution**: Scale learning rate based on parameter count relative to 2D baseline.

**Formula (d ≤ 3)**:
```python
params_2d = 2 + 2*3//2 = 5
params_current = d + d*(d+1)//2
gradient_dilution_factor = params_current / params_2d
effective_lr = base_lr * gradient_dilution_factor
```

**Formula (d > 3)**: Enhanced scaling for higher dimensions
```python
dimensional_complexity = d ** 0.8
parameter_complexity = params_current / params_2d
gradient_dilution_factor = dimensional_complexity * parameter_complexity
effective_lr = base_lr * gradient_dilution_factor
```

**Examples**:
- 2D: 5/5 = 1.0× (baseline)
- 3D: 10/5 = 2.0×
- 4D: 4^0.8 × 15/5 = 3.03 × 3.0 ≈ 7.1×

**Important**: Gradient dilution compensation is applied internally by the optimizer (`PerSplatAdam`), not by the fitting pipeline.

**Sharpness Exception**: Sharpness parameters always use `base_lr` without gradient dilution since sharpness is always a single scalar regardless of dimension.

### Parameter-Type-Specific Learning Rates

**Problem**: Different parameter types have different optimization dynamics and convergence rates.

**Solution**: Apply fixed multipliers to base learning rate for each parameter type.

**Multipliers** (applied internally by `PerSplatAdam`):
- **Position (μ)**: `×0.1` (hardcoded) - Slow position updates prevent splat migration and proliferation
- **Variance (L_diag, L_off)**: `×1.0` (hardcoded, with gradient dilution) - Normal covariance adaptation
- **Amplitude (a)**: `×2.0` (hardcoded) - Fast intensity convergence
- **Sharpness (s')**: `×0.5` (hardcoded, no gradient dilution) - Conservative shape parameter updates

**Effective Learning Rates**:
```python
# For 3D with base_lr = 0.01:
lr_position = 0.01 × 0.1 = 0.001  # Slow position updates
lr_variance = 0.01 × 1.0 × 2.0 = 0.02  # Normal adaptation with gradient dilution
lr_amplitude = 0.01 × 2.0 = 0.02  # Fast intensity convergence
lr_sharpness = 0.01 × 0.5 = 0.005  # Conservative shape updates
```

**Anti-Proliferation Rationale**:
- **Root cause**: Splats migrating away from seeded locations triggers runaway seeding cycles
- **Solution**: Slow position updates (×0.1) keep splats spatially stable
- **Complementary**: Fast amplitude updates (×2.0) allow intensity adaptation without migration
- **Result**: Eliminates splat proliferation while maintaining convergence quality

### Proportional L1 Regularization

**Problem**: L1 regularization strength needs to scale with optimization strength for consistent sparsity pressure.

**Solution**: Set L1 regularization proportional to base learning rate.

**Default Values**:
```python
l1_amp = 0.1 * lr      # 10% of base LR (5% of amplitude LR due to 2.0× multiplier)
l1_diag = 0.01 * lr    # 1% of base LR
l1_sharpness = 0.05 * lr  # 5% of base LR (10% of sharpness LR due to 0.5× multiplier)
```

**Benefits**:
1. **Consistent sparsity**: L1 pressure scales with optimization strength
2. **Dimensional invariance**: Works correctly with gradient dilution compensation (higher LR → higher L1)
3. **Auto-tuning**: No manual L1 adjustment needed when changing learning rates
4. **Sharpness sparsity**: Encourages standard Gaussians (s=2) unless sharper edges improve fit

## Implementation Notes

### Device Compatibility

**Supported Devices**:
- **CPU**: Always available, universal fallback
- **CUDA**: Preferred for GPU acceleration (10-100× speedup)
- **MPS**: Apple Silicon GPU (experimental, may be slower than CPU for some operations)

**Auto-Detection**:
```python
if device is None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
```

### Memory Management

**Memory Usage**:
- Model parameters: `O(N_splats × (d + d*(d+1)//2 + 1))` where d = dimensionality
- Gradients: Same as parameters
- Intermediate tensors: `O(V.size)` for rendered images
- Optimizer state: `O(N_splats × n_params × 2)` for exp_avg and exp_avg_sq
- Movie frames: `O(n_frames × V.size × 4)` bytes (can be large!)

**Memory Optimization**:
- Use `movie_max_frames` to limit movie memory
- Apply gradient clipping to prevent memory spikes
- Use smaller batch sizes for dynamic operations

### Numerical Stability

**Stability Measures**:
1. **Sigmoid saturation prevention**: Clamp normalized coordinates to `[1e-6, 1-1e-6]`
2. **Softplus inversion**: Use `stable_inverse_softplus()` for initialization
3. **Epsilon in Poisson loss**: Add `1e-8` to avoid `log(0)`
4. **Division by zero**: Check for uniform images (`V_range < 1e-10`)
5. **Gradient clipping**: Optional `grad_clip` parameter for stability

## Testing Requirements

### Unit Tests (Per Module)

**validation.py**:
- Valid inputs create correct FitConfig
- Invalid inputs raise appropriate errors
- Edge cases (uniform images, single pixel)
- Type validation

**preprocessing.py**:
- Normalization correctness (full range and percentile)
- Seed generation (auto and user-provided)
- L1 regularization defaults
- Convergence threshold defaults

**initialization.py**:
- Model initialization with correct parameter shapes
- Optimizer setup with correct learning rates
- Scheduler creation (plateau and exponential)
- Coordinator initialization

**losses.py**:
- MSE, Poisson, L1 loss computations
- Asymmetric penalty application
- L1 regularization terms
- Gradient flow verification

**optimization.py**:
- Training loop executes without errors
- Best state tracking works correctly
- Convergence detection triggers properly
- Dynamic operations integration

**results.py**:
- Parameter extraction and packing
- Amplitude rescaling correctness
- Statistics compilation
- Movie frame storage

**visualization.py**:
- Compression analysis calculations
- Napari movie display (mocked, no windows)

### Integration Tests

**Complete Pipeline**:
- End-to-end fitting with known ground truth
- Convergence to expected quality
- Dynamic operations improve results
- Movie recording works correctly

### Property Tests

**Mathematical Invariants**:
- Normalized data in [0, 1]
- Best state has lowest max absolute error
- Gradient dilution formula correctness
- Amplitude rescaling preserves relative magnitudes
- Parameter packing/unpacking is lossless

## Performance Characteristics

### Computational Complexity

**Per Iteration**:
- Forward pass: `O(N_splats × V.size)` (rendering)
- Loss computation: `O(V.size)`
- Backward pass: `O(N_splats × V.size)` (backpropagation)
- Optimizer step: `O(N_splats × n_params)`

**Total**:
- `O(n_iters × N_splats × V.size)` for full optimization

### Typical Performance

**2D Images** (512×512):
- 1000 splats: ~0.5s per iteration (GPU)
- 10000 splats: ~5s per iteration (GPU)

**3D Volumes** (128³):
- 1000 splats: ~2s per iteration (GPU)
- 10000 splats: ~20s per iteration (GPU)

**Speedups**:
- GPU vs CPU: 10-100× depending on volume size
- Gradient dilution: Enables higher dimensions without excessive iterations

## Extension Points

### Adding New Loss Functions

1. Implement loss function in `losses.py`:
   ```python
   def _compute_custom_loss(pred, target, asymmetric_penalty):
       # Implementation
       return loss
   ```

2. Add to `create_loss_function()`:
   ```python
   elif loss_type.lower() == "custom":
       data = _compute_custom_loss(pred, V_t, asymmetric_penalty)
   ```

3. Update validation in `validation.py`:
   ```python
   if loss_type not in ["mse", "poisson", "l1", "custom"]:
       raise ValueError(...)
   ```

### Adding New Preprocessing Steps

Add functions to `preprocessing.py` and call from `preprocess_data()`.

### Modifying Optimization Loop

Key extension points in `optimization.py`:
- **Before loop**: Add initialization
- **In loop**: Add custom operations, metrics, or callbacks
- **After loop**: Add post-processing

## References

**Related Modules**:
- `../models/gsplats/gsplat_model.py` - PyTorch model definition
- `../optim/per_splat_adam.py` - Per-splat optimizer
- `../dynamic_ops.py` - Adaptive topology operations
- `../candidates.py` - Seed generation

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) - Main Gaussian splatting specification
- [multiscale/SPECIFICATIONS.md](../multiscale/SPECIFICATIONS.md) - Multi-scale decomposition
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and naming conventions

## Version History

- **v1.0.0** (January 2025): Initial modular pipeline with 6-stage architecture
  - Refactored from monolithic 480+ line method
  - Added type-safe dataclasses for configuration
  - Implemented best state tracking (quality guarantee)
  - Added comprehensive testing (84 tests, 100% module coverage)
  - Proportional L1 regularization defaults
  - Parameter-type-specific learning rates
  - Gradient dilution compensation
  - Sharpness parameter support
