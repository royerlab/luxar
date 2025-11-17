# Per-Splat Optimization Specification

## Overview

The optim package implements specialized per-splat optimizers and schedulers for Gaussian splatting that maintain separate learning rates and momentum states for each individual splat. This enables:

1. Individual learning rate schedules per splat
2. Seamless addition/removal of splats without momentum loss
3. Better convergence for mixed-age splat populations
4. Efficient dynamic topology changes

**Prerequisite Reading**: For Gaussian splatting basics, see [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Overview and Section 2

**Related Specifications**:
- **Gradient Dilution Calculation**: [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) → Section 2
- **Model Interface**: [models/SPECIFICATIONS.md](../models/SPECIFICATIONS.md)
- **Fitting Pipeline Integration**: [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md) → Stage 3

## Design Philosophy

### Per-Splat State Management

Unlike standard PyTorch optimizers that maintain state per-parameter tensor, per-splat optimizers maintain state per-splat. This enables:

- **Momentum preservation**: When splats are added/removed, unchanged splats keep their momentum
- **Individual adaptation**: Each splat can have its own learning rate and optimization trajectory
- **Dynamic operations**: Topology changes don't disrupt optimization of existing splats

### Parameter-Type-Specific Learning Rates

Different parameter types have different optimization dynamics:

- **Position (μ)**: Slow updates prevent splat migration and proliferation
- **Variance (L_diag, L_off)**: Normal adaptation with gradient dilution compensation
- **Amplitude (a)**: Fast updates enable quick intensity matching
- **Sharpness (s')**: Moderate updates for conservative shape parameter adaptation

## Core Components

### 1. PerSplatAdam (`per_splat_adam.py`)

**Purpose**: Adam optimizer with per-splat state management and parameter-type-specific learning rates.

**Key Features**:
- Individual learning rates per splat
- Separate momentum buffers for each parameter type per splat
- Gradient dilution compensation for higher dimensions
- Efficient topology change handling
- Compatible with AMSGrad variant

#### Class Structure

```python
class PerSplatAdam:
    # Hard-coded parameter-type learning rate multipliers
    _POS_LR_MULTIPLIER = 0.1      # Position parameters (slow movement)
    _VAR_LR_MULTIPLIER = 1.0      # Variance parameters (normal adaptation)
    _AMP_LR_MULTIPLIER = 2.0      # Amplitude parameters (fast intensity matching)
    _SHARPNESS_LR_MULTIPLIER = 0.5  # Sharpness parameters (moderate adaptation)
```

#### Initialization

```python
def __init__(
    self,
    model: GaussianSplatModel,
    lr: float = 1e-3,              # Base learning rate (gradient-dilution-compensated)
    betas: Tuple[float, float] = (0.9, 0.999),  # Adam momentum decay rates
    eps: float = 1e-8,             # Numerical stability epsilon
    weight_decay: float = 0.0,     # L2 penalty coefficient
    amsgrad: bool = False          # Whether to use AMSGrad variant
):
    """
    Initialize per-splat Adam optimizer.

    Gradient dilution compensation is applied automatically:
    - Calculates effective_lr based on model dimensionality
    - Uses gradient dilution factor from utils.trils
    - Sharpness always uses base_lr (no dilution since it's 1 scalar)
    """
```

**Gradient Dilution Compensation**:
```python
def _calculate_effective_lr(self, d: int, base_lr: float) -> float:
    """
    Calculate effective learning rate with gradient dilution compensation.

    Parameters:
        d: Model dimensionality
        base_lr: Base learning rate

    Returns:
        Effective learning rate = base_lr * gradient_dilution_factor

    Gradient Dilution Factor Calculation:
    - 2D: params=5 → factor = 5/5 = 1.0× (baseline)
    - 3D: params=10 → factor = 10/5 = 2.0×
    - 4D: params=15 → factor = 4^0.8 × 15/5 ≈ 7.1×
    - d≤3: factor = params_current / params_2d
    - d>3: factor = d^0.8 × (params_current / params_2d)
    """
    from luxar.gsplats.utils.trils import calculate_gradient_dilution_factor
    gradient_dilution_factor = calculate_gradient_dilution_factor(d)
    return base_lr * gradient_dilution_factor
```

#### Per-Splat State Structure

Each splat maintains the following optimizer state:

```python
state = {
    "lr": float,                          # Gradient-dilution-compensated LR
    "base_lr": float,                     # Base LR without gradient dilution (for sharpness)
    "step": int,                          # Step counter for bias correction

    # Momentum for center position (μ)
    "exp_avg_mu": torch.Tensor,           # First moment estimate, shape (d,)
    "exp_avg_sq_mu": torch.Tensor,        # Second moment estimate, shape (d,)

    # Momentum for Cholesky diagonal (L_diag)
    "exp_avg_L_diag": torch.Tensor,       # First moment estimate, shape (d,)
    "exp_avg_sq_L_diag": torch.Tensor,    # Second moment estimate, shape (d,)

    # Momentum for Cholesky off-diagonal (L_off)
    "exp_avg_L_off": torch.Tensor,        # First moment estimate, shape (n_off_diag,)
    "exp_avg_sq_L_off": torch.Tensor,     # Second moment estimate, shape (n_off_diag,)

    # Momentum for amplitude (a)
    "exp_avg_a": torch.Tensor,            # First moment estimate, scalar
    "exp_avg_sq_a": torch.Tensor,         # Second moment estimate, scalar

    # Momentum for sharpness offset (s')
    "exp_avg_sharpness": torch.Tensor,    # First moment estimate, scalar
    "exp_avg_sq_sharpness": torch.Tensor, # Second moment estimate, scalar
}

# If AMSGrad=True, additionally:
state["max_exp_avg_sq_mu"] = torch.Tensor     # Maximum second moment for μ
state["max_exp_avg_sq_L_diag"] = torch.Tensor # Maximum second moment for L_diag
state["max_exp_avg_sq_L_off"] = torch.Tensor  # Maximum second moment for L_off
state["max_exp_avg_sq_a"] = torch.Tensor      # Maximum second moment for a
state["max_exp_avg_sq_sharpness"] = torch.Tensor  # Maximum second moment for s'
```

#### Optimization Step

```python
def step(self) -> bool:
    """
    Perform single optimization step for all splats.

    Algorithm:
    1. Check if any parameter has gradients (skip if none)
    2. Increment global step counter
    3. Detect topology changes (n_splats changed since last step)
    4. Initialize new splat states if topology changed (lazy initialization)
    5. Update each splat individually with _step_single_splat()

    Returns:
        bool: True if optimization performed, False if skipped (no gradients)

    Performance:
        - Only initializes states when topology changes (efficient)
        - Preserves momentum for unchanged splats during dynamic operations
    """
```

**Single Splat Update**:
```python
def _step_single_splat(self, splat_idx: int):
    """
    Perform Adam optimization step for a single Gaussian splat.

    Algorithm:
    1. Validate splat index and ensure state exists
    2. Extract gradients for all parameter types
    3. Update momentum buffers for each parameter type
    4. Apply bias correction
    5. Compute parameter updates using Adam formula
    6. Apply parameter-type-specific learning rate multipliers

    Parameter Update Formula (Standard Adam):
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

    Parameter-Type-Specific Learning Rates:
        - Position: α = effective_lr × 0.1
        - Variance: α = effective_lr × 1.0
        - Amplitude: α = effective_lr × 2.0
        - Sharpness: α = base_lr × 0.5 (no gradient dilution)
    """
```

**Core Update Function**:
```python
def _update_parameter(
    self,
    param: torch.Tensor,              # Parameter to update (modified in-place)
    grad: torch.Tensor,               # Gradient tensor
    exp_avg: torch.Tensor,            # First moment estimate buffer
    exp_avg_sq: torch.Tensor,         # Second moment estimate buffer
    max_exp_avg_sq: Optional[torch.Tensor],  # AMSGrad maximum (can be None)
    beta1: float,                     # First moment decay rate
    beta2: float,                     # Second moment decay rate
    lr: float,                        # Learning rate for this parameter type
    bias_correction1: float,          # First moment bias correction
    bias_correction2: float           # Second moment bias correction
):
    """
    Core Adam parameter update following PyTorch's implementation.

    Steps:
    1. Apply L2 regularization if weight_decay > 0
    2. Update first moment: exp_avg ← β₁*exp_avg + (1-β₁)*grad
    3. Update second moment: exp_avg_sq ← β₂*exp_avg_sq + (1-β₂)*grad²
    4. Compute denominator:
       - AMSGrad: use max(max_exp_avg_sq, exp_avg_sq)
       - Standard: use exp_avg_sq
    5. Apply update: param ← param - step_size * exp_avg / (√denom + ε)

    Where:
        step_size = lr / bias_correction1
        denom = √(exp_avg_sq) / √(bias_correction2) + ε
    """
```

#### Dynamic Operations Support

**Adding Splats**:
```python
def add_splats(self, n_new_splats: int, lr_new: Optional[float] = None):
    """
    Add optimizer state for newly added splats.

    Creates fresh state (zero momentum) for new splats.
    Preserves existing splat states unchanged.

    Args:
        n_new_splats: Number of new splats added to model
        lr_new: Learning rate for new splats (default: effective_lr)
    """
```

**Removing Splats**:
```python
def remove_splats(self, keep_mask: torch.Tensor):
    """
    Remove optimizer state for pruned splats.

    Reindexes remaining splat states to maintain contiguous indices.
    Preserves momentum for kept splats.

    Args:
        keep_mask: Boolean tensor indicating which splats to keep
    """
```

**Setting Individual Learning Rates**:
```python
def set_learning_rate(self, splat_idx: int, lr: float):
    """
    Set learning rate for a specific splat.

    Enables per-splat learning rate adaptation (used by schedulers).
    """

def get_learning_rate(self, splat_idx: int) -> float:
    """Get learning rate for a specific splat."""

def get_effective_learning_rates(self) -> torch.Tensor:
    """Get learning rates for all splats (for monitoring)."""
```

#### State Serialization

```python
def state_dict(self) -> Dict:
    """
    Get optimizer state for serialization.

    Returns all splat states, hyperparameters, and global step counter.
    """

def load_state_dict(self, state_dict: Dict):
    """
    Load optimizer state from serialization.

    Restores all splat states and hyperparameters.
    """
```

### 2. PerSplatReduceLROnPlateau (`per_splat_scheduler.py`)

**Purpose**: Per-splat learning rate scheduler that reduces LR when individual splats plateau.

**Key Features**:
- Tracks loss contribution per splat
- Reduces LR for individual splats independently
- Global fallback mechanism for overall convergence
- Configurable patience, factor, and minimum LR

#### Initialization

```python
def __init__(
    self,
    optimizer: PerSplatAdam,
    mode: str = "min",            # "min" for minimization (loss), "max" for maximization
    factor: float = 0.5,          # LR reduction factor (new_lr = lr * factor)
    patience: int = 10,           # Steps without improvement before LR reduction
    threshold: float = 1e-4,      # Minimum change to qualify as improvement
    cooldown: int = 0,            # Steps to wait after LR reduction
    min_lr: float = 1e-8,         # Minimum learning rate floor
    global_patience: int = 20     # Global patience for fallback mechanism
):
    """
    Initialize per-splat plateau scheduler.

    Maintains separate state for each splat:
    - best: Best metric seen for this splat
    - num_bad_epochs: Consecutive epochs without improvement
    - cooldown_counter: Remaining cooldown steps
    - lr_reductions: Total LR reductions for this splat
    """
```

#### Per-Splat State Structure

```python
splat_state = {
    "best": float,             # Best metric value seen (initialized to inf/-inf based on mode)
    "num_bad_epochs": int,     # Consecutive epochs without improvement
    "cooldown_counter": int,   # Remaining cooldown steps after LR reduction
    "lr_reductions": int       # Total number of LR reductions applied
}
```

#### Scheduler Step

```python
def step(self, metrics: Union[float, torch.Tensor, Dict[int, float]]):
    """
    Update learning rates based on metrics.

    Args:
        metrics: Can be:
            - float: Global loss (affects all splats equally)
            - torch.Tensor: Per-splat losses, shape (n_splats,)
            - Dict[int, float]: Explicit per-splat metrics {splat_idx: metric}

    Algorithm:
    1. Convert metrics to per-splat format
    2. Update global state (for fallback mechanism)
    3. Update each splat individually:
       a. Skip if in cooldown
       b. Check for improvement (metric better than best + threshold)
       c. If improved: update best, reset bad_epochs counter
       d. If not improved: increment bad_epochs counter
       e. If bad_epochs >= patience: reduce LR and enter cooldown
    4. Global fallback: If global_bad_epochs >= global_patience, reduce all LRs
    """
```

**Improvement Check**:
```python
def _is_better(self, current: float, best: float) -> bool:
    """
    Check if current metric is better than best.

    Returns:
        True if current is significantly better than best
        (considering threshold to avoid noise)

    Mode "min": current < best - threshold
    Mode "max": current > best + threshold
    """
```

**Individual Splat LR Update**:
```python
def _update_splat_lr(self, splat_idx: int, metric: float):
    """
    Update learning rate for individual splat based on its metric.

    Algorithm:
    1. Initialize state if splat is new
    2. Skip if in cooldown period
    3. Check for improvement
    4. If plateau detected (bad_epochs >= patience):
       a. Calculate new_lr = max(current_lr * factor, min_lr)
       b. Apply new learning rate to optimizer
       c. Reset bad_epochs counter
       d. Enter cooldown period
    """
```

**Global LR Reduction**:
```python
def _reduce_all_learning_rates(self):
    """
    Global learning rate reduction for all splats.

    Fallback mechanism when overall convergence stalls.
    Applied when global_bad_epochs >= global_patience.
    """
```

#### Dynamic Operations Support

```python
def add_splats(self, n_new_splats: int):
    """Add scheduler state for new splats (fresh state, no history)."""

def remove_splats(self, keep_mask: torch.Tensor):
    """Remove scheduler state for pruned splats (reindex remaining)."""
```

#### Monitoring

```python
def get_lr_reduction_counts(self) -> torch.Tensor:
    """
    Get number of LR reductions per splat (for monitoring).

    Returns:
        Tensor of shape (n_splats,) with reduction counts
    """
```

#### State Serialization

```python
def state_dict(self) -> Dict:
    """
    Get scheduler state for serialization.

    Returns:
        Dictionary containing:
            - splat_scheduler_states: Per-splat scheduler states
            - global_best: Best global metric seen
            - global_bad_epochs: Consecutive epochs without global improvement
            - global_cooldown_counter: Remaining global cooldown steps
            - last_epoch: Current epoch number
    """

def load_state_dict(self, state_dict: Dict):
    """
    Load scheduler state from serialization.

    Restores all scheduler states and global tracking variables.
    """
```

### 3. PerSplatExponentialLR (`per_splat_scheduler.py`)

**Purpose**: Per-splat exponential learning rate decay with optional age-based adaptation.

**Key Features**:
- Applies different decay rates to different splats based on age
- Newer splats decay slower (still learning)
- Older splats decay faster (fine-tuning)

#### Initialization

```python
def __init__(
    self,
    optimizer: PerSplatAdam,
    gamma: float = 0.95,           # Multiplicative LR decay factor
    age_based_decay: bool = True   # Enable age-based decay adaptation
):
    """
    Initialize per-splat exponential scheduler.

    Tracks splat ages (when they were added):
    - splat_ages: Dict[int, int] mapping splat_idx → birth_epoch
    - current_epoch: Current optimization epoch
    """
```

#### Scheduler Step

```python
def step(self):
    """
    Apply exponential decay to all splats.

    Algorithm:
    1. Increment current_epoch
    2. For each splat:
       a. Get current learning rate
       b. Calculate age = current_epoch - birth_epoch
       c. If age_based_decay enabled:
          - Calculate age_factor = 1.0 / (1.0 + age * 0.1)
          - Adjust gamma: gamma_adjusted = gamma + (1 - gamma) * age_factor
          - Newer splats get slower decay (gamma closer to 1)
       d. Apply decay: new_lr = current_lr * gamma_adjusted
       e. Set new learning rate in optimizer

    Age-Based Decay Rationale:
    - New splats need aggressive learning (slow decay)
    - Old splats need fine-tuning (fast decay)
    - Provides automatic adaptation without manual scheduling
    """
```

#### Dynamic Operations Support

```python
def add_splats(self, n_new_splats: int):
    """
    Add age tracking for new splats.

    New splats are born at current_epoch (fresh splats get slow decay).
    """

def remove_splats(self, keep_mask: torch.Tensor):
    """
    Remove age tracking for pruned splats (reindex remaining).
    """
```

**Note**: PerSplatExponentialLR does **not** provide `state_dict()` or `load_state_dict()` methods for serialization (unlike PerSplatReduceLROnPlateau). This is because the state is simple (just age tracking) and can be reconstructed easily.

### 4. ModelOptimizerCoordinator (`integration.py`)

**Purpose**: Coordinates model topology changes with optimizer and scheduler state management.

**Key Features**:
- Atomic updates to model, optimizer, and scheduler
- Ensures state consistency across all components
- Prevents momentum loss for unchanged splats
- Tracks operation count for monitoring

#### Initialization

```python
def __init__(
    self,
    model: GaussianSplatModel,
    optimizer: PerSplatAdam,
    scheduler: Optional[Union[PerSplatReduceLROnPlateau, PerSplatExponentialLR]] = None
):
    """
    Initialize coordinator.

    Maintains references to all components and operation counter.
    """
```

#### Dynamic Operations

**Pruning Splats**:
```python
def prune_splats(self, keep_mask: torch.Tensor) -> int:
    """
    Prune splats from model and sync all states.

    Algorithm:
    1. Apply keep_mask to model parameters
    2. Remove optimizer states for pruned splats
    3. Remove scheduler states for pruned splats (if present)
    4. Increment operation counter

    Returns:
        n_removed: Number of splats removed

    Atomicity:
        All components updated together - no partial states
    """
```

**Adding Splats**:
```python
def add_splats(
    self,
    centers_new: torch.Tensor,    # (n_new, d)
    Ls_new: torch.Tensor,          # (n_new, d, d)
    amps_new: torch.Tensor,        # (n_new,)
    lr_new: Optional[float] = None # Learning rate for new splats
) -> int:
    """
    Add new splats to model and sync all states.

    Algorithm:
    1. Append new splats to model parameters
    2. Add optimizer states for new splats (zero momentum)
    3. Add scheduler states for new splats (if present)
    4. Increment operation counter
    5. Assert: n_splats_after == n_splats_before + n_new

    Returns:
        n_new: Number of splats added
    """
```

**Replacing All Splats**:
```python
def replace_all_splats(
    self,
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    lr_reset: Optional[float] = None
) -> int:
    """
    Replace all splats (complete model reset).

    Algorithm:
    1. Replace model parameters completely
    2. Clear optimizer states entirely
    3. Initialize fresh optimizer states for all splats
    4. Clear scheduler states entirely (if present)
    5. Initialize fresh scheduler states for all splats (if present)
    6. Increment operation counter

    Returns:
        n_new: Number of splats after replacement

    Use Case:
        Operations where entire splat population changes structure.
    """
```

#### Monitoring

```python
def get_status(self) -> dict:
    """
    Get coordinator status for monitoring.

    Returns:
        status: Dict containing:
            - model_splats: Current number of splats in model
            - optimizer_states: Number of optimizer states
            - operation_count: Total dynamic operations performed
            - learning_rates: {mean, min, max} LR statistics
    """
```

### 5. Factory Function (`integration.py`)

**Purpose**: Convenient creation of coordinated per-splat optimizer setup.

```python
def create_per_splat_optimizer_setup(
    model: GaussianSplatModel,
    lr: float = 1e-3,
    scheduler_type: str = "plateau",  # "plateau", "exponential", or None

    # Optimizer args
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,

    # Plateau scheduler args
    patience: int = 10,
    factor: float = 0.5,
    threshold: float = 1e-4,
    cooldown: int = 0,
    min_lr: float = 1e-8,

    # Exponential scheduler args
    gamma: float = 0.95,
    age_based_decay: bool = True,

    **extra_kwargs
) -> Tuple[PerSplatAdam, Optional[Scheduler], ModelOptimizerCoordinator]:
    """
    Factory function to create coordinated per-splat optimizer setup.

    Returns:
        (optimizer, scheduler, coordinator) tuple

    Usage:
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            model,
            lr=0.01,
            scheduler_type="plateau",
            patience=50
        )
    """
```

## Mathematical Foundations

### Gradient Dilution Compensation

**Problem**: Higher dimensions have more parameters per splat, diluting gradients.

**Parameter Counts**:
- d dimensions → `d + d*(d+1)//2` parameters per splat
- 2D: 2 + 3 = 5 parameters (baseline)
- 3D: 3 + 6 = 10 parameters (2.0× dilution)
- 4D: 4 + 10 = 15 parameters (3.0× parameter dilution)

**Solution**: Scale learning rate based on parameter count and spatial complexity.

**Formula** (implemented in `utils/trils.py:calculate_gradient_dilution_factor`):

For d ≤ 3 (Conservative):
```
params_2d = 5
params_current = d + d*(d+1)//2
gradient_dilution_factor = params_current / params_2d
```

For d > 3 (Enhanced):
```
dimensional_complexity = d^0.8
parameter_complexity = params_current / params_2d
gradient_dilution_factor = dimensional_complexity × parameter_complexity
```

**Examples**:
- 2D: 5/5 = 1.0×
- 3D: 10/5 = 2.0×
- 4D: 4^0.8 × 15/5 = 3.03 × 3.0 ≈ 7.1×

**Application**: Automatically applied by `PerSplatAdam` during initialization.

**Sharpness Exception**: Sharpness parameters always use `base_lr` without gradient dilution since sharpness is a single scalar regardless of dimension.

### Parameter-Type-Specific Learning Rates

**Problem**: Different parameter types have different optimization dynamics.

**Solution**: Apply fixed multipliers to compensated learning rate.

**Multipliers** (hard-coded in `PerSplatAdam`):
- Position (μ): ×0.1 - Slow movement prevents splat migration and proliferation
- Variance (L_diag, L_off): ×1.0 - Normal covariance adaptation
- Amplitude (a): ×2.0 - Fast intensity matching
- Sharpness (s'): ×0.5 - Moderate shape parameter updates (uses base_lr, no dilution)

**Effective Learning Rates** (3D example with base_lr=0.01):
```
effective_lr = 0.01 × 2.0 = 0.02 (from gradient dilution)

lr_position = 0.02 × 0.1 = 0.002
lr_variance = 0.02 × 1.0 = 0.02
lr_amplitude = 0.02 × 2.0 = 0.04
lr_sharpness = 0.01 × 0.5 = 0.005 (base_lr, no dilution)
```

**Anti-Proliferation Rationale**:
- Splats migrating away from seeded locations triggers runaway seeding
- Slow position updates (×0.1) keep splats spatially stable
- Fast amplitude updates (×2.0) allow intensity adaptation without migration
- Result: Eliminates splat proliferation while maintaining convergence

### Adam Update Algorithm

**Standard Adam**:
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           # First moment (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          # Second moment (variance)
m̂_t = m_t / (1 - β₁^t)                        # Bias-corrected first moment
v̂_t = v_t / (1 - β₂^t)                        # Bias-corrected second moment
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)         # Parameter update
```

**AMSGrad Variant**:
```
v̂_t = max(v̂_{t-1}, v_t / (1 - β₂^t))         # Use maximum second moment
```

**Per-Splat Application**:
- Each splat has independent {m_t, v_t, v̂_max_t} for each parameter type
- Enables different optimization trajectories per splat
- Preserves momentum during dynamic operations

## Implementation Details

### Lazy State Initialization

**Strategy**: Only initialize splat states when topology changes.

**Benefits**:
- Avoids redundant initialization every step
- Efficient for static phases (no dynamic operations)
- Automatic detection via n_splats comparison

**Implementation**:
```python
def step(self):
    current_n_splats = self.model.n_splats()
    if current_n_splats != self._last_known_n_splats:
        self._initialize_all_splats()  # Only initialize new splats
        self._last_known_n_splats = current_n_splats
```

### State Reindexing

**Problem**: When splats are removed, indices need to be reindexed.

**Solution**: Map old indices to new indices using keep_mask.

**Implementation**:
```python
def remove_splats(self, keep_mask: torch.Tensor):
    new_states = {}
    keep_indices = torch.where(keep_mask)[0].cpu().numpy()

    for new_idx, old_idx in enumerate(keep_indices):
        if old_idx.item() in self.splat_states:
            new_states[new_idx] = self.splat_states[old_idx.item()]

    self.splat_states = new_states  # Atomic replacement
```

### Device Compatibility

**Supported Devices**:
- CPU: Universal fallback
- CUDA: Preferred for GPU acceleration
- MPS: Apple Silicon (experimental)

**Device Handling**:
- All tensors created on same device as model
- State tensors automatically placed on correct device
- Serialization preserves device information

### Thread Safety

**Status**: Not thread-safe

**Assumption**: Single-threaded usage per model instance

**Rationale**: Gaussian splatting optimization is inherently sequential

## Testing Requirements

### Unit Tests (Per Module)

**per_splat_adam.py**:
- Initialization with various configurations
- Single splat update correctness
- Batch update correctness
- Gradient dilution compensation accuracy
- Parameter-type-specific LR application
- Dynamic operations (add/remove splats)
- State serialization/deserialization
- AMSGrad variant correctness

**per_splat_scheduler.py**:
- Plateau detection and LR reduction
- Exponential decay application
- Age-based decay correctness
- Dynamic operations support
- Global fallback mechanism
- Monitoring functions

**integration.py**:
- Coordinator atomic operations
- State consistency across components
- Factory function correctness

### Integration Tests

**Complete Pipeline**:
- Optimization with dynamic operations
- Scheduler integration with coordinator
- State preservation during topology changes
- Multi-epoch training with LR scheduling

### Property Tests

**Mathematical Invariants**:
- Gradient dilution formula correctness
- Parameter-type LR multipliers applied correctly
- Adam update formula matches PyTorch implementation
- State indices remain contiguous after reindexing

## Performance Characteristics

### Computational Complexity

**Per Step**:
- State initialization: `O(N_new_splats × n_params)` (only when topology changes)
- Gradient extraction: `O(N_splats × n_params)`
- Momentum updates: `O(N_splats × n_params)`
- Parameter updates: `O(N_splats × n_params)`

**Total**: `O(N_splats × n_params)` per optimization step

### Memory Usage

**Per Splat**:
- Standard Adam: `2 × n_params` tensors (exp_avg, exp_avg_sq)
- AMSGrad: `3 × n_params` tensors (+ max_exp_avg_sq)
- Additional metadata: learning rate, step counter

**Total**: `O(N_splats × n_params × tensor_size)` memory

**Example** (3D with 10K splats):
- n_params per splat = 10 (3 pos + 6 cov + 1 amp + 1 sharpness = 11, but position/variance have d elements each)
- Actually: 3 + 3 + 3 + 1 + 1 = 11 scalar values
- With exp_avg and exp_avg_sq: 22 scalars per splat
- Total: 10K × 22 × 4 bytes ≈ 880 KB (negligible)

### Optimization Tips

1. **Use gradient dilution compensation**: Already applied automatically
2. **Enable AMSGrad for stability**: Set `amsgrad=True` if convergence issues
3. **Tune scheduler patience**: Higher patience for smoother convergence
4. **Monitor LR reductions**: Use `get_lr_reduction_counts()` for debugging

## Extension Points

### Adding New Schedulers

1. Implement scheduler class following same interface:
   ```python
   class PerSplatCustomScheduler:
       def __init__(self, optimizer: PerSplatAdam, **kwargs):
           ...

       def step(self, metrics):
           ...

       def add_splats(self, n_new_splats: int):
           ...

       def remove_splats(self, keep_mask: torch.Tensor):
           ...
   ```

2. Add to factory function in `integration.py`

### Adding New Optimizers

Follow same per-splat state management pattern as `PerSplatAdam`.

## See Also

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) - Core Gaussian splatting concepts
- [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) - Gradient dilution calculation
- [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md) - Pipeline integration
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and naming conventions

## Version History

- **v1.0.0** (January 2025): Initial implementation with per-splat Adam
  - Gradient dilution compensation
  - Parameter-type-specific learning rates
  - Per-splat plateau and exponential schedulers
  - Model-optimizer coordinator for dynamic operations
  - Comprehensive testing and documentation
