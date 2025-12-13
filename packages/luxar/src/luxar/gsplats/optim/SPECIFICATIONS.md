# Per-Splat Optimization Specification

**Version**: 1.0.0
**Last Updated**: 2025-11-27

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
    
    Internal state tracking:
    - splat_states: Dict[int, Dict] storing per-splat optimizer state
    - global_step: int tracking total optimization steps
    - _last_known_n_splats: int for detecting topology changes
    
    Initialization behavior:
    - Calls _initialize_all_splats() immediately to create state for current splats
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
    - 3D: params=9 → factor = 9/5 = 1.8×
    - 4D: params=14 → factor = 4^0.8 × 14/5 ≈ 8.5×
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

    # Momentum for center position (μ) - stored in raw_mu space
    "exp_avg_mu": torch.Tensor,           # First moment estimate, shape (d,)
    "exp_avg_sq_mu": torch.Tensor,        # Second moment estimate, shape (d,)

    # Momentum for Cholesky diagonal (L_diag) - stored in raw_L_diag space
    "exp_avg_L_diag": torch.Tensor,       # First moment estimate, shape (d,)
    "exp_avg_sq_L_diag": torch.Tensor,    # Second moment estimate, shape (d,)

    # Momentum for Cholesky off-diagonal (L_off)
    "exp_avg_L_off": torch.Tensor,        # First moment estimate, shape (n_off_diag,)
    "exp_avg_sq_L_off": torch.Tensor,     # Second moment estimate, shape (n_off_diag,)

    # Momentum for amplitude (a) - stored in raw_a space
    "exp_avg_a": torch.Tensor,            # First moment estimate, scalar
    "exp_avg_sq_a": torch.Tensor,         # Second moment estimate, scalar

    # Momentum for sharpness offset (s') - stored in raw space
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

**Important Note**: All momentum buffers are stored in the **raw parameter space** (e.g., raw_mu, raw_L_diag, raw_a, sharpness_offsets_raw), not in the transformed space. This is critical for proper gradient application.

#### Optimization Step

```python
def step(self) -> bool:
    """
    Perform single optimization step for all splats.

    Algorithm:
    1. Check if any parameter has gradients (return False if none)
    2. Increment global_step counter
    3. Validate current_n_splats is non-negative
    4. Detect topology changes (current_n_splats != _last_known_n_splats)
    5. Initialize new splat states if topology changed (lazy initialization)
    6. Update _last_known_n_splats to current value
    7. Update each splat individually with _step_single_splat()

    Returns:
        bool: True if optimization performed, False if skipped (no gradients)

    Performance:
        - Only initializes states when topology changes (efficient)
        - Preserves momentum for unchanged splats during dynamic operations
    
    Error Handling:
        - Validates n_splats >= 0
        - Wraps all exceptions in RuntimeError with context
    """
```

**Single Splat Update**:
```python
def _step_single_splat(self, splat_idx: int):
    """
    Perform Adam optimization step for a single Gaussian splat.

    Algorithm:
    1. Validate splat index (0 <= splat_idx < n_splats)
    2. Lazy initialize state if splat_idx not in splat_states
    3. Increment state["step"] counter
    4. Validate hyperparameters (lr > 0, base_lr > 0, 0 <= betas < 1)
    5. Calculate bias corrections: bias_correction1 = 1 - beta1^step
                                   bias_correction2 = 1 - beta2^step
    6. Extract gradients for all parameter types:
       - grad_mu from raw_mu.grad[splat_idx]
       - grad_L_diag from raw_L_diag.grad[splat_idx]
       - grad_L_off from L_off.grad[splat_idx]
       - grad_a from raw_a.grad[splat_idx]
       - grad_sharpness from sharpness_offsets_raw.grad[splat_idx]
    7. Update each parameter type with _update_parameter():
       - Position: lr × 0.1 (POS_LR_MULTIPLIER)
       - L_diag: lr × 1.0 (VAR_LR_MULTIPLIER)
       - L_off: lr × 1.0 (VAR_LR_MULTIPLIER)
       - Amplitude: lr × 2.0 (AMP_LR_MULTIPLIER)
       - Sharpness: base_lr × 0.5 (SHARPNESS_LR_MULTIPLIER, no gradient dilution)

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
        
    Gradient Extraction:
        - All gradients extracted with bounds checking (splat_idx < grad.shape[0])
        - Only updates parameters if gradient is not None
        - Handles None gradients gracefully (skips update)
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
    1. Apply L2 regularization if weight_decay > 0:
       grad = grad.add(param, alpha=weight_decay)
    2. Update first moment: exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
    3. Update second moment: exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
    4. Compute denominator:
       - AMSGrad: torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                  denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)) + eps
       - Standard: denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)) + eps
    5. Compute step_size = lr / bias_correction1
    6. Apply update: param.addcdiv_(exp_avg, denom, value=-step_size)

    Formula Details:
        step_size = lr / bias_correction1
        denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + ε
        param ← param - step_size * exp_avg / denom
        
    In-Place Operations:
        - All tensor updates use in-place operations (.mul_, .add_, .addcdiv_)
        - Modifies param, exp_avg, exp_avg_sq, and max_exp_avg_sq directly
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
        lr_new: Learning rate for new splats (default: effective_lr if None)

    Validation:
        - n_new_splats must be >= 0 (returns immediately if 0)
        - lr_new must be > 0 if provided
        
    Implementation:
        - Calculates current_n = len(self.splat_states)
        - For i in range(n_new_splats): _initialize_splat(current_n + i, lr=lr_new)
        - Uses effective_lr if lr_new is None
        
    Error Handling:
        - Raises ValueError if n_new_splats < 0
        - Raises ValueError if lr_new <= 0
        - Wraps exceptions in RuntimeError with context
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
        keep_mask: Boolean tensor (1D) indicating which splats to keep

    Validation:
        - Must be torch.Tensor
        - Must have dtype torch.bool
        - Must be 1D tensor
        - Length must match len(self.splat_states)
        
    Algorithm:
        1. Create empty new_states dict
        2. Extract keep_indices = torch.where(keep_mask)[0].cpu().numpy()
        3. For each (new_idx, old_idx) in enumerate(keep_indices):
           - If old_idx in splat_states: new_states[new_idx] = splat_states[old_idx]
        4. Atomic replacement: self.splat_states = new_states
        
    Error Handling:
        - Raises TypeError if keep_mask is not torch.Tensor or not bool dtype
        - Raises ValueError if keep_mask is not 1D or wrong length
        - Wraps exceptions in RuntimeError with context
    """
```

**Setting Individual Learning Rates**:
```python
def set_learning_rate(self, splat_idx: int, lr: float):
    """
    Set learning rate for a specific splat.

    Enables per-splat learning rate adaptation (used by schedulers).
    
    Validation:
        - lr must be > 0
        - splat_idx must be in range [0, n_splats)
        
    Behavior:
        - If splat_idx in splat_states: updates state["lr"] = lr
        - If splat_idx not in splat_states: calls _initialize_splat(splat_idx, lr=lr)
    """

def get_learning_rate(self, splat_idx: int) -> float:
    """
    Get learning rate for a specific splat.
    
    Returns:
        - splat_states[splat_idx]["lr"] if splat_idx in splat_states
        - base_lr otherwise
    """

def get_effective_learning_rates(self) -> torch.Tensor:
    """
    Get learning rates for all splats (for monitoring).
    
    Returns:
        torch.Tensor of shape (n_splats,) with current learning rates
        
    Implementation:
        - Creates zeros tensor of size n_splats
        - For each splat_idx: lrs[splat_idx] = get_learning_rate(splat_idx)
    """
```

#### State Serialization

```python
def state_dict(self) -> Dict:
    """
    Get optimizer state for serialization.

    Returns dict containing:
        - splat_states: Dict[int, Dict] with all per-splat state
        - global_step: int
        - base_lr: float
        - betas: Tuple[float, float]
        - eps: float
        - weight_decay: float
        - amsgrad: bool
        
    Note: effective_lr is NOT saved (recalculated from base_lr on load)
    """

def load_state_dict(self, state_dict: Dict):
    """
    Load optimizer state from serialization.

    Restores all fields from state_dict directly.
    Does NOT recalculate effective_lr (assumes it's reconstructed properly).
    Does NOT update _last_known_n_splats (will be updated on next step()).
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

    Internal State:
        - splat_scheduler_states: Dict[int, Dict] with per-splat state
        - global_best: Optional[float] (None initially)
        - global_bad_epochs: int (starts at 0)
        - global_cooldown_counter: int (starts at 0)
        - last_epoch: int (starts at 0)

    No automatic initialization of splat states (lazy initialization in step()).
    """
```

#### Per-Splat State Structure

```python
splat_state = {
    "best": float,             # Best metric value seen (inf for min, -inf for max)
    "num_bad_epochs": int,     # Consecutive epochs without improvement (starts at 0)
    "cooldown_counter": int,   # Remaining cooldown steps after LR reduction (starts at 0)
    "lr_reductions": int       # Total number of LR reductions applied (starts at 0)
}
```

#### Scheduler Step

```python
def step(self, metrics: Union[float, torch.Tensor, Dict[int, float]]):
    """
    Update learning rates based on metrics.

    Args:
        metrics: Can be:
            - float/int: Global loss (affects all splats equally)
            - torch.Tensor: Per-splat losses
              - Scalar tensor (shape []): treated as global metric
              - 1D tensor (shape [n_splats]): per-splat metrics
              - Other shapes: raises ValueError
            - Dict[int, float]: Explicit per-splat metrics {splat_idx: metric}

    Algorithm:
    1. Validate metrics is not None
    2. Increment last_epoch counter
    3. Convert metrics to per-splat dict (splat_metrics):
       - float/int: create dict with same value for all splats [0, n_splats)
       - torch.Tensor (scalar): same as float
       - torch.Tensor (1D): create dict from tensor values
       - dict: validate keys are non-negative ints, values are finite
    4. Calculate global_metric:
       - float/int or scalar tensor: use directly
       - 1D tensor: use torch.mean(metrics)
       - dict: use sum(values) / len(values)
    5. Call _update_global_state(global_metric)
    6. For each (splat_idx, metric) in splat_metrics:
       - Call _update_splat_lr(splat_idx, metric)

    Validation:
        - Raises ValueError if metrics is None
        - Raises ValueError if metric values are not finite
        - Raises ValueError if tensor is not scalar or 1D
        - Raises ValueError if dict keys are not non-negative ints
        - Wraps exceptions in RuntimeError
    """
```

**Global State Update**:
```python
def _update_global_state(self, metric: float):
    """
    Update global scheduler state.
    
    Algorithm:
    1. If global_best is None: set global_best = metric and return
    2. If global_cooldown_counter > 0: decrement and return
    3. Check if metric is better than global_best (using _is_better):
       - If better: update global_best, reset global_bad_epochs = 0
       - If not better: increment global_bad_epochs
    4. If global_bad_epochs >= global_patience:
       - Call _reduce_all_learning_rates()
       - Reset global_bad_epochs = 0
       - Set global_cooldown_counter = cooldown
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
    1. If splat_idx not in splat_scheduler_states: call _init_splat_state(splat_idx)
    2. Get state = splat_scheduler_states[splat_idx]
    3. If state["cooldown_counter"] > 0: decrement and return
    4. Check if metric is better than state["best"] (using _is_better):
       - If better: update state["best"] = metric, state["num_bad_epochs"] = 0
       - If not better: increment state["num_bad_epochs"]
    5. If state["num_bad_epochs"] >= patience:
       a. Get current_lr from optimizer.get_learning_rate(splat_idx)
       b. Calculate new_lr = max(current_lr * factor, min_lr)
       c. If new_lr < current_lr:
          - optimizer.set_learning_rate(splat_idx, new_lr)
          - state["lr_reductions"] += 1
          - state["num_bad_epochs"] = 0
          - state["cooldown_counter"] = cooldown
    """
```

**Splat State Initialization**:
```python
def _init_splat_state(self, splat_idx: int):
    """
    Initialize scheduler state for a splat.
    
    Sets best to:
        - float("inf") if mode == "min"
        - float("-inf") if mode == "max"
        
    Sets all counters to 0:
        - num_bad_epochs = 0
        - cooldown_counter = 0
        - lr_reductions = 0
    """
```

**Global LR Reduction**:
```python
def _reduce_all_learning_rates(self):
    """
    Global learning rate reduction for all splats.

    Fallback mechanism when overall convergence stalls.
    Applied when global_bad_epochs >= global_patience.
    
    Algorithm:
        For splat_idx in range(optimizer.model.n_splats()):
            current_lr = optimizer.get_learning_rate(splat_idx)
            new_lr = max(current_lr * factor, min_lr)
            if new_lr < current_lr:
                optimizer.set_learning_rate(splat_idx, new_lr)
                
    Note: Does NOT update per-splat lr_reductions counters
    """
```

#### Dynamic Operations Support

```python
def add_splats(self, n_new_splats: int):
    """
    Add scheduler state for new splats (fresh state, no history).
    
    Validation:
        - n_new_splats must be >= 0 (returns immediately if 0)
        
    Algorithm:
        - current_n = len(self.splat_scheduler_states)
        - For i in range(n_new_splats): _init_splat_state(current_n + i)
        
    Error Handling:
        - Raises ValueError if n_new_splats < 0
        - Wraps exceptions in RuntimeError
    """

def remove_splats(self, keep_mask: torch.Tensor):
    """
    Remove scheduler state for pruned splats (reindex remaining).
    
    Validation:
        - Must be torch.Tensor with dtype torch.bool
        - Must be 1D tensor
        - Length must match len(self.splat_scheduler_states)
        
    Algorithm:
        - Same reindexing logic as PerSplatAdam.remove_splats()
        - Creates new_states dict with reindexed entries
        - Atomic replacement: self.splat_scheduler_states = new_states
        
    Error Handling:
        - Raises TypeError if keep_mask is not torch.Tensor or not bool
        - Raises ValueError if keep_mask is not 1D or wrong length
        - Wraps exceptions in RuntimeError
    """
```

#### Monitoring

```python
def get_lr_reduction_counts(self) -> torch.Tensor:
    """
    Get number of LR reductions per splat (for monitoring).

    Returns:
        Tensor of shape (n_splats,) with reduction counts
        
    Implementation:
        - Creates zeros tensor of size n_splats
        - For each splat_idx: if in states, set counts[splat_idx] = state["lr_reductions"]
        - Returns 0 for splats without state
    """
```

#### State Serialization

```python
def state_dict(self) -> Dict:
    """
    Get scheduler state for serialization.

    Returns:
        Dictionary containing:
            - splat_scheduler_states: Dict[int, Dict]
            - global_best: Optional[float]
            - global_bad_epochs: int
            - global_cooldown_counter: int
            - last_epoch: int
    """

def load_state_dict(self, state_dict: Dict):
    """
    Load scheduler state from serialization.

    Restores all scheduler states and global tracking variables.
    Direct assignment from state_dict keys.
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

    Internal State:
        - splat_ages: Dict[int, int] mapping splat_idx → birth_epoch
        - current_epoch: int (starts at 0)
        
    Initialization:
        - Initializes ages for all current splats to 0
        - For i in range(optimizer.model.n_splats()): splat_ages[i] = 0
    """
```

#### Scheduler Step

```python
def step(self):
    """
    Apply exponential decay to all splats.

    Algorithm:
    1. Increment current_epoch
    2. For splat_idx in range(optimizer.model.n_splats()):
       a. Get current_lr = optimizer.get_learning_rate(splat_idx)
       b. If age_based_decay enabled:
          - age = current_epoch - splat_ages.get(splat_idx, 0)
          - age_factor = 1.0 / (1.0 + age * 0.1)
          - gamma_adjusted = gamma + (1 - gamma) * age_factor
       c. Else: gamma_adjusted = gamma
       d. Calculate new_lr = current_lr * gamma_adjusted
       e. Set optimizer.set_learning_rate(splat_idx, new_lr)

    Age-Based Decay Rationale:
    - New splats (age=0): age_factor=1.0, gamma_adjusted → 1.0 (slow decay)
    - Old splats (age→∞): age_factor→0.0, gamma_adjusted → gamma (fast decay)
    - Provides automatic adaptation without manual scheduling
    
    Age Factor Formula:
        age_factor = 1.0 / (1.0 + age * 0.1)
        gamma_adjusted = gamma + (1 - gamma) * age_factor
        
    Examples (gamma=0.95):
        - age=0: gamma_adjusted = 0.95 + 0.05*1.0 = 1.00 (no decay)
        - age=10: gamma_adjusted = 0.95 + 0.05*0.5 = 0.975 (slow decay)
        - age=100: gamma_adjusted = 0.95 + 0.05*0.09 = 0.9545 (near-normal decay)
    """
```

#### Dynamic Operations Support

```python
def add_splats(self, n_new_splats: int):
    """
    Add age tracking for new splats.

    New splats are born at current_epoch (fresh splats get slow decay).
    
    Algorithm:
        - current_n = len(self.splat_ages)
        - For i in range(n_new_splats):
            - splat_ages[current_n + i] = current_epoch
    """

def remove_splats(self, keep_mask: torch.Tensor):
    """
    Remove age tracking for pruned splats (reindex remaining).
    
    Algorithm:
        - Same reindexing logic as other components
        - Creates new_ages dict with reindexed entries
        - Atomic replacement: self.splat_ages = new_ages
    """
```

**Note**: PerSplatExponentialLR does **not** provide `state_dict()` or `load_state_dict()` methods for serialization (unlike PerSplatReduceLROnPlateau). This is a design decision - the state is simple (just age tracking) and can be reconstructed easily if needed.

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

    Internal State:
        - model: reference to GaussianSplatModel
        - optimizer: reference to PerSplatAdam
        - scheduler: optional reference to scheduler
        - operation_count: int (starts at 0)
    """
```

#### Dynamic Operations

**Pruning Splats**:
```python
def prune_splats(self, keep_mask: torch.Tensor) -> int:
    """
    Prune splats from model and sync all states.

    Algorithm:
    1. Get n_before = model.n_splats()
    2. Call model.prune_(keep_mask)
    3. Call optimizer.remove_splats(keep_mask)
    4. If scheduler is not None and has remove_splats method:
       - Call scheduler.remove_splats(keep_mask)
    5. Get n_after = model.n_splats()
    6. Calculate n_removed = n_before - n_after
    7. Increment operation_count
    8. Return n_removed

    Atomicity:
        All components updated together - no partial states
        
    Scheduler Check:
        Uses hasattr(scheduler, "remove_splats") to check for method existence
    """
```

**Adding Splats**:
```python
def add_splats(
    self,
    centers_new: torch.Tensor,    # (n_new, d)
    Ls_new: torch.Tensor,          # (n_new, d, d)
    amps_new: torch.Tensor,        # (n_new,)
    sharpness_new: torch.Tensor,   # (n_new,)
    lr_new: Optional[float] = None # Learning rate for new splats
) -> int:
    """
    Add new splats to model and sync all states.

    Algorithm:
    1. Get n_before = model.n_splats()
    2. Calculate n_new = centers_new.shape[0]
    3. Call model.append_(centers_new, Ls_new, amps_new)
    4. Call optimizer.add_splats(n_new, lr_new=lr_new)
    5. If scheduler is not None and has add_splats method:
       - Call scheduler.add_splats(n_new)
    6. Get n_after = model.n_splats()
    7. Assert n_after == n_before + n_new (raises AssertionError if not)
    8. Increment operation_count
    9. Return n_new

    Scheduler Check:
        Uses hasattr(scheduler, "add_splats") to check for method existence
    """
```

**Replacing All Splats**:
```python
def replace_all_splats(
    self,
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    sharpness: torch.Tensor,
    lr_reset: Optional[float] = None
) -> int:
    """
    Replace all splats (complete model reset).

    Algorithm:
    1. Calculate n_new = centers.shape[0]
    2. Call model.replace_with(centers, Ls, amps)
    3. Clear optimizer state:
       - optimizer.splat_states = {}
       - optimizer.add_splats(n_new, lr_new=lr_reset)
    4. Clear scheduler state if present:
       - If hasattr(scheduler, "splat_scheduler_states"):
           scheduler.splat_scheduler_states = {}
       - If hasattr(scheduler, "splat_ages"):
           scheduler.splat_ages = {}
       - If hasattr(scheduler, "add_splats"):
           scheduler.add_splats(n_new)
    5. Increment operation_count
    6. Return n_new

    Use Case:
        Operations where entire splat population changes structure
        (e.g., complete reseeding, dimension changes)
    """
```

#### Monitoring

```python
def get_status(self) -> dict:
    """
    Get coordinator status for monitoring.

    Returns:
        dict containing:
            - model_splats: int (current number of splats)
            - optimizer_states: int (number of optimizer states)
            - operation_count: int (total dynamic operations)
            - learning_rates: dict with:
                - mean: float
                - min: float
                - max: float
                
    Implementation:
        - Gets effective_lrs = optimizer.get_effective_learning_rates()
        - Calculates mean, min, max from effective_lrs tensor
        - Converts all values to float for JSON serialization
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
) -> Tuple[PerSplatAdam, Optional[Union[PerSplatReduceLROnPlateau, PerSplatExponentialLR]], ModelOptimizerCoordinator]:
    """
    Factory function to create coordinated per-splat optimizer setup.

    Algorithm:
    1. Create optimizer_kwargs with only optimizer-specific args:
       - betas, eps, weight_decay, amsgrad
    2. Create optimizer = PerSplatAdam(model, lr=lr, **optimizer_kwargs)
    3. If scheduler_type == "plateau":
       - Create scheduler_kwargs with plateau-specific args
       - Create scheduler = PerSplatReduceLROnPlateau(optimizer, **scheduler_kwargs)
    4. Elif scheduler_type == "exponential":
       - Create scheduler_kwargs with exponential-specific args
       - Create scheduler = PerSplatExponentialLR(optimizer, **scheduler_kwargs)
    5. Elif scheduler_type is None:
       - scheduler = None
    6. Else: raise ValueError
    7. Create coordinator = ModelOptimizerCoordinator(model, optimizer, scheduler)
    8. Return (optimizer, scheduler, coordinator)

    Returns:
        tuple: (optimizer, scheduler, coordinator)

    Scheduler Options:
        - "plateau": PerSplatReduceLROnPlateau with patience-based reduction
        - "exponential": PerSplatExponentialLR with age-based decay
        - None: No scheduler

    Note: extra_kwargs is accepted but not used (for API flexibility)

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
- 3D: 3 + 6 = 9 parameters (1.8× dilution)
- 4D: 4 + 10 = 14 parameters (2.8× parameter dilution)

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
- 3D: 9/5 = 1.8×
- 4D: 4^0.8 × 14/5 = 3.03 × 2.8 ≈ 8.5×

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

**Note**: `_initialize_all_splats()` only initializes splats that don't already have state, preserving existing momentum.

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

**Key Details**:
- Uses torch.where(keep_mask)[0] to extract kept indices
- Converts to CPU numpy for iteration
- Uses enumerate to assign new contiguous indices
- Atomic dict replacement ensures consistency

### Device Compatibility

**Supported Devices**:
- CPU: Universal fallback
- CUDA: Preferred for GPU acceleration
- MPS: Apple Silicon (experimental)

**Device Handling**:
- All tensors created on same device as model
- Device determined by: `device = next(model.parameters()).device`
- State tensors automatically placed on correct device during initialization
- Serialization preserves device information in tensor data

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
- Edge cases: zero gradients, single splat, device placement

**per_splat_scheduler.py**:
- Plateau detection and LR reduction
- Exponential decay application
- Age-based decay correctness
- Dynamic operations support
- Global fallback mechanism
- Monitoring functions
- Metrics conversion (float, tensor, dict)
- Edge cases: scalar tensors, empty metrics

**integration.py**:
- Coordinator atomic operations
- State consistency across components
- Factory function correctness
- Prune/add/replace operations
- Status monitoring
- Assertion checks

### Integration Tests

**Complete Pipeline**:
- Optimization with dynamic operations
- Scheduler integration with coordinator
- State preservation during topology changes
- Multi-epoch training with LR scheduling
- End-to-end workflows

### Property Tests

**Mathematical Invariants**:
- Gradient dilution formula correctness
- Parameter-type LR multipliers applied correctly
- Adam update formula matches PyTorch implementation
- State indices remain contiguous after reindexing
- Learning rates always positive and >= min_lr

## Performance Characteristics

### Computational Complexity

**Per Step**:
- State initialization: `O(N_new_splats × n_params)` (only when topology changes)
- Gradient extraction: `O(N_splats × n_params)`
- Momentum updates: `O(N_splats × n_params)`
- Parameter updates: `O(N_splats × n_params)`

**Total**: `O(N_splats × n_params)` per optimization step

**Where**:
- N_splats = number of splats
- n_params = total scalar parameters per splat
  - 2D: 2 (mu) + 2 (L_diag) + 1 (L_off) + 1 (a) + 1 (s') = 7 scalars
  - 3D: 3 (mu) + 3 (L_diag) + 3 (L_off) + 1 (a) + 1 (s') = 11 scalars
  - General formula: d + d + d*(d-1)/2 + 2 = 2d + d*(d-1)/2 + 2

### Memory Usage

**Per Splat**:
- Standard Adam: `2 × n_params` tensors (exp_avg, exp_avg_sq)
- AMSGrad: `3 × n_params` tensors (+ max_exp_avg_sq)
- Additional metadata: learning rate (float), base_lr (float), step counter (int)

**Total**: `O(N_splats × n_params × tensor_size)` memory

**Example** (3D with 10K splats):
- Parameters per splat: 3 (mu) + 3 (L_diag) + 3 (L_off) + 1 (a) + 1 (s') = 11 scalars
- With exp_avg and exp_avg_sq: 22 scalars per splat
- Total: 10K × 22 × 4 bytes (float32) ≈ 880 KB (negligible)

### Optimization Tips

1. **Use gradient dilution compensation**: Already applied automatically
2. **Enable AMSGrad for stability**: Set `amsgrad=True` if convergence issues
3. **Tune scheduler patience**: Higher patience for smoother convergence
4. **Monitor LR reductions**: Use `get_lr_reduction_counts()` for debugging
5. **Batch dynamic operations**: Group add/remove operations when possible

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

2. Add to factory function in `integration.py`:
   ```python
   elif scheduler_type == "custom":
       scheduler = PerSplatCustomScheduler(optimizer, **custom_kwargs)
   ```

### Adding New Optimizers

Follow same per-splat state management pattern as `PerSplatAdam`:
- Dict[int, Dict] for per-splat state storage
- Lazy initialization on topology changes
- Reindexing support for dynamic operations
- Parameter-type-specific handling
- Serialization support

## See Also

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) - Core Gaussian splatting concepts
- [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) - Gradient dilution calculation
- [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md) - Pipeline integration
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and naming conventions

## Changelog

- **v1.0.0** (January 2025): Initial implementation with per-splat Adam
  - Gradient dilution compensation
  - Parameter-type-specific learning rates
  - Per-splat plateau and exponential schedulers
  - Model-optimizer coordinator for dynamic operations
  - Comprehensive testing and documentation
