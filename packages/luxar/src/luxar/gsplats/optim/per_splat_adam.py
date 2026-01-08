"""
Per-Splat Adam Optimizer for Gaussian Splatting

A specialized Adam optimizer that maintains separate learning rates and momentum
for each Gaussian splat, enabling:
1. Individual learning rate schedules per splat
2. Seamless addition/removal of splats without momentum loss
3. Better convergence for mixed-age splat populations
4. Efficient dynamic operations
"""

from typing import Any, Dict, Optional, Tuple, cast

import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


class PerSplatAdam:
    """
    Per-Splat Adam Optimizer for Gaussian Splatting.

    This specialized optimizer maintains separate Adam optimizer state (momentum buffers,
    learning rates, step counts) for each individual Gaussian splat, rather than per-parameter
    tensor like standard PyTorch optimizers. This enables advanced dynamic operations without
    losing optimization momentum.

    Key Features:
    - Individual learning rates per splat (enables splat-specific adaptation)
    - Parameter-type-specific learning rate multipliers (prevents splat proliferation)
    - Per-splat momentum preservation during topology changes
    - Efficient handling of splat addition/removal
    - Zero momentum disruption for unchanged splats
    - Compatible with AMSGrad variant

    Parameter-Type Learning Rate Multipliers:
    - Position parameters (μ): ×0.1 (hard-coded, slow movement, prevents migration)
    - Variance parameters (L_diag, L_off): ×1.0 (hard-coded, normal adaptation)
    - Amplitude parameters (a): ×2.0 (hard-coded, fast intensity convergence)
    - Sharpness parameters (s'): ×0.5 (hard-coded, moderate shape adaptation)

    Performance Optimizations:
    - Lazy initialization: only creates state when topology changes
    - Batch processing: updates all splats in single step() call
    - Memory efficient: state tensors sized per splat dimension

    Thread Safety:
    - Not thread-safe: assumes single-threaded usage per model instance

    Example:
        >>> model = GaussianSplatModel(...)
        >>> optimizer = PerSplatAdam(model, lr=0.01)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = compute_loss(model())
        ...     loss.backward()  # type: ignore[no-untyped-call]
        ...     optimizer.step()
    """

    # Parameter-type learning rate multipliers (hard-coded for splat proliferation prevention)
    _POS_LR_MULTIPLIER = 0.1  # Position parameters (slow movement)
    _VAR_LR_MULTIPLIER = 1.0  # Variance parameters (normal adaptation)
    _AMP_LR_MULTIPLIER = 2.0  # Amplitude parameters (fast intensity matching)
    _SHARPNESS_LR_MULTIPLIER = 0.5  # Sharpness parameters (moderate shape adaptation)

    def __init__(
        self,
        model: GaussianSplatModel,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        """
        Initialize per-splat Adam optimizer.

        Args:
            model: GaussianSplatModel to optimize
            lr: Base learning rate (automatically compensated for gradient dilution in higher dimensions)
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: L2 penalty coefficient
            amsgrad: Whether to use AMSGrad variant
        """
        self.model = model
        self.base_lr = lr

        # Calculate gradient dilution compensation based on dimensionality
        # This compensates for the fact that higher dimensions have more parameters
        # per splat, which dilutes gradients across more values
        d = len(model.shape)
        self.effective_lr = self._calculate_effective_lr(d, lr)

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # Per-splat state storage
        # Each splat has its own: lr, step, exp_avg_*, max_exp_avg_sq_*
        self.splat_states: Dict[int, Dict[str, Any]] = {}
        self.global_step = 0

        # Track model state to detect topology changes
        self._last_known_n_splats = 0

        # Initialize state for current splats
        self._initialize_all_splats()

    def _calculate_effective_lr(self, d: int, base_lr: float) -> float:
        """
        Calculate effective learning rate with gradient dilution compensation.

        Gradient dilution occurs because higher dimensions have more parameters per splat,
        spreading gradients thinner. We compensate by scaling the learning rate.

        This compensation applies to position/variance/amplitude parameters.
        Sharpness is NOT affected because it's always a single scalar regardless of dimension.

        Args:
            d: Dimensionality of the model
            base_lr: Base learning rate

        Returns:
            Effective learning rate with gradient dilution compensation
        """
        from luxar.gsplats.utils.trils import calculate_gradient_dilution_factor

        gradient_dilution_factor = calculate_gradient_dilution_factor(d)
        return base_lr * gradient_dilution_factor

    def _initialize_all_splats(self) -> None:
        """
        Initialize optimizer state for all current splats in the model.

        This method is called automatically when the number of splats changes
        (detected by comparing with _last_known_n_splats). It only initializes
        state for splats that don't already have state, making it efficient
        for incremental splat additions.

        Performance Note:
            Only called when topology changes, not every step.
        """
        n_splats = self.model.n_splats()
        # Only initialize splats that don't already have optimizer state
        # This preserves existing momentum for unchanged splats during topology changes
        for splat_idx in range(n_splats):
            if splat_idx not in self.splat_states:
                self._initialize_splat(
                    splat_idx
                )  # Create fresh state with zero momentum

    def _initialize_splat(self, splat_idx: int, lr: Optional[float] = None) -> None:
        """Initialize optimizer state for a single splat."""
        # Validate splat index
        if splat_idx < 0:
            raise ValueError(f"Splat index must be non-negative, got {splat_idx}")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            raise RuntimeError("Model has no parameters to optimize")

        if lr is None:
            lr = self.effective_lr  # Use effective_lr for new splats by default

        # Validate learning rate
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        try:
            # Validate model dimensions
            if not hasattr(self.model, "shape") or len(self.model.shape) == 0:
                raise ValueError("Model shape is invalid or empty")

            if not hasattr(self.model, "L_off") or self.model.L_off.shape[-1] <= 0:
                raise ValueError("Model L_off parameter is invalid")

            # Per-splat state: separate momentum for each parameter type
            state: Dict[str, Any] = {
                "lr": lr,  # Gradient-dilution-compensated LR (used for position/variance/amplitude)
                "base_lr": self.base_lr,  # Base LR without gradient dilution (used for sharpness)
                "step": 0,
                # Momentum for mu (center): shape (d,)
                "exp_avg_mu": torch.zeros(len(self.model.shape), device=device),
                "exp_avg_sq_mu": torch.zeros(len(self.model.shape), device=device),
                # Momentum for L_diag: shape (d,)
                "exp_avg_L_diag": torch.zeros(len(self.model.shape), device=device),
                "exp_avg_sq_L_diag": torch.zeros(len(self.model.shape), device=device),
                # Momentum for L_off: shape depends on dimensionality
                "exp_avg_L_off": torch.zeros(self.model.L_off.shape[-1], device=device),
                "exp_avg_sq_L_off": torch.zeros(
                    self.model.L_off.shape[-1], device=device
                ),
                # Momentum for amplitude: shape ()
                "exp_avg_a": torch.tensor(0.0, device=device),
                "exp_avg_sq_a": torch.tensor(0.0, device=device),
                # Momentum for sharpness: shape ()
                "exp_avg_sharpness": torch.tensor(0.0, device=device),
                "exp_avg_sq_sharpness": torch.tensor(0.0, device=device),
            }

            # AMSGrad variant: track maximum of squared gradients
            if self.amsgrad:
                state.update(
                    {
                        "max_exp_avg_sq_mu": torch.zeros_like(state["exp_avg_sq_mu"]),
                        "max_exp_avg_sq_L_diag": torch.zeros_like(
                            state["exp_avg_sq_L_diag"]
                        ),
                        "max_exp_avg_sq_L_off": torch.zeros_like(
                            state["exp_avg_sq_L_off"]
                        ),
                        "max_exp_avg_sq_a": torch.zeros_like(state["exp_avg_sq_a"]),
                        "max_exp_avg_sq_sharpness": torch.zeros_like(
                            state["exp_avg_sq_sharpness"]
                        ),
                    }
                )

            self.splat_states[splat_idx] = state

        except Exception as e:
            raise RuntimeError(f"Failed to initialize splat {splat_idx}: {e}") from e

    def zero_grad(self) -> None:
        """Clear gradients of all model parameters."""
        self.model.zero_grad()

    def step(self) -> bool:
        """
        Perform single optimization step for all splats.

        Efficiently handles topology changes by only initializing new states
        when the number of splats has changed since the last step.

        Returns:
            bool: True if optimization step was performed, False if skipped
        """
        try:
            # Check if any parameter has gradients
            has_grad = any(p.grad is not None for p in self.model.parameters())
            if not has_grad:
                return False  # No gradients to optimize

            self.global_step += 1

            # Only initialize if topology changed (more efficient)
            current_n_splats = self.model.n_splats()
            if current_n_splats < 0:
                raise ValueError(f"Invalid number of splats: {current_n_splats}")

            if current_n_splats != self._last_known_n_splats:
                self._initialize_all_splats()
                self._last_known_n_splats = current_n_splats

            # Update each splat individually
            for splat_idx in range(current_n_splats):
                self._step_single_splat(splat_idx)

            return True

        except Exception as e:
            raise RuntimeError(f"Error during optimization step: {e}") from e

    def _step_single_splat(self, splat_idx: int):
        """
        Perform Adam optimization step for a single Gaussian splat.

        Updates all parameters associated with the splat (center, covariance,
        amplitude) using individual learning rate and momentum state. Follows
        standard Adam algorithm with bias correction.

        Args:
            splat_idx: Index of the splat to update (0-based)

        Raises:
            IndexError: If splat_idx is out of valid range
            ValueError: If learning rate or beta values are invalid

        Algorithm:
            1. Validate splat index and parameters
            2. Extract gradients for all splat parameters
            3. Update momentum buffers (exp_avg, exp_avg_sq)
            4. Apply bias correction
            5. Compute parameter updates using Adam formula
            6. Apply updates to model parameters in-place
        """
        # Validate splat index to prevent out-of-bounds access
        n_splats = self.model.n_splats()
        if splat_idx < 0 or splat_idx >= n_splats:
            raise IndexError(f"Splat index {splat_idx} out of range [0, {n_splats})")

        # Initialize state if this splat is new (lazy initialization)
        if splat_idx not in self.splat_states:
            self._initialize_splat(splat_idx)

        # Get per-splat optimization state and increment step counter
        state = self.splat_states[splat_idx]
        state["step"] += 1  # Used for bias correction in Adam

        # Unpack hyperparameters for this splat
        beta1, beta2 = self.betas  # Momentum decay rates
        lr = cast(float, state["lr"])
        base_lr = cast(float, state["base_lr"])

        # Validate hyperparameters to catch configuration errors early
        if lr <= 0:
            raise ValueError(f"Invalid learning rate {lr} for splat {splat_idx}")
        if base_lr <= 0:
            raise ValueError(
                f"Invalid base learning rate {base_lr} for splat {splat_idx}"
            )
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError(f"Invalid beta values: beta1={beta1}, beta2={beta2}")

        # Compute bias correction terms for Adam algorithm
        # These compensate for initialization bias in momentum estimates
        bias_correction1 = 1 - beta1 ** state["step"]  # First moment bias correction
        bias_correction2 = 1 - beta2 ** state["step"]  # Second moment bias correction

        # Extract gradients for all parameters of this specific splat
        # Each parameter type has its own gradient tensor that we index into
        grad_mu = None  # Gradient w.r.t. center position (μ)
        grad_L_diag = None  # Gradient w.r.t. Cholesky diagonal elements
        grad_L_off = None  # Gradient w.r.t. Cholesky off-diagonal elements
        grad_a = None  # Gradient w.r.t. amplitude
        grad_sharpness = None  # Gradient w.r.t. sharpness offset

        try:
            # Extract gradients with bounds checking to handle dynamic model changes
            if (
                self.model.raw_mu.grad is not None
                and splat_idx < self.model.raw_mu.grad.shape[0]
            ):
                grad_mu = self.model.raw_mu.grad[splat_idx]  # Shape: (d,)
            if (
                self.model.raw_L_diag.grad is not None
                and splat_idx < self.model.raw_L_diag.grad.shape[0]
            ):
                grad_L_diag = self.model.raw_L_diag.grad[splat_idx]  # Shape: (d,)
            if (
                self.model.L_off.grad is not None
                and splat_idx < self.model.L_off.grad.shape[0]
            ):
                grad_L_off = self.model.L_off.grad[splat_idx]  # Shape: (n_off_diag,)
            if (
                self.model.raw_a.grad is not None
                and splat_idx < self.model.raw_a.grad.shape[0]
            ):
                grad_a = self.model.raw_a.grad[splat_idx]  # Shape: scalar
            if (
                self.model.sharpness_offsets_raw.grad is not None
                and splat_idx < self.model.sharpness_offsets_raw.grad.shape[0]
            ):
                grad_sharpness = self.model.sharpness_offsets_raw.grad[
                    splat_idx
                ]  # Shape: scalar
        except IndexError as e:
            raise IndexError(
                f"Failed to extract gradients for splat {splat_idx}: {e}"
            ) from e

        # Update center position (μ) using Adam if gradients are available
        if grad_mu is not None:
            self._update_parameter(
                param=self.model.raw_mu.data[splat_idx],  # Raw parameter (logit-space)
                grad=grad_mu,  # ∂L/∂raw_mu
                exp_avg=state["exp_avg_mu"],  # First moment estimate
                exp_avg_sq=state["exp_avg_sq_mu"],  # Second moment estimate
                max_exp_avg_sq=state.get("max_exp_avg_sq_mu"),  # AMSGrad (optional)
                beta1=beta1,
                beta2=beta2,
                lr=lr * self._POS_LR_MULTIPLIER,  # Slow position updates (×0.1)
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
            )

        # Update Cholesky diagonal elements (determines splat size)
        if grad_L_diag is not None:
            self._update_parameter(
                param=self.model.raw_L_diag.data[
                    splat_idx
                ],  # Raw parameter (inverse-softplus)
                grad=grad_L_diag,  # ∂L/∂raw_L_diag
                exp_avg=state["exp_avg_L_diag"],  # First moment estimate
                exp_avg_sq=state["exp_avg_sq_L_diag"],  # Second moment estimate
                max_exp_avg_sq=state.get("max_exp_avg_sq_L_diag"),  # AMSGrad (optional)
                beta1=beta1,
                beta2=beta2,
                lr=lr * self._VAR_LR_MULTIPLIER,  # Normal variance adaptation (×1.0)
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
            )

        # Update Cholesky off-diagonal elements (determines splat orientation)
        if grad_L_off is not None:
            self._update_parameter(
                param=self.model.L_off.data[splat_idx],  # Raw parameter (unconstrained)
                grad=grad_L_off,  # ∂L/∂L_off
                exp_avg=state["exp_avg_L_off"],  # First moment estimate
                exp_avg_sq=state["exp_avg_sq_L_off"],  # Second moment estimate
                max_exp_avg_sq=state.get("max_exp_avg_sq_L_off"),  # AMSGrad (optional)
                beta1=beta1,
                beta2=beta2,
                lr=lr * self._VAR_LR_MULTIPLIER,  # Normal variance adaptation (×1.0)
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
            )

        # Update amplitude (controls splat brightness/contribution)
        if grad_a is not None:
            self._update_parameter(
                param=self.model.raw_a.data[
                    splat_idx
                ],  # Raw parameter (inverse-softplus)
                grad=grad_a,  # ∂L/∂raw_a
                exp_avg=state["exp_avg_a"],  # First moment estimate (scalar)
                exp_avg_sq=state["exp_avg_sq_a"],  # Second moment estimate (scalar)
                max_exp_avg_sq=state.get("max_exp_avg_sq_a"),  # AMSGrad (optional)
                beta1=beta1,
                beta2=beta2,
                lr=lr * self._AMP_LR_MULTIPLIER,  # Fast amplitude convergence (×2.0)
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
            )

        # Update sharpness offset (controls edge falloff profile)
        if grad_sharpness is not None:
            self._update_parameter(
                param=self.model.sharpness_offsets_raw.data[
                    splat_idx
                ],  # Raw parameter (s')
                grad=grad_sharpness,  # ∂L/∂s'
                exp_avg=state["exp_avg_sharpness"],  # First moment estimate (scalar)
                exp_avg_sq=state[
                    "exp_avg_sq_sharpness"
                ],  # Second moment estimate (scalar)
                max_exp_avg_sq=state.get(
                    "max_exp_avg_sq_sharpness"
                ),  # AMSGrad (optional)
                beta1=beta1,
                beta2=beta2,
                lr=base_lr
                * self._SHARPNESS_LR_MULTIPLIER,  # Use base LR (no gradient dilution for single-value parameter)
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
            )

    def _update_parameter(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        beta1: float,
        beta2: float,
        lr: float,
        bias_correction1: float,
        bias_correction2: float,
    ) -> None:
        """
        Core Adam parameter update following PyTorch's implementation.

        Implements the Adam update rule with optional AMSGrad variant:

        Standard Adam:
            m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            m̂_t = m_t / (1 - β₁^t)
            v̂_t = v_t / (1 - β₂^t)
            θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

        AMSGrad variant:
            v̂_t = max(v̂_{t-1}, v_t / (1 - β₂^t))

        Args:
            param: Parameter tensor to update (modified in-place)
            grad: Gradient tensor
            exp_avg: First moment estimate buffer (m_t)
            exp_avg_sq: Second moment estimate buffer (v_t)
            max_exp_avg_sq: Maximum second moment (AMSGrad only, can be None)
            beta1: First moment decay rate
            beta2: Second moment decay rate
            lr: Learning rate
            bias_correction1: First moment bias correction (1 - β₁^t)
            bias_correction2: Second moment bias correction (1 - β₂^t)
        """

        # Apply L2 regularization (weight decay) if specified
        if self.weight_decay != 0:
            grad = grad.add(param, alpha=self.weight_decay)  # grad ← grad + λ*param

        # Update first moment estimate (exponential moving average of gradients)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t ← β₁*m_{t-1} + (1-β₁)*g_t

        # Update second moment estimate (exponential moving average of squared gradients)
        exp_avg_sq.mul_(beta2).addcmul_(
            grad, grad, value=1 - beta2
        )  # v_t ← β₂*v_{t-1} + (1-β₂)*g_t²

        # Compute denominator for parameter update
        if self.amsgrad:
            # AMSGrad: use maximum of past squared gradients for stability
            if max_exp_avg_sq is None:
                raise RuntimeError("AMSGrad enabled but max_exp_avg_sq is missing")
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(self.eps)
        else:
            # Standard Adam: use current second moment estimate
            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(self.eps)

        # Compute effective step size (learning rate with bias correction)
        step_size = lr / bias_correction1  # α / (1 - β₁^t)

        # Apply parameter update: θ ← θ - step_size * m̂_t / (√v̂_t + ε)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    def add_splats(self, n_new_splats: int, lr_new: Optional[float] = None) -> None:
        """
        Add optimizer state for newly added splats.

        Args:
            n_new_splats: Number of new splats added to the model
            lr_new: Learning rate for new splats (default: base_lr)
        """
        if n_new_splats < 0:
            raise ValueError(
                f"Number of new splats must be non-negative, got {n_new_splats}"
            )

        if n_new_splats == 0:
            return  # Nothing to do

        if lr_new is not None and lr_new <= 0:
            raise ValueError(
                f"Learning rate for new splats must be positive, got {lr_new}"
            )

        current_n = len(self.splat_states)

        try:
            for i in range(n_new_splats):
                new_splat_idx = current_n + i
                self._initialize_splat(new_splat_idx, lr=lr_new)
        except Exception as e:
            raise RuntimeError(f"Failed to add {n_new_splats} new splats: {e}") from e

    def remove_splats(self, keep_mask: torch.Tensor) -> None:
        """
        Remove optimizer state for pruned splats.

        Args:
            keep_mask: Boolean mask indicating which splats to keep
        """
        if not isinstance(keep_mask, torch.Tensor):
            raise TypeError(f"keep_mask must be a torch.Tensor, got {type(keep_mask)}")

        if keep_mask.dtype != torch.bool:
            raise TypeError(f"keep_mask must be boolean tensor, got {keep_mask.dtype}")

        if len(keep_mask.shape) != 1:
            raise ValueError(
                f"keep_mask must be 1D tensor, got shape {keep_mask.shape}"
            )

        if len(keep_mask) != len(self.splat_states):
            raise ValueError(
                f"keep_mask length {len(keep_mask)} doesn't match number of splat states {len(self.splat_states)}"
            )

        try:
            # Create new state dict with only kept splats
            new_states = {}
            keep_indices = torch.where(keep_mask)[0].cpu().numpy()

            for new_idx, old_idx in enumerate(keep_indices):
                old_idx_int = old_idx.item()
                if old_idx_int in self.splat_states:
                    new_states[new_idx] = self.splat_states[old_idx_int]

            self.splat_states = new_states

        except Exception as e:
            raise RuntimeError(f"Failed to remove splats: {e}") from e

    def set_learning_rate(self, splat_idx: int, lr: float) -> None:
        """Set learning rate for a specific splat."""
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        if splat_idx < 0 or splat_idx >= self.model.n_splats():
            raise IndexError(
                f"Splat index {splat_idx} out of range [0, {self.model.n_splats()})"
            )

        if splat_idx in self.splat_states:
            self.splat_states[splat_idx]["lr"] = lr
        else:
            # Initialize splat with the specified learning rate
            self._initialize_splat(splat_idx, lr=lr)

    def get_learning_rate(self, splat_idx: int) -> float:
        """Get learning rate for a specific splat."""
        if splat_idx in self.splat_states:
            return float(self.splat_states[splat_idx]["lr"])
        return self.base_lr

    def get_effective_learning_rates(self) -> torch.Tensor:
        """Get effective learning rates for all splats (for monitoring)."""
        n_splats = self.model.n_splats()
        lrs = torch.zeros(n_splats)

        for splat_idx in range(n_splats):
            lrs[splat_idx] = self.get_learning_rate(splat_idx)

        return lrs

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for serialization."""
        return {
            "splat_states": self.splat_states,
            "global_step": self.global_step,
            "base_lr": self.base_lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from serialization."""
        self.splat_states = state_dict["splat_states"]
        self.global_step = state_dict["global_step"]
        self.base_lr = state_dict["base_lr"]
        self.betas = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.amsgrad = state_dict["amsgrad"]
